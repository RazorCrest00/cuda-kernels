// matmul C = A*B (NxN, float). compute-bound, so we report GFLOP/s.
// compares naive vs shared-memory tiled vs cuBLAS.

#include <cstdio>
#include <vector>
#include <cmath>
#include <functional>
#include <cublas_v2.h>
#include "../common/cuda_utils.h"

#define TILE 16

// naive: one thread per output, dot-products straight from global memory
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    float acc = 0.f;
    for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// tiled: block loads TILExTILE chunks into shared mem and reuses them
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.f;

  // slide tiles across the row of A / column of B
  for (int t = 0; t < N / TILE; ++t) {
    // each thread loads one element of each tile
    As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
    Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
    __syncthreads();                       // wait until tile is fully loaded

    // accumulate this tile's contribution from fast shared mem
    for (int k = 0; k < TILE; ++k)
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();                       // wait before overwriting the tile
  }
  C[row * N + col] = acc;
}

// register-tiled: each thread computes TM outputs (a column), keeping the
// running sums in registers. block tile is BM x BN, inner K-step is BK.
// each B value loaded from shared mem is reused across all TM results.
__global__ void matmul_reg(const float* A, const float* B, float* C, int N) {
  const int BM = 64, BN = 64, BK = 8, TM = 8;
  int cRow = blockIdx.y, cCol = blockIdx.x;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  int threadCol = threadIdx.x % BN;   // 0..63
  int threadRow = threadIdx.x / BN;   // 0..7  (blockDim = 512)

  // move pointers to this block's top-left
  const float* Ablk = A + cRow * BM * N;
  const float* Bblk = B + cCol * BN;
  float* Cblk = C + cRow * BM * N + cCol * BN;

  // which element each thread loads into shared mem
  int innerColA = threadIdx.x % BK, innerRowA = threadIdx.x / BK;  // 64x8
  int innerColB = threadIdx.x % BN, innerRowB = threadIdx.x / BN;  // 8x64

  float acc[TM] = {0.f};

  for (int bk = 0; bk < N; bk += BK) {
    As[innerRowA * BK + innerColA] = Ablk[innerRowA * N + innerColA];
    Bs[innerRowB * BN + innerColB] = Bblk[innerRowB * N + innerColB];
    __syncthreads();
    Ablk += BK;        // step along K (across A's columns)
    Bblk += BK * N;    // step along K (down B's rows)

    for (int dot = 0; dot < BK; ++dot) {
      float tmpB = Bs[dot * BN + threadCol];        // reused across all TM
      for (int r = 0; r < TM; ++r)
        acc[r] += As[(threadRow * TM + r) * BK + dot] * tmpB;
    }
    __syncthreads();
  }

  for (int r = 0; r < TM; ++r)
    Cblk[(threadRow * TM + r) * N + threadCol] = acc[r];
}

// 2d register tiling: each thread computes a TM x TN block of outputs.
// A-column and B-row values are pulled into registers once and reused across
// the whole TMxTN tile (TM*TN multiply-adds per pair of register loads).
__global__ void matmul_reg2d(const float* A, const float* B, float* C, int N) {
  const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  int cRow = blockIdx.y, cCol = blockIdx.x;

  int threadCol = threadIdx.x % (BN / TN);   // 0..15
  int threadRow = threadIdx.x / (BN / TN);   // 0..15  (blockDim = 256)

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const float* Ablk = A + cRow * BM * N;
  const float* Bblk = B + cCol * BN;
  float* Cblk = C + cRow * BM * N + cCol * BN;

  // gmem->smem load indices (each thread loads several elements, strided)
  int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
  int strideA = (BM * BN / (TM * TN)) / BK;   // 32
  int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;
  int strideB = (BM * BN / (TM * TN)) / BN;   // 2

  float acc[TM * TN] = {0.f};
  float regM[TM], regN[TN];

  for (int bk = 0; bk < N; bk += BK) {
    for (int o = 0; o < BM; o += strideA)
      As[(innerRowA + o) * BK + innerColA] = Ablk[(innerRowA + o) * N + innerColA];
    for (int o = 0; o < BK; o += strideB)
      Bs[(innerRowB + o) * BN + innerColB] = Bblk[(innerRowB + o) * N + innerColB];
    __syncthreads();
    Ablk += BK;
    Bblk += BK * N;

    for (int dot = 0; dot < BK; ++dot) {
      for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + dot];
      for (int i = 0; i < TN; ++i) regN[i] = Bs[dot * BN + threadCol * TN + i];
      for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n)
          acc[m * TN + n] += regM[m] * regN[n];
    }
    __syncthreads();
  }

  for (int m = 0; m < TM; ++m)
    for (int n = 0; n < TN; ++n)
      Cblk[(threadRow * TM + m) * N + threadCol * TN + n] = acc[m * TN + n];
}

// matmul does 2*N^3 flops; this turns time into GFLOP/s
static double gflops(int N, float ms) {
  return (2.0 * N * N * N) / (ms / 1e3) / 1e9;
}

int main() {
  const int N = 2048;                      // multiple of TILE
  const size_t bytes = (size_t)N * N * sizeof(float);

  // --- host buffers + inputs ---
  std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N), h_ref(N * N);
  for (int i = 0; i < N * N; ++i) { h_A[i] = (i % 13) * 0.1f; h_B[i] = (i % 7) * 0.2f; }

  // --- device buffers + copy inputs over ---
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;

  // --- cuBLAS baseline (correctness ref + speed target) ---
  // cuBLAS is column-major, so compute B*A to get row-major A*B
  cublasHandle_t handle; cublasCreate(&handle);
  float alpha = 1.f, beta = 0.f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
              &alpha, d_B, N, d_A, N, &beta, d_C, N);  // warm up
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.start();
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
              &alpha, d_B, N, d_A, N, &beta, d_C, N);
  float ms_cublas = timer.stop();
  CUDA_CHECK(cudaMemcpy(h_ref.data(), d_C, bytes, cudaMemcpyDeviceToHost));  // save as reference

  // run a kernel launch, time it, check it against the cuBLAS reference
  auto bench = [&](const char* name, std::function<void()> launch) {
    launch();  // warm up
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.start();
    launch();
    float ms = timer.stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // max difference vs reference (float rounding -> small is fine)
    double max_err = 0.0;
    for (int i = 0; i < N * N; ++i)
      max_err = fmax(max_err, fabs(h_C[i] - h_ref[i]));
    printf("%-14s %8.2f ms  %8.1f GFLOP/s  max_err=%.2e  %s\n",
           name, ms, gflops(N, ms), max_err,
           (max_err / N < 1e-3) ? "OK" : "FAIL");
  };

  // --- results ---
  printf("Matmul  N=%d  (cuBLAS = baseline)\n", N);
  printf("%-14s %8.2f ms  %8.1f GFLOP/s  (baseline)\n",
         "cublas", ms_cublas, gflops(N, ms_cublas));

  dim3 block(TILE, TILE), grid(N / TILE, N / TILE);
  bench("naive", [&]{ matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N); });
  bench("tiled", [&]{ matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N); });

  // register-tiled: 512 threads/block, each computes TM=8 outputs (64x64 tile)
  dim3 rblock(64 * 64 / 8), rgrid(N / 64, N / 64);
  bench("reg (1d tile)", [&]{ matmul_reg<<<rgrid, rblock>>>(d_A, d_B, d_C, N); });

  // 2d register-tiled: 256 threads/block, each computes 8x8 outputs (128x128 tile)
  dim3 r2block(128 * 128 / (8 * 8)), r2grid(N / 128, N / 128);
  bench("reg (2d tile)", [&]{ matmul_reg2d<<<r2grid, r2block>>>(d_A, d_B, d_C, N); });

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  return 0;
}
