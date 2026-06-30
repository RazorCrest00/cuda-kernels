// matmul C = A*B (NxN, float). compute-bound, so we report GFLOP/s.
// compares naive vs shared-memory tiled vs cuBLAS.

#include <cstdio>
#include <vector>
#include <cmath>
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

  // launch grid: one thread per output element, TILExTILE blocks
  dim3 block(TILE, TILE);
  dim3 grid(N / TILE, N / TILE);
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

  // run a kernel, time it, check it against the cuBLAS reference
  auto run = [&](const char* name, void (*kern)(const float*, const float*, float*, int)) {
    kern<<<grid, block>>>(d_A, d_B, d_C, N);  // warm up
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.start();
    kern<<<grid, block>>>(d_A, d_B, d_C, N);
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
  run("naive", matmul_naive);
  run("tiled", matmul_tiled);

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  return 0;
}
