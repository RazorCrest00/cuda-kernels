// softmax over rows of an RxC matrix: y[j] = exp(x[j] - max) / sum_k exp(x[k] - max).
// subtract the row max first so exp() can't overflow on large inputs.
// compares naive (1 thread/row) vs block reduction (1 block/row).

#include <cstdio>
#include <vector>
#include <cmath>
#include <functional>
#include "../common/cuda_utils.h"

// naive: one thread does one whole row, sequentially over columns
__global__ void softmax_naive(const float* in, float* out, int R, int C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= R) return;
  const float* x = in + (size_t)row * C;
  float* y = out + (size_t)row * C;

  float m = -INFINITY;
  for (int j = 0; j < C; ++j) m = fmaxf(m, x[j]);
  float sum = 0.f;
  for (int j = 0; j < C; ++j) sum += expf(x[j] - m);
  for (int j = 0; j < C; ++j) y[j] = expf(x[j] - m) / sum;
}

// block: one block per row, threads split the columns and reduce in shared mem
__global__ void softmax_block(const float* in, float* out, int R, int C) {
  int row = blockIdx.x;
  int t = threadIdx.x, nt = blockDim.x;
  const float* x = in + (size_t)row * C;
  float* y = out + (size_t)row * C;
  __shared__ float red[256];

  // 1. row max: each thread reduces a strided slice, then block-reduce
  float local = -INFINITY;
  for (int j = t; j < C; j += nt) local = fmaxf(local, x[j]);
  red[t] = local; __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (t < s) red[t] = fmaxf(red[t], red[t + s]);
    __syncthreads();
  }
  float m = red[0]; __syncthreads();

  // 2. sum of exp(x - m), same reduce pattern
  local = 0.f;
  for (int j = t; j < C; j += nt) local += expf(x[j] - m);
  red[t] = local; __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  float sum = red[0]; __syncthreads();

  // 3. normalize
  for (int j = t; j < C; j += nt) y[j] = expf(x[j] - m) / sum;
}

// cpu reference (stable) to check against
static void softmax_cpu(const std::vector<float>& in, std::vector<float>& out, int R, int C) {
  for (int r = 0; r < R; ++r) {
    const float* x = in.data() + (size_t)r * C;
    float* y = out.data() + (size_t)r * C;
    float m = -INFINITY;
    for (int j = 0; j < C; ++j) m = fmaxf(m, x[j]);
    float sum = 0.f;
    for (int j = 0; j < C; ++j) sum += expf(x[j] - m);
    for (int j = 0; j < C; ++j) y[j] = expf(x[j] - m) / sum;
  }
}

int main() {
  const int R = 4096, C = 1024;
  const size_t n = (size_t)R * C;
  const size_t bytes = n * sizeof(float);

  // --- host buffers + inputs (large values: would overflow without the max trick) ---
  std::vector<float> h_in(n), h_out(n), h_ref(n);
  for (size_t i = 0; i < n; ++i) h_in[i] = ((i % 17) - 8) * 12.0f;
  softmax_cpu(h_in, h_ref, R, C);

  // --- device buffers + copy input over ---
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;

  // time a launch, copy result back, check vs reference
  auto bench = [&](const char* name, std::function<void()> launch) {
    launch(); CUDA_CHECK(cudaDeviceSynchronize());  // warm up
    timer.start();
    launch();
    float ms = timer.stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) max_err = fmax(max_err, fabs(h_out[i] - h_ref[i]));
    printf("%-18s %.3f ms  max_err=%.2e  %s\n",
           name, ms, max_err, (max_err < 1e-4) ? "OK" : "FAIL");
  };

  const int threads = 256;
  printf("Softmax  R=%d C=%d\n", R, C);
  bench("naive (1 thread/row)", [&]{
    softmax_naive<<<(R + threads - 1) / threads, threads>>>(d_in, d_out, R, C);
  });
  bench("block (reduction)", [&]{
    softmax_block<<<R, threads>>>(d_in, d_out, R, C);
  });

  cudaFree(d_in); cudaFree(d_out);
  return 0;
}
