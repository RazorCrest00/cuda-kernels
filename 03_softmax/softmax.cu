// softmax over rows of an RxC matrix: y[j] = exp(x[j] - max) / sum_k exp(x[k] - max).
// subtract the row max first so exp() can't overflow on large inputs.
// naive: one thread does one whole row, sequentially over columns.

#include <cstdio>
#include <vector>
#include <cmath>
#include "../common/cuda_utils.h"

// one thread per row
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

  // --- launch: one thread per row ---
  const int threads = 256;
  const int blocks = (R + threads - 1) / threads;
  softmax_naive<<<blocks, threads>>>(d_in, d_out, R, C);  // warm up
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer timer;
  timer.start();
  softmax_naive<<<blocks, threads>>>(d_in, d_out, R, C);
  float ms = timer.stop();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  // --- verify vs cpu reference ---
  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i) max_err = fmax(max_err, fabs(h_out[i] - h_ref[i]));

  printf("softmax_naive  R=%d C=%d  %.3f ms  max_err=%.2e  %s\n",
         R, C, ms, max_err, (max_err < 1e-4) ? "OK" : "FAIL");

  cudaFree(d_in); cudaFree(d_out);
  return 0;
}
