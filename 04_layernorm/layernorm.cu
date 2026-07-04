// layernorm over rows of an RxC matrix: y[j] = (x[j] - mean) / sqrt(var + eps).
// naive: one thread does one whole row.

#include <cstdio>
#include <vector>
#include <cmath>
#include "../common/cuda_utils.h"

// one thread per row
__global__ void layernorm_naive(const float* in, float* out, int R, int C, float eps) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= R) return;
  const float* x = in + (size_t)row * C;
  float* y = out + (size_t)row * C;

  float mean = 0.f;
  for (int j = 0; j < C; ++j) mean += x[j];
  mean /= C;

  float var = 0.f;
  for (int j = 0; j < C; ++j) { float d = x[j] - mean; var += d * d; }
  var /= C;

  float inv = rsqrtf(var + eps);
  for (int j = 0; j < C; ++j) y[j] = (x[j] - mean) * inv;
}

// cpu reference
static void layernorm_cpu(const std::vector<float>& in, std::vector<float>& out,
                          int R, int C, float eps) {
  for (int r = 0; r < R; ++r) {
    const float* x = in.data() + (size_t)r * C;
    float* y = out.data() + (size_t)r * C;
    float mean = 0.f;
    for (int j = 0; j < C; ++j) mean += x[j];
    mean /= C;
    float var = 0.f;
    for (int j = 0; j < C; ++j) { float d = x[j] - mean; var += d * d; }
    var /= C;
    float inv = 1.f / sqrtf(var + eps);
    for (int j = 0; j < C; ++j) y[j] = (x[j] - mean) * inv;
  }
}

int main() {
  const int R = 4096, C = 1024;
  const float eps = 1e-5f;
  const size_t n = (size_t)R * C;
  const size_t bytes = n * sizeof(float);

  // --- host buffers + inputs ---
  std::vector<float> h_in(n), h_out(n), h_ref(n);
  for (size_t i = 0; i < n; ++i) h_in[i] = ((i % 17) - 8) * 0.5f;
  layernorm_cpu(h_in, h_ref, R, C, eps);

  // --- device buffers + copy input over ---
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  // --- launch: one thread per row ---
  const int threads = 256;
  const int blocks = (R + threads - 1) / threads;
  layernorm_naive<<<blocks, threads>>>(d_in, d_out, R, C, eps);  // warm up
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer timer;
  timer.start();
  layernorm_naive<<<blocks, threads>>>(d_in, d_out, R, C, eps);
  float ms = timer.stop();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  // --- verify vs cpu reference ---
  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i) max_err = fmax(max_err, fabs(h_out[i] - h_ref[i]));

  printf("layernorm_naive  R=%d C=%d  %.3f ms  max_err=%.2e  %s\n",
         R, C, ms, max_err, (max_err < 1e-4) ? "OK" : "FAIL");

  cudaFree(d_in); cudaFree(d_out);
  return 0;
}
