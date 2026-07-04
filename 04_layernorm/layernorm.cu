// layernorm over rows of an RxC matrix:
//   y[j] = (x[j] - mean) / sqrt(var + eps) * gamma[j] + beta[j].
// gamma/beta are per-column learned scale/shift.
// compares naive (1 thread/row) vs block reduction (1 block/row).

#include <cstdio>
#include <vector>
#include <cmath>
#include <functional>
#include "../common/cuda_utils.h"

// naive: one thread does one whole row
__global__ void layernorm_naive(const float* in, float* out,
                                const float* gamma, const float* beta,
                                int R, int C, float eps) {
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
  for (int j = 0; j < C; ++j) y[j] = (x[j] - mean) * inv * gamma[j] + beta[j];
}

// block: one block per row. single pass over x collects sum and sum-of-squares,
// then var = mean(x^2) - mean^2 (one reduce of each instead of two passes).
__global__ void layernorm_block(const float* in, float* out,
                                 const float* gamma, const float* beta,
                                 int R, int C, float eps) {
  int row = blockIdx.x;
  int t = threadIdx.x, nt = blockDim.x;
  const float* x = in + (size_t)row * C;
  float* y = out + (size_t)row * C;
  __shared__ float rsum[256];
  __shared__ float rsq[256];

  // one pass: accumulate sum and sum of squares
  float s = 0.f, sq = 0.f;
  for (int j = t; j < C; j += nt) { float v = x[j]; s += v; sq += v * v; }
  rsum[t] = s; rsq[t] = sq; __syncthreads();
  for (int k = nt / 2; k > 0; k >>= 1) {
    if (t < k) { rsum[t] += rsum[t + k]; rsq[t] += rsq[t + k]; }
    __syncthreads();
  }

  float mean = rsum[0] / C;
  float var = rsq[0] / C - mean * mean;
  float inv = rsqrtf(var + eps);

  for (int j = t; j < C; j += nt) y[j] = (x[j] - mean) * inv * gamma[j] + beta[j];
}

// cpu reference
static void layernorm_cpu(const std::vector<float>& in, std::vector<float>& out,
                          const std::vector<float>& gamma, const std::vector<float>& beta,
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
    for (int j = 0; j < C; ++j) y[j] = (x[j] - mean) * inv * gamma[j] + beta[j];
  }
}

int main() {
  const int R = 4096, C = 1024;
  const float eps = 1e-5f;
  const size_t n = (size_t)R * C;
  const size_t bytes = n * sizeof(float);
  const size_t cbytes = (size_t)C * sizeof(float);

  // --- host buffers + inputs ---
  std::vector<float> h_in(n), h_out(n), h_ref(n), h_gamma(C), h_beta(C);
  for (size_t i = 0; i < n; ++i) h_in[i] = ((i % 17) - 8) * 0.5f;
  for (int j = 0; j < C; ++j) { h_gamma[j] = 1.f + (j % 5) * 0.1f; h_beta[j] = (j % 3) * 0.1f; }
  layernorm_cpu(h_in, h_ref, h_gamma, h_beta, R, C, eps);

  // --- device buffers + copy inputs over ---
  float *d_in, *d_out, *d_gamma, *d_beta;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMalloc(&d_gamma, cbytes));
  CUDA_CHECK(cudaMalloc(&d_beta, cbytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), cbytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), cbytes, cudaMemcpyHostToDevice));

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
    printf("%-20s %.3f ms  max_err=%.2e  %s\n",
           name, ms, max_err, (max_err < 1e-4) ? "OK" : "FAIL");
  };

  const int threads = 256;
  printf("LayerNorm  R=%d C=%d\n", R, C);
  bench("naive (1 thread/row)", [&]{
    layernorm_naive<<<(R + threads - 1) / threads, threads>>>(d_in, d_out, d_gamma, d_beta, R, C, eps);
  });
  bench("block (reduction)", [&]{
    layernorm_block<<<R, threads>>>(d_in, d_out, d_gamma, d_beta, R, C, eps);
  });

  cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
  return 0;
}
