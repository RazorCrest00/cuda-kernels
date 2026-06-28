// vector add: C[i] = A[i] + B[i]. memory-bound, so we report GB/s.

#include <cstdio>
#include <vector>
#include "../common/cuda_utils.h"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 24;
  const size_t bytes = n * sizeof(float);

  // host buffers
  std::vector<float> h_a(n), h_b(n), h_c(n);
  for (int i = 0; i < n; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

  // device buffers + copy inputs over
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  // warm up
  vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // timed run
  GpuTimer timer;
  timer.start();
  vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
  float ms = timer.stop();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  // check
  bool ok = true;
  for (int i = 0; i < n; ++i) {
    if (h_c[i] != 3.0f) { ok = false; break; }
  }

  double gb = 3.0 * bytes / 1e9;  // 2 reads + 1 write
  printf("vector_add  n=%d  %.3f ms  %.1f GB/s  %s\n",
         n, ms, gb / (ms / 1e3), ok ? "OK" : "FAIL");

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return ok ? 0 : 1;
}
