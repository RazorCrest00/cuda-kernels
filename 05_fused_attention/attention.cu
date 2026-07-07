// attention: O = softmax(Q*K^T * scale) * V.  Q,K,V,O are (S, d), single head.
// scale = 1/sqrt(d). naive: one thread per query row.

#include <cstdio>
#include <vector>
#include <cmath>
#include <functional>
#include "../common/cuda_utils.h"

#define DMAX 64

// one thread per query. 3 passes over keys (recompute scores) to avoid buffers:
// pass 1 row max, pass 2 sum of exp, pass 3 weighted sum of V.
__global__ void attn_naive(const float* Q, const float* K, const float* V,
                           float* O, int S, int d, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= S) return;
  const float* q = Q + (size_t)i * d;

  float m = -INFINITY;
  for (int j = 0; j < S; ++j) {
    const float* k = K + (size_t)j * d;
    float s = 0.f;
    for (int t = 0; t < d; ++t) s += q[t] * k[t];
    m = fmaxf(m, s * scale);
  }

  float l = 0.f;
  for (int j = 0; j < S; ++j) {
    const float* k = K + (size_t)j * d;
    float s = 0.f;
    for (int t = 0; t < d; ++t) s += q[t] * k[t];
    l += expf(s * scale - m);
  }

  float* o = O + (size_t)i * d;
  for (int t = 0; t < d; ++t) o[t] = 0.f;
  for (int j = 0; j < S; ++j) {
    const float* k = K + (size_t)j * d;
    const float* v = V + (size_t)j * d;
    float s = 0.f;
    for (int t = 0; t < d; ++t) s += q[t] * k[t];
    float p = expf(s * scale - m) / l;
    for (int t = 0; t < d; ++t) o[t] += p * v[t];
  }
}

// cpu reference
static void attn_cpu(const std::vector<float>& Q, const std::vector<float>& K,
                     const std::vector<float>& V, std::vector<float>& O,
                     int S, int d, float scale) {
  std::vector<float> s(S);
  for (int i = 0; i < S; ++i) {
    float m = -INFINITY;
    for (int j = 0; j < S; ++j) {
      float dot = 0.f;
      for (int t = 0; t < d; ++t) dot += Q[(size_t)i * d + t] * K[(size_t)j * d + t];
      s[j] = dot * scale;
      m = fmaxf(m, s[j]);
    }
    float l = 0.f;
    for (int j = 0; j < S; ++j) { s[j] = expf(s[j] - m); l += s[j]; }
    for (int t = 0; t < d; ++t) {
      float acc = 0.f;
      for (int j = 0; j < S; ++j) acc += (s[j] / l) * V[(size_t)j * d + t];
      O[(size_t)i * d + t] = acc;
    }
  }
}

int main() {
  const int S = 512, d = 64;
  const float scale = 1.f / sqrtf((float)d);
  const size_t n = (size_t)S * d;
  const size_t bytes = n * sizeof(float);

  // --- host buffers + inputs (small deterministic values) ---
  std::vector<float> h_Q(n), h_K(n), h_V(n), h_O(n), h_ref(n);
  for (size_t i = 0; i < n; ++i) {
    h_Q[i] = ((int)(i % 13) - 6) * 0.1f;
    h_K[i] = ((int)(i % 7) - 3) * 0.1f;
    h_V[i] = ((int)(i % 11) - 5) * 0.1f;
  }
  attn_cpu(h_Q, h_K, h_V, h_ref, S, d, scale);

  // --- device buffers + copy inputs over ---
  float *d_Q, *d_K, *d_V, *d_O;
  CUDA_CHECK(cudaMalloc(&d_Q, bytes));
  CUDA_CHECK(cudaMalloc(&d_K, bytes));
  CUDA_CHECK(cudaMalloc(&d_V, bytes));
  CUDA_CHECK(cudaMalloc(&d_O, bytes));
  CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

  GpuTimer timer;

  auto bench = [&](const char* name, std::function<void()> launch) {
    launch(); CUDA_CHECK(cudaDeviceSynchronize());  // warm up
    timer.start();
    launch();
    float ms = timer.stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) max_err = fmax(max_err, fabs(h_O[i] - h_ref[i]));
    printf("%-16s %.3f ms  max_err=%.2e  %s\n",
           name, ms, max_err, (max_err < 1e-3) ? "OK" : "FAIL");
  };

  printf("Attention  S=%d d=%d\n", S, d);
  const int threads = 128;
  bench("naive", [&]{
    attn_naive<<<(S + threads - 1) / threads, threads>>>(d_Q, d_K, d_V, d_O, S, d, scale);
  });

  cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
  return 0;
}
