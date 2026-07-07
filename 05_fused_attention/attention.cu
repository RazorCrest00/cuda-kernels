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

// online softmax: single pass over keys. keep running max m, sum l, and output
// acc; when a new max appears, rescale l and acc by exp(old_m - new_m).
// this is the FlashAttention core -- the full score row is never materialized.
__global__ void attn_online(const float* Q, const float* K, const float* V,
                            float* O, int S, int d, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= S) return;
  const float* q = Q + (size_t)i * d;

  float m = -INFINITY, l = 0.f;
  float acc[DMAX];
  for (int t = 0; t < d; ++t) acc[t] = 0.f;

  for (int j = 0; j < S; ++j) {
    const float* k = K + (size_t)j * d;
    const float* v = V + (size_t)j * d;
    float s = 0.f;
    for (int t = 0; t < d; ++t) s += q[t] * k[t];
    s *= scale;

    float m_new = fmaxf(m, s);
    float corr = expf(m - m_new);   // rescale old running values
    float p = expf(s - m_new);      // weight of this key
    l = l * corr + p;
    for (int t = 0; t < d; ++t) acc[t] = acc[t] * corr + p * v[t];
    m = m_new;
  }

  float* o = O + (size_t)i * d;
  for (int t = 0; t < d; ++t) o[t] = acc[t] / l;
}

// flash: one block per query. threads each run online softmax over a strided
// slice of keys -> partial (m,l,acc), then the partials are merged. the merge
// is the same online-softmax rescale, so splitting the keys is exact.
__global__ void attn_flash(const float* Q, const float* K, const float* V,
                           float* O, int S, int d, float scale) {
  int i = blockIdx.x;                 // query
  int t = threadIdx.x, nt = blockDim.x;

  __shared__ float qs[DMAX];
  __shared__ float sm[128];           // per-thread max
  __shared__ float sl[128];           // per-thread sum
  __shared__ float sacc[128 * DMAX];  // per-thread output accumulator

  for (int x = t; x < d; x += nt) qs[x] = Q[(size_t)i * d + x];
  __syncthreads();

  // this thread's online softmax over its keys
  float m = -INFINITY, l = 0.f;
  float acc[DMAX];
  for (int x = 0; x < d; ++x) acc[x] = 0.f;
  for (int j = t; j < S; j += nt) {
    const float* k = K + (size_t)j * d;
    const float* v = V + (size_t)j * d;
    float s = 0.f;
    for (int x = 0; x < d; ++x) s += qs[x] * k[x];
    s *= scale;
    float m_new = fmaxf(m, s);
    float corr = expf(m - m_new);
    float p = expf(s - m_new);
    l = l * corr + p;
    for (int x = 0; x < d; ++x) acc[x] = acc[x] * corr + p * v[x];
    m = m_new;
  }
  sm[t] = m; sl[t] = l;
  for (int x = 0; x < d; ++x) sacc[t * d + x] = acc[x];
  __syncthreads();

  // thread 0 merges the per-thread partials
  if (t == 0) {
    float M = -INFINITY, L = 0.f;
    float A[DMAX];
    for (int x = 0; x < d; ++x) A[x] = 0.f;
    for (int p = 0; p < nt; ++p) {
      float M_new = fmaxf(M, sm[p]);
      float c1 = expf(M - M_new);      // rescale running
      float c2 = expf(sm[p] - M_new);  // rescale incoming
      L = L * c1 + sl[p] * c2;
      for (int x = 0; x < d; ++x) A[x] = A[x] * c1 + sacc[p * d + x] * c2;
      M = M_new;
    }
    float* o = O + (size_t)i * d;
    for (int x = 0; x < d; ++x) o[x] = A[x] / L;
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

  auto bench = [&](const char* name, const std::vector<float>& ref, std::function<void()> launch) {
    launch(); CUDA_CHECK(cudaDeviceSynchronize());  // warm up
    timer.start();
    launch();
    float ms = timer.stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) max_err = fmax(max_err, fabs(h_O[i] - ref[i]));
    printf("%-20s %.3f ms  max_err=%.2e  %s\n",
           name, ms, max_err, (max_err < 1e-3) ? "OK" : "FAIL");
  };

  printf("Attention  S=%d d=%d\n", S, d);
  const int threads = 128;
  bench("naive", h_ref, [&]{
    attn_naive<<<(S + threads - 1) / threads, threads>>>(d_Q, d_K, d_V, d_O, S, d, scale);
  });
  bench("online", h_ref, [&]{
    attn_online<<<(S + threads - 1) / threads, threads>>>(d_Q, d_K, d_V, d_O, S, d, scale);
  });
  bench("flash (block/query)", h_ref, [&]{
    attn_flash<<<S, threads>>>(d_Q, d_K, d_V, d_O, S, d, scale);  // threads must be <= 128
  });

  cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
  return 0;
}
