#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap every CUDA call: CUDA_CHECK(cudaMalloc(...));
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err__));                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                          \
  } while (0)

// Simple GPU timer using CUDA events. Returns elapsed milliseconds.
struct GpuTimer {
  cudaEvent_t start_, stop_;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start() { CUDA_CHECK(cudaEventRecord(start_)); }
  float stop() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};
