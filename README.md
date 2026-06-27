# cuda-kernels

Hand-written CUDA kernels — GEMM, softmax, layernorm, and fused attention — benchmarked against cuBLAS.

A progression of from-scratch GPU kernels, each measured against a reference (cuBLAS / PyTorch) so the
performance gap is visible. Building up toward a FlashAttention-style fused attention kernel.

## Roadmap

| # | Kernel | Concepts | Reference baseline |
|---|--------|----------|--------------------|
| 01 | Vector add | grid/block layout, memory coalescing | `cudaMemcpy` bandwidth |
| 02 | Tiled matmul (GEMM) | shared-memory tiling, occupancy | cuBLAS `sgemm` |
| 03 | Softmax | warp reductions, numerical stability | PyTorch `softmax` |
| 04 | LayerNorm | block reductions, fused mean/var | PyTorch `layer_norm` |
| 05 | Fused attention | online softmax, no materialized scores | FlashAttention / SDPA |

Each folder has the kernel, a short README with the math, and benchmark numbers.

## Build & run

CUDA can't run on a Mac — develop locally, compile/run on an NVIDIA GPU
(Google Colab free T4 is the quickest start; Lightning AI / Modal / RunPod for persistent boxes).

```bash
# on a machine with the CUDA toolkit installed
make 01_vector_add      # build one kernel
./build/01_vector_add   # run it
make all                # build everything
```

### Google Colab quickstart

1. Runtime → Change runtime type → **T4 GPU**
2. `!nvcc --version` to confirm the toolkit
3. `!git clone https://github.com/RazorCrest00/cuda-kernels && cd cuda-kernels && make all`

## Goal

Part of a 2-month ML-systems track (CUDA → inference internals → vLLM/SGLang contributions),
aimed at LLM-serving systems work.
