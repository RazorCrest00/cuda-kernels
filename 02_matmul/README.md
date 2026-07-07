# 02 — matmul

`C = A * B` (NxN, float). Each output is a dot product of a row of A and a column of B.

The single most important op in deep learning — every linear layer is a matmul.

**Compute-bound**: ~2*N^3 flops over only N^2 data, so the metric is **GFLOP/s**.
Three versions to compare:

- **naive** — one thread per output, all reads straight from global memory (slow).
- **tiled** — block stages 16x16 tiles into shared memory and reuses them (`__syncthreads`).
- **cublas** — NVIDIA's tuned library: correctness reference + speed target.

A small `max_err` vs cublas is just float rounding (different sum order), not a bug.

## results (N=2048, T4)

optimization climb (N=2048, one T4 run; cublas = 5260 GFLOP/s that run):

| version | GFLOP/s | % of cublas | what changed |
|---------|---------|-------------|--------------|
| cublas        | 5260 | 100% | baseline (Tensor Cores) |
| naive         |  438 |   8% | all global-memory reads |
| tiled         |  769 |  15% | 16x16 shared-mem tile |
| reg (1d tile) | 1922 |  37% | each thread does 8 outputs, sums in registers |
| reg (2d tile) | 4147 |  79% | each thread does an 8x8 output block |
| vec (float4)  | 4627 |  88% | float4 vectorized loads/stores |

max_err vs cublas ≈ 2.3e-3 (float rounding, expected). T4 FP32 peak ≈ 8,100 GFLOP/s.
absolute GFLOP/s drifts run-to-run (GPU boost clocks); the % / ratios are the point.
88% of cublas from a hand-written kernel. remaining gap = Tensor Cores (wmma/mma),
double-buffering/prefetch, and autotuned tile sizes.

## run

```bash
make 02_matmul && ./build/02_matmul
```
