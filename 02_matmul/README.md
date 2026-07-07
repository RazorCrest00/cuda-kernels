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

| version | GFLOP/s | vs tiled | notes |
|---------|---------|----------|-------|
| cublas        | ~2800-4400 | — | baseline (Tensor Cores) |
| naive         | ~330  | 0.6x | all global-memory reads |
| tiled         | ~510  | 1x   | basic 16x16 shared-mem tile |
| reg (1d tile) | ~1930 | ~3.8x | each thread computes 8 outputs, sums in registers |

max_err vs cublas ≈ 2.3e-3 (float rounding, expected). T4 FP32 peak ≈ 8,100 GFLOP/s.
absolute GFLOP/s drifts run-to-run (GPU boost clocks); the ratios are the point.
reg tiling closes most of the gap to cublas. remaining gap = vectorized loads,
2D register tiles, and Tensor Cores (what cublas uses).

## run

```bash
make 02_matmul && ./build/02_matmul
```
