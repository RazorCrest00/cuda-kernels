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

| version | time | GFLOP/s | notes |
|---------|------|---------|-------|
| cublas  | 3.86 ms  | 4446.4 | baseline (Tensor Cores) |
| naive   | 47.18 ms | 364.2  | all global-memory reads |
| tiled   | 29.22 ms | 588.0  | ~1.6x over naive, still 7.5x behind cublas |

max_err vs cublas ≈ 2.3e-3 (float rounding, expected). T4 FP32 peak ≈ 8,100 GFLOP/s.
tiled is a basic 16x16 tile — no register blocking / vectorization / Tensor Cores yet,
which is why cublas is still miles ahead.

## run

```bash
make 02_matmul && ./build/02_matmul
```
