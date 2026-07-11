# 04 — layernorm

Normalize each row to mean 0 / variance 1, then scale and shift:

`y[j] = (x[j] - mean) / sqrt(var + eps) * gamma[j] + beta[j]`

gamma/beta are per-column learned parameters. In every transformer block (before
attention and before the MLP) to keep activations stable.

Also includes **rmsnorm** (what Llama-style models use): drop the mean subtraction,
normalize by root-mean-square only — `y[j] = x[j] / sqrt(mean(x^2) + eps) * gamma[j]`.

Main ideas:

- **reductions** for mean and variance (threads in a block cooperate in shared memory).
- **single-pass stats** — collect sum and sum-of-squares in one pass, then `var = mean(x^2) - mean^2`, instead of two passes over the row.

Versions:

- **naive** — one thread per row (only R threads are busy).
- **block** — one block per row and fused single-pass reduction.
- **rmsnorm** — block version without mean.

Checked against CPU references.

## results (R=4096, C=1024, T4)

| version | time | notes |
|---------|------|-------|
| naive (1 thread/row) | 1.137 ms | most of the GPU idle |
| block (reduction)    | 0.143 ms | ~8x faster, single-pass stats |
| rmsnorm (block)      | 0.145 ms | no mean subtraction |

all max_err = 0 vs cpu references.

## run

```bash
make 04_layernorm && ./build/04_layernorm
```

## reference

- Mark Harris, "Optimizing Parallel Reduction in CUDA" (mean/var reduction) — https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- Karpathy, llm.c — layernorm/rmsnorm CUDA kernels — https://github.com/karpathy/llm.c
- RMSNorm — Zhang & Sennrich, "Root Mean Square Layer Normalization" — https://arxiv.org/abs/1910.07467
