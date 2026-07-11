# 03 — softmax

Softmax over each row of an RxC matrix:

`y[j] = exp(x[j] - max) / sum_k exp(x[k] - max)`

Turns a raw row into a probability distribution (positive, sums to 1).
--> transformer usage

Two main ideas:

- **numerical stability** — subtract the row max first, or `exp(big)` overflows to inf.
- **reductions** - threads in a block share memory to compute the row max and sum in parallel. First kernel where threads combine values.

Versions:

- **naive** — one thread per row, loops the columns alone (only R threads busy).
- **block** — one block per row, threads split columns + reduce in shared memory.

Checked against a CPU reference.

## results (R=4096, C=1024, T4)

| version | time | notes |
|---------|------|-------|
| naive (1 thread/row) | 0.967 ms | most of the GPU idle |
| block (reduction)    | 0.168 ms | ~5.8x faster, fills the GPU |

both max_err = 0 vs cpu reference.

## run

```bash
make 03_softmax && ./build/03_softmax
```

## reference

The block-reduction pattern (row max / sum) is the classic parallel reduction:

- Mark Harris, "Optimizing Parallel Reduction in CUDA" — https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- online softmax (the stable, streaming form) — Milakov & Gimelshein, "Online normalizer calculation for softmax" — https://arxiv.org/abs/1805.02867
