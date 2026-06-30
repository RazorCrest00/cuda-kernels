# 03 — softmax

Softmax over each row of an RxC matrix:

`y[j] = exp(x[j] - max) / sum_k exp(x[k] - max)`

Turns a row of raw scores into a probability distribution (positive, sums to 1).
Used everywhere in transformers — next-token probs and attention weights.

Two ideas this kernel teaches:

- **numerical stability** — subtract the row max first, or `exp(big)` overflows to inf.
- **reductions** — threads in a block cooperate (shared mem) to compute the row max
  and sum in parallel. First kernel where threads *combine* values.

Versions:

- **naive** — one thread per row, loops the columns alone (only R threads busy).
- **block** — one block per row, threads split columns + reduce in shared memory.

Checked against a CPU reference.

## results (R=4096, C=1024, T4)

| version | time | notes |
|---------|------|-------|
| naive (1 thread/row) | TBD | most of the GPU idle |
| block (reduction)    | TBD | fills the GPU |

## run

```bash
make 03_softmax && ./build/03_softmax
```
