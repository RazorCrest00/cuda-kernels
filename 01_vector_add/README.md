# 01 — vector add

`C[i] = A[i] + B[i]` — add two arrays element by element.

The simplest possible kernel. Point is to learn the plumbing (threads/blocks,
host↔device copies, timing), not the math.

**Memory-bound**: ~no compute, just moves 3 floats per element (2 read, 1 write).
So the number that matters is **bandwidth (GB/s)** vs the GPU's peak.

## results

| GPU | time | bandwidth | notes |
|-----|------|-----------|-------|
| T4  | 0.768 ms | 262.0 GB/s | ~82% of T4's ~320 GB/s peak |

n = 2^24 (16.7M elements).

## run

```bash
make 01_vector_add && ./build/01_vector_add
```

## reference

- Mark Harris, "An Even Easier Introduction to CUDA" — https://developer.nvidia.com/blog/even-easier-introduction-cuda/
