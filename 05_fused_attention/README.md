# 05 — fused attention

The core of a transformer:

`O = softmax(Q * K^T * scale) * V`   (scale = 1/sqrt(d))

Combines everything in this repo: matmul (Q*K^T and *V), softmax, and reductions.
Q, K, V, O are (S, d) — single head.

## the idea

Naive attention forms the full S×S score matrix (or recomputes it). **FlashAttention**
never materializes it: it streams over the keys keeping a running max `m`, sum `l`,
and output `acc`, rescaling by `exp(old_m - new_m)` whenever a bigger score shows up
(**online softmax**). That keeps the big matrix out of slow memory — which is why it's
fast, and why modern LLM serving (vLLM, SGLang) relies on it.

## versions

- **naive** — one thread per query, 3 passes over keys (recompute scores).
- **online** — one thread per query, single pass with online softmax (no score buffer).
- **flash (block/query)** — one block per query; threads split the keys, each does an
  online-softmax partial, then the partials are merged (the merge is associative).

Checked against a CPU reference. (config: S=512, d=64, threads<=128.)

## results (S=512, d=64, T4)

| version | time | notes |
|---------|------|-------|
| naive               | TBD | 3 passes, recompute scores |
| online              | TBD | single pass, flash core |
| flash (block/query) | TBD | parallel over keys |

## run

```bash
make 05_fused_attention && ./build/05_fused_attention
```

## reference

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" — https://arxiv.org/abs/2205.14135
- Milakov & Gimelshein, "Online normalizer calculation for softmax" — https://arxiv.org/abs/1805.02867

## next steps (not done yet)

- tile K/V into shared memory (cooperative loads, reuse across the block)
- causal masking (skip keys j > i)
- multi-head / batched
