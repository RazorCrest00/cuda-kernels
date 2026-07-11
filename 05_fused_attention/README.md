# 05 — fused attention

My attempt at replicating the core of a transformer:

`O = softmax(Q * K^T * scale) * V`   (scale = 1/sqrt(d))

Combines everything in this repo: matmul (Q*K^T and *V), softmax, and reductions.
Q, K, V, O are (S, d) single-head.

## The idea

Naive attention forms the full S×S score matrix (or recomputes it). **FlashAttention** streams over the keys, keeping a running max `m`, sum `l`, and output `acc`, rescaling by `exp(old_m - new_m)` whenever a bigger score shows up (**online softmax**). That keeps the big matrix out of slow memory, which is why it's fast. Used in vLLM and SGLang.

## versions

- **naive** — one thread per query, 3 passes over keys.
- **online** — one thread per query, single pass with online softmax (no buffer).
- **flash (block/query)** — one block per query; threads split the keys, each does an
  online-softmax partial, then the partials are merged (the merge is associative).
- **flash causal** — same kernel with a causal mask: query i only attends to keys j<=i
  (a token can't see the future). This is what a decoder LLM actually uses.

Checked against CPU references (non-causal and causal). (config: S=512, d=64, threads<=128.)

## results (S=512, d=64, T4)

| version | time | vs naive | notes |
|---------|------|----------|-------|
| naive               | 27.23 ms | 1x    | 3 passes, recompute scores |
| online              | 6.64 ms  | ~4x   | single pass, flash core |
| flash (block/query) | 4.31 ms  | ~6.3x | parallel over keys |
| flash causal        | 3.95 ms  | —     | causal mask (decoder attention); skips ~half the keys |

all correct (max_err ~1e-9 vs cpu reference).

## run

```bash
make 05_fused_attention && ./build/05_fused_attention
```

## reference

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" — https://arxiv.org/abs/2205.14135
- Milakov & Gimelshein, "Online normalizer calculation for softmax" — https://arxiv.org/abs/1805.02867

## next steps (not done yet)

- **query tiling** — one block per *block of queries*, so a loaded K/V tile in shared
  memory is reused across many queries. (in the current one-block-per-query design each
  K/V row is read by exactly one thread, so shared-mem tiling alone buys nothing — you
  need query tiling to create the reuse.) this is the real FlashAttention layout.
- multi-head / batched
- fp16 / bf16 inputs
