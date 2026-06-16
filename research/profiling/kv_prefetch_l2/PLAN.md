# KV-cache L2 prefetch in Triton verify-attention — gate card (PR #445)

**Lever:** prefetch the next attention layer's KV pages from HBM into the A10G L2
(`cp.async.bulk.prefetch.L2`) while the current layer computes, to overlap
HBM-read latency with compute in the Triton unified-attention kernel — byte-exact
(pure memory scheduling, identical FP arithmetic).

This is an **analysis/measurement-only** card. No served-file change, no HF job,
no leaderboard submission. The advisor pre-registered a hard self-abort gate:

- Profile `t_attn_frac_of_verify` = attention share of the verify forward wall on
  the served K=7 stack (`fa2sw_precache_kenyan`, deployed 481.53).
- **attention < 10% of verify → self-abort** (system-level prefetch gain ≤ 1 TPS;
  bank the negative, do NOT prototype).
- attention ≥ 15% → instrument a prefetch stub and A/B it.

## Baseline anchors
- Deployed incumbent (NON-equivalent, identity 0.9966): **481.53 TPS** / PPL 2.3772
  — PR #52, `fa2sw_precache_kenyan`, W&B `2x9fm2zx`.
- Realized equivalence frontier (anchor base): **467.14 TPS** — denken #423, W&B `5a6zq2yz`.
- PPL gate ≤ 2.42; identity gate strict byte-exact greedy-token.

## Authoritative committed evidence (deployed split-KV stack)
`research/profiling/frontier_decode_postsplitkv/frontier_decode_profile.json`
(2026-06-13, submission `fa2sw_precache_kenyan`, the deployed 481.53 stack)
gives the verify-forward decomposition directly:

| component | ms/cycle | of verify_gpu_ms (6.519) |
|---|---|---|
| body int4-Marlin GEMM | 4.826 | 74.0% |
| norm + elementwise | 0.595 | 9.1% |
| **attention (fa2sw Triton)** | **0.605** | **9.28%** |
| sampling | 0.233 | 3.6% |
| lmhead12k GEMM | 0.026 | 0.4% |

- **t_attn_frac_of_verify ≈ 9.28%** (attention / verify_gpu_ms), 7.6% of gpu_busy.
- This is **below the 10% self-abort gate** → expected verdict: SELF-ABORT.
- The split-KV verify lever (PR #43) already collapsed verify attention from
  19.6% of cycle (pre-split-KV, ~424.5 frontier, PR #39 `profile_attention.py`)
  to 7.6% of cycle / 9.28% of verify here. The attention headroom is already taken.

## Hardware-premise corrections (measured on this pod's A10G)
- **L2 = 6.29 MB, not 40 MB.** The PR premises "11 MB KV comfortably inside the
  40 MB L2"; A10G (GA102) has a 6 MB L2 — the full-prompt KV does NOT fit. Only
  ~1 layer-ahead of KV fits, so any prefetch is necessarily one-layer-ahead.
- Gemma-4-E4B text decoder = **37 layers** (30 sliding + 7 full), not 42.

## Mechanism risk (why even the 9.3% is likely not prefetch-addressable)
- PR #39 `profile_attention.py` found the served attention is **occupancy/launch-
  bound at conc=1** (runs well below the SWA bandwidth floor; M=8 verify uses
  ~6 CTAs on 80 SMs). L2 prefetch targets HBM-read **latency/bandwidth**, which
  is not the binding constraint for a launch-bound kernel.
- Open question (PR instruction #3): does the Triton kernel already emit
  `cp.async` (ldgsts) through its pipeline stages? If so the marginal explicit-
  prefetch benefit ≈ 0.

## Confirmatory measurement plan (this pod, local microbench)
`scripts/profiler/kv_prefetch_l2_gate.py`:
1. Re-derive end-to-end `t_attn_frac_of_verify` from the committed post-split-KV
   profile (authoritative denominator).
2. Fresh microbench of the real `vllm ... unified_attention` kernel at M=8 verify
   across the 37-layer arch → attention ms/cycle + bandwidth efficiency
   (occupancy-bound check) + split-KV speedup. (reuses PR #39 `bench_op`)
3. Decision gate + honest optimistic prefetch-ceiling bracket.
4. Self-test + W&B (`--wandb_group kv-prefetch-l2`).
