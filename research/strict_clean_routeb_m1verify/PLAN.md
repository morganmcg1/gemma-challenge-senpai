# Strict-clean route-b: does tree-verify-at-M=1 net >126.378 TPS? (PR #746)

**Owner:** stark · **W&B group:** `strict-clean-routeb-m1verify` · local A10G only, NO HF job.

## Question
Run spec-dec but execute the **verify at M=1** (K sequential single-query forwards,
byte-identical to the decode path) instead of one batched M=K+1 verify. This is the
only **strict byte-exact** (G1-immune) spec path (given input: wirbel #736 — int4
Marlin GEMV is byte-exactly M-invariant, the strict-#319 divergence is pinned to the
**attention** branch: multi-query verify vs single-query decode). Net-TPS question:
after paying K sequential M=1 verify forwards/step instead of one batched M=K+1
verify, does it still clear **126.378**?

## Bar / projection convention (from research/ar_identity_safe_tps)
- Bar: **126.378** official TPS = operative `int4_g128_lmhead` AR rung (PR #601).
- Measure **local decode `wall_tps`** in the warm-steady greedy predicate
  (MAX_NUM_SEQS=1, temp=0, single-stream, 128×512 sharegpt). wall_tps is the official
  proxy (== sglang output_throughput, PR #72); TAU=1.03524 is the small local→official
  scalar. Compare wall_tps directly to 126.378.

## Cost structure (why this is the question)
Per spec step: drafter proposes K, then verify.
- **Batched verify (#730 fire, non-byte-exact):** ONE M=K+1 target forward amortizes the
  int4 weight-read over K+1 positions → emits ~accept_len tokens per ~1 target forward.
  net_TPS ≈ AR × accept_len / (1 + drafter/target).
- **Route-b (byte-exact):** K+1 **sequential M=1** target forwards. Each M=1 forward
  re-reads the full int4 weights (memory-bound) → ~1 target forward **per emitted token**,
  same forward-count as plain AR, PLUS wasted drafter overhead.
  net_TPS ≈ AR_target − drafter_overhead.

**Prior (to be measured, not assumed):** route-b removes the batched-verify amortization
that is the entire spec speedup, so net_TPS collapses toward the target's M=1 AR rate
minus drafter waste. Risk: it is then **dominated by the existing byte-exact M=1 AR
floor-lock** (`fa2sw_strict_m1ar_int4`, 161.70 local) — byte-exact AND faster, no drafter.
The honest deliverable is the measured net-TPS K-sweep + the byte-exactness tax, and
whether route-b is a useful submission or a dominated one.

## Measurement plan (real served wall_tps, not projection — #642 de-projected)
Base submission: `int4_mtp_batchinv` (int4 W4A16 target + MTP drafter, K=NUM_SPECULATIVE_TOKENS).
1. **Anchors (real served wall_tps, warm seqs=1 temp=0):**
   - M=1 AR target, spec OFF (`SENPAI_REFERENCE_MODE=1`) → route-b ceiling + identity ref.
   - Batched spec at each K∈{2,3,4,5,6} → byte-exactness-tax denominator + accept_len(K).
2. **Per-forward cost curve:** `scripts/profiler/verify_step_m_curve.py` → t_M1 vs t_batchedK+1.
3. **Route-b realized cost:** real in-loop M=1 verify-forward firing
   (`vllm_recompute_acceptor_patch` RATE mode, my #642/#663) → measured marginal C of an
   in-loop M=1 forward (includes cudagraph-break reality), to build net_TPS(route-b,K)
   from MEASURED pieces, not projection.
4. Per K report: accept_len, M=1-verify wall-cost/accepted-token, **net TPS**, clears
   126.378? Y/N, **batched-verify TPS** (the tax), strict byte-exact identity (128/128),
   PPL ≤ 2.42.

## Gates
- Strict byte-exact greedy identity vs served AR reference (zero-tol, 128/128) —
  `gen_greedy_reference` + `greedy_gate`/`check_greedy_identity`. This is the whole point.
- PPL ≤ 2.42 (`same_path_ppl`/`ppl_runner`).
- All modalities stay enabled; greedy decode unbroken.
