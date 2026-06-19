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
ceiling**. The honest deliverable is the measured net-TPS K-sweep + the byte-exactness
tax, and whether route-b is a useful submission or a dominated one.

> **DE-PROJECTION CORRECTION (#642).** An earlier draft of this plan cited
> `fa2sw_strict_m1ar_int4 = 161.70 local` as the byte-exact M=1 AR floor-lock. That is
> WRONG: that submission's manifest says **"modeled 161.70 OFFICIAL TPS (lawine #438)"** —
> a *modeled official* projection for the heavily-approximated dixie-flatline stack
> (lm_head-prune to 12k vocab, FA-sliding, fused sparse argmax), which is byte-exact
> *within-stack* but NOT vs the canonical reference. The **genuine byte-exact static AR
> ceiling is MEASURED**: `int4_g128_lmhead` (the bar's own config) AR ≈ **126.4 local**
> ≈ the 126.378 bar (`research/ar_identity_safe_tps`, same pod). That — not 161.70 — is
> route-b's most-charitable ceiling, and it sits AT the bar (consistent with denken #740:
> static byte-exact is exhausted at the bar). Route-b ≤ this ceiling, so route-b < bar.

## Result (MEASURED, 2026-06-19)

**Verdict: route-b CANNOT be both byte-exact (G1-immune) AND clear 126.378.** Its M=1
single-query verify provides zero weight-read amortization → `route_b_tps ≤ plain
byte-exact AR`. The genuine byte-exact AR ceiling ≈ 126.4 ≈ the bar (denken #740), so
route-b lands below the bar by the no-amortization + drafter tax. Only the **non**-byte-exact
batched verify clears the bar (measured 28/128 greedy identity at K=2 — the very G1 DQ
source route-b exists to remove). G1-immunity and clearing the bar are mutually exclusive.

Measured anchors (local warm-steady, MAX_NUM_SEQS=1, temp=0, 128×512 sharegpt):
- `int4_g128_lmhead` plain byte-exact AR = **126.4 local** ≈ bar (the byte-exact ceiling).
- `int4_mtp_batchinv` M=1 AR ref, fast-kernel (batchinv-OFF, route-b's true ceiling on this
  base) = **95.034 local** / 98.383 official-proj, PPL 2.0058 (W&B `w1owpt54`).
- `int4_mtp_batchinv` M=1 AR ref, batchinv-ON = 77.843 local, PPL 2.0055 (W&B `a5vcnqy5`).
- Batched-spec K-sweep (the non-byte-exact reference): K=2 144.85 … K=4 161.13 (best) …
  K=6 156.40 local — all clear the bar locally, K=6 PPL 2.0055.
- Byte-exactness of batched verify vs M=1 AR ref: K=2 **28/128**, K=3 23/128, K=4 28/128
  identical (onset min=0) → non-byte-exact even with `VLLM_BATCH_INVARIANT=1`.
- Route-b net-TPS (fast-kernel-anchored, bounded by 95.034): K=2 upper 73.2 / drafter-incl
  62.5 … K=6 45.5 — all far below the bar. Verdict run W&B `8dl148ds`.

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
