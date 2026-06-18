# PR #636 — Option-B strict-#319 rescue: tie-deterministic M=1-recompute acceptor (stark)

## Goal
Make Option-B **strict byte-exact #319** (τ=0 vs pure M=1 greedy AR) at acceptable
TPS, eliminating the human tolerance-ruling round-trip on the #622 residual.

## The residual being rescued (stark #622, run `15g9q3wc`, MERGED)
- `break_rate_bi1_both_sides = 0.0009187` (6/6531 positions), BI=1 both sides.
- All 6 flips are int4-Marlin grid-ties: **gap = margin = 0.125 nat EXACTLY** (one
  int4 logprob grid quantum), 0 attention-path, 0 M=1-token-outside-top-k.
- Benign in graded quality (denken #626) but NOT strict τ=0 — a free-running
  greedy trajectory cascades the 0.092% per-step seed (that is wirbel #607's 47%).

## Mechanism of the acceptor (why a gap-flag is SOUND)
At a flip, M=8 picks A (its top-1), M=1 picks B. In M=8 space A>B; in M=1 space B>A.
- Let `gap_M8 = lp_M8(top1) − lp_M8(top2)` (the cheap flag scalar).
- Let `margin_M8 = lp_M8(A) − lp_M8(B)`. Since top2 ≥ B, **gap_M8 ≤ margin_M8**.
- `margin_M8 + margin_M1 = δ(A) − δ(B) ≤ 2·δ_max` where δ = per-logit M8−M1
  perturbation; margin_M1>0 at a flip ⇒ `margin_M8 < 2·δ_max`, so **gap_M8 < 2·δ_max**.
- Therefore flagging `gap_M8 < τ_flag` with **τ_flag ≥ 2·δ_max catches EVERY flip**.
  #622 measured all flip margins ≤ 0.125 nat ⇒ 2·δ_max ≲ 0.25; τ_flag = 0.5 is 2×
  that → comfortable. The free-running scan is the empirical proof of completeness.

## Acceptor
During MTP spec-verify, per verified position compute `gap_M8`; if `gap_M8 < τ_flag`
recompute that position at **M=1** (canonical strict-#319 path) and emit the M=1
argmax; else keep the fast M=8 token. Flagged positions are rare → ~1 extra M=1
forward per `1/flag_trigger_rate` tokens.

## Harness (`optionb_strict319_rescue.py`), all LOCAL A10G, BI=1 both sides, NO HF Job
**PRIMARY — `phase=scan` (rigorous, reuses validated #622 geometry):**
Teacher-forced per-step along the real M=1 AR trajectory. At each position record
`gap_M8` over ALL positions (not just flips) and the flip indicator. Accumulate per
τ_flag ∈ {0.2,0.25,0.3,0.5,0.75,1.0}:
- `flag_trigger_rate(τ) = P(gap_M8 < τ)` — the recompute frequency / TPS cost driver
  (#622 NEVER measured this; it is the decisive new number).
- `rescued_break_rate(τ) = P(flip ∧ gap_M8 ≥ τ)` — leaks the flag misses; target 0.
- `unrescued_break_rate = P(flip)` — reproduces #622's 0.092%.
- `min_tau_flag_for_zero_breaks`.
**Soundness:** if `rescued_break_rate=0` along the real AR trajectory, then by
induction the free-running rescued stream is byte-identical to M=1 AR (at every
position both emit the AR token), i.e. strict τ=0. This is the PR-permitted faithful
Python reconstruction over real generated trajectories.

**CONFIRMATORY — `phase=freerun` (literal):** size_m=8 emulated via 8 batched copies
(validated against the scan's chunk-read M=8 on teacher-forced positions). Free-run a
rescued stream and an un-rescued stream; byte-compare to M=1 AR. Demonstrates 0
rescued breaks and the un-rescued cascade. Degrades to the induction proof if the
emulation cannot assert width-8.

**SECONDARY — TPS:** project `rescued_wall_tps = 1/(1/152.291 + flag_trigger_rate/126.378)`
(un-rescued land #623 = 152.291 local; M=1 forward cost from the 126.378 AR rung,
cross-checked by a local micro-bench). `rescued_beats_126 = rescued_wall_tps > 126.378`.

## Deliverables (terminal SENPAI-RESULT)
`rescued_break_rate` (vs M=1 AR free-running, =0 target), `unrescued_break_rate`,
`min_tau_flag_for_zero_breaks`, `flag_trigger_rate` at that τ_flag,
`rescued_wall_tps_local` (or `_projected` + flag), `rescued_beats_126` (bool), VERDICT ∈
{`STRICT_319_RESCUED__TPS_VIABLE`, `STRICT_319_RESCUED__TPS_REGRESSES`, `RESCUE_INCOMPLETE`}.

## Scope
Local A10G, `analysis_only=true`, `official_tps=0`, NO HF Job / NO submission / NO
served-file change. vLLM 0.22.0, int4 W4A16 body `google/gemma-4-E4B-it-qat-w4a16-ct`
(MTP drafter not loaded — greedy temp=0 ⇒ drafter changes acceptance/speed only, never
the verify argmax, #621). W&B group `optionb-strict319-rescue-stark`.

## Baselines
- Locked rung int4_g128_lmhead (#4): **126.378** official TPS, strict-#319 byte-exact.
- Un-rescued BI=1 Option-B (land #623): **152.291** LOCAL wall_tps, official_tps=0.
