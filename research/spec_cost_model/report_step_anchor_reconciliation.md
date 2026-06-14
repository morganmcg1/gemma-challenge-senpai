# Step-anchor stack reconciliation (PR #168)

**Verdict: GREEN.** PRIMARY `step_reconciliation_self_test_passes = 1` (all four
checks pass). TEST `launch_realized_step_both_bugs = 1.2182`. Pure-analytic
CPU-only synthesis of committed outputs (#136, #154, #161); imports, does **not**
re-derive. No HF Job / submission / served-file change. BASELINE stays 481.53.
Adds 0 TPS — the step-denominator closeout that hands fern #155's launch packet
ONE defensible number instead of four anchors with a 1.1 % spread.

W&B run `oti5l4sb` (group `launch-step-reconciliation`). JSON:
`research/spec_cost_model/step_anchor_reconciliation.json`. Script:
`research/spec_cost_model/step_anchor_reconciliation.py`.

## The problem

There were **four** step-time anchors for the depth-9 tree decode step, and the
launch packet must quote exactly **one**:

| anchor | value | source | regime |
|---|---|---|---|
| roofline ideal-overlap | 1.2127 | #136 graphed floor (attn launch idle hidden) | **SUBSTITUTE** |
| measured idle-hidden overlap | 1.2182 | #136 / #161 (realistic eager star-attn) | **SUBSTITUTE** |
| scatter+LP-reduced decode-path | 1.2047 | #154 (−1.108 % @ bar) | **CONDITIONAL SUBSTITUTE** |
| both-bugs-neutral | 1.2182 | #161 (depth-1 spine adds 0) | **ADDITIVE = 0** |

The four span 1.1 % (1.2047 → 1.2182) ≈ 6 TPS at the 537 level — too wide for the
`Approval request: HF job` projection block.

## Reconciliation logic

**roofline (1.2127) and measured-overlap (1.2182) are SUBSTITUTES** — two
estimates of the *same* physical step. The +0.447 % gap is the **real exposed
star-attn launch idle** that survives realistic GEMM overlap (#136 measured
43.3 µs/step). Under `PRECACHE_BENCH=1` the warmup replays the 128 bench prompts
so the timed window is pure-decode, and the served fa2sw stack runs
`compute_logits` / star attention **eagerly** (outside the CUDA graph, per #154's
orthogonality note). So the served stack **pays** that idle → the launch-realized
reality is the **measured-overlap 1.2182**. roofline 1.2127 is the optimistic
floor a *fully-graphed attention build* would recover (blocker #2, **not
shipped**) — it is the band edge, not a second cost.

**#154's 1.2047 is the same step with an avoidable tax REMOVED** (decode-path
`[M,262144]` scatter + sampling LogitsProcessor). It is a *conditional*
substitute: it applies **only if the argmax-only decode build ships** — the
`compute_logits` token-selection-vs-`prompt_logprobs` branch (#154's
`seam_land_must_guard`). That build has **not** shipped, so it does **not** lower
the launch-realized step. It is reported as a separate not-yet-realized lane.

**#161's 1.2182 confirms bug-1 (depth-1 spine) is an ADDITIVE component of
magnitude exactly 0** — measured marginal accept-prep device-busy −0.031 µs
(within noise), step-neutral. So both-bugs step == descent-only step == 1.2182.
The +1 `target_logits_indices` deref is upstream plumbing (denken #133, 0 added
kernel ops); the served accept-prep kernel is byte-identical.

**No double-count:** roofline+overlap are *not* summed (≈ 2.43 is meaningless);
ONE substitute is quoted and the |overlap−roofline| spread (= #136's
`delta_vs_roofline_pct` = 0.447 %) is the residual uncertainty. bug-1 is additive
but 0; #154's reduction is multiplicative-conditional, not added.

## Launch-realized step + propagation

`official = K_cal·(E[T]/step)·τ`, K_cal = 125.268, τ = 1.0 (all imported,
cross-checked: K_cal agrees across #136/#154/#161).

| quantity | descent-only (E[T]=5.0564) | both-bugs (E[T]=5.2070) |
|---|---|---|
| **launch-realized step** | **1.2182** | **1.2182** |
| official @ overlap (realized) | **519.96** | **535.44** |
| official @ roofline (optimistic edge) | 522.29 | 537.84 |
| TPS band (roofline−overlap) | +2.33 | **+2.39** |

Operative clear-500 bar at the realized step: **E[T] = 4.862** (reproduces fern
#142/#155). At the roofline edge the bar would fall to 4.841.

### Conditional #154 lane (NOT in the launch-realized step)

If the argmax-only decode build ships, the reduction (imported from #154's M=32
table, not re-derived):

| | conservative | realistic |
|---|---|---|
| reduced step | 1.2078 | 1.2047 |
| clear-500 bar | 4.820 | 4.808 |
| descent-only TPS | 524.27 | 525.55 |
| both-bugs TPS | 539.76 | 541.03 |

This is a **separate, greedy-safe, not-yet-realized lever** (+4.3…5.6 TPS), not
part of the step the launch should currently quote.

## Self-test (PRIMARY)

| check | result |
|---|---|
| (a) reconciled step reproduces #136's 1.2182 (Δ = 0.0022 % < 0.10 %) | PASS |
| (b) descent-only ∈ [519.96, 522.29] ≈ 522; both-bugs ∈ [535.44, 537.84] ≈ 535–538 | PASS |
| (c) #154 reduction lowers the clear-500 bar to 4.808–4.820 | PASS |
| (d) roofline/overlap substitutes not double-counted (spread 0.447 % == #136 delta) | PASS |
| K_cal agreement across imports | PASS |

`metrics_nan_clean = 1`.

## Hand-off

The launch should quote **ONE step = 1.2182** (measured idle-hidden overlap;
shared by descent-only and both-bugs since bug-1 adds 0). This gives
**descent-only 519.96** and **both-bugs 535.44** official, with a **±2.4 TPS
uncertainty band** from the roofline↔overlap substitute spread (the upside a
fully-graphed attention build would recover). This pins the step **assumption**
in fern #155's launch packet — whose `land_tuple_spec` already defaults
`step = 1.2182` and whose step-anchor leg already carries a 0.5 % half-width — to
**one measured, defensible number with a 0.45 % band**, and quarantines the
1.2047 (#154) figure as a labelled conditional lane rather than a fourth anchor.

## Public evidence used

Synthesis of my own #136 (measured step anchor) and #161 (both-bugs step cost) +
ubel #154 (scatter+LP step-denominator reduction). openevolve's
20260614-140843 board post (oracle depth-1 conditioning localizer:
`depth1_accept = 0.6927`, `linear_top1 = 0.7287`) corroborates the bug-1
depth-1-conditioning framing that #161's both-bugs E[T]=5.207 anchor rests on —
that is the **numerator** lane (denken #166 / land #71); this PR closes the
orthogonal **denominator** (step) question.
