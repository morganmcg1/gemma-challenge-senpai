<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Liveprobe depth-budget — which depths × N for a DECISIVE GO/NO-GO (PR #197 · denken)

**PRIMARY** `depth_budget_self_test_passes` = **True** (5/5 conditions, NaN-clean)
**TEST** `total_trials_for_decisive_private` = **30,455** liveprobe trials — the Neyman-optimal budget to DECISIVELY certify the **best-case** λ=1.0 build against the PRIVATE bar (both-bugs λ≥0.9780, margin only 0.022); efficiency **1.434×** over equal-allocation.
**W&B** `liveprobe-depth-budget` group · peak ≈ 29 MiB

## Honest scope
Pure-analytic **CPU-only** synthesis — a 2-D measurement-DESIGN problem (which depths × how many trials), not a new bar and not a measurement. No GPU / vLLM / HF Job / submission / kernel build / served-file change. BASELINE stays **481.53**; **0 TPS**; greedy/PPL untouched. **Bank-the-analysis** (PRIMARY = self-test). Imports — and does NOT re-derive — denken **#193** (`2clxvlr8`) `mechanism_lambda`/`metrics_at_profile`, β_primary=0.7651, β_crit=0.9649, λ̂₁=0.3419; denken **#187** (`tloghme9`) `operating_spine`, `a_d=∂E[T]/∂q_d`, `n_prompts_to_resolve`, HW187=±0.0171; denken **#191** (`jeclr39w`) private bar 0.9780 (both) / descent UNREACHABLE, drop_both=0.0235, τ_low=0.9924; denken **#183** (`82uisrez`) public bar 0.9052; descent-E[T] **#172** (`gh8pa4f3`) backward-DP E[T]. **NOT open2. NOT a launch.**

## The question
#187 priced the measurement CI of a *single-aggregate* λ̂_built against the **public** 0.9052 bar (±0.0171 @128, on-bar unresolvable). But land #71 must certify the **β-aware ladder** against the **private-stricter** 0.9780 bar (#191), and it has a *finite* liveprobe trial budget. Two design questions follow: **(1)** given a total budget, how to split trials ACROSS depths 1..9 to make the GO/NO-GO decisive with the *fewest* trials; **(2)** how many distinct depths must be measured at all — can a 2-depth β-fit extrapolate the ladder, or must all of 2..9 be probed directly. Then **(3)** what GO/NO-GO error a lazy depth-1-only read incurs.

## 1. Decisive-certification budget — Neyman allocation on the E[T] functional (deliverable 1)
Certify the build's **E[T] → private_LCB** (not a degenerate aggregate λ̂: pooling one scalar and minimizing its variance s.t. Σn_d=N puts all trials on one depth). The per-depth physics weights `a_d = ∂E[T]/∂q_d` (imported #187 central-difference on the #172 backward DP) are FIXED and non-degenerate, so the optimum is the classic **Neyman allocation** `n_d ∝ a_d·σ_d`, `σ_d=√(q_d(1−q_d))`:

| depth d | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| `a_d=∂E[T]/∂q_d` | 2.700 | 2.325 | 2.139 | 1.684 | 1.179 | 0.749 | 0.426 | 0.180 |
| Neyman fraction | 0.2585 | 0.2111 | 0.1833 | 0.1401 | 0.0979 | 0.0603 | 0.0343 | 0.0145 |

`N_d_budget[1..9] = [0, 7873, 6428, 5581, 4268, 2982, 1837, 1044, 442]` (depth-1 **pinned 0** — both-bugs span=0, it is deployed/fixed). The allocation is **SHALLOW-heavy**: depth-2 gets **18× more trials than depth-9**, because the shallow depths move E[T] hardest (a_d falls 15× from depth 2 to 9). This is the E[T]-functional generalisation of #187's inverse-variance WLS.

`total_trials_for_decisive_private` and the Cauchy–Schwarz efficiency `N_equal/N_opt = D·Σ(a_dσ_d)²/(Σa_dσ_d)²` vs the true-λ margin to the 0.9780 bar:

| true-λ | margin to 0.9780 | **N_opt (Neyman)** | N_equal | efficiency |
|---|---|---|---|---|
| 0.980 | 0.0020 | 3,781,857 | 5,441,263 | 1.439× |
| 0.985 | 0.0070 | 305,054 | 438,513 | 1.438× |
| 0.990 | 0.0120 | 103,261 | 148,305 | 1.436× |
| **1.000** | **0.0220** | **30,455** | 43,664 | **1.434×** |

**Punchline:** the private bar is so high that even a *perfect* λ=1.0 build sits only **+0.022** above it, so a decisive private GO costs **~30k trials** at best; a build at +0.002 costs **~3.8M**. Neyman buys a flat **~1.44×** over equal-allocation throughout.

## 2. Minimum depth-COUNT for β-identification (deliverable 2) — `min_depths_for_decisive = full-ladder`
The salvage-staleness law is `λ_d = λ̂₁·β^(d−1)`, so a log-linear WLS fit `ln λ_d = ln λ̂₁ + (d−1)·ln β` over depths {1..k} *could* in principle extrapolate to depth 9. The extrapolation half-width to depth 9 vs #187's single-depth precision (±0.0171):

| depths measured k | lever arm | β̂ | hw[λ₉] | within #187 CI? |
|---|---|---|---|---|
| 1..2 | 1 | 0.7651 | ±1.1001 | ❌ |
| 1..3 | 2 | 0.7651 | ±0.0653 | ❌ |
| 1..4 | 3 | 0.7651 | ±0.0447 | ❌ |
| 1..5 | 4 | 0.7651 | ±0.0370 | ❌ |
| 1..6 | 5 | 0.7651 | ±0.0333 | ❌ |
| 1..7 | 6 | 0.7651 | ±0.0300 | ❌ |
| 1..8 | 7 | 0.7651 | ±0.0288 | ❌ |
| 1..9 | 8 | 0.7651 | (all measured) | ✅ |

A **2-depth** fit's β-CI is `[0.0153, 38.3]` — effectively *unidentified* (β could be <0.02 or >1). The extrapolation hw exceeds ±0.0171 for **every** k<9: the deep, β-decayed, survival-thinned tail is exactly where extrapolation is worst, so `depth1_plus_2_suffices = False` and `min_depths_for_decisive = full-ladder`. **Land #71 must probe all of depths 2..9 DIRECTLY** — a few-depth β-fit cannot stand in for the ladder.

## 3. The under-measurement error — depth-1-only is a FALSE GO (deliverable 3)
Read λ̂₁ only, assume flat (β=1) → claim a GO; truth at the grounded β=0.7651 decays the ladder → MISS the 500 bar:

| λ̂₁ | naive-flat priv_LCB | naive GO? | TRUE mech priv_LCB (β=0.765) | true GO? | overstatement |
|---|---|---|---|---|---|
| 0.9052 | 484.6 | no | 412.4 | no | 72.1 TPS |
| 0.9500 | 493.9 | no | 415.8 | no | 78.1 TPS |
| **1.0000** | **504.9** | **YES** | **419.6** | **NO** | **85.2 TPS** |

`false_go_risk_depth1_only = True`. At λ̂₁=1.0 the naive-flat read claims a private GO (504.9 ≥ 500) while the true mechanism is a hard NO-GO (419.6) — an **85.2 TPS** overstatement. Every depth-1 read that clears the flat public bar maps, under β-decay, to a true private LCB of ~412–420 TPS.

## 4. Mechanism feasibility — no real build clears the private bar at the grounded β
`mechanism_can_clear_private_bar = False`: at β=0.7651, *even perfect* depth-1 recovery (λ̂₁=1.0) yields private_LCB = **419.6 << 500**. A GO build must have **β≈1** (no salvage staleness). So the 30k-trial budget sizes the measurement that **CONFIRMS β≈1 across the full ladder** — it is structurally a β-confirmation, not a point-λ̂ check, and cannot be shortcut by a depth-1 probe.

## 5. Self-test (PRIMARY) — `depth_budget_self_test_passes` = True
| # | condition | result |
|---|---|---|
| a | reproduce #187's single-depth N_resolve (flat, public 0.9052): {0.86:19, 0.88:60, 0.905:717062, 0.93:62, 0.95:19} via `D187.n_prompts_to_resolve` on #187's OWN bar | ✅ |
| b | reproduce #193 `beta_crit_depth1_sufficient` = 0.9649 | ✅ |
| c | reproduce #191 private bar 0.9780 (both) + descent UNREACHABLE (null) + forward map `private_lcb(0.9780,β=1)=500.0` | ✅ |
| d | conservative ordering — β-aware (decayed, lower-q) ladder resolve-load Σq(1−q)/span² ≥ flat ladder at every λ̂₁∈{0.5, 0.9052, 1.0} (staleness cannot make MEASUREMENT easier) | ✅ |
| e | NaN-clean — key scalars finite; unreachable/unidentified stored as **null**, not NaN | ✅ |

Condition (a) note: #187's published `resolve_table` was generated with its OWN truncated bar constant `0.905229`; the on-bar entry (margin 0.0002, N≈7·10⁵) is pathologically sensitive to that 6th-decimal truncation (715067 vs 717062), so faithful reproduction uses #187's bar, not the full-precision #193/#183 re-derivation.

## 6. Hand-off
**land #71 (measurement SPEC):** to DECISIVELY certify self-KV recovery against the private bar λ≥0.9780, **measure depths 2..9 DIRECTLY** (depth-1 is pinned/deployed, span 0) — do NOT extrapolate the ladder from a 2-depth β-fit (`min_depths_for_decisive=full-ladder`; `depth1+2 suffices=False`). Neyman-allocate the budget SHALLOW-heavy `N_d[1..9]=[0, 7873, 6428, 5581, 4268, 2982, 1837, 1044, 442]` (best-case λ=1.0; total≈30,455); tighter true-λ margins need quadratically more. A depth-1-only read assuming flat is a FALSE GO worth **85 TPS**.

**fern #185 (GO/NO-GO integrator):** this DESIGNS the measurement; you WIRE the verdict. Consume the certified tuple (per-depth budget, full-ladder requirement, private bar 0.9780 both / descent UNREACHABLE). The decisive-GO margin is structurally **≤0.022 in λ** (the bar is so high that λ=1.0 is only +0.022), and at β=0.765 NO build clears it — so the GO hinges on **confirming β≈1** across the measured ladder, not on a point λ̂.

**Scope:** this sizes the measurement DESIGN for the GATE; it does **NOT** move the bar (#191 owns the private bar, #183 the public) or authorize a launch. **NOT open2. NOT a launch.**
