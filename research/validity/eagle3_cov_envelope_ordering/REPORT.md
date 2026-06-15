<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #340 — Coverage→envelope ordering: solve c\* for speed-500 vs the 0.9213 identity bar

**PRIMARY `cov_envelope_ordering_self_test_passes` = True** (all 14 checks)
**TEST `c_star_central_for_500` = 0.9089** (coverage at which the central envelope == 500)
**REPORT `c_star_worst_for_500` = 0.9256** (coverage at which the worst envelope == 500; the strict bar)
**W&B `jwv1vbug`** (group `eagle3-cov-envelope-ordering`) · LOCAL read-only analytic, 0 GPU, 0 TPS

> **Verdict: `c*_central < identity_bar < c*_worst` — CONFIRMED.** Inverting stark #337's banked
> envelope map (`envelope_X(c) = X_ANCHOR · E[T](c)/E[T](0.9213)`, E[T](c)=1+Σ_{d=1..7} c^d) gives
> **c\*_central = 0.9089** and **c\*_worst = 0.9256**, so the ordering is
> **0.9089 < 0.9213 < 0.9256**. Clearing the strict **identity bar (0.9213) is SUFFICIENT for
> central-500** (0.0124 of coverage to spare) **but NOT sufficient for worst-case 500**
> (0.0043 short). Tying to lawine #336: the **+0.031** retrain lands the head exactly at 0.9213, so
> worst-500 needs **0.0043 more** coverage than the retrain delivers (total lift **0.0353** from the
> honest 0.8903 prior, just past the +0.031 budget). Standing caveat (fern #335): coverage is the
> **demand** axis only — **even at c\*_worst the supply floor must still revive (φ≥0.255)**; coverage
> alone never reaches 500.

## 1. Envelope map (deliverable 1)

The same banked map stark #337 built, written as an explicit function of coverage `c` (= per-depth
effective acceptance c_eff, the E[T](c) axis):

> **E[T](c) = 1 + Σ_{d=1..7} c^d**  ·  **envelope_X(c) = X_ANCHOR · E[T](c) / E[T](0.9213)**

| corner | anchor @E[T]=6.1112 | round-trip @0.9213 | #337 round-trip @0.8903 |
|---|---|---|---|
| central (cap-bound, λ-ceiling) | **520.95** | 520.95 ✅ | **470.35** ✅ |
| worst (uncapped private-tax) | **492.87** | 492.87 ✅ | **444.99** ✅ |

Worst/central tile factor = 492.87/520.95 = **0.9461**. Both envelopes are **strictly monotone
increasing in c** (verified on a 1000-point grid), so each `env_X(c)=500` has a **unique** root.

## 2. Inverse solve — PRIMARY (deliverable 2)

Monotone bisection on `env_X(c) = 500` (residuals at machine precision, ≈1e-13):

| corner | target E[T] = 500·E[T](0.9213)/anchor | **c\*** |
|---|---|---|
| central | 5.8654 | **c\*_central = 0.908936** |
| worst | 6.1997 | **c\*_worst = 0.925604** |

Both roots lie in (0,1); `e_t(c*)` reproduces the target E[T] to 1e-6 (inverse self-consistency).

## 3. The ordering verdict (deliverable 3)

> **0.9089 (c\*_central)  <  0.9213 (identity_bar)  <  0.9256 (c\*_worst)**

- `identity_bar_suffices_for_central_500` = **True** (c\*_central ≤ 0.9213)
- `identity_bar_suffices_for_worst_500` = **False** (c\*_worst > 0.9213)

The anticipated ordering (central anchor 520.95 > 500 already at the bar; worst anchor 492.87 < 500 at
the bar) is **CONFIRMED** by the solved roots. Clearing greedy-identity buys central-500; private-stable
worst-500 demands a **stricter** coverage than identity.

## 4. Coverage headroom / shortfall vs the +0.031 retrain budget (deliverable 4)

| quantity | value |
|---|---|
| `central_500_headroom` = 0.9213 − c\*_central | **+0.0124** (slack the bar gives) |
| `worst_500_shortfall` = c\*_worst − 0.9213 | **+0.0043** (extra beyond the bar) |
| total lift for worst-500 from honest prior (c\*_worst − 0.8903) | **0.0353** |
| lawine #336 retrain budget (0.9213 − 0.8903) | **0.031** |
| worst-500 within +0.031 budget? | **No** (needs +0.0043 more; shortfall = **13.9%** of budget) |

The shortfall is **small but nonzero**: because the +0.031 retrain is designed to land exactly at the
identity bar, a worst-500 retrain must **target c\*_worst = 0.9256**, not merely 0.9213.

## 5. Sub-cliff structure — c\* are head-coverage targets (deliverable 5)

The inverse map scales the envelope at the **sub-cliff W=4** operating point (verify step ×1). stark
#337 proved widening to restore E[T] crosses the M=32→33 cliff (**μ=1.16981 > μ_tie=1.1076**, a net
loss), so the **only lever for higher coverage is a better head** — exactly what lawine #336's retrain
delivers. Hence c\*_central / c\*_worst are **head-coverage targets at fixed W=4**, not tree-width
changes. Mapping to top-4 rank coverage (salvage `cov=(c_eff−a₁)/(1−a₁)`, a₁=0.7731, color only):
**c\*_worst needs rank-cov 0.6722 > the deployed linear spine's 0.6532** — worst-500 asks for a head
*stronger than the linear spine itself*.

## 6. Decision framing (deliverable 6)

- **If the #319 retrain targets CENTRAL-500:** set the coverage target = **identity bar 0.9213**. The
  +0.031 lawine #336 lift suffices; identity is the binding constraint.
- **If it targets WORST-case (private-stable) 500:** set the coverage target = **c\*_worst = 0.9256**,
  i.e. **+0.0043 beyond the identity bar** and **+0.0353 total** from the honest 0.8903 prior — just
  past the +0.031 budget.
- **Supply caveat (fern #335):** coverage is the **demand** axis only. The binding axis is **supply**:
  even at c\*_worst the supply floor must still revive (**φ≥0.255**). c\* is a *necessary* demand-side
  target, not sufficient on its own — coverage alone never reaches 500.

## 7. Self-test (NaN-clean, deterministic)

All **14** PRIMARY checks pass: (a) anchor round-trip 520.95/492.87 @0.9213 · (b) reproduce #337
470.35/444.99 @0.8903 · (c) E[T]/both envelopes monotone increasing · (d) c\*_central solves
env=500 · (e) c\*_worst solves env=500 · (f) worst anchor < central ⇒ c\*_worst > c\*_central · (g)
imports exact (round to displayed 0.9213/520.95/492.87/0.8903) · (h) NaN-clean · (i) roots ∈ (0,1) ·
(j) anticipated ordering confirmed · (k) sufficiency central-YES/worst-NO · (l) headroom & shortfall
both positive · (m) worst-500 exceeds the +0.031 budget · (n) inverse self-consistency (e_t(c\*) ==
target E[T]).

## 8. Honest caveats (carried in the artifact)

1. **INVERSE of stark #337 — re-prices nothing measured.** It inverts the *same* banked envelope map
   (anchors × E[T] lever ratio) to find c\* at env=500. No EAGLE-3 fusion checkpoint runs here; **not a
   running EagleProposer.**
2. **Central anchor is cap-bound** at the λ-ceiling 520.95. Above the identity bar the central envelope
   is **flat at the ceiling** (a higher E[T] cannot exceed the cap), so `envelope_central(c)` scaling is
   load-bearing **only below the bar** — exactly where the 500 crossing (c\*_central) lies. The worst
   corner is uncapped and scales throughout.
3. **c is per-depth c_eff** (the E[T](c) axis), in the **same units** as the identity bar 0.9213, the
   cov prior 0.8903, and lawine #336's +0.031. The rank-cov map (§5) is color only.
4. **Demand axis only.** c\* is a necessary coverage target; fern #335's supply floor (φ≥0.255) is the
   complementary binding constraint. **NOT a launch / build / served-file change / HF Job / submission /
   open2.**

## Greedy/PPL-safety certificate

`analysis_only = True`. No served-file change, no emitted-token change, no HF Job, no submission, NOT a
launch, NOT a build. Coverage/E[T] is the **speed** axis: the draft acceptance rate changes *how fast*
tokens are verified, not *which* token is emitted (the target model's argmax remains the emitted token),
so **greedy identity is invariant to c**. BASELINE **481.53 TPS unchanged**; this leg adds **0 TPS**.

## Hand-off

The SPEED-500 envelope needs coverage **c\*_central=0.9089** (central) / **c\*_worst=0.9256** (worst),
so the ordering is **c\*_central < 0.9213 < c\*_worst** → clearing the strict identity bar buys
central-500 but is **not** sufficient for worst-case 500 (shortfall **0.0043** vs lawine #336's +0.031
retrain budget; total lift **0.0353**, just past it), and **even at c\*_worst supply must still revive**
(fern #335, φ≥0.255) — coverage alone never reaches 500. A necessary input for sizing the
human-approval-gated #319 retrain coverage target: **central-500 → target 0.9213; worst-500 → target
0.9256.**

## Public evidence used

Banked W&B numbers only (all `wandb-applied-ai-team/gemma-challenge-senpai`): stark #337 `lbuirkpt`
(E[T](c) chain law, anchors central 520.95 / worst 492.87, honest corners 470.35/444.99, cliff
μ=1.16981, μ_tie=1.1076); lawine #330 `hfrscdai` (cov prior 0.8903, identity bar 0.9213); lawine #336
`krroookz` (+0.031 retrain lift); fern #335 `5pos499e` (supply floor φ≥0.255, binding axis = supply).
Official frontier 481.53 TPS (PR #52, `2x9fm2zx`).
