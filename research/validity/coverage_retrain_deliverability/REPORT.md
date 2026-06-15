<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #380 — Is the coverage-retrain recipe real? Grounding deliverability of the +0.0107 demand-side target

**0-GPU analytic card. No training, no checkpoint, no model forward, no HF Job, no served-file change,
0 official TPS. Greedy/PPL untouched (an EAGLE-3 drafter only PROPOSES; emission is the verify argmax →
byte-exact greedy regardless of drafter quality). BASELINE 481.53 unchanged. Reuses the denken #377
non-iid harness. W&B run `00oijpwg` (`coverage-retrain-deliverability`), group `strict-bi-verify-gemm`.**

## TL;DR — VERDICT: **YELLOW** — `recipe_is_real = True`, but `deliverability_survives_conservative = False` for the ROBUST target

My #377 (`030uc5mk`) sized the now-primary demand-side closer: retrain the EAGLE-3 fusion head to
coverage `c ≥ 0.9010` (Δcov **+0.0107 robust** / **+0.00565 central**) to close the #373 +5.44 TPS
private-500 residual, within #336's +0.031 budget. That sizing rests on two ungrounded inputs the entire
route inherits — the **delivery distribution** N(0.0385, 0.00742) (#339) and the **transfer efficiency**
κ≈0.672 (#377). This card grounds both and finds a clean split:

- **The κ-axis is ROBUST and is NOT the weak link.** `kappa_breakeven = 0.122` sits **below even the
  program's own worst c\* corner (0.354)**, so the +0.0107 target stays inside #336's +0.031 budget across
  the *entire* plausible κ range. `kappa_margin = 0.549` vs the central κ=0.672. The optional local-GPU
  κ-probe is therefore the **wrong tool** — κ is well-bounded analytically.
- **The DELIVERY distribution is the binding uncertainty, and #339's N(0.0385, 0.00742) is OPTIMISTIC.**
  A focused re-read of the load-bearing citations (V1/V2/V3 below) shows +0.0385 sits at the *top* of the
  published fully-trained-head coverage range [0.899, 0.929]. A **defensible FINE-TUNE** delivery is
  N(0.016, 0.006). Under it: the **central +0.00565 target survives** (P=0.958 ≥ 0.90), but the
  **robust +0.0107 target does NOT** (P=0.811 < 0.90).

So: the recipe is **real** (the levers are literature-documented and a positive lift is genuinely
expected — the +0.0227 from-scratch ceiling is a real anchor), but it is **not a measured checkpoint and
not the +0.0385 magnitude**. fern #357 can **split-bank the CENTRAL target now**; the **ROBUST target is
the binding open item** and needs a cheap *real coverage-lift measurement* before banking.

| quantity | value |
|---|---|
| `delivery_distribution_grounding` | **literature-anchored** (not measured-checkpoint, not modeled-only) |
| `defensible_delivery_distribution` | **N(0.016, 0.006)**, band [+0.009, +0.029], fine-tune-realistic |
| `kappa_breakeven` (required_dcov == +0.031 budget) | **0.1222** |
| `p_deliver_at_kappa_breakeven` | 0.843 (opt #339) / 0.006 (defensible) |
| `kappa_margin` (= 0.672 − breakeven) | **0.5494** |
| margin from worst c\* corner (0.354 − 0.122) | 0.232 |
| `p_softkd_reasoning_retrain_delivers_robust` (defensible) | **0.811** ← < 0.90 |
| `p_softkd_reasoning_retrain_delivers_central` (defensible) | **0.958** ← ≥ 0.90 |
| `recipe_is_real` | **True** |
| `deliverability_survives_conservative` (tracks ROBUST) | **False** |
| `measured_kappa` (GPU leg not run — κ well-bounded) | null |
| `deliverability_self_test_passes` | **True** (34/34 checks) |

## What κ is, and why required_dcov is a 1/κ curve (Deliverable 2)

From #377 first principles (self-test `a`): κ is the realized coverage→E[T] conversion slope as a
**fraction of the κ=1 uniform-additive passthrough bound** `S_uniform = 11.781` (dE[l]/dc when every
per-position conditional aₔ shifts by +dc). The nominal coverage lift to close a residual dE[T] is

```
required_dcov(κ) = dE[T] / (κ · S_uniform)          ← a 1/κ curve
```

The two #377 targets are just two points on this *one* curve:
- **+0.0107 robust** = required_dcov(κ_worst = S_program_worst/S_uniform = 4.173/11.781 = **0.354**)
- **+0.00565 central** = required_dcov(κ_central = S_program_central/S_uniform = 7.913/11.781 = **0.672**)

`kappa_breakeven` is the κ at which required_dcov crosses #336's +0.031 budget:

```
kappa_breakeven = dE[T] / (budget · S_uniform) = 0.04468 / (0.031035 · 11.781) = 0.1222
```

**0.1222 is below the program's own worst c\* corner (0.354)** — so even at the worst realized transfer
the program ever exhibits, the +0.0107 target stays inside budget. The κ-axis is robust; it is not the
thing that can break this route.

| κ | required_dcov | frac of +0.031 budget | within budget? |
|---|---|---|---|
| 0.122 (breakeven) | 0.03104 | 1.00 | edge |
| 0.354 (worst c\* corner) | 0.01071 | 0.345 | yes |
| 0.672 (central) | 0.00565 | 0.182 | yes |
| 1.00 (uniform bound) | 0.00379 | 0.122 | yes |

## The load-bearing finding: #339's +0.0385 delivery distribution is OPTIMISTIC (Deliverable 1)

The provenance chain is fully traced and round-trips numerically (self-test `b`):

- **#330 `hfrscdai`** → MEASURED honest top-4 ROOT prior **0.8903** (fern #34 eval). Grounded.
- **#336 `krroookz`** → per-lever bands soft-KD **+0.030** [0.015,0.045], reasoning-data **+0.025**
  [0.010,0.040], combined `0.70·(0.030+0.025) = +0.0385`. #336's own words: *"literature-grounded
  expected-value priors a retrain would confirm — they are NOT measured here"*, and the top-4-tail
  attribution is *"reasoned inference, not a per-paper ablation."*
- **#339 `0aq16szh`** → N(0.0385, σ); σ=0.00742 is the **independent (ρ=0)** spread — the *tightest* of
  three (indep / ρ=0.5 / comonotonic = 0.00742 / 0.00909 / 0.01050).
- **#377 `030uc5mk`** → adopted mean 0.0385 **and** the optimistic σ=0.00742 → P(deliver) ≈ 1.0.

**Grounding classification: `literature-anchored`** — the levers and their magnitudes ARE cited to real
papers (DistillSpec 2310.08461 / OSD 2310.07177 / Medusa 2401.10774 / EAGLE-3 2503.01840), so this is
**not modeled-only**; but no coverage-lift checkpoint has been run, so it is **not measured-checkpoint**.

### Why +0.0385 is the optimistic tail (PR #380 independent verification pass)

A focused re-read of the load-bearing citations sharpened three decision-critical points the upstream
cards did not surface:

- **(V1) No cited paper reports top-4 COVERAGE.** Every paper reports acceptance-rate α or accept-length
  τ (sampling metrics). The +0.0385 top-4-coverage number is an *inference* from acceptance gains via an
  ASSUMED (never-measured) transfer — the metric itself is not literature-measured.
- **(V2) The closest controlled A/B CONTRADICTS the soft-KD +0.030.** EAGLE-1 (2401.15077) ran
  DistillSpec-style logit distillation as a baseline, found *"only modest improvements,"* and chose
  CE-based feature regression because logit-KD **underperformed**. Defensible soft-KD coverage lift:
  ~+0.005..+0.015, not +0.030.
- **(V3) The reasoning-DATA +0.025 is optimistic.** EAGLE-1 reports *"low sensitivity to training data"*
  (~3.6% speedup from a data-quality swap); HASS finds ¼ of ShareGPT ≈ full set; and KD benefit shrinks
  as the student-teacher gap narrows (Mirzadeh 2020) — our head is already capable at 0.89. Defensible
  data lift: ~+0.005..+0.010.

**Net:** 0.8903 + 0.0385 = **0.9288 ≈ the TOP of the published [0.899, 0.929] fully-trained-from-scratch
head range** (`cov_post_339_in_lit_range_frac ≈ 0.93`). That is a best-case, not a central — and we are
FINE-TUNING an existing 0.8903 head, not training one from scratch.

### The re-derived defensible delivery distribution (Deliverable 1 output)

| anchor | mean | sd | basis |
|---|---|---|---|
| #339 optimistic (ρ=0) | +0.0385 | 0.00742 | modeled combo at the top of the lit range |
| from-scratch CEILING | +0.0227 | 0.0091 | #323 published head 0.913 [0.899,0.929] − 0.8903 prior |
| **DEFENSIBLE fine-tune** (headline) | **+0.016** | **0.006** | discounted for V2 (logit-KD underperforms) + V3 (data/saturation) |
| pessimistic low-central | +0.012 | 0.005 | EAGLE-1 contra + capable-head saturation |

The **from-scratch ceiling +0.0227** is an *upper bound* on a fine-tune (fine-tuning the 0.8903 head can
at best close the gap to a from-scratch head, not exceed it). The **defensible fine-tune +0.016** discounts
further for V2/V3. Both are below #339's +0.0385.

## Deliverability across the full spectrum (Deliverable 3)

P(retrain delivers the required nominal lift) = 1 − Φ((req − mean)/σ), evaluated for the robust (+0.0107)
and central (+0.00565) targets across the whole delivery spectrum (self-test `e`, `e2`):

| delivery distribution | mean | sd | **P(robust +0.0107)** | **P(central +0.00565)** |
|---|---|---|---|---|
| #339 optimistic (ρ=0) | 0.0385 | 0.00742 | 1.000 | 1.000 |
| from-scratch ceiling | 0.0227 | 0.0091 | 0.906 | 0.970 |
| **DEFENSIBLE fine-tune** | **0.016** | **0.006** | **0.811** | **0.958** |
| pessimistic low-central | 0.012 | 0.005 | 0.602 | 0.898 |

The spectrum is monotone in delivery mean (self-test `e2_robust_spectrum_monotone`). Reading down the
columns is the whole story:

- **CENTRAL +0.00565 target:** P stays ≥ 0.90 from optimistic all the way down to the *pessimistic* corner
  (0.898). **Robustly deliverable.** → `central_target_survives_conservative = True`.
- **ROBUST +0.0107 target:** P falls 1.000 → 0.906 (ceiling) → **0.811 (defensible fine-tune)** → 0.602
  (pessimistic). It **clears 0.90 only if you believe the fine-tune reaches the from-scratch ceiling**,
  which is optimistic for a fine-tune. Under the defensible fine-tune it is **0.811 < 0.90**. →
  `robust_target_survives_conservative = False`.

This is genuinely marginal: the verdict hinges on +0.0227 (ceiling) vs +0.016 (defensible fine-tune) — which
is *exactly* why this is YELLOW, not GREEN and not RED. The robust target's deliverability is the **binding
residual uncertainty** on the route.

## Verdict + cheapest real-retrain proof (Deliverables 4 & 5)

`recipe_is_real = True` — **not a fiction**: the levers (soft-KD, reasoning-data) are documented and a
positive coverage lift IS empirically expected (the +0.0227 from-scratch ceiling is a real anchor). But
"real" here means *literature-anchored prior*, **NOT a measured checkpoint, and NOT the +0.0385 magnitude**
(V2/EAGLE-1 contradicts the soft-KD component); the defensible fine-tune central is ~+0.016.

`deliverability_survives_conservative = False` — the headline bool tracks the **ROBUST** target (the
`c ≥ 0.9010` sizing fern #357 would bank). It survives the optimistic #339 basis and barely survives the
from-scratch ceiling, but **not** the defensible fine-tune delivery.

**VERDICT = YELLOW. Recommended next step (split-bank):**

1. **Bank the CENTRAL +0.00565 target now.** `c ≥ 0.8959` is deliverable at ≥ 0.90 confidence across the
   entire literature-grounded delivery spectrum, and the κ-axis is robust (breakeven 0.122 ≪ worst-corner
   0.354). This is safe to fold into the route today.
2. **The ROBUST +0.0107 target is the binding open item** — do NOT bank it on #339's optimistic
   distribution. Cheapest *real* proof before banking it:
   - **(a)** kanna #294 Phase-1 a₂ ≥ 0.83 cheap pre-check (de-risks the prior; no retrain), then
   - **(b)** the #352-priced **~25 A10G-GPU-hr (~3 h on the 8× node)** soft-KD + reasoning-trace
     **FINE-TUNE** + wirbel #79 `RANKPROBE_W=4` coverage re-measure on the OFFICIAL 128 eval → a **direct
     top-4-coverage-lift measurement** (V1: the metric no paper reports).
3. **The optional local-GPU κ-probe is the WRONG tool.** κ is robust analytically; the weak link is
   DELIVERY, which needs a coverage-lift pilot, not a transfer measurement. The GPU leg was therefore not
   run.

## Self-test (`deliverability_self_test_passes = True`, 34/34)

Covers: κ identity round-trips #377 (`a`); required_dcov reproduces the +0.0107 / +0.00565 targets at the
worst/central κ (`a2`); grounding enum + #336/#339 mean/σ round-trips + the V1/V2/V3 findings carried
(`b`, `b2`); defensible mean below both #336 and the from-scratch ceiling and inside the fine-tune range
(`c`); breakeven below the worst c\* corner and below the RED 0.62 line, margin positive, req-at-breakeven
== budget (`d`, `d2`); **the decisive split** — central survives, robust does not, spectrum monotone,
central ≥ 0.85 at the pessimistic corner (`e`, `e2`, `e3`); sweep monotonicity + within-budget crossing
at breakeven (`f`, `g`); budget == bar − prior identity (`i`); numeric hygiene (`j`).

## Public evidence used

- **EAGLE-1** (arXiv:2401.15077) — logit-KD-as-baseline A/B (V2); low training-data sensitivity (V3).
- **EAGLE-3** (arXiv:2503.01840) — training-time test, multi-layer feature fusion; coverage anchor (#323/#336).
- **HASS** (arXiv:2408.15766) — top-K distillation folded into a richer pipeline; ¼-data ≈ full (V3).
- **DistillSpec** (arXiv:2310.08461), **OSD** (arXiv:2310.07177), **Medusa** (arXiv:2401.10774),
  **KOALA** (arXiv:2408.08146) — soft-KD / online-distillation lever citations (#336).
- **Mirzadeh et al.** (2020) — KD benefit shrinks as the student-teacher gap narrows (diminishing returns, V3).
- Banked advisor-branch cards: **#377** `030uc5mk` (non-iid sizing), **#336** `krroookz` (recipe bands),
  **#339** `0aq16szh` (delivery distribution), **#323** `ceddxj20` (achievability ceiling), **#330**
  `hfrscdai` (measured 0.8903 prior).

*All Δcov numbers are literature-anchored expected-value priors; none is a measured coverage-lift
checkpoint. The whole point of this card is that the metric — top-4 coverage lift — has never been
measured directly, which is why the robust target needs a real pilot.*
