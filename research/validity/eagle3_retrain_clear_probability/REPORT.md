<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #339 — Demand-half retrain clear-probability: convolve #336 Δcov bands into P(clears 0.9213)

**0-GPU analytic card. No training, no checkpoint, no model forward, no served-file change, 0 TPS.
Greedy/PPL untouched. BASELINE 481.53 unchanged. W&B run `0aq16szh`
(`eagle3-retrain-clear-probability`).**

## TL;DR — VERDICT: retrain ROI is **JUSTIFIED** (`p_clears_identity_bar_0p9213 = 0.843` independent / `0.794` at ρ=+0.5)

lawine #336 (`krroookz`) sized the demand-half retrain recipe and called it **REACHABLE-MARGINAL** —
*"central +0.0385 clears, low band misses."* That is a **verdict**, not a **probability**: the human's
#319 build/measure decision can't price the retrain ROI from "marginal." This card converts #336's
per-lever Δcov bands into a posterior over post-retrain coverage and reports the decision-grade number:
**if we spend the GPU on the retrain, P(it clears the 0.9213 identity bar) ≈ 0.84 (independent levers),
≈ 0.79 (conservative +0.5 correlation), ≈ 0.76 (worst-case comonotonic = #336's own reported band).**
All three are **≥ 0.5 → coin-flip-or-better → retrain is JUSTIFIED**, and the cheap de-risker before the
human-gated spend is kanna #294's Phase-1 a₂≥0.83 gate.

| quantity | value |
|---|---|
| cov_post mean (= 0.8903 + combo 0.0385) | **0.9288** |
| recipe σ — independent / ρ=+0.5 / comonotonic | 0.00742 / 0.00909 / **0.01050** |
| **`p_clears_identity_bar_0p9213`** (TEST, independent) | **0.843** |
| P(clears 0.9213) — ρ=+0.5 / comonotonic(=#336 band) | 0.794 / **0.762** |
| **`p_clears_speed500_bar`** at c*=0.908 (independent) | **0.997** |
| full-4-lever optimistic upper bound P(identity) | 0.995 |
| **`retrain_roi_verdict`** | **JUSTIFIED** |

## The load-bearing refinement: #336's reported band was the **worst-case (ρ=1) correlation**

#336 built the combo band `[+0.0175, +0.0595]` by **adding the two lever band endpoints** (low+low,
high+high) and applying the 0.70 non-additivity haircut: `0.70·[0.025, 0.085] = [0.0175, 0.0595]`.
Adding endpoints is the **comonotonic (ρ=1)** spread — σ = (0.0595−0.0175)/4 = **0.0105**. Re-doing it as
a proper **convolution** of the two lever bands gives:

- **independent (ρ=0)** σ = 0.70·√(0.0075²+0.0075²) = **0.00742** ← the tightest, base case;
- **ρ=+0.5** σ = 0.70·0.0075·√3 = **0.00909** ← the PR's conservative bound;
- **comonotonic (ρ=1)** σ = 0.70·(0.0075+0.0075) = **0.01050** ← **exactly reproduces #336's band** (self-test 05).

The **mean is 0.0385 in every case** — correlation widens the spread, it does not move the mean
(cov_post mean = 0.9288 > bar 0.9213 always, so P > 0.5 always). Because cov_post mean is **above** the
bar, a **tighter** spread concentrates more mass above the bar → **higher** clear-probability. So the
independent base case (0.843) is strictly more optimistic than #336's implicit ρ=1 band (0.762), and the
realistic answer sits in the **0.76–0.84** band — **all of which clears 0.5**.

## Deliverable 1 — per-lever distributions (band → mean, σ)

Each #336 band `[low, high]` is modeled as a ±2σ (95%) interval → σ = (high−low)/4, truncated-normal
(Δcov ≥ 0). Truncation is negligible (the combo mean sits ~3.7σ above 0).

| lever | central (mean) | band | σ = (hi−lo)/4 | citation |
|---|---|---|---|---|
| soft-KD top-k distill | +0.030 | [0.015, 0.045] | 0.00750 | DistillSpec/OSD/Medusa |
| reasoning-data | +0.025 | [0.010, 0.040] | 0.00750 | EAGLE-3 |
| deeper/wider head | +0.012 | [0.005, 0.020] | 0.00375 | Medusa/EAGLE-3 (directional) |
| on-policy TTT | +0.002 | [0.000, 0.005] | 0.00125 | lawine #316 (TTT lifts depth≥2, not ROOT) |

**Robustness (uniform-over-band alternative):** modeling each lever as Uniform[low,high] (σ=range/√12)
instead of normal gives P(clears identity) = **0.793 independent / 0.677 comonotonic** (MC) — the
verdict is robust to the distribution-shape assumption.

## Deliverable 2 — recipe convolution → posterior over `cov_post`

Recommended recipe = soft-KD + reasoning-data (the #336 combo), `cov_post = 0.8903 + 0.70·(Δ_KD + Δ_data)`.
Mean 0.9288 (all ρ). Percentiles:

| case | σ | cov_post p05 | p50 | p95 |
|---|---|---|---|---|
| independent | 0.00742 | 0.9166 | 0.9288 | 0.9410 |
| ρ=+0.5 | 0.00909 | 0.9138 | 0.9288 | 0.9437 |

The 5th percentile stays **below** the 0.9213 bar in both cases (0.9166 / 0.9138) — i.e. the low tail
still misses, consistent with #336's "low band misses." The probability mass **above** the bar is what
the clear-probability quantifies: **0.843 / 0.794**.

## Deliverable 3 — clear probabilities (PRIMARY decision numbers)

**P(cov_post ≥ 0.9213) — identity bar:** independent **0.843**, ρ=+0.5 **0.794**, comonotonic **0.762**.

**P(cov_post ≥ c*) — speed-envelope bar sweep** (stark #stark-cov-envelope-ordering solves the exact c*;
this brackets it):

| c* | independent | ρ=+0.5 | comonotonic | which bar binds |
|---|---|---|---|---|
| 0.880 | 1.0000 | 1.0000 | 1.0000 | identity |
| 0.900 | 0.9999 | 0.9992 | 0.9970 | identity |
| **0.908** | **0.9974** | **0.9888** | **0.9761** | identity |
| 0.920 | 0.8811 | 0.8325 | 0.7982 | identity |

**Which bar binds:** the **identity bar (0.9213) is the higher threshold than every c* in the sweep**
(max c* = 0.9200 < 0.9213), so **identity binds across the whole sweep** — the speed-envelope coverage
bar is strictly looser than the identity bar at every swept c* (self-test 15). The speed bar would only
become binding if the true c* exceeded 0.9213, which the {0.88…0.92} bracket does not reach.

## Deliverable 4 — full-recipe (all 4 levers) optimistic upper bound

Stacking deeper-head (+0.012) and TTT (+0.002) additively on the combo (no extra haircut — the optimistic
case): cov_post mean 0.9428, P(clears identity) = **0.995** (independent). This is **≥** the 2-lever combo
P (0.843), as expected (self-test 10). **TTT barely moves it:** dropping TTT gives 0.990 → adding it
0.995, a **+0.004** bump (self-test 11) — consistent with #316 (TTT lifts depth≥2 continuation, not the
ROOT token the 0.9213 bar measures).

**Over-optimistic naïve bound (no haircut):** if the two levers stacked additively with no overlap
(combo +0.055 instead of +0.0385), P would be 0.988 — reported only as a ceiling; it double-counts the
shared mmlu_pro drag and is **not** the headline.

## Deliverable 5 — decision framing (retrain ROI)

`p_clears_identity_bar_0p9213 = 0.843` (independent) **≥ 0.5 → JUSTIFIED** (coin-flip-or-better). The
verdict holds across the full correlation range (0.762 at worst-case ρ=1 still ≥ 0.5) and the
distribution-shape robustness check (0.677 uniform-comonotonic still ≥ 0.5). It is **not** a slam-dunk
(not ≥ 0.9), so the disciplined order is: **run kanna #294's Phase-1 a₂≥0.83 cheap pre-check first** — it
de-risks this probability on the existing base before committing the human-gated full retrain spend
(#319). If Phase-1 passes, the ROI here is realized; if it misses, hold.

## Deliverable 6 — greedy-safety

The retrain **target** is **greedy-IDENTICAL by construction**: an EAGLE-3 drafter only *proposes*;
emission is the verify-model **argmax**, so accepted tokens are byte-exact greedy and **PPL is pinned**.
Coverage is the **SPEED/acceptance axis** (E[T]), **not** the validity axis. `clear_prob_card_is_cpu_analytic
= True`, `retrain_run_is_human_gated = True`. 0 TPS; no served-file change; no HF Job; not a launch.

## Hand-off (one sentence)

> *The EAGLE-3 demand-half retrain clears the 0.9213 identity bar with probability **0.843** (independent
> levers) / **0.794** (+0.5-correlated), and the looser speed-envelope bar c*≈0.908 with probability
> **0.997** — so the retrain ROI is **JUSTIFIED**, and the cheap de-risker is kanna #294's Phase-1
> a₂≥0.83 pre-check before the human-gated full spend.*

## Self-test — **18/18 PASS**, NaN-clean, deterministic, peak mem 101.1 MiB

Verifies: convolution preserves the mean (cov_post mean = 0.8903 + 0.0385 = 0.9288); independent σ =
RSS of the scaled lever σ's; correlation widens σ monotonically (indep < ρ=0.5 < comonotonic); **the
comonotonic σ reproduces #336's reported band σ 0.0105 exactly** and the reconstructed band equals
`[0.0175, 0.0595]` (= 0.70 × naïve-sum endpoints); P monotone-decreasing in the threshold; central
clears with P ≥ 0.5 in every ρ case and P orders by ρ; full-recipe P ≥ combo P and TTT barely moves it;
imported constants exact (rounded display + unrounded banked, with min_lift = identity_bar − cov_prior);
NaN-clean probabilities ∈ [0,1]; percentiles ordered; identity binds over the whole c* sweep; the Monte-
Carlo cross-check (numpy, seed-fixed) agrees with the closed form within 5e-3 (MC indep 0.8426 vs
analytic 0.8427); greedy-safety/human-gated flags; and the imported bands/bar/prior match the on-disk
lawine #336 + #330 artifacts (drift ≤ 1e-9 / 1e-6).

## Honesty / caveats

This card **prices** the retrain decision; it does **not** measure coverage. The probability inherits
#336's bands, which are **literature-grounded expected-value priors a retrain would confirm**, not
measurements — so the "0.84" is a probability *conditional on those priors being calibrated*, not a
guarantee. The 0.9213 bar is position-1 **ROOT** coverage: necessary, **not** sufficient — the deployed
deep spine still caps E[T] (#316), the private-tax robustness (fern #325 pass-(a)) is a **separate** gate,
and the supply-half φ-recovery (denken #332) and the joint (φ,Δcov) AND-gate (fern #335 / #fern-joint-
revival-isocline) are orthogonal axes this card does not own. The 0.70 non-additivity haircut is #336's
modeling assumption (kept fixed at its central value), distinct from the lever-uncertainty correlation
swept here. No training, no checkpoint, no model forward; greedy/PPL untouched; BASELINE 481.53 unchanged.

## Public evidence used

- **lawine #336** (`krroookz`, MERGED `research/validity/eagle3_head_coverage_lift_target/`): the
  per-lever Δcov bands, the 0.70 non-additivity haircut, the combo central +0.0385 / band [+0.0175,
  +0.0595], and the REACHABLE-MARGINAL verdict this card converts into a probability. Cross-checked live
  against its on-disk results JSON (drift ≤ 1e-9).
- **lawine #330** (`hfrscdai`, MERGED `research/validity/eagle3_sharegpt_coverage_prior/`): the honest
  top-4 ROOT coverage prior 0.8903 (the official 128 eval is 100% reasoning/STEM) and the −0.031 gap.
  Cross-checked against its on-disk JSON (drift ≤ 1e-6).
- **lawine #316 / #323**: the regime-invariant identity build bar 0.9213 and the TTT depth≥2-not-root
  measurement.
- **stark #325**: the λ=1 ceiling 520.95 and the speed-envelope anchors; the exact c* is stark's parallel
  `#stark-cov-envelope-ordering` card (this card sweeps it).
- **fern #335** (`5pos499e`): the joint compliant-500 AND-gate this demand-half clear-probability feeds.
- **kanna #294** (`j0ss47bv`, MERGED): the Phase-1 a₂≥0.83 cheap gate cited as the de-risker, and the
  EAGLE-3 E[T]=4.69 < 4.9029 demand-miss on the speed axis.

## Reproduce

```bash
cd target/ && .venv/bin/python \
    research/validity/eagle3_retrain_clear_probability/eagle3_retrain_clear_probability.py \
    --self-test --wandb_group eagle3-retrain-clear-probability \
    --wandb_name lawine/eagle3-retrain-clear-probability
```

Primary `retrain_clear_probability_self_test_passes = True` (18/18). Test
`p_clears_identity_bar_0p9213 = 0.843` (independent). Report `p_clears_speed500_bar (c*=0.908) = 0.997`,
`retrain_roi_verdict = JUSTIFIED`. W&B run `0aq16szh`.
