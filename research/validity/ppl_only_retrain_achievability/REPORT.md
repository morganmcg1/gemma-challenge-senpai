<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #346 — PPL-only retrain achievability: does a feasible retrain hit the lowered +0.0186 central target?

**0-GPU analytic card. No training, no checkpoint, no model forward, no served-file change, 0 TPS.
Greedy/PPL untouched. BASELINE 481.53 unchanged. W&B run `ifurfu5l`
(`ppl-only-retrain-achievability`).**

## TL;DR — VERDICT: **central-500 = LIKELY-WIN** (`p≈0.99999` tri-indep, `0.9963–1.0000` across shapes) / **worst-500 = LEAN-WIN** (`p≈0.695` tri-indep, `0.641–0.695` across shapes, `0.659` at ρ=+0.5)

wirbel #343 (`kklof4wr`, MERGED) **sized** the PPL-only >500 coverage target: central-500 needs
`c*=0.9089` (**+0.0186** lift from the honest 0.8903 prior), inside lawine #336's **+0.031** reachable
budget; worst-500 needs `c*=0.9256` (**+0.0353**, ~14% past budget). "Within budget" is a *feasibility*
verdict, not an *expected outcome*. This card converts #343's target into an **achievability probability**
by treating each lawine #336 recipe lever's Δcov band as a distribution and convolving the recommended
**soft-KD + reasoning-data** combination (combo central **+0.0385**, support **[0.0175, 0.0595]**), then
asks `P(combo lift ≥ target)` for both targets — re-pointing lawine #339's (`0aq16szh`) machinery from the
strict 0.9213 identity bar (lift +0.031) to the **lower** PPL-only targets.

| quantity | value |
|---|---|
| Combo lift (soft-KD + reasoning-data, 0.70 haircut) | mean **0.0385**, support **[0.0175, 0.0595]** |
| Central-500 target lift / coverage | **+0.0186** / c*=0.9089 (within +0.031 budget) |
| Worst-500 target lift / coverage | **+0.0353** / c*=0.9256 (13.9% past budget) |
| **`p_retrain_clears_ppl_only_central_500`** (TEST, triangular indep) | **0.99999** (range **0.9963–1.0000**) |
| **`p_retrain_clears_ppl_only_worst_500`** (TEST, triangular indep) | **0.6946** (range **0.6406–0.6946**) |
| central / worst at ρ=+0.5 (triangular) | 0.9998 / **0.6586** |
| **`central_500_verdict`** / **`worst_500_verdict`** | **LIKELY-WIN** / **LEAN-WIN** |

## Deliverable 1 — imported bands & targets (re-derive NONE)

Per-lever Δcov bands and the combo, imported EXACT from lawine #336 (`krroookz`, drift ≤1e-6 vs on-disk):
soft-KD +0.030 [0.015,0.045], reasoning-data +0.025 [0.010,0.040], deeper-head +0.012 [0.005,0.020], TTT
+0.002 [0.000,0.005]; combo soft-KD+data central **+0.0385**, band **[+0.0175,+0.0595]** under the 0.70
non-additivity haircut. Targets imported EXACT from wirbel #343 (`kklof4wr`, drift ≤1e-9 vs on-disk):
`coverage_lift_for_ppl_only_central_500 = 0.01863633` (= c* − 0.8903, rounds to +0.0186),
`_worst_500 = 0.03530365` (rounds to +0.0353); c*_central=0.9089363, c*_worst=0.9256036 (stark #340 map).
Honest prior 0.8903 (eval 100% reasoning/STEM): lawine #330 (`hfrscdai`).

## Deliverable 2 — achievability distribution (shape × correlation)

Each lever band → a distribution. The PR default is **triangular** over [lo, central, hi] (headline);
**uniform** over [lo, hi] is the shape sensitivity; **normal ±2σ** is carried as the #339 bridge. Both
training-lever bands are symmetric (central = midpoint), so the **independent** triangular combo is exact
(sum of two equal-half-width symmetric triangulars = **Irwin-Hall(4)**: L = 0.0175 + 0.0105·IH4), the
**independent** uniform combo is exact (two equal-width uniforms → Triangular[0.0175, 0.0385, 0.0595]), and
the normal combo is exact for any ρ via σ = haircut·√(σ²·(2+2ρ)). The +0.5-correlated triangular/uniform
cells (no elementary closed form) come from a **deterministic Gaussian-copula Monte-Carlo** (4M samples,
seed 20260615), matching #339's z-mixing and cross-checked against the closed forms (≤5e-3).

`P(combo lift ≥ target)`:

| shape | corr | σ | **central (+0.0186)** | **worst (+0.0353)** |
|---|---|---|---|---|
| **triangular** (default) | independent | 0.00606 | **0.99999** | **0.6946** |
| triangular | ρ=+0.5 | 0.00742 | 0.9998 | 0.6586 |
| triangular | comonotonic | 0.00857 | 0.9986 | 0.6408 |
| uniform (sensitivity) | independent | 0.00857 | 0.9985 | 0.6406 |
| uniform | ρ=+0.5 | 0.01050 | 0.9915 | 0.6008 |
| uniform | comonotonic | 0.01212 | 0.9730 | 0.5764 |
| normal (#339 bridge) | independent | 0.00742 | 0.9963 | 0.6666 |
| normal | ρ=+0.5 | 0.00909 | 0.9855 | 0.6374 |
| normal | comonotonic | 0.01050 | 0.9709 | 0.6197 |

**Robustness:** across all **9** cells, central ≥ **0.9709** and worst ≥ **0.5764** — the verdict
(central LIKELY-WIN, worst LEAN-WIN) holds under every shape and every correlation, including the
worst-case comonotonic uniform.

## Deliverable 3 — the decision number

- **Central-500 is a LIKELY-WIN.** The target +0.0186 sits only **+0.0011 above the combo's hard support
  floor 0.0175** and far below the combo central +0.0385, so it is a near-certain clear under every shape
  (0.996–1.000). Read the 0.99999 headline as "near-certain", not literal certainty — the exact decimal
  is shape-driven (triangular density → 0 at the floor); the load-bearing fact is the **range floor 0.971**.
- **Worst-500 is a LEAN-WIN (coin-flip-OR-BETTER, not a slam-dunk).** The target +0.0353 sits just below
  the combo central +0.0385 and ~14% past #336's +0.031 budget, so achievability is **0.64–0.69**
  independent / **0.60–0.66** at ρ=+0.5 — above 0.5 in every cell, but materially short of a confident win.
- **Continuity:** re-pointed to #339's strict identity lift (+0.031035), the normal engine reproduces
  #339's **0.843** (computed 0.8427) — the lowered PPL-only central target buys a large achievability jump
  (0.843 → ~1.0) precisely because #124's PPL-only world drops the strict identity bar.

## Deliverable 4 — honest caveats

- The per-lever Δcov bands are **workload-dependent literature-grounded point estimates** (ranges carried,
  not false precision); the achievability is conditional on those priors being calibrated, not a guarantee.
- Even a lit-central fully-trained head (~0.913) clears the central target, but the official eval is
  **100% reasoning/STEM** (lawine #330) — the hard distribution; a head hitting 0.913 on a generic mix may
  fall short here.
- Worst-500 achievability rises **above 0.9 only if the capacity levers are added** (full-4-lever
  optimistic bound: deeper-head + TTT on the combo → worst ≈ 0.997) — reported as an upper bound, not the
  headline 2-lever recipe.
- This is an **achievability SCREEN**, not a build recommendation. A definitive number needs the
  **measured** post-retrain coverage read (human-gated, #319/#322).

## Greedy-safety / scope

The retrain TARGET is greedy-IDENTICAL by construction (EAGLE-3 drafter only *proposes*; emission = verify
argmax) and **PPL-pinned ≤ 2.42**. In the #124 PPL-only world the strict greedy-identity bar is
intentionally dropped (accepted risk) while PPL stays the guardrail. Coverage is the SPEED/acceptance axis
(E[T]), not the validity axis. `card_is_cpu_analytic=True`, `retrain_run_is_human_gated=True`. 0 TPS; no
served-file change; no HF Job; not a launch. BASELINE 481.53 unchanged.

## Self-test — **13/13 PASS**, NaN-clean, deterministic, peak mem ~404 MiB

(a) #336 recipe bands round-trip ≤1e-6 (on-disk drift guard pr336+pr330); (b) wirbel #343 targets imported
exact (≤1e-9 on-disk, round to +0.0186/+0.0353, = c*−0.8903); support [0.0175,0.0595] and target bracketing
(central<budget<worst); σ ordering triangular<normal<uniform and comonotonic-normal σ=0.0105 (=#336 band);
correlation widens σ; both convolutions NaN-clean ∈[0,1]; **monotone P(≥+0.0186) ≥ P(≥+0.0353)** every
cell; independent vs ρ=+0.5 both reported and ordered; headline triangular is closed-form; MC matches
closed-form ≤5e-3; normal reproduces #339's 0.843 at the identity lift; central LIKELY-WIN / worst LEAN-WIN
every shape; full-4-lever optimistic ≥ 2-lever combo on worst.

## Public evidence used (internal banked anchors)

- **wirbel #343** (`kklof4wr`, MERGED `research/validity/ppl_only_gate_500_envelope/`): the PPL-only >500
  coverage targets c*_central=0.9089 (+0.0186) / c*_worst=0.9256 (+0.0353) and the +0.031-budget framing
  this card prices. Cross-checked live against its on-disk JSON (drift ≤1e-9). *Extending* its target sizing
  into an achievability distribution.
- **lawine #336** (`krroookz`, MERGED): the per-lever Δcov bands + 0.70 haircut + combo central +0.0385
  band [+0.0175,+0.0595] + the +0.031 reachable budget. Drift ≤1e-6.
- **lawine #330** (`hfrscdai`, MERGED): the honest 0.8903 top-4 ROOT coverage prior and the 100%
  reasoning/STEM eval misnomer.
- **lawine #339** (`0aq16szh`, MERGED): the independent-vs-+0.5 correlation convolution method this card
  re-points; reproduced exactly at the identity lift (0.843) as a continuity anchor.
- **stark #340** (`jwv1vbug`): the c* speed-envelope map backing #343's targets.

Non-collision: this card owns ACHIEVABILITY (P a retrain delivers the lift). Orthogonal to wirbel #343
(target SIZING — cited), the ρ-axis private-stability card (whether a realized 500 survives private),
stark #345 (non-EAGLE-3 methods), and denken #344 (speed).

## Reproduce

```bash
cd target/ && .venv/bin/python \
    research/validity/ppl_only_retrain_achievability/ppl_only_retrain_achievability.py \
    --self-test --wandb_group ppl-only-retrain-achievability \
    --wandb_name lawine/ppl-only-retrain-achievability
```

Primary `retrain_achievability_self_test_passes = True` (13/13). Test
`p_retrain_clears_ppl_only_central_500 = 0.99999` (triangular indep; 0.9963–1.0000 across shapes),
`p_retrain_clears_ppl_only_worst_500 = 0.6946` (0.6406–0.6946 across shapes). Report
`central_500_verdict = LIKELY-WIN`, `worst_500_verdict = LEAN-WIN`. W&B run `ifurfu5l`.
