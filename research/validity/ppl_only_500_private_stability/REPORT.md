<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #347 — PPL-only 500 private-stability: does PUBLIC c\*=0.9089 survive private?

**PRIMARY `private_stability_self_test_passes` = True** (all 13 checks)
**TEST `ppl_only_central_500_is_private_stable` = False** (does public c\*=0.9089 clear 500 private?)
**TEST `coverage_lift_for_private_500` = +0.0366** (rho_priv_e3) / **+0.0319** (deployed gap) — coverage delta 0.8903 → private-500
**W&B `0hxhs0kd`** (group `ppl-only-500-private-stability`) · LOCAL read-only analytic, 0 GPU, 0 TPS · BASELINE **481.53 UNCHANGED**

> **Verdict: NO — the PUBLIC central-500 target is private-UNSTABLE by construction.** At the public-500
> operating point realized private TPS = `500 · ρ_priv` with `ρ_priv < 1` always, so it can **never**
> stay ≥ 500. With lawine #300's `ρ_priv_e3 = 0.9421` (deployed-effective) realized private = **471.06**;
> with the organizer-measured deployed gap `g = 0.9571` it is **478.53** — both < 500 (worst/raw push to
> ~390–396). Restoring private-500 needs over-provisioned coverage `c*_private ≈ 0.922–0.927`
> (**+0.0319 to +0.0366**), which **EXCEEDS** lawine #336's **+0.031** budget under *every* private model
> — even the lightest (the 4.3% deployed gap). The cheap public-500 isocline fern #341 found is
> **PUBLIC-only**. This is the **ρ-axis closure** of fern #341's dropped 3rd axis.

## Public→private imports (deliverable 1 — re-derive nothing)

| import | value | source |
|---|---|---|
| deployed public→private | 481.53 → 460.85 (Δ **4.29%**, `g = 0.95705`) | lawine #300 `8t5q6sr0` (organizer-verified) |
| `ρ_priv_e3` (deployed-effective) | **0.9421** (deep fidelity, a₁ tree-recovered) | lawine #300 |
| `ρ_priv_e3_raw` (CI lower) | **0.7797** (no tree-recovery) | lawine #300 |
| `ρ_worst_xdataset` | **0.7923** (EAGLE-3 lit, a₁ not credited) | fern #318 `xe8ff7hq` / arXiv:2503.01840 T1 |
| ρ robustness gate (full ceiling) | **0.8038** = 500 / 622.08 | fern #335 / #310 |
| c\*_central / c\*_worst (PUBLIC-500) | **0.9089** / **0.9256** | wirbel #343 `kklof4wr` |
| retrain budget | **+0.031** | lawine #336 `krroookz` |

`ρ_priv_e3` and the deployed 4.3% gap are **two estimates of the SAME** private/public haircut (lawine
#300 derived `ρ_priv_e3` from the per-position collapse that *also* produces the deployed gap) — they
**bracket** the realized ratio at **[0.9421, 0.9571]**; they are **not** multiplied.

## Private-realized envelope @ the PUBLIC central-500 target (deliverable 2 — TEST)

`private(c) = envelope_central(c) · ρ_priv`. At `c*_central = 0.9089` public TPS = 500 by construction:

| ratio | ρ | realized private | clears 500? | regime |
|---|---|---|---|---|
| `ρ_priv_e3` (headline) | 0.9421 | **471.06** | **No** | realistic |
| `deployed_gap` | 0.9571 | **478.53** | **No** | realistic |
| `ρ_worst_xdataset` | 0.7923 | 396.14 | No | downside |
| `ρ_priv_e3_raw` | 0.7797 | 389.86 | No | downside |

**Every** ratio misses 500. The public-500 target leaves **zero private margin** — staying ≥ 500 private
would need `ρ ≥ 1.0`, impossible. `ppl_only_central_500_is_private_stable = False`.
*(The "apply both multiplied" value 500·0.9421·0.9571 = 450.8 double-counts the one haircut and is
rejected — shown only for completeness.)*

## Solve the private-500 coverage target (deliverable 3 — TEST)

Solve `envelope_central(c) · ρ = 500`:

| ratio | c\*_private | lift from 0.8903 | within +0.031? | reachable? |
|---|---|---|---|---|
| `deployed_gap` | 0.92215 | **+0.0319** | **No** (just over) | Yes |
| `ρ_priv_e3` (headline) | 0.92686 | **+0.0366** | **No** | Yes |
| `ρ_worst_xdataset` | 0.97772 | +0.0874 | No | Yes |
| `ρ_priv_e3_raw` | 0.98233 | +0.0920 | No | Yes |

`c*_private` is **reachable** (coverage solution exists, `< 1`) but the lift **exceeds +0.031 under every
model** — even the organizer-measured 4.3% gap lands at +0.0319. `c*_private ≥ c*_central` is structural
(`ρ < 1` ⇒ more coverage). The public-500 (+0.0186) fit the budget; the **private-500 requirement does
not**.

## Worst-corner reconcile (deliverable 4 — pin or refute the identity)

The identity `c*_private == c*_worst` holds **exactly iff** the private ratio `g == worst/central anchor
ratio = 492.87/520.95 = 0.94608`.

- measured `g_dep = 0.95705` ≠ 0.94608 (Δ 0.01097) · `ρ_priv_e3 = 0.94212` ≠ 0.94608 (Δ 0.00396) →
  **REFUTED** (`identity_holds_exactly = False`).
- but 0.94608 sits **between** them, so `c*_worst = 0.9256` is **bracketed** by the two realistic
  `c*_private` (0.9222 deployed / 0.9269 ρ_e3) — a sensible **central private proxy**, not an identity.
- **scale confound:** #343's 492.87 anchor = `honest611(622.08) · ρ_worst(0.7923)` — it mixes the
  honest-ceiling uplift (×1.194) with a worst-ρ tax, while the central anchor 520.95 is the λ-**capped**
  public ceiling. The 0.9461 net lands in the realistic bracket **partly by coincidence**.

**Verdict: REFUTED-BUT-BRACKETED.** The "worst-budget gap" is *not* identically the public→private
robustness margin, but it is a reasonable conservative envelope in the right ballpark.

## ρ-gate closure: necessary, NOT sufficient (deliverable 5)

At the **FULL honest ceiling** (622.08) private = `622.08·ρ`, clears 500 iff `ρ ≥ 0.8038`. Realistic ρ
(0.9421/0.9571 → **586/595** private) **clears** it — that is fern #318's YELLOW *build* verdict. But the
**same** ρ **misses** at the budget-minimal public-500 point, where the implied gate is `ρ ≥ 1.0`. So the
ρ ≥ 0.8038 gate is **necessary** (a full build needs it) but **not sufficient** (the cheap public-500
isocline point still fails private). **This is exactly the ρ-axis fern #341 dropped** — it bites at the
operating point its 2D (φ, Δcov) card never covered.

## Self-test (NaN-clean, deterministic)

All **13** PRIMARY checks pass: (a) 481.53/460.85 round-trips ≤ 1e-6 + Δ = 4.3% · (b) ρ_priv + c\*
targets imported exact · (c) private envelope NaN-clean · (d) `c*_private ≥ c*_central` every model · (e)
worst-corner identity explicit (refuted + bracketed + 492.87 reconstruction) · (f) public-500 NOT
private-stable (all miss) · (g) private-500 over budget every realistic model · (h) c\*_private reachable
∈ (0,1) · (i) ρ-gate necessary-not-sufficient · (j) imports round to display · (k) NaN-clean · (l)
private < 500 at public target · (m) realistic band entirely < 500.

## Honest caveats (carried in the artifact)

1. **`ρ_priv_e3` is a modeled point estimate** (lawine #300 deployed-effective), not a measured
   fusion-head private tax. Carry its CI bracket **[0.7797 raw, 0.9421 deployed-effective]**; the worst
   cross-dataset 0.7923 is a literature lower bound. Under the downside bracket private-500 needs
   **+0.087–0.092** (far over budget).
2. **Private eval is held-out.** The deployed 4.3% gap is an aggregate; it may **not** be
   coverage-uniform across depths/prompts, so a scalar haircut on `env(c)` is a first-order
   approximation (lawine #300's per-position `c_deep = 0.97135` on j≥2, a₁ held, is the finer structure).
3. **ρ_priv_e3 and the deployed gap are not independent** and are **not** multiplied — they bracket the
   one ratio at [0.9421, 0.9571]; the 450.8 "both-multiplied" is a double-count, shown to reject it.
4. **Scale confound** in #343's "worst" anchor (492.87 = 622.08 × 0.7923) — its agreement with the
   realistic private bracket is partly coincidental.
5. **A definitive private number needs the gated private read** (organizer-side). This is a CPU envelope
   projection. NOT a launch / build / served-file change / HF Job / submission / open2.

## Greedy/PPL-safety certificate

`analysis_only = True`. No served-file change, no emitted-token change, no HF Job, no submission, NOT a
launch, NOT a build. BASELINE **481.53 TPS unchanged**; this leg adds **0 TPS**.

## Hand-off

The PPL-only **PUBLIC** central-500 target (wirbel #343 c\*=0.9089, +0.0186, within budget) does **NOT**
survive private: realized private = 500·ρ = **471.06** (ρ_e3) / **478.53** (deployed) — both < 500, zero
private margin. Restoring private-500 needs `c*_private ≈ 0.922–0.927` (**+0.0319 to +0.0366**), **OVER**
lawine #336's +0.031 under every private model. `c*_worst = 0.9256` is a central private proxy but **not**
an exact identity (REFUTED-BUT-BRACKETED). The ρ ≥ 0.8038 gate clears at the full ceiling (586/595, fern
#318 YELLOW) but is **not sufficient** at the cheap public-500 point. **ρ-axis closure of fern #341: the
demand-led public-500 isocline is PUBLIC-only; private-stability costs more coverage than the retrain
budget allows.**

## Public evidence used

Banked W&B numbers only (all `wandb-applied-ai-team/gemma-challenge-senpai`): wirbel #343 `kklof4wr`
(PPL-only envelope + c\*_central 0.9089 / c\*_worst 0.9256 / prior 0.8903 / identity bar 0.9213 / anchors
520.95/492.87); lawine #300 `8t5q6sr0` (ρ_priv_e3 0.9421 / 0.7797 raw; deployed 481.53→460.85 Δ4.3%;
private_bar 500); fern #318 `xe8ff7hq` (honest611 622.08, ρ_worst 0.7923, full-build private 622.08·ρ,
worst 492.87, YELLOW); fern #335/#310 (ρ ≥ 0.8038 gate = 500/622.08); fern #341 `o4rzy1k6` (demand-led
isocline + dropped ρ axis); lawine #336 `krroookz` (+0.031 retrain budget). Official frontier 481.53 TPS
(PR #52, `2x9fm2zx`); private-verified 460.85.
