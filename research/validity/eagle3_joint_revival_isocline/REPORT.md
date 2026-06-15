<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Joint compliant-500 revival ISOCLINE (PR #341, fern)

CPU-only analytic over banked W&B numbers. 0 GPU, 0 HF quota, no served-file change, no build, no
launch. BASELINE stays 481.53; this card adds 0 TPS. W&B run `o4rzy1k6`
(`wandb-applied-ai-team/gemma-challenge-senpai`, group `eagle3-joint-revival-isocline`).

## What it does

Turns fern #335's binary DEAD-at-the-corner verdict into a quantified margin. Maps the continuous
locus of `(phi_supply, Delta_cov_demand)` pairs where the compliant envelope = 500, with

    envelope(phi, Delta_cov) = ceiling(phi) * E[T](0.8903 + Delta_cov) / E[T](0.9213)
      ceiling(phi)  = 469.68 + phi*(520.95 - 469.68)        # denken #327 linear BW-gap law
      E[T](c)       = 1 + sum_{d=1..7} c^d                  # stark #337 chain law

## Headline

| metric | value |
|---|---|
| `joint_revival_isocline_self_test_passes` (PRIMARY) | **True** (13/13) |
| `min_joint_effort_to_500` (TEST, normalized distance) | **1.547** |
| `dDeltacov_dphi_at_closest` (REPORT, substitution) | **−0.0320** |
| closest feasible point | phi=**0.100**, Delta_cov=**0.0464** (coverage 0.9367) |
| isocline at demand break-even Delta_cov=0.031 | phi=**0.5913** |
| cheaper axis | **demand** |
| feasibility verdict | **REACHABLE-BUT-OPTIMISTIC** |

## LOAD-BEARING CORRECTION to the PR's break-even anchor

The PR asks to validate `envelope(0.255, 0.031) ≈ 500` and have the isocline "pass through
(0.255, 0.031)". **Under the PR's own model this is false:**

    envelope(0.255, 0.031) = ceiling(0.255) * E[T](0.9213)/E[T](0.9213) = 482.76    (NOT 500)

482.76 is exactly fern #335's banked `ceiling_at_breakeven_B_edge`. The two anchors that *do* pin
the model — `envelope(0,0)=424` and `envelope(1,0.031)=520.95` — over-determine it, forcing
`envelope(0.255,0.031)=482.76`.

**Root cause: a three-axis conflation.** #335's "rho=0.8038 internal 500" is a point on the
*private-tax rho* axis, not on the `(phi, Delta_cov)` plane:

    #335 500 = min(HONEST_PUBLIC_611 * rho, LAMBDA_CEIL) = min(622.08*0.8038, 520.95) = 500

It lives at the **full-B ceiling (phi=1 → 520.95)** with demand clear and rho at its break-even
0.8038. The supply break-even `phi*=0.255` is only the *discrete* C→B threshold; in the continuous
linear law the ceiling there is just 482.76. The **true** 500-isocline at Delta_cov=0.031 passes
through **phi=0.5913**, not 0.255. The self-test validates the corrected round-trips and documents
the discrepancy rather than asserting a value the model cannot produce.

## The isocline (envelope = 500)

| Delta_cov | coverage | phi needed | feasible? |
|---|---|---|---|
| 0.0000 | 0.8903 | +1.642 | no — supply alone can't reach 500 even at full recovery |
| 0.0150 | 0.9053 | +1.124 | no — phi>1 |
| 0.0310 | 0.9213 | +0.591 | yes |
| 0.0450 | 0.9353 | +0.145 | yes |
| 0.0600 | 0.9503 | −0.316 | no — demand alone overshoots 500 |

Feasible segment runs from `(phi=1, Delta_cov=0.0187)` [supply maxed, least demand] down to
`(phi=0, Delta_cov=0.0497)` [demand alone].

## Closest feasible point + substitution

Minimum-effort 500 crossing (axes normalized by their break-evens, phi/0.255, Delta_cov/0.031):
**phi=0.100 (0.39 supply break-even units) + Delta_cov=0.0464 (1.50 demand break-even units)**,
normalized distance **1.547**. Both axes move, but **demand carries it** — supply spends *less* than
its own break-even while demand spends ~1.5× its break-even. The pure-demand endpoint
`(phi=0, Delta_cov=0.0497)` is only ~3% farther, so the supply assist barely helps: `phi*=0.255` buys
only ~13 TPS of ceiling. Substitution `dDelta_cov/dphi = −0.0320` (normalized −0.26): one supply
break-even unit buys back only 0.26 demand break-even units — a losing trade, confirming demand is
the cheaper axis.

## Feasibility verdict — REACHABLE-BUT-OPTIMISTIC

The cheapest 500 crossing needs Delta_cov=0.0464 — **above** lawine #336's central combination
(0.0385) but inside its optimistic band [0.0175, 0.0595] — plus only phi=0.100 supply recovery
(well **below** the C→B threshold 0.255). So the burden falls almost entirely on demand, near the
top of its plausible range; supply contributes a small assist and does **not** require the full C→B
kernel revival #335's discrete corner implied. Reconciling #335's `p_both=0` at the measured corner:
the measured-dead corner `envelope(0,0)=424` sits **1.547 break-even units** from the live
500-isocline. (424 is the EAGLE-3 lane's own envelope at zero lift — below the 481.53 deployed
fallback; the rho≥0.8038 robustness gate is a separate check on top.)

## Greedy safety

Both axes preserve greedy identity by construction (BI int4 verify is bit-exact; EAGLE-3 emission is
verify-gated). `isocline_card_is_cpu_analytic = True`, no GPU, no served change.

## Reproduce

```bash
cd target/ && uv run python research/validity/eagle3_joint_revival_isocline/eagle3_joint_revival_isocline.py \
  --self-test --wandb_group eagle3-joint-revival-isocline --wandb_name fern/eagle3-joint-revival-isocline
```

Sources (all `wandb-applied-ai-team/gemma-challenge-senpai`): fern #335 `5pos499e`, denken #327
`kcjlr5ny`, lawine #336 `krroookz`, stark #337 `lbuirkpt`.
