<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# TPS-vs-private-risk exchange rate — how many TPS does each 1pp of accepted draw-risk buy? (PR #240)

**ubel · `ubel/tps-risk-exchange-rate` · W&B `cl6poy6t` · BANK-THE-ANALYSIS (adds 0 TPS, no draw, no launch)**

## The question

The launch-σ lane prices **TPS(λ)** and **private-draw risk(λ)** on *separate* axes. Issue #124's
actual green-light decision is the **trade** between them: *"is the extra speed of a lower-λ build
worth the extra private-draw risk it accepts?"* This leg composes the two banked curves into the
**exchange rate** — a single slope a human can read instead of two unrelated numbers:

```
dTPS/drisk(λ) = (dTPS/dλ) / (drisk/dλ)        # public TPS bought per unit private-draw risk
```

## The two curves (imported verbatim as modules; NOT re-derived)

| curve | law | source | round-trip |
|---|---|---|---|
| **TPS(λ)** | `mu_pub(λ) = 520.953·E[T](λ)/E[T](1) = K_cal·(E[T](λ)/step)·τ` (both_bugs) | ubel #234/#222 `binding_gate` (`izpjgncc`) | 513.557 @ gate, 520.953 @ λ=1 — **resid 0.0** |
| **risk(λ)** | `risk(λ) = 1 − Φ((μ_priv(λ) − 500)/σ_draw)`, assumed + grounded f_priv | kanna #237 `publishfirst_accepted_risk` (`8x7i38jh`) | bit-for-bit at the shared grid — **resid 0.0** |

Both modules are imported and called directly, so the only **new** object is the ratio of their
λ-derivatives (central finite difference, `h=1e-6`). `K_cal=125.268`, `step=1.2182`,
`σ_draw=7.391`, `f_priv=0.9691` (assumed) / `0.9571` (grounded #224).

## The honest finding — the headline CONTRADICTS the PR's directional premise

The PR hypothesis assumes *"a lower-λ build buys more TPS but accepts more private-draw risk"*
(self-test (b) as written: *"TPS monotone ↓ in λ"*). **The banked maps say the OPPOSITE about
TPS.** Along the build-λ axis both curves *improve* as λ rises:

```
dTPS/dλ  > 0   EVERYWHERE   (higher acceptance λ → higher E[T] → higher public TPS)
drisk/dλ < 0   EVERYWHERE   (higher acceptance λ → higher private mean → lower fail-risk)
⇒  dTPS/drisk(λ) < 0   EVERYWHERE                  (co-monotone, dominated axis)
```

TPS and private-clearance are **co-monotone** — a faster build is the *same* as a safer build,
because both want higher λ. There is **no speed-for-risk trade to optimise on this axis**: lowering
λ to "buy speed" actually **loses TPS *and* adds risk**. We record
`pr_premise_tps_decreasing_in_lambda_holds = False` and report the negative slope faithfully rather
than bending the imported maps to manufacture a positive one.

## The deliverable — exchange rate on the shared λ grid

| λ | TPS | risk (assumed) | risk (grounded) | dTPS/dλ | drisk_a/dλ | **TPS/pp (assumed)** | TPS/pp (grounded) |
|---|---|---|---|---|---|---|---|
| **0.9138** (floor, μ_priv=500) | 501.80 | 0.5000 | 0.7999 | +214.1 | −11.41 | **−0.188** | −0.271 |
| 0.9500 | 509.67 | 0.1468 | 0.4224 | +220.8 | −6.78 | **−0.326** | −0.194 |
| **0.9675** (speed gate, #229) | 513.56 | 0.0583 | 0.2394 | +224.2 | −3.49 | **−0.643** | −0.244 |
| **0.9780** (P95 bar, #191) | 515.93 | 0.0297 | 0.1536 | +226.2 | −2.03 | **−1.112** | −0.320 |
| 0.9970 | 520.26 | 0.0069 | 0.0557 | +229.9 | −0.59 | **−3.909** | −0.675 |
| 1.0000 (ceiling) | 520.95 | 0.0053 | 0.0462 | +230.5 | −0.47 | **−4.926** | −0.784 |

- **HEADLINE** `tps_per_pct_risk_at_speed_gate` (TEST, assumed f_priv) = **−0.6431 TPS/pp**
  (grounded **−0.2443**): each +1pp of accepted private-draw risk near the 0.9675 gate comes with a
  TPS **loss**, not a gain. The slope only *steepens* (more negative) toward λ=1, where risk is tiny
  and a pp of risk costs ever more TPS — never positive anywhere on the grid.
- The floor row is a clean sanity check: at μ_priv=500 the risk is exactly **0.5000 = 1−Φ(0)**.
- **Grounded vs assumed:** the grounded risk curve is both *higher* and *steeper* in λ (check (c):
  `|drisk_g/dλ| > |drisk_a/dλ|` at the gate, 9.18 vs 3.49), so a given speed change spans **more**
  risk-pp ⇒ the |TPS bought per pp| is *smaller* under grounded (−0.244 vs −0.643). Both stay
  negative — the dominated-axis verdict is invariant to the #224 f_priv calibration.

## The decision read (the words) — the 0.9780 bar → 0.9675 gate secant

> Dropping from the **0.9780** P95 bar to the **0.9675** speed gate changes public TPS by
> **−2.374** (a LOSS) while changing accepted private-draw risk by **+2.87pp** (assumed) / **+8.58pp**
> (grounded). Because TPS and clearance both fall with λ, the move is **strictly dominated**: it buys
> **−0.828 TPS/pp** (assumed) of extra risk. There is no speed-for-risk trade on the build-λ axis.

The most TPS-efficient **AND** least-risky publish-first build in the band
`[floor 0.9138, P95-bar 0.9780]` is therefore the **TOP** of the band:
`efficient_lambda = 0.9780` (the p95_bar) — *not* an interior point, because a dominated axis has no
trade-off frontier. λ=1 would be even better but sits outside the publish-first band.

## What the #124 tension actually is

Speed and safety **move together** on the build-λ axis, so the genuine green-light tension is
**not** speed-vs-safety. It is **build-λ vs. how HARD a high-λ build is to LAND** (land #71): a
higher-λ build is both faster and safer but harder to *construct and certify*. That difficulty
axis lives **outside** this speed/risk composition; this leg only prices the (negative) speed/risk
slope and authorizes nothing.

## Self-test (PRIMARY) — `tps_risk_exchange_rate_self_test_passes = True`

(a) TPS(λ) round-trips the composition anchors (513.557 @ gate, 520.953 @ λ=1; resid 0.0) and
risk(λ) round-trips #237's curve bit-for-bit (resid 0.0); (b) **observed** monotonicity over a fine
101-node grid — TPS strictly ↑, risk strictly ↓ ⇒ exchange rate **< 0 at every node** (definite
sign; note this is the *opposite* TPS direction to the PR's assumed premise, recorded as
`pr_premise_…holds = False`); (c) the grounded exchange rate differs from assumed and grounded risk
is steeper at the gate; (d) `efficient_lambda` in the publish-first band; (e) NaN-clean across all
reported scalars. All five pass.

## Hand-off (fern decision-card + Issue #124)

> Along the build-λ axis each +1pp of accepted private-draw risk near the 0.9675 gate buys
> `tps_per_pct_risk_at_speed_gate = −0.6431` TPS (assumed f_priv; −0.2443 grounded) — the slope is
> **NEGATIVE** because TPS(λ) and private-clearance are **co-monotone** (both rise with λ): a faster
> build is the *same* as a safer build, so there is **no speed-for-risk trade** to optimise here. The
> most TPS-efficient *and* least-risky publish-first build sits at `efficient_lambda = 0.9780` (the
> top of the band). Read the #124 tension as **build-λ vs. how hard a high-λ build is to land**
> (land #71), NOT as speed-vs-safety — those move together. Conditional on a measured tunable-λ
> build existing (land #71); this leg authorizes nothing.

## Reproduce

```bash
cd target/ && CUDA_VISIBLE_DEVICES="" \
  python research/validity/tps_risk_exchange_rate/tps_risk_exchange_rate.py \
    --self-test --wandb_group issue192-reading-calibration --wandb_name ubel/tps-risk-exchange-rate
```

CPU-only, peak ≈ 36 MiB, < 0.1 s. Imports: ubel #234/#222 `binding_gate` mu_pub TPS(λ) map (513.557
@ gate, 520.953 @ λ=1), ubel #229 speed gate λ=0.9675, kanna #237 `publishfirst_accepted_risk`
risk(λ) curve (assumed + grounded #224 f_priv), stark #191 P95 bar 0.9780, kanna #228 publish-first
floor 0.9138. BASELINE 481.53 untouched; adds 0 TPS (PRIMARY = self-test); greedy/PPL untouched.
NOT a launch. NOT open2.
