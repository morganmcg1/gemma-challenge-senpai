<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #343 — PPL-only gate (#124): what LIFTING greedy-identity (#192) buys on >500

**PRIMARY `ppl_only_envelope_self_test_passes` = True** (all 16 checks)
**TEST `coverage_lift_for_ppl_only_central_500` = +0.0186** (coverage delta from 0.8903 to PPL-only central-500)
**TEST `ppl_only_500_reachable_via_coverage` = True** (the central-500 lift fits lawine #336's +0.031 retrain budget)
**W&B `kklof4wr`** (group `ppl-only-gate-500-envelope`) · LOCAL read-only analytic, 0 GPU, 0 TPS

> **Verdict: lifting #192 converts the >500 lane from IMPOSSIBLE to ACHIEVABLE.** STRICT (gate ON) is
> supply-capped at **473.5 < 500** for every realizable deterministic schedule (denken #332, round-trips
> to 1e-13) — `strict_500_reachable = False`. PPL-ONLY (gate LIFTED) drops the determinism tax, so the
> operative envelope is stark #340's **demand-only** map; the existing head still doesn't reach 500
> (470.35/444.99 @ measured 0.8903), but a **coverage retrain of +0.0186 to c\*_central = 0.9089**
> clears central-500 **within** the +0.031 budget (worst-500 needs +0.0353, marginally past it). PPL is
> structurally decoupled from emission (wirbel #324, 2.3772 ≤ 2.42), so the gate-lift adds **no new PPL
> risk** — it's the SAME deployed serve that already passes the official scorer.

## The two worlds (deliverable 1)

| world | supply tax | best ceiling | 500 status | round-trip |
|---|---|---|---|---|
| **STRICT** (gate ON, #192) | denken #332 floor **0.09103** @ geometric φ | 520.95·(1−0.09103) = **473.53** | **IMPOSSIBLE** | == #332 `473.5296` (≤1e-6) |
| **PPL-ONLY** (gate LIFTED, #124) | **0** (no determinism) | **520.95** (λ-ceiling) | **ACHIEVABLE** | anchors 520.95/492.87 (≤1e-9) |

- **STRICT** pays the determinism tax. Even at **perfect coverage** (central corner = λ-ceiling 520.95)
  the best the strict world does is 473.53 < 500 — supply-capped for **every** realizable deterministic
  schedule (denken #332 proved `phi_realizable ≥ 1`), AND coverage must additionally clear the identity
  bar 0.9213. The strict-gated lane is DEAD end-to-end (supply RED + demand insufficient).
- **PPL-ONLY** drops determinism → the supply-φ tax **vanishes** → the operative envelope is stark
  #340's **demand-only** map (anchors already exclude the supply tax):
  `envelope_X(c) = X_ANCHOR · E[T](c)/E[T](0.9213)`, `E[T](c) = 1 + Σ_{d=1..7} c^d`. Coverage need only
  keep **PPL ≤ 2.42** (NOT the 0.9213 identity bar). The >500 lane is **not** supply-capped — reachable
  purely via coverage.

## Price @ the measured coverage + solve the lift (deliverable 2 — PRIMARY/TEST)

At the **measured** fusion coverage 0.8903 (lawine #330) the PPL-only envelope is **470.35 / 444.99**
(round-trips stark #337) — **both < 500**, so the existing head does **not** give a free >500 even
gate-lifted. Solving `env(c)=500` on the demand-only map gives the **same** roots as stark #340 (supply
tax just absent):

| corner | c\* (env=500) | lift from 0.8903 | within +0.031 budget? |
|---|---|---|---|
| central | **0.9089** (== #340) | **+0.0186** | **Yes** (`ppl_only_central_500_within_budget`) |
| worst | **0.9256** (== #340) | **+0.0353** | **No** (marginal, 13.9% past) |

Note **c\*_central = 0.9089 < identity_bar 0.9213**: PPL-only central-500 needs *less* coverage than the
strict identity bar **and** no supply revival.

## PPL gate holds in the gate-lifted config (deliverable 3 — cite, do not re-run)

wirbel #324 (`pespixw1`) showed the M=8 argmax divergence is structurally **DECOUPLED** from PPL: PPL is
a `prompt_logprobs` reference-forward over **fixed token-IDs**, so it passes by construction
(`ppl_delta_under_eagle3_verify = 0.0`), M-binary. **PPL stays 2.3772 ≤ 2.42.** The PPL-only config is
the **SAME deployed serve** that already passes the official scorer (481.53 frontier, 128/128) — lifting
#192 removes a self-imposed identity check the scorer never ran, introducing **no new PPL risk**.
*Caveat:* the literal served greedy-rate would need an HF Job (gated, ubel #322) — not drawn here.

## The #124 delta (deliverable 4 — the deliverable the human needs)

> **STRICT: >500 IMPOSSIBLE** (supply-capped 473.5; no retrain reaches it).
> **PPL-ONLY: >500 ACHIEVABLE** via a coverage retrain to **0.9089** (central) / **0.9256** (worst).

**Lifting #192 converts the >500 lane from IMPOSSIBLE under strict identity to a SIZED, FEASIBLE
coverage-retrain target** — central within the +0.031 budget; worst marginally past it
(`impossible_to_feasible_conversion = True`).

## Key precision (do not over-claim a free win)

The deployed 481.53 is **already** the fast, non-deterministic, PPL-passing config (its 56%
AR-divergence proves batch-invariant determinism is OFF in deployment). Lifting #192 does **not** change
the deployed config or give a free one-run jump to ~520 — it means we never pay denken #332's
determinism tax (paid only to chase strict identity), and the >500 path becomes a coverage/E[T] retrain
on the already-fast config that **actually reaches 500** (vs being supply-capped at 473.5 under strict).

## Self-test (NaN-clean, deterministic)

All **16** PRIMARY checks pass: (a) strict ceiling 473.5 round-trips denken #332 + `strict_500_reachable`
False · (b) PPL-only envelope round-trips #340 anchors 520.95/492.87 and #337 corners 470.35/444.99
@0.8903 · (c) c\*_central=0.9089 / c\*_worst=0.9256 reproduced · (d) PPL-decoupling caveat carried · (e)
#124 delta NaN-clean + impossible→feasible · (f) central lift +0.0186 within budget · (g) worst lift
+0.0353 over budget · (h) `ppl_only_500_reachable_via_coverage` True · (i) imports exact · (j) NaN-clean
· (k) roots ∈ (0,1) · (l) PPL-only ceiling 520.95 > strict 473.5 · (m) c\*_central below identity bar.

## Honest caveats (carried in the artifact)

1. **INVERTS stark #340 into the PPL-only world — re-prices nothing measured.** The PPL-only envelope is
   #340's same demand-only map; the only change is the strict world *additionally* pays denken #332's
   supply tax. No EAGLE-3 fusion checkpoint runs here.
2. **Not a free win.** The deployed config is unchanged; the gate-lift sizes a coverage retrain, it does
   not jump TPS by itself.
3. **STRICT 473.5 is the supremum over coverage** (central corner taxed); `phi_realizable ≥ 1` makes
   `strict_500_reachable = False` robust, not a point estimate.
4. **PPL safety is structural** (wirbel #324); the literal served greedy-rate needs a gated HF Job
   (ubel #322). NOT a launch / build / served-file change / HF Job / submission / open2.

## Greedy/PPL-safety certificate

`analysis_only = True`. No served-file change, no emitted-token change, no HF Job, no submission, NOT a
launch, NOT a build. Coverage/E[T] is the **speed** axis; the PPL-only config is the deployed serve
(PPL 2.3772 ≤ 2.42, structurally decoupled per wirbel #324). BASELINE **481.53 TPS unchanged**; this leg
adds **0 TPS**.

## Hand-off

LIFTING #192 (PPL-only, #124) converts the >500 lane from **IMPOSSIBLE** under strict identity
(supply-capped 473.5, denken #332) to **ACHIEVABLE** via a coverage retrain on the same already-fast
deployed serve: **c\*_central = 0.9089 (+0.0186, within lawine #336's +0.031 budget)** / c\*_worst =
0.9256 (+0.0353, marginally past it). The existing head does not give a free >500 even gate-lifted
(470.35/444.99 @0.8903). The #124 deliverable: lifting #192 sizes the >500 path as a **feasible
coverage-retrain target**, not a build/launch.

## Public evidence used

Banked W&B numbers only (all `wandb-applied-ai-team/gemma-challenge-senpai`): stark #340 `jwv1vbug`
(demand-only envelope + c\*_central 0.9089 / c\*_worst 0.9256 / identity bar 0.9213); stark #337
`lbuirkpt` (E[T](c) chain law, anchors 520.95/492.87, honest corners 470.35/444.99 @0.8903); denken #332
`y5cl0ena` (supply floor 0.09103 @ geometric φ, strict ceiling 473.5, `phi_realizable ≥ 1`, revive
breakeven 0.255); lawine #330 `hfrscdai` (cov prior 0.8903, identity bar 0.9213); lawine #336 `krroookz`
(+0.031 retrain budget); wirbel #324 `pespixw1` (PPL-decoupling 2.3772, `ppl_delta = 0.0`, M-binary).
Official frontier 481.53 TPS (PR #52, `2x9fm2zx`).
