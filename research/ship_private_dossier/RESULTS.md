# PR #508 — Surgical-357 private-outcome dossier + floor-lock portfolio price

**Run:** `fn2v5wox` · `--wandb_group ship-private-dossier` · analysis_only, official_tps=0, CPU-only.
NO serve, NO HF job, NO `--launch`, NO submission, NO served-file change.

## One-page dossier verdict (the number for the human at reopen)

> **Ship surgical-357 as primary; hold floor-lock-166.23 as the pre-staged fallback.**
> Expected private TPS **341.9** [95% **335.2–348.6**], a **4.3% breach** off the 357.2 local
> public anchor (realized propagation factor **1.00 — linear**, kanna #504). P(private <
> 0.95×public) = **23.1%**; **P(private < floor-lock 166.23) ≈ 0** (51σ away). On raw TPS,
> surgical-357 **dominates floor-lock under every plausible private draw** — even the refuted
> 24% worst-case breach at its 95% σ_hw downside (**266.2**) beats floor-lock's best case
> (169.5). **Floor-lock is never a speed case — it is purely invalidation insurance.** The
> ship/keep decision therefore hinges entirely on the organizer's private **validity rule**
> (the #474 crux), not on speed.

## Portfolio price — bracketed verdict (`portfolio_verdict = "bracketed"`)

| private validity rule | surgical-357 | floor-lock-166.23 | dominant |
|---|---|---|---|
| **(a) penalize-breach** (score = realized private TPS) | **341.9** [335.2–348.6]; 24%-WC 95%-lo 266.2 | 166.23 | **surgical** (+175.6 TPS / +106%; holds across the whole band & WC) |
| **(b1) invalidate @ speed-threshold** (private < 0.95×public → 0) | E-value **264.1** (P_inval 23.1%, E[·\|valid]=343.2); worst outcome **0** | 166.23 guaranteed | **bracketed**: E-value→**surgical** (264.1 > 166.2); maximin→**floor-lock** (0 < 166.2) |
| **(b2) invalidate @ literal greedy identity** (any private divergence → 0) | **0** (operative-1.0; spec-alive diverges off-public by construction) | 166.23 (literal-1.0) | **floor-lock** outright (166.2 vs 0) |

**Decisive fork:** does the reopen rule key on **speed-drift** (→ ship surgical) or **literal
private greedy identity** (→ keep floor-lock)? And is the objective **expected-value** (→
surgical, even under the speed rule) or **maximin/guaranteed-floor** (→ floor-lock)?

## Composition (PR item 1 — the surgical-357 private band)

`private_TPS = 357.22 × (1 − 0.04295 [#504 linear breach]) × (1 ± 0.01 [#478 σ_hw one-shot])`

| stat | value |
|---|---|
| private mean | **341.88** TPS |
| 68% band | [338.46, 345.30] |
| **95% band** | **[335.18, 348.58]** |
| σ_hw (1% of mean) | 3.42 TPS |
| **P(private < 0.95×public = 339.36)** | **0.2306** |
| **P(private < floor-lock 166.23)** | **≈ 0** (point 0.0e+00 @ −51.4σ; convolution 0.0) |

The band is σ_hw-driven (the breach point is the realized/expected 4.3%, not a worst case). Rounding
the PR-literal `357 × (1 − 0.043)` moves the mean by <0.3 TPS (341.67 vs 341.88) — immaterial.

## Why the floor-lock case is purely invalidation insurance (PR item 2)

- **Expected private gap** surgical − floor-lock = **+175.65 TPS (+105.7%)**.
- **Downside gap**: surgical's 95% **downside** (335.18) vs floor-lock's 95% **upside** (169.49) =
  **+165.69** — *surgical's worst case beats floor-lock's best case*.
- **Even at the refuted 24% worst-case breach**: mean 271.49, 95%-lo 266.17 — still **+96.68** over
  floor-lock's best case. There is **no plausible private draw** where floor-lock wins on raw speed.
- So `P_surgical_below_floorlock ≈ 0`: the only mechanism by which floor-lock can "win" is an
  **invalidation** rule that zeroes surgical's score — i.e. a *validity* decision, not a speed one.

Floor-lock (166.23, `submissions/fa2sw_strict_m1ar_int4`, literal-1.0) is the guaranteed-valid floor:
zero breach (M=1 AR, no spec → private mean == public), and #478 confirmed it is structurally safe
(10/10 completions, `floorlock_draw_is_safe=1`).

## Inputs (my own merged work — reused, not re-derived)

| input | value | source |
|---|---|---|
| ship public TPS | 357.22 (PR "357") | kanna #504 `0urxqwob` / recert l0attso0 |
| linear breach frac | 0.04295 (PF 1.00) | kanna #504 `0urxqwob` |
| σ_hw fractional (one-shot) | 0.01 (between-dominated 13.9×) | kanna #478 `mssuss3f` |
| floor-lock TPS | 166.23 (literal-1.0, zero-breach) | stark #485 `pavotwci` |
| ship run / classification | `j7qao5e9`, operative-1.0, spec-alive | stark #499 |

**σ_hw note:** the composition uses the **scale-invariant 1% fractional** convention (1% of the 341.9
mean = **3.42 TPS**, not the 481-scale 4.864). The PR's "σ_hw 4.864" is the directly-measured
between-allocation leg at its ~505-mean pool — provenance only, not composed with.

## Self-test (`self_test.passes = True`, 25/25)

Reproduces #504 exactly (mean / 95-band / P_below to ≤1e-6); both bands ordered; P's valid &
NaN-clean; raw-TPS dominance (surg 95-lo > floor 95-hi, and 24%-WC 95-lo > floor 95-hi); speed-rule
expected-score discounting consistent; rule winners and bracketed verdict assert as expected; σ_hw
convention roundtrips (0.01×481.53 = 4.8153; 0.010128×481.53 = 4.877).

## Command

```bash
.venv/bin/python -m research.ship_private_dossier.compose_dossier \
    --name kanna/ship-private-dossier --group ship-private-dossier
```

Peak memory: negligible (pure-Python CPU composition; no model load, no serve). W&B run `fn2v5wox`
(`ship_private_dossier` artifact attached).
