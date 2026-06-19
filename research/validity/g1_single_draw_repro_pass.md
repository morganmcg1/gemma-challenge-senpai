# Single-draw P(G1-pass) for the fire — acceptance-gap × σ_hw (PR #756)

**TL;DR.** The literal G1 gate is a **single private reproduction run**: the
organizer reruns the fire (`submissions/int4_mtp_batchinv`) on a private prompt
mix and invalidates it if `private_TPS < 0.95 × reported_TPS`. That single draw
carries **two** downside sources — the *systematic* public→private acceptance gap
(`δ_stock`, my #754) and the *aleatoric* single-draw hardware/scheduling jitter
(`σ_hw`, my #159/#478). #754 modeled only the first and gave `P(DQ)=0.0137`. Folding
`σ_hw` onto that distribution gives the literal gate quantity:

> **`p_g1_single_draw_pass = 0.985`** — on a single private reproduction draw, the
> fire clears the 5% rule **98.5%** of the time. σ_hw moves it only **−0.13 pp**
> off #754's systematic-only `0.9863`. The 95%-worst single draw still runs at
> **97.7%** of reported (R_05 = 0.977), a **+2.74 pp** cushion inside the 5% rule.
> The fire's official number need only beat the 126.378 bar by
> **`g1_margin_tps_at_95 = 2.9 TPS`** for the 95%-worst single private draw to
> still clear the bar.

This completes the G1-safety leg: **#749** faithful central (−7.73%) → **#754**
well-posed `P(DQ)=0.0137` → **this** single-draw `P(G1-pass)=0.985`.

**Verdict: `SINGLE_DRAW_G1_SAFE`.** Analysis-only — no HF Job, no submission, no
served-file change. Reuses only my own merged artifacts on the advisor branch.

---

## 1. The gate, made literal

The G1 reproduction rule is one-sided on the realized TPS ratio
`R = private_TPS / reported_TPS`:

```
G1-pass  ⟺  private_TPS ≥ 0.95 × reported_TPS  ⟺  R ≥ 0.95
```

`reported_TPS` is the fire's posted (public) number — a fixed anchor. The single
private rerun realizes `R = R_systematic × (1 + ε)`:

* **Systematic** `R_systematic = 1 − δ_stock/100` — the private prompt mix has a
  different acceptance profile. From #754's faithful 57/57/14 public-faithful blend
  (`f_nonmcq ≈ 0.109`), `δ_stock` central **−7.73%** (private *faster*; the only
  slow corner is free-response math at +19.6%). The #749 report confirms
  `δ_stock_by_tps ≈ δ_stock_by_eaccept` to <1 pt, so the e_accept gap **is** the
  TPS-ratio gap.
* **Aleatoric** `ε ~ N(0, σ_hw)` — which physical A10G the official scorer lands
  you on (clock/bandwidth/contention). My #159 measured this; my #478 canonicalised
  it as a **~1% fractional, multiplicative** per-draw CV.

Sign convention (carried from #754): `δ_stock < 0` ⇒ private faster;
`P(DQ) = P(δ_stock > +5%)` = private >5% slower = the reproduction-fail side, so
systematic-only `P(G1-pass) = 1 − P(DQ) = 1 − 0.0137 = 0.9863`.

## 2. Method (LOCAL, no HF Job)

* **Arm A — systematic R.** Regenerate the **exact** #754 well-posed block
  bootstrap (seed 730, 50 000 draws over the #749 faithful serving run's ~10 s
  spec-log windows) by importing its machinery
  (`scripts/validity/deltastock_wellposed.py`). Reproduces #754 byte-for-byte:
  central −7.729%, SE 6.047%, CI95 **[−19.93, +3.68]**, `P(DQ)=0.0137`,
  e_pub-fixed SE 2.878%. Map each draw to `R_sys = 1 − δ_stock/100`.
* **Arm B — fold σ_hw.** `R_real = R_sys × (1 + ε)`, `ε ~ N(0, 0.01)`, independent
  aleatoric draw. Report `p_g1_single_draw_pass = P(R_real ≥ 0.95)`, the two-sided
  `P(within ±5%)`, and the move vs the systematic-only 0.9863.
* **Arm C — invert.** `R_05` (single-draw 5th percentile) → the headroom the fire's
  number must carry over the bar: `g1_margin_tps_at_95 = 126.378 × (1/R_05 − 1)`.

Cross-checked with a closed-form symmetric-Normal model and swept over a σ_hw
robustness bracket.

## 3. Results

| quantity | value |
|---|---|
| **`p_g1_single_draw_pass`** (primary) | **0.985** |
| sym-Normal cross-check | 0.981 |
| gate-literal (e_pub-fixed) | 1.000 |
| systematic-only (#754, 1−P(DQ)) | 0.9863 |
| **σ_hw moves pass by** | **−0.0013** |
| `P(within ±5%)` two-sided | 0.341 |
| `P(faster than +5%)` | 0.644 |
| R_real mean / median | 1.075 / 1.072 |
| **R_05** (95%-worst draw) | **0.9774** |
| reproduction margin at 95% | **+2.74 pp** inside the 5% rule |
| **`g1_margin_tps_at_95`** (test) | **2.92 TPS** over 126.378 |
| required reported for 95% bar-clear | ≥ 129.30 TPS |

**σ_hw robustness sweep** (p_g1 holds ≥ 0.95 across every basis, including the
deliberately loose absolute bound):

| σ_hw basis | σ_hw | p_g1 | R_05 | margin TPS |
|---|---|---|---|---|
| #159 cross CV (frantic-penguin) | 0.96% | 0.9851 | 0.977 | 2.92 |
| #478 canonical convention | 1.00% | 0.9850 | 0.977 | 2.92 |
| #478 one-shot measured | 1.01% | 0.9850 | 0.977 | 2.93 |
| absolute bound @ ~160 TPS (loose) | 3.01% | 0.9721 | 0.966 | 4.43 |

**Findings.**

1. **σ_hw barely moves the gate.** It shifts `p_g1` only −0.13 pp (0.9863 → 0.9850)
   because the systematic gap is favorable and far from the +5% fail line; the 1%
   aleatoric jitter is small next to the 6.05% systematic SE. Even the loose 3%
   absolute bound leaves `p_g1 = 0.972 ≥ 0.95`.
2. **The fire is so much faster on the private mix that the two-sided "reproduces
   within ±5%" band is the wrong lens** — only 34% of single draws land inside ±5%
   because **64%** are *more than 5% faster* than reported. That fails the symmetric
   band on the high side but passes the one-sided G1 rule. The gate is one-sided;
   being faster is not a reproduction failure.
3. **Margin.** At the 95%-worst single draw the private run is still 97.7% of
   reported. Anchored to the only hard official number (126.378), the fire must post
   ≥ **129.3 TPS** (+2.92 over the bar) so the worst-case single private draw still
   clears the bar. Since the int4 fire is expected at ~150–168 TPS, it clears with
   room: at reported 157 the 95%-worst private draw is 153.5 TPS (+27 over the bar).

## 4. Honesty carry-forward (the #754 caveats, propagated)

* **Wide CI.** The systematic central CI95 is **[−19.9%, +3.7%]** — the upper end
  *touches* positive (slightly slower), which is exactly why the residual `P(DQ)`
  isn't zero. The point and bootstrap-mean both sit firmly favorable.
* **Common-mode dominance.** 5.32 pp of the 6.05 pp systematic SE is the **shared
  official-128 public-anchor sampling noise** (common-mode across the three
  corners), not corner spread. The gate divides by that *fixed* official-128 public
  TPS, so the gate-literal e_pub-fixed reading (SE 2.88 pp, `P(DQ)=0`) gives
  `p_g1 = 1.0`. I report the conservative full-bootstrap as the headline.
* **No double-count.** `σ_hw` is hardware-allocation scatter; the δ_stock
  common-mode noise is public-anchor *prompt* sampling — independent sources, so
  they add in quadrature, not double-counted.
* **σ_hw provenance.** Same hardware (A10G, sm_86): within-allocation measured
  n=12 fresh-server **locally** (≈0.011%); the dominant cross-allocation term is
  frantic-penguin's **3 same-submission official a10g-small draws**
  (489.63/483.80/480.41, CV 0.96%). **Not** same-config as the fire — the cross
  term was measured on the split-KV/K7 ~485 TPS frontier, not `int4_mtp_batchinv`.
  #478 established σ_hw is a *multiplicative* clock/bandwidth draw, so the ~1%
  fractional model transfers across configs (a model assumption, not a direct
  fire-config measurement). I deliberately **reloaded** rather than re-measured:
  local repeated draws on one pod capture only the negligible within-allocation
  term (~0.01%) and **cannot** see the cross-allocation scatter that dominates a
  real official single draw (each official rerun lands on a different physical GPU).
* **Block-bootstrap conservatism.** The ~10 s spec-log window block bootstrap is
  conservative vs an i.i.d.-prompt resample, widening the systematic SE.

## 5. Files

* `scripts/validity/g1_single_draw_repro_pass.py` — self-contained analysis
  (imports the #754 bootstrap machinery; no server, no HF Job).
* `research/validity/g1_single_draw_repro_pass/results/g1_single_draw_repro_pass.json`
  — full record.
* `research/validity/g1_single_draw_repro_pass/results/single_draw_distribution.png`
  — folded single-draw R distribution.
* W&B run `pu7mell9` (group `g1_single_draw_repro_pass`).

**Reproduce:**
```bash
.venv/bin/python scripts/validity/g1_single_draw_repro_pass.py \
    --draws 50000 --wandb_group g1_single_draw_repro_pass
```
