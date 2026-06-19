# σ_hw breakdown-point — how fragile is the G1-safety claim to hardware-variance mis-spec? (PR #763)

**TL;DR.** The board G1-safety headline (#756 single-draw `P(G1-pass)=0.98504`;
#758 position survival `0.9958`) rests on **one** load-bearing modeling
assumption: the local→official hardware-variance transfer `σ_hw`. It is **not** a
direct fire-config measurement — the canonical **1% fractional, multiplicative**
CV was characterized on the split-KV/K7 ~485 TPS frontier (#159 frantic-penguin
3-draw CV 0.962%) and *assumed* to transfer to `int4_mtp_batchinv` (#478). This
capstone does **not** re-measure `σ_hw`; it asks the robustness question — *how
badly would `σ_hw` have to be mis-specified before the claim breaks?* — by holding
everything else (#754 systematic `δ_stock` bootstrap, #758 common-mode ρ,
idiosyncratic SE, every seed) **fixed** and sweeping **only** `σ_hw` upward.

> **`sigma_hw_breakdown_mult_at_095 = 4.72×`** — single-draw `P(G1-pass)` survives
> `σ_hw` mis-specification up to **4.72× the modeled 1%** (i.e. σ_hw ≈ 4.7%) before
> it drops to 0.95, and **7.24×** before 0.90. The joint leaderboard-position
> survival is **even more robust**: ~**12.8×** (correlated) / ~**10.2×**
> (physical backstop) before 0.95. At the multiple where the *fire-alone* claim
> hits 0.95 (4.72×), position survival is still **0.993** because the deterministic
> 126.378 live submission fails only via σ_hw, **independently** of the fire's
> `δ_stock`-driven failure.

**Verdict: `G1_CLAIM_ROBUST_TO_SIGMA_HW` (`g1_claim_robust = 1`).** The breakdown
multiple (4.72×) is **>2×**, so the headline is robust — `σ_hw` would have to be a
large multiple of its modeled value, far outside any measured hardware scatter
(measured CV <1%; the loosest absolute bound was 3%), to break the claim.
Analysis-only — LOCAL CPU-numpy, no HF Job, no submission, no served-file change.
Reuses my own merged #754/#756/#758 machinery byte-for-byte.

---

## 1. The robustness question

`σ_hw` is the single biggest modeling assumption under the G1-safety leg, and the
#756 card flagged it itself: it is an **assumption** that the ~1% multiplicative CV
measured on a *different* config (the ~485 TPS frontier) transfers to the fire, the
systematic CI95 is wide (`[−19.9, +3.7]`), and the common-mode dominates 77.3% of
the `δ_stock` variance. The board post is about to claim G1-safety to @cmpatino, so
before it does we want a **defensible fragility bound**.

The literal gate is a single private rerun realizing `R = R_sys × (1 + ε)`,
`R_sys = 1 − δ_stock/100`, `ε ~ N(0, σ_hw)`, scored one-sided `R ≥ 0.95`. We sweep
`σ_hw → m · σ_hw^modeled` and read off the multiple `m` at which each survival
curve first crosses 0.95 (then 0.90).

**Sensitivity-sweep determinism.** `convolve`/`joint_survival` draw `ε` with a
*fixed* rng seed and fixed size, so `ε = σ · z` for the **same** standard normals
`z` at every multiple. Scaling `σ` scales the same draws → the breakdown curve is
**smooth and monotone with no Monte-Carlo jitter across the sweep**, which is
exactly what isolates the `σ_hw` axis. At `m = 1×` the pipeline reproduces #756
(`P_fire=0.98504`, `R_05=0.97739`, `g1_margin=2.9231`) and #758 (`P_pos=0.9958`)
**byte-identically** (asserted in-code).

## 2. Method (LOCAL, no HF Job)

Reuse my own merged machinery verbatim — `load_systematic_delta`, `convolve`
(from #756) and `joint_survival` (from #758):

* **Modeled baseline.** `σ_hw^modeled = 0.01` (1% fractional) — #159 cross-allocation
  frantic-penguin official 3-draw CV 0.962% (dominant), #478 canonicalized to 1%.
* **Sweep.** `m` from 0.5× to 20× in 0.05× steps (391 points). The 0.5–4× band is
  the requested detail window; the grid extends to 20× only to *locate* the
  crossings (which lie beyond 4×). For each `m`, recompute (a) single-draw
  `P(G1-pass)`, `R_05`; (b) joint position survival `P(≥1 of {126.378-live, fire}
  clears G1)` under the common-mode-correlated reading, the comonotonic floor, the
  independence ceiling, and the **physical deterministic-126 backstop**; and (c) the
  126.378 deterministic marginal `Φ(0.05/σ_hw)`.
* **Held fixed** (so the sweep isolates `σ_hw`): the #754 bootstrap (seed 730, 50k),
  the #758 common-mode ρ (0.7515 realized) / idiosyncratic SE / all hw + permutation
  seeds. δ_stock, ρ, and the idiosyncratic SE do **not** move.
* **Anchors.** `P(G1-pass)`/`R_05` are **scale-free** in the official TPS (the gate
  is the ratio rule), so the breakdown multiple is identical at any anchor; we report
  the **absolute** 95%-worst private TPS at the predicted fire anchor **157**
  (fern #750) and the bar-adjacent stress anchor **130**.

## 3. Results — the breakdown multiples

| survival curve | value @ 1× | →0.95 at | →0.90 at |
|---|---|---|---|
| **fire single-draw `P(G1-pass)`** (PRIMARY) | 0.98504 | **4.72×** | 7.24× |
| position survival (common-mode correlated) | 0.99580 | 12.78× | >20× |
| position survival (physical det-126 backstop) | 1.00000 | 10.21× | 16.66× |
| 126.378 deterministic marginal (`Φ(0.05/σ)`) | 1.00000 | 3.04× | 3.89× |

**Detailed 0.5–4× band** (every curve holds far above 0.95 across the entire
plausible σ_hw range):

| `m` | σ_hw | fire `P(G1)` | `R_05` | pos. corr | pos. phys | 126-det |
|---|---|---|---|---|---|---|
| 0.5× | 0.50% | 0.98582 | 0.9784 | 0.99602 | 1.0000 | 1.0000 |
| 1.0× | 1.00% | 0.98504 | 0.9774 | 0.99580 | 1.0000 | 1.0000 |
| 2.0× | 2.00% | 0.98056 | 0.9733 | 0.99490 | 0.9998 | 0.9940 |
| 3.0× | 3.00% | 0.97216 | 0.9662 | 0.99340 | 0.9986 | 0.9519 |
| 4.0× | 4.00% | 0.96048 | 0.9576 | 0.99172 | 0.9955 | 0.8935 |

**The deterministic backstop (the reason position survival is more robust).** At
the multiple where the **fire-alone** claim hits 0.95 (`m = 4.72×`, σ_hw ≈ 4.7%):

* position survival (correlated) = **0.99008** ✓ above 0.95
* position survival (physical) = **0.99282** ✓ above 0.95

So **yes** — the deterministic 126.378 backstop keeps the leaderboard position
survival at ~0.993 even when the fire's single-draw claim has just broken. The
126.378 live submission (`int4_g128_lmhead`, non-spec greedy, deterministic)
carries **no** `δ_stock` and fails **only** via σ_hw — a failure mode *independent*
of the fire's `δ_stock`-driven failure — so the joint `P(fire ∨ 126 clears G1)`
breaks at a **larger** σ_hw multiple than the fire alone.

**Absolute margin — scale-free claim vs scale-dependent bar-clearance.** The
breakdown *multiple* is identical at the 157 and 130 anchors (the G1 reproduction
gate is the ratio rule). What differs is the absolute 95%-worst private TPS:

| | worst-priv @ 157 (margin over 126.378) | worst-priv @ 130 (margin) |
|---|---|---|
| modeled 1× | 153.45 (**+27.07**) | 127.06 (**+0.68**) |
| fire 0.95 breakdown (4.72×) | 149.18 (**+22.80**) | 123.53 (**−2.85**) |

At the 130 stress anchor the absolute worst-private margin over the 126.378 bar is
razor-thin even at the modeled σ_hw (+0.68), and goes negative well before the
scale-free G1-reproduction claim breaks. **This is a property of the 130 op-point
being barely above the bar, not of G1 fragility** — at the predicted 157 anchor the
worst-case draw clears the bar by +22.8 even at the 4.72× breakdown.

## 4. Findings

1. **The headline is robust by a wide margin.** `σ_hw` would have to be **4.72×**
   its modeled value — σ_hw ≈ 4.7%, ~5× the measured <1% CV and well past the loose
   3% absolute bound — before single-draw `P(G1-pass)` even reaches 0.95. The
   refutation condition (breakdown ≲ 1.5×) is nowhere near met. `g1_claim_robust = 1`.
2. **Why σ_hw is such a weak lever.** The systematic `δ_stock` gap is *favorable*
   (central −7.73%, R_sys ≈ 1.077) and sits far from the +5% fail line; the 6.05 pp
   systematic SE dwarfs a 1% multiplicative jitter. σ_hw has to grow several-fold
   before its widened tail pushes meaningful mass across the 0.95 floor.
3. **Position survival is strictly more robust than the fire alone**, exactly as
   pre-registered in #758 — the deterministic 126.378 backstop (independent σ_hw-only
   failure) lifts the 0.95 crossing from 4.72× (fire) to ~10–13× (joint). Even when
   the fire-alone claim breaks, the position holds at ~0.993.
4. **The 126.378 marginal alone is the least robust leg (3.04×)** because its *only*
   defense is σ_hw and the gate `Φ(0.05/σ_hw)` degrades once σ_hw ≈ 3% — but this is
   a single-submission marginal; the *joint* (fire ∨ 126) is far tougher because both
   must fail simultaneously and they fail through independent mechanisms.

## 5. Honesty carry-forward

* **This is a sensitivity bound, not a measurement.** The claim is "the headline
  survives σ_hw up to **N× modeled**," **not** "σ_hw is N×." No new σ_hw measurement
  was taken; the modeled 1% is unchanged and remains the best estimate.
* **#756 caveats propagate.** Wide systematic CI95 `[−19.9, +3.7]` (upper end touches
  positive); common-mode dominates **77.3%** of the `δ_stock` variance (5.32 of
  6.05 pp SE is shared official-128 public-anchor sampling noise); the gate is
  **one-sided** (being faster is not a failure — 64.4% of single draws are >5%
  faster).
* **σ_hw transfer assumption is the whole point.** The cross-allocation CV was
  measured on the ~485 TPS split-KV/K7 frontier, not `int4_mtp_batchinv`; #478
  establishes σ_hw is a multiplicative clock/bandwidth draw, so the fractional model
  transfers across configs — *a model assumption*. This card quantifies exactly how
  wrong that assumption could be (≥4.72×) and still leave the claim intact.
* **Consistency with #756.** The #756 loose absolute corner (3% at ~160 TPS ≈ 3×
  modeled) already reported `P(G1-pass)=0.972`; this sweep reproduces it (3.00× →
  0.97216) and extends it to the crossing.

## 6. Files

* `scripts/validity/g1_sigma_hw_breakdown.py` — self-contained sweep (imports the
  #756/#758 machinery; no server, no HF Job).
* `research/validity/g1_sigma_hw_breakdown/results/g1_sigma_hw_breakdown.json` —
  full 391-point sweep + breakdown multiples.
* `research/validity/g1_sigma_hw_breakdown/results/sigma_hw_breakdown.png` —
  survival curves vs σ_hw multiple with 0.95/0.90 thresholds and the modeled-1× and
  2× bars marked.
* W&B run `wnac29oo` (group `g1_safety_robustness`).

**Reproduce:**
```bash
python scripts/validity/g1_sigma_hw_breakdown.py \
    --draws 50000 --mult-lo 0.5 --mult-hi 20 --mult-step 0.05 \
    --wandb_group g1_safety_robustness
```
