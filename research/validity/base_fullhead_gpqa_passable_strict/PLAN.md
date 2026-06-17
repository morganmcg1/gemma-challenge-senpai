# GPQA passable-strict verdict: paired bootstrap + Beta-posterior (PR #574, #564 follow-up)

**Analysis-only, NO FIRE.** `analysis_only=true`, `official_tps=0`. No HF Job, no
`train.py --launch`, no `/v1/jobs:run`, no submission, no served-file change. 0-GPU CPU
re-analysis of data already banked in my #564 (`7bi4e2ne`).
`--wandb_group base-fullhead-gpqa-passable-strict`.

## The footnote this closes (#564 follow-up)
My #564 (`7bi4e2ne`) showed MMLU-Pro strict Wilson CI-lb PASSES at n=1500 (+0.0327) but
GPQA-Diamond "FALSE" — and proved that miss is **un-passable by construction**: a Wilson
CI-lb on a fixed 198-item set has half-width pinned by n, so it can never clear 0.90x even
when the true gap is zero (it would need n≈830). #564 Stage-3 paired McNemar (p=0.761,
discordant 20/23) showed the base_fullhead↔ple_fold GPQA gap is sampling noise, not a
surgical-attention decrement.

This card replaces the un-passable Wilson lens with the standard small-n replacements that
answer the *actual* question ("is base_fullhead ≥ 0.90 × the denominator?") using the PAIRED
structure (which the unpaired Wilson lens throws away): a **paired bootstrap CI** on the gate
margin and a **Beta-posterior** on the clear-probability.

## Banked data source (#564 `7bi4e2ne`, byte-identical 198-item set, `n_prompt_mismatch=0`)
Verified paired 2×2 over the same 198 GPQA-Diamond items:

|                   | ple_fold correct | ple_fold wrong | row |
|-------------------|------------------|----------------|-----|
| base_fullhead ✓   | 75               | 20             | 95  |
| base_fullhead ✗   | 23               | 80             | 103 |
| col               | 98               | 100            | 198 |

base_fullhead = 95/198 = 0.4798; ple_fold = 98/198 = 0.4949. Point margin
mean(x) − 0.90·mean(y) = +0.0343.

## Denominators (Morgan #515 gate = base_fullhead ≥ 0.90 × vanilla base)
Reported against all three the program has used (wirbel #568 flagged the spread):
- **ple_fold #557/#564 (0.4949, PAIRED per-item) — PRIMARY/strictest.** gate 0.4455.
- ubel #511 vanilla-base anchor (0.470). gate 0.4230.
- banked `base_gpqa.json` (0.4444). gate 0.4000.

## Stages (all PRIMARY)
- **Stage 1 — paired bootstrap CI on the gate margin.** Resample ITEMS (pairs) with
  replacement (B=20000, fixed seed). `m_b = mean(x) − 0.90·mean(y)`. Report
  `gpqa_bootstrap_margin_point`, `gpqa_bootstrap_ci95`, `gpqa_bootstrap_passes` = (CI-lb > 0).
- **Stage 2 — Beta-posterior of clearing the gate.** `gpqa_beta_posterior_p` =
  P(p_fh ≥ 0.90·p_pf) under independent Jeffreys-Beta(0.5,0.5) (unpaired) AND a paired
  Dirichlet(0.5,…) over the 4 cells (paired). MC ≥100k, fixed seed.
  Pre-register `gpqa_beta_passes` = (p ≥ 0.95).
- **Stage 3 — exemption + consolidated verdict.** Capstone-spec amendment paragraph;
  `gpqa_passable_strict_verdict` ∈ {PASS, FAIL} folding Stages 1–2 + the #564 McNemar noise
  finding; `n_for_wilson_cilb_pass` (≈830 confirm) plus the analogous n the bootstrap/Beta
  lenses need.

## Cites
#564 `7bi4e2ne` (data + the strict-CI-lb miss this resolves); #557 `yw6vwk1w` (ple_fold
denominator); #542 `92pcnx6a` (two-prop-z GO); Morgan #515 (≥90% gate); #511 ubel anchor;
kanna #563 owns the disjoint SAMPLING-protocol axis (this is the GREEDY protocol).

`primary_metric = gpqa_passable_strict_verdict` (or the bootstrap CI-lb margin).
NO HF FIRE — hardens the quality half of the two-gate, does not touch the speed half.
