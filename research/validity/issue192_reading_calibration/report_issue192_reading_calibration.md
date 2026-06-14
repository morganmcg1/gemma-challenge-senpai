<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Issue #192 enforcement-reading calibration — per-reading pass-fractions (PR #219)

**denken · `denken/issue192-reading-calibration` · W&B `0unwptbz` · BANK-THE-ANALYSIS (adds 0 TPS, no draw, no launch)**

## The question

Issue #192 (greedy token-identity) is the live launch gate. The human asked the board
**which reading** of the contract applies: **(A)** strict literal per-sequence
token-identity, **(B)** per-token tolerance, or **(C)** operational / PPL acceptance.
kanna #114 (`9q5yy9l1`) measured **one** divergence number — **56.08%** of tokens flip
argmax under the int4-Marlin batch-variant spec-verify GEMM vs plain greedy AR — but a
single per-token rate does not tell the human the per-reading **pass-fraction**. This leg
converts #114's one number into the per-reading **menu** the #192 ruling picks from. It is
a **read-only re-read** of #114's banked divergence (it does NOT re-measure #114).

## Source granularity — the finest #114 banked, used directly

#114's `interlock_report.json` banks the **per-sequence** split directly (`num_identical=16`,
`num_divergent=112` of 128; aggregate `token_div_frac=0.5607757568359375`; onset
min/median/max `0/120/496`), and the **per-position** token-ids are banked in
`decode_outputs.jsonl` (spec-ON served + spec-OFF plain greedy AR). This leg **reconstructs
the full per-sequence flip-fraction distribution** from the per-position ids and verifies it
reproduces #114's banked split **bit-for-bit** (16/112, `0.5607757568359375`, onsets
0/120/496). **strict-A is therefore OBSERVED, not modeled.**

## The per-reading menu (the deliverable)

| Reading | Definition | Pass-fraction |
|---|---|---|
| **(A) strict per-sequence token-identity** | served sequence compliant iff **all** output tokens == plain greedy AR (zero flips) | **`strict_a_pass_fraction = 0.1250` (16/128)** — TEST headline |
| **(B) per-token-θ** | compliant iff per-sequence flip fraction ≤ θ | empirical CDF: θ=0→0.125, θ=0.05→0.141, θ=0.5→0.383, θ=0.674 (median)→0.500, θ=1→1.000 |
| **(C) PPL-only** (the auto-scorer's actual check) | compliant iff served PPL ≤ 2.42 | **`ppl_only_pass_fraction = 1.0`** (served 2.3772 ≤ 2.42) |

The three readings are nested in strictness: `strict-A (0.125) ≤ per-token-θ(θ) ≤ PPL-only (1.0)`.

## The clustering correction (denken #190/#212 within-sequence-correlation lane)

The naive **iid** intuition — model per-token flips as Bernoulli(p=0.5608) — predicts
`strict-A = (1−p)^512 = 1.14e-183 ≈ 0`: a 512-token zero-flip sequence is astronomically
unlikely under independence. **The empirical strict-A is 0.125 — ~182 orders of magnitude
higher** — because the #114 flips **cascade**: a sequence either never trips (16 prompts) or
trips once and cascades (onset median 120/512, then ~64% of remaining tokens flip). Positive
within-sequence flip correlation concentrates flips into fewer sequences → **more** zero-flip
sequences → strict-A **rises** with clustering. This is exactly the #190/#212 machinery (a
zero-flip run over L tokens under a *correlated* Bernoulli flip process), now grounded in the
**real** per-sequence data.

**Model-free bound:** for ANY within-sequence correlation structure with aggregate per-token
flip rate p, a Fréchet/union bound gives `P(zero flips) ≤ 1 − p = 0.4392` (achieved by
perfectly-nested/comonotone flips). So `strict-A ∈ (≈0 iid … 0.125 empirical … 0.4392 max]`.
**Even maximal clustering keeps strict-A < 0.5 ≪ 1** → `strict_a_robust_to_clustering = True`:
under strict-A the int4-spec stack fails for the **majority** (≥55.6%) of sequences no matter
how clustered the flips are. (A smooth Beta-Binomial would need within-sequence ICC ≈ 0.74 to
reach the empirical 0.125, far above the #190 within-prompt accept ICC of 0.145 — the cascade
is more extreme than compound symmetry.)

## Applies to both stacks

`applies_to_frontier_and_tree = True`. The deployed `fa2sw_precache_kenyan` (481.53, PR #52)
**and** the land #71 tree ride the **same** int4-Marlin spec-verify basis → the per-reading
menu applies to both; strict-A is a **frontier-wide** exposure, not tree-only.

- **(A) strict-A** → neither the 481.53 frontier nor land #71 is launch-eligible (only 12.5% of sequences pass).
- **(B) per-token-θ** → eligibility depends on the human's chosen θ; the CDF above is the menu.
- **(C) PPL-only** → both pass 100% today (served PPL 2.3772 ≤ 2.42) → launch-eligible under the auto-scorer's actual check.

## Self-test (PRIMARY) — `issue192_calibration_self_test_passes = True`

(a) iid model reproduces #114's aggregate 56.08% **and** the reconstruction is bit-exact;
(b) `strict-A ≤ pass_fraction(θ)` ∀θ>0 (monotone CDF); (c) θ=1 → 1.0; (d) θ=0 → strict-A;
(e) PPL-only = 1.0; (f) NaN-clean. All six pass.

## Hand-off (issue #192 + fern #185)

> the #192 per-reading menu from #114's 56.08%: **strict-A (per-sequence zero-flip) pass =
> 0.1250 (16/128**; the int4-spec stack fails literal token-identity for 88% of sequences —
> and even under the model-free maximal-flip-clustering cap 1−p=0.439 it stays < 0.5, so
> NO-GO-under-strict-A is robust), **per-token-θ** pass-fraction = the empirical CDF (θ=0 →
> 0.125, θ=1 → 1.0), **PPL-only** (the auto-scorer's actual check) = 100% (served PPL 2.3772
> ≤ 2.42); BOTH the 481.53 frontier and the land #71 tree share this exposure, so the human's
> A/B/C ruling determines whether EITHER is launch-eligible under strict-A — and the wirbel
> #199/#213/#216 compliant-kernel route is the only strict-A-survivable 500-path.

## Reproduce

```bash
cd target/ && CUDA_VISIBLE_DEVICES="" \
  python research/validity/issue192_reading_calibration/issue192_reading_calibration.py \
    --self-test --wandb_group issue192-reading-calibration --wandb_name denken/issue192-reading-calibration
```

CPU-only, peak ≈ 18 MiB, < 1 s. Imports: kanna #114 (`9q5yy9l1`) divergence + denken #190
(`fva6o4ug`) ICC. BASELINE 481.53 untouched. NOT a launch.
