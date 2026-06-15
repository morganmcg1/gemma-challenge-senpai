<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #336 — What head recipe could lift fusion top-4 coverage +0.031 to clear the 0.9213 bar?

**0-GPU analytic card. No training, no checkpoint, no model forward, no served-file change, 0 TPS.
Greedy/PPL untouched. BASELINE 481.53 unchanged. W&B run `krroookz`.**

## TL;DR — VERDICT: **REACHABLE-MARGINAL** (`min_aggregate_lift_required = 0.0310`)

lawine #330 (`hfrscdai`) established that the official 128 "ShareGPT" eval is 100 % reasoning/STEM, so
the honest unconditional top-4 **ROOT** coverage prior for the fern #34 (`gua9x68j`) {2,21,39} fusion
head is **0.8903**, **−0.0310 below** the 0.9213 build bar. The lane revives only if a better-trained
head delivers **+≥0.031 aggregate** coverage. This card scopes that demand-side revival target.

**+0.031 is reachable, but at the optimistic edge.** No *single* recipe central-clears the gap; the
**soft-KD top-k + reasoning-data combination** (central **+0.0385**, the only architecture-preserving
clear) does, with a thin margin and a low band (+0.0175) that misses. Even the published fully-trained
lit-central head (0.913) misses this 100 %-reasoning eval, so the clear is **not guaranteed** — a
deeper/wider head may be needed for margin. **Reachable, not a slam-dunk.**

| quantity | value |
|---|---|
| Aggregate baseline top-4 (official 57/57/14 weighting) | **0.8903** |
| Build bar | 0.9213 |
| **`min_aggregate_lift_required`** (gap, policy-independent) | **+0.0310** |
| Weighted-sum lift (gap × 128) | **3.97** |
| Cheapest clearing recipe | **soft-KD top-k + reasoning-CoT data** (combo central +0.0385) |
| **`feasibility_verdict`** | **REACHABLE-MARGINAL** |

## Deliverable 1 — per-source lift-target table (three allocation policies)

Baseline per-source unconditional top-4 (fern #34, teacher-forced; the official-eval per-source curve):
aime **0.9570** (w=14), gpqa **0.9176** (w=57), mmlu_pro **0.8465** (w=57); total 128; aggregate
**0.8903**. The bar demands an aggregate lift of **+0.0310** (= 3.97 weighted-sum). How that maps to
per-source demands depends on allocation; **all three policies re-aggregate to exactly 0.9213** (verified
≤1e-9):

| source | base | (a) mmlu_pro-only | (b) uniform | (c) proportional-to-gap |
|---|---|---|---|---|
| aime | 0.9570 | +0.0000 → 0.9570 | +0.0310 → 0.9880 | +0.0122 → 0.9692 |
| gpqa | 0.9176 | +0.0000 → 0.9176 | +0.0310 → 0.9486 | +0.0233 → 0.9409 |
| **mmlu_pro** | 0.8465 | **+0.0697 → 0.9162** | +0.0310 → 0.8776 | **+0.0434 → 0.8899** |
| **max single-source lift** | — | **0.0697** | **0.0310** | 0.0434 |

- **(a) mmlu_pro-only** concentrates the whole lift on the binding source: one **+0.0697** jump
  (0.8465 → 0.9162). The PR's "Δ≈0.070" figure. Largest single jump of the three.
- **(b) uniform** asks **+0.0310 of every source** (Δ = gap exactly, since the weights sum to 1) and is
  **minimax-optimal**: its hardest single-source lift (0.0310) is the *smallest possible* — no allocation
  that raises a weighted mean by `gap` can keep every source below `gap`. The cheapest "hardest ask".
- **(c) proportional-to-gap** lifts each source toward 1.0 by `k·(1−cov)`, `k = gap/(1−agg) = 0.2828`:
  aime +0.0122, gpqa +0.0233, **mmlu_pro +0.0434** (still the binding source). Between (a) and (b).

**No policy demands a coverage > 1.0** — every per-source target is physically reachable. The
decision-relevant reading: the **minimum aggregate lift is +0.031 regardless of allocation**; whether a
training recipe must move *one* source by 0.070 (concentrate) or *all three* by 0.031 (spread) is the
allocation knob, and uniform is the lowest-stress target.

## Deliverable 2 — diagnosing the mmlu_pro drag → **TRAINING-OBJECTIVE/DATA-LIMITED** (capacity secondary)

The binding source is mmlu_pro (0.8465, **−0.1105 below aime** under the *same* head). From lawine #330's
decomposition, the drag is the **reasoning-CoT vocabulary breadth**, **not** the answer-letter tail
(literal committed answer-letter token ≈ 0.0002 of generated tokens, ~420× too small to bind). Token
classes the head misses:

1. **multi-domain technical vocabulary** — mmlu_pro's 14 subject domains (law / philosophy / business /
   health / psychology / economics / engineering …), the highest-entropy token mix;
2. **rare / long-tail domain terms** outside the head's training distribution;
3. **LaTeX / symbolic spans and numerals** interleaved in the CoT.

**Is this head-capacity-limited or training-limited? Primarily TRAINING-limited**, on three grounds:

- **Objective.** Top-4 (not top-1) coverage is the drafter's top-k **tail**. fern #34 was trained with
  **hard cross-entropy** (argmax target), which gives **no gradient to ranks 2–4** — the tail is
  uncalibrated by construction. This is directly fixable by soft-KD.
- **Capacity headroom.** The *same* 1-layer fusion head already reaches **0.957 on aime**; the shortfall
  tracks domain **breadth** (a data-mix property), not an inherent ceiling — the head demonstrably *has*
  the capacity to cover a domain once its tokens are trained.
- **Undertraining.** fern #34 is K=1, hard-CE, no soft-KD, no TTT (the only trained head).

Capacity is a **secondary** ceiling at the very top: even the published fully-trained lit-central head
(0.913) still misses 0.9213 on a pure-reasoning eval, leaving a thin residual a deeper/wider head would
address — which is why capacity ranks #3, not #1.

## Deliverable 3 — literature-grounded recipe ranking (by expected aggregate top-4 ROOT-coverage lift)

Strict total order (descending central Δcov). **These Δcov bands are honest literature-grounded
expected-value priors a retrain would confirm — they are not measured here.**

| # | recipe | central Δcov | band | clears alone? | citation |
|---|---|---|---|---|---|
| **1** | **soft-KD / top-k logit distillation** | **+0.030** | [+0.015, +0.045] | no (central); yes (high) | DistillSpec (Zhou et al., ICLR 2024, arXiv:2310.08461); OSD (Liu et al., ICML 2024, arXiv:2310.07177); Medusa (Cai et al., ICLR 2024, arXiv:2401.10774) |
| **2** | **more on-distribution reasoning data** | +0.025 | [+0.010, +0.040] | no | EAGLE-3 (Li et al., 2025, arXiv:2503.01840) |
| **3** | deeper/wider fusion head | +0.012 | [+0.005, +0.020] | no | Medusa / EAGLE-3 (directional only — no published width ablation) |
| **4** | **on-policy TTT** | +0.002 | [+0.000, +0.005] | no | **lawine #316 (internal measurement)** |

1. **soft-KD / top-k distillation — PRIMARY.** The bar *is* the drafter's top-k tail accuracy; hard-CE
   leaves ranks 2–4 uncalibrated, and matching the teacher top-k softmax trains exactly that tail.
   DistillSpec measures general draft-KD acceptance gains (10–45 % speedup); the **top-4-tail-specific
   attribution is our reasoned inference, not a per-paper ablation** (flagged honestly).
2. **more reasoning data — targets the binding source.** EAGLE-3 removed EAGLE-2's feature-prediction
   constraint specifically so the head "fully benefits from scaling up training data"; adding
   mmlu_pro/gpqa CoT directly covers mmlu_pro's breadth (weight 0.445, lowest coverage). Ranked below
   soft-KD because data-mix shifts raise top-1 more reliably than the top-4 tail per se.
3. **deeper/wider head — secondary capacity.** Lifts the ceiling for the broadest domains but capacity
   is not the binding constraint (the head already hits 0.957 on aime), and it costs per-step draft
   latency + VRAM (ubel #299/#306: current head fits 24 GB at 20.16 GiB / 3.84 GiB headroom). **No
   published head width/depth ablation exists — directional only.**
4. **on-policy TTT — RANKED LOWEST (wrong axis).** lawine #316 **measured** that TTT lifts depth≥2
   (continuation) acceptance, **not** the ROOT token; the 0.9213 bar is a top-4 **ROOT** bar, so TTT's
   expected lift here is ≈0. The per-position split is *our* internal measurement — external lit reports
   only aggregate acceptance and does not ablate root-vs-continuation. (EAGLE-3's "TTT" = *training-time
   test*, a positive training technique, distinct from this test-time-training recipe.)

**Which combination clears +0.031?** No single lever central-clears. The two **training** levers
(soft-KD + reasoning-data) are complementary (objective tail-shaping vs data breadth). Naive sum +0.055;
under a conservative **0.70 non-additivity haircut** (they partly target the same mmlu_pro shortfall, and
the head saturates) → **combo central +0.0385**, which **clears** the +0.0310 gap with a thin margin.
Band [+0.0175, +0.0595]: central and high clear, the low (both levers at their floors) **misses**. If the
training levers land low, add the deeper/wider head (#3, +0.012 central) for margin and to lift the
residual ceiling.

## Deliverable 4 — feasibility verdict

**REACHABLE-MARGINAL.** +0.031 is reachable with the **soft-KD top-k + reasoning-CoT data combination**
(central +0.0385 clears), but at the optimistic edge: a single lever (soft-KD central +0.030) lands just
under, the combination's low band (+0.0175) misses, and the published fully-trained lit-central head
(0.913) still misses this 100 %-reasoning eval. **Not out of reach, not a slam-dunk.**

- **Cheapest clearing recipe:** soft-KD top-k distillation + reasoning-CoT data augmentation on the
  **existing {2,21,39} fusion architecture** — *no capacity change → same VRAM, same deploy/latency
  path* (ubel #299/#306 24 GB fit preserved; only the drafter weights change).
- **Cost to validate:** **one cluster drafter retrain** (open `instructions/training-request.md`, **not**
  a TPS run) → **re-measure** per-source unconditional top-4 coverage on the 240-record benchmark-matched
  reasoning holdout via the fern #34 / lawine #330 RANKPROBE_W=4 protocol (CPU/1-GPU coverage eval, **no
  HF Job, 0 TPS**). Gate: retrained aggregate ≥ 0.9213 flips the demand-half **GO** for the fern #335
  joint AND-gate.

## Self-test — **23/23 PASS**, NaN-clean, deterministic, peak mem 15.2 MiB

Reproduces aggregate 0.8903 / gap 0.031 / weighted-sum 3.97 from per-source × weights; verifies the three
allocation policies each re-aggregate to exactly 0.9213 (≤1e-9), no policy demands cov>1.0, mmlu_pro is
binding, uniform is minimax-optimal; the recipe ranking is a strict total order with soft-KD #1 and TTT
last carrying the #316 root-vs-deep rationale; the combination central-clears while every single lever is
marginal; the verdict is REACHABLE-MARGINAL; and the imported per-source constants match the on-disk
lawine #330 artifact (drift ≤1e-6).

## Honesty / caveats

PRIOR / **feasibility scoping, not a measurement**. The per-recipe Δcov bands are literature-grounded
expected-value priors — no training, no model forward here. The 0.9213 bar is position-1 **ROOT**
coverage: necessary, **not sufficient** — the deployed deep spine still caps E[T] (#316), and the
private-tax robustness (fern #325 pass-(a)) is a **separate** gate. This card scopes the **demand-half**
revival target only; the fern #335 joint AND-gate folds in both halves. fern #34 is the only trained head
and is undertrained; no fusion checkpoint is deployed and the drafter build stays human-gated. 0 TPS;
greedy/PPL untouched; BASELINE 481.53 unchanged; no checkpoint, no publish, no HF Job.

## Public evidence used

- **lawine #330** (`hfrscdai`, MERGED `research/validity/eagle3_sharegpt_coverage_prior/`): the
  per-source coverage decomposition (aime 0.9570 / gpqa 0.9176 / mmlu_pro 0.8465 → aggregate 0.8903), the
  −0.031 gap, the reasoning/STEM misnomer, and the "what would flip it = a better head lifting per-source
  coverage by ≥0.031" open question this card answers. Cross-checked live against its on-disk results
  JSON (drift ≤1e-6). *Extending* this banked verdict.
- **fern #34** (`gua9x68j`, train `56ksyxgw`): the only trained {2,21,39} fusion head; per-source
  unconditional top-4 + native a₁=0.7714 provenance.
- **lawine #316** (`5lnz5jgb`): the build bar (0.9213) and the TTT depth≥2-not-root measurement that
  ranks recipe #4 low.
- **Literature** (verified): DistillSpec (arXiv:2310.08461), Online Speculative Decoding
  (arXiv:2310.07177), Medusa (arXiv:2401.10774), EAGLE-3 (arXiv:2503.01840).
- **ubel #299/#306**: the 24 GB VRAM fit that the architecture-preserving cheapest recipe keeps.

## Reproduce

```bash
cd target/ && .venv/bin/python \
    research/validity/eagle3_head_coverage_lift_target/eagle3_head_coverage_lift_target.py \
    --self-test --wandb_group eagle3-head-coverage-lift-target \
    --wandb_name lawine/eagle3-head-coverage-lift-target
```

Primary `coverage_lift_target_self_test_passes = True` (23/23). Test `min_aggregate_lift_required =
0.0310`. Report `feasibility_verdict = REACHABLE-MARGINAL`. W&B run `krroookz`.
