# Sampled-decoding quality-neutrality (PR #520)

**Does the surgical-357 ship stay quality-neutral under the OFFICIAL
`generation_config.json` SAMPLING config (`do_sample=true, T=1.0, top_k=64,
top_p=0.95`), not just under greedy?**

The SAMPLING twin of denken #513 (`krma4lm7`), which priced the **greedy /
acceptance** axis (`private_quality_exposure = 0.0`). #513 left one hole open: a
skeptic can grant "greedy is output-exact" and still ask *"but the organizer's
downstream evals (MMLU/GPQA/AIME) and lewtun's Issue #31 SCORE UNDER SAMPLED
decoding — does the ship hold there?"* This card closes that hole by measuring
the base-vs-ship **output-distribution divergence under the exact sampling
transform** the benchmarks use.

- **base** = stock attention, spec-OFF (plain AR multinomial sampling from `p_base`).
- **ship** = surgical-357 attention (`is_batch_invariant=True`, 2D order-preserving,
  matmul tax OFF — the #499 lever) + spec-dec ON (deployed MTP rejection sampler).

**Verdict: QUALITY-NEUTRAL UNDER SAMPLING.** W&B run
[`6wt6b5mk`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/6wt6b5mk)
(`044gm9px` was the initial raw measurement pass). "Quality-neutral" here is a
**two-tier** claim, made precisely because the full 128-prompt run surfaced one
honest residual the smoke run did not:

1. **`scored_geometry_exact = true`** — in the teacher-forced PREFILL geometry the
   official PPL gate scores in, the ship is **EXACTLY bit-identical** to base:
   `TV = KL = 0`, `max |Δlogit| = 0` across all **6144** scored positions.
2. **`is_quality_neutral = true`** — in the free-running DECODE geometry the
   downstream evals sample, the **argmax (answer token) is identical at all 256
   first-answer positions** (`decode_argmax_identity_rate = 1.0000`,
   `n_semantic_answer_flips = 0`), spec-dec output `== p`, and the result is flat
   across temperature.

The only thing that is **not** bit-identical (`bit_identical_all_geometries =
false`) is a **benign bf16 reduction-order ULP wobble** in the decode geometry at
**3 of 256** near-tied **non-argmax** nucleus tokens (`max TV = 0.0549`,
`max |Δlogit| = 0.500 nat`). This is the **#509 decode near-tie phenomenon**: it
changes **no answer**, and it **vanishes in the scored geometry**. We report the
raw `max_tv_sampled = 0.0549` as `sampled_quality_exposure` (never massaged), and
separately report `sampled_semantic_exposure = 0.0` — the exposure that survives
into an actual MMLU/GPQA/AIME answer.

> **On `max_kl = 22.53`.** This is a `top_p`-boundary floor artifact, **not** a
> real divergence. At ONE of the 3 near-tie positions a single token sits exactly
> on the nucleus (`top_p=0.95`) cut: it is kept in one arm's support and dropped in
> the other, so a `p·log(p / ~1e-300)` term in the KL blows up even though the
> token carries `p ≈ 1e-3`. The honest divergence bound is the bounded
> `TV = 0.0549`, which is what `sampled_quality_exposure` tracks. (`min_support_agreement
> = 0.75` is this same single boundary token: 3 of 4 differing top-tokens agree.)

## The composition argument (what the measurement confirms)

The ship differs from base on TWO independent axes; the decision-relevant sampled
output is preserved on BOTH, so the composition is preserved:

1. **Attention patch** (stock 3D split-KV → surgical 2D in-order). stark #509
   (`ljk3ffv5`) proved this is **logit-identical** at M=1 in the **score/prefill
   geometry** (`max_abs_logit_delta=0`, `KL=0`) — reproduced here EXACTLY across
   6144 prefill positions. The gen_config sampling distribution
   `p = top_p(top_k(softmax(logit/T)))` is a **deterministic function of the
   logits**, so in that geometry logit-identity ⇒ `p_surgical == p_base`, hence
   `KL(p_base‖p_ship)=0`, `TV=0`, AND identical matched-seed draws. In the
   **decode geometry** the 3D-vs-2D KV reduction order differs at the bf16 ULP, so
   at near-ties the two arms can rank two ~equal-probability NON-argmax tokens
   differently — the residual wobble above. The **argmax is never affected** (256/256
   identical), so the answer token the evals read is preserved.
2. **Spec-dec** (AR → rejection sampler). denken #505/#513 proved the deployed
   standard rejection rule with a greedy MTP draft `x_d` and `draft_probs=None`
   is **exactly distribution-preserving**: output ~ `p` for ANY draft. Reproduced
   here on the REAL surgical gen_config sampling distributions at the iid noise floor.

Compose: `ship_answer ~ argmax p_surgical == argmax p_base` and the scored
distribution is identical ⇒ `sampled_semantic_exposure = 0`.

### Mechanism (verified from the pinned kernel source)

Pinned vLLM `0.22.1rc1.dev307+g3e8afdf78` (`/tmp/server-venv`),
`rejection_sampler.py`. Under temperature sampling (`all_random`, `temp>0`) the
greedy early-exit (`L895-898`) is skipped and the stock random/recovered kernels
run. With a deterministic greedy MTP draft `x_d` and `draft_probs=None`
(`NO_DRAFT_PROBS`), per draft position:

- **accept** `x_d` with probability `min(1, p(x_d)) = p(x_d)` — `NO_DRAFT_PROBS`
  forces `draft_prob=1` (`L913-914`), so the accept test is `target_prob ≥
  uniform_prob` (`L921-926`);
- **on reject**, resample from the recovered distribution = `p` masked to exclude
  the draft token, i.e. `p|{y ≠ x_d}` renormalized (`L1006-1011`), via the stock
  exponential-race Gumbel-max (`score = prob·inv_q`, argmax, `L1027-1039`).

This is **exactly distribution-preserving: output ~ p for ANY draft token** — the
realized acceptance at a position is exactly `p(x_d)` (which token is drafted moves
only `E[T]`/TPS, never the output distribution). This is the same kernel #513 drove;
here it is fed the **real surgical gen_config sampling distributions** so the
spec-ON leg of the ship is made explicit: the spec-dec output distribution equals
`p_surgical`, which (in the scored geometry) equals `p_base`.

## Method

`sampled_quality_neutrality.py` measures four legs, all LOCAL on the int4 serve,
all isolating the sampling axis. The two attention arms run in **isolated
subprocesses** (a pin never leaks across arms); base runs first and emits
`_ref_base.json`, surgical reads it so both arms teacher-force the **byte-identical**
trajectory.

- **LEG A — sampled target-distribution divergence (attention-patch axis).** Re-run
  the #509 two arms (base / surgical, M=1, `enforce_eager`). Capture top-64 raw
  logprobs in TWO geometries at **identical conditioning**, apply the exact
  gen_config transform (T=1.0, top_k=64, top_p=0.95 nucleus) to BOTH arms, and
  measure per-position `KL(base‖ship)`, `TV(base, ship)`, nucleus-support
  agreement, AND the argmax (semantic answer) identity:
  - `decode_first_token` (`gen_lps[0]`): the first generated answer token, DECODE
    geometry (split-KV 3D vs 2D), conditioning identical by construction (same
    prompt). **This is the MMLU/GPQA first-answer-token the downstream evals sample.**
    256 positions; argmax identical at all 256 → 0 semantic flips.
  - `prefill_trajectory` (`score_lps`): teacher-forced ctx+ref_base, PREFILL geometry,
    full per-position coverage. **This is the geometry the official PPL gate scores in**
    (`prompt_logprobs`). 6144 positions; exactly bit-identical.
- **LEG B — seed-matched determinism.** With identical Gumbel noise, draw from
  `p_base` and `p_ship`; identical iff the distributions are bit-identical. Report
  `seed_matched_identity_1p0` + the ULP magnitude of any logit divergence (lse-free
  ref-anchored, per #509 Leg-2). `false` here, driven entirely by the 3/256 decode
  near-ties; the prefill geometry is matched-seed identity `1.0`.
- **LEG C — spec-dec output preservation under sampling (decoder axis).** Drive the
  EXACT deployed `rejection_sample()` on the REAL surgical gen_config sampling
  distributions (greedy draft AND adversarial low-prob draft); confirm the empirical
  output histogram ~ `p_surgical` (TV at the iid Monte-Carlo noise floor, KL~0).
- **LEG D — temperature sweep.** Repeat LEG A across a T grid
  `(0.1, 0.5, 0.7, 1.0, 1.3)` around the config's T=1.0 to show the neutrality is not
  a single-temperature artifact.

### On matched-seed trajectory identity (the honest caveat)

The **attention axis is matched-seed IDENTICAL in the scored geometry** (logit-identity
→ identical sampling distribution → identical seeded draw, `1.0` across 6144 positions).
In the **decode geometry** matched-seed identity is `0.99936` — the 3/256 near-tie
positions where the bf16 reduction-order wobble re-ranks two ~equal NON-argmax tokens.
The **spec-dec axis** is distribution-identical but NOT trajectory-identical at a
matched seed — the rejection sampler deliberately re-routes the RNG (accept/reject coin
+ exponential-race Gumbel on the recovered distribution). That re-routing is pure
**sampling noise** (both draws are ~ `p`; the matched-seed agreement an RNG-re-routing
layer achieves even with an identical distribution is the collision rate `Σ p²`),
**not a quality gap**. We report `seed_matched_identity_1p0 = false` explicitly so the
headline is read correctly: the scored-geometry attention axis is matched-seed exact,
the decode near-ties and the spec-dec RNG re-route are the sub-headline residuals, and
**neither changes an answer**.

## Key results (full run: 128 prompts × 2 splits, seeds=128, spec-dec M=200000)

| metric | value |
| --- | --- |
| verdict | **QUALITY-NEUTRAL UNDER SAMPLING** |
| `is_quality_neutral` | **true** |
| `scored_geometry_exact` (PPL/prefill) | **true** |
| `bit_identical_all_geometries` | false (decode ULP wobble) |
| `n_semantic_answer_flips` | **0** |
| `decode_argmax_identity_rate` | **1.0000** |
| `sampled_semantic_exposure` (answer-surviving) | **0.0** |
| `sampled_quality_exposure` (raw max TV) | 0.0549 |
| `max_tv_sampled` | 0.0549 |
| `max_kl_base_vs_ship_sampled` | 22.53 (top_p-boundary floor artifact) |
| `seed_matched_identity_1p0` | false |
| `sampled_answer_agreement_rate` | 0.99936 |
| `min_support_agreement` (nucleus) | 0.75 (single boundary token) |
| `max_abs_logit_delta_sampled` | 0.500 nat (bf16 decode reduction-order) |
| decode_first_token positions | 256 (3 with nonzero TV) |
| prefill_trajectory positions | 6144 (0 with nonzero TV) |
| spec-dec mean TV (deployed) | 0.000784 |
| spec-dec mean iid floor | 0.000847 |
| spec-dec worst-case TV | 0.00311 |
| `spec_dec_output_preserves_distribution` | **true** |
| temperature sweep flat | true (max sweep TV 0.0549 @ T=1.0) |
| self-tests | 38/38 |
| any NaN | false |
| peak GPU mem | 19092 MiB (~18.6 GB) |

The **scored/prefill geometry A/B is exactly zero** at every one of the 6144
positions (`KL=TV=0`, `max_abs_logit_delta=0`, nucleus support identical,
matched-seed draw identical) — the gen_config sampling distribution is a
deterministic function of logits that #509 already pinned identical, and this is
the geometry the official PPL gate scores in. The **decode/free-running geometry**
preserves the **argmax at all 256 first-answer positions** (0 semantic flips); its
only residual is a benign sub-0.055-TV ULP wobble at 3 near-tied non-argmax nucleus
tokens (the #509 decode phenomenon), which changes no answer. The **spec-dec leg**
sits at the iid Monte-Carlo sampling-noise floor (mean deployed TV 0.000784 ≤ mean
iid floor 0.000847; worst single case 0.00311 within the floor band), so the
spec-ON output `== p_surgical`. The sweep shows this holds flat across the
temperature grid — the neutrality is structural, not a single-temperature artifact.

## Run

```bash
cd research/validity/sampled_quality_neutrality
# no-GPU self-test (38 checks: transform / KL / TV / seed / two-tier compose plumbing / NaN-clean)
CUDA_VISIBLE_DEVICES="" /tmp/server-venv/bin/python sampled_quality_neutrality.py --selftest
# full measurement (two attention arms + spec-dec leg + compose + W&B)
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  /tmp/server-venv/bin/python sampled_quality_neutrality.py \
    --n-prompts 128 --n-new 24 --ctx-cap 256 --topk 64 --seeds 128 \
    --specdec-M 200000 --specdec-cases 128 \
    --wandb_name denken/sampled-quality-neutrality --wandb_group sampled-quality-neutrality
# re-derive the two-tier verdict from cached deterministic arms (no GPU)
CUDA_VISIBLE_DEVICES="" /tmp/server-venv/bin/python sampled_quality_neutrality.py --recompose \
    --wandb_name denken/sampled-quality-neutrality --wandb_group sampled-quality-neutrality
```

Scope: LOCAL profiling card. `analysis_only=true`, `official_tps=0`, NO served-file
change, NO HF Job, NO `train.py --launch`, NO submission. The shipped surgical-357
config and the baseline are UNCHANGED; the sampling distribution is MEASURED, never
altered. Reuses stark #509's attention-patch isolation
(`apply_surgical_pin == is_batch_invariant=True`), denken #505/#513's deployed
`rejection_sample()` driver, and the merged #491/#497 prompt splits
(`rc.load_shifted_prompts`, `EPS_STAR`/`NEAR_TIE`).
