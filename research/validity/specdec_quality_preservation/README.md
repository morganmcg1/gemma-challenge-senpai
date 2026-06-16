# PR #505 — Spec-dec sampled-distribution quality preservation under sampling

**Analysis-only** (`analysis_only=true`, `official_tps=0`, no served-file change,
no HF job, no submission). W&B group `specdec-quality-preservation`, run
[`bg03bq0d`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/bg03bq0d).

## Question

Surgical-357 (PR #499) is **spec-alive** (MTP K=7 drafter). Spec-dec is provably
lossless for *greedy*. The organizer's downstream eval (MMLU/GPQA/AIME) may run
with the model's `generation_config.json` **sampling** params (lewtun #31). Does
the deployed spec path preserve the *sampled* output distribution, or only the
greedy one?

`google/gemma-4-E4B-it` `generation_config.json`: `do_sample:true,
temperature:1.0, top_k:64, top_p:0.95` — real sampling.

## Leg 1 — deployed acceptance rule (code inspection)

vLLM `0.22.1rc1.dev307+g3e8afdf78` (the manifest's pinned wheel). The serve.py
dixie SMP-02 patch modifies `vllm/v1/sample/rejection_sampler.py` but **only adds
an `all_greedy` fast-path short-circuit**; `rejection_sample()`, the random
kernel, and the recovered kernel are stock.

Two paths, selected by `sampling_metadata.all_greedy`:

1. **temp=0 / all_greedy** (leaderboard + manifest `OVERRIDE_GENERATION_CONFIG`
   default): `rejection_greedy_sample_kernel` — `token = target_argmax;
   rejected = draft != target_argmax`. Pure **greedy-verify** → emits target's
   greedy tokens exactly (the challenge greedy-identity gate).
2. **temp>0 / sampling**: `all_greedy=False` skips the dixie fast-path and falls
   through to stock `rejection_sample()`. vLLM defaults
   `rejection_sample_method="standard"`, `draft_sample_method="greedy"` (manifest
   does not override) ⇒ `_enable_probabilistic_draft_probs=False` ⇒ MTP draft is
   **greedy** (deterministic), `draft_probs=None` (`NO_DRAFT_PROBS`):
   - accept the drafted token `x_d` w.p. `min(1, p(x_d))` (random kernel:810,
     `draft_prob=1`),
   - on reject, resample from `p` **restricted to `≠ x_d`** (recovered kernel:889,
     `vocab_offset != draft_token_id`).
   This is the **standard speculative-sampling / Leviathan-Chen rule specialized
   to a deterministic draft**, which is **exactly distribution-preserving**:
   `output ~ p` for any temp/top_k/top_p, regardless of drafter quality (drafter
   quality only affects acceptance rate = speed; vLLM source comment
   `llm_base_proposer.py:1677-1680` states it "does not affect the distribution
   of the generated tokens after rejection sampling").

⇒ acceptance rule is **NOT greedy-only**. Under sampling it is the
distribution-matching standard rejection sampler.

## Leg 2 — empirical (drives the EXACT deployed kernel)

`specdec_dist_preserve.py` calls the deployed `rejection_sample()` (pinned
server-venv) in the deployed MTP config (`draft_probs=None`, greedy draft,
standard rejection), histograms the first emitted token over many trials, and
compares to target `p` via TV and `KL(p‖p̂)` against the i.i.d. Monte-Carlo noise
floor. Also tested with an **adversarial** (rare-token) draft to show the rule
corrects any draft.

- **Synthetic** (M=200k; confident-MCQ, two-way, graded-4way, power-law, broad):
  mean TV deployed `0.00398` vs noise floor `0.00379`; mean KL `1.6e-4`.
  `accept_rate_top ≈ p_max` (the `min(1,p)` signature).
- **Real #497 reasoning_stem** (`extract_real_logits.py` → 128 first-token
  sampling dists from the deployed int4 target at T=1.0/top_k=64/top_p=0.95;
  support median 4, p_max median 0.59 — the MMLU/GPQA answer regime; M=50k):
  **mean TV deployed `0.00219` ≤ noise floor `0.00235`**, max TV `0.0062`,
  **mean KL `2.7e-5`**, max KL `1.6e-4`. The deployed-spec output distribution is
  statistically indistinguishable from sampling directly from the target.

## Verdict

`quality_preserving_verdict = preserving`. The spec-alive surgical-357 serve is
**sampled-distribution-preserving** under temperature. Downstream-eval exposure
from the spec-dec acceptance rule = **0**: whether MMLU/GPQA/AIME is scored greedy
(logprob-argmax, typical) or sampled at the generation_config params, the spec
path emits exactly the target model's distribution.

**Boundary (vs stark `ship-quality-safety-proof`):** this leg proves the spec-dec
*decoding-algorithm* reproduces the target's own distribution `p`. Whether `p`
itself equals base gemma's distribution (int4-quant / lm_head-prune / model-config
identity) is the **separate** model-config axis (stark). The manifest also forces
temp=0 by default via `OVERRIDE_GENERATION_CONFIG`, a generation-config choice
that is likewise orthogonal to the acceptance rule.

## Files
- `specdec_dist_preserve.py` — deployed-kernel TV/KL harness.
- `extract_real_logits.py` — #497 reasoning first-token sampling dists.
- `synthetic_results.json`, `real_reasoning_results.json`, `real_reasoning_p.pt`.
