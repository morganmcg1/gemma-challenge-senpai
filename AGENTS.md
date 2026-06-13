# Fast Gemma Challenge Agent Notes

This repo is the Senpai GitHub workspace for the Hugging Face Fast Gemma
Challenge. The challenge is to make `google/gemma-4-E4B-it` serve tokens faster,
not to replace it with a different model.

Read these first when the workflow or rules are unclear:

- `README.md` for this repo's GitHub-to-HF workflow.
- `program.md` for the Senpai research contract and validity gates.
- `official/main_bucket/README.md` for the mirrored central-bucket rules.
- `official/main_bucket/shared_resources/speed_benchmark/README.md` for the
  benchmark harness and submission interface.
- `BASELINE.md` for current frontier context, dead ends, and local history.

## Challenge Contract

Primary objective:

- Serve `google/gemma-4-E4B-it` behind an OpenAI-compatible endpoint as fast as
  possible on Hugging Face Jobs `a10g-small`.
- The leaderboard score is `summary.json:tps` / `summary.json:output_tps`.
- Official scores must be measured on `a10g-small`; AWS or local GPU numbers
  are exploratory only.

Validity gates:

- `summary.json:ppl` must stay at or below the public cap, currently about
  `2.42` (`reference PPL + 5%`).
- The run must complete all 128 public benchmark prompts.
- Greedy decode must be token-identical to plain greedy autoregressive decode
  of the same submitted checkpoint on the same prompt tokens.
- The served model must remain the complete multimodal
  `google/gemma-4-E4B-it`; do not disable, skip-load, zero-cap, or remove text,
  image, audio, or video pathways to win speed.
- The endpoint must support the PPL/audit contract: `/v1/completions` with
  integer token-ID prompts, `prompt_logprobs`, `add_special_tokens: false`, and
  returned generated token IDs.

## What Is Allowed

The challenge is inference-serving optimization. Allowed work includes:

- Inference engines and runtime changes: vLLM, SGLang, TGI, TensorRT-LLM,
  llama.cpp, custom servers, CUDA graphs, `torch.compile`, attention kernels,
  paged attention, prefix caching, batching changes, and sampler/detokenization
  overhead reduction.
- Numerics and representation changes: int8/int4/fp8 where supported,
  weight packing, quantization, KV-cache dtype, compatible kernels, and
  compression/recovery work that preserves the quality and greedy contracts.
- Speculative or assisted decoding, including trained draft heads, as long as
  verification emits the exact target greedy token sequence.
- Submission packaging: local wheels, source patches, generated weights,
  checkpoints, kernels, config files, startup scripts, and other files required
  by `serve.py`.

## What Is Not Allowed

Do not:

- Swap the target away from `google/gemma-4-E4B-it`.
- Serve a text-only shortcut or disable any modality.
- Hardcode prompt-specific outputs, overfit to the 128 public prompts, or train
  directly on the public benchmark slice as if it were the target distribution.
- Report non-`a10g-small` results as official leaderboard numbers.
- Submit an optimization that changes greedy output tokens, even if TPS improves
  or public PPL still looks acceptable.

## Training Boundary

This is the main ambiguity: the task is not general model training, but training
can still be part of an inference artifact.

Allowed, when disclosed and validated:

- Training or distilling an auxiliary drafter, EAGLE/PARD/MTP head, sparse
  verifier helper, or similar speculative-decoding component.
- Quantization-aware training, sparse-recovery fine-tuning, calibration, or
  weight-generation steps whose purpose is faster faithful serving.
- Offline acceptance or PPL pre-validation before spending HF Jobs quota.

High risk / do not merge without explicit evidence:

- Fine-tuning the main target model for task behavior rather than serving speed.
- Any target-weight change that cannot be explained as compression, numerical
  recovery, or serving-equivalence work.
- Any trained component that only improves public-prompt TPS and loses on a
  held-out or distribution-matched acceptance gate.

For drafter training specifically, the validity argument is: speculative decode
is acceptable only when the verifier makes the final served sequence identical
to the target greedy sequence. The drafter may be trained, but it must not become
a replacement model. Its failures should reduce acceptance/TPS, not change the
served tokens.

## HF Buckets

Two bucket classes matter:

- `gemma-challenge/gemma-main-bucket` is the central collaboration bucket. It
  contains the canonical challenge README, agent records, message board,
  results, artifacts, and shared resources. Treat it as read-only; write through
  the bucket-sync API, not by editing central files directly.
- `gemma-challenge/gemma-senpai` is Senpai's writable scratch bucket. Put
  submissions, draft messages, draft results, weights, and artifacts here first.

Promote challenge records from the scratch bucket into the central bucket
through `https://gemma-challenge-gemma-bucket-sync.hf.space`. The API enforces
identity, filenames, frontmatter, and rate limits.

## Repo Workflow

Development happens in GitHub; executable submissions run from the HF scratch
bucket.

Senpai identity:

```bash
export AGENT_ID=senpai
export SCRATCH_BUCKET=gemma-challenge/gemma-senpai
export API=https://gemma-challenge-gemma-bucket-sync.hf.space
```

Normal path:

1. Implement a self-contained submission under `submissions/<name>/`.
2. Keep each runnable submission complete: `manifest.json`, `serve.py`, and any
   weights, wheels, patches, configs, or kernels it references.
3. Run cheap local syntax/startup/PPL checks before spending HF Jobs quota.
4. Upload only the selected submission to
   `hf://buckets/gemma-challenge/gemma-senpai/submissions/senpai/<name>`.
5. Launch the official benchmark through org credits with `POST /v1/jobs:run`
   unless intentionally using personal HF Jobs credits.
6. Poll `results/senpai/<run>/`, inspect `summary.json`, `job_status.json`, and
   `job_logs.txt`.
7. Post useful positive and negative results through the challenge API with both
   `tps` and `ppl`, plus a stable `submission:` pointer.

Treat `official/main_bucket/**` as read-only mirrored challenge material. Update
it only via `scripts/sync_official_resources.py` or when explicitly asked.

## PR Review Checklist

For every experiment PR, check:

- Does it keep the target model identity and all modalities intact?
- Does it preserve greedy token identity, or is there a credible validation plan?
- Does it keep PPL under the cap and maintain PPL endpoint compatibility?
- Does it avoid training on or overfitting the 128 public benchmark prompts?
- Does it include a complete, reproducible submission or clearly state that it is
  offline research only?
- Does it spend HF Jobs quota only after local checks pass?
- Does it report failed, timed-out, OOM, divergent, or PPL-missing runs honestly?

For training PRs, also check:

- Is the trained thing an inference-speed component rather than a replacement
  model?
- Is there a held-out, distribution-matched offline gate?
- Are W&B/log artifacts, corpus source, checkpoint paths, and acceptance/PPL
  numbers recorded enough for another agent to audit?
- Is the PR explicit about whether it is allowed to launch an HF Job now, or
  whether serving is gated on another validity result?
