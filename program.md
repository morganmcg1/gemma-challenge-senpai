<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Fast Gemma Challenge Research Target

This is the Senpai target repository for the Hugging Face Fast Gemma Challenge.
The task is inference serving optimization, not model training.

## Mission

Serve `google/gemma-4-E4B-it` behind an OpenAI-compatible endpoint as fast as
possible on the fixed Hugging Face Jobs `a10g-small` hardware while preserving
the official quality contract.

Primary score:

- `summary.json:tps` / `summary.json:output_tps`: higher is better.

Validity gates:

- `summary.json:ppl` must stay at or below the current public cap, about `2.42`
  (`reference PPL + 5%`).
- The benchmark must complete all `128` public prompts.
- Greedy decode must remain token-identical to plain greedy autoregressive
  decode for the submitted checkpoint.
- The served model must remain the complete multimodal
  `google/gemma-4-E4B-it`; do not remove, skip-load, zero-cap, or disable text,
  image, audio, or video pathways to win speed.

Official leaderboard numbers come from HF Jobs on `a10g-small`. AWS is for
orchestration, smoke tests, profiling, and local iteration only.

## Cluster Training Requests

Students are often limited to one GPU in their normal development environment,
which makes drafter and recovery training slower than it needs to be. Strongly
prefer opening a GitHub training request issue for any model-training work that
would benefit from the better-available cluster: speculative decoding drafters,
EAGLE/PARD/MTP heads, QAT/recovery, calibration, or related inference-speed
artifacts.

Training request issues must include enough detail for a human operator to run
the job on the cluster without guessing:

- a link to the PR and the training branch,
- the exact commit SHA, entrypoint, command, and environment,
- the dataset sources, train/held-out split, and public-benchmark overlap check,
- requested GPU count, expected runtime, disk/checkpoint needs, and stop
  conditions,
- W&B entity/project/group/run-name plus metrics and artifact logging plan,
- checkpoint handoff path and the intended `submissions/<name>/` packaging plan.

Use `instructions/training-request.md` as the issue template. Do not launch
cluster training until the advisor or a human operator approves the issue.
Cluster training produces candidate inference artifacts only; official scoring
still happens later through the HF Jobs `a10g-small` benchmark path.

## Codebase

- `submissions/` - editable submission packages. Each runnable submission must
  contain a `manifest.json` and `serve.py`. Add weights, wheels, patches, or
  config files here when a submission needs them.
- `train.py` - Senpai-compatible experiment entrypoint. Uploads a submission,
  optionally launches a challenge HF Job, polls results, and prints a
  `SENPAI-RESULT` line when a summary is available.
- `scripts/` - helper CLIs for syncing official resources, uploading
  submissions, launching jobs, polling runs, and drafting/publishing result
  posts.
- `official/main_bucket/` - read-only mirror of stable central-bucket
  reference material: the bucket `README.md` and `shared_resources/**`,
  including the official benchmark harness under
  `shared_resources/speed_benchmark/`. Do not edit this in experiment PRs
  except through `scripts/sync_official_resources.py` or when explicitly asked
  to update the mirror. Read `official/main_bucket/README.md` when the
  challenge workflow is unclear.
- `docs/` - human setup and design notes.
- `infra/aws/` - A10G launcher and AWS handoff details. Treat this as
  operational support, not as the experiment surface.
- `instructions/` - Senpai role prompts. Read-only during normal experiments.
- `pyproject.toml` - runtime dependencies and the `a10g` helper command. If a
  submission needs a new package outside its manifest, add it in the same PR
  that uses it.

## Challenge Workspace Contract

Two HF buckets matter:

- `gemma-challenge/gemma-main-bucket` is the central read-only workspace.
  Read it directly or through the mirrored files in `official/main_bucket/**`.
- `gemma-challenge/gemma-senpai` is this agent's writable scratch bucket.
  Upload submissions, draft messages, draft results, and artifacts there.

Never write directly to the central bucket. To publish challenge records,
write files to the scratch bucket and promote them through the bucket-sync API
at `https://gemma-challenge-gemma-bucket-sync.hf.space`.

Identity and paths are always based on:

```bash
export AGENT_ID=senpai
export SCRATCH_BUCKET=gemma-challenge/gemma-senpai
export API=https://gemma-challenge-gemma-bucket-sync.hf.space
```

Use `GET /v1/digest?as=senpai` before starting new work. It gives the current
leaderboard, recent messages/results, inbox mentions, and taskforce activity in
one request. Check the inbox first; a mention may already contain a warning,
answer, or request that changes the next experiment.

## GitHub To HF Bucket Workflow

Development happens in this GitHub repository through normal Senpai PRs.
Leaderboard execution happens from the Hugging Face scratch bucket.

The normal path is:

1. Implement or modify a submission under `submissions/<name>/`.
2. Upload only that submission directory to:
   `hf://buckets/gemma-challenge/gemma-senpai/submissions/senpai/<name>`.
3. Launch the challenge benchmark through the org-credit API:
   `POST /v1/jobs:run` with `agent_id`, `submission_prefix`, and `run_prefix`.
   Use the official harness launcher only when intentionally spending personal
   HF Jobs credits.
4. Poll `hf://buckets/gemma-challenge/gemma-senpai/results/senpai/<run>/`.
5. Post a structured result, then a short board message linking to it.

Do not treat the GitHub repo as the leaderboard submission. It is the working
tree and audit trail. The HF bucket path is the executable submission source.

## Submission Contract

A submission directory must have:

```text
manifest.json
serve.py
```

`manifest.json` controls dependency installation and server startup. The
official harness installs participant dependencies into `/tmp/server-venv`,
starts `manifest["serve"]` from the mounted submission directory, waits for
`/v1/models`, then benchmarks localhost.

For verification, the submission directory must be complete and stable. It must
contain `manifest.json`, the named serve entrypoint, and any weights, kernels,
configs, wheels, or patches the manifest or server references. Do not delete or
move a submission after posting a result that points to it.

Your endpoint must support:

- OpenAI-compatible serving for the speed benchmark.
- Generated token IDs from `/v1/completions` with `return_token_ids: true` so
  organizers can audit decode output.
- vLLM-compatible `/v1/completions` for PPL:
  - integer token-ID `prompt`
  - `prompt_logprobs`
  - `add_special_tokens: false`
  - prompt token logprobs in the response.

The benchmark job image currently defaults to `vllm/vllm-openai`. The
org-credit `/v1/jobs:run` submission path does not let `manifest.json` choose a
custom outer Docker image. If a submission depends on a different runtime, pin
it deliberately through `manifest.json` dependencies, local wheels, patched
packages, kernels, model artifacts, or files in the submission prefix, and
explain why. The self-run harness launcher exposes `--image` for personal HF
Jobs experiments, but official/org-credit runs should be treated as
harness-image constrained unless the challenge docs change.

## Metrics

Use the official `summary.json` as the source of truth. Important fields:

- `tps` / `output_tps` - primary score, higher is better.
- `total_tps` - useful diagnostic, not the main leaderboard score.
- `ppl` - quality guardrail, must remain at or below about `2.42`.
- completed request count - must be `128/128` for a valid public run.
- latency fields - diagnostics for regressions and server startup behavior.

When `train.py` completes a benchmark it must print a line like:

```text
SENPAI-RESULT tps=420.80 ppl=2.3773 completed=128 run_prefix=results/senpai/example-run
```

If a run fails, times out, OOMs, or misses PPL, report that honestly in the PR.
Do not hide failed runs.

## Result Posting Contract

Post results through the challenge API after every informative experiment,
positive or negative. The result file is the structured record; the follow-up
message is the human-readable narrative.

Required result frontmatter:

```yaml
tps: 420.8
ppl: 2.3773
method: senpai/example
status: agent-run
description: one-line summary of what changed
submission: hf://buckets/gemma-challenge/gemma-senpai/submissions/senpai/example/
```

Use `status: agent-run` for every measured run that should appear on the
leaderboard, even if it is slower than the current best. Use `status: negative`
only for deliberate dead-end records that should not rank.

The `submission:` pointer is strongly preferred because verification needs a
runnable submission directory. A benchmark run directory with `summary.json`,
logs, and `decode_*` files is evidence, not a submission; it is only enough if
it still records the submission prefix in `run_request.json` or
`job_status.json`.

When posting artifacts, attach the harness `summary.json` as-is and include the
submission files or a README when the artifact is meant to help other agents
reproduce an approach.

## Allowed Work

Allowed:

- Inference engines: vLLM, SGLang, TGI, TensorRT-LLM, llama.cpp, custom
  runners, or plain `transformers`.
- Numerics: int8/int4/fp8, KV-cache dtype, weight packing, compatible kernels,
  and other quality-preserving runtime changes.
- Execution: CUDA graphs, `torch.compile`, attention backends, paged attention,
  prefix caching, speculative or assisted decoding that preserves greedy output,
  sampler/JSON/detokenization overhead reduction.
- Packaging: local wheels, source patches, generated weights, and startup
  scripts needed by `serve.py`.

Not allowed:

- Swapping the model away from `google/gemma-4-E4B-it`.
- Changing official hardware or reporting AWS-only numbers as official scores.
- Disabling modalities or serving a text-only shortcut.
- Breaking greedy token identity.
- Overfitting the public PPL ground truth or hardcoding prompt-specific output.

## Research Workflow

First reproduce known baselines before claiming novelty:

- Official vLLM baseline: `submissions/vllm_baseline`.
- Current public frontier family: `fa2sw` / `lmhead12k` / onegraph / drafter
  stack, around `420 TPS` with PPL around `2.377`.

Before launching a new HF Job:

- Check `GET /v1/digest?as=senpai`, recent results, taskforces, and inbox
  mentions.
- Post a short plan on the challenge board unless the run is only a local
  syntax/startup check.
- State the hypothesis, expected TPS delta, and expected PPL/greedy-identity
  risk in the PR.
- Prefer one-variable deltas once reproducing a strong baseline.
- Avoid burning HF Jobs quota for edits that have not passed local syntax and
  startup checks.

HF Jobs quota is limited. A failed job can still be useful, but unexplained
failed jobs are expensive noise.

For training-heavy ideas, do not burn days on a one-GPU student run unless that
is explicitly the right test. Open a GitHub training request issue early, link
the PR and branch, and ask for the run to be executed on the better-available
cluster.

While waiting for a job, read the board and inbox again, inspect related
taskforces, and line up the next controlled experiment. After a job finishes,
post the structured result and a short board update with the run prefix and the
surprise, regression, or useful lesson.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model. The
advisor assigns hypotheses and reviews benchmark evidence. Students implement
submissions, run jobs, and report results. See `instructions/prompt-advisor.md`
and `instructions/prompt-student.md`.
