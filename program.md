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
  to update the mirror.
- `docs/` - human setup and design notes.
- `infra/aws/` - A10G launcher and AWS handoff details. Treat this as
  operational support, not as the experiment surface.
- `instructions/` - Senpai role prompts. Read-only during normal experiments.
- `pyproject.toml` - runtime dependencies and the `a10g` helper command. If a
  submission needs a new package outside its manifest, add it in the same PR
  that uses it.

## GitHub To HF Bucket Workflow

Development happens in this GitHub repository through normal Senpai PRs.
Leaderboard execution happens from the Hugging Face scratch bucket.

The normal path is:

1. Implement or modify a submission under `submissions/<name>/`.
2. Upload only that submission directory to:
   `hf://buckets/gemma-challenge/gemma-senpai/submissions/senpai/<name>`.
3. Launch the challenge benchmark through the org-credit API or the official
   harness launcher.
4. Poll `hf://buckets/gemma-challenge/gemma-senpai/results/senpai/<run>/`.
5. Post only meaningful results back to the challenge board.

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

Your endpoint must support:

- OpenAI-compatible serving for the speed benchmark.
- vLLM-compatible `/v1/completions` for PPL:
  - integer token-ID `prompt`
  - `prompt_logprobs`
  - `add_special_tokens: false`
  - prompt token logprobs in the response.

The benchmark image currently defaults to `vllm/vllm-openai`. If a submission
depends on a different runtime, pin it deliberately and explain why.

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

- Check the challenge digest and recent results.
- State the hypothesis and expected delta in the PR.
- Prefer one-variable deltas once reproducing a strong baseline.
- Avoid burning HF Jobs quota for edits that have not passed local syntax and
  startup checks.

HF Jobs quota is limited. A failed job can still be useful, but unexplained
failed jobs are expensive noise.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model. The
advisor assigns hypotheses and reviews benchmark evidence. Students implement
submissions, run jobs, and report results. See `instructions/prompt-advisor.md`
and `instructions/prompt-student.md`.
