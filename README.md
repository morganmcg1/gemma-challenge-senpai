# Gemma Challenge Senpai Workspace

Senpai target repository for the Hugging Face Fast Gemma Challenge.

This repo is where we develop, review, and preserve experiment history through
GitHub. The actual challenge submissions are synced from this repo into the
Hugging Face scratch bucket for agent `senpai`.

## Challenge Identity

```bash
export AGENT_ID=senpai
export SCRATCH_BUCKET=gemma-challenge/gemma-senpai
```

Canonical HF submission destination:

```text
hf://buckets/gemma-challenge/gemma-senpai/submissions/senpai/<submission-name>
```

## Layout

- `program.md` - Senpai research contract: metrics, editable boundaries,
  allowed work, validity gates, and GitHub-to-HF workflow.
- `instructions/` - advisor and student prompts loaded by Senpai.
- `submissions/` - editable challenge submission packages. Each runnable
  package contains `manifest.json` and `serve.py`.
- `scripts/` - helpers for syncing official resources, uploading
  submissions, launching HF Jobs, polling runs, and posting results.
- `official/main_bucket/` - read-only mirror of stable central-bucket
  reference material: the bucket `README.md` and `shared_resources/**`.
- `docs/` - setup notes and Senpai infrastructure design notes.
- `infra/aws/` - A10G launcher and AWS handoff docs.
- `research/` - plans, postmortems, leaderboard notes, and human-readable
  experiment context.

## Development Flow

Development happens in GitHub. Submissions run from the HF bucket.

1. Open a PR against the active Senpai advisor branch.
2. Modify code under `submissions/<name>/`, plus scripts/docs/research notes as
   needed.
3. Run cheap local checks.
4. Sync only the selected submission package to the HF scratch bucket.
5. Launch the official benchmark on HF Jobs `a10g-small`.
6. Poll `summary.json`, then post a structured result if the run is useful.

This split is intentional. GitHub gives us review history and Senpai PR routing;
the HF bucket gives the challenge verifier an executable, stable submission
directory.

## Central Bucket Mirror Policy

Do not clone the full central bucket into Git. The live directories
`agents/`, `message_board/`, `results/`, `inbox/`, `artifacts/`, and
`taskforces/` are collaborative workspace state and should be queried from HF
when needed.

This repo vendors only stable reference material from the central bucket:

```text
official/main_bucket/README.md
official/main_bucket/shared_resources/**
```

Develop and review code in this GitHub repo. For leaderboard runs, sync only
the selected `submissions/<name>/` package to the `senpai` HF scratch bucket,
then launch the official HF Job from that bucket path.

## Common Commands

Install project dependencies:

```bash
uv sync
```

The helper scripts use `hf` if it is on `PATH`; otherwise they fall back to
`uv run hf` from this project environment.

Upload the baseline submission:

```bash
python scripts/upload_submission.py \
  --path submissions/vllm_baseline \
  --name vllm-baseline
```

Upload, launch, wait, and emit a Senpai result line:

```bash
python train.py \
  --submission submissions/vllm_baseline \
  --name vllm-baseline \
  --method senpai/vllm-baseline \
  --launch \
  --wait
```

Launch without waiting:

```bash
python train.py \
  --submission submissions/vllm_baseline \
  --name vllm-baseline \
  --method senpai/vllm-baseline \
  --launch
```

Poll a run:

```bash
python scripts/poll_run.py \
  --run-prefix results/senpai/vllm-baseline-YYYYMMDDTHHMMSSZ \
  --wait
```

Sync the official central-bucket resources mirror:

```bash
python scripts/sync_official_resources.py
```

## Official Resources

The mirrored harness defaults to the challenge benchmark image
`vllm/vllm-openai`. Treat `official/main_bucket/**` as read-only challenge
source. Put experiment code in `submissions/**`.

The submission manifest controls dependencies, environment variables, and the
server command; it does not select the outer HF Job image for org-credit
`/v1/jobs:run` submissions. Ship runtime changes as pinned Python packages,
local wheels, patched files, kernels, model artifacts, or other files inside
the uploaded submission prefix.

If you need to run the official launcher directly:

```bash
cd official/main_bucket/shared_resources/speed_benchmark
uv run --with huggingface_hub python scripts/run_hf_bucket_benchmark.py \
  --submission-bucket gemma-challenge/gemma-senpai \
  --submission-prefix submissions/senpai/vllm-baseline \
  --run-prefix results/senpai/vllm-baseline-$(date -u +%Y%m%dT%H%M%SZ) \
  --flavor a10g-small \
  --wait
```

This spends HF Jobs quota/credits, so do it deliberately.

## A10G EC2 Node

The AWS helper starts and terminates EC2 G5 instances with NVIDIA A10G GPUs:

```bash
uv run a10g check-auth
uv run a10g plan
uv run a10g launch
```

For current infrastructure handoff details, see
[`infra/aws/README.md`](infra/aws/README.md).

## Docs

- [`docs/GEMMA_CHALLENGE_SETUP.md`](docs/GEMMA_CHALLENGE_SETUP.md)
- [`docs/SENPAI_INFRA_AGNOSTIC_PLAN.md`](docs/SENPAI_INFRA_AGNOSTIC_PLAN.md)
- [`program.md`](program.md)
