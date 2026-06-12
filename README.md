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
- `scripts/` - helpers for syncing the official harness, uploading
  submissions, launching HF Jobs, polling runs, and posting results.
- `official/speed_benchmark/` - exact synced mirror of the official challenge
  benchmark harness from `gemma-challenge/gemma-main-bucket`.
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

## Common Commands

Install project dependencies:

```bash
uv sync
```

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

Sync the official harness mirror:

```bash
python scripts/sync_official_harness.py
```

## Official Harness

The mirrored harness defaults to the challenge benchmark image
`vllm/vllm-openai`. Treat `official/speed_benchmark/` as read-only challenge
source. Put experiment code in `submissions/**`.

If you need to run the official launcher directly:

```bash
cd official/speed_benchmark
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
