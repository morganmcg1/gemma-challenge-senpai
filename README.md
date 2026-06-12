# Gemma Challenge Senpai Workspace

Workspace for running a Senpai-backed loop against the Hugging Face Fast Gemma
Challenge. The official score is produced by the challenge HF Jobs benchmark on
`a10g-small`; AWS is used here for orchestration, smoke tests, and iteration.

## What Is Here

- `GEMMA_CHALLENGE_SETUP.md` - current challenge setup, registered agent state,
  submission paths, benchmark image, and next steps.
- `speed_benchmark/` - official HF challenge benchmark harness synced from
  `gemma-challenge/gemma-main-bucket/shared_resources/speed_benchmark/`.
- `speed_benchmark/examples/vllm_baseline/` - official baseline submission
  package with `manifest.json` and `serve.py`.
- `SENPAI_INFRA_AGNOSTIC_PLAN.md` - notes for making `wandb/senpai` usable
  outside its original Kubernetes cluster.
- `AWS_INFRA.md` - current A10G node handoff and operational commands.
- `src/a10g_node/` - small `uv` CLI for launching/terminating an AWS EC2 G5
  instance with an NVIDIA A10G GPU.

Current challenge identity:

```bash
export AGENT_ID=senpai
export SCRATCH_BUCKET=gemma-challenge/gemma-senpai
```

The baseline submission has been uploaded to:

```text
hf://buckets/gemma-challenge/gemma-senpai/submissions/senpai/vllm-baseline
```

## Run The Official Harness

The synced harness defaults to the same benchmark image used by the challenge:
`vllm/vllm-openai`.

To launch a benchmark intentionally, from `speed_benchmark/`:

```bash
export AGENT_ID=senpai
export SCRATCH_BUCKET=gemma-challenge/gemma-$AGENT_ID
export SUBMISSION_PREFIX=submissions/$AGENT_ID/vllm-baseline
export RUN_PREFIX=results/$AGENT_ID/vllm-baseline-$(date -u +%Y%m%dT%H%M%SZ)

uv run --with huggingface_hub python scripts/run_hf_bucket_benchmark.py \
  --submission-bucket "$SCRATCH_BUCKET" \
  --submission-prefix "$SUBMISSION_PREFIX" \
  --run-prefix "$RUN_PREFIX" \
  --flavor a10g-small \
  --wait
```

This spends HF Jobs quota/credits, so do it deliberately.

## A10G EC2 Node Launcher

Small `uv` project for starting an AWS EC2 G5 instance with an NVIDIA A10G GPU.

For the current running AWS resources and handoff details, see [AWS_INFRA.md](AWS_INFRA.md).

## Setup

```bash
uv sync
```

Fill `.env` with either an AWS profile:

```dotenv
AWS_PROFILE=my-profile
AWS_REGION=us-east-1
```

Or direct credentials:

```dotenv
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
AWS_REGION=us-east-1
```

## Check Auth

```bash
uv run a10g check-auth
```

## Preview Launch Inputs

```bash
uv run a10g plan
```

This resolves:

- the current AWS identity
- latest Deep Learning AMI ID from the configured public SSM parameter
- a default-VPC subnet in an AZ that offers the selected G5 instance type
- SSH security group details
- your SSH CIDR, auto-detected as your current public IPv4 `/32` if blank

## Launch

```bash
uv run a10g launch
```

Defaults:

- `INSTANCE_TYPE=g5.xlarge`
- `VOLUME_GB=200`
- `MARKET_TYPE=on-demand`
- `DLAMI_SSM_PARAMETER=/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.8-ubuntu-24.04/latest/ami-id`

If the EC2 key pair named by `KEY_NAME` does not exist, launch creates it and writes the private key to `KEY_PATH` with `0600` permissions.

## Terminate

```bash
uv run a10g terminate i-0123456789abcdef0
```

Current node:

```bash
cd /Users/mmcguire/ML/gemma_chall
uv run a10g terminate i-031a9640edf2921c5
```

Instance details:

- `i-031a9640edf2921c5`
- `g5.xlarge`
- `us-east-1a`
- `ec2-3-87-184-234.compute-1.amazonaws.com`
- `3.87.184.234`

## Notes

AWS documents G5 instances as the EC2 family using NVIDIA A10G GPUs, and the AWS Deep Learning AMI docs publish SSM parameter paths for current DLAMI IDs. The default here uses the PyTorch Ubuntu 24.04 DLAMI path; set `AMI_ID` or `DLAMI_SSM_PARAMETER` in `.env` if you want a different image.

CoreWeave note: if this work moves from AWS to CoreWeave, verify and use the latest supported CoreWeave GPU image at launch time instead of reusing an old image tag.
