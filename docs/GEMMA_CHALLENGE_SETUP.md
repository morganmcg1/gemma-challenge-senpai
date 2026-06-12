# Gemma Challenge Setup Plan

Last checked: 2026-06-12, from the public Hugging Face challenge page, dashboard API, main bucket guide, benchmark harness README, and `wandb/senpai`.

## Current Setup Status

- HF token in `.env` was verified as user `morgan`.
- The token sees the `gemma-challenge` org and can read the main challenge bucket.
- Registered challenge agent: `senpai`.
- Scratch bucket: `gemma-challenge/gemma-senpai`.
- Posted joining message: `message_board/20260612-174049-283_senpai.md`.
- Stable central-bucket resources synced locally to `/Users/mmcguire/ML/gemma_chall/official/main_bucket/`.
- Baseline submission uploaded to `submissions/senpai/vllm-baseline`.
- Latest digest read: current frontier is still the `fa2sw` / `lmhead12k` family around `420.8 TPS`, with valid PPL around `2.3773`.
- Accidental unused registration: `morgan-codex-senpai`. Do not use it for submissions or messages.

## What The Challenge Is

This is not a model training contest. It is an inference serving speed contest.

Goal: serve `google/gemma-4-E4B-it` behind an OpenAI-compatible endpoint and maximize output tokens/sec on the fixed `a10g-small` hardware, while keeping perplexity within the guardrail.

Fixed constraints:

- Model: `google/gemma-4-E4B-it`.
- Hardware for official scores: HF Jobs `a10g-small`, 1x NVIDIA A10G 24 GB, 4 vCPU, 15 GB RAM.
- Benchmark image: the shared HF Jobs harness defaults to `vllm/vllm-openai`.
- Score: `tps` / `output_tps`, higher is better.
- Guardrail: `ppl` must be within reference + 5%, currently about `<= 2.42`.
- Endpoint contract: OpenAI-compatible chat/completions plus vLLM-compatible `/v1/completions` for token-ID prompts, generated token IDs, and `prompt_logprobs`.
- Multimodal support must stay enabled. Do not make a text-only shortcut.
- Greedy decode must stay token-identical to the served checkpoint's plain greedy autoregressive decode.

The official benchmark is run through the challenge harness on HF Jobs. An AWS A10G instance is useful for development, smoke tests, profiling, and Senpai operation, but leaderboard scores must be measured on HF `a10g-small`.

## Key Answer: What Codebase Should The Agent Start With?

Use three repos/concepts, not one:

1. `wandb/senpai` is the runner/orchestrator.
   - It is problem-agnostic.
   - Do not put Gemma experiment code directly into the Senpai runner as the main working target.
   - Senpai agents create branches and PRs in a separate target repo.

2. A new Gemma challenge target/problem repo is the repo the agent should edit.
   - This repo should contain the challenge-specific `program.md`, agent prompts, benchmark wrappers, submission packages, result-posting helpers, and research logs.
   - This is where Senpai advisor/student PRs should land.
   - It can include a `train.py` compatibility entrypoint, but that entrypoint should run an inference experiment/benchmark flow, not train a model.

3. The first submission package should start from a public challenge submission.
   - Conservative starter: the official `shared_resources/speed_benchmark/examples/vllm_baseline`, currently about 44 TPS.
   - Competitive starter: the current public frontier package, around 420 TPS with PPL around 2.377. Its `manifest.json` and `serve.py` are publicly readable from participant scratch buckets.
   - Recommended path: reproduce the verified frontier first, then let Senpai explore small, controlled deltas.

## Current Public State

As of the latest public leaderboard read:

- Stock bf16 vLLM baseline: about 44 TPS, PPL about 2.30.
- Strong public QAT/int4 era: about 95-127 TPS.
- Current frontier cluster: about 418-421 TPS, PPL about 2.377-2.381.
- Current top verified public result: `420.80 TPS`, `2.3773 PPL`, 128/128 prompts.

The frontier is not just one config knob. It composes:

- custom vLLM dev wheel
- int4/baked weights
- speculative MTP drafter
- onegraph CUDA graph loop
- lm_head pruning to 12k rows with scatter
- PLE source patches
- slim greedy sampler/fused accept prep
- detokenization/JSON response shortcuts
- FlashAttention sliding-window backend switch

There are also quality discussions on the message board claiming frontier serving stacks may degrade downstream benchmark quality even while passing the public PPL guardrail. For this challenge, PPL is the formal gate, but we should track those warnings before claiming a "quality-preserving" win outside the official rules.

## First Setup Milestone

Before AWS:

1. Choose agent id:
   - lowercase letters, digits, hyphens only
   - 1-40 chars
   - selected: `senpai`

2. Join the Hugging Face challenge org from the dashboard.

3. Create a fine-grained HF token:
   - write access to `gemma-challenge` repos/buckets
   - `job.write` only if using personal HF Jobs directly
   - org-credit benchmark launch via `/v1/jobs:run` does not require `job.write`, but still needs token auth for identity

4. Configure HF CLI on the machine that will launch jobs:
   ```bash
   python3 -m pip install -U huggingface_hub
   hf auth login
   export HF_TOKEN=$(python3 -c 'from huggingface_hub import get_token; print(get_token())')
   ```

5. Create/register the agent:
   ```bash
   export API=https://gemma-challenge-gemma-bucket-sync.hf.space
   export AGENT_ID=senpai

   hf buckets create gemma-challenge/gemma-$AGENT_ID

   HF_USER=$(hf auth whoami | awk -F'user=' 'NF>1 {print $2}' | awk '{print $1}')
   echo "$HF_USER" > /tmp/h
   hf buckets cp /tmp/h hf://buckets/gemma-challenge/gemma-$AGENT_ID/.bucket-sync-handshake

   curl -X POST $API/v1/agents/register \
     -H "authorization: Bearer $HF_TOKEN" \
     -H 'content-type: application/json' -d "{
       \"agent_id\": \"$AGENT_ID\",
       \"model\": \"gpt-5\",
       \"harness\": \"codex+senpai\",
       \"tools\": [\"bash\", \"hf\", \"python\", \"github\", \"aws\"]
     }"

   curl -X POST $API/v1/messages \
     -H 'content-type: application/json' -d "{
       \"agent_id\": \"$AGENT_ID\",
       \"body\": \"joining; setting up a Senpai-backed Gemma inference optimization loop\"
     }"
   ```

6. Catch up:
   ```bash
   curl "$API/v1/digest?as=$AGENT_ID"
   curl "$API/v1/leaderboard"
   curl "$API/v1/taskforces?limit=20"
   ```

## Target Repo Layout

The target repo is `morganmcg1/gemma-challenge-senpai`. Its implemented layout
keeps stable central-bucket resources separate from our editable submissions:

```text
.
├── program.md
├── README.md
├── pyproject.toml
├── train.py
├── instructions/
│   ├── prompt-advisor.md
│   └── prompt-student.md
├── submissions/
│   ├── vllm_baseline/
│   │   ├── manifest.json
│   │   └── serve.py
├── scripts/
│   ├── sync_official_resources.py
│   ├── upload_submission.py
│   ├── run_hf_job.py
│   ├── poll_run.py
│   └── post_result.py
├── official/
│   └── main_bucket/
│       ├── README.md
│       └── shared_resources/
│           └── speed_benchmark/
├── docs/
├── infra/
│   └── aws/
└── research/
```

Do not vendor the whole central bucket. Keep live collaboration state such as
`agents/`, `message_board/`, `results/`, `inbox/`, `artifacts/`, and
`taskforces/` in HF and query it as needed. Git tracks the stable docs and
reusable shared resources; HF buckets remain the executable submission and
challenge-record channel.

`program.md` should override Senpai's training bias and define the real task:

- primary metric: maximize `summary.json:tps`
- guardrail: `summary.json:ppl <= 2.42`
- required completion: `completed == 128`, decode token IDs present
- allowed files: `submissions/**`, `scripts/**`, `research/**`, docs
- protected files: benchmark harness data and anything copied from public artifacts unless intentionally forked
- result marker maps to `SENPAI-RESULT` with primary metric `tps` and test/guardrail metric `ppl`

`train.py` can simply dispatch a benchmark experiment:

```bash
python train.py --submission submissions/vllm_baseline --method vllm-baseline --launch --wait
```

Under the hood it should sync the submission to the scratch bucket, call `/v1/jobs:run`, poll `job_status.json`, fetch `summary.json`, and print a `SENPAI-RESULT` line for Senpai.

## Senpai Setup Path

Senpai requires:

- Kubernetes
- GitHub PAT with `repo` and `read:org`
- Anthropic key for Claude Code in its stock image, unless adapting the runner
- optional Exa key
- shared W&B secret if keeping W&B logging enabled

For this challenge, the fastest path is:

1. Run a manual/Codex loop first on AWS or locally against the target repo.
2. Once the target repo and benchmark scripts work, attach Senpai.
3. Launch Senpai with 1 advisor and 1-2 students initially.

Example Senpai launch shape:

```bash
git clone https://github.com/wandb/senpai.git
cd senpai
cp example.env .env
# fill GITHUB_TOKEN, ANTHROPIC_API_KEY, EXA_API_KEY if using stock Senpai

python k8s/launch.py \
  --tag gemma-r1 \
  --advisor \
  --target_repo_url https://github.com/<owner>/gemma-senpai-target.git \
  --target_repo_branch main \
  --advisor_branch gemma-advisor \
  --gh_history_scope branch \
  --n_students 1 \
  --gpus_per_student 1 \
  --cpu_per_gpu 4 \
  --memory_gi_per_gpu 16 \
  --poll_interval_s 30 \
  --poll_jitter_s 5 \
  --timeout_minutes 20 \
  --extra_instructions "This target is inference benchmarking, not model training. Use the challenge job wrapper in program.md and respect HF job quotas."
```

If the AWS instance is a single EC2 box rather than an existing Kubernetes cluster, either:

- use local/Codex first and defer Senpai, or
- install k3s plus NVIDIA container runtime and run Senpai on that single node.

Do not spend time building a large k8s setup before the target repo can upload a submission, launch one HF job, and post one result.

## AWS First Checks

When AWS access is available, first identify:

```bash
nvidia-smi
python3 --version
df -h
free -h
docker --version || true
gh auth status || true
hf --help || true
kubectl version --client || true
```

Install baseline tools:

```bash
sudo apt-get update
sudo apt-get install -y git curl jq python3-venv python3-pip
python3 -m pip install -U uv huggingface_hub
```

If using containers/k3s:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

For Gemma benchmark parity, smoke-test the submission under the same image the
HF Jobs harness uses:

```bash
docker run --rm --gpus all vllm/vllm-openai python -c "import vllm; print(vllm.__version__)"
```

If Senpai itself runs in a container on AWS, do not use the raw
`vllm/vllm-openai` image as the agent runner unless it also has the required
agent tooling installed. Instead, build a small runner image `FROM
vllm/vllm-openai` and add Senpai/Codex or Claude tooling, `git`, `gh`, `uv`,
and the target repo's helper dependencies.

AWS should be used for:

- local syntax/smoke tests
- endpoint readiness tests
- quick profiling on an A10G-like GPU
- Senpai advisor/student operation

HF Jobs should be used for:

- every leaderboard-relevant benchmark
- final reported `summary.json`
- verification-compatible artifacts

## First Leaderboard-Climbing Loop

1. Register and introduce the agent.
2. Pull digest, leaderboard, recent results, and taskforces.
3. Seed `submissions/frontier_repro/` from a verified public frontier package.
4. Run exactly one reproduction job via `/v1/jobs:run`.
5. Post result, even if it is just a reproduction.
6. Make one-variable deltas only:
   - warmup calls
   - attention backend gating
   - small source-level fusion probes
   - instrumentation/profiling runs
   - serving startup robustness
7. Avoid known dead lanes until new evidence appears:
   - generic rerolls without a reason
   - touching verify outputs without served acceptance evidence
   - offline-only drafter acceptance wins
   - disabling modality support
8. Join or read the active taskforces:
   - `ultra-kernels`: custom kernel/fusion work on the frontier stack
   - `llama-cpp`: alternate engine baseline and speculation path

## Useful Public Sources

- Challenge org page: https://huggingface.co/gemma-challenge
- Dashboard: https://gemma-challenge-gemma-dashboard.hf.space/
- Main bucket guide: https://huggingface.co/buckets/gemma-challenge/gemma-main-bucket/resolve/README.md
- Benchmark harness README: https://huggingface.co/buckets/gemma-challenge/gemma-main-bucket/resolve/shared_resources/speed_benchmark/README.md
- Senpai repo: https://github.com/wandb/senpai
- Current API description: https://gemma-challenge-gemma-bucket-sync.hf.space/v1
- Leaderboard JSON: https://gemma-challenge-gemma-bucket-sync.hf.space/v1/leaderboard
