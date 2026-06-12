# Senpai Compute-Agnostic Plan For AWS A10G

Last checked: 2026-06-12 against `wandb/senpai` `main`.

## Short Version

Yes: using a single AWS A10G node changes how we should use Senpai.

The good news is that Senpai's core research loop is not fundamentally Kubernetes-specific. The GitHub PR protocol, advisor/student roles, target repo model, labels, and result review flow are portable.

The Kubernetes coupling lives mostly in:

- `k8s/launch.py`
- `k8s/launch_helpers.py`
- `k8s/*-deployment.yaml`
- `k8s/entrypoint-advisor.sh`
- `k8s/entrypoint-student.sh`
- `system_instructions/CLAUDE-ADVISOR.md` references to `kubectl`
- README/operator docs and cutoff scripts

For the Gemma challenge specifically, we probably do not need a GPU pod per student. Official scores run on HF Jobs. The AWS A10G node is best used for local smoke tests, profiling, and optionally one GPU-backed student. The advisor and most students can be CPU-only workers that edit code, upload submissions, launch HF Jobs, poll results, and post to the challenge board.

## Current Couplings

### 1. Launcher Does Too Much

`k8s/launch.py` currently handles:

- CLI/config parsing
- GitHub token resolution and preflight
- target repo branch creation
- GitHub label creation
- student name expansion
- environment/config generation
- secret rendering
- Kubernetes manifest rendering
- `kubectl apply`

Only the last three are actually Kubernetes-specific. The rest should be shared.

### 2. Role Runtime Assumes A K8s Pod

The deployment YAMLs clone the Senpai runner into `/workspace/senpai`, inject ConfigMaps/Secrets, mount a PVC, then call:

- `bash k8s/entrypoint-advisor.sh`
- `bash k8s/entrypoint-student.sh`

The shell entrypoints are mostly portable, but their names and comments assume the deployment has already cloned the repo and mounted data.

### 3. Resource Model Is K8s-Shaped

Current config fields:

- `gpus_per_student`
- `cpu_per_gpu`
- `memory_gi_per_gpu`
- `pvc_claim_name`
- `pvc_mount_path`

These map naturally to Kubernetes resources and PVCs, but not to:

- host processes
- Docker Compose
- one EC2 instance
- SSH-launched remote workers
- HF Jobs as the actual benchmark executor

### 4. Advisor Instructions Mention `kubectl`

The advisor system instruction explicitly lists `kubectl` as a tool and tells the advisor to monitor student pods with:

```bash
kubectl get deployments -l app=senpai
```

That should become a backend-neutral status command.

### 5. Agent Runtime Is Also Coupled

This is a separate axis from compute infrastructure.

Stock Senpai currently runs Claude Code via `run-senpai-claude.sh`, Anthropic credentials, Claude settings, and Claude-specific skills. If we want Codex to be the actual worker, we either:

- use Senpai's GitHub PR protocol manually with Codex for now, or
- add an `agent_runtime` abstraction after the compute backend abstraction.

For the AWS A10G setup, compute agnosticism is the first necessary step. Codex runtime support is useful but not required to get the Gemma target repo moving.

## Proposed Refactor

### Layer 1: Shared Launch Model

Extract shared config and preflight from `k8s/launch.py` into a backend-neutral module:

```text
senpai/
  launch/
    config.py
    preflight.py
    roles.py
    backends/
      base.py
      kubernetes.py
      local.py
      ssh.py
```

Core structures:

```python
@dataclass
class LaunchConfig:
    tag: str
    repo_url: str
    repo_branch: str
    target_repo_url: str
    target_repo_branch: str
    advisor_branch: str
    problem_dir: str
    student_names: list[str]
    env: dict[str, str]
    secrets: dict[str, str]
    resources: ResourceConfig

@dataclass
class RoleSpec:
    role: Literal["advisor", "student"]
    name: str
    env: dict[str, str]
    secrets: dict[str, str]
    resources: ResourceConfig
    command: list[str]
```

Shared functions:

- resolve credentials
- preflight target repo access
- ensure advisor branch
- ensure GitHub labels
- expand student names
- build advisor env
- build student env
- encode extra instructions

### Layer 2: Backend Interface

Add a small backend contract:

```python
class ComputeBackend(Protocol):
    def preflight(self, config: LaunchConfig) -> None: ...
    def launch(self, roles: list[RoleSpec]) -> None: ...
    def status(self, tag: str) -> None: ...
    def logs(self, tag: str, role: str | None = None) -> None: ...
    def stop(self, tag: str) -> None: ...
```

Initial backends:

- `kubernetes`: wraps the existing YAML renderer and `kubectl apply`.
- `local`: runs advisor/student loops as local supervised processes.
- `ssh`: copies/env-renders onto a remote host and starts the local backend there.

`ssh` can come later. For one AWS node, `local` is enough once we SSH into the box.

### Layer 3: Runtime Entrypoints

Move portable entrypoints out of `k8s/`:

```text
runtime/
  bootstrap-runner.sh
  entrypoint-advisor.sh
  entrypoint-student.sh
  run-role.sh
```

Kubernetes deployments can keep calling them. Local backend can call them too.

`bootstrap-runner.sh` should handle:

- clone/update Senpai runner
- install package
- clone target repo
- configure git auth
- write role-specific `CLAUDE.md`

The backend should only supply environment variables, secrets, working directory, and resource limits.

### Layer 4: Backend-Neutral Status

Add:

```bash
senpai status --tag <tag>
senpai logs --tag <tag> [--role advisor|student:name]
senpai stop --tag <tag>
```

Then update advisor instructions:

- replace `kubectl get deployments -l app=senpai`
- with `senpai status --tag "$RESEARCH_TAG"`

Kubernetes backend can implement status with `kubectl`.
Local backend can implement status from pidfiles and log dirs.

## Local Backend MVP

This is the fastest path for AWS A10G.

### Files

```text
runtime/local_launch.py
runtime/local_status.py
runtime/local_stop.py
runtime/run_local_role.sh
```

### Runtime Directory

```text
~/.senpai/runs/<tag>/
  env/
    advisor.env
    student-fern.env
  secrets.env              # chmod 600
  worktrees/
    advisor/
    student-fern/
  logs/
    advisor.log
    student-fern.log
  pids/
    advisor.pid
    student-fern.pid
  shared/
```

### Launch Command Shape

```bash
python -m senpai.launch \
  --backend local \
  --tag gemma-r1 \
  --advisor \
  --target_repo_url https://github.com/<owner>/gemma-senpai-target.git \
  --target_repo_branch main \
  --advisor_branch gemma-advisor \
  --names fern \
  --student_gpu_ids "" \
  --shared_dir ~/.senpai/runs/gemma-r1/shared \
  --poll_interval_s 30 \
  --poll_jitter_s 5 \
  --extra_instructions "This target launches HF Jobs for official benchmarks; do not assume every student owns a local GPU."
```

For a single local GPU-backed worker:

```bash
--student_gpu_ids fern:0
```

Then local backend sets:

```bash
CUDA_VISIBLE_DEVICES=0
GPUS_PER_STUDENT=1
```

For CPU-only workers:

```bash
CUDA_VISIBLE_DEVICES=
GPUS_PER_STUDENT=0
```

This requires relaxing the current launcher validation that requires `gpus_per_student >= 1`.

## Docker Local Backend Variant

If the AWS box has Docker + NVIDIA runtime, local backend can run the existing image:

```bash
docker run -d \
  --name senpai-gemma-r1-fern \
  --gpus '"device=0"' \
  --env-file ~/.senpai/runs/gemma-r1/env/student-fern.env \
  --env-file ~/.senpai/runs/gemma-r1/secrets.env \
  -v ~/.senpai/runs/gemma-r1/shared:/mnt/senpai \
  -v ~/.senpai/runs/gemma-r1/logs:/workspace/logs \
  ghcr.io/wandb/senpai:latest \
  bash runtime/entrypoint-student.sh
```

For the Gemma challenge, direct host processes may be simpler because the students mostly launch HF Jobs. Docker is more reproducible if we later run local vLLM/SGlang smoke tests.

## Gemma-Specific Operational Changes

### 1. Allow CPU-Only Students

The challenge's official benchmark compute is external HF Jobs. A student does not need a reserved GPU to:

- edit `serve.py`
- upload a submission
- call `/v1/jobs:run`
- poll bucket results
- post challenge messages/results

Change validation from:

```python
gpus_per_student >= 1
```

to:

```python
gpus_per_student >= 0
```

Then make Kubernetes omit `nvidia.com/gpu` resources when zero.

### 2. Add `compute_mode`

Add:

```yaml
compute_mode: external-hf-jobs  # or local-gpu, train-local
```

The target `program.md` can use this to tell students how to evaluate:

- `external-hf-jobs`: upload + launch challenge job
- `local-gpu`: run local smoke/profile only
- `train-local`: old Senpai training behavior

### 3. Add HF Credentials To Launch Secrets

For this challenge, workers need:

- `HF_TOKEN`
- `AGENT_ID`
- `GEMMA_CHALLENGE_API=https://gemma-challenge-gemma-bucket-sync.hf.space`

Those should become first-class launch env/secrets, not ad hoc target repo setup.

### 4. Quota-Aware Job Launching Belongs In Target Repo

HF challenge jobs are capped. The target repo should provide a wrapper that:

- uses unique run prefixes
- records `429` retry-after responses
- avoids launching duplicate jobs
- polls `job_status.json`
- fetches `summary.json`
- prints the Senpai result marker

This should live in the Gemma target repo, not in Senpai core.

## Minimal Patch Sequence

### Patch 1: Extract Shared Env Builders

Move `render_student` / `render_advisor` data construction into pure functions:

- `build_student_env(args, student_name, tag)`
- `build_advisor_env(args, student_list, tag)`

Keep Kubernetes rendering behavior unchanged.

### Patch 2: Add Local Backend Script

Add `runtime/local_launch.py` that reuses the same env builders and starts processes with:

```bash
nohup env $(cat env files) bash runtime/entrypoint-student.sh > logs/student.log 2>&1 &
```

Initially this can assume Senpai is already cloned locally on the AWS box.

### Patch 3: Make Entrypoints Portable

Rename or copy:

- `k8s/entrypoint-advisor.sh` -> `runtime/entrypoint-advisor.sh`
- `k8s/entrypoint-student.sh` -> `runtime/entrypoint-student.sh`

Keep the old paths as wrappers for compatibility.

Replace comments and hardcoded `/workspace/senpai` with:

```bash
WORKDIR="${SENPAI_WORKDIR:-/workspace/senpai}"
```

### Patch 4: Backend-Neutral Status

Add local status/log/stop scripts and update advisor instructions from `kubectl` to `senpai status`.

### Patch 5: Optional Docker/K8s Cleanup

Let K8s backend omit GPU resources when `gpus_per_student=0`.

### Patch 6: Agent Runtime Abstraction Later

Once compute backend works, consider:

```yaml
agent_runtime: claude-code | codex
agent_command: [...]
```

This would replace `run-senpai-claude.sh` with a runtime adapter. Not needed for the first AWS setup unless we insist on fully automated Codex workers inside Senpai.

## Recommended Path For Us

Do not start by making Senpai beautifully generic. Start with a thin local backend:

1. Create the Gemma target repo and benchmark wrapper.
2. Run a manual Codex loop against it.
3. Add a local Senpai backend that can run one advisor and one CPU-only student on the AWS node.
4. Add one GPU-backed student only if local profiling becomes useful.
5. Keep official scoring through HF Jobs.
6. Upstream the backend split once it proves useful.

This preserves Senpai's useful part, the PR-based research protocol, without forcing a single AWS instance to pretend it is the original Kubernetes cluster.
