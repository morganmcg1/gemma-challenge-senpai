# Fast Gemma Challenge Agent Notes

## AWS A10G Access

- Before using AWS APIs, try direct SSH to the recorded A10G node. SSH only
  needs the local key and can still work when AWS SSO or temporary credentials
  are expired:

  ```bash
  ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
    ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com
  ```

- If SSH times out at TCP connect, the public IP/DNS may be stale or the
  security group may only allow a previous operator IP. Then refresh AWS access
  with `aws sso login --profile sandbox-sso` or renewed temporary credentials,
  discover the current instance address, and update SSH ingress for the current
  `/32`.
- Do not print `.env`, `.credentials`, or Kubernetes Secret values in logs or
  final answers.
- Treat AWS SSO/API access as a fallback for discovery or security-group repair.
  Most day-to-day investigation, Docker management, and Senpai restarts can be
  done over SSH with the local key.

## Active AWS Senpai Runtime

- The active Gemma advisor/student workers are Docker containers on the AWS A10G
  node, not pods in the local `kubectl` context. Check the AWS node first when
  asked about current advisor/student status, experiments, or Weave logging.
- A local `kubectl get pods` view can be misleading for this launch. Old
  `open2-*` local deployments belonged to another launch and may be scaled down
  or deleted; do not infer current student health from them.
- Current run tag and host-side state directory:

  ```bash
  RUN_TAG=gemma-8gpu-progress-20260613
  RUN_DIR=/home/ubuntu/.senpai/runs/$RUN_TAG
  ```

- Safe status command:

  ```bash
  ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
    ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
    'sudo docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" | grep "^senpai-gemma"'
  ```

- Expected active container names use this pattern:

  ```text
  senpai-gemma-8gpu-progress-20260613-advisor
  senpai-gemma-8gpu-progress-20260613-student-{denken,fern,kanna,land,lawine,stark,ubel,wirbel}
  ```

- Host logs live under `$RUN_DIR/logs/` and per-role workdirs under
  `$RUN_DIR/workdirs/`. Secret env files live under `$RUN_DIR/secrets/`; do not
  print them.

### AWS Runtime Investigation Checklist

Use this checklist before concluding that a pod, student, advisor, or Weave is
missing:

```bash
ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
  ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
  'set -euo pipefail
   RUN_TAG=gemma-8gpu-progress-20260613
   RUN_DIR=/home/ubuntu/.senpai/runs/$RUN_TAG
   hostname
   printf "run_dir=%s\n" "$RUN_DIR"
   sudo docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | grep "^senpai-$RUN_TAG" || true
   ls -1 "$RUN_DIR"/logs | head'
```

If a student is absent from `docker ps -a`, check its pidfile and last log
before restarting:

```bash
ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
  ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
  'set -euo pipefail
   RUN_TAG=gemma-8gpu-progress-20260613
   RUN_DIR=/home/ubuntu/.senpai/runs/$RUN_TAG
   STUDENT=kanna
   PIDFILE="$RUN_DIR/pids/student-$STUDENT.pid"
   if [ -f "$PIDFILE" ]; then
     pid=$(cat "$PIDFILE")
     if kill -0 "$pid" 2>/dev/null; then echo "$pid alive"; else echo "$pid dead"; fi
   else
     echo "missing pidfile"
   fi
   tail -80 "$RUN_DIR/logs/student-$STUDENT.log"'
```

### Docker And GPU Mapping

- Student GPU placement is encoded both in Docker `DeviceRequests` and in
  `$RUN_DIR/env/student-*.docker.env` as `CUDA_VISIBLE_DEVICES`.
- Before re-spawning a student, inspect the live mapping and use the student's
  original GPU. Do not guess from container order.

```bash
ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
  ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
  'set -euo pipefail
   RUN_TAG=gemma-8gpu-progress-20260613
   RUN_DIR=/home/ubuntu/.senpai/runs/$RUN_TAG
   ids=$(sudo docker ps -q --filter "name=senpai-$RUN_TAG-student-")
   if [ -n "$ids" ]; then
     sudo docker inspect $ids --format "{{.Name}} {{json .HostConfig.DeviceRequests}}" | sort
   fi
   for f in "$RUN_DIR"/env/student-*.docker.env; do
     name=$(basename "$f" .docker.env)
     gpu=$(grep -E "^CUDA_VISIBLE_DEVICES=" "$f" | cut -d= -f2- || true)
     printf "%s CUDA_VISIBLE_DEVICES=%s\n" "$name" "$gpu"
   done | sort'
```

### Re-Spawning A Missing Student

Use this when a student container is absent or exited but the run should
continue. The command uses the existing non-secret env file plus the existing
secret env file path without printing secret values.

1. Preserve dirty target work before restart. The target repo may be owned by
   root because it is container-managed, so use `sudo git`.

   ```bash
   ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
     ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
     'set -euo pipefail
      RUN_TAG=gemma-8gpu-progress-20260613
      RUN_DIR=/home/ubuntu/.senpai/runs/$RUN_TAG
      STUDENT=kanna
      TARGET="$RUN_DIR/workdirs/student-$STUDENT/target"
      sudo git -C "$TARGET" status --short
      if [ -n "$(sudo git -C "$TARGET" status --porcelain)" ]; then
        sudo git -C "$TARGET" stash push -u -m "pre-respawn-$STUDENT-$(date -u +%Y%m%dT%H%M%SZ)"
      fi'
   ```

2. Re-spawn with the existing run volumes and the student's assigned GPU.

   ```bash
   ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
     ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
     'set -euo pipefail
      RUN_TAG=gemma-8gpu-progress-20260613
      RUN_DIR=/home/ubuntu/.senpai/runs/$RUN_TAG
      STUDENT=kanna
      GPU=$(grep -E "^CUDA_VISIBLE_DEVICES=" "$RUN_DIR/env/student-$STUDENT.docker.env" | cut -d= -f2-)
      NAME=senpai-$RUN_TAG-student-$STUDENT
      LOG="$RUN_DIR/logs/student-$STUDENT.log"
      PIDFILE="$RUN_DIR/pids/student-$STUDENT.pid"
      if sudo docker ps -a --format "{{.Names}}" | grep -qx "$NAME"; then
        sudo docker rm -f "$NAME" >/dev/null
      fi
      printf "\n=== Manual re-spawn at %s ===\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
      nohup sudo docker run --rm \
        --name "$NAME" \
        --workdir /workspace/senpai \
        --env-file "$RUN_DIR/env/student-$STUDENT.docker.env" \
        --env-file "$RUN_DIR/secrets/student-$STUDENT.docker.env" \
        --volume "$RUN_DIR/workdirs/student-$STUDENT:/workspace/senpai" \
        --volume "$RUN_DIR:/senpai-run" \
        --gpus "device=$GPU" \
        ghcr.io/wandb/senpai:latest \
        bash k8s/entrypoint-student.sh >> "$LOG" 2>&1 &
      printf "%s\n" "$!" > "$PIDFILE"
      sleep 3
      sudo docker ps --format "{{.Names}}\t{{.Status}}" | grep "$NAME"'
   ```

3. Verify the student picked up its assigned PR and did not immediately exit.

   ```bash
   ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
     ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
     'set -euo pipefail
      RUN_TAG=gemma-8gpu-progress-20260613
      RUN_DIR=/home/ubuntu/.senpai/runs/$RUN_TAG
      STUDENT=kanna
      NAME=senpai-$RUN_TAG-student-$STUDENT
      sudo docker ps --format "{{.Names}}\t{{.Status}}" | grep "$NAME"
      pid=$(cat "$RUN_DIR/pids/student-$STUDENT.pid")
      if kill -0 "$pid" 2>/dev/null; then echo "pidfile=$pid alive"; else echo "pidfile=$pid dead"; fi
      sudo docker exec "$NAME" pgrep -af "claude|node /usr/bin/weave-claude-plugin daemon" || true
      grep -E "Student Heartbeat|Assigned PRs|Switched to a new branch|Claude exited" "$RUN_DIR/logs/student-$STUDENT.log" | tail -30'
   ```

### Weave Logging Checks

- The expected project is `wandb-applied-ai-team/gemma-challenge-senpai`.
- `WEAVE_PROJECT` may be unset in the container; that is OK if
  `weave-claude-plugin status` reports the expected project from
  `settings.json` and the daemon log's OTel init line uses the same project.
- To check Weave inside a worker without exposing credentials:

  ```bash
  ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
    ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
    'sudo docker exec senpai-gemma-8gpu-progress-20260613-advisor weave-claude-plugin status'
  ```
- To audit all active workers:

  ```bash
  ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
    ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
    'set -euo pipefail
     RUN_TAG=gemma-8gpu-progress-20260613
     for c in $(sudo docker ps --format "{{.Names}}" | grep "^senpai-$RUN_TAG-" | sort); do
       echo "== $c =="
       sudo docker exec "$c" bash -lc '"'"'
         printf "env_WEAVE_PROJECT=%s\n" "${WEAVE_PROJECT:-<unset>}"
         printf "env_WANDB_ENTITY=%s\n" "${WANDB_ENTITY:-<unset>}"
         printf "env_WANDB_PROJECT=%s\n" "${WANDB_PROJECT:-<unset>}"
         weave-claude-plugin status 2>&1 | grep -E "Weave project|Status:" || true
         tail -200 "$HOME/.weave_claude_plugin/logs/daemon.log" 2>/dev/null |
           grep -E "OTel tracer initialized|Daemon started" | tail -3 || true
       '"'"'
     done'
  ```
- If `weave-claude-plugin status` says ready but Weave traces are missing, check
  `~/.weave_claude_plugin/logs/hook-errors.log` inside the container. The known
  failure mode is a stale `daemon.sock` with `nc: unix connect failed:
  Connection refused`. Hook errors before a daemon restart indicate lost spans;
  fresh daemon logs plus new `PreToolUse`/`PostToolUse` lines indicate tracing is
  alive again. Fix the stale socket without restarting Claude or experiments by
  removing the stale socket and starting one daemon with detached Docker exec:

  ```bash
  ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
    ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
    'sudo docker exec senpai-gemma-8gpu-progress-20260613-advisor \
      bash -lc "rm -f \$HOME/.weave_claude_plugin/daemon.sock" && \
      sudo docker exec -d senpai-gemma-8gpu-progress-20260613-advisor \
      bash -lc "weave-claude-plugin daemon"'
  ```

- After restart, verify both status and timestamps:

  ```bash
  ssh -i /Users/mmcguire/ML/gemma_chall/gemma-a10g.pem \
    ubuntu@ec2-107-22-25-10.compute-1.amazonaws.com \
    'sudo docker exec senpai-gemma-8gpu-progress-20260613-advisor bash -lc "
      weave-claude-plugin status
      stat -c \"%y %s %n\" \$HOME/.weave_claude_plugin/logs/hook-errors.log 2>/dev/null || true
      stat -c \"%y %s %n\" \$HOME/.weave_claude_plugin/logs/daemon.log 2>/dev/null || true
      pgrep -af \"node /usr/bin/weave-claude-plugin daemon\" || true
      tail -40 \$HOME/.weave_claude_plugin/logs/daemon.log 2>/dev/null | grep -E \"OTel tracer initialized|Daemon started|PreToolUse|PostToolUse|ERROR\" || true
    "'
  ```

- `no parent span` errors in the daemon log mean partial trace continuity for
  that Claude session, often after daemon inactivity or restart. They do not by
  themselves mean W&B auth failed or that the project is wrong.

### GitHub Operational Issues

- When fixing an advisor-filed operational issue, comment with enough evidence
  for the advisor to resume: container name, GPU, assigned PR, pid/Claude status,
  and Weave status when relevant.
- If the user asks for a human-visible control message, prefix the GitHub
  comment exactly as requested, for example `human: fixed ...`.
