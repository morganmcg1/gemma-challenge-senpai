#!/usr/bin/env bash
# PR #646 resume supervisor.
#
# The first FULL int8 window was launched manually at 05:54Z and will not finish both
# cells inside its 82-min soft-cap (GPQA alone is ~92 min at conc1). This supervisor waits
# for that window (PID passed as $1) to exit GRACEFULLY at its soft-cap — at which point
# LocalServer.__exit__ has already torn down the vLLM child and freed the GPU — then hands
# off to run_ladder_windows.sh, the idempotent multi-window orchestrator that:
#   * resumes int8 GPQA from its per-item jsonl,
#   * runs int8 AIME,
#   * measures the int4 GPQA rung (the genuinely-missing endpoint; cannot be borrowed under
#     launch isolation),
# each as its own <82-min window with clean LocalServer teardown.
#
# ANALYSIS-ONLY. One server per port 8000; this never runs concurrently with the window.
set -u
cd /workspace/senpai/target
WPID="${1:?usage: resume_supervisor.sh <window_pid>}"
DIR=research/validity/int8_bf16_reasoning_ladder
SUPLOG="$DIR/results/_supervisor.log"

echo "[sup] START $(date -u +%FT%TZ) waiting for window pid=$WPID to exit" >> "$SUPLOG"
while kill -0 "$WPID" 2>/dev/null; do sleep 30; done
echo "[sup] window pid=$WPID exited $(date -u +%FT%TZ)" >> "$SUPLOG"

# Defensive: a graceful soft-cap already tore the server down, but if the window was hard-
# killed the vLLM api_server could be orphaned on port 8000. The pattern below is specific
# to the server process, so it never matches this supervisor's own command line.
stray=$(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true)
if [ -n "$stray" ]; then
  echo "[sup] killing orphaned vllm api_server: $stray" >> "$SUPLOG"
  kill $stray 2>/dev/null || true; sleep 8
  kill -9 $stray 2>/dev/null || true; sleep 4
fi

# Wait (bounded) for VRAM to actually drop before the orchestrator boots its own server.
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ "${used:-99999}" -lt 3000 ] 2>/dev/null; then
    echo "[sup] vram free (${used} MiB) after ${i} checks" >> "$SUPLOG"; break
  fi
  sleep 5
done

echo "[sup] handing off to orchestrator $(date -u +%FT%TZ)" >> "$SUPLOG"
exec bash "$DIR/run_ladder_windows.sh"
