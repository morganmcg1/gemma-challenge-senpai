#!/usr/bin/env bash
# PR #627 -- continuation orchestrator. Chains AFTER the run_serverH_arms.sh driver
# finishes its last arm (H_conc1_mt3072): tears down the mml=6144 healthy server,
# frees the GPU, brings up the EXACT submission serving config (serve.py:
# MAX_MODEL_LEN=4096, MAX_NUM_BATCHED_TOKENS=512, default max_num_seqs) on vLLM
# 0.22.0, and runs the DECISIVE single-stream GPQA arm twice:
#   sub_conc1        limit=100  (apples-to-apples with H_conc1's matched 100 Qs)
#   sub_conc1_full   n=198      (binding gate number)
#
# Usage: continue_to_submission.sh <driver_pid> <h_server_pid>
set -u
cd /workspace/senpai/target
DRIVER_PID="${1:?run_serverH_arms.sh driver pid}"
HSRV_PID="${2:?mml=6144 H api_server pid}"
HERE=research/validity/optionb_crater_config_axis
VLLM_PY=/workspace/senpai/target/.venv/bin/python
LOG="$HERE/logs/_continue.log"
mkdir -p "$HERE/logs"
exec >>"$LOG" 2>&1
echo "[continue] START $(date -u +%FT%TZ) driver=$DRIVER_PID hsrv=$HSRV_PID"

# 1) Wait for the config-axis driver to finish its last arm, then exit.
while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 20; done
echo "[continue] driver $DRIVER_PID exited $(date -u +%FT%TZ)"
sleep 5

# 2) Tear down the healthy mml=6144 server and free the GPU before re-serving.
echo "[continue] stopping H server pid=$HSRV_PID"
kill "$HSRV_PID" 2>/dev/null || true
# cascade to the process session (setsid leader) in case children linger
kill -- -"$HSRV_PID" 2>/dev/null || true
for i in $(seq 1 30); do
  kill -0 "$HSRV_PID" 2>/dev/null || { echo "[continue] H api_server down after ${i}x2s"; break; }
  sleep 2
done
# poll until GPU memory actually frees (engine core child may outlive the api proc)
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "[continue] gpu_used=${used}MiB (wait ${i})"
  if [[ -n "$used" && "$used" -lt 2000 ]]; then echo "[continue] GPU free (${used}MiB)"; break; fi
  # after 20s of waiting, hard-kill any leftover vllm engine cores
  if [[ "$i" -ge 10 ]]; then pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true; pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true; fi
  sleep 3
done

# 3) Bring up the EXACT submission serving config.
echo "[continue] serving submission config $(date -u +%FT%TZ)"
if ! bash "$HERE/serve_submission_config.sh" "$VLLM_PY" 8000; then
  echo "[continue] FATAL: submission serve failed to come up"; exit 1
fi

# 4) DECISIVE arm: matched-100 first (vs H_conc1), then full n=198 binding number.
echo "[continue] decisive sub_conc1 limit=100 $(date -u +%FT%TZ)"
bash "$HERE/run_decisive_submission.sh" 100

# Rename the matched-100 output so the full run does not clobber it.
if [[ -f "$HERE/runs/gpqa_sub_conc1.json" ]]; then
  mv "$HERE/runs/gpqa_sub_conc1.json" "$HERE/runs/gpqa_sub_conc1_n100.json"
  rm -rf "$HERE/logs/gpqa_sub_conc1_n100"; mv "$HERE/logs/gpqa_sub_conc1" "$HERE/logs/gpqa_sub_conc1_n100" 2>/dev/null || true
fi

echo "[continue] decisive sub_conc1 FULL n=198 $(date -u +%FT%TZ)"
bash "$HERE/run_decisive_submission.sh" 0

echo "[continue] DONE $(date -u +%FT%TZ)"
# leave the submission server up for any follow-up; harvest step will stop it.
