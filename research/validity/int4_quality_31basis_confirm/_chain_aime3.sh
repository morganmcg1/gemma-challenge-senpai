#!/usr/bin/env bash
# PR #696 -- AIME leg RE-LAUNCH after the run_aime.py base-url fix.
#
# Why this exists: _chain_aime2.sh's thinking smoke aborted with HTTP 404 because run_aime.py
# passed --base-url http://127.0.0.1:8001/v1 to aime_eval.py, which itself appends
# '/v1/chat/completions' -> the request path doubled to '/v1/v1/...'. (The inspect openai-api
# harness wants the '/v1' suffix; aime_eval.py does NOT -- different convention.) Fixed: run_aime.py
# now passes the server ROOT. The smoke gate did its job: zero GPU was wasted on the broken config.
#
# This re-launch waits for renew09 (the GPQA seeds-0-9 venv-homogenize re-run) to free the single
# A10G, then re-runs the AIME leg with the corrected client. Idempotent (run_aime.py skips existing).
#   think   leg -- decision-grade (gate 0.420; reconcile int4-body #31 point to lawine #693's 0.3467)
#   nothink leg -- SECONDARY floor cross-check (gate 0.090)
# LOCAL, analysis_only, NO HF JOB, NO submission, NO FIRE.
set -u
cd /workspace/senpai/target/research/validity/int4_quality_31basis_confirm
PY=/tmp/senpai-venvs/5f4c623f772358a2/bin/python

RENEW_PID="$(cat _chain_renew09.pid 2>/dev/null || echo '')"
echo "[aime3] $(date +%H:%M:%S) waiting for renew09 pid=${RENEW_PID} (GPQA seeds 0-9 homogenize) to exit"
if [ -n "${RENEW_PID}" ]; then
  while kill -0 "${RENEW_PID}" 2>/dev/null; do sleep 30; done
fi
echo "[aime3] $(date +%H:%M:%S) renew09 exited; waiting for GPU memory to free (<4000 MiB)"
used=99999
for _ in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${used:-99999}" -lt 4000 ]; then break; fi
  sleep 10
done
echo "[aime3] $(date +%H:%M:%S) GPU used=${used}MiB; thinking smoke (2 problems) to gate the sweep"
if ! "$PY" -u run_aime.py --thinking --smoke; then
  echo "[aime3] $(date +%H:%M:%S) THINKING SMOKE FAILED -- aborting"
  exit 1
fi
echo "[aime3] $(date +%H:%M:%S) smoke OK; THINK leg: greedy (wall decomposition) + #31-sampled 0,1,2"
"$PY" -u run_aime.py --thinking --greedy --sampled-seeds 0,1,2
echo "[aime3] $(date +%H:%M:%S) THINK leg done; NOTHINK cross-check (secondary): #31-sampled 0,1,2"
"$PY" -u run_aime.py --sampled-seeds 0,1,2
echo "[aime3] $(date +%H:%M:%S) AIME ALL DONE"
