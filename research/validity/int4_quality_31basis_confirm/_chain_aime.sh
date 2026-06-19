#!/usr/bin/env bash
# PR #696 -- chain the AIME leg (run_aime.py) to start AFTER the GPQA 30-seed chain
# (chainD pid in _chainD.pid runs GPQA seeds 23-29) exits and the single A10G frees.
# Smoke (2 problems greedy) gates the full sweep so a misconfig can't waste hours.
# LOCAL, analysis_only, NO HF JOB, NO FIRE. Idempotent: run_aime.py skips existing files.
set -u
cd /workspace/senpai/target/research/validity/int4_quality_31basis_confirm
PY=/tmp/senpai-venvs/5f4c623f772358a2/bin/python

CHAIND_PID="$(cat _chainD.pid 2>/dev/null || echo '')"
echo "[chainAIME] $(date +%H:%M:%S) waiting for chainD pid=${CHAIND_PID} (GPQA seeds 23-29) to exit"
if [ -n "${CHAIND_PID}" ]; then
  while kill -0 "${CHAIND_PID}" 2>/dev/null; do sleep 30; done
fi
echo "[chainAIME] $(date +%H:%M:%S) chainD exited; waiting for GPU memory to free (<4000 MiB)"
used=99999
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${used:-99999}" -lt 4000 ]; then break; fi
  sleep 10
done
echo "[chainAIME] $(date +%H:%M:%S) GPU used=${used}MiB; running AIME smoke (2 problems)"
if ! "$PY" -u run_aime.py --smoke; then
  echo "[chainAIME] $(date +%H:%M:%S) SMOKE FAILED -- aborting full sweep"
  exit 1
fi
echo "[chainAIME] $(date +%H:%M:%S) smoke OK; launching greedy anchor + 10 #31-sampled seeds"
exec "$PY" -u run_aime.py --greedy --sampled-seeds 0,1,2,3,4,5,6,7,8,9
