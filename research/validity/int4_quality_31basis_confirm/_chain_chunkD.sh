#!/usr/bin/env bash
# PR #696 -- chain GPQA boot-D (seeds 23-29) to start AFTER chunkC (driver pid in
# _chunkC.driver.pid) exits and port 8000 frees, so the 30-seed pool completes
# without two servers fighting for the single A10G. Idempotent: run_gpqa_seeds.py
# skips existing seed files. LOCAL, analysis_only, NO FIRE.
set -u
cd /workspace/senpai/target/research/validity/int4_quality_31basis_confirm

CHUNKC_PID="$(cat _chunkC.driver.pid 2>/dev/null || echo '')"
echo "[chainD] $(date +%H:%M:%S) waiting for chunkC pid=${CHUNKC_PID} to exit"
if [ -n "${CHUNKC_PID}" ]; then
  while kill -0 "${CHUNKC_PID}" 2>/dev/null; do sleep 20; done
fi
echo "[chainD] $(date +%H:%M:%S) chunkC exited; waiting for port 8000 to free"
for _ in $(seq 1 30); do
  if ss -ltn 2>/dev/null | grep -q ':8000'; then sleep 5; else break; fi
done
echo "[chainD] $(date +%H:%M:%S) launching GPQA seeds 23-29"
exec /tmp/senpai-venvs/5f4c623f772358a2/bin/python -u run_gpqa_seeds.py \
  --seeds 23,24,25,26,27,28,29
