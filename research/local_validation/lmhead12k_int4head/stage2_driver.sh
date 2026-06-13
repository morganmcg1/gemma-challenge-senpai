#!/usr/bin/env bash
# Stage 2 driver: unpruned-int4 control + same-session isolated head-dtype TPS
# delta + bandwidth refit. Run AFTER stage1 finishes (needs the free GPU + the
# stage1 int4-head TPS in stage1_evidence/evidence.json).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/workspace/senpai/target
VENVPY=/tmp/server-venv/bin/python
ts() { date -u +%FT%TZ; }

free_gpu() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  local deadline=$(( $(date +%s) + 90 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    echo "[free_gpu] used=${used}MiB $(ts)"
    [ "${used:-99999}" -lt 2500 ] && return 0
    sleep 5
  done
  echo "[free_gpu] WARN memory still high"; return 0
}

cd "$ROOT" || exit 1
echo "=== int4head stage2 (control + isolated TPS + bandwidth) start $(ts) ==="
free_gpu
$VENVPY research/local_validation/lmhead12k_int4head/stage2_isolated_and_control.py
rc=$?
echo "[stage2] rc=$rc $(ts)"
free_gpu
echo "=== int4head stage2 DONE $(ts) (rc=$rc) ==="
