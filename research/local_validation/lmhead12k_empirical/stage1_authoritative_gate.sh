#!/usr/bin/env bash
# Stage 1: AUTHORITATIVE served-vs-served greedy gate for the pruned submission,
# using the canonical harness (scripts/local_validation), per advisor 15:42Z.
#
#   ref  = gen_greedy_reference --mode served --submission <dir> --spec-off
#          (serves MY serve.py + plugin, M=1 AR, writes research/greedy_reference/<tag>/)
#   cand = validate_submission --submission <dir>
#          (serves the SAME stack on the real path, captures candidate, runs the
#           greedy gate vs the served reference + local PPL + exploratory TPS)
#
# Both serve the pruned checkpoint with the IDENTICAL manifest env (the candidate's
# _headroom_overrides resolve to {} because the manifest already sets all three
# headroom keys), so any divergence is pure single-stream FP non-determinism of
# the served path -> the self-consistency gate. CVD is normalized to 0 by the
# harness, but we export it too for the kill/guard helpers here.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0

ROOT=/workspace/senpai/target
VENVPY=/tmp/server-venv/bin/python
SUB=submissions/lmhead12k_empirical
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
echo "=== stage1 authoritative gate start $(ts) ==="
free_gpu

echo "=== STEP A: served spec-off reference (pruned, MY stack) $(ts) ==="
$VENVPY -m scripts.local_validation.gen_greedy_reference \
  --mode served --submission "$SUB" --spec-off \
  --server-python "$VENVPY"
rc_ref=$?
echo "[ref] rc=$rc_ref $(ts)"
free_gpu

if [ "$rc_ref" -ne 0 ]; then
  echo "[FATAL] reference generation failed (rc=$rc_ref); aborting before validate."
  exit "$rc_ref"
fi

echo "=== STEP B: validate_submission (served candidate + gate + PPL + TPS) $(ts) ==="
$VENVPY -m scripts.local_validation.validate_submission \
  --submission "$SUB" \
  --server-python "$VENVPY"
rc_val=$?
echo "[validate] rc=$rc_val $(ts)"
free_gpu

echo "=== stage1 authoritative gate DONE $(ts) (ref rc=$rc_ref validate rc=$rc_val) ==="
