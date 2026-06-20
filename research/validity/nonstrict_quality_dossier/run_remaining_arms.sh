#!/usr/bin/env bash
# PR #762 wirbel -- chain the two remaining quality-panel arms after bi1_fire.
# Waits for the in-flight bi1_fire panel process to exit, frees the GPU between
# arms (one server fits the A10G), then runs bi0_nonstrict and int4_base. LOCAL
# ONLY: no HF Job, no --launch, no served-file change.
set -u
ROOT=/workspace/senpai/target
DOSSIER="$ROOT/research/validity/nonstrict_quality_dossier"
OUT="$DOSSIER/out"
PY=/tmp/eval-serve-venv/bin/python
cd "$ROOT" || exit 2

log() { echo "[driver $(date -u +%H:%M:%S)] $*"; }

# 1) wait for the running bi1_fire panel to finish
BI1_PID="$(cat "$OUT/run_bi1_fire.pid" 2>/dev/null || true)"
if [ -n "${BI1_PID:-}" ]; then
  log "waiting for bi1_fire panel pid=$BI1_PID to exit"
  while kill -0 "$BI1_PID" 2>/dev/null; do sleep 20; done
  log "bi1_fire panel exited"
fi

wait_gpu_free() {
  # block until <8 GiB used (server torn down) or 180s elapsed
  local i=0
  while [ "$i" -lt 36 ]; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$used" ] && [ "$used" -lt 8000 ]; then
      log "gpu free (${used} MiB used)"; return 0
    fi
    sleep 5; i=$((i+1))
  done
  log "WARN gpu still busy after wait (${used:-?} MiB)"; return 0
}

run_arm() {
  local arm="$1"
  wait_gpu_free
  log "START arm=$arm"
  "$PY" "$DOSSIER/run_panel.py" --arm "$arm" > "$OUT/run_${arm}.log" 2>&1
  local rc=$?
  log "DONE arm=$arm rc=$rc"
  return $rc
}

run_arm bi0_nonstrict
run_arm int4_base
log "ALL REMAINING ARMS DONE"
