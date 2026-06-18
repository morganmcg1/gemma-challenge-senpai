#!/usr/bin/env bash
# PR #632 K-sweep driver: after the in-flight k3 paired_tps_ab job (which banks the
# reusable k7 baseline) exits, run K=4,5,6 as candidate-only arms reusing that k7
# baseline (valid: PR #72 restart-invariance). BI=1 fixed; only NUM_SPECULATIVE_TOKENS
# varies. LOCAL only, no HF Job.
set -u
ROOT=/workspace/senpai/target
PY="$ROOT/.venv/bin/python"
KS="$ROOT/research/walltps_ab/optionb_bi1_stock_int4/ksweep"
BASE_JSON="$KS/k3/paired_ab.json"
WAIT_PID="${1:-610727}"     # k3 job pid to wait on

log(){ echo "[k456 $(date -u +%H:%M:%S)] $*"; }

# 1) wait for the k3 job to finish (numeric-pid kill -0, never pgrep -f)
log "waiting for k3 job pid=$WAIT_PID to exit"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
log "k3 job exited"

if [ ! -f "$BASE_JSON" ]; then
  log "FATAL: $BASE_JSON missing (k3 job did not produce reusable k7 baseline); aborting"
  exit 1
fi

gpu_free(){  # block until GPU mostly free (<1500 MiB used) or timeout
  local t=0
  while [ "$t" -lt 180 ]; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 1500 ] && return 0
    sleep 5; t=$((t+5))
  done
  log "WARN: GPU still busy after ${t}s (used=${used:-?}MiB); proceeding anyway"
}

run_k(){
  local K=$1
  local OUT="$KS/k${K}"
  if [ -f "$OUT/paired_ab.json" ]; then
    log "k${K} already has paired_ab.json; skipping"
    return 0
  fi
  gpu_free
  log "=== launching K=${K} (reuse k7 baseline) ==="
  cd "$ROOT"
  "$PY" scripts/profiler/paired_tps_ab.py \
    --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
    --baseline-label k7 --candidate-label "k${K}" \
    --reuse-baseline-from "$BASE_JSON" \
    --baseline-env VLLM_BATCH_INVARIANT=1 \
    --baseline-env MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct \
    --baseline-env DRAFTER_MODEL=/tmp/qat-assistant \
    --baseline-env NUM_SPECULATIVE_TOKENS=7 \
    --candidate-env VLLM_BATCH_INVARIANT=1 \
    --candidate-env MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct \
    --candidate-env DRAFTER_MODEL=/tmp/qat-assistant \
    --candidate-env "NUM_SPECULATIVE_TOKENS=${K}" \
    --n 3 --num-prompts 128 --output-len 512 --seed 1 \
    --out-dir "$OUT" --no-project \
    --wandb-name "land/optionb-bi1-ksweep-k${K}" --wandb-group optionb-bi1-k-sweep \
    >"$KS/inv_k${K}.log" 2>&1
  local rc=$?
  log "K=${K} exited rc=$rc"
  return $rc
}

for K in 4 5 6; do
  run_k "$K" || log "K=${K} returned nonzero (see inv_k${K}.log)"
done

touch "$KS/sweep_456.done"
log "ALL DONE -> $KS/sweep_456.done"
