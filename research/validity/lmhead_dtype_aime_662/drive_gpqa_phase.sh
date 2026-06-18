#!/usr/bin/env bash
# PR #662 -- GPQA-Diamond guard (n=198, greedy CoT) for the 3 head-dtype arms.
# 3rd-priority guard: confirm the head dtype does NOT trade AIME for GPQA
# (precision-insensitive per #646 -> expected flat). Runs AFTER the AIME phase.
#
# Per arm: serve via serve_arm.sh (vLLM 0.22.0, BI=1, MAX_NUM_SEQS=1, identical
# boot to the AIME arms) -> GPQA-D via the inspect_ai eval venv on --base-url ->
# teardown. One server at a time (int4/bf16 + KV fills the A10G).
#
# GREEDY (temperature 0), the same decode protocol as the #639/baseline GPQA so
# the >= bar comparison stays apples-to-apples. analysis_only; no served change.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_VENV=/tmp/eval-serve-venv
RUN_EVAL=/workspace/senpai/target/research/validity/downstream_quality_eval/run_eval.py
PORT="${PORT:-8000}"
ALL_MARKER="$HERE/_all_gpqa_done.marker"
rm -f "$ALL_MARKER"
log(){ echo "[gpqa-drive $(date -u +%T)Z] $*"; }

# arm name -> checkpoint dir
declare -A DIRS=(
  [shipped_g128]=/workspace/gemma_build/int4_g128_lmhead
  [our_g128_int8head]=/workspace/gemma_build/our_g128_int8head
  [our_g128_bf16head]=/workspace/gemma_build/our_g128_bf16head
)
ORDER=(shipped_g128 our_g128_int8head our_g128_bf16head)

for ARM in "${ORDER[@]}"; do
  MODEL_DIR="${DIRS[$ARM]}"
  OUT="$HERE/results/gpqa_${ARM}.json"
  if [[ -f "$OUT" ]]; then log "skip $ARM (already have $OUT)"; continue; fi
  log "=== $ARM serve ($MODEL_DIR) ==="
  ARM="$ARM" MODEL_DIR="$MODEL_DIR" PORT="$PORT" bash "$HERE/serve_arm.sh" \
    > "$HERE/_gpqa_serve_${ARM}.log" 2>&1
  if [[ $? -ne 0 ]]; then log "$ARM SERVE FAILED"; continue; fi
  SRV_PID="$(cat "$HERE/_server_${ARM}.pid")"

  log "=== $ARM GPQA-D n=198 greedy ==="
  "$EVAL_VENV/bin/python" "$RUN_EVAL" \
    --task gpqa_diamond --arm "$ARM" \
    --base-url "http://127.0.0.1:${PORT}/v1" --model gemma-4-e4b-it \
    --max-tokens 2048 --temperature 0 --top-p 1.0 --max-connections 16 \
    --seed 12345 --out "$OUT" \
    > "$HERE/_gpqa_${ARM}.log" 2>&1
  log "$ARM GPQA rc=$? $(grep -o 'acc=[0-9.]*' "$HERE/_gpqa_${ARM}.log" | tail -1)"

  kill "$SRV_PID" 2>/dev/null || true
  for i in $(seq 1 20); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SRV_PID" 2>/dev/null || true
  sleep 5
done

echo "all_gpqa_done $(date -u +%FT%TZ)" > "$ALL_MARKER"
log "ALL GPQA ARMS DONE"
