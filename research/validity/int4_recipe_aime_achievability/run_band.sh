#!/usr/bin/env bash
# Multi-session int4-AR greedy AIME band for ONE group-size arm (PR #679).
# For each of N FRESH serve processes: start server -> eval @12288 greedy
# conc=16 over the 60-problem (2024 + 2025-I + 2025-II) set -> kill server ->
# free GPU. Fresh process per session is what controls the 0.828 cross-session
# non-determinism floor (#672 §5).
#
# Usage: run_band.sh <arm-label> <model-dir> <n-sessions> [limit]
set -uo pipefail
ARM="${1:?arm label, e.g. g32}"
MODEL="${2:?model dir}"
N="${3:?n sessions}"
LIMIT="${4:-0}"   # >0 => smoke (cap problems)
START="${5:-0}"   # first session index (lets us extend a band without clobbering)
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE"
PORT=8000
PY=/tmp/vllm0220-srv/bin/python
EVAL=research/downstream_quality_aime/aime_eval.py
MASTER="$OUT/_band_${ARM}.log"

log(){ echo "[band:$ARM] $* $(date -u +%H:%M:%S)" | tee -a "$MASTER"; }

limit_args=()
[ "$LIMIT" -gt 0 ] && limit_args=(--limit "$LIMIT")

log "START model=$MODEL n=$N limit=$LIMIT start=$START"
for ((s=START; s<N; s++)); do
  SLOG="$OUT/_serve_${ARM}s${s}.log"
  OUTJSON="$OUT/${ARM}_session${s}.json"
  log "=== session $s: starting fresh server ==="
  CUDA_VISIBLE_DEVICES=0 bash "$HERE/serve_int4.sh" "$MODEL" "$SLOG" "$PORT" &
  SRV_PID=$!
  # wait for readiness (/v1/models 200), max 240s
  ready=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then log "server DIED during startup (see $SLOG)"; break; fi
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    log "session $s: server NOT ready -> skip"; kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; sleep 5; continue
  fi
  log "session $s: server READY; running eval -> $OUTJSON"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$EVAL" \
    --base-url "http://127.0.0.1:$PORT" \
    --model gemma-4-e4b-it \
    --years 2024,2025-I,2025-II \
    --k 1 --temperature 0 --max-tokens 12288 --min-tokens 8 \
    --no-thinking --client-concurrency 16 --seed 0 \
    --label "${ARM}_s${s}" "${limit_args[@]}" \
    --out "$OUTJSON" >> "$MASTER" 2>&1
  rc=$?
  acc=$("$PY" -c "import json;d=json.load(open('$OUTJSON'));print(f\"{d['maj_k_accuracy']:.4f} ({d['n_correct_maj']}/{d['n_problems']}) extract_fail={d['extract_fail_rate']:.3f}\")" 2>/dev/null || echo "PARSE_FAIL")
  log "session $s: eval rc=$rc acc=$acc"
  kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null
  # wait for GPU memory to free before next fresh process
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
    [ "${used:-9999}" -lt 500 ] && break
    sleep 2
  done
  log "session $s: server down, gpu used=${used:-?}MiB"
done
log "ALL DONE"
