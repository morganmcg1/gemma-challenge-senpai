#!/usr/bin/env bash
# PR #632 finalize (run AFTER the K-sweep, with GPU free):
#   A) BI=1 AR M=1 greedy reference (drafter OFF) for the int4_mtp_batchinv stack.
#   B) #319 byte-identity gate: official greedy verifier, reference vs K=7 control
#      and vs K* (reuses the A/B arms' own greedy decode/run00.jsonl captures).
#   C) teacher-forced PPL at K* (BI=1, <=2.42 gate).
# BI=1 fixed; only NUM_SPECULATIVE_TOKENS varies. LOCAL only, no HF Job.
set -u
KSTAR="${1:?usage: finalize_kstar.sh <Kstar>}"
ROOT=/workspace/senpai/target
PY="$ROOT/.venv/bin/python"
KS="$ROOT/research/walltps_ab/optionb_bi1_stock_int4/ksweep"
REF="$KS/ar_ref_bi1/decode_outputs.jsonl"
HARN="$ROOT/official/main_bucket/shared_resources/speed_benchmark"
GT="$HARN/data/ppl_ground_truth_tokens.jsonl"
PPLPY="$HARN/scripts/ppl_endpoint.py"
SUB="$ROOT/submissions/int4_mtp_batchinv"
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
PORT=8000; BASE="http://127.0.0.1:$PORT"; MODEL=gemma-4-e4b-it
PPLOUT="$KS/ppl_k${KSTAR}"; mkdir -p "$PPLOUT"
log(){ echo "[fin $(date -u +%H:%M:%S)] $*"; }
cd "$ROOT"

# ---------- Phase A: BI=1 AR M=1 reference (drafter OFF) ----------
if [ -s "$REF" ]; then
  log "reference exists: $REF (skip gen)"
else
  log "generating BI=1 AR M=1 reference -> $REF"
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_BATCH_INVARIANT=1 \
  "$PY" -m scripts.local_validation.gen_greedy_reference \
    --mode served --submission submissions/int4_mtp_batchinv --spec-off \
    --ref-env VLLM_BATCH_INVARIANT=1 --ref-env VLLM_USE_FLASHINFER_SAMPLER=0 \
    --out "$REF" --num-prompts 128 --output-len 512 --seed 1 \
    >"$KS/ref_gen.log" 2>&1
  log "reference gen rc=$? (see ref_gen.log)"
fi

# ---------- Phase B: #319 byte-identity gate (CPU; official verifier) ----------
gate(){  # $1=label $2=candidate_jsonl
  local L=$1 C=$2
  if [ ! -f "$C" ]; then log "GATE $L: candidate MISSING $C"; return 3; fi
  "$PY" -m scripts.local_validation.greedy_gate --reference "$REF" --candidate "$C" --json \
    >"$KS/gate_${L}.json" 2>"$KS/gate_${L}.err"
  local rc=$?
  log "GATE $L exit=$rc (0=IDENTICAL 1=DIVERGENT 2=INCOMPARABLE) -> gate_${L}.json"
  return $rc
}
gate k7 "$KS/k3/k7/decode/run00.jsonl"
gate "k${KSTAR}" "$KS/k${KSTAR}/k${KSTAR}/decode/run00.jsonl"

# ---------- Phase C: teacher-forced PPL at K* (BI=1) ----------
free_port(){ fuser -k ${PORT}/tcp 2>/dev/null; sleep 2; }
wait_ready(){ local t=0; while [ "$t" -lt "$1" ]; do curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0; sleep 3; t=$((t+3)); done; return 1; }
ppl_one(){  # $1=num_spec $2=tag
  local NS=$1 TAG=$2
  free_port
  log "PPL serve BI=1 num_spec=$NS tag=$TAG"
  CUDA_VISIBLE_DEVICES=0 MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct \
  DRAFTER_MODEL=/tmp/qat-assistant NUM_SPECULATIVE_TOKENS=$NS VLLM_BATCH_INVARIANT=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.90 \
  MAX_NUM_BATCHED_TOKENS=512 MAX_NUM_SEQS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HOST=127.0.0.1 PORT=$PORT SERVED_MODEL_NAME=$MODEL \
  setsid "$VENV" "$SUB/serve.py" >"$PPLOUT/server_${TAG}.log" 2>&1 &
  local PID=$!
  if ! wait_ready 480; then
    log "PPL server NOT ready (tag=$TAG); tail:"; tail -20 "$PPLOUT/server_${TAG}.log"
    kill -TERM -- -$PID 2>/dev/null; sleep 3; kill -KILL -- -$PID 2>/dev/null; return 2
  fi
  log "PPL scoring (tag=$TAG)..."
  "$VENV" "$PPLPY" --base-url "$BASE" --model "$MODEL" --dataset-path "$GT" \
    --output-file "$PPLOUT/ppl_results_${TAG}.jsonl" \
    --summary-file "$PPLOUT/ppl_summary_${TAG}.json" --request-timeout-s 180
  local RC=$?
  kill -TERM -- -$PID 2>/dev/null; sleep 4; kill -KILL -- -$PID 2>/dev/null; free_port
  return $RC
}
if ppl_one "$KSTAR" "k${KSTAR}_spec"; then
  log "PPL K=$KSTAR spec-on OK"
else
  log "PPL K=$KSTAR spec-on failed (rc=$?); fallback drafter-off (identical target PPL)"
  ppl_one 0 "k${KSTAR}_specoff" && log "PPL K=$KSTAR specoff OK" || log "PPL K=$KSTAR specoff ALSO FAILED"
fi

log "DONE. summaries:"
for f in "$KS"/gate_*.json "$PPLOUT"/ppl_summary_*.json; do echo "--- $f"; cat "$f" 2>/dev/null; echo; done
touch "$KS/finalize_k${KSTAR}.done"
