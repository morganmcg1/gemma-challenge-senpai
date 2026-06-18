#!/usr/bin/env bash
# PR #623 instruction #3: teacher-forced PPL gate (<=2.42) for both arms.
# Arms differ ONLY in VLLM_BATCH_INVARIANT. PPL is teacher-forced prefill
# (ppl_endpoint.py: prompt_logprobs=1, max_tokens=1), so it depends only on the
# int4 W4A16 target + this flag; the drafter never runs during scoring. We serve
# the FULL Option-B spec stack (faithful to the arm); if prompt_logprobs is
# rejected under spec, we fall back to NUM_SPECULATIVE_TOKENS=0 (provably identical
# target-only PPL) and tag the record.
set -u
ROOT=/workspace/senpai/target
SUB="$ROOT/submissions/int4_mtp_batchinv"
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
HARNESS="$ROOT/official/main_bucket/shared_resources/speed_benchmark"
GT="$HARNESS/data/ppl_ground_truth_tokens.jsonl"
PPL="$HARNESS/scripts/ppl_endpoint.py"
OUT="$ROOT/research/walltps_ab/optionb_bi1_stock_int4/ppl"
PORT=8000
BASE="http://127.0.0.1:$PORT"
MODEL=gemma-4-e4b-it
mkdir -p "$OUT"

wait_ready () {  # $1 = timeout_s
  local t=0
  while [ "$t" -lt "$1" ]; do
    if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 3; t=$((t+3))
  done
  return 1
}

free_port () { fuser -k ${PORT}/tcp 2>/dev/null; sleep 2; }

serve_one () {  # $1=BI  $2=num_spec  $3=logtag
  # VLLM_USE_FLASHINFER_SAMPLER=0 below matches the paired_tps_ab harness (and the
  # whole profiler family): locally /usr/local/cuda lacks curand.h so the flashinfer
  # sampling JIT fails at profile_run. PPL is teacher-forced (prompt_logprobs), so the
  # sampler is never on the scored path -> the flag has zero effect on the PPL number.
  local BI=$1 NS=$2 TAG=$3
  free_port
  echo "[ppl] === arm BI=$BI num_spec=$NS tag=$TAG ==="
  CUDA_VISIBLE_DEVICES=0 \
  MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct \
  DRAFTER_MODEL=/tmp/qat-assistant \
  NUM_SPECULATIVE_TOKENS=$NS \
  VLLM_BATCH_INVARIANT=$BI \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.90 \
  MAX_NUM_BATCHED_TOKENS=512 MAX_NUM_SEQS=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HOST=127.0.0.1 PORT=$PORT SERVED_MODEL_NAME=$MODEL \
  setsid "$VENV" "$SUB/serve.py" >"$OUT/server_bi${BI}_${TAG}.log" 2>&1 &
  local PID=$!
  echo "[ppl] serve pid=$PID (pgid)"
  if ! wait_ready 480; then
    echo "[ppl] SERVER NOT READY (BI=$BI tag=$TAG) -- last log lines:"
    tail -25 "$OUT/server_bi${BI}_${TAG}.log"
    kill -TERM -- -$PID 2>/dev/null; sleep 3; kill -KILL -- -$PID 2>/dev/null
    return 2
  fi
  echo "[ppl] ready; scoring PPL..."
  "$VENV" "$PPL" --base-url "$BASE" --model "$MODEL" \
    --dataset-path "$GT" \
    --output-file "$OUT/ppl_results_bi${BI}_${TAG}.jsonl" \
    --summary-file "$OUT/ppl_summary_bi${BI}_${TAG}.json" \
    --request-timeout-s 180
  local RC=$?
  kill -TERM -- -$PID 2>/dev/null; sleep 4; kill -KILL -- -$PID 2>/dev/null
  free_port
  return $RC
}

for BI in 0 1; do
  if serve_one "$BI" 7 spec; then
    echo "[ppl] BI=$BI spec-on OK"
  else
    echo "[ppl] BI=$BI spec-on failed (rc=$?); falling back to drafter-off (identical target PPL)"
    serve_one "$BI" 0 specoff && echo "[ppl] BI=$BI specoff OK" || echo "[ppl] BI=$BI specoff ALSO FAILED"
  fi
done

echo "[ppl] DONE. summaries:"
for f in "$OUT"/ppl_summary_*.json; do echo "--- $f"; cat "$f"; echo; done
