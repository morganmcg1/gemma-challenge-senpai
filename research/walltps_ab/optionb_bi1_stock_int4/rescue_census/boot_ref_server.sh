#!/usr/bin/env bash
# PR #651: boot the spec-OFF M=1 AR reference engine (the SAME engine that generated
# ar_ref_bi1/decode_outputs.jsonl) via the submission's own launcher. serve.py honors
# SENPAI_REFERENCE_MODE=1 by forcing num_speculative_tokens=0 -> plain int4 W4A16 target,
# drafter OFF -> the canonical greedy M=1 AR path. This is stark #636's "M=1 recompute"
# engine; it is K-INDEPENDENT (no spec), so we boot it ONCE and reuse for every K's
# fired-position recompute. BI=1, FlashInfer-sampler off, MAX_NUM_SEQS=1 -> faithful M=1
# decode (the int4 Marlin GEMM is M-dependent: a batched/prefill path would change the
# argmax at the very near-ties we are probing, so the recompute MUST be single-seq M=1).
# Faithful to ar_ref_bi1/meta.json ref_env + ref_gen.log.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
SERVE="$ROOT/submissions/int4_mtp_batchinv/serve.py"
LOG="$HERE/ref_server.log"
PIDF="$HERE/ref_server.pid"

# stale inherited CUDA_VISIBLE_DEVICES=7 -> physical GPU is index 0 (memory: gpu_env.md)
export CUDA_VISIBLE_DEVICES=0
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export SENPAI_REFERENCE_MODE=1            # spec OFF -> M=1 AR reference (the recompute engine)
export MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct
export DRAFTER_MODEL=/tmp/qat-assistant   # ignored under reference mode (num_spec forced 0)
export SERVED_MODEL_NAME=gemma-4-e4b-it
export HOST=127.0.0.1
export PORT=8000
export MAX_MODEL_LEN=4096
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_SEQS=1
export MAX_NUM_BATCHED_TOKENS=512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[boot] M=1 AR reference (SENPAI_REFERENCE_MODE=1, spec OFF) via serve.py -> $LOG"
nohup "$VENV" "$SERVE" >"$LOG" 2>&1 &
echo $! > "$PIDF"
echo "[boot] pid $(cat "$PIDF"); polling /health ..."

for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "[boot] healthy after ${i}0s"
    # confirm spec really is OFF in the log (defensive)
    if grep -q "forcing num_speculative_tokens=0" "$LOG"; then
      echo "[boot] confirmed reference mode (spec OFF)"
    else
      echo "[boot] WARNING: reference-mode banner not found in log; check spec is OFF"
    fi
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "[boot] server process died; tail log:"; tail -30 "$LOG"; exit 1
  fi
  sleep 10
done
echo "[boot] timeout waiting for health; tail log:"; tail -30 "$LOG"; exit 1
