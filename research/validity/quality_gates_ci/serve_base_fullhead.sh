#!/usr/bin/env bash
# PR #590 -- serve `base_fullhead` for the multi-seed quality-gate CI card.
#
# base_fullhead = the STOCK int4_g32 QAT checkpoint google/gemma-4-E4B-it-qat-w4a16-ct
# served vanilla. The checkpoint's quantization `ignore` list contains `lm_head`
# (config.json verified) and vocab_size=262144, tie_word_embeddings=true, so the head
# is the FULL native 262k bf16 (tied) embedding matrix -- i.e. "full head" comes free
# from the stock checkpoint, no prune/transplant needed. Body is int4 compressed-tensors.
#
# Recipe (anchor, wirbel #553 / PR #590): spec-OFF (no --speculative-config),
# VLLM_USE_FLASHINFER_SAMPLER=0 (box has no curand headers for the JIT sampler),
# TRITON_ATTN auto-forced by gemma4 heterogeneous head dims. Decode protocol is set
# PER-REQUEST by the eval harnesses (lewtun #31: temp=1.0 top_p=0.95 top_k=64) + a
# request-level min_tokens=8 EOS-guard (#541), so the server generation-config default
# is irrelevant. MAX_NUM_SEQS raised to 32 for eval batching/throughput (vanilla vLLM
# decode is per-sequence; batch size does not change the sampled distribution).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=/tmp/eval-serve-venv
PORT="${PORT:-8000}"
MODEL_DIR=/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0
LOG="$HERE/_server_base_fullhead.log"
PIDFILE="$HERE/_server_base_fullhead.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
export VLLM_USE_FLASHINFER_SAMPLER=0
# pck04_inject sitecustomize installs the prometheus-instrumentator route-name compat
# shim (vLLM 0.22.1rc1 + newer FastAPI otherwise 500s every request incl. /v1/models).
# With PCK04_KEEPSET UNSET the lm_head patch is a no-op -> the base arm stays numerically
# vanilla; only the metrics-labeling shim applies. Same dir start_server.sh's base arm uses.
unset PCK04_KEEPSET || true
export PYTHONPATH="/workspace/senpai/target/research/validity/downstream_quality_eval/pck04_inject${PYTHONPATH:+:$PYTHONPATH}"

echo "[serve] base_fullhead model=$MODEL_DIR port=$PORT max_num_seqs=32 spec=OFF flashinfer_sampler=OFF $(date -u +%H:%M:%SZ)" | tee "$LOG"

setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --trust-remote-code \
  --disable-log-stats \
  >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] pid=$SRV_PID -> $LOG"

for i in $(seq 1 300); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve] FAILED: process exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve] READY after ${i}x5s ($(date -u +%H:%M:%SZ))"; exit 0
  fi
  sleep 5
done
echo "[serve] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
