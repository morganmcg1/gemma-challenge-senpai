#!/usr/bin/env bash
# PR #614 root-cause probe -- serve the UNQUANTIZED vanilla bf16 base
# google/gemma-4-E4B-it on the dev307 engine, faithfully reproducing kanna #563's
# PLAIN-vLLM fp16 serve (the recipe that measured base GPQA-Diamond = 0.5236), so we
# can isolate which of MY serve deltas (max-model-len 6144 / max-num-batched-tokens
# 2048) corrupts the token stream. #563-faithful flags: dtype bf16, gpu-mem 0.90,
# max-num-seqs 16, VLLM_USE_FLASHINFER_SAMPLER=0, --override-generation-config, and
# crucially NO max-num-batched-tokens override (vLLM default) unless explicitly asked.
#
# Usage: serve_probe.sh <max_model_len> [max_num_batched_tokens|none] [port]
#   serve_probe.sh 4096            -> #563-exact reference (no batched-tokens override)
#   serve_probe.sh 6144            -> PR-mandated len, no batched override
#   serve_probe.sh 6144 2048       -> reproduce my corrupted serve (len 6144 + chunk 2048)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=/tmp/senpai-venvs/5f4c623f772358a2          # pinned dev307 (#580); same engine #563 used
PYV="$VENV/bin/python"

MML="${1:-4096}"
MNBT="${2:-none}"
PORT="${3:-8000}"
LOG="$HERE/_server_probe_mml${MML}_mnbt${MNBT}.log"
PIDFILE="$HERE/_server_probe.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# PLE_FOLD=1 -> apply the #557-root-caused Gemma4 Per-Layer-Embedding embed-scale fold
# (=sqrt(256)=16.0). dev307 dropped the runtime PLE x16 multiply and gated it behind this
# env (model_loader/utils.py:130). Without it the vanilla bf16 base feeds 16x-too-small
# per-layer embeddings into every decoder layer -> corrupted long-CoT decode. TARGET_MODEL
# must equal --model so the native fold safety-check applies it.
if [[ "${PLE_FOLD:-0}" == "1" ]]; then
  export PLE_FOLD_EMBED_SCALE=1
  export PLE_FOLD_TARGET_MODEL="google/gemma-4-E4B-it"
fi

ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model google/gemma-4-E4B-it
  --served-model-name gemma-4-e4b-it
  --host 127.0.0.1 --port "$PORT"
  --dtype bfloat16
  --max-model-len "$MML"
  --gpu-memory-utilization 0.90
  --max-num-seqs 16
  --seed 0
  --trust-remote-code
  --no-enable-log-requests
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":64}'
)
if [[ "$MNBT" != "none" ]]; then
  ARGS+=(--max-num-batched-tokens "$MNBT")
fi
# ENFORCE_EAGER=1 disables torch.compile + CUDA-graph capture (the prime suspect for
# build-level token corruption on this dev307 / torch 2.11+cu130 stack).
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

echo "[probe-serve] model=google/gemma-4-E4B-it bf16 max_model_len=$MML max_num_batched_tokens=$MNBT gpu_util=0.90 max_num_seqs=16 flashinfer_sampler=OFF dev307 $(date -u +%H:%M:%SZ)" | tee "$LOG"
echo "[probe-serve] argv: ${ARGS[*]}" | tee -a "$LOG"

setsid "$PYV" "${ARGS[@]}" >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[probe-serve] pid=$SRV_PID -> $LOG"

for i in $(seq 1 360); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[probe-serve] FAILED: process exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[probe-serve] READY after ${i}x5s ($(date -u +%H:%M:%SZ))"; exit 0
  fi
  sleep 5
done
echo "[probe-serve] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
