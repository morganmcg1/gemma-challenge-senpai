#!/usr/bin/env bash
# PR #661 -- serve the shipped int4_g128_lmhead body OR the bf16 base, on vLLM
# 0.22.0, autoregressive (NO spec/draft), at the gb6144 M=1-AR panel with
# **MAX_NUM_SEQS=1** -- the advisor's #515 four-axis gate panel. This is the
# seqs=1 counterpart of #638's serve_{int4ar,bf16}_0p22.sh (which ran seqs=16);
# the ONLY deltas vs #638 are (a) MAX_NUM_SEQS 16 -> 1 and (b) the venv path (the
# old /tmp/vllm0220-srv was wiped; the dep-keyed survivor is reused verbatim).
#
# Why seqs=1: under VLLM_BATCH_INVARIANT=1 the aten::mm override is bit-identical
# across the decode-batch row count M (kanna #19: M=1 vs M=7 max|diff|=0), but the
# int4 **Marlin** weight-GEMM is a `_C` op the override cannot reach, so the int4
# body can carry residual cross-request (MAX_NUM_SEQS) batch-variance. seqs=1
# isolates the body at the exact panel that produced the banked AIME 0.4000 /
# GPQA-D 0.4798 cells, so all four gate axes sit on ONE panel.
#
# Usage: serve_panel.sh {int4ar|bf16}
set -euo pipefail

ARM="${1:-}"
[[ "$ARM" == "int4ar" || "$ARM" == "bf16" ]] || { echo "[serve] usage: serve_panel.sh int4ar|bf16 (got '$ARM')"; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV="${VENV:-/tmp/senpai-venvs/20f658587e8a6643}"   # vLLM 0.22.0 + transformers 5.9.0 survivor
SERVE_PY="$ROOT/submissions/bf16_base_aime/serve.py"  # plain AR wrapper, no drafter (same as #638)
PORT="${PORT:-8000}"

case "$ARM" in
  int4ar)
    CKPT=/workspace/gemma_build/int4_g128_lmhead        # PR #4 live-rung body (built local)
    ;;
  bf16)
    CKPT=/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187
    ;;
  *) echo "[serve] FATAL: arm must be int4ar|bf16, got '$ARM'"; exit 2;;
esac

LOG="$HERE/_server_${ARM}.log"
PIDFILE="$HERE/_server_${ARM}.pid"

if [[ ! -f "$VENV/bin/python" ]]; then
  echo "[serve] FATAL: serve venv missing at $VENV"; exit 2
fi
if [[ ! -f "$CKPT/config.json" ]]; then
  echo "[serve] FATAL: checkpoint missing at $CKPT"; exit 2
fi

# One mapped A10G is NVML index 0; inherited host CUDA_VISIBLE_DEVICES / void NVIDIA_* break NVML.
export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true

# prometheus _get_route_name .path-guard shim (else every request 500s under this box's FastAPI).
# PCK04_KEEPSET stays UNSET so the lm_head patch is inert -- both bodies stay numerically pure.
INJECT_DIR="$ROOT/research/validity/downstream_quality_eval/pck04_inject"
export PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset PCK04_KEEPSET || true

export MODEL_ID="$CKPT"
export SERVED_MODEL_NAME=gemma-4-e4b-it
export HOST=127.0.0.1
export PORT
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"               # <-- the panel knob: seqs=1
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_SEED=0
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[serve] arm=$ARM model=$CKPT mml=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS BI=$VLLM_BATCH_INVARIANT venv=$VENV $(date -u +%FT%TZ)" | tee "$LOG"
"$VENV/bin/python" -c "import vllm; print('[serve] vllm', vllm.__version__)" | tee -a "$LOG"

setsid "$VENV/bin/python" "$SERVE_PY" >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] pid=$SRV_PID log=$LOG"

# int4 (~9 GB) / bf16 (~16 GB) cold load + (int4) Marlin repack + compile: allow up to ~25 min.
for i in $(seq 1 300); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve] FAILED: server exited early; tail:"; tail -60 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve] READY arm=$ARM after ${i}x5s ($(date -u +%FT%TZ))"; exit 0
  fi
  sleep 5
done
echo "[serve] TIMEOUT; tail:"; tail -80 "$LOG"; exit 1
