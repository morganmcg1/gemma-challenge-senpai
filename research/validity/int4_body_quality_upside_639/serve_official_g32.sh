#!/usr/bin/env bash
# PR #639 Arm 1 -- serve Google's OFFICIAL int4-QAT recipe
# google/gemma-4-E4B-it-qat-w4a16-ct (group_size 32, tied bf16 lm_head, bf16
# vision/audio towers) on vLLM *0.22.0* (the engine the locked-rung submission
# int4_g128_lmhead pins). This is the REFERENCE arm: Google's own g32 recipe served
# through the same 0.22.0 Marlin path as the ubel #628 bf16 denominator, so its
# GPQA-D sampled number is apples-to-apples with bf16 base 0.5404 / Option-B 0.4652.
#
# Config is BYTE-FOR-BYTE the ubel #628 serve_bf16_0p22.sh (max-model-len 8192,
# max-num-seqs 16, BI=1, gpu-mem-util 0.90, max-num-batched-tokens 2048,
# VLLM_USE_FLASHINFER_SAMPLER=0, pck04 prometheus shim, PCK04_KEEPSET unset) EXCEPT
# the served model path. Reuses submissions/bf16_base_aime/serve.py verbatim; the
# int4 weights come from the checkpoint's compressed-tensors quantization_config
# (Marlin), --dtype bfloat16 only sets the activation dtype.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV="${VENV:-/tmp/vllm0220-srv}"
SERVE_PY="$ROOT/submissions/bf16_base_aime/serve.py"
# Google's official int4-QAT g32 checkpoint (this student's HF cache snapshot).
SNAP="$(ls -d /senpai-run/home/student-lawine/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/*/ 2>/dev/null | head -1)"
PORT="${PORT:-8000}"
LOG="$HERE/_server_official_g32.log"
PIDFILE="$HERE/_server_official_g32.pid"

if [[ -z "$SNAP" || ! -f "$SNAP/model.safetensors" ]]; then
  echo "[serve] FATAL: official g32 snapshot not found / incomplete: '$SNAP'"; exit 2
fi

# One mapped A10G is NVML index 0; the inherited host CUDA_VISIBLE_DEVICES and
# NVIDIA_VISIBLE_DEVICES=void both break NVML -> override/unset.
export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true

# vLLM mounts prometheus_fastapi_instrumentator whose _get_route_name does
# route.path on every app.route; under this box's FastAPI some entries are
# _IncludedRouter objects with no .path -> EVERY request 500s. The #614 harness
# sitecustomize shim (auto-imported via PYTHONPATH in BOTH APIServer and the
# spawned EngineCore) installs a .path-guarded _get_route_name. PCK04_KEEPSET is
# left UNSET so only the prometheus shim fires (the pck04 lm_head patch stays
# inert) and the arm is numerically pure-vanilla int4 Marlin.
INJECT_DIR="$ROOT/research/validity/downstream_quality_eval/pck04_inject"
export PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset PCK04_KEEPSET || true

export MODEL_ID="$SNAP"
export SERVED_MODEL_NAME=gemma-4-e4b-it
export HOST=127.0.0.1
export PORT
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_SEED=0
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[serve] official-g32@0.22.0 model=$SNAP mml=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS BI=$VLLM_BATCH_INVARIANT $(date -u +%FT%TZ)" | tee "$LOG"
"$VENV/bin/python" -c "import vllm; print('[serve] vllm', vllm.__version__)" | tee -a "$LOG"

setsid "$VENV/bin/python" "$SERVE_PY" >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] pid=$SRV_PID log=$LOG"

# Cold load (~11 GB int4+bf16 multimodal) + Marlin repack + compile: allow ~25 min.
for i in $(seq 1 300); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve] FAILED: server exited early; tail:"; tail -60 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve] READY after ${i}x5s ($(date -u +%FT%TZ))"; exit 0
  fi
  sleep 5
done
echo "[serve] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
