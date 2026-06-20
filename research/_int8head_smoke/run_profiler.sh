#!/usr/bin/env bash
# Reproduce the PR #781 decode op-profiler (eager mode, M=1, spec OFF) on ONE
# checkpoint to isolate the per-token lm_head GEMV time. profile_decode.py calls
# paths.prepare_local_gpu_env() itself (sets CUDA_VISIBLE_DEVICES=0 +
# VLLM_USE_FLASHINFER_SAMPLER=0), so no extra env is needed here. Runs the bare
# in-process LLM (no server, no speculation) -> faithful per-kernel CUDA-time
# composition. The bf16-head base reproduces #781's 2.776 ms/token lm_head GEMV;
# the int8-head build measures the AllSpark W8A16 GEMV that replaces it.
# Usage: run_profiler.sh <model_id_or_path> <out_subdir>
set -uo pipefail
MODEL=$1; OUT=$2
ROOT=/workspace/senpai/target
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
OUTDIR="$ROOT/research/_int8head_smoke/$OUT"
mkdir -p "$OUTDIR"
echo "[prof] model=$MODEL out=$OUTDIR $(date -u +%H:%M:%S)"
cd "$ROOT"
"$VENV" -m scripts.local_validation.profile_decode \
  --model-id "$MODEL" \
  --mode eager \
  --profile-mode op \
  --out-dir "$OUTDIR"
echo "[prof] done rc=$? $(date -u +%H:%M:%S)"
