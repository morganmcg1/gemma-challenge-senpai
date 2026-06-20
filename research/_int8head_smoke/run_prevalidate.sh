#!/usr/bin/env bash
# Local pre-validation runner for the lm_head-bytes experiment (PR #788).
# Runs scripts/local_prevalidate.py for ONE submission under a fixed local-GPU
# env so the int8-head candidate and the bf16-head bi0 control are measured
# apples-to-apples (same GPU, same sampler, same prompts/seed). Local A10G TPS is
# exploratory (NOT the official a10g-small number); PPL + greedy token_ids carry
# over. Usage: run_prevalidate.sh <submission_dir> <port> <out_subdir>
set -uo pipefail
SUB=$1; PORT=$2; OUT=$3
ROOT=/workspace/senpai/target
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
OUTDIR="$ROOT/research/_int8head_smoke/$OUT"
mkdir -p "$OUTDIR"

# Single in-container GPU.
export CUDA_VISIBLE_DEVICES=0
# PyTorch-native sampler: the canonical container fix for the missing cuRAND
# headers (see scripts/local_validation/paths.default_native_sampler). Sampler
# backend does NOT touch logits -> greedy(argmax)/PPL identical; only exploratory
# TPS is marginally affected, and BOTH arms use it, so the delta is fair. Kept in
# the shell env (NOT the manifest) so the shipped submission stays a 1-delta diff.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "[run] submission=$SUB port=$PORT out=$OUTDIR $(date -u +%H:%M:%S)"
"$VENV" "$ROOT/scripts/local_prevalidate.py" \
  --submission "$ROOT/submissions/$SUB" \
  --venv-python "$VENV" \
  --port "$PORT" \
  --decode-num-prompts 128 \
  --ppl-records 0 \
  --output-dir "$OUTDIR" \
  --server-log "$OUTDIR/serve.log"
echo "[run] done rc=$? $(date -u +%H:%M:%S)"
