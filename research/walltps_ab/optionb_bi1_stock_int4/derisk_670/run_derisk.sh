#!/usr/bin/env bash
# PR#670 local-drafter de-risk orchestrator.
# Completes the interrupted headline candidate (ft-v1@topk64) by reusing the
# already-done stock@topk32 baseline, then runs the 2x2 off-diagonal cells
# (stock@topk64, ftv1@topk32) to decompose the espec edge into retrain vs top_k.
# Each paired call is < 90 min (the per-run cap). Population workload = 128x512 seed1.
set -u
cd /workspace/senpai/target
PY=.venv/bin/python
AB=scripts/profiler/paired_tps_ab.py
D=research/walltps_ab/optionb_bi1_stock_int4/derisk_670
COMMON_BASE="--baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --num-prompts 128 --output-len 512 --seed 1 \
  --wandb-group local-drafter-derisk-land"
ENV4() { # drafter_path -> repeated --<role>-env flags
  local role=$1 drafter=$2
  echo "--${role}-env VLLM_BATCH_INVARIANT=1 --${role}-env NUM_SPECULATIVE_TOKENS=6 \
    --${role}-env MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct --${role}-env DRAFTER_MODEL=${drafter}"
}

echo "[derisk] START $(date -u +%FT%TZ)"

# ---- 1. headline candidate ft-v1@topk64 (reuse done stock@topk32 baseline), n=3 ----
echo "[derisk] === headline candidate ftv1_topk64 (n=3, reuse stock baseline) ==="
$PY $AB $COMMON_BASE \
  --baseline-label stock_topk32 --candidate-label ftv1_topk64 \
  $(ENV4 baseline /tmp/stock-topk32) $(ENV4 candidate /tmp/qat-assistant) \
  --reuse-baseline-from "$D/headline/reuse_baseline_stock32.json" \
  --n 3 --out-dir "$D/headline_v2" \
  --wandb-name land/derisk-headline-ftv1 \
  > "$D/headline_v2.console.log" 2>&1
echo "[derisk] headline candidate rc=$? $(date -u +%FT%TZ)" | tee "$D/headline_v2.done"

# ---- 2. 2x2 off-diagonal: stock@topk64 baseline + ftv1@topk32 candidate, n=2 ----
echo "[derisk] === 2x2 off-diagonal stock_topk64 vs ftv1_topk32 (n=2 each) ==="
$PY $AB $COMMON_BASE \
  --baseline-label stock_topk64 --candidate-label ftv1_topk32 \
  $(ENV4 baseline /tmp/stock-topk64) $(ENV4 candidate /tmp/ftv1-topk32) \
  --n 2 --out-dir "$D/grid_offdiag" \
  --wandb-name land/derisk-2x2-offdiag \
  > "$D/grid_offdiag.console.log" 2>&1
echo "[derisk] off-diagonal rc=$? $(date -u +%FT%TZ)" | tee "$D/grid_offdiag.done"

echo "[derisk] ALL DONE $(date -u +%FT%TZ)" | tee "$D/run_derisk.alldone"
