#!/usr/bin/env bash
# PR#670 finish orchestrator (resumes after the prior pod-session died mid-off-diagonal).
# headline (stock@32) + headline_v2 (ftv1@64) are already DONE on disk; this script runs
# only the two MISSING pieces, each a separate paired_tps_ab.py call < 90 min (per-run cap):
#   1. off-diagonal 2x2 cells: stock@topk64 (baseline) + ftv1@topk32 (candidate), n=2
#      -> decomposes the +0.31 espec edge into retrain vs the top_k 32->64 knob (decisive)
#   2. subsample-seed sweep: n=64 x seeds{1,2,3} on the headline pair (stock@32 vs ftv1@64)
#      -> a genuine prompt-resampling CI on the delta (single-seed-artifact robustness)
# Population workload for the grid = 128x512 seed1. analysis_only, official_tps=0, fires=false.
set -u
cd /workspace/senpai/target
PY=.venv/bin/python
AB=scripts/profiler/paired_tps_ab.py
D=research/walltps_ab/optionb_bi1_stock_int4/derisk_670
ENV4() { local role=$1 d=$2; echo "--${role}-env VLLM_BATCH_INVARIANT=1 --${role}-env NUM_SPECULATIVE_TOKENS=6 --${role}-env MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct --${role}-env DRAFTER_MODEL=${d}"; }

echo "[finish] START $(date -u +%FT%TZ)"

# ---- 1. off-diagonal 2x2 (decisive): stock@topk64 vs ftv1@topk32, n=2 ----
echo "[finish] === off-diagonal stock_topk64 vs ftv1_topk32 (n=2) ==="
$PY $AB \
  --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --num-prompts 128 --output-len 512 --seed 1 \
  --baseline-label stock_topk64 --candidate-label ftv1_topk32 \
  $(ENV4 baseline /tmp/stock-topk64) $(ENV4 candidate /tmp/ftv1-topk32) \
  --n 2 --out-dir "$D/grid_offdiag" \
  --wandb-group local-drafter-derisk-land --wandb-name land/derisk-2x2-offdiag \
  > "$D/grid_offdiag.console.log" 2>&1
echo "[finish] off-diagonal rc=$? $(date -u +%FT%TZ)" | tee "$D/grid_offdiag.done"

# ---- 2. subsample-seed sweep on the headline pair (robustness) ----
for S in 1 2 3; do
  echo "[finish] === subsample n=64 seed=$S (stock@32 vs ftv1@64) ==="
  $PY $AB \
    --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
    --baseline-label stock_topk32 --candidate-label ftv1_topk64 \
    $(ENV4 baseline /tmp/stock-topk32) $(ENV4 candidate /tmp/qat-assistant) \
    --n 1 --num-prompts 64 --output-len 512 --seed $S \
    --out-dir "$D/sub64_seed$S" \
    --wandb-group local-drafter-derisk-land --wandb-name "land/derisk-sub64-s$S" \
    > "$D/sub64_seed$S.console.log" 2>&1
  echo "[finish] sub64 seed=$S rc=$? $(date -u +%FT%TZ)" | tee "$D/sub64_seed$S.done"
done

echo "[finish] ALL DONE $(date -u +%FT%TZ)" | tee "$D/run_finish.alldone"
