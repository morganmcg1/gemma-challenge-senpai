#!/usr/bin/env bash
# PR #523 chained GPU campaign (serialized, runs after Batch A inspected):
#   B  geometry sweep  : bx_T16_S16, bx_T8_S32, bx_T2_S128  (n=2, PPL)
#   M  microbench 0/8  : adaptive + 4 fixed (T16/8/4/2)      (no serve)
#   C  lever arms      : bx_fisampler, bx_eager_drafter      (n=2, no PPL)
# Each step is independent; `|| true` keeps the chain alive past a crash
# (e.g. FlashInfer sampler JIT on this pod). All LOCAL, no HF job.
set -u
cd /workspace/senpai/target
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=0
G=byteexact-realization-gap

echo "=== [chain] B geometry sweep $(date -u +%H:%M:%S) ==="
$PY -m research.speed.byteexact_realization_gap.run_realization_gap \
    --arms bx_T16_S16,bx_T8_S32,bx_T2_S128 --n-decodes 2 --tag geometry \
    --wandb-name lawine/realization-gap-geometry --wandb-group "$G" \
    > logs/pr523/batchB_geometry.log 2>&1 || true

echo "=== [chain] M microbench 0/8 sweep $(date -u +%H:%M:%S) ==="
$PY -m research.speed.byteexact_realization_gap.run_microbench_sweep \
    > logs/pr523/microbench_sweep.log 2>&1 || true

echo "=== [chain] C lever arms (sampler + eager-drafter) $(date -u +%H:%M:%S) ==="
$PY -m research.speed.byteexact_realization_gap.run_realization_gap \
    --arms bx_fisampler,bx_eager_drafter --n-decodes 2 --no-ppl --tag levers \
    --wandb-name lawine/realization-gap-levers --wandb-group "$G" \
    > logs/pr523/batchC_levers.log 2>&1 || true

echo "=== [chain] DONE $(date -u +%H:%M:%S) ==="
