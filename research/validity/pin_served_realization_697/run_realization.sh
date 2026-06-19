#!/usr/bin/env bash
# PR #697 — served-stack speed realization of the attention pin.
# LOCAL A10G, analysis_only. Fresh process per run (pin is process-global).
# Run A (core pin A/B): unpinned spec (BI=0,K=5) vs bi1-pinned spec (BI=1,K=5).
# Run B (anchor):       AR M=1 (K=0) vs bi1-pinned spec (BI=1,K=5) cross-check.
set -uo pipefail
cd /workspace/senpai/target
export CUDA_VISIBLE_DEVICES=0
PY=./.venv/bin/python
AB=scripts/profiler/paired_tps_ab.py
GROUP=pin-served-tps-realization-land

echo "=== [697] Run A: core pin A/B (unpinned vs bi1), N=3, 128x512 ==="
$PY $AB \
  --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --baseline-env VLLM_BATCH_INVARIANT=0 --baseline-env NUM_SPECULATIVE_TOKENS=5 \
  --candidate-env VLLM_BATCH_INVARIANT=1 --candidate-env NUM_SPECULATIVE_TOKENS=5 \
  --baseline-label unpinned_k5 --candidate-label bi1_pinned_k5 \
  --n 3 --num-prompts 128 --output-len 512 --seed 1 \
  --tag pin_served_realization_697/core --no-project \
  --wandb-name land/pin-realization-unpinned-vs-bi1 --wandb-group $GROUP
rcA=$?
echo "=== [697] Run A exit=$rcA ==="

echo "=== [697] Run B: AR anchor (K=0) vs bi1 cross-check, N=2, 128x512 ==="
$PY $AB \
  --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --baseline-env NUM_SPECULATIVE_TOKENS=0 \
  --candidate-env VLLM_BATCH_INVARIANT=1 --candidate-env NUM_SPECULATIVE_TOKENS=5 \
  --baseline-label ar_m1 --candidate-label bi1_pinned_k5_xcheck \
  --n 2 --num-prompts 128 --output-len 512 --seed 1 \
  --tag pin_served_realization_697/anchor --no-project \
  --wandb-name land/pin-realization-ar-anchor --wandb-group $GROUP
rcB=$?
echo "=== [697] Run B exit=$rcB ==="
echo "=== [697] ALL DONE rcA=$rcA rcB=$rcB ==="
