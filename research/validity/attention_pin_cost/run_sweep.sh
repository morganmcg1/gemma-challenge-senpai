#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# PR #691 (land) -- long-ctx 3D-vs-2D attention-pin crossover sweep driver.
# Runs each config in a FRESH process (MIN_LAUNCH_GRID_SIZE_2D patch + batch-invariant
# snapshot are process-global, fixed in the metadata builder __init__), then decides+logs.
# LOCAL A10G only. analysis_only -- NO HF Job.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${VENV_PY:-/tmp/senpai-venvs/20f658587e8a6643/bin/python}"
SYS_PY="${SYS_PY:-/usr/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0}"   # inherited =7 is stale
export WANDB_GROUP="${WANDB_GROUP:-strict319-attention-pin-cost-land}"
export WANDB_NAME="${WANDB_NAME:-land/pin-ctx-crossover}"

CTX="${CTX:-512 1024 2048 4096 8192 16384}"
N_NEW="${N_NEW:-32}"
REPS="${REPS:-4}"
CONFIGS="${CONFIGS:-baseline fixed2d bi1}"

mkdir -p "$HERE/runs"
echo "[run_sweep] ctx=[$CTX] n_new=$N_NEW reps=$REPS configs=[$CONFIGS] gpu=$CUDA_VISIBLE_DEVICES"

for cfg in $CONFIGS; do
  echo "================= CONFIG $cfg $(date -u +%T) ================="
  "$VENV_PY" "$HERE/ctx_crossover_sweep.py" \
    --config "$cfg" --ctx $CTX --n-new "$N_NEW" --reps "$REPS" \
    2>&1 | tee "$HERE/runs/${cfg}.log"
done

echo "================= DECIDE + W&B $(date -u +%T) ================="
"$SYS_PY" "$HERE/decide_and_log.py"
echo "[run_sweep] done $(date -u +%T)"
