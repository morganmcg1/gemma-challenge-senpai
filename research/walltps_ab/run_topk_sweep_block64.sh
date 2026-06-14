#!/usr/bin/env bash
# PR #138 — CENTROID_TOP_K {64,128} probe at FUSED_SPARSE_ARGMAX_BLOCK=64, K=7.
#
# Step 3 of #138 asks: with the best block size (block64) and best K (K=7), does
# widening the sparse-argmax candidate set CENTROID_TOP_K 64->128 buy any wall_tps?
# More active vocab candidates can only raise E[accept] if the drafter's true next
# token sits outside its top-64 centroid candidates; otherwise it just adds argmax
# work. The manifest default is CENTROID_TOP_K=64, so topk64 is the incumbent and
# topk128 is the candidate. Greedy identity is verifier-enforced and unaffected by
# the candidate-set width (it only changes which/how-many drafter proposals are
# scored, never the verifier's accept rule), so this is a pure wall_tps probe.
#
# A prior interactive probe got topk64 {454.04, 454.09, 454.04} vs topk128
# {449.97, 450.21, <interrupted>}; this script re-runs the full paired A/B to N=3
# for a clean paired_ab.json + W&B curve.
#
# Usage: bash research/walltps_ab/run_topk_sweep_block64.sh
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root (target/)

PY=.venv/bin/python
RUNNER=scripts/profiler/paired_tps_ab.py
GROUP=k-sweep-block64-reopt
N=3
SUB=fa2sw_precache_kenyan
MODEL=/tmp/qat-assistant
SPEC='{"method":"mtp","model":"'"$MODEL"'","num_speculative_tokens":7}'
LOG=research/validity/block64_ksweep/topk_sweep_block64.log

mkdir -p research/validity/block64_ksweep

echo "[topk64] START $(date -u +%FT%TZ)" | tee -a "$LOG"
$PY $RUNNER \
    --baseline "$SUB" --candidate "$SUB" \
    --baseline-env "FUSED_SPARSE_ARGMAX_BLOCK=64" \
    --baseline-env "CENTROID_TOP_K=64" \
    --baseline-env "SPECULATIVE_CONFIG=$SPEC" \
    --baseline-label topk64 \
    --candidate-env "FUSED_SPARSE_ARGMAX_BLOCK=64" \
    --candidate-env "CENTROID_TOP_K=128" \
    --candidate-env "SPECULATIVE_CONFIG=$SPEC" \
    --candidate-label topk128 \
    --n $N --tag topk_block64 \
    --wandb-name kanna/k-sweep-topk128-block64 --wandb-group "$GROUP" \
    >> "$LOG" 2>&1
echo "[topk64] exit=$? $(date -u +%FT%TZ)" | tee -a "$LOG"
echo "[topk64] DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
