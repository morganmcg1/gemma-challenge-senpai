#!/usr/bin/env bash
# PR #138 — MTP draft-length K re-characterization at FUSED_SPARSE_ARGMAX_BLOCK=64.
#
# lawine #90 established K=7 optimal at block16 (K7 ref wall_tps 454.338). stark #137
# moves the sparse-argmax tile from 16->64. A cheaper per-step argmax lowers step_time,
# which can shift the K* tradeoff toward longer draft chains. This sweep re-measures
# wall_tps vs K in {6,7,8,9} at block64 to find the new K*.
#
# Design (one paired_tps_ab.py process per arm, each well under SENPAI_TIMEOUT_MINUTES):
#   * Arm A: FRESH K7-block16 anchor baseline (override {} = unmodified manifest, must
#     reproduce lawine 454.338 +/-2%) + K7-block64 candidate. The paired delta is the
#     pure block16->block64 step-time effect at K=7, and the saved baseline records are
#     reused by the other arms (restart-invariant per #72/#82).
#   * Arms K=6,8,9: reuse the K7-block16 anchor baseline; candidate = K{6,8,9}-block64.
#
# Block size only re-tiles the drafter's sparse-argmax kernel (constexpr BLOCK_SELECTED)
# with leftmost tie-break preserved at both block and reduce stages => bit-identical
# drafter proposals => E[accept] invariant at fixed K, greedy identity verifier-enforced.
# So only step_time moves. FUSED_SPARSE_ARGMAX_REQUIRE=1 means a block64 compile failure
# hard-errors rather than silently reverting to block16.
#
# Usage: bash research/walltps_ab/run_k_sweep_block64.sh
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root (target/)

PY=.venv/bin/python
RUNNER=scripts/profiler/paired_tps_ab.py
GROUP=k-sweep-block64-reopt
N=3
SUB=fa2sw_precache_kenyan
MODEL=/tmp/qat-assistant
BLOCK=64
LOG=research/validity/block64_ksweep/k_sweep_block64.log

mkdir -p research/validity/block64_ksweep
spec() { printf '{"method":"mtp","model":"%s","num_speculative_tokens":%s}' "$MODEL" "$1"; }

echo "[k64] START $(date -u +%FT%TZ)" | tee -a "$LOG"

# --- Arm A: fresh K7-block16 anchor baseline + K7-block64 candidate ---
echo "[k64] === Arm A: K7-block16 anchor + K7-block64 $(date -u +%FT%TZ) ===" | tee -a "$LOG"
$PY $RUNNER \
    --baseline "$SUB" --candidate "$SUB" \
    --candidate-env "FUSED_SPARSE_ARGMAX_BLOCK=$BLOCK" \
    --candidate-env "SPECULATIVE_CONFIG=$(spec 7)" \
    --candidate-label k7_block64 \
    --n $N --tag k7_block64 \
    --wandb-name kanna/k-sweep-k7-block64 --wandb-group "$GROUP" \
    >> "$LOG" 2>&1
echo "[k64] Arm A exit=$? $(date -u +%FT%TZ)" | tee -a "$LOG"

BASE_JSON=research/walltps_ab/k7_block64/paired_ab.json
if [ ! -f "$BASE_JSON" ]; then
    echo "[k64] FATAL: anchor baseline json missing ($BASE_JSON); aborting reuse arms" | tee -a "$LOG"
    exit 1
fi

# --- Arms K=6,8,9 at block64: reuse the fresh K7-block16 anchor baseline ---
for K in 6 8 9; do
    echo "[k64] === K=$K block64 (reuse anchor) $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    $PY $RUNNER \
        --baseline "$SUB" --candidate "$SUB" \
        --reuse-baseline-from "$BASE_JSON" \
        --candidate-env "FUSED_SPARSE_ARGMAX_BLOCK=$BLOCK" \
        --candidate-env "SPECULATIVE_CONFIG=$(spec $K)" \
        --candidate-label "k${K}_block64" \
        --n $N --tag "k${K}_block64" \
        --wandb-name "kanna/k-sweep-k${K}-block64" --wandb-group "$GROUP" \
        >> "$LOG" 2>&1
    echo "[k64] K=$K exit=$? $(date -u +%FT%TZ)" | tee -a "$LOG"
done

echo "[k64] DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
