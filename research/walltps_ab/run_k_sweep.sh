#!/usr/bin/env bash
# PR #90 — empirical MTP draft-length K sweep on the robust wall_tps runner (#82).
#
# Sweeps K in {5,6,7,8,9} for submissions/fa2sw_precache_kenyan by overriding ONLY
# num_speculative_tokens in the serve-time SPECULATIVE_CONFIG env (no served-file
# change). K=7 is the deployed manifest default and serves as the paired baseline;
# it is run fresh ONCE (in the K=6 arm) and reused for the other arms
# (restart-invariant per #72). Each python invocation is its own process, well
# under SENPAI_TIMEOUT_MINUTES.
#
# Usage: bash research/walltps_ab/run_k_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root (target/)

PY=.venv/bin/python
RUNNER=scripts/profiler/paired_tps_ab.py
GROUP=mtp-k-sweep-wall-tps
N=3
SUB=fa2sw_precache_kenyan
MODEL=/tmp/qat-assistant
LOG=research/walltps_ab/k_sweep.log

spec() { printf '{"method":"mtp","model":"%s","num_speculative_tokens":%s}' "$MODEL" "$1"; }

echo "[k-sweep] START $(date -u +%FT%TZ)" | tee -a "$LOG"

# --- Arm K=6: fresh K=7 baseline + K=6 candidate (saves baseline records) ---
echo "[k-sweep] === K=6 (fresh baseline K=7) $(date -u +%FT%TZ) ===" | tee -a "$LOG"
$PY $RUNNER \
    --baseline "$SUB" --candidate "$SUB" \
    --candidate-env "SPECULATIVE_CONFIG=$(spec 6)" \
    --candidate-label mtp_k6 \
    --n $N --tag mtp_k6 \
    --wandb-name lawine/k-sweep-k6 --wandb-group "$GROUP" \
    >> "$LOG" 2>&1
echo "[k-sweep] K=6 exit=$? $(date -u +%FT%TZ)" | tee -a "$LOG"

BASE_JSON=research/walltps_ab/mtp_k6/paired_ab.json
if [ ! -f "$BASE_JSON" ]; then
    echo "[k-sweep] FATAL: baseline json missing ($BASE_JSON); aborting reuse arms" | tee -a "$LOG"
    exit 1
fi

# --- Arms K=5,8,9: reuse the K=7 baseline records ---
for K in 5 8 9; do
    echo "[k-sweep] === K=$K (reuse baseline) $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    $PY $RUNNER \
        --baseline "$SUB" --candidate "$SUB" \
        --reuse-baseline-from "$BASE_JSON" \
        --candidate-env "SPECULATIVE_CONFIG=$(spec $K)" \
        --candidate-label "mtp_k$K" \
        --n $N --tag "mtp_k$K" \
        --wandb-name "lawine/k-sweep-k$K" --wandb-group "$GROUP" \
        >> "$LOG" 2>&1
    echo "[k-sweep] K=$K exit=$? $(date -u +%FT%TZ)" | tee -a "$LOG"
done

echo "[k-sweep] DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
