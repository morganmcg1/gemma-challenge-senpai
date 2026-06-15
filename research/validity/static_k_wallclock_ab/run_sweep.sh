#!/usr/bin/env bash
# PR #273 — static-K wall-clock A/B over the REAL served stack.
#
# Question: does static draft-depth K=4/5 actually beat the deployed K=7 in
# MEASURED local wall-clock TPS, or is the +4.28%/+4.00% from the #256/#266
# *composition* bookkeeping (which prices a draft-pass saving against E[T]/step
# but ignores the large FIXED serving overhead that does not shrink with fewer
# draft passes)?
#
# This sweeps K in {3,4,5,6,7} (7 = deployed manifest default) by overriding ONLY
# num_speculative_tokens in the serve-time SPECULATIVE_CONFIG env. NOTHING else
# changes: same model, same KV cache, same greedy sampler, same prompts, same
# seed. The verify step is greedy-exact, so emitted token-ids are identical across
# all K by construction (greedy-safe). LOCAL only: no HF Job, no submission, no
# served-file edit, NOT a launch.
#
# Reuses scripts/profiler/paired_tps_ab.py (the PR #82 paired wall_tps runner) and
# the #90 reuse-baseline pattern: K=7 is run fresh ONCE (in the K=4 arm) and reused
# for K=3,5,6 (restart-invariant per PR #72). Each python invocation is its own
# process, well under SENPAI_TIMEOUT_MINUTES.
#
# Usage:
#   bash research/validity/static_k_wallclock_ab/run_sweep.sh           # seed=1 N=3 full
#   SEED=2 KS="4" N=3 bash research/validity/static_k_wallclock_ab/run_sweep.sh   # seed=2 K4 only
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root (target/)

PY=.venv/bin/python
RUNNER=scripts/profiler/paired_tps_ab.py
GROUP=static-k-wallclock-ab
OUTROOT=research/validity/static_k_wallclock_ab
N=${N:-3}
SEED=${SEED:-1}
KS=${KS:-"4 3 5 6"}          # candidate K order; K=4 first so the fresh K=7 baseline is saved there
SUB=fa2sw_precache_kenyan
MODEL=/tmp/qat-assistant
LOG=$OUTROOT/sweep_seed${SEED}.log

spec() { printf '{"method":"mtp","model":"%s","num_speculative_tokens":%s}' "$MODEL" "$1"; }
arm_dir() { echo "$OUTROOT/seed${SEED}_mtp_k$1"; }

echo "[sweep] START seed=$SEED N=$N KS='$KS' $(date -u +%FT%TZ)" | tee -a "$LOG"

BASE_JSON=""
for K in $KS; do
    AD=$(arm_dir "$K")
    if [ -z "$BASE_JSON" ]; then
        # First arm: run the fresh K=7 baseline + this K candidate (saves baseline records).
        echo "[sweep] === K=$K (fresh K=7 baseline) $(date -u +%FT%TZ) ===" | tee -a "$LOG"
        $PY $RUNNER \
            --baseline "$SUB" --candidate "$SUB" \
            --candidate-env "SPECULATIVE_CONFIG=$(spec "$K")" \
            --candidate-label "mtp_k$K" --baseline-label mtp_k7 \
            --n "$N" --seed "$SEED" --tag "seed${SEED}_mtp_k$K" --out-dir "$AD" \
            --wandb-name "stark/static-k-ab-k${K}-seed${SEED}" --wandb-group "$GROUP" \
            >> "$LOG" 2>&1
        rc=$?
        echo "[sweep] K=$K exit=$rc $(date -u +%FT%TZ)" | tee -a "$LOG"
        BASE_JSON="$AD/paired_ab.json"
        if [ ! -f "$BASE_JSON" ]; then
            echo "[sweep] FATAL: baseline json missing ($BASE_JSON); aborting reuse arms" | tee -a "$LOG"
            exit 1
        fi
    else
        echo "[sweep] === K=$K (reuse K=7 baseline) $(date -u +%FT%TZ) ===" | tee -a "$LOG"
        $PY $RUNNER \
            --baseline "$SUB" --candidate "$SUB" \
            --reuse-baseline-from "$BASE_JSON" \
            --candidate-env "SPECULATIVE_CONFIG=$(spec "$K")" \
            --candidate-label "mtp_k$K" --baseline-label mtp_k7 \
            --n "$N" --seed "$SEED" --tag "seed${SEED}_mtp_k$K" --out-dir "$AD" \
            --wandb-name "stark/static-k-ab-k${K}-seed${SEED}" --wandb-group "$GROUP" \
            >> "$LOG" 2>&1
        echo "[sweep] K=$K exit=$? $(date -u +%FT%TZ)" | tee -a "$LOG"
    fi
done

echo "[sweep] DONE seed=$SEED $(date -u +%FT%TZ)" | tee -a "$LOG"
