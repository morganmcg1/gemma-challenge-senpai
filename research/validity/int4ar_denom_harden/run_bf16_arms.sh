#!/usr/bin/env bash
# PR #650 -- bf16-base side of the denominator-hardening panel, run sequentially
# against ONE already-running bf16 server (serve_bf16_0p22.sh @ mml=16384, BI=1,
# VLLM_USE_FLASHINFER_SAMPLER=0). Mirror of the int4-AR side that was already
# collected on the int4 server. Every leg is resumable (skips a non-empty output),
# so a mid-run stop / re-invoke keeps prior legs.
#
# Order chosen for value-if-interrupted:
#   1. AIME greedy budget grid 12288 -> 8192 -> 6144   (Arm A: completes the decisive
#      fl->0 gap + bf16 truncation curve; 12288 first so the verdict point lands early)
#   2. GPQA-D sampled 10-seed                           (Arm B: the long pole ~100 min)
#   3. AIME sampled @12288 5-seed                        (Arm A real-CI, matched to the
#                                                         int4-AR 5-seed sampled run)
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/int4ar_denom_harden
STATUS="$HERE/_bf16_arms.status"
echo "BF16-ARMS START $(date -u +%FT%TZ)" | tee "$STATUS"

# pre-flight: server must answer /v1/models with the served name
if ! curl -s --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q gemma-4-e4b-it; then
  echo "FATAL: bf16 server not READY on :8000" | tee -a "$STATUS"; exit 2
fi

# ---- Arm A: bf16 greedy AIME budget grid ----
for MT in 12288 8192 6144; do
  OUT="$HERE/results/bf16_aime_greedy_mt${MT}.json"
  if [[ -s "$OUT" ]]; then echo "[A] bf16 greedy mt$MT exists, skip $(date -u +%T)" | tee -a "$STATUS"; continue; fi
  echo "[A] bf16 greedy mt$MT START $(date -u +%T)" | tee -a "$STATUS"
  TAG=bf16 MT=$MT CONC=16 bash "$HERE/aime_budget.sh" >>"$STATUS" 2>&1
  echo "[A] bf16 greedy mt$MT rc=$? $(date -u +%T)" | tee -a "$STATUS"
done

# ---- Arm B: bf16 GPQA-D sampled 10-seed ----
echo "[B] bf16 GPQA 10-seed START $(date -u +%T)" | tee -a "$STATUS"
SEEDS="0 1 2 3 4 5 6 7 8 9" MT=6144 bash "$HERE/gpqa_bf16_sampled.sh" >>"$STATUS" 2>&1
echo "[B] bf16 GPQA 10-seed rc=$? $(date -u +%T)" | tee -a "$STATUS"

# ---- Arm A real-CI: bf16 sampled AIME @12288, matched 5 seeds to int4-AR ----
echo "[A-CI] bf16 sampled AIME mt12288 START $(date -u +%T)" | tee -a "$STATUS"
TAG=bf16 MT=12288 SEEDS="0 1 2 3 4" CONC=16 bash "$HERE/aime_sampled.sh" >>"$STATUS" 2>&1
echo "[A-CI] bf16 sampled AIME mt12288 rc=$? $(date -u +%T)" | tee -a "$STATUS"

echo "BF16-ARMS DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
