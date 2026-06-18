#!/usr/bin/env bash
# PR #650 Arm A CI (and Arm C if TAG=optionb) -- AIME 10-seed *sampled* at the clean
# budget against the already-running server. Mirrors the GPQA-sampled protocol so the
# AIME number gets a real CI: T=1, top_p=0.95, top_k=64, k=1 per seed, sampling-seeds
# 0..9 -> n = 10 x 60 = 600 samples; per-seed accuracy gives the SE/range.
# --no-thinking to measure the SAME no-think generation mode as the greedy anchor
# (sampled vs greedy of the same quantity), not gemma's long <think> channel.
# Each seed writes its own JSON so a mid-run stop keeps prior seeds (resumable: an
# existing non-empty seed file is skipped).
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/int4ar_denom_harden
CLIENT=/tmp/eval-serve-venv/bin/python
TAG="${TAG:?set TAG=int4ar|bf16|optionb}"
MT="${MT:?set MT (clean budget)}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
CONC="${CONC:-16}"
RES="$HERE/results/sampled_${TAG}"; mkdir -p "$RES"
STATUS="$HERE/_aime_sampled_${TAG}.status"
echo "AIME-SAMPLED-$TAG-mt$MT START seeds=[$SEEDS] $(date -u +%FT%TZ)" | tee -a "$STATUS"
for s in $SEEDS; do
  OUT="$RES/${TAG}_aime_sampled_mt${MT}_s${s}.json"
  if [[ -s "$OUT" ]]; then echo "  s=$s exists, skip $(date -u +%H:%M:%S)" | tee -a "$STATUS"; continue; fi
  $CLIENT research/downstream_quality_aime/aime_eval.py \
    --base-url http://127.0.0.1:8000 --model gemma-4-e4b-it \
    --years 2024,2025-I,2025-II --k 1 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --max-tokens "$MT" --min-tokens 8 \
    --no-thinking --seed "$s" --save-text --client-concurrency "$CONC" \
    --label "${TAG}_aime_sampled_mt${MT}_s${s}" --out "$OUT" \
    > "$HERE/_aime_sampled_${TAG}_s${s}.out" 2>&1
  echo "  s=$s rc=$? $(date -u +%H:%M:%S): $(grep -oE 'maj@1=[0-9.]+' "$HERE/_aime_sampled_${TAG}_s${s}.out" | head -1)" | tee -a "$STATUS"
done
echo "AIME-SAMPLED-$TAG-mt$MT DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
