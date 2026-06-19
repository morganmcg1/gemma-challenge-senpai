#!/usr/bin/env bash
# PR #719 -- chained base-bf16 greedy comparability measurement against the live
# bf16 base server. (1) canonical anchor 2024+2025 via the SAME harness (aime_eval),
# (2) past-AIME year-bands via pastaime_eval, + greedy determinism check. gb6144
# greedy config (T=0, max_tokens 6144, min_tokens 8, no-thinking), conc 16 (BI=1
# batch-invariant -> identical to sequential). Analysis-only, no HF job.
set -uo pipefail
cd /workspace/senpai/target
source .venv/bin/activate
RD=research/validity/ci_world_pool_feasibility_719/results
mkdir -p "$RD"
BASE=http://127.0.0.1:8000
CONC=16
MAXTOK=6144

echo "[run] $(date -u +%FT%TZ) ANCHOR 2024+2025 (60) ..."
python research/downstream_quality_aime/aime_eval.py --base-url "$BASE" \
  --years 2024,2025 --k 1 --temperature 0 --top-p 1.0 --top-k -1 \
  --max-tokens "$MAXTOK" --min-tokens 8 --no-thinking \
  --client-concurrency "$CONC" --request-timeout-s 1200 \
  --label anchor_2024_2025 --out "$RD/anchor_2024_2025.json" \
  && echo "[run] ANCHOR done $(date -u +%FT%TZ)" || echo "[run] ANCHOR FAILED"

echo "[run] $(date -u +%FT%TZ) BANDS past-AIME ..."
python research/validity/ci_world_pool_feasibility_719/pastaime_eval.py \
  --base-url "$BASE" --bands 1983-1994,2005-2014,2015-2023 --per-band 40 \
  --client-concurrency "$CONC" --max-tokens "$MAXTOK" --min-tokens 8 \
  --determinism-check 4 --request-timeout-s 1200 \
  --out "$RD/band_results.json" \
  && echo "[run] BANDS done $(date -u +%FT%TZ)" || echo "[run] BANDS FAILED"

echo "[run] ALL DONE $(date -u +%FT%TZ)"
touch "$RD/.MEASURE_DONE"
