#!/usr/bin/env bash
# PR #719 -- append the 1995-2004 past-AIME band for a complete 4-era comparability
# table (the original chained run sampled 1983-94 / 2005-14 / 2015-23). Waits for the
# first run's .MEASURE_DONE marker so both share the SAME live bf16 base server, then
# measures 1995-2004 with the IDENTICAL greedy config. Analysis-only, no HF job.
set -uo pipefail
cd /workspace/senpai/target
source .venv/bin/activate
RD=research/validity/ci_world_pool_feasibility_719/results
BASE=http://127.0.0.1:8000

echo "[band9504] $(date -u +%FT%TZ) waiting for first run .MEASURE_DONE ..."
for _ in $(seq 1 240); do        # bounded: <=240*15s = 60 min cap
  [ -f "$RD/.MEASURE_DONE" ] && break
  sleep 15
done
[ -f "$RD/.MEASURE_DONE" ] || { echo "[band9504] first run never finished; aborting"; exit 1; }

echo "[band9504] $(date -u +%FT%TZ) running 1995-2004 ..."
python research/validity/ci_world_pool_feasibility_719/pastaime_eval.py \
  --base-url "$BASE" --bands 1995-2004 --per-band 40 \
  --client-concurrency 16 --max-tokens 6144 --min-tokens 8 \
  --request-timeout-s 1200 \
  --out "$RD/band_results_1995_2004.json" \
  && echo "[band9504] done $(date -u +%FT%TZ)" || echo "[band9504] FAILED"
touch "$RD/.BAND9504_DONE"
