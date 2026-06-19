#!/usr/bin/env bash
# PR #719 -- chained SECONDARY-source base-greedy measurement (AMC + MATH level-5).
# Fulfils instruction #2 ("measure base greedy on candidate expansion sources") for
# the two named secondary candidates, on the SAME live bf16 base server / IDENTICAL
# greedy config as the past-AIME bands. Waits for the 1995-2004 band's .BAND9504_DONE
# so all measurements share one server. Analysis-only, no HF job.
#   * AMC  : AI-MO/aimo-validation-amc, 77 integer-gradeable -> tests DIFFICULTY
#            comparability (AMC is the AIME-qualifier exam -> expect "too easy").
#   * MATH5: nlile/hendrycks-MATH-benchmark test level==5, 65 integer-gradeable
#            (48.5% of 134 L5 golds; 51.5% non-integer -> grader unfaithful on most).
set -uo pipefail
cd /workspace/senpai/target
source .venv/bin/activate
RD=research/validity/ci_world_pool_feasibility_719/results
BASE=http://127.0.0.1:8000

echo "[secondary] $(date -u +%FT%TZ) waiting for .BAND9504_DONE ..."
for _ in $(seq 1 320); do        # bounded: <=320*15s = 80 min cap
  [ -f "$RD/.BAND9504_DONE" ] && break
  sleep 15
done
[ -f "$RD/.BAND9504_DONE" ] || { echo "[secondary] bands never finished; aborting"; exit 1; }

echo "[secondary] $(date -u +%FT%TZ) running AMC + MATH-L5 ..."
python research/validity/ci_world_pool_feasibility_719/secondary_eval.py \
  --base-url "$BASE" --sources amc,math_l5 \
  --client-concurrency 16 --max-tokens 6144 --min-tokens 8 \
  --request-timeout-s 1200 \
  --out "$RD/secondary_results.json" \
  && echo "[secondary] done $(date -u +%FT%TZ)" || echo "[secondary] FAILED"
touch "$RD/.SECONDARY_DONE"
