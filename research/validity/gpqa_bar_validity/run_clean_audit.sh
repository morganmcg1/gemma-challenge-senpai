#!/usr/bin/env bash
# PR #614 -- the truncation-clean GPQA-Diamond base denominator audit, run against the
# CLEAN PLE-folded fp16 vanilla serve (PLE_FOLD_EMBED_SCALE=1; root-caused in #557).
# One greedy@4096 run (the bar's regime anchor + truncation audit: a single 4096 run
# yields BOTH the 4096 and the derived <=2048 finish_length rate and -- since greedy is
# deterministic -- the exact derived accuracy@2048) then 5 SAMPLED@4096 seeds (lewtun
# #31; cluster-bootstrap CI). Each run_eval.py is a separate process under the 90-min
# wall; idempotent (skips any run whose json already exists). NO FIRE / analysis_only.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNONE="$HERE/run_one.sh"
echo "[clean-audit] START $(date -u +%FT%TZ)"

g4="$HERE/runs/greedy_4096.json"
if [ -s "$g4" ]; then
  echo "[clean-audit] greedy_4096 exists, skip"
else
  bash "$RUNONE" greedy_4096 0.0 1.0 0 4096 0 "$g4"; rc=$?
  echo "[clean-audit] greedy_4096 rc=$rc $(date -u +%FT%TZ)"
  [ "$rc" -ne 0 ] && { echo "[clean-audit] ABORT greedy_4096 rc=$rc"; exit $rc; }
fi

for s in 1 2 3 4 5; do
  out="$HERE/runs/sampled_4096_s${s}.json"
  if [ -s "$out" ]; then echo "[clean-audit] sampled seed=$s exists, skip"; continue; fi
  bash "$RUNONE" "sampled_4096_s${s}" 1.0 0.95 64 4096 "$s" "$out"; rc=$?
  echo "[clean-audit] sampled seed=$s rc=$rc $(date -u +%FT%TZ)"
  [ "$rc" -ne 0 ] && { echo "[clean-audit] ABORT sampled seed=$s rc=$rc"; exit $rc; }
done
echo "[clean-audit] ALL DONE $(date -u +%FT%TZ)"
