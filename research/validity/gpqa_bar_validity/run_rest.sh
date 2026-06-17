#!/usr/bin/env bash
# PR #614 -- after greedy@4096, run the rest on the SAME serve process (so greedy and
# sampled are on one bf16 process; cross-session argmax flips would otherwise add noise):
#   * greedy@2048 (real run) -- assumption-free greedy old-cap accuracy, and a cross-check
#     of the prefix-derived greedy@2048.
#   * sampled@4096 x5 seeds (lewtun #31) -- primary sampled + cluster-bootstrap CI.
# Idempotent: skips any run whose output json already exists. Each run_eval.py call is a
# separate process under the 90-min wall.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNONE="$HERE/run_one.sh"
echo "[rest] START $(date -u +%FT%TZ)"

g2="$HERE/runs/greedy_2048.json"
if [ -s "$g2" ]; then
  echo "[rest] greedy_2048 exists, skip"
else
  bash "$RUNONE" greedy_2048 0.0 1.0 0 2048 0 "$g2"; rc=$?
  echo "[rest] greedy_2048 rc=$rc $(date -u +%FT%TZ)"
  [ "$rc" -ne 0 ] && { echo "[rest] ABORT greedy_2048 rc=$rc"; exit $rc; }
fi

bash "$HERE/run_sampled_5seeds.sh"; rc=$?
echo "[rest] sampled5 rc=$rc $(date -u +%FT%TZ)"
echo "[rest] DONE $(date -u +%FT%TZ)"
exit $rc
