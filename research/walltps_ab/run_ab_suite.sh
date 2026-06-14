#!/bin/bash
# PR #82 wall_tps A/B suite: re-baseline + self-null + known-different + #56 re-screen.
# LOCAL-only, no HF launch. Single GPU, sequential (one server at a time).
# Run 1 (self-null) produces the locked baseline records the rest reuse (valid:
# PR #72 restart-invariance). Resilient: no `set -e`, each run logged separately.
set -u
cd /workspace/senpai/target
PY=.venv/bin/python
RUNNER="scripts/profiler/paired_tps_ab.py"
SELFNULL="research/walltps_ab/selfnull/paired_ab.json"
G="walltps-ab-runner"
OUT="research/walltps_ab"
SUB="fa2sw_precache_kenyan"

echo "=== [$(date -u +%H:%M:%S)] RUN 1/5: self-null + re-baseline (N=3 baseline + N=3 candidate) ==="
$PY "$RUNNER" \
  --baseline "$SUB" --candidate "$SUB" \
  --candidate-label selfnull --n 3 --tag selfnull \
  --wandb-name lawine/walltps-ab-selfnull --wandb-group "$G" \
  > "$OUT/selfnull.out" 2>&1
echo "RUN1_RC=$?  [$(date -u +%H:%M:%S)]"

echo "=== [$(date -u +%H:%M:%S)] RUN 2/5: known-different (DETOK_ENDONLY=0), reuse baseline ==="
$PY "$RUNNER" \
  --baseline "$SUB" --candidate "$SUB" \
  --candidate-env DETOK_ENDONLY=0 \
  --candidate-label detok_off --reuse-baseline-from "$SELFNULL" \
  --n 3 --tag known_diff_detok_off \
  --wandb-name lawine/walltps-ab-detok-off --wandb-group "$G" \
  > "$OUT/detok_off.out" 2>&1
echo "RUN2_RC=$?  [$(date -u +%H:%M:%S)]"

for MBT in 2048 4096 8192; do
  echo "=== [$(date -u +%H:%M:%S)] RUN re-screen MAX_NUM_BATCHED_TOKENS=$MBT, reuse baseline ==="
  $PY "$RUNNER" \
    --baseline "$SUB" --candidate "$SUB" \
    --candidate-env "MAX_NUM_BATCHED_TOKENS=$MBT" \
    --candidate-label "mbt$MBT" --reuse-baseline-from "$SELFNULL" \
    --n 3 --tag "rescreen_mbt$MBT" \
    --wandb-name "lawine/walltps-ab-mbt$MBT" --wandb-group "$G" \
    > "$OUT/mbt$MBT.out" 2>&1
  echo "MBT${MBT}_RC=$?  [$(date -u +%H:%M:%S)]"
done

echo "=== ORCHESTRATION_DONE [$(date -u +%H:%M:%S)] ==="
