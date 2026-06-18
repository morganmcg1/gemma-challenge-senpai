#!/usr/bin/env bash
# PR #656 — chain the AR arm to auto-start the instant the SPEC eval finishes.
# Waits on the captured SPEC eval PID (run_gpqa_10seed.sh), records how many SPEC
# seed result files landed, then hands off to run_ar_arm.sh (which stops the SPEC
# server, brings up the k=0 AR server on the same stack, and runs the 10-seed AR
# sweep). No broad pgrep: we wait on the exact PID in _run_spec.pid.
set -uo pipefail
cd /workspace/senpai/target
DIR=research/validity/specdec_gpqa_10seed
LOG=$DIR/_chain.log
: > "$LOG"
say() { echo "[$(date -u +%FT%TZ)] [chain] $*" | tee -a "$LOG"; }

SPEC_PID=$(cat "$DIR/_run_spec.pid")
say "waiting for SPEC eval pid=$SPEC_PID to finish"
while kill -0 "$SPEC_PID" 2>/dev/null; do sleep 30; done
NDONE=$(ls -1 "$DIR"/results/gpqa_spec_k6_s*.json 2>/dev/null | wc -l)
say "SPEC eval pid=$SPEC_PID exited; spec seed result files: $NDONE/10"
tail -3 "$DIR/_run_spec_k6.status" | sed 's/^/[chain]   /' | tee -a "$LOG"
say "handing off to run_ar_arm.sh"
bash "$DIR/run_ar_arm.sh" >> "$LOG" 2>&1
say "run_ar_arm.sh returned rc=$?"
NAR=$(ls -1 "$DIR"/results/gpqa_ar_m1_s*.json 2>/dev/null | wc -l)
say "AR seed result files: $NAR/10 — chain complete"
