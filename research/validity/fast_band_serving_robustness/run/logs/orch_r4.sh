#!/bin/bash
# PR #671 orchestrator: wait for the running r3 cal cell to free the GPU, then
# run a 2nd fresh FlashInfer server (r4) so c3 meets the deliverable's >=2-server
# bar uniformly with the native cells. Bounded local poll on a non-child PID.
set -u
cd /workspace/senpai/target
PY=/tmp/senpai-venvs/a341b8bdf5ec1fe0/bin/python
RUNDIR=research/validity/fast_band_serving_robustness/run
LOG="$RUNDIR/logs/r4.log"
SCRIPT=research/validity/fast_band_serving_robustness/serving_robustness_sweep.py
R3PID=$(cut -d= -f2 "$RUNDIR/logs/r3.pid" 2>/dev/null)

echo "[orch] start $(date -u +%H:%M:%S) waiting on r3 pid=$R3PID" >> "$LOG"
if [ -n "${R3PID:-}" ]; then
  while kill -0 "$R3PID" 2>/dev/null; do sleep 15; done
fi
echo "[orch] r3 exited $(date -u +%H:%M:%S); cal_spec_ar_m1 recorded:" >> "$LOG"
grep -c cal_spec_ar_m1 "$RUNDIR/r3/records.jsonl" >> "$LOG" 2>&1 || echo "0" >> "$LOG"

echo "[orch] launching r4 c3_flashinfer_fullpiece $(date -u +%H:%M:%S)" >> "$LOG"
"$PY" "$SCRIPT" --round r4 --cells c3_flashinfer_fullpiece --reps 2 >> "$LOG" 2>&1
rc=$?
echo "[orch] r4 finished rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"
touch "$RUNDIR/logs/r4.done"
exit $rc
