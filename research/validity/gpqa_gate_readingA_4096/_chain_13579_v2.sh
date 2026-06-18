#!/usr/bin/env bash
# PR #643 n=10 chain (v2): wait for the seed-90123 driver to exit, then launch the
# final canonical sampled seed 13579. The original detached waiter died on a session
# boundary; this one is launched with setsid (own session) so it survives. Safe to run
# alongside a manual ScheduleWakeup launch: _launch_seed.sh refuses to double-launch
# (guards on _gate.pid AND any live run_gate.py/run_eval.py), so only ONE 13579 driver
# can ever start. The 13579 run_gate.py invocation self-finalizes (writes the n=10
# gate_summary.json, logs the W&B artifact, finishes the run).
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/gpqa_gate_readingA_4096
LOG="$HERE/_chain_13579.log"
TARGET_PID=502671   # seed-90123 sampled driver (n=9)
{
  echo "[chain-13579 v2] started $(date -u +%H:%M:%SZ); waiting for seed-90123 driver $TARGET_PID"
  while kill -0 "$TARGET_PID" 2>/dev/null; do sleep 20; done
  echo "[chain-13579 v2] driver $TARGET_PID exited $(date -u +%H:%M:%SZ); reaping 12s before launch"
  sleep 12
  bash "$HERE/_launch_seed.sh" 13579 sampled
  rc=$?
  echo "[chain-13579 v2] _launch_seed returned rc=$rc at $(date -u +%H:%M:%SZ)"
} >> "$LOG" 2>&1
