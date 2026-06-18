#!/usr/bin/env bash
# Phase-2 orchestrator for PR #675 (runs AFTER the dev307 sweep frees the GPU).
#  1. wait (bounded) for the dev307 sweep PID to exit  -> base2/base3 done
#  2. re-run analyze.py                                 -> fresh dev307 results.json
#  3. run the SHIP-version (0.22.0) deterministic control (2 fresh servers)
#  4. analyze_v0220.py                                  -> results_v0220.json
# Single GPU: steps are strictly sequential so two servers never overlap.
set -u
ROOT=/workspace/senpai/target
HERE="$ROOT/research/int4_gemv_kernel_audit"
DEV307PY=/tmp/senpai-venvs/5f4c623f772358a2/bin/python
LOG="$HERE/phase2.log"
cd "$ROOT" || exit 2

say () { echo "[phase2] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

SWEEP_PID="$(cat "$HERE/sweep.pid" 2>/dev/null || echo "")"
say "BEGIN  sweep_pid=$SWEEP_PID"

# 1. bounded wait for the dev307 sweep to finish base2+base3 (max ~40 min)
if [ -n "$SWEEP_PID" ]; then
  for _ in $(seq 1 240); do
    kill -0 "$SWEEP_PID" 2>/dev/null || break
    sleep 10
  done
fi
say "sweep exited (or wait elapsed)"

# 2. fresh dev307 analysis (folds in base2/base3 -> median-of-3 anchor)
say "analyze.py (dev307) START"
"$DEV307PY" "$HERE/analyze.py" >>"$LOG" 2>&1
say "analyze.py DONE rc=$?"

# 3. SHIP-version determinism control (its own venv inside the script)
say "0.22.0 control START"
bash "$HERE/run_v0220_control.sh" >>"$LOG" 2>&1
say "0.22.0 control DONE rc=$?"

# 4. control analysis
say "analyze_v0220.py START"
"$DEV307PY" "$HERE/analyze_v0220.py" >>"$LOG" 2>&1
say "analyze_v0220.py DONE rc=$?"
say "PHASE2 COMPLETE"
