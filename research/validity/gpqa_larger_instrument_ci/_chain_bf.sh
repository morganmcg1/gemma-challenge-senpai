#!/bin/bash
# PR #598 auto-handoff: wait for the base driver to fully exit (which means all
# base seeds done AND the base vLLM server killed -> GPU freed), then launch the
# base_fullhead K=5 driver on the now-free GPU. Conservative: refuses to launch
# if base produced <5 seed files (a watchdog/early-kill case) so we never start
# base_fullhead on an incomplete denominator. LOCAL, NO FIRE.
set -u
HERE=/workspace/senpai/target/research/validity/gpqa_larger_instrument_ci
RES="$HERE/results"
PY=/tmp/senpai-venvs/5f4c623f772358a2/bin/python   # dev307 build venv (#557/#564/#589)
BASE_DRIVER_PID="${1:?need base driver pid}"
cd /workspace/senpai/target

echo "[chain] $(date -u +%H:%M:%S) waiting for base driver PID $BASE_DRIVER_PID to exit"
while kill -0 "$BASE_DRIVER_PID" 2>/dev/null; do sleep 30; done
echo "[chain] $(date -u +%H:%M:%S) base driver exited"

n=$(ls "$RES"/base_gpqa_main_mt8_s*.json 2>/dev/null | wc -l)
echo "[chain] base seed files present: $n/5"
if [ "$n" -lt 5 ]; then
  echo "[chain] base INCOMPLETE ($n/5) -- NOT launching base_fullhead. Relaunch base to resume."
  exit 1
fi

sleep 15  # let CUDA memory fully release before the next server binds the GPU
echo "[chain] $(date -u +%H:%M:%S) launching base_fullhead K=5 on freed GPU"
exec "$PY" research/validity/gpqa_larger_instrument_ci/run_seeds.py \
  --config base_fullhead --seeds 0,1,2,3,4 --conc 16 --max-num-seqs 16 --gpu-mem-util 0.92
