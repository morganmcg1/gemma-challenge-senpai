#!/usr/bin/env bash
# PR #811 pass-2: launches AFTER pass-1 (PID arg) releases the single GPU.
# Cells: maxlen3072 (the ONLY valid right-size, floor 2939) x2 reps + a control
# re-anchor (guards clock/thermal drift between pass-1 ~17:41 and pass-2 ~18:30).
set -u
PASS1_PID="${1:?need pass-1 pid}"
ROOT=/workspace/senpai/target
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
cd "$ROOT" || exit 2
echo "[pass2] waiting for pass-1 pid=$PASS1_PID to exit ..."
until ! kill -0 "$PASS1_PID" 2>/dev/null; do sleep 15; done
echo "[pass2] pass-1 gone; letting GPU/port settle 15s"
sleep 15
echo "[pass2] launching sweep cells"
exec "$VENV" research/serve_config_sweep/sweep.py \
  maxlen3072 maxlen3072 control \
  --num-prompts 128 --output-len 512 --port 8033
