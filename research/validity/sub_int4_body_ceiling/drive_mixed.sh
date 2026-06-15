#!/bin/bash
# PR #372 hands-off chunk driver: wait for the in-flight chunk to exit, then
# relaunch resumable chunks (checkpointed per-module) until finalize writes the
# results JSON. Each chunk self-stops at --max-seconds (< the 90-min run cap).
set -u
cd /workspace/senpai/target || exit 2
DIR=research/validity/sub_int4_body_ceiling
RESULTS=$DIR/measure_mixed_precision_results.json
LOG=$DIR/drive_mixed.log
CURRENT_PID="${1:-}"

echo "[drive] $(date -u +%H:%M:%S) start; waiting for in-flight PID '$CURRENT_PID'" >> "$LOG"
if [ -n "$CURRENT_PID" ]; then
  while kill -0 "$CURRENT_PID" 2>/dev/null; do sleep 15; done
  echo "[drive] $(date -u +%H:%M:%S) in-flight chunk $CURRENT_PID exited" >> "$LOG"
fi

i=0
while [ ! -f "$RESULTS" ]; do
  i=$((i+1))
  if [ "$i" -gt 6 ]; then
    echo "[drive] $(date -u +%H:%M:%S) GIVE UP after 6 chunks without results" >> "$LOG"
    break
  fi
  echo "[drive] $(date -u +%H:%M:%S) launching chunk #$i" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online .venv/bin/python \
    "$DIR/measure_mixed_precision.py" \
    --max-seconds 4200 --wandb_group mixed-precision-bit-allocation \
    --wandb_name lawine/mixed-precision-subint4 >> "$DIR/mixed_chunk_${i}.out" 2>&1
  rc=$?
  done_n=$(.venv/bin/python -c "import json;print(len(json.load(open('$DIR/mixed_precision_checkpoint.json'))['modules']))" 2>/dev/null || echo "?")
  echo "[drive] $(date -u +%H:%M:%S) chunk #$i exited rc=$rc modules=$done_n/258" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "[drive] $(date -u +%H:%M:%S) nonzero rc=$rc; sleep 30 then retry" >> "$LOG"
    sleep 30
  fi
done
echo "[drive] $(date -u +%H:%M:%S) DONE results=$([ -f "$RESULTS" ] && echo yes || echo no)" >> "$LOG"
