#!/usr/bin/env bash
# PR #679 step-3 speed gate wrapper. Mirrors the proven _chain_bands.sh launch
# pattern (bash wrapper that spawns vLLM servers) which survives backgrounding,
# whereas invoking the python driver directly under the task runner died early.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd /workspace/senpai/target
OUT="$HERE/_speed_ab.out"
PY=/tmp/vllm0220-srv/bin/python
rm -f "$HERE/_speed_ab.DONE"
echo "[speedwrap] start $(date -u +%H:%M:%S)" | tee "$OUT"

# clear any stale server on the GPU first
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && sleep 8 || true

"$PY" -u "$HERE/_speed_ab.py" \
  --num-prompts 8 --output-len 512 --reps 3 --warmups 1 --port 8000 >> "$OUT" 2>&1
rc=$?
echo "[speedwrap] python rc=$rc $(date -u +%H:%M:%S)" | tee -a "$OUT"
echo "$rc" > "$HERE/_speed_ab.rc"
touch "$HERE/_speed_ab.DONE"
