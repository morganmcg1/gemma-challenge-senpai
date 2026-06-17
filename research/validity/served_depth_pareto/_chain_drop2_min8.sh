#!/usr/bin/env bash
# Wait for the as-served GPQA (pid $1) to finish, then run the drop=2 EOS-guard
# min_tokens=8 arm (MMLU + GPQA) against the still-alive drop=2 server.
set -uo pipefail
GPQA_PID="$1"
echo "[chain] waiting for as-served GPQA pid=$GPQA_PID $(date -u +%H:%M:%SZ)"
until ! kill -0 "$GPQA_PID" 2>/dev/null; do sleep 15; done
echo "[chain] as-served GPQA done $(date -u +%H:%M:%SZ); launching drop2 min8"
./run_arm.sh bf16_drop2_min8 8
echo "[chain] drop2 min8 DONE $(date -u +%H:%M:%SZ)"
