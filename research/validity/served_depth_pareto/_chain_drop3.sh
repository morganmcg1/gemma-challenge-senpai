#!/usr/bin/env bash
# drop=3 {36,37,38} full chain: as-served then min8 EOS-guard, MMLU+GPQA each.
set -uo pipefail
echo "[chain-drop3] START $(date -u +%H:%M:%SZ)"
./run_arm.sh bf16_drop3 0
echo "[chain-drop3] as-served DONE $(date -u +%H:%M:%SZ)"
./run_arm.sh bf16_drop3_min8 8
echo "[chain-drop3] min8 DONE $(date -u +%H:%M:%SZ); ALL ARMS COMPLETE"
