#!/usr/bin/env bash
# int4 drop=2 [37,38] confirm chain: as-served then min8 EOS-guard, MMLU+GPQA each.
# Mirrors _chain_drop3.sh so the int4 arm is byte-identical-harness to the bf16 arms.
set -uo pipefail
echo "[chain-int4-drop2] START $(date -u +%H:%M:%SZ)"
./run_arm.sh int4_drop2 0
echo "[chain-int4-drop2] as-served DONE $(date -u +%H:%M:%SZ)"
./run_arm.sh int4_drop2_min8 8
echo "[chain-int4-drop2] min8 DONE $(date -u +%H:%M:%SZ); ALL ARMS COMPLETE"
