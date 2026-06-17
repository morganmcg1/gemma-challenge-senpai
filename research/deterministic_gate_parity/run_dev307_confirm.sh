#!/usr/bin/env bash
# PR #606 confirmation: a 2nd dev307 arm (fresh cross-start) to test whether the
# dev307 PPL inflation (2.626 vs 0.22.0's 2.019) is SYSTEMATIC or a per-process
# kernel-draw VARIANCE artifact, and to measure a fresh in-PR dev307 cross-start
# greedy-identity (dev307_ppl vs dev307_ppl2) as the "before" baseline contrast.
set -u
cd /workspace/senpai/target
DEV307=/tmp/senpai-venvs/5f4c623f772358a2/bin/python   # vllm 0.22.1rc1.dev307
OUT=research/deterministic_gate_parity
LOG=logs/pr606
RUN_ARM="research.ar_identity_safe_tps.run_arm"
CHECK=submissions/int4_g128_lmhead/check_greedy_identity.py

echo "=== dev307_ppl2 (dev307, decode+ppl, cross-start) $(date -u +%FT%TZ) ==="
"$DEV307" -m "$RUN_ARM" --arm-name dev307_ppl2 --with-ppl --out-dir "$OUT/dev307_ppl2" > "$LOG/dev307_ppl2.log" 2>&1
echo "dev307_ppl2 rc=$?  $(date -u +%FT%TZ)"

echo "=== dev307 CROSS-START COMPARE: dev307_ppl vs dev307_ppl2 (official byte-exact) $(date -u +%FT%TZ) ==="
"$DEV307" "$CHECK" --phase compare \
  --reference "$OUT/dev307_ppl/decode_outputs.jsonl" \
  --candidate "$OUT/dev307_ppl2/decode_outputs.jsonl" > "$OUT/dev307_crossstart.json" 2>&1
echo "dev307 cross-start compare rc=$?  $(date -u +%FT%TZ)"
cat "$OUT/dev307_crossstart.json"
echo "=== DONE $(date -u +%FT%TZ) ==="
