#!/usr/bin/env bash
# PR #606 deterministic-gate + submission-parity arms (sequential, single GPU).
#   ARM1 ref_v22   : vLLM 0.22.0  -> decode (identity ref) + PPL + wall_tps
#   ARM2 ref2_v22  : vLLM 0.22.0  -> decode (identity ref2, cross-start) + wall_tps
#   ARM3 dev307_ppl: dev307       -> decode (3rd dev307 sample) + PPL + wall_tps
# D1 verdict = compare ref_v22 vs ref2_v22 (target GREEDY_IDENTICAL 0/128).
set -u
cd /workspace/senpai/target
V22=/tmp/senpai-venvs/20f658587e8a6643/bin/python      # vllm 0.22.0 (manifest pin)
DEV307=/tmp/senpai-venvs/5f4c623f772358a2/bin/python   # vllm 0.22.1rc1.dev307
OUT=research/deterministic_gate_parity
LOG=logs/pr606
RUN_ARM="research.ar_identity_safe_tps.run_arm"

echo "=== ARM 1: ref_v22 (0.22.0, decode+ppl) $(date -u +%FT%TZ) ==="
"$V22" -m "$RUN_ARM" --arm-name ref_v22 --with-ppl --out-dir "$OUT/ref_v22" > "$LOG/ref_v22.log" 2>&1
echo "ARM1 rc=$?  $(date -u +%FT%TZ)"

echo "=== ARM 2: ref2_v22 (0.22.0, decode) $(date -u +%FT%TZ) ==="
"$V22" -m "$RUN_ARM" --arm-name ref2_v22 --out-dir "$OUT/ref2_v22" > "$LOG/ref2_v22.log" 2>&1
echo "ARM2 rc=$?  $(date -u +%FT%TZ)"

echo "=== ARM 3: dev307_ppl (dev307, decode+ppl) $(date -u +%FT%TZ) ==="
"$DEV307" -m "$RUN_ARM" --arm-name dev307_ppl --with-ppl --out-dir "$OUT/dev307_ppl" > "$LOG/dev307_ppl.log" 2>&1
echo "ARM3 rc=$?  $(date -u +%FT%TZ)"

echo "=== D1 COMPARE: ref_v22 vs ref2_v22 (official byte-exact, target GREEDY_IDENTICAL 0/128) $(date -u +%FT%TZ) ==="
CHECK=submissions/int4_g128_lmhead/check_greedy_identity.py
"$V22" "$CHECK" --phase compare \
  --reference "$OUT/ref_v22/decode_outputs.jsonl" \
  --candidate "$OUT/ref2_v22/decode_outputs.jsonl" > "$OUT/d1_ref_vs_ref2.json" 2>&1
echo "D1 compare rc=$?  $(date -u +%FT%TZ)"
cat "$OUT/d1_ref_vs_ref2.json"

echo "=== ALL ARMS DONE $(date -u +%FT%TZ) ==="
