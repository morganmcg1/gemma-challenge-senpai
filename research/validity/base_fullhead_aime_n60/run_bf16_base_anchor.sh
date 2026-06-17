#!/usr/bin/env bash
# ubel #580: UNQUANTIZED bf16 base AIME over the FULL n=60 set (AIME-2024 +
# AIME-2025 I/II), greedy maj@1, conc=32, request min_tokens=8 -- the EXACT #567
# harness (run ns5l6i28) so the number is directly comparable to base_fullhead
# int4 (0.1167) and plain int4 base (0.0667). This grounds the ">=90% of base"
# AIME gate DENOMINATOR: confirms/refutes the cited unquantized-base AIME=0.400
# and quantifies the int4-quant tax. UNQUANTIZED bf16 google/gemma-4-E4B-it, full
# native 262k head, no patches/spec/prune/quant -- stock vLLM (pinned 0.22.1rc1),
# the apples-to-apples bf16 counterpart of submissions/int4_base_aime.
set -u
cd /workspace/senpai/target
OUT=research/validity/base_fullhead_aime_n60
PYV=/tmp/senpai-venvs/5f4c623f772358a2/bin/python
COMMON="--server-python $PYV --years 2024,2025-I,2025-II --k 1 --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens 3072 --seed 1234 --no-thinking --max-num-seqs 32 --save-text"
STATUS=$OUT/run_bf16_base_anchor.status
echo "START $(date -u +%FT%TZ)" > "$STATUS"

echo "[driver] === UNQUANTIZED bf16 base n60 min_tokens=8 ===" | tee -a "$STATUS"
t0=$(date +%s)
/usr/bin/python3 research/downstream_quality_aime/aime_eval.py --submission submissions/bf16_base_aime $COMMON \
  --min-tokens 8 \
  --label bf16_base_aime_min8 --out $OUT/aime_bf16_base_min8_n60.json \
  > $OUT/aime_bf16_base_min8_n60.driver.log 2>&1
rc=$?
echo "[driver] bf16 base anchor rc=$rc elapsed=$(( $(date +%s)-t0 ))s" | tee -a "$STATUS"
echo "DONE $(date -u +%FT%TZ) rc=$rc" | tee -a "$STATUS"
