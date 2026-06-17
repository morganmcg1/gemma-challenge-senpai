#!/usr/bin/env bash
# ubel #567 Driver B (supplementary): plain int4 base AIME over the FULL n=60 set,
# greedy maj@1, conc=32, same harness, request min_tokens=8 (matches the gate arm).
# Purpose: a fresh same-n / same-harness apples-to-apples denominator for
# pct_of_base, alongside the advisor's reference vanilla-base AIME=0.400 (the
# explicit #524 gate denominator, pass bar 0.36). Plain base is stock vLLM int4
# (no fast kernels / no 262k-head EOS artifact) so min8 ~= as-served here.
set -u
cd /workspace/senpai/target
OUT=research/validity/base_fullhead_aime_n60
PYV=/tmp/senpai-venvs/5f4c623f772358a2/bin/python
COMMON="--server-python $PYV --years 2024,2025-I,2025-II --k 1 --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens 3072 --seed 1234 --no-thinking --max-num-seqs 32 --save-text"
STATUS=$OUT/run_base_anchor.status
echo "START $(date -u +%FT%TZ)" > "$STATUS"

echo "[driver] === plain int4 base n60 min_tokens=8 ===" | tee -a "$STATUS"
t0=$(date +%s)
/usr/bin/python3 research/downstream_quality_aime/aime_eval.py --submission submissions/int4_base_aime $COMMON \
  --min-tokens 8 \
  --label int4_base_aime_min8 --out $OUT/aime_base_anchor_min8_n60.json \
  > $OUT/aime_base_anchor_min8_n60.driver.log 2>&1
rc=$?
echo "[driver] base anchor rc=$rc elapsed=$(( $(date +%s)-t0 ))s" | tee -a "$STATUS"
echo "DONE $(date -u +%FT%TZ) rc=$rc" | tee -a "$STATUS"
