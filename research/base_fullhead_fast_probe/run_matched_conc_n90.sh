#!/usr/bin/env bash
# PR #535 matched-conc n~=90 AIME arm: both arms at conc=32, greedy maj@1, on the
# identical AIMO-validation item set (AIME 2022+2023+2024 = 90 problems).
set -u
cd /workspace/senpai/target
OUT=research/base_fullhead_fast_probe
PYV=/tmp/senpai-venvs/5f4c623f772358a2/bin/python
BASE_INT4=/senpai-run/home/student-fern/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0
COMMON="--server-python $PYV --years aimo-2022-2024 --k 1 --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens 3072 --seed 1234 --no-thinking --max-num-seqs 32 --save-text"
STATUS=$OUT/matched_conc_n90.status
echo "START $(date -u +%FT%TZ)" > "$STATUS"

echo "[driver] === ARM 1: base_fullhead conc=32 n90 ===" | tee -a "$STATUS"
t0=$(date +%s)
python research/downstream_quality_aime/aime_eval.py --submission submissions/fa2sw_strict_surgical357 $COMMON \
  --label base_fullhead_conc32_n90 --out $OUT/aime_base_fullhead_conc32_n90.json \
  --serve-env LOCAL_MODEL_DIR=$BASE_INT4 --serve-env PLE_FOLD_TARGET_MODEL=$BASE_INT4 \
  --serve-env LM_HEAD_PRUNE=0 --serve-env LM_HEAD_PRUNE_REQUIRE=0 --serve-env PCK04_KEEPSET= \
  > $OUT/aime_base_fullhead_conc32_n90.driver.log 2>&1
rc1=$?
echo "[driver] ARM1 rc=$rc1 elapsed=$(( $(date +%s)-t0 ))s" | tee -a "$STATUS"

echo "[driver] === ARM 2: plain base conc=32 n90 ===" | tee -a "$STATUS"
t1=$(date +%s)
python research/downstream_quality_aime/aime_eval.py --submission submissions/int4_base_aime $COMMON \
  --label int4_plain_conc32_n90 --out $OUT/aime_int4_plain_conc32_n90.json \
  > $OUT/aime_int4_plain_conc32_n90.driver.log 2>&1
rc2=$?
echo "[driver] ARM2 rc=$rc2 elapsed=$(( $(date +%s)-t1 ))s" | tee -a "$STATUS"
echo "DONE $(date -u +%FT%TZ) rc1=$rc1 rc2=$rc2" | tee -a "$STATUS"
