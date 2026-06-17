#!/usr/bin/env bash
# ubel #567 Driver A: base_fullhead AIME over the FULL n=60 set (AIME-2024 +
# AIME-2025 I/II), greedy maj@1, conc=32. Two arms on the identical served
# checkpoint (stock int4 base + full native 262k head, prune OFF, baked-bucket
# OFF, MIN_TOKENS_FLOOR disabled so the REQUEST min_tokens fully controls EOS):
#   ARM 1 = as-served, request min_tokens=0  -> base_fullhead_aime_asserved + empty_rate
#   ARM 2 = guarded,   request min_tokens=8  -> base_fullhead_aime_min8 (the gate figure)
# The min8 arm is the apples-to-apples #524 gate number (wirbel #541 EOS-guard).
set -u
cd /workspace/senpai/target
OUT=research/validity/base_fullhead_aime_n60
PYV=/tmp/senpai-venvs/5f4c623f772358a2/bin/python
MY_BASE_INT4=/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0
COMMON="--server-python $PYV --years 2024,2025-I,2025-II --k 1 --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens 3072 --seed 1234 --no-thinking --max-num-seqs 32 --save-text"
FH_ENV="--serve-env LOCAL_MODEL_DIR=$MY_BASE_INT4 --serve-env PLE_FOLD_TARGET_MODEL=$MY_BASE_INT4 --serve-env LM_HEAD_PRUNE=0 --serve-env LM_HEAD_PRUNE_REQUIRE=0 --serve-env PCK04_KEEPSET= --serve-env PLE_FOLD_EMBED_SCALE=1 --serve-env MIN_TOKENS_FLOOR="
STATUS=$OUT/run_fh_arms.status
echo "START $(date -u +%FT%TZ)" > "$STATUS"

echo "[driver] === ARM 1: base_fullhead n60 AS-SERVED (min_tokens=0) ===" | tee -a "$STATUS"
t0=$(date +%s)
/usr/bin/python3 research/downstream_quality_aime/aime_eval.py --submission submissions/fa2sw_strict_surgical357 $COMMON \
  --min-tokens 0 \
  --label base_fullhead_aime_asserved --out $OUT/aime_fh_asserved_n60.json \
  $FH_ENV > $OUT/aime_fh_asserved_n60.driver.log 2>&1
rc1=$?
echo "[driver] ARM1 rc=$rc1 elapsed=$(( $(date +%s)-t0 ))s" | tee -a "$STATUS"

echo "[driver] === ARM 2: base_fullhead n60 MIN_TOKENS=8 (gate figure) ===" | tee -a "$STATUS"
t1=$(date +%s)
/usr/bin/python3 research/downstream_quality_aime/aime_eval.py --submission submissions/fa2sw_strict_surgical357 $COMMON \
  --min-tokens 8 \
  --label base_fullhead_aime_min8 --out $OUT/aime_fh_min8_n60.json \
  $FH_ENV > $OUT/aime_fh_min8_n60.driver.log 2>&1
rc2=$?
echo "[driver] ARM2 rc=$rc2 elapsed=$(( $(date +%s)-t1 ))s" | tee -a "$STATUS"
echo "DONE $(date -u +%FT%TZ) rc1=$rc1 rc2=$rc2" | tee -a "$STATUS"
