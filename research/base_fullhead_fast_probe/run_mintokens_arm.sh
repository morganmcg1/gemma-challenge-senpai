#!/usr/bin/env bash
# PR #535 in-flight steer (advisor, wirbel #541): does a first-token-EOS serving
# artifact depress base_fullhead AIME, and does request-level min_tokens=8 recover
# it (as it recovered GSM8K 0.762->0.854)? Same served checkpoint, same conc=32,
# same n90 item set (AIME 2022+2023+2024). No served-file change; min_tokens is a
# request-level vLLM param only.
#
# ARM A = base_fullhead + min_tokens=8 (the treatment).
# ARM B = base_fullhead as-served, fresh (chaos control: the fast stack is non-
#         deterministic run-to-run, so this brackets accuracy noise vs the existing
#         as-served arm 0.1444, isolating any genuine min_tokens effect).
set -u
cd /workspace/senpai/target
OUT=research/base_fullhead_fast_probe
PYV=/tmp/senpai-venvs/5f4c623f772358a2/bin/python
BASE_INT4=/senpai-run/home/student-fern/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0
COMMON="--server-python $PYV --years aimo-2022-2024 --k 1 --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens 3072 --seed 1234 --no-thinking --max-num-seqs 32 --save-text"
FH_ENV="--serve-env LOCAL_MODEL_DIR=$BASE_INT4 --serve-env PLE_FOLD_TARGET_MODEL=$BASE_INT4 --serve-env LM_HEAD_PRUNE=0 --serve-env LM_HEAD_PRUNE_REQUIRE=0 --serve-env PCK04_KEEPSET="
STATUS=$OUT/mintokens_arm.status
echo "START $(date -u +%FT%TZ)" > "$STATUS"

echo "[driver] === ARM A: base_fullhead conc=32 n90 min_tokens=8 ===" | tee -a "$STATUS"
t0=$(date +%s)
python research/downstream_quality_aime/aime_eval.py --submission submissions/fa2sw_strict_surgical357 $COMMON \
  --min-tokens 8 \
  --label base_fullhead_conc32_n90_mintok8 --out $OUT/aime_base_fullhead_conc32_n90_mintok8.json \
  $FH_ENV > $OUT/aime_base_fullhead_conc32_n90_mintok8.driver.log 2>&1
rcA=$?
echo "[driver] ARM_A rc=$rcA elapsed=$(( $(date +%s)-t0 ))s" | tee -a "$STATUS"

echo "[driver] === ARM B: base_fullhead conc=32 n90 as-served (fresh chaos control) ===" | tee -a "$STATUS"
t1=$(date +%s)
python research/downstream_quality_aime/aime_eval.py --submission submissions/fa2sw_strict_surgical357 $COMMON \
  --label base_fullhead_conc32_n90_rerun --out $OUT/aime_base_fullhead_conc32_n90_rerun.json \
  $FH_ENV > $OUT/aime_base_fullhead_conc32_n90_rerun.driver.log 2>&1
rcB=$?
echo "[driver] ARM_B rc=$rcB elapsed=$(( $(date +%s)-t1 ))s" | tee -a "$STATUS"
echo "DONE $(date -u +%FT%TZ) rcA=$rcA rcB=$rcB" | tee -a "$STATUS"
