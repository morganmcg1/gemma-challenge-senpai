#!/usr/bin/env bash
# ubel #567 smoke: stand up base_fullhead (stock int4 QAT base + FULL native 262k
# head, prune OFF) on the surgical-357 fast stack and eval 2 AIME-2024 problems.
# Validates the serve recipe (fern #535) + the unguarded as-served path
# (MIN_TOKENS_FLOOR disabled, request min_tokens=0) before the full n=60 run.
set -u
cd /workspace/senpai/target
OUT=research/validity/base_fullhead_aime_n60
PYV=/tmp/senpai-venvs/5f4c623f772358a2/bin/python
MY_BASE_INT4=/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0
# base_fullhead = surgical-357 stack with the head-prune turned OFF + stock base
# int4 weights (NOT the osoi5 baked bucket). MIN_TOKENS_FLOOR= disables the
# server-side floor so the request-level min_tokens fully controls the EOS guard.
FH_ENV="--serve-env LOCAL_MODEL_DIR=$MY_BASE_INT4 --serve-env PLE_FOLD_TARGET_MODEL=$MY_BASE_INT4 --serve-env LM_HEAD_PRUNE=0 --serve-env LM_HEAD_PRUNE_REQUIRE=0 --serve-env PCK04_KEEPSET= --serve-env PLE_FOLD_EMBED_SCALE=1 --serve-env MIN_TOKENS_FLOOR="
echo "SMOKE START $(date -u +%FT%TZ)"
/usr/bin/python3 research/downstream_quality_aime/aime_eval.py \
  --submission submissions/fa2sw_strict_surgical357 \
  --server-python "$PYV" \
  --years 2024 --limit 2 --k 1 --temperature 0.0 --top-p 1.0 --top-k -1 \
  --max-tokens 3072 --seed 1234 --no-thinking --max-num-seqs 32 --save-text \
  --min-tokens 0 \
  --label smoke_fh_asserved --out "$OUT/_smoke_fh.json" \
  $FH_ENV
echo "SMOKE RC=$? $(date -u +%FT%TZ)"
