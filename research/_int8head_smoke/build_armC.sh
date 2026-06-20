#!/usr/bin/env bash
# PR #796 arm C build: int4 g32 lm_head with MSE/clip-aware scales.
# Same int4 body / src as arms A/B and the merged-#788 control (byte-identical
# 2762 tensors copied); ONLY delta vs control = --observer mse at g32. This
# de-risks the FIRE candidate group size (g32) at identical bytes/speed.
set -euo pipefail
SRC=/senpai-run/home/student-lawine/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0
OUT=/workspace/gemma_build/lawine_bf_g32_mse
PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
cd /workspace/senpai/target/submissions/int4_mtp_bi0_lmhead_bytefloor
echo "[build:C] $(date -u +%H:%M:%S) src=$SRC out=$OUT g32 mse"
"$PY" build_lmhead_quant.py --src "$SRC" --out "$OUT" \
  --num-bits 4 --head-group-size 32 --observer mse
echo "[build:C] done $(date -u +%H:%M:%S)"
