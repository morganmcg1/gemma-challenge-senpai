#!/usr/bin/env bash
# PR #639 Arm 2 -- rebuild OUR exact int4 pipeline at group_size 32 (body + head),
# keeping the untied/quantized lm_head. Isolates the g128->g32 group-size delta vs
# the live submission int4_g128_lmhead (everything else identical: same QAT-unq
# source, same 343-module set, same min-max RTN observer, same compressed-tensors
# pack-quantized -> Marlin path). CPU-ONLY build (safe_open device=cpu); does NOT
# touch the GPU, so it can run while the Arm 1 server/eval holds the A10G.
set -euo pipefail
ROOT=/workspace/senpai/target
cd "$ROOT"
PY="${PY:-/tmp/vllm0220-srv/bin/python}"      # torch 2.11.0 + compressed_tensors 0.15.0.1
SRC="${SRC:-/workspace/gemma_build/qat_unq}"
OUT="${OUT:-/workspace/gemma_build/int4_g32_lmhead}"
echo "[build] ours-g32 src=$SRC out=$OUT $(date -u +%FT%TZ)"
"$PY" -c "import compressed_tensors as ct; print('[build] compressed_tensors', ct.__version__)"
"$PY" submissions/int4_g128_lmhead/build_quant.py \
  --src "$SRC" \
  --out "$OUT" \
  --group-size 32 \
  --head-group-size 32
echo "[build] DONE $(date -u +%FT%TZ)"
ls -la "$OUT"
