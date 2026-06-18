#!/usr/bin/env bash
# Build the exact int4_g128_lmhead checkpoint locally (deterministic minmax quant,
# no calibration data -> reproduces the PR #4 submission build). CPU only.
set -euo pipefail
PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
SRC=/senpai-run/home/student-wirbel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized/snapshots/dfc5b925ddb1d41aaf1fe9679abdcfb0805e1aa6
CT=/senpai-run/home/student-wirbel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0
OUT=/workspace/gemma_build/int4_g128_lmhead
cd /workspace/senpai/target
echo "[build] start $(date -u +%H:%M:%S)"
$PY submissions/int4_g128_lmhead/build_quant.py \
  --src "$SRC" --out "$OUT" \
  --group-size 128 --head-group-size 128 --no-verify-official
echo "[build] quant done $(date -u +%H:%M:%S); copying tokenizer/processor assets"
for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json processor_config.json special_tokens_map.json preprocessor_config.json; do
  [ -f "$CT/$f" ] && cp -v "$CT/$f" "$OUT/$f" || true
done
echo "[build] DONE $(date -u +%H:%M:%S)"
ls -la "$OUT"
