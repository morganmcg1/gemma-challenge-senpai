#!/usr/bin/env bash
# PR #638 -- build the EXACT live-rung int4_g128_lmhead checkpoint locally so the
# int4-AR denominator leg serves the same body the submission ships. Recipe is
# byte-for-byte research/zoomout_ar_speed_screen/build_int4head.sh (PR #4 repro:
# deterministic minmax, no calibration, g128 body + untied int4 g128 lm_head),
# only the venv differs (that script's /tmp/senpai-venvs interpreter is gone).
# CPU-only -- safe to run while the bf16 server still holds GPU 0.
#
#   SRC = google/gemma-4-E4B-it-qat-q4_0-unquantized  (TRUE bf16 QAT master == qat_unq)
#   CT  = google/gemma-4-E4B-it-qat-w4a16-ct          (tokenizer/processor assets)
# Both are public Google checkpoints; here read from a shared HF hub cache.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/tmp/vllm0220-srv/bin/python                       # vLLM 0.22.0 venv: ct 0.15.0.1 (matches serve engine)
SRC=/senpai-run/home/student-wirbel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized/snapshots/dfc5b925ddb1d41aaf1fe9679abdcfb0805e1aa6
CT=/senpai-run/home/student-wirbel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0
OUT=/workspace/gemma_build/int4_g128_lmhead
LOG="$HERE/_build_int4ar.log"
cd /workspace/senpai/target

echo "[build] start $(date -u +%FT%TZ) py=$PY" | tee "$LOG"
"$PY" -c "import compressed_tensors as c; print('[build] ct', c.__version__)" | tee -a "$LOG"
"$PY" submissions/int4_g128_lmhead/build_quant.py \
  --src "$SRC" --out "$OUT" \
  --group-size 128 --head-group-size 128 --no-verify-official >>"$LOG" 2>&1
echo "[build] quant done $(date -u +%FT%TZ); copying tokenizer/processor assets" | tee -a "$LOG"
for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json processor_config.json special_tokens_map.json preprocessor_config.json; do
  [ -f "$CT/$f" ] && cp -v "$CT/$f" "$OUT/$f" >>"$LOG" 2>&1 || true
done
echo "[build] DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
ls -la "$OUT" | tee -a "$LOG"
