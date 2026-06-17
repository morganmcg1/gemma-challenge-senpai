#!/usr/bin/env bash
# PR #609 ops: shared-disk cleanup reclaimed /workspace/gemma_build, so the named
# int4_g128_lmhead base (126.378 official-TPS baseline, W&B 905tbujn) must be
# rebuilt from its canonical source. Faithful reproduction of the Jun-13 build:
#   download QAT-unquantized -> build_quant.py (g128 body + untied int4 lm_head)
#   -> validate_offline.py PPL gate (expect ~2.019, cap 2.42) -> free 16G source.
# LOCAL/analysis-only. No HF Job, no submission, no served-file change.
set -euo pipefail

V=/tmp/senpai-venvs/20f658587e8a6643/bin/python
ROOT=/workspace/senpai/target
SRC=/workspace/gemma_build/qat_unq
OUT=/workspace/gemma_build/int4_g128_lmhead
OUTDIR="$ROOT/research/validity/ngram_spec_dec_int4/_build"
cd "$ROOT"

echo "[rebuild $(date -u +%H:%M:%S)] === STEP 1/4: download source (15.88GB) ==="
mkdir -p "$SRC" "$OUTDIR"
"$V" - <<PY
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    "google/gemma-4-E4B-it-qat-q4_0-unquantized",
    local_dir="$SRC", token=os.environ.get("HF_TOKEN"),
    allow_patterns=["model.safetensors","config.json","tokenizer.json",
                    "tokenizer_config.json","chat_template.jinja",
                    "generation_config.json","processor_config.json",
                    "special_tokens_map.json","preprocessor_config.json"],
)
print("[download] ->", p)
PY

echo "[rebuild $(date -u +%H:%M:%S)] === STEP 2/4: build_quant.py (g128 body + g128 head) ==="
"$V" submissions/int4_g128_lmhead/build_quant.py \
  --src "$SRC" --out "$OUT" \
  --group-size 128 --head-group-size 128 --no-verify-official

echo "[rebuild $(date -u +%H:%M:%S)] === STEP 3/4: validate_offline.py PPL gate ==="
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$V" submissions/int4_g128_lmhead/validate_offline.py \
  --ckpt "$OUT" --device cuda \
  --out "$OUTDIR/ppl_offline.json"

echo "[rebuild $(date -u +%H:%M:%S)] === STEP 4/4: free 16G source ==="
rm -rf "$SRC"

echo "[rebuild $(date -u +%H:%M:%S)] DONE; checkpoint:"
du -sh "$OUT"
ls -la "$OUT"
