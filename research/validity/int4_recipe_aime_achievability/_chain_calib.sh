#!/usr/bin/env bash
# PR #679 CALIBRATION leg (axis 2: observer, grid fixed).
#
# The PR asks "generic vs math/reasoning-domain calibration corpus". The repo
# ships NO corpus-driven quantizer: compressed_tensors 0.15 carries the observer
# field as metadata only, and GPTQ/AWQ/llmcompressor are NOT installed, and the
# model is a custom multimodal MatFormer whose shipped build does manual
# safetensors surgery precisely to avoid transformers quant. So a true
# activation/math-domain calibration is a CLUSTER-TRAINING-REQUEST follow-up.
#
# The only data-free "calibration-ish" lever the repo actually ships is the
# build_quant.py `mse` observer: a per-group clip search (40-pt grid, <=45%
# shrink) that minimises int4 round-trip WEIGHT MSE (guaranteed <= minmax error).
# It is output-blind and corpus-free -- NOT the PR's hypothesised math-cal -- but
# it is the honest in-repo probe of "does a better clip threshold recover AIME?".
#
# Two arms, both vs their same-grid minmax control:
#   A) mse@g128  -- SPEED-FREE probe. Identical byte layout to the 126.378 anchor
#                   (observer changes scale VALUES, not the #scales). If THIS
#                   clears AIME it is a quality-safe int4 body at ZERO speed cost
#                   -- the only recipe that could reach INT4_RECIPE_CLEARS_AIME if
#                   g32 fails the speed gate. 4 sessions (decisive).
#   B) mse@g32   -- PR-specified "calibration at the best grid". g32 already == the
#                   QAT-native q4_0 32-elem grid (rel_err ~0.0667, uniform), so mse
#                   has little clip slack left -> expected ~no-op vs minmax@g32.
#                   3 sessions (confirmatory).
#
# Builds run on CPU (safetensors surgery), so they don't need the GPU. Each
# checkpoint (~10 GB) is deleted right after its band to stay under disk; the
# _build_meta.json (rel_err provenance) is copied out first. LOCAL, no HF Job.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd /workspace/senpai/target
OUT="$HERE/_chain_calib.out"
PY=/tmp/vllm0220-srv/bin/python
BUILD=research/validity/int4_recipe_aime_achievability/build_offline.py
SRC=/senpai-run/home/student-wirbel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized/snapshots/dfc5b925ddb1d41aaf1fe9679abdcfb0805e1aa6
rm -f "$HERE/_chain_calib.DONE"
log(){ echo "[calib] $* $(date -u +%H:%M:%S)" | tee -a "$OUT"; }

# safety: clear any stale server before we start
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && sleep 8 || true

build_arm() {  # $1=out-dir $2=group-size
  local out="$1" gs="$2"
  log "=== build mse@g${gs} -> $out ==="
  "$PY" "$BUILD" --src "$SRC" --out "$out" --body-group-size "$gs" \
    --body-observer mse >> "$OUT" 2>&1 || { log "build mse@g${gs} FAILED"; return 1; }
  cp -f "$out/_build_meta.json" "$HERE/_build_meta_g${gs}mse.json" 2>/dev/null || true
}

# ---- ARM A: mse @ g128 (speed-free probe, 4 sessions) ----
G128MSE=/workspace/gemma_build/int4_g128body_mse_lmhead
if build_arm "$G128MSE" 128; then
  log "=== band g128mse (4 sessions) ==="
  bash "$HERE/run_band.sh" g128mse "$G128MSE" 4 0 0
  log "=== free $G128MSE ==="; rm -rf "$G128MSE"
fi

# ---- ARM B: mse @ g32 (PR best-grid calibration, 3 sessions) ----
G32MSE=/workspace/gemma_build/int4_g32body_mse_lmhead
if build_arm "$G32MSE" 32; then
  log "=== band g32mse (3 sessions) ==="
  bash "$HERE/run_band.sh" g32mse "$G32MSE" 3 0 0
  log "=== free $G32MSE ==="; rm -rf "$G32MSE"
fi

log "=== CALIB CHAIN DONE ==="
touch "$HERE/_chain_calib.DONE"
