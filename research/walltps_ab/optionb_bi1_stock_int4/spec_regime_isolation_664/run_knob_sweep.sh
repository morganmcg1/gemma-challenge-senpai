#!/usr/bin/env bash
# PR #664 spec-regime isolation -- OWN-CONFIG knob sweep on land's 170 K6 regime.
#
# In-scope half (advisor 12:34Z): does a shippable, identity-safe latency knob exist
# that moves MY un-rescued K6 wall_tps? Vary one knob at a time vs the #660 K6 baseline
# (sampler=0 / cudagraph ON / TRITON_ATTN), re-measure paired wall_tps. NO stark data.
#
# Baseline (#660 fast regime): VLLM_BATCH_INVARIANT=1, DRAFTER=/tmp/qat-assistant,
#   NUM_SPECULATIVE_TOKENS=6, build_serve_env default VLLM_USE_FLASHINFER_SAMPLER=0,
#   ENFORCE_EAGER unset (cudagraph ON), VLLM_ATTENTION_BACKEND unset (Gemma4->TRITON_ATTN).
# Knobs (each = baseline + ONE change):
#   sampler1 : VLLM_USE_FLASHINFER_SAMPLER=1
#   eager1   : ENFORCE_EAGER=1                (engine cudagraph OFF)
#   flashattn: VLLM_ATTENTION_BACKEND=FLASH_ATTN (Gemma4 may override -> read server.log)
#
# LOCAL only. analysis_only. NO HF Job, NO submission. official_tps stays 0.
set -u
ROOT=/workspace/senpai/target
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/research/walltps_ab/optionb_bi1_stock_int4/spec_regime_isolation_664"
AB="$ROOT/scripts/profiler/paired_tps_ab.py"
N="${N:-3}"
export CUDA_VISIBLE_DEVICES=0
log(){ echo "[knobsweep $(date -u +%H:%M:%S)] $*"; }

BASE_ENV=(
  --baseline-env VLLM_BATCH_INVARIANT=1
  --baseline-env MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct
  --baseline-env DRAFTER_MODEL=/tmp/qat-assistant
  --baseline-env NUM_SPECULATIVE_TOKENS=6
)
# candidate carries the same 4 base vars + the one knob under test
cand_env(){ printf -- '--candidate-env %s ' \
  VLLM_BATCH_INVARIANT=1 MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct \
  DRAFTER_MODEL=/tmp/qat-assistant NUM_SPECULATIVE_TOKENS=6 "$@"; }

cd "$ROOT"

# ---- Run 1: baseline k6 (FRESH, reusable) + sampler1 candidate ----
log "=== RUN 1: baseline k6 + sampler1 (2 arms, N=$N) ==="
"$PY" "$AB" --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --baseline-label k6_base --candidate-label sampler1 \
  "${BASE_ENV[@]}" $(cand_env VLLM_USE_FLASHINFER_SAMPLER=1) \
  --n "$N" --num-prompts 128 --output-len 512 --seed 1 \
  --out-dir "$OUT/sampler" --no-project --no-wandb \
  >"$OUT/run1_sampler.log" 2>&1
log "RUN 1 exit=$? -> $OUT/sampler/paired_ab.json"

BASE_JSON="$OUT/sampler/paired_ab.json"
if [ ! -f "$BASE_JSON" ]; then
  log "FATAL: baseline json missing; aborting remaining knobs"; touch "$OUT/sweep.FAILED"; exit 1
fi

# ---- Run 2: eager1 (reuse baseline) ----
log "=== RUN 2: eager1 (reuse baseline, N=$N) ==="
"$PY" "$AB" --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --baseline-label k6_base --candidate-label eager1 \
  --reuse-baseline-from "$BASE_JSON" \
  "${BASE_ENV[@]}" $(cand_env ENFORCE_EAGER=1) \
  --n "$N" --num-prompts 128 --output-len 512 --seed 1 \
  --out-dir "$OUT/eager" --no-project --no-wandb \
  >"$OUT/run2_eager.log" 2>&1
log "RUN 2 exit=$? -> $OUT/eager/paired_ab.json"

# ---- Run 3: flashattn (reuse baseline) ----
log "=== RUN 3: flashattn (reuse baseline, N=$N) ==="
"$PY" "$AB" --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --baseline-label k6_base --candidate-label flashattn \
  --reuse-baseline-from "$BASE_JSON" \
  "${BASE_ENV[@]}" $(cand_env VLLM_ATTENTION_BACKEND=FLASH_ATTN) \
  --n "$N" --num-prompts 128 --output-len 512 --seed 1 \
  --out-dir "$OUT/attn" --no-project --no-wandb \
  >"$OUT/run3_attn.log" 2>&1
log "RUN 3 exit=$? -> $OUT/attn/paired_ab.json"

touch "$OUT/sweep.done"
log "ALL DONE -> $OUT/sweep.done"
