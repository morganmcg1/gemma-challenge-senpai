#!/usr/bin/env bash
# PR #664 spec-regime isolation -- RECOVERY knob sweep (sweep1 aborted when the sampler1
# candidate boot-failed: FlashInfer JIT ninja compile fails in this container, and the knob
# is greedy-moot -> recorded sampler/BOOT_FAIL.json).
#
# This sweep measures the two BOOTABLE, throughput-relevant knobs on land's 170 K6 regime,
# REUSING the 3 clean baseline runs from sweep1 (median 169.950, restart-invariant per PR#72):
#   eager1   : ENFORCE_EAGER=1                 (engine cudagraph OFF)
#   flashattn: VLLM_ATTENTION_BACKEND=FLASH_ATTN (Gemma4 may override -> read server log)
# Each knob is an INDEPENDENT paired_tps_ab.py call (one boot-fail cannot kill the other),
# and neither re-measures the baseline. NO stark data. LOCAL only. analysis_only. official_tps=0.
set -u
ROOT=/workspace/senpai/target
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/research/walltps_ab/optionb_bi1_stock_int4/spec_regime_isolation_664"
AB="$ROOT/scripts/profiler/paired_tps_ab.py"
REUSE="$OUT/baseline_k6_reuse.json"
N="${N:-3}"
export CUDA_VISIBLE_DEVICES=0
log(){ echo "[knobsweep2 $(date -u +%H:%M:%S)] $*"; }

BASE_ENV=(
  --baseline-env VLLM_BATCH_INVARIANT=1
  --baseline-env MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct
  --baseline-env DRAFTER_MODEL=/tmp/qat-assistant
  --baseline-env NUM_SPECULATIVE_TOKENS=6
)
cand_env(){ printf -- '--candidate-env %s ' \
  VLLM_BATCH_INVARIANT=1 MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct \
  DRAFTER_MODEL=/tmp/qat-assistant NUM_SPECULATIVE_TOKENS=6 "$@"; }

cd "$ROOT"

if [ ! -s "$REUSE" ]; then log "FATAL: reuse baseline missing $REUSE"; touch "$OUT/sweep2.FAILED"; exit 1; fi

# ---- eager1 (reuse baseline) ----
log "=== eager1: ENFORCE_EAGER=1 (reuse baseline, N=$N) ==="
"$PY" "$AB" --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --baseline-label k6_base --candidate-label eager1 \
  --reuse-baseline-from "$REUSE" \
  "${BASE_ENV[@]}" $(cand_env ENFORCE_EAGER=1) \
  --n "$N" --num-prompts 128 --output-len 512 --seed 1 \
  --out-dir "$OUT/eager" --no-project --no-wandb \
  >"$OUT/run_eager.log" 2>&1
log "eager1 exit=$? -> $OUT/eager/paired_ab.json"

# ---- flashattn (reuse baseline) ----
log "=== flashattn: VLLM_ATTENTION_BACKEND=FLASH_ATTN (reuse baseline, N=$N) ==="
"$PY" "$AB" --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --baseline-label k6_base --candidate-label flashattn \
  --reuse-baseline-from "$REUSE" \
  "${BASE_ENV[@]}" $(cand_env VLLM_ATTENTION_BACKEND=FLASH_ATTN) \
  --n "$N" --num-prompts 128 --output-len 512 --seed 1 \
  --out-dir "$OUT/attn" --no-project --no-wandb \
  >"$OUT/run_attn.log" 2>&1
log "flashattn exit=$? -> $OUT/attn/paired_ab.json"

touch "$OUT/sweep2.done"
log "ALL DONE -> $OUT/sweep2.done"
