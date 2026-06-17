#!/usr/bin/env bash
# PR #536 — measure AIME-greedy on the osoi5-body + base-262k-head transplant.
#
# Serves submissions/fa2sw_strict_surgical357 (the osoi5 body recipe: PLE-fold +
# surgical attn + split-KV) but points it at /tmp/osoi5-transplant-tie, whose
# config.json has tie_word_embeddings=True + lm_head in `ignore`. vLLM ties the
# output head to the full 262k BF16 embed_tokens (== base head, byte-identical)
# and SKIPS the stale osoi5 16k int4 lm_head. The ONLY moved variable vs the
# osoi5-16k row is the head.
#
# Spec is held OFF (SENPAI_REFERENCE_MODE=1): greedy spec output == M=1 AR output
# (target argmax governs), so AIME quality is invariant to spec on/off, and the
# drafter-side patches (FUSED_SPARSE_ARGMAX / ONEGRAPH / LOOPGRAPH / DIXIE, all
# keyed to the pruned 16k vocab) never import -> zero head-shape interaction risk.
# The target emits a true full-vocab argmax over 262144 (stock vLLM sampler).
#
# Usage:
#   run_transplant_aime.sh <out_dir> <label> [--limit N] [--max-tokens T] [extra aime_eval args...]
set -euo pipefail

OUT_DIR="${1:?out_dir}"; LABEL="${2:?label}"; shift 2
ROOT="/workspace/senpai/target"
SUBM="$ROOT/submissions/fa2sw_strict_surgical357"
TRANSPLANT="/tmp/osoi5-transplant-tie"
mkdir -p "$OUT_DIR"

cd "$ROOT"
exec python3 research/downstream_quality_aime/aime_eval.py \
  --submission "$SUBM" \
  --years 2024 \
  --k 1 --temperature 0.0 --top-p 1.0 --top-k -1 \
  --seed 1234 --no-thinking \
  --max-num-seqs 32 \
  --label "$LABEL" \
  --out "$OUT_DIR/aime_${LABEL}.json" \
  --save-text \
  --serve-env LOCAL_MODEL_DIR="$TRANSPLANT" \
  --serve-env PLE_FOLD_TARGET_MODEL="$TRANSPLANT" \
  --serve-env LM_HEAD_PRUNE=0 \
  --serve-env LM_HEAD_PRUNE_REQUIRE=0 \
  --serve-env PCK04_KEEPSET= \
  --serve-env FUSED_SPARSE_ARGMAX=0 \
  --serve-env FUSED_SPARSE_ARGMAX_REQUIRE=0 \
  --serve-env SENPAI_REFERENCE_MODE=1 \
  --serve-env LOOPGRAPH_REQUIRE_CAPTURE=0 \
  --serve-env DIXIE_FUSED_ACCEPT_PREP=0 \
  --serve-env DIXIE_FUSED_ACCEPT_PREP_REQUIRE=0 \
  "$@"
