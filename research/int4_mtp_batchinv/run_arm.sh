#!/usr/bin/env bash
# Batch-invariant greedy-identity runner for ONE arm. LOCAL ONLY (A10G); NOT an
# HF Job. Holds the spec-decode stack fixed (int4 target + gemma4_assistant MTP
# drafter + K) and varies ONLY VLLM_BATCH_INVARIANT (INV=1 ON vs INV=0 OFF), to
# test whether batch-invariant kernels make the M=K+1 spec-verify forward argmax
# equal the M=1 AR forward argmax (-> greedy-token-identical spec decode).
#
# Per arm: (1) serve target with speculation OFF -> capture the M=1 AR greedy
# reference; (2) serve target with speculation K -> capture the spec candidate;
# (3) strict bit-exact official verifier (--json); (4) flip_rate.py -> per-token
# flip rate (censored-geometric MLE). BOTH phases run with the SAME
# VLLM_BATCH_INVARIANT (the submission config) and ENFORCE_EAGER=1, so within an
# arm the ONLY config delta is speculative_config, and across arms the ONLY delta
# is VLLM_BATCH_INVARIANT.
#
# Env: ARM (label), INV (0/1, default 1), TARGET_MODEL_ID, QUANT (optional, e.g.
#      fp8), K (default 6), NPROMPTS (default 32), MAXLEN (default 4096),
#      OUTDIR (default /tmp/arms_bi)
set -uo pipefail
cd "$(dirname "$0")"
ARM="${ARM:?arm label}"
INV="${INV:-1}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:-google/gemma-4-E4B-it-qat-w4a16-ct}"
QUANT="${QUANT:-}"
K="${K:-6}"
NPROMPTS="${NPROMPTS:-32}"
MAXLEN="${MAXLEN:-4096}"
OUTDIR="${OUTDIR:-/tmp/arms_bi}"
BASE="http://127.0.0.1:8000"
MODEL_NAME="gemma-4-e4b-it"
PY="/workspace/senpai/target/.venvs/vllm022/bin/python"
VDIR="/workspace/senpai/target/official/main_bucket/shared_resources/gemma_greedy_identity_verifier_flowian-powers"
mkdir -p "$OUTDIR"

free_gpu() {
  for _ in $(seq 1 80); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ "$used" -lt 800 ] && return 0
    sleep 3
  done
  return 1
}

serve_phase() {
  local tag="$1" nspec="$2"
  local log="$OUTDIR/${ARM}_${tag}_server.log"
  local out="$OUTDIR/${ARM}_${tag}_decode"
  echo "=== [$ARM/$tag] launch nspec=$nspec INV=$INV model=$TARGET_MODEL_ID quant=${QUANT:-none} maxlen=$MAXLEN $(date +%H:%M:%S) ==="
  MODEL_ID="$TARGET_MODEL_ID" QUANTIZATION="$QUANT" VLLM_BATCH_INVARIANT="$INV" \
    NUM_SPECULATIVE_TOKENS="$nspec" ENFORCE_EAGER=1 MAX_MODEL_LEN="$MAXLEN" \
    setsid bash launch_server.sh >"$log" 2>&1 &
  local spid=$!
  sleep 1
  local pgid; pgid=$(ps -o pgid= -p "$spid" 2>/dev/null | tr -d ' ')
  echo "[$ARM/$tag] pid=$spid pgid=$pgid log=$log"
  if bash wait_ready.sh "$log" 900; then
    if bash capture_decode.sh "$BASE" "$MODEL_NAME" "$out" "$NPROMPTS" \
        >"$OUTDIR/${ARM}_${tag}_decode.capture.log" 2>&1; then
      echo "[$ARM/$tag] decode wrote ${out}.jsonl"
    else
      echo "[$ARM/$tag] DECODE FAILED -- see ${ARM}_${tag}_decode.capture.log"
      [ -n "${pgid:-}" ] && kill -TERM -"$pgid" 2>/dev/null; free_gpu; return 1
    fi
  else
    echo "[$ARM/$tag] SERVER NOT READY -- see $log"
    [ -n "${pgid:-}" ] && kill -TERM -"$pgid" 2>/dev/null; free_gpu; return 1
  fi
  [ -n "${pgid:-}" ] && kill -TERM -"$pgid" 2>/dev/null
  kill -TERM "$spid" 2>/dev/null
  free_gpu || echo "[$ARM/$tag] WARN gpu not freed"
  return 0
}

serve_phase ref 0    || { echo "ARM $ARM FAILED at ref $(date +%H:%M:%S)"; exit 1; }
serve_phase cand "$K" || { echo "ARM $ARM FAILED at cand $(date +%H:%M:%S)"; exit 1; }

REF="$OUTDIR/${ARM}_ref_decode.jsonl"
CAND="$OUTDIR/${ARM}_cand_decode.jsonl"
REPORT="$OUTDIR/${ARM}_verify.json"
echo "=== [$ARM] verify $(date +%H:%M:%S) ==="
( cd "$VDIR" && "$PY" check_greedy_identity.py --reference "$REF" --candidate "$CAND" --json ) \
  >"$REPORT" 2>"$OUTDIR/${ARM}_verify.err" || true
"$PY" flip_rate.py "$REPORT" --arm "$ARM" | tee "$OUTDIR/${ARM}_fliprate.txt"
echo "=== ARM $ARM DONE $(date +%H:%M:%S) ==="
