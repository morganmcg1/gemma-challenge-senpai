#!/usr/bin/env bash
# Precision-localization greedy-identity runner for ONE arm. LOCAL ONLY (A10G);
# NOT an HF Job. Holds the spec-decode stack fixed (same MTP drafter + K) and
# varies ONLY the target precision, to localize whether the temp=0 spec-decode
# divergence is int4 quantization near-tie noise or the M=K+1 verify GEMM shape.
#
# Per arm: (1) serve target with speculation OFF -> capture the M=1 AR greedy
# reference; (2) serve target with speculation K -> capture the spec candidate;
# (3) strict bit-exact verifier (--json); (4) flip_rate.py -> per-token flip rate.
# Both phases use ENFORCE_EAGER=1 so the ONLY config delta is speculative_config.
#
# Env: ARM (label), TARGET_MODEL_ID, QUANT (optional, e.g. fp8), K (default 6),
#      NPROMPTS (default 32), MAXLEN (default 4096), OUTDIR (default /tmp/arms)
set -uo pipefail
cd "$(dirname "$0")"
ARM="${ARM:?arm label}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:?target model id}"
QUANT="${QUANT:-}"
K="${K:-6}"
NPROMPTS="${NPROMPTS:-32}"
MAXLEN="${MAXLEN:-4096}"
OUTDIR="${OUTDIR:-/tmp/arms}"
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
  echo "=== [$ARM/$tag] launch nspec=$nspec model=$TARGET_MODEL_ID quant=${QUANT:-none} maxlen=$MAXLEN $(date +%H:%M:%S) ==="
  MODEL_ID="$TARGET_MODEL_ID" QUANTIZATION="$QUANT" \
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
