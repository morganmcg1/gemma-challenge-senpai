#!/usr/bin/env bash
# PR #699 (greedy pivot, advisor decision 2026-06-19T05:00Z): the int4 SAMPLED 2x2 is
# unmeasurable -- the int4 body degenerates into repetition-loops under T=1.0 sampling on
# the only working engine (vLLM 0.22.0; the pinned dev307 that stabilized it is gone AND
# lawine #606 ruled dev307 INVALID for accuracy). Greedy (argmax) is immune, reconciles to
# the banked anchors (int4=0.350 / base=0.4667 @6144), and is the valid accuracy substrate.
#
# This driver serves ONE body once at MML=13312 (fits the 12288 budget) and evals GREEDY at
# two token budgets {6144, 12288}. The ONLY axis that varies between a body's two cells is
# the request max_tokens; engine/model-len/batch-width are held fixed. Budgets run 6144-FIRST
# so the reconciliation cell lands before the long 12288 cell.
#
# Greedy protocol = the banked anchor protocol VERBATIM (research/validity/
# optionb_denom_0p22_gb6144 meta): k=1, T=0.0, top_p=1.0, top_k=-1, min_tokens=8,
# no-thinking, seed=1234, years=2024,2025-I,2025-II (canonical n=60), BI=1.
# analysis_only: local serve only, NO HF Job, NO submission.
set -euo pipefail

BODY="${1:?body label: int4 | base}"
MODEL="${2:?model dir or HF id}"
PORT="${3:-8000}"
BUDGETS="${BUDGETS:-6144 12288}"   # space-separated; 6144 first (reconciliation cell)

K="${K:-1}"                        # greedy = 1 sample/problem
SEED="${SEED:-1234}"
MML="${MML:-13312}"                # 12288 budget + prompt headroom
MNS="${MNS:-16}"                   # served-anchor batch width; HOLD FIXED across both cells
CC="${CC:-16}"                     # client concurrency; greedy is BI-invariant to it
YEARS="${YEARS:-2024,2025-I,2025-II}"   # canonical 60

HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/results"; mkdir -p "$RES"
PY="/workspace/senpai/target/.venv/bin/python"          # eval CLIENT venv (stdlib only)
AIME="/workspace/senpai/target/research/downstream_quality_aime/aime_eval.py"

echo "[greedy2x2] BODY=$BODY MODEL=$MODEL budgets={$BUDGETS} K=$K mns=$MNS cc=$CC seed=$SEED BI=1 $(date -u +%H:%M:%SZ)"

# Serve with BATCH-INVARIANT ON (greedy-identity reproducibility; the banked anchors used BI=1).
VLLM_BATCH_INVARIANT=1 bash "$HERE/serve_body.sh" "$MODEL" "$PORT" "$MML" "$MNS"
SRV_PID="$(cat "$HERE/_server_${PORT}.pid")"
cleanup() { echo "[greedy2x2] stopping server pid=$SRV_PID"; kill "$SRV_PID" 2>/dev/null || true; sleep 3; kill -9 "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

run_cell () {
  local budget="$1"
  local out="$RES/${BODY}_greedy_${budget}.json"
  if [[ -s "$out" ]]; then echo "[greedy2x2] SKIP existing $out"; return 0; fi
  echo "[greedy2x2] $(date -u +%H:%M:%SZ) body=$BODY budget=$budget years=$YEARS -> $out"
  local t0=$(date +%s)
  "$PY" "$AIME" \
    --base-url "http://127.0.0.1:${PORT}" \
    --model gemma-4-e4b-it \
    --years "$YEARS" \
    --k "$K" --seed "$SEED" \
    --temperature 0.0 --top-p 1.0 --top-k -1 \
    --max-tokens "$budget" --min-tokens 8 \
    --no-thinking \
    --client-concurrency "$CC" \
    --save-text \
    --label "${BODY}_greedy_${budget}" \
    --out "$out" 2>&1 | tail -6
  echo "[greedy2x2] cell done in $(( $(date +%s) - t0 ))s -> $out"
}

for b in $BUDGETS; do run_cell "$b"; done
echo "[greedy2x2] DONE body=$BODY $(date -u +%H:%M:%SZ)"
