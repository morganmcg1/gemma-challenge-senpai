#!/usr/bin/env bash
# PR #699: for ONE body, serve once then eval BOTH token budgets (gate + high) against
# the SAME warmed instance on the #31 sampled basis. The only varied axis between the
# two output JSONs is the request max_tokens. Writes results/<body>_<budget>.json with
# per-item finish_reasons (the truncation mechanism).
set -euo pipefail

BODY="${1:?body label: int4 | base}"
MODEL="${2:?model dir or HF id}"
PORT="${3:-8000}"
K="${K:-16}"                      # maj@k samples per problem (sampled-basis draws)
SEED="${SEED:-1234}"
GATE="${GATE:-6144}"
HIGH="${HIGH:-12288}"
MML="${MML:-13312}"
MNS="${MNS:-16}"                  # HOLD FIXED across both budgets
CC="${CC:-1}"                     # client concurrency (1 = sequential problems; k samples batch within a request)
YEARS="${YEARS:-2024,2025-I,2025-II}"   # n=60, EXACT banked-anchor basis (base 0.4667 / int4 0.350 @gb6144 greedy)
LIMIT_ARG=""; [[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit ${LIMIT}"

HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/results"; mkdir -p "$RES"
PY="/workspace/senpai/target/.venv/bin/python"
AIME="/workspace/senpai/target/research/downstream_quality_aime/aime_eval.py"

echo "[sweep] BODY=$BODY MODEL=$MODEL K=$K seed=$SEED budgets={$GATE,$HIGH} mml=$MML mns=$MNS cc=$CC years=$YEARS $(date -u +%H:%M:%SZ)"

# --- serve once ---
bash "$HERE/serve_body.sh" "$MODEL" "$PORT" "$MML" "$MNS"
SRV_PID="$(cat "$HERE/_server_${PORT}.pid")"
cleanup() { echo "[sweep] stopping server pid=$SRV_PID"; kill "$SRV_PID" 2>/dev/null || true; sleep 3; kill -9 "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

run_budget () {
  local budget="$1"
  local out="$RES/${BODY}_${budget}.json"
  if [[ -s "$out" ]]; then echo "[sweep] SKIP existing $out"; return 0; fi
  echo "[sweep] $(date -u +%H:%M:%SZ) body=$BODY budget=$budget -> $out"
  "$PY" "$AIME" \
    --base-url "http://127.0.0.1:${PORT}" \
    --model gemma-4-e4b-it \
    --years "$YEARS" $LIMIT_ARG \
    --k "$K" --seed "$SEED" \
    --temperature 1.0 --top-p 0.95 --top-k 64 \
    --max-tokens "$budget" --min-tokens 8 \
    --no-thinking \
    --client-concurrency "$CC" \
    --label "${BODY}_${budget}" \
    --save-text \
    --out "$out" 2>&1 | tail -6
}

run_budget "$GATE"
run_budget "$HIGH"
echo "[sweep] DONE body=$BODY $(date -u +%H:%M:%SZ)"
