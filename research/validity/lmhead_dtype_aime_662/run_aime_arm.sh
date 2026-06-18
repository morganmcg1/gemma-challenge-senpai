#!/usr/bin/env bash
# PR #662 -- end-to-end AIME(n=60) + decode-TPS proxy for ONE head-dtype arm, on the
# byte-identical #653/#639 serve recipe (MAX_NUM_SEQS=1, BI=1, FLASHINFER_SAMPLER=0,
# vLLM 0.22.0). Serves -> AIME -> TPS -> tears the server down -> writes a done marker.
# Only the lm_head dtype (baked into MODEL_DIR) varies across arms.
#
#   ARM=<name> MODEL_DIR=<checkpoint dir> bash run_aime_arm.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV=/tmp/vllm0220-srv
ARM="${ARM:?set ARM=<name>}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR=<checkpoint dir>}"
PORT="${PORT:-8000}"
MARKER="$HERE/_done_${ARM}.marker"
rm -f "$MARKER"

echo "[arm $ARM] START $(date -u +%FT%TZ)"

# 1) serve (blocks until ready, writes _server_<arm>.pid)
ARM="$ARM" MODEL_DIR="$MODEL_DIR" PORT="$PORT" bash "$HERE/serve_arm.sh"
if [[ $? -ne 0 ]]; then echo "[arm $ARM] SERVE FAILED"; echo "serve_failed" > "$MARKER"; exit 1; fi
SRV_PID="$(cat "$HERE/_server_${ARM}.pid")"

# 2) AIME n=60 maj@1 greedy, identical to the live bf16head run
"$VENV/bin/python" "$ROOT/research/downstream_quality_aime/aime_eval.py" \
  --base-url "http://127.0.0.1:${PORT}" --model gemma-4-e4b-it \
  --years 2024,2025 --k 1 --temperature 0 --top-p 1.0 --top-k -1 \
  --max-tokens 6144 --min-tokens 8 --no-thinking --seed 1234 --save-text \
  --label "$ARM" --out "$HERE/results/aime_${ARM}.json" \
  > "$HERE/_aime_${ARM}.log" 2>&1
AIME_RC=$?
echo "[arm $ARM] AIME rc=$AIME_RC $(date -u +%FT%TZ)"

# 3) decode-TPS proxy (single-stream), same server
"$VENV/bin/python" "$HERE/tps_proxy.py" --base-url "http://127.0.0.1:${PORT}" \
  --model gemma-4-e4b-it --arm "$ARM" --out "$HERE/results/tps_${ARM}.json" \
  > "$HERE/_tps_${ARM}.log" 2>&1
TPS_RC=$?
echo "[arm $ARM] TPS rc=$TPS_RC $(date -u +%FT%TZ)"

# 4) tear the server down so the next arm gets the GPU
kill "$SRV_PID" 2>/dev/null || true
for i in $(seq 1 20); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
kill -9 "$SRV_PID" 2>/dev/null || true

echo "aime_rc=$AIME_RC tps_rc=$TPS_RC $(date -u +%FT%TZ)" > "$MARKER"
echo "[arm $ARM] DONE $(date -u +%FT%TZ)"
