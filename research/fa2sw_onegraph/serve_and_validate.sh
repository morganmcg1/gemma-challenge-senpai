#!/usr/bin/env bash
# Authoritative server-based validation for one fa2sw/onegraph lever config.
# Starts serve.py, waits for /v1/models, runs modality smoke + the official
# decode_outputs.py (greedy capture) + ppl_endpoint.py, then shuts the server.
#
# Usage: serve_and_validate.sh <label> <FA2SW> <ONEGRAPH> [output_len]
set -u

LABEL="${1:?label}"; FA2SW_V="${2:?FA2SW}"; ONEGRAPH_V="${3:?ONEGRAPH}"; OUTLEN="${4:-512}"
ROOT=/workspace/senpai/target
SUB="$ROOT/submissions/fa2sw_onegraph"
SR="$ROOT/official/main_bucket/shared_resources/speed_benchmark"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/research/fa2sw_onegraph/serve_runs/$LABEL"
PORT=8000
BASEURL="http://127.0.0.1:$PORT"
MODEL=gemma-4-e4b-it

mkdir -p "$OUT"
LOG="$OUT/validate.log"
: > "$LOG"
echo "=== [$LABEL] FA2SW=$FA2SW_V ONEGRAPH=$ONEGRAPH_V OUTLEN=$OUTLEN start $(date -u +%H:%M:%S) ===" | tee -a "$LOG"

cd "$SUB" || exit 9
CUDA_VISIBLE_DEVICES=0 \
  FA2SW="$FA2SW_V" ONEGRAPH="$ONEGRAPH_V" \
  QUANTIZATION=compressed-tensors MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.90 \
  MAX_NUM_BATCHED_TOKENS=512 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  BACKEND_MAP_OUT="$OUT/backend_map.json" HOST=127.0.0.1 PORT="$PORT" \
  "$PY" serve.py >> "$OUT/server.log" 2>&1 &
SPID=$!
echo "server pid=$SPID" | tee -a "$LOG"

# --- wait for readiness (single long-lived poller; aborts if server dies) ---
"$PY" - "$BASEURL" "$MODEL" "$SPID" <<'PYEOF' | tee -a "$LOG"
import sys, os, time, json, urllib.request
base, model, spid = sys.argv[1], sys.argv[2], int(sys.argv[3])
deadline = time.time() + 420
while time.time() < deadline:
    try:
        os.kill(spid, 0)
    except OSError:
        print("STARTUP server_died"); sys.exit(2)
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
            data = json.loads(r.read().decode())
        if any(m.get("id") == model for m in data.get("data", [])):
            print("STARTUP ready"); sys.exit(0)
    except Exception:
        pass
    time.sleep(3)
print("STARTUP timeout"); sys.exit(3)
PYEOF
READY_RC=${PIPESTATUS[0]}

if [ "$READY_RC" != "0" ]; then
  echo "[$LABEL] NOT READY (rc=$READY_RC) -> abort" | tee -a "$LOG"
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
  echo "=== [$LABEL] ABORTED $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
  exit 11
fi

# --- modality smoke (text/image/audio + /v1/models) ---
echo "--- [$LABEL] modality smoke ---" | tee -a "$LOG"
BASE_URL="$BASEURL" SERVED_MODEL_NAME="$MODEL" \
  "$PY" "$ROOT/research/fa2sw_onegraph/modality_smoke.py" >> "$LOG" 2>&1
echo "[$LABEL] modality rc=$?" | tee -a "$LOG"

# --- greedy capture on the OFFICIAL ShareGPT prompt set ---
echo "--- [$LABEL] decode_outputs (official ShareGPT, outlen=$OUTLEN) ---" | tee -a "$LOG"
"$PY" "$SR/scripts/decode_outputs.py" \
  --base-url "$BASEURL" --model "$MODEL" \
  --dataset-path "$SR/data/eval_prompts_sharegpt.json" \
  --output-file "$OUT/decode_outputs.jsonl" --summary-file "$OUT/decode_summary.json" \
  --num-prompts 128 --output-len "$OUTLEN" --seed 1 >> "$LOG" 2>&1
echo "[$LABEL] decode rc=$?" | tee -a "$LOG"

# --- served PPL (prefill path) on the ground-truth tokens ---
echo "--- [$LABEL] served PPL ---" | tee -a "$LOG"
"$PY" "$SR/scripts/ppl_endpoint.py" \
  --base-url "$BASEURL" --model "$MODEL" \
  --dataset-path "$SR/data/ppl_ground_truth_tokens.jsonl" \
  --output-file "$OUT/ppl.jsonl" --summary-file "$OUT/ppl_summary.json" >> "$LOG" 2>&1
echo "[$LABEL] ppl rc=$?" | tee -a "$LOG"

# --- greedy-identity self-check: this run's served greedy vs the plain int4
#     base reference (REQUIRED leaderboard validity rule). REF_DECODE defaults to
#     the base served capture; skip if comparing base against itself's source. ---
REF_DECODE="${REF_DECODE:-$ROOT/research/fa2sw_onegraph/serve_runs/base/decode_outputs.jsonl}"
VERIFY="$ROOT/official/main_bucket/shared_resources/gemma_greedy_identity_verifier_flowian-powers/check_greedy_identity.py"
if [ -f "$REF_DECODE" ] && [ "$REF_DECODE" != "$OUT/decode_outputs.jsonl" ]; then
  echo "--- [$LABEL] greedy-identity vs $REF_DECODE ---" | tee -a "$LOG"
  "$PY" "$VERIFY" --reference "$REF_DECODE" --candidate "$OUT/decode_outputs.jsonl" >> "$LOG" 2>&1
  echo "[$LABEL] greedy-identity exit=$? (0=GREEDY_IDENTICAL 1=DIVERGENT 2=INCOMPARABLE)" | tee -a "$LOG"
fi

# --- shutdown ---
kill "$SPID" 2>/dev/null
for _ in $(seq 1 20); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
kill -9 "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo "=== [$LABEL] DONE $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
