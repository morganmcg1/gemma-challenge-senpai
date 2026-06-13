#!/usr/bin/env bash
# Focused re-check of the served int4 base modality gate (text/image/audio),
# now that the local .venv has the vllm[audio] decode backend (soundfile/av).
# Starts serve.py with both levers OFF, waits for /v1/models, runs the modality
# smoke only, then shuts the server down. The full decode/PPL gate already
# passed (serve_runs/base): this only closes the earlier audio:false record.
set -u
ROOT=/workspace/senpai/target
SUB="$ROOT/submissions/fa2sw_onegraph"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/research/fa2sw_onegraph/serve_runs/base_modality_recheck"
PORT=8000
BASEURL="http://127.0.0.1:$PORT"
MODEL=gemma-4-e4b-it

mkdir -p "$OUT"
LOG="$OUT/recheck.log"
: > "$LOG"
echo "=== [base recheck] FA2SW=0 ONEGRAPH=0 start $(date -u +%H:%M:%S) ===" | tee -a "$LOG"

cd "$SUB" || exit 9
CUDA_VISIBLE_DEVICES=0 \
  FA2SW=0 ONEGRAPH=0 \
  QUANTIZATION=compressed-tensors MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.90 \
  MAX_NUM_BATCHED_TOKENS=512 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  BACKEND_MAP_OUT="$OUT/backend_map.json" HOST=127.0.0.1 PORT="$PORT" \
  "$PY" serve.py >> "$OUT/server.log" 2>&1 &
SPID=$!
echo "server pid=$SPID" | tee -a "$LOG"

# wait for readiness (aborts if server dies)
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
  echo "[base recheck] NOT READY (rc=$READY_RC) -> abort" | tee -a "$LOG"
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
  echo "=== [base recheck] ABORTED $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
  exit 11
fi

echo "--- [base recheck] modality smoke ---" | tee -a "$LOG"
BASE_URL="$BASEURL" SERVED_MODEL_NAME="$MODEL" \
  "$PY" "$ROOT/research/fa2sw_onegraph/modality_smoke.py" >> "$LOG" 2>&1
echo "[base recheck] modality rc=$?" | tee -a "$LOG"

kill "$SPID" 2>/dev/null
for _ in $(seq 1 20); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
kill -9 "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo "=== [base recheck] DONE $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
