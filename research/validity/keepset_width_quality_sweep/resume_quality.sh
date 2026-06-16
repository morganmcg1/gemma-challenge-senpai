#!/usr/bin/env bash
# Resume the keepset-width QUALITY sweep after the prior driver was killed at a
# session boundary mid-16k-GPQA. The 16k vLLM server (pck04 ship, /tmp/osoi5-v0-baked)
# is still live and healthy, so we finish 16k GPQA against it, then run the full 12k
# leg (start server -> 5q smoke -> MMLU-Pro n=500 -> GPQA-Diamond 198). Same run_eval.py
# invocations as run_width_quality.sh so every width stays apples-to-apples.
set -uo pipefail

HERE="/workspace/senpai/target/research/validity/keepset_width_quality_sweep"
QE="/workspace/senpai/target/research/validity/downstream_quality_eval"
VENV=/tmp/eval-serve-venv
PY="$VENV/bin/python"
PORT=8000
MASTER="$HERE/results/_sweep.log"

log(){ echo "[resume $(date -u +%H:%M:%SZ)] $*" | tee -a "$MASTER"; }

stop_server(){
  local pf="$QE/_server_ship.pid"
  if [[ -f "$pf" ]]; then kill "$(cat "$pf")" 2>/dev/null || true; fi
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then break; fi
    sleep 2
  done
  sleep 3
}

acc_of(){ $PY -c "import json;d=json.load(open('$1'));print(f\"{d['accuracy']:.4f} scored={d['n_scored']} err={d['n_error']}\")" 2>/dev/null || echo FAIL; }

log "=== RESUME: finish 16k GPQA on live server, then 12k leg ==="

# ---- 16k GPQA on the already-live 16k server (PID per _server_ship.pid) ----
if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
  RDIR="$HERE/results/16k"; mkdir -p "$RDIR"
  log "16k: live server OK -> GPQA-Diamond 198 START"
  $PY "$QE/run_eval.py" --task gpqa_diamond --arm ship --out "$RDIR/ship_gpqa.json" \
      --seed 12345 --max-tokens 3072 --max-connections 16 \
      --base-url "http://127.0.0.1:$PORT/v1" --model gemma-4-e4b-it >"$RDIR/_gpqa.out" 2>&1
  log "16k: GPQA-Diamond acc=$(acc_of "$RDIR/ship_gpqa.json")"
else
  log "16k: live server NOT responding -> will restart in 16k block"
  stop_server
  bash "$QE/start_server.sh" ship /tmp/osoi5-v0-baked /tmp/osoi5-v0-baked/pck04_keepset.json >"$HERE/results/16k/_server_restart.out" 2>&1 \
    && { RDIR="$HERE/results/16k"; \
         $PY "$QE/run_eval.py" --task gpqa_diamond --arm ship --out "$RDIR/ship_gpqa.json" --seed 12345 --max-tokens 3072 --max-connections 16 --base-url "http://127.0.0.1:$PORT/v1" --model gemma-4-e4b-it >"$RDIR/_gpqa.out" 2>&1; \
         log "16k: GPQA-Diamond acc=$(acc_of "$RDIR/ship_gpqa.json")"; } \
    || log "16k: server restart FAILED"
fi

# ---- 12k leg: clean server start -> smoke -> MMLU -> GPQA ----
W=12k; MODEL=/tmp/osoi5-12k-baked; KEEP=/tmp/osoi5-12k-baked/pck04_keepset.json
RDIR="$HERE/results/$W"; mkdir -p "$RDIR"
log "----- width=$W model=$MODEL keep=$KEEP -----"
stop_server
log "$W: starting pck04 ship server ..."
if ! bash "$QE/start_server.sh" ship "$MODEL" "$KEEP" >"$RDIR/_server_start.out" 2>&1; then
  log "$W: SERVER START FAILED"; tail -20 "$RDIR/_server_start.out" | tee -a "$MASTER"; stop_server; exit 1
fi
log "$W: server ready"

if $PY "$QE/run_eval.py" --task mmlu_pro --arm ship --out "$RDIR/_smoke.json" \
      --n 5 --seed 12345 --max-tokens 2048 --max-connections 5 \
      --base-url "http://127.0.0.1:$PORT/v1" --model gemma-4-e4b-it >"$RDIR/_smoke.out" 2>&1; then
  sc=$($PY -c "import json;print(json.load(open('$RDIR/_smoke.json'))['n_scored'])" 2>/dev/null || echo 0)
  log "$W: SMOKE ok (scored=$sc/5)"
else
  log "$W: SMOKE FAILED"; tail -20 "$RDIR/_smoke.out" | tee -a "$MASTER"; stop_server; exit 1
fi

log "$W: MMLU-Pro n=500 START"
$PY "$QE/run_eval.py" --task mmlu_pro --arm ship --out "$RDIR/ship_mmlu_pro.json" \
    --n 500 --seed 12345 --max-tokens 2048 --max-connections 16 \
    --base-url "http://127.0.0.1:$PORT/v1" --model gemma-4-e4b-it >"$RDIR/_mmlu.out" 2>&1
log "$W: MMLU-Pro acc=$(acc_of "$RDIR/ship_mmlu_pro.json")"

log "$W: GPQA-Diamond 198 START"
$PY "$QE/run_eval.py" --task gpqa_diamond --arm ship --out "$RDIR/ship_gpqa.json" \
    --seed 12345 --max-tokens 3072 --max-connections 16 \
    --base-url "http://127.0.0.1:$PORT/v1" --model gemma-4-e4b-it >"$RDIR/_gpqa.out" 2>&1
log "$W: GPQA-Diamond acc=$(acc_of "$RDIR/ship_gpqa.json")"

stop_server
log "=== RESUME quality COMPLETE (16k GPQA + 12k full leg done) ==="
