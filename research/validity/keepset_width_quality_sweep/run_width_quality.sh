#!/usr/bin/env bash
# Keepset-width quality sweep driver: for each width K, serve the baked osoi5
# 37L-int4 checkpoint through the REAL pck04 ship path (start_server.sh ship),
# 5-question self-test, then MMLU-Pro n=500 + GPQA-Diamond 198 (greedy, byte-
# identical prompts, reused #511 run_eval.py). ONE GPU -> widths run SEQUENTIALLY.
#
# Uniform serve path across the whole sweep: every K (incl. full) goes through the
# pck04 ParallelLMHead(K)+scatter, so the only variable is the lm_head keepset width.
# full_pck04 is a cross-check against the plain-vLLM full result (results/full/).
set -uo pipefail

HERE="/workspace/senpai/target/research/validity/keepset_width_quality_sweep"
QE="/workspace/senpai/target/research/validity/downstream_quality_eval"
VENV=/tmp/eval-serve-venv
PY="$VENV/bin/python"
PORT=8000
MASTER="$HERE/results/_sweep.log"
mkdir -p "$HERE/results"

log(){ echo "[sweep $(date -u +%H:%M:%SZ)] $*" | tee -a "$MASTER"; }

stop_server(){
  # kill the ship-arm server (pidfile) + any lingering vllm on this venv, wait for GPU free
  local pf="$QE/_server_ship.pid"
  if [[ -f "$pf" ]]; then kill "$(cat "$pf")" 2>/dev/null || true; fi
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then break; fi
    sleep 2
  done
  sleep 3
}

# width  model_dir                 keepset_json
WIDTHS=(
  "full_pck04 /tmp/osoi5-full-baked /tmp/osoi5-full-baked/pck04_keepset.json"
  "32k        /tmp/osoi5-32k-baked  /tmp/osoi5-32k-baked/pck04_keepset.json"
  "16k        /tmp/osoi5-v0-baked   /tmp/osoi5-v0-baked/pck04_keepset.json"
  "12k        /tmp/osoi5-12k-baked  /tmp/osoi5-12k-baked/pck04_keepset.json"
)

log "=== keepset-width quality sweep START (widths: full_pck04 32k 16k 12k) ==="
for row in "${WIDTHS[@]}"; do
  read -r W MODEL KEEP <<<"$row"
  RDIR="$HERE/results/$W"; mkdir -p "$RDIR"
  log "----- width=$W model=$MODEL keep=$KEEP -----"

  stop_server
  log "$W: starting pck04 ship server ..."
  if ! bash "$QE/start_server.sh" ship "$MODEL" "$KEEP" >"$RDIR/_server_start.out" 2>&1; then
    log "$W: SERVER START FAILED -> skip width"; tail -20 "$RDIR/_server_start.out" | tee -a "$MASTER"; continue
  fi
  log "$W: server ready"

  # ---- 5-question self-test (server responds + parseable answers) ----
  if $PY "$QE/run_eval.py" --task mmlu_pro --arm ship --out "$RDIR/_smoke.json" \
        --n 5 --seed 12345 --max-tokens 2048 --max-connections 5 \
        --base-url "http://127.0.0.1:$PORT/v1" --model gemma-4-e4b-it >"$RDIR/_smoke.out" 2>&1; then
    sc=$($PY -c "import json;print(json.load(open('$RDIR/_smoke.json'))['n_scored'])" 2>/dev/null || echo 0)
    log "$W: SMOKE ok (scored=$sc/5)"
  else
    log "$W: SMOKE FAILED -> skip width"; tail -20 "$RDIR/_smoke.out" | tee -a "$MASTER"; stop_server; continue
  fi

  # ---- MMLU-Pro n=500 ----
  log "$W: MMLU-Pro n=500 START"
  $PY "$QE/run_eval.py" --task mmlu_pro --arm ship --out "$RDIR/ship_mmlu_pro.json" \
      --n 500 --seed 12345 --max-tokens 2048 --max-connections 16 \
      --base-url "http://127.0.0.1:$PORT/v1" --model gemma-4-e4b-it >"$RDIR/_mmlu.out" 2>&1
  ma=$($PY -c "import json;d=json.load(open('$RDIR/ship_mmlu_pro.json'));print(f\"{d['accuracy']:.4f} scored={d['n_scored']} err={d['n_error']}\")" 2>/dev/null || echo "FAIL")
  log "$W: MMLU-Pro acc=$ma"

  # ---- GPQA-Diamond 198 ----
  log "$W: GPQA-Diamond 198 START"
  $PY "$QE/run_eval.py" --task gpqa_diamond --arm ship --out "$RDIR/ship_gpqa.json" \
      --seed 12345 --max-tokens 3072 --max-connections 16 \
      --base-url "http://127.0.0.1:$PORT/v1" --model gemma-4-e4b-it >"$RDIR/_gpqa.out" 2>&1
  ga=$($PY -c "import json;d=json.load(open('$RDIR/ship_gpqa.json'));print(f\"{d['accuracy']:.4f} scored={d['n_scored']} err={d['n_error']}\")" 2>/dev/null || echo "FAIL")
  log "$W: GPQA-Diamond acc=$ga"
  log "$W: DONE  MMLU=$ma | GPQA=$ga"
done

stop_server
log "=== keepset-width quality sweep COMPLETE ==="
