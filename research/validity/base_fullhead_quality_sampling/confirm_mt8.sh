#!/usr/bin/env bash
# PR #563 confirmatory: the advisor's 08:00 reminder names a mandatory min_tokens=8
# EOS-guard (wirbel #541). The full sampled sweep ran WITHOUT it, but empty_rate=0
# across all 16 cells means the guard (which only masks EOS for the first 8 tokens to
# rescue immediate-EOS empties) cannot have changed any cell. This re-runs the seed-0
# MMLU-Pro + GPQA cells on BOTH arms WITH --min-tokens 8 to prove accuracy is unchanged
# (guard is a verified no-op for these CoT tasks). Writes *_mt8.json beside the originals.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/results"
EVAL_PY="/workspace/senpai/target/.venv/bin/python"
RUN_EVAL="/workspace/senpai/target/research/validity/downstream_quality_eval/run_eval.py"
PORT=8000
BASE_URL="http://127.0.0.1:${PORT}/v1"

run_cell () { # arm task max_tokens out
  local arm="$1" task="$2" mt="$3" out="$4"
  echo "[confirm] $(date -u +%H:%M:%SZ) arm=$arm task=$task min_tokens=8 -> $out"
  "$EVAL_PY" "$RUN_EVAL" --task "$task" --arm "$arm" --out "$RES/$out" \
    --seed 12345 --n 500 --max-tokens "$mt" \
    --base-url "$BASE_URL" --model gemma-4-e4b-it --max-connections 16 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --sampling-seed 0 --min-tokens 8 2>&1 | tail -3
}

stop_server () { # pidfile
  local pf="$1"
  [[ -f "$pf" ]] || return 0
  local pid; pid="$(cat "$pf")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[confirm] stopping server pid=$pid"
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  fi
}

# --- fp16 arm (assumed already serving on :8000) ---
echo "[confirm] === fp16 arm (already up) ==="
run_cell fp16 mmlu_pro     2048 fp16_mmlu_pro_sampled_s0_mt8.json
run_cell fp16 gpqa_diamond 3072 fp16_gpqa_sampled_s0_mt8.json

# --- swap fp16 -> int4 ---
echo "[confirm] === swap to int4 arm ==="
stop_server "$HERE/_server_fp16.pid"
sleep 3
ENFORCE_EAGER=0 "$HERE/serve.sh" int4 "$PORT"
if [[ $? -ne 0 ]]; then echo "[confirm] FATAL: int4 server failed to come up"; exit 1; fi

run_cell int4 mmlu_pro     2048 int4_mmlu_pro_sampled_s0_mt8.json
run_cell int4 gpqa_diamond 3072 int4_gpqa_sampled_s0_mt8.json

echo "[confirm] DONE $(date -u +%H:%M:%SZ)"
