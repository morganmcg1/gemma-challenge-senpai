#!/usr/bin/env bash
# PR #563: run the downstream-quality evals for ONE arm against the already-running
# serve.sh server on :PORT. Measures BOTH protocols on the same stack:
#   - GREEDY (1 run; deterministic)  -> within-harness greedy anchor for the delta
#   - SAMPLED (>=3 seeds; temp=1.0 top_p=0.95 top_k=64) -> the lewtun #31 protocol
# for BOTH MMLU-Pro (n=500) and GPQA-Diamond (full 198). Dataset seed fixed at the
# #547 value (12345) so prompts are byte-identical across arms/seeds/#547.
set -euo pipefail

ARM="$1"                       # int4 | fp16   (label only; server must already serve this arm)
SEEDS="${2:-0 1 2}"            # sampling-seeds
PORT="${3:-8000}"

HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/results"; mkdir -p "$RES"
EVAL_PY="${EVAL_PY:-/workspace/senpai/target/.venv/bin/python}"
RUN_EVAL="/workspace/senpai/target/research/validity/downstream_quality_eval/run_eval.py"
BASE_URL="http://127.0.0.1:${PORT}/v1"
DATASET_SEED=12345
MMLU_N=500

run_one () { # task max_tokens extra_decode_args... -> writes $RES/$OUT
  local task="$1"; local mt="$2"; local out="$3"; shift 3
  # Idempotent: skip a cell already written, so the greedy gate (SEEDS="") and the
  # later sampled sweep (SEEDS="0 1 2") compose without re-running the greedy anchor.
  if [[ -s "$RES/$out" ]]; then echo "[run_arm] $(date -u +%H:%M:%SZ) arm=$ARM SKIP existing $out"; return 0; fi
  echo "[run_arm] $(date -u +%H:%M:%SZ) arm=$ARM -> $out"
  "$EVAL_PY" "$RUN_EVAL" \
    --task "$task" --arm "$ARM" --out "$RES/$out" \
    --seed "$DATASET_SEED" --n "$MMLU_N" --max-tokens "$mt" \
    --base-url "$BASE_URL" --model gemma-4-e4b-it \
    --max-connections 16 "$@" 2>&1 | tail -3
}

# --- GREEDY anchor (1 run each) ---
run_one mmlu_pro     2048 "${ARM}_mmlu_pro_greedy.json"
run_one gpqa_diamond 3072 "${ARM}_gpqa_greedy.json"

# --- SAMPLED, mandated protocol, one run per seed ---
for s in $SEEDS; do
  run_one mmlu_pro     2048 "${ARM}_mmlu_pro_sampled_s${s}.json" \
    --temperature 1.0 --top-p 0.95 --top-k 64 --sampling-seed "$s"
  run_one gpqa_diamond 3072 "${ARM}_gpqa_sampled_s${s}.json" \
    --temperature 1.0 --top-p 0.95 --top-k 64 --sampling-seed "$s"
done

echo "[run_arm] DONE arm=$ARM $(date -u +%H:%M:%SZ)"
