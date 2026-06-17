#!/usr/bin/env bash
# PR #579: run MMLU-Pro + GPQA-Diamond for int4_g128_lmhead against the already-
# running serve.sh server (one task, both protocols), matching #563's harness
# byte-for-byte (run_eval.py, dataset seed 12345, MMLU n=500) with the PR-mandated
# min_tokens=8 EOS-guard added throughout. Writes one JSON per (task, regime, seed)
# into results/. Idempotent: an existing cell is skipped.
set -euo pipefail

TASK="$1"                      # mmlu_pro | gpqa_diamond
MT="$2"                        # max_tokens (mmlu_pro 2048, gpqa 3072)
SEEDS="${3:-0 1 2}"            # sampling seeds
PORT="${4:-8000}"

HERE="$(cd "$(dirname "$0")" && pwd)"
# PR #610: optional output-dir override so the de-biased 6144 sweep can write to a
# separate dir (results_6144/) without clobbering the committed 4096 baseline that
# the debias-delta + truncation audit compare against. Defaults to results/ (#579).
RES="${RES_DIR:-$HERE/results}"; mkdir -p "$RES"
EVAL_PY="/workspace/senpai/target/.venv/bin/python"
RUN_EVAL="/workspace/senpai/target/research/validity/downstream_quality_eval/run_eval.py"
BASE_URL="http://127.0.0.1:${PORT}/v1"
DATASET_SEED=12345
MMLU_N=500
ARM=int4g128

run_one () { # out extra... -> writes $RES/$out
  local out="$1"; shift
  if [[ -s "$RES/$out" ]]; then echo "[run_mc] $(date -u +%H:%M:%SZ) SKIP existing $out"; return 0; fi
  echo "[run_mc] $(date -u +%H:%M:%SZ) task=$TASK -> $out"
  "$EVAL_PY" "$RUN_EVAL" \
    --task "$TASK" --arm "$ARM" --out "$RES/$out" \
    --seed "$DATASET_SEED" --n "$MMLU_N" --max-tokens "$MT" \
    --base-url "$BASE_URL" --model gemma-4-e4b-it \
    --max-connections 16 --min-tokens 8 "$@" 2>&1 | tail -3
}

# GREEDY anchor (1 run; deterministic) -- context for the sampled/greedy delta.
run_one "${ARM}_${TASK}_greedy.json"

# SAMPLED, lewtun-#31 mandated protocol, one run per seed (the gate number).
for s in $SEEDS; do
  run_one "${ARM}_${TASK}_sampled_s${s}.json" \
    --temperature 1.0 --top-p 0.95 --top-k 64 --sampling-seed "$s"
done

echo "[run_mc] DONE task=$TASK $(date -u +%H:%M:%SZ)"
