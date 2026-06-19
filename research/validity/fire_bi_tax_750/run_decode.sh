#!/usr/bin/env bash
# Capture greedy completion_token_ids from an already-running served endpoint
# using the OFFICIAL decode_outputs.py verbatim (the organizer-side identity/audit
# protocol): /v1/completions, integer-token chat-templated prompt, temperature 0,
# max_tokens 512, add_special_tokens false, ignore_eos true, return_token_ids true.
# 128 prompts, seed 1 — identical prompt set & order to the official decode pass.
#
# Usage: run_decode.sh <PORT> <out_jsonl> <summary_json> [num_prompts]
set -euo pipefail
PORT="${1:-8000}"; OUT="${2:?need out jsonl}"; SUMM="${3:?need summary json}"; NP="${4:-128}"
PY=/tmp/bench-venv/bin/python   # has transformers (tokenizer only; no torch needed)
DEC=/workspace/senpai/target/official/main_bucket/shared_resources/speed_benchmark/scripts/decode_outputs.py
DATA=/workspace/senpai/target/official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json

"$PY" "$DEC" \
  --base-url "http://127.0.0.1:${PORT}" \
  --model gemma-4-e4b-it \
  --tokenizer google/gemma-4-E4B-it \
  --dataset-path "$DATA" \
  --output-file "$OUT" \
  --summary-file "$SUMM" \
  --num-prompts "$NP" \
  --output-len 512 \
  --seed 1 \
  --request-timeout-s 600
