#!/usr/bin/env bash
# Reproduce the official summary.json:tps protocol locally, byte-for-byte on
# args, against an already-running served endpoint. Mirrors
# official/.../hf_bucket_single_job.py::run_benchmark (NUM_PROMPTS=128,
# OUTPUT_LEN=512, MAX_CONCURRENCY=1, REQUEST_RATE=inf, WARMUP=4, SEED=1,
# ignore_eos, vllm-chat backend, temperature=0 greedy).
#
# Usage: run_bench.sh <PORT> <output_jsonl>
set -euo pipefail
PORT="${1:-8000}"; OUT="${2:?need output jsonl}"
BENCH=/tmp/bench-venv/bin/python
DATA=/workspace/senpai/target/official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json

"$BENCH" -m sglang.bench_serving \
  --backend vllm-chat \
  --base-url "http://127.0.0.1:${PORT}" \
  --model gemma-4-e4b-it \
  --tokenizer google/gemma-4-E4B-it \
  --dataset-name sharegpt \
  --dataset-path "$DATA" \
  --sharegpt-output-len 512 \
  --num-prompts 128 \
  --max-concurrency 1 \
  --request-rate inf \
  --warmup-requests 4 \
  --seed 1 \
  --extra-request-body '{"ignore_eos": true}' \
  --output-file "$OUT" \
  --output-details \
  --disable-stream \
  --disable-tqdm
