#!/usr/bin/env bash
# PR #503 chained runner: AR cross-session control + private-Δ easy/hard.
# Self-gates on the public-screen probe (PID arg) exiting so the single GPU is
# never double-booked. Each probe starts/stops its own server; all resumable.
set -u
cd /workspace/senpai/target

PUB_PID="${1:?need public-screen probe PID}"
D=research/validity/ngram_spec_dec
PY=/usr/bin/python3
EASY=research/validity/private_attention_flip_bound/shifted_reasoning_stem.jsonl
HARD=research/validity/private_attention_flip_bound/shifted_hard_ood.jsonl

echo "[run_remaining] waiting for public-screen probe PID $PUB_PID to exit..."
while kill -0 "$PUB_PID" 2>/dev/null; do sleep 10; done
echo "[run_remaining] public screen done at $(date -u +%H:%M:%SZ); starting remaining legs"

run() {  # name configs source out
  echo "[run_remaining] === $1 ($(date -u +%H:%M:%SZ)) ==="
  "$PY" "$D/probe.py" --configs-json "$D/$2" --prompt-source "$3" \
    --num-prompts 32 --output-len 256 --out "$D/$4" --resume
  echo "[run_remaining] $1 rc=$? ($(date -u +%H:%M:%SZ))"
}

run ctrl_ar2     configs_ctrl_ar2.json "public" results_ctrl_ar2.json
run private_easy configs_private.json  "$EASY"  results_private_easy.json
run private_hard configs_private.json  "$HARD"  results_private_hard.json

echo "[run_remaining] ALL REMAINING LEGS DONE $(date -u +%H:%M:%SZ)"
touch "$D/_remaining.done"
