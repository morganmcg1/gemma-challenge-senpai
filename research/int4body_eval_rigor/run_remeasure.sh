#!/usr/bin/env bash
# Re-measure the SHIPPED int4_g128_lmhead (g128 body + int4 lm_head) on
# AIME-2024 (maj@k) and gpqa_diamond under three decode bases:
#   (a) greedy            (the basis the binding gates currently rest on)
#   (b) #31 sampling      (generation_config.json: T=1.0/top_p=0.95/top_k=64)
#   (c) #31 sampling + min_tokens=8  (triad #541 EOS-guard)
# All against the already-running deterministic vllm-0.22.0 endpoint.
# Summaries -> tracked root (KB-scale). Verbose logs/.eval -> gitignored out/.
set -u
ROOT=/workspace/senpai/target
WD="$ROOT/research/int4body_eval_rigor"
LOG="$WD/out/remeasure.log"
INSPECT_LOGS="$WD/out/logs"
# aime_eval.py appends /v1/chat/completions itself -> bare host.
# run_eval.py (inspect_ai / OpenAI client) wants the /v1 suffix.
AIME_URL="http://127.0.0.1:8000"
GPQA_URL="http://127.0.0.1:8000/v1"
MODEL="gemma-4-e4b-it"
PYV="$ROOT/.venv/bin/python"          # AIME client (urllib only)
GPQAV="/tmp/eval-serve-venv/bin/python" # gpqa client (inspect_ai)
mkdir -p "$INSPECT_LOGS"

ts() { date +%H:%M:%S; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

run_aime() {
  local label="$1"; shift
  say "START aime/$label : $*"
  "$PYV" "$ROOT/research/downstream_quality_aime/aime_eval.py" \
    --base-url "$AIME_URL" --model "$MODEL" --years 2024 \
    --max-tokens 3072 --seed 1234 --no-thinking \
    --label "$label" --out "$WD/aime_${label}.json" "$@" \
    >>"$LOG" 2>&1
  say "DONE  aime/$label rc=$?"
}

run_gpqa() {
  local arm="$1"; shift
  say "START gpqa/$arm : $*"
  "$GPQAV" "$ROOT/research/validity/downstream_quality_eval/run_eval.py" \
    --task gpqa_diamond --base-url "$GPQA_URL" --model "$MODEL" \
    --max-tokens 3072 --seed 0 --max-connections 8 \
    --log-dir "$INSPECT_LOGS" \
    --arm "$arm" --out "$WD/gpqa_${arm}.json" "$@" \
    >>"$LOG" 2>&1
  say "DONE  gpqa/$arm rc=$?"
}

say "==== REMEASURE START (shipped int4_g128_lmhead) ===="

# 1) AIME greedy — fast early signal (maj@1, 30 problems)
run_aime greedy            --k 1 --temperature 0.0 --top-p 1.0 --top-k 0
# 2) AIME #31 sampled — KEY (maj@8, 240 samples)
run_aime sampled           --k 8 --temperature 1.0 --top-p 0.95 --top-k 64
# 3) gpqa #31 sampled + guard — KEY (matches #598/#682 instrument basis)
run_gpqa sampled_mintok8   --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 --sampling-seed 0
# 4) gpqa greedy — the basis #542 0.4697 rests on, now on SHIPPED model
run_gpqa greedy            --temperature 0.0 --top-p 1.0 --top-k 0 --min-tokens 0 --sampling-seed 0
# 5) AIME #31 sampled + guard — isolates EOS-guard on AIME
run_aime sampled_mintok8   --k 8 --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8
# 6) gpqa #31 sampled (no guard) — isolates EOS-guard on gpqa
run_gpqa sampled           --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 0 --sampling-seed 0

say "==== REMEASURE DONE ===="
touch "$WD/out/REMEASURE_DONE"
