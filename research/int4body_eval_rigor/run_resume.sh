#!/usr/bin/env bash
# Resume the 4 remaining decode-basis arms (AIME greedy + AIME #31-sampled
# already landed). Runs against the int4_g128_lmhead endpoint now served at
# MAX_MODEL_LEN=6144 so gpqa_diamond can be scored at the truncation-clean
# max_tokens=4096 basis the #515 base gate was set on (bars_verdict #614:
# base 6144/4096; <=2048 depresses GPQA 0.07-0.14). AIME stays at 3072 to match
# base_aime.json (same-basis apples-to-apples; base AIME maj@8=0.40 was 3072).
set -u
ROOT=/workspace/senpai/target
WD="$ROOT/research/int4body_eval_rigor"
LOG="$WD/out/resume.log"
INSPECT_LOGS="$WD/out/logs"
AIME_URL="http://127.0.0.1:8000"
GPQA_URL="http://127.0.0.1:8000/v1"
MODEL="gemma-4-e4b-it"
PYV="$ROOT/.venv/bin/python"
GPQAV="/tmp/eval-serve-venv/bin/python"
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
    --max-tokens 4096 --seed 12345 --max-connections 8 \
    --log-dir "$INSPECT_LOGS" \
    --arm "$arm" --out "$WD/gpqa_${arm}.json" "$@" \
    >>"$LOG" 2>&1
  say "DONE  gpqa/$arm rc=$?"
}

say "==== RESUME START (int4_g128_lmhead @ MAX_MODEL_LEN=6144, gpqa@4096) ===="

# Most decisive first: the #31-compliant + guard gpqa arm (does int4 clear 0.471?)
run_gpqa sampled_mintok8   --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 --sampling-seed 0
# gpqa greedy on shipped int4 at the 4096 basis (reproduce ~base greedy 0.5051?)
run_gpqa greedy            --temperature 0.0 --top-p 1.0 --top-k 0 --min-tokens 0 --sampling-seed 0
# AIME #31 sampled + guard — isolates EOS-guard on AIME (3072, base-matched)
run_aime sampled_mintok8   --k 8 --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8
# gpqa #31 sampled (no guard) — isolates EOS-guard on gpqa
run_gpqa sampled           --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 0 --sampling-seed 0

say "==== RESUME DONE ===="
touch "$WD/out/RESUME_DONE"
