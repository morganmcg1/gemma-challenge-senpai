#!/usr/bin/env bash
# Spec-break quality-materiality census (PR #682, wirbel) -- run ONE benchmark leg
# against the already-running arm server at http://127.0.0.1:8000.
#
# Usage: run_leg.sh <arm: ar|spec> <leg: gsm8k|mmlu|gpqa|aime> [N] [SEED]
#
# All legs GREEDY (temp=0), min_tokens=8 EOS-guard (wirbel #541), gb6144 budget.
# GPQA-Diamond is GREEDY here (NOT the sampled genbudget): the strict-#319 break
# is a greedy-token phenomenon, and the AR reference is greedy AR. One leg per
# invocation keeps every command well under the 90-min cap.
set -uo pipefail
cd /workspace/senpai/target

ARM="${1:?arm: ar|spec}"
LEG="${2:?leg: gsm8k|mmlu|gpqa|aime}"
N="${3:-}"
SEED="${4:-}"

PY=/tmp/eval-serve-venv/bin/python
AIME_PY=/usr/bin/python3
DIR=research/validity/specbreak_quality_materiality
RES="$DIR/results/$ARM"
mkdir -p "$RES"
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
MT=6144
TS() { date -u +%H:%M:%S; }

case "$LEG" in
  gsm8k)
    NN="${N:-300}"
    echo "[$ARM/gsm8k] n=$NN 8-shot greedy start $(TS)"
    $PY research/downstream_quality_gsm8k/gsm8k_eval.py \
      --base-url "$BASE" --model "$MODEL" --label "${ARM}_greedy" \
      --regimes greedy --n "$NN" --n-shot 8 --seed 1234 \
      --max-tokens "$MT" --min-tokens 8 --concurrency 16 --save-text \
      --out-dir "$RES"
    echo "[$ARM/gsm8k] rc=$? $(TS)"
    ;;
  mmlu)
    NN="${N:-300}"
    echo "[$ARM/mmlu] n=$NN greedy start $(TS)"
    $PY research/validity/downstream_quality_eval/run_eval.py \
      --task mmlu_pro --arm "$ARM" --out "$RES/mmlu_pro.json" \
      --n "$NN" --seed 12345 --max-tokens "$MT" --min-tokens 8 \
      --max-connections 16 --base-url "$BASE/v1" --model "$MODEL"
    echo "[$ARM/mmlu] rc=$? $(TS)"
    ;;
  gpqa)
    SS="${SEED:-12345}"
    echo "[$ARM/gpqa] diamond greedy seed=$SS start $(TS)"
    $PY research/validity/downstream_quality_eval/run_eval.py \
      --task gpqa_diamond --arm "$ARM" --out "$RES/gpqa_diamond_s${SS}.json" \
      --seed "$SS" --max-tokens "$MT" --min-tokens 8 --max-connections 16 \
      --temperature 0.0 --top-p 1.0 --base-url "$BASE/v1" --model "$MODEL"
    echo "[$ARM/gpqa] rc=$? $(TS)"
    ;;
  aime)
    echo "[$ARM/aime] 2024+2025 k=1 no-thinking greedy start $(TS)"
    # client-concurrency 16: greedy score is concurrency-independent (aime_eval.py
    # L323-329); sequential (default 1) would put 60x6144-tok problems over the
    # 90-min/command cap. Matches the other 3 legs' conc=16 panel config.
    $AIME_PY research/downstream_quality_aime/aime_eval.py \
      --base-url "$BASE" --model "$MODEL" --years 2024,2025-I,2025-II --k 1 \
      --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens "$MT" --min-tokens 8 \
      --no-thinking --seed 1234 --save-text --client-concurrency 16 \
      --label "${ARM}_aime" --out "$RES/aime.json"
    echo "[$ARM/aime] rc=$? $(TS)"
    ;;
  *)
    echo "unknown leg: $LEG" >&2; exit 2;;
esac
