#!/usr/bin/env bash
# PR #661 -- capped paired bf16-base MMLU-Pro leg. Honors the hard
# SENPAI_TIMEOUT_MINUTES=90 per-run bound by evaluating only a token-budgeted
# RANDOM paired subset of int4's 300 ids (subset_ids_mmlu.json), so the run
# finishes well under 90 min on its own; an 86-min watchdog is a hard guard in
# case bf16 CoT runs longer than the int4-length budget. Same gb6144 seqs=1 BI=1
# min_tokens=8 greedy panel, byte-identical prompts on the shared ids. No HF Job.
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/quality_gate_4axis
OUT="$HERE/results/bf16_mmlu_pro_greedy.json"
LOG="$HERE/_mmlu_bf16_capped.out"
ST="$HERE/_mmlu_bf16_capped.status"

/tmp/eval-serve-venv/bin/python research/validity/downstream_quality_eval/run_eval.py \
  --task mmlu_pro --arm bf16_mmlu --out "$OUT" \
  --n 300 --seed 12345 --ids-file "$HERE/subset_ids_mmlu.json" \
  --max-tokens 6144 --min-tokens 8 --max-connections 16 \
  --base-url http://127.0.0.1:8000/v1 --model gemma-4-e4b-it \
  > "$LOG" 2>&1 &
PYPID=$!
echo "$PYPID" > "$HERE/_mmlu_bf16_capped.pid"
echo "MMLU-CAPPED-START $(date -u +%FT%TZ) pid=$PYPID n=61 budget=60000tok" >> "$ST"

# Hard 86-min bound guard (graceful SIGINT lets inspect finalize the .eval log).
( sleep 5160
  if kill -0 "$PYPID" 2>/dev/null; then
    echo "WATCHDOG-FIRE $(date -u +%FT%TZ) SIGINT pid=$PYPID (86min bound)" >> "$ST"
    kill -INT "$PYPID" 2>/dev/null; sleep 12
    kill -0 "$PYPID" 2>/dev/null && { kill -TERM "$PYPID" 2>/dev/null; sleep 4; }
    kill -0 "$PYPID" 2>/dev/null && kill -KILL "$PYPID" 2>/dev/null
    touch "$HERE/_mmlu_bf16_capped.WATCHDOG_FIRED"
  else
    echo "WATCHDOG-NOOP $(date -u +%FT%TZ) run already finished" >> "$ST"
  fi
) &

wait "$PYPID"
echo "MMLU-CAPPED-END $(date -u +%FT%TZ) rc=$?" >> "$ST"
