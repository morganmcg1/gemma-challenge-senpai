#!/usr/bin/env bash
# PR #661 -- auto-chain the bf16-base GSM8K capped paired leg AFTER the in-flight
# bf16 MMLU capped leg (pid in _mmlu_bf16_capped.pid) exits. Keeps the single
# A10G busy (no idle gap) without ever running two eval clients against the
# seqs=1 server at once (that would break M=1 isolation). Each leg is an
# independent run with its OWN 86-min watchdog, so the hard 90-min-per-run bound
# (SENPAI_TIMEOUT_MINUTES) is respected per leg, not across the chain.
#
# GSM8K bf16 leg: paired PREFIX of int4's seed-1234 500 via --n 500 --limit 200
# (the seed-1234 shuffle is over the full split, so order[:200] is a strict
# prefix of order[:500] -> byte-identical prompts + fewshot on the shared 200
# ids). limit=200 projects ~58 min at bf16's 15 tok/s (int4 first-200 ~= 48k
# decode tok), comfortably under the 86-min watchdog. GSM8K is already a
# STRUCTURAL PASS (int4=0.9220 => pct_of_base >= 92.2% for any base <= 1.0); this
# leg only fills the table cell with a measured bf16 base. No HF Job, no submission.
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/quality_gate_4axis
RES="$HERE/results"
CLIENT=/tmp/eval-serve-venv/bin/python
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
ST="$HERE/_gsm8k_bf16_capped.status"
LOG="$HERE/_gsm8k_bf16_capped.out"
: > "$ST"

MMLU_PID="$(cat "$HERE/_mmlu_bf16_capped.pid" 2>/dev/null || echo 0)"
echo "CHAIN-WAIT $(date -u +%FT%TZ) for bf16 MMLU pid=$MMLU_PID" >> "$ST"
while [[ "$MMLU_PID" != "0" ]] && kill -0 "$MMLU_PID" 2>/dev/null; do sleep 20; done
echo "CHAIN-MMLU-EXITED $(date -u +%FT%TZ)" >> "$ST"
sleep 3  # let the MMLU json flush

if [[ -f "$RES/bf16_mmlu_pro_greedy.json" ]]; then
  echo "CHAIN-MMLU-JSON-OK $(date -u +%FT%TZ)" >> "$ST"
else
  echo "CHAIN-MMLU-JSON-MISSING (watchdog likely fired; recover from .eval) $(date -u +%FT%TZ)" >> "$ST"
fi

# Launch bf16 GSM8K capped paired prefix.
$CLIENT research/downstream_quality_gsm8k/gsm8k_eval.py \
  --base-url "$BASE" --model "$MODEL" --label bf16_gsm8k \
  --regimes greedy --n 500 --limit 200 --n-shot 8 --seed 1234 \
  --max-tokens 6144 --min-tokens 8 --concurrency 16 \
  --out-dir "$RES" \
  > "$LOG" 2>&1 &
GPID=$!
echo "$GPID" > "$HERE/_gsm8k_bf16_capped.pid"
echo "GSM8K-CAPPED-START $(date -u +%FT%TZ) pid=$GPID n=200(prefix of seed-1234 500)" >> "$ST"

# Hard 86-min bound guard (graceful first).
( sleep 5160
  if kill -0 "$GPID" 2>/dev/null; then
    echo "WATCHDOG-FIRE $(date -u +%FT%TZ) pid=$GPID (86min bound)" >> "$ST"
    kill -INT "$GPID" 2>/dev/null; sleep 12
    kill -0 "$GPID" 2>/dev/null && { kill -TERM "$GPID" 2>/dev/null; sleep 4; }
    kill -0 "$GPID" 2>/dev/null && kill -KILL "$GPID" 2>/dev/null
    touch "$HERE/_gsm8k_bf16_capped.WATCHDOG_FIRED"
  else
    echo "WATCHDOG-NOOP $(date -u +%FT%TZ) run already finished" >> "$ST"
  fi
) &

wait "$GPID"
RC=$?
echo "GSM8K-CAPPED-END $(date -u +%FT%TZ) rc=$RC" >> "$ST"
[[ -f "$RES/bf16_gsm8k_greedy.json" ]] && echo "GSM8K-JSON-OK $(date -u +%FT%TZ)" >> "$ST" || echo "GSM8K-JSON-MISSING $(date -u +%FT%TZ)" >> "$ST"
touch "$HERE/_chain_gsm8k.DONE"
