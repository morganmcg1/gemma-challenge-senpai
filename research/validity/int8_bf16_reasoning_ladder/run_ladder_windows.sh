#!/usr/bin/env bash
# PR #646 — bounded multi-window orchestrator for the int4->int8->bf16 reasoning ladder.
#
# Each eval_ladder.py invocation is ONE window: boots its own LocalServer child (vLLM
# 0.22.0, conc1, BI=1, gb6144), generates+scores until the 82-min soft-cap (< the 90-min
# hard bound), then exits and tears the server down. rc=0 => that body's evals are all
# complete; rc=3 => soft-cap hit mid-cell (resume next window); rc=2 => fatal (body path
# missing) -> abort that body. Idempotent per-item jsonl resume makes relaunch safe.
#
# Cells RUN here (the genuinely missing rungs): int8 GPQA+AIME, int4 GPQA. The int4 AIME
# rung is the cited dh0tbwpp (int4_g128_lmhead, denken #637, conc1 greedy gb6144 = 0.4000)
# and the bf16 rungs are cited g3cig1xo/zoszxnb0 — NOT re-run (PR: cite, don't re-run).
#
# ANALYSIS-ONLY. No HF Job, no submission. Start only AFTER any prior driver on port 8000
# has exited (one server per port).
set -u
cd /workspace/senpai/target
PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
DRV=research/validity/int8_bf16_reasoning_ladder/eval_ladder.py
LOG=research/validity/int8_bf16_reasoning_ladder/results/_orchestrator.log
MAXIT=6

run_body() {  # $1=body  $2=evals
  local body="$1" evals="$2" i=0 rc=1 before after
  while [ "$i" -lt "$MAXIT" ]; do
    i=$((i + 1))
    before=$(cat research/validity/int8_bf16_reasoning_ladder/results/${body}_*.jsonl 2>/dev/null | wc -l)
    echo "[orch] === ${body} (${evals}) window ${i} $(date -u +%FT%TZ) records_before=${before} ===" >> "$LOG"
    "$PY" "$DRV" --body "$body" --evals "$evals" --mode full --soft-cap-min 82 >> "$LOG" 2>&1
    rc=$?
    after=$(cat research/validity/int8_bf16_reasoning_ladder/results/${body}_*.jsonl 2>/dev/null | wc -l)
    echo "[orch] ${body} window ${i} rc=${rc} records_after=${after}" >> "$LOG"
    [ "$rc" -eq 0 ] && return 0
    [ "$rc" -eq 2 ] && { echo "[orch] ${body} FATAL rc=2 (body missing) — abort" >> "$LOG"; return 2; }
    if [ "$after" -le "$before" ]; then
      echo "[orch] ${body} window ${i} made NO progress (${before}->${after}) — abort to avoid spin" >> "$LOG"
      return 4
    fi
    sleep 5
  done
  echo "[orch] ${body} hit MAXIT=${MAXIT} without completing" >> "$LOG"
  return 3
}

echo "[orch] START $(date -u +%FT%TZ)" >> "$LOG"
run_body int8 gpqa,aime ; echo "[orch] int8 final rc=$?" >> "$LOG"
run_body int4 gpqa       ; echo "[orch] int4-gpqa final rc=$?" >> "$LOG"
echo "[orch] ALL DONE $(date -u +%FT%TZ)" >> "$LOG"
