#!/usr/bin/env bash
# PR #646 — windowed supervisor for the ONE missing rung: int4 GPQA-D.
#
# int8 GPQA+AIME are already complete (summaries written). The int4 AIME rung is the
# cited dh0tbwpp (0.4000) and the bf16 rungs are cited g3cig1xo/zoszxnb0 (PR: cite,
# don't re-run). The int4 GPQA-D greedy endpoint is measured here in-harness so the
# GPQA ladder is self-consistent and in launch-isolation scope.
#
# Robustness: eval_ladder.py now serves from a stable submission copy OUTSIDE the git
# tree (/workspace/gemma_build/sub_int4_base_aime), so the entrypoint's ~10-min branch
# checkout can no longer unlink the live server CWD. Each window self-caps at 82 min
# (< the 90-min hard bound) and resumes idempotently from per-item jsonl.
set -u
cd /workspace/senpai/target
PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
DRV=research/validity/int8_bf16_reasoning_ladder/eval_ladder.py
RES=research/validity/int8_bf16_reasoning_ladder/results
LOG="$RES/_int4_gpqa_supervisor.log"
MAXIT=4

echo "[sup] START $(date -u +%FT%TZ)" >> "$LOG"
i=0
while [ "$i" -lt "$MAXIT" ]; do
  i=$((i + 1))
  before=$(wc -l < "$RES/int4_gpqa.jsonl" 2>/dev/null || echo 0)
  echo "[sup] === int4/gpqa window ${i} $(date -u +%FT%TZ) records_before=${before} ===" >> "$LOG"
  "$PY" "$DRV" --body int4 --evals gpqa --mode full --soft-cap-min 82 --no-wandb >> "$LOG" 2>&1
  rc=$?
  after=$(wc -l < "$RES/int4_gpqa.jsonl" 2>/dev/null || echo 0)
  echo "[sup] int4/gpqa window ${i} rc=${rc} records_after=${after}" >> "$LOG"
  [ "$rc" -eq 0 ] && { echo "[sup] int4/gpqa COMPLETE $(date -u +%FT%TZ)" >> "$LOG"; break; }
  [ "$rc" -eq 2 ] && { echo "[sup] int4/gpqa FATAL rc=2 (body missing) — abort" >> "$LOG"; break; }
  if [ "$after" -le "$before" ]; then
    echo "[sup] int4/gpqa window ${i} NO progress (${before}->${after}) — abort to avoid spin" >> "$LOG"
    break
  fi
  sleep 5
done
echo "[sup] DONE $(date -u +%FT%TZ) final_rc=${rc}" >> "$LOG"
