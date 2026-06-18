#!/usr/bin/env bash
# PR #661 -- finalize + aggregate the 4-axis #515 gate once both bf16 legs land.
# 1) Ensure bf16_mmlu_pro_greedy.json exists: run_eval.py writes it only on normal
#    completion, so if the 86-min watchdog SIGINT'd the leg, recover the SAME-schema
#    JSON from the capped MMLU .eval (scored samples preserved; prompt_sha recomputed
#    byte-identically for the paired key). GSM8K's custom harness has no .eval, but
#    GSM8K is a structural PASS (int4=0.9220 => pct_of_base>=92.2% for any base<=1.0),
#    so a missing bf16 GSM8K only blanks one table cell, never the verdict.
# 2) Run aggregate.py (system python3 has wandb; eval venvs do not) with --wandb so the
#    panel lands in W&B group quality-gate-4axis-denken. No HF Job, no submission.
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/quality_gate_4axis
RES="$HERE/results"
MM="$RES/bf16_mmlu_pro_greedy.json"
CAPPED_EVAL="$RES/_inspect_logs/2026-06-18T14-38-57-00-00_task_kFvkFCKYgUy9fzdyFkcyBL.eval"

if [[ ! -f "$MM" ]]; then
  echo "[agg] bf16 MMLU JSON missing -> recovering from capped .eval"
  EVAL="$CAPPED_EVAL"
  [[ -f "$EVAL" ]] || EVAL="$(ls -t "$RES"/_inspect_logs/*.eval | head -1)"
  /tmp/eval-serve-venv/bin/python "$HERE/recover_eval_json.py" \
    --eval-log "$EVAL" --out "$MM" --arm bf16_mmlu \
    --seed 12345 --n-requested 300 --max-tokens 6144 --min-tokens 8
else
  echo "[agg] bf16 MMLU JSON present (normal completion)"
fi

WANDB_FLAG=""
[[ "${1:-}" == "--no-wandb" ]] && WANDB_FLAG="" || WANDB_FLAG="--wandb"
echo "[agg] running aggregate.py $WANDB_FLAG"
python3 "$HERE/aggregate.py" $WANDB_FLAG --wandb-group quality-gate-4axis-denken
