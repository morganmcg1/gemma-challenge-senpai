#!/usr/bin/env bash
# PR #615 -- faithful #547 reproduction on the SHIPPED int4_g128_lmhead under 0.22.0:
# GREEDY (temp=0, top_p=1, top_k=0), min_tokens=0 (no #548 EOS-guard, exactly the
# pre-#548 protocol #547 used; #547 reported "0% EOS-empty"). Tests whether the
# "0.22.0 craters MMLU" belief TRANSFERS to the shipped checkpoint. max_tokens=4096
# so truncation is directly comparable to the sampling panel.
set -u
cd /workspace/senpai/target
VENV=/senpai-run/home/student-lawine/eval-client-venv/bin/python
RUN_EVAL=research/validity/downstream_quality_eval/run_eval.py
URL=http://127.0.0.1:8000/v1
OUT=research/validity/eval_stack_accuracy_validity/runs
LOGD=research/validity/eval_stack_accuracy_validity/logs
export HF_HOME=/senpai-run/home/student-lawine/.cache/huggingface
echo "[repro547] START $(date -u +%FT%TZ)"
# MMLU greedy (the eval #547 named)
$VENV "$RUN_EVAL" --task mmlu_pro --arm ship_v0220_greedy --n 200 --seed 12345 \
  --temperature 0.0 --top-p 1.0 --top-k 0 --min-tokens 0 --max-tokens 4096 \
  --max-connections 32 --base-url "$URL" \
  --out "$OUT/mmlu_v0220_greedy.json" --log-dir "$LOGD/mmlu_v0220_greedy" \
  > "$LOGD/mmlu_v0220_greedy.log" 2>&1 && echo "[repro547] mmlu greedy OK" || echo "[repro547] mmlu greedy FAIL"
# GPQA greedy (the keystone)
$VENV "$RUN_EVAL" --task gpqa_diamond --arm ship_v0220_greedy --seed 12345 \
  --temperature 0.0 --top-p 1.0 --top-k 0 --min-tokens 0 --max-tokens 4096 \
  --max-connections 32 --base-url "$URL" \
  --out "$OUT/gpqa_v0220_greedy.json" --log-dir "$LOGD/gpqa_v0220_greedy" \
  > "$LOGD/gpqa_v0220_greedy.log" 2>&1 && echo "[repro547] gpqa greedy OK" || echo "[repro547] gpqa greedy FAIL"
echo "[repro547] DONE $(date -u +%FT%TZ)"
