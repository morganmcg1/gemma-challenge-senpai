#!/usr/bin/env bash
# PR #703 -- bounded post-debias finalizer. Waits for the running MMLU driver
# (run_mmlu_int4.sh, pid arg) to exit, verifies all 5 debiased seeds, then runs
# finalize_mmlu_agg.sh (-> summaries/mmlu_debiased.json) and assemble_panel.py
# (-> panel.json + the single W&B run). Does NOT restart the server or re-run evals.
set -u
TGT=/workspace/senpai/target
HERE="$TGT/research/int4body_gate_panel"
LOG="$HERE/logs/finalize_driver.log"
MMLU_PID="${1:?usage: finalize_driver.sh <mmlu_driver_pid>}"
PYI=/tmp/eval-serve-venv/bin/python
VENV="$TGT/.venv/bin/python"
N=2000
exec >>"$LOG" 2>&1
echo "[finalize] === START $(date -u +%H:%M:%SZ) waiting for MMLU driver pid=$MMLU_PID ==="
deadline=$(( $(date +%s) + 3600 ))
while kill -0 "$MMLU_PID" 2>/dev/null; do
  if (( $(date +%s) > deadline )); then
    echo "[finalize] $(date -u +%H:%M:%SZ) DEADLINE exceeded; MMLU driver still running -> abort"
    exit 1
  fi
  sleep 20
done
echo "[finalize] $(date -u +%H:%M:%SZ) MMLU driver exited; verifying 5 debias seeds"
miss=0
for s in 1 2 3 4 5; do
  f="$HERE/runs/mmlu_debias_n${N}_s${s}.json"
  if [[ -f "$f" ]] && $PYI -c "import json,sys;d=json.load(open('$f'));sys.exit(0 if len(d.get('per_sample',[]))>=$N else 1)" 2>/dev/null; then
    echo "[finalize]   debias s$s OK acc=$($PYI -c "import json;print(round(json.load(open('$f'))['accuracy'],4))")"
  else
    echo "[finalize]   debias s$s MISSING/INCOMPLETE"; miss=1
  fi
done
if (( miss )); then echo "[finalize] FATAL: not all 5 debias seeds ready -> abort (will retry on next wakeup)"; exit 2; fi
echo "[finalize] $(date -u +%H:%M:%SZ) running finalize_mmlu_agg.sh"
if ! bash "$HERE/finalize_mmlu_agg.sh"; then echo "[finalize] finalize_mmlu_agg.sh FAILED"; exit 3; fi
echo "[finalize] $(date -u +%H:%M:%SZ) running assemble_panel.py (with wandb, from panel dir, WANDB_DIR=/tmp)"
if ! ( cd "$HERE" && WANDB_DIR=/tmp "$VENV" "$HERE/assemble_panel.py" ); then echo "[finalize] assemble_panel.py FAILED"; exit 4; fi
echo "[finalize] === DONE $(date -u +%H:%M:%SZ) panel.json + W&B run created ==="
