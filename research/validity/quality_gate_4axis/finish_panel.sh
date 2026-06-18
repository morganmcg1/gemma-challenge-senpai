#!/usr/bin/env bash
# PR #661 orchestrator. After the in-flight int4ar arm (run_panel.sh) finishes,
# swap the served body to bf16-base @fee6332c and run the PAIRED bf16 arm
# (MMLU-Pro n=300 seed 12345 + GSM8K n=500 seed 1234) at the SAME gb6144 M=1-AR
# (seqs=1, BI=1) panel. Writes the two bf16 result JSONs run_panel.sh emits so
# aggregate.py can assemble the 4-axis gate. No HF Job, no submission.
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/quality_gate_4axis
LOG="$HERE/_finish_panel.log"
exec >>"$LOG" 2>&1
echo "=== FINISH-PANEL START $(date -u +%FT%TZ) ==="

INT4_RUN_PID="${1:?need int4ar run_panel pid}"
SERVER_PIDFILE="$HERE/_server_int4ar.pid"

# 1) Wait for the int4ar run_panel.sh (both MMLU + GSM8K legs) to exit.
echo "[wait] int4ar run_panel pid=$INT4_RUN_PID ..."
while kill -0 "$INT4_RUN_PID" 2>/dev/null; do sleep 20; done
echo "[wait] int4ar run_panel exited $(date -u +%FT%TZ)"
sleep 3  # let the final JSON flush

# 2) Validate int4ar outputs are complete enough to pair.
MM="$HERE/results/int4ar_mmlu_pro_greedy.json"
GS="$HERE/results/int4ar_gsm8k_greedy.json"
if ! python3 - "$MM" "$GS" <<'PY'
import json, sys
mm = json.load(open(sys.argv[1])); gs = json.load(open(sys.argv[2]))
print(f"[int4ar] MMLU n_samples={mm['n_samples']} n_scored={mm['n_scored']} acc={mm['accuracy']:.4f} min_tok={mm.get('min_tokens')}")
print(f"[int4ar] GSM8K n_problems={gs['n_problems']} acc={gs['accuracy']:.4f}")
assert mm['n_samples'] >= 250, f"MMLU int4ar only {mm['n_samples']} samples"
assert gs['n_problems'] >= 400, f"GSM8K int4ar only {gs['n_problems']} problems"
PY
then
  echo "[FATAL] int4ar outputs incomplete; NOT starting bf16 arm"; exit 1
fi

# 3) Stop the int4ar server, free the GPU for the bf16 body.
SRV="$(cat "$SERVER_PIDFILE")"
echo "[swap] TERM int4ar server pgid=$SRV $(date -u +%FT%TZ)"
kill -TERM -- -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null || true
for i in $(seq 1 60); do kill -0 "$SRV" 2>/dev/null || { echo "[swap] int4 server gone after ${i}s"; break; }; sleep 1; done
kill -KILL -- -"$SRV" 2>/dev/null || true
sleep 2
for i in $(seq 1 80); do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  echo "[swap] gpu used=${used}MiB ($i)"
  [[ "${used:-99999}" -lt 2500 ]] && { echo "[swap] GPU freed $(date -u +%FT%TZ)"; break; }
  sleep 3
done

# 4) Start the bf16-base server (blocks until /v1/models healthy or fails).
echo "[serve] starting bf16 $(date -u +%FT%TZ)"
if ! bash "$HERE/serve_panel.sh" bf16; then
  echo "[FATAL] bf16 serve_panel failed"; exit 2
fi

# 5) Run the bf16 paired arm (identical sizes/seeds to int4ar).
echo "[panel] run_panel.sh bf16 300 500 0 $(date -u +%FT%TZ)"
bash "$HERE/run_panel.sh" bf16 300 500 0
echo "[panel] bf16 rc=$? $(date -u +%FT%TZ)"

echo "=== FINISH-PANEL DONE $(date -u +%FT%TZ) ==="
touch "$HERE/_finish_panel.DONE"
