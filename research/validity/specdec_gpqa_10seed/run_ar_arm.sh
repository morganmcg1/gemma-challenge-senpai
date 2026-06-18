#!/usr/bin/env bash
# PR #656 — AR-arm half of the AR-vs-spec GPQA 10-seed contrast.
# Transition orchestrator: stop the running SPEC (k=6) server, bring up the AR
# (k=0, drafter OFF, M=1) server on the SAME dev307 stack / same int4_g128_lmhead
# body+head, wait for it to be ready, then run the identical 10-seed GPQA-Diamond
# sampled sweep (convention A: vary dataset --seed, sampling_seed=0) via
# run_gpqa_10seed.sh. Single-variable contrast — the ONLY change vs SPEC is k.
set -uo pipefail
cd /workspace/senpai/target
DIR=research/validity/specdec_gpqa_10seed
SEEDS="12345 13579 23456 34567 45678 56789 67890 78901 89012 90123"
SERVE_PY=/tmp/senpai-venvs/a341b8bdf5ec1fe0/bin/python   # dev307 server venv (cached)
LOG=$DIR/_ar_transition.log
: > "$LOG"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# --- 1. stop the SPEC server gracefully (SIGTERM -> serve_arm.py clean shutdown) ---
if [ -f "$DIR/_serve.pid" ]; then
  OLD=$(cat "$DIR/_serve.pid")
  if kill -0 "$OLD" 2>/dev/null; then
    say "stopping SPEC server pid=$OLD (SIGTERM)"
    kill -TERM "$OLD" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$OLD" 2>/dev/null || break; sleep 1; done
    kill -0 "$OLD" 2>/dev/null && { say "SPEC server still up; SIGKILL"; kill -9 "$OLD" 2>/dev/null || true; sleep 3; }
  fi
fi
# wait for GPU memory to drain below ~3 GiB so the AR server has the full A10G
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  say "gpu mem used=${used}MiB"
  [ "${used:-99999}" -lt 3000 ] && break
  sleep 3
done

# --- 2. start the AR (k=0) server in background ---
rm -f "$DIR/serve.ready"
say "starting AR server (k=0)"
nohup "$SERVE_PY" "$DIR/serve_arm.py" --k 0 --max-model-len 8192 \
  > "$DIR/_serve_ar_m1.boot" 2>&1 &
NEW=$!
echo "$NEW" > "$DIR/_serve.pid"
say "AR server pid=$NEW; waiting for ready"

# --- 3. wait for ready (serve.ready third line == ar_m1) ---
ok=0
for _ in $(seq 1 300); do
  if [ -f "$DIR/serve.ready" ] && grep -qx "ar_m1" "$DIR/serve.ready" 2>/dev/null; then ok=1; break; fi
  kill -0 "$NEW" 2>/dev/null || { say "AR server died during boot; see _serve_ar_m1.boot"; exit 1; }
  sleep 2
done
[ "$ok" = 1 ] || { say "AR server not ready after 600s"; exit 1; }
say "AR server ready: $(tr '\n' ' ' < "$DIR/serve.ready")"

# --- 4. run the 10-seed AR sweep (same runner, same seeds, convention A) ---
say "launching AR 10-seed GPQA sweep"
bash "$DIR/run_gpqa_10seed.sh" ar_m1 $SEEDS >> "$LOG" 2>&1
rc=$?
say "AR sweep done rc=$rc"
exit $rc
