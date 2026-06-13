#!/usr/bin/env bash
# Sweep num_speculative_tokens K, probing acceptance + exploratory TPS per K and
# logging each to W&B (group int4-mtp-drafter). Launches the spec-decode server
# fresh per K, waits ready, probes, then tears down and waits for the GPU to free
# before the next K. Greedy identity (temp=0) is K-independent, so pass the
# verdict from the separate reference comparison via GREEDY=.
#
# Set EAGER=0 to capture under CUDA graphs (the submission's ENFORCE_EAGER=0
# path): this validates that cudagraph capture works with the patched, split
# attention groups and yields a more representative exploratory TPS. EAGER=1
# (default) is faster to start when only relative acceptance per K matters.
#
# Env: KS="5 6 7", OUTDIR=/tmp/val, GREEDY=true|false|unknown, EAGER=0|1, PPL=
set -uo pipefail
cd "$(dirname "$0")"
OUTDIR="${OUTDIR:-/tmp/val}"
KS="${KS:-5 6 7}"
GREEDY="${GREEDY:-unknown}"
PPL="${PPL:-}"
EAGER="${EAGER:-1}"
PY="/workspace/senpai/target/.venvs/vllm022/bin/python"
mkdir -p "$OUTDIR"

free_gpu() {
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ "$used" -lt 800 ] && return 0
    sleep 3
  done
  return 1
}

for K in $KS; do
  echo "=== K=$K launch $(date +%H:%M:%S) ==="
  LOG="$OUTDIR/server_k${K}.log"
  NUM_SPECULATIVE_TOKENS=$K ENFORCE_EAGER=$EAGER setsid bash launch_server.sh >"$LOG" 2>&1 &
  SPID=$!
  sleep 1
  PGID=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
  echo "server pid=$SPID pgid=$PGID log=$LOG"
  if bash wait_ready.sh "$LOG" 900; then
    pplarg=()
    [ -n "$PPL" ] && pplarg=(--ppl "$PPL")
    "$PY" log_probe_wandb.py --k "$K" --greedy-identical "$GREEDY" "${pplarg[@]}" \
      | tee "$OUTDIR/probe_k${K}.txt"
  else
    echo "server K=$K NOT READY; see $LOG"
  fi
  echo "=== K=$K teardown $(date +%H:%M:%S) ==="
  [ -n "${PGID:-}" ] && kill -TERM -"$PGID" 2>/dev/null
  kill -TERM "$SPID" 2>/dev/null
  if free_gpu; then echo "GPU freed"; else echo "WARN: GPU not freed: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; fi
done
echo "SWEEP DONE $(date +%H:%M:%S)"
