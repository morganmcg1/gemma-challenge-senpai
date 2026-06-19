#!/usr/bin/env bash
# PR #703 -- wait for the running GSM8K driver to finish, then launch the MMLU
# sweep on the SAME live int4 server. Bounded: exits once GSM8K is done and the
# MMLU driver is started. Does NOT restart the server.
set -u
cd /workspace/senpai/target
GSM_PID="${1:?usage: chain_mmlu_after_gsm8k.sh <gsm8k_driver_pid>}"
LOGD=research/int4body_gate_panel/logs
mkdir -p "$LOGD"
echo "[chain] $(date -u +%H:%M:%SZ) waiting for GSM8K driver pid=$GSM_PID"
# Bounded local wait: GSM8K is 10 seeds ~110s each, hard-cap ~40min.
deadline=$(( $(date +%s) + 2400 ))
while kill -0 "$GSM_PID" 2>/dev/null; do
  if (( $(date +%s) > deadline )); then
    echo "[chain] $(date -u +%H:%M:%SZ) DEADLINE exceeded; GSM8K still running, aborting chain"
    exit 1
  fi
  sleep 15
done
echo "[chain] $(date -u +%H:%M:%SZ) GSM8K driver exited; verifying 10 result files"
n=$(ls research/int4body_gate_panel/runs/int4g128_{guard,noguard}_sampled_s{1,2,3,4,5}.json 2>/dev/null | wc -l)
echo "[chain] gsm8k result files present: $n/10"
echo "[chain] $(date -u +%H:%M:%SZ) launching MMLU sweep"
exec bash research/int4body_gate_panel/run_mmlu_int4.sh
