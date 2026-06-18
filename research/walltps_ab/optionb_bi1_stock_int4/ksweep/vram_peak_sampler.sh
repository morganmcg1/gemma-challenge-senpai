#!/usr/bin/env bash
# Read-only peak-VRAM sampler for the K-sweep arms. Samples nvidia-smi every 20s,
# tracks the running max memory.used (MiB) and appends a tagged peak line whenever
# a new max is seen. Self-terminates when sweep_456.done appears or after 3h.
# Harmless: pure query, never touches the GPU compute path.
set -u
KS=/workspace/senpai/target/research/walltps_ab/optionb_bi1_stock_int4/ksweep
OUT="$KS/vram_peak.log"
maxu=0
t=0
while [ "$t" -lt 10800 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
  utl=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
  if [ -n "$used" ] && [ "$used" -gt "$maxu" ]; then
    maxu=$used
    echo "$(date -u +%H:%M:%SZ) new_peak memory.used=${maxu}MiB util=${utl}%" >>"$OUT"
  fi
  [ -f "$KS/sweep_456.done" ] && { echo "$(date -u +%H:%M:%SZ) sweep done; final_peak=${maxu}MiB" >>"$OUT"; break; }
  sleep 20; t=$((t+20))
done
echo "$(date -u +%H:%M:%SZ) sampler exit final_peak=${maxu}MiB" >>"$OUT"
