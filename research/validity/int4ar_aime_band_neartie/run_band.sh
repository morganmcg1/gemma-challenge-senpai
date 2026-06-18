#!/usr/bin/env bash
# Run a range of fresh-process AIME band sessions for one arm (PR #672).
# Usage: run_band.sh <int4|bf16> <start_idx> <count> [max_num_seqs]
# Each session is a fresh server process (the "process epoch" axis). Sessions run
# sequentially (single GPU). Per-session hard cap 80 min (< SENPAI 90-min bound).
set -uo pipefail
ARM="$1"; START="$2"; COUNT="$3"; MAXSEQS="${4:-16}"
DIR=/workspace/senpai/target/research/validity/int4ar_aime_band_neartie
PY=/tmp/vllm0220-srv/bin/python
master="$DIR/_band_${ARM}_master.log"
echo "[band:$ARM] START $(date -u) start=$START count=$COUNT maxseqs=$MAXSEQS" | tee -a "$master"
end=$((START + COUNT - 1))
for i in $(seq "$START" "$end"); do
  echo "[band:$ARM] === session $i begin $(date -u +%H:%M:%S) ===" | tee -a "$master"
  timeout 4800 bash "$DIR/serve_and_eval.sh" "$ARM" "$MAXSEQS" "${ARM}s$i" \
    "$PY" "$DIR/band_neartie.py" session --arm "$ARM" --session-idx "$i" \
      --base-url http://127.0.0.1:8000 --client-concurrency 16 \
      --max-tokens 12288 --min-tokens 8 --seed 0 \
      --out "$DIR/${ARM}_session$i.json" >> "$master" 2>&1
  rc=$?
  echo "[band:$ARM] === session $i end rc=$rc $(date -u +%H:%M:%S) ===" | tee -a "$master"
done
echo "[band:$ARM] ALL DONE $(date -u)" | tee -a "$master"
