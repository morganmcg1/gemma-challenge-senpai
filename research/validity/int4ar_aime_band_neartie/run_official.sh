#!/usr/bin/env bash
# Official aime_eval.py (logprobs-OFF) full-60 fresh sessions = the distributional
# cross-check that the logprobs-on band is unbiased (the guard already showed
# logprobs-on==logprobs-off aggregate accuracy at limit-8; this confirms at n=60).
# Usage: run_official.sh <start_idx> <count> [max_num_seqs]
set -uo pipefail
START="$1"; COUNT="$2"; MAXSEQS="${3:-16}"
DIR=/workspace/senpai/target/research/validity/int4ar_aime_band_neartie
PY=/tmp/vllm0220-srv/bin/python
master="$DIR/_band_official_master.log"
echo "[official] START $(date -u) start=$START count=$COUNT" | tee -a "$master"
end=$((START + COUNT - 1))
for i in $(seq "$START" "$end"); do
  echo "[official] === session $i begin $(date -u +%H:%M:%S) ===" | tee -a "$master"
  timeout 4800 bash "$DIR/serve_and_eval.sh" int4 "$MAXSEQS" "officials$i" \
    "$PY" /workspace/senpai/target/research/downstream_quality_aime/aime_eval.py \
      --base-url http://127.0.0.1:8000 --model gemma-4-e4b-it \
      --years 2024,2025-I,2025-II --k 1 --temperature 0 --max-tokens 12288 \
      --min-tokens 8 --no-thinking --client-concurrency 16 --seed 0 \
      --out "$DIR/official_session$i.json" >> "$master" 2>&1
  echo "[official] === session $i end rc=$? $(date -u +%H:%M:%S) ===" | tee -a "$master"
done
echo "[official] ALL DONE $(date -u)" | tee -a "$master"
