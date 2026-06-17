#!/usr/bin/env bash
# PR #614 -- run the 5 SAMPLED GPQA-Diamond realizations (lewtun #31: temp=1.0,
# top_p=0.95, top_k=64, min_tokens=8) at max_tokens=4096 against the live bf16 base.
# Fixed dataset choice-shuffle seed (12345, byte-identical prompts); the per-request
# sampling RNG seed varies 1..5 (decode-noise -> cluster-bootstrap CI). Each seed is a
# SEPARATE run_eval.py process (well under the 90-min per-run wall); this driver just
# chains them sequentially on the single GPU so KV pressure stays low.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNONE="$HERE/run_one.sh"
echo "[sampled5] START $(date -u +%FT%TZ)"
for s in 1 2 3 4 5; do
  out="$HERE/runs/sampled_4096_s${s}.json"
  if [ -s "$out" ]; then echo "[sampled5] seed=$s exists, skip"; continue; fi
  bash "$RUNONE" "sampled_4096_s${s}" 1.0 0.95 64 4096 "$s" "$out"
  rc=$?
  echo "[sampled5] seed=$s rc=$rc $(date -u +%FT%TZ)"
  if [ "$rc" -ne 0 ]; then echo "[sampled5] ABORT on seed=$s rc=$rc"; exit $rc; fi
done
echo "[sampled5] DONE $(date -u +%FT%TZ)"
