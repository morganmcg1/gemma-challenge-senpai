#!/usr/bin/env bash
# PR #590 -- AIME multi-seed decode realizations on the live base_fullhead server.
#
# n=60 fixed benchmark (--years 2024,2025-I,2025-II == the #567/#580 anchor's 60
# byte-identical ids). FIVE request-seeds, each k=5 samples/problem -> S=25 decode
# samples per question merged downstream by aggregate_ci.py. Protocol: lewtun #31
# sampling (T=1.0 top_p=0.95 top_k=64) + min_tokens=8 EOS-guard, --no-thinking.
#
# Seeds run CONCURRENTLY: 5 clients x k=5 = 25 in-flight <= server max_num_seqs=32,
# so vLLM batches them and the whole sweep finishes in ~one k=5 pass (~31 min) rather
# than 5x sequentially. Each individual client pass is ~31 min, well under the
# 90-min/run bound. Batch size does not change the per-sequence sampled distribution.
set -u
cd /workspace/senpai/target
PY=/usr/bin/python3                       # aime_eval uses stdlib urllib (no inspect_ai)
URL0=http://127.0.0.1:8000                # aime_eval appends /v1/chat/completions itself
OUT=research/validity/quality_gates_ci/runs
LOGD=research/validity/quality_gates_ci/logs
SEEDS=(1234 2345 3456 4567 5678)
SAMP="--temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8"

mkdir -p "$OUT" "$LOGD"
echo "[aime] multiseed START $(date -u +%H:%M:%SZ) seeds=${SEEDS[*]} k=5 -> S=25"
t0=$(date +%s)
pids=()
for s in "${SEEDS[@]}"; do
  $PY research/downstream_quality_aime/aime_eval.py --base-url $URL0 \
    --years 2024,2025-I,2025-II --k 5 $SAMP --max-tokens 3072 --seed "$s" \
    --no-thinking --save-text --label "aime_bfh_s${s}" \
    --out "$OUT/aime_base_fullhead_s${s}.json" \
    >"$LOGD/aime_s${s}.log" 2>&1 &
  pids+=($!)
  echo "[aime]   launched seed=$s pid=${pids[-1]} -> $OUT/aime_base_fullhead_s${s}.json"
done

fail=0
for i in "${!SEEDS[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[aime]   seed=${SEEDS[$i]} OK ($(grep -h 'DONE' "$LOGD/aime_s${SEEDS[$i]}.log" | tail -1))"
  else
    echo "[aime]   seed=${SEEDS[$i]} FAILED rc=$? (tail below)"; tail -8 "$LOGD/aime_s${SEEDS[$i]}.log"; fail=1
  fi
done
echo "[aime] multiseed DONE wall=$(( $(date +%s)-t0 ))s $(date -u +%H:%M:%SZ) fail=$fail"
exit $fail
