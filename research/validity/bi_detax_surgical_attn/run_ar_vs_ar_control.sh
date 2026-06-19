#!/bin/bash
# AR-vs-AR determinism control: regenerate each arm's served spec-off (M=1 AR)
# reference a SECOND time (run B) and compare to the already-captured canonical
# run A. If run A == run B (0 divergent over the overlap), the served path is
# deterministic run-to-run, so the spec-on-vs-spec-off break measured by the main
# arms is a REAL M=K-verify-vs-M=1-AR divergence, NOT cross-process FP noise.
# This reconstructs kanna #673 / stark #690 (AR-vs-AR 0/41,984) in-scope on the
# full-vocab served substrate this PR validates on.
set -e
cd /workspace/senpai/target
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/workspace/senpai/target/.venvs/vllm022/bin/python
N="${1:-32}"
OUT=research/validity/bi_detax_surgical_attn/ar_vs_ar_control
mkdir -p "$OUT"

run_control () {
  local sub="$1" tag="$2" port="$3" refA="$4"
  echo "[$(date -u +%H:%M:%S)] CONTROL $tag — run B (spec-off, N=$N) submission=$sub"
  $PY -m scripts.local_validation.gen_greedy_reference \
    --mode served --submission "$sub" --spec-off \
    --num-prompts "$N" --output-len 512 --server-python "$PY" --port "$port" \
    --out "$OUT/${tag}_runB.jsonl" \
    > "$OUT/${tag}_runB.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] CONTROL $tag — comparing run A vs run B (greedy gate)"
  $PY -m scripts.local_validation.greedy_gate \
    --reference "$refA" --candidate "$OUT/${tag}_runB.jsonl" --json \
    > "$OUT/${tag}_arVSar.json" 2> "$OUT/${tag}_arVSar.err" || true
  $PY -m scripts.local_validation.greedy_gate \
    --reference "$refA" --candidate "$OUT/${tag}_runB.jsonl" \
    > "$OUT/${tag}_arVSar.txt" 2>&1 || true
  echo "[$(date -u +%H:%M:%S)] CONTROL $tag DONE"
  tail -8 "$OUT/${tag}_arVSar.txt"
}

REF_BASE=research/greedy_reference/workspace__senpai__target__submissions__int4_mtp_batchinv__google__gemma-4-E4B-it-qat-w4a16-ct/decode_outputs.jsonl
REF_SURG=research/greedy_reference/workspace__senpai__target__submissions__int4_mtp_bi0_surgattn__google__gemma-4-E4B-it-qat-w4a16-ct/decode_outputs.jsonl

run_control submissions/int4_mtp_batchinv      baseline_bi1   8011 "$REF_BASE"
run_control submissions/int4_mtp_bi0_surgattn  recover_bi0    8012 "$REF_SURG"
echo "[$(date -u +%H:%M:%S)] ALL CONTROLS DONE"
