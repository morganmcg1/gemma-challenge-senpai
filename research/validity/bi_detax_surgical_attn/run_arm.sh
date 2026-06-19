#!/bin/bash
# Run one arm: served spec-off M=1 AR reference, then spec-on candidate + greedy gate.
# Usage: run_arm.sh <submission-dir> <num-prompts> <tag> [extra-validate-flags...]
set -e
cd /workspace/senpai/target
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/workspace/senpai/target/.venvs/vllm022/bin/python
SUB="$1"; N="$2"; TAG="$3"; shift 3
OUT="research/validity/bi_detax_surgical_attn/${TAG}"
mkdir -p "$OUT"
echo "[$(date -u +%H:%M:%S)] ARM=$TAG submission=$SUB N=$N — generating served spec-off M=1 AR reference"
$PY -m scripts.local_validation.gen_greedy_reference \
  --mode served --submission "$SUB" --spec-off \
  --num-prompts "$N" --output-len 512 --server-python "$PY" --port 8001 \
  > "$OUT/reference_gen.log" 2>&1
echo "[$(date -u +%H:%M:%S)] reference done — serving spec-on candidate + greedy gate"
$PY -m scripts.local_validation.validate_submission \
  --submission "$SUB" --server-python "$PY" \
  --num-prompts "$N" --output-len 512 --port 8000 \
  --out-dir "$OUT/validate" "$@" \
  > "$OUT/validate.log" 2>&1
echo "[$(date -u +%H:%M:%S)] ARM=$TAG DONE"
