#!/usr/bin/env bash
# PR #816 — run the downstream quality panel for k in {2,4,8}, one server boot per
# k, matching the completed k=1 control config EXACTLY (only TOPK_ACCEPT_K varies):
#   max_model_len=12288, max_num_seqs=32, num_spec=6, mmlu_n=250, gsm8k_n=500,
#   seed=12345, gpqa_max_tokens=6144, aime_max_tokens=3072, tasks as below.
# LOCAL A10G only (CUDA_VISIBLE_DEVICES=0). No HF job.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1   # -> target/
HERE="research/topk_match_accept_816"
PROG="$HERE/quality_k248_progress.log"
: > "$PROG"
echo "[driver] start $(date -u +%FT%TZ)" | tee -a "$PROG"
for k in 2 4 8; do
  echo "[driver] === k=$k start $(date -u +%FT%TZ) ===" | tee -a "$PROG"
  CUDA_VISIBLE_DEVICES=0 uv run python "$HERE/quality_panel.py" \
    --k "$k" \
    --tasks mmlu_pro,gpqa_diamond,aime_greedy,gsm8k \
    --max-model-len 12288 --max-num-seqs 32 --num-spec 6 \
    --mmlu-n 250 --gsm8k-n 500 --seed 12345 \
    --wandb-group bi0-int4head-topk-accept \
    > "$HERE/quality_k${k}_driver.log" 2>&1
  rc=$?
  band=$(python3 -c "import json;print(json.load(open('$HERE/runs/quality_k${k}/panel.json')).get('in_band'))" 2>/dev/null || echo "NA")
  echo "[driver] === k=$k done rc=$rc in_band=$band $(date -u +%FT%TZ) ===" | tee -a "$PROG"
done
echo "[driver] all-done $(date -u +%FT%TZ)" | tee -a "$PROG"
