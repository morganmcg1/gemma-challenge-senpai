#!/usr/bin/env bash
# PR #796 GSM8K quick-screen orchestrator (greedy, n=200, seed 1234, 8-shot).
# Serializes one GPU server at a time (LocalServer ctx mgr tears each down before
# the next). Three arms that hold the screen gate (PPL<=2.42 AND TPS>=g32 control):
#   ctrl_g32_mm   = merged #788 g32 minmax lm_head  (my int4_g32_lmhead build)
#   armA_chan_mm  = channelwise minmax  (lawine_bf_chan_mm)
#   armB_chan_mse = channelwise MSE     (lawine_bf_chan_mse)
# Output: research/_int8head_smoke/gsm8k/<label>_greedy.json
set -uo pipefail
ROOT=/workspace/senpai/target
cd "$ROOT"
DRV=/usr/bin/python3
SRVPY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
OUT=research/_int8head_smoke/gsm8k
SUB=submissions/int4_mtp_bi0_lmhead_bytefloor

run_arm () {
  local label=$1 model=$2 port=$3
  echo "[gsm8k-orch] === $label model=$model port=$port $(date -u +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES=0 "$DRV" research/downstream_quality_gsm8k/gsm8k_eval.py \
    --submission "$SUB" --label "$label" --regimes greedy \
    --n 200 --n-shot 8 --seed 1234 --max-num-seqs 32 \
    --port "$port" --server-python "$SRVPY" \
    --serve-env MODEL_ID="$model" --out-dir "$OUT" 2>&1
  echo "[gsm8k-orch] $label rc=$? $(date -u +%H:%M:%S)"
}

run_arm ctrl_g32_mm   /workspace/gemma_build/int4_g32_lmhead   8041
run_arm armA_chan_mm  /workspace/gemma_build/lawine_bf_chan_mm  8042
run_arm armB_chan_mse /workspace/gemma_build/lawine_bf_chan_mse 8043
echo "[gsm8k-orch] ALL DONE $(date -u +%H:%M:%S)"
