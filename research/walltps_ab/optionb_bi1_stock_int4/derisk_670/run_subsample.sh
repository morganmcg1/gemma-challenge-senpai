#!/usr/bin/env bash
# PR#670 subsample robustness sweep: the eval set is a hard 128-prompt population,
# so to test the single-seed-artifact worry we resample. decode_outputs.py shuffles
# by --seed THEN truncates to --num-prompts, so n=64 x seeds{1,2,3} are 3 genuinely
# different 64-prompt resamples. Run BOTH endpoints (stock@topk32, ftv1@topk64) on each
# -> a prompt-resampling CI on espec and on the delta. n=1 per arm (the seed is the
# resampling axis). Each n=64 paired call ~ 11 min, 3 calls ~ 34 min total.
set -u
cd /workspace/senpai/target
PY=.venv/bin/python
AB=scripts/profiler/paired_tps_ab.py
D=research/walltps_ab/optionb_bi1_stock_int4/derisk_670
ENV4() { local role=$1 d=$2; echo "--${role}-env VLLM_BATCH_INVARIANT=1 --${role}-env NUM_SPECULATIVE_TOKENS=6 --${role}-env MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct --${role}-env DRAFTER_MODEL=${d}"; }

echo "[sub] START $(date -u +%FT%TZ)"
for S in 1 2 3; do
  echo "[sub] === subsample n=64 seed=$S (stock@32 vs ftv1@64) ==="
  $PY $AB \
    --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
    --baseline-label stock_topk32 --candidate-label ftv1_topk64 \
    $(ENV4 baseline /tmp/stock-topk32) $(ENV4 candidate /tmp/qat-assistant) \
    --n 1 --num-prompts 64 --output-len 512 --seed $S \
    --out-dir "$D/sub64_seed$S" \
    --wandb-group local-drafter-derisk-land --wandb-name "land/derisk-sub64-s$S" \
    > "$D/sub64_seed$S.console.log" 2>&1
  echo "[sub] seed=$S rc=$? $(date -u +%FT%TZ)" | tee "$D/sub64_seed$S.done"
done
echo "[sub] ALL DONE $(date -u +%FT%TZ)" | tee "$D/run_subsample.alldone"
