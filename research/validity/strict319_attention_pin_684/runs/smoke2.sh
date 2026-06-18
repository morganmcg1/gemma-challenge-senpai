set -e
PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
export CUDA_VISIBLE_DEVICES=0
for C in baseline fixed2d; do
  echo "===== SMOKE $C ====="
  $PY attn_pin_cost.py --config $C --n-prompts 4 --n-new 8 --ctx-cap 384 --det-prompts 2 \
    --tps-warmup 16 --tps-long 32 --tps-short 8 --tps-reps 2 --tps-ctx-prompts 1 \
    --out runs/smoke_$C.json 2>&1 | tail -8
done
echo "ALL_SMOKES_DONE"
