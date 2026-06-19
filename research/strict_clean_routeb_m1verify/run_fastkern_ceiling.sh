#!/usr/bin/env bash
# PR #746 follow-on: after the batchinv-ON batched K-sweep finishes (single A10G
# -> one server at a time), measure route-b's TRUE ceiling = the FAST-KERNEL
# (VLLM_BATCH_INVARIANT=0) M=1 AR rate. Route-b's byte-exactness comes from M=1
# single-query SHAPE identity with decode (wirbel #736: int4 Marlin GEMV is
# M-invariant; the strict-#319 divergence is the multi-query-attention branch),
# NOT from batch-invariant kernels -- so route-b need not pay the batchinv tax
# the batched M=K+1 verify needed. This arm is the decisive number: route-b's
# net TPS can never exceed this ceiling * (a+1)/(K+1).
set -u
cd /workspace/senpai/target
ROOT=research/strict_clean_routeb_m1verify
PY=.venv/bin/python

echo "[fastkern] waiting for batched sweep marker ($ROOT/_sweep_done.marker)..."
# bounded guard: ~80 min max wait (5 arms * ~12 min + compose)
for _ in $(seq 1 800); do
  [ -f "$ROOT/_sweep_done.marker" ] && break
  sleep 6
done
if [ ! -f "$ROOT/_sweep_done.marker" ]; then
  echo "[fastkern] sweep marker never appeared; NOT serving (avoid GPU contention)."
  exit 1
fi
echo "[fastkern] sweep done. GPU free. Serving fast-kernel M=1 AR ceiling @ $(date -u +%H:%M:%SZ)."

# Fast-kernel M=1 AR ceiling (+ PPL guardrail + greedy-reference outputs).
$PY -m research.strict_clean_routeb_m1verify.run_arm \
    --mode arref --no-batch-invariant \
    --out-dir "$ROOT/arref_fastkern" \
    --num-prompts 128 --output-len 512 --ppl \
    --wandb-name "stark/routeb-arref-fastkern" \
    > "$ROOT/arref_fastkern.log" 2>&1
echo "[fastkern] arref_fastkern rc=$? @ $(date -u +%H:%M:%SZ)"

# Re-compose with the fast-kernel ceiling (route-b's TRUE upper bound).
echo "[fastkern] composing route-b table vs FAST-KERNEL ceiling @ $(date -u +%H:%M:%SZ)"
$PY -m research.strict_clean_routeb_m1verify.compose_routeb \
    --root "$ROOT" --ar-dir arref_fastkern \
    --out "$ROOT/routeb_table_fastkern.json" \
    > "$ROOT/routeb_table_fastkern.txt" 2>&1
echo "[fastkern] DONE @ $(date -u +%H:%M:%SZ). table -> $ROOT/routeb_table_fastkern.txt"
touch "$ROOT/_fastkern_done.marker"
