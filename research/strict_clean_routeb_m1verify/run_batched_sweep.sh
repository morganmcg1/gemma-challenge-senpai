#!/usr/bin/env bash
# PR #746: batched-verify K-sweep orchestrator. Self-sequences AFTER the AR
# anchor finishes (single A10G -> one server at a time). For each K it serves
# int4_mtp_batchinv with NUM_SPECULATIVE_TOKENS=K, captures 128x512 warm-steady
# greedy decode wall_tps + spec acceptance, then composes the route-b table.
set -u
cd /workspace/senpai/target
ROOT=research/strict_clean_routeb_m1verify
PY=.venv/bin/python

echo "[sweep] waiting for AR anchor to finish ($ROOT/arref/arm_result.json)..."
# bounded guard: ~40 min max wait for the AR anchor
for _ in $(seq 1 400); do
  [ -f "$ROOT/arref/arm_result.json" ] && break
  sleep 6
done
if [ ! -f "$ROOT/arref/arm_result.json" ]; then
  echo "[sweep] AR anchor never produced arm_result.json; aborting sweep." ; exit 1
fi
echo "[sweep] AR anchor done. Starting batched K-sweep at $(date -u +%H:%M:%SZ)."

for K in 2 3 4 5 6; do
  PPL_FLAG=""
  [ "$K" = "6" ] && PPL_FLAG="--ppl"   # one batched PPL datapoint (manifest default K)
  echo "[sweep] === batched K=$K $PPL_FLAG @ $(date -u +%H:%M:%SZ) ==="
  $PY -m research.strict_clean_routeb_m1verify.run_arm \
      --mode batched --k "$K" \
      --out-dir "$ROOT/batched_k$K" \
      --num-prompts 128 --output-len 512 $PPL_FLAG \
      --wandb-name "stark/routeb-batched-k$K" \
      > "$ROOT/batched_k$K.log" 2>&1
  echo "[sweep] K=$K rc=$? @ $(date -u +%H:%M:%SZ)"
done

echo "[sweep] composing route-b table @ $(date -u +%H:%M:%SZ)"
$PY -m research.strict_clean_routeb_m1verify.compose_routeb \
    --root "$ROOT" --out "$ROOT/routeb_table.json" \
    > "$ROOT/routeb_table.txt" 2>&1
echo "[sweep] DONE @ $(date -u +%H:%M:%SZ). table -> $ROOT/routeb_table.txt"
touch "$ROOT/_sweep_done.marker"
