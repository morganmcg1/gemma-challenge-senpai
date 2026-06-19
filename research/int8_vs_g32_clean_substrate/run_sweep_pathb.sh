#!/usr/bin/env bash
# PR #726 Path-B sweep: for each arm build the dense bf16 fake-quant checkpoint
# from the qat_unq bf16 master, serve it once (BI=1, generation_config sampling),
# run the #31 gate-basis AIME for 5 seeds via --base-url, stop, delete the build.
# Sequential (one build on disk at a time). Protocol = my #702 verbatim (sampled):
# years 2024,2025-I,2025-II (n=60), k=1, T=1.0 top_p=0.95 top_k=64, max_tokens=12288,
# min_tokens=8, no-thinking, conc 16, seeds 0-4 -> pooled 300.
set -u
cd /workspace/senpai/target
ROOT=research/int8_vs_g32_clean_substrate
L=$ROOT/logs; R=$ROOT/results
mkdir -p "$L" "$R"
SERVE_PY=/tmp/vllm0220-srv/bin/python
BUILD_PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
EVAL_PY=/usr/bin/python3
BUILD_ROOT=/workspace/pathb_build
mkdir -p "$BUILD_ROOT"
# flashinfer JIT needs curand.h; expose ONLY curand*.h via an absolute CPATH shim
# (do NOT shim cuda.h/cuda_runtime.h -- those must resolve from the real 13.2 toolkit).
WHEEL_INC=/tmp/vllm0220-srv/lib/python3.12/site-packages/nvidia/cu13/include
CURAND_INC=/workspace/senpai/target/$ROOT/_curand_shim
mkdir -p "$CURAND_INC"
ln -sf "$WHEEL_INC"/curand*.h "$CURAND_INC"/ 2>/dev/null || true
PORT=${PORT:-8000}
YEARS="2024,2025-I,2025-II"
SEEDS="${SEEDS:-0 1 2 3 4}"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"
ARMS="${ARMS:-full_g32 int8_locus bf16_locus}"

stamp(){ date -u +%FT%TZ; }

# vLLM's EngineCore is a separate child owning the GPU alloc; reap it explicitly by
# proctitle and wait for VRAM to drain so the next serve starts clean.
reap_serve(){
  [ -n "${1:-}" ] && kill -9 "$1" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for _ in $(seq 1 15); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${used:-9999}" -lt 1000 ] 2>/dev/null && break
    sleep 2
  done
}

run_arm(){
  arm=$1; dir=$BUILD_ROOT/fq_$arm
  if [ ! -f "$dir/model.safetensors" ]; then
    echo "[chain] BUILD $arm $(stamp)  free=$(df -h / | awk 'NR==2{print $4}')"
    CUDA_VISIBLE_DEVICES="" "$BUILD_PY" "$ROOT/build_path_b.py" --arm "$arm" --out "$dir" \
      > "$L/build_$arm.log" 2>&1 || { echo "[chain] BUILD FAILED $arm"; tail -8 "$L/build_$arm.log"; return 1; }
  fi
  reap_serve
  echo "[chain] SERVE $arm $(stamp)  free=$(df -h / | awk 'NR==2{print $4}')  gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)"
  CUDA_VISIBLE_DEVICES=0 MODEL_ID="$dir" SERVED_MODEL_NAME=gemma-4-e4b-it HOST=127.0.0.1 PORT=$PORT \
    MAX_MODEL_LEN=13312 GPU_MEMORY_UTILIZATION=0.92 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    VLLM_BATCH_INVARIANT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CPATH="$CURAND_INC" \
    nohup "$SERVE_PY" submissions/int4_base_aime/serve.py > "$L/serve_$arm.log" 2>&1 &
  spid=$!
  ready=0
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 $spid 2>/dev/null || { echo "[chain] SERVE DIED $arm"; tail -25 "$L/serve_$arm.log"; reap_serve $spid; return 1; }
    sleep 5
  done
  [ $ready -eq 1 ] || { echo "[chain] SERVE TIMEOUT $arm"; reap_serve $spid; return 1; }
  echo "[chain] READY $arm $(stamp)"
  rc=0
  for s in $SEEDS; do
    echo "[chain] EVAL $arm seed=$s $(stamp)"
    "$EVAL_PY" research/downstream_quality_aime/aime_eval.py --base-url "http://127.0.0.1:$PORT" \
      --model gemma-4-e4b-it --years "$YEARS" --k 1 --temperature 1.0 --top-p 0.95 --top-k 64 \
      --max-tokens 12288 --min-tokens 8 --no-thinking --client-concurrency 16 $LIMIT_ARG \
      --seed "$s" --label "${arm}_seed$s" --out "$R/${arm}_seed$s.json" --save-text \
      > "$L/eval_${arm}_seed$s.log" 2>&1 || { echo "[chain] EVAL FAIL $arm seed=$s"; rc=1; }
    acc=$("$EVAL_PY" -c "import json;d=json.load(open('$R/${arm}_seed$s.json'));print('acc=%.4f n=%d corr=%d fail=%.3f wall=%.0fs'%(d['maj_k_accuracy'],d['n_problems'],d['n_correct_maj'],d['extract_fail_rate'],d['wall_s']))" 2>/dev/null || echo "NO-JSON")
    echo "[chain]   -> $arm seed=$s $acc"
  done
  reap_serve $spid
  echo "[chain] STOP $arm rc=$rc gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1) $(stamp)"
  [ "${KEEP_BUILD:-0}" = "1" ] || rm -rf "$dir"
  return $rc
}

echo "[chain] === START arms='$ARMS' seeds='$SEEDS' limit='${LIMIT:-full}' $(stamp) ==="
for arm in $ARMS; do
  run_arm "$arm" || { echo "[chain] ABORT at $arm $(stamp)"; exit 1; }
done
echo "[chain] === ALL DONE $(stamp) ==="
