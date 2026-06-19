#!/usr/bin/env bash
# PR #699: serve ONE body (int4 build dir OR bf16 HF id) on a SINGLE warmed vLLM
# instance, at a configurable max-model-len so the SAME instance can serve both the
# gate (6144) and high (12288) token budgets -- the only varied axis between the two
# eval calls is the request max_tokens (the #610 clean single-instance design; holding
# the served instance fixed removes the model-len / batch-width confound that muddied
# #610's 4096->6144 per-seed deltas).
#
# ENGINE NOTE (2026-06-19): the pinned 0.22.1rc1.dev307 venv (.venvs/vllm0221) is
# UNUSABLE -- its package files are symlinks into uv's archive cache and that cache was
# evicted, leaving all 341 vllm/* symlinks dangling so `import vllm` degrades to an empty
# namespace package. We fall back to the self-contained .venvs/vllm022 (vLLM 0.22.0,
# compressed-tensors 0.15.0.1 == the int4 build's pack-quantized format version). The
# verdict here is a RELATIVE int4/base budget-sensitivity ratio measured with BOTH bodies
# on the SAME engine, so a 0.22.0-vs-0.22.1rc1 bump cancels in the ratio; base@6144 is
# sanity-checked against the banked 0.4667 as an engine-transport guard.
# analysis_only: local serve only, NO HF Job.
set -euo pipefail

MODEL="${1:?model dir or HF id}"   # /workspace/gemma_build/int4_g128_lmhead | google/gemma-4-E4B-it
PORT="${2:-8000}"
MML="${3:-13312}"                  # max-model-len: 12288 budget + prompt headroom
MNS="${4:-16}"                     # max-num-seqs: HOLD FIXED across both budgets

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-/workspace/senpai/target/.venvs/vllm022}"   # 0.22.1rc1 venv has dangling uv-cache symlinks; 0.22.0 is the working fallback
INJECT="/workspace/senpai/target/research/validity/base_fullhead_quality_sampling/_inject"

# Local dir must have weights; an HF id (contains '/' but not a leading '/') is fetched by vLLM.
if [[ "$MODEL" == /* && ! -f "$MODEL/model.safetensors" ]]; then
  echo "[serve] FATAL: no checkpoint at $MODEL"; exit 1
fi

LOG="$HERE/_server_${PORT}.log"
PIDFILE="$HERE/_server_${PORT}.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
export VLLM_USE_FLASHINFER_SAMPLER=0          # flashinfer sampler JIT needs curand.h (absent) -> use torch sampler
# BATCH-INVARIANT (2026-06-19): default OFF for this QUALITY eval. With BI=1 the int4
# body's T=1.0 *sampled* decode degenerates on hard/long AIME problems (30-56% of n=16
# samples collapse to token-salad / repetition loops -> ~0% acc, 48-69% truncation),
# while greedy (argmax) stays fine. BI pins reductions but cannot cleanly reach the int4
# Marlin GEMM (#122), leaving an inconsistent numeric path that corrupts long batched
# sampling. BI's only purpose is greedy-IDENTITY reproducibility, which this sampled
# accuracy measurement does not need. Held at the SAME value across all 4 cells so the
# int4/base budget ratio stays clean. Override with VLLM_BATCH_INVARIANT=1 to reproduce.
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}"
export PYTHONPATH="$INJECT${PYTHONPATH:+:$PYTHONPATH}"

# generation-config override: DEFAULT EMPTY for the greedy basis. The banked greedy
# anchor (serve.py, int4=0.350 clean) used NO --override-generation-config -- it let the
# model's native generation_config.json (do_sample:true,T=1.0,top_k:64,top_p:0.95) stand
# as the request DEFAULT, which the per-request greedy params (T=0.0) cleanly overrode.
# Passing --override-generation-config with the SAME sampled values changed the precedence
# enough that T=1.0 leaked past the greedy request -> the int4 body collapsed into
# repetition-loop gibberish (acc 0.0667 vs banked 0.350) EVEN THOUGH the harness recorded
# T=0.0. So: greedy serves pass NO override (match banked verbatim). Opt back in only for a
# deliberately-sampled serve by setting OVERRIDE_GEN_CONFIG.
OVERRIDE_GEN_CONFIG="${OVERRIDE_GEN_CONFIG:-}"

OVERRIDE_FLAG=()
if [[ -n "$OVERRIDE_GEN_CONFIG" ]]; then
  OVERRIDE_FLAG=(--override-generation-config "$OVERRIDE_GEN_CONFIG")
fi

# ENFORCE-EAGER (2026-06-19): on the substitute .venvs/vllm022 engine the inductor-
# compiled + CUDA-graph int4-Marlin decode path corrupts GREEDY decode (acc collapses to
# ~0.03 vs banked 0.350, coherent-but-WRONG answers, RUN-TO-RUN NON-DETERMINISTIC under
# nominal greedy -- the banked /tmp/vllm0220-srv venv that decoded int4 cleanly is gone).
# enforce_eager disables compilation+graph capture -> the eager Marlin path, which is the
# trustworthy accuracy substrate and should reconcile to the banked 0.350. Default ON for
# this QUALITY eval; bf16 base is held at the SAME setting so the ratio stays clean.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
EAGER_FLAG=()
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  EAGER_FLAG=(--enforce-eager)
fi

echo "[serve] model=$MODEL port=$PORT max_model_len=$MML max_num_seqs=$MNS gpu_mem=${GPU_MEM:-0.90} bi=$VLLM_BATCH_INVARIANT eager=$ENFORCE_EAGER override='${OVERRIDE_GEN_CONFIG:-<none>}' $(date -u +%H:%M:%SZ)" | tee "$LOG"

# Engine flags match the banked greedy serve.py wrapper (submissions/bf16_base_aime/
# serve.py) so the 6144 greedy cells reconcile to the banked anchors. The ONLY
# intended change vs banked is --max-model-len (8192 -> 13312) to fit the 12288
# budget; greedy argmax is invariant to model-len on a BI=1 stack. seed=0 and
# max-num-batched-tokens=2048 mirror the banked wrapper (both irrelevant to greedy
# argmax, kept for byte-fidelity insurance).
setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MML" \
  --gpu-memory-utilization "${GPU_MEM:-0.90}" \
  --max-num-seqs "$MNS" \
  --max-num-batched-tokens "${MNBT:-2048}" \
  --seed "${VLLM_ENGINE_SEED:-0}" \
  --trust-remote-code \
  --disable-log-stats \
  "${EAGER_FLAG[@]}" \
  "${OVERRIDE_FLAG[@]}" \
  >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] server pid=$SRV_PID logging to $LOG"

for i in $(seq 1 360); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve] FAILED: server exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve] READY after ${i}x5s ($(date -u +%H:%M:%SZ))"; exit 0
  fi
  sleep 5
done
echo "[serve] TIMEOUT; tail:"; tail -40 "$LOG"; exit 1
