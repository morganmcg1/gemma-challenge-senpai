#!/usr/bin/env bash
# PR #748 phase 2b (recovery): STEP1/STEP2 of run_phase2.sh OOMed because the full-vocab (262k)
# prompt_logprobs forward over long PPL prompts (max 2943 tok) allocated a >1.3 GiB logits tensor
# in a single un-chunked prefill. selfconsist_ppl.py is now patched (enable_chunked_prefill +
# max_num_batched_tokens=512, util 0.85) so each prefill step's logits stay ~0.5 GiB. This script
# re-runs only the two selfconsist steps; the determinism-floor rep (STEP3, a normal decode arm
# with no prompt_logprobs) already succeeded inside the in-flight run_phase2.sh and is reused.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
RUNS="$HERE/runs"

# 1. wait for the in-flight phase2 (determinism rep) to finish so we don't contend for GPU
OLD=$(cat "$RUNS/phase2.pid" 2>/dev/null || echo "")
if [ -n "$OLD" ]; then
  echo "[phase2b] waiting for in-flight phase2 pid=$OLD (determinism rep) $(date -u +%H:%M:%SZ)"
  while kill -0 "$OLD" 2>/dev/null; do sleep 20; done
fi
echo "[phase2b] in-flight phase2 exited $(date -u +%H:%M:%SZ)"

# 2. wait until the GPU is actually free (the rep api_server must release its ~19 GiB)
echo "[phase2b] waiting for GPU to free $(date -u +%H:%M:%SZ)"
for i in $(seq 1 40); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo 99999)
  [ "${u:-99999}" -lt 2000 ] && { echo "[phase2b] GPU free (${u} MiB)"; break; }
  sleep 10
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0

# 3. fixed selfconsist bi1 (gap_probe + PPL) -- the advisor-mandated benign-tie self-consistency
echo "[phase2b] STEP1 selfconsist_ppl --bi 1 (chunked prefill) $(date -u +%H:%M:%SZ)"
CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/selfconsist_ppl.py" --bi 1 \
  > "$RUNS/selfconsist_ppl_bi1.out" 2>&1
echo "[phase2b] STEP1 exit=$? $(date -u +%H:%M:%SZ)"

# 4. fixed selfconsist bi0 (PPL only) -- BI-neutrality of PPL
echo "[phase2b] STEP2 selfconsist_ppl --bi 0 (PPL only, chunked prefill) $(date -u +%H:%M:%SZ)"
CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/selfconsist_ppl.py" --bi 0 --do-gapprobe 0 \
  > "$RUNS/selfconsist_ppl_bi0.out" 2>&1
echo "[phase2b] STEP2 exit=$? $(date -u +%H:%M:%SZ)"

echo "[phase2b] ALL DONE $(date -u +%H:%M:%SZ)"
