#!/usr/bin/env bash
# PR #782 — re-run the MTP control cell once the GPU-serial grid is done.
# The first control attempt died at record 59/128 with a transient client-side
# RemoteDisconnected (server logged 60 clean 200 OKs, no error/OOM/traceback),
# so this just needs a clean 128/128 re-run on the freed GPU. Bounded waits +
# up-to-2 attempts so a transient drop can't leave us with no control.
set -u
cd /workspace/senpai/target
export CUDA_VISIBLE_DEVICES=0
LOG=research/ngram_spec_782/cells

# 1) Wait (bounded ~60 min) for the running grid to finish and free the GPU.
for i in $(seq 1 180); do
  [ -f "$LOG/grid.done" ] && break
  sleep 20
done
sleep 15   # let the last cell's server fully release GPU memory

# 2) Preserve the failed 59-record control artifacts (crash evidence) once.
if [ -f "$LOG/control_mtp_k6/decode_outputs.jsonl" ] && [ ! -f "$LOG/control_mtp_k6/cell_summary.json" ]; then
  rm -rf "$LOG/control_mtp_k6.failed59"
  mv "$LOG/control_mtp_k6" "$LOG/control_mtp_k6.failed59"
fi

# 3) Re-run control (bi0 MTP K=6 + PPL); retry once if a transient drop recurs.
for attempt in 1 2; do
  echo "=== [rerun] $(date -u +%H:%M:%SZ) control attempt $attempt ==="
  python3 research/ngram_spec_782/run_cell.py \
    --submission submissions/int4_mtp_bi0_surgattn --label control_mtp_k6 \
    --extra-env '{"VLLM_USE_FLASHINFER_SAMPLER":"0"}' \
    --num-prompts 128 --output-len 512 --ppl \
    > "$LOG/control_rerun_attempt${attempt}.log" 2>&1
  n=$(wc -l < "$LOG/control_mtp_k6/decode_outputs.jsonl" 2>/dev/null || echo 0)
  echo "=== [rerun] attempt $attempt produced $n records ==="
  [ "${n:-0}" -ge 128 ] && break
  rm -rf "$LOG/control_mtp_k6.attempt${attempt}_fail"
  mv "$LOG/control_mtp_k6" "$LOG/control_mtp_k6.attempt${attempt}_fail" 2>/dev/null || true
done

touch "$LOG/control.done"
echo "=== [rerun] $(date -u +%H:%M:%SZ) DONE ==="
