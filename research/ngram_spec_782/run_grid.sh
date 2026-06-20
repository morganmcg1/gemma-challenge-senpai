#!/usr/bin/env bash
# PR #782 — full bi0 ngram-vs-MTP A/B grid, GPU-serial (one A10G).
# Control C = bi0 MTP K=6 (+PPL). Variant A = ngram k in {3,5} x prompt_lookup_max
# in {3,4}, prompt_lookup_min=2 fixed (+PPL on k5_plm3). All cells add
# VLLM_USE_FLASHINFER_SAMPLER=0 (local flashinfer JIT lacks curand.h; native
# sampler is numerically inert at temp=0 and applied identically to every arm).
# Continues past a single-cell failure so one bad boot can't sink the grid.
set -u
cd /workspace/senpai/target
export CUDA_VISIBLE_DEVICES=0
RC=research/ngram_spec_782/run_cell.py
LOGDIR=research/ngram_spec_782/cells
mkdir -p "$LOGDIR"

run() {  # $1=label $2=submission $3=extra_env_json $4=ppl_flag
  local label="$1" sub="$2" env="$3" ppl="$4"
  echo "=== [grid] $(date -u +%H:%M:%SZ) START $label ppl=$ppl ==="
  python3 "$RC" --submission "$sub" --label "$label" \
    --extra-env "$env" --num-prompts 128 --output-len 512 $ppl \
    > "$LOGDIR/$label.driver.log" 2>&1
  echo "=== [grid] $(date -u +%H:%M:%SZ) END   $label rc=$? ==="
}

# 1) MTP control (bi0 stack, K=6 from manifest). +PPL.
run control_mtp_k6 submissions/int4_mtp_bi0_surgattn \
  '{"VLLM_USE_FLASHINFER_SAMPLER":"0"}' --ppl

# 2-5) ngram grid (int4_ngram_bi0_surgattn). PPL only on k5_plm3.
run ngram_k3_plm3 submissions/int4_ngram_bi0_surgattn \
  '{"VLLM_USE_FLASHINFER_SAMPLER":"0","SPECULATIVE_METHOD":"ngram","NUM_SPECULATIVE_TOKENS":"3","PROMPT_LOOKUP_MAX":"3","PROMPT_LOOKUP_MIN":"2"}' ""
run ngram_k3_plm4 submissions/int4_ngram_bi0_surgattn \
  '{"VLLM_USE_FLASHINFER_SAMPLER":"0","SPECULATIVE_METHOD":"ngram","NUM_SPECULATIVE_TOKENS":"3","PROMPT_LOOKUP_MAX":"4","PROMPT_LOOKUP_MIN":"2"}' ""
run ngram_k5_plm3 submissions/int4_ngram_bi0_surgattn \
  '{"VLLM_USE_FLASHINFER_SAMPLER":"0","SPECULATIVE_METHOD":"ngram","NUM_SPECULATIVE_TOKENS":"5","PROMPT_LOOKUP_MAX":"3","PROMPT_LOOKUP_MIN":"2"}' --ppl
run ngram_k5_plm4 submissions/int4_ngram_bi0_surgattn \
  '{"VLLM_USE_FLASHINFER_SAMPLER":"0","SPECULATIVE_METHOD":"ngram","NUM_SPECULATIVE_TOKENS":"5","PROMPT_LOOKUP_MAX":"4","PROMPT_LOOKUP_MIN":"2"}' ""

echo "=== [grid] $(date -u +%H:%M:%SZ) ALL CELLS DONE ==="
touch "$LOGDIR/grid.done"
