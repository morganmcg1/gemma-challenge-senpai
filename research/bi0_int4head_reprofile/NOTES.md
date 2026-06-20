# PR #798 — Post-int4head decode re-profile (stark)

Analysis-only profiling card. LOCAL A10G, no HF job, no submission.
W&B group: `bi0-int4head-reprofile`.

## Goal
Re-attribute the conc=1 decode cycle on the MERGED int4head config
(`submissions/int4_mtp_bi0_int4head`, model `/workspace/gemma_build/bi0_int4head_g32`)
now that #788 quantized the lm_head bf16 GEMV -> int4 W4A16 g32 Marlin
(1.342 -> 0.378 GB/tok; 2.777 -> 0.750 ms/firing). Split body-Marlin verify-GEMM
from lm_head GEMV; before/after vs #781; rung-3 verdict.

## Methodology to reuse (#781 / #786 / #789 era)
- `scripts/local_validation/serve_profile.py` (STEPTIME drafter/verify split +
  torch-profiler trace categorization over a steady decode window, CUDA graphs ON).
- bi0-compatible STEPTIME probe = `/tmp/steptime_patch_786.py` (wraps
  `GPUModelRunner.execute_model` + `Gemma4Proposer.propose`; emits the same
  `[steptime] raw` lines serve_profile.parse_steptime reads).
- Isolation must be adapted for the int4head serve.py: spec_off = NUM_SPECULATIVE_TOKENS=0
  (not SPECULATIVE_CONFIG=""); lm_head pruning isolation (LM_HEAD_PRUNE) is kenyan-only.

## BLOCKER (2026-06-20) — disk too small to serve the int4head stack
- Only writable FS is `/` with **~7.6 GB free** (overlay 100% used; no larger volume).
- Source ckpt `google/gemma-4-E4B-it-qat-w4a16-ct` `model.safetensors` = **11.5 GB**
  (multimodal towers stay bf16). `/workspace/gemma_build/bi0_int4head_g32` is NOT
  built on this pod and cannot be built (source download alone > free disk; build
  needs source + output simultaneously ~ 22 GB).
- `.venv` (8.9 GB) is fully provisioned with the matching server stack and must stay.

## Fallback plan (fits in disk)
1. Recover the pre-int4head bi0-surgattn ABSOLUTE per-cycle component ms (#781/#786/#789).
2. Direct microbench of the int4 lm_head GEMV: download ONLY the 1.342 GB bf16
   lm_head tensor, quantize to int4 g32 (build_lmhead_quant primitives), run vLLM
   Marlin at M=1/M=7 -> fresh int4 lm_head ms (the single new variable).
3. Re-attribute: swap the lm_head term, recompute GPU-busy composition + wall TPS impact.
4. Rung-3 verdict (body-Marlin 1-wave wall vs fresh lever).
Reported to advisor; full served re-profile needs the model staged on a bigger-disk pod.
