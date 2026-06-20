# lm_head GEMV time: bf16 vs int8 (eager, M=1, spec OFF) — PR #788

Reproduces the PR #781 decode op-profiler on one checkpoint at a time, same harness
(`scripts.local_validation.profile_decode --mode eager --profile-mode op`), same GPU
(A10G, CUDA_VISIBLE_DEVICES=0), native sampler. lm_head runs once per generated token,
so the per-token GEMV = (kernel cuda_ms total) / gen_tokens with gen_tokens=256.

## int8 head — /workspace/gemma_build/bi0_int8head_ch  [profile_int8head/]
- gen_tokens 256, eager_tps 8.4, total_cuda_ms 5016.93
- lm_head GEMV kernel: `_C::allspark_w8a16_gemm` (op) -> `void allspark::ampere_hgemm_W8A16_perc_f16_...`
  - count 256 (1 / token), tot 366.00 ms, **per-token 1.4297 ms** (op wrapper 1.4308 ms; same op listed twice)
  - AllSpark W8A16 channelwise, Ampere sm_86 path — as hypothesized.
- body matmuls = Marlin (int4 body, unchanged): _C::marlin_gemm 32.25% + Marlin<...> 16.83%+15.27%.

## bf16 head — google/gemma-4-E4B-it-qat-w4a16-ct  [profile_bf16head/]
- gen_tokens 256, eager_tps 7.95, total_cuda_ms 7130.97
- lm_head GEMV kernel: `aten::linear` -> `void gemv2T_kernel_val<int,int,__nv_bfloat16,...>` (cuBLAS bf16 GEMV)
  - count 256 (1 / token), tot 710.92 ms, **per-token 2.7770 ms** (9.97%) — native bf16 Linear, NOT Marlin.
  - Reproduces #781 anchor 2.776 ms/token (1.34218 GB/token, 80.6% peak BW) to 3 decimals, same harness/GPU/session.
- body matmuls = Marlin int4 (unchanged): _C::marlin_gemm + Marlin<...> identical kernel set to int8 arm.

## int4 head — /workspace/gemma_build/bi0_int4head_g32  [profile_int4head/]
- gen_tokens 256, eager_tps 8.11, total_cuda_ms 4666.93
- lm_head GEMV is Marlin W4A16 g32 -- SAME kernel template as the int4 body, so it has NO separate
  count==256 entry; it folds into `_C::marlin_gemm`. Isolated by differencing vs the byte-identical int8 body:
  - `_C::marlin_gemm` op-count: int8 64768 -> int4 65024 = **exactly +256** (lm_head, once/token).
  - cuda_ms delta: op-level 1809.6-1617.7 = +191.9 ms; kernel-level 1809.4-1617.4 = +192.0 ms (agree).
  - **int4 lm_head GEMV = 191.9 ms / 256 = 0.750 ms/token.**

## Mechanism (same-session, same-harness, eager M=1 spec OFF) -- bandwidth-bound, all three consistent
- lm_head GEMV per token:  bf16 **2.777** -> int8 **1.430** -> int4 **0.750** ms/token.
  bytes/token 1.342 / 0.672 / 0.378 GB; speedups 1.0x / 1.94x / 3.70x track the byte ratios (BW-bound, ~470-500 GB/s).
- bf16 head = native bf16 cuBLAS GEMV (gemv2T); int8 head = AllSpark W8A16 channelwise; int4 head = Marlin W4A16 g32
  (same kernel family as body). int4 body (Marlin) byte-identical across all candidate arms.
- Explains measured decode TPS (local_prevalidate, spec ON, 128 prompts):
  bf16 control 219.34 -> int8 241.09 (+9.9%) -> int4 256.74 (+17.0%); PPL preserved 2.0057 / 2.0051 / 2.0029.
