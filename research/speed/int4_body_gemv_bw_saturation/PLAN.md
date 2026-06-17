# PR #602 — AR int4-body GEMV: Marlin M=1 HBM-saturated or byte-identical lever?

**stark · group `int4-body-gemv-bw-saturation` · LOCAL A10G (sm_86) microbench ·
NO served-file change, NO kernel BUILD, NO HF Job. `analysis_only=True`,
`official_tps=0`.**

## Decisive question
On the shipped `int4_g128_lmhead` AR (spec-OFF, M=1) decode path, is the int4
**body** Marlin W4A16 GEMV already **HBM-bandwidth-saturated** (→ no body speed
lever; the whole +10-over-126.378 hunt belongs to lawine's attention/graph axis),
or is there a real **byte-identical** gap between the measured M=1 body-GEMV time
and the pure weight-read bandwidth floor (→ a concrete +TPS lever on the dominant
44.4% component)?

## Phase 1 — body-GEMV bandwidth-saturation verdict (primary, decisive)
1. Self-built g=128 int4-Marlin body GEMMs (qkv/o/gate_up/down) at the SERVED
   fused shapes (hidden 2560, inter 10240, 37 layers, head_dim 256, GQA 8:2),
   timed L2-cold (n_distinct cold weights, working set >> 6 MiB L2) via CUDA-graph
   replay — the same `apply_gptq_marlin_linear` the served kernel calls. Co-measure
   peak HBM BW (STREAM read 1x / copy 2x). Exact per-token body weight+scale bytes
   from the served safetensors (`/tmp/osoi5-v0-baked`).
2. Headline: **M=1 achieved body-GEMV bandwidth vs peak** = measured GB/s /
   {read-peak, 600 spec}. ≥~0.90 → `body_gemv_bw_saturated=True`. Else quantify
   `body_gemv_headroom_ms` / `body_gemv_headroom_tps`.
3. **M-sweep (1,2,4,8,16,32,64)**: if the body-GEMV time is flat (M-invariant)
   the kernel is already weight-read-**bandwidth**-bound at M=1 (not a
   latency-bound regime with a free tiling fix). AI ≈ 30 flop/byte ≪ A10G ridge
   208 → memory-bound by construction up to M≈52.
4. Decompose any gap: near-saturated gate_up/down vs under-saturated small qkv/o
   (wave-quantization/occupancy), recoverable-only-by-retile (reduction-order
   change → identity break).

## Phase 2 — byte-identical recovery (only if Phase 1 finds a gap)
- Dispatch enumeration (`choose_mp_linear_kernel`): on sm_86 only **Marlin**
  `can_implement` (Cutlass-W4A8/Machete = Hopper sm_90-only; AllSpark no-g128 on
  Ampere; Conch absent; Exllama fp16-only). Confirms my #448.
- Only selectable Marlin knob `use_fp32_reduce=False`: measure speedup AND
  bit-compare GEMM output vs served default. Prior (#448): +0.18% but BREAKS
  byte-exactness on split-K layers → identity FAIL.
- Verdict: any faster int4 W4A16 path requires a kernel BUILD and/or changes the
  reduction order → NOT byte-identical to the shipped Marlin M=1 reference.

## Gate
Only `identity==1.0` variants vs shipped `int4_g128_lmhead` plain-greedy-AR
reference (official `check_greedy_identity.py`, zero tol) are eligible speed
levers. A GEMM-output bit-diff is a STRONGER (cheaper) disqualifier than served
identity: if the kernel output bits differ, served greedy identity cannot hold.

## Anchors
- Rung to beat: `int4_g128_lmhead` @ 126.378 official TPS, PPL 2.019, 128/128.
  +10 ⇒ ≥ 136.4.
- Cycle decomp (lawine #591 `b001enxl`): int4 BODY 44.4% / 6.728 ms; free-head
  ceiling 311.27 TPS.
- #448 (`fn4iz0dz`): body M-invariant (M8/M1=1.011); only-Marlin on sm_86;
  fp32_reduce=False non-byte-exact; no byte-exact selectable headroom.
- gemm_roofline (`roofline_ceiling.json`): M=8 body GEMM = 433 GB/s = 84%
  read-peak (518) / 72% of 600 spec.
- τ ≈ 1.03524 local→official (#267). A10G ~600 GB/s spec HBM (read-peak ~518).
