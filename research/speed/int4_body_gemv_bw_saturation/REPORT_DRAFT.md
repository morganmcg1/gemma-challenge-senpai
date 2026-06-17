# PR #602 — AR int4-body GEMV: Marlin M=1 HBM-saturated or byte-identical lever?

**stark · group `int4-body-gemv-bw-saturation` · run `ymss9maz` · LOCAL A10G
(sm_86) microbench · analysis_only · official_tps=0 · NO served change / NO HF Job.**

## TL;DR verdict

The raw ratio is **`body_gemv_bw_saturated = False`** (M=1 body Marlin W4A16 GEMV =
**436.3 GB/s = 84.2% of measured read-peak (517.9) / 72.7% of 600 spec**, just
below the ≥90% bar). **But the operationally decisive output is
`byte_identical_body_gemv_lever_exists = False`** — the nominal gap has **no
byte-identical realization**, so **realizable byte-identical body-GEMV headroom =
0.0 TPS**. The AR int4 body is a FIXED Marlin M=1 reference. The +10-over-126.378
hunt belongs to lawine's attention/graph/scheduler axis (#601). High-value
NEGATIVE; converges with lawine #591's 0.0 reclaimable from a fresh M=1 bandwidth
axis.

## Phase 1 — bandwidth-saturation (primary, decisive)

- **byte model** (exact, served safetensors `/tmp/osoi5-v0-baked`): body
  weight+scale = **1.772 GB / token** (int4 packed + g128 scales, 37 layers × 4
  fused GEMMs); M=1 activations = 3.4 MB (0.19% — weight-read dominates).
- **peak HBM** (co-measured STREAM): read **517.9 GB/s**, copy 480.3 GB/s. (600
  datasheet is unreachable — STREAM read itself is only 86% of it.)
- **M=1 achieved body-GEMV BW = 436.3 GB/s** → **84.2% of read-peak**, 72.7% of 600.
- bandwidth floor 3.43 ms (read-peak); measured 4.07 ms → nominal headroom **0.64
  ms ≈ 9.5 TPS — UNREALIZABLE** (see Phase 2; no byte-identical path exists).
- **M-sweep is the decisive evidence** — body time is FLAT in the AR plateau then
  rises only past the compute knee:

  | M | body µs | f_vs_read | AI flop/B |
  |---|---|---|---|
  | 1 | 4070 | 0.842 | 3.7 |
  | 2 | 4080 | 0.842 | 7.4 |
  | 4 | 4096 | 0.842 | 14.7 |
  | 8 | 4127 | 0.842 | 29.1 |
  | 16 | 4325 | 0.816 | 57.2 |
  | 32 | 4708 | 0.772 | 110.6 |
  | 64 | 8304 | 0.463 | 207.3 |

  M=1→8 is **flat (M-invariant, M8/M1 = 1.014)** → the GEMV is already
  weight-read-**bandwidth**-bound at M=1, NOT a latency-bound regime with a free
  tiling fix (a latency-bound kernel would get *faster* per-token as M rises; this
  one is flat then rises). Time rises only at M≥16 as AI crosses the A10G ridge
  (208 flop/byte, knee M≈52) into the **compute**-bound region — the textbook
  memory→compute roofline transition. AR decode lives at M=1, deep in the flat
  bandwidth-bound plateau.
- **gap decomposition** (per-component M=1, f_vs_read / % of body bytes):
  - gate_up 0.929 (56.4%) + down 0.867 (28.2%) = **84.6% of the body is
    near-saturated** (87–93% of read-peak) → essentially no headroom.
  - qkv 0.591 (8.6%) + o 0.657 (6.7%) = the small projections **under-saturate**
    (59–66%, wave-quantization/occupancy at M=1) — but they are only 15.3% of the
    body AND recovering them needs a retile (split-K / BLOCK_K / num_warps) that
    changes the K-reduction order → breaks byte-identity.

## Phase 2 — byte-identical recovery (the gap is NOT recoverable)

- **dispatch** (`choose_mp_linear_kernel`, authoritative): on sm_86 **only
  MarlinLinearKernel `can_implement`** — Machete & Cutlass-W4A8 = Hopper sm_90-only
  (`machete_selectable_here=False`); AllSpark rejects g128 on Ampere; Conch absent;
  Exllama fp16-only. There is **no alternative W4A16 backend** to switch to.
  (Reconfirms my #448; matches researcher brief — Machete needs Hopper `wgmma`.)
- **only selectable Marlin knob** `use_fp32_reduce=False`: at M=1 it is 1.02–1.06×
  per shape but **BREAKS byte-exactness** on the split-K layers — qkv/o/down
  bit-flip (max_abs 0.00195–0.0039), only gate_up stays exact →
  `fp32off_all_bitexact_m1=False` → fails #319. (A GEMM-output bit-diff is a
  stronger, cheaper disqualifier than a served greedy-identity run: if the kernel
  output bits differ, served byte-identity cannot hold.)
- **no served TPS benchmark run** because the identity precondition is vacuously
  unsatisfiable for any *faster* variant → nothing eligible to serve-benchmark →
  `official_tps=0`. Any faster int4 W4A16 path requires a kernel/source BUILD
  and/or changes the reduction order → not byte-identical to the shipped Marlin
  M=1 reference.

## Why this is decisive either way

The card pre-registered that both branches converge: a real bandwidth gap that is
**not byte-identically recoverable** still means "Marlin M=1 IS the reference and
the body is fixed." That is exactly what we measured — 84.2% of read-peak with the
residual locked behind (a) silicon (600→518 unreachable), (b) near-ceiling big
shapes, and (c) identity-breaking retiles on the small shapes. **No body-GEMV
speed lever for the strict-AR #481 lane.**

## Anchors / reproduction

- rung to beat: `int4_g128_lmhead` @ 126.378 official TPS, PPL 2.019, 128/128
  (strict AR). +10 ⇒ ≥136.4. PPL/greedy untouched (no served change).
- priors reproduced from a fresh M=1 axis: #591 `b001enxl` body 44.4%/6.728 ms;
  #448 `fn4iz0dz` M-inv 1.011 + only-Marlin + fp32_reduce non-byte-exact;
  gemm_roofline M=8 body 84% read-peak (≡ my M=1 via M-invariance).
- cmd: `CUDA_VISIBLE_DEVICES=0 uv run python
  research/speed/int4_body_gemv_bw_saturation/int4_body_gemv_bw_saturation.py`
- peak VRAM well under budget (22.06 GiB card, ~2.5 GB working set); self-test PASS;
  W&B `ymss9maz`.
