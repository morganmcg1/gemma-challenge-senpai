<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Verify-GEMM M=8 roofline audit — is the 53% block free to widen?

**PR:** #68 · **Author:** denken · **Date:** 2026-06-13 · **Builds on:** #30 (decode
block-share), #51/#28/#37 (int4 Marlin verify staircase, tile cliffs M=33/49)
**Question:** The deployed `fa2sw_precache_kenyan` stack verifies **M = K+1 = 8**
tokens/step (MTP drafter `num_speculative_tokens=7`). #30 attributes **53.2% of the
decode step** to the int4 Marlin verify-GEMM — the #1 block, long treated as the
untouchable "bandwidth floor." **Is that block weight-bandwidth-bound at M=8 (so
verifying MORE candidate rows per weight-read is nearly free until a tile cliff /
compute roofline), or already compute/tile-bound (no free headroom)?**

**LOCAL profiling only** — no HF Job, no submission, no serve-path change, lossless
by construction (the isolated GEMM microbenchmark never touches the emitted token
stream). Harness: `scripts/profiler/verify_gemm_roofline.py`.

## Verdict / headline

> **At M=8 the int4 Marlin verify-GEMM is unambiguously WEIGHT-BANDWIDTH-BOUND, not
> compute-bound. It runs at ~77% of the A10G HBM roofline (~600 GB/s) while using
> only ~20% of the FP16 tensor-core compute peak. Free verification headroom EXISTS
> and is bounded by the Marlin M=33 tile cliff: widening verify from M=8 to M=32
> (4× more candidate positions) costs +18% of the verify-GEMM (~+0.9 ms, ~+8% of a
> decode step), ≈ 37 µs per extra verified row — cheap but not literally zero. Past
> M=32 a hard +53% tile cliff makes wider verification expensive.**

| quantity (A10G, int4 W4A16 Marlin, M=8) | value |
|---|--:|
| measured realizable FP16 tensor peak (compute ceiling) | **64.3 TFLOPS** (A10G datasheet 70; A10 = 125) |
| HBM roofline | 600 GB/s |
| roofline ridge point (peak/BW) | **107 FLOP/byte** |
| arithmetic intensity at M=8 | **28.0 FLOP/byte** (3.8× below ridge) |
| aggregate verify-GEMM: achieved HBM bandwidth | **462 GB/s = 77.1% of peak** |
| aggregate verify-GEMM: achieved compute | 13.0 TFLOP/s = **20.2% of FP16 peak** |
| → **bound at M=8** | **BANDWIDTH-BOUND** |
| marginal cost per extra verified row (M=8→32) | **≈ 37 µs/row** (~0.32% of an 11.6 ms decode step) |
| M=8→16 (verify 8 extra rows) | +310 µs (+6.4% GEMM, +2.7% of a decode step) |
| M=8→32 (verify 24 extra rows) | +898 µs (+18.4% GEMM, +7.7% of a decode step) |
| Marlin tile cliff | **M=33: +53%**, M=49: +100% (hard ceiling on free widening) |

**One-line result:** the "53% bandwidth floor" is bandwidth-bound *because the
weights are read once and only 8 rows ride along* — it is **not** compute-irreducible
at M=8. Verifying up to **M=32** is cheap (≈37 µs/row, dominated by the flat MLP
GEMMs); the hard wall is the **M=33 Marlin tile cliff**, not the compute roofline.

## 0. The roofline framing — why the compute ceiling is FP16, not int4

The PR phrases Step-0 as "achieved int4 GFLOP/s vs int4 peak." That premise needs a
correction that turns out to *strengthen* the bandwidth-bound conclusion. **Marlin
W4A16 does not compute in int4.** It dequantizes the int4 weights to FP16 on-chip
(the `lop3` trick) and runs **FP16×FP16 tensor-core MACs** (Marlin paper, arXiv
2408.11743 §3; vLLM's GPTQ-Marlin inherits this). The 4-bit format is a weight
*storage* format that cuts HBM traffic 4× — it never touches the int4 tensor path.

So the correct **compute ceiling is the FP16 tensor peak**, which we **measured**
directly (a large-M GEMM, launch-free) at **64.3 TFLOPS** — resolving the
A10G-vs-A10 datasheet ambiguity in favour of the A10G's ~70 TFLOPS (the 125 TFLOPS
figure is the data-center A10). The int4 tensor peak (~280 TOPS) is irrelevant; if
anyone used it as the ceiling they would *understate* utilisation 4× and conclude
even more headroom.

For a W4A16 GEMM reading each int4 weight (0.5 byte) once and contributing 2 FLOPs
per output row, arithmetic intensity ≈ **4·M FLOP/byte**. Ridge point =
64.3e12 / 600e9 = **107 FLOP/byte** → compute/memory crossover at **M ≈ 27**. At
M=8, AI ≈ 32 (measured aggregate 28.0, slightly below the weight-only 4·M because
activation+output traffic is folded in) — **3.8× below the ridge → deeply
memory-bound**, exactly as the empirical numbers confirm.

## 1. Method — isolated Marlin GEMM microbenchmark

`verify_gemm_roofline.py` loads the int4 base (`google/gemma-4-E4B-it-qat-w4a16-ct`,
the same compressed-tensors W4A16 weights the deployed stack bakes PLE/fa2sw on top
of — **baking changes weight values, not GEMM shapes/dtype/kernel**, so the roofline
is faithful to the deployed verify path), auto-discovers the **168 weight-GEMM
instances** in the 42 text decoder layers, which collapse to **6 unique Marlin
shapes**, and times each `module(x)` with **synthetic bf16 activations [M, in]**
while sweeping M. Weights are fixed; only the verify row-count M changes.

| role | in → out | layers | class |
|---|--:|--:|---|
| mlp.gate_up_proj | 2560 → 20480 | 42 | MergedColumnParallelLinear |
| mlp.down_proj | 10240 → 2560 | 42 | RowParallelLinear |
| self_attn.qkv_proj (sliding) | 2560 → 3072 | 35 | QKVParallelLinear |
| self_attn.o_proj (sliding) | 2048 → 2560 | 35 | RowParallelLinear |
| self_attn.qkv_proj (full-attn) | 2560 → 6144 | 7 | QKVParallelLinear |
| self_attn.o_proj (full-attn) | 4096 → 2560 | 7 | RowParallelLinear |

These all dispatch to `vllm._custom_ops.marlin_gemm` via
`CompressedTensorsWNA16 → MarlinLinearKernel → apply_gptq_marlin_linear` (no
M-dependent kernel switch on sm_86). The dominant weight traffic is the **MLP**
(gate_up 26 MB + down 13 MB per layer × 42 ≈ 1.6 GB of the ~2 GB int4 weight).

**Timing basis = launch-free CUDA-graph replay** (the same capture mechanism vLLM
serves these layers with). Eager per-call timing carries a fixed **~55 µs/call
launch+dispatch floor** that masks the true marginal cost; it is logged only as a
cross-check (the floor is constant in M, so the M-marginal cancels it either way —
both methods agree on the marginal). The compute-ceiling probe and per-shape
roofline use the launch-free time.

## 2. M=8 is bandwidth-bound — every GEMM, not just on average

Launch-free roofline at the deployed verify width M=8:

| GEMM (× count) | t (µs) | achieved GB/s | **% HBM peak** | % FP16 compute peak |
|---|--:|--:|--:|--:|
| gate_up 2560→20480 (×42) | 62.9 | 475 | **79%** | 21% |
| down 10240→2560 (×42) | 33.6 | 445 | **74%** | 19% |
| qkv sliding 2560→3072 (×35) | 9.6 | 471 | **79%** | 20% |
| o sliding 2048→2560 (×35) | 7.3 | 413 | **69%** | 18% |
| qkv full 2560→6144 (×7) | 22.7 | 395 | **66%** | 17% |
| o full 4096→2560 (×7) | 9.7 | 622* | **~100%** | 27% |
| **aggregate verify-GEMM** | **4868** | **462** | **77.1%** | **20.2%** |

*o-full >100% is L2-residency in isolation (its 5 MB weight partly survives the
replay loop in the 6 MB L2). The big MLP GEMMs (>L2) are HBM-accurate and dominate
the bytes; the small attn GEMMs are L2-optimistic in isolation but *even more*
bandwidth-bound in the real forward — so the verdict only hardens. **Every GEMM sits
at 66–100% of the HBM roofline and 17–27% of compute → uniformly memory-bound.**

## 3. Free-verification headroom — bounded by the M=33 tile cliff

Aggregate verify-GEMM cost vs M (launch-free), relative to the single-token (M=1)
weight-read floor:

| M | gemm µs | vs M=1 | vs M=8 | %HBM | %compute | AI |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4798 | — | −1.4% | 77% | 3% | 3.5 |
| **8** | **4868** | **+1.5%** | **0.0%** | **77%** | **20%** | **28** |
| 12 | 5141 | +7.2% | +5.6% | 73% | 29% | 42 |
| 16 | 5178 | +7.9% | +6.4% | 73% | 38% | 55 |
| 24 | 5692 | +18.6% | +16.9% | 68% | 52% | 82 |
| 32 | 5766 | +20.2% | +18.4% | 68% | 68% | 108 |
| **33** | **7465** | **+55.6%** | **+53.3%** | 52% | 54% | 111 |
| 48 | 7581 | +58% | +56% | 53% | 78% | 157 |
| **49** | **9727** | **+103%** | **+100%** | 41% | 62% | 160 |

Two regimes, split at the **M=33 Marlin tile cliff**:

1. **M ≤ 32 — cheap widening.** The dominant **MLP GEMMs are flat to M=32**
   (gate_up 62→67 µs across M=1→32, +8.9% total; down 33→38 µs, +13%) — the spec
   amortization working: one weight read serves up to 32 rows. The aggregate rises
   more (+20% to M=32) because the smaller attention GEMMs have finer tile steps,
   but the marginal is only **≈ 37 µs per extra verified row** (~0.32% of an 11.6 ms
   decode step). Verifying 8 extra rows (M=8→16) = +310 µs; 24 extra (M=8→32) =
   +898 µs (~+8% of a decode step). **Cheap, but not literally zero** — the
   "completely free" impression from eager timing was the ~55 µs launch-floor
   masking the real per-row cost.
2. **M ≥ 33 — hard wall.** +53% at M=33, +100% at M=49 — the Marlin 16-row M-tile
   boundaries (3rd and 4th tiles), reproducing #51's directly-measured M=33/49
   cliffs from an independent isolated-kernel angle. **Widening past M=32 is
   expensive.**

So the free window is **M ∈ [8, 32]**: up to **4× more candidate positions** at
~37 µs/row, with a hard ceiling at M=33.

## 4. What this justifies (and what it rules out)

- **Rules OUT "the 53% block is irreducible at M=8."** It is bandwidth-bound with
  ~80% of its weight-read time spent moving weights that only 8 rows consume.
  Single-candidate linear verify at M=8 leaves the per-weight-read amortization
  ~4× under-utilised before the tile cliff.
- **Justifies a multi-candidate / small-width tree verify sized to land at M ≤ 32**
  as the next lever. Break-even: an extra verified row costs ≈ 37 µs; a decode step
  (~7.9 ms on the 481.53-TPS deployed stack, emitting E[accept]≈3.8 tok) has a
  ~2 ms/token budget, so a batch of 24 extra rows (M=8→32, +898 µs) is net-positive
  if it adds **> ~0.43 accepted tokens/step** — a low bar for a width-2…4 tree.
- **Does NOT change drafter K or the dynamic-K rule** (PR #68 scope guard; that is
  the AdaEDL/#54 lane). This audit only *prices* the verify width; acting on it
  (tree/multi-candidate verification) is the separate downstream experiment this
  audit greenlights, and bounds to M ≤ 32.

## 5. Reproduce

```bash
cd target
.venv/bin/python scripts/profiler/verify_gemm_roofline.py \
  --m-sweep 1,8,12,16,24,32,33,48,49 --ceiling-m 256,512,1024 \
  --iters 200 --warmup 40 \
  --output research/spec_cost_model/verify_gemm_roofline.json \
  --wandb_group verify-gemm-m8-audit --wandb_name verify-gemm-roofline-int4
```

**Peak GPU mem:** 18.54 GiB (model load; the GEMM timing itself is negligible).
**Env:** A10G, vLLM 0.22.0 (deployed wheel 0.22.1rc1.dev307 — Marlin W4A16 path
stable across), torch 2.11.0+cu130, `CUDA_VISIBLE_DEVICES=0`,
`VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
**Artifacts:** `verify_gemm_roofline.json`, this report.
**W&B:** group `verify-gemm-m8-audit`, run `av8a5wh8` (launch-free CUDA-graph) and
`av98bjsw` (eager cross-check showing the launch floor). Local only, no HF Job.

## 6. Caveats

1. **Int4 base, not the PLE-folded osoi5 weights** — GEMM shapes/dtype/kernel are
   identical; folding changes values not shapes, and the roofline is value-
   independent. Faithful to the deployed verify path.
2. **L2 residency** makes the small attn GEMMs' *absolute* isolated times optimistic
   (weights < 6 MB L2 survive the replay loop); the >L2 MLP GEMMs are HBM-accurate
   and dominate. The bandwidth-bound verdict and the M-marginal (constant L2 effect
   → cancels) are robust.
3. **% of a decode step** uses 11.6 ms (#51 graph-mode int4-base verify step at
   M≈8). The deployed stack is faster (~8 ms; lm_head pruned to 12k, fa2sw,
   split-KV), so the per-row % is a mild *under*-statement of how dominant the GEMM
   is on the real stack (where it is 53.2% of decode).
4. **Dense-M, not tree-masked.** The microbenchmark times a full [M, K] GEMM; a real
   width-W tree only changes the *attention* mask, not the weight GEMM (which
   processes all M rows regardless), so the GEMM roofline is unaffected.
