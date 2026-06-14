<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# LUT/GANQ W4A16 GEMM feasibility at M=8 on sm_86 — build or kill?

**PR:** #113 · **Author:** denken · **Date:** 2026-06-14 · **Builds on:** #68 (verify-GEMM
M=8 roofline — MEASURED 77.1% HBM-bound), #105/#109 (tree-free ship-readiness corner),
kanna #96 (self-referential greedy gate)
**Question:** Does **LUT-based W4A16 GEMM** (LUT-GEMM, Park et al. ICLR 2024
arXiv:2206.09557; GANQ, ICML 2025) beat **int4 Marlin** at the deployed verify width
**M=8 on the A10G (sm_86)** by enough to move denken #109's conservative ship corner from
*straddle* to **≥500 with margin** — alone or composed with ubel #108's SplitK — and is it
worth a multi-day kernel build?

**LOCAL only**: analytic roofline anchored on #68's MEASURED Marlin numbers, plus an
OPTIONAL single-GPU INT8-TC substrate probe (`torch._int_mm`, no model load, no token
stream). No HF Job, no submission, no served-file change. This is a **SIZING gate** (decide
whether to BUILD), not a kernel build.

## Verdict / headline

> **🔴 RED — do NOT build LUT-GEMM for the M=8 verify-GEMM. KILL the lane.**
> The M=8 verify-GEMM is **bandwidth-bound** (#68: 77.1% HBM, 20.2% compute). LUT-GEMM is a
> **compute-path** lever — it replaces dequant+FP16-MAC with table lookups. But at M=8
> compute is **80% idle**, so cutting it changes wall-time by **~0**. LUT reads the **same
> 4-bit weight bytes** (no byte win; at iso-PPL its LUT metadata is comparable-to-*larger*
> than Marlin's group scales), with **no better memory schedule** (no utilisation win). The
> entire +29.8% verify-GEMM headroom at M=8 is a **bandwidth-UTILISATION** ceiling that
> belongs to **SplitK**, not LUT. **LUT-GEMM is dominated by SplitK, not additive to it.**

| metric | value |
|---|--:|
| **PRIMARY `lut_gemm_m8_speedup_vs_marlin_pct`** (best-case, iso-bytes) | **0.0%** |
| realistic (LUT-GEMM BCQ B=4 at iso-PPL g=32 metadata) | **−24.7%** |
| per-group non-uniform codebook at iso-PPL (g=32) | −62.2% |
| **TEST `lut_gemm_ppl_projected`** (4-bit non-uniform, iso-bit-width) | **2.3777** (cap 2.42; moot) |
| Marlin M=8 (MEASURED #68): HBM util / compute util | 77.1% / 20.2% → **BW-bound** |
| bandwidth-gap ceiling 77.1%→100% HBM (= **SplitK's**, not LUT's) | **+29.8%** |
| compute floor as % of Marlin time (hidden under memory stalls) | 20.2% |
| INT8-TC substrate (`torch._int_mm`) at M=8 / M=16 | **refused (M must be > 16)** |
| #109 corner at LUT alone (0%) / needs SplitK 14.34% for 500 | **466.8 → no clear** |
| SplitK 8.5% alone / SplitK + LUT combined | 487.0 / **487.0 (LUT adds +0.00)** |

**One-line result:** LUT-GEMM attacks the one resource that is **slack** at M=8 (compute)
and touches **neither** resource that **binds** (bandwidth utilisation, byte count). Its
M=8 speedup ceiling is **0%**, realistic **negative**. It cannot give #109's corner its
missing 500 margin, and it is strictly worse than SplitK per unit build effort.

## 1. The roofline framing — why a compute lever can't move a bandwidth-bound GEMM

verify-GEMM time = **bytes / achieved_bandwidth**. The committed `tree_free_500_ceiling`
model classifies the two factors exactly:

- **achieved_bandwidth (UTILISATION).** Marlin runs at **77.1%** of the 600 GB/s HBM peak
  (#68, measured 462 GB/s aggregate). Closing the gap to 100% is the **+29.8%** ceiling.
  It needs **better memory scheduling** — more K-dim thread blocks → **SplitK** (ubel #108).
- **bytes (BYTE COUNT).** 4-bit weight + scale/codebook metadata + activations/outputs. The
  byte lever is double-quant/palette (#104; INT8-dq KILLED, palette ~0.3%).

**Where does LUT-GEMM act?** On *neither*:

1. **No utilisation win.** LUT reads the same int4 weight layout. Its memory access pattern
   is not inherently better than Marlin's hand-tuned coalesced 128-byte pipeline; LUT's
   activation-indexed lookups add *indirection* that, if anything, hurts coalescing. So its
   achieved bandwidth ≤ Marlin's. It does **not** claim the +29.8% utilisation headroom.
2. **No byte win.** LUT does **not** reduce weight bit-width (still 4-bit). Its metadata at
   iso-PPL is comparable-to-larger than Marlin's group scales (§2). Byte-neutral at best,
   byte-negative realistically.
3. **Only a compute win** — replacing dequant+FP16-MAC with lookups. But at M=8 the GEMM
   uses **20.2%** of compute peak; the compute floor is **20.2% of Marlin's wall-time**,
   fully **hidden** under memory stalls. Zeroing compute can't speed a bandwidth-bound
   kernel.

This is the crux: **the M=8 headroom is bandwidth (SplitK's home); LUT-GEMM is a compute
optimisation for a regime that is 80% compute-idle.** Wrong tool.

## 2. The byte model — LUT does not read fewer bytes at iso-PPL

All variants read the **same 4-bit weight matrix** (`0.5·out·in`). They differ only in
metadata. BW-bound time = bytes / (Marlin's measured achieved BW), so the LUT/Marlin time
ratio = the byte ratio. Aggregate over the 6 real verify shapes (#68):

| byte model | metadata | M=8 speedup vs Marlin |
|---|---|--:|
| `lut_iso` (optimistic: metadata == Marlin scales) | 1× group scales | **0.0%** |
| `lut_bcq_b4_g32` (LUT-GEMM BCQ, iso-PPL g=32) | **4×** scales (B=4 planes) | **−24.7%** |
| `lut_nu_g32` (per-group non-uniform codebook, iso-PPL) | 16-entry LUT **per group** | **−62.2%** |
| `lut_nu_perchannel` (per-channel codebook) | 16-entry LUT per row | +10.2% † |

† The only "positive" row buys its win by adopting **coarser** per-channel quantization
(less metadata than g=32) — a **granularity/byte trade Marlin could also take**, and a PPL
risk (per-channel int4 is far coarser than the deployed g=32). It is **not** a property of
the LUT lookup mechanism; it is the #104/palette byte lever wearing a LUT hat. At true
iso-PPL (g=32), LUT metadata is **4×–16×** Marlin's → strongly byte-negative.

So the honest **LUT-mechanism** ceiling (iso-quantization, iso-bytes) is **0%**, and any
realistic iso-PPL LUT format is **−25% to −62%** in this BW-bound GEMM.

## 3. INT8 tensor-core substrate — measured: not available at M=8

The hypothesis posits "INT8-TensorCore LUT lookups." The Ampere INT8 MMA is **m16n8k32** —
minimum M-tile **16**. At M=8 it is structurally tile-underfilled. **Measured on the A10G**
(`torch._int_mm`, cuBLASLt IMMA):

| shape | M=8 | M=16 | M=32 | M=64 |
|---|---|---|--:|--:|
| gate_up 2560→20480 | **refused** | **refused** | 203.6 µs | 213.2 µs |
| down 10240→2560 | **refused** | **refused** | 98.0 µs | 177.2 µs |

`RuntimeError: self.size(0) needs to be greater than 16` for M≤16; M=17/24 →
`CUBLAS_STATUS_NOT_SUPPORTED` (not 8-aligned). The INT8-TC GEMM substrate a TC-LUT kernel
would issue **does not serve the M=8 verify width at all**. Where it does run (M=32),
int8 gate_up = **203.6 µs vs Marlin int4's ~67 µs at M=32** — ~3× slower, because int8
reads **2× the bytes**. In the BW-bound regime, **byte count is destiny**: an INT8-substrate
LUT cannot beat 4-bit Marlin.

## 4. Greedy-safety (by construction) + PPL gate

Under **kanna #96**, the official greedy gate is **self-referential per checkpoint**
(program.md 27-28: token-identical to plain greedy AR *for the submitted checkpoint*), so
any **deterministic** verify kernel is greedy-safe by the acceptance rule — kernel-agnostic.
LUT-GEMM is deterministic ⇒ greedy-safe. The only numerics risk is **PPL**: a 4-bit
non-uniform LUT (GANQ) is at least as expressive as uniform 4-bit GPTQ at iso-bit-width, so
PPL projects **≈2.38** (≤ the 2.42 cap, cushion ~0.04). **PPL holds — but it is moot**: the
speed ceiling is ≤0, so there is nothing to gate.

## 5. Step-4 — does it give #109's corner its missing 500 margin? (No)

A verify-GEMM speedup `s` enters denken #109's ship model identically to SplitK:
`vg = 0.53·(1−f_dq)/(1+s)`. The conservative corner needs **SplitK ≥ 14.34%** for a
confident 500.

| scenario | corner official TPS | clears 500? |
|---|--:|:--:|
| LUT-GEMM alone (best-case 0%) | **466.8** | no |
| SplitK alone 8.5% (ubel plausible) | 487.0 | no |
| **SplitK + LUT-GEMM combined** | **487.0** | no |

**LUT adds +0.00 TPS on top of SplitK.** They do **not stack**: both speed up the *same*
BW-bound verify-GEMM; once SplitK saturates HBM the kernel is still BW-bound on the same
bytes, so LUT (a compute lever) contributes ~0. **LUT-GEMM is dominated by SplitK** on
official-TPS-per-build-effort.

## 6. What this rules out, and the climb-ROI ranking (feeds fern #111)

- **Rules OUT** "LUT-GEMM is a fresh kernel ceiling for the verify-GEMM." It is not — at
  M=8 it is a compute optimisation for a bandwidth-bound, compute-idle GEMM. The
  researcher-agent's generic "+12–22%" projection is **LUT-GEMM's M=1 GEMV latency win** (a
  *compute/dequant-latency*-bound regime), which **does not transfer** to M=8 BW-bound
  verify. Independent literature pass (LUT-GEMM arXiv:2206.09557, GANQ ICML2025 OpenReview
  pkKQGJ5d99, T-MAC arXiv:2407.00088, Marlin arXiv:2408.11743): **no published sm_86/sm_80
  LUT W4A16 kernel beats Marlin at M=4–16**; GANQ's 2.57× is **RTX-4090 (sm_89) at M=1**;
  T-MAC is **CPU-only**. Verdict corroborated RED (−5% to +3%, ~0%).
- **Do NOT pivot ubel #108 from SplitK to LUT.** SplitK is the correct lever for the M=8
  headroom (it is a *utilisation* lever; the headroom is utilisation). LUT-GEMM ranks
  **below SplitK** in the fern #111 climb-ROI: same verify-GEMM slice, ~0 gain, multi-day
  build, and the INT8-TC substrate isn't even available at M=8.
- **The missing 500 margin must come from elsewhere** — a higher SplitK realization, the
  palette byte lever (#104, ~0.3%), or the tree (land #71). LUT-GEMM is not a source.

## 7. Reproduce

```bash
cd target
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python scripts/profiler/lut_gemm_feasibility.py --int8-tc-probe --wandb
```

**Inputs:** `research/spec_cost_model/verify_gemm_roofline.json` (denken #68 MEASURED Marlin
per-shape M=8 times/bytes). **Outputs:** `lut_gemm_feasibility_results.json`, this report.
**Peak GPU mem:** negligible (no model load; the INT8-TC probe times bare `torch._int_mm`).
**Env:** A10G sm_86, torch 2.11.0+cu130. **W&B:** group `lut-gemm-feasibility`, run
`htk6wnof`. Analytic core is CPU-only; the GPU probe is corroboration only. Local, no HF Job.

## 8. Caveats

1. **The LUT kernel is not built or microbenched** (PR Step-2 fallback): IST-DASLab/GANQ is
   research-grade sm_89 CUDA with no sm_86 wheel, T-MAC is CPU-only, and a sm_86 LUT-TC
   kernel is exactly the multi-day ubel-class build this gate decides whether to fund. The
   byte-counting roofline (LUT reads ≥ the same int4 bytes at iso-PPL → BW-bound floor ≥
   Marlin's) is **decisive without a build**; the INT8-TC probe corroborates the tile-floor
   sub-claim on the real card.
2. **Byte model is iso-PPL-conservative to LUT.** The `lut_iso` 0% assumes LUT metadata ==
   Marlin scales (best case). Real LUT-GEMM BCQ/codebook metadata at g=32 is 4×–16× larger
   → the realistic number is negative.
3. **A non-uniform 4-bit requant could shave bytes via coarser grouping** (the +10.2%
   per-channel row) — but that is the orthogonal #104/palette byte lever (not LUT-specific,
   PPL-risky), already separately sized at ~0.3% realizable.
4. **#68 caveats inherited** (L2 residency on small attn GEMMs; int4 base vs PLE-folded
   weights — shapes/dtype identical). The BW-bound verdict and byte ratios are robust to all.
