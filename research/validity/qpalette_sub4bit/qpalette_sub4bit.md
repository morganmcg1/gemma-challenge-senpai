<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #132 — Q-Palette sub-4-bit weights: 🔴 RED at Step 1 (no servable sub-4-bit GEMM on A10G sm_86)

**Hypothesis.** The deployed frontier (int4 W4A16 Marlin, 481.53 official TPS) is
memory-bandwidth-bound at concurrency=1 / 512-token decode. Q-Palette fractional-bit
quantization (NeurIPS 2025, OpenReview `l4F50jpiVH`, SNU) reaches 3.0–3.5 average
bits → 15–25% weight-byte reduction → ~proportional decode-TPS uplift on the plain
M=1 AR path (a spec-decode-INDEPENDENT, ruling-independent lever).

**Gate design (advisor).** Staged, cheap→expensive, KILL-fast. Step 1 (cheapest,
make-or-break): *is there a servable sub-4-bit weight-only GEMM kernel that beats
int4 Marlin's effective M=1 decode bandwidth on A10G sm_86 — in the pinned vLLM
wheel or as a clean drop-in?* If the only route to serve sub-4-bit weights is
dequant-to-bf16 (no native sub-4-bit GEMM), realized TPS ≤ int4 (cf. #113 LUT-GEMM
KILL) → STOP, RED. Do NOT invest in PTQ before confirming the kernel.

**Verdict: 🔴 RED.** No servable sub-4-bit GEMM exists for A10G sm_86, neither in
the pinned wheel nor as a clean drop-in. Step 2 (PPL) and Step 3 (projection) are
NOT run — there is no servable kernel to realize any byte reduction. This banks the
definitive closure of the below-int4 territory for this hardware/wheel.

---

## A. Pinned-wheel inventory (authoritative for "in the wheel")

Static source parse of every weight-only mixed-precision GEMM kernel the pinned
vLLM wheel ships (`vllm/model_executor/kernels/linear/mixed_precision/*.py`,
extends the #122 kernel-mapping). Reproducible, no GPU:
`scripts/validity/qpalette_kernel_gate.py` → `kernel_gate.json`.

| kernel | min weight bits | min cap | runs on sm_86? | sub-4-bit on sm_86? | note |
|---|---|---|---|---|---|
| **marlin** (deployed) | **4** | 75 | yes | **NO** | `uint4/uint4b8/uint8b128`; `num_bits must be 4 or 8` |
| machete | 4 | **90** | **no** | NO | sm_90a Hopper-only (`is_device_capability(90)`); falls back to Marlin on A10G |
| cutlass (W4A8) | 4 | **90** | **no** | NO | sm_90-gated |
| exllama | 4 | 60 | yes | **NO** | `[uint4b8, uint8b128]`; src comment: *"in theory supports uint2b2, uint3b4 too but currently untested so not added"* — **3-bit explicitly disabled**; also fp16-act-only |
| conch | 4 | 80 | yes | NO | `uint4/uint8/uint4b8/uint8b128` |
| dynamic_4bit | 4 | — | yes | NO | `int4` |
| triton_w4a16 | 4 | — | yes | NO | `uint4b8/uint4` |
| allspark | 8 | 80 | yes | NO | `uint8b128` (8-bit only) |
| cpu / xpu | — | — | n/a | NO | non-CUDA backends |

**Min servable weight-bits on sm_86 across the whole wheel = 4. Sub-4-bit servable
kernels in the wheel = 0.** Every CUDA weight-only GEMM caps at 4 bits (or 8). The
single kernel whose *underlying* CUDA could do 3-bit (Exllama / ExLlamaV2) has
`uint3b4` deliberately left out of `SUPPORTED_QUANT_TYPES` and requires fp16
activations (the deployed stack is bf16). Marlin (the baseline) is hard-asserted to
4/8-bit. There is **no native sub-4-bit GEMM to dispatch to**.

## B. Literature "clean drop-in" matrix (the other half of the gate)

Research pass (arxiv/OpenReview/GitHub). "Servable" = a real kernel that, at M=1
decode, streams fewer weight bytes than int4 Marlin AND turns that into wall-clock
decode speedup on sm_86 — *not* dequant-to-fp16, *not* large-batch-only.

| scheme | native low-bit GEMM? | sm_86 kernel? | vLLM path? | M=1 vs **Marlin**? | servable & beats Marlin on sm_86 |
|---|---|---|---|---|---|
| **Q-Palette** (named) | unknown | **no/unknown** (RTX 4090 sm_89 Ada only) | **none** | none (Ada 190–200 tok/s, doesn't transfer) | **NO** |
| Machete | yes | **no** (sm_90a) | in-wheel, arch-gated off | n/a on sm_86 | **NO** |
| QTIP | yes (BW-bound) | yes (RTX3090/A6000) | **none** (standalone) | 119 tok/s @3b vs **fp16** 52.5 (not Marlin) | unknown (no vLLM path) |
| AQLM | yes | partial demo | old demo only | "up to 3x" vs **fp16** (not Marlin) | unknown |
| VPTQ | **no (dequant-to-fp16)** | n/a | planned | n/a (compute-bound) | **NO** (#113 mode) |
| QuIP# | yes (E8 lattice) | unknown | none | no Ampere batch=1 numbers | unknown |
| FLUTE (nearest) | yes (fused LUT, not dequant) | **yes** (Ampere-opt) | **old** (~0.5.x monkeypatch, not the 0.22 `MPLinearKernel` API) | A6000 W3G128 108.1 vs **FLUTE**-W4G128 98.1 (~10% 3b>4b); **no Marlin row** | unknown |

**No scheme is a clean drop-in that beats int4 Marlin at M=1 on sm_86.**

- **Q-Palette itself** (the named lever) ships only RTX 4090 (sm_89 Ada) kernels;
  no documented sm_86 support, no vLLM integration. Porting its CUDA kernels Ada→
  Ampere + writing a vLLM-0.22 quant plugin is a multi-week kernel project, not a
  local probe — and not a "clean drop-in."
- **FLUTE** is the only candidate with both an Ampere fused-LUT GEMM (a real
  native low-bit kernel, *not* dequant-to-fp16) and any vLLM history. But (1) its
  integration targets ~vLLM-0.5.x; the pinned wheel's quant API is the entirely
  different `kernels/linear/mixed_precision/MPLinearKernel` structure → the patch
  does not apply, integration is a from-scratch port; (2) its only published M=1
  comparison is FLUTE-3bit vs **FLUTE**-4bit (~10%), never vs **Marlin**-int4; and
  (3) roofline: Marlin is the SOTA int4 kernel and FLUTE-4bit is slower than
  Marlin-4bit, so FLUTE-3bit ≈ Marlin-4bit — a wash, not the 15–25% the hypothesis
  needs. The 25%-fewer-bytes only converts to TPS if the kernel is as tight as
  Marlin AND stays BW-bound; FLUTE's LUT-decode overhead + general (non-Marlin)
  scheduling erodes exactly that.
- **QTIP/AQLM/QuIP#** have Ampere kernels but no vLLM serving path and no
  proven-vs-Marlin M=1 number; **VPTQ** is the literal #113 dequant-to-fp16 failure
  mode; **Machete** is Hopper-only.

## Why it matters / how to apply

- **Sub-4-bit on Ampere sm_86 is a footprint-saving territory, not a decode-speedup
  territory, for vLLM serving.** The pinned wheel has no sub-4-bit GEMM, and every
  external scheme either is Hopper-only, has no vLLM path, dequants to fp16
  (≤int4 TPS), or has never been shown to beat *Marlin* (only fp16/itself) at M=1.
  The 15–25%-byte-reduction → proportional-TPS premise has **no kernel to realize
  it** without a from-scratch CUDA port whose own roofline (FLUTE-3b ≈ Marlin-4b)
  predicts a wash at best.
- **Do NOT propose Q-Palette/QTIP/AQLM/VPTQ/QuIP# as a sub-4-bit TPS lever on this
  stack.** Q-Palette specifically is Ada-only with no vLLM integration. A green
  sub-4-bit path would require a NEW native sub-4-bit Marlin/Machete-class kernel
  for sm_86 (does not exist in or out of the wheel) — far beyond a local probe.
- **Hardens the int4-Marlin floor.** Combined with #122 (no batch-invariant Marlin)
  and #117 (SplitK W4A16 capped 1.56%), the int4 Marlin GEMM is confirmed as both
  the divergence source AND the practical weight-byte floor on A10G: you cannot go
  below 4 bits with a real decode win, and you cannot make the 4-bit verify
  batch-invariant. The ruling-independent BW lever the PR hoped for does not exist
  at the kernel level on this hardware.

**Result marker.** `primary_metric` `qpalette_projected_official_tps` = None (no
servable kernel → nothing to project); `test_metric`
`qpalette_servable_and_clears_500` = 0. W&B run `g8dgvmkd`
(`kanna/qpalette-sub4bit-kernel-gate`, group `qpalette-sub4bit`).
