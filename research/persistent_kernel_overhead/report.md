<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Persistent-kernel overhead gate — is the ~32% "other" GPU-idle (reclaimable) or GPU-busy (#65 extended)? (#97)

**PR:** #97 · **Author:** denken · **Date:** 2026-06-14 · **Type:** LOCAL CPU-only
trace analysis (no HF Job, no GPU spend — reads the committed #43 decode trace) ·
**Builds on:** #65 (CUDA-graph → decode 99.41% GPU-bound), #43/#30
frontier_decode (decode composition + STEPTIME), #94 (A10G HBM bus saturated at
82% peak, two memory-bound streams serialize), #85 (tree non-GEMM per-op
budget), #67 (norm/elementwise fusion ceiling <0.5%)

**Question.** The parallel-advisor LEVER 1 prices a persistent-kernel / megakernel
scheduler at **+8–15%** by assuming the **~32% "other/overhead"** in the conc=1
decode budget is reclaimable **GPU-idle** (launch latency + host round-trips +
inter-kernel bubbles). This is in direct tension with **denken #65** (decode is
99.41% GPU-bound → ~0 launch-overhead headroom). Both cannot be naively true. Is
the 32% **GPU-IDLE** (reclaimable by a resident kernel) or **GPU-BUSY** (a long
tail of small real kernels + bus-bound spillover, already on the critical path,
NOT reclaimable, #65 extended)?

## Verdict / headline

> **AMBER ceiling → recommend CLOSE. The coarse "~32% overhead" is ~93% GPU-BUSY
> real kernels — LEVER 1's reclaimable-idle premise is refuted.** A trace-direct
> timeline analysis of the deployed frontier decode step (CUDA graphs ON,
> conc=1) shows the GPU is **97.83% busy / 2.17% idle** in steady decode (mean
> 2.173%, std 0.024% across 39 cycles). Of the coarse 32% "other", only **2.17pp
> is GPU-idle**; **29.8pp is GPU-busy** under-counted attention + drafter +
> norm/sampling/lm_head/elementwise — all real kernels a megakernel reorders but
> cannot remove (the bus is the wall, #94). The residual **2.17% idle is the
> persistent-kernel CEILING** — and it is not a host-stall bucket: it is **~1000
> sub-microsecond inter-kernel gaps per step** (no gap >10 µs in steady decode),
> overwhelmingly **inside the CUDA-graph replay**. It lands in the AMBER band but
> below the 3% build-worth bar, **anti-compounds with the M=32 tree (2.17 →
> 1.64%)**, realizes well under 2% (#67 measured the comparable fusion lever at
> <0.5%), and would cost a full int4-Marlin + fa2sw-attention + sampling
> megakernel to chase. **Not worth building.**

| metric | value | band |
|---|--:|---|
| **`persistent_kernel_reclaimable_pct`** (primary; idle ceiling) | **2.17%** | AMBER (1–3%) |
| `decode_gpu_idle_fraction` (test) | **2.17%** (0.0217) | — |
| TPS reclaim, ceiling (now, M=8) | +2.22% → 492.2 official | — |
| TPS reclaim, ceiling (M=32 tree) | +1.67% | anti-compounds |
| realizable after floor haircut (optimistic) | +1.76% → 490.0 official | low-AMBER |

## Method

Pure-CPU analysis of the committed conc=1 decode chrome trace (torch profiler,
**CUDA graphs ON**, frontier `fa2sw_precache_kenyan` post-split-KV, the deployed
stack) at `research/profiling/frontier_decode_postsplitkv/trace_frontier/`. The
trace carries the full GPU kernel timeline (56 728 kernel events). I:

1. Identified decode cycles from the 53 `execute_context_0(0)_generation_1(8)`
   CUDA-graph-replay annotations (the M=8 verify steps).
2. Isolated steady decode = the longest contiguous run of clean cycles (cycle
   wall < 1.5× median) = **cycles 13–51, 39 cycles, 8.16 ms/cycle** (matches the
   #43 FINDING's 8.011 ms). Warmup cycles 0–10 (~15 ms, 46% idle) and the 221 ms
   request-switch stall at cycle 11 are excluded as non-steady.
3. Merged all GPU kernel intervals → **GPU-busy**; gaps between them →
   **GPU-idle**. Decomposed idle into (a) launch/API, (b) host-sync/round-trip,
   (c) inter-kernel bubble (CPU-runtime overlap), and by location (intra-graph
   vs inter-step).

`decode_gpu_idle_fraction = 2.173% ± 0.024%` (std across 39 cycles; insensitive
to window trimming). Reproduce:
`.venv/bin/python scripts/profiler/persistent_kernel_overhead_gate.py`.

## Step 1 — decompose the "~32% other" into GPU-idle vs GPU-busy

**GPU-IDLE (a+b+c) = 2.17% of the decode step. GPU-BUSY = 97.83%.** The decode
step runs **1049 GPU kernels per 8.16 ms cycle**.

| idle bucket | % of decode wall |
|---|--:|
| (a) kernel-launch / API overhead | 0.53% |
| (b) host-device sync / Python round-trip | 0.33% |
| (c) inter-kernel GPU-idle bubble | 1.31% |
| **total GPU-idle (a+b+c)** | **2.17%** |

| idle by location | % of decode wall |
|---|--:|
| intra-CUDA-graph inter-kernel bubble | **1.51%** |
| inter-step host window (sampling + dispatch + HtoD input copies) | **0.66%** |

The idle is **entirely sub-3 µs gaps** (38% in <0.5 µs, 22% in 0.5–1 µs, 37% in
1–3 µs, 2% in 3–10 µs, **zero gaps >10 µs**). There is **no host-stall bucket**
in steady decode — it is death-by-a-thousand-kernel-boundaries: ~1000 tiny
relaunch gaps per step, mostly between the captured graph's small kernels (the
int4-Marlin GEMVs are 16–37 µs, the fused triton kernels 1.5–3.5 µs, so the
~0.5–2 µs grid-relaunch latency is not hidden).

**GPU-BUSY tail (d).** The non-(big-GEMM/attention/drafter) small-kernel tail is
**13.9% of the decode wall** — norm, elementwise/copy, sampling, KV-cache
writes, activation, small GEMMs. These are **real GPU work**: a megakernel
reorders/fuses them but the SM-cycles and bytes remain (the bus is the wall,
#94: verify pulls 82% HBM peak solo, two memory-bound streams serialize).

## Step 2 — reconciliation

### Coarse-budget split: the "32% other" is 93% GPU-BUSY

The parallel-advisor budget {verify-GEMM 53% / drafter 7% / attn 8% / **other
32%**} lumps under-counted attention (real ≈19.6% pre-split-KV / 7.6% post),
under-counted drafter (real 15.5–18.1%), and norm/sampling/lm_head/elementwise
into one "other 32%" bucket and mislabels it "host-device scheduling, Python
round-trips." The trace shows that of those 32 points, **only 2.17pp is
GPU-idle; 29.8pp (93%) is GPU-busy real kernels.** LEVER 1's premise — that the
32% is reclaimable idle worth +8–15% — is **refuted**.

### #65 reconciliation: agrees, then refines

- **Agreement.** My **inter-step host window = 0.66%** ≈ #65's implied GPU-idle
  **0.59%** (99.41% GPU-bound) ≈ #43 STEPTIME host-overhead 0.59%. Three
  independent measures of the inter-step host gap agree. #65 was right: the
  launch/host-roundtrip headroom a CUDA graph leaves is ~0.6%.
- **Refinement.** My trace-direct method exposes an **additional 1.51%
  intra-CUDA-graph inter-kernel idle** that #65's measurement was blind to. #65
  (and STEPTIME, and CUDA-event timing) measure at **CUDA-graph-step
  granularity** — they time the whole captured replay and count its internal
  inter-kernel gaps as "busy." A persistent megakernel operates at **finer
  sub-kernel granularity** (one resident kernel, no per-op grid launch), so it
  targets exactly the intra-graph boundaries the graph preserves. **This is the
  persistent-kernel's distinct mechanism vs the CUDA graph #65 already has.** It
  is real, but it is 1.5%, not the 8–15% LEVER 1 assumed.

### Anti-compounding with the M=32 tree (land #71)

The per-step idle is ~fixed in absolute terms (kernel-count-bound: verify is
per-layer GEMM+attn regardless of M, just wider kernels). The M=32 tree grows
GPU-busy work per step (verify 6.5 → 9.0 ms, drafter → 1.6 ms, #94 projection)
without adding many kernel boundaries, so the **idle fraction shrinks 2.17% →
1.64%** (TPS reclaim +2.22% → +1.67%). The levers **partially anti-compound**:
the tree (the #1 lever) amortizes the fixed per-step kernel-boundary tax, making
the persistent kernel worth ~0.5pp less once the tree lands. Lever-ordering
input: build the tree first; it eats part of this bucket.

### Realizability haircut

The 2.17% is a **ceiling** assuming a perfect megakernel removes every one of
~1000 heterogeneous kernel boundaries/step. Pulls toward RED:

- **38% of the idle is sub-0.5 µs gaps** at the A10G warp-issue / CUPTI
  resolution floor — partly irreducible even in a resident kernel (instruction
  issue, smem barriers between fused stages).
- **#67 measured the comparable lever** (norm/elementwise kernel fusion, which
  removes exactly these boundaries for a subset) at a **<0.5% ceiling**.
- **#94: the bus is the wall** — a megakernel cannot prefetch into a saturated
  bus (82% used), capping latency-hiding.

Optimistic floor-haircut realizable = **1.76%** (still AMBER); the empirical
#67 cross-check suggests the realized small-kernel-boundary reclaim is well
under 1%.

## Step 3 — gate

| input | value | threshold | result |
|---|--:|---|---|
| `persistent_kernel_reclaimable_pct` (idle ceiling) | **2.17%** | GREEN ≥3 / AMBER 1–3 / RED <1 | **AMBER** |
| realizable (floor haircut, optimistic) | 1.76% | — | low-AMBER |
| under M=32 tree | 1.64% | — | anti-compounds |

**AMBER → recommend CLOSE the persistent-kernel / megakernel lane.** It is not
the clean RED the #65-extension predicted, but it does not clear the build bar:

1. **LEVER 1's premise is refuted.** The 32% is 93% GPU-busy real kernels, not
   reclaimable idle. The real idle is 2.17% (ceiling), not 8–15%.
2. **#65 extends, with a refinement.** #65's GPU-bound finding holds (decode is
   GPU-busy); the megakernel's distinct sub-kernel-granularity mechanism targets
   a finer 1.5% intra-graph bucket #65 couldn't see — real, but sub-build-worth
   and at the #67 fusion-ceiling floor.
3. **It anti-compounds with the #1 lever.** The M=32 tree shrinks the prize to
   1.64% before a megakernel is even written.
4. **Disproportionate build.** Fusing int4-Marlin GEMM + fa2sw attention +
   sampling for Gemma-4-E4B into one resident kernel is megakernel-class
   engineering for a sub-2%, shrinking, bus-capped prize.

**Re-label the decode budget:** the "~32% other/overhead" should read **"~30%
GPU-busy small-kernel tail (norm/sampling/elementwise/KV + under-counted
attention & drafter) + ~2% intra-graph kernel-boundary idle."** It is not a
host-scheduling slack bucket.

## Validity / provenance

- **Read-only CPU analysis. Zero submission-file changes**, no HF Job, no GPU
  spend (the advisor consumes no GPU). Greedy-identity is untouched by
  construction (this is measurement only; a scheduler/fusion change would
  reorder WHEN/WHERE ops run, never WHAT they compute — program.md 27–28 safe).
- Trace = committed #43 frontier decode (CUDA graphs ON, conc=1, the deployed
  stack). Local A10G probe; composition fractions and the idle ratio are the
  trustworthy output, not absolute TPS.
- `decode_gpu_idle_fraction = 2.173% ± 0.024%` (39 steady cycles); robust to
  window trimming (2.170–2.173%).
- W&B: group `persistent-kernel-overhead-gate`, run `denken/...` (`gro3qa0d`).

## Reproduce

```bash
cd target/
.venv/bin/python scripts/profiler/persistent_kernel_overhead_gate.py            # + W&B
.venv/bin/python scripts/profiler/persistent_kernel_overhead_gate.py --no-wandb # offline
```

Artifact: `research/persistent_kernel_overhead/gate.json`.
