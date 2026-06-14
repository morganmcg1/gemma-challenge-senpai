<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Drafter-forward roofline — is the 15.5% block bandwidth-bound? (stark #70 premise audit)

**PR:** #75 · **Author:** denken · **Date:** 2026-06-14 · **Builds on:** #68 (verify-GEMM
roofline, MERGED — method + FP16-ceiling correction reused here), #69 (wirbel decode
composition: drafter is now the #2 block at ~15.5–18.1%), #43 (split-KV) ·
**Audits:** stark #70 (int4 drafter weights)
**Question:** wirbel #69 made the **drafter forward (~15.5–18.1%) the largest decode
block after the 53% verify-GEMM**, and stark #70 is building **int4 drafter weights on
the unaudited premise that the drafter is weight-bandwidth-bound at M=1×K=7**. Is that
premise right (→ int4 helps) or is the drafter forward **launch/latency-bound** (→ int4
won't move TPS)?

**LOCAL roofline only** — no HF Job, no submission, no serve-path change, audit-only.
The drafter linears are **unquantized bf16** (`quant_config=None` → `F.linear`/cuBLAS),
so reconstructing the exact-shape `nn.Linear` modules from the real drafter weights and
timing them is **bit-identical in kernel to the served path** (same faithful-modules /
synthetic-activations design as #68). Harness: `scripts/profiler/drafter_forward_roofline.py`.

## Verdict / headline

> **stark #70's micro-premise is TRUE but its macro-conclusion is REFUTED. The drafter
> GEMMs at M=1 ARE in the memory-bound regime (arithmetic intensity ≈ 1.0 FLOP/byte,
> 86× below the A10G ridge; 0.45% of FP16-compute peak) — but they are NOT
> bandwidth-*saturated*: the deployed 7-pass GEMM chain runs at only ~47% of the HBM
> roofline (the dominant small q/o GEMVs sit at ~19%), i.e. latency/occupancy-floored,
> so int4's 3.5× byte cut buys sub-linear time. More decisively, the int4-addressable
> bytes are a tiny slice of decode: the whole 7-pass drafter GEMM chain is 566 µs =
> 4.88% of the 11.6 ms step, and GEMMs are only ~27–31% of the drafter forward (the
> other ~70% is non-GEMM: attention SDPA, the centroid sparse sampler, the 262k-vocab
> masked-embedder gather, norms/rotary/sampling — all untouched by int4 weights). Hard
> ceiling if int4 made every drafter GEMM FREE: +5.13% TPS. Realistic int4: +1.5–3.6%.
> The "bandwidth-bound → int4 ≈ 3.5× faster drafter" framing implies +12.5–14.9% —
> overstated ~3–5×. int4 drafter weights are a LOW-VALUE TPS lever; flag for stark
> BEFORE the build.**

| quantity (A10G, drafter bf16, deployed M=1 × K=7) | value |
|---|--:|
| **primary metric — `drafter_forward_pct_hbm_peak_at_M1K7`** | **47.2%** (7-pass GEMM chain, launch-free graph) |
| measured realizable FP16/BF16 tensor peak (compute ceiling) | **52.1 TFLOPS** (A10G datasheet 70) |
| HBM roofline | 600 GB/s |
| roofline ridge point (peak/BW) | **86.8 FLOP/byte** |
| arithmetic intensity at M=1 | **1.0 FLOP/byte** (86× below ridge) |
| achieved compute at M=1 | 0.24 TFLOP/s = **0.45% of FP16 peak** |
| → **bound at M=1** | **memory-bound regime, but bandwidth-UNDER-saturated (47% peak → latency/occupancy-floored)** |
| 7-pass GEMM chain (deployed onegraph proxy) | **566 µs = 4.88% of an 11.6 ms decode step** |
| GEMM share of the drafter forward (vs #69's 1798–2100 µs budget) | **~27–31%** (the other ~70% is non-GEMM) |
| int4 weight-byte ratio (4-bit + group scales) | **0.284 (~3.52× fewer bytes)** |
| **int4 TPS impact — hard ceiling (GEMMs → 0 µs)** | **+5.13%** |
| **int4 TPS impact — realistic** | **+1.5 … +3.6%** |
| premise-implied naive ("3.5× faster 15.5–18.1% drafter") | +12.5 … +14.9% (**overstated ~3–5×**) |

**One-line result:** the drafter forward is *not* a bandwidth wall you can knock down
with int4 — it is a **small (~4.9% of decode), latency-floored GEMM slice wrapped in a
~70%-non-GEMM forward**, so int4 drafter weights move TPS by low single digits at best.

## 0. Roofline framing — the FP16-ceiling correction carries over and matters here too

The drafter is **currently bf16** (unquantized; `Gemma4MultiTokenPredictor`'s linears
are built with `quant_config=None`). stark #70 would store the weights as int4. Per the
#68 correction, **W4A16 dequantizes to FP16 on-chip and runs FP16 tensor MACs** — so the
hypothetical int4 drafter's **compute ceiling stays the FP16 peak (52.1 TFLOPS measured),
not an int4 peak**. int4 is a *weight-storage* format that cuts HBM bytes ~3.5×; it never
raises the compute ceiling.

Consequence: int4 only helps **if and where the GEMM is bandwidth-bound**. At M=1 the
weight-only AI is ≈ 2 FLOP / 2 bytes = 1 FLOP/byte (bf16); int4 raises it to ≈ 3.5
FLOP/byte — **both 25–87× below the ridge of 86.8**, i.e. the drafter GEMMs stay deeply
in the memory-bound *regime* even after int4. That is the part of stark's premise that is
correct. The error is conflating "in the memory-bound regime" with "bandwidth-saturated
and therefore int4 ≈ 3.5× faster" — §2 and §3 show the chain is only 47%-saturated and a
tiny fraction of decode.

## 1. Method — isolated bf16 GEMM microbenchmark of the real drafter weights

`drafter_forward_roofline.py` loads the deployed drafter (`/tmp/qat-assistant`, =
`drafter-ft/ft-v1-epoch_001`, 78.78 M params / 157 MB, internal hidden 256, backbone
2560, 4 layers, vocab 262144), reconstructs each `nn.Linear` at its **exact served shape
and bf16 dtype** from `model.safetensors` (fusing `gate_proj`+`up_proj` into the served
`gate_up` [4096,256]), and times each `module(x)` with **synthetic bf16 activations
[M,in]** while sweeping M ∈ {1,2,4,8}. One MTP pass = **19 weight GEMMs**:

| role | in → out | count/pass | note |
|---|--:|--:|---|
| pre_projection | 5120 → 256 | 1 | backbone(2×2560) → drafter hidden |
| q_proj (sliding) | 256 → 1024 | 3 | layers 1–3 (sliding attn) |
| o_proj (sliding) | 1024 → 256 | 3 | layers 1–3 |
| q_proj (full) | 256 → 2048 | 1 | layer 4 (full attn) |
| o_proj (full) | 2048 → 256 | 1 | layer 4 |
| gate_up | 256 → 4096 | 4 | MLP, all layers |
| down_proj | 2048 → 256 | 4 | MLP, all layers |
| post_projection | 256 → 2560 | 1 | drafter hidden → backbone |
| centroids_sampler | 256 → 2048 | 1 | centroid sparse sampler |

**Timing basis = launch-free CUDA-graph replay**, matching deployment: the served drafter
runs **inside blake's `onegraph` (`ONEGRAPH:"1"`)** — `sitecustomize.py:propose_onegraph`
captures the *entire* 7-pass propose into ONE graph and `graph.replay()`s it (the centroid
sampler is separately CUDA-graphed too). So the deployed drafter is **launch-free**; it does
**NOT** pay #68's ~55 µs/call eager launch floor. I report the launch-free chain as the
deployed-representative number and the eager chain only as a cross-check. The
compute-ceiling probe times a large square GEMM launch-free → **52.1 TFLOPS** (resolving
the A10G ~70 datasheet vs A10 125 ambiguity, same as #68).

## 2. The drafter GEMMs are memory-bound but NOT bandwidth-saturated at M=1

Launch-free per-GEMM roofline at the deployed M=1:

| GEMM (× count/pass) | t (µs) | achieved GB/s | **% HBM peak** | % FP16 compute peak | bound |
|---|--:|--:|--:|--:|---|
| gate_up 256→4096 (×4) | 5.80 | 363 | **60.5%** | 0.69% | ~bandwidth |
| pre_projection 5120→256 (×1) | 8.64 | 305 | **50.8%** | 0.58% | ~bandwidth |
| post_projection 256→2560 (×1) | 5.55 | 237 | **39.5%** | 0.45% | mid |
| down_proj 2048→256 (×4) | 4.50 | 234 | **39.0%** | 0.45% | mid |
| q_proj full 256→2048 (×1) | 4.54 | 232 | **38.7%** | 0.44% | mid |
| centroids_sampler 256→2048 (×1) | 4.55 | 231 | **38.6%** | 0.44% | mid |
| o_proj full 2048→256 (×1) | 4.65 | 227 | **37.8%** | 0.43% | mid |
| **q_proj sliding 256→1024 (×3)** | 4.61 | 114 | **19.1%** | 0.22% | **latency-floored** |
| **o_proj sliding 1024→256 (×3)** | 4.57 | 115 | **19.2%** | 0.22% | **latency-floored** |
| **aggregate (Σ individual /pass)** | **96.7** | 237 | **39.5%** | 0.45% | mixed |
| **7-pass chain (one graph)** | **566 /step (80.9/pass)** | **283** | **47.2%** | — | mixed |

Two things the premise misses:

1. **Saturation, not regime.** A *bandwidth-bound and saturated* kernel sits at ~75–100%
   of HBM peak (cf. #68's verify-GEMM at 77%). The drafter chain is at **47%**, and its
   most-repeated GEMMs — the small sliding-attn q/o GEMVs (256↔1024, 6 of the 19 per
   pass) — are at **~19%**. At 19% of peak the time is set by **launch/occupancy/latency
   inside the graph**, not by bytes delivered; halving their bytes barely moves them.
   int4 helps the genuinely bandwidth-bound members (gate_up 60%, pre_proj 51%) and
   barely touches the latency-floored majority.
2. **The chain at 47% confirms it.** 7 passes × 22.9 MB = 159.6 MB of dense weight reads;
   at 600 GB/s that is **266 µs if perfectly saturated**, but the measured chain is
   **566 µs → 47% efficiency**. int4 (45.6 MB/step) at the same efficiency → ~161 µs, but
   the latency-floored GEMVs won't scale, so realized int4 time is higher.

## 3. The decisive number — int4-addressable bytes are 4.88% of decode

The drafter GEMMs are not just under-saturated; they are a **small slice of the step**:

| slice | µs | % of 11.6 ms step |
|---|--:|--:|
| 7-pass drafter GEMM chain (deployed graph) | 566 | **4.88%** |
| drafter forward total (wirbel #69 budget) | 1798–2100 | 15.5–18.1% |
| → **GEMM share of the drafter** | | **~27–31%** |
| → **non-GEMM drafter** (attn SDPA, centroid sampler, 262k masked-embed gather, norms/rotary/sampling) | ~1230–1530 | **~69–73%** |

int4 weights touch **only the GEMM weight bytes** — a fraction of the 4.88%. So:

- **Hard ceiling (int4 makes every drafter GEMM cost 0 µs):**
  11600 / (11600 − 566) − 1 = **+5.13% TPS**. Unarguable upper bound.
- **int4 bandwidth-scaling (chain × 0.284, i.e. same 47% efficiency holds):**
  566 → 161 µs, save 405 µs → **+3.62% TPS** — itself optimistic (assumes the
  latency-floored 19% GEMVs also scale by bytes; they won't).
- **Realistic (only the bandwidth-bound GEMMs scale): ≈ +1.5 … +3%.**
- **Premise-implied naive** ("drafter is 15.5–18.1% and bandwidth-bound → int4 ≈ 3.5×
  faster drafter") = save 15.5–18.1% × (1 − 1/3.52) → **+12.5 … +14.9% TPS**.

**The premise overstates the achievable TPS win by ~3–5× against the hard ceiling, and
~4–8× against the realistic estimate.** Even the ceiling (+5.1%) is below what the framing
implies, and int4 cannot reach the ceiling.

## 4. Decompose the 7 passes — are they the binding latency? Is the drafter graphed?

- **Graphed, not eager.** The deployed drafter is **inside `onegraph`** → launch-free
  (`graph.replay()`), and the centroid sampler is separately CUDA-graphed. It does **not**
  pay the ~55 µs/call eager floor. (Cross-check: eager 7-pass chain = 2859 µs vs graph
  566 µs — the ~2.3 ms gap is exactly the launch/dispatch overhead the graph removes. The
  deployed stack already harvested it; this is not free headroom.)
- **The 7 passes ARE sequential and re-read weights 7×.** per-pass GEMM chain = 80.9 µs,
  ×7 = 566 µs. The weights are re-read every pass because the per-pass working set
  (**22.9 MB dense**, **6.5 MB int4**) exceeds the **6 MB A10G L2** — they spill between
  passes either way. int4 (6.5 MB) still > 6 MB L2, so int4 does **not** make the weights
  L2-resident across passes.
- **The GEMMs are NOT the binding drafter latency.** They are ~27–31% of the drafter
  forward; the binding cost is the ~70% non-GEMM (centroid sparse sampler + 262k-vocab
  masked-embedder gather + attention SDPA + sampling). int4 weights do nothing for that.

## 5. Pass-count lever — Step-0 FEASIBILITY (per deliverable 4; NOT implemented)

**Can K=7 draft tokens be produced in fewer weight reads with UNCHANGED drafter outputs?
→ NO, infeasible.** Three routes, all change outputs:

1. **Batch the 7 passes into one wider forward.** Impossible without changing outputs: the
   MTP passes are **autoregressive** — pass *i* consumes the token *sampled* at pass *i−1*
   (`gemma4_mtp.py` / `propose_onegraph` feed each draft token back as the next input). You
   cannot compute token *i+1* before token *i* exists, so the passes are intrinsically
   sequential; there is no single wider GEMM that yields the identical 7-token chain.
2. **Keep weights L2-resident to avoid re-reads.** Needs the per-pass weight set < 6 MB L2.
   Dense is 22.9 MB; int4 is 6.5 MB — **still > L2**. Shrinking below 6 MB means dropping
   weights = a different (smaller) model = different outputs.
3. **Fewer passes (K < 7).** Changes the number of draft tokens → changes accept behavior
   and outputs. That is fern #34's acceptance axis / a K-tuning experiment, not a
   contract-safe "same outputs, fewer reads" lever.

**So pass-count reduction with unchanged outputs has NO headroom.** The real drafter-latency
levers (out of scope; flagged for future work) are: **kernel fusion** within a pass (fuse
the 19 GEMMs + elementwise to cut the launch/occupancy overhead that pins the chain at 47%
and the small GEMVs at 19%), **shrinking the ~70% non-GEMM** (faster centroid sampler / embed
gather / SDPA), or a **fundamentally cheaper drafter**. int4 weights are a minor contributor
to any of these.

## 6. What this means for stark #70 (flag — share before the build)

- **int4 drafter weights are a LOW-VALUE TPS lever:** realistic **+1.5 … +3%**, hard
  ceiling **+5.1%**, versus the premise-implied +12.5–14.9%. The drafter is not a
  bandwidth wall — it is a small, latency-floored GEMM slice (4.88% of decode) inside a
  ~70%-non-GEMM forward.
- **The micro-premise is not wrong, but it doesn't carry the conclusion.** The GEMMs are in
  the memory-bound regime (AI ≈ 1, 0.45% compute) — but they are 47%-saturated, not 80–100%,
  and they are a minority of the drafter. int4 helps the few bandwidth-bound GEMMs (gate_up
  60%, pre_proj 51%) and barely the latency-floored majority (q/o GEMVs 19%).
- **If stark proceeds anyway,** expect ≤ +3% TPS, validate the small-M (256↔1024 GEMV)
  speedup empirically first (those won't benefit), and treat any VRAM saving (157 → ~45 MB,
  ~112 MB freed) as the actual justification — **not** a TPS win. The TPS rationale in the
  premise does not hold.

## 7. Reproduce

```bash
cd target
# timing on the faithful served torch (vLLM venv), JSON only:
/tmp/server-venv/bin/python scripts/profiler/drafter_forward_roofline.py \
  --drafter-dir /tmp/qat-assistant --k 7 --m-sweep 1,2,4,8 \
  --ceiling-m 256,512,1024,2048 --iters 300 --warmup 60 \
  --decode-step-ms 11.6 --drafter-budget-pct-lo 15.5 --drafter-budget-pct-hi 18.1 \
  --frontier-tps 481.53 --no-wandb \
  --output research/spec_cost_model/drafter_forward_roofline.json
# W&B logging (run from /tmp to avoid the local wandb/ dir shadowing the import):
( cd /tmp && /workspace/senpai/target/.venv/bin/python \
  /workspace/senpai/target/scripts/profiler/drafter_forward_roofline.py \
  --log-only /workspace/senpai/target/research/spec_cost_model/drafter_forward_roofline.json \
  --wandb_group drafter-forward-roofline --wandb_entity wandb-applied-ai-team \
  --wandb_name denken/drafter-forward-roofline )
```

**Peak GPU mem:** 0.27 GiB (real drafter weights + activations; timing is negligible).
**Env:** A10G, torch 2.11.0+cu130 (deployed wheel `vllm-0.22.1rc1.dev307+g3e8afdf78.cu129`),
`CUDA_VISIBLE_DEVICES=0`. **Artifacts:** `drafter_forward_roofline.json`, this report.
**W&B:** group `drafter-forward-roofline`, run `uknpbk94`. Local only, no HF Job.

## 8. Caveats

1. **bf16 drafter, faithful by construction.** The served drafter linears are
   `quant_config=None` → `F.linear`/cuBLAS; reconstructing exact-shape bf16 `nn.Linear`
   from the real weights uses the identical kernel/shape/dtype. The roofline is value-
   independent, so this is faithful to the served path.
2. **L2 residency** makes the small attn GEMVs' *absolute* isolated times mildly
   optimistic (< 6 MB survives the replay loop); the chain-in-one-graph number (47% HBM)
   already accounts for cross-GEMM packing and is the deployed-representative figure. The
   under-saturation verdict only hardens in the real forward.
3. **Non-GEMM share is inferred,** not directly timed: GEMM chain 566 µs (measured) vs
   wirbel #69's drafter budget 1798–2100 µs (cited baseline) → ~70% non-GEMM. A direct
   nsys trace of the served onegraph would sharpen the split but cannot change the
   conclusion (int4 touches only the 566 µs GEMM bytes, ≤ 4.88% of decode).
4. **% of a decode step** uses 11.6 ms (the #51/#68 graph-mode step). The deployed stack
   is faster (~8 ms; 481.53 TPS), which makes the GEMM-chain fraction a mild *over*-state
   of int4's reach — i.e. the real int4 TPS win is, if anything, **smaller** than the
   +1.5–3.6% reported.
