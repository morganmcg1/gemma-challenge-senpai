<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# int4 decode-step cost model vs verify-width M — drafter TPS ceiling on A10G

**PR:** #18 · **Author:** denken · **Date:** 2026-06-13
**Question:** On the local int4 QAT W4A16 Gemma base (the same checkpoint as the
PR #7 ablation, ~96.89 TPS at M=1), how does the **batched-verify forward**
(one sequence, M = K+1 query positions, shared KV context) scale with M, and what
realistic TPS ceiling does that impose on any int4 drafter (MTP / EAGLE-3 / PARD)?

This is a **pure latency/throughput** study with **synthetic** candidate tokens —
no drafter, no greedy gate, **LOCAL ONLY, no HF Job**. It replaces the literature
ceiling (480–550 TPS) we have been quoting with a measured curve on our hardware.

## Verdict / headline

| quantity (canonical = **graph**, ctx 256) | value |
|---|---|
| M=1 calibration | **11.51 ms → 86.9 TPS** (PR #7 ref 96.89; BASELINE int4 ~95.4) — harness within ~10% |
| verify-step latency M=1 → M=16 | 11.51 → 11.82 ms (**+2.6 %** over 16× the query positions) |
| marginal cost per added query token (M≥4) | **≈0.02 ms/token** — accepted tokens are nearly free |
| **knee M\*** (last M within +10 % of M=1) | **16** (the knee is **beyond** the tested range; *not* the expected ≈4–8) |
| lm_head cost (262k-vocab verify) | **fixed ≈2.85→2.95 ms, ~25 % of the step**, ~flat in M |
| attention share (incl. RoPE+KV-write) | **6 % at M=1 → ~17–18 % at M≥2** (ctx 256); **7 % → ~27 %** (ctx 512) |
| ideal all-accepted ceiling (`K/lat`, K*=15) | **1269 TPS** (theoretical upper bound, not realistic) |
| realistic ceiling, QAT-MTP accept a≈3.3 | **291 TPS @ K\*=3** (matches BASELINE MTP 273–286) |
| realistic ceiling, EAGLE-3-like accept a≈4.5 | **395 TPS @ K\*=5** |
| realistic ceiling, geometric p=0.8 | **411 TPS @ K\*=15** (approaches the ~420 frontier) |

**Three answers this curve gives the team:**

1. **The verify forward is so weight-bandwidth-bound that proposing deeper is
   nearly free up to M=16.** The limit on useful draft depth K is **acceptance
   economics and drafter proposal cost, not verify latency.** Target **K≈3–6**
   because acceptance saturates there (a≈3.3 → K*=3, a≈4.5 → K*=5), *not* because
   deep verify is expensive.
2. **The ~420 public frontier is *not* reachable by verify-amortization alone.**
   At realistic acceptance the amortization ceiling is ~290 TPS (current MTP-like)
   to ~395–411 TPS (EAGLE-3-like / high-p). The remaining gap to 420 lives in the
   **lm_head verify-cost cut** — this model shows the 262k-vocab projection is a
   **fixed ~25 % of every graph step**, exactly what the frontier's `lmhead12k`
   sparse-verify lever attacks — plus precache/warmup.
3. **fa2sw / attention levers are *not* dead under deep spec.** At M=1 conc=1
   (PR #7's regime) attention is ~6 % of the graph step, consistent with PR #7
   finding fa2sw a dead standalone lever. But under any spec depth (M≥2) attention
   rises to **~17 % (ctx 256) and ~27 % (ctx 512)** of the real graph step. This
   **closes the open question from PR #7**: revisit fa2sw *for the spec path*,
   prioritising longer contexts. (Caveat: my `attention` bucket also counts RoPE
   and the KV-cache write, so this is an *upper bound* on the fa2sw-addressable
   core-attention share — see §6.)

## 1. Method

`scripts/profiler/spec_cost_model.py`. Loads the local int4 QAT W4A16 base
`google/gemma-4-E4B-it-qat-w4a16-ct` in vLLM 0.22.0 (`quantization=compressed-tensors`,
bf16, Marlin W4A16 kernel, TRITON_ATTN backend forced by the model's heterogeneous
head dims) and drives vLLM's own `GPUModelRunner._dummy_run` — the blessed
"forward of N tokens" primitive — to issue the exact **1-request, M-query-token**
decode shape with `profile_seq_lens = ctx + M` so attention attends over a real KV
context. The lm_head is timed separately via `model.compute_logits` over all M
positions (verify needs logits at every position) followed by `argmax` (the greedy
target token). Per-step latency is `t_step(M) = t_forward(M) + t_lmhead(M)`.

- **Sweep:** M ∈ {1,2,4,6,8,10,12,16}; KV context ∈ {256, 512}; modes
  {eager, graph}. **200 timed steps after 20 warmup**, median reported.
- **graph mode** pins `cudagraph_mode=PIECEWISE` with
  `cudagraph_capture_sizes` = the exact M sweep, so every M dispatches to a captured
  piecewise graph (verified in the log: `Capturing CUDA graphs … PIECEWISE 8/8`).
  This is the real serving path; **eager** is the no-graph reference.
- **Latency basis** is the *pipelined* GPU-event time (back-to-back enqueue, one
  final sync) — the throughput condition real async-scheduled serving achieves,
  and the basis on which PR #7's 96.89 TPS was measured. A serialized
  isolated-latency number is recorded alongside as a diagnostic (`pipeline_speedup`
  ≈ 1.05).
- **Component shares** are torch.profiler self-device time, categorised exactly
  like the official `gemma_decode_profiler` (GEMM / attention / lm_head / norm /
  activation / elementwise / other).

**GPU-memory isolation (the bug fixed in this PR):** vLLM V1's in-process engine
does **not** release GPU memory on `del llm` + `empty_cache()`, so a second
`LLM()` in the same process fails (`Free memory … 3.1/22.06 GiB < 19.85 GiB`).
The script therefore runs **each mode in its own worker subprocess** (a CUDA-free
orchestrator spawns one worker per mode, merges the partial JSONs, then builds the
cost model). Process exit frees the GPU between modes. This is why the run is one
reproducible command despite needing two fresh engines.

**Environment** (single local A10G; see `project-local-a10g-gpu-env` memory):
`CUDA_VISIBLE_DEVICES=0`, `VLLM_USE_FLASHINFER_SAMPLER=0` (avoids a curand.h JIT
failure; greedy argmax is unaffected), `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process
runner so the profiler sees the decode kernels).

**Peak memory:** model weights **9.81 GiB**, KV cache **~8.5 GiB** (≈215k tokens),
CUDA-graph pool **0.07 GiB**, at `gpu_memory_utilization=0.90` (≈19.85 GiB reserved
of the 22.06 GiB A10G). Well within budget.

## 2. Calibration

Graph M=1 ctx=256 = **11.512 ms → 86.86 TPS**, vs the PR #7 int4 reference
**96.89 TPS** and the BASELINE int4-as-is **~95.4 TPS** — within ~10 %. The
microbenchmark forces attention-metadata creation over a 256-token context and
excludes sampling/detokenisation/scheduling, so a ~10 % offset from the full
generate loop is expected; the **shape** of the curve, not the absolute M=1 point,
is the deliverable. (Eager M=1 = 43.0 ms / 23 TPS is the *no-CUDA-graph* path and
is not the calibration anchor — the graph collapses the per-step overhead, as
PR #7 noted.)

## 3. The verify-step latency curve (strongly sub-linear)

**graph mode** — median `t_step` (ms):

| M | ctx256 ms | Δ/tok ms | ctx512 ms | Δ/tok ms |
|--:|--:|--:|--:|--:|
| 1 | 11.512 | — | 13.054 | — |
| 2 | 11.285 | −0.228 | 12.270 | −0.784 |
| 4 | 11.346 | 0.031 | 12.330 | 0.030 |
| 6 | 11.384 | 0.019 | 12.385 | 0.027 |
| 8 | 11.421 | 0.018 | 12.426 | 0.021 |
| 10 | 11.667 | 0.123 | 12.673 | 0.123 |
| 12 | 11.711 | 0.022 | 12.717 | 0.022 |
| 16 | 11.815 | 0.026 | 12.821 | 0.026 |

The marginal cost of an extra query position is **~0.02 ms** in the flat region —
two orders of magnitude below the ~0.5 ms/token a *new* weight read would cost.
The verify forward amortises one weight read over all M positions essentially for
free up to M=16, so **knee M\* = 16** (latency stays within +10 % of M=1 across the
whole sweep). The hypothesis of a "modest knee M\*≈4–8" is **refuted on this
hardware** — the knee is past the tested range.

The **M=1 → M=2 transition is a regime change**, not amortization: M=1 is a pure
single-token decode (FlashDecode attention path, low attention share); M≥2 is a
"mixed prefill-decode" shape with attention metadata over M positions. In **eager**
this shows as a one-time +8.5 ms jump (43.0 → 51.5 ms at ctx256) and pins the eager
knee metric at M=1 (an artifact of M=1 being *cheaper* than the M≥2 plateau, not of
rising cost). In **graph** the per-step overhead is collapsed so M=1 and M=2 are
within noise (11.51 vs 11.29 ms); only the attention *share* jumps (§4). At ctx512
the M=1 point is slightly *slower* than M=2 in both modes (first-timed-shape /
pure-decode attention routine), after which the curve is flat.

**eager mode** (no-graph reference): ctx256 plateaus at ~50.5–51.8 ms (M≥2);
ctx512 at ~50.5–52.1 ms. Same sub-linearity, ~4.5× slower per step than graph.

## 4. Component shares vs M — the lm_head and attention story

**graph mode, % of full-step GPU-busy:**

| M | lm_head% ctx256 | attn% ctx256 | gemm% ctx256 | lm_head% ctx512 | attn% ctx512 | gemm% ctx512 |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 24.8 | 6.2 | 69.8 | 21.9 | 7.1 | 69.0 |
| 2 | 24.8 | 17.6 | 61.8 | 22.8 | 27.2 | 54.6 |
| 4 | 24.8 | 17.7 | 61.8 | 22.8 | 27.2 | 54.6 |
| 8 | 25.0 | 17.5 | 61.6 | 22.9 | 27.0 | 54.5 |
| 16 | 25.0 | 17.0 | 61.7 | 23.0 | 26.3 | 54.8 |

Two load-bearing facts:

- **lm_head is a fixed ~2.85 → 2.95 ms tax (~25 % of the graph step), almost flat
  in M.** Computing logits at 16 positions costs essentially the same as at 1,
  because the `[M×2560]·[2560×262144]` projection is **bandwidth-bound on the 1.34 GB
  vocab weight** at these M — the weight read dominates the FLOPs until M is far
  larger. So verifying many positions is nearly free on the lm_head too, **and** the
  lm_head is a quarter of every step. This is the precise, quantified case for the
  frontier's **`lmhead12k`** sparse-/reduced-vocab verify lever: cutting the 262k
  projection cost lifts the *whole* ceiling by up to ~25 %.
- **Attention share rises with M and with context.** At M=1 it is ~6–7 % (the PR #7
  conc=1 regime). Under spec depth (M≥2) it is **~17 % at ctx256 and ~27 % at
  ctx512**, roughly flat thereafter. Crossings (graph, both ctx): **>5 % at M=1,
  >10 % at M=2, >15 % at M=2.** GEMM correspondingly falls from ~70 % (M=1) to
  ~62 % (ctx256) / ~55 % (ctx512).

## 5. The TPS ceiling curves (ideal + realistic)

`TPS_ideal(K) = K / latency(M=K+1)` (all K drafted tokens accepted). Because
latency is ~flat, this rises monotonically in K — the all-accepted ceiling is the
*upper bound*, not a realistic operating point:

| | graph ctx256 | graph ctx512 |
|---|--:|--:|
| ideal `K/lat` @ K*=15 | **1269.5 TPS** | 1170.0 TPS |
| ideal-with-bonus `(K+1)/lat` @ K*=15 | 1354.2 TPS | 1248.0 TPS |

**Realistic ceiling** `TPS_real(K) = E[accepted | K] / latency(M=K+1)`, where
flat acceptance = `min(a, K+1)` and geometric (Leviathan i.i.d.) =
`(1−p^(K+1))/(1−p)`:

| acceptance model | graph ctx256: K\* / TPS | graph ctx512: K\* / TPS | anchor |
|---|--:|--:|---|
| flat a=2.2 | 3 / **193.9** | 3 / 178.4 | low MTP |
| flat a=3.3 | 3 / **290.8** | 3 / 267.6 | **QAT-MTP (BASELINE 273–286)** |
| flat a=4.5 | 5 / **395.3** | 5 / 363.4 | EAGLE-3-like |
| geom p=0.6 | 7 / 215.2 | 7 / 197.8 | |
| geom p=0.7 | **15** / 281.2 | 15 / 259.1 | |
| geom p=0.8 | 15 / **411.3** | 15 / 379.0 | high-acceptance |

The realistic band is **~290 TPS (today's MTP-like acceptance) to ~395–411 TPS
(EAGLE-3-like / high-p)**. The ~420 public frontier sits *at or just above* the top
of this verify-amortization band — i.e. it needs both high acceptance **and** the
lm_head/precache levers, consistent with §4.

(Eager realistic ceilings are ~4.5× lower and noisier in K* due to latency jitter
on the flat plateau — e.g. eager ctx512 reports K*=11 purely from a low M=12 sample;
graph is the canonical path and the eager K* values should not be over-read.)

## 6. Knee M\*, optimal K\*, and what bounds K

- **Verify-latency knee M\* = 16** (graph, both ctx): deep proposals are nearly
  free on the verify side. Draft depth is therefore bounded by **(a) acceptance
  saturation** — flat a=3.3 → K*=3, a=4.5 → K*=5; geometric high-p keeps rewarding
  depth with sharply diminishing returns — and **(b) drafter proposal cost**, which
  this study does *not* measure (synthetic candidates). Net practical guidance:
  **target K≈3–6**, set by acceptance, not by verify cost.
- **Attention-bucket caveat:** the `attention` category sums the core attention
  kernel **plus RoPE plus `reshape_and_cache` (the KV write)**. fa2sw (sliding-window
  flash attention) only addresses the core attention compute, so the §4 attention
  shares are an **upper bound** on the fa2sw-addressable fraction. Even discounted,
  ~17–27 % under deep spec at ctx≥256 is large enough that fa2sw is worth re-testing
  on the spec path — the qualitative "not dead under deep spec" conclusion is robust
  to the bucket composition. A clean core-attention-vs-RoPE/KV-write split is the
  obvious follow-up if we decide to spend on fa2sw.

## 7. Reproduce

```bash
cd target
python scripts/profiler/spec_cost_model.py \
  --int4-base google/gemma-4-E4B-it-qat-w4a16-ct \
  --m-sweep 1,2,4,6,8,10,12,16 --ctx-sweep 256,512 --modes eager,graph \
  --steps 200 --warmup 20 --profile-steps 30 \
  --accept-models flat:2.2,flat:3.3,flat:4.5,geom:0.6,geom:0.7,geom:0.8 \
  --output research/spec_cost_model/results.json \
  --wandb_project gemma-challenge-senpai --wandb_entity wandb-applied-ai-team \
  --wandb_group spec-cost-model --wandb_name spec-cost-model-int4
```

Artifacts: `results.json` (raw rows + derived `cost_model`), `run.log`, this report.
W&B run: `wandb-applied-ai-team/gemma-challenge-senpai/runs/pvj0qogp`
(group `spec-cost-model`). Wall time ~22 min on one A10G; no HF Job, no submission.

## 8. Suggested follow-ups (not implemented — PR scope is the cost model)

1. **Quantify the `lmhead12k` lever directly:** re-time `compute_logits` with a
   reduced/sparse verify vocab and confirm the ~25 % step-cost cut this model
   predicts; that is the clearest remaining path from ~395 to the ~420 frontier.
2. **Split core-attention from RoPE/KV-write** in the profiler categories, then
   re-test **fa2sw on the spec path** at ctx∈{256,512,1024} — §3–4 say this is where
   attention-side levers can finally pay, unlike the PR #7 M=1 conc=1 regime.
3. **Overlay the drafter proposal cost** (the term this study omits) to convert the
   verify-only ceiling into an end-to-end K* — couples to fern #16 (EAGLE-3) and the
   MTP drafter; the verify side says K is acceptance-bound, so the drafter cost will
   set the real optimum.
