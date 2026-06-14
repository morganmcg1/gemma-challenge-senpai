# FINDING — Split-KV attention roofline audit (PR #69): at the conc=1 latency floor, NEGATIVE

_Submission_: `submissions/fa2sw_precache_kenyan` (int4-pck04 + MTP K=7 + PLE-fold + fa2sw + onegraph + precache + **#43 split-KV verify, ACTIVE**)
_Workload_: M=8 spec-verify decode attention, served ctx 128→1024 (3D split-KV / FlashDecoding path, `max_seqlen_q=1` route).
_W&B_: run **`rajcg6an`**, group `attention-splitkv-audit`. Companions: deployed microbench `research/profiling/splitkv_verify/attention_patched.json`; served re-profile `r0ahjs45` (`frontier_decode_postsplitkv`).

> **Local A10G in-container probe — NOT the official a10g-small TPS.** Absolute µs/GB·s are single-GPU probes against the real deployed Triton kernel; the **roofline ratios** (% of peak BW, occupancy, oracle-vs-deployed n_seg saving) are the trustworthy output. Zero submission-file changes, so PPL/greedy are definitionally unchanged.

## TL;DR — terminal NEGATIVE

The PR frames attention as "the #2 block (19.6%)" from the #30 profile. **#43 split-KV already collapsed it to 7.6% of GPU-busy (the #3 block)** — my own post-#43 re-profile (`r0ahjs45`): 1836→605 µs/step, a 3.03× drop. This audit asks the only remaining question: *is the post-#43 kernel at its roofline, or is there residual lossless headroom?* It is **at the floor**:

1. **Kernel is 100% vLLM-native Triton `unified_attention`** (3D split-KV) — not a custom submission kernel, not Inductor-managed. The fa2sw FA2 router is **inert** (0 FlashAttention kernels in the served trace; vLLM forces TRITON_ATTN for the heterogeneous head_dims). Only tunable surface = dispatch + the global `num_par_softmax_segments=16`.
2. **Occupancy-saturated at the deployed config.** n_seg=16 launches ~96 CTAs ≥ 80 SMs at every served shape; the split-KV (#43) already bought the 4.4× under-occupancy win that motivated the pre-#43 audit.
3. **Sub-peak BW is the irreducible conc=1 latency floor, not slack.** Deployed M=8 aggregate = **96.6 GB/s = 20.0% of measured peak** (16.1% of 600 spec). At conc=1 the per-layer KV read is 0.25–2 MB — too small to saturate HBM regardless of kernel; the optimal 3D path tops out at 18–38% of peak.
4. **The one named lever (a #53-style per-ctx split heuristic) is worth +0.13% TPS and is un-CUDA-graph-able.** Oracle per-(layer,ctx) n_seg saves only **1.66%** of ctx-weighted attention time → **+0.126% TPS ceiling**; deployed n_seg=16 is already optimal at the dominant ctx 256–512. Even this is unrealisable: n_seg is a capture-shape constexpr under onegraph.

→ **Same fail-fast as #65/#67.** Attention is irreducible on this stack; do not burn budget. Even a *free* attention→0 is only +8.2% TPS (de-prioritised per `r0ahjs45`); a *realisable* n_seg tweak is +0.13% and breaks single-graph replay.

## #67 check — what kind of kernel is this?

| question | answer | evidence |
|---|---|---|
| Custom submission kernel? | **No** | served attention = `kernel_unified_attention` (vLLM Triton), 98.1% of attention-category time; #43 is a *dispatch* wrapper (`splitkv_verify_patch.py` spoofs `max_seqlen_q=1`), not a kernel |
| Inductor/compiler-managed? | **No** | stock vLLM Triton template; `inductor_managed=False`. Hand-tunable surface exists in principle (n_seg, block sizes) but is owned by vLLM, not us |
| fa2sw FA2 path active? | **INERT** | 0 FlashAttention/FMHA kernels in the served decode trace; only `reshape_and_cache_kernel_flash`. vLLM forces TRITON_ATTN for heterogeneous head_dims (sliding=256, full=512); FA2 caps at 256 |

The deployed `fa2sw` name is historical: the FA2 sliding router never fires in serving. Harmless (FA2 sliding M=1 = 58 µs vs Triton 12 µs in the bake-off — FA2 would be *slower*), but the only knob #69 leaves us is the split-count `n_seg`.

## Roofline headline (deployed config, M=8 served)

| quantity | value | reference |
|---|--:|---|
| attention % of GPU-busy (post-#43) | **7.6%** | `r0ahjs45` (was 19.6% @ #30) |
| attention µs/step (served) | **605** | `r0ahjs45` (was 1836) |
| deployed M=8 aggregate BW | **96.6 GB/s** | `attention_patched.json` served_verify_aggregate |
| → vs measured peak (482 GB/s copy) | **20.0%** | |
| → vs spec peak (600 GB/s) | **16.1%** | |
| nominal CTAs @ n_seg=16 | **96** (≥ 80 SMs) | occupancy-saturated |
| split-KV speedup already captured (#43) | **4.4× sliding / 3.9× full** | force-2D vs 3D, M=1 |
| free attention→0 TPS ceiling | **+8.2%** | de-prioritised (`r0ahjs45`) |

**Why 20% of peak is the floor, not slack.** conc=1 decode attention reads one sequence's KV (sliding 0.25–1 MB, full 2.2 MB per layer). That is far below the working-set needed to hide HBM latency on 80 SMs; the kernel is **memory-latency-bound**, not bandwidth-bound. The per-M and per-ctx sweeps confirm BW *rises* monotonically with the read size (ctx128 sliding 8% → ctx512 full 28% of peak) — i.e. the kernel gets *more* efficient as the read grows, exactly the latency-bound signature. There is no kernel rewrite that reaches 80% peak at conc=1; that regime only exists at large batch (which this single-stream submission never sees).

## The named lever — per-ctx n_seg split heuristic (à la #53): +0.13% ceiling

`num_par_softmax_segments` trades attention-kernel parallelism against `reduce_segments` merge cost. Swept {1,2,4,8,16,32,64} × {sliding,full} × ctx {128,256,512,1024} at M=8 (powers of 2 only — `reduce_segments` requires a pow-2 segment count). Deployed = global 16.

| shape | deployed n_seg=16 (µs) | best n_seg | best (µs) | speedup |
|---|--:|--:|--:|--:|
| sliding ctx128 | 8.37 | 8 | 8.21 | 1.02× |
| sliding ctx256 | 9.70 | **16** | 9.70 | **1.00× (optimal)** |
| sliding ctx512 | 12.23 | 32 | 12.00 | 1.02× |
| sliding ctx1024 | 14.45 | 32 | 12.99 | 1.11× |
| full ctx128 | 10.89 | 8 | 10.43 | 1.04× |
| full ctx256 | 13.65 | 8 | 12.94 | 1.05× |
| full ctx512 | 16.77 | **16** | 16.77 | **1.00× (optimal)** |
| full ctx1024 | 23.70 | **16** | 23.70 | **1.00× (optimal)** |

The deployed n_seg=16 is **exactly optimal at the served-dominant shapes** (sliding ctx256 = 43.8% of cycles, full ctx512/1024). The only non-trivial gap is sliding ctx1024 (1.11×) which is 2.6% of served cycles. Cost-weighting the per-shape best-vs-deployed saving by this submission's real post-#43 ctx distribution (ctx<256 8.2%, 256–512 89.2%, ≥1024 2.6%; 30 sliding + 7 full layers/cycle):

> **oracle per-shape n_seg saves 1.66% of ctx-weighted attention time → +0.126% TPS ceiling.**

And that ceiling is **unrealisable**:
- It's an *oracle* (perfect per-(layer,ctx) n_seg); a single global value can't capture it — deployed n_seg=16 is already the best single global value at the dominant shapes.
- n_seg is a **capture-shape constexpr** under onegraph (`ONEGRAPH=1`, `LOOPGRAPH_REQUIRE_CAPTURE=1`): a per-ctx n_seg makes the K=7 propose + M=8 verify loop a different graph per ctx, breaking the single-graph replay that is the whole point of the substrate. Decode is 99.4% GPU-bound *because* of that single capture.

## Verdict: NEGATIVE — attention is at the conc=1 roofline

| gate (PR #69) | result |
|---|---|
| At HBM-bandwidth roofline (≥80% peak, occupancy-saturated)? | **occupancy-saturated YES** (96 CTAs ≥ 80 SMs); BW 20% is the **latency floor**, the conc=1 analogue of the roofline |
| Residual headroom (under-saturated BW / low occupancy / ctx-suboptimal split)? | **NO** — occupancy saturated; sub-peak BW is irreducible latency-bound; deployed n_seg optimal at dominant ctx; oracle n_seg = +0.13% and un-graph-able |
| → Verdict | **terminal NEGATIVE** — no lossless fix worth prototyping |

This matches the #30→#43 trajectory: the pre-#43 audit (`fa2sw_attention`) found attention occupancy-bound at M=8 (2D grid, ~6 CTAs) and that *did* have headroom → it motivated #43, which captured 4.4× and dropped attention 19.6%→7.6%. **#43 already harvested the headroom.** The residue is the conc=1 small-read latency floor, which no kernel can cross. No fix prototyped (the only candidate is +0.13% and breaks onegraph).

## Suggested follow-ups (NOT implemented — out of #69 scope)

- **fa2sw dead-config cleanup.** The FA2 sliding router is inert in serving (0 flips). A cleanup PR could drop `FA_SLIDING`/`fa_sliding_patch.py` from the stack — pure simplification, no perf/PPL change. Not a correctness bug; flagged for the advisor.
- **Cross-layer KV read coalescing (YOCO/CLA family).** The only structural attention lever left attacks the *latency floor* itself: 30 sliding layers each re-stream their KV. Sharing/coalescing KV reads across layers would cut HBM traffic, but (a) it changes numerics → fails the lossless 128/128 gate, and (b) the prize is bounded by the +8.2% attention→0 ceiling. De-prioritised vs the GPU-compute levers (stark #47 drafter W8A8, land #9 acceptance).

## Validity / provenance

- **Read-only audit. Zero submission-file changes** (`git status`: only `research/profiling/splitkv_nseg/` added). PPL/greedy/serving definitionally untouched; #43 already validated PPL 2.3767 ≤ 2.42 + greedy-equivalence for this exact stack.
- Probe drives the **real deployed Triton kernel** via `scripts/local_validation/profile_attention.py` helpers (paged KV build, rotary, SDPA reference validation: max_abs_err 6.1e-5). No HF launch (local profiling only, per PR + operator rule).
- Public anchor: attention is universally Triton/FlashDecoding on this model class; no leaderboard submission claims a custom conc=1 attention kernel — corroborating that the floor is structural, not a tuning miss.

## Reproduce

```bash
cd target/
# venv with vllm + cuda (container GPU = index 0):
VENV=/tmp/senpai-venvs/5f4c623f772358a2/bin/python
CUDA_VISIBLE_DEVICES=0 $VENV research/profiling/splitkv_nseg/nseg_sweep.py     # n_seg sweep -> nseg_sweep.json
python research/profiling/splitkv_nseg/aggregate.py                            # ctx-weighted oracle -> roofline_summary.json
python research/profiling/splitkv_nseg/log_wandb.py                            # -> W&B rajcg6an (base python, has wandb)
```
