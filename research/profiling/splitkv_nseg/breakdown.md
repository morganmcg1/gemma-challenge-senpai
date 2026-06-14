# split-KV n_seg roofline — component breakdown (post-#43 sequel)

_Submission profiled_: `submissions/fa2sw_precache_kenyan` (served frontier stack, **#43 split-KV ACTIVE**)
_Method_: local A10G op-microbench driving the **real** vLLM Triton attention kernel
(`vllm.v1.attention.ops.triton_unified_attention.unified_attention`) on a paged KV
cache with L2-defeating buffer rotation; device time via `torch.profiler`; the
verify batch is forced onto the **3D split-KV** path (`max_seqlen_q=1`, the #43
route) and `num_par_softmax_segments` is swept.
_Artifacts_: `nseg_sweep.json`, `roofline_summary.json`. _Captured_: 2026-06-14.
_Local A10G probe — roofline/efficiency evidence, NOT an official TPS._

## This closes the loop the pre-#43 breakdown opened

`fa2sw_attention/breakdown.md` (pre-#43) ended with **"Next lever = enable 3D
split-KV for the M>1 spec-verify"**. That became **#43** and shipped. The pre-#43
predictions vs what #43 delivered:

| pre-#43 breakdown said | #43 delivered (served `r0ahjs45`) |
|---|---|
| M=8 verify stuck on 2D, ~6 CTAs, 4.7% of BW floor | M=8 verify on 3D, ~96 CTAs, **20% of measured peak** |
| 2D→3D recovers ~4× at M=1 | served attention 1836→**605 µs (3.03×)**, 19.6%→**7.6%** |
| reachable verify saving ≈ 50–82% → ≈471–505 TPS | local +16.4% TPS (391→455); official +4.3% (#30 re-run) |

**The headroom the pre-#43 audit found is spent.** This breakdown asks what's
left in the *same* kernel now that it's on the 3D path: the answer is the
`num_par_softmax_segments` split count, and it's empty.

## n_seg sweep @ M=8 — attn-kernel vs reduce_segments split (device µs)

Deployed = global **n_seg=16**. `reduce_segments` requires a power-of-2 segment
count, so the realisable sweep is {1,2,4,8,16,32,64}. `total = attn_kernel +
reduce_segments`.

### sliding layers (head_dim 256, 30 layers/cycle)

| ctx | n_seg | attn µs | reduce µs | total µs | % meas-peak |
|---|--:|--:|--:|--:|--:|
| 128 | 8 (**best**) | 5.81 | 2.40 | **8.21** | 8.3% |
| 128 | 16 (dep) | 5.89 | 2.48 | 8.37 | 8.1% |
| 256 | 16 (**best=dep**) | 7.00 | 2.70 | **9.70** | 12.6% |
| 512 | 16 (dep) | 9.51 | 2.72 | 12.23 | 18.9% |
| 512 | 32 (**best**) | 8.40 | 3.60 | **12.00** | 19.3% |
| 1024 | 16 (dep) | 11.74 | 2.71 | 14.45 | 16.0% |
| 1024 | 32 (**best**) | 9.40 | 3.59 | **12.99** | 17.8% |

### full layers (head_dim 512, 7 layers/cycle)

| ctx | n_seg | attn µs | reduce µs | total µs | % meas-peak |
|---|--:|--:|--:|--:|--:|
| 128 | 8 (**best**) | 7.91 | 2.51 | **10.43** | 13.0% |
| 128 | 16 (dep) | 7.97 | 2.92 | 10.89 | 12.5% |
| 256 | 8 (**best**) | 10.41 | 2.53 | **12.94** | 18.9% |
| 256 | 16 (dep) | 10.39 | 3.26 | 13.65 | 17.9% |
| 512 | 16 (**best=dep**) | 13.51 | 3.26 | **16.77** | 27.6% |
| 1024 | 16 (**best=dep**) | 19.82 | 3.88 | **23.70** | 37.9% |

**Reading the split.** Raising n_seg shrinks the attention kernel (more parallel
softmax segments) but grows `reduce_segments` (more partials to merge). The
optimum is the balance point. At the deployed n_seg=16 the kernel already launches
**96 CTAs ≥ 80 SMs** at every shape → occupancy is saturated; pushing n_seg=32/64
adds reduce cost faster than it shaves the (already SM-bound) kernel except at the
longest, rarest ctx. The attention kernel's own BW *rises* with ctx (8%→38% of
peak) — the latency-bound signature: bigger reads amortise launch/latency better.

## Per-ctx oracle vs deployed (ctx-weighted by served distribution)

Cycle = 30 sliding + 7 full layers. Weights = post-#43 served decode ctx
distribution (`frontier_decode_postsplitkv/ctx_gate_analysis`).

| ctx | weight | deployed µs/cyc | oracle-best µs/cyc | saving | best n_seg (sliding/full) |
|---|--:|--:|--:|--:|--:|
| 128 | 0.082 | 327.4 | 319.3 | 2.5% | 8 / 8 |
| 256 | 0.438 | 386.4 | 381.4 | 1.3% | **16** / 8 |
| 512 | 0.454 | 484.4 | 477.4 | 1.5% | 32 / **16** |
| 1024 | 0.026 | 599.5 | 555.4 | 7.4% | 32 / **16** |
| **ctx-wtd** | 1.000 | **431.6** | **424.4** | **1.66%** | — |

`reduce_segments` requires pow-2 segments, so the deployed **16** is already the
single global value that is optimal at the two dominant regimes (sliding ctx256 =
43.8% of cycles is `16`; full ctx512/1024 are `16`). The only material per-shape
gap is sliding ctx1024 (32 vs 16), which carries 2.6% weight.

## Oracle → TPS ceiling

Attention is **7.6% of GPU-busy** (`r0ahjs45`); cycle ≈ GPU-busy at 99.4%
GPU-bound. An oracle that recovered the full 1.66% attention-time saving:

`TPS_uplift = 0.076 · 0.0166 / (1 − 0.076 · 0.0166) = ` **+0.126%**.

Bounds that make even this unreachable:
- **oracle, not a knob** — a single global n_seg can't realise per-(layer,ctx)
  optima; deployed 16 is already best-global at the dominant shapes.
- **onegraph constexpr** — n_seg is a CUDA-graph capture-shape constant
  (`ONEGRAPH=1`, `LOOPGRAPH_REQUIRE_CAPTURE=1`). Per-ctx n_seg = a distinct graph
  per ctx, which destroys the single-graph replay that makes decode 99.4%
  GPU-bound in the first place.

## Verdict

**No lossless lever ≥ measurement noise.** Occupancy saturated (96 CTAs ≥ 80 SMs);
sub-peak BW (20% measured) is the irreducible conc=1 small-read latency floor;
deployed n_seg=16 optimal at dominant ctx; oracle n_seg = **+0.13% TPS** and
breaks onegraph. The pre-#43 audit's headroom is spent by #43; the residue is the
floor. **Terminal NEGATIVE** — consistent with #65 (CUDA-graph already deployed)
and #67 (compiler-managed kernels carry no hand-tunable headroom).
