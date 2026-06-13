# fa2sw attention deep-profile — component breakdown

_Submission profiled_: `submissions/fa2sw_precache_kenyan` (served frontier stack)
_Method_: local A10G op-microbench driving the **real** vLLM Triton attention
kernel (`vllm.v1.attention.ops.triton_unified_attention.unified_attention`) on a
paged KV cache with L2-defeating buffer rotation; device time read with
`torch.profiler` (same instrument as the served 88.6 ms / 19.6% figure).
_Artifact_: `attention_detail.json`. _Captured_: 2026-06-13.
_Local A10G probe — composition/efficiency evidence, NOT an official TPS._

## What the 19.6% "fa2sw attention" actually is

| kernel | ms (trace window) | note |
|---|---:|---|
| `kernel_unified_attention` (vLLM **Triton**) | 88.57 | 98.1% of the attention category |
| `reshape_and_cache_kernel_flash` (KV **write**) | 1.69 | not attention compute |
| `reduce_segments` (3D split-KV merge) | 3.95 | from the **M=1 drafter** path |
| FA2 / FMHA / SDPA compute kernels | **0** | full-trace scan: none present |

**The `fa2sw` FlashAttention-2 router never fires.** vLLM forces `TRITON_ATTN`
for this model's heterogeneous head dims (sliding=256, full=512); FA2 caps at
head_dim 256, so it cannot serve the 7 full layers. The 19.6% is 100% the Triton
unified-attention kernel.

## Numerical validation (vs dense torch-SDPA reference)

| kernel | max_abs_err | mean_abs_err | ref_abs_mean |
|---|---:|---:|---:|
| Triton sliding (hd 256) | 6.1e-5 | 4.0e-6 | 3.4e-3 |
| Triton full (hd 512) | 6.1e-5 | 3.8e-6 | 3.5e-3 |
| FA2 sliding | 6.1e-5 | 3.9e-6 | — |

All kernels are bit-faithful to SDPA at bf16 epsilon → the microbench drives them
correctly.

## Step 1 — per-M sweep @ ctx=528 (served dispatch: M=1→3D, M>1→2D)

| layer | M | path | device µs | GB/s | % measured-peak |
|---|---:|---|---:|---:|---:|
| sliding | 1 | **3D split-KV** | 12.2 | 86.5 | 18.0% |
| sliding | 7 | 2D | 53.5 | 20.7 | 4.3% |
| sliding | 17 | 2D | 53.4 | 22.3 | 4.6% |
| sliding | 25 | 2D | 53.3 | 23.5 | 4.9% |
| sliding | 45 | 2D | 53.4 | 26.5 | 5.5% |
| full | 1 | **3D split-KV** | 16.4 | 132.5 | 27.5% |
| full | 7 | 2D | 65.0 | 35.0 | 7.3% |
| full | 45 | 2D | 65.1 | 44.5 | 9.2% |

**The 2D verify path is pinned at ~53 µs (sliding) / ~65 µs (full) — flat from
M=7 to M=45.** Device time does not grow with M ⇒ the kernel is **occupancy /
launch-latency bound, not compute-bound and not bandwidth-bound**. (wall/device =
1.37: the eager launch gap; device µs is the CUDA-graph-comparable number.)

## Step 2 — SWA bandwidth floor vs achieved

Per verify-cycle the kernel must stream KV once for all 37 layers (flat-in-M):
30 sliding × min(ctx,512) × 2048 B + 7 full × ctx × 4096 B.
Using the real decode context distribution (mean ctx 527.7, mean min(ctx,512)
434.7 from `decode_frontier.jsonl`):

| quantity | value |
|---|---:|
| KV floor bytes / cycle | 41.84 MB |
| floor time @ measured peak (482 GB/s) | 0.087 ms |
| floor time @ spec peak (600 GB/s) | 0.070 ms |
| **served attention / cycle** | **1.836 ms** |
| **bandwidth efficiency (floor / served)** | **4.7%** (vs measured peak) / 3.8% (vs spec) |

**Served attention runs at 4.7% of the bandwidth floor** — 21× above the floor —
because the M=8 spec-verify uses the 2D Triton path (~6 CTAs / 80 SMs).
Cross-check: microbench 30·sliding + 7·full = **2.06 ms/cycle ≈ served 1.836 ms
(1.12×)** → the per-op device times are faithful to the live stack.

## Split-KV headroom (the lever) — 2D vs 3D at identical work (M=1)

| layer | 2D (force) µs | 3D split-KV µs | speedup |
|---|---:|---:|---:|
| sliding | 53.1 | 12.2 | **4.36×** |
| full | 64.6 | 16.5 | **3.91×** |

Same bytes, same math, **only the launch grid differs** — the 2D→3D speedup is
the pure FlashDecoding (split-KV) effect. The verify (M>1) is stuck on 2D because
`unified_attention` gates 3D off for `max_seqlen_q > 1`.

## Step 3 — M=1 kernel bake-off

| kernel | device µs | verdict |
|---|---:|---|
| **Triton unified (3D split-KV)** | **12.2** | **winner — the served kernel** |
| FA2 `flash_attn_varlen_func` (paged) | 58.2 | 4.8× slower (what `fa2sw` would call) |
| torch SDPA (dense) | 97.9 | 8.0× slower |

The served Triton kernel is already the **best** available decode-attention
kernel; switching to FA2 (the inert `fa2sw` path) would make decode *slower*.

## Step 4 — TPS-uplift projection (public anchor 424.5 TPS, attn_frac 0.196)

`TPS_new = 424.5 / (1 − 0.196 · saving)`

| attention saving | TPS | Δ | crosses |
|---:|---:|---:|---|
| 10% | 433.0 | +8.5 | — |
| 25% | 446.4 | +21.9 | 440 |
| 50% | 470.7 | +46.2 | 440, 460 |
| 100% | 528.2 | +103.7 | 440, 460, 500 |

Measured split-KV speedup (4.1× at M=1) ⇒ **reachable verify saving ≈ 82%** if the
M=8 verify reached the 3D kernel's bandwidth; even a conservative 2× (50% saving,
M=8 already has 3× the CTAs of M=1) lands **≈ 471 TPS**.

## Verdict

**`verdict_attn_reduction_worth_pursuing = 1`.** Attention is **not** near-optimal:
it is 21× above its bandwidth floor and the greedy-**exact** split-KV path (already
in vLLM, only gated off for `max_seqlen_q>1`) demonstrably recovers ~4× at M=1.
Primary metric **`fa2sw_bandwidth_efficiency_fraction = 0.047`**.

## Next lever (named)

**Enable 3D split-KV (FlashDecoding) for the M>1 spec-verify forward** — patch the
`max_seqlen_q > 1` guard in `unified_attention` and extend the segment softmax
reduction to multiple query rows. Exact (no quality risk), A10G-feasible, ~2–4×
on the 19.6% attention component → projected **≈ 471–505 TPS** from the 424.5
frontier.
