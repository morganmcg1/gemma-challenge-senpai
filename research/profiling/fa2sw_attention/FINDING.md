# FINDING — the 19.6% decode-attention lever is occupancy-bound, not bandwidth-bound

_PR #39 (fa2sw attention deep-profile). Submission profiled:
`submissions/fa2sw_precache_kenyan`. Local A10G op-microbench — no server, no
submission, no leaderboard number._

> **Local A10G exploratory probe.** Absolute GB/s and TPS here are single-GPU
> in-container measurements; treat them as composition/efficiency evidence, not a
> leaderboard number. Public anchor: the `osoi5-…-precache` leaderboard row is
> ~424.5 TPS on a10g-small.

## Headline (this contradicts the "attention is near-optimal" prior)

The 19.6% decode-attention component is **100% the vLLM Triton
`kernel_unified_attention`** — the `fa2sw` FlashAttention-2 router is **inert**
(vLLM forces `TRITON_ATTN` for the heterogeneous head dims; a full trace scan
finds zero FA2/FMHA compute kernels). That Triton kernel runs at **4.7% of the
sliding-window KV-bandwidth floor** — it is **occupancy/launch-bound at conc=1,
not bandwidth-bound**. There is a concrete, greedy-**exact** lever worth ~2–4×.

| quantity | value | source |
|---|---:|---|
| KV floor bytes / verify-cycle | 41.84 MB | config + real ctx dist |
| floor time @ measured peak (482 GB/s) | 0.087 ms | bandwidth model |
| **served attention / cycle** | **1.836 ms** | frontier trace (PR #30) |
| **`fa2sw_bandwidth_efficiency_fraction`** | **0.047** | floor / served |
| microbench Σ(30 sliding + 7 full) @M=8 | 2.06 ms | this probe (1.12× of served — validates) |

## Why it's occupancy-bound: the M=8 verify never uses split-KV

vLLM's `unified_attention` dispatches a **3D split-KV (FlashDecoding)** kernel only
when `max_seqlen_q == 1`. The MTP spec-verify processes **M = 1 + K = 8** query
positions, so it falls to the **2D** kernel, which at conc=1 launches only
`q.shape[0]//BLOCK_Q + 1 ≈ 6` CTAs on **80 SMs** (7.5% occupancy). The 2D path is
**latency-bound and flat in M**:

| layer | M=7 | M=17 | M=25 | M=45 |
|---|---:|---:|---:|---:|
| sliding device µs | 53.5 | 53.4 | 53.3 | 53.4 |

Device time does not move from M=7 to M=45 (6.4× more query rows, same KV) → the
kernel is neither compute- nor bandwidth-limited; it is stuck on a fixed occupancy
floor. The drafter (M=1) *does* use 3D split-KV (12.2 µs, 86 GB/s) and is already
efficient — the gap is the verify only.

## The lever, measured directly (not modeled)

At **identical work** (M=1, ctx=528, same paged cache) the only difference between
the 2D and 3D kernels is the launch grid:

| layer | 2D µs | 3D split-KV µs | speedup |
|---|---:|---:|---:|
| sliding | 53.1 | 12.2 | **4.36×** |
| full | 64.6 | 16.5 | **3.91×** |

This is the pure FlashDecoding effect — same bytes, same softmax, exact attention.
The verify forward is pinned on the slow side of this 4× purely because of the
`max_seqlen_q > 1` guard.

## Cross-checks against priors

- **PR #30 (attention = 19.6% of decode GPU-busy, 1.836 ms/cycle).** Reproduced:
  the microbench sums to **2.06 ms/cycle (1.12×)**, and the served kernel is
  confirmed to be `kernel_unified_attention` (98.1% of the attention category;
  the remaining 1.7% is the `reshape_and_cache` KV write).
- **fableous / `fa2sw` premise ("FA2 sliding-window attention").** Refuted at the
  kernel level: FA2 never runs (head_dim 512 > FA2's 256 cap on the 7 full
  layers). In a M=1 bake-off the Triton kernel (12 µs) **beats** FA2 (58 µs,
  4.8×) and torch SDPA (98 µs, 8×) — switching to `fa2sw` would *slow* decode.
- **denken #18 ("verify flat-in-M / bandwidth-bound").** Flat-in-M: **yes**
  (53 µs across M=7→45). Bandwidth-bound: **no** at conc=1 — it is occupancy-bound
  well below the bandwidth ceiling. #18's flat-in-M observation is the *symptom*
  of the occupancy floor, not bandwidth saturation.

## Next lever (named) + TPS projection

**Enable 3D split-KV (FlashDecoding) for the M>1 spec-verify forward.** Patch the
`max_seqlen_q > 1` guard in `vllm.v1.attention.ops.triton_unified_attention` and
extend the per-segment softmax reduction to multiple query rows. It is **exact**
(greedy decode identity preserved), **A10G-feasible** (no new hardware features),
and ~90% present in vLLM (the 3D kernel already exists; only the dispatch and the
multi-row segment reduction need work).

Projection (`TPS_new = 424.5 / (1 − 0.196·saving)`):

| attention saving | TPS | crosses |
|---:|---:|---|
| 25% | 446 | 440 |
| 50% (conservative ~2×) | **471** | 440, 460 |
| 82% (reachable, verify at 3D BW) | **505** | 440, 460, 500 |

A second-order lever (the body int4 GEMM at 53.2%, PR #30) remains the larger
single component, but it is bandwidth-bound and hard to move without a quality
risk; **this attention lever is exact and unusually high-leverage for a 19.6%
component because the kernel change is mostly already written.**

## Verdict

`verdict_attn_reduction_worth_pursuing = **1**`. Attention is **not** near-optimal.
The honest redirect is *not* "drop attention, chase body GEMM" — it is **"flip the
verify onto the split-KV kernel that the drafter already uses."**

## Caveats / provenance

- Local A10G single-GPU in-container microbench; absolute TPS is **not** the
  official a10g-small metric. The composition fractions, the 2D↔3D speedup, and
  the bandwidth-floor ratio are the trustworthy outputs.
- The **reachable** verify saving (82%) assumes the M=8 verify could reach the
  M=1 3D kernel's bandwidth; the M=8 3D kernel does not yet exist (the guard
  blocks it), so this is an upper bound. The **conservative** estimate is ~2×
  (M=8 already has 3× the CTAs of M=1) → 50% saving → ~471 TPS. Both cross 460.
- FA2's paged decode used `num_splits=0` (auto); a forced split might narrow its
  gap, but it is irrelevant to the served path (Triton, already optimal at M=1).
- **No served-compute change.** This PR adds only the read-only profiler
  `scripts/local_validation/profile_attention.py`; the frontier serve path and
  greedy spec-decode identity are untouched.

## Reproduce

```bash
cd target/
python -m scripts.local_validation.profile_decode \
    --profile-mode attention-detail \
    --M-values 1,7,17,25,45 --attn-iters 100 \
    --output research/profiling/fa2sw_attention/attention_detail.json
```
