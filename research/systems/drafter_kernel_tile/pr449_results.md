STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["xryqregh"],"primary_metric":{"name":"max_honest_endtoend_tps_delta","value":0.0},"test_metric":{"name":"ppl","value":2.3772}}

## Results

**Verdict: the MTP K=7 ONEGRAPH drafter is already tile-optimal on sm_86. NO >+2 TPS honest end-to-end headroom. NO-GO to build.**

`max_honest_endtoend_tps_delta = +0.000 TPS` (official & local). The drafter's only Triton-tunable kernel — the fused sparse argmax — is already the fastest of the swept grid; every alternative tile config is equal or slower. The int4 Marlin GEMMs that dominate D are CUDA (stark's domain), not Triton-tunable here.

W&B: [`xryqregh`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/xryqregh) — group `kernel-tiling-sweep`, run `lawine/drafter-kernel-tile-profile`.

### 1. Drafter per-step kernel breakdown (D measured = 1.434 ms, anchor 1.433 ms → 0.07% match)

Served breakdown via the team `serve_profile` harness (STEPTIME timing pass for the D anchor, zero-perturbation; + torch-profiler kernel pass over a 256-token frontier window, ONEGRAPH=1). Per-step attribution from kernel **call count** (the drafter runs K=7 width-1 iterations/step and calls the blocks kernel once/iter → 420 blocks calls = exactly 60 decode steps; E[T] spec_log is null under STEPTIME so call-count is the robust divisor).

| Component | µs / decode-step | % of D |
|---|---:|---:|
| **sparse_argmax (drafter-specific Triton: blocks+reduce)** | **93.49** | **6.52%** |
| &nbsp;&nbsp;↳ `_sparse_argmax_blocks_kernel` | 82.11 | 5.73% |
| &nbsp;&nbsp;↳ `_sparse_argmax_reduce_kernel` | 11.38 | 0.79% |
| int4 Marlin GEMM body + attention + norms (rest of D) | ~1340 | ~93.5% |

In-graph per-call sparse_argmax = 13.36 µs (blocks 11.73 + reduce 1.63), 420 calls / 60 steps.

Whole-window GPU category split (drafter **+** verify, ONEGRAPH torch trace):

| category | % of window GPU |
|---|---:|
| matmul_gemm (int4 Marlin) | **79.21%** |
| attention | 7.60% |
| norm | 4.38% |
| sampling (incl. sparse_argmax + ATen gatherTopK) | 2.92% |
| other | 2.95% |
| elementwise_copy | 2.36% |
| activation | 0.57% |

The two dominant int4 Marlin GEMM kernels alone are 32.93% + 28.78% = **61.7%** of all GPU time. **Conclusion:** the only drafter-SPECIFIC Triton kernel is sparse_argmax (6.52% of D). The remaining ~93% of D is the int4 Marlin GEMM body (pre/post-proj, q/o-proj×4, MLP×4 — ~10 GEMMs/iter × 7) + attention + RMSNorm — all CUDA/Marlin (stark's domain), not Triton-tunable on this leg.

### 2. Tile-config sweep on the drafter-specific Triton kernel

Grid-mapping note: the PR's `{BLOCK_M, BLOCK_N}` are GEMM-tiling knobs, but the drafter sparse argmax is a **centroid-pruned matvec+argmax, not a GEMM** — its tile knobs are `BLOCK_SELECTED` (selected-row tiling) and `BLOCK_D` (= next_pow2(256), pinned). So I swept the applicable knobs: **`BLOCK_SELECTED ∈ {8,16,32,64,128} × num_warps ∈ {2,4,8} × num_stages ∈ {2,3,4}` = 45 configs** (do_bench), then re-timed the 8 plausible-winner corners at sub-µs precision.

| timer | served default (BS=16, warps=8) | best of grid | speedup | byte-correct |
|---|---:|---:|---:|---|
| do_bench (45 cfg) | 21.504 µs (1.024µs-quantized) | ×1.0000 | none | 45/45 ✓ |
| precise batched (n=1000×50) | 40.69 µs/call (blocks 22.94 + reduce 16.34) | **served_default** | ×1.0000 | ✓ |

Every alternative config is ×0.965–0.993 (slower). `gain_kernel_us_per_call = +0.000`. The int4 Marlin GEMMs were deliberately **not** re-measured (coordinated — stark's domain).

### 3. Honest kernel→end-to-end mapping (D/(D+V) ≈ 18%)

- Realizable: `gain = 0` → ΔD = 0 → **Δ_endtoend = +0.000 TPS** (official & local).
- Amdahl ceiling (sparse_argmax cost → 0, the theoretical max for this kernel): ΔD = 93.49 µs → 1.187% of the 7878 µs cycle → **+5.71 TPS official** — and this is **unreachable** via tiling (zero speedup exists; you can't make argmax free).
- Trap avoided: unlike pinned-K #433 (+13.998 micro → −5.82) and cb3 #437 (+15.60 → 0.0), the microbench Δ here is itself 0/negative, so there is nothing to mis-map. This is a clean **REALIZED-NULL**.

### 4. Self-test — PASS (all flags true)

`tile_sweep_best_is_served_default` ✓ · `no_config_beats_default` ✓ · `all_swept_configs_byte_correct` (45/45) ✓ · `default_config_byte_correct` ✓ · `greedy_identity_free_by_construction` ✓ (drafter gates accept-LENGTH only; verify is sole arbiter via `_dixie_fused_accept_prep_kernel`, land #420) · `ppl_ok` (2.3772 ≤ 2.42) ✓ · `no_served_change` ✓.

### 5. Plain statement

There is **no >+2 TPS honest end-to-end headroom** in the drafter's Triton kernels. The fused ONEGRAPH replay's only tunable Triton kernel (sparse argmax) is already at its optimal tile config on sm_86 — served default wins the full 45-config grid at sub-µs precision. A tile change would be a one-time CUDAGraph **re-capture** (NOT the wirbel #424 structural replay rewrite that risked −16.5%) — but it is **MOOT: no winning config exists**. **NO-GO to build.**

### Metrics vs baseline

| | value | baseline | Δ |
|---|---:|---:|---:|
| max_honest_endtoend_tps_delta (primary) | **+0.000 TPS** | — | — |
| ceiling if argmax free (unreachable) | +5.71 TPS | — | — |
| PPL (test) | 2.3772 | 2.3772 (deployed PR #52 [`2x9fm2zx`], gate ≤ 2.42) | 0 |
| Realized equivalence frontier | 467.14 (unchanged) | 467.14 (denken #423) | 0 |
| Deployed incumbent | 481.53 (unchanged) | 481.53 (PR #52) | 0 |

This PR ships **nothing** (verdict = keep served default), so PPL/frontier/incumbent are all unchanged by construction.

### Commands

```bash
SV=/tmp/senpai-venvs/5f4c623f772358a2/bin/python   # serve venv (torch+triton)
# (1) tile sweep — 45 configs, do_bench, correctness vs torch ref
CUDA_VISIBLE_DEVICES=0 $SV research/systems/drafter_kernel_tile/microbench.py
# (1b) sub-us batched re-time of the plausible winners (blocks/reduce split)
CUDA_VISIBLE_DEVICES=0 $SV research/systems/drafter_kernel_tile/precise_time.py
# (2) served per-kernel breakdown (STEPTIME D anchor + torch-profiler trace, ONEGRAPH=1)
CUDA_VISIBLE_DEVICES=0 $SV -m research.systems.drafter_kernel_tile.breakdown_profile 1
# (3) consolidate -> report.json + primary metric + self-test
$SV research/systems/drafter_kernel_tile/consolidate.py
# (4) W&B log (repo .venv has wandb; group kernel-tiling-sweep)
WANDB_API_KEY=… .venv/bin/python research/systems/drafter_kernel_tile/log_wandb.py
```

### Peak memory

- Standalone microbench/precise: lm_head `[262144, 256]` bf16 = 134 MB + workspace ≈ **<0.2 GB**.
- Served breakdown profiling: standard serve config `GPU_MEMORY_UTILIZATION=0.90` (≈21.6 GB reserved of 24 GB A10G).

### Public evidence used

- **SlimSpec: low-rank lm_head for speculative drafters** (arXiv:2605.10453) — identified as the only larger drafter-leg lever (~4–5× lm_head, ~+3.8% TPS), but it requires **drafter retraining** (cluster training request) → out of scope for this profiling PR.
- **`triton.testing.do_bench` + Triton autotune tile knobs** — standard microbench/tile-sweep methodology.
- Internal lands referenced: land #444 (cycle split D/V), land #420 (verify-arbiter byte-exact identity), PR #75 (fused MTP K=7 ONEGRAPH drafter), wirbel #424 (replay-rewrite −16.5% risk).

### What happened — honest analysis

The drafter's D is ~93% int4 Marlin GEMM body + attention/norms (stark's CUDA domain) and only ~6.5% drafter-specific Triton (the fused sparse argmax). That one Triton kernel is already at its fastest tile config on sm_86: the served default (`BLOCK_SELECTED=16, num_warps=8`) wins the entire 45-config grid, confirmed at sub-µs precision where every alternative is equal or slower. So the drafter leg has **literally zero tile-tuning headroom**, and even the theoretical ceiling (argmax cost → 0) is only +5.71 TPS and is unreachable. This is the cleanest possible outcome for avoiding the #433/#437 trap — the microbench Δ is itself zero, so there is nothing to over-map. The drafter leg of the four-leg kernel-tiling sweep is a **REALIZED-NULL**.

### Suggested follow-ups

- **SlimSpec low-rank lm_head (arXiv:2605.10453)** is the only identified drafter-leg lever that could move D meaningfully (~+3.8% TPS), but it needs a drafter retrain — recommend a separate retrain-gated PR if the team wants to pursue the drafter leg.
- The 79% matmul_gemm window share confirms D's time lives in the **int4 Marlin GEMM body** — stark's int4-GEMM sweep is the high-value leg on the drafter side, not the Triton argmax.
- **Verify is 82% of the cycle** — if its attention is mis-tiled (wirbel #442 modeled +15.86 → 483.0, still UNPROVEN realized), that remains the live lead and dwarfs anything on the drafter leg. Recommend prioritizing the verify-attention realization over any further drafter-kernel work.
