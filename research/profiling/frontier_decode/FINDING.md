# FINDING — Where decode time goes at the ~420-TPS frontier (next-lever finder)

_Submission_: `submissions/fa2sw_precache_kenyan` (int4-pck04 + MTP drafter K=7 + PLE-fold + fa2sw + onegraph + precache)
_Workload_: conc=1, 128 official sharegpt prompts, output_len 512, **CUDA graphs ON** (not `--enforce-eager`).
_Isolation variants_ (`spec_off`, `lmhead_off`): light 32×256 — they only feed a `verify_gpu_ms` p50 and a GEMM-category trace, both stable with far fewer steps.
_W&B_: run **`07kg6bn7`**, group `frontier-decode-profile` (authoritative; a stale-code intermediate `og7z6w0c` was superseded — see Caveats).

> **Local A10G exploratory probe — NOT the official a10g-small TPS.** Absolute tok/s here are a single-GPU in-container probe; treat them as composition evidence, not a leaderboard number. Public anchor: the leaderboard `osoi5-…-precache` row is ~424.5 TPS on a10g-small.

## Headline

At steady single-stream decode the frontier is **GPU-bound (99.3% of the cycle wall is GPU-busy)**. The mean spec-decode cycle is:

| quantity | ms | source |
|---|---|---|
| drafter forward (GPU) | 1.446 | STEPTIME `kind=draft` p50 |
| verify forward (GPU) | 7.906 | STEPTIME `kind=exec` p50 |
| **GPU-busy / cycle** | **9.352** | drafter + verify |
| host overhead / cycle | 0.064 | cycle wall − GPU-busy |
| **cycle wall (host-to-host)** | **9.416** | exec_cpu + inter-step gap, p50 |

**E_accept (mean acceptance length) = 3.82 tokens/cycle** — agreed across three independent sources: vLLM server-log counters `1 + K·acc/draft` = **3.817**, Prometheus `mean_acceptance_length` = **3.824**, server-log per-interval mean = **3.818**. Draft acceptance rate ≈ **0.40**.

**TPS reconstruction** (local A10G): `E_accept / cycle_wall` = **405 tok/s** (drafter-overlapped upper bound); drafter-inclusive lower bound **351 tok/s**. The whole-run engine meter measures **391 tok/s** — it sits inside the bracket, and the independent `E_accept/steady_tps` cycle-wall estimate (9.76 ms) matches the STEPTIME wall (9.42 ms) to ~3.5%, validating the cycle accounting. (The warm single-burst probe reads 862 tok/s and overstates steady ~2× — reported only for continuity with #22's ~867.)

## Decode GPU-busy composition (share of GPU-busy / cycle)

| component | % of GPU-busy | ms/cycle | measured / inferred |
|---|---:|---:|---|
| **verify body int4-Marlin GEMM** | **53.2%** | 4.97 | trace − drafter − lmhead |
| verify attention (fa2sw) | 19.6% | 1.84 | trace (direct) |
| drafter forward | 15.5% | 1.45 | STEPTIME (direct) |
| verify norm / elementwise | 6.7% | 0.62 | trace (direct) |
| sampling | 2.6% | 0.25 | trace (direct) |
| verify lmhead12k GEMM | 1.0% | 0.09 | isolation 16k↔12k + bandwidth model |

## Cross-checks against priors

- **fableous (drafter ≈ 1.4 ms / verify ≈ 7 ms).** Measured **1.45 / 7.91 ms** — matches. The spec-decode split is unchanged at the frontier.
- **PR #8 int4 base (lm_head GEMV = 26.4% of decode GPU).** At the frontier the 12 288-row pruned head is **1.0%** of GPU-busy. Per-row scaling implies a *full* 262 k head would be **20.5%** — same order as the 26.4% prior, i.e. the pck04c-12k prune cut the head **~21×** and **lm_head is no longer a lever**. Hypothesis confirmed.
- **denken #18 (int4 verify bandwidth-bound, flat-in-M up to M≈16).** Isolation: `spec_off` verify (M=1) = **6.326 ms** vs frontier verify (M=8) = **7.906 ms** — only **+25% for 8× the positions**. Verify is near-flat-in-M / bandwidth-bound, as #18 found. The GEMM category share is also stable across the two (≈88% of the spec_off window is matmul).

## Next lever (named)

**`verify_body_int4_gemm` — the int4-Marlin body GEMM, at 53.2% of decode GPU-busy (~4.97 ms/cycle).** It is the single largest addressable component and dominates the verify forward.

Because the body GEMM is **bandwidth-bound and flat-in-M** (denken #18, re-confirmed above), the lever is **reducing the weight bytes moved per decode step** — e.g. a lower-bit or sparser body-weight format, a faster Marlin schedule, or weight-streaming/residency tricks — **not** packing more tokens per verify (M is already cheap; going M=1→8 costs only 25%). A second-order lever is **attention (fa2sw) at 19.6%**. Drafter (15.5%) is already small and well-amortised by E_accept≈3.8, so spending there has poor leverage.

## Caveats / provenance

- Local A10G single-GPU in-container probe; **absolute TPS is not the official a10g-small metric**. Composition fractions are the trustworthy output.
- Cycle TPS carries a known **±~10% drafter-overlap ambiguity** (the drafter runs as its own scheduler step; whether its GPU time overlaps verify host work is wheel-dependent). The 351–405 tok/s bracket spans it; measured steady 391 confirms partial overlap.
- Raw per-step STEPTIME logs (`server_*_timing.log`) and chrome traces (`trace_*/`) are **local-only and regeneratable** — they are not committed (multi-MB). The committed `frontier_decode_profile.json` carries every derived number; `breakdown.md` is the human table.
- **No served-compute change.** The only edit to the frontier stack is an inert, default-off `PROFILER_CONFIG` env forward in `serve.py` (forwards vLLM's torch-profiler config when set; absent on the leaderboard path). Greedy spec-decode identity is preserved.

## Reproduce

```bash
cd target/
python -m scripts.local_validation.profile_decode \
    --submission submissions/fa2sw_precache_kenyan \
    --num-prompts 128 --output-len 512 \
    --variants frontier,lmhead_off,spec_off \
    --wandb-name wirbel/frontier-decode-profile-128 \
    --wandb-group frontier-decode-profile \
    --out-dir research/profiling/frontier_decode/
```
