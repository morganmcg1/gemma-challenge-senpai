STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["5kpb73tb"],"primary_metric":{"name":"gap_decomp_self_test_passes","value":1},"test_metric":{"name":"irreducible_gap_floor_pct","value":0.6334},"headline":{"irreducible_gap_floor_pct":0.6334,"clears_3p2_knife_edge":true,"gap_bucket_acceptance_pct":85.25,"gap_bucket_ctxlen_pct":14.75,"gap_bucket_outlen_pct":0.0,"gap_bucket_irreducible_pct":0.0,"gap_shrink_contribution_per_coverage":489.76,"gap_after_max_coverage_retrain":1.142,"coverage_target_for_3p2":0.9011,"demand_side_route_has_path":true,"verdict_band":"GREEN_demand_side_route_has_path"}}

## Results

**GREEN — the demand-side route HAS a path.** The irreducible gap floor is **0.633%** (central), with a banked-convention corner of **0.0%** and a pessimistic corner of **1.65%** — every corner **clears the 3.2% knife-edge** with ≥1.5pp of margin. The 4.295% gap is ~85–100% **acceptance** (coverage-addressable); the residual is a small, roofline-bounded context-length term. **denken #377's sizing has a validated ceiling.**

### The decisive structural finding (refutes the card's hypothesis d)

The card hypothesized that a *fixed numerics/framework tax* might be the irreducible floor capping the gap above 3.2%. **It is not — that tax contributes exactly 0 to the GAP.** Writing the deployed step as `T_step = B + A` (B = body GEMM + lm_head + framework/sampler/batch-invariance tax, ctx- and distribution-**independent**; A = attention, ctx-dependent):

```
gap = 1 − TPS_priv/TPS_pub
    = 1 − (E[T]_priv/E[T]_pub)·((B+A_pub)/(B+A_priv))
    = g_a + r_a·g_s                          (exact additive identity)
g_s = (A_priv − A_pub)/(B + A_priv)          ← B CANCELS in the numerator
```

B (the numerics tax) sits only in the **denominator** of `g_s` — a larger fixed tax *dilutes* the context-length sensitivity, it does **not create gap**. The numerics tax is a floor on **absolute** TPS (it is why even the public side is "only" 481.53), **not** a floor on the public→private gap. This is decision-critical: it means the irreducible floor is the (small) context-length bucket, not the (large, fixed) numerics tax.

### The four buckets (central ΔP = 50 tok; sum to 4.295% exactly, residual 6.3e-17)

| Bucket | abs % of TPS | % of the 4.295% gap | Coverage-addressable? |
|---|---:|---:|---|
| **(a) acceptance** | 3.661% | **85.25%** | **YES** — soft-KD + reasoning-trace retrain |
| **(b) ctxlen** | 0.633% | **14.75%** | No — prompt-length dist (the irreducible floor) |
| **(c) outlen** | 0.000% | **0.00%** | N/A — output fixed at 512 tok (#282), no shift |
| **(d) numerics** | 0.000% | **0.00%** | N/A — cancels in the public→private step difference |

- **(c) = 0 hard:** the speed benchmark generates a fixed 512 output tokens for every prompt (#282 `measured_result`: `n_completion_tokens=512` ×128), so there is no output-length distribution shift to amortize prefill differently.
- **(d) = 0 hard:** shown above — identical B on both distributions cancels in the step difference.

### Irreducible floor & the 3.2% knife-edge

`irreducible_gap_floor_pct` = gap remaining after a **perfect** coverage retrain (closes bucket a) = ctxlen + outlen + numerics = **0.633%** (central).

| Corner | private prompt shift ΔP | irreducible floor | clears 3.2%? |
|---|---:|---:|:---:|
| banked (#318/#373 pure-rho convention) | +0 tok | 0.000% | ✅ |
| **central** (modest held-out shift) | +50 tok | **0.633%** | ✅ |
| pessimistic (public high-decile, #282) | +130 tok | 1.647% | ✅ |
| **breakeven to 3.2%** | **+253 tok** | 3.200% | — |

The ctxlen bucket is pinned by the #257 roofline: attention is only **557.9 µs of the 7983 µs** deployed step (≤7.0% — and this is an **upper bound**, since Gemma's sliding-window layers cap attention beyond the window so most of A does *not* scale with L). To push the floor to the 3.2% knife-edge, private prompts would need to be **+253 tokens longer on average — a ~93% increase over the public mean prompt (~272 tok)**, implausible for a same-methodology held-out split. A second-order effect makes the floor *even safer*: #282 found longer prompts have **higher** E[T], so a private length increase partially self-offsets (slower step, but better acceptance), shrinking the net ctxlen bucket below this attention-only bound.

### Coverage → gap map (reconciles with denken #377)

- `gap_shrink_contribution_per_coverage` = **489.8 TPS per unit coverage** (4.90 TPS per +0.01), from `dE[T]/dcov ≈ #289 a₁-only leverage T2=3.9097`, `dTPS/dcov = T2·K_cal`.
- `coverage_target_for_3p2` = **0.9011** (+0.0108 from the #336 baseline 0.8903) — **well within #336's +0.031 REACHABLE-MARGINAL envelope.**
- **Reconciliation:** denken #377's +5.44 TPS residual needs **Δcov = +0.0111**; my independent knife-edge sizing needs **+0.0108**. The two agree to within 0.0003 coverage. ✅
- `gap_after_max_coverage_retrain` (full #336 +0.031 envelope) = **1.142%** — even the *achievable* retrain clears 3.2% (it closes 73% of the acceptance bucket, leaving residual acceptance + ctxlen).

### Comparison vs. PR baselines

| Quantity | PR baseline | This card |
|---|---|---|
| public→private gap | 4.295% (denken #373 `oqs8lddd`) | 4.2946% reconstructed (resid <0.01pp; matches #318 degradation 4.2946%) |
| knife-edge to private-500 GO | <3.2% | irreducible floor 0.633% << 3.2% → **clears** |
| residual to close | +5.44 TPS central (denken #377) | +0.0108 cov = +5.27 TPS to knife-edge (reconciles +5.44) |
| coverage envelope | +0.031 (#336) | target +0.0108, **3× headroom** |
| official TPS | 481.53 (unchanged) | **+0 (analysis-only)** |

### Honest analysis — what happened

The gap is **genuine OOD acceptance weakness, and it is coverage-addressable to well below the knife-edge.** The decomposition is robust because the verdict rides on the *non-acceptance* buckets, which are structurally pinned rather than estimated: outlen is a hard 0 (fixed-length benchmark), the numerics tax is a hard 0 *to the gap* (cancels), and the ctxlen bucket is roofline-bounded small. The only modeled degree of freedom is the private prompt-length shift ΔP, and the breakeven (+253 tok, a near-doubling) is far outside any plausible same-methodology held-out shift — so the floor verdict does not hinge on the ΔP central choice. This is why I **skipped the optional GPU accept-gap leg**: a per-distribution E[ℓ] measurement would only refine how bucket (a) splits between coverage-recoverable and intrinsic acceptance — it cannot move the floor, which is the non-acceptance residual.

One honesty note on the acceptance bucket: I deliberately did **not** import the #258 decode-proxy private E[T] (3.090) as the benchmark private E[T]. That proxy is an adversarial slice that over-reads the gap 3–5× (19.6% vs the 4.295% benchmark). The benchmark private E[T] is `ρ·E_T_pub = 0.9571·3.844 = 3.679` (#318), and the acceptance bucket is backed out from the *benchmark* gap, not the proxy.

**Bottom line:** the demand-side coverage route is **not** capped by an irreducible floor — #373's cheapest-lever (`coverage_retrain_b`) conclusion stands, and denken #377's Δcoverage sizing now has a confirmed ceiling well inside the #336 envelope. fern #357's composite gets a costed, ceiling-checked demand-side closer.

### Suggested follow-ups

1. **Pin ΔP with one cheap measurement** (de-risks the only modeled DOF): if/when private-VALID prompt token-length stats are available (or measurable on the int4-ct proxy over a private-like slice), drop the real ΔP into bucket (b) to replace the +50 tok central. Even the pessimistic +130 corner clears, so this is confirmation, not a gate.
2. **Optional GPU accept-gap leg** (`--gpu --proxy google/gemma-4-E4B-it-qat-w4a16-ct --measure-accept-gap`, scaffolded but skipped): measure per-distribution E[ℓ] to split bucket (a) into coverage-recoverable vs intrinsic — would sharpen `gap_after_max_coverage_retrain` but not the floor verdict.
3. **Hand denken #377 the validated slope:** `gap_shrink_contribution_per_coverage = 489.8 TPS/unit` and `coverage_target_for_3p2 = 0.9011` are the ceiling-checked inputs for their non-iid accept-model sizing.

### Reproduce

```bash
cd target/ && python research/validity/public_private_gap_decomposition/public_private_gap_decomposition.py \
    --decompose-gap --anchor-289-decay \
    --wandb_group strict-bi-verify-gemm --wandb_name ubel/public-private-gap-decomp
```

- **Self-test:** `gap_decomp_self_test_passes` = **True** (22/22 checks: buckets sum to 4.295%, both hard-zero buckets, floor = non-acceptance buckets, all corners clear 3.2%, monotone in ΔP, coverage knife-edge within #336 envelope + reconciles denken #377, benchmark E[T] distinct from decode proxy, constants exact, NaN-clean).
- **Peak memory:** 12.12 MiB (pure-stdlib CPU-analytic; no torch/numpy in the core path).
- **W&B run:** `5kpb73tb` (group `strict-bi-verify-gemm`).
- **Public-evidence note:** 0 official TPS, 0 HF Job, 0 `--launch`, 0 submission, 0 served-file change. CPU-analytic over banked W&B anchors (denken #373 gap `5k3px8p1`, ubel #318 ρ, #257 roofline, #289 a₁ decay, #336 coverage budget, #282 fixed-output). The optional local-A10G accept-gap leg was scaffolded but **not run** (floor verdict robust without it).
