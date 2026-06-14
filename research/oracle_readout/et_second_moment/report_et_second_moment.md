<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# E[T] second moment — finite-sample TPS CI + land distribution gate (PR #175 · wirbel)

**PRIMARY** `et_second_moment_self_test_passes` = **True** (18/18 checks, NaN-clean)
**TEST** `tps_finite_sample_ci_halfwidth` = **±10.906 TPS** (both-bugs, 16384-token budget, measured step, τ=1)

## Honest scope
Pure-analytic **CPU-only** second-moment read of wirbel #160's E[T]-DP. No GPU / vLLM / HF Job / submission / kernel build / served-file change. BASELINE stays 481.53; **0 TPS**; greedy untouched by construction. Imports #160 (`x8vffgbs`) topology + `score_tree_depthrank`, #165 (`laxllfjl`) composed ceiling, #170 (`ne7p642c`) over-accept locus — **does NOT re-derive any of them**. The second moment is read off the **same `reach[]` object** that yields the first moment. **NOT open2** (tree economics, not drafter architecture). **Does not authorize a launch.**

## The model (imported; the second moment falls out of the first-moment object)
The descent walk commits `L = (edges descended) + 1 bonus` tokens per step, so `E[L] = E[T]`. #160's `score_tree_depthrank` gives `reach[c] = P(reach node c)` (path-product of edge marginals). The walk **stops** at `c` (deepest reached) with prob `reach[c]·(1 − s[c])`, `s[c]` = sum of `c`'s child-edge marginals. Hence the exact pmf:

```
P(L = k) = Σ_{c : depth_c = k−1} reach[c] · (1 − s[c])     (sums to 1; mean = E[T])
Var[L] = E[L²] − E[L]²        σ_L = √Var[L]
```

Topology (imported #160): **M=32, depth 9, max-branch 3, 7 leaves**.

## 1. Accepted-length pmf (both topologies)
`L` = committed length (incl. bonus); `accepted_drafts = L−1` (the literal "k=0..depth" view). `E[L]=E[T]`, so the mean falls out as the first-moment consistency check.

**both-bugs (E[T]=5.206954, σ_L=3.0354, Var=9.2139):**

| committed L | accepted drafts | P(L) |
|---|---|---|
| 1 | 0 | 0.116248 |
| 2 | 1 | 0.112633 |
| 3 | 2 | 0.159055 |
| 4 | 3 | 0.076578 |
| 5 | 4 | 0.115901 |
| 6 | 5 | 0.069122 |
| 7 | 6 | 0.076719 |
| 8 | 7 | 0.065641 |
| 9 | 8 | 0.055451 |
| 10 | 9 | 0.152650 |

**descent-only (E[T]=5.056405, σ_L=3.0593, Var=9.3592):**

| committed L | accepted drafts | P(L) |
|---|---|---|
| 1 | 0 | 0.137564 |
| 2 | 1 | 0.114347 |
| 3 | 2 | 0.160703 |
| 4 | 3 | 0.074400 |
| 5 | 4 | 0.110319 |
| 6 | 5 | 0.066336 |
| 7 | 6 | 0.072978 |
| 8 | 7 | 0.062428 |
| 9 | 8 | 0.058695 |
| 10 | 9 | 0.142231 |

First-moment consistency: `Σ_k k·P(L=k)` = 5.206954309 (both-bugs) / 5.056404569 (descent-only) — reproduces the imported #160 ceilings exactly. `Σ_k P(L=k)` = 1.000000000000.

## 2. DP-exactness certificate
Two **independent** enumerations of the walk's stop-node distribution — a recursive DFS path-product and a per-node parent-walk — match the DP pmf to **max-abs-diff 0.0e+00** (DFS) / **2.8e-17** (parent-walk), tol 1e-12. `dp_distribution_exact` = **True**. A 2,000,000-trial Monte-Carlo histogram agrees to 5.6e-04. The propagated second moment is therefore exact, not a DP artifact.

## 3. Second moment → finite-benchmark TPS CI
Budget `B=16384` tokens (128×128, PR contract); `N_steps ≈ B/E[T]`. `SE[L̄]=σ_L/√N_steps`; `official=K_cal·(L̄/step)·τ` is linear in `L̄`, so the 95 % half-width is `1.96·(K_cal·τ/step)·SE[L̄]`.

| topology | E[T] | σ_L | N_steps | central TPS | ±95% half (TPS) | 95% CI | lower clears 500? |
|---|---|---|---|---|---|---|---|
| both-bugs | 5.2070 | 3.0354 | 3147 | 535.43 | ±10.906 | [524.53, 546.34] | **True** |
| descent-only | 5.0564 | 3.0593 | 3240 | 519.95 | ±10.832 | [509.12, 530.78] | **True** |

**vs lawine #168's ±2.4 TPS roofline↔overlap band:** the both-bugs finite-sample half-width ±10.91 is **LARGER** (4.54× lawine's band) — i.e. the single-draw sampling scatter is the larger of the two terms.

**Budget sensitivity:** at the 512-token/prompt budget (B=65536, the #170 convention) the half-width shrinks to ±5.453 TPS (∝ 1/√N_steps). The PR's 16384-token contract is the conservative (wider) case.

## 4. Quadrature composition with kanna #159 (σ_hw) + land #71 distribution gate
**(a)** Total single-shot TPS variance = this accept-length **numerator** term ⊕ kanna #159's σ_hw **denominator** step-jitter term (independent → quadrature):

```
halfwidth_total = √( halfwidth_acceptlen²  +  halfwidth_σ_hw² )
                = √( 10.906²  +  σ_hw_term² )      ← σ_hw slot ARMED (kanna #159, pending)
```

The accept-length half-width ±10.906 TPS is supplied here; plug kanna #159's measured σ_hw TPS half-width into the armed slot to close the total single-shot CI.

**(b)** land #71 distributional gate (validates the build at the **shape** level):

```
land_histogram_in_band(measured_pmf, predicted_pmf, tol=0.02) -> bool
    # normalize(measured); return max_k |measured[k] − predicted[k]| <= tol
```

Composition with #170: **#170 = mean gate** ("is E[T] trustworthy, not over-accept-inflated?"); **this = distribution gate** ("does the whole accepted-length shape match prediction?"). A build bug that preserves the mean but distorts the histogram passes #170 yet fails this — together a **mean + distribution** build gate. (Demo: a mean-preserving 5 %-mass shape distortion is caught — measured mean 5.2070 ≈ predicted 5.2070, but `in_band=False`.)

## 5. Self-validate (PRIMARY)
18/18 checks pass: DP mean reproduces 5.0564/5.2070 and equals `score_tree_depthrank`; pmf sums to 1 and is non-negative; brute-force exactness (two enumerations); σ_L ≥ 0; CI brackets the point estimate; lower-clears-500 verdict explicit for both topologies; NaN-clean. **`et_second_moment_self_test_passes` = True**.

## Hand-off
This is the **finite-sample sampling-uncertainty stamp** for fern #167's launch packet — how much the single irreversible benchmark draw could scatter from the central TPS by chance alone, distinct from every input-band (fern #174) and modeling bound (denken #172 / wirbel #170) already in the packet. Composes in quadrature with kanna #159 (σ_hw) for the total single-shot TPS CI, and hands land #71 a distributional readout gate alongside wirbel #170's mean gate. Does **not** authorize a launch.

## Public / banked evidence used
- wirbel #160 (`x8vffgbs`): descent E[T]-DP — topology + measured rising spine + `score_tree_depthrank` (imported; second moment read off the same `reach[]`).
- wirbel #165 (`laxllfjl`): composed ceiling 5.206954 (both-bugs anchor).
- wirbel #170 (`ne7p642c`): over-accept locus — the **mean** trustworthiness gate this **distribution** gate composes with.
- lawine #168: measured step 1.2182 + the ±2.4 TPS roofline↔overlap band compared against.
- kanna #159: σ_hw step-jitter — the denominator quadrature twin (armed, pending).
