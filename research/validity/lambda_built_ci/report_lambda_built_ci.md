<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Margin-aware λ̂_built measurement-CI — the gate-INPUT resolvability stamp (PR #187 · denken)

**PRIMARY** `lambda_built_ci_self_test_passes` = **True** (6/6 conditions, NaN-clean)
**TEST** `lambda_built_halfwidth` = **±0.017140** (WLS/MLE, both-bugs at λ̂=0.905, default 128-prompt × 512-token bench) → CI **[0.8881, 0.9224]**
**W&B** `tloghme9` · group `lambda-built-ci` · peak 27.1 MiB

## Honest scope
Pure-analytic **CPU-only** synthesis. No GPU / vLLM / HF Job / submission / kernel build / served-file change. BASELINE stays **481.53**; **0 TPS**; greedy untouched. **Bank-the-analysis** (PRIMARY = self-test). Imports denken **#183** (`82uisrez`) `q_d(λ)` + inverse map + λ*_LCB=0.9052, denken **#178** (`zjdc7hhh`) forward map / E[T](λ) geometry, wirbel **#175** (`zh1accmi`) accepted-length pmf (→ survival ladder + ±10.9 OUTPUT-side numerator), kanna **#159** σ_hw=4.86 TPS, K_cal=125.268, step 1.2182 — **does NOT re-derive any of them, and does NOT move the bar (#183 owns it)**. **NOT open2. NOT a launch.**

## The question
#183 made the GO gate crisp **on the bar** (build-acceptance λ*_LCB=0.9052 both-bugs). But land #71 will report `λ̂_built` *inferred from a finite-sample measured per-depth ladder* `q̂[2..9]` — each `q̂_d` a binomial over the accepted positions at depth d. So **λ̂_built carries its own sampling CI**, and if that CI straddles 0.9052 the single-run GO is not decisive: land #71 could read a point λ̂=0.91 whose 95% interval dips below the bar. This is the **INPUT-side dual of wirbel #175's OUTPUT-side TPS CI** — #175 priced the noise on the projected TPS; this prices the noise on the measured λ̂ that drives it, and converts it into a measurement protocol (how many prompts land #71 needs so its λ̂ cleanly resolves the bar).

## 1. Measurement model + the depth-thinning trial ladder `n[2..9]`
Per depth d, accepted/total = **Binomial(n_d, q_d)** where n_d = the number of verify-positions that REACH depth d across the 128-prompt × 512-token bench. The walk is a chain (mutually-exclusive siblings), so depths thin out by the **survival** of #175's accepted-length pmf: `n_d = N_steps · S(d)`, `S(d)=P(L≥d)=Σ_{k≥d} pmf[k]`, `N_steps = n_prompts·output_len/E[T]`. At the bar (E[T]=4.9992): **N_steps=13109**.

| depth | q\*(λ\*) | span = q_full−q_floor | S(d)=P(L≥d) | **n_d** | Var[q̂_d]=q(1−q)/n_d | Var[λ̂_d]=Var[q̂]/span² |
|---|---|---|---|---|---|---|
| 2 | 0.7363 | 0.2397 | 0.8838 | **11585** | 1.68e-05 | 2.92e-04 |
| 3 | 0.7724 | 0.2125 | 0.7605 | **9970** | 1.76e-05 | 3.91e-04 |
| 4 | 0.8050 | 0.1764 | 0.5914 | **7752** | 2.03e-05 | 6.51e-04 |
| 5 | 0.8196 | 0.1549 | 0.5097 | **6682** | 2.21e-05 | 9.22e-04 |
| 6 | 0.8200 | 0.1611 | 0.3902 | **5115** | 2.89e-05 | 1.11e-03 |
| 7 | 0.8254 | 0.2306 | 0.3199 | **4194** | 3.44e-05 | 6.46e-04 |
| 8 | 0.8254 | 0.2306 | 0.2413 | **3164** | 4.55e-05 | 8.57e-04 |
| 9 | 0.8254 | 0.2306 | 0.1764 | **2313** | 6.23e-05 | **1.17e-03** |

`n_ladder = [11585, 9970, 7752, 6682, 5115, 4194, 3164, 2313]`. Depth-9 gets **5× fewer trials** than depth-2 — the deep tail of the ladder is the noisy end.

## 2. q̂ → λ̂_built CI (delta-method through the inverse map)
Per depth, `λ̂_d = (q̂_d − q_floor[d])/span_d`, so `Var[λ̂_d] = Var[q̂_d]/span_d²`. Pool across depths two ways:
- **OLS** (= #183's actual simple-mean inverse map): `λ̂ = mean_d λ̂_d` → **half-width ±0.019044**
- **WLS / MLE** (inverse-variance weighting, primary): `λ̂ = Σ w_d λ̂_d / Σ w_d`, `w_d=1/Var[λ̂_d]` → **half-width ±0.017140**

**`lambda_built_halfwidth` = ±0.017140** (WLS), **`lambda_built_ci` = [0.8881, 0.9224]** at the both-bugs operating point λ̂=0.9052. WLS is ~10 % tighter than the simple mean because it down-weights the thin deep depths. **Dominant raw-variance depth = 9** (deepest = fewest trials), confirming the depth-thinning physics.

## 3. The resolvability gate — `n_prompts_to_resolve(margin)` (the deliverable)
Half-width scales as `hw(N)=hw(N0)·√(N0/N)`; a build at true-λ resolves vs the bar iff its half-width ≤ its margin `|λ_true−0.9052|`, giving `N_resolve = N0·(hw_ref/margin)²`. `hw_ref` = the at-the-bar noise floor (±0.017140) — the resolvability curve is a **pure, monotone function of margin** (the operating-point half-width varies <2 % across λ∈[0.86,0.95], so the bar reference is a fair single noise level).

| true-λ | side | margin | **N≥ to resolve @95 %** | decisive @ default 128? |
|---|---|---|---|---|
| 0.86 | NO-GO | 0.0452 | **19** | ✅ True |
| 0.88 | NO-GO | 0.0252 | **60** | ✅ True |
| 0.905 | ON-BAR | 0.0002 | **717 062** | ❌ False |
| **0.93** | **GO** | **0.0248** | **62** | ✅ **True** |
| 0.95 | GO | 0.0448 | **19** | ✅ True |

**Punchline:** to make a **true-λ=0.93** build read as a **decisive GO** vs the 0.9052 bar, measure `q[2..9]` over **N≥62 prompts** at output_len 512 — already decisive at the default 128. A build sitting **on** the bar (λ≈0.905) needs ~**717 k** prompts → effectively **unresolvable**: the gate INPUT is decisive only with margin from the bar, and a point λ̂ within **±0.017** of 0.9052 is an **indecisive GO**, not a decision.

## 4. Compose INPUT-CI ⊕ OUTPUT-CI (double-count audit) — `input_output_compose`
Both the λ̂-route (INPUT) and #175's L̄-route (OUTPUT) are linear functionals of the **same** per-depth accept fluctuations δq̂_d:
`δλ̂ = Σ_d (w_λ,d/span_d)·δq̂_d` and `δL̄ = Σ_d (∂E[T]/∂q_d)·δq̂_d` (∂E[T]/∂q_d via central difference on the imported #172 backward DP). Mapping λ̂ to TPS with the local forward-map slope **216.48 TPS/unit-λ** (matches #178's 0.838→486.2, 0.9052→500, 1.0→520.95):

| leg | half-width (TPS) |
|---|---|
| H_in (λ̂-route) = slope·±0.017140 | **±3.710** |
| H_out (#175 L̄-route) | **±5.178** |
| naive quadrature √(H_in²+H_out²) | 6.370 |
| **overlap-corrected (shared bench)** | **5.319** |

**Verdict: partial-overlap, `overlap_fraction = 0.8929` (ρ=0.9449).** The two CIs are **NOT independent on a shared bench** — 89 % of the λ̂-route variance already lives inside #175's L̄-draw. Regime rule for the integrator fern #185: **independent benches** (ladder bench ≠ official-TPS bench) → add in quadrature (6.37 TPS); **same bench** (one run yields both q̂[2..9] and L̄) → use the overlap-corrected 5.32 TPS — stacking the two CIs in quadrature on a shared bench **double-counts** (the ubel #181 audit, on the variance instead of the mean).

## 5. Self-test (PRIMARY) — `lambda_built_ci_self_test_passes` = True
| # | condition | result |
|---|---|---|
| a | liveprobe ladder λ̂=0.342 inverts back to 0.342 (both poolings, both topologies) | ✅ |
| b | endpoints λ=1→1.0 and λ=0→0.0 recover | ✅ |
| c | half-width shrinks as 1/√N (hw·√N const over N∈{32…1024}) | ✅ |
| d | `resolve_table` monotone in margin | ✅ |
| e | deep-depth q̂_d dominates raw variance (**depth 9**) | ✅ |
| f | NaN-clean (every payload scalar finite) | ✅ |

**Serial-correlation sensitivity** (effective-N deflation, conservative): VIF=1.0 → ±0.0171; VIF=1.5 (lag-1 ρ≈0.20) → ±0.0210; VIF=2.0 (ρ≈0.33) → ±0.0242. If land #71's per-step accepts are autocorrelated the half-width inflates — measure the accept autocorrelation and deflate N accordingly.

## 6. Hand-off
**land #71:** *measure your `q[2..9]` ladder over **N≥62 prompts** at output_len 512 so your implied λ̂_built resolves the 0.9052 bar **decisively** at 95 % — otherwise a point λ̂ within ±0.0171 of the bar is an **indecisive GO**, not a decision. Pool the per-depth λ̂_d by inverse-variance (WLS); the deep depths thin out and dominate the raw variance. Report q̂[2..9] **with** their per-depth trial counts n_d so the CI is auditable.*

**fern #185 (integrator):** the INPUT-side λ̂_built CI (±0.0171 at N=128) and #175's OUTPUT-side ±10.9 TPS CI **share overlap_fraction=0.893** of variance on a common bench — compose in quadrature **only** if the ladder and official-TPS benches are independent; on a shared bench subtract the overlap.

**Scope:** this prices the measurement noise on the **GATE INPUT**; it does **NOT** change the bar (#183 owns it) or authorize a launch. **NOT open2. NOT a launch.**
