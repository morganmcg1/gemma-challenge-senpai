<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SPRT liveprobe budget — expected-N early-stop vs #197's fixed-N 30k (PR #205 · denken)

**PRIMARY** `sprt_budget_self_test_passes` = **True** (6/6 conditions, NaN-clean)
**TEST** `expected_n_sprt_nogo` = **405** liveprobe trials — the REALISTIC sequential cost to certify the **likely NO-GO** (grounded β=0.765, private_LCB 419.6 ≪ 500), a **≈75× collapse** vs #197's truth-independent fixed-N **30,455**.
**W&B** `sprt-liveprobe-budget` group · run `eijqklu2` · peak ≈ 30 MiB

## Honest scope
Pure-analytic **CPU-only** synthesis — the SEQUENTIAL analog of #197's fixed-N measurement DESIGN. It MODELS a Wald SPRT over #197's banked per-depth accept weights; it takes **NO draws** and authorizes none. No GPU / vLLM / HF Job / submission / served-file change. BASELINE stays **481.53**; **0 TPS**; greedy/PPL untouched. **Bank-the-analysis** (PRIMARY = self-test). Imports — and does NOT re-derive — denken **#197** (`wqr94io4`) `neyman_budget` (a_d=∂E[T]/∂q_d, Neyman fractions, fixed-N 30,455 @λ=1, private forward map); denken **#193** (`2clxvlr8`) β_primary=0.7651, β_crit=0.9649; denken **#187** (`tloghme9`) per-depth σ_d; stark **#191** (`jeclr39w`) private bar 0.9780 / descent UNREACHABLE; wirbel **#190** (`fva6o4ug`) within-prompt ICC 0.1446, Deff 4.41. **NOT open2. NOT a launch.**

## The question
#197 priced the **fixed-N** Neyman budget — 30,455 trials to decisively certify the best-case λ=1 build — and proved that at the grounded β=0.765 even *perfect* depth-1 recovery yields private_LCB 419.6 ≪ 500: a clear NO-GO. A fixed-N design spends the **full 30k even when the build is obviously below the bar**. The sequential question: with Wald early-stopping on the per-depth accept stream, what is the **expected** trial count under each truth — and how much does the common (clear-NO-GO) case collapse below 30k?

## Design — matched-strength sequential test (deliverable 1)
**H0** GO-capable (ladder clears 0.9780 at P≥0.95, anchor μ1=1.0) vs **H1** NO-GO (μ0=bar=0.9780); indifference zone width = #197's margin **m=0.022**. The per-trial LLR on the Neyman-weighted λ-equivalent read is

```
z_i = (m/σ₁²)·(x_i − μ_mid),   x_i ~ N(λ_eq, σ₁²),   μ_mid=(0.9780+1)/2=0.9890
```

with Wald boundaries **A = ln((1−β)/α) = +2.9444** (decide GO) and **B = ln(β/(1−α)) = −2.9444** (decide NO-GO) at target (α,power)=(0.05,0.95). #197's decisive fixed-N is **adopted as the fixed-sample reference**, calibrating the per-trial drift so the no-early-stop sample size reproduces 30,455. The SPRT **keeps #197's shallow-heavy Neyman weighting** (the likelihood-ratio increment *is* inverse-variance).

## 2. Expected-N under each truth (the core, deliverable 2)
ASN/OC via Wald; truths parameterized by staleness β at perfect depth-1 (λ̂₁=1.0):

| β (truth) | λ_eq | private_LCB | P(decide GO) | **E[N] (iid)** | ×Deff=4.41 | GO? |
|---|---|---|---|---|---|---|
| 0.700 | 0.459 | 407.7 | 0.000 | **344** | 1,517 | ❌ |
| **0.765** | **0.539** | **419.6** | 0.000 | **405** | **1,788** | ❌ |
| 0.820 | 0.619 | 432.1 | 0.000 | 492 | 2,170 | ❌ |
| 0.880 | 0.721 | 449.4 | 0.000 | 680 | 3,000 | ❌ |
| 0.920 | 0.801 | 464.0 | 0.000 | 969 | 4,275 | ❌ |
| 0.9649 | 0.905 | 484.6 | 0.000 | 2,176 | 9,596 | ❌ |
| 0.980 | 0.944 | 492.8 | 0.000 | 4,088 | 18,027 | ❌ |
| 1.000 | 1.000 | 504.9 | 0.950 | 14,915 | 65,775 | ✅ |

**Headline:** the likely NO-GO (β=0.765) costs **E[N]≈405** trials — a **≈75× collapse** vs the fixed 30,455. Cost rises to **≈14,915** only if the build is genuinely at the best-case (β=1.0), staying **≤ 30k**.

## 3. Sequential depth order / shallow screen (deliverable 3)
`sequential_depth_order = [2,3,4,5,6,7,8,9]` — **unchanged** from #197 (the LLR is Neyman/inverse-variance weighted). Cumulative decisive-information fraction: depths **{2,3,4} carry 65.3%**, deep **{7,8,9} only 10.9%**. A **shallow-first stage rejects most clear-NO-GO builds** (large drift) before any depth-7–9 probing.

## 4. Operating characteristic at the 0.022 knife-edge (deliverable 4)
Realized **(α,power) = (0.050, 0.950) ≥ target** (Wald boundaries are conservative). The ASN **peaks at the indifference point** (λ_eq=μ_mid=0.989, β≈0.9962): `worst_case_expected_n = 24,398` (≤ 30k). **A build genuinely at the bar is the only expensive case; everything clearly below is cheap.**

## 5. Realism band — within-prompt ICC (deliverable 5, secondary)
#190's Deff=4.41 inflates absolute counts if liveprobe steps are autocorrelated: `expected_n_sprt_nogo` **405 (iid) → 1,788 (×Deff)**; the fixed-N reference **30,455 → 134,327**. The **≈75× saving ratio is Deff-invariant** (both scale by Deff).

## 6. Self-test (PRIMARY) — 6/6
(a) truncated SPRT (early-stop disabled) reproduces #197's fixed-N **30,455** (no-early-stop limit; FSS round-trips); (b) E[N|near-bar]=14,915 ≤ 30k and E[N|NO-GO]=405 ≪ 30k; (c) realized (α,power)=(0.05,0.95) ≥ target; (d) ASN monotone down the NO-GO side and peaks at indifference; (e) reproduces #193 β_crit 0.9649 + #191 private bar 0.9780; (f) NaN-clean. **Calibration-invariance check:** matched (anchor 30,455) and physical-info (anchor 85,799) calibrations give the **identical 75.12× saving**.

## Hand-off
- **land #71:** REALISTIC liveprobe cost is **E[N]≈405** to certify the likely NO-GO (vs fixed-N 30,455), rising to **≈14,915** only at the bar; a shallow-first stage (depths 2–4 = 65% of info) rejects most NO-GO builds before deep probing; worst-case ≈24,398; ×Deff=4.41 for the autocorrelated-step band.
- **fern #185:** consume the **(expected-N, OC)** sequential row — early-stop ≈405 (likely NO-GO) … ≈24,398 (worst-case at the bar) at (α,power)=(0.05,0.95) — as the realistic measurement-cost row, not #197's truth-independent fixed-N.
