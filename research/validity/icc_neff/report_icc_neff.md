<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Within-prompt ICC → realistic launch CI + GO-robustness verdict (PR #190 · wirbel)

**PRIMARY** `icc_neff_self_test_passes` = **True** (7/7 conditions, NaN-clean)
**TEST** `lcb_bothbugs_realistic_icc` = **510.63 TPS** (both-bugs launch LCB at realistic ICC=0.145, §4 convention — clears 500 by +10.6)
**W&B** group `icc-neff-launch-ci`

## Honest scope
Pure-analytic **CPU-only** synthesis. No GPU / vLLM / HF Job / submission / kernel build / served-file change. BASELINE stays **481.53**; **0 TPS**; greedy untouched. Imports (does **NOT** re-derive): wirbel **#175** (`zh1accmi`) σ_L/±10.906/central; denken **#184** N_eff two-level model + the ICC=0/ICC=1 bracket (reproduced as a self-test); denken **#183** (`82uisrez`) forward-map spine + λ\*_LCB=0.9052; kanna **#159** σ_hw=4.86, K_cal, step 1.2182; the launch-packet LCB(P≥0.9) convention. **NOT open2. Does NOT authorize a launch.**

## The question
#175 priced the single-shot finite-sample TPS CI at **±10.906** assuming the ~3147 decode steps are **IID**; #184 stress-tested perfect clustering **ICC=1 → ±54.9** (LCB 480.5, a MISS). The clear-500 margin lives in that 5× bracket. This card asks the **data** where: what is the **realistic** within-prompt ICC, and does the **both-bugs GO survive it**?

## Estimator + data source
`icc_data_source` = **research/rank_coverage/pr86/rankprobe_records.jsonl.118860** (PR #86 rankprobe, **17169 decode steps**), reconstructed into the **128 benchmark prompts** by accumulating committed tokens to 512/prompt (research/rank_coverage/pr86/decode_rank_coverage.jsonl). Per-step quantity = **L = accepted-run (fd) + 1 bonus** (E[L]=E[T]; Var[L]=σ_L² — the #175 CI driver). One-way random-effects ANOVA: **ICC = σ²_b/(σ²_b+σ²_w)**, σ²_b=(MSB−MSW)/m₀, σ²_w=MSW.

**Cluster-size honesty:** the within-prompt correlation is **serial** (ρ(1)=0.258, ρ(2)=0.168, ρ(3)=0.118, ~AR decay), not pure exchangeable, so the ANOVA ICC depends on the window. **PRIMARY = the TPS-window** (128 tok/prompt = #175's B=16384/128, ~33 steps); the band below brackets it.

| ICC estimate (cluster window) | ICC |
|---|---|
| **token-window 128 tok/prompt (PRIMARY, #175 scale)** | **0.1446** |
| full-prompt (512 tok, ~134 steps) | 0.0562 |
| first-24-step (≈ m̄) | 0.1546 |
| first-25-step | 0.1505 |
| first-33-step | 0.1291 |
| ACF-Deff equiv @ m̄ (pure serial, no prompt-mean heterogeneity) | 0.0752 |

**`icc_hat` = 0.1446**, **`icc_ci` = [0.1043, 0.1857]** (prompt-level cluster bootstrap, 3000 resamples, TPS-window ICC).

## 1. Realistic CI (between #175 IID and #184 ICC=1)
m̄ = 24.583 (imported #184). `Deff = 1+(m̄−1)·ICC` = **4.411**; `N_eff` = N_steps/Deff = **713** (from 3147). Only the accept-length term inflates (σ_hw is fixed denominator jitter):

| | half-width (accept, TPS) |
|---|---|
| #175 IID floor (ICC=0) | ±10.91 |
| **realistic (ICC=0.145)** | **±22.90** |
| #184 ICC=1 ceiling | ±54.07 |

`halfwidth_realistic` = ±22.90 TPS — **2.1× the IID floor, 0.42× the ICC=1 ceiling.**

## 2. GO-robustness verdict (THE deliverable)
§4 convention (central=535.43 both / 519.95 descent, z95, accept ⊕ σ_hw=4.86 in quadrature). The ICC=0 and ICC=1 rows **reproduce #184's bracket exactly** (520.95 / 480.53):

| ICC | Deff | accept ± | total ± | **both-bugs LCB** | clears? | **descent LCB** | clears? |
|---|---|---|---|---|---|---|---|
| 0 (IID #175) | 1.00 | — | ±14.48 | 520.95 | ✓ | 505.53 | ✓ |
| 0.104 (CI-lo) | 3.46 | ±20.28 | ±22.41 | **513.02** | ✓ | — | — |
| **0.145 (ĤAT)** | **4.41** | **±22.90** | **±24.81** | **510.63** | **✓** | **495.04** | **✗** |
| 0.186 (CI-hi) | 5.38 | ±25.30 | ±27.03 | **508.40** | ✓ | 492.79 | ✗ |
| 1 (#184 worst) | 24.58 | — | ±54.91 | 480.53 | ✗ | 464.63 | ✗ |

- `lcb_bothbugs_realistic_icc` = **510.63** (clears 500 by **+10.6**)
- `lcb_descent_realistic_icc` = **495.04** (MISSES)
- `icc_at_which_bothbugs_breaks_500` = **0.3728** (= **2.6×** the realistic ICC); descent breaks already at ICC=0.0666.

**HEADLINE — both-bugs STAYS >=500 (robust GO) at the realistic ICC 0.145 (LCB 510.6) and across the entire CI [0.104,0.186] (LCB 508.4..513.0); it only breaks 500 at ICC=0.373 (2.6x the realistic value). descent-only MISSES (LCB 495.0); it breaks 500 already at ICC=0.067, below the realistic estimate -- NOT robust to realistic within-prompt correlation.**

## 3. Refined build bar (#183) under realistic correlation
The #183 bar λ\*_LCB=0.9052 solved `central(λ)−z95·√(SE(λ)²+σ_hw²)=500` at Deff=1. The finite-sample SE inflates by √Deff. Machinery check: **Deff=1 reproduces 0.9052** (= published 0.9052).

| Deff (ICC) | λ\*_LCB | shift vs iid |
|---|---|---|
| 1.00 (iid) | 0.9052 | +0.0000 |
| 3.46 (CI-lo) | 0.9404 | +0.0352 |
| **4.41 (ĤAT)** | **0.9513** | **+0.0461** |
| 5.38 (CI-hi) | 0.9613 | +0.0561 |

`lambda_star_lcb_realistic_icc` = **0.9513**, `bar_shift_from_icc` = **+0.0461** (vs iid 0.9052). #183 section-5 used an asymptotic AR(1) VIF (<=2.0 -> lambda*=0.9213); the correct finite-cluster design effect Deff=1+(m_bar-1)*ICC is larger, so the realistic build bar is higher. Still reachable (<1) and bracketed by full recovery.

## 4. Launch-packet LCB(P≥0.9) — secondary handoff
Folding the realistic-ICC finite-sample relative term (×√Deff) into the launch packet's published 3-term combined (σ_hw retired on a separate axis):

| | combined_rel (1σ) | LCB(P≥0.9) | GO? |
|---|---|---|---|
| published (3-term, finite-sample PENDING) | 0.02067 | 514.88 | GO |
| + IID finite-sample fold | 0.02313 | 513.21 | GO |
| **+ realistic-ICC finite-sample fold** | **0.03006** | **508.51** | **GO** |

Consistent with the §4 verdict (510.6): both-bugs clears 500 under the realistic ICC in **both** LCB conventions.

## 5. Self-validate (PRIMARY)
7/7 conditions pass: (a) ICC=0 reproduces #184 ±10.906/LCB 521; (b) ICC=1 reproduces ±54.9/LCB 480.5; (c) ICC_hat∈[0,1] with finite CI; (d) §4 LCB monotone-decreasing in ICC; (e) λ\*_LCB(realistic) ≥ 0.9052 AND Deff=1 reproduces 0.9052; (f) NaN-clean. **`icc_neff_self_test_passes` = True**.

## Operating-point caveat
the rankprobe is at the liveprobe operating point lambda_hat=0.342 (E[L]=3.85), not the launch's full-recovery lambda=1 (E[L]=5.21). The CORRELATION STRUCTURE (ICC, rho(l)) is dimensionless and transported to lambda=1; if longer accept-runs at lambda=1 are MORE correlated, the true ICC could exceed this estimate -- the breakpoint headroom (2.6x) absorbs a 2x miss.

## Hand-off
**fern #185 / launch packet:** the realistic within-prompt ICC is **0.145** [0.104, 0.186], placing the single-shot CI at **±22.9 TPS** (Deff=4.41, N_eff=713) — 2.1× #175's IID floor but only 0.42× #184's ICC=1 ceiling. **both-bugs stays a robust GO** (LCB 510.6 §4 / 508.5 P≥0.9), breaking 500 only at ICC=0.373 (2.6× realistic); **descent-only is NOT robust** (LCB 495.0, breaks at ICC=0.067 < realistic). **land #71's build bar tightens to λ ≥ 0.9513** (vs iid 0.9052, +0.046).

## Public / banked evidence used
- wirbel **#175** (`zh1accmi`): σ_L, ±10.906 IID half-width, central 535.43 — the CI floor + driver.
- denken **#184**: N_eff two-level model (m̄=24.58, exchangeable Deff), the ICC=0/ICC=1 bracket (520.95/480.53) reproduced as the self-test.
- denken **#183** (`82uisrez`): forward-map spine (E[T](λ), σ_L(λ)) + λ\*_LCB=0.9052 — the build bar refined here.
- kanna **#159**: σ_hw=4.86 TPS, K_cal, step 1.2182.
- launch packet: LCB(P≥0.9) convention (proj_private 528.89, z_p90 1.2816) — secondary fold.
