<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Margin-aware λ-acceptance card — the finite-sample-LCB build bar (PR #183 · denken)

**PRIMARY** `lambda_acceptance_card_self_test_passes` = **True** (6/6 conditions, NaN-clean)
**TEST** `both_bugs_lambda_star_lcb` = **0.9052** (both-bugs, full finite-sample LCB clears 500, τ=1) — **stricter than #178's 0.8384 point estimate by Δλ=+0.0668**
**W&B** `82uisrez` · group `lambda-acceptance-card` · peak 27.1 MiB

## Honest scope
Pure-analytic **CPU-only** composition. No GPU / vLLM / HF Job / submission / kernel build / served-file change. BASELINE stays **481.53**; **0 TPS**; greedy untouched. **Bank-the-analysis** (PRIMARY = self-test). Imports denken **#178** (`zjdc7hhh`) E[T](λ) interpolation + endpoints, wirbel **#175** (`zh1accmi`) accepted-length pmf → σ_L second moment, kanna **#159** σ_hw=4.86 TPS, lawine **#168** step 1.2182, K_cal=125.268 (#148, tree-invariant #169) — **does NOT re-derive any of them**. **NOT open2. NOT a launch.**

## The question
#178 reported the clear-500 self-KV thresholds as POINT estimates (the λ at which the **central** TPS = 500: 0.838 both-bugs / 0.909 descent-only at τ=1). But the single official `summary.json:tps` is one finite, irreversible draw. "The built kernel clears 500" must mean the **95% lower confidence bound** clears 500, not the central point. This card converts #178's point bar into the **margin-aware LCB bar** land #71 actually has to hit.

## Composition (imported legs)
```
central(λ) = K_cal·(E[T](λ)/step)·τ                              # #172/#178
H(λ)       = z95·√( SE_tps(λ)²  +  σ_hw² )                       # PR #183 full half-width
  SE_tps(λ) = (K_cal·τ/step)·σ_L(λ)/√N_steps(λ)                 # wirbel #175 numerator leg
  N_steps(λ) = B/E[T](λ),  B=16384                              # wirbel #175 budget
  σ_L(λ)     = √Var[L] of the accepted-length pmf at spine(λ)   # wirbel #175 second moment
  σ_hw       = 4.86 TPS  (1σ)                                    # kanna #159 denominator leg
lcb_tps(λ) = central(λ) − H(λ)
```
**Provenance lock:** σ_L(λ) is read from wirbel #175's `dp_accepted_length_pmf` on the **same** spine #178's `et_backward` consumes; the pmf-mean reproduces E[T](λ) to **resid ≤ 1.8e-15** at every λ. The two imported machineries describe one model.

## 1. λ=1 reproduces wirbel #175 (resid check)
At λ=1 the **numerator-only** LCB (σ_hw off) must reproduce wirbel #175's published bounds; the **full** LCB then adds σ_hw on top.

| topology | σ_L (ref) | central | lcb_num | wirbel ref | resid | **lcb_full (+σ_hw)** |
|---|---|---|---|---|---|---|
| both-bugs | 3.035437 (3.035437) | 535.433 | 524.5268 | 524.5270 | **2.1e-04** | **520.95** |
| descent-only | 3.059275 (3.059275) | 519.952 | 509.1202 | 509.1204 | **2.0e-04** | **505.53** |

The 2e-4 TPS resid is exactly the K_cal rounding gap (wirbel's `125.268` vs the canonical tree-invariant `125.26795`); ≪ the 0.5 TPS self-test tolerance. **σ_L matches to 1e-15.** Both full LCBs (with σ_hw) still clear 500 at λ=1, so a finite λ* exists.

## 2. Margin-aware λ* (the build-acceptance bar) vs #178's point estimate
Solve `lcb_tps(λ*) = 500`:

| topology | τ | **λ\*_LCB (build bar)** | #178 point λ* (central=500) | Δλ (extra recovery needed) |
|---|---|---|---|---|
| **both-bugs** | 1.0 | **0.9052** | 0.8384 | **+0.0668** |
| both-bugs | 0.9924 | 0.9234 | 0.8570 | +0.0664 |
| descent-only | 1.0 | 0.9750 | 0.9091 | +0.0659 |
| descent-only | 0.9924 | 0.9926 | 0.9271 | +0.0655 |

The finite-sample bar is **uniformly ~6.6 points of recovery stricter** than #178's central point. **At #178's own point estimate λ=0.838, the both-bugs LCB is only 486.2 — a 14-TPS miss.** Building to the point bar would fail the finite-sample test; the build must reach λ≥0.9052.

## 3. Per-depth `q[2..9]` acceptance ladder at λ\* (both-bugs, τ=1, the card land #71 tests against)
`q_d(λ*) = (1−λ*)·q_floor[d] + λ*·q_full[d]`, λ*=0.9052. q[2..9] = the self-KV-governed depths (depth-1 is the separate BUG-1 axis). The spine parameterizes depths 1–7; depths 8–9 are flat-extrapolated (= depth-7), exactly as the imported E[T] DP's `qd_at` clamps.

| depth | **q\*(λ\*) must hit** | [floor → full] | headroom to full |
|---|---|---|---|
| 2 | **0.7363** | 0.5193 → 0.7590 | 0.0227 |
| 3 | **0.7724** | 0.5800 → 0.7925 | 0.0201 |
| 4 | **0.8050** | 0.6453 → 0.8217 | 0.0167 |
| 5 | **0.8196** | 0.6794 → 0.8343 | 0.0147 |
| 6 | **0.8200** | 0.6742 → 0.8353 | 0.0153 |
| 7 | **0.8254** | 0.6167 → 0.8473 | 0.0219 |
| 8 | 0.8254 (flat) | 0.6167 → 0.8473 | 0.0219 |
| 9 | 0.8254 (flat) | 0.6167 → 0.8473 | 0.0219 |

All bracketed by #178's endpoints. **Most binding = depth 7** (highest absolute accept it must reach, 0.8254). Depth 5 has the thinnest headroom (0.0147) — i.e. the build is closest to its full-recovery ceiling there.

## 4. Forward map (the go/no-go calculator) — both-bugs, τ=1
`measured λ → predicted LCB-TPS`, `card_is_monotone = True`:

| λ | E[T] | predicted LCB | clears 500? | |
|---|---|---|---|---|
| 0.342 | 4.0485 | 404.1 | ✗ | ← liveprobe λ̂ (reproduces #178 miss) |
| 0.838 | 4.8624 | 486.2 | ✗ | ← #178 point λ* (LCB still misses!) |
| **0.9052** | 4.9992 | **500.0** | ✓ | ← **LCB build bar** |
| 0.95 | 5.0953 | 509.7 | ✓ | |
| 1.00 | 5.2070 | 520.95 | ✓ | |

**Inverse map for land #71** — `λ̂_d = (q_meas[d]−q_floor[d])/(q_full[d]−q_floor[d])`, pooled `λ̂_built = mean_d`. Self-consistency demo (both-bugs): the floor ladder → λ̂=0.000000, the full ladder → λ̂=1.000000, the λ* spine → λ̂=0.905229 (= target). land feeds its measured q[2..9] → implied λ̂_built → reads the predicted LCB off this table; **clear iff predicted-LCB ≥ 500**.

## 5. Serial-correlation sensitivity (conservative bar)
wirbel #175's ±10.9 assumes iid steps; positive lag-1 autocorrelation ρ shrinks effective N (SE × √VIF, VIF≈(1+ρ)/(1−ρ)) and **raises** λ*:

| VIF (ρ≈) | both-bugs λ\*_LCB (τ=1) | descent-only λ\*_LCB (τ=0.9924) |
|---|---|---|
| 1.0 (0.00) | 0.9052 | 0.9926 |
| 1.5 (0.20) | 0.9137 | **unreachable** (LCB@λ=1 = 499.75) |
| 2.0 (0.33) | 0.9213 | **unreachable** (LCB@λ=1 = 498.06) |

**both-bugs is the only path robust to plausible serial correlation:** under the conservative-τ floor, descent-only's LCB cannot clear 500 even at full recovery once ρ≳0.2. The conservative both-bugs bar is **λ ≥ 0.921** (VIF=2).

## Self-validate (PRIMARY)
6/6 conditions pass: (a) λ=1 numerator-LCB reproduces wirbel 524.5/509.1 (resid 2e-4 < 0.5); (b) margin-aware λ* ≥ #178 point estimate AND ∈[0,1] for both topologies; (c) per-depth q[2..9] ladder reported + bracketed by #178 endpoints (8 depths); (d) forward map monotone AND λ̂=0.342 row reproduces #178's 404/416 central; (prov) pmf-mean reproduces E[T] to 1e-15; (central) λ*-central cross-check reproduces #178's point estimates to <1e-3. **`lambda_acceptance_card_self_test_passes = True`**, NaN-clean.

## Hand-off
**fern #179 packet (build line):** land #71's kernel must demonstrate per-depth self-KV recovery **λ ≥ 0.9052** (both-bugs, finite-sample-LCB-clears-500 bar — wirbel #175's ±10.9 ⊕ kanna #159's σ_hw=4.86), **stricter than #178's 0.8384 point estimate by Δλ=+0.067**; the measured q[2..9] ladder is tested against the per-depth card (depth 7 most binding); the one pre-build measured point λ̂=0.342 (liveprobe) is far below.

**land #71 calculator:** hit per-depth q[2..9] ≥ [0.7363, 0.7724, 0.8050, 0.8196, 0.8200, 0.8254, 0.8254, 0.8254] (both-bugs, LCB bar λ*=0.9052); depth 7 hardest. Feed measured q[2..9] → implied λ̂_built → read predicted LCB-TPS; clear iff ≥ 500. (At the unbuilt λ̂=0.342: LCB≈404, a clear miss.)

## Public / banked evidence used
- denken **#178** (`zjdc7hhh`): E[T](λ) graded recovery curve + endpoints (5.0564 / 3.5346 / 5.2070), liveprobe λ̂=0.342, point-estimate λ* (0.838/0.909) — imported.
- wirbel **#175** (`zh1accmi`): accepted-length pmf → σ_L (3.0354 both-bugs / 3.0593 descent), ±10.906 finite-sample CI, LCB [524.5 / 509.1] — imported (σ_L machinery + reproduction targets).
- kanna **#159**: σ_hw=4.86 TPS denominator leg (the quadrature twin wirbel #175 left armed/pending).
- lawine **#168**: step 1.2182. K_cal=125.268 (#148, tree-invariant #169). Issue #124 RESOLVED greedy-exact.
