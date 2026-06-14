<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Salvage-staleness λ(depth) — mechanism vs flat depth-transfer (PR #193 · denken)

**PRIMARY** `lambda_depth_profile_self_test_passes` = **True** (5/5 conditions, NaN-clean)
**TEST** `both_bugs_mechanism_floor_tps` = **396.72** TPS (primary β, at liveprobe λ̂₁=0.342, τ=1) — **misses 500, and LOWER than #178's flat-transfer 416.3**
**W&B** `2clxvlr8` · group `lambda-depth-staleness-profile` · peak 25.6 MiB

## Honest scope
Pure-analytic **CPU-only** synthesis. No GPU / vLLM / HF Job / submission / kernel build / served-file change. BASELINE stays **481.53**; **0 TPS**; greedy/PPL untouched. **Bank-the-analysis** (PRIMARY = self-test). Imports denken **#183** (`82uisrez`) finite-sample-LCB machinery + 0.9052 bar + q[2..9] card → **#178** (`zjdc7hhh`) graded E[T](λ) + endpoint spines + liveprobe λ̂₁=0.342 → **#172** (`gh8pa4f3`) backward-DP + composition constants, wirbel **#175** (`zh1accmi`) accepted-length pmf σ_L (through #183), wirbel **#135** measured salvage-no-descend conditional ladder — **does NOT re-derive any of them**. **NOT open2. NOT a launch.**

## The question
Every finite-sample result in the denken self-KV lane — #178 (`REALISTIC-FLOOR-MISSES-BOTH`), #183 (the build bar λ≥0.9052), #187 (the λ̂_built measurement-CI) — rests on ONE shared, un-grounded modelled assumption: that the depth-1 recovery fraction transfers **FLAT** across depths 2..9. The single measured point is the depth-1 liveprobe `λ̂₁ = (0.6927−0.674)/(0.7287−0.674) = 0.342`, carried *constant* across depth (constant-λ primary; a geometric γ∈[0.7,0.9] band as the only sensitivity). The clear-500 question is decided almost entirely by the **shape** of λ(depth) — and that shape was an assumption, not a mechanism.

## The mechanism: geometric staleness decay (DERIVED, not guessed)
λ(depth) is **not** free. It is set by the BUG-2 self-KV **salvage staleness** physics. The salvage-no-descend path (`_dixie_fused_accept_prep_kernel` in the served `fa2sw_precache_kenyan` stack; root-caused in wirbel #135's `bug2_salvage_descent`) reuses the **parent's** KV without re-running the descend, so at depth `d` the salvaged KV is `d−1` steps stale. Each descend step adds exactly **one** stale step, and the stale-KV / true-KV divergence compounds multiplicatively, so the per-depth recovery fraction obeys a **geometric staleness-decay law**

```
λ_d = λ̂₁ · β^(d−1)        (staleness s = d−1 ; β = per-step self-KV retention ; α = −ln β)
```

This is *exactly* the geometric profile #178 already swept — but #178 **guessed** γ∈[0.7,0.9]; here the geometric **FORM** is **derived** from the salvage construction (one stale step per depth → multiplicative compounding) and β is **grounded** in the kernel's own measured staleness fingerprint. α=0 (β=1) is the flat #178 special case.

## 1. Grounding β in the salvage construction
β is read from wirbel #135's measured salvage-no-descend conditional ladder (the in-scope kernel's own stale-KV output), `salvage_cond = [0.674, 0.5193, 0.580, 0.6453, 0.6794, 0.6742, 0.6167]`:

| construction measure | value | meaning |
|---|---|---|
| `beta_reach_absolute` | **0.6165** | geomean per-step survival of the salvage conditional ladder (absolute reach) |
| `beta_rel_staleness_only` | **0.9496** | salvage ladder ÷ full-recovery ladder per step (divides out the depth trend; staleness-only) |
| **`beta_primary_geomean`** | **0.7651** | geomean of the two (α = 0.2677) |

**Honest caveat:** β is **bounded but not point-identified** by the single depth-1 anchor — the absolute-reach and staleness-only constructions bracket `[0.6165, 0.9496]`. Primary β=0.765 sits **inside** #178's guessed [0.7,0.9] band. **The verdict is invariant across the entire construction range** (§3) — that invariance, not the point value of β, is the deliverable.

Effective λ(depth 1..9) at primary β (the E[T] DP's 7-entry spine flat-extrapolates depths 8–9 to depth-7, exactly as #178's `qd_at` clamps — mildly **optimistic** at depths 8–9 vs the pure law):
```
effective (DP-clamped) = [0.342, 0.262, 0.200, 0.153, 0.117, 0.090, 0.069, 0.069, 0.069]
pure mechanism (uncl.) = [0.342, 0.262, 0.200, 0.153, 0.117, 0.090, 0.069, 0.053, 0.040]
```

## 2. The realistic floor under the mechanism (at the liveprobe λ̂₁=0.342)

| topology | E[T] | TPS τ=1 | TPS τ=0.9924 | clears 500? |
|---|---|---|---|---|
| descent-only | 3.7465 | **385.3** | 382.3 | ✗ |
| **both-bugs** | 3.8580 | **396.7** | 393.7 | ✗ |

Both **miss 500**, and both are **LOWER** than #178's flat-transfer floor (404 / 416) — because staleness decay pulls the deep-spine recovery below the constant-λ plateau. The full β-sweep at λ̂₁ (β=1 ⇒ flat #178):

| β | α | descent TPS | both-bugs TPS | clears 500? | |
|---|---|---|---|---|---|
| 1.0000 | 0.000 | 404.1 | 416.3 | ✗ | ← flat #178 (OPTIMISTIC plateau) |
| 0.9496 | 0.052 | 398.8 | 410.8 | ✗ | ← β_rel (staleness-only) |
| 0.9000 | 0.105 | 394.4 | 406.2 | ✗ | |
| 0.8000 | 0.223 | 387.3 | 398.8 | ✗ | |
| **0.7651** | **0.268** | **385.3** | **396.7** | ✗ | ← **primary β** |
| 0.7000 | 0.357 | 381.9 | 393.2 | ✗ | |
| 0.6165 | 0.484 | 378.3 | 389.5 | ✗ | ← β_reach (absolute) |

**No β in the construction range clears 500.** Flat (β=1) is the ceiling and already misses by 84 TPS.

## 3. Verdict robustness — `misses_both_robust_to_mechanism = True`
MISSES-BOTH is robust **iff** even the OPTIMISTIC plateau (β=1, the #178 flat case) misses at the realistic λ̂₁ — because any β<1 only **lowers** E[T] (conservative ordering, §4d). It does: flat both-bugs = 416.3 < 500. **So the #178/#183/#187 MISSES-BOTH verdict is a property of the mechanism, not an artifact of the flat-transfer assumption.** The mechanism only makes the miss *wider* (416→397).

## 4. Inverse map — depth-1 sufficiency, `both_bugs_lambda1_star`
Under the mechanism the #183 bar is a **constant-λ** bar. With β<1 deeper depths are capped at `λ̂₁·β^(d−1)`, so asking "what depth-1 λ̂₁ clears the 0.9052-equivalent LCB bar?" gives:

| β | `both_bugs_lambda1_star` (LCB bar) | LCB-TPS at λ̂₁=1.0 |
|---|---|---|
| **1.0000** | **0.9052** (= #183's bar exactly) | 521.0 |
| 0.9496 | **UNREACHABLE** | 492.2 |
| 0.9000 | **UNREACHABLE** | 470.9 |
| 0.8000 | **UNREACHABLE** | 440.9 |
| 0.7651 (primary) | **UNREACHABLE** | 433.0 |
| 0.6165 | **UNREACHABLE** | 408.3 |

At any β<1 the bar is **unreachable even at λ̂₁=1.0** (perfect depth-1 recovery). The critical retention at which a depth-1-only probe is still sufficient is **`beta_crit_depth1_sufficient = 0.9649`** (≈ zero staleness). **Consequence for land #71: the depth-1 liveprobe is necessary but NOT sufficient — unless per-step retention β≥0.965, land must measure the q[2..9] ladder directly, not infer it from depth-1.**

## Self-validate (PRIMARY) — 5/5, NaN-clean
| # | condition | check | residual |
|---|---|---|---|
| a | flat (β=1) reproduces #178 floors | descent 3.9294 / both-bugs 4.0485 → 404.1 / 416.3 | **0.0** |
| b | λ̂₁=0 reproduces #172 lower bound | floor E[T] = 3.5346 | **8.9e-16** |
| c | flat inverse reproduces #183 + #178 | bb λ*_LCB = 0.9052 **and** bb λ*_central = 0.8384 | **0.0 / 0.0** |
| d | conservative ordering mech ≤ flat | min slack (flat−mech) over λ̂₁-grid × β-ladder × both topologies | **0.0** (≥0) |
| prov | pmf-mean reproduces E[T] | wirbel #175 pmf on the same spine | machine-eps |

`lambda_depth_profile_self_test_passes = True`, NaN-clean (all unreachable inverse maps stored as `null`, not NaN).

## Hand-off
**SALVAGE-STALENESS λ(depth) (denken #193):** replaces the FLAT depth-transfer behind #178/#183/#187 with the mechanism-derived `λ_d = λ̂₁·β^(d−1)` (β grounded in wirbel #135's measured salvage ladder; primary β=0.765, construction range [0.616, 0.950], inside #178's [0.7,0.9]). At the liveprobe λ̂₁=0.342: descent **385** / both-bugs **397** TPS — both MISS, *lower* than #178's flat 404/416, so **MISSES-BOTH = ROBUST** (flat is the optimistic plateau and already misses; staleness only widens it). The 0.9052 build bar is a constant-λ bar; under the mechanism the depth-1 inverse bar is **unreachable** and a depth-1 probe is sufficient only if β≥0.965. **land #71 confirms/refutes against measured q[2..9]**; this re-grounds denken #183 (the bar) and #187 (depth-9 — the dominant-variance end — is also the lowest-recovery end). NOT open2. NOT a launch.

## Public / banked evidence used
- denken **#183** (`82uisrez`): finite-sample-LCB machinery, 0.9052 both-bugs build bar, 0.8384 #178 point, q[2..9] per-depth card — imported (the bar this re-grounds).
- denken **#178** (`zjdc7hhh`): graded E[T](λ) interpolation, endpoint spines, liveprobe λ̂₁=0.342, flat-transfer floors 404/416, geometric γ∈[0.7,0.9] sensitivity band — imported (the FLAT assumption this replaces).
- denken **#172** (`gh8pa4f3`): backward-DP `et_backward`, composition constants K_cal=125.268 / step 1.2182, the 3.5346 descent lower bound — imported (λ̂₁=0 special case).
- wirbel **#175** (`zh1accmi`): accepted-length pmf → σ_L second moment (through #183) — imported (provenance lock).
- wirbel **#135** (`bug2_salvage_descent`): the measured salvage-no-descend conditional ladder + "salvage-fires-but-does-not-descend" physics — the staleness fingerprint β is grounded in.
