<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Realistic self-KV E[T] floor — graded recovery curve anchored to the liveprobe (PR #178)

**PRIMARY** `realistic_selfkv_floor_self_test_passes` = **True** (4/4 conditions, NaN-clean)
**TEST** `descent_only_realistic_floor_E_T` = **3.9294** (constant-λ at the liveprobe anchor λ̂=0.342)
**W&B** run `zjdc7hhh` · group `descent-realistic-selfkv-floor` · peak 12.1 MiB

## TL;DR — the graded verdict that replaces #172's binary floor

denken #172 left a **binary** caveat on the descent-only numerator: full self-KV → central
**5.0564** (→520, clears 500); 100%-starved → adversarial floor **3.5346** (→363, fails). This
PR grades that floor with a per-depth self-KV recovery curve **E[T](λ)** anchored to
openevolve's live depth-1 deficit, and the measured anchor lands the realistic floor **far
below** the clear-500 line:

> **Descent-only clears 500 iff built deep-spine self-KV recovery λ ≥ 0.909. The
> liveprobe-anchored realistic estimate is λ̂ = 0.342 → E[T] = 3.9294 → MISSES 500 by ~96
> TPS (404 TPS).** Both-bugs (depth-1 BUG-1 fix on top) needs λ ≥ 0.838 and at λ̂ also misses
> 500 (416 TPS) — the *safer first shot* (lower threshold, more top-margin) but **not robust
> at the measured floor**. The binding constraint for **both** paths is self-KV recovery, not
> BUG-1.

`VERDICT: REALISTIC-FLOOR-MISSES-BOTH`.

## Honest scope

Pure-analytic **CPU-only** synthesis. No GPU / vLLM / HF Job / submission / kernel build /
served-file change. BASELINE stays 481.53; **adds 0 TPS**; greedy untouched. It IMPORTS denken
#172's (`gh8pa4f3`) descent E[T]-DP machinery (`et_backward` renewal-reward DP + `et_pathenum`,
the committed endpoint spines, `K_cal`, step) and openevolve's liveprobe numbers — it does
**not** re-derive 5.0564, 3.5346, 5.2070, K_cal=125.268, or step=1.2182. This is the **graded
realistic twin** of #172's adversarial floor on the E[T] numerator axis. **NOT open2** (tree
economics, not drafter architecture). **NOT a launch.**

The single load-bearing modelling step (flagged below): the liveprobe is **one measured
depth-1 point**, the depth-1→depth>0 transfer and the deeper-depth λ profile are **modelled**.

## 1. Per-depth self-KV recovery map `q_d(λ_d)` (endpoints reproduce #172 exactly)

`λ_d ∈ [0,1]` = fraction of the depth-d self-KV deficit recovered. Linear per-depth map between
two committed #172 endpoint spines:

```
q_d(λ_d) = (1 − λ_d)·q_floor[d]  +  λ_d·q_full[d]
```

| endpoint | spine `q[d]` (depth 1..7) | E[T] | #172 import | resid |
|---|---|---|---|---|
| `q_full`  (λ=1, full self-KV)       | [0.679, 0.759, 0.793, 0.822, 0.834, 0.835, 0.847] | **5.056405** | central 5.0564 | 0.0 |
| `q_floor` (λ=0, full self-KV starve)| [0.674, 0.519, 0.580, 0.645, 0.679, 0.674, 0.617] | **3.534581** | floor 3.5346 | 8.9e-16 |

`endpoints_reproduce = True`. Unlike #172's λ-knob — which pinned depth-1 at 0.679 and swept
only depth≥2, so its λ=0 was 3.5445 (not the true floor) — this map interpolates **depth-1 too**
(0.674↔0.679), so all-λ=0 reproduces 3.5346 and all-λ=1 reproduces 5.0564 exactly. Cross-method
M1 (`et_backward`) == M2 (`et_pathenum`) to **4.4e-16** at the realistic anchor.

## 2. Anchoring λ to the live liveprobe

openevolve's liveprobe measured the depth-1 deficit on the as-built stack:
`walk_topw0_hit = 0.6927` (tree-walk depth-1 top-1 hit) **<** `linear_top1 = 0.7287` (linear
chain depth-1 = full self-KV). The recovery fraction = the fraction of the **maximum self-KV
deficit** the as-built stack has closed:

```
λ̂_1 = (walk_topw0_hit − q_floor_d1) / (linear_top1 − q_floor_d1)
     = (0.6927 − 0.674) / (0.7287 − 0.674)  =  0.34186
```

`q_floor_d1 = 0.674` is the oracle's self-KV-starved depth-1 (the λ=0 floor); `linear_top1` is
the liveprobe's own full-self-KV reference. λ̂_1 is a dimensionless deficit-closure fraction
that transfers onto the model's per-depth segments. **The model's depth-1 full endpoint (0.679,
BUG-1-unfixed descent-only) is held distinct from the liveprobe's full (0.7287, BUG-1-fixed) —
that gap is the separate BUG-1 axis, NOT part of λ.**

`main0_accept = 0.4974` and `tok/step = 2.583` are the as-built *realized* (compounded) rates —
both **below** the λ=0 floor 3.5346 because the as-built walk also lacks the descent re-seeding
topology (consistent with #172 `ORACLE_E_T`=2.621). Reported as context, **not** used as the
conditional λ anchor.

**Deeper-depth profile** (only depth-1 is measured — the rest is MODELLED, stated):
- **constant-λ (PRIMARY):** `λ_d = λ̂_1 ∀d`. Minimal assumption — carry the one measured point
  flat. This is the *optimistic-among-realistic* choice (self-KV starvation in fact worsens with
  depth), so a miss here is robust to the assumption.
- **geometric decay (conservative band):** `λ_d = λ̂_1·γ^(d−1)`, γ<1.

| profile | λ_d (depth 1..7) | E[T] | TPS @τ=1 | clears 500? |
|---|---|---|---|---|
| constant (γ=1.0, **primary**) | [0.342]×7 | **3.9294** | 404.1 | **No** |
| geom γ=0.90 | [0.342, 0.308, …, 0.182] | 3.8353 | 394.4 | No |
| geom γ=0.80 | [0.342, 0.273, …, 0.090] | 3.7664 | 387.3 | No |
| geom γ=0.70 | [0.342, 0.239, …, 0.040] | 3.7139 | 381.9 | No |

Decay only lowers E[T] → the miss is robust to the deeper-depth assumption; constant-λ is the
ceiling of the realistic band.

## 3. Realistic-floor E[T] + clear-500 verdict

`official = K_cal·(E[T]/step)·τ`, K_cal=125.268, step=1.2182, τ∈{1.0 central, 0.9924 conservative}.

| corner | bar E[T] | realistic E[T] | TPS | clears 500? | margin |
|---|---|---|---|---|---|
| τ=1.0 (central)       | 4.8624 | 3.9294 | **404.06** | **No** | −95.94 |
| τ=0.9924 (conservative)| 4.8996 | 3.9294 | **400.99** | **No** | −99.01 |

**λ-threshold (minimum self-KV recovery to clear 500):** `λ* = 0.9091` (τ=1.0) / `0.9271`
(τ=0.9924). The realistic anchor λ̂=0.342 sits **far below** λ*. (λ*=0.909 matches #172's
spread-recovery threshold 0.908 — consistent.)

`E_T(λ)` curve (monotone, bracketed by [3.5346, 5.0564]): crosses 500 only at λ≈0.91; at λ̂=0.342
it is 3.93 (404 TPS).

## 4. Both-bugs cross-check (the same self-KV λ governs depth≥2)

Both-bugs fixes depth-1 (BUG-1 → 0.7287) **and** depends on the same depth≥2 self-KV recovery λ
(BUG-2 is a shared kernel property; the both-bugs delta is only the depth-1 fix).

| both-bugs | E[T] | TPS @τ=1 | clears 500? |
|---|---|---|---|
| full (λ=1) | 5.2070 | 535.4 | Yes (+35) |
| floor (λ=0)| 3.6427 | 374.6 | No |
| **realistic λ̂=0.342** | **4.0485** | **416.3** | **No** (−84) |

**λ*_bb = 0.8384** (τ=1.0) / `0.8570` (τ=0.9924); `robust_across_realistic_range = False`. So
the depth-1 BUG-1 fix buys only **~+0.12 E[T] (~+12 TPS)** at the realistic floor — both-bugs is
the *safer first shot* (lower threshold than descent-only's 0.909, +35 vs +20 TPS top-margin) but
**not robust at the measured floor**. This refines fern #174's GO path: both-bugs' robustness was
the *λ=1* central; the realistic floor shows that assumption is load-bearing for both-bugs too.

## 5. Self-test (PRIMARY) — `realistic_selfkv_floor_self_test_passes = True`

- `endpoints_reproduce_5p0564_3p5346`: **True** (5.056405 / 3.534581, resid ≤8.9e-16)
- `E_T_monotone_and_bracketed`: **True** (monotone in λ; bracketed by [3.5346, 5.0564])
- `lambda_hat_in_unit_and_clear500_explicit`: **True** (λ̂=0.342 ∈ [0,1]; clear-500 bool at both τ)
- `lambda_star_reported`: **True** (0.9091 / 0.9271)
- NaN-clean: **True**

## Hand-off for fern #167's packet (replaces #172's binary floor)

> **Descent-only clears 500 iff the built deep-spine self-KV recovery λ ≥ 0.909; the
> liveprobe-anchored realistic estimate is λ̂ = 0.342 → E[T] = 3.9294 → MISSES 500 by ~96 TPS
> (404).** Both-bugs needs λ ≥ 0.838 and at λ̂ also misses (416) — safer first shot, not robust at
> the measured floor; the binding constraint for both paths is self-KV recovery, not BUG-1.

**Honest scope / what is NOT closed:** the liveprobe is one measured depth-1 point; the
depth-1→depth>0 transfer (anchoring the BUG-2 self-KV λ on a depth-1 deficit the PR itself calls
BUG-1) and the deeper-depth λ profile are **modelled**. λ̂ is also sensitive to the floor
reference (0.674) — a more-starved floor raises λ̂, a less-starved one lowers it. The one truly
unmeasurable closure remains **land #71's built-kernel descent ladder q[2..9]** (#172's suggested
follow-up): measuring it converts this liveprobe-anchored modelled floor into a measured one and
collapses the λ uncertainty.
