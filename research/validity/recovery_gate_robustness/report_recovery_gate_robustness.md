<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #709 — Recovery gate-clearing robustness (RE-ANCHOR + POWER)

**Verdict: `RECOVERY_GATE_KNIFE_EDGE`.** W&B `j2884s0i` (group
`recovery-gate-robustness-denken`). analysis_only=1, official_tps=0,
no_hf_job=1, fires=0. Self-test 11/11. Peak RSS 12.1 MB, runtime 0.85 s.

## The question

denken #706 (`c5obav63`) closed the SPEED axis of the int4-body recovery on a
linear impact-energy→AIME proxy anchored at the **bf16 base 0.46**. But the
MEASURED full-g32 ceiling is **0.438** (ubel #679 `1z5vq2ej`,
`aime_g32_mean`=0.4375), which clears the 0.420 gate by only **0.018**. Re-anchor
on 0.438 and ask: is the gate-clearing statistically robust, or a small-n
knife-edge that AIME variance can't resolve — and what n does ubel #702 need?

## (1) Re-anchor — the cost of using the achievable ceiling

| | floor | ceiling | lift | f\* clearing 0.420 | min-N modules |
|---|---|---|---|---|---|
| #706 (0.46 base) | 0.3467 | 0.460 | 0.1133 | 0.6470 | **14** |
| re-anchored (0.438) | 0.3467 | 0.438 | 0.0913 | **0.8028** | **37 (scalar) / 48+ (pareto, unreachable)** |

Re-anchoring raises the required cum-energy fraction 0.647→0.803. The 48-module
localized subset captures only ~0.80 of total impact energy (ubel #700 pareto
rank-48 = 0.7996), so the selective-g32-on-48 prediction **collapses from 0.4373
(clears by +0.017) to 0.4197 (ON the gate, −0.0003)**. The cheap-selective-fix
margin is erased: min-N jumps from 14 modules to essentially the whole subset.
(Non-monotonicity flagged: g64=0.446 > g32=0.438; anchoring on g64 gives min-N≈38.)

## (2) Power — the core deliverable (best case: true rate = the 0.438 ceiling)

| n | Wilson half-width | Wilson CI | point-clears 0.420 | straddles | power(clear) |
|---|---|---|---|---|---|
| 30 | ±0.167 | [0.274, 0.608] | no | yes | 0.055 |
| 60 | ±0.122 | [0.316, 0.559] | no | yes | 0.053 |
| 120 | ±0.088 | [0.356, 0.531] | no | yes | 0.072 |
| 240 | ±0.062 | [0.376, 0.501] | no | yes | 0.089 |
| 480 | ±0.044 | [0.394, 0.482] | no | yes | 0.130 |

- **PRIMARY `min_n_for_robust_gate_clear` (95% power) = 9851 trials** — 33× ubel
  #702's planned 5-seed×60 = 300. 80% power = 5966. Even **observing exactly
  0.438**, the Wilson-lo clears 0.420 only at **n ≥ 2889**.
- At ubel #702's n=300: Wilson-lo = 0.382 (< 0.420), power = 0.098. The arm
  straddles the gate at every feasible n.
- **Empirical confirmation:** ubel #679's OWN measured g32 CI **[0.3977, 0.4773]**
  already straddles 0.420 at 4 sessions.

## (3) Pre-registered bands for ubel #702 (n=300 Wilson)

| arm | predicted point | n=300 band | anchor |
|---|---|---|---|
| full-g128 | 0.3467 | [0.295, 0.402] | fixed (lawine #693) |
| full-g32 | 0.4380 | [0.382, 0.493] | fixed (ubel #679) |
| selective-g32-on-48 | **0.4197** | [0.366, 0.477] | predicted |

**Falsification rule:** reject the linear proxy iff the measured selective arm
lands outside point ± (proxy-spread ⊕ Wilson half-width). At n=300 the Wilson
half-width is ±0.056, so the falsification band is so wide the proxy is
**unfalsifiable at the planned n**; falsifying the selective arm needs n ≥ 2889.
The three arms are **not separable** at 300 (full-g32 − full-g128 = 0.091 <
2×half-width).

**SPEED-verdict tolerance:** even if the proxy is wrong and the fix needs the
WHOLE 48-module subset, the tax is 0.66 TPS < the ±2.48 noise band → sub-noise,
so #706's SPEED verdict is **robust** to the re-anchoring. Re-anchoring breaks the
QUALITY margin, not the speed conclusion.

## (4) Proxy-shape sensitivity

| shape | f\* | min-N modules (pareto) | reachable in 48? | selective-48 |
|---|---|---|---|---|
| linear | 0.803 | 48.4 | no | 0.4197 (fails) |
| concave | 0.645 | 22.1 | yes | 0.4283 (clears) |
| convex | 0.896 | 58.6 | no | 0.4051 (fails) |

Module-count gate-clearing-n is **shape-fragile** (2.7× spread; convex →
unreachable within the subset). But the PRIMARY power-n is **shape-INVARIANT** —
it depends only on the fixed endpoints, so the ~9851-trial requirement is
trustworthy regardless of the impact-energy→AIME curvature.

## Self-test (PRIMARY, 11/11)

Wilson 24/60 reproduces the fleet-logged [0.2857, 0.5263]; Wilson 50/100 =
textbook [0.4038, 0.5962]; Clopper-Pearson 0/10 upper = 0.3080, 10/10 lower =
0.6920; exact binomial tail P(X≥1|10,0.5) = 1−0.5¹⁰; betai monotone; min-N
boundary tight; power-n within 25% of the normal approximation; exact power at
min-N ≥ 0.95; #706 reproduced (f\*=0.647, min-N≈13.66); re-anchor strictly raises
min-N; shape block consistent.

## Hand-off to ubel #702

Scale seeds to **n ≈ 9851** (95%) / **5966** (80%) for a Wilson-lo > 0.420 if the
true recovery is 0.438; at the planned 300 the verdict CI straddles the gate
(inconclusive). The selective-g32-on-48 arm is predicted ON the gate (~0.420) and
is unresolvable at any finite n — measure full-g32 and full-g128 as the decisive
anchors, not the selective arm. This card hardens the recovery program's central
proxy and powers ubel #702's design; it does NOT itself clear any gate (the fire
stays separately blocked on quality + a MEASURED official speed > 136.378).
