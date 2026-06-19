<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #710 — Realizable-recovery frontier (FUSED-BLOCK ALGEBRA + POWER)

**Verdict: `REALIZABLE_RECOVERY_DEAD`.** W&B `66rhys58` (group
`recovery-frontier-denken`). `analysis_only=1, official_tps=0, no_hf_job=1,
fires=0`. Self-test 14/14. Peak RSS 12.1 MB. PURE CPU on logged census.

## The question

Under land #708's fused-block servability algebra (vLLM fuses q/k/v→`qkv_proj`
and gate/up→`gate_up_proj`; a fused block gets exactly one group_size), is there
ANY realizable g32-promotion set whose **predicted AIME Wilson-lower-bound** clears
the 0.420 gate at a **feasible** eval budget (n ≤ ~1000)? Or is the realizable
recovery arm dead?

Inputs cross-read (auth #666, logged data only): ubel #700 `vjhzcvmu`
(full 343-module impact-energy census + names), land #708 `8yf0622s` (per-block
op-bench taxes + fused-vs-standalone map). Proxy reproduces denken #709
`j2884s0i` exactly: `AIME(f)=0.3467+0.0913·shape(f)`, ceiling 0.438 (ubel #679),
shapes linear=f / concave=f^0.5 / convex=f^2, `f*`(0.420)=0.8028.

## (1) Fused-block energy correction — the NEW result over #709's 0.7996

The targeted top-48 subset (40 PLIG + 3q+3k+2v) carries cum-energy **0.7996**,
predicting AIME 0.4197 — #709's knife-edge (ON the gate, −0.0003). The 8
attention modules are fused-blocked: their q/k/v live in **5 layers**
(L0, L1, L18, L40, L41). vLLM forces whole-`qkv_proj` promotion.

| route | servable? | realized cum-energy | predicted AIME (linear) | point clears 0.420? |
|---|---|---|---|---|
| **A** — 40 PLIG only (drop attn) | yes (40 standalone) | **0.6055** | **0.4020** | **NO** (−0.018) |
| top-48 idealized (#709) | no (8 fused-blocked) | 0.7996 | 0.4197 | knife-edge |
| **B** — 40 PLIG + whole-qkv ×5 | yes | **0.8172** | **0.4213** | **YES** (+0.0013) |

- **Route A is point-dead:** dropping the fused-blocked attention loses 0.194
  energy → 0.4020, clearly under the gate.
- **Route B crosses on the point:** whole-qkv promotion drags in *incidental*
  untargeted heads **{q@L1, q@L18, v@L18} = 0.01769 energy**, which EXCEEDS the
  knife-edge gap (`f*`−0.7996 = **0.0032**). Realized energy 0.8172 > `f*`=0.8028
  → predicted point **0.4213 moves OFF #709's knife-edge**. The fused-block drag
  is real and decisive for *point-viability*. (Only L1/L18 contribute: L40/L41
  are KV-shared so their qkv block is Q-only — no incidental k/v; L0's q/k/v are
  all already targeted.)

## (2) The binding wall is POWER, not energy — and it kills even full-g32

The predicted point caps at the ceiling 0.438 (any realizable set has f ≤ 1.0,
shape(1)=1). At the feasible budget the Wilson-lo of *even the maximal set* fails:

| realizable set | point | Wilson-lo n=300 | Wilson-lo n=1000 | clears 0.420? |
|---|---|---|---|---|
| Route B (selective) | 0.4213 | 0.367 | 0.391 | no |
| all-PLIG + all-qkv | 0.4343 | 0.379 | 0.404 | no |
| **full-g32 (max, =ceiling)** | **0.438** | **0.383** | **0.4075** | **no (−0.0125)** |

`max_realizable_predicted_wilson_lo` at n=1000 = **0.4075 < 0.420** (PRIMARY).
**No realizable set — not even full-g32 — clears the Wilson-lo at n ≤ 1000.**
Budget to *prove* even the 0.438 ceiling: n ≥ **2889** (Wilson-lo of an observed
0.438) / n ≥ **9828** (95% power) — both ≫ 1000, reproducing #709. Selective sets
predict < 0.438 → need *more* n still. The recovery is bottlenecked by AIME
small-n variance, not by the energy frontier.

## (3) Proxy-shape sensitivity (compose with stark)

| shape | `f*` | Route B point | Route B point clears? | max-real Wilson-lo n=1000 | DEAD? |
|---|---|---|---|---|---|
| linear | 0.803 | 0.4213 | yes | 0.4075 | **yes** |
| concave | 0.645 | 0.4292 | yes | 0.4075 | **yes** |
| convex | 0.896 | 0.4077 | **no** | 0.4075 | **yes** |

Route B's *point-viability* is **shape-fragile** — it fails under a convex
(superlinear) activation→AIME map. But the DEAD verdict is **shape-INVARIANT**:
the max-realizable Wilson-lo is 0.4075 under every shape because the argmax is
always full-g32 at point = ceiling = 0.438, independent of curvature. **stark's
shape measurement only moves the (moot) point-viability of sub-maximal selective
sets; it cannot flip the power-bound DEAD verdict.** The two cards compose: the
shape would have to be sharply concave *and* the eval budget would have to jump
to thousands of seeds before any selective arm became both point-viable and
provable — and even then full-g32 is the easier, equally-blocked target.

## (4) Speed is not the binding constraint (secondary)

Op-bench additive model (TPS = 1e6/(7889.5 µs + Σ iso Δµs); conservatively
over-taxes vs land #708 measured). All selective sets cost ≤ 0.2 TPS:

| set | additive dTPS | land #708 measured |
|---|---|---|
| Route A (40 PLIG) | −0.13 | **−0.07** (126.68) |
| Route B (≈whole-qkv-8) | −0.19 | **−0.11 to −0.17** |
| full-g32 | −5.67 | **−5.48** (121.27) |

Selective taxes are sub-noise vs the ±2.48 TPS AIME band (denken #706/#709). The
recovery dies on QUALITY provability, not speed.

## Self-test (14/14)

Wilson 24/60 = [0.2857, 0.5263] (fleet-logged); Wilson 50/100 = [0.4038, 0.5962];
`f*`(linear)=0.8028; top-48 energy 0.7996; selective-48 point 0.4197; concave
0.4283 / convex 0.4051 (all reproduce #709); min-n(0.438→Wilson-lo>0.420)=2889;
min-n95 power ≈ 9828 (#709: 9851); energy increments sum to 1.0; full-set point =
ceiling; Route A < top-48 < Route B; fused units partition all 343 body modules
exactly once.

## Hand-off (3-leg recovery-axis closure)

- **denken (this card):** the realizable g32 frontier is **power-DEAD** at feasible
  n. The fused-block correction makes Route B *point-viable* (0.4213, off the
  knife-edge) but cannot rescue *provability*: even full-g32's Wilson-lo = 0.4075
  at n=1000 (need n ≥ 2889/9828).
- **stark (activation→energy shape):** only affects sub-maximal point-viability;
  does not flip the power-bound verdict (shape-invariant max Wilson-lo).
- **land (strict-#319 identity of servable config):** orthogonal — even a clean
  identity cannot clear the gate at feasible n.

The recovery QUALITY axis is closed for FIRING: selective g32 recovery is
unprovable at any feasible eval budget. The fire stays blocked (needs a MEASURED
official speed > 136.378 AND a proven quality clear); this card clears no gate.
