# PR #337 — Does honest fusion coverage (0.8903) collapse the E[T] lever + force the M=32 cliff?

**PRIMARY `cov_et_tile_self_test_passes` = True** (all 18 checks)
**TEST `honest_envelope_worst` = 444.99** (worst-case compliant TPS under honest fusion coverage; **below 500**)
**REPORT `honest_envelope_clears_500` = False** (worst corner; the binding compliant-500 test)
**W&B `lbuirkpt`** (group `eagle3-cov-et-tile-coupling`) · LOCAL read-only analytic, 0 GPU, 0 TPS

> **Verdict: ENVELOPE-COLLAPSES-ON-E[T]-AXIS.** lawine #330's honest fusion per-depth acceptance
> **c_eff=0.8903** drops the chain-law lever to **E[T]=5.52** (from the linear **E[T]=6.11** at
> c_eff=0.9213). Restoring E[T]=6.11 demands coverage **0.6532** (the linear spine's cov4), which the
> fusion head only reaches by widening to **W≥5 → M≥36 > knee 32**, paying the measured **×1.16981**
> verify-GEMM cliff. The **+16.98% cliff penalty exceeds the +10.76% E[T] gain**, so the best
> operating point is **STAY sub-cliff at W=4, E[T]=5.52**. Scaling fern #325's banked envelope by
> **5.52/6.11 = 0.903** gives **central 470.35 / worst 444.99 — BOTH below 500.** The honest coverage
> kills compliant-500 on the **E[T] (demand) axis alone**, a **second failure mode** independent of the
> binary identity/acceptance bar (0.8903 < 0.9213).

## 1. E[T](c_eff) curve (deliverable 1)

Chain law (stark #331 convention, K=7 deployed depth):

> **E[T] = 1 + Σ_{d=1..7} c_eff^d**

| draft head | c_eff | E[T] | note |
|---|---|---|---|
| deployed LINEAR spine | **0.9213** = a₁ + (1−a₁)·cov4 | **6.1112** | reproduces stark #331's banked 6.11 ✅ |
| honest FUSION head (lawine #330) | **0.8903** | **5.5176** | **−9.71%** lever drop |

E[T] is **strictly monotone increasing in c_eff** (verified on a 0–1 grid), so the coverage shortfall
maps monotonically to a lower E[T] lever — the lever loss is unavoidable, not an artifact of a tie.

## 2. Coverage → width → tile map (deliverable 2)

Salvage relation (denken #320): `c_eff(W) = a₁ + (1−a₁)·cov_W`, a₁=0.7731. Inverting:

- **cov needed to restore c_eff=0.9213** = (0.9213−0.7731)/0.2269 = **0.6532** — exactly the linear
  spine's `cov4` (wirbel #79). Self-consistency: the fusion head must reach the *same* top-4 coverage
  the linear spine already carries.
- **fusion seed** cov4_fusion = (0.8903−0.7731)/0.2269 = **0.5166** (PR seed 0.5165 used rounded a₁).
- **shortfall** 0.6532 − 0.5166 = **0.1366**.

Tile map `M = W·K+1`, knee_Mstar=32 (cliff @ M=33), anchors reproduced: W=1→M=8 (deployed), W=4→M=29
(sub-cliff, 2 blocks, == size-29 corpus).

**Model-free lower bound (load-bearing):** coverage is monotone in W and cov4_fusion=0.5166 <
0.6532, so restoring c_eff needs **W ≥ 5**. The smallest such tree, **W=5 → M=36 > knee 32**, already
crosses the cliff — so the M=33 cliff is **unavoidable for any restoration**.

**Geometric sensitivity (labeled, not load-bearing):** a saturating model `cov_W = 1 − ρ^(W−1)`
anchored at the single fusion seed gives W≈5.37 → **M=43**, also supra-cliff. A larger W only worsens
the supra-cliff case (W=7 → M=50 would also cross the M=49 second cliff). Lower bound and estimate
**agree**: restoration crosses.

## 3. Sub-cliff (A) vs supra-cliff (B) decision (deliverable 3)

Effective TPS lever = E[T] / step.

| operating point | E[T] | step | **lever** | compliant worst |
|---|---|---|---|---|
| **(A) STAY sub-cliff** W=4, M=29 | 5.5176 | ×1 | **5.5176** | **444.99** |
| **(B) WIDEN supra-cliff** W≥5, M≥36 | 6.1112 (restored) | ÷1.16981 | **5.2241** | 421.32 |

**(A) wins** (lever 5.5176 > 5.2241; +5.6%). The two tie at **μ_tie = E[T](0.9213)/E[T](0.8903) =
1.1076**; the real cliff μ=**1.16981** is well past it, so widening can never recover the lost E[T]
without a net loss. (B)'s corners (central 445.33 / worst 421.32) reproduce stark #331's "if-crossed"
numbers exactly — the cross-check that the divide-by-μ branch is consistent.

## 4. Honest envelope re-price (deliverable 4)

Scale fern #325's banked corners (central 520.95 = λ-ceiling cap-bound, worst 492.87 uncapped, both
@E[T]=6.11) by the winning sub-cliff lever ratio **E[T]_honest / E[T]_611 = 5.5176/6.1112 = 0.9029**:

| corner | fern #325 @6.11 | **honest @5.52** | vs 500 | clears 500? |
|---|---|---|---|---|
| central | 520.95 | **470.35** | **−5.93%** | **False** |
| worst | 492.87 | **444.99** | **−11.00%** | **False** |

**Both corners below 500.** `honest_envelope_clears_500` (worst, binding) = **False**.

*Cap subtlety (carried in the artifact):* the central corner is cap-bound at the λ-ceiling; a strict
re-price holding the ceiling fixed would leave central near 520.95 (>500), but the **worst** corner is
the binding compliant-500 test either way — and it is **below 500**. The verdict does not depend on the
central-corner convention.

## 5. Verdict + hand-off (deliverable 5)

**ENVELOPE-COLLAPSES-ON-E[T]-AXIS.** The honest fusion coverage collapses fern #325's compliant-500
envelope to central 470.35 / worst 444.99 — **both below 500** — on the **E[T] (demand) axis alone**.
This is a **second, independent failure mode**: the binary identity/acceptance bar already fails
(0.8903 < 0.9213), but the *same* shortfall *also* kills compliant-500 through the E[T] lever, without
needing the identity bar.

**Hand-off to fern #335.** This refines fern #335's joint compliant-500 AND-gate **demand axis** from a
binary `[c_eff ≥ 0.9213]` gate into the **continuous consequence**
`honest_envelope = fern_envelope × E[T](c_eff_honest)/E[T](0.9213)`. At c_eff=0.8903 the demand axis is
**RED (worst 444.99 < 500)** *before* any private-tax or identity term is applied.

## 6. Self-test (NaN-clean, deterministic)

All **18** PRIMARY checks pass (`tol ≤ 1e-9`, E[T] reproduction `tol ≤ 2e-2`):
01 E[T](0.9213)=6.11 reproduced · 02 E[T](0.8903)=5.52 < linear · 03 E[T] monotone in c_eff · 04
chain-law spot-check · 05 M=8 deployed anchor · 06 M=29 sub-cliff anchor (2 blocks) · 07 cov_needed ==
linear cov4 · 08 salvage round-trip · 09 restore crosses cliff (W≥5→M=36, lower bound) · 10 geom
estimate also crosses · 11 μ from measured tile · 12 A/B decision consistent (A wins) · 13 μ_tie
recovered, real μ exceeds it · 14 supra-cliff corners match stark #331 · 15 honest envelope = fern ×
ratio · 16 both clears_500 = False · 17 verdict collapse · 18 NaN-clean. Two runs produce **identical**
synthesis (determinism verified).

## 7. Honest caveats (carried in the artifact)

1. **DERIVED, not measured:** no EAGLE-3 fusion checkpoint runs here (training-gated). This prices the
   E[T] and tile consequences the measured coverages (linear cov4=0.6532, fusion c_eff=0.8903) imply
   under stark #331's tile map — not a running `EagleProposer`.
2. **Restoring W is model-dependent; the verdict is not.** It rests only on the model-free lower bound
   **W≥5 (M≥36 > 32)**, which any monotone coverage curve satisfies. The geometric estimate (W≈5.4,
   M≈43) is a labeled sensitivity that only strengthens the supra-cliff loss.
3. **Cap-fixed central re-price** would leave central >500; the **worst** corner (444.99 < 500) is the
   binding compliant-500 test in either convention.
4. **0 TPS / re-pricing property:** depends only on the chain law, integer node counts, the measured
   tile boundary, and fern #325's banked corners — not tensor values. **NOT a launch / build /
   served-file change / HF Job / submission.**

## Greedy/PPL-safety certificate

`analysis_only = True`. No served-file change, no emitted-token change, no HF Job, no submission, NOT a
launch, NOT a build. BASELINE **481.53 TPS unchanged**; this leg adds **0 TPS**; greedy decode and PPL
untouched.

## Hand-off

The honest fusion coverage (lawine #330, c_eff=0.8903) collapses fern #325's compliant-500 envelope to
**central 470.35 / worst 444.99 — both below 500** — on the **E[T] (demand) axis alone**, the cheapest
operating point being STAY-sub-cliff (W=4, E[T]=5.52) because the +16.98% M=33 cliff penalty exceeds
the +10.76% E[T] gain from restoring coverage. fern #335's joint AND-gate should adopt the continuous
demand consequence `honest_envelope = fern_envelope × E[T](c_eff_honest)/E[T](0.9213)`: at the honest
fusion coverage the demand axis is RED before private-tax/identity terms. The single cheapest path back
to GREEN is **raising the realized fusion coverage** (closing 0.8903 → 0.9213), not widening the tree —
widening crosses the cliff and loses net.
