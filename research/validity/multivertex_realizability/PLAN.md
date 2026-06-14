<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #208 — Multi-vertex realizability: is 0.9780 the worst case over ALL realizable blends?

LOCAL CPU-only analytic synthesis over EXISTING banked numbers (#176 per-axis
deficits + #203 reach-weights + #193 β). No GPU, no vLLM, no HF Job, no
submission, no served-file change, no kernel deploy. Greedy/PPL untouched,
BASELINE unchanged (481.53). It optimizes over numbers already computed — takes
NO draws, authorizes nothing, adds 0 TPS. `--wandb_group private-drop-shape-robustness`.
Output under `research/validity/multivertex_realizability/`.

## The last open assumption (the one this closes)

My #203 (`hexhagf6`, MERGED) proved the both-bugs private bar is a **monotone
function of the reach-weighted deficit Σ_d w_d·δ_d**, and that #176's
non-Latin-script (NLS) vertex is the worst case **over a single-axis synthetic
shape family** (re-scaling the ONE measured NLS centered deviation) → worst-case
bar 0.977978 = the operative 0.9780. But #203 flagged honestly that the claim
"NLS maximizes Σ w·δ among ALL realizable domain blends" was **argued from #176's
construction, not proven** over a richer polytope. If a convex BLEND of #176's
measured per-axis deficits produces a HIGHER Σ w·δ at the reach-weights than the
pure NLS axis, the true worst-case bar is STRICTER than 0.9780. This is the LAST
open assumption in the private-validity chain (#176 → #191 → #198 → #203).

## What #208 produces

Close the assumption with an explicit optimization over #176's SIX banked
per-axis adverse domains (NLS + code + casual + sharegpt + math + long-context):

1. **Realizable blend polytope.** Each axis a is one of #176's measured
   adversarial domains, count-pooled with the public reference in CUMULATIVE-ladder
   space at the weight W_a that lands the decode drop on GT-4.3%:
   `C_mix[d]=(1−W_a)·C_pub[d]+W_a·C_hard_a[d]`, `q_adv_a[d]=C_mix[d]/C_mix[d−1]`,
   `δ^(a)_d = 1 − q_adv_a[d]/q_pub[d]`. This reproduces #176's banked NLS adverse
   vertex (`q_native`) byte-exactly. A realizable blend is the convex combination
   `δ_d(p)=Σ_a p_a·δ^(a)_d` (p≥0, Σp=1). Report `realizable_axes` + `blend_constraints`.

2. **The core LP.** `max_p Σ_d w_d·δ_d(p)` at #203's reach-weights `w_d_at_bar`.
   The objective is LINEAR in p → optimum at a VERTEX (single axis); the LP reduces
   to "which calibrated axis maximizes Σ w·δ?" Report `argmax_blend`,
   `max_weighted_deficit_pp`, `optimum_exceeds_nls` (bool), margin. Vertex-optimality
   reconfirmed by a 200k-point Dirichlet interior sweep.

3. **Map the optimum to the bar (TEST).** Feed `δ_d(argmax)` through #198/#191's
   forward map + #183's bisection → `both_bugs_bar_worstcase_blend`. Report
   finite-or-UNREACHABLE, `delta_vs_203_worstcase`, and whether 0.9780 STANDS.

4. **NO-GO robustness at the true worst case.** Does the realistic floor λ̂₁=0.342
   still miss the bar? Report `nogo_robust_worstcase_blend` + floor→bar gap in λ.

5. **β-robustness (SECONDARY).** Re-solve the LP with `w_d(β)` at #193's β-range
   endpoints [0.6165, 0.9496]. Report `argmax_blend_beta_stable` + worst-case bar
   per β endpoint. (Low β steepens the falloff → more weight on shallow rungs →
   STRENGTHENS front-loaded-is-worst.)

6. **Self-test (PRIMARY).** (a) NLS vertex → #203's 0.977978 EXACTLY (resid 0.0);
   (b) LP optimum ≥ NLS deficit (NLS is feasible → sanity floor); (c) flat blend →
   #203's c=0 flat bar 0.945302 EXACTLY; (d) bar monotone in Σ w·δ; (e) β-endpoint
   LPs NaN-clean with per-β argmax; (f) NaN-clean (UNREACHABLE → null). Report
   `multivertex_self_test_passes` (PRIMARY) + `both_bugs_bar_worstcase_blend` (TEST).

7. **Hand-off** to fern #185 + land #71's per-rung co-log (one sentence each).

## The constraint-set call (stated explicitly, per PR step 1)

The PR asked to "carry the SAME total-mass constraint Σδ=0.04169 you used in #203."
Applying that **fixed-mass** normalization to the multi-axis case is a DEGENERATE
counterfactual: the near-mass-balanced axes (math Σδ=0.0084, longctx Σδ=−0.056)
get scaled 5–9× (or sign-flipped) into >5% per-rung deficits that correspond to NO
decode-4.3% domain — they violate the very calibration #176 held. The PHYSICAL
invariant #176 actually held is **decode drop = GT-4.3%**, not Σδ. So the headline
realizable polytope is **decode-drop-realizable (natural mass)**: each vertex is a
real #176 domain at the 4.3% calibration, and the deficit mass Σδ VARIES across
axes (NLS happens to carry the LARGEST, Σδ=0.04169). The fixed-mass arm is reported
as a flagged SECONDARY shape-ordering arm — its "winners" are super-NLS shapes →
private-UNREACHABLE (exactly #203's more-shallow-than-NLS c_crit finding), a
STRONGER NO-GO, not a higher finite go-bar. Both framings agree 0.9780 stands.

## What it finds (honest)

Among the six DECODE-DROP-realizable axes, **NLS maximizes Σ w·δ = 2.349 pp** at
the bar's reach-weights, beating the runner-up (code, 2.331 pp) by ~0.018 pp, and
stays the argmax across #193's whole β-range (worst-case bar band [0.977978,
0.978015]). `optimum_exceeds_nls=False`; `both_bugs_bar_worstcase_blend=0.977978`
(resid 0.0 vs #203); NO-GO robust (floor→bar gap 0.636 in λ). **0.9780 STANDS** as
the operative worst-case FINITE go-bar over all realizable blends. NLS is the
unique vertex that pairs a front-loaded SHAPE with the LARGEST realizable MASS;
every other axis is either less front-loaded or carries a smaller realizable mass
→ smaller reach-weighted deficit, looser bar (down to 0.9608 for long-context). The
only way to beat NLS is the fixed-mass degenerate (super-NLS) counterfactual, which
is private-UNREACHABLE — a strictly stronger NO-GO.

## Reused tools (imported, NOT re-derived)

- `research/validity/lambda_private_drop_shape/shape_robustness.py` — #203's
  `Mechanism`, `reach_weights`, the forward Σ w·δ → bar map + bisection, constants.
- #203 manifest `lambda_private_drop_shape/results.json` — w_d_at_bar, δ_d_meas,
  Σδ=0.04169, flat bar 0.945302, NLS bar 0.977978.
- #176 manifests `private_adverse_skew/proxies_native_6axis.json` + `results.json`
  — the six per-axis components, pool weights, the NLS adverse vertex (`q_native`).
- #198 `lambda_private_drop/lambda_private_drop.py` — the coupled forward map anchor.

## Constants (banked, confirmed from #203/#176 results.json)

w_d_at_bar=[0.4128, 0.3359, 0.2710, 0.2395, 0.1717, 0.1033, 0.0895] (4.61× falloff
@β=0.7651), Σδ_nls=0.04168998, τ_low=0.9924318649, β_primary=0.765124, β-range
[0.6165, 0.9496], λ_floor (λ̂₁)=0.34186, NLS bar 0.977978, flat bar 0.945302,
c_crit=−1.672 → UNREACHABLE.

## Plan / status

1. [setup] Assemble the six decode-drop-realizable axes from #176 + the polytope. **DONE**
2. [LP] Vertex-enumeration argmax + Dirichlet interior-sweep confirmation. **DONE**
3. [map] Forward map of each vertex → bar; worst-case + delta-vs-203. **DONE**
4. [robustness] NO-GO at the worst case + β-endpoint re-solves. **DONE**
5. [self-test] Conditions a–f + NaN-clean; W&B run in the group. **DONE** (`wi4gxxx8`)
6. [report] reconciliation JSON + PR SENPAI-RESULT (incl. fixed-mass methodology note).
