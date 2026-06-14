# PR #198 — λ-dependent private drop: mechanism coupling vs #191's fixed-drop bar

**Status:** in progress (stark). LOCAL CPU-only analytic synthesis. NOT a launch. Adds 0 TPS.

## Question
#191 composed #176's adverse-skew private drop as a CONSTANT (2.35% both / 2.30%
descent) applied uniformly across all λ. denken #193 proved the realized recovery
is NOT flat in depth: λ_d = λ̂₁·β^(d−1). Is the private drop itself a function of
λ̂₁ — and if so, does the λ-coupling move #191's both-bugs private bar (0.9780)?

## Imports (do NOT re-derive)
- #176 `uzl7ixll` (`private_adverse_skew/results.json`): adverse vertex = pure
  non-Latin-script, W_hard=0.2904; per-rung q_pub & q_adv; tree drop both 2.3503% /
  descent 2.2999% measured at λ=1; decode drop 4.2946% (λ-independent DQ anchor).
- #191 `jeclr39w` (`private_build_bar.py`): forward map
  private_LCB(λ)=public_LCB(λ)·(1−drop)·τ_low; λ*_LCB,private=0.9780 (both-bugs);
  descent UNREACHABLE; τ_low=0.9924318649.
- #193 `2clxvlr8` (`lambda_depth_profile`): λ_d=λ̂₁·β^(d−1); β_primary=0.7651,
  construction range [0.6165,0.9496]; β_crit=0.9649; liveprobe λ̂₁=0.342.
- #183 `82uisrez` (`lambda_acceptance_card`): public_LCB(λ) card (constant-λ map).

## Model (the one new equation)
Keep #191's forward map verbatim; replace the scalar drop with **drop(λ̂₁)**:

    drop(λ̂₁) = drop_176 · m(λ̂₁) / m(1)
    m(λ̂₁)    = Σ_d w_d(λ̂₁)·δ_d / Σ_d w_d(λ̂₁)      (depth-mass-weighted deficit)
    w_d(λ̂₁)  = Π_{k≤d} a_pub(k; λ̂₁,β)               (reach mass at depth d)
    a_pub(d)  = q_floor[d] + λ_d·(q_full[d]−q_floor[d]),  λ_d = λ̂₁·β^(d−1)   (#193)
    δ_d       = 1 − q_adv[d]/q_pub[d]                (per-rung adverse deficit, #176)

- **quality component** = drop_176 (depth-flat-δ baseline; λ-independent).
- **acceptance component** = drop_176·(m(λ̂₁)/m(1) − 1) (depth-variation of δ_d; 0 at
  λ=1, >0 below — the coupling). drop RISES as λ̂₁ falls because accepted mass
  concentrates at shallow depths where the adverse domain drafts WORST (δ_d>0 at
  rungs 0–2, δ_d≤0 deep).
- Then private_LCB(λ)=public_LCB(λ)·(1−drop(λ))·τ_low; re-solve λ*_private.

## Self-tests (PRIMARY)
(a) depth-flat-δ limit (acceptance component→0) ⇒ drop≡drop_176 ⇒ bar=#191's 0.9780.
(b) reproduce #176 drop at λ=1 (drop(1)=drop_176) and #191 public import points.
(c) reproduce #191's public leg #183 import points.
(d) conservative ordering drop(λ_low) ≥ drop(λ_high) (positive coupling).
(e) NaN-clean; unreachable bars as null.

## Outputs
`both_bugs_lambda_star_lcb_private_coupled`, `drop_at_lambda_floor`,
`drop_at_lambda_bar`, `drop_is_lambda_dependent`, `private_nogo_more_robust_under_coupling`,
`lambda_private_drop_self_test_passes` (PRIMARY). `--wandb_group lambda-dependent-private-drop`.
