<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #191 — Private-side build bar: does the adverse-skew private drop demand a stricter λ than public 0.9052?

LOCAL CPU-only analytic composition over EXISTING banked certificates. No GPU /
vLLM / HF Job / submission / served-file change. BASELINE stays 481.53.
Greedy/PPL untouched. Bank-the-analysis (PRIMARY = self-test, adds 0 TPS).
`--wandb_group private-build-bar`. Output under
`research/validity/private_build_bar/`. **NOT a launch.** NOT open2.

## The un-composed seam this closes

`#176` banked an adverse-skew private certificate that PASSES at λ=1: descent-only
τ-low **504.15 (+4.15)**. But that margin was banked against the public **CENTRAL**
at λ=1 (519.95). `#183`'s BINDING public build bar is on the finite-sample **LCB**
(`public-LCB(λ=0.9052)=500.0`, `public-LCB(λ=1)=520.95` both-bugs / **505.53**
descent). The two certificates live at DIFFERENT lower-bound notions and have
never been multiplied. If land #71's measured λ̂_built lands AT the public bar,
applying #176's adverse private drop to the finite-sample LCB (not the central)
is the launch's most dangerous un-composed corner.

## Composition (imports — NOT re-derived)

```
private_LCB(λ) = public_LCB(λ) · (1 − drop) · τ_corner          # the deliverable bar
private_central(λ) = public_central(λ) · (1 − drop) · τ_corner  # the #176-consistency leg
```

- `public_LCB(λ)`, `public_central(λ)`: denken **#183** (`82uisrez`,
  `lambda_acceptance_card`) `metrics_at(...)["lcb_full_tps" / "central_tps"]` at
  τ=1 — IMPORTED by executing #183's committed machinery (guarantees byte-exact
  reproduction of its forward map: 0.342→404.1, 0.838→486.2, 0.9052→500.0,
  1.0→520.95 both-bugs; 1.0→505.53 descent). LCB = central − z95·√(SE_tps² + σ_hw²),
  σ_hw=4.86 (#159), iid ±10.906 (#175).
- `drop`: stark **#176** (`uzl7ixll`, `private_adverse_skew/results.json`)
  `adverse_vertex` worst-corner tree drop — **2.2999% descent / 2.3503% both**
  (axis = pure non-Latin-script, W_hard 0.290). #176 designs this as the
  worst-realistic-skew CEILING over the diversity-capped (cap 0.5) domain
  simplex, i.e. the conservative UPPER edge — NOT a parametric sampling CI
  (#176 explicitly disclaims a sampling CI; the true private draw is the one
  unmeasurable leg). So `drop_LCB` is operationalized as the adverse-vertex
  ceiling; no further inflation is available from the banked artifacts without
  re-deriving.
- `τ_corner = τ_low = 0.9924318649123313` (#181/#176 tree-class τ floor).

## Deliverables

1. `private_forward_map_spec` — the composition string + provenance.
2. **valid_at_bar** (MUST-HAVE): at λ=0.9052, worst-corner `private_drop_at_bar_pct`
   ≤ 5% disqualification gate → NOT-DISQUALIFIED; `private_lcb_at_public_bar`.
3. **lambda_star_lcb_private** (TEST): solve `private_LCB(λ*)=500`;
   `private_bar_shift_from_public = λ* − 0.9052`; `binding_bar ∈
   {public-0.9052, private-stricter}`.
4. Both bug-paths: `lambda_star_lcb_private` descent (drop 2.2999%) AND both
   (2.3503%); `both_bugs_required_at_private_bar`.
5. **PRIMARY** `private_build_bar_self_test_passes`: (a) private_central at λ=1
   reproduces #176's 504.149 within tol; (b) private_LCB monotone↑ in λ; (c) λ*≥0.9052
   (private ≤ public always) — flag if violated; (d) drop_at_bar reproduces #176
   ≤2.300% & valid_at_bar=True; (e) public leg reproduces #183 import points; (f) NaN-clean.

## Expected (pre-run, from smoke check)

- private_LCB descent at λ=1 ≈ **490** (< 500): #176's +4.15 → ~−10. Descent-only
  private LCB UNREACHABLE even at full recovery → `both_bugs_required_at_private_bar=True`
  (FLIPS #176's central-based False).
- both-bugs λ*_lcb_private ≈ **0.977** (vs public 0.9052) → `binding_bar=private-stricter`.
- valid_at_bar = **True** (drop 2.30%/2.35% ≪ 5% DQ gate).

## Honest scope

Worst-CORNER product of 4 orthogonal CI axes (finite-sample LCB ⊕ adverse domain
skew ⊕ τ-low; the conservative robust target). Does NOT change the central
projection; does NOT run a private draw; authorizes nothing. Tightens the
private-side bar now; land #71's served per-step traces + a real private draw
confirm. fern #185 consumes this as the private-validity ledger row (and decides
worst-corner-product vs RSS combination of the orthogonal axes).
