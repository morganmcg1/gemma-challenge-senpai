# Depth-1 spine (BUG-1) build-spec verification (PR #160)

**Gate: GREEN** — spec'd depth-1 fix (contamination f->0) reproduces the both-bugs ceiling 5.2068 with idealization gap 0.00e+00 (≈0); all merged anchors reproduced; greedy-identity safe.

- **PRIMARY** `spine_spec_self_test_passes` = **True**
- **TEST** `both_bugs_E_T_specced` = **5.2070** (idealized override 5.2070; idealization gap **0.00e+00**)
- `spine_fix_greedy_identity_safe` = **True**

## Public / banked evidence used
- denken #133 (MERGED): rank-2 contamination root-cause (`target_logits_indices`), as-built depth-1 0.598, `q1 = (1-f)*q_true + f*rho2`, f≈0.419, rho2=0.4165.
- wirbel #135 (`bug2_salvage_descent.py`) + #152 topology DP: the E[T] DP (`build_depth_pvecs_measured`/`score_tree_depthrank`) on the rho-optimal M=32/depth-9/max-branch-3 topology; idealized both-bugs override 5.207.
- fern #134 / #125 official-TPS matrix: descent-only 5.0564, both-bugs 5.2068.
- ubel #154 lowered clear-500 bar 4.808–4.820; lawine #136 measured step 1.2182.

## Contamination model — the spec'd accept
```
q1(f) = (1 - f) * q_true + f * rho2
  q_true = 0.728740   (verifier rank-1 acceptance)
  rho2   = 0.416505   (rank-2 marginal)
```
| f (contamination) | meaning | q1 |
|---|---|---|
| 0.4190 | denken #133 as-built (deployed) | 0.5979 ≈ 0.598 |
| 0.1753 | openevolve oracle measured | 0.674 |
| 0.1593 | descent-only residual spine | 0.679 |
| **0.0000** | **THE SPEC'D FIX** | **0.72874 = q_true** |

The fix is `f -> 0`. Because `q1(0) ≡ q_true` identically, the spec'd fix and the idealized override coincide **exactly** — there is **no idealization gap**.

## Anchors reproduced (self-test gate)
| anchor | target | modelled |
|---|---|---|
| as-built depth-1 | 0.598 | 0.5979 |
| ET_tree(0.598) (denken #128) | 4.8112 | 4.8112 |
| descent-only E[T] | 5.0564 | 5.0564 |
| both-bugs E[T] (ceiling) | 5.2068 | 5.2070 |

## Official projection — both-bugs spec'd ceiling
- @ measured step 1.2182: **535.4** (τ=1.0) … 531.4 (τ=0.9924)
- @ roofline step 1.2127: **537.9** (τ=1.0) — the fleet's ~537.8 anchor.
- descent-only @ measured step τ=1.0: 520.0 (roofline 522.3).
- clear-500 bar @ measured step: 4.8624 (τ=1.0) … 4.8996 (τ=0.9924); ubel band [4.808, 4.82].
- both-bugs margin: **+0.345** over the measured bar, **+0.387** over ubel's.

## MC cross-check
- Monte-Carlo both-bugs E[T] = 5.1954 vs DP 5.2070 (|Δ| = 0.0116).

See `SPINE_FIX_SPEC.md` for the buildable kernel interface + exact diff.
