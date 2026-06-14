# Tightened private-safe 500-frontier + land #71 min-recovery build gate (PR #162)

**Gate: ARMED** — `tightened_frontier_self_test_passes=1` (all four assertions). Under ubel #154's lowered step bar the public green-area WIDENS 3.00%→3.99%; stark #151's private haircut SHRINKS the safe sub-region to 1.72% (P≥0.5). Inverting the surface gives land #71's build target: at the realistic bar (E[T]=4.809) and the GT private drop 4.3%, the descent kernel must reach **(λ_min=0.8809 spread @ full width, μ_min=0.7353 width @ full spread)** for P(clear-500)≥0.5. `lambda_mu_min_private_safe=[0.8809, 0.7353]`.

This is a **pure-analytic CPU synthesis**: it adds 0 TPS, does not touch the build, and BASELINE stays 481.53. It composes three merged legs — #149 joint (λ,μ) frontier, ubel #154 step-bar reduction, stark #151 private acceptance gap — into one private-conditioned build gate, then self-validates that the composition reproduces each leg's banked anchors.

## Public evidence used
- **#149 joint frontier** (`deep_spine_width_spread_decomp.py`): λ=deep-spine-spread (q[2:] measured→ρ-opt), μ=branch-width (ρ_cond 0→opt), depth-1 held at ρ-opt q76[0]=0.7287. Reproduces public green-area=**0.0300** at bar E[T]=4.8624. Surface machinery (`joint_frontier_surface`, `extract_iso_contour`, `green_area_fraction`) imported verbatim.
- **#155 consolidator** (`approval_projection_consolidator.py`): P(clear-500)=Φ((proj−500)/σ), `combined_rel=√(samp²+calib²+step²)` (step-time-invariant), calib_downside_rel=0.00787, z90=1.28155. Probability model replicated on the private projection (the merged file is not mutated).
- **ubel #154 step bar** (`step_denominator_reduction_audit.json`): recoverable step-pct → step 1.2182 → conservative 1.2078 (−0.857%) / realistic 1.2047 (−1.108%); bars E[T] 4.8624 → 4.8207 / 4.8085.
- **stark #151 private gap** (`tree_private_acceptance_gap/results.json`): banked breakeven tolerances both-bugs **9.88%** / descent-only **5.89%**, band [4%,9%]; depth-1 public 0.72874 / descent-only 0.679.
- **land #71 anchors**: realized (λ,μ) recovery is the build target this gate constrains; oracle E[T]=2.621, both-bugs ceiling 5.207, boundary 4.862.

## Map
```
private_official = K_cal · E_public(λ,μ) · r_tree(d) / step      (K_cal=125.268)
P(clear-500)     = Φ( (private_official − 500) / σ )             (σ from #155 combined_rel)
```

## Step 1 — public frontier re-drawn at three bars (lower bar widens green)
| bar | E[T] | step | step Δ | public green-area | iso-500 λ@μ1 | iso-500 μ@λ1 |
|---|---|---|---|---|---|---|
| original | 4.8624 | 1.2182 | 0.000% | **3.00%** (== #149) | 0.838 | 0.649 |
| conservative | 4.8207 | 1.2078 | −0.857% | **3.74%** | 0.817 | 0.608 |
| realistic | 4.8085 | 1.2047 | −1.108% | **3.99%** | 0.811 | 0.596 |

A −1.1% step reduction lowers the E[T] needed to hit 500, moving the iso-500 contour toward the origin: public green grows +0.99pp (one grid column). The original bar reproduces #149's 0.0300 exactly — the import is faithful.

## Step 2 — private retention r_tree(d) (stark #151 banked; breakevens reproduce exactly)
stark's tolerances are NOT a flat E[T]/TPS haircut (a flat 6.6% scaling would mis-place the breakeven). r_tree(d) is calibrated piecewise-linearly through stark's banked anchors per topology, so the breakeven drops reproduce to machine precision:

| drop d | r_tree (both-bugs) | r_tree (descent-only) |
|---|---|---|
| 4.3% (GT) | 0.9717 | 0.9721 |
| 9.0% (band ceiling) | 0.9398 | 0.9411 |
| 11.3% | 0.9242 | 0.9259 |
| 19.6% | 0.8678 | 0.8711 |
| **breakeven** | **9.880%** | **5.891%** |

Retention is ~topology-independent at a given drop (0.9717 vs 0.9721 at GT, 0.04% apart) — the private haircut acts almost purely on the level, not the shape.

## Step 3 — INVERSION: land #71's build gate
Back-solving the private P=0.5 (and P=0.9 LCB) contour for the minimum per-bug realized recovery at the realistic bar, GT drop 4.3% (r_tree=0.9717):

| target | λ_min @ μ=1 (min spread) | μ_min @ λ=1 (min width) | private-safe area |
|---|---|---|---|
| **P ≥ 0.5** | **0.8809** | **0.7353** | 1.72% |
| P ≥ 0.9 (LCB) | 0.9465 | 0.8764 | 0.42% |

`λ_min(μ)` curve (P≥0.5): the gate is **unreachable for μ ≲ 0.73** — below ~73% width, no amount of spread clears 500 privately. Above it:

| μ (width) | 0.74 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|
| λ_min (spread) | 0.998 | 0.971 | 0.926 | 0.881 |

So land #71 has a trade: full width buys spread headroom down to 88%; at 80% width it needs 97% spread; below ~73% width it cannot clear the private bar at any spread.

## Step 4 — self-test (PRIMARY = all four pass)
- **(a) reproduces stark's 9.88%** — at d = both-bugs breakeven (9.8799%), the (1,1) corner sits exactly on the 500 bar (proj=500.000) and λ_min(μ=1)=1.0 (full spine needed). The composition recovers stark's both-bugs tolerance by construction. ✓
- **(b) green-areas move in opposite directions** — public widens 3.00%→3.99% (+0.99pp, ≥#149's 0.0300); private-safe shrinks to 1.72% (−1.28pp vs #149). Net: the lower bar's public gain is more than cancelled by the private haircut. ✓
- **(c) bracketing anchors survive the lower bar** — oracle 2.621 → proj 272.5 **RED**; both-bugs 5.207 → proj 541.4 **GREEN**; boundary tracks the updated realistic bar 4.8085 → proj 500.0 **INDETERMINATE**. ✓
- **(d) build gate sits on P=0.5 by construction** — each axis intercept maps to P(clear-500)≈0.500 (λ-intercept 0.50023, μ-intercept 0.49999). ✓

`tightened_frontier_self_test_passes = 1`.

## Step 5 — hand-off: is the BUG-1 depth-1 spine (wirbel #160) mandatory for the private shot?
Question: with ubel #154's lower bar ALONE — full BUG-2 descent recovery (λ=μ=1) but depth-1 left UNFIXED at descent-only 0.679 — does the corner still land private-safe-GREEN?

| bar | drop | descent-only corner proj | P(clear) | clears P50? | BUG-1 mandatory here |
|---|---|---|---|---|---|
| realistic | 4.3% (GT) | 511.1 | 0.845 | ✅ | **No** |
| realistic | 6.0% | 505.2 | 0.684 | ✅ | No |
| realistic | 9.0% (ceiling) | 494.8 | 0.317 | ❌ | **Yes** |
| original | 6.0% | 499.6 | 0.486 | ❌ | Yes |
| original | 9.0% | 489.3 | 0.161 | ❌ | Yes |

**Verdict — BUG-1 is a CONDITIONAL requirement, binding only in the upper half of stark's band:**
- The **primary binding margin sits on BUG-2** (the descent kernel: spread+width). The gate (λ_min=0.881, μ_min=0.735) is entirely a BUG-2 target — depth-1 is held at ρ-opt. At the GT drop 4.3%, BUG-2-alone with depth-1 UNFIXED still clears (proj 511.1), so BUG-1 is **not** mandatory at the ground-truth operating point even on the original bar's neighbour.
- **BUG-1 (wirbel #160 depth-1 spine 0.679→0.7287) becomes mandatory once the private drop exceeds ≈6–7%** — at the 9% band ceiling the BUG-1-unfixed corner falls to 494.8 < 500. The lower bar buys ~5.7 TPS of headroom (505.2→511.1 at realistic vs original), enough to keep BUG-1 optional through ~6% drop, but not to the conservative ceiling.
- Practical read for the build: **land #71 (BUG-2 descent) is the unconditional gate; wirbel #160 (BUG-1 spine) is the insurance leg** that must land iff the realized private drop comes in above ~6%. If stark #156 pins the drop at/under ~4.3%, BUG-1 is deferrable; if it lands in the 7–9% conservative band, BUG-1 is required for a P≥0.5 private shot.

## Reproduce
```
cd target && python scripts/profiler/tightened_private_500_frontier.py \
  --wandb --wandb_group tightened-private-500-frontier \
  --wandb_name fern/tightened-private-500-frontier
# CPU-only, ~30 s. --private-drop 0.043 (GT) default; stark #156 pins drop later.
```
Output: `research/spec_cost_model/tightened_private_500_frontier_results.json`. W&B run `0il5xhji`.

**Primary:** tightened_frontier_self_test_passes = 1. **Test:** lambda_mu_min_private_safe = [0.8809, 0.7353].
