# BUG-2 salvage-descent root-cause (PR #135)

**Gate: GREEN** — descent defect NAMED (salvage-fires-but-does-not-descend (deep-spine decay + rescue-leaf-not-reseeded); spread+width co-dominant); bug2_et_recovery=2.420 E[T] (>> bug1 0.125) -> the build has a concrete BUG-2 target distinct from BUG-1, and BUG-2 alone CLEARS 500.

## Public evidence used
- Oracle readout of `tree-488-pw-fp32-v0` (openevolve board `20260614-100550-487`): E[T]=2.621, depth-1≈0.674/0.679, cumulative ladder [0.674, 0.35, 0.203, 0.131, 0.089, 0.06, 0.037], salvages=391, full=37 over 1024 steps, drafts=2417.
- Banked wirbel E[T] DP (`treeshape_measured_accept.build_depth_pvecs_measured` / `score_tree_depthrank`) on the rho-optimal M=32/depth-9/max-branch-3 topology (#83/#86) + measured rho-ladder [0.4165,0.2655,0.1908] (#79/#86) + deployed rising conditional spine (#76).
- denken #128 anchors reproduced: ET_tree(0.598)=4.8112, ET_tree(0.7287)=5.2070.

## Step 1 — reconstruct realized E[T]=2.621
- spine-only linear DP = **2.5440** (identity 1+ΣC = 2.5440).
- salvage residual vs realized 2.621 = **+0.0770** (2.9% of E[T]) — the rescue the non-descending salvage adds.
- reconstruction residual = 0.00e+00.
- the SAME ladder on the descending mb3 tree → 3.5346 (the descent headroom).

## Step 2 — measured vs ρ-optimal at q1=0.674; localize
- ρ-optimal E[T] @ measured q1 = **5.0413**; descent gap **+2.4203**.
- A(declining,linear)=2.5440, B(rising,linear)=3.6308, C(declining,mb3)=3.5346, D(rising,mb3)=5.0413.
- spread(B−A)=**+1.0868**, width(C−A)=**+0.9906**, interaction=+0.4199 → dominant **spread** (spine reaches depth 7, NOT depth-truncated).

## Step 3 — descent dynamics
- mean accepted depth: measured **1.621** vs ρ-optimal **4.041** (gap +2.420).
- full-tree reach: measured **3.6%** vs ρ-optimal MC **14.1%** (salvage rate 38.2%).
- **Named defect:** salvage-fires-but-does-not-descend (deep-spine decay + rescue-leaf-not-reseeded) — 38% of steps salvage (rank>=2 rescues a divergence) but only 3.6% reach full depth, and the walk realizes mean accepted depth 1.62 vs the rho-optimal 4.04. The spine reaches depth 7, so this is NOT depth-truncation -- it is two co-dominant descent pathologies: (1) the deep spine's conditional acceptance DECAYS with depth (0.67->0.52->0.58..) instead of RISING as the same drafter+verifier does in the linear chain (0.73->..->0.85) -- the walk loses the 'easy run' once it descends; and (2) each rank>=2 rescue is committed as a TERMINAL LEAF rather than becoming a new spine root that RE-DESCENDS its subtree (full reach 14.1% rho-optimal vs measured 3.6%). Both are build defects (not drafter-capacity), recoverable together for +2.42 E[T] at the measured q1.

## Step 4 — BUG-1 vs BUG-2 decomposition
- deficit to clear-500 (4.841) = +2.2200.
- **BUG-2 (descent only)** recovery = **+2.4203** (2.621→5.0413); clears 500 alone: **True**.
- **BUG-1 (spine→0.7287 only)** recovery = **+0.1254** (2.621→2.7464); clears 500 alone: False.
- both fixed → 5.2070 (supply ceiling).
- **BUG-2/BUG-1 = 19.3× → bug2_is_dominant_ceiling = 1**.

### fern hand-off — E[T] columns (official-TPS recovery matrix)
| config | E[T] |
|---|---|
| as_measured | 2.6210 |
| bug1_fix_spine_only | 2.7464 |
| bug2_fix_descent_only | 5.0413 |
| both_fixed_rho_optimal | 5.2070 |

**Primary:** bug2_et_recovery = 2.4203.  **Test:** bug2_is_dominant_ceiling = 1.
