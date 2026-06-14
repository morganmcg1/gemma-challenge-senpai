# Measured-ρ-optimal M=32/M=16 draft-tree topology + salvage oracle (PR #83)

**Closure of the acceptance-cost-model axis (#49 → #74 → #76 → #79) for the tree-verify build.**
#74 built the served M=32/M=16 parent arrays with a BORROWED, FLAT cumulative rescue
ρ=0.565 (EAGLE-3). #79 MEASURED the real per-rank rescue ladder on our own deployed
stack and it is steeply DECLINING — ρ₂=0.4165 ≫ ρ₃=0.2655 > ρ₄=0.1908. This PR
re-runs the Sequoia/DP tree optimisation with the measured ladder (replacing the borrowed
flat ρ) to produce the true-ρ-optimal topology land #71 should build, plus the per-position
salvage oracle for land's debug-gate.

Run: `6tghbnjn` (W&B). Local, CPU-only, analytic. No GPU, no vLLM, no HF Job, no
served-file change. Reuses #49/#76/#79 machinery verbatim.

---

## Headline

| quantity | value |
|---|---|
| **primary** `measured_rho_optimal_M32_gain_pct` | **+18.2%** (drafter-aware) · +23.3% (PR-literal #68 M-only, depth-matched) |
| **test** `expected_pooled_branch_hit_salvage` (ρ₂ gate) | **0.4165** |
| measured-ρ-optimal M=32 | **depth 9, max-branch 3**, E[T]=5.207, wall_tps **536.6**, official **569.0** |
| #74 borrowed-ρ M=32 | depth 9, max-branch 4, E[T]=5.157, wall_tps 531.5 |
| **re-optimisation win over #74** | **+0.96% E[T] = +1.13pp TPS = +5.1 wall_tps** (cost-model-INDEPENDENT) |
| M=16 optimal | depth 9, max-branch 2, E[T]=4.631, wall_tps 505.6 |
| M=32 dominates M=16 | **True** |
| width-4 used by optimum? | **No** — max-branch-3 is optimal; width-4 adds 0 E[T]; width-5+ never pays |
| **decision** | **hand land #71 the re-optimised M=32 array** (re-opt > wall_tps MDE), but the win is modest — confirm with a direct wall_tps A/B |

**The one substantive finding:** the measured DECLINING ladder pulls the M=32 optimum from
#74's **max-branch-4 down to max-branch-3**. The 4th child never clears its node-budget
opportunity cost under the real ρ ladder — a node spent on a rank-4 branch
(marginal ≈ ρ₂ρ₃-survived·ρ₄·(1−q) ≈ 0.022) is always beaten by a rank-2 child or a spine
extension elsewhere. This is the PR's hypothesised mechanism ("rank-4 worth less"),
confirmed and sharpened: rank-4 is not just worth less, it is worth **nothing at the M=32
margin**. Re-allocating those nodes is the entire +0.96% E[T] re-opt win.

---

## 1. The depth-dependent Sequoia DP (the one new piece of machinery)

#74's `build_sequoia_tree` optimises a POSITION-INDEPENDENT per-rank vector p[r]. The
measured model is DEPTH-DEPENDENT: the rank-1 spine acceptance q[d] RISES with depth
(0.729→0.847, #76) and the rank≥2 rescue follows the measured chain-rule ladder applied
to the residual (1−q[d]):

```
pv[d][1] = q[d]                                       (#76 conditional acceptance)
pv[d][r] = Π_{j<r}(1−ρ_j) · ρ_r · (1−q[d])   r=2..W   (#79 measured ladder)
```

so a rank-r child's edge weight depends on its ABSOLUTE depth d. `build_depth_dp`
generalises the Sequoia Alg-1 DP by indexing the cost-to-go table on absolute depth
(`Tmax[m][d]`, `G[m][d][b]`); the recursion is #74's with the single substitution
`p[b] → pv[d+1][b]`. Because all children of a node share depth d+1 and pv[d+1][·] is
monotone non-increasing in rank, the Sequoia exchange argument still proves optimality.

**Validated rigorously the same way #49/#74 did:**
- `selfcheck PASS`: depth-DP == EXHAUSTIVE brute force over all labelled rooted trees for
  n≤7 (incl. depth-VARYING pv, max-branch ∈ {2,4}, depth caps {n,3});
- depth-DP == original position-independent `build_sequoia_tree` to 1e-9 when pv is held
  depth-constant (n ∈ {8,16,24,32}).
- **Anchor:** F_linear(8) under the measured pv = **3.84445** == measured E[T]=3.8441
  (#76), |gap| = 3.2e-4 (log rounding).
- **MC cross-check:** 400k-trial greedy simulation of the optimal M=32 tree gives
  E[T]=5.214 vs analytic 5.207 (|err| 0.007).

---

## 2. Measured-ρ-optimal topology (parent arrays for land #71)

**M=32 (build target):**
```
[-1,0,0,0,1,1,1,2,3,4,4,5,7,9,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,28,29]
```
depth 9 · max-branch 3 · 4 rank-2+ branch points · width-by-depth {1:3, 2:5, 3:4, 4:5,
5:4, 6:4, 7:3, 8:2, 9:1} · spine branch-widths [3,3,2,2,1,1,1,1,1].

**M=16 (cross-check / fallback):**
```
[-1,0,0,1,1,2,3,4,5,6,8,9,11,12,13,14]
```
depth 9 · max-branch 2 · spine branch-widths [2,2,1,1,1,1,1,1,1]. At the M=16 budget the
optimum can only afford width-2 branches at the top two spine positions over a near-linear
depth-9 chain.

---

## 3. ⚠ Cost-model note for advisor review (why the headline is +18.2%, not #79's +21.8%)

The PR asks for the "#68 real GEMM cost curve (acceptance-independent, unchanged)" — i.e.
the **M-only** cost where the decode-step multiplier depends only on the verify node-budget
M. That is correct for **re-pricing a fixed shape** (#74/#79), but it **breaks when the DP
is allowed to choose depth** (this PR): under the M-only cost a deeper rank-1 spine is
nearly free (rising q[d] makes deep spine cheap), so the unconstrained DP balloons the
M=32 spine to **depth 18** for a degenerate **+32.4%** "optimum" that would actually tank
wall_tps.

The physical regulariser is the **drafter-depth cost**. From my own #69/#77 drafter
profiling, the MTP drafter runs `depth` SEQUENTIAL weight-re-reading passes and is
**15.5–18.1% of the 11.6 ms decode step at the deployed K=7 chain** (central 16.8%). So the
decode-step multiplier vs the deployed (M=8, depth=7) chain is:

```
step_mult(M, depth) = (1 − g_v − g_d) + g_v·GEMM68(M)/GEMM68(8) + g_d·(depth / 7)
                       g_v = 0.532 (verify-M, #68)   g_d = 0.168 (drafter-depth, #69/#77)
```

`g_d = 0` recovers the PR-literal M-only cost. With `g_d = 0.168` the deep-spine artifact
collapses (depth-18 → +6.7%) and the optimum sits at **depth 9, +18.2%** — same depth as
#74, which is why the comparison below is clean.

**Crucially, the re-optimisation DELTA over #74 is cost-model-INDEPENDENT:** the re-opt tree
and #74 are BOTH depth-9, so they pay identical cost, and the +0.96% E[T] / +1.13pp TPS win
holds for any g_d. The cost-model choice only moves the ABSOLUTE headline (+23.3% M-only ↔
+18.2% drafter-aware), not the topology, not the re-opt win, not the salvage oracle.

Reconciliation with #79's +21.8%: under the SAME M-only cost as #79, my re-pricing of
#74's shape reproduces **+22.2%** (523.3 on the old 428.37 base — matches #79's +21.8% /
521.6), and the re-optimised tree is **+23.3%** (depth-matched), i.e. **+1.1pp over #74**.
The headline dropping to +18.2% is the drafter-depth correction, NOT a worse topology.

---

## 4. Re-optimisation vs #74 — what shifted, and is it worth a re-build?

Both at depth 9, priced identically (cost_mult 1.146):

| | E[T] | max-branch | spine widths | gain (drafter-aware) | wall_tps |
|---|---|---|---|---|---|
| #74 (borrowed flat ρ=0.565) | 5.157 | 4 | [4,3,3,2,2,1,1,1,1] | +17.04% | 531.5 |
| **measured-ρ-optimal** | **5.207** | **3** | [3,3,2,2,1,1,1,1,1] | **+18.17%** | **536.6** |
| **Δ re-opt** | **+0.96%** | −1 | width pulled off rank-4 | **+1.13pp** | **+5.1** |

The declining ladder pulls breadth **off the deep rank-4 child** (#74 spent a 4th child at
spine position 1 and rank-3 children deeper) and toward **rank-2 breadth + spine** exactly
as hypothesised. `reopt_within_wall_mde = False` (+1.13pp ≫ lawine #72's 0.1–0.2% MDE),
`materially_better = True`.

**Decision: hand land #71 the re-optimised M=32 array.** Caveat — the win is **modest**
(+0.96% E[T], ~5 wall_tps, ~1/18th of the tree's total +18% advantage). #74 is NOT broken;
it is structurally within ~1% E[T] and merely over-invested in width-4 under the flat-ρ
assumption. The +0.96% is a clean, cost-model-independent DP result, but it rests on
chain-rule independence + depth-pooled ρ₃/ρ₄, so the realised gain should be **confirmed by
a direct wall_tps A/B of the two arrays** (lawine #82's runner) before treating it as
banked. If the A/B can't resolve +1.1pp, banking #74 as-is is also a clean outcome.

---

## 5. Per-position salvage / branch-hit ORACLE (land #71's debug-gate target)

For the chosen M=32 topology, the EXPECTED fraction of first-divergence steps salvaged at
each spine position (using per-depth ρ₂ from #79 where measured, pooled ρ₃/ρ₄):

| spine pos k | branch width | q_spine | ρ₂(k) | E[salvage] full-width | E[salvage] rank-2 only | first-div weight |
|---|---|---|---|---|---|---|
| 1 | 3 | 0.729 | 0.397 | **0.557** | 0.397 | 4403 |
| 2 | 3 | 0.759 | 0.431 | **0.582** | 0.431 | 2842 |
| 3 | 2 | 0.792 | 0.413 | **0.413** | 0.413 | 1887 |
| 4 | 2 | 0.822 | 0.428 | **0.428** | 0.428 | 1286 |
| 5–9 | 1 | 0.83–0.85 | 0.41–0.44 | 0 (no branch) | 0 | 972/832/647/… |

- **Universal rank-2 gate** (any width-2 branch at a first divergence, topology-independent):
  **ρ₂ = 0.4165** = `expected_pooled_branch_hit_salvage` (the test metric). A correctly
  functioning width-2 branch must read **≈0.41**, NOT byteshark's broken tree-v2 **0.033**.
- **Full width-4 ceiling** (all ranks 2–4): cov₄ = **0.6532** of divergences rescuable.
- Topology-weighted pooled salvage for THIS tree: full-width **0.422**, rank-2-portion
  **0.334** (lower than 0.4165 because positions 5–9 have width-1 / no branch).

The per-position table is the gate land #71 checks against — a number to hit at each
position, not just ">3%".

---

## 6. Width / branch-factor verdict under the measured ladder

Best drafter-aware tree achievable under each branch cap (each depth-swept identically):

| branch cap | M=32 E[T] | M=32 gain | M=16 E[T] | M=16 gain |
|---|---|---|---|---|
| ≤2 | 5.182 | +17.60% | 4.631 | +11.34% |
| ≤3 | **5.207** | **+18.17%** | 4.631 | +11.34% |
| ≤4 | 5.207 | +18.17% | 4.631 | +11.34% |

- **M=32: width-3 buys +0.57pp over width-2; width-4 buys +0.00pp over width-3** → the
  optimum is **max-branch-3**; allowing 4 (or, in a stress test, up to 6 with an optimistic
  ρ₅=ρ₄=0.19) leaves max-branch at 3 and never places a rank-5 child.
- **M=16: max-branch-2** (width-3 buys nothing).
- **Beyond width-4 never pays:** the best rank-5 leaf marginal (0.0179) < the least-valuable
  node already placed in the width-4 tree (0.0272), and 34.7% of steps are hard-miss beyond
  top-4 (unrescuable by any width). `beyond_width4_pays = False`, as the PR expected — but
  note the optimum stops one branch SHORT of 4, at 3.

This refines #79's "full max-branch-4 justified" (a marginal-ρ-threshold argument that
ignored node-budget competition): re-optimising the FULL allocation under the measured
ladder, the M=32 optimum is max-branch-**3**.

---

## 7. Three-base re-pricing (lawine #72: relative is the robust number)

Measured-ρ-optimal M=32 (drafter-aware / [PR-literal M-only, depth-matched]):

| base | optimal M=32 | #74 M=32 | re-opt Δ |
|---|---|---|---|
| relative | **+18.2%** / [+23.3%] | +17.0% / [+22.2%] | **+1.1pp** |
| local wall_tps (×454.1) | **536.6** / [560.0] | 531.5 / [554.7] | +5.1 |
| official (×481.53) | **569.0** / [593.8] | 563.6 / [588.2] | +5.4 |
| legacy local (×428.37) | 506.2 / [528.4] | 501.4 / [523.3] | +4.8 |

**Robustness:**
- g_v (verify-GEMM share) sweep ∈ [0.42, 0.70]: re-opt Δ stays **+1.10 to +1.15pp** (flat).
- g_d (drafter share) sweep ∈ [0, 0.25]: M=32-opt gain moves +23.3% → +15.8%; re-opt Δ
  unchanged (both depth-9). g_d is the dominant headline uncertainty; the re-opt win is not.
- M=33 hard cliff (#68 Marlin tile): step_mult 1.284 (+28%) — M≤32 cap holds.

---

## Commands

```bash
# Re-optimise + salvage oracle (run 6tghbnjn):
.venv/bin/python scripts/profiler/rho_optimal_topology.py \
  --wandb-group rho-optimal-topology --wandb-name wirbel/rho-optimal-topology
#   inputs: research/rank_coverage/rank_coverage_results.json (#79 measured ladder),
#           research/accept_calibration/accept_calibration_results.json (#76 depth-q),
#           research/spec_cost_model/verify_gemm_roofline.json (#68 GEMM curve).
#   cost:   g_verify=0.532 (#68), g_drafter=0.168 (#69/#77), base_depth=7.
```

## Peak memory
CPU-only analytic script — **no GPU allocated, no vLLM, no served file touched**. Peak host
RSS **40.8 MiB** (numpy DP tables). W&B run `6tghbnjn`.

## What happened
The hypothesis holds: the measured DECLINING ladder changes the DP-optimal branch
allocation. Re-optimising the M=32 tree with the real ρ ladder (vs #74's borrowed flat
0.565) pulls the optimum from **max-branch-4 to max-branch-3** and buys **+0.96% E[T] /
+1.13pp TPS** — a clean, cost-model-independent win (both depth-9), above the wall_tps MDE,
so land #71 should build the new array. The win is modest; #74 was structurally close and
merely over-weighted width-4. The salvage oracle hands land a per-position gate
(ρ₂≈0.41 universal, 0.56/0.58/0.41/0.43 at positions 1–4) to replace byteshark's broken
3.3% read. **One honest deviation flagged for review:** following the PR-literal M-only
#68 cost while letting the DP choose depth produces a degenerate depth-18 +32% spine; I
added the physically-measured drafter-depth term (#69/#77) to regularise it to depth-9
+18.2%, which is why the headline is below #79's +21.8% (cost model, not topology). Both
cost models agree on the topology, the re-opt win, and the oracle.

## Suggested follow-ups
- **land #71:** build the re-optimised M=32 array (max-branch-3); wire the per-position
  salvage gate (ρ₂≈0.41) into the debug check. Treat the +1.1pp over #74 as a hypothesis to
  confirm, not a banked number.
- **lawine #82:** the re-opt vs #74 A/B is the ideal first wall_tps job for the runner — a
  +1.1pp-MDE-sized, cost-model-independent, served-shape-only delta.
- **Cost-model closure:** the #68 M-only cost should carry the drafter-depth term whenever
  depth is a free variable (it is degenerate otherwise). Worth folding g_d into
  `treeshape_real_cost.py` as a first-class input if future PRs re-optimise depth — flagged
  here rather than changing #74's merged tool unilaterally.
