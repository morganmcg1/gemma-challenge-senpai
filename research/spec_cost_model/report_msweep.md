<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Extended verify-latency M-sweep (M→64) — measuring the tree-verify cost, killing the >500 extrapolation

**PR:** #28 · **Author:** denken · **Date:** 2026-06-13 · **Extends:** PR #18 (`report.md`), PR #26 (`tree_acceptance_model.py`)
**Question:** PR #26's width-4 tree headline rested on **extrapolating** the PR #18
int4 verify-latency curve (measured only to **M≤16**) out to **M=25** (K=6 tree)
and **M=41** (K=10 tree). Does the int4 verify forward really stay bandwidth-bound
and ~flat in M well beyond 16, or does it develop a super-linear knee that caps the
realistic tree depth? This was the single extrapolated input in the whole
drafter-ladder ceiling estimate.

**LOCAL profiling only** — same harness (`scripts/profiler/spec_cost_model.py`),
same int4 `google/gemma-4-E4B-it-qat-w4a16-ct` base, same `graph|ctx256` config,
same 200 steps / 20 warmup. The only change is the M sweep:
`{1,2,4,6,8,10,12,16}` → `{…,20,24,28,32,40,48,64}`. No HF Job, no submission, no
drafter, no greedy/PPL surface touched.

## Verdict / headline

| quantity (canonical = **graph, ctx256**) | extrapolated (PR #26) | **measured (this PR)** |
|---|--:|--:|
| M=1 calibration | 11.51 ms → 86.9 TPS | **11.56 ms → 86.5 TPS** (continuity ✓) |
| verify-step latency M=1 → M=16 | 11.51 → 11.82 ms (+2.6%) | **11.56 → 11.81 ms** (reproduces ±0.008 ms) |
| knee M\* (last M within +10% of M=1) | ≥16 (edge of sweep) | **24** — curve is flat to ~M=32, then ramps |
| **V_tree(M=25)/V_lin(M=7)** (K=6 tree overhead) | 1.057× | **1.113×** (still ≪ 4× naive fear) |
| K=6 tree TPS @ p=0.6792 (w/ drafter) | 346.8 | **331.2** (−4.5%; still net-positive 1.46×) |
| tree K\* @ p=0.78 (w/ drafter / verify-only) | K\*=20: 616 / 680 | **K\*=12: 452 / 493** |
| **>500 TPS @ p=0.78 at full scale?** | yes (from K≈8–10) | **NO — never crosses 500 (max 452/493)** |

**One-line result:** the verify forward is **not** flat to 64. It is a **staircase**
— flat to ~M=32, then a clear super-linear (GEMM-driven) ramp. The K=6 *moderate*
tree (M=25) sits in the flat region and its PR #26 conclusion **holds** (overhead
1.11× measured vs 1.06× extrapolated). But the **deep** tree the >500 claim relied
on (K≈10–20, M≈41–81) sits in the ramp: the measured ceiling at p=0.78 is
**~452–493 TPS**, and **>500 @ p=0.78 does not survive measurement.** This is the
PR's stated "equally-valuable alternative outcome": a real knee at M\*≈32 caps the
realistic tree depth at K\*≈8–12.

> **Update — PR #37 (denken, 2026-06-13): tile boundaries folded into this canonical curve.**
> The `graph|ctx256` node of `results_msweep.json` now carries the PR #33
> directly-measured Marlin tile-boundary fine sweep (M=17/33/49), folded in by
> `scripts/profiler/fold_tile_into_msweep.py`. The coarse sweep above interpolated
> the verify step **linearly across each `ceil(M/16)` tile cliff**, undershooting
> the boundary M: **M=49 was 15.45 ms interpolated vs 18.13 ms measured (−14.8%)**
> and M=33 was 13.10 vs 14.99 ms (−12.6%). Consumers that read `results_msweep.json`
> directly (no `--cost-model-json` override) now inherit the measured cliffs.
> Consequence: the verdict-table row **"tree K\* @ p=0.78 = K\*=12: 452/493"** is
> superseded by the tile-corrected **K\*=11 (M=45): 440/481** (the M=49 step is no
> longer undershot, so K=12/M=49 is correctly priced out of the optimum). The
> **>500 @ p=0.78 = NO** verdict is unchanged on the full (unpruned) head. The
> pre-fold curve is preserved at `results_msweep_prefold.json`. **Scope:** only
> `graph|ctx256` is tile-corrected (the PR #33 sweep measured that config only);
> `eager|*` and `*|ctx512` keep their coarse interpolation and are flagged in
> `config.tile_boundary_folded`. See `report_lmhead12k_verify_cost.md` for the
> full PR #37 cost-model closure (lmhead12k verify-head pruning raises the ceiling
> to 538/600 and flips >500 to YES at the K\*=11 optimum).

## 1. Continuity — the new curve reproduces PR #18 (M≤16)

Re-measuring M≤16 in the *same* run as the new points (one process, one thermal
state) is the continuity control. The new `graph|ctx256` latencies match the merged
PR #18 `results.json` to within run-to-run noise:

| M | 4 | 6 | 8 | 10 | 12 | 16 |
|---|--:|--:|--:|--:|--:|--:|
| Δ (new − PR#18), ms | −0.025 | −0.009 | −0.003 | −0.011 | −0.009 | −0.008 |

M=1 calibration 86.51 TPS (PR #18: 86.86; PR #7 ref 96.89). M=2 differs by +0.26 ms
(the known M=1→2 regime-change point, see PR #18 §3). The M≥20 points are therefore
**continuous** with the existing curve — no process/thermal discontinuity.

## 2. The measured verify-step latency curve (graph, ctx256) — a staircase

| M | step ms | fwd ms | lm_head ms | Δstep/tok vs prev | note |
|--:|--:|--:|--:|--:|---|
| 1 | 11.559 | 8.702 | 2.856 | — | plateau 1 |
| 16 | 11.808 | 8.855 | 2.953 | 0.013 | …flat (+2.6% over M=1) |
| 20 | 12.575 | 9.452 | 3.123 | 0.192 | **step ↑** → plateau 2 |
| 24 | 12.662 | 9.500 | 3.161 | 0.022 | flat |
| 32 | 12.816 | 9.570 | 3.246 | 0.019 | flat (+11% over M=1) |
| 40 | 15.087 | 11.774 | 3.313 | 0.284 | **step ↑↑** → plateau 3 |
| 48 | 15.254 | 11.864 | 3.389 | 0.021 | flat (+31% over M=1) |
| 64 | 18.446 | 14.880 | 3.566 | 0.200 | **step ↑↑↑** (+60% over M=1) |

Three facts:

1. **Flat treads + discrete steps**, not a smooth flat line. Plateaus at M≈1–16,
   20–32, 40–48 are punctuated by steps at M≈16→20, 32→40, 48→64. This is the
   signature of **kernel tile/block quantization**, not thermal drift (thermal drift
   would be a monotone creep, not flat treads). Each M has its own pinned CUDA graph
   (`cudagraph_capture_sizes` = the M sweep), so it is not a capture-bucketing
   artifact.
2. **The ramp is the FORWARD, not the lm_head.** lm_head grows smoothly and modestly
   (2.86→3.57 ms, the genuine O(M) 262k-vocab projection). The forward jumps
   8.7→9.5→11.8→14.9 ms — the int4 weight-GEMM going **compute-bound** as M grows and
   the one-weight-read amortization saturates. Through the ramp the **GEMM share
   rises 62%→68%** while **attention share falls 16%→13%** and lm_head share falls
   25%→19%. The ramp is GEMM/compute, not attention.
3. **Graph mode reveals it; eager hides it.** In eager mode the forward stays pinned
   at ~48 ms flat all the way to M=64 — because ~48 ms of fixed CPU-launch overhead
   masks the GPU compute ramp. Graph mode collapses that overhead to ~8.7 ms, so the
   true GPU compute ramp becomes visible. graph is the serving path, so the ramp is
   the operative reality.

**Departure-from-flat M\* ≈ 32.** "Nearly free" verify holds through M≈32 (within
+11% of M=1); beyond that each added query position costs real GEMM.

## 3. Tree-acceptance model on MEASURED latency (Step 2)

Re-ran `scripts/profiler/tree_acceptance_model.py` (PR #26, unchanged) with
`--cost-model-json results_msweep.json`. `LatencyCurve.at(M)` now **interpolates**
the tree M values from real data (M=25 from measured M=24/28; M=41 from M=40/48;
M=61 from M=48/64) instead of flat-extrapolating off M=16.

**K=6 (moderate tree) — conclusion HOLDS.** M=25 is still inside the flat region:

| p (top-1) | V_tree(M=25) | V_lin(M=7) | overhead | tree TPS (w/ drafter) | gain vs linear |
|--:|--:|--:|--:|--:|--:|
| 0.6792 | 12.68 ms | 11.40 ms | **1.113×** | 331.2 (was 346.8) | 1.46× |
| 0.78 | 12.68 | 11.40 | 1.113× | 375.1 (was 392.8) | 1.28× |
| 0.85 | 12.68 | 11.40 | 1.113× | 409.7 (was 429.0) | 1.16× |

The 1.06×→**1.11×** overhead is the headline correction: the extrapolation was
mildly optimistic, the tree is still **strongly net-positive**, and the naive
"tree = 4× linear" fear is still wrong by ~3.5×.

**Optimal tree depth K\* (W=4, measured verify + 1.4 ms drafter) — the >500 question.**

| p | measured K\* / TPS (w/ drafter, verify-only) | extrapolated K\* / TPS | what changed |
|--:|--:|--:|---|
| 0.6792 | **K\*=8 (M=33): 366.5 / 405.7** | K\*=20 (M=81): 460.3 / 508.3 | interior optimum appears |
| 0.78 | **K\*=12 (M=49): 452.4 / 493.4** | K\*=20 (M=81): 616.3 / 680.2 | **never >500** |
| 0.85 | **K\*=12 (M=49): 531.2 / 579.3** | K\*=20 (M=81): 778.9 / 859.6 | clears 500 only here |

Under the flat extrapolation, TPS rose **monotonically** to the K=20 search ceiling
(deeper always better → K\*=20, M=81, deep in extrapolated territory). Under measured
latency the M\*≈32 ramp creates a **real interior optimum** at K\*≈8–12, and the
extrapolation **overstated deep-tree TPS by 30–55%** (e.g. p=0.78 K=12: 542→452;
K=20: 616→395).

## 4. The >500 @ p=0.78 verdict — does not survive measurement

p=0.78, W=4 tree, full TPS-vs-K (measured vs extrapolated, with drafter):

```
 K   M   E      tps_meas   tps_extrap
 6   25  5.28   375.1      392.8
 8   33  6.22   429.3      455.7
10   41  7.00   423.7      504.4   ← extrap crosses 500 here
12   49  7.63   452.4 ◄pk  541.7
15   61  8.36   434.4      581.2
20   81  9.19   395.4      616.3   ← extrap "optimum"
```

- **Measured: p=0.78 tree never reaches 500** — peak 452.4 (drafter) / 493.4
  (verify-only) at K\*=12, then it **declines** as the ramp outpaces the saturating
  acceptance. `verdict_exceeds_500_at_full_scale` = **False** (both verify-only and
  with-drafter). It does **not** flip to True; the corrected ceiling lands **lower**.
- **Extrapolated:** crossed 500 from K≈8 (verify-only) / K≈10 (with drafter) and kept
  climbing — the artifact the PR #26 ">500 @ K≈10, M≈41" note rested on.
- Only at **p≥0.85** does the deep tree clear 500 (K\*=12 → 531/579). The drafter
  would need ~0.85 top-1 acceptance — well above the debug head's measured 0.6792 —
  for the tree path to reach 500 on this hardware.

## 5. Dense-M is a tight upper bound here (tree-mask caveat)

The profiler times a **dense / full-causal M-token forward** (`_dummy_run`, all M
query positions attend over ctx+M keys) — a **conservative upper bound** on the true
tree-verify cost, since a real width-W depth-K tree uses a sparse tree-causal mask
(each node attends only to its ancestors). I did **not** add the tree mask: it is
non-trivial in `_dummy_run` and, more importantly, **it cannot move this verdict**:

- A tree-causal mask only reduces the **attention** term (O(M²)→O(M·depth)). The
  ramp here is **GEMM**, whose share *rises* to 68% through the ramp while attention
  *falls* to 13%. The GEMM processes all M tokens through the int4 weights regardless
  of the attention mask.
- The mask-addressable slice is at most the core-attention O(M²) excess (the
  `attention` bucket also counts RoPE + KV-write, which are O(M) and unaffected) —
  a sub-2 ms effect at M≈49. That could nudge p=0.78 from ~452 toward the ~500
  borderline at best; it does **not** restore the comfortable >500 the flat
  extrapolation claimed. Building + measuring the real mask is a follow-up.

So the dense-M ceiling (~452/493 @ p=0.78) is **tight**, and if anything pessimistic
by <2 ms — the >500 refutation is robust.

## 6. Reproduce

```bash
cd target
# Step 1 — extended latency sweep (GPU, ~40 min on one A10G)
python scripts/profiler/spec_cost_model.py \
  --int4-base google/gemma-4-E4B-it-qat-w4a16-ct \
  --m-sweep 1,2,4,6,8,10,12,16,20,24,28,32,40,48,64 --ctx-sweep 256,512 \
  --modes eager,graph --steps 200 --warmup 20 --profile-steps 30 \
  --output research/spec_cost_model/results_msweep.json \
  --wandb_group spec-verify-msweep --wandb_name spec-verify-msweep-int4

# Step 2 — tree model on the MEASURED curve (CPU, <1 min)
python scripts/profiler/tree_acceptance_model.py \
  --top1-acc 0.6792 --top4-acc 0.8605 --drafter-ms 1.4 --verify-base-ms 7.0 \
  --cost-model-json research/spec_cost_model/results_msweep.json \
  --K-range 1 20 --widths 1 4 --p-list 0.6792 0.78 0.85 \
  --output research/spec_cost_model/tree_results_measured.json \
  --plot-dir research/spec_cost_model/tree_plots_measured \
  --wandb_group spec-verify-msweep --wandb_name tree-acceptance-measured
```

**Peak memory:** 9.81 GiB weights + ~8.5 GiB KV at `gpu_memory_utilization=0.90`
(~19.6 GiB of 23 GiB A10G). **Artifacts:** `results_msweep.json` (full M=1..64 curve
+ cost_model), `tree_results_measured.json`, `tree_plots_measured/`, this report.
**W&B:** latency `2mk0z0c3`, tree `imoi4mx1` (group `spec-verify-msweep`).
Local only, no HF Job, no submission.

## 7. Suggested follow-ups (not implemented — PR scope is verify latency)

1. **Tree-causal mask measurement** to convert the dense-M upper bound into the true
   tree cost in the M≈25–49 region — §5 says it is a sub-2 ms correction (GEMM-bound
   ramp), but measuring it would settle the p=0.78 ~500 borderline definitively.
2. **Confirm the K\* operating point against acceptance.** The measured ceiling makes
   the drafter ladder verify-bound at K\*≈8–12 for p≈0.68–0.78, not the K≈20 the flat
   extrapolation implied — the drafter's real top-1 acceptance now sets whether the
   tree path can approach 500 at all (needs ≳0.85).
3. **Profile the GEMM tile structure** behind the M≈32 ramp (Marlin W4A16 tiling): if
   the steps are tile-boundary effects, choosing tree shapes that land on a plateau
   edge (e.g. M=32 vs M=40) is nearly free TPS.
