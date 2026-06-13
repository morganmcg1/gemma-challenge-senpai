<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# lmhead12k verify-forward cost model + tile-corrected tree ceiling

**PR:** #37 · **Author:** denken · **Date:** 2026-06-13
**Builds on:** PR #14 (ubel, `lmhead12k_empirical`, merged) · PR #28 (`report_msweep.md`, canonical verify curve) · PR #33 (Marlin tile boundaries + tree-causal mask) · PR #26 (`tree_acceptance_model.py`)
**W&B group:** `spec-verify-lmhead12k` · runs `klvpfk7g` (derive+measure), `ruch259z` (tree, measured curve), `6c9r3lih` (tree, analytic curve)

**LOCAL profiling / cost-model only.** No HF Job, no submission, no drafter run,
no greedy/PPL surface touched. The int4 base is unchanged
(`google/gemma-4-E4B-it-qat-w4a16-ct`, `graph|ctx256`, 200 steps / 30 warmup). The
only thing measured here is the **spec-VERIFY forward step latency** when ubel #14's
pruned lm_head (262,144 → 12,288 rows) is swapped into the verify path, and what
that does to the width-4 tree TPS ceiling on the #33 tile-corrected curve.

---

## Verdict / headline

| quantity (canonical = **graph, ctx256**) | full head (#33 tile-corrected) | **lmhead12k (measured)** | analytic ceiling |
|---|--:|--:|--:|
| lm_head verify cost @ M=45 | 3.367 ms | **0.348 ms** (scatter floor) | 0.158 ms (×0.0469) |
| V_tree step @ M=45 | 15.235 ms | **12.212 ms** (−3.02 ms) | 12.022 ms (−3.21 ms) |
| %-step reduction @ M=45 | — | **19.8%** | 21.1% |
| tree K\* @ p=0.6792 (w/ drafter / verify-only) | K11/M45: 359.9 / 393.0 | **K7/M29: 446.6 / 510.2** | K7/M29: 451.7 / 516.9 |
| tree K\* @ p=0.78 (w/ drafter / verify-only) | K11/M45: **440.4 / 480.8** | **K11/M45: 538.1 / 599.8** | K11/M45: 545.8 / 609.4 |
| tree K\* @ p=0.85 (w/ drafter / verify-only) | K15/M61: 511.6 / 558.4 | **K11/M45: 624.9 / 696.6** | K11/M45: 633.8 / 707.7 |
| **>500 TPS @ p=0.78, K\*-optimum, w/ drafter?** | **NO** (440.4) | **YES (538.1)** | YES (545.8) |

**One-line result:** pruning the verify-path lm_head to the top-12,288 vocab rows
removes **~3.0 ms** of memory-bandwidth-bound projection from every verify step (the
weight stream drops from ~1.34 GB to ~63 MB). That is a **~20% step reduction at
M=45** and it **lifts the realistic p=0.78 width-4 tree ceiling from 440 → 538 TPS
with drafter (599 verify-only), flipping the PR #33 ">500 @ p=0.78 = NO" verdict to
YES at the K\*=11 (M=45) optimum.** The win is real but **bounded by a residual
floor**: ubel #14's `compute_logits` still scatters the 12k logits back into a full
[M, 262144] −inf tensor and argmaxes over the full vocab (identity-correctness
requirement), so the measured verify-head cost is **~2.2× the naive ×0.0469
analytic** and the realised saving is **~94% of the analytic ceiling**, not 100%.

### Honest caveats (carried from the PR hypothesis)

1. **lm_head is a *fixed-ish* fraction, not M-dominated.** The full-head lm_head term
   is **86% fixed + 14% M-linear** (`2.900 + 0.01047·M` ms over M∈[16,64]). At the
   K=6 *moderate* tree (M=25) it is only **25%** of the step; at M=45 it is **22%**.
   So the verify-head saving at the serving M is **modest in step terms** (−3 ms of a
   ~15 ms step) — exactly the "small fixed fraction" outcome the PR flagged. It moves
   the ceiling because the tree TPS denominator is `V_tree/E[accept]` and −3 ms is
   ~20% of V, but it is **not** a step-count collapse.
2. **The >500 flip is verdict-lens dependent.** It holds at the **K\*-optimum**
   (the frame PR #33 reported "440/481 @ p=0.78" in). At the **fixed K=sim_K=6 (M=25)
   headline** the same measured curve gives **476.5 with drafter (still <500)** and
   **545.4 verify-only (>500)** — see §3.2. The SENPAI-RESULT `test_metric` is keyed
   to the K\*-optimum lens to match the PR's reference number.
3. **At realistic p=0.6792 the with-drafter optimum stays <500** (446.6). The >500
   flip needs the higher acceptance regimes (p≥0.78). Verify-only crosses 500 already
   at p=0.6792 (510.2).
4. **Analytic vs measured gap is the scatter floor.** Analytic (×0.0469, no scatter)
   = 545.8; measured (real `compute_logits`) = 538.1. The 7.7-TPS gap @ p=0.78 is the
   cost of the retained full-vocab scatter + argmax. The **measured** number is the
   production-faithful one and is what the headline reports.

### Public evidence used

This cost model is parameterised by **publicly documented, measured-on-this-pod**
quantities only: the int4 Marlin GEMM tile structure (`thread_m_blocks =
ceil(M/16)`, vLLM/Marlin public kernel), the gemma-3n hidden size 2560 and vocab
262,144 (public model card), bf16 = 2 bytes/elt bandwidth arithmetic, and the
A10G HBM2 bandwidth implied by the directly-measured lm_head stream time. No
private or AWS-only numbers feed the headline; every latency in the tables is a
CUDA-event measurement from this pod (`klvpfk7g`) or a transparent arithmetic
transform of one.

---

## Step 1 — analytic re-derivation of V_tree(M) with the 12k head

`V_tree(M) = t_forward(M) + t_lmhead(M)`. ubel #14 prunes **only** the lm_head GEMM
weight (the body GEMM and attention are untouched), and the projection is
**memory-bandwidth-bound** (streaming the [2560, vocab] weight once). So to first
order the lm_head term scales with the kept-vocab fraction:

```
KEPT_FRAC = 12288 / 262144 = 0.046875
t_lmhead_12k(M) ≈ KEPT_FRAC · t_lmhead_full(M)
t_forward(M)  unchanged
```

**Does the #28 lm_head term scale with M?** Yes, weakly. Fitting the measured
full-head lm_head curve over M∈[16,64]:

```
t_lmhead_full(M) = 2.900 + 0.01047·M  ms     (R² > 0.99)
fixed share @ M=45 = 2.900 / 3.371 = 86.0%
```

The 86% fixed part is the weight stream (~1.34 GB bf16, M-independent); the 14%
M-linear part is the [M, vocab] write + argmax. Pruning the weight to 12k scales the
**whole** term by ×0.0469 (both the stream and the write shrink with vocab), so the
analytic 12k term is `0.0469·(2.900 + 0.01047·M)`.

**Step-1 analytic table** (`lmhead12k_verify_cost.json::step1_analytic`):

| M | t_forward | lm_head_full | lm_head_12k | head saving | V_full | V_12k | %step ↓ |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1  | 8.702 | 2.856 | 0.134 | 2.722 | 11.558 | 8.836 | 23.6% |
| 7  | 8.554 | 2.842 | 0.133 | 2.709 | 11.396 | 8.688 | 23.8% |
| 17 | 9.432 | 3.093 | 0.145 | 2.948 | 12.525 | 9.577 | 23.5% |
| 25 | 9.511 | 3.171 | 0.149 | 3.023 | 12.682 | 9.660 | 23.8% |
| 33 | 11.731 | 3.255 | 0.153 | 3.102 | 14.986 | 11.884 | 20.7% |
| 45 | 11.864 | 3.371 | 0.158 | 3.213 | 15.235 | 12.022 | 21.1% |
| 49 | 14.732 | 3.402 | 0.159 | 3.243 | 18.134 | 14.891 | 17.9% |

The %-step reduction **falls as M climbs the tile staircase** (23.6% at M=1 → 17.9%
at M=49): the body-GEMM tile cliffs (M=33, M=49) inflate `t_forward` while the head
saving is ~flat, so the head is a shrinking slice of a growing step.

> `t_forward` and `lm_head_full` here are the **#33 tile-corrected merged
> components** (`merged_forward_by_M` / `merged_lmhead_by_M` in
> `lmhead12k_verify_cost.json`): tile-boundary measured rows override the coarse
> msweep at shared M, so V_full already carries the M=17/33/49 cliffs.

---

## Step 2 — direct measurement (the scatter floor)

Rather than reload the full vLLM plugin, Step 2 slices the **committed**
`submissions/lmhead12k_empirical/kept_ids.json` (12,288 ids, *not* re-picked) into a
standalone CUDA-event microbenchmark that replicates ubel #14's `compute_logits`
exactly, at M∈{1,7,16,17,25,33,45,49}, 200 steps / 30 warmup
(`lmhead12k_verify_cost.py::step2_measured`). Three lm_head variants:

- **`full`** — `F.linear(hs, W[262144,2560])` + softcap + full-vocab argmax (control).
- **`k12_gemm_only`** — `F.linear(hs, W[12288,2560])` + softcap + argmax over 12k
  (the *pure* pruned-GEMM cost ≈ what the analytic ×0.0469 models).
- **`k12_scatter`** — the **real** ubel #14 path: 12k GEMM, then scatter into a full
  [M, 262144] −inf tensor, then **full-vocab** argmax (identity-correctness).

**Calibration:** the standalone `full` head matches the in-profiler lm_head time to
**ratio 0.997–1.001** across all M (`step2_calibration`) — the microbenchmark is
faithful to the profiler kernel.

**Measured lm_head medians (ms):**

| M | full | k12_gemm_only | k12_scatter | analytic 12k (×0.0469) |
|--:|--:|--:|--:|--:|
| 1  | 2.802 | 0.155 | 0.171 | 0.134 |
| 7  | 2.844 | 0.148 | 0.173 | 0.133 |
| 25 | 3.171 | 0.163 | 0.250 | 0.149 |
| 33 | 3.250 | 0.189 | 0.304 | 0.153 |
| 45 | 3.367 | 0.193 | 0.348 | 0.158 |
| 49 | 3.394 | 0.194 | 0.365 | 0.159 |

**Reconciliation (the >0.3 ms gap the PR asked about):** the bare pruned GEMM
(`k12_gemm_only`) lands within **~0.04 ms** of the analytic ×0.0469 — the
bandwidth model is right. But the **production** `k12_scatter` path is **~2.2×**
that (0.348 vs 0.158 @ M=45). The extra ~0.19 ms is the **retained residual floor**:
allocating + writing the full [M, 262144] −inf tensor and argmaxing over the full
vocab. ubel #14 keeps this on purpose — it is what preserves exact greedy identity
(the argmax is over the same full vocab as the unpruned head). The microbenchmark
makes the floor visible and the reduced **measured** curve uses `k12_scatter`, so the
ceiling is *not* over-claimed.

**Net measured saving @ M=45:** `full − k12_scatter = 3.367 − 0.348 = 3.019 ms` =
**94.0% of the analytic 3.213 ms.** The scatter floor costs back ~6% of the saving.

---

## Step 3 — tree TPS ceiling on the lmhead12k-reduced curve

Two reduced curves were built by subtracting the per-M head saving from the #33
tile-corrected flopideal merged curve, then re-running `tree_acceptance_model.py`
(same config as #33: top1=0.6792, top4=0.8605, rescue=0.5651, drafter=1.4 ms,
verify_base=7.0 ms, widths {1,4}, K∈[1,15]):

- **measured** (`merged_treemask_flopideal_lmhead12k_measured.json`) — subtracts
  `full − k12_scatter` (production-faithful, carries scatter floor). → run `ruch259z`.
- **analytic** (`..._analytic.json`) — subtracts `full − full·0.0469` (optimistic
  GEMM-bandwidth ceiling). → run `6c9r3lih`.
- **baseline** = full head (`merged_treemask_flopideal.json`), reproduces the #33
  headline exactly → run `tree_results_lmhead12k_baseline.json`.

### 3.1 K\*-optimum ceiling (the PR's reference frame)

The baseline column **reproduces #33's "440/481 @ p=0.78" K\*=11/M=45 headline
exactly** (440.4 / 480.8) — validates the pipeline.

| p | baseline (full head) | **lmhead12k measured** | analytic | Δ (meas − base), w/drafter |
|--:|--:|--:|--:|--:|
| 0.6792 | K11/M45: 359.9 / 393.0 | **K7/M29: 446.6 / 510.2** | K7/M29: 451.7 / 516.9 | **+86.7 (+24.1%)** |
| 0.78 | K11/M45: 440.4 / 480.8 | **K11/M45: 538.1 / 599.8** | K11/M45: 545.8 / 609.4 | **+97.7 (+22.2%)** |
| 0.85 | K15/M61: 511.6 / 558.4 | **K11/M45: 624.9 / 696.6** | K11/M45: 633.8 / 707.7 | **+113.3 (+22.1%)** |

Two structural shifts from the cheaper verify step:
- **p=0.6792:** the optimum **moves shallower** (K11→K7, M45→M29). A cheaper step
  makes shorter trees more efficient per token, so the argmax-K drops. Ceiling still
  rises (359.9→446.6) but stays **<500 with drafter**.
- **p=0.85:** the optimum **moves off the K-range cap** (K15→K11). On the full head
  the deep tree was still climbing at the K=15 boundary; the cheaper step makes the
  M=45 knee the true interior optimum.

### 3.2 Fixed-K=6 (M=25) headline — the other verdict lens

`tree_acceptance_model.py`'s `verdict` block keys off the **fixed sim_K=6** (W=4 →
M=25) headline, *not* the K\*-optimum:

| p=0.78, M=25 fixed | full head | **lmhead12k measured** |
|--:|--:|--:|
| tps_tree, w/ drafter | 420.8 | **476.5  (<500, NO)** |
| tps_tree, verify-only | 481.6 | **545.4  (>500, YES)** |

So **`verdict.exceeds_500_at_full_scale_withdrafter = false`** in the JSON (476.5),
while **`...verifyonly = true`** (545.4). This is consistent with §3.1: at the fixed
moderate tree (M=25) the with-drafter number is below 500; you need the K\*=11 (M=45)
optimum to cross 500 with the drafter included. **The report headline and
SENPAI-RESULT use the K\*-optimum lens** because that is the frame PR #33 reported
its "440/481" baseline in; both lenses are documented here so the choice is explicit.

---

## Step 4 — tile boundaries folded into the canonical curve

Done and documented in `report_msweep.md` (see the "Update — PR #37" block after the
verdict table). Summary: `scripts/profiler/fold_tile_into_msweep.py` overlays the
#33 directly-measured Marlin tile-boundary rows (M=17/33/49) onto the `graph|ctx256`
node of `results_msweep.json` **in place** and re-runs `build_cost_model`, so #26/#28
consumers that read `results_msweep.json` **without** a `--cost-model-json` override
now inherit the tile cliffs. Continuity at the 5 shared M is ≤0.054 ms (sub-thermal).

Headline effect of the fold (full head, p=0.78): the coarse curve interpolated
**linearly across each `ceil(M/16)` cliff**, undershooting M=49 (15.45 interp vs
**18.13 measured**, −14.8%) and M=33 (13.10 vs **14.99**, −12.6%). With the cliffs
priced in, the full-head optimum corrects from **K\*=12 (M=49): 452/493** →
**K\*=11 (M=45): 440/481** — K=12/M=49 is no longer undershot, so it is correctly
priced out of the optimum. The pre-fold curve is preserved at
`results_msweep_prefold.json`. **Scope:** only `graph|ctx256` is tile-corrected (the
#33 sweep measured that config); `eager|*` and `*|ctx512` keep coarse interpolation
and are flagged in `config.tile_boundary_folded`.

---

## Step 5 — reconciling the `optimal_k_*` summary scalars

The PR Step 5 asked which acceptance scenario each `optimal_k_<label>` scalar is
keyed to, and to confirm the realistic p∈{0.6792, 0.78} serving optimum is **K=11
(M=45)** so the guidance to kanna #24 is locked.

- **Where the `optimal_k_*` scalars live.** They are logged by `spec_cost_model.py`'s
  `build_cost_model` → the `realistic` field, which is a **LINEAR (W=1, M=K+1)**
  per-accept-model sweep (`flat:2.2/3.3/4.5`, `geom:0.6/0.7/0.8`), **not** the W=4
  tree. The run that carries them on the tile curve is **`36hkaj14`**
  (`denken/marlin-tile-boundary`), not the tree run `aid45far`
  (`denken/tree-model-tree-masked`, which has **no** `optimal_k_*` scalars — it logs
  `tps_tree_*` / `optimal_kstar` tree fields instead). The PR Step 5 text attributed
  them to `aid45far`; that is the correction.
- **Why they read K=15 (the range cap).** `optimal_k_geom:0.8 = 15` and
  `optimal_k_flat:4.5 = 15` on the tile curve are the **linear-chain** optima
  **floored at the K-range minimum**: the tile curve starts at M=16, so the smallest
  in-range K is M−1 = 15, and these high-acceptance linear models still want to go
  deeper than the curve's M cap — the argmax sits at the boundary. They are an
  artifact of the linear lens on a curve that starts at M=16, **not** a serving
  recommendation.
- **The serving optimum (W=4 tree, the kanna #24 frame).** Confirmed from the tree
  runs above: on the **full head**, the realistic tree optimum is **K=11 (M=45)** at
  **both** p=0.6792 (359.9/393.0) and p=0.78 (440.4/480.8). lmhead12k keeps K=11
  (M=45) at p=0.78 and only moves shallower (K=7/M=29) at p=0.6792.

**One-line Step-5 confirmation:** the realistic serving optimum on the canonical
full head is **K=11 (M=45)** at p=0.6792 and p=0.78; the `optimal_k_*=15` scalars in
run `36hkaj14` are **linear-chain (W=1) optima floored at the K-range minimum of the
M≥16 tile curve**, not the W=4 tree serving optimum — so the K=11/M=45 guidance to
kanna #24 is locked.

---

## Reproduce

```bash
cd target
export PYTHONPATH=scripts/profiler
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0

# Steps 1+2: analytic table + direct microbenchmark + build reduced curves
python scripts/profiler/lmhead12k_verify_cost.py \
  --wandb_name denken/lmhead12k-derive-measure --wandb_group spec-verify-lmhead12k

# Step 4: fold tile boundaries into the canonical msweep (in place; keeps prefold copy)
python scripts/profiler/fold_tile_into_msweep.py

# Step 3: tree ceiling on each curve (baseline / measured / analytic)
for tag in baseline measured analytic; do
  case $tag in
    baseline) C=research/spec_cost_model/merged_treemask_flopideal.json;;
    measured) C=research/spec_cost_model/merged_treemask_flopideal_lmhead12k_measured.json;;
    analytic) C=research/spec_cost_model/merged_treemask_flopideal_lmhead12k_analytic.json;;
  esac
  python scripts/profiler/tree_acceptance_model.py \
    --cost-model-json "$C" --key graph\|ctx256 \
    --out research/spec_cost_model/tree_results_lmhead12k_$tag.json \
    --wandb_name denken/tree-lmhead12k-$tag --wandb_group spec-verify-lmhead12k
done
```

## Artifacts

| file | what |
|---|---|
| `lmhead12k_verify_cost.py` | Steps 1+2 driver (analytic + microbenchmark + curve builder) |
| `fold_tile_into_msweep.py` | Step 4 in-place tile fold |
| `lmhead12k_verify_cost.json` | Step 1 table + Step 2 measurements + calibration |
| `merged_treemask_flopideal_lmhead12k_{measured,analytic}.json` | reduced tree curves |
| `merged_dense_corrected_lmhead12k_{measured,analytic}.json` | reduced dense (no tree-mask) curves |
| `tree_results_lmhead12k_{baseline,measured,analytic}.json` | Step 3 tree ceilings |
| `results_msweep.json` (folded) · `results_msweep_prefold.json` (provenance) | Step 4 canonical curve |

**W&B (project `gemma-challenge-senpai`, group `spec-verify-lmhead12k`):**
`klvpfk7g` (derive+measure), `ruch259z` (tree/measured), `6c9r3lih` (tree/analytic).
