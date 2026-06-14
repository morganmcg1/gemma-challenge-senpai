<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Tree-verify NON-GEMM overhead audit — does the M=32 tree's systems machinery eat the +21.8% verify-GEMM gain? (#79 cost-model follow-up)

**PR:** #85 · **Author:** denken · **Date:** 2026-06-14 · **Builds on:** #79 (wirbel
2-sided drafter-ceiling: priced the **+21.8% GROSS** gain = acceptance × #68 verify-GEMM
savings at M=32, but treats the tree's non-GEMM systems overhead as *free*), #77 (drafter
non-GEMM faithful CUDA-graph method), #68 (verify-GEMM roofline), #43 (split-KV
FlashDecoding, K\*=11 linear) · **Hands off to:** #71 (land the M=32 tree — this PR is the
**performance-oracle half** of its debug gate, paired with wirbel #83's salvage oracle)
**Question:** wirbel #79 prices a **+21.8%** tree gain from the GEMM side only (more accepted
tokens per verify pass × the #68 verify-GEMM saving). The deployed path is a **linear M=8
chain**; land #71 builds an **M=32 tree**. A tree is not free on the *systems* side: it adds
an ancestor attention-mask, candidate scatter/gather, an M-row sampler, M-row greedy verify
argmax, and accepted-prefix scheduling — all **non-GEMM** glue #79's cost model omits. **Is
that overhead small enough that the +21.8% gross survives, or does the tree's machinery eat
the gain?** And does attention stay at its floor when the verify batch grows 8 → 32 rows?

## Verdict / headline

> **The tree's non-GEMM machinery is ~8× smaller than the GEMM gain it unlocks — the +21.8%
> gross SURVIVES as ~+19.8% net.** The full non-GEMM tree footprint at M=32 (static tree, the
> shape land #71 builds) is **301 µs/step = 2.597% of the 11.6 ms decode step** — and the
> slice that is genuinely NEW versus the M=8 linear chain it replaces is only **192 µs =
> +1.65pp**. Eroding the +21.8% gross by that delta leaves **net +19.82% (static) / +19.74%
> (dynamic)**. There is **no O(M²) blow-up**: the only [M,M]-shaped tensor (the ancestor
> mask) is **0.32% decode** and, for a static tree, is precomputed → **0/step**. The two ops
> that grow with M are both ~**O(M)** linear: the drafter's M-row sparse sampler (2.09%,
> drafter-side) and the full-vocab greedy verify argmax (0.43%). **Attention does NOT scale
> with the verify batch:** #43's split-KV routes all M ≤ 64 verify rows to 3D FlashDecoding,
> the shared-prefix KV is read once, so **M=32 attention = 1.06× M=8** (not 4×). The
> verify-side-only machinery (excluding the drafter sampler, which is arguably already in the
> drafter's budget) is just **59 µs = 0.512% decode**. **GO: the tree net-wins on every base;
> land #71.** This PR ships #71's per-op cost-budget oracle (expected µs/step + 1.5× ceiling)
> as the performance half of its debug gate.

| quantity (A10G, M=32 DP-tree vs M=8 linear, 11.6 ms decode step) | value |
|---|--:|
| **primary metric — `tree_overhead_nongemm_pct_decode` (M=32, static)** | **2.597%** (301 µs/step) — vs **+21.8%** GEMM savings → **~8× smaller** |
| &nbsp;&nbsp;└ NEW vs M=8 linear (the slice that erodes the gross gain) | **192 µs = +1.65pp** decode |
| &nbsp;&nbsp;└ verify-side ONLY (excl. drafter M-row sampler) | **59 µs = 0.512%** decode |
| dynamic-tree variant (mask rebuilt per step, graph-captured) | 338 µs/step = **2.917%** decode |
| **test metric — `net_tree_gain_after_overhead_pct` (static)** | **+19.82%** (gross 21.8% − 1.98pp erosion) |
| &nbsp;&nbsp;└ dynamic-tree net | **+19.74%** |
| **attention amortization — M=32 / M=8 (split-KV 3D FlashDecoding)** | **1.06×** (CONFIRMED ≪ 4×; KV read shared) |
| only super-linear-shaped op (ancestor [M,M] mask), static-tree cost | 0.32% decode → **0/step precomputed** |
| largest M-growing op — drafter `centroid_sampler_Mrows` (O(M), drafter-side) | 242 µs = 2.09% decode |
| largest verify-side M-growing op — `verify_argmax_Mrows` (O(M)) | 50 µs = 0.43% decode |
| net-gain on 3 bases (rel% / wall_tps×454 / official×481.53) | **+19.82% / 543.98 / 576.96** |
| peak GPU mem (profiler) | 0.38 GiB |

**One-line result:** the M=32 tree's non-GEMM systems overhead is **2.6% decode** (1.65pp new
vs M=8 linear), attention amortizes to **1.06×**, and the +21.8% gross verify-GEMM gain
survives as **+19.8% net** — the tree net-wins; land #71.

---

## Method — launch-free per-op timing, and why static-tree is the operative number

The deployed serve path runs the whole verify step under **ONEGRAPH + torch-compile** (one
CUDA graph). Timing each glue op in **eager** mode pays a separate kernel launch that does
not exist in the deployed graph — the #77 lesson — so eager numbers are a launch-overhead
**upper bound**. We therefore time every op the way #68/#77 do: **reps-in-one-CUDA-graph**
(`time_op_graph`), which is the deployed onegraph basis, and report the eager number only as
a contrast. (The gap is large and exactly the expected direction: e.g. the ancestor mask is
**37 µs graph vs 327 µs eager** at M=32 — 8.8× launch-overhead inflation.)

Every op is the **real deployed object**, not a stand-in:
- **centroid sparse sampler** — the VERBATIM deployed fused triton kernel from #77
  (`_get_fused_sparse_argmax_kernels`), run over M candidate rows.
- **greedy verify argmax** — `argmax` over the full **[M, 262144]** vocab, the deployed DIXIE
  SMP-02 greedy fast-path (`serve.py` scatters pruned logits into the full-vocab buffer, so
  verify argmax is genuinely over 262144).
- **accepted-prefix kernel** — the real vLLM `rejection_greedy_sample_kernel` at `grid=(1,)`
  (conc=1), walking the M−1 draft tokens for the longest accepted prefix (all-accept worst
  case = full walk).
- **ancestor mask / scatter / gather / sampling-meta / seq-lens** — the tensor ops a tree
  verify step issues, at served target shapes (hidden 2560, 42 layers, vocab 262144).

**Static vs dynamic tree — the operative distinction.** A **static** tree bakes its topology
into the CUDA graph: the ancestor mask is computed **once** at graph-capture and replayed →
**0/step**. A **dynamic** tree rebuilds the mask every step. **Land #71 builds a static
tree**, so the **primary** number *excludes* the mask (precomputed). We report the dynamic
variant too (mask rebuilt per step, now graph-captured at 37 µs) as the pessimistic bound —
both net-win.

> **Drafter-side vs verify-side.** The biggest M-growing op, `centroid_sampler_Mrows`
> (2.09% decode), is the **drafter** emitting M tree-candidate rows instead of 1 — arguably
> already inside the drafter's existing per-step budget (#43: 1.446 ms flat). We report it in
> the inclusive total (2.597%) to be conservative, but the cleanest "pure tree-verify
> machinery" footprint — mask + scatter + gather + verify-argmax + accepted-prefix + seq-lens,
> **excluding** the drafter sampler — is just **0.512% decode**.

## Non-GEMM tree-overhead per-op table (launch-free µs/step, deployed onegraph basis)

`x` = M=8→M=32 growth ratio · `exp` = fitted scaling exponent (0 ≈ flat, 1 ≈ linear O(M),
2 ≈ quadratic O(M²)).

| op | side | µs@8 | µs@16 | µs@32 | %dec@8 | %dec@16 | %dec@32 | x | exp |
|---|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `tree_mask_construct` ([M,M] ancestor) | V | 29.68 | 33.47 | 37.07 | 0.256 | 0.289 | **0.320** | 1.25 | **0.16** |
| `centroid_sampler_Mrows` (drafter M rows) | **D** | 85.18 | 116.61 | 241.97 | 0.734 | 1.005 | **2.086** | 2.84 | **0.75** |
| `verify_argmax_Mrows` ([M,262144]) | V | 13.60 | 26.28 | 50.33 | 0.117 | 0.227 | **0.434** | 3.70 | **0.94** |
| `gather_accepted_spine` ([M,2560]) | V | 5.23 | 9.27 | 1.42 | 0.045 | 0.080 | 0.012 | 0.27 | −0.94 |
| `accepted_prefix_kernel` (rej-greedy) | V | 1.88 | 2.57 | 4.03 | 0.016 | 0.022 | 0.035 | 2.15 | 0.55 |
| `scatter_tree_tokens` (index_copy) | V | 1.37 | 1.37 | 1.36 | 0.012 | 0.012 | 0.012 | 0.99 | −0.01 |
| `sampling_meta_expand` ([1]→[M]) | V | 1.14 | 1.12 | 1.12 | 0.010 | 0.010 | 0.010 | 0.99 | −0.01 |
| `seq_lens_handoff` ([1] scalar) | V | 1.07 | 1.07 | 1.08 | 0.009 | 0.009 | 0.009 | 1.01 | 0.00 |

**Reading the table:**
- **No op scales O(M²).** The only [M,M]-shaped tensor — the ancestor mask — has exp **0.16**
  (nearly flat: it is dominated by fixed launch + an O(depth) parent-walk, not the M² cells),
  and at M=32 it is **0.32% decode** even rebuilt every step. The [M,M] among-token cost the
  cost model worried about is a non-event.
- The two genuinely M-growing ops are both **≈O(M) linear** (exp 0.75 / 0.94): the drafter's
  M-row sparse sampler and the full-vocab greedy verify argmax. Linear growth in M is exactly
  what a tree should cost — M candidates verified per step.
- Everything else (scatter, sampling-meta, seq-lens) is **flat and < 0.013% each** — fixed
  tiny launches independent of M. (`gather_accepted_spine`'s non-monotone 5.2→9.3→1.4 µs is
  sub-10-µs measurement jitter on a 0.01%-decode op; immaterial either direction.)

## Faithful overhead accounting (static tree = land #71's shape)

```
TOTAL non-GEMM tree overhead @ M=32 (static, mask precomputed → excluded):
    301 µs/step  =  2.597% decode            ← PRIMARY metric
  ├ NEW vs M=8 linear chain (Δ that erodes #79's gross gain):
  │     192 µs   =  +1.65pp decode           ← feeds net-gain
  ├ drafter-side (centroid_sampler_Mrows, M rows; arguably already in drafter budget):
  │     242 µs   =  2.086% decode
  └ verify-side ONLY (mask+scatter+gather+verify-argmax+accepted-prefix+seq-lens):
        59 µs    =  0.512% decode            ← cleanest "pure tree machinery" footprint

  dynamic-tree variant (mask rebuilt/step, graph-captured 37 µs):
    338 µs/step  =  2.917% decode
```

Why the net-gain uses the **Δ vs M=8 linear**, not the absolute total: wirbel #79's **+21.8%**
is itself measured *relative to the deployed M=8 linear chain*, which already pays the M=8
versions of these ops (scatter, gather, an 8-row sampler, an 8-row verify argmax). The only
overhead the tree *adds* on top of that baseline is the **Δ = 192 µs (+1.65pp)** — so that,
not the 301 µs absolute, is the apples-to-apples erosion of the gross gain.

| overhead, by tree size | static µs/step | static %dec | Δ vs M=8 | verify-side µs (%dec) |
|---|--:|--:|--:|--:|
| M=8 (linear baseline) | 109.5 | 0.944% | — | 24.3 (0.209%) |
| M=16 (DP-tree) | 158.3 | 1.365% | +48.8 µs / +0.42pp | 41.7 (0.359%) |
| **M=32 (DP-tree)** | **301.3** | **2.597%** | **+191.9 µs / +1.65pp** | **59.3 (0.512%)** |

## Attention amortization — does the verify batch (8→32 rows) erode the KV floor? **No.**

The cost model's load-bearing assumption is that attention does **not** pay 4× when the
verify batch grows 8 → 32. #43's **split-KV FlashDecoding** partitions the KV *reduction*
axis (orthogonal to query rows): the shared-prefix KV is streamed **once** and reused across
all M tree-query rows; only the tiny [M,M] among-token block grows. Confirmed two ways:

1. **By code** — `splitkv_verify_patch.would_redirect` gates `1 < M ≤ 64` verify batches to
   the 3D split-KV path. **All of M = 1/8/16/32 route 3D** (`routes_3d_splitkv = True`).
2. **By SDPA proxy** at the served target attention shape (8 heads / 2 KV / head-dim 256,
   GQA), swept over M query rows at a fixed KV length:

| M (query rows) | proxy µs/step | roofline µs/step (KV bytes / 600 GB/s) | 3D split-KV |
|--:|--:|--:|:--:|
| 1 | 536.0 | 36.8 | ✓ |
| 8 | 554.3 | 37.8 | ✓ |
| 16 | 562.7 | 39.0 | ✓ |
| 32 | 586.7 | 41.3 | ✓ |

**M=32 / M=8 = 1.06×** (proxy) — far from the naïve 4×. The roofline floor moves only
37.8 → 41.3 µs (+9%, the [M,M] among-token term); KV bytes/step barely change (22.7 → 24.8
MB) because the shared prefix dominates and is read once. **Attention stays at its floor at
M=32. `attention_amortizes_M32 = True`.**

## Net-tree-gain after overhead — on 3 bases

Eroding wirbel #79's **+21.8% gross** by the measured non-GEMM overhead Δ
(`net = (1+gross)/(1+ov) − 1`, with `ov` = Δ-vs-M8 overhead fraction):

| base | baseline | gross (#79, +21.8%) | **net (static)** | net (dynamic) | erosion |
|---|--:|--:|--:|--:|--:|
| relative % | 0.0 | +21.80% | **+19.82%** | +19.74% | −1.98 / −2.06 pp |
| local wall_tps (×454) | 454.0 | 552.97 | **543.98** | 543.64 | −8.99 / −9.34 |
| official tps (×481.53 proj) | 481.53 | 586.50 | **576.96** | 576.60 | −9.54 / −9.90 |

The overhead erosion is **< 2.1pp on every base**, against a **21.8pp** gross gain — the tree
net-wins by an order of magnitude over its own systems cost. (`local wall_tps ×454` and
`official ×481.53` are projections off the deployed frontier — wall_tps ≈454, official
481.53, PPL 2.3767, 128/128 — *not* measured submission numbers; this PR runs no HF Job.)

## Cost-budget oracle for land #71 (the performance half of its debug gate)

Per-op **expected µs/step at M=32** + a **1.5× ceiling**. land #71 asserts each op against
its budget on a debug step; a measured op **over budget** ⇒ a systems regression (an
un-fused glue op, a mis-routed attention batch, an accidental full-vocab gather) — fail the
gate before it silently eats the +19.8%. Pairs with wirbel #83's **salvage/divergence
oracle** (≈0.41) as the correctness half.

| op | side | expected µs@M32 | **budget (1.5×)** | %dec@M32 |
|---|:--:|--:|--:|--:|
| `tree_mask_construct` | V | 37.07 | 55.60 | 0.320% (static: precomputed → 0) |
| `centroid_sampler_Mrows` | D | 241.97 | 362.96 | 2.086% |
| `verify_argmax_Mrows` | V | 50.33 | 75.50 | 0.434% |
| `accepted_prefix_kernel` | V | 4.03 | 6.04 | 0.035% |
| `gather_accepted_spine` | V | 1.42 | 2.13 | 0.012% |
| `scatter_tree_tokens` | V | 1.36 | 2.03 | 0.012% |
| `sampling_meta_expand` | V | 1.12 | 1.68 | 0.010% |
| `seq_lens_handoff` | V | 1.08 | 1.61 | 0.009% |
| **TOTAL (dynamic, incl. mask)** | | **338.4** | **507.6** | 2.917% |
| **STATIC (excl. mask)** | | **301.3** | **452.0** | **2.597%** |

> Suggested gate wiring for #71: assert `measured_static_total ≤ 452 µs` (and per-op ≤
> column) on a debug verify step. The drafter `centroid_sampler_Mrows` row dominates the
> budget but is drafter-side — if #71 reuses the drafter's existing M-row sampler output, it
> can assert only the **verify-side budget ≤ 89 µs** (59.3 µs × 1.5).

## GO conclusion

The M=32 tree's non-GEMM systems machinery is **2.597% decode** (1.65pp new vs the M=8 linear
it replaces), with **no O(M²) term**, and attention **amortizes to 1.06×** rather than 4×.
wirbel #79's **+21.8% gross** verify-GEMM gain therefore survives as **net +19.82% (static) /
+19.74% (dynamic)** — the tree net-wins by ~10× over its own overhead on every base.
**Recommendation: land #71.** This PR hands #71 the per-op cost-budget oracle above as the
performance half of its debug gate.

## Caveats

1. **Drafter-sampler attribution.** `centroid_sampler_Mrows` (2.09% decode) is the largest
   single line and is **drafter-side** — the drafter emitting M rows instead of 1. We bill it
   into the inclusive 2.597% to be conservative, but if #71 consumes the drafter's existing
   M-row output it is *already paid* and the relevant tree footprint is the **0.512%
   verify-side** number. The conclusion (tree net-wins) holds under either attribution.
2. **SDPA attention proxy is an upper bound, not the deployed paged kernel.** Its absolute
   µs is launch-overhead-dominated; only its **ratio across M** (1.06×) is load-bearing, and
   the ratio is corroborated by the roofline (KV bytes barely grow) and by `would_redirect`
   routing all M ≤ 64 to 3D split-KV. Nothing depends on the proxy's absolute value.
3. **+21.8% gross is wirbel #79's number, imported as-is.** This audit prices only the
   *non-GEMM* erosion of that gain; it does not re-derive the GEMM-side acceptance×savings. If
   #79's gross moves, the net moves with it (net ≈ gross − ~2pp).
4. **Static-tree is assumed for the primary.** It is what #71 plans to build (topology baked
   into the CUDA graph). The dynamic variant (2.917%) is reported as the pessimistic bound and
   also net-wins; the mask is now graph-captured (37 µs) so even the dynamic number is tight.
5. Profiler is `submissions/`-faithful (real deployed sampler kernel, real vLLM rejection
   kernel, real split-KV router) but is **not** a serve-path change and runs **no** HF Job —
   audit-only, single assigned A10G, peak 0.38 GiB.

## Reproduce

```bash
# full per-op profile + attention sweep + net-gain + cost-budget oracle + JSON + W&B
# (server venv has vLLM; wandb installed into it for this audit)
/tmp/server-venv/bin/python scripts/profiler/tree_nongemm_overhead.py \
  --iters 200 --warmup 50 \
  --wandb_group tree-overhead-audit --wandb_name denken/tree-nongemm-overhead
```

- **JSON:** `research/spec_cost_model/tree_nongemm_overhead.json`
- **W&B run:** `denken/tree-nongemm-overhead` (id `f0c8mb39`, project
  `wandb-applied-ai-team/gemma-challenge-senpai`, group `tree-overhead-audit`)
- **Primary metric:** `tree_overhead_nongemm_pct_decode = 2.597%` (M=32, static)
- **Test metric:** `net_tree_gain_after_overhead_pct = 19.82%` (static)
