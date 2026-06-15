<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Read-reduction PPL Pareto — can a PPL-safe body-read cut reach 500? (PR #287)

**fern · `fern/read-reduction-ppl-pareto` · W&B `17en3hus` (rev — supersedes `uc2mqt82`) · BANK-THE-ANALYSIS (adds 0 TPS, no served-file change, no launch)**

> **Revision note.** v1 reported `read_reduction_lever_clears_500 = True` using a TPS model that treated the
> deployed point as **100% read-bound** (`BW_eff = 481.53·1.76/3.844 = 220.47 GB/s`). The advisor's cross-check
> against **denken #283** (`vmxuwxm0`, now MERGED) flagged this as an over-credit: denken **measured** the int4
> body-read is only **38% of the honest 1/K_cal wall**. The PPL Pareto + per-layer ranking are **unchanged**;
> this revision re-prices the TPS attribution honestly — and the verdict flips to **does NOT clear 500**.

## The question

denken #278 proved the deployed decode reads the int4 target body from HBM each step; denken #283 (`vmxuwxm0`,
MERGED) **measured** how much of the honest 1/K_cal = **7982.9µs** step that read actually costs: **3037.2µs =
38.05%** of the wall. The other ~62% (draft + verify-compute + host) is **fixed** and a body-read-byte cut does
**not** touch it. At fixed E[T]=3.844 there are two ceiling-raising levers: **more tokens per read** (E[T]-raise —
my #281 numerator, a built drafter) and **FEWER bytes per read** (the body-read denominator). This leg probes the
denominator: **can a PPL-safe body-read-byte reduction (holding projected deployed PPL ≤ 2.42; deployed int4
anchor 2.3772, headroom 0.0428) reach the byte-reduction % that moves the DEPLOYED 481.53 → 500?**

## Method (local in-memory fake-quant — OPTIMISTIC proxy, 0 TPS)

Decompress the int4 body (`google/gemma-4-E4B-it-qat-w4a16-ct`, pack-quantized int4 group_size=32 symmetric;
342 body Linears, code+scale = 2.2192 GB), then for each config **dequant → re-round onto a lower-bit grid / 2:4
mask → measure PPL** on the official corpus with the EXACT official teacher-forced arithmetic (61797 scored
tokens, bit-faithful), and compute the **analytic** read reduction (code+scale bytes vs the int4 baseline). No
kernel build, no served write.

## TPS attribution — denken #283 MEASURED read-fraction (replaces the v1 full-read-bound model)

A body-read-byte cut of fraction `X` shrinks **only** the measured read-portion `f_read = 0.3805` of the wall;
the fixed `1 − f_read = 0.6195` (draft + verify-compute + host) does not move. So at fixed E[T] the deployed TPS
is

```
tps(X) = 481.53 / (1 − f_read · X) = 481.53 / (1 − 0.3805 · X)
```

- **required reduction for 500** = `(1 − 481.53/500) / f_read` = **9.709%**  (round-trips to 500.0 TPS, resid 0.0).
- **TPS at the PPL-safe max** (`X = 8.43%`) = `481.53/(1 − 0.3805·0.0843)` = **497.49 TPS** (short of 500).

## The Pareto frontier — **UNCHANGED** (projected deployed PPL = 2.3772 + offset-corrected delta)

| config | read-reduction % | local PPL | projected deployed PPL | Δ vs int4 | ≤ 2.42 |
|---|---:|---:|---:|---:|:--:|
| int4 baseline (anchor) | 0.00 | 2.0067 | 2.3772 | — | ✅ |
| mixed demote 1L | 0.52 | 1.9993 | 2.3698 | −0.0074 | ✅ |
| mixed demote 2L | 1.03 | 1.9968 | 2.3673 | −0.0099 | ✅ |
| mixed demote 4L | 2.20 | 2.0027 | 2.3732 | −0.0040 | ✅ |
| mixed demote 6L | 3.23 | 2.0037 | 2.3742 | −0.0030 | ✅ |
| mixed demote 8L | 4.28 | 2.0041 | 2.3746 | −0.0026 | ✅ |
| mixed demote 10L | 5.33 | 2.0063 | 2.3768 | −0.0004 | ✅ |
| mixed demote 12L | 6.36 | 2.0082 | 2.3786 | +0.0014 | ✅ |
| **mixed demote 16L** | **8.43** | 2.0271 | **2.3975** | +0.0203 | ✅ **← max safe** |
| mixed demote 21L | 11.14 | 2.0947 | 2.4651 | +0.0879 | ❌ first breach |
| mixed demote 24L | 12.68 | 2.1474 | 2.5179 | +0.1407 | ❌ |
| mixed demote 32L | 16.89 | 2.2983 | 2.6688 | +0.2916 | ❌ |
| uniform int3 | 22.22 | 4.7765 | 5.1470 | +2.7698 | ❌ |
| 2:4 sparsity | 21.91 | 6.8757 | 7.2462 | +4.8690 | ❌ |

The **sensitivity-ranked efficient frontier** (demote the LEAST-int3-sensitive layers first; ranking by
single-layer int3 PPL-delta on a 32-rec subset) is far better than the pessimistic corners. Least-sensitive
layers: **L8, L30, L29, L5, L4**; most-sensitive: **L17, L27, L24, L12, L16**. The **max PPL-safe reduction is
8.43%** (demote-16L, proj 2.3975). 2:4 sparsity IS realizable on A10G sm_86 (Ampere, Sparse-Marlin) but is
PPL-catastrophic here (+4.87); uniform int3 also blows the budget (+2.77) — only the sensitivity-ranked frontier
is safe.

## Verdict — `read_reduction_lever_clears_500 = False` (on the MEASURED read-fraction)

`max_ppl_safe_read_reduction_pct = 8.43%` **<** `required = 9.71%` → at the PPL-safe ceiling the body-read lever
delivers only **497.49 TPS**, short of 500. The bytes-per-read denominator lever is a **real but INSUFFICIENT**
ceiling-raiser: even spending the entire PPL headroom on read-reduction does not reach 500 at fixed E[T]. This is
**fully consistent** with denken #283's non-binding verdict (read is not the binding constraint — the pure-read
ceiling is 1265.6 ≫ 500), not a contradiction of it. **The path to 500 remains E[T]-raise (a built drafter, my
#281 numerator); the non-speculative denominator lever does not get there on its own.**

## Reconciliation — one named quantity, three premises

`read_reduction_pct_for_500_at_fixed_et` came out **+3.694%** (my v1), **−153%** (denken #283), and **+9.71%**
(this honest deployed model). They differ only in **what fraction of the wall is read-bound**:

| premise | f_read | question answered | result |
|---|---:|---|---:|
| v1 full-read-bound (`BW_eff=220.47`) | **1.00** | "if the deployed point were 100% read-bound, what cut → 500?" | **+3.694%** |
| denken #283 ceiling-framing | 0.38 | "what body change moves the PURE-READ ceiling (1265.6) to 500?" | **−153%** (enlarge body; non-binding) |
| **honest deployed movement (this rev)** | **0.38** | "what cut moves the DEPLOYED 481.53 → 500, read = 38% of wall?" | **+9.71%** |

My v1 `BW_eff = 220.47 GB/s` (≪ denken's measured 520.95 GB/s) is the bandwidth that makes the full 1.76 GB body
read consume the **entire** 7982.9µs step — i.e. it silently set `f_read = 1.0`, absorbing all the non-read cost
(draft + verify-compute + host = the other 62%) into a pretend-low bandwidth and crediting a read cut as scaling
the **whole** wall. That is the **same class of over-credit as denken #278's draft-side bridge**. denken #283's
**−153%** is the *pure-read-ceiling* framing: the read-bound ceiling (1265.6) is already 2.6× above 500, so to pull
it **down** to 500 you'd have to **enlarge** the body 2.53× → negative reduction → non-binding. The honest
deployed-movement question — the one that matters for this lever — gives **+9.71%**, which **exceeds** the 8.43%
PPL-safe ceiling. denken's "read isn't binding" and this rev's "PPL can't fund enough read-cut" agree: **no free
path to 500 via the denominator.**

## Honesty — why the Pareto is a candidate map, not a build (unchanged from v1)

1. **The local anchor does NOT reproduce the deployed 2.3772.** Local QAT int4 PPL = **2.0067** (resid
   **−0.3705**). The deployed PTQ weights (`osoi5-v0-baked`) are **pod-absent**, so the public **QAT-w4a16-ct** is
   the only local int4 body. This is a genuine model difference, not a harness bug — faithfulness is proven three
   ways: exact **61797** official scored tokens, int8 round-trip near-lossless (Δ 0.00133), int4-recomputed
   bounded (Δ 0.0322). The PPL verdict is built in **offset-corrected DELTA space** so the absolute QAT-vs-PTQ gap
   cancels and only the int4→int3 transcode deltas matter.
2. **Two stacked optimisms** make the Pareto a lower bound on PPL cost: (a) **fake-quant** uses fp accumulation —
   a real int3 kernel adds group/scale-quant + transcode error; (b) **QAT > deployed PTQ robustness** — the
   deployed PTQ int3 deltas would be LARGER. **The fake-quant-QAT-vs-real-kernel-PTQ PPL gap is the dominant PPL
   sensitivity.** (Note: the verdict now MISSES even on this optimistic PPL proxy, so a real-kernel penalty only
   widens the miss.)
3. **0 TPS. NO served-file change, NO re-quantization of the served checkpoint, NOT a launch, NOT open2, NO HF
   Job, NO submission. BASELINE stays 481.53.** The launch gate remains land #245's **MEASURED ≥500 at λ̂ ≥ 0.9780**.

## Self-test (PRIMARY) — `read_reduction_ppl_pareto_self_test_passes = True` (11/11)

harness scores the exact **61797** official tokens; PPL NaN-clean; int8 near-lossless (<0.01); int4-recomputed
round-trip <0.05; required-% round-trips to 500 through the **measured** movement formula (resid 0.0); uniform int3
reduction (22.22%) > required (9.71%); reduction monotone in demoted-layer count; **all imported constants EXACT**
(481.53 / 520.953 / K_cal 125.268 / step 1218.2 / E[T] 3.844 / 2933.83µs / 2.42 / 2.3772 / headroom 0.0428 / τ_lo
1.03524 / **denken #283 f_read 0.38046 / ceiling 1265.64 / −153.13%**); max-safe well-defined; **verdict_tps_consistent**
(the boolean lever verdict matches the priced deployed TPS — new attribution-bug guard); 6 caveats carried.

**Sub-check (a) deviation (unchanged, still flagged):** the anchor does NOT reproduce 2.3772 (local QAT ≠ deployed
PTQ). I substitute the three harness-faithfulness proxies and report `anchor_reproduces_2p3772 = False` as
informational, because the absolute gap is the model difference the verdict cancels in delta space. Advisor to
judge whether to source the deployed `osoi5-v0-baked` PTQ for a faithful absolute anchor.

## Hand-off (one sentence)

> The PPL-vs-read-reduction Pareto shows the largest PPL-safe body-read reduction (PPL ≤ 2.42, fake-quant
> optimistic) is **8.43%**, but on denken #283's MEASURED 38%-read-fraction wall that only reaches **497.5 TPS** —
> **below** the **9.71%** needed for 500 — so the bytes-per-read denominator lever **does NOT clear** 500 (a real
> but insufficient lever, consistent with denken #283's non-binding read ceiling), and the path to 500 remains the
> E[T]-raise numerator (a built drafter, my #281).

## Reproduce

```bash
cd target/ && CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python research/validity/read_reduction_ppl_pareto/read_reduction_ppl_pareto.py \
    --self-test --wandb_group read-reduction-ppl-pareto --wandb_name fern/read-reduction-ppl-pareto
```

GPU, peak **16.99 GB**, ≈ 6.5 min. **W&B run `17en3hus`** (rev; v1 `uc2mqt82`). Imports (EXACT): kanna #217
E[T]=3.844 / step 1218.2 / K_cal 125.268 / 481.53 / λ=1 ceiling 520.953; denken #278 read floor 2933.83µs;
**denken #283 (`vmxuwxm0`) body-read fraction 0.38046 / honest wall 7982.9µs / body read 3037.2µs / HBM ceiling
1265.64 / read_reduction_pct_for_500_at_fixed_et −153.13%**; PPL gate 2.42, anchor 2.3772 (private 2.3777),
headroom 0.0428; lawine #267 τ_lo 1.03524. Deployed config from `submissions/fa2sw_precache_kenyan/manifest.json`
(osoi5-baked PTQ pod-absent → public QAT-w4a16-ct proxy; serve venv absent → `.venv`). BASELINE 481.53 untouched.
NOT a launch. NOT open2.
