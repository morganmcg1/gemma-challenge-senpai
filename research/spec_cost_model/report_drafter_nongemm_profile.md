<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Drafter NON-GEMM profile — what is the other ~70% of the drafter forward, and is any of it a contract-safe TPS lever? (#75 caveat-3 follow-up)

**PR:** #77 · **Author:** denken · **Date:** 2026-06-14 · **Builds on:** #75 (drafter-forward
roofline, MERGED — GEMM half: 566 µs = 4.88% decode, latency-floored, int4 refuted; this
audit closes its caveat 3), #69 (wirbel decode composition: drafter is the #2 block at
15.5–18.1%), #68 (faithful isolated-kernel method) · **Relates to:** #71 (tree-verify, the
#1 lever)
**Question:** #75 timed the drafter's **GEMM** half and *inferred* (did not time) that the
other **~70% of the drafter forward is non-GEMM** (attention SDPA, the centroid sparse
sampler / 262k-vocab masked-embed gather, RMSNorms, rotary, residuals, sampling, glue).
That non-GEMM block — **~1230–1530 µs/step, the largest un-audited drafter cost** — is what
this PR TIMES and decomposes. Is there a single fat **contract-safe (unchanged-outputs)**
non-GEMM sub-block worth a build, or is the drafter already near its floor?

**LOCAL profiling + Step-0 feasibility only** — no HF Job, no submission, no serve-path
change, audit-only. Real served vLLM modules (`RMSNorm`, `get_rope`, `get_act_and_mul_fn`,
`Gemma4MTPMaskedEmbedder`) with **real drafter weights**, plus a **VERBATIM copy of the
deployed fused sparse-argmax triton kernel**. Attention = roofline (memory-bound decode) +
an SDPA proxy. All M=1, value-independent. Harness: `scripts/profiler/drafter_nongemm_profile.py`.

## Verdict / headline

> **There is NO contract-safe non-GEMM lever — the drafter is already near its floor on
> BOTH halves.** The ~70% non-GEMM block (#69 anchor: **1232–1534 µs/step = 10.6–13.2% of
> the 11.6 ms decode step**) is NOT dominated by one fat reducible kernel. It is
> "death by a thousand cuts": the two genuine **standalone** non-GEMM kernels — the centroid
> sparse sampler (**178 µs/step = 1.54% decode**, the binding sub-block) and attention
> (**61 µs/step = 0.53% decode at its memory-bound floor**) — sum to only ~240 µs (2.1%
> decode). The remaining **992–1294 µs/step (8.6–11.2% decode)** is the long tail of tiny
> elementwise/norm/rotary/residual "glue" ops **already fused into the GEMM epilogues by
> torch-compile + ONEGRAPH** plus per-kernel graph-replay dispatch — no single addressable
> hotspot. The PR's hypothesized contract-safe reduction (gather only the candidate set,
> not the full 262k vocab) is **ALREADY DEPLOYED**: the masked-embedder gathers 8192/262144
> rows (3.1% of vocab, ~31× cheaper than a full gather). Combined with #75 (GEMM half
> latency-floored, int4 refuted), the **realistically-addressable drafter TPS headroom is
> ~0%**. FAIL-FAST: do not build a drafter non-GEMM optimization. Spend the cycle on the
> #1 lever — land tree-verify (#71).**

| quantity (A10G, deployed M=1 × K=7, 11.6 ms decode step) | value |
|---|--:|
| **primary metric — `drafter_nongemm_binding_subblock_pct_of_decode`** | **1.54%** (centroid sparse sampler, the largest standalone non-GEMM kernel) |
| binding non-GEMM sub-block | **`centroid_sampler_fused`** = 178 µs/step (9.2% of drafter) |
| non-GEMM TOTAL (authoritative; #69 drafter 15.5–18.1% − #75 GEMM 566 µs) | **1232–1534 µs/step = 10.6–13.2% decode** |
| &nbsp;&nbsp;├ resolved STANDALONE kernels (sampler + attn@roofline floor) | 240 µs/step = **2.1% decode** |
| &nbsp;&nbsp;└ un-attributable long-tail (fused glue + graph-replay dispatch + python) | **992–1294 µs/step = 8.6–11.2% decode** — no single hotspot |
| attention (3 sliding KV≤512 + 1 full KV=512) — roofline floor | 61 µs/step = 0.53% decode (memory-bound) |
| attention — SDPA proxy (launch-overhead-dominated upper bound) | 912 µs/step (proxy ≠ deployed paged kernel) |
| **Step-0: masked-embed gather already candidate-restricted?** | **YES — 8192/262144 = 3.1% of vocab** |
| &nbsp;&nbsp;gather 8192 cand vs full-262k counterfactual | 17.8 µs vs 543 µs (**~31× already saved**) |
| GEMM half (#75, for the total-drafter cross-check) | 566 µs/step = 4.88% decode (latency-floored, int4 refuted) |
| **whole drafter forward (this audit CONFIRMS #69)** | 1798–2100 µs/step = **15.5–18.1% decode** |
| **realistically-addressable contract-safe drafter TPS headroom** | **~0%** (gather restricted, attn at floor, glue fused, GEMM floored) |
| sanity: isolated-unfused SUM vs #69 whole-drafter budget | 3319 µs > 2100 µs → **proves the glue FUSES in deployment** |
| peak GPU mem (profiler) | 0.39 GiB |

**One-line result:** the drafter's non-GEMM ~70% is a memory-floored attention kernel + an
already-fused-and-candidate-restricted sampler + a large irreducible long-tail of
fused glue — **no contract-safe lever**; with #75 this closes the drafter as a TPS target.

---

## Method — why the isolated sub-block table is an UPPER BOUND, and what is authoritative

The deployed drafter runs under **ONEGRAPH=1** (the whole 7-pass propose is one CUDA graph)
**plus `@support_torch_compile`** on `Gemma4MultiTokenPredictor`. Under torch-compile,
inductor **fuses** the elementwise / norm / rotary / residual "glue" ops into the backbone
GEMM epilogues (the kernel-priority log confirms `rms_norm=['native']` so inductor *can*
fuse the norms). Timing each glue op **in isolation** therefore pays a separate kernel
launch + global read/write that **does not exist in the deployed graph** — so the
isolated-unfused per-op numbers are an **upper bound**, not the deployed cost.

> **Proof the glue fuses:** the isolated-unfused SUM of all sub-blocks is **3319 µs/step**,
> which **EXCEEDS wirbel#69's entire measured drafter budget of 2100 µs/step** (#69 hi).
> That is physically impossible if the isolated per-op costs were the real deployed costs —
> so most of the glue must collapse into the GEMM epilogues in deployment.

We therefore report **three faithful layers** rather than trusting the isolated sum:

1. **AUTHORITATIVE non-GEMM total** = wirbel#69's end-to-end drafter budget (15.5–18.1% of
   decode, measured in the *real server*) minus #75's 566 µs GEMM chain = **1232–1534 µs/step**.
   This audit thus **decomposes and confirms #69**, it does not replace it.
2. **STANDALONE deployed kernels we can time faithfully** — the only non-GEMM ops that do
   *not* fuse into the GEMM epilogues because they are their own kernels:
   - **centroid sparse sampler**: the **VERBATIM deployed fused triton kernel** (its own CUDA
     graph) — a faithful deployed number, **25.5 µs/pass → 178 µs/step**.
   - **attention**: its own paged kernel; the **roofline** (KV bytes / 600 GB/s) is the
     memory-bound floor, **8.74 µs/pass → 61 µs/step**.
3. **FUSIBLE GLUE** (gather, RMSNorms, rotary, residuals, activation): reported as an
   explicit **isolated-unfused upper bound**, fused away in the deployed graph.

The **binding non-GEMM sub-block** is the largest genuine **standalone** kernel (the only
thing you could actually target with a kernel-level optimization): the centroid sampler.

## NON-GEMM sub-block table (per decode step, ×K=7)

`cls`: **[S]** standalone deployed kernel · **[u]** standalone upper bound (SDPA proxy) ·
**[U]** fusible-glue ISOLATED UPPER BOUND (fused away in the deployed onegraph).

| sub-block | cls | cnt | µs/pass | µs/step | % drafter | % decode |
|---|:--:|--:|--:|--:|--:|--:|
| rmsnorm_hidden | U | 17 | 184.78 | 1293.5 | 66.4% | 11.15% |
| attention_sdpa_proxy | u | 4 | 130.30 | 912.1 | 46.8% | 7.86% |
| rotary_sliding | U | 3 | 45.93 | 321.5 | 16.5% | 2.77% |
| qnorm_sliding | U | 3 | 37.27 | 260.9 | 13.4% | 2.25% |
| **centroid_sampler_fused** | **S** | **1** | **25.48** | **178.3** | **9.2%** | **1.54%** |
| rotary_full | U | 1 | 14.26 | 99.9 | 5.1% | 0.86% |
| qnorm_full | U | 1 | 12.82 | 89.7 | 4.6% | 0.77% |
| residual_add | U | 8 | 9.32 | 65.2 | 3.3% | 0.56% |
| **attention_sdpa_roofline** | **S** | **4** | **8.74** | **61.2** | **3.1%** | **0.53%** |
| activation_gelu | U | 4 | 5.49 | 38.4 | 2.0% | 0.33% |
| layer_scalar_mul | U | 4 | 5.45 | 38.1 | 2.0% | 0.33% |
| embed_gather | U | 1 | 1.71 | 12.0 | 0.6% | 0.10% |
| embed_concat | U | 1 | 1.27 | 8.9 | 0.5% | 0.08% |

> The big `%decode` rows (rmsnorm 11.15%, SDPA-proxy 7.86%) are **[U]/[u] upper bounds that
> do not survive deployment**: the norms fuse into GEMM epilogues, and the SDPA proxy is
> launch-overhead-dominated (130 µs/pass) versus the memory-bound paged kernel it stands in
> for (roofline floor 8.74 µs/pass). Reading those rows as deployed cost is exactly the
> over-count #69 disproves. The deployed-faithful rows are the two **[S]** standalone kernels.

## Faithful non-GEMM accounting

```
AUTHORITATIVE non-GEMM total (#69 drafter 15.5–18.1% − #75 GEMM 566µs):
    1232 – 1534 µs/step   (10.6 – 13.2% decode)
  ├ resolved STANDALONE kernels:
  │     centroid_sampler_fused   178 µs  (1.54% decode)   [verbatim deployed triton kernel]
  │     attention @ roofline floor 61 µs  (0.53% decode)   [memory-bound; deployed paged kernel ≥ this]
  │     = 240 µs/step  (2.1% decode)
  └ UN-ATTRIBUTABLE long-tail:
        992 – 1294 µs/step  (8.6 – 11.2% decode)
        = fused glue (norms/rotary/residual/act, collapsed into GEMM epilogues)
        + per-kernel graph-replay dispatch + python propose-loop glue
        → NO single reducible hotspot
```

The long tail dominates the non-GEMM block, but **by construction it is not a single
sub-block** — it is the aggregate of dozens of tiny ops the deployed graph has already
fused, plus the irreducible cost of replaying a multi-kernel CUDA graph. There is nothing
fat to cut. (Even if attention's deployed paged kernel runs at, say, 2–4× its roofline
floor, that 120–240 µs is *still* not contract-safely reducible — it is the attention
kernel doing its job; and it merely moves cost from the long-tail bucket into a kernel that
is itself at/near its memory floor.)

## Step-0 — is the PR's hypothesized contract-safe reduction available? **Already deployed.**

The PR asked specifically: *does the 262k-vocab masked-embed gather touch the full vocab
when only the candidate set is needed downstream?* Answer, from the served kernel
(`gemma4_mtp.py` `Gemma4MTPMaskedEmbedder._select_and_score` →
`embeddings = lm_head_weight[selected.reshape(-1)]`) **and** measurement:

- The masked-embedder gathers only `num_selected = top_k × (vocab / num_centroids) =
  64 × 128 = **8192** rows of `lm_head[262144, 256]`, i.e. **3.1% of the vocab**, never the
  full 262144.
- Measured: gather 8192 candidate rows = **17.8 µs** vs a full-262k gather counterfactual =
  **543 µs** → the deployed sparse path is **~31× cheaper** than the non-masked design the
  PR worried about.
- Internally, the sampler's dominant cost is the **top-k over the 2048 centroids (19.0 µs)**,
  not the gather (17.8 µs unfused → fused into the 25.5 µs whole-sampler kernel). There is no
  full-vocab materialization anywhere on the path.

**So the hypothesized "gather only the candidate set" reduction is ALREADY in production.**
The contract-safe non-GEMM levers are exhausted: gather restricted, attention at its memory
floor, glue already fused by torch-compile. `contract_safe_nongemm_lever_exists = False`.

## Total-drafter headroom cross-check (GEMM #75 + non-GEMM this audit) vs the #1 lever (#71)

| drafter half | µs/step | % decode | contract-safe addressable? |
|---|--:|--:|---|
| GEMM chain (#75) | 566 | 4.88% | **~0%** — memory-bound *regime* but only 47% HBM-saturated (latency-floored); int4 **refuted** (≈+1.5–3.6% TPS, overstated ~3–5×) |
| non-GEMM (this audit) | 1232–1534 | 10.6–13.2% | **~0%** — sampler fused+candidate-restricted; attention at memory floor; glue already fused; long-tail is graph-dispatch |
| **whole drafter forward** | **1798–2100** | **15.5–18.1%** | **~0% realistically addressable** |

Even the *hard ceiling* of making the entire drafter forward **free** is bounded by its
15.5–18.1% decode share — but that is unreachable (you need the drafter to propose K=7), and
the **contract-safe** slice of it is ~nil on both halves. The drafter is a **near-floor,
low-value** TPS target.

By contrast, the **#1 decode block is the 53% verify-GEMM**, and the live lever there is
**tree-verify (#71)**, which lifts accepted-tokens-per-step / verify efficiency — a different
axis that the drafter audits do not touch. **Recommendation: stop optimizing the drafter
forward; land #71.**

## FAIL-FAST conclusion

Per the PR's fail-fast instruction: the non-GEMM block **is irreducible by any contract-safe
lever**. No single sub-block is both fat and reducible — the one nameable standalone kernel
(centroid sampler, 1.54% decode) is already fused and candidate-restricted, attention is at
its memory-bound floor, and the bulk (8.6–11.2% decode) is fused-glue + graph-replay dispatch
with no hotspot. **Do not build a drafter non-GEMM optimization.** This closes the drafter
forward (GEMM #75 + non-GEMM #77) as a TPS target and redirects the cycle to tree-verify (#71).

## Caveats

1. **Attention deployed cost lives between the roofline floor (61 µs/step) and the SDPA
   proxy (912 µs/step).** The roofline is byte-exact for a perfectly memory-bound decode
   kernel; the deployed paged `TRITON_ATTN` will run somewhat above it (kernel overhead) but
   the SDPA proxy is launch-overhead-dominated (130 µs/pass at M=1) and is *not* the deployed
   kernel — it brackets the upper bound only. Either way attention is small and at/near its
   memory floor; nothing in the conclusion depends on the exact point in [61, ~250] µs.
2. **The 992–1294 µs long-tail is bounded by subtraction (#69 anchor − resolved standalone),
   not timed op-by-op.** That is deliberate: the deployed onegraph fuses those ops, so there
   is no faithful isolated number to time. Its *composition* (fused glue + graph dispatch +
   python) is inferred from the fusion proof, but its *magnitude* is anchored to #69's real
   server measurement, and its key property — **no single reducible hotspot** — holds
   regardless of the internal split.
3. **#69's 15.5–18.1% drafter budget is the load-bearing external anchor.** If a future
   re-measure of #69 moves that band, the non-GEMM total moves with it; the standalone-kernel
   numbers (sampler, attention) and the Step-0 gather facts are independent of #69 and do not.
4. Profiler is `submissions/`-faithful but **not** a serve-path change and runs **no** HF Job
   — audit-only, single assigned A10G, peak 0.39 GiB.

## Reproduce

```bash
# full profile + JSON + W&B (server venv has vLLM)
/tmp/server-venv/bin/python scripts/profiler/drafter_nongemm_profile.py \
  --iters 200 --warmup 50 --l-sweep 128,256,512,1024,2048 --l-headline 512 \
  --wandb_group drafter-nongemm-profile --wandb_name denken/drafter-nongemm-profile \
  --output research/spec_cost_model/drafter_nongemm_profile.json
# (W&B push can also be replayed from JSON: --log-only <json>)
```

- **JSON:** `research/spec_cost_model/drafter_nongemm_profile.json`
- **W&B run:** `denken/drafter-nongemm-profile` (id `q9p4vetv`, project
  `wandb-applied-ai-team/gemma-challenge-senpai`, group `drafter-nongemm-profile`)
- **Primary metric:** `drafter_nongemm_binding_subblock_pct_of_decode = 1.54%`
