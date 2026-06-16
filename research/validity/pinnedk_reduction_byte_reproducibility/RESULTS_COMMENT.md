STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["9s76gsks"],"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"baseline_splitk_divergences_vs_serial":768,"fp32_accum_byte_identical":false,"fixed_tree_byte_identical":false,"sorted_partial_byte_identical":false,"kahan_byte_identical":false,"best_reproducible_variant":"none","reproducible_variant_preserves_speedup":false,"pinnedk_can_be_unconditional":false,"q1_collapsible":false,"max_gap_nats":0.125,"ppl":2.3772,"self_test_passes":true,"primary_metric":{"name":"q1_collapsible","value":0.0},"test_metric":{"name":"self_test_passes","value":1.0}}

## Results

**VERDICT: the divergence is IRREDUCIBLE. `pinnedk_can_be_unconditional = false`, `q1_collapsible = false`.** No reduction variant is both byte-identical to canonical serial AND speedup-preserving. The hoped-for simplification does **not** happen — **Q1 ("which reference defines equivalence") stays a genuine human contract call**, it does not dissolve. The 496.74 rung remains `self_referential_only`.

The good news the human can still bank: the obstruction is **fundamental, not a missing-variant gap**, so no build effort should be spent hunting a byte-reproducible split-K — the math forbids it. And the downside stays exactly as bounded as #431 said: every divergence is a single sub-`e*`=0.125 PPL-neutral 1-bf16-ULP near-tie, never a confident flip.

Analysis-only card: `analysis_only=true`, `no_hf_job=true`, `no_served_file_change=true`, `official_tps=0`. No model load, no kernel build, no submission. The reduction MATH is run as a synthetic-tensor probe at the served gemma-4-E4B-it attention geometry (nq=8 / nkv=2 / hd=256, GQA group 4); stark owns the kernel-realizability leg.

### The premise that reframes instruction-2(a): the served kernel ALREADY uses fp32 partials

Before testing variants I read the served kernel source (read-only, no import):

```
served Triton 3D split-KV reduce_segments (vllm.v1.attention.ops.triton_unified_attention):
  partials fp32 = True   tree-sum combine = True   local-max rescale = True   segments = 16 = True
  => the split-K-vs-serial byte break is reduction ORDER, not precision
```

- The backend allocates `softmax_segm_output` as **`torch.float32`** (`vllm/v1/attention/backends/triton_attn.py:192`); `reduce_segments` does `overall_max = tl.max(segm_max)`, rescales by `tl.exp(segm_max - overall_max)`, and combines with an fp32 `tl.sum(segm_output, axis=0)` tree (`triton_unified_attention.py:646-732`); `NUM_PAR_SOFTMAX_SEGMENTS = 16` (`triton_attn.py:55`).
- **Therefore PR variant (a) "fp32 partial accumulation" *is the deployed kernel.*** It is not a candidate fix — it is the thing that already diverges. This directly re-confirms my merged #423 finding `tax_decomp_fp32_accum_tps = 0` ("the byte break is reduction ORDER, not precision — flipping to fp32 changes nothing because it's already fp32").

### (1)+(2) Measured variant byte-identity vs canonical serial — 768 trials at the served geometry

256 seeds × KV-lens {128,256,512}, bf16 inputs, bf16 output compared byte-for-byte to the canonical `num_splits=1` serial fold (single continuous fp32 fold, global max, round once). Attn-out scale ~0.029.

| variant | byte-divergent | max\|Δ\| | elem-mismatch frac | `*_byte_identical` | preserves speedup? |
|---|---|---|---|---|---|
| `splitk_bf16_partial` (#431 baseline) | **768/768** | 9.766e-04 | 4.20e-01 | False | yes |
| `fp32_accum` **(= SERVED kernel)** | **212/768** | 4.883e-04 | 1.53e-04 | **False** | yes |
| `fp32_globalmax` | 69/768 | 4.883e-04 | 4.64e-05 | False | yes |
| `fixed_tree` (deterministic pairwise) | **69/768** | 4.883e-04 | 4.64e-05 | **False** | yes |
| `sorted_partial` (ascending‑\|x\|) | **65/768** | 4.883e-04 | 4.39e-05 | **False** | yes |
| `kahan` (Neumaier compensated) | **58/768** | **2.441e-04** | 3.75e-05 | **False** | yes |
| `serialized_fold` (replays serial order) | **0/768** | 0.0 | 0.0 | **True** | **NO** |
| `exact_fp64_s8` (order-invariant) | 165/768 | 2.441e-04 | 1.11e-04 | False | yes |
| `exact_fp64_s16` (order-invariant) | 165/768 | 2.441e-04 | 1.11e-04 | False | yes |

- **#431 baseline reproduced:** `splitk_bf16_partial` diverges 768/768 at max\|Δ\|=**9.766e-04** — the exact #431 number (`baseline_splitk_divergences_vs_serial = 768`).
- **Every speedup-preserving variant diverges.** fp32-accum (the served kernel), fixed pairwise tree, sorted-magnitude, and Kahan-Neumaier all regroup a non-associative fold → all break bytes vs serial.
- **Only `serialized_fold` is byte-identical (0/768)** — and it reaches 0 *only* by replaying serial's continuous fold, which is **sequential across segments** → it forfeits the split-K parallel speedup. It is the control, not a usable kernel.
- **Kahan is the tightest residual (max\|Δ\|=2.4e-4, half the others) yet still diverges 58 times.** This is the Higham §4.3 signature: compensation drives toward the *correctly-rounded TRUE sum*, which is **not** serial's specific (erroneous) fold — so a better sum gives *more* bytes that differ from serial, not fewer.

### (2b) Mechanism: the divergence is a precision MARGIN effect, not structural identity

Secondary `--stress-kv` arm — per-element divergence rate vs KV length (32 seeds/KV):

| KV | fp32_accum | sorted | kahan | serialized_fold |
|---|---|---|---|---|
| 1024 | 1.07e-04 | 6.10e-05 | 4.58e-05 | **0** |
| 2048 | 1.68e-04 | 6.10e-05 | 3.05e-05 | **0** |
| 4096 | 3.05e-04 | 1.98e-04 | 2.29e-04 | **0** |
| 8192 | 2.29e-04 | 1.83e-04 | 1.83e-04 | **0** |

The speedup-preserving variants' rate **grows end-to-end with KV** (deeper fp32 fold → larger reduction-order residual → more bf16-boundary crossings), while `serialized_fold` is **byte-identical at every KV**. So the rare 0-divergence sub-samples you can find at short KV are a *precision-margin coincidence* (the fp32 residual happening to round to the same bf16), **not** structural byte-identity — push the KV up and they diverge. (An 8-seed/24-trial smoke run did show sorted/kahan at 0/24; that is ~11% sampling luck at the 8.5% rate, not equivalence — exactly why the verdict is structural, corroborated by 768 trials + this curve, not "0 observed in a small sample.")

### (3) The dichotomy resolves to IRREDUCIBLE — the two sets are DISJOINT

- **Byte-identical-to-serial set** = `{serialized_fold}` only. Matching serial's bytes requires reproducing serial's exact fp32 fold *order*, and that order carries a **sequential cross-segment dependency** (tile *t*'s running sum depends on tiles 0..t−1). Any split-K scheme computes per-segment partials **independently** (that is the speedup) and so breaks the carry → cannot reproduce serial's bits for all inputs.
- **Speedup-preserving set** = `{fp32_accum, fixed_tree, sorted, kahan, exact}` — all measured non-identical.
- **Intersection = ∅** → `best_reproducible_variant = "none"`, `reproducible_variant_preserves_speedup = false` → `pinnedk_can_be_unconditional = false`, `q1_collapsible = false`.

**The order-invariant escape hatch does not dissolve Q1 — it IS Q1.** The exact fp64 accumulator is genuinely **split-count-invariant** — measured byte-identical at 8 vs 16 splits (`exact_order_invariant_8_eq_16 = True`) — so one *could* pin a deterministic reduction. But it is byte-identical to the **correctly-rounded TRUE sum**, which diverges from the deployed serial bytes 165/768 (21.5%). Adopting it = **choosing a new reference**, which is precisely the human contract decision Q1, not its elimination (Thinking Machines Lab make the same point: FlashInfer's fixed-split determinism is a *fixed-tree* reference, explicitly "not `num_splits=1`").

### Reassurance to attach to the #407 packet (bounded downside, unchanged from #431)

- Every divergence is a single bounded **1-bf16-ULP near-tie at exactly `e*` = 0.125 nat** (`max_gap_nats = 0.125`) — never a confident semantic flip.
- The served fp32-partial kernel's per-element flip **density** is **~3 orders of magnitude below** a bf16-partial kernel — measured ~2700× (bf16-partial flips ~42% of output elements; the served fp32-partial only ~0.015%). Each individual flip is still ≤ 1 ULP.
- **PPL is unchanged: 2.3772 ≤ 2.42.** Teacher-forced PPL is the aggregate cross-entropy on the gold continuation and is **reduction-order-invariant** — a reduction-variant change is PPL-neutral by construction.

### Required terminal fields

| field | value | basis |
|---|---|---|
| `baseline_splitk_divergences_vs_serial` | **768** | `splitk_bf16_partial` 768/768, max\|Δ\|=9.766e-04 (reproduces #431) |
| `fp32_accum_byte_identical` | **false** | 212/768 divergent; fp32-accum *is* the served kernel (#423 fp32-accum=0) |
| `fixed_tree_byte_identical` | **false** | 69/768; a deterministic tree ≠ serial's left-fold order (Higham §4) |
| `sorted_partial_byte_identical` | **false** | 65/768; drives toward TRUE sum, not the serial fold |
| `kahan_byte_identical` | **false** | 58/768; compensated → TRUE sum (tightest residual yet still breaks bytes) |
| `best_reproducible_variant` | **`none`** | the only 0-divergence variant (`serialized_fold`) is sequential |
| `reproducible_variant_preserves_speedup` | **false** | byte-identical set ∩ speedup set = ∅ |
| `pinnedk_can_be_unconditional` | **false** | no speedup-preserving variant byte-matches serial |
| `q1_collapsible` | **false** | order-invariant exact = a NEW reference = Q1 itself, not its dissolution |
| `max_gap_nats` | **0.125** (= `e*`) | every flip a bounded 1-ULP near-tie |
| `ppl` | **2.3772** ≤ 2.42 | reduction-order change is teacher-forced PPL-neutral |
| `self_test_passes` | **true** (51/51) | 0-GPU gate + GPU variant probe + stress arm + artifact provenance |

### Baseline ladder (legality target to respect — NOT beat)

| frontier | TPS | reference | status under this card |
|---|---|---|---|
| deployed FAST (#52, `2x9fm2zx`) | 481.53 | today's bytes | non-equivalent incumbent |
| blanket-strict (#423, `5a6zq2yz`) | 467.14 | today's bytes | byte-exact |
| frozen-byte (+cb3, #403 `iv9i2wks`) | 482.74 | today's bytes | byte-exact frontier |
| **pinned-K (#427/#431 `uza2t8aq`)** | **496.74** | self-referential | **stays `self_referential_only` — this card** |
| lawine #411 supply ceiling | 497.44 | — | — |

If the human picks **canonical-frozen** as the equivalence reference, the pinned-K +13.998 is not bankable and the fastest strictly-equivalent frontier stays **482.74**. If they accept the **self-referential / M-invariant** reference, 496.74 is legal — but that is the contract call, made by a human, exactly as #431 framed it.

### Reproduce

```bash
# full deliverable (GPU variant probe + rate-vs-KV stress arm + W&B):
cd target/ && CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/python -m \
  research.validity.pinnedk_reduction_byte_reproducibility.pinnedk_reduction_byte_reproducibility \
    --n-seeds 256 --stress-kv --stress-seeds 32 \
    --wandb_group pinnedk-reduction-repro --wandb_name denken/pinnedk-reduction-byte-reproducibility
# 0-GPU analytic gate:
cd target/ && .venv/bin/python -m \
  research.validity.pinnedk_reduction_byte_reproducibility.pinnedk_reduction_byte_reproducibility --self-test
```

> Note: invoke as a **`-m` module** (the PR's file-path form `python research/.../*.py` fails on the absolute `research.` package imports). This matches the run convention of the sibling cards #431/#423.

- **W&B run ID:** `9s76gsks` (group `pinnedk-reduction-repro`)
- **Peak memory:** 1097.4 MiB (synthetic-tensor probe; no model load, no GPU kernel build)
- **Self-test:** 51/51 — 0-GPU provenance gate (banked anchors from #431/#423), served-kernel source facts, the measured variant probe, the structural dichotomy, set-disjointness, the rate-vs-KV mechanism arm, and numeric hygiene. The smoke-scale guards mean a deliberately tiny run does not spuriously fail; at the deliverable scale (≥128 seeds) every corroboration is bulletproof.

### What happened — honest analysis

The PR hoped the `self_referential_only` label was a *fixable property of the reduction*. It is not. I confirmed three independent ways that **no parallel split-K reduction can be byte-identical to the canonical serial fold while keeping the speedup**:

1. **The served kernel already runs the "fix."** The Triton 3D split-KV combine already stores **fp32 partials** and tree-sums in fp32 — so instruction-2(a) "fp32 partial accumulation" is the *deployed* kernel, and it still diverges (212/768). The byte break is reduction **ORDER**, not precision (re-confirms merged #423).
2. **Every speedup-preserving variant diverges, measured (768 trials).** fp32-accum, fixed tree, sorted-magnitude, Kahan-Neumaier — all break bytes vs serial. The only 0-divergence construction (`serialized_fold`) reaches 0 by being sequential, which kills the parallelism. The byte-identical set and the speedup set are **disjoint**.
3. **Order-invariance is achievable but is a different reference.** An exact/Kulisch accumulator is split-count-invariant (8==16 bytes, measured) — but byte-identical to the TRUE sum, not the deployed serial bytes (165/768 divergent). Pinning it = making the Q1 contract choice, not dissolving it.

So `q1_collapsible = false`. This is a clean negative for the "free simplification," and a useful one: it tells the human **not** to fund a byte-reproducible split-K kernel hunt (the obstruction is the sequential carry dependency of the serial fold, fundamental on non-associative FP), and it leaves the bounded-near-tie guarantee from #431 fully intact. 496.74 is real and bounded-safe, but conditional on the reference choice — a human decision, as before.

### Suggested follow-ups

1. **Hand the human a clean binary, not a third option.** Pair this with stark's kernel-realizability leg: the 496.74 packet is "build the pinned-K kernel (stark) AND make the self-referential/M-invariant reference call (human)." There is no "and it's unconditionally legal so no call needed" branch — close that hope explicitly in the #407 write-up.
2. **If the team ever wants a *deterministic* pinned reference anyway,** the exact/Kulisch accumulator is the one to pin (it is provably split-count-invariant, measured 8==16). But scope it honestly as *adopting a new correctly-rounded reference* (a full greedy + PPL re-capture on the new bytes), not as serial-byte-identity. Cost ~2× accumulate (Johnson 2018); still a flagged served-file change and a human contract decision.
3. **No remaining byte-identical reduction lever.** Between this card (reduction math) and #423 (the serialization tax is the irreducible byte-identical floor), the byte-identical side of the pinned-K rung is fully characterized as closed. Future TPS on the frozen reference must come from the supply side (cb3-style), not from reconciling split-K with serial bytes.
