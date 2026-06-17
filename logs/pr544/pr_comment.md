STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["d44b61gj"],"primary_metric":{"name":"base_fullhead_optimized_tps_ceiling","value":292.1},"test_metric":{"name":"head_verify_tax_pct","value":82.2}}

## Results — base_fullhead TPS-ceiling decomposition (LOCAL, analysis-only)

**TL;DR.** The quality-safe ship is **structurally slow**. The ~100-TPS base_fullhead→osoi5 gap is **82.2% the 262k-head verify-tax** and **17.8% the +5 transformer layers**; the K=7 MTP drafter is a **non-factor** (E[T] is actually *slightly higher* on the full-head body). The only identity-preserving lever is a **lower-precision full head** (int4/fp8 — argmax is already free), recovering **+38.3 TPS → ceiling 292.1 local (~302 official)**. **Even a magically-free head tops out at 328.9 local — still −25% under the unsafe-class 442 frontier.** Both top-line bools: `quality_safe_ship_can_beat_442=FALSE`, `argmax_tax_is_dominant_gap_driver=TRUE`.

`analysis_only=true`, `official_tps=0` — no HF job, no submission; official BASELINE (481.53) untouched.

### Measured serve (MAX_NUM_SEQS=1, two back-to-back decodes each, 128/128 prompts, 65536 tokens/decode)

| arm | local TPS (median wall) | anchor | PPL | peak GPU | exec gpu/step | t_cycle | E[T] |
|---|---|---|---|---|---|---|---|
| **base_fullhead** (42L int4 body + **262k bf16** head) | **252.31** (251.61 / 253.00) | 253.78 | **2.0057** byte-exact | 19411 MiB | 12.214 ms | 15.138 ms | 3.819 |
| **osoi5 ship** (37L baked + **16k int4** pruned head) | **350.76** (349.25 / 352.27) | 353.73 | 2.3767 | 19395 MiB | 7.928 ms | 10.845 ms | 3.804 |
| **gap** | **−98.46** (anchor −99.95) | | +0.371 PPL | | **−4.286 ms** | −4.293 ms | +0.015 |

Both arms reproduce their anchors within 0.6% / 0.8%. Full-head guard fired green: `lm_head.shape[0]=262144` (no silent 16k fallback could masquerade as full-head). E[T] is steptime-derived (`median_tps × t_cycle`); vLLM `/metrics` accept-counters weren't exposed this run.

### Four-way decomposition of the 4.286 ms / ~100-TPS step-time gap

| # | component | ms | % of step gap | TPS if removed | verdict |
|---|---|---|---|---|---|
| **C2** | **262k-head verify-tax** (vs 16k pruned) | **3.524** (3.22–3.83) | **82.2%** (75–89%) | +76.6 | **dominant** |
| **C1** | **+5 transformer layers** (42→37L, +13.5% depth) | 0.762 (0.46–1.06) | 17.8% | +23.1 | **irreducible** (kanna #539) |
| **C3** | MTP K=7 drafter E[T] | — | second-order | +1.42 | **non-driver** (base accepts *better*) |
| **C4** | residual kernel/overhead | 0.100 | recon | — | negligible |

**C1 + C2 = 4.286 ms = 100% of the exec-step gap** (17.8% + 82.2%). C3 (drafter, +0.015 E[T]) and C4 (residual, exec↔t_cycle reconciliation) are second-order, outside the step-ms split.

**C2 is the verify-step logits matmul, NOT the argmax.** Micro-bench (A10G, real bf16 `hidden@head.T` + greedy argmax, CUDA events, M=8 verify rows):

- 262k bf16 head = **2.698 ms**; 16k/12k int4 head = 0.069 / 0.057 ms; **argmax-over-262k = 0.032 ms** (free).
- Intrinsic head delta (262k_bf16 − 12k_int4) = **2.641 ms = 61.6% of the exec gap, measured directly.**
- The cost is the **1.342 GB dense bf16 weight read** (eff HBM 500.5 GB/s back-solved on *this* GPU), not FLOPs and not the reduction → **no faster-argmax trick helps**, only lower-precision head *weights*.

**Cross-check (hold E[T]=base, peel components):** base 252.3 →(−head) 328.9 →(−5L) **352.0** ≈ osoi5 **350.8**, residual error **1.19 TPS**. The four components are exhaustive.

### Realized E[T] (component 3, resolved)

base_fullhead **3.819** vs osoi5 ship **3.804** → **+0.015 (base is +0.4% higher).** The K=7 MTP drafter accepts *equally well — marginally better —* on the deeper 262k-head body. **No E[T] penalty; no re-keying needed.** The drafter is a ~1.4-TPS *mitigant*, not a cost.

### Optimized ceiling & irreducible floor

The full head's whole point is keeping all 262k token rows, so the only identity-preserving lever is **lower-precision head weights** (keeps every token; argmax already free):

| lever | head saves | recoverable TPS | ceiling (local) | ceiling (official proj, ×1.0352) |
|---|---|---|---|---|
| **int4 full head** | 1.996 ms | **+38.3** | **292.1** | **302.4** |
| fp8 full head (conservative) | 1.341 ms | +24.5 | 278.3 | — |
| **free head (upper bound)** | 2.698 ms | +76.6 | **328.9** | — |

- **`base_fullhead_optimized_tps_ceiling` = 292.1 local** (= 253.78 anchor + 38.3 int4-head) ≈ **302.4 official**.
- **`base_fullhead_irreducible_floor` — irreducible part of the gap = 61.6 TPS** = **23.1 (+5-layer depth tax, kanna #539)** + **~38.5 (the 262k head still reads 16× more rows than the 16k pruned head even at int4** — only *pruning* closes that, and pruning is exactly what breaks quality-safety).
- **⚠ The 292.1 ceiling is an UPPER bound:** int4 head weights perturb logits → may flip greedy near-ties → byte-exactness NOT guaranteed (Gemma int4 is already near-tie). fp8 is lower-risk (278.3). A strictly *byte-exact* full head cannot beat bf16 = the measured 252–253 serve.

### Top-line bools

- **`quality_safe_ship_can_beat_442` = FALSE.** Decisive: even a **free** full head caps base_fullhead at **328.9 local ≪ 442**; the realistic int4 ceiling 292.1 misses by 34%. Removing *only* the head (→ 328.9) still leaves it −113 under 442 — the +5 layers alone disqualify it. **The quality-safe ship is structurally slow — a clean, documented Morgan #524 input.**
- **`argmax_tax_is_dominant_gap_driver` = TRUE** (head verify-tax 82.2% central, ≥75% even at the most body-favorable bracket) — with the **correction** that the dominant cost is the dense **logits matmul (weight read)**, not the argmax reduction (0.032 ms). The highest-value engineering target for a faster quality-safe ship is **a lower-precision full head**, not a faster argmax.

### Structured terminal marker

```
SENPAI-CEILING {"pr": 544, "analysis_only": true, "official_tps": 0, "gap_tps": 100.0, "head_verify_tax_pct": 82.2, "body_5layer_pct": 17.8, "et_delta": 0.0154, "head_intrinsic_delta_ms": 2.641, "base_fullhead_optimized_tps_ceiling": 292.1, "base_fullhead_irreducible_floor": 292.1, "irreducible_5layer_tps": 23.1, "quality_safe_ship_can_beat_442": false, "argmax_tax_is_dominant_gap_driver": true}
```

### Public evidence used

Public challenge digest (`GET /v1/digest?as=senpai`, pulled 2026-06-17): the **osoi5 + lmhead12k pruned-head lineage** — the "unsafe ship" fast end of this gap — is live and **verification-valid** on the public leaderboard: **frantic-penguin #4** `osoi5-feopt2-w20-e1-lmhead12k-fa2sw-precache-skv64-v1` **489.63 TPS / PPL 2.3774 (valid)**; **need-for-speed #7** 488.07 / PPL 2.3774 (valid); **byteshark #9** 484.62 / PPL 2.3769 (valid). My `osoi5_ship` arm reproduces this exact pruned-head stack locally at **PPL 2.3767** (byte-matching the public frontier's 2.377), confirming the gap's fast end is the publicly-shipped 12k/16k-pruned-head lineage whose **PPL 2.377 > base_fullhead's byte-exact 2.006** — the speed is bought with quality the full-head ship refuses to spend. Active public taskforces: evals, ultra-kernels, llama-cpp.

### Reproduce (LOCAL, analysis-only, ~20 min on one A10G)

```bash
# 1) A/B serve: base_fullhead (full 262k head) vs osoi5 ship, 2 decodes each + STEPTIME + E[T] + PPL
.venv/bin/python -m research.validity.base_fullhead_tps_ceiling.serve_measure \
    --arms base_fullhead,osoi5_ship --n-decodes 2
#   base_fullhead bakes fern #535's recipe onto submissions/fa2sw_strict_surgical357:
#     LOCAL_MODEL_DIR=<qat-w4a16-ct snapshot>  PLE_FOLD_TARGET_MODEL=<same>
#     LM_HEAD_PRUNE=0  LM_HEAD_PRUNE_REQUIRE=0  PCK04_KEEPSET=""   (guard: raise if lm_head rows < 262144)

# 2) Head micro-bench: real bf16 262k-head matmul+argmax, int4/fp8 projection (serve venv, GPU free)
/tmp/senpai-venvs/5f4c623f772358a2/bin/python \
    research/validity/base_fullhead_tps_ceiling/microbench_head.py

# 3) Decompose + ceiling model + wandb (group base-fullhead-tps-ceiling)
.venv/bin/python -m research.validity.base_fullhead_tps_ceiling.decompose
```

Artifacts: `research/validity/base_fullhead_tps_ceiling/{decomposition.json, head_results.json, run/serve_results.json}`. **W&B run `d44b61gj`** (group `base-fullhead-tps-ceiling`). Peak GPU 19411 MiB (base_fullhead) / 19395 MiB (osoi5).

### What happened

The gap is real and it is **almost entirely the full 262k head**. Two independent bounds bracket it — the direct micro-bench head delta (2.641 ms = 61.6% floor) and the osoi5-per-layer-cap argmax (≥75.2%); central 82.2%. The cross-check (peel head + 5 layers off base, land on osoi5 within 1.19 TPS) proves the four components are exhaustive. The surprise vs the hypothesis: it is **not the argmax** (0.032 ms, free) — it's the **dense weight read** of the bf16 head, so the lever is *precision*, not a clever reduction. The drafter-E[T] worry (component 3) is fully dead: the 42L full-head body accepts *as well or better* than the 37L pruned body. Net — a fully-optimized quality-safe ship reaches ~292 local / ~302 official, far below the 442 unsafe class, and **even a physically-impossible free head can't close it** because the +5 layers (kanna #539's irreducible quality tax) alone cap it at 328.9. **base_fullhead is structurally a ~290–300 ship, not an un-optimized 442.**

### Suggested follow-ups

- **Validate the int4-head byte-exactness risk directly** (the one soft spot in the 292.1 ceiling): serve base_fullhead with an int4/fp8-quantized *full* 262k head, run the served-vs-served greedy-identity gate vs the bf16 full-head reference. If byte-exact → 292.1 is a *real* quality-safe ceiling; if it flips near-ties → the honest quality-safe ceiling is the bf16 252–253 serve.
- **Feed 292.1 local / 302.4 official into Morgan #524** as the quantified "good faster TPS" ceiling for the quality-safe class, beside the triad's "is it quality-safe" verdict — together they fully resolve the two-gate.
- The +5-layer 23.1-TPS tax is the only structural lever left and it's locked by kanna #539 (layer-drop = body damage); not worth re-opening unless a depth-preserving distill changes that.
