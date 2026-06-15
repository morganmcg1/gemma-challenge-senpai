STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["n384zrxq"],"primary_metric":{"name":"token_identity_rate","value":0.98779296875},"test_metric":{"name":"detector_self_test_passes","value":1},"headline":{"token_identity_rate":0.98779296875,"min_selective_eta_at_identity_1":0.26594761439732145,"best_m8only_precision_at_recall1":0.04904364884747425,"L_star_min_separating":null,"cheap_detector_clears_500_budget":false,"selective_beats_blanket":false,"verdict_band":"RED_closes_selective_lever"}}

## Results

**Verdict: RED — CLOSES the selective-DVR lever.** No cheap+correct detector lands below the **9.841%** blanket, let alone the **4.02%** >500 budget. The decisive evidence is the answer to the card's *central* question — *"because the divergence is hidden-driven, does a signal that reads the residual-stream perturbation **directly** beat the margin's precision?"* — and the answer is a clean **NO**: the partial-M=1 residual-disagreement `‖h_M8(L*)−h_M1(L*)‖` has **AUC ≈ 0.5 at every depth L\*∈{3…42}** (0.445–0.597). The residual perturbation that *causes* the flip does not have larger *magnitude* at flip positions — flips are set by the **logit margin** (proximity to the argmax boundary), not by how big the hidden-state kick is. So the margin is not a lossy proxy for a better hidden-state signal; it is already the sufficient statistic, and it is **base-rate precision-capped** exactly as #364 found. Required headline fields: `token_identity_rate = 0.987793`, `best_m8only_precision_at_recall1 = 0.0490`, `L_star_min_separating = None`, `min_selective_eta_at_identity_1 = 26.59%`, `cheap_detector_clears_500_budget = False`, `selective_beats_blanket = False`, W&B **`n384zrxq`**.

### Primary
| metric | value | meaning |
|---|---|---|
| **`token_identity_rate`** (PRIMARY) | **0.987793** | 8092/8192 M=8-verify positions match plain greedy AR (M=1) |
| `verify_divergence` | 0.012207 | **100/8192** HF positions flip (16 prompts × 512 new) |
| `hf_vllm_agreement` | **0.9844** | HF instrument agrees with #364's vLLM flip oracle on 98.4% of positions |
| `determinism_verify_geometry` | **1.0** | M=8 re-forward reproduces the argmax stream bit-exactly |
| **`detector_self_test_passes`** (TEST) | **True** | all 14 self-test checks pass (0 fails) |

> **Instrument note (honesty).** vLLM exposes no per-layer hidden states/attentions, so the detector frontier is traced in a self-contained HF-transformers dual-geometry harness (M=1 AR vs M=8 chunked-verify, `sdpa`) on the **same int4 deployed substrate** and the **same #364 teacher-forced token sequences** (reconstructed from the merged `_marginscan.json`). The HF instrument flips **100/8192 (1.22%)** vs the deployed vLLM **32/8192 (0.39%)** — the documented cross-engine bf16-GEMV vs Marlin-int4 gap (#362 ~9–13%); the two agree on 98.4% of positions and the verify geometry is bit-exact-deterministic. `selective_eta` is reported **both** at the HF rate and normalized ×0.32 to the deployed 0.39% (see the normalization caveat below — the ×0.32 linear model is a lower bound, not the operative number).

### Stage 1 — free M=8-only detectors: NOTHING beats the margin (precision @ recall=1.0)
All signals are read off the M=8 verify forward we already run (cost = free); `selective_eta = frac_flagged × full-M=1-forward`.

| free detector | AUC | frac flagged = eta | precision @ recall 1.0 |
|---|---|---|---|
| **`margin_m8`** (top1−top2) | **0.9905** | **24.89%** | **0.0490** ✅ best |
| `lens_min_margin` (logit-lens depth) | 0.703 | 96.1% | 0.0127 |
| `lens_instab` (lens argmax-instability) | 0.776 | 100.0% | 0.0122 |
| `attn_entropy_max` (bf16-inject layers) | 0.431 | 99.8% | 0.0122 |
| `attn_maxw_min` | 0.445 | 98.5% | 0.0124 |
| `resid_norm_anom` (‖h‖ z-score) | 0.611 | 99.6% | 0.0123 |
| **free fusion** (held-out logistic, 5 splits) | 0.959 | 19.5% | 0.0824 |

**Answer to "does any free M=8-internal signal break the margin's precision ceiling?" — NO.** The margin is the *only* free signal that separates (AUC 0.990); every other internal signal (logit-lens, attention entropy/concentration, residual-norm anomaly) is at-or-near chance and flags ~all positions. A held-out logistic **fusion** of all six lifts precision to 8.2% at frac 19.5% — better than margin alone but still `selective_eta ≈ 19.5%` (×0.32 → 6.2%), nowhere near the 4.02% budget. The margin's 4.9% precision here vs #364's 2.3% is purely the 3.1× higher HF base rate (precision scales with base rate); the *ranking* — margin ≫ everything else — is the invariant.

### Stage 2 — the direct hidden-state read FAILS (this is the result)
For the stage-1 margin pre-filtered set (frac 24.89%, τ1=1.091 nats, recall 1.0), run M=1 geometry only to depth L\* and threshold `dh = ‖h_M8(L*)−h_M1(L*)‖`, passing only the doubly-flagged to the full M=1 forward.

| L\* | L/n | dh AUC vs flips | frac doubly | two-stage eta | held-out recall_min |
|---|---|---|---|---|---|
| 3 | 0.07 | 0.550 | 24.82% | **26.59%** | 1.000 |
| 5 | 0.12 | 0.549 | 24.30% | 27.27% | 0.958 |
| 8 | 0.19 | 0.597 | 24.11% | 28.85% | 0.958 |
| 11 | 0.26 | 0.543 | 24.32% | 30.84% | 0.964 |
| 16 | 0.38 | 0.463 | 24.67% | 34.15% | 0.905 |
| 21 | 0.50 | 0.445 | 24.80% | 37.25% | 0.958 |
| 26 | 0.62 | 0.454 | 24.87% | 40.27% | 0.958 |
| 32 | 0.76 | 0.519 | 24.80% | 43.77% | 0.821 |
| 42 | 1.00 | 0.586 | 24.88% | 49.77% | 0.958 |

- **`dh` AUC ≈ 0.5 at every depth** (independently re-derived from the raw scan: 0.45–0.60). The residual disagreement is real and varies (finite at all 8192 positions, ‖·‖ grows with depth), but it is **uncorrelated with flips**. On the stage-1 set, flip vs no-flip dh means are nearly identical (L=8: 0.570 vs 0.531; L=21: 2.13 vs 1.60; L=42: 4.96 vs 4.37) — a hair higher at flips, not separable.
- **`frac_doubly ≈ stage1_frac` (≈24.8%) at every L\***: because dh doesn't separate, the recall-1 stage-2 threshold flags essentially the *entire* stage-1 set. The two-stage gate therefore **adds the partial-M=1 depth cost without shrinking the flagged set** → strictly worse than the margin alone. `min two-stage eta = 26.59% at L\*=3`, rising monotonically with depth.
- **`L_star_min_separating = None`**: no layer clears the honest gate (AUC ≥ 0.90 **and** held-out recall_min ≥ 0.999). L\*=3 reaches held-out recall 1.0 only by flagging everything (AUC 0.55), which the AUC gate correctly rejects.

### Stage 3 — frontier + verdict (decision-gate mapping)
| detector (achieves identity 1.0 in-sample) | selective_eta (HF rate) | ×0.32 → deployed | vs budget 4.02% | vs blanket 9.84% |
|---|---|---|---|---|
| margin pre-filter alone (#364 mechanism) | 24.89% | 7.97% | ✗ | ✗ (HF) |
| **two-stage margin→dh (best, L\*=3)** | **26.59%** | **8.51%** | ✗ | ✗ (HF) |
| free fusion (logistic) | 19.50% | 6.24% | ✗ | ✗ |
| — reference floors — | | | | |
| blanket batch-invariant GEMM (#360) | **9.841%** | — | — | — |
| >500 kernel budget (λ=1 ceiling 520.953) | **4.022%** | — | — | — |
| probe-gated floor (#362, perfect detector) | **0.97%** | — | — | — |

→ **`cheap_detector_clears_500_budget = False`, `selective_beats_blanket = False`, `selective_beats_probe_gated_362 = False`.** Decision gate ⇒ **"no detector < 9.841% → bank the definitive RED that CLOSES the selective-DVR lever."** The eta-axis must clear 500 via the **blanket** pinned-split (stark #365 territory), not selective repair. For fern #357's composite: **the composite cannot rely on a cheap selective-repair term.**

### Normalization caveat (read this before quoting the 8.5% number)
The script also reports `min_selective_eta_normalized_to_deployed = 8.51%`, which dips *below* the 9.841% blanket. **This is a linear-scaling artifact, not a real "beats blanket."** The ×0.32 factor assumes `frac_flagged ∝ base_rate`, but the recall-1 flagged set is dominated by the **natural low-margin background (~17% of positions), which is base-rate-independent** (it's a property of the language, not the flip count). The honest deployed anchor is #364's **direct vLLM measurement**: margin recall-1 flags **17.0% (in-sample τ100) / 44.6% (provable τ_robust)** at the true 0.39% rate — both **> blanket**. Since `dh` adds AUC≈0.5 (zero separation), it cannot reduce that 17%; it can only add cost. So the operative deployed selective eta is **≥17% > blanket**, and the ×0.32 → 8.5% is a lower bound that undershoots the measured deployed frac by ~2×. **RED holds under the HF rate (24.9%), the honest deployed rate (17%, #364), and the fusion (19.5%) — only the naive linear normalization dips under blanket.**

### Determinism / validity controls (signal, not noise)
- `determinism_verify_geometry = 1.0` (M=8 re-forward bit-exact on 2 prompts) ⇒ the 100 flips are the deterministic reduction-geometry effect, not jitter.
- `hf_vllm_agreement = 0.9844`, `m1_argmax_eq_teacherforced = 0.8597` (cross-engine, ≥0.80 expected) — the HF instrument tracks the #364 vLLM oracle.
- `corroborates_364_identity_order = True` (|0.9878 − 0.9961| = 0.0083 ≤ 0.05). 5 held-out splits (≥2 seeds). nan-clean. Self-test 14/14.
- Margin AUC (0.9905) and all dh AUCs independently re-derived from the raw `_detscan.json` — match `compose` exactly (no sign/aggregation bug).

### Secondary (LOCAL — 0 official TPS)
- Cost model: full bf16 lm_head GEMV = 2778.8 µs, body 1-layer = 422.1 µs, `f_lmhead_bandwidth = 0.4106`. A partial-M=1 to L\* costs `L/n` of a body forward — but with dh non-separating this buys nothing.
- Peak GPU: **16.34 GB** (A10G 23 GB; the dual-geometry scan caches per-layer hidden states, hence > #364's 11.9 GB).

### Baseline comparison (per PR)
- Official frontier **481.53 TPS / PPL 2.3772 (#52)** — **UNCHANGED**. Local strict-mechanism screen: **0 official TPS, no served-file change, no HF job, no submission, no `--launch`.** ✅
- λ=1 ceiling **520.953**; budget eta **4.022%** (`520.953·(1−0.0402)=500`); blanket **9.841%**; probe-gated **0.97%** — all advisor-provided PR anchors, reused not re-derived.

### Command
```
cd target/ && CUDA_VISIBLE_DEVICES=0 python research/validity/selective_dvr_detector/selective_dvr_detector.py \
    --gpu --n-prompts 16 --n-new 512 --ctx-cap 256 --det-prompts 2 --n-splits 5 \
    --wandb_group selective-dvr-detector --wandb_name ubel/selective-dvr-detector-eta
```
GPU phases run as isolated subprocesses (`CUDA_VISIBLE_DEVICES=0`) on the on-target pod A10G (GA102/sm_86); int4 substrate is the deployed `gemma-4-E4B-it-qat-w4a16-ct` snapshot. **W&B run `n384zrxq`** (group `selective-dvr-detector`), free-detector + dh-depth frontiers logged as Tables.

### What happened
The card **definitively closes the selective-DVR lever.** It tested the advisor's sharpest remaining hypothesis — that because the divergence is hidden-state-driven, a signal reading the **residual-stream perturbation directly** could beat the logit-margin's base-rate precision cap — and **refuted it on both fronts**:
1. **Stage 1:** no free M=8-internal signal (logit-lens depth trajectory, attention entropy/concentration at the bf16-injection layers, residual-norm anomaly) separates flips; the margin (AUC 0.990) is the only one, and a 6-signal held-out fusion only reaches frac 19.5% — still ~5× the budget.
2. **Stage 2:** the direct partial-M=1 residual read has **AUC ≈ 0.5 at every depth 3→42** — the magnitude of the M8−M1 hidden-state disagreement is **uncorrelated** with whether the argmax flips. The flip is governed by the *logit margin* (boundary proximity), which the M=8 logits already expose for free. So the two-stage gate flags the whole pre-filter set and only adds cost.

The mechanistic punchline: the divergence *is* hidden-driven, but its **effect** is gated by margin, not by perturbation size — so there is no cheaper-than-margin correct detector, and the margin is precision-capped at ~17% (deployed) ≫ 9.841% blanket. **The cheap+correct detector wirbel #362 needs does not exist among these candidates; the eta-axis to >500 runs through the blanket pinned-split (stark #365), not selective repair.**

### Suggested follow-ups
1. **Stop searching the detector axis.** Stage-1 (logit) and Stage-2 (hidden-state) are now both closed for cheap selective repair; the remaining theoretical floor (0.97% probe-gated, #362) requires a *perfect* detector that this card shows none of the natural cheap signals approximate. Selective repair should be retired as a >500 route.
2. **Bank the closure for fern #357's composite** — the composite's eta term must come from the **blanket** (9.841%, #360) or a sub-blanket batch-invariant-kernel lift (e.g. pinned-split #366 isolation 0.39%, pending stark #365's end-to-end paged-KV confirmation), **not** from a selective detector.
3. **If selective is ever revisited:** the only lever the data leaves open is the **complement** — since margin recall=1.0, `margin>τ` cleanly gates the ~75–83% of positions that are *safe to accept without verify* (a throughput lever, not an identity lever). Out of scope here, but the AUC-0.99 margin supports it.
4. **Margin-as-sufficient-statistic** is the reusable lesson: for this reduction-geometry divergence, top1−top2 at the M=8 verify logits is the maximal cheap signal; no upstream internal feature adds separating information.

### Repro / notes
- Three GPU subphases (detector_scan `sdpa`, attn_entropy `eager`, recompute_timing) self-isolate as subprocesses; analysis re-runs cheaply via `--reuse-scan` over the cached `_detscan.json`/`_entropy.json`. `--relog-wandb` re-logs an existing `_results.json` without GPU.
- The HF instrument's 3.1× flip inflation vs deployed is expected cross-engine numerics; it is handled by base-rate normalization **and** cross-checked against #364's direct vLLM 17%/44.6% deployed measurement, so the RED verdict does not hinge on the normalization model.

_Public-evidence note: builds directly on my own merged **#364** (`i08xlqbg`, logit-margin selective-DVR closure: recall 1.0 / precision 2.3% / eta 17.0–44.6%). Reference floors (blanket 9.841% #360; probe-gated 0.97% / lm_head-only-invalid #362; budget 4.022% from λ=1 ceiling 520.953; frontier 481.53/PPL 2.3772 #52) are the advisor-provided PR anchors, reused not re-derived. Checked the HF challenge digest (`as=senpai`) for `@senai` interrupts before reporting — none blocking; this card makes no submission and adds 0 official TPS. No files outside `research/validity/selective_dvr_detector/` changed._
