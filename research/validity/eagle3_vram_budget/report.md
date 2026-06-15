# PR #299 — EAGLE-3 build VRAM budget: does the fusion drafter fit ≤ 24 GiB?

**PRIMARY `eagle3_vram_budget_self_test_passes` = True**
**TEST `eagle3_build_fits_24gb` = True** · `fits_23_usable` = True · `fits_device_visible(22.058)` = True
**`eagle3_net_memory_delta_gb` = 0.800** (conservative, hold-KV-capacity) · **`eagle3_build_resident_gb` = 20.10 GiB**
**`dominant_memory_term` = extra_kv** · W&B `u2sv81kl` (group `eagle3-vram-budget`)

> **Verdict:** the EAGLE-3 fusion drafter **FITS** with room to spare. Deployed resident **19.30 GiB** + net delta **0.800 GiB** = **20.10 GiB** build resident — **3.90 GiB** under 24-hard, **2.90 GiB** under 23-usable, and under the measured **22.058 GiB** device-visible cap. VRAM is **not** the binding constraint on the human-gated EAGLE-3 retrain. The drafter could be ~16× larger (decoder layers) before busting 23 GiB.

## 1. Deployed resident memory MAP (reconcile to #284 19.30 GiB)

All terms log-measured from the #284 deployed rig (`research/validity/decode_host_overhead/server_deployed_decode.log`):

| term | GiB | source |
|---|---|---|
| model loading (int4 body + linear MTP drafter) | 8.85 | log:128 |
| KV cache (376,880 tokens) | 9.46 | log:151/154 |
| CUDA-graph pool | 0.04 | log:159 |
| **torch subtotal** | **18.35** | sum |
| non-torch CUDA context (residual) | 0.95 | balancing; ∈ plausible band [0.5, 1.2] |
| **reconciled resident** | **19.30** | **resid 0.000 vs #284 anchor** |

Cross-check: util budget = 0.8974 × 22.058 = **19.79 GiB** (gap 0.49 to resident = transient activation peak).

## 2. EAGLE-3 fusion-drafter NET memory delta (vs the linear drafter it replaces)

Parametric from live config dims (hidden 2560, heads 8/kv 2, head_dim 256, intermediate 10240, vocab 262144), **corroborated by a random-init GPU spot-check on the live A10G** (rel_err **0.22%**, param count exact 119,285,760; `device_total` measured 22.058 GiB = banked).

| term | GiB | basis |
|---|---|---|
| drafter_weights | +0.0374 | EAGLE-3 1-layer 2H-input Llama decoder (99.6M) + final norm − linear drafter (0.148 GiB) |
| fusion_fc | +0.0366 | `[3H→H]` + bias (19.7M params) |
| hidden_state_retention | +0.0073 | L_FUSE(3) × H(2560) × draft_positions(512) × 2B |
| **extra_kv** (dominant) | **+0.7188** | +1 drafter attention layer @ 376,880 tokens (full-attn, conservative) |
| **net_delta (conservative)** | **+0.800** | hold-KV-capacity |
| net_delta (elastic / weights-only) | +0.081 | extra_kv is elastic → see below |

**Elastic-KV caveat:** vLLM sizes total KV to fill the util budget, so the drafter's +1 KV layer does **not** grow resident — it trades **~7.6% of KV tokens** (376,880 → ~348,000). The deployment-honest build resident is therefore **19.38 GiB** (weights-only); the 20.10 GiB headline is the conservative hold-capacity upper bound. **Both fit.**

## 3. Fit verdict + embed/lm_head sensitivity

EAGLE-3 has `draft_dim == backbone_dim == 2560`, so it **reuses the target embed + lm_head** (the linear drafter keeps its own [262144,256] lm_head only because draft_dim≠backbone_dim; gemma4.py:176). PRIMARY (S0) assumes reuse.

| scenario | build resident (GiB) | fits 24 | fits 23 | fits 22.058 |
|---|---|---|---|---|
| **S0 reuse embed+lm_head (PRIMARY)** | **20.10** | ✅ | ✅ | ✅ |
| S1 separate full-vocab lm_head (+1.25) | 21.35 | ✅ | ✅ | ✅ |
| S2 untied separate embed+lm_head (+2.50) | 22.60 | ✅ | ✅ | ❌ |
| S3 reduced 32k-vocab lm_head (+0.16) | 20.26 | ✅ | ✅ | ✅ |

Only the doubly-pessimistic S2 (untied separate full-vocab embed **and** lm_head **and** hold-KV) brushes the 22.058 GiB visible cap — unrealistic for EAGLE-3. **Every scenario fits both PR ceilings.**

## Self-test (a–i, NaN-clean)
a deployed-map reconciles (resid 0.000) · b decomposition sums · c EAGLE-3 params exact + spot-check agrees (0.22%) · d build-resident identity · e dominant = extra_kv · f fit-flags consistent + fits · g constants exact · i all sensitivity scenarios fit 24 → **PRIMARY PASS**.

## Greedy/PPL-safety certificate
`analysis_only = True`. No served-file change, no emitted-token change, no HF Job, no submission, NOT a launch, NOT a build (the GPU spot-check allocates random-init weight tensors only, then frees them). BASELINE **481.53 TPS unchanged**; this leg adds **0 TPS**.

## Hand-off
EAGLE-3 fusion drafter fits ≤24 GiB (20.10 conservative / 19.38 elastic build resident) with ~2.9 GiB of 23-usable headroom; the only material new allocation is the drafter's extra KV layer (+0.72 GiB, elastic → ~7.6% fewer KV tokens, not +resident), weights add just +0.08 GiB, embed/lm_head reuse keeps the head free, and VRAM is **not** the binding constraint on the human-gated retrain.
