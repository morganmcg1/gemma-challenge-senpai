<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Greedy-Safety Boundary of the Served Verify Path (+ Closed-Lever Annex v2)

**PR #460 · denken · `equivalence-escalation-anchors` · analysis-only (0 TPS, no HF job, no submission, no served-file change) · PPL 2.3772 (gate ≤ 2.42) · 2026-06-16**

The #456 annex established **WHAT** is closed (20 levers; best realized byte-exact +0.26; all material headroom greedy-unsafe). This card states the **PRINCIPLE** behind it — the auditable *WHY* a skeptical reviewer demands:

> **A change to the served verify path is byte-exact (greedy-identical) if and only if it preserves the order of every floating-point reduction. Every material bandwidth-saving lever recovers its bandwidth by *reassociating* a reduction, and is therefore greedy-unsafe.**

The decisive new evidence is wirbel #442 ([`gyw2ksvs`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/gyw2ksvs) / census [`grrc3zms`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/grrc3zms) / floor [`cy0ijlit`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/cy0ijlit)): the Triton **3D split-KV attention** reduction is greedy-**UNSAFE** under a tile change (served greedy census **53.1% identical**, floor-proven REAL). #450's FP-reassociation hazard is **NOT confined to the Marlin GEMM** — it lives in the attention reduction too. That closes the unifying statement and adds **lever #21** to the ledger.

Model: `google/gemma-4-E4B-it`, int4-Marlin W4A16, Linear MTP K=7 (M=8 verify) + 3D split-KV attention (deployed `fa2sw_precache_kenyan`, PR #52, 481.53 TPS / PPL 2.3772, [`2x9fm2zx`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/2x9fm2zx)).

**Anchors (the frame, σ_hw ≈ 4.8 TPS):** deployed non-equivalent **481.53** ([`2x9fm2zx`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/2x9fm2zx), identity 0.9966) · realized blanket-strict byte-exact frontier **467.14** ([`5a6zq2yz`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/5a6zq2yz), #423) · roofline perfect-retile ceiling **510.87** (greedy-UNSAFE, [`c5oyb7gv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/c5oyb7gv), #450) · verify-BW λ=1 wall **520.95** ([`nvsbctji`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/nvsbctji), #436). Order: 467.14 < 481.53 < 510.87 ≤ 520.95; gap realized→deployed = 14.39 (−2.99 σ_hw).

---

## TL;DR

- **The boundary is a clean dichotomy.** For every kernel in the served verify path, each speed lever is either **REDUCTION-ORDER-PRESERVING** (pipeline/scheduling knob or no-knob → byte-exact, ≤ +0.26 TPS) or **REASSOCIATING** (re-partitions / re-combines an FP reduction → greedy-unsafe). There is no third class.
- **Every material headroom is on the reassociating side.** The only > σ_hw bandwidth slack — the ~16% int4-Marlin GEMM achieved-BW headroom (#450) and the (FLAG-2) larger Triton-attention surface — is recoverable *only* by reassociating a reduction. `byte_exact_levers_all_capped_le_0p26 = true`; `every_material_headroom_is_reassociating = true`.
- **Completeness.** Four FP-reduction families exhaust the verify path. Each is either pinned to deployed order (byte-exact, no BW headroom) or reassociable-for-BW-but-greedy-unsafe. `reduction_count_enumerated = 4`.
- **The trap:** a reassociating change can be **PPL-neutral yet identity-breaking**. #442 passed PPL (2.3767 ≤ 2.42) while its greedy census was only 53.1% identical. **PPL-pass ≠ greedy-identical.**
- `closed_lever_count = 21` (lever #21 = #442 Triton-attn tile-retune greedy-unsafe). Conclusion robust to the pending FLAG-2 correction (parameterized slot below).

---

## 1. The greedy-safety boundary table

For each kernel in the served verify path, every known speed lever, classified. **PRESERVING** = does not change the order in which the FP reduction accumulates (a pipeline/scheduling knob, or no exposed knob → stays at deployed order) → byte-exact. **REASSOCIATING** = re-partitions or re-combines the reduction → different float rounding order → can flip an argmax → greedy-unsafe.

### 1a. int4-Marlin W4A16 GEMM — ~85% of verify ([`crrq2e1y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/crrq2e1y): `int4_gemm_frac_of_verify=0.8509`)

Dominant weight read (qkv_proj, o_proj, gate_up_proj, down_proj × all decoder layers). Reduction = the K-dim (contraction) accumulation.

| lever | class | run · PR | result |
|---|---|---|---|
| **tile / kernel-config** (no exposed Python `num_splits`/`max_par` knob → stays deployed in-kernel order) | **PRESERVING** | [`fn4iz0dz`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/fn4iz0dz) · #448 | **+0.00** byte-exact-safe; Marlin is the UNIQUE sm_86 int4 GEMM |
| **split-K / BLOCK_K / num_warps re-tile** (re-partitions the K reduction) | **REASSOCIATING** | [`c5oyb7gv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/c5oyb7gv) · #450 | **greedy-UNSAFE** (`realistic_splitk_greedy_safe=false`); this is where the ~16% BW slack lives (+12.6…+31.4 TPS, unsafe) |
| **`use_fp32_reduce=False`** (changes the K-reduce accumulation dtype/order) | **REASSOCIATING** | [`fn4iz0dz`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/fn4iz0dz) · #448 | breaks byte-exactness on **3/4 shapes**; identity-breaking UB **+0.64 < +2** |

The M-dependence is the fingerprint: Marlin chooses its split-K geometry as `f(M)`, so M=1 decode and M=8 verify already reduce K in *different* float order (#122 [`n5bypf5h`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/n5bypf5h)) — the deployed 481.53 is itself non-equivalent (identity 0.9966) for exactly this reason. The order is *pinned*; any retile that moves it is reassociating.

### 1b. Triton 3D split-KV attention — head-256 sliding (×35) + head-512 global (×7), ~14.19% of verify ([`crrq2e1y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/crrq2e1y): `attn_frac_of_verify=0.1419`)

The served M=8 verify routes to vLLM's 3D split-KV (FlashDecoding) path (`splitkv_verify_patch.py`). Reduction = the per-query-token **online-softmax segment merge** (`reduce_segments`: max-track + exp-rescale + weighted-PV sum) across the KV-partition axis.

| lever | class | run · PR | result |
|---|---|---|---|
| **`num_stages`** (software-pipeline depth — prefetch scheduling; the accumulation order is identical) | **PRESERVING** | [`crrq2e1y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/crrq2e1y) · #447 (ref #428) | **+0.2613** byte-exact (kernel +6.11% on the 75.3µs tunable slice) |
| **BLOCK_M / BLOCK_Q tile** (BLOCK_Q 4→1 ⇒ grid 96→288 CTAs; re-partitions the split-KV reduction) | **REASSOCIATING** | **NEW** [`grrc3zms`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/grrc3zms)/[`cy0ijlit`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/cy0ijlit)/[`gyw2ksvs`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/gyw2ksvs) · #442 | **greedy-UNSAFE** — census **53.1% identical** (`byte_exact=0`, 30/64 diverge, first flip @ tok 290), floor-proven REAL; wall **−5.65 TPS** (evaporates) — **lever #21** |

**This is the decisive new finding.** The #442 injector's a-priori argument — "`reduce_segments` is BLOCK_Q-independent, `num_stages` is a pure pipeline knob ⇒ byte-exact" — was **empirically refuted** by a served census. Changing BLOCK_M→4 / BLOCK_Q→1 perturbs the 3D split-K reduction order → FP-reassociation → flipped argmax. The clean attribution holds because the two knobs are *isolable*: `num_stages` alone is byte-exact (#447, +0.26), so the identity break is owned by the BLOCK_M/BLOCK_Q **partition** change, exactly the reduction-order-preserving ⟺ byte-exact dichotomy.

> **The floor control is the lynchpin.** Default-vs-default census = **64/64 = 100% identical** across two independent processes ([`cy0ijlit`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/cy0ijlit)) → the eager M=8 path is *not* a cross-process FP-noise source (the lawine-#438 confound does not apply), so the 53.1% bm4 divergence is a **REAL identity break**, not measurement noise.

### 1c. lm_head vocab GEMM — 262144 vocab, ~0.64% of verify ([`crrq2e1y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/crrq2e1y): `lm_head_frac_of_verify=0.0064`)

Reduction = the hidden→vocab projection K-accumulation that produces logits (argmax over vocab selects the token).

| lever | class | run · PR | result |
|---|---|---|---|
| **full-hidden read** (deployed order; the verify GEMM reads the full 12288 hidden) | **PRESERVING** | #144 (lm_head verify audit) | pinned; candidate-restricted read is **2.1× SLOWER + correctness-impossible** → no byte-exact lever, no BW headroom |
| (split-K on lm_head) | REASSOCIATING (immaterial) | — | would reorder → greedy-unsafe, but lm_head is 0.64% of verify → ~0 BW headroom |

### 1d. RMSNorm — Σx² over hidden, < 0.1% of verify

Reduction = sum-of-squares over the hidden dim (input/post-attention/pre-&-post-feedforward norms, Gemma q/k norms, final norm).

| lever | class | run · PR | result |
|---|---|---|---|
| **deployed reduction order** | **PRESERVING** | — | pinned; memory-trivial at bs=1, not BW-bound → no headroom either way |
| (tree-reduce reorder) | REASSOCIATING (immaterial) | — | greedy-unsafe but ~0 BW |

### 1e. Drafter kernels (not verify, listed for completeness) — fused sparse argmax

| lever | class | run · PR | result |
|---|---|---|---|
| **fused-sparse-argmax tile** | **PRESERVING** | [`xryqregh`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/xryqregh) · #449 | **+0.00** (served default wins the full 45-config grid sub-µs) |

**Boundary summary:** every PRESERVING lever realizes ≤ **+0.26 TPS** (Marlin +0.00, drafter +0.00, attention num_stages +0.26, lm_head/RMSNorm pinned). Every material BW headroom sits on the REASSOCIATING side (Marlin split-K ~16% slack; the FLAG-2 attention surface) and is greedy-unsafe.

---

## 2. Enumerate the reductions — proof of completeness

The principle is only airtight if *every* FP reduction in the verify path is accounted for. The served Gemma-4-E4B M=8 verify forward (+ lm_head) contains exactly **four reduction families**; each is classified below. `reduction_count_enumerated = 4`.

| # | reduction | where | order status | BW headroom | reassociable lever (greedy-unsafe) |
|---|---|---|---|---|---|
| R1 | **GEMM K-reduction** (W4A16 contraction accumulate) | int4-Marlin: qkv/o/gate_up/down × all layers (~85%) | **pinned** (Marlin in-kernel split-K = f(M); no Python knob, #448/#122) | **~16% achieved-BW slack** (#450) | split-K / BLOCK_K / num_warps / `fp32_reduce=False` → #450 / #448 |
| R2 | **Attention split-KV online-softmax merge** (`reduce_segments`: max + exp-rescale + ΣPV across KV partitions) | Triton 3D: head-256 sliding ×35 + head-512 global ×7 (~14.19%) | **pinned** (deployed BLOCK_M=16 / BLOCK_Q=4 partition) | byte-exact knob `num_stages` → +0.26 only; material BW only via partition change | BLOCK_M / BLOCK_Q / num_splits re-tile → **#442** (census 0.531) |
| R3 | **lm_head vocab GEMM K-reduction** (hidden→vocab → logits) | lm_head (~0.64%) | **pinned** (full-hidden read, #144) | **none** (candidate-restrict 2.1× slower + correctness-impossible) | split-K (immaterial: 0.64% slice) |
| R4 | **RMSNorm Σx²** (sum-of-squares over hidden) | all norms (< 0.1%) | **pinned** | **none** (memory-trivial at bs=1) | tree-reduce reorder (immaterial) |

**The closure statement.** Every FP reduction in the verify path is either:
- **(a) pinned to the deployed order** — byte-exact, and carries **no material BW headroom** (R2-`num_stages`, R3, R4 and the no-knob R1 tile-config: max realizable **+0.26**); or
- **(b) reassociable to recover bandwidth** — but reordering the accumulation flips low-bit argmaxes → **greedy-unsafe** (R1 split-K = the ~16% GEMM slack; R2 partition re-tile = the FLAG-2 surface).

There is no reduction in class (a) with material BW headroom, and there is no member of class (b) that is greedy-safe. **byte-exact ⟺ reduction-order-preserving** — proven by enumeration. This is the answer to *"are you SURE there's no byte-exact lever you missed?"*: **no** — here is every reduction in the path, and the only BW-bearing ones are reassociating.

---

## 3. The trap — PPL-pass ≠ greedy-identical

A reassociating change can be **PPL-neutral yet identity-breaking**, because the two metrics probe different things:

- **PPL** is teacher-forced: each step conditions on the *ground-truth* prefix and scores `log p(next | gt-prefix)`. A sub-ULP per-logit perturbation almost never moves the *probability mass*, so PPL is robust to reassociation.
- **Greedy identity** is free-running: each step conditions on the model's *own* previous argmax. A **single** argmax flip cascades — every subsequent token can diverge.

**#442 is the live proof:** the bm4 retile **passed PPL** (2.3767 ≤ 2.42, anchor 2.3772 — [`grrc3zms`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/grrc3zms) `ppl_bm4=2.3767`) while its greedy census was only **53.1% identical** (`verdict_byte_exact_and_ppl_pass=False`). A PPL-only gate would have **waved bm4 through as valid**. The lesson for the relax-prize cards: **certify greedy identity by a served same-path census, never by PPL or an a-priori byte-exact assertion** — #442's assertion was wrong, and only the census caught it.

---

## 4. Annex v2 — lever #21 and the FLAG-2 parameterized slot

**Lever #21** (added to the ledger, `closed_lever_count = 20 → 21`):

> **`supply-verify-attn-triton-tile-reassoc`** — re-tile the Triton 3D split-KV attention (BLOCK_M/BLOCK_Q) to cash the attention BW. Closed: **greedy-UNSAFE** (#442 census 53.1% identical, floor-proven real) **and** wall-negative (−5.65 TPS, evaporates). Reason class: **physics** (reassociates the split-KV reduction) + measurement (−5.65 realized). The reduction-order-preserving sibling (`num_stages`, lever #20 / #447) stays the byte-exact +0.26 ceiling; #21 is its reassociating complement.

This makes #442 the **4th independent kernel-tiling strict-NULL** (verify-wall #447 / int4-GEMM #448 / drafter-kernel #449 / **attn-tile #442**) and the **5th** isolated-op collapse (pinned-K +13.998→−5.82, cb3 +15.60→0.0, static-K +13.2%→−8.63%, autotune-isolated +15.86→**−5.65**).

### FLAG-2 — parameterized slot (NOT blocking)

#442 FLAG-2: the Triton verify-attention surface includes **head-256 sliding** (forced-log `head=256 q_rows=8 IS_3D=True`), routed through Triton 3D split-KV in verify — **larger** than #447's head-512-only **1.27%** map. wirbel is reseated on the served-stack measurement (group `equivalence-escalation-anchors`, metric `triton_verify_attn_frac_of_verify`).

**Status: PENDING** — as of 2026-06-16 the `equivalence-escalation-anchors` group has no wirbel `triton_verify_attn_frac_of_verify` run yet. So the #447 spine number is carried as a **parameterized slot**, not a blocker:

| slot field | value |
|---|---|
| `triton_verify_attn_frac_of_verify` (carried, lower bound, #447 measured) | **0.0127** (1.27%, head-512-only tunable Triton-3D) |
| upper bound (if all sliding+global route through Triton 3D in verify) | **0.1419** (14.19%, the full attention slice; 35 sliding : 7 global) |
| corrected value | **pending wirbel reseat** — revise the #447 spine 1.27% → measured when it lands |

**Robustness — the correction cannot reopen a strict win.** Even at the 14.19% upper bound:
1. The **byte-exact** knob on that surface is `num_stages`, which is **partition-invariant** — its realized ceiling is the #447 +0.26 (a bigger slice does not make a pipeline-depth retune greedy-unsafe, but #442's served wall A/B on the *full* head-256+512 surface realized **−5.65**: the 3D path is occupancy-saturated past 80 SMs, so retiling adds CTA overhead, not speedup). Deleting the *entire* tunable kernel caps at +4.27 (#447).
2. The **material BW** of the larger surface is recoverable only by a **partition** change (BLOCK_M/BLOCK_Q/num_splits), which is **greedy-unsafe** (#442 identity 0.531).

So the corrected (larger) Triton surface only **enlarges the greedy-unsafe column** — it does not move headroom into the byte-exact column. `closed_lever_count` stays **21**; the conclusion *"every material headroom is reassociating → greedy-safety is the binding constraint"* is **robust** to the pending measurement.

---

## 5. Consequence for the relax-decision (#407)

The #456 thesis stands and is now *principled*, not just enumerated: **physics does not close the frontier below 481.53; greedy-safety does.** Roofline leaves a real ~16% GEMM BW slack (and possibly a larger attention surface), but **every** lever that cashes it reassociates a reduction → greedy-unsafe. The strict (byte-exact) frontier is closed at the realized **467.14** (best byte-exact lever **+0.26**); the only way past it is to relax strict equivalence — exactly the question for the human (#407).

---

## 6. Self-test & W&B fields

`greedy_safety_boundary_self_test.py` (0-GPU) loads `closed_lever_ledger_v2.json`, asserts the boundary table is a total dichotomy (every lever ∈ {preserving, reassociating}), every reduction family is classified (a)/(b), the 21-lever ledger is internally consistent and every run id is rendered in this markdown, the FLAG-2 slot robustness invariants hold, and logs:

`reduction_count_enumerated=4` · `byte_exact_levers_all_capped_le_0p26=true` · `every_material_headroom_is_reassociating=true` · `closed_lever_count=21` · `greedy_safety_boundary_self_test_passes` · `analysis_only=true` · `no_served_file_change=true` · `official_tps=0` · `ppl=2.3772`.

```bash
cd target/ && CUDA_VISIBLE_DEVICES="" .venv/bin/python \
  research/equivalence_escalation/greedy_safety_boundary/greedy_safety_boundary_self_test.py \
  --wandb_group equivalence-escalation-anchors \
  --wandb_name denken/greedy-safety-boundary
# 0-GPU gate only (no W&B): add --no-wandb
```

## Public evidence used

Spine: denken #447 verify-wall map ([`crrq2e1y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/crrq2e1y)) + #456 closed-lever annex ([`k33t25ct`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/k33t25ct)). Decisive new lever: wirbel #442 served bm4 A/B ([`gyw2ksvs`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/gyw2ksvs)) + greedy census ([`grrc3zms`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/grrc3zms)) + eager floor ([`cy0ijlit`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/cy0ijlit)) + isolated autotune ([`e5n9a2dc`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/e5n9a2dc)). Reassociation-hazard anchors: ubel #450 roofline ([`c5oyb7gv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/c5oyb7gv)), stark #448 int4-GEMM audit ([`fn4iz0dz`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/fn4iz0dz)), kanna #122 batch-variance ([`n5bypf5h`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/n5bypf5h)). Incumbent: PR #52 ([`2x9fm2zx`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/2x9fm2zx)).
