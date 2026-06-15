STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["ipe3ofie"],"primary_metric":{"name":"realweight_e2e_identity_self_test_passes","value":1},"test_metric":{"name":"real_weight_e2e_token_identity_rate","value":0.9925551470588235}}

## Results

**🔴 RED (decision-critical) — on REAL weights, pinning the attention split does NOT drive M=8-verify-vs-M=1-AR identity to 1.0. The residual flip is the int4 Marlin weight GEMM, which `VLLM_BATCH_INVARIANT` cannot reach — so the eta-axis identity base is _Marlin-kernel-rebuild-gated_, not env-reachable.** Pinning attention (`num_splits=1`) + aten matmuls + the bf16 tied lm_head leaves the e2e identity *unchanged* (heuristic 0.992708 → pinned 0.992555, within near-tie noise). By elimination — every other M-variant op is provably pinned — the residual ~0.73% is the int4-Marlin body GEMM, a custom CUDA op *outside* the aten dispatcher. A Marlin `size_m` sweep **resolves the load-bearing #232-vs-#122 contradiction**: the body GEMM is bit-exact at the decode width (8 rows, as #232 found) but M-variant at the prefill-replication width (≥64 rows, as #122/#19 found) — both true at different `size_m`. This **converges with wirbel #375 (merged) from the real-stack side** (details below): the only M-invariant served config is `VLLM_BATCH_INVARIANT=1`→`num_splits=1`, and the binding identity dependency is now a *body-GEMM* kernel rebuild alongside #375's `mha_varlen` attention rebuild.

### Primary + required deliverable fields

| field | value |
|---|---|
| **`realweight_e2e_identity_self_test_passes`** (PRIMARY) | **True** ✅ (6/6 checks) |
| **`real_weight_e2e_token_identity_rate`** (pinned, TEST) | **0.992555** (flip 0.007445) |
| `heuristic_e2e_token_identity_rate` | **0.992708** (flip 0.007292) |
| `pinned_restores_identity_vs_heuristic` (GREEN bar) | **False** |
| `residual_flip_rate_pinned` | **0.007445** |
| `heuristic_flip_reproduces_362` (vs 0.0052) | **True** (within ±0.004 band) |
| `residual_divergence_locus` | **int4 Marlin weight GEMM** (size_m-variant split-K; custom CUDA op outside the `VLLM_BATCH_INVARIANT` aten override; first divergent `size_m=64`, M-variant at the e2e width 2048) |
| **VERDICT** | **RED** (locus isolated) |

Self-test breakdown (PRIMARY, all True): `heuristic_controls_eq_1`, `pinned_controls_eq_1`, `pin_engaged_aten_mm_bitexact`, `pin_attn_is_batch_invariant_flag`, `heuristic_arith_consistent`, `pinned_arith_consistent`. The self-test gates **harness calibration** (controls clean, pin live, arithmetic consistent), not the science outcome — RED is the scientific finding, and it is trustworthy because the calibration passed.

### Controls + pin-engaged positive control (both arms, 32,640 positions)

| control | heuristic | pinned |
|---|---|---|
| `determinism_M1_vs_M1` | 1.000000 ✅ | 1.000000 ✅ |
| `determinism_M8_vs_M8` | 1.000000 ✅ | 1.000000 ✅ |
| `within_batch_copy0_vs_copy1` | 1.000000 ✅ | 1.000000 ✅ |
| **`aten_mm` bit-exact M1-vs-M8** (pin probe) | **False** (max_abs_diff 1.0) | **True** (max_abs_diff 0.0) |
| `attn_is_batch_invariant` flag | False | **True** |

The within-dtype within-batch determinism holds at exactly 1.0 over all 32,640 positions in **both** arms, so the M1-vs-M8 signal is clean (no run-to-run nondeterminism). The `aten_mm` probe is the decisive positive control: the pin **flips the aten path from M-variant (max_abs_diff 1.0) to bit-exact (0.0)** and forces `attn_is_batch_invariant=True` — yet the e2e identity does **not** move (0.992708 → 0.992555). That is the elimination proof that the residual is not attention/aten/lm_head.

### The decider: Marlin `size_m` sweep (resolves #232 vs #122/#19)

Identical in both arms (the override does not patch the custom Marlin CUDA op), row-0 bit-exactness of all four body GEMMs (`qkv_proj`/`o_proj`/`gate_up_proj`/`down_proj`) vs `size_m=1`:

| `size_m` | bit-exact (row 0 vs M=1) | max_abs_diff | regime |
|---|---|---|---|
| 1 | True | 0.0 | trivial |
| **8** (decode-verify width) | **True** | 0.0 | **#232 was right here** |
| **64** (first divergent) | **False** | 7.81e-3 | crossover |
| 256 | False | 1.95e-3 | M-variant |
| **2048** (e2e prefill-repl. width = 8×seq_len) | **False** | 1.56e-2 | **#122/#19 were right here** |

The int4-Marlin split-K geometry is chosen internally as a function of `size_m`, so it is genuinely bit-exact at the small decode width (8 rows) **and** genuinely M-variant at the prefill-replication width (≥64 rows). #232's decode-width bit-exactness (residual attributed to bf16 lm_head) and #122/#19's "Marlin is the M-variant residual" are **both correct at different `size_m`** — this sweep is the missing piece that reconciles them. The deployed e2e prefill-replication geometry sits deep in the M-variant regime, so that is where my RED headline is measured.

### Decision-critical for the composite (fern #357) + convergence with wirbel #375

This RED is **not merely a failed pin — it relocates the eta-axis identity base to a kernel rebuild.** Because the residual is the int4-Marlin body GEMM (a custom CUDA op *outside* the aten dispatcher that `VLLM_BATCH_INVARIANT` structurally cannot patch), the `(real-weight identity = 1.0)` factor in fern #357's composite is **NOT reachable via any `VLLM_*` env knob on the deployed Marlin kernel** — it is **Marlin-kernel-rebuild-gated**, a *second* kernel-rebuild line item alongside #375's `mha_varlen` rebuild.

**Two independent angles, one answer (cite #375, merged).** wirbel #375 established from the *kernel-export* side that the #365 synthetic `num_splits=8` pin lives on `mha_fwd_kvcache` — a kernel the served vLLM V1 varlen decode *never calls* — and that the only M-invariant served attention config is `VLLM_BATCH_INVARIANT=1`→`num_splits=1` (un-packed). This leg confirms the SAME conclusion from the *real-stack* side: Gemma-4-E4B's heterogeneous head dims (256/512) force TRITON_ATTN (server log: "Forcing TRITON_ATTN backend"), so flash_attn's `num_splits` knob does not even apply, and the faithful pin is `VLLM_BATCH_INVARIANT=1`→`num_splits=1`. #375 pins the *attention* rebuild; this leg adds the *body-GEMM* rebuild as the binding one for end-to-end identity.

**Minimal Marlin change that WOULD pin it (to size against #375's rebuild).** The deployed `ops.marlin_gemm(..., size_m=reshaped_x.shape[0], ...)` chooses its split-K (K-axis reduction) partition *internally as a function of `size_m`* and exposes **no `num_splits` / `max_par` knob** (#122 mechanism analysis). The minimal change to reach strict M-invariance is a **fixed-split-K int4 Marlin/Machete GEMM**: force the K-axis partition count to be `size_m`-independent (a single deterministic split, or a fixed split count pinned across M) so that M=1 and M=8×seq_len reduce K in the *same* float order. On A10G (SM8.6, bf16) the Marlin reduction is *already* non-atomic/deterministic (`should_use_atomic_add_reduce=False`, #122), so this is purely a split-count-selection change, **not** an accumulation-mode change — a contained kernel edit, but a kernel **rebuild** nonetheless (not an env flag). In kind it is comparable to #375's `mha_varlen` rebuild; both are deployed-kernel rebuilds, neither reachable by `VLLM_*` env.

**Verdict for the composite:** GREEN (pinned=1.0) would have closed the eta-axis end-to-end on real weights via env alone; this **RED-on-Marlin** instead reframes the eta-axis identity base as **kernel-rebuild-gated** — load-bearing input for #357's composite (the `identity=1.0` factor now carries a Marlin-body-GEMM-rebuild dependency, in addition to #375's attention-rebuild dependency).

### Geometry — what the headline measures, and the honest caveat

- **Headline (prefill-replication, the #221/#232 geometry):** M=8 = 8 identical co-batched prefill replicas → GEMM M-dim = 8×seq_len = 2048. This is the only M=1-vs-M=8 geometry tractable through vLLM's high-level `prompt_logprobs` API, and it is apples-to-apples with #232. Here the Marlin GEMM is M-variant → pinning everything-else does not reach 1.0 → **RED, locus = Marlin**.
- **The literal decode-verify width (the #362/#365 attention-split mechanism, 8 query rows vs KV cache):** the Marlin sweep shows the body GEMM is **bit-exact at width 8**. So at *that* geometry the Marlin residual vanishes, and with attention (`num_splits=1`) + the bf16 lm_head both batch-invariant in the pinned arm, the remaining divergence sources are removed — **strongly indicating** the decode-width pinned identity would reach (or approach) 1.0. I did **not** directly e2e-measure the pure decode width (vLLM's high-level path does prefill scoring, not an 8-row decode against a cache), so this is an inference from the diagnostics, not a measured number. **This is the key follow-up below.**

So the card's question splits by geometry: at the prefill-replication width the answer is RED (Marlin-dominated); at the decode-verify width the diagnostics indicate the attention-split pin is sufficient on the GEMM side but this needs a direct decode-width e2e measurement to confirm.

### Cross to the fleet anchors

| anchor | value | this leg |
|---|---|---|
| #232 `nxwv6pam` identity (prefill-repl.) | 0.992708 | heuristic **0.992708** (delta **+0.000000**, exact) |
| #232 strict-A pass fraction | 0.164062 | both arms **0.164062** (exact) |
| #362 `5k3px8p1` deployed flip | 0.0052 | heuristic flip 0.007292 (delta +0.0021; same ~0.5–1% regime, but #362 is the *decode-verify* geometry) |
| #122 `n5bypf5h` Marlin-locus claim | Marlin is the M-variant residual | **confirmed at the clean per-token level** (not the 56% spec-decode artifact) |
| #19 batch-invariant scope | covers attention/aten/lm_head, not Marlin | **confirmed** (aten probe False→True, identity unchanged) |
| #375 (wirbel, merged) kernel-export side | served pin = `VLLM_BATCH_INVARIANT`→`num_splits=1`; `num_splits=8` lives on `mha_fwd_kvcache` (never called) | **converges from the real-stack side** (TRITON_ATTN forced; flash_attn `num_splits` N/A) — and this leg adds the *body-GEMM* rebuild beside #375's `mha_varlen` |

The heuristic arm **reproduces #232 exactly** (same geometry). My result is the clean-per-token confirmation of #122's mechanism: `VLLM_BATCH_INVARIANT` pins attention+aten+lm_head, the residual ~0.73% flip is entirely the Marlin GEMM and is left **unchanged** (it merely reshuffled a handful of near-tie flips, consistent with #122's Jaccard 0.829) — but now isolated at the per-token level, free of the #114/#122 spec-decode-trajectory 56% artifact.

### Comparison against baseline

- **Official baseline 481.53 TPS** (PPL 2.3772 served, 128/128; PR #52). **This leg adds 0 TPS — it is a LOCAL correctness measurement, not a speed change.** The served int4 path is READ, never modified. Greedy/PPL untouched. **0 HF Job / 0 submission / 0 served-file change / 0 official TPS draw / no `train.py --launch`.**

### Exact command

```bash
cd target/ && .venv/bin/python research/validity/realweight_e2e_token_identity/realweight_e2e_token_identity.py \
  --gpu --real-lmhead --real-int4-body \
  --proxy google/gemma-4-E4B-it-qat-w4a16-ct \
  --eval-prompts 128 --pin-splits 8 \
  --wandb_group strict-bi-verify-gemm --wandb_name stark/realweight-e2e-identity
```

(Smoke pre-flight: same script `--smoke --no-wandb` — 6 prompts, both arms, self-test passed, RED, Marlin first-divergent `size_m=64` — before the full run. Each arm runs as an isolated subprocess under `CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0`; the `pinned` arm additionally gets `VLLM_BATCH_INVARIANT=1`.)

### Run facts

- **Peak GPU memory:** 12.25 GB (gpu_mem_util 0.55 on the single A10G; comfortable headroom). Both arms ~12.2 GB.
- **W&B run ID:** `ipe3ofie` (`wandb-applied-ai-team/gemma-challenge-senpai`, group `strict-bi-verify-gemm`).
- **Model:** `google/gemma-4-E4B-it-qat-w4a16-ct` (compressed-tensors int4-Marlin w4a16 body, bf16 tied lm_head) — the canonical Hub checkpoint #221/#232 probed; 128 prompts × max_len 256 = 32,640 positions/arm, M_verify = K_spec(7)+1 = 8.
- **Pin realisation:** Gemma-4 forces **TRITON_ATTN** (heterogeneous head dims 256/512 → flash_attn cannot run; confirmed in the server log "Forcing TRITON_ATTN backend"), so flash_attn's `num_splits` knob does not apply. The faithful real-stack pin is `VLLM_BATCH_INVARIANT=1` (attention `num_splits=1` single-segment, M-independent, + batch-invariant aten/lm_head/norm). `--pin-splits 8` is recorded for parity with the #365 skeleton.

### What happened — honest analysis

**It worked, and it is RED — pinning the attention split does not restore strict M-invariance on the deployed int4 stack, and I isolated exactly why.** The pin engaged cleanly (the `aten_mm` probe went from M-variant max_abs_diff 1.0 to bit-exact 0.0, and `attn_is_batch_invariant=True`), all determinism controls held at exactly 1.0 in both arms, yet the e2e identity was unchanged (0.992708 → 0.992555). With attention, the aten matmuls, and the bf16 tied lm_head all provably pinned, the only forward op the override cannot reach — the int4-Marlin body GEMM — is the residual by elimination, and the `size_m` sweep confirms it is M-variant at the prefill-replication width (first divergent `size_m=64`, max_abs_diff 1.56e-2 at the e2e width 2048). The same sweep shows it is bit-exact at the decode width 8, which is the missing reconciliation between #232 (decode-width bit-exact → lm_head residual) and #122/#19 (Marlin is the residual): both are right at different `size_m`.

**Honest band:** (a) the headline is the **prefill-replication geometry** (8×seq_len GEMM M-dim) — the only M=1-vs-M=8 geometry tractable through vLLM's high-level API, apples-to-apples with #232; the literal decode-verify attention width (8 rows) is covered by the in-process Marlin `size_m` diagnostic + the documented `num_splits=1` override, **not** a separate decode kernel. (b) Because the Marlin GEMM is **bit-exact at the decode width 8**, the decode-verify geometry is *not* Marlin-limited — the attention-split pin would plausibly restore identity there — but I did **not** directly e2e-measure the pure decode width, so I do not claim decode-width GREEN. (c) The pin is `VLLM_BATCH_INVARIANT` (pins attention+aten+lm_head), which is *more* than just the attention split; the elimination argument uses that breadth, and the aten probe proves it is live. (d) LOCAL probe on the canonical Hub int4 checkpoint; the served path is READ, never modified; the deployed lmhead12k differs in row count but shares the bf16 dtype, and since the residual is the Marlin *body* (not the lm_head) the qualitative RED carries. (e) Anchors (#232 0.992708, #362 0.0052, #122, #19) imported unchanged.

### Hand-off (one sentence)

> *On real weights, pinning the attention split (`VLLM_BATCH_INVARIANT`, the faithful real-stack `num_splits=1`) leaves the M=8-vs-M=1 e2e identity at **0.992555** (unchanged from heuristic 0.992708) — RED — because the residual ~0.73% flip is the int4-Marlin body GEMM (M-variant at `size_m≥64`, bit-exact at the decode width 8), the one op the override cannot reach; this settles the #232-vs-#122 split (both right at different `size_m`) and means a strict greedy-identity GREEN with the drafter ON requires a fixed-split-K Marlin kernel, not a batch-invariant override.*

### Suggested follow-ups

- **Direct decode-width e2e identity (the cleanest close of the card):** measure the pinned vs heuristic identity at the *literal* 8-query-row decode-verify width (a custom decode loop / spec-verify hook against a real KV cache, avoiding spec-trajectory branching). The Marlin sweep predicts the pinned decode-width identity reaches ~1.0 (Marlin bit-exact at 8); a direct measurement would convert that inference into a GREEN/RED at the exact #362 geometry.
- **Fixed-split-K int4 Marlin kernel (decision-critical for #357):** the only path to a strict-identity GREEN with the drafter ON is a Marlin/Machete GEMM with `size_m`-independent split-K (not in the pinned wheel). It is the binding `identity=1.0` dependency in fern #357's composite and should be sized as a *second* deployed-kernel rebuild alongside wirbel #375's `mha_varlen` rebuild — on A10G it is a split-count-selection edit only (the reduction is already deterministic, #122), not an accumulation-mode change.
- **lm_head ablation under the pin:** force an fp32 / batch-invariant lm_head and re-confirm the residual does not move (it should not — the locus is the body GEMM), pinning the margin-gate supply cap to exactly the Marlin `size_m`-variance.
