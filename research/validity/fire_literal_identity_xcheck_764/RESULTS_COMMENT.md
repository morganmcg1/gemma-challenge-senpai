STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["ki2b2r4g"],"primary_metric":{"name":"fire_literal_greedy_identity","value":0.2031},"test_metric":{"name":"xcheck_consistent_with_751","value":1}}

## Results

**VERDICT: `XCHECK_REPRODUCES_751_NOT_BYTEEXACT` — `xcheck_consistent_with_751 = 1`.** On a **second, independent harness** (my own merged #748 machinery + the merged fire submission `submissions/int4_mtp_batchinv`, BI=1, MTP drafter), the full fire config's literal served-greedy identity is **26/128 byte-exact (frac 0.2031)** vs my independent served spec-off M=1 AR reference. That is **within ±10/128 of wirbel #751's 20/128 (0.156)** (|26−20| = 6 ≤ 10; consistency band [10,30]). **The fire is self-consistent-gate + PPL-clean but NOT literal-byte-exact vs an independent AR reference — independently corroborated.** A skeptical reviewer's first question ("did anyone check that 108/128 on a second harness?") now has a YES.

The honest direction of the small gap: **my harness sees slightly *more* identity than wirbel's (26 vs 20), not less** — but in the **same regime** (far from both 128/128 byte-exactness and 0), and all four qualitative facts reproduce. A 6/128 difference is ≈1.4σ of binomial sampling noise at p≈0.18 (σ≈4.3 prompts) on top of genuine harness-dependence (different reference constructions) — exactly the kind of within-tolerance agreement that *hardens* the board claim rather than a refutation.

### 1. Primary identity (the load-bearing measurement)

| comparison | identical | per-token flip hazard | meaning |
|---|---|---|---|
| **fire spec-ON (BI=1, MTP num_spec=6) vs my spec-OFF M=1 AR ref** (PRIMARY / strict-#319) | **26/128** (0.2031) | **0.368 %/tok** | the fire config is NOT literally byte-exact |
| **DETERMINISM FLOOR** (spec-OFF AR run A vs run B, N=32) | **32/32** (1.000) | 0.000 %/tok | the M=1 AR stack is bit-reproducible within-config |

- Both arms boot the **fire submission's OWN `serve.py`** (verified unmodified — `git diff` clean). The **only** between-arm difference is the MTP spec path: `SENPAI_REFERENCE_MODE=1` (the fire's own documented reference contract) forces `num_speculative_tokens=0`, drafter OFF. BI=1, target model, kernels, attention backend (TRITON_ATTN), CUDA graphs — all held identical. So a divergent prompt is attributable to the **MTP batched-verify path**, not cross-engine noise.
- **BI=1 engagement confirmed live** in both arms (`batch_invariant.py:913` custom kernels register; `matmul_persistent` traced). **MTP drafter confirmed loaded only in spec-ON** (`SpeculativeConfig(method='mtp', model='…q4_0-unquantized-assistant', num_spec_tokens=6)`, "Detected MTP model"; 1 drafter-load in spec-ON vs **0** in the reference). This is denken's exact 128/128-self-consistent-gate config.
- The **determinism floor = 32/32** is the control that makes 26/128 interpretable: the served greedy stack is bit-reproducible when nothing changes, so the 26/128 is a **real, reproducible per-step reduction-order divergence on the verify path**, not run-to-run noise. (This matches #751's AR-vs-AR control "0/32 divergent".)

### 2. xcheck reconciliation vs wirbel #751

| quantity | mine (independent harness) | wirbel #751 (relayed FACT) | consistent? |
|---|---|---|---|
| literal greedy identity | **26/128 = 0.2031** | 20/128 = 0.156 | ✅ \|Δ\|=6 ≤ tol 10 |
| fraction diverging at all | 0.797 (102/128) | ~0.84 (108/128) | ✅ same regime |
| AR-vs-AR determinism | 32/32 identical (0 diverge) | 0/32 diverge | ✅ identical conclusion |

`xcheck_consistent_with_751 = 1`. Tolerance pre-registered at ±10/128 in the PR.

### 3. Divergence distribution (mechanism evidence for #751's int4 near-tie cascade, NOT attention)

Per-prompt **first-divergence position** over the 102 diverging prompts:

| token-pos bin | 0 | 1–3 | 4–15 | 16–63 | 64–127 | 128–255 | 256–512 |
|---|---|---|---|---|---|---|---|
| # prompts | 1 | 4 | 7 | 24 | 22 | 26 | 18 |

- **min / median / max first-divergence = 0 / 98 / 509**; **88.2 %** of diverging prompts first flip **after token 16**. The first flip is **SPREAD across the entire 512-token rollout, NOT root-clustered** — the signature of **per-token near-tie argmax flips compounding**, exactly the mechanism wirbel #751 named (int4 quant-grid near-ties), and **not** a single deterministic attention-divergence point. (Consistent with land #680: the int4 Marlin GEMM is byte-identical across M, so the verify break is reduction-order at the near-tie margin, not the matmul.)
- **Per-token flip hazard 0.368 %/tok** (constant-hazard MLE, first-divergence as event + full-length survival as right-censoring). A 512-token greedy rollout is fragile to *any* single near-tie flip, so a sub-0.4 %/tok rate still compounds to ~20 % full-completion identity. (Measured 0.203 sits a touch above the naive `(1−h)^512 ≈ 0.151` because hazard is heterogeneous across prompts — the survivors are the low-near-tie prompts; I don't over-rely on the closed form.)

### 4. Reconciliation — this does NOT overturn denken's gate, and is NOT a DQ

- My **literal-identity-vs-independent-AR** number is a **different notion of strict** than denken's **128/128 self-consistent** gate. denken's gate compares the spec loop against a reference whose construction may differ from mine; **both can be true at once.** This corroborates the #747/#752/#751 framing — it does **not** refute denken's gate.
- **NOT a DQ.** The organizer's scorer is token-identity-blind; the fire passes **PPL 2.0057 / completion 128/128 / all modalities** regardless. This card changes only the **honesty wording** of the board post (self-consistent + PPL-clean + official-gate-PASS, but *not* literal-byte-exact vs an independent AR reference), now backed by **two independent harnesses**.

### 5. TPS deliverables (LOCAL probe — NON-transferable, `official_tps=0`)

| arm | output_tps (local) | peak mem |
|---|---|---|
| fire spec-ON (BI=1, MTP verify) | 156.59 | 19677 MiB |
| spec-OFF M=1 AR reference | 77.97 | 19313 MiB |
| spec-OFF AR run B (N=32) | 78.22 | 19313 MiB |

**These are raw full-vocab `api_server` LOCAL probes** (no deployed pruned-16k lm_head / fa2sw / precache / onegraph) and **do NOT transfer to official TPS** — the local 156.59 is a coincidence near fern #750's 156.95 *official* anchor and must not be read as a TPS result. Only the **identity / divergence-distribution / determinism** results transfer. The locked `int4_g128_lmhead`@126.378 baseline is untouched.

### Command

```bash
# 3 served arms (resumable), each = one live vLLM 0.22.0 api_server booted through the fire
# submission's OWN serve.py; BI=1; greedy temp=0 (#319, NOT generation_config.json); 128 ShareGPT
# prompts (seed 1), 512 tok, ignore_eos. spec_off_ref/repB add SENPAI_REFERENCE_MODE=1 (drafter OFF).
cd target/ && research/validity/fire_literal_identity_xcheck_764/run_all.sh
research/validity/fire_literal_identity_xcheck_764/analyze_xcheck.py
python3 research/validity/fire_literal_identity_xcheck_764/wandb_log.py
```

- **Engine + sampling:** vLLM 0.22.0 v1 `api_server` (online, CUDA graphs), `AttentionBackendEnum.TRITON_ATTN` (vLLM auto-forces for Gemma4's heterogeneous head dims), **greedy τ=0** — the strict-#319 identity protocol, NOT a downstream-quality eval, so it does **not** follow `generation_config.json`.
- **Peak GPU mem:** 19677 MiB (spec-ON) on the single A10G.
- **W&B run:** `ki2b2r4g` (group `fire_literal_identity_xcheck`).
- **`analysis_only=1, official_tps=0, no_hf_job=1, fires=0`.** No served-file change, no submission, no `--launch`.

### What happened — honest analysis

The hypothesis is **reproduced**: measured on my own independent #748 harness, the full fire config (`int4_mtp_batchinv`, BI=1, MTP num_spec=6) gives **26/128 literal served-greedy byte-exact identity (0.2031)** vs my independent spec-off M=1 AR reference — **within ±10/128 of wirbel #751's 0.156**, so `xcheck_consistent_with_751=1`. The "self-consistent + PPL-clean but **not** literal-byte-exact vs an independent AR reference" claim no longer rests on one harness; **two independent harnesses agree within tolerance.** The 32/32 determinism floor proves the residual is a real verify-path reduction-order effect (not noise), and the first-divergence positions are **spread across the rollout** (median 98, 88 % after tok 16), corroborating #751's int4 near-tie-cascade mechanism rather than an attention break. My number lands **6/128 above** wirbel's — slightly *more* identity, well inside the pre-registered tolerance and binomial sampling noise — which is the reassuring outcome: no harness-dependence blocker for the board. This **does not** overturn denken's 128/128 self-consistent gate (a different notion of strict, different reference) and is **not a DQ**.

### Suggested follow-ups

- **If the board wants a tighter cross-harness match**, the residual 6/128 gap is almost certainly reference-construction: my reference is `serve.py + SENPAI_REFERENCE_MODE=1` (BI=1 retained, drafter OFF); if wirbel's M=1 AR reference ran with a different BI or split-KV setting, that would shift a handful of marginal near-tie prompts. A one-line reconciliation of the two reference constructions would close it, but it does not change the verdict.
- **A don't-care near-tie band** (cf. land #654's tie-tolerant residual): since the divergence is a benign int4/bf16 near-tie cascade and PPL is neutral, a tie-tolerant identity gate would classify the fire as quality-equivalent without bit-exact reduction alignment — the principled way to state "not literal-byte-exact but quality-identical" to the board.

### Public evidence used

This card cross-validates an **identity measurement** relayed as a FACT from wirbel #751 (`BI_DETAX_SURGICAL_REFUTED`, `o05emuic`: fire BI=1 = 108/128 divergent / 20/128 identical, 0.156) — his branch was **not** read; the reference was reconstructed in-scope from my own merged #748 method (`fikec7di`, `research/validity/strict_clean_served_byteexact_748/`). Anchors: my #760 (`FULL_BI_NECESSARY`, `2rmeroz8`), fern #750 (`cdkvekkn`, full BI=1 strict ≈156.95 official). DQ-risk-free analysis over the locked `int4_g128_lmhead`@126.378 anchor (`905tbujn`); no leaderboard movement.
