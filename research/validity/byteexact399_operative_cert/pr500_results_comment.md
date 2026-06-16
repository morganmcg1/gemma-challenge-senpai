STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["m76qbs3l","feof8wtk","rvl5w50z"],"primary_metric":{"name":"served_warm_tps_128x512","value":444.82},"test_metric":{"name":"ppl","value":2.3766643358900286}}

## Results — Package + serve-certify the 399.75 byte-exact split-KV (LOCAL, analysis_only, NO launch)

All four deliverables landed. Headline: the **attention kernel is byte-exact (#496 confirmed, not refuted)** and the candidate is fast + quality-safe + served-self-deterministic — **but the end-to-end operative census is LOOSER than surgical-357 (5 ULP near-ties vs 1)**, and the residual flips are **matmul-induced, not attention-induced**. Honest verdict below; I did not paper over the gap.

### (1) Package — `submissions/fa2sw_strict_byteexact_splitkv399/` ✅
Byte-faithful copy of `fa2sw_precache_kenyan`, differing only by the fixed-order split-KV lever:
- `model_id=google/gemma-4-E4B-it`, weights/drafter/keepset/`SPECULATIVE_CONFIG` (MTP K=7) **identical**.
- Lever baked in manifest `env`: `BYTEEXACT_FIXED_TPS=4`, `BYTEEXACT_NUM_SEGMENTS=64`.
- Armed on a **stock wheel** via `sitecustomize.py` → env-gated `import byteexact_splitkv_patch` → `if ENABLED: _arm()` (meta-path finder pre-vLLM / in-memory re-jit post-vLLM). **No installed-wheel file edit** (same discipline as stark's surgical-357). `VLLM_BATCH_INVARIANT` is NOT set → fast Marlin matmul kept.

### (2) Harness fix (#496 follow-up #4) ✅ — the advisor's proposed fix is a no-op; the real confound was elsewhere
- **"Drive the M=1-AR ref through cudagraph not eager" is a NO-OP.** `serve.py::disable_speculation_for_reference_mode()` only clears `SPECULATIVE_CONFIG`; there is no `enforce_eager` anywhere → the ref **already** captures+replays a width-1 cudagraph (confirmed by static read + both `server.logs`).
- **`full_flag` is the WRONG control** (can't ever read ~1.0): `VLLM_BATCH_INVARIANT=1` patches only the matmul family + NCCL/cuBLAS/TF32 env — **never the attention kernel** — so `full_flag`'s Gemma global attn stays the stock adaptive-3D split-KV = M-dependent on the very axis under test.
- **The actual fix:** bump `--ref-decodes` 2→3 so the ref's **warm** (r1-r2) self-determinism is measurable. The 2-round default only ever measured cold(r0)-vs-warm(r1), mislabeling cudagraph/cache warmup as ref non-determinism.
- **Validated on the byte-exact-by-construction arms** (the correct controls): `byteexact` **warm r1-r2 = 1.000000** (0/128 flipped) AND `byteexact_ref` **warm r1-r2 = 1.000000**. The cold r0-r1 = 0.6212 is exactly the confound. (The free-running greedy M=8-vs-M=1AR rate of 0.3977 in the 128×512 run is the cascade-amplified noise metric — non-certifying by construction.)

### (3) Operative-1.0 certification — reported exactly, including the gap
**Part 1 — served self-determinism (warm r1-r2):** `byteexact` = **1.0**, `byteexact_ref` = **1.0**. ✅

**Part 2 — margin census** (#461 logit-margin locus: teacher-forced M=8 `prompt_logprobs` vs M=1 AR, 127 prompts × ctx 224, the **same methodology surgical-357 was certified on**), wandb `rvl5w50z`:

| arm | identity | flips | ULP near-tie | **semantic** | max margin |
|-----|----------|-------|------|----------|-----------|
| deployed (adaptive 3D) | 0.99663 | 3 | 3 | **0** | 0.125 nat |
| **byteexact (fixed 3D, candidate)** | 0.99438 | **5** | 5 | **0** | 0.25 nat |
| surgical attn_only (2D in-order) | 0.99888 | 1 | 1 | **0** | 0.125 nat |

`byteexact` has **0 semantic flips** (quality-safe) but **5 ULP near-tie flips** — more than deployed's 3 and 5× surgical's 1. It closes deployed's 1 *attention*-induced flip (`mmlu_pro-00996c6808`) but opens 3 new ties (all `gpqa_diamond`). So it does **not** match surgical-357's census tightness.

**Reconciliation with #496 (this is the important part — #496 is CONFIRMED, not refuted):** the raw-attention-output-byte microbench (`verify_packaged_patch.py`, `torch.equal` int16, M=8 row-i vs M=1 AR same abs pos) at the **exact census positions** reads **0/8 flips, max_abs_err 0.0** for the fixed scheme — at base 224 (census chunk), 250 (straddle256), 192 (seg-start), 100 (control). The byteexact **attention kernel is byte-exact**, including the partial-last-segment region the census exercises. So the 5 census *token* flips are **not** attention-induced — they come from the **un-taxed fast Marlin matmul** (M-dependent; byteexact deliberately keeps it for the +42 TPS, exactly as surgical's `attn_only` does — `aten_mm_bitexact_M1_vs_M8=false` in both). Evidence chain: (a) microbench attn byte-exact 0/8; (b) `attn_only` (M-inv 2D attn + un-taxed matmul) **still** has 1 flip → matmul contributes; (c) byteexact's particular 3D attention output lands on more of the M-dependent matmul's knife-edge ties (5) than surgical's 2D output (1). The adaptive contrast confirms the lever still matters where it should: adaptive is byte-exact at 224 (0/8) but breaks 6/8 at the 256-straddle.

### (4) Fire-time recert dry-run (LOCAL, analysis_only, NO `--launch`) ✅
- **128×512 fire-time** (wandb `m76qbs3l`): **444.82 warm TPS**, **PPL 2.37666**, **128/128**, peak 19535 MiB, armed + onegraph captured, 0 fatal tracebacks, `[byteexact] fixed split-KV armed: tiles_per_segment=4 num_par_softmax_segments=64`.
- **Matched 32×256 recert vs the #496 399.75 proxy** (wandb `feof8wtk`): **399.97 warm TPS** — inside the σ_hw band [394.89, 404.61], **0.046σ** from 399.75. (The deliverable's "within σ_hw of 399.75 AND 128/128" is internally inconsistent because 399.75 was a 32×256 proxy; I report both the matched-workload recert AND the 128/128 fire-time.)
- PPL gate 2.42: **2.37666 PASS**.

### Comparison vs baseline
| | TPS | PPL | completed | census flips (semantic) |
|---|---|---|---|---|
| surgical-357 (the rung this targets) | 357.6 | 2.3767 | 128/128 | 1 ULP (0 sem) |
| **byteexact (this card)** | **399.97 / 444.82** | **2.37666** | **32/32 · 128/128** | **5 ULP (0 sem)** |
| deployed (non-strict ref) | 481.53 pub | 2.3772 | 128/128 | 3 ULP (0 sem) |

### Exact commands
```bash
# (2)+(4) serve + fixed harness (ref-decodes 3):
.venv/bin/python -m research.speed.byteexact_attn.run_byteexact_serve --arms deployed,byteexact,byteexact_ref \
  --fixed-tps 4 --num-segments 64 --num-prompts 128 --output-len 512 --ref-decodes 3 \
  --wandb_name lawine/byteexact399-firetime --wandb_group byteexact-splitkv399-package
# (3) margin census (same #461 locus as surgical-357):
.venv/bin/python -m research.validity.byteexact399_operative_cert.run_byteexact_census \
  --n-prompts 128 --arms deployed,byteexact,attn_only \
  --wandb_name lawine/byteexact399-flip-census --wandb_group byteexact-splitkv399-package
# attention-byte reconciliation at census positions:
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/speed/byteexact_attn/verify_packaged_patch.py --mode fixed \
  --bases "224:census_chunk,250:straddle256,192:seg3_start,100:control"
```

### Peak memory
19535 MiB (speed arm), 19349 MiB (ref arm), 12.3 GB (census arm). A10G 23 GB.

### Pristine restore (verify 0 markers) ✅
Both venvs' `triton_unified_attention.py` keep the adaptive `cdiv_fn` (lines 302/681), no baked `tiles_per_segment = 4`; serve-venv backend `NUM_PAR_SOFTMAX_SEGMENTS = 16`. The package re-jits **in-memory only** (inspect+exec) — it never edits installed-wheel files, so 0 markers by construction.

### Public evidence used
`#496` (`42qroec1`, fixed-order split-KV 399.75 / 0-8 kernel flips), surgical-357 (`#488 ko01dcyy` / stark `#494`), deployed (`#52 2x9fm2zx`). Local harness `summary.json` attached at `research/validity/byteexact399_operative_cert/summary.json`.

### What happened — honest analysis
- The **package is correct** and the **attention byte-exactness claim of #496 holds** (microbench 0/8 at the census positions, exact-zero error). The candidate is **fast** (+42 TPS over surgical-357, recert 0.046σ from #496) and **quality-safe** (0 semantic flips, PPL identical 2.37666), and **served-self-deterministic** (warm r1-r2 = 1.0).
- **But it is NOT operative-1.0 at surgical-357's strict census standard.** The end-to-end #461 census has **5 ULP near-tie flips vs surgical's 1**. Crucially these are **matmul-induced** (the un-taxed fast Marlin path byteexact keeps for speed), not attention-induced — byteexact's byte-exact 3D attention output simply lands on more of the M-dependent matmul's bf16 knife-edges than surgical's 2D output. To make the end-to-end token census as tight as surgical (≤1 flip), byteexact would *also* need the ~48% matmul tax — which defeats the entire purpose of the lever.
- Net: this is a **faster, byte-exact-attention, quality-safe** draw candidate, but it trades end-to-end census tightness for the +42 TPS. The 5 residuals are all sub-0.25-nat near-ties (no meaning change), so it is a legitimate *quality-equivalent* candidate — just not a *byte-identical-token* one at surgical's bar.

### Suggested follow-ups
1. **Definitive matmul-causation proof:** rerun the census `byteexact` arm with `VLLM_BATCH_INVARIANT=1` (byteexact attn + matmul tax). Expectation: flips → 0, confirming the 5 are 100% matmul-induced. (Cheap, ~5 min; I did not run it because it tests a config byteexact doesn't ship.)
2. **If the human wants a strictly byte-identical-token rung at >357 TPS:** the only route is byteexact-attn + a *cheaper* M-invariant matmul than the full batch-invariant tax (e.g. a fixed-split-K Marlin), since the residual is entirely the matmul. Worth a scoping card.
3. **Decision for the advisor:** is "byte-exact attention + 0 semantic + served r1-r2=1.0, but 5 ULP near-ties e2e" an acceptable draw standard, or must a draw candidate match surgical's ≤1-flip token census? That gates whether this candidate is human-approvable as-is.

Per the HARD RULE this card is analysis_only (official_tps=0) — **no HF job, no `--launch`, no submission**. The candidate is staged and certified locally; surfacing for human approval is the advisor's call once surgical-357 lands.
