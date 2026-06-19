# PR #722 — 2:4 sparse-int4 feasibility: TERMINAL VERDICT

## ⛔ `SPARSE_INT4_KERNEL_ABSENT` — the lever is closed on sm_86

**There is no servable 2:4-structured-sparse INT4 (W4A16) path on A10G (sm_86) in
vLLM 0.22.0 — the validated challenge engine.** Both candidate kernels named in
the PR (Sparse-Marlin `gptq_marlin_24`, and compressed-tensors
`CompressedTensors24` / `sparse-24`+`w4a16`) have been **removed upstream**. The
screen terminates at step 1 (servability), exactly as the PR scoped
("no sparse-int4 kernel … → report `SPARSE_INT4_KERNEL_ABSENT` and stop").
Nothing fired; locked `int4_g128_lmhead`@126.378 untouched. `analysis_only=1`,
`no_hf_job=1`, `fires=0`.

W&B run: `w6rfxvdb`
(wandb-applied-ai-team/gemma-challenge-senpai, group `sparse24_int4_722`).

---

## Decisive evidence — three independent direct-source lines (installed vLLM 0.22.0)

Probed the **actual installed engine** (`/senpai-run/home/student-stark/.venvs/vllm022`:
`vllm==0.22.0`, `compressed_tensors==0.15.0.1`, `transformers==5.9.0`),
`torch.cuda.get_device_capability(0) == (8, 6)` (A10G sm_86 confirmed).

### 1. No 2:4 sparse GEMM op compiled into `_C` (robust by-name probe)
The probe is **validated** — it correctly finds the dense int4 kernel the anchor
serves on, and finds every sparse variant absent:

| `torch.ops._C` op | status |
|---|---|
| `marlin_gemm` (dense int4 — anchor's kernel) | **PRESENT** ✅ (probe sanity) |
| `gptq_marlin_repack` (dense) | **PRESENT** ✅ |
| `gptq_marlin_24_gemm` (Sparse-Marlin 2:4) | **ABSENT** ❌ |
| `gptq_marlin_24_repack` | **ABSENT** ❌ |
| `marlin_qqq_gemm` | **ABSENT** ❌ |
| `marlin_24_gemm` | **ABSENT** ❌ |
| `cutlass_sparse_scaled_mm` / `cutlass_scaled_sparse_mm` | **ABSENT** ❌ |
| `cutlass_sparse_compress` / `semi_structured_sparse_mm` | **ABSENT** ❌ |

The only sparse-GEMM machinery bundled in the tree is CUTLASS **headers** for
`sm90` (Hopper) and `sm120` (Blackwell), block-scaled/fp8 — not int4, not sm_86.

### 2. compressed-tensors sparsity path hard-rejected at load
`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`
`_parse_sparsity_config` (lines 261-266) **raises** on any non-empty
`sparsity_config`:

```python
if sparse_scheme_map:
    raise DeprecationWarning(
        "Sparsity support has been removed from compressed-tensors. "
        "Please use a model without sparsity configuration.")
```

So a `sparse-24-bitmask` + `w4a16` checkpoint **cannot even load** — it crashes
at config-parse before any kernel selection.

### 3. No sparse scheme / no sparse quant-method in the registries
- Scheme dir `compressed_tensors/schemes/` has only dense
  `…_w4a16_nvfp4 / _w4a8_* / _w8a8_* / _wNa16` — **no `compressed_tensors_24`**.
- `vllm … QUANTIZATION_METHODS` (30 methods: gptq_marlin, awq_marlin, moe_wna16,
  fp8, …) has **zero** entries containing `24` / `sparse`.
- `compressed_tensors 0.15.0.1` still *names* `marlin_24` / `sparse_24_bitmask`
  in its `CompressionFormat` enum, but ships **zero sparse-compressor
  implementations** (`compressors/**/*sparse*` → empty) — a dead format string.

---

## Corroboration

**Upstream-PR timeline (literature research agent):**
- `gptq_marlin_24` / `marlin_24`: deprecated vLLM v0.14, **removed v0.15**
  (PR #32688, ~Jan 2026; rationale: maintainer burden, ~zero downloads).
- `CompressedTensors24` / Sparse24: **fully excised** (PR #36799, ~Mar 2026) —
  kernel, integration, tests, build configs.
- `llmcompressor` removed Sparse24 in v0.11.0 (last builder ≤ v0.10.0.2).
- sm_86 is **hardware-capable** (sparse tensor cores need cc ≥ 8.0; A10G = 8.6).
  **The blocker is software (removed kernels), not silicon.**

**Independent public confirmation (challenge board):** openevolve's dead-lever
map (`message_board/20260616-062754-273_openevolve.md`) — scanning the same A10G
substrate — states verbatim: *"sub-int4 has no 2/3-bit kernel on A10G (marlin
4/8 only, machete=Hopper sm90, **no 2:4 sparse**)."* A second team reached the
same conclusion by measurement.

---

## What the lever WOULD have been worth (theoretical, UNREALIZABLE here)

For context only — none of this is achievable without a serving kernel:
- 2:4 int4 ≈ **2.65 eff bits/wt** (50% zeros + 2-bit/4 metadata) ≈ **−34% bytes**.
- Ideal HBM-bound M=1: 126.378 × 1.34 ≈ **~169 TPS**; Sparse-Marlin's measured
  ~1.2× (A10/Llama-2, batch=1) ≈ **~151 TPS**. Either clears the +10 bar (136.378).
- So the *physics* of the lever is real and high-ceiling — but it is not
  serv­able on this stack.

**Second, independent blocker even if the kernel existed:** building a 2:4
checkpoint needs a SparseGPT calibration pass (~75 GB VRAM for this model
class); A10G = 24 GB → the build is locally infeasible too, and no pre-built
Gemma-4 sparse-int4 checkpoint exists on the Hub.

---

## Why no escape hatch within the challenge envelope
- **Version bump?** Counterproductive — newer vLLM has *less* sparse support
  (removed), not more. Rolling **back** to ≤ v0.14 would restore the kernel but
  break the validated Gemma-4 / transformers-5.9 / Marlin / onegraph stack, and
  the org-credit `/v1/jobs:run` path is locked to the `vllm/vllm-openai` image
  anyway (program.md L246-253) — no custom outer image.
- **Store 2:4-pruned weights in dense int4?** Reads the zeros as ordinary int4 →
  same bytes, same dense Marlin kernel → **zero speedup** (and a quality hit).
  Not a lever.

## Verdict for the program
`SPARSE_INT4_KERNEL_ABSENT`. The one untried physics-valid fewer-bytes M=1 lever
(2:4 structured sparsity) is **software-closed** on sm_86 in the validated
engine — the sparse-int4 kernels were removed upstream and the loader rejects
sparse configs. This is the clean negative the PR anticipated: the AR speed
frontier remains HBM-walled at ~126.378 with no servable fewer-bytes path on
this hardware. (Mandate note moot: a SHIP would need a human ruling, but there
is nothing to ship.)
