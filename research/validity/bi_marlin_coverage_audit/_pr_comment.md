STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["fa1f9vm1"],"primary_metric":{"name":"official_tps","value":0},"test_metric":{"name":"marlin_m8_maxdiff_vs_m1","value":0.0}}

## Results — BI-coverage audit (vLLM 0.22.0, single A10G sm_86, `analysis_only`, NO build / NO HF Job)

**Verdict: `MARLIN_OUTSIDE_BI__REQUIRES_CUSTOM_KERNEL`** — but with a decisive rider that flips the recommended action:

> **BI=1 does NOT cover the int4-Marlin GEMM (proven), AND Marlin does not need covering: it is already bit-exact at the served spec-verify width. The 31048/65536 break wirbel #607 measured is NOT attributable to Marlin M-dependence. Do not build a batch-invariant Marlin kernel — investigate the non-Marlin ops.** (i.e. the *consequence* matches the `BI_ALREADY_COVERS_MARLIN` branch "the break is elsewhere — investigate", even though the *mechanism* is "Marlin is outside BI but coincidentally already M-invariant at M≤16".)

The hypothesis is **half-confirmed, half-refuted**:
- ✅ **Confirmed:** `VLLM_BATCH_INVARIANT=1` covers only BF16 reduction/dense-aten ops; the int4-Marlin GEMM is routed entirely around it.
- ❌ **Refuted:** the premise that "a batch-invariant Marlin kernel is the single missing piece for identity-preserving spec." Marlin is **already batch-invariant (bit-exact) at the verify width** — so a custom BI-Marlin kernel would fix nothing.

### Task 1 — What does `VLLM_BATCH_INVARIANT=1` actually patch?
`enable_batch_invariant_mode()` (`batch_invariant.py:905`) installs a single `torch.library.Library("aten", "IMPL")` and overrides **only ATen ops**:

| Overridden op | Replacement | Scope |
|---|---|---|
| `aten::mm`, `aten::addmm`, `aten::matmul`, `aten::linear` | Triton persistent matmul | **SM80-family branch only** (A10G qualifies, cc 8.6) |
| `aten::bmm` (+ `torch.bmm`) | Triton batched matmul | all CUDA |
| `aten::_log_softmax`, `aten::softmax`, `aten::_softmax` | deterministic softmax | all CUDA |
| `aten::mean.dim` | deterministic mean | all CUDA |

Plus backend flags: fp16/bf16 reduced-precision reduction **OFF**, `preferred_blas_library=cublaslt`, TF32→`ieee`. Attention is made batch-invariant **separately** (`flash_attn.py:1194/1219`: `num_splits=1 if VLLM_BATCH_INVARIANT`), and the Python-level dense-linear hook is wired into **`UnquantizedLinearMethod.apply` only** (`linear.py:226`).

**It touches GEMMs only at the ATen level (dense bf16). It does not register any `torch.ops._C.*` custom-kernel override.**

### Task 2 — Is int4-Marlin among the BI-covered ops? **No.**
Full dispatch trace of the served int4 W4A16 (compressed-tensors, group_size=128 symmetric) path:
```
CompressedTensorsWNA16.apply_weights      (compressed_tensors_wNa16.py:228)  — NO BI check
 └─ MarlinLinearKernel.apply_weights       (kernels/linear/mixed_precision/marlin.py:185) — NO BI check
     └─ apply_gptq_marlin_linear           (marlin_utils.py:510)              — NO BI check
         └─ ops.marlin_gemm  ==  torch.ops._C.marlin_gemm   ← custom _C op, NOT aten
```
- `grep VLLM_BATCH_INVARIANT` over the entire `quantization/` tree (incl. `compressed_tensors/`, `marlin_utils.py`) → **zero matches**. The compressed-tensors WNA16 path never reads the flag.
- `marlin_gemm` lives in the `_C` namespace, so the `aten`-IMPL override **structurally cannot** intercept it.
- On sm_86+bf16, `should_use_atomic_add_reduce()` is hard-**False** for all M (`marlin_utils.py:461`), so even the Python-visible reduce knobs (`use_atomic_add`/`use_fp32_reduce`) don't vary with M — any M-dependence would be *inside* the precompiled CUDA schedule.

**Runtime proof (in-process):** enabling `enable_batch_invariant_mode()` **changed** dense `aten::mm` output bits (maxdiff `1.95e-3` → BI genuinely active on this A10G) while `_C.marlin_gemm` was **bit-identical before vs after** (maxdiff `0.0`). BI is on; Marlin is untouched.

### Task 3 — Is the #607 M-dependence attributable to Marlin? **No — Marlin is bit-exact at the verify width.**
Direct M-sweep of `_C.marlin_gemm` on the four served fused shapes (faithful int4 g128-symmetric synthetic weights via vLLM's own `marlin_quantize`; compare output **row 0** across M to the M=1 result — the canonical batch-invariance test). The M-invariance of the kernel is a property of (M,N,K)+schedule, *not* of weight values, so synthetic weights are faithful here.

| shape | K | N | maxdiff M=7 | maxdiff M=8 | maxdiff M=16 | first divergence |
|---|---|---|---|---|---|---|
| qkv | 2560 | 3072 | **0.0** | **0.0** | 0.0 | none (≤128) |
| o_proj | 2048 | 2560 | **0.0** | **0.0** | 0.0 | none (≤128) |
| gate_up | 2560 | 20480 | **0.0** | **0.0** | 0.0 | **M=32** (~9.8e-4, 1 bf16-ULP) |
| down | 10240 | 2560 | **0.0** | **0.0** | 0.0 | **M=128** (~2.4e-4, 1 bf16-ULP) |

The submission runs `MAX_NUM_SEQS=1` with `NUM_SPECULATIVE_TOKENS=6` (verify width M=7) / K_spec=7 (M=8). **At those widths every served Marlin GEMM is bit-exact to the M=1 AR forward.** Divergence only appears at M≥32 and is a single-bf16-ULP reduction-order artifact. So Marlin contributes **zero** divergence to the spec-verify-vs-AR comparison at the deployed width — it is not even the first domino. (Independently reproduces my prior decode-width finding; threshold is shape-dependent.)

A GEMM M-dependence produces ULP-magnitude diffs that only flip *near-tie* argmaxes (sub-1%); a 31048/65536 = 47% break is categorically larger and, in greedy AR decode, indicates either a structural divergence in the spec path or a single early non-Marlin flip cascading over the 65536-token gate. Either way the trigger is upstream of Marlin.

### Why this verdict and not `EXISTING_PATH_AVAILABLE`
There is no in-wheel **fast byte-exact** batch-invariant int4 path for compressed-tensors WNA16 on sm_86: the only BI-covered GEMM is the dense `aten::mm` triton-persistent path (i.e. dequant-to-bf16 = the ~48% matmul tax the int4 stack exists to avoid). The `awq_marlin`/`scaled_mm-marlin` BI branches are different schemes/numerics (re-quant or fp8, not a free route). So *if* one ever needed a fast BI Marlin it would `REQUIRE_CUSTOM_KERNEL` — but per Task 3 that need does not exist at the verify width.

### Public evidence used
- **wirbel #607** (`yuvztndu`): the 31048/65536 spec-break on the clean 0.22.0 gate — the motivating measurement whose attributed *mechanism* (Marlin M-dependence) this card tests and refutes.
- **stark #613** (`eqvdyntw`): confirmed dense `marlin_gemm` present in-wheel and reused its served-shape int4-Marlin scaffolding.
- Submission under audit: `submissions/int4_mtp_batchinv` (`vllm==0.22.0`, `VLLM_BATCH_INVARIANT=1`, `MAX_NUM_SEQS=1`, `NUM_SPECULATIVE_TOKENS=6`).

### Reproduction
```bash
cd target/
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  research/validity/bi_marlin_coverage_audit/bi_marlin_coverage_audit.py
```
- **W&B run:** `fa1f9vm1` (group `bi-marlin-coverage-audit`, `analysis_only=true`, `official_tps=0`).
- **Peak GPU memory:** ~1.5 GiB incl. CUDA context (synthetic int4 weights; **no checkpoint loaded, no kernel build** — disk-neutral, respects the ENOSPC constraint).
- vLLM 0.22.0 (`.venv`), single A10G sm_86.

### What happened
The audit cleanly separates two questions the PR conflated:
1. **Coverage (structural):** BI=1 patches dense ATen ops + attention `num_splits`; the int4-Marlin custom `_C` op is outside it. Confirmed by source trace **and** runtime (Marlin bit-identical before/after BI, aten::mm changed).
2. **Attribution (causal):** Marlin is *already* batch-invariant at the deployed verify width (M≤16 bit-exact). So the #607 break originates in a **non-Marlin** op. The submission's claim that BI=1+MAX_NUM_SEQS=1 makes the verify forward batch-invariant-with-AR is false for Marlin *coverage*, but Marlin is not why identity breaks.

### Suggested follow-ups
- **Localize the real #607 trigger** with the same row-0 bit-exact probe on the *non-Marlin* verify ops at M=8 vs M=1: (a) the served **attention** backend's varlen paged-KV combine (is `num_splits=1` actually taken for this submission's backend, or is a TRITON_ATTN/splitkv-verify 3D redirect bypassing it?), (b) the **MTP drafter** forward, (c) the **sampler/argmax** tie-break. The cascade hypothesis predicts a single early flip in one of these.
- **Do NOT open a custom-kernel build issue** for a batch-invariant Marlin (would be #319-class risk for zero identity gain at the verify width).
- If the advisor wants belt-and-suspenders confirmation on the *real* baked checkpoint, a tiny `/tmp/osoi5-v0-baked` load + the same M=8-vs-M=1 GEMM bit-compare would confirm the synthetic-weight result (expected identical, since M-invariance is schedule- not data-dependent).
