# PR #617 — Does `VLLM_BATCH_INVARIANT=1` cover the int4-Marlin verify path?

**stark · group `bi-marlin-coverage-audit` · LOCAL A10G (sm_86) · `analysis_only=true`,
`official_tps=0` · CPU/GPU introspection, NO build, NO training, NO HF Job.**

Same disciplined no-build mode as #613 (`int4_gemm_kernel_config_audit`).

## Question
wirbel #607 (`yuvztndu`): on a clean vLLM **0.22.0** gate the int4+MTP spec-verify path
breaks greedy identity on **31048/65536 tokens** vs a 0/65536 plain-AR floor. The
`int4_mtp_batchinv` submission sets `VLLM_BATCH_INVARIANT=1` + `MAX_NUM_SEQS=1` and claims
that makes the M=8 spec-verify forward batch-invariant with the M=1 AR forward. The break is
the evidence it does NOT — at least for the int4-Marlin GEMM.

**Hypothesis:** `BI=1` covers only BF16 reduction ops (attention / layernorm / logsoftmax /
sampler) and the int4-Marlin GEMM is routed *around* the batch-invariant path entirely
(straight to the custom `torch.ops._C` CUDA kernel). If true, a batch-invariant Marlin kernel
is the missing piece for identity-preserving spec.

## Plan (static dispatch trace on 0.22.0 source)
1. **What does `BI=1` patch?** Locate `vllm/model_executor/layers/batch_invariant.py`,
   enumerate every op it overrides / monkey-patches (`torch.library` registrations, `Linear`
   forward swaps, attention/sampler hooks). Classify: reductions only, or GEMMs too?
2. **Is int4-Marlin among them?** Trace WNA16 forward:
   `compressed_tensors_wNa16.py` → `marlin_utils.apply_gptq_marlin_linear` →
   `torch.ops._C.gptq_marlin_gemm`. Does that call route through a BI override, or straight to
   the un-wrapped custom kernel? State explicitly: is `marlin_gemm` batch-invariant under `BI=1`?
3. **Verdict** (exactly one):
   - `BI_ALREADY_COVERS_MARLIN` — break is elsewhere, investigate.
   - `MARLIN_OUTSIDE_BI__EXISTING_PATH_AVAILABLE` — route-able, low effort.
   - `MARLIN_OUTSIDE_BI__REQUIRES_CUSTOM_KERNEL` — custom-kernel build, out of scope/human-gated.
   - `IMPOSSIBLE_ON_SM86`.

## Anchors / tension to reconcile
- **wirbel #607** (`yuvztndu`): 31048/65536 break on 0.22.0; clean 0/65536 floor; attributes
  mechanism to int4-Marlin M-dependence.
- **#613** (`eqvdyntw`): dense `marlin_gemm` present in the served wheel.
- **stark #381** (my prior validity finding): int4-Marlin body GEMM measured **bit-exact at
  `size_m=8`** (1/8/16 max_diff 0.0; first divergence at M=64) under `BI=1`. If true, the
  structural BI-coverage question and the *attribution* of the #607 break can decouple: Marlin
  can be (a) structurally outside BI coverage AND (b) still bit-exact at the decode-verify
  width — which would point the break at a non-Marlin op. The audit must distinguish the
  structural claim (does BI patch Marlin?) from the attribution claim (is Marlin the cause?).

## Boundaries
Source: `/workspace/senpai/target/.venv/.../site-packages/vllm` (0.22.0). stark-owned paths
only. No build, no checkpoint load unless a tiny load is strictly needed to trace dispatch.
