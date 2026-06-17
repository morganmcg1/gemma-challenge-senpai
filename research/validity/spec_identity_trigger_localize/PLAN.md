# PR #621 — Localize the real spec-#319 trigger (non-Marlin op probe, vLLM 0.22.0)

`analysis_only=true`, `official_tps=0`, **NO build / NO training / NO HF Job**, single A10G (sm_86).
Reuse the #617 row-0 bit-exact probe (output row 0, M=8 vs M=1) on the **non-Marlin** verify ops.

## Background (what's already established)
- **#617 (`fa1f9vm1`)**: int4-Marlin GEMM bit-exact at the served verify width (maxdiff 0.0 at M=7/8/16, all 4 served shapes; first divergence M=32, 1 bf16-ULP). `VLLM_BATCH_INVARIANT=1` covers only dense ATen + attention; Marlin custom `_C` op is outside it but needs no covering at verify width → the 47% break is a **non-Marlin** trigger.
- **#607 (`yuvztndu`)**: 31048/65536 (47%) spec-#319 greedy break on the clean 0.22.0 gate (0/65536 ref-vs-ref floor).
- Submission: `submissions/int4_mtp_batchinv` (`vllm==0.22.0`, `VLLM_BATCH_INVARIANT=1`, `MAX_NUM_SEQS=1`, `NUM_SPECULATIVE_TOKENS=6`; backend = TRITON_ATTN per `vllm_attn_group_patch.py`).

## Source findings (pre-probe)
- `batch_invariant.py`: BI overrides ONLY ATen ops (mm/addmm/matmul/linear/bmm/softmax/mean). No direct attention override.
- Sole attention BI mechanism: per-backend `num_splits=1` / `is_batch_invariant` guards.
  - `flash_attn.py:1194/1219`: `num_splits=1 if VLLM_BATCH_INVARIANT` (FlashAttention only).
  - `triton_unified_attention.py:931`: `is_batch_invariant` forces `use_3d=False` (2D, num_segments=1). Also `max_seqlen_q>1` (line 929) forces 2D for the M=K+1 verify step regardless of BI.

## Probe plan
1. Confirm the submission's actual attention backend (selector, no checkpoint load).
2. Row-0 probe on attention: M=8 vs M=1, BI=1 (expect 0 if guard taken) and BI=0 (expect >0 if guard is what closes it).
3. If attention clean: row-0 probe the MTP drafter forward + sampler/argmax tie-break; extend the Marlin probe to the lm_head shape.
4. Verdict: `SPEC_TRIGGER_RECOVERABLE__<op>__<knob>` (surface knob, NEVER auto-fire) or `SPEC_TRIGGER_FUNDAMENTAL__<op>`.

W&B group `spec-identity-trigger-localize`.
