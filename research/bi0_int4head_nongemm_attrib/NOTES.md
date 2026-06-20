# PR #806 — Non-GEMM verify-step attribution (stark)

LOCAL A10G synthetic-microbench, analysis-only. NO HF job, NO submission,
NO full checkpoint on disk. W&B group: `bi0-int4head-nongemm-attrib`.

## Goal
Split the ~3.28 ms "verify non-GEMM" residual of the int4head conc=1 decode
cycle (#798, W&B `dpc36210`): total 12.42 ms = 256.74 TPS; MTP drafter 2.48 ms
(20.0%), body verify-GEMM 5.92 ms (47.6%), lm_head GEMV 0.75 ms (6.0%),
**verify non-GEMM 3.28 ms (26.4%) ← target**. Lumped: attention kernel, RMSNorm
(~6/layer), RoPE, KV-cache write, sampling. Nobody has split it.

## Verdict question
Is any class a real >=+2% lever (e.g. launch-bound fusable RMSNorms, or an
attention cost that bounds surgattn-3D), or is it irreducible (already-fused
attention + BW-bound norms) -> decode is definitively body-MLP-bound, kernel
side exhausted?

## Method (reuse #798 scaffolding, value-independent at conc=1)
research/bi0_int4head_reprofile/gemm_attrib.py: CVD=0, time_call_graph (CUDA
-graph launch-free + eager fallback w/ launch floor), roofline (%HBM-BW), W&B.
Non-GEMM timings are shape-bound (norms/RoPE/KV/sampling) or shape+KV-length
-bound (attention) -> synthetic activations faithful, no weights needed.

## Status
WIP — enumerating kernels from gemma4.py + serving config.
