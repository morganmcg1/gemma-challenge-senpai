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

## Result (W&B azlgrinu, A10G, vllm 0.22.0 / torch 2.11.0+cu130)

**Kernel inventory / step (M=7 verify, conc=1):** RMSNorm ×302 (210 hidden
[input/post_attn/pre_ff/post_ff/post_ple ×42] + 1 final fused-add + 1 PLE-proj +
42 q_norm + 24 k_norm + 24 v_norm), RoPE ×42, elementwise resid/scale ~×168
(3 adds + scalar-mul / layer), attention ×42 (35 sliding + 7 full), KV-write ×24
(non-shared), sampling ×1.

**Build fact that pins the lever:** this vLLM has NO hand `_C` norm/rope kernel
(`torch.ops._C.{rms_norm,fused_add_rms_norm,rotary_embedding}` absent;
`RMSNorm/RoPE.enabled()=False`) -> forward_native (pure torch), which the
shipped default torch.compile (Inductor) fuses. So realizable = fused
(compile + FULL@7 CUDA graph). No "switch to fast CUDA kernel" lever exists.

**Three-rung ladder (ms/step, sum over all non-GEMM):**
eager (no graph/compile) 60.0 -> graphed (CUDA graph) 8.91 -> fused (+compile)
7.28. Launch lever (eager->graphed) 51.1 ms + fusion lever (graphed->fused)
1.64 ms = **52.7 ms/step ALREADY HARVESTED** by the shipped FULL@7 graph +
compile. Within the graph, fusion barely helps norms (15.7->14.5us) because each
is reduction-FLOOR-bound, not launch-bound.

**Per-class realizable (fused×mult), UPPER BOUNDS:** RMSNorm 4.27 ms (58.7%),
attention 2.38 ms (32.7%), elementwise 0.22, RoPE 0.22, KV-write 0.12, sampling
0.065. All ≤6% HBM-BW (sampling 28%) -> non-GEMM at M=7 is fixed-per-kernel-FLOOR
bound (~14us/reduction, ~5us/pointwise), set by kernel COUNT, not bandwidth.

**Calibration:** SUM(fused)=7.28 ms > 3.28 ms (#798). Per-op isolation can't
capture the full model's cross-op fusion and the synthetic paged-KV attn
over-costs vs real cached-KV flash -> per-class are UPPER BOUNDS; the dominant
inflation is the ~14us RMSNorm reduction floor (in-model these may run nearer the
~5us pointwise floor). Ranking + lever verdict robust regardless.

**Attention (surgattn-3D #791 headroom):** 32.7% share -> ~1.07 ms of the 3.28 ms.
Sliding attn (×35) plateaus at 52.8us (window cap 512); full attn (×7) scales
linearly with KV (19.6us@128 -> 230us@2048). Already launch-free Triton flash at
3-6% BW near its small-shape floor -> realizable surgattn headroom is a fraction
of 1.07 ms (it restructures, can't zero, attention).

**VERDICT — NO non-GEMM lever.** The launch+fusion lever (52.7 ms) is fully
harvested in the shipped path. What remains is fixed-floor/count-bound tiny
kernels: ~302 architecturally-required RMSNorm reductions + 42 launch-free
attention kernels. Cutting it needs FEWER kernels (fusion maxed) or FEWER ops
(Gemma4 sandwich + q/k/v + PLE norms are fixed). Decode is GEMM-bound (body-MLP
wall, #798); kernel side EXHAUSTED. Only non-GEMM-adjacent lever is attention via
surgattn-3D (#791, wirbel), capped ≤1.07 ms.
