# W4A16 GEMM kernel inventory — vLLM 0.22.0 on A10G (sm_86)
Model: google/gemma-4-E4B-it-qat-w4a16-ct (compressed-tensors, symmetric int4, group_size=32)
Method: code inspection of the pinned wheel at /tmp/senpai-venvs/20f658587e8a6643/lib/python3.12/site-packages/vllm/

## Dispatch (the SELECTED kernel)
compressed-tensors detected -> CompressedTensorsWNA16.create_weights (schemes/compressed_tensors_wNa16.py:109)
-> choose_mp_linear_kernel (kernels/linear/__init__.py:611) iterates _POSSIBLE_KERNELS[CUDA]:
| prio | class | min_cap | sm_86 outcome |
|---|---|---|---|
| 1 | CutlassW4A8LinearKernel | 90 | SKIPPED (cap 90>86) |
| 2 | MacheteLinearKernel | 90 | SKIPPED (cap 90>86) — also is_device_capability(90) inside can_implement |
| 3 | AllSparkLinearKernel | 80 | REJECT — only uint8b128 (W8), not uint4b8 |
| 4 | MarlinLinearKernel | 75 | **SELECTED** |
| 5 | ConchLinearKernel | 80 | unreachable; conch-triton-kernels not installed |
| 6 | ExllamaLinearKernel | 60 | rejects bf16 (exllama.py:52 act_type!=float16) |
=> Marlin is the SOLE servable W4A16 kernel for this checkpoint on sm_86.

## Knobs (all dead ends on sm_86)
- VLLM_MARLIN_USE_ATOMIC_ADD (envs.py:160,1321; default False): NO-OP on sm_86+bf16.
  should_use_atomic_add_reduce (marlin_utils.py:445) hard-returns False when device_capability[0]<9 AND dtype==bfloat16 (sm8x lacks native bf16 atomicAdd). Also returns False for n>=2048 or k<2048 (most bi0 linears).
  EMPIRICALLY CONFIRMED: serving with VLLM_MARLIN_USE_ATOMIC_ADD=1 still logs "Using MarlinLinearKernel for CompressedTensorsWNA16" (research/_localrun/lever_probes/atomic_add_on.server.log:30) — the toggle changes nothing at kernel selection, and the in-kernel reduce path stays the fp32 non-atomic one on Ampere+bf16.
- VLLM_MARLIN_INPUT_DTYPE (int8/fp8): W8A8-style activation quant; not W4A16; changes numerics (asserts reject for uint4b8 path).
- use_fp32_reduce: hardcoded USE_FP32_REDUCE_DEFAULT=True (marlin_utils.py:36); NOT env-exposed; MarlinLinearKernel.apply_weights doesn't pass it. Flipping would change reduction precision (numerics).
- Tile/thread/split: ALL compile-time constants (GPTQ_MARLIN_TILE=16, MIN_THREAD_N=64, MIN_THREAD_K=128, MAX_PARALLEL=16). ops.marlin_gemm() takes NO tile/num_warps/num_splits args. Zero user-facing config.
- VLLM_DISABLED_KERNELS=MarlinLinearKernel -> falls to Conch (absent) or Exllama (rejects bf16) => NO servable kernel (would fail to boot).
- --quantization <method>: HARD-REJECTED when method != the checkpoint's self-declared method. ModelConfig pydantic validation raises ValidationError "Quantization method specified in the model config (compressed-tensors) does not match the quantization method specified in the `quantization` argument (machete)" and the server fails to boot. EMPIRICALLY CONFIRMED for machete (see research/_localrun/lever_probes/quant_machete.server.log). This is config-method enforcement BEFORE kernel dispatch: every alt method name (machete/marlin/gptq_marlin/awq_marlin) mismatches compressed-tensors and is refused the same way; only --quantization compressed-tensors (the no-op match) is accepted. So the CLI cannot force a different W4A16 kernel.

## Conclusion
There is NO numerics-equivalent faster W4A16 GEMM kernel or config available on sm_86 in vLLM 0.22.0. The kernel cannot be swapped (Marlin is the only one) or reconfigured (no exposed knobs; the one toggle is HW-gated off). => the kernel-swap lever does not exist on this hardware.
