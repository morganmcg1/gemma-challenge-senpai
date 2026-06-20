# PR #789 — Drafter CUDA-graph capture: are bi0's M=1 proposer passes eager?

Base: bi0 = `int4_mtp_bi0_surgattn`, official TPS 218.02, PPL 2.0058, 128/128,
W&B `s63tb03x`. Control timing anchor (#786 `h1nsfad1`): drafter_gpu_ms 2.434
(6× M=1), verify_gpu_ms 11.752 (M=7), drafter/verify 0.207, drafter = 17.2% of
GPU-busy, accept 0.3856, E_accept 3.314, local steady decode 210.1 TPS.

**Key question:** do the 6 sequential MTP-drafter proposer passes run EAGER (one
kernel-launch per op, per pass) or CUDA-graph CAPTURED (collapsed to graph
replay)? If eager → capture recovers the launch-latency slice of the 2.434 ms at
byte-identical outputs. If captured → 2.434 ms is irreducible M=1 tiny-GEMM
compute (clean null).

## Step 1 — capture state (code-inspection + #786 server log, GPU-free)

### Serve config (serve.py)
- No `--enforce-eager` unless `ENFORCE_EAGER=1` (default off) → main model CUDA
  graphs ON.
- No `--compilation-config`, no `cudagraph_capture_sizes` override.
- `--speculative-config` = `{"model": drafter, "num_speculative_tokens": 6}` only
  (no `enforce_eager` key, no drafter compilation override).

### Server startup census (#786 control/server.log, vLLM 0.22.0)
- engine `enforce_eager=False`; `cudagraph_mode=FULL_AND_PIECEWISE`;
  `cudagraph_capture_sizes=[1,2,4,8]`; `max_cudagraph_capture_size=8`.
- L85 `gemma4.py:137 Gemma4 MTP: captured centroids CUDA graphs for sizes
  [1,2,4,8,16,32,64]` → only the drafter **centroid output head**
  (`get_top_tokens`) is captured (see `Gemma4Proposer._setup_centroids_cuda_graphs`),
  NOT the transformer body.
- L95 `backends.py:1089 Using cache directory: .../eagle_head for vLLM's
  torch.compile` → drafter body IS torch.compile/inductor-compiled.
- L101 `Profiling CUDA graph memory: PIECEWISE=1 (largest=7), FULL=1 (largest=7)`;
  L107-108 capture progress: 1 PIECEWISE + 1 FULL graph, largest=7 (= M=K+1=7
  verify shape) → these are the MAIN model's verify-shape graphs. No separate
  drafter-graph census line.
- L130-134 EAGLE spec-decode glue kernels JIT-compiled during inference
  (`eagle_prepare_inputs_padded_kernel`, `eagle_prepare_next_token_padded_kernel`,
  `kernel_unified_attention`, `rejection_greedy_sample_kernel`).

### Behavioral STEPTIME probe (#786 control, steady-state i=32..71)
- `kind=draft` (proposer propose(), 6× M=1): cpu ≈ 5.5 ms, gpu ≈ 2.0–2.3 ms
  → **CPU/launch-bound** (CPU ≈ 2.7× GPU work).
- `kind=exec` (main-model verify, M=7): cpu ≈ 6.5 ms, gpu ≈ 11.1 ms
  → **GPU-bound** (CPU < GPU = captured/compiled).
- The contrast is the signature: the verify pass is GPU-bound (captured), the
  proposer spends 5.5 ms CPU to issue only 2.0 ms of GPU work.

### vLLM 0.22.0 drafter-capture support (source)
- `SpecDecodeBaseProposer.initialize_cudagraph_keys` (llm_base_proposer.py:380):
  drafter supports **PIECEWISE only**, gated on
  `not speculative_config.enforce_eager` AND main `cudagraph_mode.mixed_mode()`
  in {PIECEWISE, FULL}. FULL cudagraph of the drafter is NOT supported.
- propose() (llm_base_proposer.py:427) calls
  `_determine_batch_execution_and_padding(num_tokens)` → cudagraph_runtime_mode
  per pass, then `self.model(**model_kwargs)`.

**Open item (settling EAGER vs PIECEWISE-captured):** is the drafter's M=1 pass
actually dispatched PIECEWISE at runtime, or NONE (eager)? The 5.5 ms CPU is more
consistent with per-op eager dispatch than piecewise replay. Confirming via the
gpu_model_runner spec-decode capture flow + a runtime cudagraph-mode probe.
