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

### Why only the M=7 verify shape is captured (the mechanism)
- `compilation.py:adjust_cudagraph_sizes_for_spec_decode` rounds EVERY
  `cudagraph_capture_sizes` entry UP to a multiple of `uniform_decode_query_len
  = 1 + num_spec = 7`: [1,2,4,8] → {7,7,7,14}; 14 > max_capture(8) dropped →
  **`cudagraph_capture_sizes = [7]`, `max_cudagraph_capture_size = 7`**. So only
  size-7 is ever captured (matches the census `largest=7, PIECEWISE=1, FULL=1`).
- `speculative_config.enforce_eager` is only forced True for `deepseek_v32` MTP
  (speculative.py:554) AND only when `self.model is None`; serve.py provides the
  drafter model explicitly, so that block is skipped → spec `enforce_eager`
  stays None → `not None == True` → the drafter PIECEWISE path is ENABLED.
- During capture, `gpu_model_runner._dummy_run` calls `drafter.dummy_run` with
  `use_cudagraphs=True` ONLY during the main model's PIECEWISE pass (gate
  gpu_model_runner.py:5854-5863), so the drafter captures a **PIECEWISE graph at
  size 7** and no FULL graph. `Gemma4Proposer.initialize_cudagraph_keys`
  (llm_base_proposer.py:380) hard-codes drafter mode to **PIECEWISE or NONE
  only** — FULL drafter capture is NOT supported.

### Runtime dispatch of the M=1 loop passes
- `propose()` calls `_determine_batch_execution_and_padding` twice: once for the
  first pass (num_tokens), once for the loop (batch_size=1 at max_num_seqs=1).
- `CudagraphDispatcher.dispatch(1)`: `_bs_to_padded_graph_size[1] = 7` (pads
  1→7), then the relaxed PIECEWISE key `(num_tokens=7, num_reqs=None,
  uniform=False)` MATCHES the captured size-7 key → returns
  **(PIECEWISE, padded=7)**. So by code, the M=1 loop passes are expected to run
  PIECEWISE-at-padded-width-7, NOT raw eager.

### Reconciling PIECEWISE-replay with the 5.5 ms launch-bound timing
PIECEWISE cudagraph splits the model at `unified_attention_with_output`
(splitting_ops), so per drafter forward pass: ~5 captured graph segments + **4
eager TRITON_ATTN attention regions** (reshape_and_cache + unified_attention +
metadata kernels). ×6 passes + EAGLE input-prep kernels + centroid-head replays
+ per-pass sampling ≈ ~150 launches ≈ the measured ~5.5 ms CPU. The launch
overhead lives in the parts PIECEWISE CANNOT capture (attention + piece
boundaries), not in un-captured linear GEMMs.

**Emerging verdict (pending probe confirmation of PIECEWISE vs NONE):** the
drafter is NOT fully eager — vLLM-native PIECEWISE capture is already active on
its linear/norm regions. Collapsing the 6 M=1 passes into one graph replay
requires FULL-cudagraph / whole-loop capture of the proposer, which vLLM 0.22.0
**does not expose** (PIECEWISE-only by design) — it would need a vLLM
spec-decode internals patch. That whole-loop "LOOPGRAPH" capture is exactly
**kanna #771's axis** (fa2sw custom sitecustomize), not a vLLM-native config
lever. → CAP/STOP per #784: a valid "config-capture-not-exposed" null for the
DRAFTER-native axis, with the remaining lever flagged as kanna's lane.

### Runtime probe (settling PIECEWISE vs NONE behaviorally)
`research/bi0_drafter_cudagraph/cudagraph_probe.py` (CGPROBE=1) wraps
`_determine_batch_execution_and_padding` to log the dispatched mode per pass.
Run via `run_probe.py`. Results below (local A10G, control bi0 config, decode
1536 tokens / 8 prompts; W&B `bi0-drafter-cudagraph` group, run `cgprobe-control`).

**Dispatch mode (decisive) — `runs/control_probe/server.log`, steady-state:**
- n=1000 dispatches: `mode_counts={'PIECEWISE': 993, 'NONE': 7}`.
- shapes: `7->7:PIECEWISE: 493` (first proposer pass, num_tokens=K+1=7),
  `1->7:PIECEWISE: 500` (the M=1 loop passes — num_tokens=1 PADDED→7, dispatched
  PIECEWISE), and the 7 NONE are ALL large prefill shapes (`356,457,199,328,159,
  171,157`→self, all >> max_capture=7, so they run eager — these are prompt
  prefill, NOT decode-path drafter passes).
- **VERDICT: 100% of decode-path drafter passes (first pass AND the 6 M=1 loop
  passes) dispatch as PIECEWISE — never eager (NONE).** Confirms the Step-1
  code reading (`_bs_to_padded_graph_size[1]=7` → relaxed PIECEWISE size-7 key).

**Timing (steady-state, `[cgprobe] agg timing`, n≈460 draft / 470 exec):**
- `kind=draft` (proposer propose(), 6× M=1): cpu p50 **5.596 ms**, gpu p50
  **2.482 ms** → CPU ≈ 2.3× GPU = **CPU/launch-bound** (reproduces #786
  drafter_gpu_ms 2.434 / cpu≈5.5).
- `kind=exec` (main-model verify, M=7): cpu p50 **7.631 ms**, gpu p50
  **11.874 ms** → **GPU-bound** = captured/compiled (reproduces #786
  verify_gpu_ms 11.752).
- The launch-bound proposer signature survives WITH PIECEWISE active, confirming
  the residual ~3 ms CPU-over-GPU is launches PIECEWISE **cannot** capture:
  4 eager TRITON_ATTN attention regions/pass × 6 + piece boundaries + EAGLE
  input-prep + centroid-head replays + per-pass sampling (≈150 launches/step),
  NOT un-captured linear GEMMs.

### Reconciliation with lawine #787 cross-pollination (advisor 2026-06-20 13:15Z)
Advisor relayed lawine's bi0 init-dump census: `cudagraph_capture_sizes=[1,2,4,8]`,
`max_cudagraph_capture_size=8`, `cudagraph_mode=FULL_AND_PIECEWISE`,
`enforce_eager=False` — "a size-1 graph DOES exist" — and lawine measured the M=7
VERIFY pass FULL-captured (0 paddings, 100%). Sharpened question: do the 6× M=1
DRAFTER proposer passes **dispatch** to a size-1 graph or **bypass** to eager?

My own run's startup log settles it empirically:
- **Config LIST vs captured graphs differ.** Line 22 dump confirms the *raw*
  `compilation_config.cudagraph_capture_sizes=[1,2,4,8]`, `max=8` (= lawine's
  census, the pre-spec-adjustment list). **But** line 101
  `Profiling CUDA graph memory: PIECEWISE=1 (largest=7), FULL=1 (largest=7)` and
  line 107-108 (`PIECEWISE 1/1`, `decode FULL 1/1`) show the *actually captured*
  graphs are **only at size 7** — `adjust_cudagraph_sizes_for_spec_decode` rounds
  every [1,2,4,8] entry UP to a multiple of `uniform_decode_query_len=7` and drops
  >max → **no size-1/2/4/8 graph is ever captured.** So "size-1 graph exists" is
  true of the *config list* but **false of the realized captures**.
- **One shape (7), TWO graphs — the proposer/verifier asymmetry.** At size 7 there
  is a FULL@7 graph AND a PIECEWISE@7 graph. lawine's VERIFY pass dispatches to
  **FULL@7** (GPU-bound, fully amortized → lawine's null). My probe shows the
  PROPOSER dispatches to **PIECEWISE@7**: `1->7:PIECEWISE` (M=1 loop, padded 1→7)
  and `7->7:PIECEWISE` (first pass). The proposer gets only PIECEWISE because
  `Gemma4Proposer.initialize_cudagraph_keys` is hard-coded PIECEWISE-or-NONE
  (FULL proposer capture unsupported in 0.22.0).
- **Answer to the sharpened question:** the 6× M=1 proposer passes **DISPATCH to a
  captured graph (PIECEWISE@7), not a fully-eager bypass, and not a size-1 graph
  (none is captured).** But because it is PIECEWISE (not FULL), the advisor's
  binary "dispatch ⇒ all irreducible compute" is too strong: the linear/norm GEMMs
  ARE amortized, yet each of the 6 passes still issues eager TRITON_ATTN attention
  + piece-boundary launches that PIECEWISE structurally cannot capture (the real
  source of the draft cpu 5.6ms ≫ gpu 2.5ms launch-bound gap). That residual is
  **real launch overhead, not compute** — but it is **not config-recoverable**:
  removing it needs FULL/whole-loop proposer capture, which 0.22.0 does not expose.
  → CAP/STOP per #784; whole-loop lever = kanna #771's LOOPGRAPH axis.

## VERDICT (Step 1 → close)

The bi0 drafter's 6 M=1 proposer passes are **already PIECEWISE CUDA-graph
captured by vLLM 0.22.0-native machinery** (padded 1→7), NOT eager. Step 2's
config-capture premise (passes are eager) is therefore **false** — there is no
eager linear/GEMM slice to recover via `--compilation-config` /
`cudagraph_capture_sizes`. The remaining launch overhead lives in regions
PIECEWISE structurally cannot capture (attention + piece boundaries + spec glue);
collapsing the 6 passes into ONE replay needs **FULL / whole-loop drafter
capture**, which vLLM 0.22.0 **hard-disables**
(`SpecDecodeBaseProposer.initialize_cudagraph_keys` is PIECEWISE-or-NONE only —
FULL drafter capture is not supported, llm_base_proposer.py:380). That whole-loop
"LOOPGRAPH" capture is a vLLM-internals patch, not a config lever → **CAP/STOP
per #784: a valid "config-capture-not-exposed (already PIECEWISE-captured)" null
for the DRAFTER-native axis.** The whole-loop lever is **kanna #771's axis**
(fa2sw custom sitecustomize LOOPGRAPH), NOT duplicated here.

Probe artifacts (`cudagraph_probe.py`, `run_probe.py`, `runs/control_probe/`)
are research-only; the submission `sitecustomize.py` was reverted byte-pristine
(no CGPROBE hook ships).
