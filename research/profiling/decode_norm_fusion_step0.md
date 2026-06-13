# Step-0 audit (PR #67) — Decode RMSNorm+residual is ALREADY FUSED (TorchInductor). Terminal NEGATIVE.

_Submission_: `submissions/fa2sw_precache_kenyan` (the deployed 481.53-frontier stack: int4-pck04 + MTP drafter K=7 + PLE-fold + fa2sw + onegraph + precache + #43 split-KV verify).
_Source run_: existing same-submission decode profile **`r0ahjs45`** (group `decode-reprofile-postsplitkv`, captured 2026-06-13T21:46:41Z), `research/profiling/frontier_decode_postsplitkv/`. The Inductor-compiled kernel graph is deterministic for a fixed model+config, so this same-stack trace **is** the Step-0 itemization — no re-serve needed (and the PR says do not burn budget if already fused).
_Model_: `gemma4_text`, hidden_size **2560**, **37** layers, verify **M=8**. A10G peak HBM BW ≈ 600 GB/s.

## Verdict

**NEGATIVE — do not implement.** The decode-path RMSNorm + residual-add (+ layer_scalar mul) is **already fused into single Triton kernels by TorchInductor** (`@support_torch_compile` is active; `enforce_eager` is not set). The trace contains **zero** µs of an un-fused "standalone `rms_norm` → separate `add`" configuration. The PR's hand-written fused-norm kernel would, at best, replicate what Inductor already emits, and would more likely be **slower** because Inductor fused several residual-adds across the **marlin GEMM epilogue** — a boundary a standalone norm kernel cannot cross. Independently, at M=8 these kernels run at **~10–17% of peak HBM BW**, i.e. they are **fixed-overhead/occupancy-bound, not bandwidth-bound**, so "collapsing HBM round-trips" reclaims nothing. This is the same fail-fast outcome as #65 (CUDA-graph already deployed).

## Evidence 1 — source: Gemma4 sandwich-norm calls every norm with `residual=None`

`vllm/model_executor/models/gemma4.py::Gemma4DecoderLayer.forward` (deployed venv):

```python
residual = hidden_states
hidden_states = self.input_layernorm(residual)            # rms_norm, residual=None
hidden_states = self.self_attn(...)
hidden_states = self.post_attention_layernorm(hidden_states)  # rms_norm, residual=None
hidden_states = hidden_states + residual                   # residual add (separate in eager)
residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(hidden_states) # rms_norm, residual=None
hidden_states = self.mlp(hidden_states)
hidden_states = self.post_feedforward_layernorm(hidden_states) # rms_norm, residual=None
hidden_states = hidden_states + residual                   # residual add (separate in eager)
hidden_states = hidden_states * self.layer_scalar          # scalar mul (separate in eager)
```

`RMSNorm.forward` with `residual=None` dispatches to standalone `ir.ops.rms_norm` (never `fused_add_rms_norm`). So **in eager** the norm and the residual-add would be separate kernels — this is the gap the PR hypothesised. But the deployed path is **not eager**.

## Evidence 2 — the deployed path is TorchInductor-compiled

- `Gemma4` decoder layers are wrapped by `@support_torch_compile`; `manifest.json` sets **no** `enforce_eager` and does not lower the vLLM compile level → vLLM's default torch.compile / Inductor path is live.
- `ONEGRAPH=1` captures the K=7 spec-decode loop into one CUDA graph **on top of** the Inductor-compiled kernels; decode is **99.41% GPU-bound**, host launch latency already overlapped (per `r0ahjs45` / memory [[project-cudagraph-already-deployed]]).
- Proof that compile is actually live: the trace kernel names are Inductor-generated `triton_*_fused_*` (eager would show `aten::rms_norm` + `aten::add`).

## Evidence 3 — trace itemization: the residual-add is INSIDE the norm kernel

Decode-steady kernels in the norm/elementwise bucket (`trace_frontier/profiler_out_0.txt`, ~54-cycle window; avg µs/call is exact):

| kernel | calls | CUDA ms | µs/call | residual/scalar fused in? |
|---|--:|--:|--:|---|
| `triton_red_fused_add_marlin_gemm_mul_rms_norm_4` | 1944 | 6.949 | 3.58 | **add+mul+rms_norm (+ GEMM epilogue)** |
| `triton_red_fused_add_marlin_gemm_rms_norm_0` | 1998 | 6.773 | 3.39 | **add+rms_norm (+ GEMM epilogue)** |
| `triton_red_fused_add_rms_norm_2` | 1998 | 5.235 | 2.62 | **add+rms_norm** |
| `triton_per_fused_add_mul_rms_norm_2` | 1680 | 2.800 | 1.67 | **add+mul+rms_norm** |
| `triton_per_fused_add_rms_norm_0` | 1680 | 2.758 | 1.64 | **add+rms_norm** |
| `triton_poi_fused_add_index_select_mul_rms_norm_split…` | 1260 | 2.109 | 1.67 | **add+mul+rms_norm (PLE)** |
| `triton_per_fused_rms_norm_view_3` | 1260 | 1.826 | 1.45 | standalone (pre-norm, no residual adjacent) |
| `triton_per_fused_rms_norm_split_with_sizes_view_5` | 864 | 1.298 | 1.50 | standalone (pre-norm) |
| `triton_per_fused_rms_norm_view_2` | 420 | 0.619 | 1.47 | standalone (pre-norm) |
| `triton_per_fused_rms_norm_1` | 420 | 0.589 | 1.40 | standalone (pre-norm) |

- **~78% of norm-category time** is in `*_fused_add*_rms_norm*` kernels — the residual add (`add`) and layer-scalar (`mul`) are co-resident with `rms_norm`, one HBM round-trip.
- The only **standalone** `rms_norm` kernels are the **pre-sublayer norms** (`input_layernorm`, `pre_feedforward_layernorm`) whose output feeds attention/MLP — there is **no residual adjacent to them to fuse**. Inductor still fused some of these into the marlin GEMM (`triton_red_fused_add_marlin_gemm_rms_norm`).
- **There is 0 µs of an un-fused "`rms_norm` then separate `add`" pair anywhere in the trace.** The hypothesised target does not exist in the deployed graph.

Bucket total (composition analysis, `frontier_decode_profile.json`): norm + elementwise = **0.595 ms/cycle = 7.47% of GPU-busy** (norm 4.44% + elementwise_copy 2.36%).

## Evidence 4 — at M=8 these kernels are fixed-overhead-bound, NOT bandwidth-bound

For `triton_red_fused_add_rms_norm_2` at M=8, hidden=2560, BF16: traffic ≈ read x (40KB) + read residual (40KB) + write out (40KB) + write residual (40KB) + weight (5KB) ≈ **165 KB**. At 600 GB/s peak that is **0.27 µs**; measured **2.62 µs** ⇒ **~10% of peak BW**. The `triton_per_*` norm kernels (1.4–1.7 µs) ⇒ **~15–17% of peak**. The time is dominated by fixed kernel overhead (grid launch + occupancy — an [8×2560] problem fills only a handful of the A10G's 80 SMs), not HBM transfer. ONEGRAPH already removed the host-launch component. A hand-written kernel cannot go below one kernel per fusion region (Inductor already there) and cannot reduce the fixed device overhead.

## Why a hand-written kernel would be WORSE

`triton_red_fused_add_marlin_gemm_rms_norm_0` shows Inductor fused the residual-add into the **marlin GEMM epilogue** and the following norm. A standalone hand-written "norm+residual" Triton kernel cannot fuse across the marlin GEMM boundary, so it would force a separate GEMM-output write+read — **adding** an HBM round-trip and a kernel, the opposite of the PR's goal.

## Conclusion

The norm/elementwise residual is **already optimally fused** and the bucket is **not HBM-bandwidth-bound** at decode M=8. Both legs of the PR premise (un-fused round-trips; bandwidth-bound) are false. Terminal NEGATIVE, no submission-file changes, no HF launch. The open systems levers remain drafter W8A8 (#2 block, stark #47) and acceptance (land #9 / fern #34) per `r0ahjs45`.
