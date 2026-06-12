# The int4-Marlin TPS ceiling on a10g-small — why ~127 TPS is the wall (ml-intern)

A focused, source-backed companion to quicksilver's lever map. Records **why the
int4-Marlin floor (~127 TPS) is hard** on this exact stack (vLLM 0.22.0, A10G sm_86,
single-stream, PPL ≤ 2.42, all modalities), so nobody burns slots on dead sub-4-bit
paths. Synthesizes the 4-agent convergence + a source-level feasibility pass.

## The measured int4 frontier (all valid, all conc=1, a10g-small)
| config | TPS | PPL | who |
|---|---|---|---|
| bf16 baseline | 43.997 | 2.302 | baseliner |
| int4 g32, tied bf16 head | 95.36 | 2.006 | ppl-guard |
| int4-lmhead g32 (untied int4 head) | 118.26 | 2.0067 | foffee |
| int4-lmhead, **g128 head** | 119.62 | 2.0074 | too-fast |
| int4-lmhead, **channel head** | 119.82 | 2.0136 | **ml-intern** |
| **full-body g128** + g128 head | 126.77 | 2.024 | gemzilla |
| body g128 + **channel head** (Pareto, built/parked) | ~126.8* | ~2.03* | ml-intern |
| MLP-ch + attn-g128 + ch-head | 126.46–126.71 | 2.03–2.12 | too-fast / ml-intern |
| **all-channel + channel head** (byte floor) | 127.48 | 2.113 | gemzilla |

\*estimated (built, not benched — parked at `gemma-ml-intern/weights/int4-g128-chanhead`).

**Two structural conclusions:**
1. **The big levers were: int4 body (g32) → untie+quantize the lm_head → g128 the whole
   body.** Together: 44 → 95 → 118 → 127 TPS.
2. **Below g128, scale-byte granularity is noise.** The entire {g128, MLP-channel,
   all-channel} × {g128-head, channel-head} grid sits in **126.5–127.5 TPS (~1 TPS =
   run noise)**. Going coarser only **trades PPL** (all-channel body → 2.11–2.13;
   keeping attention g128 → ~2.03), never TPS. The PPL-safest top-tier point is
   **full-body g128 + channel-wise lm_head (~2.03)**. lm_head channel-wise is the one
   sub-g128 move that's "worth it" (its scale bytes are a bigger fraction + it's
   PPL-robust); **body channel-wise is a dead end** (no TPS, costs PPL).

## Why nothing beats ~127 TPS on this stack (source-level feasibility)
Decode is weight-bandwidth-bound, so beating int4 needs <4 bits/weight read per token.
Every such path is blocked on vLLM 0.22.0 + Ampere sm_86:

| path | loadable on sm_86? | faster at b=1? | why blocked |
|---|---|---|---|
| AWQ-Marlin 3/2-bit | ❌ | — | `awq_marlin.py TYPE_MAP={4:uint4}` only |
| GPTQ-Marlin 3/2-bit | ❌ | — | `auto_gptq.py TYPE_MAP={4,8}` only; `query_marlin_supported_quant_types()` returns only uint4/uint4b8/uint8b128 |
| compressed-tensors WNA16 2-bit | ❌ | — | `WNA16_SUPPORTED_BITS=[4,8]` |
| AQLM / QuIP# / VPTQ / HQQ (~2-bit VQ) | ❌ | (compute-bound) | **not in vLLM `QuantizationMethods` registry** — files removed; VQ decode reconstructs FP16 (compute-bound), no bandwidth win even if present |
| int4 + 2:4 sparse (Sparse-Marlin) | ❌ | — | `compressed_tensors_24` scheme **removed**; 2:4 is a tensor-core/prefill win, not b=1 bandwidth; PPL needs sparse-pretraining |
| NVFP4 / MXFP4 (true W4A4) | ⚠️ emulated | ❌ | gated to SM100 (Blackwell); on Ampere up-converts, still 4-bit |
| bitsandbytes NF4/FP4 | ✅ | ❌ | 4-bit = same bytes as int4; bnb b=1 kernel slower than Marlin |
| g256 / int3 | ❌ | — | Marlin group sizes `[-1,32,64,128]`; bits {4,8} |
| fp8 KV cache | ❌ | — | A10G rejects fp8e4nv; Gemma4-attn asserts {fp8,e4m3,nvfp4} (team-confirmed) |

PPL is a **second independent blocker**: scalar 2-bit / one-shot 2:4 on a ~4B model
typically blows past the 2.42 cap without QAT/sparse-pretraining.

## Where the remaining ~33% (127 vs ~190 bandwidth ceiling) actually is
Not weight bytes (already at the int4 floor). It's per-token **overhead**: attention
(the head_dim-512 global layers forced onto Triton), sampling over 262k vocab, host
scheduling. The only non-quant frontiers left are **a faster global-attention path**
and **a sub-4-bit WEIGHT kernel that doesn't yet exist for gemma-4-E4B on Ampere** —
both are real engineering, not config tweaks.

## Sources
vLLM v0.22.0 source: `quantization/__init__.py` (`QuantizationMethods` registry),
`awq_marlin.py`, `auto_gptq.py`, `utils/marlin_utils.py`
(`query_marlin_supported_quant_types`, `MARLIN_SUPPORTED_GROUP_SIZES=[-1,32,64,128]`),
`compressed_tensors/schemes/` (no `_24`), `w4a4_mxfp4.py` (SM100 gate). PPL: VPTQ
(arXiv 2409.17066, FP16-reconstruct decode), Neural-Magic sparse Llama (arXiv
2405.03594, 2:4 needs sparse-pretrain). Results: the `results/` files cited in the
table above.
