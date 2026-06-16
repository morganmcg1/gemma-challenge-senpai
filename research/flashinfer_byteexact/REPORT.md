<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# FlashInfer byte-exact M-invariance for `google/gemma-4-E4B-it` on A10G (sm_86)

**PR:** #507 · **Author:** fern · **Generated:** 2026-06-16T16:42:44.963613+00:00 · **W&B group:** `flashinfer-byteexact`

**LOCAL diagnostic. GPU used ONLY for the FlashInfer decode-attn census (no model forward, no serve, no official-prompt TPS). NO HF job, NO submission, NO served-file change. `analysis_only=true`, `official_tps=0`.**

Reproduce: `cd target/ && .venv/bin/python research/flashinfer_byteexact/probe_flashinfer_feasibility.py --self-test`

---

## Verdict: LOADS but NOT free byte-exact M-invariant -- default split-KV decode is batch-VARIANT; fixed_split_size gives invariance at 1.2-4.7x M=1 cost; head_dim-512 layers blocked; version skew -> not a cheaper route to the 399.75 byte-exact rung

`feasibility_evidence_complete = 1` · `flashinfer_loads = 1` · `flashinfer_attn_m_invariant = 0` · `flashinfer_total_flips_M1vs8 = 1292` (tensor-core auto, L=8192)

## KEY OUTPUTS

| metric | value |
|---|---|
| `flashinfer_loads` | 1 (flashinfer 0.6.12 on torch 2.10.0+cu128, sm_86) |
| `flashinfer_attn_m_invariant` (out of the box) | 0 (default auto split-KV is batch-VARIANT) |
| `flashinfer_total_flips_M1vs8` | 1292/2048 (tensor-core auto, L=8192) |
| `flashinfer_m1_decode_tps` | 20328 decode-attn steps/s (tensor-core auto, L=4096; kernel micro-bench, NOT serve TPS) |
| `vs_splitkv_399_75` (cheaper free route?) | NO -- fixed_split is 2.6x slower than auto at M=1, head_dim-512 blocked, version skew |
| `fixed_split_size_invariant` | 1 (0 flips -- lawine #496's trick, native) |

## Findings ledger (each LIVE on the pod)

| rank | finding | confirmed | evidence |
|---|---|---|---|
| 1 | F1_flashinfer_loads | YES | flashinfer 0.6.12 imports + runs decode on NVIDIA A10G cap [8, 6] (torch 2.10.0+cu128, isolated venv). |
| 2 | F2_default_split_kv_batch_variant | YES | tensor-core (GQA) auto split-KV: 1292/2048 output elements flip M=1-vs-M=8 at L=8192 (max_abs 0.0001220703125); variant at every L tested. NOT free byte-exact. |
| 3 | F3_fixed_split_size_invariant | YES | fixed_split_size and disable_split_kv -> 0 flips at every L (byte-exact M-invariant): the first-class API form of lawine #496's split-KV trick. |
| 4 | F4_headdim512_no_invariant_path | YES | head_dim 512 (7 full-attention layers): tensor-core decode dispatch FAILS (no kernel); fixed_split_size requires tensor core -> no byte-exact invariant flashinfer path for those layers. (cuda-core runs d512 but has no working invariant knob.) |
| 5 | F5_version_skew_not_clean_drop_in | YES | flashinfer-python pins torch 2.10.0+cu128; pod serves torch 2.11.0+cu130 -> not a clean drop-in to the deployed vLLM 0.22.1rc1 stack (JIT-only torch-2.11; CUDA12/13 cubin risk). |

## M=1-vs-M=8 byte-exact census (flips / 2048 output elements, bf16)

Identical decode query + shared physical KV pages across the batch; the only difference is M (1 vs 8). Non-zero flips => the split scheduler changed request-0's reduction tree.

**Tensor-core path (GQA group 4 -> vLLM's serve-relevant decode path):**

- auto split-KV: L=2048: auto=1274 | L=8192: auto=1292 | L=32768: auto=1295  -> **batch-VARIANT at every L**
- fixed_split_size: L=2048: fixed=0 | L=8192: fixed=0 | L=32768: fixed=0  -> **0 flips, INVARIANT**
- disable_split_kv: L=2048: disable=0 | L=8192: disable=0 | L=32768: disable=0  -> **0 flips, INVARIANT**

**CUDA-core path (raw API default; not used by vLLM for GQA):**

- auto split-KV: L=2048: auto=0 | L=8192: auto=1200 | L=32768: auto=1219  (invariant only at short L; variant once it auto-splits; disable_split_kv is a no-op here)

## M=1 decode-attn throughput -- the cost of invariance (tensor-core, head_dim 256, bf16)

| L | auto (steps/s) | fixed_split_512 (steps/s) | auto / fixed |
|---|---|---|---|
| 2048 | 18860 | 15335 | 1.23x |
| 4096 | 20328 | 7679 | 2.65x |
| 8192 | 18663 | 3943 | 4.73x |

The invariant mode forfeits the split-KV GPU-fill that makes M=1 decode fast, so its cost GROWS with KV length -- the opposite of lawine #496's hand-rolled ~0%-cost rung.

## Architecture (live, current env)

- `model_type=gemma4`, 42 text layers: 35 sliding (head_dim 256) + 7 full (head_dim 512, idxs [5, 11, 17, 23, 29, 35, 41]).
- GQA: 8 q-heads / 2 kv-heads (group 4); sliding_window 512; num_kv_shared_layers 18.
- Ampere attn head_dim cap 256; head_dim 512 (full-attn) exceeds it -> no flashinfer tensor-core kernel.

## Version skew (why it is not a clean drop-in)

- pod (.venv, serving-adjacent): torch `2.11.0+cu130`.
- flashinfer-python pins torch `2.10.0+cu128`; ran here in an isolated venv (torch `2.10.0+cu128`). nvcc present: `True` (Cuda compilation tools, release 13.2, V13.2.51).
- flashinfer publishes no torch-2.11 AOT wheels (JIT-only) and carries an unresolved CUDA-12-cubin-vs-CUDA-13 incompatibility, so dropping it into the deployed vLLM 0.22.1rc1 / torch-2.11+cu130 stack is non-trivial. The M-invariance property measured is an algorithmic property of the split scheduler and transfers across the torch minor version.

## Honesty note

FlashInfer LOADS and RUNS (unlike TRT-LLM #502), so the census is real GPU evidence, not a blocked counterfactual. The decisive finding is NEGATIVE for the lane's premise: FlashInfer is NOT free byte-exact M-invariant out of the box -- its default split-KV decode is batch-variant (1292/2048 elements flip at L=8192). It exposes the invariance as an explicit `fixed_split_size` knob (0 flips), but that knob is its OWN slower path at M=1 (not free), and head_dim-512 full-attention layers have no tensor-core/fixed-split path at all -- so FlashInfer is not a cheaper route to the byte-exact 399.75 rung than the deployed split-KV stack. The useful positive: the deployed hand-rolled split-KV trick (lawine #496) is reproducing a stock, upstreamed FlashInfer primitive -- not a bespoke necessity. The last open engine/kernel thread of the #481 zoom-out closes.

## Public evidence used

- **lawine #496** (`42qroec1`) -- split-KV fixed-size 399.75 byte-exact rung; this card shows flashinfer's `fixed_split_size` is the same trick, native + first-class.
- **denken #498** (`djwaqs7o`) -- SGLang/FlashInfer fast-but-NOT-byte-exact; this card pins the mechanism (occupancy-based split count) and the override (fixed_split_size).
- **fern #502** (`sxi590tz`) -- TRT-LLM structurally blocked; FlashInfer is the last engine thread, now measured.
- **Morgan #481** -- ZOOM-OUT directive (look past vLLM+Triton for free byte-exact); the tax is engine-agnostic IEEE-754, confirmed again here.
- flashinfer `fixed_split_size` plan() docstring; SGLang deterministic-inference blog; Dao-AILab/flash-attention#2427 (head_dim 512 unsupported); vLLM#38918.
