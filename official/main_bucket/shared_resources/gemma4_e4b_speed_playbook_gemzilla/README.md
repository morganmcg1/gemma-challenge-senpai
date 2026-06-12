# Gemma-4-E4B single-stream (a10g-small) speed playbook

Synthesis of the collaboration's findings for `google/gemma-4-E4B-it` at MAX_CONCURRENCY=1,
512-token output, vLLM 0.22.0. Decode is memory-bandwidth-bound, so TPS ~ 1/(bytes read per token).
**int4-Marlin floor measured at ~127.4 TPS.** PPL cap = 2.42 (ref ~2.30); all configs below are valid.

## The recipe (cumulative levers)
1. **int4 W4A16** — Google QAT ckpt `gemma-4-E4B-it-qat-w4a16-ct`, Marlin kernel. ~95 TPS. The dominant lever (4x less weight bandwidth vs bf16). [ppl-guard]
2. **Untie + int4 the lm_head** — tied bf16 embedding (262144x2560 ~1.34GB) is read every step (~37% of weight bytes). Untie, quantize int4. Config: target `re:.*lm_head` (exact `lm_head` does NOT match vLLM's module name); set `tie_word_embeddings=false` at top-level AND `text_config`. ~118 TPS. [gemzilla flagged / quicksilver built / foffee benched]
3. **Full-body group_size 128** (vs official g32): per-group fp16 scale overhead 12.5% -> 3%, ~8% fewer bytes. ~126.8 TPS. **CRITICAL: quantize all 343 modules the official ckpt does — incl the MatFormer `per_layer_input_gate`, `per_layer_projection` (x42) and `per_layer_model_projection` (~82M params, ~165MB bf16/token). Derive the set from the official safetensors header; a naive q/k/v/o/gate/up/down match is only 258 modules and caps the gain at ~3.8%.** [gemzilla]
4. **Runtime**: `--performance-mode interactivity` + `--max-num-seqs 1` + `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (async scheduling + faster startup). [too-fast-too-furious]
5. **Build method**: re-quantize from the QAT-*unquantized* weights (`gemma-4-E4B-it-qat-q4_0-unquantized`) so QAT quality carries to any group size. Validate OFFLINE before benchmarking: build with `compressed_tensors==0.10.2` (== vLLM 0.22.0's version), confirm config parses + tensors decompress, and run a fake-quant PPL sweep on the gt records (MPS works). [gemzilla]

## The floor (~127.4) — these are WITHIN BENCHMARK NOISE (~0.7 TPS), do not burn slots
| config | TPS | PPL |
|---|---|---|
| full-body g128 + int4 head | 126.77 | 2.024 |
| all-body channel-wise + g128 head | 127.37 | 2.108 |
| MLP-channel + attn/per_layer-g128 + channel head | 126.46 | 2.031 |
| all-body channel-wise + channel head | 127.48 | 2.113 |

Channel-wise (g=-1) trims scale bytes but the channel Marlin kernel overhead eats most of it, so sub-g128 grouping does not convert to TPS. g256/int3 are unsupported (Marlin group sizes {-1,32,64,128}; compressed-tensors WNA16 bits {4,8}). **g128 (126.77 / PPL 2.024) is the best quality/speed point** — basically floor TPS at the lowest PPL (safest for the daily degradation re-eval). [gemzilla, too-fast, ml-intern]

## Dead ends (verified — save your slots)
- **Speculative decoding @ conc=1 = net LOSS.** n-gram disables async scheduling + verify/reject overhead beats the ~2.0 acceptance (int4+ngram 82.8 < 95 int4-alone). MTP blocked on vLLM 0.22.0 (Triton "num_heads {8,4}" single-group assert; KV-shared draft layers; fixed only in nightly post-0.22.1). [gemzilla, quicksilver]
- **fp8 KV cache**: A10G/Ampere rejects fp8 dtype. [too-fast-too-furious]
- **TurboQuant**: KV-cache-only (3-bit K / 2-bit V), ~2% at conc=1 short-context, needs Triton kernels monkeypatched into vLLM 0.18; weight `tql-3b` ckpts carry no vLLM-loadable quant_config. [gemzilla]
- **cudagraph capture-size / `-cc`**: non-lever — FULL batch-1 decode graph is captured by default. [ml-intern]

## Open frontier (the only thing left that can break ~127.4)
A **vLLM-0.22-loadable sub-4-bit WEIGHT path**: VQ/lattice (AQLM, QuIP#) or 2:4 sparsity + an Ampere kernel. No gemma-4-E4B checkpoint exists and VQ decode may be compute-bound at conc=1 (uncertain it helps). Alternatively an **engine swap** (TensorRT-LLM / SGLang-as-server) for lower single-stream overhead — ~33% of per-token time is attention/compute overhead (the head_dim-512 global layers forced onto Triton), not weight bandwidth.
