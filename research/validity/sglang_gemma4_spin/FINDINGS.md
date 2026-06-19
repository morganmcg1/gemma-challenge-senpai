# PR #745 — SGLang gemma-4-E4B spin (local A10G)

**Verdict: NO-GO.** SGLang cannot serve the challenge model `google/gemma-4-E4B-it`
(`Gemma4ForConditionalGeneration`, `model_type: gemma4`) on the official
`a10g-small` (NVIDIA A10G, sm_86) in **any** version. No tokens served; no TPS to
report. Baseline vLLM int4 floor (126.378 TPS) stands unchallenged.

W&B: `ppss56mt` (group `sglang-gemma4-spin`). Reproduce: `bash probe_sglang_gemma4.sh`.

## Two independent, empirically-verified blockers

### L1 — the harness-pinned stack (sglang 0.5.2 + transformers 5.9.0) cannot load gemma4
- `sglang[srt]==0.5.2` **hard-pins `transformers==4.56.1`** → unsatisfiable with the
  `transformers==5.9.0` that the challenge serves on. (uv resolver: "your
  requirements are unsatisfiable".)
- `transformers==4.56.1` **cannot parse `model_type: gemma4`** — `AutoConfig`
  raises *"Transformers does not recognize this architecture."* (`gemma4` is not in
  `CONFIG_MAPPING`; `gemma3n` is). gemma4 is strictly newer than this transformers.
- sglang 0.5.2's model registry ships `Gemma3nForConditionalGeneration` but **no
  `Gemma4` class**; the only generic fallback is `TransformersForCausalLM`, which is
  **text-only** — it cannot serve the required multimodal (text+image+audio) model,
  and would violate the "keep all modalities enabled" rule anyway.

### L2 — every gemma4-capable sglang (≥0.5.11) is Hopper/Blackwell-only, broken on A10G
- Native `Gemma4ForConditionalGeneration` support in sglang first appears in
  **v0.5.11** (transformers 5.6.0) and v0.5.13 (transformers 5.8.1). **No** sglang
  version pins `transformers==5.9.0`, so none matches the served baseline exactly.
- All of 0.5.11→0.5.13 **hard-depend on `flash-attn-4>=4.0.0b9`** (FlashAttention-4,
  a Blackwell-era pre-release) and pull `torch==2.11.0+cu130`.
- Installed 0.5.13 + transformers 5.8.1: torch CUDA works on the A10G and
  transformers 5.8.1 **does** parse gemma4 (`AutoConfig -> Gemma4Config`). **But**
  the prebuilt **`sgl_kernel` wheel ships `common_ops` only for `sm90/` and
  `sm100/` — there is no `sm86` binary.** On the A10G (cc 86) sgl_kernel cannot
  load its core ops ("Expected variant: SM86 … found sm100 only"), every model
  module import fails, and the model registry comes up **empty** — gemma4 is
  unservable. (Also `libnuma.so.1` is missing system-wide.)

## Why this lane is low-value even if the kernels were rebuilt for sm_86
- **Single-stream batch=1 is SGLang's weakest regime.** SGLang's architectural wins
  (RadixAttention prefix reuse, overlap scheduler) need concurrency / shared
  prefixes. At concurrency=1 with fresh prompts — exactly this challenge's scored
  regime (`MAX_CONCURRENCY=1`) — public benchmarks put SGLang ≈ vLLM (often slightly
  worse from RadixAttention trie overhead). No single-stream speedup is expected.
- **The Gemma-4 cookbook's headline latency lever is speculative decoding**
  (`--speculative-algorithm NEXTN` with the `-assistant` MTP draft) — which our
  vLLM frontier already exploits (the 489-TPS leaderboard stack). SGLang brings no
  new lever here, only a re-implementation.
- Packaging would also require re-proving greedy-identity + PPL + all-modalities on
  a different transformers minor (5.8.1 vs 5.9.0) and a from-source sm_86 kernel
  build — large risk for ~0 expected gain.

## Useful cookbook learnings (still worth keeping)
- For Gemma-4 on non-Hopper/Blackwell, **Triton is the auto-selected attention
  backend** (required for bidirectional image-token attention in prefill); `fa3`/`fa4`
  are Hopper/Blackwell-only and unavailable on sm_86.
- SGLang's quant menu (AWQ, GPTQ, gptq_marlin, awq_marlin, compressed-tensors,
  bitsandbytes, torchao int4wo) is in principle sm_86-compatible — but moot here
  because the runtime can't bootstrap on sm_86.
