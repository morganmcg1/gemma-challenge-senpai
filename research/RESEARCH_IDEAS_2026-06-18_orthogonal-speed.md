# Orthogonal Speed Levers — HF Fast Gemma Challenge
**Date:** 2026-06-18  
**Research scope:** Draft-free / self-speculative decoding, int4 kernels, alternative serving stacks, 2024–2026 single-stream latency literature  
**Competition target:** google/gemma-4-E4B-it, vLLM 0.22.1rc1.dev307, single A10G (sm_86, 24GB)  
**Locked submission baseline:** 126.378 official TPS / PPL 2.019 / 128/128 greedy match  
**Required improvement:** +10 TPS → ≥136.378 TPS  
**Hard gates:** PPL ≤ 2.42 · strict byte-exact greedy-token-identity (128/128) · int4 mandatory  

---

## Research Method

Thirteen Exa/semantic-scholar searches across two sessions, reading methodology sections of key papers and vLLM PRs/issues. Citation graph traversal from anchor papers (Marlin, EAGLE3, Lookahead, SAM, ConfLayers, DFlash, SGLang). GitHub source reading for vLLM 0.22.x speculative decoding backend implementations.

---

## Idea 1 — Ngram / Prompt-Lookup Speculative Decoding

**(a) What it is:** vLLM-native drafter-free speculative decoding that extracts n-gram candidates directly from the prompt/KV context (no auxiliary model). The speculator re-uses tokens from the visible context as a lookahead draft; if the verifier accepts them, the step processes multiple tokens per forward pass.

**(b) Expected TPS upside:** +5–15% on chat/instruction prompts with repeated phrases; lower on fully novel generation. Acceptance rate varies strongly with prompt repetition. Competition prompts (IF benchmark-style) tend to have moderate repetition, suggesting ~+8% is a reasonable central estimate.

**(c) Greedy-identity-safe by construction?** YES. Speculative decoding with exact (lossless) verification is mathematically equivalent to greedy decoding: the verifier only accepts a draft token if the argmax of the full model agrees. Rejection resets to the correct token. No stochastic acceptance used here.

**(d) Implementation path — exact flags for vLLM 0.22.1rc1:**

```bash
# Method A: compact JSON on CLI
--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":2,"prompt_lookup_max":6}'

# Method B: YAML server config (if using config file)
speculative_config:
  method: ngram
  num_speculative_tokens: 5
  prompt_lookup_min: 2
  prompt_lookup_max: 6
```

Key tuning levers:
- `num_speculative_tokens`: Try 4, 5, 6, 8. Larger drafts increase upside IF acceptance rate is high; diminishing returns past the typical sequence repeat length.
- `prompt_lookup_min`: 2 is minimum safe. Setting 1 causes over-matching on single tokens. 2–3 is optimal for instruction-following.
- `prompt_lookup_max`: 5–8. Larger means more aggressive search; marginal compute cost is negligible.

vLLM V1 engine compatibility: PR #12193 (merged 2025-02-16, title "[V1][Spec Decode] Ngram Spec Decode") fixed V1 support. PR #15348 ("[V1][Spec Decode] Respect prompt_lookup_max") also merged. Issue #16883 (ngram broken in V1 for 0.8.x) is CLOSED. vLLM 0.22.1rc1.dev307 is significantly newer and should carry both fixes.

Known issue #40875 (tool-call token corruption with prompt_lookup_min=1): not applicable to this competition (no tool-call generation expected).

**(e) #1 risk:** Benchmark prompts may be insufficiently repetitive, yielding near-zero acceptance rate and no speedup (possibly slight overhead). Measure acceptance_rate from W&B or vLLM logs on a few sample prompts before committing to a full benchmark run.

---

## Idea 2 — Suffix Decoding (Arctic Inference / vLLM-integrated)

**(a) What it is:** A more powerful drafter-free speculative method that builds a suffix tree over the entire KV cache / generation history (not just the current prompt), enabling longer exact matches and higher acceptance rates than n-gram lookup. Integrated into vLLM via PR #25784 (merged November 2025) from Snowflake AI Research's Arctic Inference library.

**(b) Expected TPS upside:** +8–18% on repetition-rich chat completions; the suffix tree enables deeper matches than n-gram, so acceptance rate is systematically higher. Single-stream batch-1 is the ideal regime because the suffix tree does not compete with batching overhead.

**(c) Greedy-identity-safe by construction?** YES. Uses the same lossless exact-verification protocol as ngram speculative decoding. Acceptance is mathematically equivalent to greedy argmax.

**(d) Implementation path — exact flags for vLLM 0.22.1rc1:**

```bash
--speculative-config '{"method":"suffix","num_speculative_tokens":5}'
```

The `method:"suffix"` backend was added via vLLM PR #25784 and is available in vLLM main after November 2025. vLLM 0.22.1rc1.dev307 (June 2026 competition) should carry this. If the method name differs in 0.22.x, consult `python -c "from vllm.config import SpeculativeConfig; help(SpeculativeConfig)"`.

Additional tuning: `num_speculative_tokens` 4–8 same sweep as ngram. Arctic Inference's original implementation defaults to 5 with good results on conversational benchmarks.

**(e) #1 risk:** Method name or API may differ between Arctic Inference's standalone integration and the vLLM-upstreamed version in 0.22.1rc1; verify method availability by checking vLLM version changelog or help output before running benchmark. If unavailable, fall back to Idea 1.

---

## Idea 3 — CUDA Graph Mode Audit + Decode-Path CPU Overhead Reduction

**(a) What it is:** vLLM uses CUDA graphs to eliminate Python-layer overhead on the decode step (the tight loop). Default settings in vLLM batch `enforce_eager=False` (CUDA graphs enabled), but the capture configuration (full vs piecewise, max batch size covered, cudagraph pool allocation) has meaningful tuning knobs that are often left at defaults. On single-stream batch-1, per-step CPU overhead from the scheduler, tokenizer callbacks, and streaming output is proportionally large relative to the compute kernel.

**(b) Expected TPS upside:** +3–8%. The gain is smaller than speculative decoding because the current stack likely already uses CUDA graphs, but aggressive tuning of cudagraph pool sizes and disabling fallback paths can reduce step latency meaningfully at batch-1.

**(c) Greedy-identity-safe by construction?** YES. CUDA graph mode does not affect the numeric outputs of the forward pass; it only changes when Python/CUDA synchronization happens. No quality risk.

**(d) Implementation path:**

```bash
# Confirm CUDA graphs are active (default is True when enforce_eager=False)
# Tune cudagraph pool allocation
export VLLM_GRAPH_RESERVED_MEM=0.05      # fraction of VRAM reserved for graph pool
export VLLM_GRAPH_PADDING_SIZE=1         # reduce unnecessary padding captures

# Piecewise CUDA graph (if supported in 0.22.1rc1)
# Some vLLM versions expose piecewise capture for multi-module models
# Check: grep -r "piecewise" /path/to/vllm/ for flag names

# Single-batch-specific: ensure batch-1 is captured explicitly
# Some vLLM versions skip capturing small batch sizes for VRAM efficiency
export VLLM_CUDAGRAPH_MIN_BS=1
```

Auditing steps for the student:
1. Enable vLLM metrics logging and confirm `cuda_graph_capture_success` is True.
2. Check `--enforce-eager False` is set (should be default).
3. Profile with `nsys profile` or `torch.profiler` to identify if CPU synchronization points remain in the decode loop.
4. If the model uses prefix caching, verify CUDA graph is not disabled by it.

**(e) #1 risk:** CUDA graphs may already be fully optimized in the current config; the gain may be <1% or noise-level. This is a diagnostic/audit idea rather than a guaranteed win. Worth running as a cheap confirmation pass.

---

## Idea 4 — ConfLayers Self-Speculative Decoding (Confidence-Based Layer Skip)

**Paper:** "ConfLayers: Confidence-Based Adaptive Layer Skipping for Self-Speculative Decoding" (arxiv 2604.14612, April 2026)

**(a) What it is:** A plug-and-play self-speculative decoding method where the model runs a shallow "draft" pass by skipping later transformer layers when an early-exit confidence threshold is met, then runs the full model as verifier only on tokens where the draft was accepted. No auxiliary model trained. No weight changes. The draft is the same model with a subset of layers active.

**(b) Expected TPS upside:** Up to 1.4x (40%) reported on standard chat benchmarks in the paper. This is the single highest upside idea that requires no drafter training and no stack change. The improvement is strongest in single-stream batch-1 where each verify step runs the full model serially.

**(c) Greedy-identity-safe by construction?** YES with correct configuration. The full model is always used as the verifier; draft tokens that don't match the verifier's argmax are rejected and replaced. The output is mathematically identical to greedy decoding from the full model. The confidence threshold only affects efficiency (how often the full model is invoked), not correctness.

**(d) Implementation path:**

The GitHub repo (from arxiv 2604.14612) provides a plug-and-play integration. Since this is not natively in vLLM 0.22.x, it requires patching the model forward pass:

1. Install ConfLayers or copy the layer-skip forward pass logic from their GitHub.
2. Monkey-patch the Gemma-4 forward function in vLLM's model registry to insert confidence-based early-exit logic at the layer level.
3. The confidence gate uses a small MLP head on the hidden state at each candidate exit layer to predict whether the full-model argmax would agree. This head can be zero-shot (using softmax entropy as the confidence signal) or trained (small overhead).

Key hyperparameters:
- `exit_layer`: Which layer to attempt early exit from (typically layers 60–80% depth for Gemma-4E4B).
- `confidence_threshold`: Controls draft acceptance rate vs quality.
- Zero-shot entropy threshold requires no training; use `threshold=0.9` (high confidence) to start.

**(e) #1 risk:** Patching vLLM's model forward pass for a quantized int4 model is non-trivial; the quantized weight representations complicate layer indexing and early-exit tensor shapes. This is a medium-complexity implementation that could take 1–2 days to get right. The 1.4x claim is on bf16 models; int4 may reduce the gain if the quantized model has different confidence calibration. Test on a few prompts with confidence logging before committing to full benchmark.

---

## Idea 5 — SGLang + FlashInfer Stack Swap for Single-Stream int4

**Paper:** "Efficient LLM Scheduling by Learning to Schedule" (NeurIPS 2024); SGLang: "Fast and Expressive LLM Inference with RadixAttention" (OSDI 2024-adjacent, arxiv 2312.07104)

**(a) What it is:** Replace vLLM with SGLang as the serving engine. SGLang uses FlashInfer as its attention backend (instead of FlashAttention-2/3), enabling per-sequence radix attention caching and a more aggressively optimized single-stream decode path. FlashInfer's batched decode kernel is specifically tuned for batch-1 through batch-8.

**(b) Expected TPS upside:** +10–20% on single-stream int4 decode, based on SGLang's published throughput numbers vs vLLM. RadixAttention benefits compound when there is prefix reuse across the benchmark prompts. FlashInfer's int4 decode kernels for Ampere (sm_86) are separate from Hopper-optimized paths and should be effective on the A10G.

**(c) Greedy-identity-safe by construction?** YES if configured for greedy decoding (`temperature=0, top_p=1`). The underlying forward pass is numerically equivalent. However: SGLang's int4 implementation may use different quantization backends (it primarily supports AWQ and GPTQ), and the competition model uses int4 QAT weights. Verify that SGLang can load the gemma-4-E4B-it int4 QAT format without accuracy degradation.

**(d) Implementation path:**

```bash
# Install SGLang (requires separate environment or Dockerfile)
pip install sglang[all]

# Launch SGLang server for gemma-4-E4B-it with int4
python -m sglang.launch_server \
  --model google/gemma-4-E4B-it \
  --quantization awq \          # or gptq; verify QAT format compatibility
  --dtype auto \
  --tp 1 \
  --mem-fraction-static 0.9

# Or via HuggingFace job with SGLang container
# SGLang 0.3.x Docker: lmsys/sglang:latest
```

Key investigation before full run:
- Verify SGLang + FlashInfer supports gemma-4-E4B-it int4 QAT weight format.
- Run greedy identity check (128/128) vs bf16 base before any benchmark submission.
- SGLang version: 0.3.x+ has FlashInfer integrated; check that Ampere sm_86 is not a fallback path.

**(e) #1 risk:** The competition's evaluation harness may be locked to vLLM; switching stacks may require special approval or may simply not be permitted by competition rules. Before investing in SGLang integration, confirm that the HF competition submission format allows non-vLLM serving engines. If vLLM is mandatory, this idea is blocked.

---

## Idea 6 — Lookahead / Jacobi Decoding

**Paper:** "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding" (ICML 2024); GitHub: hao-ai-lab/LookaheadDecoding

**(a) What it is:** Runs multiple parallel Jacobi iterations on a "lookahead window" of future token positions simultaneously, using a verification step to accept correct lookahead tokens. The parallelism exploits the fact that modern GPUs are significantly underutilized at batch-1 decode. Requires no auxiliary model; operates on the same model weights.

**(b) Expected TPS upside:** 1.5–2.3x reported on MT-bench (2.5x max observed), 1.8x average. These are model and prompt dependent. At batch-1 where the GPU is memory-bandwidth-bound, the parallel Jacobi steps amortize the per-step kernel launch overhead.

**(c) Greedy-identity-safe by construction?** YES for the exact (lossless) variant. The verification step accepts only tokens where the speculated positions match the model's argmax. Lookahead decoding paper Section 3 proves distribution equivalence for exact verification.

**(d) Implementation path:**

Lookahead is not natively in vLLM 0.22.x. Two options:
1. Use the hao-ai-lab/LookaheadDecoding library directly with HuggingFace `generate()` API (bypasses vLLM). Simpler but loses vLLM's batching/scheduling overhead reduction.
2. Port the lookahead speculative backend into vLLM by registering a custom speculative decoding worker. This requires more effort but keeps the vLLM scaffolding.

Key hyperparameters (from paper):
- `window_size W`: Number of parallel Jacobi positions. Paper uses W=5 as default.
- `ngram_size n`: History n-grams maintained in the lookahead cache. Default n=7.
- Both are tunable; W=7, n=8 tends to improve acceptance on code/math.

```python
# hao-ai-lab/LookaheadDecoding quick-start
from lookahead.decoding import LookaheadDecoding
ld = LookaheadDecoding(model, tokenizer, window_size=7, ngram_size=8)
output = ld.generate(prompt, max_new_tokens=512)
```

**(e) #1 risk:** Not in vLLM natively; requires either bypassing vLLM entirely (losing competition-format compatibility if vLLM is mandatory) or a substantial vLLM fork/patch. Implementation complexity is high for a competition context. Lower priority than Ideas 1/2 which are vLLM-native and lower-risk.

---

## Idea 7 — SAM Decoding (Suffix Automaton Method)

**Paper:** "SAM Decoding: Speculative Decoding via Suffix Automaton" (arxiv 2411.10666, NeurIPS 2024 workshop)

**(a) What it is:** An O(1) retrieval-based speculative decoding method using a suffix automaton data structure built over the KV cache. The automaton enables finding the longest suffix match in the current context in constant time (vs O(n) linear scan for n-gram lookup), yielding higher acceptance rates at the same draft budget.

**(b) Expected TPS upside:** Paper reports 18%+ speedup over other retrieval-based speculative decoding methods (including standard n-gram); ~+15–25% over no speculative decoding baseline.

**(c) Greedy-identity-safe by construction?** YES. Same lossless verification as n-gram; the suffix automaton only changes how draft candidates are selected, not how they are verified.

**(d) Implementation path:** SAM is NOT natively in vLLM 0.22.x as of this research date. The suffix-tree-based backend from Snowflake Arctic (Idea 2 / method:"suffix") is a related implementation that may cover similar functionality. If `method:"suffix"` in vLLM uses a suffix structure, it is the vLLM-integrated version of this idea. The original SAM paper implementation is standalone and would require integration.

For competition context: test `method:"suffix"` first (Idea 2). If that method name is not available in 0.22.1rc1, the SAM paper's code could be adapted to register as a custom speculative decoding backend.

**(e) #1 risk:** Functional overlap with Idea 2 (suffix decoding); if suffix is available in 0.22.1rc1, SAM standalone implementation is redundant. Treat as backup to Idea 2.

---

## Idea 8 — TensorRT-LLM Stack Swap

**Source:** NVIDIA TRT-LLM blog 2024-02-21; TRT-LLM GitHub; Gemma support tracking issue

**(a) What it is:** NVIDIA's production inference compiler (TensorRT-LLM) compiles model weights + architecture into a highly optimized engine with XQA kernels, fused attention, and native int4 AWQ support on Ampere. Reported to outperform vLLM for single-stream decode on NVIDIA hardware.

**(b) Expected TPS upside:** +10–25% on single-stream from XQA kernel gains and fused decode operations. The gain is model and batch-size dependent; A10G (sm_86) uses the RTX/Ampere code path.

**(c) Greedy-identity-safe by construction?** YES if engine is compiled for greedy decoding and the same weight format. TRT-LLM engine compilation is deterministic for fixed precision.

**(d) Implementation path:**

```bash
# Install TRT-LLM (Docker preferred)
docker run --gpus all --rm -it \
  nvcr.io/nvidia/tensorrt_llm:latest bash

# Build Gemma-4 int4 engine
python -m tensorrt_llm.examples.gemma.convert_checkpoint \
  --model_dir google/gemma-4-E4B-it \
  --output_dir ./gemma4e4b_trtllm \
  --dtype float16 \
  --use_weight_only \
  --weight_only_precision int4

trtllm-build \
  --checkpoint_dir ./gemma4e4b_trtllm \
  --output_dir ./gemma4e4b_engine \
  --gemm_plugin float16 \
  --max_batch_size 1 \
  --max_input_len 4096 \
  --max_output_len 1024
```

Key risk: TRT-LLM uses int4 AWQ internally; the competition model uses int4 QAT (quantization-aware training). The weight format may not be directly loadable. A conversion step may be needed.

**(e) #1 risk:** Competition submission format may require vLLM; if so, TRT-LLM is not applicable. Also, int4 QAT → TRT-LLM format conversion is non-trivial and may introduce accuracy drift that fails the greedy identity gate. Validate format compatibility before investing in this path.

---

## Idea 9 — Marlin Kernel Forcing + W4A8 Investigation

**Paper:** "Marlin: A Mixed-Precision Matrix Multiplication Kernel for Ampere" (arxiv 2408.11743)

**(a) What it is:** Marlin is the native Ampere (sm_86) int4 mixed-precision GEMM kernel. It is already auto-selected by vLLM for A10G W4A16 workloads. This idea is about auditing whether the kernel is actually being selected in every layer and investigating W4A8 (int4 weights + int8 activations) as a potential speedup over W4A16.

**(b) Expected TPS upside:** W4A8 can be 10–20% faster than W4A16 for pure GEMM throughput on Ampere, because int8 MAC units have 2x throughput over fp16 MAC units. Whether this translates to end-to-end TPS depends on whether GEMM is the bottleneck.

**(c) Greedy-identity-safe by construction?** W4A16 (current) → YES. W4A8 introduces activation quantization; accuracy impact must be verified. Since the competition model is int4 QAT (weights), the activations are currently bf16. Switching to W4A8 would require activation quantization which changes numerics and WILL affect greedy token output. This path requires a full greedy identity check before use.

**(d) Implementation path for W4A16 audit:**

```bash
# Check Marlin is selected in the current config
python -c "
from vllm import LLM
llm = LLM('google/gemma-4-E4B-it', quantization='gptq_marlin')
print(llm.llm_engine.model_executor.driver_worker.model_runner.model)
"
# Look for MarlinLinear in the printed model tree
```

vLLM issue #38063 (int4 scalar_types unsupported) affects W4A8 only; W4A16 (our use case) is confirmed fine.

**(e) #1 risk:** If Marlin is already active (likely), the W4A16 path cannot be improved further from kernel selection. W4A8 would require re-quantizing activations and is outside the "zero quality risk" category; it becomes an accuracy experiment, not a pure speed experiment.

---

## Summary Ranking

| Rank | Idea | Greedy-safe | vLLM-native | Est. TPS gain | Complexity | Priority |
|------|------|-------------|-------------|---------------|------------|----------|
| 1 | Ngram/Prompt-Lookup SD | YES | YES | +5–15% | Low | **ASSIGN NOW** |
| 2 | Suffix Decoding | YES | YES (verify) | +8–18% | Low | **ASSIGN NOW** |
| 3 | CUDA graph audit | YES | YES | +3–8% | Low | **ASSIGN NOW** |
| 4 | ConfLayers self-speculative | YES | Patch needed | up to +40% | Medium | Worth testing |
| 5 | SGLang + FlashInfer swap | YES (verify) | NO | +10–20% | High | Block-check format first |
| 6 | Lookahead/Jacobi | YES | Patch needed | +50–130% | High | High upside, high effort |
| 7 | SAM decoding | YES | NO (overlap 2) | +15–25% | Medium | Try Idea 2 first |
| 8 | TRT-LLM stack swap | YES (verify) | NO | +10–25% | High | Block-check format first |
| 9 | Marlin W4A16 audit | YES | YES | +0–5% | Low | Diagnostic only |

---

## Recommended Assignments

**Student Assignment 1 — ngram + suffix speculative decoding sweep**

Test both ngram (`method:"ngram"`) and suffix (`method:"suffix"`) with a grid over `num_speculative_tokens` ∈ {4, 5, 6, 8}. Measure acceptance_rate and TPS. If suffix method is not available in the build, fall back to ngram. Expected wall time: 2–3 hours with 4 benchmark runs.

**Student Assignment 2 — ConfLayers self-speculative patch**

Fork the model forward pass, instrument layer-exit confidence gates using hidden-state entropy (zero-shot, no training), and verify 128/128 greedy identity. Run a TPS benchmark with W=3–5 exit layers tested. Expected wall time: 3–4 hours for integration + benchmark.

**Student Assignment 3 — CUDA graph audit + SGLang compatibility check**

Audit CUDA graph capture status in current config. Separately, test SGLang serving of gemma-4-E4B-it int4 and confirm format compatibility + greedy identity. This is a fork: CUDA graph audit is low-risk; SGLang check is exploratory. Report both outcomes.

---

## vLLM 0.22.1rc1.dev307 Compatibility Summary

| Method | Status | PR/Issue | Notes |
|--------|--------|----------|-------|
| ngram speculative | CONFIRMED FIXED | PR #12193 (Feb 2025), Issue #16883 CLOSED | V1 engine support added |
| suffix speculative | LIKELY AVAILABLE | PR #25784 (Nov 2025) | Verify method name in 0.22.1 |
| DFlash | AVAILABLE but NO E4B DRAFT | PR #36847 (Mar 2026) | No draft model for gemma-4-E4B-it |
| EAGLE3 | AVAILABLE in vLLM | Merged 2025-2026 | Requires custom trained draft |
| Marlin W4A16 | ACTIVE (auto-selected) | Default kernel for sm_86 | No change needed |
| Lookahead | NOT IN vLLM | External: hao-ai-lab | Requires integration patch |
| SAM | NOT IN vLLM | Overlap with suffix | Test suffix method first |
| ConfLayers | NOT IN vLLM | arxiv 2604.14612 (Apr 2026) | Requires forward pass patch |

---

## Key References

1. **Ngram speculative decoding in vLLM** — vLLM PR #12193 "[V1][Spec Decode] Ngram Spec Decode", merged 2025-02-16. https://github.com/vllm-project/vllm/pull/12193
2. **Suffix Decoding (Arctic Inference)** — vLLM PR #25784, merged November 2025. Snowflake AI Research. https://github.com/vllm-project/vllm/pull/25784
3. **SAM Decoding** — Xu et al., "SAM Decoding: Speculative Decoding via Suffix Automaton" (NeurIPS 2024 workshop), arxiv:2411.10666. https://arxiv.org/abs/2411.10666
4. **Lookahead Decoding** — Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding" (ICML 2024). https://arxiv.org/abs/2402.02057 · GitHub: https://github.com/hao-ai-lab/LookaheadDecoding
5. **ConfLayers** — "ConfLayers: Confidence-Based Adaptive Layer Skipping for Self-Speculative Decoding" (April 2026), arxiv:2604.14612. https://arxiv.org/abs/2604.14612
6. **DFlash** — arxiv:2602.06036, vLLM PR #36847 (March 2026). Supported draft models: gemma-4-31B-it, gemma-4-26B-A4B-it. NOT gemma-4-E4B-it.
7. **EAGLE3** — Xu et al. "EAGLE-3: Scaling Up Inference Acceleration of Large Language Models via Training-Time Test" (March 2025), arxiv:2503.01840. 6.5x speedup; requires trained draft. https://arxiv.org/abs/2503.01840
8. **Marlin kernel** — Frantar & Alistarh, "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models" (arxiv:2408.11743). Native sm_86 Ampere int4 GEMM. https://arxiv.org/abs/2408.11743
9. **SGLang + FlashInfer** — Zheng et al., "Efficiently Programming Large Language Models using SGLang" (OSDI 2025-adjacent, NeurIPS 2024). Up to 6.4x throughput vs SOTA. https://arxiv.org/abs/2312.07104
10. **TRT-LLM int4 AWQ for Gemma** — NVIDIA blog "Speed up Gemma with TensorRT-LLM" (2024-02-21). https://developer.nvidia.com/blog/speed-up-inference-for-large-language-models-with-tensorrt-llm/
11. **vLLM V1 prompt_lookup_max fix** — vLLM PR #15348 "[V1][Spec Decode] Respect prompt_lookup_max". https://github.com/vllm-project/vllm/pull/15348
12. **Marlin W4A8 issue** — vLLM Issue #38063 "scalar_types.int4 unsupported in Marlin for W4A8". W4A16 confirmed fine. https://github.com/vllm-project/vllm/issues/38063
