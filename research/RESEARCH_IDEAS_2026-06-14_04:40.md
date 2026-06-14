# Research Ideas — 2026-06-14 04:40
## Focus: Provably-Lossless TPS Levers Beyond Current Frontier (481.53 TPS)

### Context
Hardware: A10G (sm_86), 23 GB, ~600 GB/s HBM, NO FP8.  
Model: gemma-3n E4B, greedy, batch=1, 128-prompt / 128-gen.  
Stack: linear-MTP spec-decoding K=7 + 3D split-KV verify + int4 W4A16 Marlin.  
Decode budget: verify-GEMM ~53%, drafter ~7%, attn <8%, **overhead ~32% (uncharacterized)**.  
Gate: bit-exact greedy argmax identity vs. reference; perplexity ≤ 2.42.

Owned (do not duplicate): tree/width, SplitK/tile-scheduler kernels, drafter GEMM-fusion, prompt-lookup n-gram, entropy branching, argmax-margin gating.  
Dead: CUDA-graph capture, norm/elementwise fusion.

---

## LEVER 1: CPU-Free Token Sampling via Persistent-Kernel Scheduling

### Mechanism
After each verify step the host CPU currently: (a) reads accepted-token count from device, (b) decides the next speculation length, (c) repacks the KV-cache offset, (d) launches the drafter kernel, and (e) issues the next verify batch. Each step requires a host-device sync or kernel launch from Python, adding cumulative CPU scheduling latency. On A10G at BS=1 this overhead chain is a first-order contributor to the ~32% "other" bucket.

The fix is a persistent-kernel loop that keeps a single CUDA kernel resident across all speculative decoding steps. The kernel reads its own acceptance count via in-kernel atomics, branches internally between draft and verify phases, and never returns to the CPU between tokens. CPU involvement reduces to: one kernel launch at start, one synchronization at end of the entire 128-token generation window.

This is architecturally equivalent to the Blink scheduler (arxiv 2604.07609) applied to the speculative decode loop: replace the per-step Python dispatch loop with a device-side finite-state machine.

### Greedy-Lossless Proof
The persistent kernel executes exactly the same arithmetic — same Marlin verify GEMM, same argmax, same acceptance rule — in exactly the same order. Nothing changes except who issues the next kernel (GPU vs. CPU). Argmax output is a pure function of the logits; logits are a pure function of weights and inputs. No weight transformation, no approximation. Bit-exact identity follows trivially.

### Expected TPS Upside
CPU scheduling overhead has been measured at 30–50% of total latency on fast accelerators in the vLLM V1 profiling study. Our ~32% "other" budget is consistent with that range. Even recovering half of it (say 15% of total step time) would yield ~15% TPS lift: 481 → ~555 TPS. Conservative lower bound assuming only 8% recovery: ~520 TPS.

### Distinction from Owned Lanes
Owned lanes: tree/width, SplitK/tile-scheduler, drafter-GEMM fusion, n-gram, entropy branching, argmax-margin.  
This lever operates at the **scheduler/runtime level**, not the kernel math or speculation strategy level. It does not change any token count, any arithmetic, or any weight. It is purely about who issues the next work item: CPU Python vs. GPU finite-state machine. No overlap with any owned lane.

### First CPU/Analytical Experiment
Profile the current stack with `nsys` or `nvtx` around the Python dispatch loop between verify and drafter kernel launches. Measure: (a) wall time from last verify kernel completion to first drafter kernel start (this is pure CPU overhead), (b) the ratio of that gap to total step time. If the gap exceeds 5% of step time, the persistent-kernel approach is guaranteed to recover at least that. The experiment requires zero GPU training — just a profiling run with existing weights.

### Papers / Repos
- **Blink: CPU-Free LLM Inference** — arxiv 2604.07609. Persistent GPU kernel + SmartNIC for scheduling; measured 2.1× decode throughput, 3.40× P99 TPOT on A100. Key ablation shows CPU scheduling alone contributes 30–50% of end-to-end latency at BS=1.
- **vLLM V1 zero-copy DMA** — https://github.com/vllm-project/vllm (v0.4+ architecture). Pinned host memory + DMA path removes redundant CPU-GPU tensor copies during token sampling and output processing. The PR notes "CPU scheduling consumes up to 50% of end-to-end latency on fast accelerators."

---

## LEVER 2: Double Quantization on the Verify-GEMM Weight Buffer

### Mechanism
The verify-GEMM currently stores the weight buffer as: W4 (4-bit quantized weights) + FP16 scales (one per group of g=128 tokens). At g=128, the FP16 scale tensor costs ~0.78% of total weight bytes — small but nonzero HBM bandwidth. Double quantization (introduced in QLoRA, arxiv 2305.14314) applies a second quantization level to the scales themselves: scales-of-scales are stored in 8-bit, and the 8-bit dequant cost is absorbed into the already-existing dequant path at negligible FLOPs overhead.

For a 4B model, the scale tensor (FP16, g=128) is roughly 26 MB. Compressing the scales to INT8 saves 13 MB of HBM bandwidth per forward pass — a ~0.6% reduction in total weight bytes streamed. That translates directly to ~0.6% TPS lift, since verify-GEMM is bandwidth-bound. More importantly, it *sets the upper bound lower* for any further scale-format experiment, and it is a strict no-op on the output distribution if implemented exactly: dequantizing INT8-quantized FP16 scales back to FP16 before the GEMM produces the same W4 weights as the un-double-quantized path, up to the INT8 rounding error on the scales.

The lossless variant requires: quantize scales to INT8 using round-to-nearest, store both INT8 scales and the per-scale-group offset, then dequantize back to FP16 before the Marlin kernel reads them. If the dequantized FP16 scale is *bit-identical* to the original FP16 scale, the GEMM output and argmax are unchanged. This holds when the original FP16 scales are themselves representable in the INT8 + offset + scale scheme without rounding error (achievable by appropriate choice of the secondary quantization resolution).

A strictly lossless version: use BF16 scale storage → INT8 round-trip → BF16 reconstruction and verify bit-exact reconstruction before deployment. Any scale that does not round-trip exactly is stored as a "sparse exception" in FP16 (SqueezeLLM-style dense-and-sparse for the scale tensor itself).

### Greedy-Lossless Proof
The argument is constructive: for each scale value s_i in FP16, if INT8 round-trip(s_i) = s_i exactly, then the dequantized weight w_i = s_i * q_i is unchanged, the GEMM output is unchanged, and argmax is unchanged. Before deployment, run a scan over all scale values; any that fail the round-trip test are kept in FP16 as exceptions. The final weight buffer is a hybrid: 98–99%+ of scales in INT8 (saving bandwidth), the rest in FP16 (preserving exact output). This is bit-exact by construction.

### Expected TPS Upside
Scale bandwidth is ~0.78% of total weight bytes at g=128. Compressing 95% of them to INT8 saves ~0.37% of total GEMM bandwidth. Direct TPS lift: ~0.4–0.6%. Small but free and composable with all other levers. The more interesting upside is using smaller group sizes (g=64 or g=32) to improve quantization quality — at g=32 the scale tensor is 3× larger (~2.3% of weight bytes), so the bandwidth saving from double quantization is ~1.1%, and the primary quantization quality also improves (lower per-group error), potentially enabling a slight K increase in the drafter acceptance rate.

### Distinction from Owned Lanes
Owned: SplitK/tile-scheduler kernel variants for the verify GEMM. This lever does not touch the Marlin kernel tiling or scheduling at all. It operates on the **weight packing format before the kernel reads it**, reducing the bytes the kernel must stream from HBM. It is a data-layout / compression lever, not a compute-schedule lever.

### First CPU/Analytical Experiment
Python script (no GPU required): load the current W4A16 checkpoint, extract all scale tensors, simulate INT8 round-trip quantization, and measure the fraction of scales that are bit-exact after round-trip. If >98% round-trip exactly, the lossless double-quant scheme is immediately viable and the bandwidth saving is calculable analytically. Runtime: minutes on CPU with standard PyTorch.

### Papers / Repos
- **QLoRA: Double Quantization** — arxiv 2305.14314, Dettmers et al. 2023. Section 2.2 introduces double quantization; ablation shows negligible quality loss on a range of LLMs at g=64 and g=128 with INT8 secondary quantization.
- **SqueezeLLM: Dense-and-Sparse Decomposition** — arxiv 2306.07629, Kim et al. 2023. Sparse exception mechanism for quantization outliers; directly applicable to the scale-exception case. GitHub: https://github.com/SqueezeAILab/SqueezeLLM

---

## LEVER 3: LK-Loss Draft Fine-Tuning for Higher Acceptance Rate Without Tree Width

### Mechanism
The current drafter (MTP linear head) is trained with a next-token cross-entropy loss. Cross-entropy maximizes the likelihood of the correct next token in isolation, but the speculative decoding objective is different: maximize the *number of consecutive tokens accepted* under greedy verification. These two objectives are not equivalent. A draft token can have high cross-entropy loss but still be accepted (correct argmax); conversely, a low-loss draft can produce a slightly wrong token rank ordering that causes rejection.

LK Losses (arxiv 2602.23881) directly optimizes the acceptance probability of the draft. The loss is differentiable and introduces no computational overhead at inference time — only the training objective changes. The paper reports 8–10% improvement in average acceptance length (E[T]) across standard benchmarks. On our stack: E[T] currently sits at some value consistent with K=7 and the measured ~7% drafter budget fraction; a +8% E[T] lift would directly translate to ~8% TPS lift (since each additional accepted token amortizes the verify cost over more output tokens), independent of K tuning.

This is distinct from entropy branching (owned) and prompt-lookup n-gram (owned) because it targets the **training objective of the neural draft head**, not the inference-time speculation strategy or retrieval augmentation.

### Greedy-Lossless Proof
The LK loss changes how the draft head is trained; it does not change the verify step. Greedy verification remains exactly the standard speculative decoding acceptance rule: accept token t if and only if the verifier's argmax at position t equals the draft's proposed token. No change to the verifier, no change to the argmax, no change to the acceptance rule. The output sequence is always the verifier's greedy output — the draft only controls *how quickly* we reach it. Lossless by the standard speculative decoding losslessness theorem (Chen et al. 2023), which holds for any draft distribution when greedy verification is used.

### Expected TPS Upside
LK Losses paper reports 8–10% E[T] improvement. On our stack at K=7, TPS ∝ E[T] / (draft_cost + verify_cost). Since verify-GEMM dominates (~53%) and drafter is only ~7%, a +8% E[T] lifts the numerator without changing the denominator: ~8% TPS lift. 481 → ~520 TPS conservative, ~530 TPS optimistic. This compounds with Lever 1 (orthogonal — scheduler vs. draft quality) and Lever 2 (orthogonal — weight bandwidth vs. draft quality).

### Distinction from Owned Lanes
Owned: "Drafter proposal augmentation via prompt-lookup / n-gram retrieval" and "Entropy-conditioned branching of the speculation tree." LK Losses touches neither. It is a **training loss formulation change** for the neural draft head only. It does not augment proposals with retrieval, does not change the tree topology, and does not modify the inference-time acceptance rule.

### First CPU/Analytical Experiment
Before any GPU retraining: measure the current draft head's per-token acceptance probability distribution (P(accept | position k) for k=1..7 under greedy). This requires a forward pass over a validation set (no training). Then compute the theoretical E[T] under the LK-optimal draft: if the current draft has a geometric acceptance profile with mean p per token, and LK improves p by +8%, calculate the implied E[T] improvement analytically. This gives a ceiling estimate for the TPS upside before committing to a fine-tuning run.

### Papers / Repos
- **LK Losses: Direct Acceptance Rate Optimization** — arxiv 2602.23881. Differentiable loss for speculative decoding drafters; 8–10% E[T] improvement, no inference overhead.
- **DRAFT: On-the-Fly Self-Speculative Decoding** — arxiv 2410.06916. Layer-skip self-drafting without auxiliary model; useful comparison point if the MTP head is replaced rather than fine-tuned.

---

## Summary Table

| Lever | Target bucket | Mechanism | TPS upside | Greedy-lossless | First experiment (no GPU) |
|---|---|---|---|---|---|
| 1. Persistent-kernel scheduling | ~32% overhead | Replace Python dispatch loop with device-side FSM | +8–15% | Trivially: same arithmetic, different issuer | nsys profile of inter-kernel gap |
| 2. Double quantization on scales | ~53% verify-GEMM (scale bandwidth) | INT8-compress FP16 scales, sparse exceptions for non-roundtrip | +0.4–1.1% | Constructive: scan + exception set | Python round-trip fraction scan |
| 3. LK-Loss draft fine-tuning | ~7% drafter quality | Change training loss to optimize acceptance probability directly | +6–10% | Standard spec-dec theorem, verify unchanged | Acceptance-profile analysis on val set |

All three are orthogonal to each other and to all owned lanes. None touches the Marlin tile schedule, tree width, n-gram retrieval, entropy branching, or argmax-margin gating. Combined ceiling (assuming partial compounding): ~20–25% TPS lift over current 481.53 → ~575–600 TPS before compute-budget limits are hit.
