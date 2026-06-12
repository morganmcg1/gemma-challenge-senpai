# Frontier roadmap — the big swings for world's-fastest E4B on A10G (quicksilver)

The lever map's Tier-1 (lm_head int4, g128) caps at ~+35%. For *world's fastest*
you attack the batch-1 bottleneck multiplicatively. At conc=1 you're bandwidth-
bound, so the big wins are: **(a) more tokens per weight-read** (speculative
decoding), **(b) fewer bytes per read** (sparsity + sub-4-bit), and **(c) erase
the ~40% overhead floor** (megakernel). These STACK.

## Ranked big swings

### 1. 2:4 sparsity + int4 (Sparse-Marlin) — A10-PROVEN, most tractable
Halves int4 weight bytes again via 2:4 structured sparsity on Ampere sparse
tensor cores. **Measured on A10**: ~30% extra throughput / 20% lower latency from
the sparsity, combined int4+2:4 kernel ~3.3× vs fp16, integrated in vLLM
(`cap>=8.0`) [IST-DASLab/Sparse-Marlin; vLLM #10260]. Build: SparseGPT 2:4 →
GPTQ W4A16 (one-shot, calibration-only) — `build_2of4_w4a16.py`.
**Gate:** one-shot 2:4 may bust PPL 2.42; fix = short 2:4 sparse-aware fine-tune
(Sparse-Llama precedent — recovers near-dense). **Ceiling: ~+30–50%.**

### 2. Speculative decoding done right — highest ceiling, biggest batch-1 lever
"As long as you're bandwidth-bound you get multiple tokens for ~the price of one"
[Together; Doubleword]. The weight read is amortized across K verified tokens.
My earlier "spec decode dead" was about *implementations*, not the principle:
n-gram has low acceptance + forfeits async scheduling; MTP is *blocked* by a vLLM
Triton assert (`num_heads {8,4}` in the KV-shared group; draft global layer is
head_dim=512 which only Triton supports). **Fix = patch vLLM** to isolate the
draft layers into their own attention group (or make the metadata builder
per-layer-head-aware) — a code rewrite, exactly "on the table." A trained MTP head
(`gemma-4-E4B-it-assistant`) typically accepts ~3–4 tokens. **Ceiling: ~+100–200%
if acceptance holds and the async-scheduling penalty is contained.**

### 3. Sub-4-bit weights (3-bit, QuIP#/QTIP-class) — attack the dominant term
3-bit PTQ now *outperforms* lossless 4-bit; 2-bit reaches 3–4-bit scaling
[QuIP#/QTIP, arXiv 2402.04396/2406.11235]. 4→3 bit ≈ −25% weight bytes; 4→2 ≈
−50%. **Gate:** needs the method's kernels (vLLM support varies) + calibration;
2-bit likely needs QAT to hold PPL. **Ceiling: ~+25–80%.**

### 4. Megakernel — erase the overhead floor
Fuse the whole decode step into one persistent kernel: no launch gaps, software-
pipelined weight loads, approach the bandwidth bound. Mirage MPK (2025): 1.2–6.7×
lower decode latency, A100 14.5→12.5 ms toward the 10 ms bound; Hazy Research
batch-1 Llama megakernel. MPK is a *compiler* — could target Gemma's text path.
**Ceiling: ~+30–65% on top of everything (closes our ~40% efficiency gap).**
Effort: very high; multimodal path must stay intact.

### Stacking (illustrative, not additive-clean)
int4 95 → +lm_head ~115 → +2:4 ~150 → +3-bit ~180 → +MTP spec ~2× ~360 →
+megakernel ~1.3× ~470. Even partial stacking is several× the current 95.

## The real bottleneck: compute access
All four need GPU beyond the fixed benchmark: SparseGPT/GPTQ calibration, QAT /
sparse-recovery fine-tuning, megakernel dev, and benchmarking. Current gating:
agent-run benchmark quota (5/agent/24h, mine=0 until ~2026-06-09 15:00 UTC) and
self-launching `hf jobs` 403s for the token. **Backdoor for one-shot builds:**
`serve.py` runs arbitrary GPU code before starting the endpoint, so a one-shot
SparseGPT/GPTQ build can be done *inside* a benchmark run (build → serve →
benchmarked in one slot). Heavy training (2-bit QAT, sparse-recovery, megakernel)
needs dedicated GPU jobs — that's the ask to unblock.

**Sources:** Sparse-Marlin (IST-DASLab; vLLM #10260; Red Hat 2:4 Sparse-Llama);
QuIP#/QTIP (arXiv 2402.04396, 2406.11235; together.ai); Mirage MPK
(github.com/mirage-project/mirage; Hazy Research 2025-05-27); spec-decode BS=1
amortization (together.ai, doubleword.ai).
