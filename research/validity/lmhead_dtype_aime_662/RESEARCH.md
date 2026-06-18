# PR #662 research pass — lm_head dtype effect on AIME (literature)

Run before the quantization change (per program.md). Full synthesis from the
researcher-agent; key load-bearing points for THIS card:

## Mechanistic prior: head-dtype IS plausibly a large AIME lever
- **lm_head is the highest-sensitivity position** in the net (SliderQuant: first/last
  layers most quant-sensitive; AMD Quark lists `*lm_head` as an exclusion candidate).
  GPTQ/AWQ skip lm_head by default — not because it's safe, but because the cost only
  shows up on hard generative tasks, not PPL/MMLU.
- **LFQ (arxiv 2605.29756, ICML 2026):** block-wise PTQ that omits the unembedding
  drops greedy **AIME 46.67%→30.00%** (35% rel); fixing the final-block/head recovers
  to **43.33%**. Mechanism = autoregressive compounding: one wrong top-1 at a tie
  steers the whole CoT off the solution path. AIME greedy + integer-exact-match is the
  worst-case regime for head-quant error.
- **QAT-specific (our exact case):** Gemma-4-E4B was QAT'd with a **tied bf16 lm_head**;
  the int4 BODY was globally calibrated against a bf16 head. Post-hoc int4-quantizing
  that head breaks the QAT loss alignment with no compensation. "What Makes Low-Bit QAT
  Work for Reasoning" (2601.14888) + d-Matrix (2410.14570): PTQ's local proxy decouples
  from global loss; QAT coupling is precision-specific. => restoring head precision
  should *repair* AIME. **Raises prior toward HEAD_RECOVERS_AIME — but must MEASURE.**

## int4 -> int8 -> bf16 ladder on the unembedding
- HF transformers issue #31474: **int8 unembedding "nearly lossless"** (PPL increase
  within body-int8 noise); NF4 measurable but "acceptable". Per-channel/per-row int8
  (one scale per vocab token) is essentially exact for a 262144xN matrix.
- => expect int8head to capture MOST of bf16head's recovery at ~half the head read.

## CRITICAL vLLM 0.22.0 gotcha (de-risk the int8 arm)
- vLLM PR #37291 / commit e8b055a (Mar 2026) fixed `get_quant_method` not routing
  `ParallelLMHead` when the layer name matched a quant target. **0.22.0 PREDATES this** —
  an explicitly-targeted lm_head *could* silently fall back to unquantized bf16.
- BUT our shipped submission's **int4** lm_head (same `targets:["re:.*lm_head"]`) *does*
  load as int4 on 0.22.0 (PR #4 served PPL 2.0190, int4-head-specific). So 4-bit routing
  works. The residual risk is **8-bit specifically**. The research-recommended SAFE path
  is exactly what build_head_variant.py does: **bake int8 weights into the safetensors**
  + declare a `targets:["re:.*lm_head"], num_bits:8` group.
- **ACTION:** after serving int8head, EMPIRICALLY confirm the head is int8 (not silent
  bf16): (a) server-log kernel line for lm_head, (b) decode-TPS signature must sit
  BETWEEN shipped(int4) and bf16head(bf16); if int8head TPS == bf16head TPS, the int8
  silently fell back and the int8 AIME number is really bf16's.

## bf16head arm safety
- Research: "for bf16 lm_head, omit it from quant targets — loads bf16 automatically."
  build_head_variant.py does this (removed group_1, lm_head in `ignore`, bf16
  lm_head.weight present). Confirmed body byte-identical to shipped (2762/2762 tensors).
