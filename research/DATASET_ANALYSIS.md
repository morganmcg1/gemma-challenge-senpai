# DATASET ANALYSIS — Fast Gemma Challenge benchmark inputs

The "dataset" for this serving challenge is the fixed benchmark input: the 128 public
eval prompts (TPS) + the 128 PPL ground-truth records (quality guardrail). Source files:
`official/main_bucket/shared_resources/speed_benchmark/data/`.

## Public eval prompts — `eval_prompts_sharegpt.json` (128 records)

- Format: `{id, conversations:[{from:human,value}, {from:gpt/assistant,value}]}`. **All 128 are single-turn** (one human prompt). The harness benchmarks generation against the human turn.
- **Source mix (by `id` prefix):** `mmlu_pro` 57 · `gpqa_diamond` 57 · `aime2026` 14.
  → **100% reasoning / STEM** (multiple-choice science + competition math). Despite the
  filename, the public 128 are **not** ShareGPT chit-chat. MMLU-Pro/GPQA prompts ask for
  step-by-step reasoning ending in `ANSWER: $LETTER`; AIME are math problems.
- Human-prompt length: **min 331 / median 730 / max 5842 chars.** A few long prompts, most medium.
- Decode request is fixed: `max_tokens=512, temperature=0.0, ignore_eos=true, add_special_tokens=false, return_token_ids=true`.
  **`ignore_eos=true` + 512 output** means every request decodes a full 512 tokens regardless of
  natural stopping → **TPS is dominated by steady-state decode**, and prompt length only affects
  the (small) prefill fraction. This is why decode-bandwidth levers dominate.

## PPL ground truth — `ppl_ground_truth_tokens.jsonl` (128 records)

- Format: `{id, context_token_ids, target_token_ids}`. **IDs map 1:1 to the eval prompts.**
- `context_token_ids` (the prompt as token IDs): min 114 / median 233 / max 2431 tokens.
- `target_token_ids` (reference continuation to be scored, teacher-forced): mostly **512** (min 216).
- Generated once by the **`gemma-4-31B-it` reference** through the multimodal chat template
  (temperature 1.0, top_k 64, top_p 0.95, fixed seed). E4B shares the 31B tokenizer, so the
  token IDs are scored directly. A correctly-served bf16 E4B scores aggregate **PPL ≈ 2.30**.
- PPL is **teacher-forced** (feeds the correct next token each step) → it is *blind to which
  tokens the model would actually emit*. That is why a separate **greedy-identity** check is the
  binding correctness gate, not PPL (PPL can't see token drift from a lossy kernel/quant/spec-decode).

## Implications for hypotheses

1. **Drafter acceptance is workload-specific — and the benchmark workload is REASONING, not chat.**
   Public acceptance is measured on MMLU-Pro/GPQA/AIME continuation. Two distinct corpus goals,
   do not conflate them:
   - **Benchmark-match (maximize public acceptance — fern #25):** train on **reasoning CoT** matching
     the 57/57/14 mix (MMLU-Pro + GPQA + competition-math, distilled from the served target).
     **ShareGPT is a poor match and a measured negative control, NOT a lever.** Empirical (fern #25,
     2026-06-13): MATH-only EAGLE-3 acceptance **plateaus ~0.68** (0.6603@500 → 0.6816@898, +0.02 for
     a 2nd epoch on 1.76M tokens) → the bottleneck is **data distribution, not training steps**.
     Breaking 0.68→0.78 needs benchmark-matched reasoning data, not more epochs or ShareGPT.
   - **Private-robustness (avoid the 5% repro gap that invalidates — land #9):** the **private** set
     differs from these public 128 enough to cost 4–9% TPS if the drafter overfits public. Hedge with
     a **wider** distribution (reasoning core + ShareGPT/misc breadth), **dedup against these 128**
     (ids `mmlu_pro*`/`gpqa_diamond*`/`aime2026*`), hold out ≥200 disjoint.
   These can tension: narrow-reasoning maximizes public, wide hedges private. The per-source
   acceptance breakdown (fern #25) is the evidence that tells us how much breadth costs on public.
2. **Prefill is cheap, decode is everything** (512 forced output tokens). Optimize steady-state
   decode bandwidth + accepted-tokens/step, not prompt handling.
3. **The PPL memory headroom recipe matters:** `prompt_logprobs` materializes a full-vocab
   float32 `log_softmax` whose peak scales with prefill-chunk length. `MAX_NUM_BATCHED_TOKENS=512`
   + `gpu_memory_utilization=0.90` + `expandable_segments:True` bound it; the longest context
   (2431 tokens) is what OOMs the PPL stage if you remove that cap. Any engine/numerics change
   must keep equivalent headroom or the PPL stage errors (likely cause of a missing
   `ppl_summary.json`).
4. **Vocab-prune levers (lmhead12k / PCK04)** exploit that the 262k vocab is far larger than the
   per-step token mass; restricting verify/logits to the top-k is a real lever — but it must
   provably preserve greedy identity (top accepted token always in the kept set, else fall back).

_Last updated: 2026-06-13 (cycle 16 — fern #25 confirmed MATH-only acceptance plateaus ~0.68; data-distribution is the drafter bottleneck; benchmark-matched reasoning CoT, not ShareGPT, is the path to 0.78)._
