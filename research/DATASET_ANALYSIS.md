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

1. **Drafter acceptance is workload-specific.** Public acceptance is measured on
   reasoning/math continuation. A drafter tuned to MMLU-Pro/GPQA/AIME will look great on the
   public 128 — but the **private** set differs enough to cost 4–9% TPS (the 5% repro gap that
   invalidates submissions). The corpus_spec fix: train the drafter on a **wide** distribution
   (ShareGPT 50% + MMLU-Pro/GPQA/MATH-AIME + misc), **dedup against these 128**, hold out ≥900.
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

_Last updated: 2026-06-13._
