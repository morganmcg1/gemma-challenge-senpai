# Corpus design — KL-distill drafter training

The verifier invalidates results whose private-set TPS deviates from public by >5% (see `shared_resources/tps_repro_gap_itaca/`). The drafter is a function from prefix → drafted distribution; if that function is fit to a distribution narrower than the verifier's, gains in accepted-tokens/step on the public corpus may not carry to the private set.

`@kenyan-duma` flagged this directly: a 128-prompt corpus is the exact class of gain the verifier was designed to catch. kduma1's training corpus was ~9.2k distribution-matched prompts.

This document fixes a corpus design that should survive the private-set re-run.

## Hard requirements

1. **Size: >= 9,000 prompts** producing >= 1M propose-call traces.
2. **No overlap with the public bench.** Hash every prompt's first 512 tokens. Drop any whose 512-bigram-prefix matches anything in `data/eval_prompts_sharegpt.json`.
3. **Distribution-matched.** The eval set spans MMLU-Pro / GPQA-Diamond / AIME 2026 / ShareGPT — match all four.
4. **Held-out shard: >= 900 prompts** (10%) reserved for the offline acceptance gate. Never trained on. Selected by stratified sampling across the four distributions before any training run sees data.

## Suggested source mix

| source | weight | notes |
|---|---|---|
| ShareGPT (Vicuna fork) | 50% | Public bench is half ShareGPT-flavored; capacity-match. |
| MMLU-Pro train split | 20% | Multi-choice STEM. The bench includes MMLU-Pro. |
| GPQA train split | 10% | Same. |
| MATH / AIME 1995–2025 | 10% | Match the AIME 2026 slice in the public bench. |
| Misc instruct (e.g. Tulu, OpenOrca) | 10% | Distributional padding. |

The four target distributions are public; the *exact prompts in the eval split* are also public (`data/eval_prompts_sharegpt.json`), so dedup is straightforward.

## Per-prompt trace capture

For each prompt:
1. Run a full bench-style decode (output_len=512, temperature=0, max_concurrency=1) on the verified osoi5 substrate.
2. **Capture per propose-call:** the prefix token IDs, the int4 target's **top-2048 logits** (softmax-applied), and the kduma1 drafter's argmax token. The vocab is PCK04-pruned to ~16k, so top-2048 covers >99.9% of mass.
3. Write one record per propose-call to `trace_stream.jsonl`. Record schema in `train_kl_drafter.py:TRACE_SCHEMA`.

Approximate trace volume: 9k prompts × ~120 propose-calls/prompt ≈ 1.08M records.

`@paxenos-gemma-2`'s `kltrace-v0` captures the right format already (TRACE_TARGET_LOGITS hook on compute_logits, top-2048 per call). The corpus diversity is the missing piece — kltrace-v0 captures from the public 128 prompts.

## Why this should work

The drafter learns a conditional distribution over the next token given a prefix. KL-loss on top-2048 softmax aligns the drafter's distribution with the target's at every position. The training data needs to span the prefix-distribution the *verifier* draws from — not the prefix-distribution of the public bench.

If the four target distributions are sampled at scale, the resulting prefix-distribution is a superset of both public and private bench prompt-sets (they're both drawn from those same distributions). The drafter then generalizes to either at parity.

## Cheaper alternative: argmax+KL hybrid

If full-distribution capture is infeasible, a **two-loss combination** has been shown in the LK-losses literature (cite: paxenos's `20260611-222359-244` update mentions "hybrid KL+LK loss"):

```
L = α · CE(target_argmax) + (1 - α) · KL(top-k softmax)
```

with `α ∈ [0.3, 0.5]`. Captures most of the distillation benefit at lower trace-storage cost. Still needs the >=9k corpus.

## What to skip

- **Don't seed from `data/eval_prompts_sharegpt.json` or the 128 public bench prompts directly**. Even if the eventual training corpus is large, those 128 are higher-weight than they should be — the drafter will overfit. Use the four source distributions, not the eval slice.
- **Don't use only 128 prompts to get started.** This is the class of gain that evaporates on the private set. If you're capacity-constrained, run fewer epochs on a wider corpus, not full-epoch on the bench prompts.
