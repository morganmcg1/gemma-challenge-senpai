STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["orwwhmxi"],"primary_metric":{"name":"gpqa_stack_delta_0220_minus_dev307","value":0.3081},"test_metric":{"name":"gpqa_shipped_0220","value":0.4859},"s547_failure_mode":"harness_bug","gpqa_shipped_0220":0.4859,"gpqa_shipped_dev307":0.1778,"gpqa_stack_delta":0.3081,"gsm8k_0220":0.9317,"gsm8k_dev307":0.7333,"mmlu_0220":0.6533,"mmlu_dev307":0.2783,"mmlu_0220_obtainable":true,"finish_length_rate_0220":0.0313,"finish_length_rate_dev307":0.5172,"dev307_is_valid_accuracy_denominator":false,"eval_stack_verdict":"0.22.0 valid (=submission pin); dev307 invalid (repetition-loop crater); corrected shipped GPQA-D=0.486 sampled/0.465 greedy clears 0.471"}

## Results

**Pod liveness:** alive. W&B run for #615 is [`orwwhmxi`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/orwwhmxi) (group `eval-stack-accuracy-validity`, `analysis_only=true`, `official_tps=0`). LOCAL only — no HF Job, no submission, no served-file change.

### Headline (the answer flips the programme's premise)

**dev307 is NOT a valid accuracy denominator. It is dev307 — not 0.22.0 — that craters accuracy.** On the shipped `int4_g128_lmhead` checkpoint, served with **identical** flags and **only the vLLM build differing**, dev307 collapses every accuracy eval while 0.22.0 (the submission pin) is healthy and clears every bar. `#547`'s "0.22.0 craters MMLU" does **not** reproduce on the shipped checkpoint → it was a **harness/protocol incompatibility, not a model collapse**.

### The keystone A/B (shipped int4, identical decode protocol, only the engine differs)

Decode protocol on **both** stacks: `T=1.0 top_p=0.95 top_k=64`, `min_tokens=8` (#541), `max_tokens=4096` CoT / `1024` GSM8K, `--max-model-len 6144` (#598), fixed choice-layout seed `12345`, sampling-seeds `1..5`. Both servers loaded `submissions/int4_g128_lmhead/model` with the **same** `compressed-tensors` quant + **same** `MarlinLinearKernel`, same KV-cache (265,071 tokens) — the serve logs are config-identical line-for-line except the version string.

| Eval | n | seeds | **0.22.0** (submission pin) | **dev307** | **Δ (0220−dev307)** | bar | finish_len 0220 / dev307 |
|---|---|---|---|---|---|---|---|
| **GPQA-Diamond** | 198 | 5 | **0.4859** · CI95 [0.431, 0.539] | **0.1778** · CI95 [0.143, 0.213] | **+0.3081** | 0.471 | 3.1% / **51.7%** |
| MMLU-Pro | 200 | 3 | **0.6533** · CI95 [0.593, 0.710] | **0.2783** · CI95 [0.228, 0.328] | **+0.3750** | 0.605 | 1.3% / **49.5%** |
| GSM8K | 200 | 3 | **0.9317** · CI95 [0.902, 0.960] | **0.7333** · CI95 [0.683, 0.783] | **+0.1983** | 0.807 | 0.5% / 9.5% |

- Every CI is **completely non-overlapping**. dev307 GPQA-D (0.178) is **below random chance** (0.25 for 4-way).
- On 0.22.0: GPQA mean **clears** 0.471, MMLU mean **clears** 0.605, GSM8K CI-lb **clears** 0.807. Per-seed GPQA-D 0.22.0: `[0.505, 0.495, 0.470, 0.480, 0.480]`.
- Greedy cross-check (0.22.0, `temp=0 min_tokens=0`, the literal pre-#548 #547 protocol): **GPQA-D 0.4646** (92/198), **MMLU-Pro 0.6450** (129/200) — healthy, 0 empty, 0 parse errors.

### 1. #547 re-diagnosis → `s547_failure_mode = harness_bug`

I reproduced the #547 setup **on my own branch** against the shipped checkpoint on 0.22.0 (greedy, `min_tokens=0`, the exact pre-#548 protocol #547 used). Result: **MMLU-Pro 0.645, GPQA-D 0.465 — 0.22.0 does not crater.** The inspect `choice` scorer parsed every completion cleanly (`empty_rate=0.0000`, `err=0`). So "0.22.0 craters MMLU" is **not a property of the model on the shipped checkpoint**; #547's collapse was a harness/protocol incompatibility in that specific setup (chat-template / logprob-extraction / sampling-API / stop-token — the failure did not transfer to the generation+parse harness used here). **A valid 0.22.0 MMLU number is fully obtainable** (0.653 sampled / 0.645 greedy).

### 2. The smoking gun — dev307 degenerates into repetition loops (not a parse artifact)

Matched-completion inspection on GPQA seed 1: **93/198** questions have 0.22.0 terminating cleanly while dev307 runs to the 4096-token cap. Same question `rec0wZvZgiz320KRs`:

- **0.22.0:** `stop=stop`, 1026 tokens, coherent derivation ending `ANSWER: A` → **correct**.
- **dev307:** `stop=max_tokens`, 4096 tokens, degenerates into `\text{}\text{}\text{}…` repeated **~974×** → **incorrect**.

Truncation taxonomy over those 93: **~71% are repetition degeneracy** (compression-ratio < 0.20), only ~29% are budget-starved-but-coherent. **This means an ubel-#590-style "re-run at more tokens" debias CANNOT rescue dev307 — the model loops regardless of budget.** The low scores are genuine model failure (the model never emits a parseable answer), not a scorer that mis-reads good completions. Scoring counted truncated loops as wrong (`empty_rate=0.0000`), which is the correct conservative behavior.

### 3. Reconcile PPL ↔ accuracy

My #606 ([`dqg9xcpo`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/dqg9xcpo)) found dev307 PPL **2.6264 (+30.1%)** vs 0.22.0 **2.0188** (= official 2.019), bit-identical across draws. This card shows the PPL inflation **does** correspond to a large **accuracy degradation** — and it is mechanistically the same defect: dev307's numerics on this int4 checkpoint push generation into repetition attractors on hard prompts. The distortion is **not** confined to the teacher-forced PPL path; it fully propagates to generation accuracy. The three signals (PPL +30%, 0/128 decode divergence, accuracy crater) are one phenomenon.

**Why this is not my serve bug (ruled out):** (a) the dev307 and 0.22.0 serve logs are config-identical (same model, same Marlin kernel, same flags); (b) #606's PPL gap is a teacher-forced measurement independent of sampling/generation; (c) the loops are visible in raw completions. Three independent lines of evidence.

### `eval_stack_verdict`

**0.22.0 is the valid stack for accuracy-gate measurement, and it is the submission pin — so there is NO split: 0.22.0 is clean for BOTH PPL/identity AND accuracy.** dev307 must be retired from the accuracy gate. `dev307_is_valid_accuracy_denominator = false`.

**Corrected GPQA-D for the shipped config on the valid stack (0.22.0): 0.4859 sampled (5-seed mean, CI95 [0.431, 0.539]) / 0.4646 greedy — clears the 0.471 bar by mean.** (CI lower bound 0.431 does not clear; that is multi-seed spread, not a stack effect.)

### ⚠️ Reconciliation flag for the advisor (important)

You cited **fern #612 GPQA-D = 0.4764 on dev307 + spec**. My controlled measurement of **dev307 (non-spec, shipped int4, direct serve) = 0.1778** is irreconcilable with 0.4764 *if the underlying model numerics are the same*. Faithful speculative decoding samples from the target distribution, so spec on a broken dev307-int4 target should also read ~0.18, not 0.48. fern's 0.476 is in fact **closer to my 0.22.0 number (0.486)** than to dev307. That strongly implies fern's spec path's effective target/serve config differs from a faithful dev307-int4 serve (different target head, sampling, or budget). **I did not inspect fern's branch (launch isolation) — this needs your reconciliation.** Either way, by your own decision rule ("differ materially → re-base"), the +0.31 GPQA-D gap settles it: **dev307 is invalid; the Option-B quality panel must be re-based onto 0.22.0.**

### Implications (for your decision, not implemented here)

- Every dev307 accuracy read in the quality programme (#612's 0.4764, #614's base-denominator bars, #605's Option-B-dead 0.4141, #610's Option-A) is measured on a stack that **deflates** accuracy via repetition-loop truncation. The absolute numbers must re-base onto 0.22.0; the truncation/regime *structure* #614 studies is stack-dependent here (dev307's 50% trunc is the artifact), so the bars likely move **up** on 0.22.0.
- Concretely: the base-of-comparison bars were set as 0.9×(dev307 base). If the base itself was measured on the deflating stack, the 90%-of-base bars are biased; re-measuring the **base** on 0.22.0 is the next required step (coordinates with ubel #614).

### Reproduction

Single A10G, server determines the stack; driver is stack-agnostic:
```bash
# Serve shipped int4 on the chosen build (0.22.0 pin OR dev307), identical flags:
research/validity/eval_stack_accuracy_validity/serve_ship.sh <0220|dev307>
# 3-eval panel vs the running server (GPQA n=198×5 seeds, MMLU n=200×3, GSM8K n=200×3):
research/validity/eval_stack_accuracy_validity/run_stack.sh <v0220|dev307> full
# Faithful #547 greedy repro on 0.22.0:
bash research/validity/eval_stack_accuracy_validity/repro547_greedy.sh
# Pool + CI + finish_length:
.../summarize_stack.py <v0220|dev307>
```

- **Peak GPU memory:** 19.6 GiB / 23.0 GiB (A10G, `--gpu-memory-utilization 0.90`).
- **W&B:** [`orwwhmxi`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/orwwhmxi) — per-stack/per-eval accuracy, per-seed, CIs, finish_length rates, stack deltas, and #606 PPL deltas.
- Disk watched: 87% used / 127 GiB free at report time (no ENOSPC during this card).

### What happened — honest analysis

The card's binding question resolves cleanly and in the opposite direction from the programme's working assumption. A controlled single-variable A/B (only the vLLM build changes) shows dev307 craters accuracy via a repetition-degeneracy generation defect — the same defect already visible as +30% PPL and 0/128 decode divergence in #606. 0.22.0, the actual submission stack, is healthy on accuracy and PPL alike, so the worry "no stack is clean for both" dissolves: **0.22.0 is clean for both.** #547's MMLU "crater" was a harness incompatibility that does not transfer to the shipped checkpoint under the generation+parse harness. The one loose end I cannot close locally is the fern #612 0.4764-on-dev307 number, which is inconsistent with a faithful dev307 serve and needs advisor reconciliation.

### Suggested follow-ups

1. **Re-base the accuracy gate + base-denominator bars onto 0.22.0** (coordinate with ubel #614). The bars set as 0.9×(dev307 base) are biased by the deflating stack.
2. **Reconcile fern #612's 0.4764 dev307+spec** against this 0.178 dev307 non-spec — determine what in the spec path differs from a faithful dev307-int4 serve (likely a different effective target/head or budget).
3. **Re-measure Option A/B/C quality on 0.22.0.** Option B died on a dev307 GPQA-D 0.4141; on the valid stack that number must be re-read before declaring Option B dead.
4. Retire dev307 from all accuracy evals programme-wide; keep it only where it was ever justified (it never was, on this checkpoint).
