STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["e2t9l2n5"],"primary_metric":{"name":"total_answer_diffs_at_tau0p3","value":1},"test_metric":{"name":"net_graded_delta_pooled","value":-0.0043}}

## Results

### Decision statement for Morgan

> **Under a τ=0.3-nat tolerance #319 contract, Option-B (int4+spec) changes `0 / 698` GRADED-OUTCOME answers vs the strict int4-AR rung** — and `1 / 698` extracted-choice *strings* (one GPQA item, **graded-immaterial: both arms wrong**) — **versus `51 / 698` grade-flips (`86 / 698` choice-diffs) at strict byte-exact (τ=0).** Every one of those strict flips roots in a **≤0.375-nat int4-Marlin grid-tie** (0 tokens outside the AR top-k anywhere on the stack), and the net graded Δ is **−0.0043 [−0.024, +0.016] ∋ 0**, inside the AR rung's own run-to-run nondeterminism band. Measured paired-greedy on **GPQA-D + GSM8K (n=698)**; **MMLU-Pro + AIME have no paired int4-AR stream and are flagged for routing** (below).

**One-word verdict: `TOLERANCE_ANSWER_SAFE` (graded)** — no accuracy-moving answer survives τ=0.3 on either measured benchmark. The lone extracted-choice survivor is a both-wrong near-tie, so it costs the eval score nothing.

### Headline table (Option-B int4+spec vs strict int4-AR, GREEDY paired)

| bench | n_q | tok-divergent items | ans-diffs (strict, choice) | grade-flips (strict) | **choice-resid @τ0.3** | **grade-resid @τ0.3** | net Δ (spec−ar) | verdict |
|---|---|---|---|---|---|---|---|---|
| GPQA-D | 198 | 196 | 59 (29.8%) | 38 | **1** | **0** | −0.0101 [−0.071, +0.051] | TOLERANCE_RESCUES_ALL |
| GSM8K | 500 | 161 | 27 (5.4%) | 13 | **0** | **0** | −0.0020 [−0.016, +0.012] | TOLERANCE_RESCUES_ALL |
| **POOLED** | **698** | **357** | **86** | **51** | **1** | **0** | **−0.0043 [−0.024, +0.016]** | **TOLERANCE_ANSWER_SAFE** |

- **`choice-resid @τ0.3`** = answer-divergent items whose ROOT token divergence gap **> 0.3 nat** (or spec token outside AR top-k) → the diffs a τ=0.3 acceptor would NOT rescue.
- **`grade-resid @τ0.3`** = the subset of those that are **accuracy-moving** (one arm correct, the other wrong). This is the decision-critical number: **0**.
- Cross-check at the #626 NEARTIE=0.5-nat band: choice-resid=0, grade-resid=0 (reproduces denken #626's "0 large-margin answer flips" exactly). The τ=0.3 cut is *stricter* than #626's 0.5-nat and surfaces the single boundary item below.

### The one τ=0.3 survivor (GPQA `recK9F5aqdaybl8bb`)

```
gold=B   spec→A (✗)   ar→C (✗)   root_gap = 0.375 nat   spec_outside_topk = False   grade_changed = False
```

The two arms disagree on the *letter* (A vs C) but **both are wrong**, so it moves the GPQA score by **0**. Its root divergence is a 0.375-nat int4-grid tie (= 3 × the 0.125-nat Marlin grid step) — above τ=0.3 but still a sub-0.5-nat coin-flip, not a decisive/structural flip. It is the **only** answer-string difference in 698 questions that a τ=0.3 contract leaves un-rescued, and it vanishes entirely at τ≥0.375.

### Why the strict-vs-tolerance gap is so large but graded-immaterial

At **strict byte-exact (τ=0)** the two arms diverge on **86/698** extracted answers (GPQA 29.8%!) — because a single mid-CoT grid-tie flip cascades the whole downstream generation (GPQA token-divergent items 196/198, but the un-cascaded per-step hazard is only **0.40%**, #616-consistent). Yet of the **51** *accuracy-moving* flips, McNemar is symmetric (GPQA b=18/c=20 p=0.87; GSM8K b=6/c=7 p=1.0) and **every single one roots in a ≤0.25-nat tie** — so τ=0.3 rescues **100%** of accuracy-moving flips. The strict contract's "30% of GPQA answers differ" is real at the token level but is pure quality-neutral CoT-path roulette.

### Scoped out — MMLU-Pro + AIME (flag for routing, per your terminal-marker instruction)

Neither has a **paired int4-AR** per-question greedy stream anywhere on the branch or in W&B (confirmed by run scan — `ubel #638` / `denken #637` / `int4-ar` / `live-rung` return zero matching runs). I have the **Option-B-only** greedy legs (MMLU-Pro 0.664 @0p22 / 0.664 @dev307; AIME maj@1 0.3667 @0p22 / 0.400 @dev307) but **no int4-AR counterpart to diff against**. The denken #626 harness that produced the GPQA/GSM8K pairs only builds `gpqa`/`gsm8k` items, so producing these two would need **net-new evalset builders + scorers AND a fresh paired generation run** (≈10 min MMLU + ≈40 min AIME on the local A10G, int4 body + drafter are present). That is exactly the "needs a fresh full eval → flag it" case you named, so I did **not** spend the GPU. **Say the word and I'll add the MMLU-Pro/AIME builders to the #626 harness and run the paired greedy gen locally**, or route to another slot.

Note on the in-flight runs you cited: `ubel #638` is a *sampled* int4-AR GPQA denominator (a different question — the quality *gate*, not the answer-level diff), and GPQA's answer-level question is already fully closed here by the #626 paired greedy. `denken #637` produces the *Option-B* AIME greedy leg, which I already have — it does not supply the missing *int4-AR* AIME side, so it would not unblock the AIME pairing on its own.

### Provenance / method (consolidation — nothing re-run)

- **Source data:** denken #626 (W&B [`bj8d88gf`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/bj8d88gf)) on-branch paired-greedy streams `research/validity/optionb_319_answer_materiality/results/{spec,ar,gaps}_{gpqa,gsm8k}.jsonl`. GREEDY matched-arm: spec = int4 body + MTP-K7 ON (M=8 verify), ar = same int4 body, spec OFF (M=1), vLLM 0.22.0, BI=1, MAX_NUM_SEQS=1 serial, prompt_sha gate PASS (0 mismatches), serving errors symmetric (0/0).
- **Added by this card:** re-cut the residual at the **exact τ=0.3-nat** contract (#626 flagged at 0.5-nat) at two answer levels (extracted-choice and accuracy-moving grade-flip), reproduced #626's answer-div (59/27) and net-Δ (−0.0101/−0.0020) independently, and pooled the net Δ via cluster bootstrap (each question = 1 cluster).
- **Token-level anchor (consistent):** stark #622 [`15g9q3wc`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/15g9q3wc) = 0.092% both-sides break, 100% sub-0.5-nat, 0 at τ=0.3; #616 per-step 0.43%; wirbel #633 [`33ulzfg8`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/33ulzfg8) all <0.5-nat. All corroborate the int4-Marlin M-variance grid-tie family as the *only* divergence source (0 outside-top-k across 357 probed divergences).

### Honest caveat (the framing you asked for)

The competition's #319 is a **token** gate, so τ=0.3 tolerance is a **contract-change request, not a quality finding**. This card answers only *"what is the answer-level cost of that contract change?"* — and on the two benchmarks with paired data the graded-answer cost is **zero** (1 immaterial choice-string). The decision of whether to *adopt* a tolerance contract at all is yours; this just removes the "but does it silently change answers?" unknown from that call. It also does **not** touch Option-B's separate open legs (the GPQA/AIME quality-gate reading A-vs-B, and official speed) — it is purely the identity/#481 leg.

### Run details

- **Command:** `python3 research/validity/optionb_tolerance_eval_answer_dossier/consolidate_tolerance_dossier.py --wandb_group optionb-tolerance-eval-answer-dossier`
- **W&B run:** `e2t9l2n5` — [optionb-tolerance-eval-answer-dossier](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/e2t9l2n5) (`analysis_only=1`, `official_tps=0`)
- **Peak memory:** N/A — CPU-only consolidation, **no GPU used**, no model loaded, no HF Job, no served-file change.
- **summary.json fields:** N/A (no benchmark/submission run; `analysis_only=true`, `official_tps=0`).

### What happened

The hypothesis ("τ=0.3 rescues 100% of token divergences AND changes ZERO graded answers") is **confirmed at the graded level and sharpened at the choice level**: 0/698 accuracy-moving answers survive τ=0.3, but exactly **1/698** extracted-choice strings do (a both-wrong GPQA item at gap 0.375 nat, between τ=0.3 and #626's 0.5-nat band). So "ZERO" is right for what a grader scores; the precise figure for what a byte-comparator sees at τ=0.3 is **1**, and it's quality-neutral. Cross-suite is **2/4 measured** (GPQA-D, GSM8K); MMLU-Pro + AIME are blocked only by the missing int4-AR pairing, not by any contrary evidence.

### Suggested follow-ups

1. **If you want the full 4/4:** approve adding MMLU-Pro + AIME builders to the #626 paired harness and a single local paired-greedy gen (no submission/HF Job). I expect the same result — the divergence family is a stack-wide int4-grid tie (0 outside-top-k over 357 probed), not benchmark-specific — but AIME's long free-form maj@1 has the most cascade surface, so it's the one worth actually measuring rather than inferring.
2. **Sensitivity line for the ruling:** the answer-safety is threshold-clean at τ≤0.25 and τ≥0.375 (0/698 either way) and shows its single boundary item only inside (0.3, 0.375]. If you set the contract at τ=0.375-nat instead of 0.3, the dossier is a literal **0/698** with no caveat.
3. This closes the **identity leg** of #481 as a measured input; the remaining Option-B blockers are the quality-gate reading (A vs B) and official speed — unchanged by this card.
