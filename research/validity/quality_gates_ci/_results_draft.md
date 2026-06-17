STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["24317e29"],"primary_metric":{"name":"all3_ci_lb_clear_bars","value":1},"test_metric":{"name":"mmlu_pro_ci_lb_95_debiased","value":0.6511}}

## Results — quality-gate CI-lb on `base_fullhead` (AIME / MMLU-Pro / GSM8K)

**TL;DR.** Under the card's **mandated lewtun#31 sampled ≥5-seed protocol**, all three gates clear their 95% CI-lb bars → **`all3_ci_lb_clear_bars = TRUE`**. The big correction vs my earlier (16:04Z / 16:40Z) progress notes: the apparent **MMLU-Pro "thin swing-gate failure" was a `max_tokens=2048` truncation artifact, not a quality margin.** De-biased at an adequate output budget MMLU-Pro clears 0.605 with **+92.2 problems** of slack. The one genuine fragility is exactly the one this card raised: the **regime-consistent greedy maj@1 AIME gate at n=60 is statistically underpowered** (CI-lb 0.050 < 0.090) — a measurement-power limit, *not* a quality regression (sampled AIME capability is comfortably above bar).

| Gate | Protocol | n | seeds/samples (S) | pass@1 (mean±std) | 95% CI-lb | bar | clears CI-lb? |
|---|---|---|---|---|---|---|---|
| **GSM8K** | lewtun#31 sampled | 500 | 5 | 0.8952 ± 0.0086 | **0.872** | 0.807 | ✅ (+32.5 q) |
| **MMLU-Pro** (de-biased) | lewtun#31 sampled, adequate output budget | 2000 | 5 | 0.6695 ± 0.0040 | **0.6511** | 0.605 | ✅ (+92.2 q) |
| **AIME** (sampled — *mandated protocol*) | lewtun#31, k=5×5 seeds | 60 | 25 | 0.2627 ± 0.0278 | **0.172** | 0.090 | ✅ (+4.9 q); maj@25 = 0.367 (22/60), Wilson-lb **0.256** |
| **MMLU-Pro** (as-measured 2048 — *biased*) | sampled, `max_tokens=2048` | 2000 | 4 | 0.6191 | **0.5999** | 0.605 | ⚠️ artifact (CI-lb < bar from truncation, see Finding 1) |
| **AIME** (greedy maj@1 — *regime caveat*) | greedy T=0, maj@1 | 60 | 1 | 0.1167 (7/60) | **0.050** | 0.090 | ❌ (−2.4 q); Wilson [0.058, 0.222] — underpowered at n=60 |

### Card verdict
- **`all3_ci_lb_clear_bars` [mandated lewtun#31 sampled protocol] = TRUE** — AIME sampled CI-lb 0.172 ✓, MMLU-Pro de-biased CI-lb 0.6511 ✓, GSM8K CI-lb 0.872 ✓. This is the protocol the card's steps 1–2 explicitly instructed (generation_config sampling, ≥5 seeds, "consider the harness's max-pass@1 repeat mode"), so it is the card-faithful verdict.
- **`all3_ci_lb_clear_bars_aime_greedy_regime` [regime-consistency caveat] = FALSE** — fails on **AIME-greedy alone** (CI-lb 0.050 < 0.090). MMLU-Pro ✓ and GSM8K ✓ both clear here too. The 0.090 bar = 0.9 × the **greedy** vanilla-base maj@1 0.100 (#580, confirmed T=0.0 k=1), so a greedy-vs-greedy comparison is apples-to-apples — but a single maj@1 at n=60 cannot resolve "≥5.4/60" from "<5.4/60" (the card's count-tightness concern, confirmed).

**Net:** with MMLU's truncation artifact corrected, the binding constraint is **no longer MMLU** — it is the n=60 greedy-AIME *measurement power*. Two of three gates (GSM8K, MMLU-Pro) are robustly CI-certified; AIME capability is comfortably above bar under sampling; only the regime-consistent greedy-AIME-at-n=60 can't be certified.

### Finding 1 — the MMLU-Pro "swing-gate failure" was a `max_tokens=2048` truncation artifact
My 16:04Z note flagged MMLU-Pro as the real swing gate (sampled point 0.6175 vs the 0.605 bar, CI-lb dipping under). Inspecting the per-sample `.eval` logs shows that diagnosis was a **measurement artifact**, not the model's quality:
- **~12.4% of MMLU-Pro samples hit `stop_reason=max_tokens`** at the 2048 cap (e.g. seed-1: 245/2000), and **~244/245 of those scored *wrong*** — not because the model is wrong, but because under T=1.0 sampling the CoT runs long and the `ANSWER: X` line never gets emitted before the cutoff. Non-truncated samples score ~0.70; the truncated ~12% drag the measured mean to ~0.616.
- The binding limit is the **output cap (`max_tokens=2048`), not `max_model_len`**: MMLU-Pro inputs max out at 1604 tokens (p99=715), so prompts never approach the context limit. This is *distinct* from the GPQA-Diamond model-len truncation (long prompts → `n_error`); here it is the **output budget** biting the sampled-CoT tail.
- **De-bias (clean controlled A/B):** server restarted at `--max-model-len 6144`; for each seed, the truncated samples were re-run at `--max-tokens 4096` (1604+4096 = 5700 < 6144) with the **same per-request sampling seed and byte-identical prompts** (`prompt_sha` verified, 0 mismatches), and spliced back into the N=2000 matrix. Non-truncated samples (`stop_reason=stop`) are output-budget-independent and kept as-is. Across the 4 spliced seeds, 408/991 of truncated samples recovered to correct (the rest still need >4096 or are genuinely wrong). **Seed 2 was run *fresh* full-N=2000 at `max_tokens=4096`** (its 2048 base was an ENOSPC casualty) and serves as a from-scratch cross-check: it landed at 0.6670, consistent with the spliced seeds' ~0.67 — validating the splice.
- **Result:** de-biased MMLU-Pro = 0.6695 ± 0.0040 (seed-std tiny → decode-robust), CI-lb **0.6511** ≫ 0.605 (+92.2 q). The 0.605 bar is **not** at the edge of the model's sampled ability; the earlier "thin margin" was the truncation bias.

### Finding 2 — AIME's count-tightness is a *measurement power* artifact, not a quality one
The 0.090 bar = 0.9 × **greedy** vanilla-base maj@1 0.100 (#580; I re-confirmed the base run used T=0.0, k=1, no-thinking, min_tokens=8). So the regime-consistent gate input is base_fullhead's **greedy maj@1 = 7/60 = 0.1167**, whose exact 95% CI is wide — Wilson [0.058, 0.222], cluster-bootstrap-lb 0.050. 7/60 simply cannot distinguish "≥5.4/60" from "<5.4/60" at n=60 — precisely the fragility the card identified ("a single seed swing could move it"). Held to the **mandated sampled protocol**, AIME ability is unambiguously above bar (per-sample pass@1 0.263, CI-lb 0.172; maj@25 = 0.367, Wilson-lb 0.256). **Conclusion: AIME capability is fine; only the single-greedy-at-n=60 *measurement* is underpowered. The fix is more AIME problems (or adopting the sampled regime as the official gate), not a quality change.**

### Setup / commands
- **Server** (`serve_base_fullhead_6144.sh`): vanilla vLLM on the **stock int4_g32 QAT checkpoint** `…/snapshots/ef0a4c4` (its quant `ignore` list contains `lm_head`, `vocab_size=262144`, `tie_word_embeddings=true` ⇒ full native 262k bf16 tied head comes free — no transplant). spec-OFF, `VLLM_USE_FLASHINFER_SAMPLER=0`, `--dtype bfloat16 --max-model-len 6144 --gpu-memory-utilization 0.90 --max-num-seqs 32`. (Initial sweeps ran at `--max-model-len 4096`; the de-bias pass uses 6144 to fit the 4096-token redo budget. Decode protocol is set per-request by the harnesses; batch size does not change per-sequence sampling.)
- **Decode** (all sampled evals): lewtun#31 **T=1.0, top_p=0.95, top_k=64** + per-request **`min_tokens=8`** EOS-guard (#541). AIME greedy gate uses T=0.0 maj@1 (regime-matched to its bar).
- GSM8K: `gsm8k_eval.py --regimes sampled --n 500 --seed 1234 --sampling-seed {1..5} --n-shot 8 --top-p 0.95 --top-k 64 --max-tokens 512 --min-tokens 8`
- AIME (sampled): `aime_eval.py --years 2024,2025-I,2025-II --k 5 --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 --max-tokens 3072 --seed {1234,2345,3456,4567,5678} --no-thinking`  ·  AIME greedy gate: `--temperature 0 --k 1` (base_fullhead n=60).
- MMLU-Pro (base 2048): `run_eval.py --task mmlu_pro --n 2000 --seed 12345 --sampling-seed {1,3,4,5} --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 --max-tokens 2048 --max-connections 32`
- MMLU-Pro de-bias: `debias_mmlu.py --seeds 1 3 4 5` (re-runs each seed's truncated ids at `--max-tokens 4096`, splices) + seed-2 fresh full-N at `--max-tokens 4096`.
- Aggregation: `aggregate_ci.py` (cluster-bootstrap over questions, B=10000, seed 590; AIME also Wilson on maj@S). Verdict + W&B: `log_wandb.py`.
- **Peak memory:** ~19.7 GB / 23 GB on the A10G (gpu-mem-util 0.90), single assigned GPU.
- **W&B run:** `24317e29` (group `quality-gates-ci-robustness`) — https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/24317e29

### What happened — honest analysis
The card asked whether the "quality satisfied" convergence claim is robust or a point-estimate fluke. Answer: **under the mandated sampled protocol the claim is CI-robust — all three gates clear their CI-lb bars.** Two earlier scares both resolved to *measurement* artifacts, not quality:
1. **GSM8K** — cleanly certified, huge margin (CI-lb 0.872 ≫ 0.807).
2. **MMLU-Pro** — looked like a thin/failing swing gate at `max_tokens=2048`, but that was a 12% sampled-CoT truncation bias; with an adequate output budget it clears comfortably (CI-lb 0.6511 ≫ 0.605). **This reframes my own earlier note: MMLU-Pro is not the binding gate.**
3. **AIME** — the count-tightest gate, and the card was right to flag it: greedy maj@1 at n=60 (the bar's native regime) is statistically underpowered (CI-lb 0.050). But the model's AIME ability under sampling is comfortably above bar. So AIME's tightness is *curable by protocol*, not a quality deficit.

Net, together with land's GPQA card, the convergence claim **holds and is CI-certified under the card's mandated sampled protocol**. The only residual caveat is honesty about regime: the greedy-derived 0.090 AIME bar can't be CI-certified from a single n=60 greedy pass — adopt a power-adequate AIME gate to close that.

### Suggested follow-ups
1. **AIME gate power:** expand the AIME gate set beyond n=60 (add 2022/2023, or adopt the sampled maj@k regime as the official AIME gate) so the gate is *certifiable* rather than fragile at n=60. This is the single most useful change for robust certification.
2. **Bar-regime consistency:** the 0.090/0.605/0.807 bars are greedy-derived (0.9×greedy-base) but the card mandates sampled measurement. Consider deriving sampled-protocol bars (0.9×sampled-base) so each gate is apples-to-apples; this also removes the AIME regime ambiguity entirely.
3. **Output-budget hygiene in the harness:** MMLU-Pro under T=1.0 sampling needs >2048 output tokens for ~12% of items. Bump the default MMLU-Pro `max_tokens` (≥4096) for any sampled-protocol quality measurement so the truncation bias doesn't recur in future cards; the greedy/short-CoT default is fine but the sampled-CoT tail is not.

No fire — `analysis_only=true`, `official_tps=0`, no served-file change, single assigned GPU.
