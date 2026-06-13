# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-13 (cycle 15, ~15:50Z)
- **Advisor branch:** `approval-gated-8gpu-20260613`
- **Most recent human directive:** Morgan (human) approved both int4 HF jobs ~13:00Z (issues #11 int4-qat, #12 int4-g128-lmhead). **Still operating under launch operator rules: no automatic HF Jobs / no `/v1/jobs:run` / no `train.py --launch` without a human-approved GitHub issue. Advisor consumes no GPU.**

---

## MILESTONE (cycle 13, 2026-06-13 14:38Z)

**Four PRs merged this cycle. The linchpin is now DEFINITIVELY RESOLVED (negatively). The next lane is verify-rollback.**

| PR | student | result | type |
|---|---|---|---|
| #4 `int4_g128_lmhead` | lawine | **126.378 TPS / PPL 2.019 / 128/128 / GREEDY_IDENTICAL** (official a10g-small, job `6a2d5a96`) | **LEADERBOARD WINNER** — new baseline |
| #19 `batch-invariant-vllm-spec` | kanna | **flip_rate 0.376%/tok ON — DIVERGENT.** bf16 control: 0.111%/tok. Two independent un-coverable causes identified. | **LINCHPIN DEFINITIVE NEGATIVE** |
| #16 `eagle3-training-pipeline` | fern | **tf_acceptance_rate_debug_1k = 0.6816**, val_loss 1.3372, W&B 30bgs1rs | Keeper research artifact |
| #18 `spec-verify-cost-model` | denken | **TPS ceiling ideal K*=15: 1,269.5 TPS** (p=0.7), W&B pvj0qogp | Keeper research artifact |

**Current official baseline: `submissions/int4_g128_lmhead` (PR #4) — 126.378 TPS / PPL 2.019 / GREEDY_IDENTICAL. 2.87× over bf16. 1.32× over int4 base.**

---

## THE LINCHPIN — RESOLVED (cycle 13, 2026-06-13)

**`VLLM_BATCH_INVARIANT=1` does NOT rescue greedy-valid spec decode at any precision in vLLM 0.22.0.**

Two independent un-coverable root causes (decomposed by kanna's bf16 discriminator arm):

- **(a) int4 Marlin `_C` op** — batch-variant, outside aten scope. Contributes ~0.265%/tok excess above bf16 floor. Cannot be intercepted by batch-invariance (aten-scoped).
- **(b) Spec verify path non-aten residual** — ~0.111%/tok irreducible. A non-aten component in the spec verify forward (attention-metadata build / rejection-sampler logits compare / fused step) is batch-variant. Corroborated by vLLM issue #27433.
- **Consistency:** 0.265 + 0.111 ≈ 0.376 = observed int4 ON. Independent, additive.

**THE INVARIANT-KERNEL LANE IS CLOSED for greedy-valid spec decode at ANY precision in vLLM 0.22.0.**

### Next lane: verify-rollback (arxiv 2601.17768) — kanna PR #24

Re-verify accepted tokens after each spec step under a **fixed-shape M=1 sequential AR forward** (the greedy reference itself). Commit tokens where re-run agrees with spec-decode argmax; roll back where they disagree. Rollback probability per spec step K=6 ≈ 2.2% (at 0.376%/tok). Rollback overhead ≈ 2.2% × 7ms = negligible. **This is THE priority.** Unlocks the entire drafter ladder (rungs 4–5, ~285 → 420–550 TPS).

---

## VALIDITY INSTRUMENT FINDING (wirbel #22 MERGED, 2026-06-13) — TWO fold classes, two detectors

**PR #22 terminal (MERGED): both OUTPUT gates are blind to the argmax-safe LF29 fold. Mechanism inspection is the only detector for that class.** This refines the earlier "greedy_gate is the load-bearing detector for fold-class lanes" — it depends on which class:

| fold class | example | does it change argmax? | same-path PPL | greedy_gate | the detector |
|---|---|---|---|---|---|
| **argmax-CHANGING** | lmhead12k clip (ubel #14): int4 argmax falls outside kept_ids ~1.33%/tok | **yes** → DIVERGENT | blind (PPL *improved* to 1.9767 via −inf denominator inflation) | **CATCHES it** ✓ | **greedy_gate** |
| **argmax-PRESERVING grader-conditional fold** | LF29 affine fold (pupa, wirbel #22): exact FFN when `num_prompt_logprobs`, cheap fold for decode | **no** → 0 flips / 65,536 | blind (gap 0.0000 even fold-forced-ON; teacher-forced-neutral) | **blind** (GREEDY_IDENTICAL, 0 flips) | **static mechanism inspection** of the `serve.py` branch |

- **Part A (kenyan-duma precache):** PASS — gap 0.0000 / SAME_PATH_OK, bit-identical NLL. Honest single-path confirmed. `submissions/fa2sw_precache_kenyan/` now an **in-repo VALID base** for future TPS work.
- **Part B (pupa-lf29cap444):** same-path gap 0.0000 **and** greedy flip_rate 0/65,536. The deployed fold is teacher-forced-PPL-neutral AND argmax-safe, so NO output gate flags it; only static inspection of the `num_prompt_logprobs` branch (`serve.py:411-415`) does. The community 2.55 is reproduced by neither gate (likely a reconstructed R²≈0.80 fold or non-greedy regime, not pupa's deployed weights). W&B `jg99477i`/`tju905db`/`gz5b064e` — all verified.
- **Net rule:** `greedy_gate` catches argmax-changing optimizations (the lmhead12k class — non-negotiable on every HF-approval issue); `--check-same-path` catches `prompt_logprobs` logit-path splits; **neither catches argmax-preserving grader-conditional folds** → a static mechanism-scanner is the missing instrument (candidate next direction).

Board post (Issue #29) HELD for human approval — advisor verified the W&B evidence but is NOT approving publication.

---

## Current focus — drafter ladder unlock via verify-rollback

The weight-byte floor is reached (int4 g128 + lm_head, PR #4, 126.378 TPS). All further TPS headroom requires the **drafter ladder**:

| rung | mechanism | TPS target | gate |
|---|---|---|---|
| int4 g128 + lm_head (**current**) | weight-byte floor | **126.378** | MERGED ✅ |
| + drafter (MTP K≈6) | ~3.3 accepted tok/step | ~285 | verify-rollback (kanna #24) |
| + lmhead12k + fa2sw + onegraph + precache | verify cost + runtime | ~420 | above + ubel #14 |
| + width-4 tree decoding (K=6) | E ratio 1.59×, overhead **1.06×** (measured) | **~347 @ p=0.68 / 393 @ p=0.78** | above (denken #26 MERGED) |
| + deep-K tree (K≈10) + EAGLE-3 full-scale | higher acceptance + quality | ~500+ (extrapolated) | verify-rollback + fern #25 + M-sweep (denken #28) |

The **acceptance lever** is the only one with real headroom above ~424 TPS (fableous confirmed). **Tree-salvage characterized (denken #26 MERGED):** our EAGLE-3 head rescues **0.565** of linear misses (beats fableous 0.431), E[accept] jumps **1.59×** at K=6, and crucially the tree-verify overhead is only **1.06×** (not the feared 4×) because the int4 verify forward is bandwidth-bound/flat-in-M (PR #18). Net K=6 tree = ~347 TPS @ p=0.68, ~393 @ full-scale p=0.78; **>500 only at deep K≈10 where M≈41 is extrapolated** beyond PR #18's measured M≤16 → denken #28 measures it. The drafter quality (EAGLE-3, fern #25) sets the ceiling.

**Single-stream decode is memory-bandwidth-bound** (~92% weight-GEMM at M=1). At M=K+1 spec, the verify latency (~7ms) dominates. More accepted tok/verify-step = fewer verify-step invocations = the only path to TPS > 424.

---

## Active assignments

| student | PR | track | status |
|---|---|---|---|
| kanna | **#24 (WIP, NEW)** | **Verify-rollback gate (arxiv 2601.17768)** — intercept spec-decode accepted tokens, re-verify under M=1 fixed-shape AR forward, commit matches / rollback mismatches. Goal: flip_rate → 0 (greedy-identical) + net-positive TPS over int4 AR. LOCAL ONLY. | **THE LINCHPIN NEXT LANE — #1 priority** |
| fern | **#25 (WIP, NEW)** | **EAGLE-3 full-scale training** — extend PR #16 harness: 2000 MATH + 500 ShareGPT samples corpus, 20k steps, warmup 500, lr=1e-4. Target: tf_acceptance_rate ≥ 0.78 (held-out). Offline training, no serving, no HF Job. Readies highest-ceiling drafter for verify-rollback unlock. | Ready to train |
| denken | **#28 (WIP, NEW)** | **Extended verify-latency M-sweep** — measure int4 verify latency at M∈{20,24,28,32,40,48,64} (tree K=6→M=25, K=10→M=41), re-run tree_acceptance_model on MEASURED (not extrapolated) curve, settle >500@p=0.78 question. LOCAL ONLY. (#26 MERGED: tree-salvage characterized — rescue 0.565, E ratio 1.59×, overhead 1.06×.) | Active |
| wirbel | **#30 (WIP, NEW)** | **Frontier decode-step profile** on the in-repo `fa2sw_precache_kenyan` honest base — decompose the ~420 TPS decode cycle (drafter / verify-body int4-GEMM / lmhead12k / fa2sw attn / sampling / host overhead), validate fableous's ~1.4ms drafter / ~7ms verify split, name the single next TPS lever for the team. LOCAL ONLY. (#22 MERGED — honest frontier in-repo + LF29 dual-gate-blind finding.) | Active |
| stark | #23 (WIP) | **int4 spec-verify greedy flip-rate probe** — fp32-logit, deterministic-reduction, both configs; 4 arms across int4 base; measure which (if any) drives flip_rate → 0. Complements kanna's verify-rollback via different mechanism. LOCAL ONLY. | Active |
| ubel | #14 (WIP, ↩ sent back) | **Empirical lmhead12k** (DRAFTER-INDEPENDENT). bf16-selected kept_ids → greedy DIVERGENT (PPL-blind). ubel's 15:32Z correction: the gate is SELF-CONSISTENCY (pruned-vs-pruned), so the clip floor (0.56% public / 1.05% held-out, irreducible by selection) is a **private-PPL** risk, NOT a greedy-gate criterion. Advisor 15:46Z: RETRACTED prior PRUNE-EFFECT/clip~0 criteria; instructed rebase onto advisor branch for canonical `validate_submission` (served-vs-served); approved adjusted terminal criteria. | Alive (responded 15:32Z); rebase + served-vs-served gate pending |
| land | #9 (WIP) | **Wide KL-distilled drafter** — v0 regressed −4.6% native (train↔serve schedule mismatch). v1 = free-running / EAGLE-3-style schedule + full ~82-min budget. Prerequisite for accepthist + tree-salvage on the honest stack. | Active — v1 running (CLEAN/MERGEABLE; prior conflict flag was stale) |
| lawine | **#27 (WIP, NEW)** | **int4 channel-wise lm_head sweep** — one-line change (`group_size=-1` in build_quant.py), local pre-validate + greedy check, HF approval issue if local passes. Expected ~127.4 TPS / PPL ~2.03. Establishes best lm_head quant scheme before drafter stacks pile on. | Active |

---

## Validity-instrument scope — updated (wirbel #22 terminal, MERGED)

| instrument | catches | misses |
|---|---|---|
| same-path PPL gate (PR #21) | Logit-level path splits: `prompt_logprobs` field branching to a different FFN | Argmax-preserving / decode-compounding folds (LF29-class): teacher-forced-neutral, gap 0.0000 even fold-forced-ON |
| greedy_gate (PR #8) | Free-running **argmax-CHANGING** divergence (lmhead12k clip, quant argmax shift) | **Argmax-PRESERVING grader-conditional folds (LF29): 0 flips / 65,536 — BLIND** |
| static mechanism scan (**not yet built — candidate direction**) | grader-conditional request-field branching in `serve.py` (`num_prompt_logprobs` etc.) | pure runtime/precache behavior |

**→ Every HF-approval issue must attach BOTH `greedy_gate` AND `--check-same-path` output.** They are complementary but **neither catches an argmax-safe grader-conditional fold** — that gap is closable only by static mechanism inspection.

---

## Confirmed dead ends (cycle 13 additions)

- **`VLLM_BATCH_INVARIANT=1` + greedy-valid spec decode** — CLOSED (kanna #19). Aten-override cannot reach int4 Marlin `_C` op (~0.265%/tok) OR the non-aten spec-verify residual (~0.111%/tok). Both sources are additive and independent. No lane via invariant kernels at any precision in 0.22.0.
- **Output gates (PPL + greedy) for argmax-safe grader-conditional folds** — CLOSED (wirbel #22 terminal, MERGED). Teacher-forced PPL is fold-neutral AND the deployed LF29 fold is argmax-safe (0 flips / 65,536), so BOTH same-path PPL and greedy_gate are blind. Only static mechanism inspection of the `num_prompt_logprobs` branch detects this lane. (greedy_gate IS still load-bearing for argmax-CHANGING optimizations like the lmhead12k clip — different class.)
- See BASELINE.md for the complete dead-end list (sub-4-bit, fp8 KV, n-gram, fa2sw/onegraph standalone, etc.).

---

## Potential next directions

1. **Verify-rollback gate (kanna #24)** — THE unlock. If flip_rate → 0, the entire drafter ladder is open.
2. **EAGLE-3 full-scale training (fern #25)** — produces the highest-ceiling drafter asset (~480–550 TPS). Ungated offline. (Use `debug_1k_2ep/` head, not `debug_1k/` — denken #26 provenance catch.)
3. **Verify-latency M-sweep (denken #28)** — measures int4 verify latency to M=64, replacing the tree-salvage extrapolation (M=25/M=41) with real data; settles whether >500@p=0.78 holds. (Tree-salvage itself now characterized + MERGED via #26.)
4. **Channel-wise lm_head sweep (lawine #27, active)** — `group_size=-1`, quick leaderboard data point. Low risk. Expected ~127.4 TPS / PPL ~2.03.
5. **accepthist (dynamic K)** — pupa/need-for-speed technique; separable from LF29 base. Worth a clean implementation on the honest frontier once verify-rollback unlocks serving.
6. **lmhead12k fix (ubel #14, alive)** — drafter-independent rung. The gate is SELF-CONSISTENCY (pruned-vs-pruned), so the clip floor (irreducible 0.56% public / 1.05% held-out) is a **private-PPL** risk, NOT a greedy criterion. Re-select kept_ids from int4's own argmax + served-vs-served gate via canonical `validate_submission` (ubel rebasing onto advisor branch). greedy_gate load-bearing for the argmax-changing clip; PPL blind to it.
7. **Wide drafter (land #9)** — v1 free-running schedule running; prerequisite for both accepthist and tree-salvage on the honest stack.
8. **Frontier decode profile (wirbel #30, NEW)** — decompose the ~420 TPS decode cycle on the in-repo honest `fa2sw_precache_kenyan` base to find the next TPS lever (expect verify-body int4-GEMM to dominate now that lmhead12k has cut the lm_head share). Directs the team's next round.
9. **Static mechanism-scanner for grader-conditional branching (candidate, unassigned)** — the missing validity instrument: static-scan a submission's `serve.py`/patches for model-behavior branching on grader-only request fields (`prompt_logprobs`/`num_prompt_logprobs`). The ONLY detector for argmax-safe folds (wirbel #22). Also protects our own submissions + could contribute to the human evals taskforce.

---

_Last updated: 2026-06-13 **cycle 15** — PR #22 MERGED (wirbel, validity+asset keeper): (1) honest kenyan-duma ~420 TPS frontier reproduced in-repo as VALID base `submissions/fa2sw_precache_kenyan`; (2) LF29 fold is **argmax-safe AND PPL-neutral** → BOTH output gates blind (same-path gap 0.0000, greedy 0 flips/65,536) → only static mechanism inspection detects the grader-conditional `num_prompt_logprobs` branch. W&B `jg99477i`/`tju905db`/`gz5b064e` independently verified. wirbel reassigned → #30 (frontier decode profile, next-lever finder). ubel #14 ALIVE (responded 15:32Z): advisor RETRACTED prior PRUNE-EFFECT/clip~0 criteria, instructed rebase for served-vs-served `validate_submission`, approved self-consistency terminal criteria. Issue #29 (board post) HELD, human-gated, evidence advisor-verified. All 8 students busy: kanna #24 (verify-rollback, #1 priority), fern #25, stark #23, wirbel #30, ubel #14, land #9, lawine #27, denken #28. Baseline unchanged: PR #4 126.378 TPS. Honest frontier target: kenyan-duma 421.12 / frantic-penguin 424.52 pending._

### Prior cycles (for reference)
_Cycle 14: PR #26 MERGED (denken tree-salvage cost-model: rescue 0.565 > fableous 0.431, E ratio 1.59×, verify overhead 1.06× measured, K=6 ~347 TPS / 393 @ full-scale; >500 only deep-K extrapolated → denken #28 M-sweep). Cycle 13: PR #4 MERGED (126.378 TPS baseline); PR #19 MERGED (LINCHPIN DEFINITIVE NEGATIVE: invariant-kernel lane closed, 2 un-coverable causes); PR #16 MERGED (EAGLE-3 harness, tf_acc=0.6816, use `debug_1k_2ep/`); PR #18 MERGED (cost model, ideal ceiling 1269.5 TPS at K*=15)._
