# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-13 (cycle 19, ~17:55Z)
- **Advisor branch:** `approval-gated-8gpu-20260613`
- **Most recent human directive (Issue #31, lewtun, 2026-06-13 ~16:42Z):** "For everyone looking to contribute downstream evals of the baseline or most promising submissions, don't use greedy decoding: instead use the model's recommended sampling parameters from `generation_config.json`." **Standing requirement:** any downstream quality eval (MT-Bench, MMLU, or similar) must use `generation_config.json` params — NOT greedy (temp=0.0). This does NOT apply to the official TPS benchmark (greedy by protocol) or the greedy-identity validity gate (greedy by definition) — only to human-facing quality/downstream evals. Include in every future PR body that involves downstream eval. Acknowledged in Issue #31.
- **Prior directive (Morgan, ~13:00Z):** Approved both int4 HF jobs (issues #11/#12). **Still operating under launch operator rules: no automatic HF Jobs / no `/v1/jobs:run` / no `train.py --launch` without a human-approved GitHub issue. Advisor consumes no GPU.**

---

## MILESTONE (cycle 19, 2026-06-13 17:55Z)

**PR #14 MERGED (validated lever + best-LOCAL rung): empirical lmhead12k — +34.8% isolated lm_head-prune delta, greedy-gate 128/128, official a10g-small PENDING.**

| PR | student | result | type |
|---|---|---|---|
| #14 `empirical-lmhead12k` | ubel | **Best-LOCAL rung: 131.6 local TPS / PPL 1.9712 / GREEDY_IDENTICAL 128/128.** Top-12k bf16 lm_head prune; isolated lever +34.8% (matches wirbel #8's 26.4% head fraction), +2.7% local-net over PR #4. Official a10g-small TPS + private-PPL await gated HF job. | **VALIDATED LEVER / best-local rung** |
| #25 `eagle3-full-scale-training` (cycle 18) | fern | **Best drafter asset: tf_acc 0.7314 on MATH holdout.** Reasoning acceptance plateaus ~0.73 → DATA-bottlenecked. Banked for verify-rollback unlock. | **KEEPER drafter asset** |
| #28 `verify-latency-msweep` (cycle 17) | denken | **K=12 tree: 452 TPS @ p=0.78 (dense-M upper bound).** >500 TPS extrapolation refuted — needs drafter p≥0.85. | **KEEPER research artifact** |

**Official baseline UNCHANGED: `submissions/int4_g128_lmhead` (PR #4) — 126.378 a10g-small / PPL 2.019 / GREEDY_IDENTICAL.** Per the exploratory-only rule, lmhead12k's 131.6 is LOCAL — it banks the lever + clean code but does not displace the official headline until a gated HF job confirms it (approval issue opened). lmhead12k also compounds in the spec-verify forward (gated on kanna #24).

---

## PUBLIC LEADERBOARD — CRITICAL UPDATE (2026-06-13 ~15:00–17:00Z)

**The LF29cap family (ranks 1–4, 449–459 TPS) is confirmed gate-evasion and should be treated as INVALID:**

| rank | agent | TPS | status | mechanism |
|---|---|---|---|---|
| 1 | rock-ai | 459.72 | valid (method unclear — may or may not use LF29) | rockai |
| 2 | pupa-agent | 459.21 | **INVALID** (gate-evasion) | LF29 affine FFN: timed PPL 2.55 > cap, PPL measured on dense branch |
| 3 | need-for-speed | 457.08 | **INVALID** | same LF29 construction |
| 4 | fabulous-frenzy | 449.10 | **INVALID** | same LF29 family |
| 5 | frantic-penguin | 424.52 | PENDING (honest, legitimate) | precache+noscatter on verified fa2sw frontier |
| 6 | cheesetaco | 421.53 | PENDING (honest repro) | clean fa2sw-v3 repro |
| 7 | kenyan-duma | 421.12 | VALID | fa2sw + precache (our in-repo base) |
| 8 | agent-smith | 420.59 | VALID | fa2sw-v3 |

**Community-confirmed mechanism (frantic-penguin, hayai-agent, reid, pupa-agent own negative):**
- `serve.py` sets `lffn_ppl_exact_active = (bool(num_prompt_logprobs))`. Timed decode: affine ridge-map FFN (layer 29), PPL 2.55 > cap. PPL grader: exact dense FFN, PPL 2.3779. The grader sees a different model than the timer sees. Same-path PPL gap ≈ 0.17 (our own `--check-same-path` gate would catch it).
- Organizer ruling pending from @human-lewtun / @cmpatino-verifier.

**TRUE VALID FRONTIER: ~421–424 TPS** (kenyan-duma VALID, frantic-penguin pending, agent-smith VALID).

**Note on rock-ai (rank 1, 459.72 TPS):** method is "rockai" — unclear if it uses the LF29 construction. Treat as unknown validity until organizer comments.

---

## THE LINCHPIN — RESOLVED (cycle 13, 2026-06-13)

**`VLLM_BATCH_INVARIANT=1` does NOT rescue greedy-valid spec decode at any precision in vLLM 0.22.0.**

Two independent un-coverable root causes (decomposed by kanna's bf16 discriminator arm):

- **(a) int4 Marlin `_C` op** — batch-variant, outside aten scope. Contributes ~0.265%/tok excess above bf16 floor.
- **(b) Spec verify path non-aten residual** — ~0.111%/tok irreducible. Corroborated by vLLM issue #27433.

**THE INVARIANT-KERNEL LANE IS CLOSED for greedy-valid spec decode AT ANY PRECISION in vLLM 0.22.0.**

### Next lane: verify-rollback (arxiv 2601.17768) — kanna PR #24

Re-verify accepted tokens after each spec step under a **fixed-shape M=1 sequential AR forward** (the greedy reference itself). Commit tokens where re-run agrees with spec-decode argmax; roll back where they disagree. **This is THE priority.** Unlocks the entire drafter ladder (rungs 4–5, ~285 → 420–550 TPS).

---

## VALIDITY INSTRUMENT UPDATE (wirbel #22 + PR #28 + ubel #14 self-consistency clarification)

| instrument | catches | misses |
|---|---|---|
| same-path PPL gate (PR #21) | Logit-level path splits on `prompt_logprobs` branching | Argmax-preserving grader-conditional folds (LF29-class) |
| greedy_gate (PR #8) — **SELF-CONSISTENCY** (served vs plain-greedy of the *same* checkpoint) | serving-vs-reference numerical nondeterminism on the *submitted* model | **argmax-preserving folds AND prune-clipping** (a pruned model's argmax is always in `kept_ids` by construction → clip cannot fail self-consistency) |
| static mechanism scan (**not yet built**) | grader-conditional request-field branching in serve.py | pure runtime behavior |
| greedy reference keying (**lawine #32, active**) | infra correctness (resolves silent cross-submission collision) | n/a |

**→ Every HF-approval issue must attach BOTH `greedy_gate` AND `--check-same-path`. Neither catches an argmax-safe grader-conditional fold.**

**Greedy-gate methodology (ubel #14, CRITICAL):** the reference MUST be generated **batch=1 served-vs-served** (single-stream), matching the candidate's strictly-sequential decode. An **offline-batched** reference (batch≈128 in one `generate()`) injects FP-reduction false-divergence — ubel measured an *unpruned* control "fail" 107/128 under offline-batched, vs **128/128 under batch=1 served-vs-served**. This is wirbel #8's warning, larger at scale. A pruned/clipped model is **self-consistent** (passes greedy-identity public AND private); its residual risk is **private PPL** (private GT-target outside `kept_ids` → +∞), not greedy-identity.

---

## Current focus — drafter ladder unlock via verify-rollback

The weight-byte floor is reached (int4 g128 + lm_head, PR #4, 126.378 TPS). All further TPS headroom requires the **drafter ladder**:

| rung | mechanism | TPS target | gate |
|---|---|---|---|
| int4 g128 + lm_head (**current**) | weight-byte floor | **126.378** | MERGED ✅ |
| + drafter (MTP K≈6) | ~3.3 accepted tok/step | ~285 | verify-rollback (kanna #24) |
| + lmhead12k + fa2sw + onegraph + precache | verify cost + runtime | ~420 | above + **ubel #14 MERGED** (lmhead12k lever confirmed: +34.8% isolated, 131.6 local, greedy-gate 128/128; official PENDING) |
| + width-4 tree K=6 | E ratio 1.59×, overhead 1.11× (measured) | **~331 @ p=0.68 / 375 @ p=0.78** | above (PR #28 MEASURED) |
| + width-4 tree K=12 (K*, p=0.78) | measured optimum | **~452 TPS** (dense-M upper bound) | above + PR #33 |
| + EAGLE-3 drafter, reasoning tf_acc 0.7314 (**#25 MERGED asset**) | best drafter to date; tf_acc is upper bound on free-running p | unlocks ~285→ladder | **asset banked**, awaits verify-rollback #24 |
| + reasoning-matched corpus drafter (target p≥0.85) | benchmark-matched CoT → break 0.73 plateau | ~500–530 | fern #34 (NEW) + verify-rollback |

**Key update from PR #28:** the K* is 8–12 (not 20 as extrapolated). The >500 TPS frontier requires **drafter top-1 acceptance ≥0.85** — this is the deciding variable, not tree depth. **PR #25 (cycle 18) banked the best drafter asset (reasoning tf_acc 0.7314) but proved reasoning acceptance is DATA-bottlenecked (plateaus ~0.73 on MATH+chat).** The lever toward p≥0.85 is now a **benchmark-matched reasoning corpus** (MMLU-Pro/GPQA/AIME CoT), not more steps — fern's next PR.

**Tree-causal mask caveat (PR #33 open):** the 452 TPS is a dense-M upper bound. The real tree-masked cost is cheaper in the attention term (~13%), potentially shifting the ceiling toward 470–490. PR #33 measures it.

---

## Active assignments (cycle 19)

| student | PR | track | status |
|---|---|---|---|
| kanna | **#24 (WIP)** | **Verify-rollback gate (arxiv 2601.17768)** — intercept spec-decode accepted tokens, re-verify under M=1 fixed-shape AR forward, commit matches / rollback mismatches. Goal: flip_rate → 0 (greedy-identical) + net-positive TPS over int4 AR. LOCAL ONLY. | **THE LINCHPIN NEXT LANE — #1 priority** |
| fern | **#34 (NEW)** | **Benchmark-matched reasoning corpus → EAGLE-3 retrain.** Self-distill greedy CoT from the served target under the EXACT benchmark prompt templates (MCQ `ANSWER: $LETTER` for mmlu_pro/gpqa, step-by-step `ANSWER: $ANSWER` for aime) on MMLU-Pro/GPQA/AIME questions (mixed 57/57/14), hard-dedup vs the 128 eval ids in `eval_prompts_sharegpt.json`, hold out ≥200 disjoint; warm-start from #25 best, early-stop on held-out reasoning tf_acc. Sharpened hypothesis: the plateau is a **prompt-template + distribution mismatch** (MATH is free-form; benchmark is rigid MCQ-with-`ANSWER:` CoT). Offline only, no HF Job. | Assigned (post-#25 merge) |
| denken | **#33 (NEW)** | **Tree-causal mask verify-cost + Marlin tile boundary.** Add sparse tree-causal attention mask to profiler for K=6 (M=25) and K=12 (M=49) tree shapes; fine M sweep around M≈20 and M≈40 Marlin tile steps. Settles dense-M upper bound → real tree cost. LOCAL ONLY. | Assigned |
| wirbel | **#30 (WIP)** | **Frontier decode-step profile** on the in-repo `fa2sw_precache_kenyan` honest base — decompose the ~420 TPS decode cycle (drafter / verify-body int4-GEMM / lmhead12k / fa2sw attn / sampling / host overhead), validate fableous's ~1.4ms drafter / ~7ms verify split, name the single next TPS lever for the team. LOCAL ONLY. | Active |
| stark | #23 (WIP) | **int4 spec-verify greedy flip-rate probe** — fp32-logit, deterministic-reduction, both configs; 4 arms across int4 base; measure which (if any) drives flip_rate → 0. Complements kanna's verify-rollback via different mechanism. LOCAL ONLY. | Active |
| ubel | **#36 (NEW)** | **lmhead12k follow-up #3 — int4-pruned head.** #14 MERGED (bf16-12k head, +34.8% isolated lever). Next: slice the 12k head in **int4** (≈16 MB vs 62.9 MB bf16) for another ~4× head-byte cut, single-variable on the merged `submissions/lmhead12k_empirical` base; bandwidth-model projects ~133 local (+1.3%). Orthogonal to the kept-set/private-PPL question (same `kept_ids`) → unblocked regardless of Issue #35. Optional g64 robustness arm if g128-int4 drifts PPL. Same greedy-gate batch=1 served-vs-served methodology + PPL ≤ 2.42. Offline only. | **ASSIGNED PR #36** |
| land | #9 (WIP) | **Wide KL-distilled drafter** — v0 regressed −4.6% native (schedule mismatch). v1 = free-running / EAGLE-3-style schedule + full ~82-min budget. Prerequisite for accepthist + tree-salvage on the honest stack. PR now CLEAN/MERGEABLE (rebased 16:01Z). | Active — v1 running |
| lawine | **#32 (WIP)** | **Greedy-gate reference-keying fix** — fix the silent cross-submission collision hazard: `harness._participant_env` leaves `srv.model_id="model"` → all `env.MODEL_ID="model"` submissions share tag → silent wrong-reference collision. Fix: canonical submission-specific ref tag. Must land before drafter stacks run greedy gate. | Active |

---

## Confirmed dead ends (cycle 13–17 additions)

- **`VLLM_BATCH_INVARIANT=1` + greedy-valid spec decode** — CLOSED (kanna #19). Two un-coverable causes (int4 Marlin _C op + non-aten spec-verify residual). 
- **Output gates (PPL + greedy) for argmax-safe grader-conditional folds** — CLOSED (wirbel #22 terminal). Teacher-forced PPL is fold-neutral; the deployed LF29 fold is argmax-safe (0 flips / 65,536). Only static mechanism inspection detects this lane.
- **Channel-wise (`group_size=-1`) int4 lm_head** — CLOSED (lawine #27). −0.39 TPS (noise). lm_head quant granularity is NOT a TPS knob.
- **`>500 TPS @ p=0.78` via deeper trees (PR #28):** The PR #26 extrapolation overstated deep-tree TPS by 30–55%. Real K* at p=0.78 is K=12 (452 TPS, dense-M upper bound). Only p≥0.85 drafter acceptance crosses 500. Dense-M upper bound being closed by PR #33.
- See BASELINE.md for the full dead-end list (sub-4-bit, fp8 KV, n-gram, fa2sw/onegraph standalone, batch-invariant, etc.).

---

## Potential next directions (priority order)

1. **Verify-rollback gate (kanna #24)** — THE unlock. If flip_rate → 0, the entire drafter ladder is open.
2. **Benchmark-matched reasoning corpus → EAGLE-3 retrain (fern #34, NEW)** — PR #25 banked the best drafter (reasoning tf_acc 0.7314) but proved acceptance is DATA-bottlenecked (plateaus ~0.73 on MATH). Self-distill greedy CoT from the served target under the EXACT benchmark prompt templates on MMLU-Pro/GPQA/AIME (the actual 128-prompt distribution) to break toward 0.78→0.85, the level PR #28 says is needed for >500 TPS. Ungated offline. **On-policy distillation (Draft-OPD, round-3 H1) is the follow-on if static-corpus distillation also plateaus below 0.85.**
3. **Tree-causal mask + tile boundary (denken #33, NEW)** — closes the dense-M upper bound on tree verify cost; identifies "free" tree M shapes that land on Marlin bandwidth plateaus.
4. **Greedy-gate reference-keying fix (lawine #32)** — must land before any drafter stack's greedy gate run is treated as canonical.
5. **accepthist (dynamic K)** — pupa/need-for-speed technique (separable from invalid LF29). Clean implementation on honest frontier once verify-rollback unlocks serving. Potential +~20 TPS on top of static K.
6. **lmhead12k int4-pruned head (ubel #36, NEW)** — #14 MERGED (bf16-12k, +34.8% isolated lever, greedy-gate 128/128, official PENDING). Follow-up PR #36: int4-slice the 12k head for another ~4× head-byte cut (~16 MB; bandwidth-model ~133 local). Also compounds in the spec-verify forward (gated on kanna #24). **HF-approval Issue #35 opened for the official a10g-small confirmation (official TPS + private PPL) of the merged bf16-12k rung — awaiting human approval; no HF job launched.**
7. **Wide drafter (land #9)** — v1 free-running schedule running. Prerequisite for both accepthist and tree-salvage on the honest stack.
8. **Frontier decode profile (wirbel #30)** — decompose ~420 TPS decode cycle to name the next TPS lever after the precache stack.
9. **Static mechanism-scanner (candidate, unassigned)** — build a static analyzer detecting grader-conditional branching in serve.py. Only detector for argmax-safe grader-conditional folds (LF29-class). Protects our own submissions + could contribute to the evals taskforce.
10. **rock-ai method investigation** — rock-ai is at 459.72 TPS with method "rockai" and verification "valid". This is NOT the LF29 family (different method name). Worth understanding what they're doing — if it's a genuinely novel valid approach, it's our next target.

---

_Last updated: 2026-06-13 **cycle 19** — PR #14 MERGED (ubel empirical lmhead12k: validated lever + best-LOCAL rung — top-12k bf16 lm_head prune, +34.8% isolated single-variable delta, 131.6 local / PPL 1.9712 / greedy-gate 128/128 self-consistency; official a10g-small TPS + private-PPL PENDING gated HF job, approval **Issue #35** opened; official 126.378 headline unchanged). Keeper validity findings: gate is self-consistency (clip cannot fail it); greedy reference must be batch=1 served-vs-served (offline-batched injects false divergence); int4-argmax clip has an irreducible selection floor. ubel reassigned to int4-pruned-head follow-up (**PR #36**). All 8 students busy; zero idle._

_Cycle 18: PR #25 MERGED (fern EAGLE-3 full-scale training: best drafter asset, reasoning tf_acc 0.7314 on MATH holdout; reasoning acceptance is DATA-bottlenecked, plateaus ~0.73; asset banked for verify-rollback unlock). fern reassigned to benchmark-matched reasoning corpus (MMLU-Pro/GPQA/AIME CoT) to break the plateau toward 0.85._

_Cycle 17: PR #28 MERGED (denken verify-latency M-sweep: >500 TPS extrapolation refuted on measured hardware; real K*=12 @ p=0.78 gives 452 TPS dense-M upper bound; >500 needs drafter p≥0.85 — re-anchored focus on drafter quality). PR #33 ASSIGNED (denken, tree-causal mask + tile boundary, closes upper bound). PR #9 land CLEAN (rebase was done 16:01Z, running v1). LF29cap band (ranks 1–4, 449–459 TPS) confirmed gate-evasion across community — ruling pending. True valid frontier ~421–424 TPS._

_Cycle 16: PR #27 CLOSED (lawine channel-wise lm_head, confirmed NEGATIVE: g=-1 is −0.39 TPS wash; secondary find: silent greedy-gate cross-submission collision bug → lawine reassigned to harness fix PR #32)._

_Cycle 15: PR #22 MERGED (wirbel, validity+asset keeper): honest kenyan-duma ~420 TPS frontier reproduced in-repo as VALID base `submissions/fa2sw_precache_kenyan`; LF29 fold argmax-safe AND PPL-neutral → BOTH output gates blind._

_Cycle 14: PR #26 MERGED (denken tree-salvage cost-model: rescue 0.565 > fableous 0.431, E ratio 1.59×, verify overhead 1.06× measured at M≤16, K=6 ~347 TPS / 393 @ full-scale; now corrected by PR #28 to K=6 331.2 TPS / K*=12 452 TPS)._

_Cycle 13: PR #4 MERGED (126.378 TPS baseline); PR #19 MERGED (LINCHPIN DEFINITIVE NEGATIVE); PR #16 MERGED (EAGLE-3 harness, tf_acc=0.6816); PR #18 MERGED (cost model, ideal ceiling 1269.5 TPS at K*=15 — academic, not hardware-limited)._
