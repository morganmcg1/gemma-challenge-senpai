# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-13 (cycle 20, ~18:58Z)
- **Advisor branch:** `approval-gated-8gpu-20260613`
- **Most recent human directive (Issue #31, lewtun, 2026-06-13 ~16:42Z):** "For everyone looking to contribute downstream evals of the baseline or most promising submissions, don't use greedy decoding: instead use the model's recommended sampling parameters from `generation_config.json`." **Standing requirement:** any downstream quality eval (MT-Bench, MMLU, or similar) must use `generation_config.json` params — NOT greedy (temp=0.0). This does NOT apply to the official TPS benchmark (greedy by protocol) or the greedy-identity validity gate (greedy by definition) — only to human-facing quality/downstream evals. Include in every future PR body that involves downstream eval. Acknowledged in Issue #31.
- **Prior directive (Morgan, ~13:00Z):** Approved both int4 HF jobs (issues #11/#12). **Still operating under launch operator rules: no automatic HF Jobs / no `/v1/jobs:run` / no `train.py --launch` without a human-approved GitHub issue. Advisor consumes no GPU.**
- **HF JOB APPROVED + IN FLIGHT (Issue #35, Morgan, 2026-06-13 ~18:10Z):** "HF Job launch authorized - godspeed" for `lmhead12k_empirical` (merged PR #14). Advisor cannot launch HF jobs → routed the single-launch to **ubel** (submission owner) on PR #36 (`train.py --submission submissions/lmhead12k_empirical --launch --wait`, launch-exactly-once). **AWAITING official a10g-small `tps / ppl / completed`.** Decisive checks: (1) does official TPS beat 126.378 (PR #4 headline)? local probe 131.60; (2) **private-set PPL ≤ 2.42** — the one risk not closable locally (private GT-target outside frozen `kept_ids` → +∞ PPL). If both pass → promote lmhead12k to OFFICIAL baseline + update BASELINE.md.

---

## MILESTONE (cycle 19 CLOSED, 2026-06-13 ~18:30Z)

**Cycle 19 closed five PRs:**
- **#24 MERGED (kanna — verify-rollback lane CLOSED):** spec-decode-for-speed under strict M=1 greedy-identity gate is DEAD in vLLM 0.22.0, proven by cost theorem. The `TPS_VR = 1/(1/TPS_AR + 1/TPS_spec) < TPS_AR` invariant is exact and implementation-independent. Only net-positive route left: source-level batch-invariance of M=K+1 verify forward (stark #23).
- **#30 MERGED (wirbel — frontier decode composition):** 99.3% GPU-bound; verify-body int4-GEMM 53.2% (walled at floor), **fa2sw attention 19.6% (second lever — most addressable)**, drafter 15.5%, lm_head 1.0% (validates lmhead12k #14). Verify bandwidth-flat-in-M. This is the authoritative cost breakdown of the ~420 TPS stack.
- **#32 MERGED (lawine — greedy-gate reference-keying fix):** Collision hole closed (keyed `<submission_dir>::<model_id>`). `fa2sw_precache_kenyan` DIVERGENT 27/32 vs own M=1 AR under correct keying → routes to kanna's served-gate audit.
- **#33 MERGED (denken — cost-model closure):** Tree-mask dead; Marlin `ceil(M/16)` tile-boundary fix; K=11 (M=45)/440 TPS @ p=0.78; >500 stays FALSE.
- **#14 MERGED (ubel — empirical lmhead12k):** Best-LOCAL rung 131.6 / PPL 1.9712 / GREEDY_IDENTICAL 128/128. Official a10g-small PENDING (Issue #35).

| PR | student | result | type |
|---|---|---|---|
| #24 `verify-rollback-gate` | kanna | **LANE CLOSED by cost theorem.** TPS_VR 15.48 (0.69×AR) eager / 66.32 (0.71×AR) cudagraph. Identity RESTORED (flip→0, 32/32 GREEDY_IDENTICAL) but net-NEGATIVE always. | **DEFINITIVE NEGATIVE (cost theorem)** |
| #30 `frontier-decode-profile` | wirbel | **99.3% GPU-bound; verify GEMM 53.2% (floor), fa2sw attn 19.6% (addressable), drafter 15.5%, lm_head 1.0%.** E_accept=3.817. | **KEEPER characterization — ranked lever list** |
| #32 `greedy-gate-ref-keying-fix` | lawine | **collision_free=1.0, distinct_tags=2.** fa2sw DIVERGENT 27/32 vs M=1 AR (correct keying). | **KEEPER infra fix — routes to kanna** |
| #33 `tree-causal-mask-verify-cost` | denken | **Tree-mask dead (SDPA saves 0 ms); Marlin tile-boundary fix; K=11/440 @ p=0.78; >500 FALSE (firmer).** | **KEEPER cost-model fix** |
| #14 `empirical-lmhead12k` | ubel | **Best-LOCAL rung: 131.6 local TPS / PPL 1.9712 / GREEDY_IDENTICAL 128/128.** | **VALIDATED LEVER / best-local rung** |
| #37 `lmhead12k-verify-cost` | denken | **538.1 TPS K\*=11/M=45 p=0.78 with drafter (+22% over #33's 440); scatter floor 0.348 ms; K=11/M=45 serving config LOCKED; tile-fold into canonical msweep.** | **KEEPER cost-model closure + infra** |
| #40 `greedy-ref-128prompt` | lawine | **128/128 served spec-off reference generated (514.75s); bare-tag assertion wired at both sites; 8/8 tests; self-consistent at batch=1. Unblocks kanna #38 at full 128-prompt scale.** | **KEEPER infra closure** |
| #39 `fa2sw-attn-profile` | wirbel | **Premise refuted: fa2sw FA2 inert, 19.6% is Triton unified (2D, 4.7% BW floor). Root cause: `max_seqlen_q>1` gates 3D split-KV off. 4.14× measured. Projects ~471–505 TPS. HIGHEST-LEVERAGE GREEDY-SAFE LEVER.** | **KEEPER lever discovery** |

**Official baseline UNCHANGED: `submissions/int4_g128_lmhead` (PR #4) — 126.378 a10g-small / PPL 2.019 / GREEDY_IDENTICAL.**

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

**TRUE VALID FRONTIER: ~421–424 TPS** (kenyan-duma VALID, frantic-penguin pending, agent-smith VALID).

**Note on rock-ai (rank 1, 459.72 TPS):** method is "rockai" — unclear if it uses the LF29 construction. Treat as unknown validity until organizer comments.

**Cap-tier characterization (firfir-cast, 3 INVALID runs, 2026-06-13):** cap=445/448/475 all show ~7.2–7.3% private-set TPS drop → INVALID by the 5% repro rule (verifier post `20260613-185613-207_cmpatino-verifier.md`). Confirms the 445–459 `DECODE_TPS_CAP` tier dies on private re-run. **Our target stays the legitimate ~420 VALID frontier (kenyan-duma 421.12).**

---

## THE LINCHPIN — FULLY CLOSED (cycle 19, 2026-06-13)

**Spec-decode-for-speed under a strict M=1-greedy-identity gate is DEAD in vLLM 0.22.0.**

Two closed routes:
- **(a) `VLLM_BATCH_INVARIANT=1` — kanna #19:** Definitive negative. Two un-coverable causes: int4 Marlin `_C` op (batch-variant, outside aten) + non-aten spec-verify residual.
- **(b) Verify-rollback (arxiv 2601.17768) — kanna #24:** DEAD by cost theorem. `TPS_VR = 1/(1/TPS_AR + 1/TPS_spec) < TPS_AR` always. Detecting which ~2.2% of steps roll back *is* running M=1 forward for 100% of tokens. Per-token M=1 → identity ✓ speed ✗; batched M=K → speed ✓ identity ✗.

**ONE NET-POSITIVE ROUTE REMAINS:** source-level batch-invariance of the M=K+1 verify forward (stark #23) — would make spec valid with zero rollback, strictly dominating VR.

**OPEN QUESTION (kanna #38):** Is our strict M=1 bar over-conservative vs the leaderboard's served gate? fa2sw_precache_kenyan is leaderboard-valid at ~424.5 TPS but DIVERGENT 27/32 vs M=1 AR under correct keying (#32). The leaderboard likely enforces a served-spec-on vs served-spec-off gate (not strict M=1) — if so, spec submissions through the weaker gate are already valid and stark #23's batch-invariance work becomes even higher priority.

---

## VALIDITY INSTRUMENT UPDATE

| instrument | catches | misses |
|---|---|---|
| same-path PPL gate (PR #21) | Logit-level path splits on `prompt_logprobs` branching | Argmax-preserving grader-conditional folds (LF29-class) |
| greedy_gate (PR #8) — **SELF-CONSISTENCY** (served vs plain-greedy of the *same* checkpoint) | serving-vs-reference numerical nondeterminism on the *submitted* model | **argmax-preserving folds AND prune-clipping** |
| reference keying fix (**lawine #32, MERGED**) | infra correctness (resolves silent cross-submission collision) | n/a |
| **Served-gate reconciliation (kanna #38, ACTIVE)** | whether leaderboard's served gate matches our strict M=1 bar | TBD |

**→ Every HF-approval issue must attach BOTH `greedy_gate` AND `--check-same-path`. Neither catches an argmax-safe grader-conditional fold.**

---

## DRAFTER TRAIN↔SERVE GAP — NEW FINDING (land #9 v1b, 2026-06-13 ~18:58Z)

**The offline teacher-forced (tf) acceptance gate is NOT a faithful proxy for native HF assisted-generation acceptance.** Two independent drafter training schedules (v0 teacher-forced, v1b free-running) each lifted the tf gate +10–16% while native serving acceptance *regressed* −5–6% (stock 3.553 → v1b 3.341 native). tf and native are **anti-correlated** under our training → rules out exposure bias, points at **interface fidelity**: our objective conditions the draft's step-0 hidden on the target's ground-truth hidden, but HF native assisted-generation feeds the draft its *own* running hidden over accumulated KV. Fine-tuning to excel on the former drifts off the serving optimum (the un-fine-tuned stock draft sits ON it at 3.553).

**Programme implications:**
- Drafter-quality work must be gated on **native** accepted-tok/step (`heldout_native_accept_per_step`), NOT the tf proxy. Propagated to fern #34 (native cross-check requested alongside `tf_acc`).
- The lever is **serve-faithful (HASS-style) training** — feed the draft its own running hidden, matching the serve path. land #9 pivoted to this (the only thing that should convert tf→native).
- Directly serves the #1 programme risk (**private-stable acceptance**): the verifier just invalidated `firfir-cast` for a 7.2% private TPS drop; the legitimate ~420 frontier is won on acceptance, not on `DECODE_TPS_CAP` gaming.

---

## Current focus — decode lever prioritization

The weight-byte floor is reached (PR #4, 126.378 TPS). The frontier stack (`fa2sw_precache_kenyan`) is at ~420 TPS. PR #30 has now **ranked the remaining levers by cost fraction:**

| lever | fraction | addressable? | who |
|---|---|---|---|
| Verify-body int4 GEMM | 53.2% | **NO** — walled at int4-Marlin floor in vLLM 0.22.0 | (closed) |
| **Triton verify-attention (2D path, occupancy-bound)** | **19.6%** | **YES — GREEDY-EXACT** — patch `max_seqlen_q > 1` guard → 3D split-KV; measured **4.14×** at M=1; projects ~471–505 TPS | **wirbel #43 (IMPLEMENTATION)** |
| Drafter (quality / acceptance rate) | 15.5% | YES — better drafter acceptance (p toward 0.85) | fern #34 |
| lm_head | 1.0% | NO — already exploited by lmhead12k (#14) | (exhausted) |

**KEY UPDATE (PR #39):** The "fa2sw attention" label was wrong. The `fa2sw` FA2 path is INERT (vLLM forces Triton for heterogeneous head dims). The 19.6% is 98.1% Triton `kernel_unified_attention` running at 4.7% BW efficiency (occupancy-bound, not BW-bound). Root cause: M=8 verify falls on 2D path because `max_seqlen_q > 1` gates 3D split-KV off. **This is the single highest-leverage greedy-safe lever in the programme** — the fix is already ~90% in vLLM.

**The k* drafter ladder (spec-decode angle):** requires either (a) batch-invariance fix (stark #23) or (b) leaderboard's served gate being more permissive than our strict M=1 bar (kanna #38 reconciliation). If (b) is true, the drafter ladder is already unlocked for the frontier — stark's fix becomes a hardening step, not a gate.

| rung | mechanism | TPS target | gate |
|---|---|---|---|
| int4 g128 + lm_head (**current official**) | weight-byte floor | **126.378** | MERGED ✅ |
| + drafter (MTP K≈6) | ~3.3 accepted tok/step | ~285 | batch-invariance (stark #23) OR served-gate wider than M=1 (kanna #38) |
| + lmhead12k + fa2sw + onegraph + precache | verify cost + runtime | ~420 | above + **ubel #14 MERGED** |
| + width-4 tree K=11 (K*, p=0.78, tile-corrected, full head) | measured optimum, M=45 plateau | **~440 TPS** | above + PR #33 |
| + lmhead12k on verify path (K*=11/M=45, p=0.78) | −19.8% verify step (PR #37 measured) | **~538 TPS ceiling** (measured, scatter floor included) | above + PR #37 |
| + EAGLE-3 drafter, reasoning tf_acc 0.7314 (**#25 MERGED asset**) | best drafter to date | unlocks ~285→ladder | asset banked |
| + reasoning-matched corpus → p≥0.85 | benchmark-matched CoT | ~500–530 | fern #34 + gate |
| + **3D split-KV verify dispatch (wirbel #43, ACTIVE)** | 50–82% attn saving, greedy-exact | **~471–505 TPS** | wirbel #43 (patch `max_seqlen_q>1` guard) |
| + int4-pruned 12k head | ~4× head-byte cut (ubel #36) | ~133 local | ubel #36 |

---

## Active assignments (cycle 19 closure)

| student | PR | track | status |
|---|---|---|---|
| kanna | **#38 (NEW)** | **Served-gate validity audit.** Does `fa2sw_precache_kenyan` pass the SERVED greedy gate (spec-on vs spec-off, batch=1)? Reconcile 27/32 M=1-offline divergence (#32) with leaderboard-valid ~424.5 TPS status. Is our strict M=1 bar over-conservative? LOCAL ONLY. | **Assigned (post-#24 merge)** |
| wirbel | **#43 (NEW)** | **3D split-KV dispatch for M>1 verify.** Patch `max_seqlen_q > 1` guard in Triton `unified_attention`; extend per-segment softmax reduction to multiple query rows; measure served TPS. Greedy-exact (bit-identical). Projects ~471–505 TPS. LOCAL ONLY. | **Assigned (post-#39 merge)** |
| lawine | **#42 (NEW)** | **`--spec-off` contract fix + validator N-mismatch legibility.** Teach spec `serve.py` to honor `SENPAI_REFERENCE_MODE`; clean up `--ref-env` workaround; surface `num_records` into `evidence.json` with N-mismatch warning. LOCAL ONLY. | **Assigned (post-#40 merge)** |
| fern | **#34 (WIP)** | **Benchmark-matched reasoning corpus → EAGLE-3 retrain.** Self-distill greedy CoT from served target under EXACT benchmark prompt templates (MCQ `ANSWER: $LETTER` for mmlu_pro/gpqa, step-by-step `ANSWER: $ANSWER` for aime) on MMLU-Pro/GPQA/AIME (57/57/14), hard-dedup vs 128 eval ids, early-stop on held-out reasoning tf_acc. Target: break 0.73 plateau toward 0.78–0.85. | Active — **native accept/step cross-check requested** (land #9: tf_acc may not convert to native) |
| denken | **#41 (NEW)** | **Scatter floor elimination in `compute_logits`.** Prove `kept_ids[argmax(partial_12k_logits)]` == scatter+full-argmax (empirical greedy-identity guarantee), then implement the skip to save ~0.155 ms/step @ M=45. Ceiling 538→~546 TPS. LOCAL ONLY. | **Assigned (post-#37 merge)** |
| stark | **#23 (WIP)** | **Source-level batch-invariance probe** — fp32-logit, deterministic-reduction arms; measure which (if any) drives flip_rate → 0. The only remaining net-positive route to greedy-valid spec decode in vLLM 0.22.0. | Active — **#1 spec-decode priority** |
| ubel | **#36 (WIP)** | **lmhead12k int4-pruned head.** Slice the 12k head in int4 (≈16 MB vs 62.9 MB bf16) for another ~4× head-byte cut. Bandwidth-model ~133 local (+1.3%). Same `kept_ids`, orthogonal to private-PPL question. **PLUS high-priority operational interrupt: launch the human-approved (Issue #35) HF benchmark for the merged `lmhead12k_empirical` — official TPS + private-PPL test; run in parallel with #36.** | Active + HF launch (**Option A approved**, Issue #35: host ckpt→repoint MODEL_ID→smoke greedy 128/128→launch-once) |
| land | **#9 (WIP)** | **Wide KL-distilled drafter** — v1b free-running: tf +15.9% but native −6.0% → **tf gate is not serve-faithful** (interface fidelity, not exposure bias). **Sent back: rebase (heldout.jsonl conflict) + pivot to HASS-style serve-faithful objective**, gate on native accept/step. | Active — request-changes pivot |

**All 8 students busy. Zero idle GPUs.**

---

## Confirmed dead ends (cycle 13–19 additions)

- **`VLLM_BATCH_INVARIANT=1` + greedy-valid spec decode** — CLOSED (kanna #19). Two un-coverable causes.
- **Verify-rollback (arxiv 2601.17768)** — CLOSED (kanna #24). Cost theorem: `TPS_VR < TPS_AR` always.
- **Tree-causal attention mask** — CLOSED (denken #33). Production SDPA saves 0 ms by construction.
- **>500 TPS @ p=0.78 via deeper trees** — CLOSED (denken #28/#33). Real optimum K=11 (M=45)/440 TPS; only p≥0.85 drafter acceptance crosses 500.
- **Channel-wise (`group_size=-1`) int4 lm_head** — CLOSED (lawine #27). −0.39 TPS (noise).
- **Output gates for argmax-safe grader-conditional folds** — CLOSED (wirbel #22). Teacher-forced PPL is fold-neutral.
- **Sub-4-bit, fp8 KV, n-gram, prompt-lookup, fa2sw/onegraph standalone (without precache)** — various PRs.
- See BASELINE.md for the full list.

---

## Potential next directions (priority order)

1. **3D split-KV for M>1 verify (wirbel #43)** — **THE SINGLE HIGHEST-LEVERAGE GREEDY-SAFE LEVER.** Patch `max_seqlen_q > 1` guard; ~90% already in vLLM; 4.14× measured speedup; projects 471–505 TPS on the already-valid frontier. Zero gate risk (bit-identical attention). Active.
2. **Source-level batch-invariance (stark #23)** — THE unlock for strict M=1-greedy-valid spec decode. Only net-positive route in vLLM 0.22.0.
3. **Served-gate reconciliation (kanna #38)** — if leaderboard gate is weaker than our strict M=1 bar, the drafter ladder is already unlocked for frontier submissions; stark #23 becomes a hardening step, not a gate.
4. **Benchmark-matched corpus → drafter p≥0.85 (fern #34)** — PR #25 proved reasoning acceptance is DATA-bottlenecked at 0.73; MCQ-template CoT on the actual 128-prompt distribution is the lever toward p≥0.85 (needed for >500 TPS).
5. **lmhead12k int4-pruned head (ubel #36)** — another ~4× head-byte cut; compounds with spec if batch-invariance unlocks.
6. **Greedy-ref infra 128-prompt (lawine #40)** — feeds kanna's served-gate audit with full 128-prompt data.
7. **Serve-faithful (HASS) drafter (land #9)** — tf-acc gains do NOT convert to native (land #9 v1b finding: tf +16% / native −6%). Interface-fidelity training (draft's own running hidden, matching the serve path) is the real acceptance lever and the path to **private-stable acceptance** (the #1 programme risk). Gate on native accept/step, not tf.
8. **Eliminate the scatter floor (denken follow-up from #37)** — kernel argmax over 12k partial + remap to full-vocab id; needs correctness proof that top-1 never falls outside kept_ids; ceiling ~546 vs 538 measured. Local profiling only.
9. **accepthist (dynamic K)** — clean implementation on honest frontier once spec is unlocked. Potential +~20 TPS on top of static K.
10. **rock-ai method investigation** — 459.72 TPS, method "rockai", validity status unclear. If genuinely valid and novel, it's our next target.

---

_Cycle 20 (review pass, ~18:58Z): land #9 reviewed → **request-changes pivot** (tf gate not serve-faithful; rebase + HASS serve-faithful objective, gate on native). ubel #36 → back to wip (Option A approved on Issue #35: host lmhead12k ckpt → repoint → smoke → launch-once). fern #34 → native accept/step cross-check requested. Public intake: frontier #1 rock-ai 459.72 but 445–459 cap-tier confirmed INVALID on private re-run (firfir-cast −7.2%); legitimate frontier ~420 unchanged. All 8 students busy; zero idle._

_Last updated: 2026-06-13 **cycle 22/23** — PR #39 MERGED (wirbel: fa2sw attention premise refuted; Triton 2D occupancy-bound at 4.7% BW floor; 3D split-KV fix greedy-exact; projects 471–505 TPS; HIGHEST-LEVERAGE LEVER). wirbel→#43 (implement 3D split-KV for M>1 verify). Awaiting: ubel HF result (Issue #35), land v1 verdict (PR #9, training ~done). All 8 students busy; zero idle._

_Cycle 20: Issue #35 approved (Morgan, "HF Job launch authorized"); routed single-launch to ubel (PR #36). awaiting official a10g-small tps/ppl. Cycle 19 CLOSED (~18:30Z): PRs #24/#30/#32/#33/#14 ALL MERGED. kanna→#38 (served-gate audit), wirbel→#39 (fa2sw deep-profile), lawine→#40 (greedy-ref 128-prompt + assert)._

_Cycle 18: PR #25 MERGED (fern EAGLE-3 full-scale training: best drafter asset, reasoning tf_acc 0.7314, DATA-bottlenecked). fern reassigned #34 (benchmark-matched corpus)._

_Cycle 17: PR #28 MERGED (denken verify-latency M-sweep: K*=12/452 @ p=0.78, >500 needs drafter p≥0.85). PR #33 ASSIGNED (denken tree-causal mask + tile boundary). LF29cap band (ranks 1–4) confirmed gate-evasion; true valid frontier ~421–424 TPS._

_Cycle 16: PR #27 CLOSED (lawine channel-wise lm_head, NEGATIVE). lawine reassigned #32 (greedy-gate keying fix)._

_Cycle 15: PR #22 MERGED (wirbel honest frontier ~420 TPS in-repo; LF29 fold argmax-safe AND PPL-neutral → both output gates blind)._

_Cycle 14: PR #26 MERGED (denken tree-salvage cost-model; corrected by #28→#33)._

_Cycle 13: PR #4 MERGED (126.378 TPS baseline); PR #19 MERGED (LINCHPIN DEFINITIVE NEGATIVE); PR #16 MERGED (EAGLE-3 harness); PR #18 MERGED (cost model)._
