# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-13 (cycle 29, ~22:35Z)
- **Advisor branch:** `approval-gated-8gpu-20260613`
- **Most recent human directive (Issue #31, lewtun, 2026-06-13 ~16:42Z):** "For everyone looking to contribute downstream evals of the baseline or most promising submissions, don't use greedy decoding: instead use the model's recommended sampling parameters from `generation_config.json`." **Standing requirement:** any downstream quality eval (MT-Bench, MMLU, or similar) must use `generation_config.json` params — NOT greedy (temp=0.0). This does NOT apply to the official TPS benchmark (greedy by protocol) or the greedy-identity validity gate (greedy by definition) — only to human-facing quality/downstream evals. Include in every future PR body that involves downstream eval. Acknowledged in Issue #31.
- **Prior directive (Morgan, ~13:00Z):** Approved both int4 HF jobs (issues #11/#12). **Still operating under launch operator rules: no automatic HF Jobs / no `/v1/jobs:run` / no `train.py --launch` without a human-approved GitHub issue. Advisor consumes no GPU.**
- **HF JOB Issue #35 — CLOSED (Morgan, 2026-06-13 ~21:36Z: "ok i think we can close this and move on").** The lmhead12k **bf16-head** launch arm was **retired** (cross-session argmax non-determinism — officially safe per #38 but a worse artifact). The strictly-dominant lmhead12k artifact is ubel's **int4-head (#36b)**: cross-session deterministic (int4 Marlin GEMV bit-exact), 133.3 local, PPL 1.9713 — kept as a **documented LOCAL rung** (no HF launch; official bar stays 126.378). Priority quota spend is the **split-KV frontier launch** (Issue #46 / lawine #52, ~471–493 projected). No lmhead12k launch lane remains open.
- **NEW human direction (morganmcg1, 2026-06-13 ~20:39–20:49Z):** Directly opened **4 student assignments** — stark #47 (W8A8 drafter precision probe), kanna #48 (token-frequency logit bias), wirbel #49 (Sequoia DP-optimal draft tree), lawine #50 (official_gate HF-launch preflight + staging) — and **Issue #46: HF-Job approval request for the MERGED split-KV patch (PR #43)** to measure official TPS on a10g-small. **Issue #46 APPROVED (Morgan, 20:49Z: "approved, lessgo!") → advisor routed the launch to lawine #52 → COMPLETED 481.53 official TPS, NEW PUBLIC #1 (#52 MERGED 22:13Z).** [cycle 27: #50 = the official_gate→HF-launch preflight interlock, MERGED 21:22Z; the one-shot split-KV launch ran as lawine #52.] Advisor filled the one remaining idle slot: **denken #51 (accepthist dynamic-K)**. All 8 students now busy.
- **🏆 HEADLINE — split-KV launch (#52, lawine) COMPLETED → NEW PUBLIC #1 / NEW OFFICIAL FRONTIER:** the Issue-#46-approved one-shot official HF-Jobs launch of the fa2sw split-KV frontier (linear MTP K=7 + #43 split-KV) **landed at 481.53 official a10g-small TPS** (job `6a2dce05871c005b5352c0b9`, PPL 2.3772 ✓, 128/128, all modalities). **New public #1** vs prior rock-ai 459.72 (**+4.74%**); +13.4% over the ~424.5 frontier-repro baseline; landed mid-projection (#43 projected 471–493). **The official bar all submissions must beat moves 126.378 → 481.53 TPS** (first gated HF job to confirm a frontier-scale rung; PR #52 MERGED 22:13Z, BASELINE/EXPERIMENTS_LOG updated). Pre-launch `official_gate=PASS@128` with the split-KV patch engaged (M=8 → 3D FlashDecoding every step, zero 2D fallback). Leaderboard row pending organizer re-sync. **Standing open axis (the programme's #1 risk): the private re-run gate** — kanna #44 probe predicts ~12.4% public→private on a pure-chat proxy (WOULD-FAIL >5%); kanna #55 is calibrating on this exact stack. The 481.53 is the **public** number; private stability is not yet certified.
  - _Underlying lever (#43, wirbel, MERGED cycle 25):_ 3D split-KV dispatch (`SPLITKV_VERIFY_MAX_Q=64`) routes M=8 verify through FlashDecoding instead of the occupancy-bound 2D Triton path. Local: 428.37 TPS steady-state (+10.86%), attention op **4.38×** (53.24→12.15 µs), verify GPU ms −17.5%, PPL 2.3767 — the highest-leverage greedy-safe lever, now officially confirmed at frontier scale.

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

**ANSWERED (kanna #38, MERGED 2026-06-13 ~19:14Z):** The strict M=1 bar is **NOT over-conservative** — and the leaderboard gate is **weaker than any token-identity bar**: the official HF-Jobs harness (`hf_bucket_single_job.py`) runs **no greedy-identity check** (validity = PPL + completion + modalities). So **spec-decode stacks are leaderboard-legal** (this is why the entire ~420 VALID frontier ships MTP spec). fa2sw_precache_kenyan's divergence is **run-to-run FP nondeterminism**, not spec (same-GPU spec-OFF control diverges 28/32; plain int4 baseline 29/32; `FA_SLIDING=0` → 0/32). **Consequence: stark #23's batch-invariance is a HARDENING step, NOT a gate — the drafter ladder is already unlocked for official submission.** The binding constraint is now the **private-set TPS re-run** (kanna reassigned → #44 local private-gap probe). Full audit: `research/validity/served_gate_reconciliation.md`.

---

## VALIDITY INSTRUMENT UPDATE

| instrument | catches | misses |
|---|---|---|
| same-path PPL gate (PR #21) | Logit-level path splits on `prompt_logprobs` branching | Argmax-preserving grader-conditional folds (LF29-class) |
| greedy_gate (PR #8) — **SELF-CONSISTENCY** (served vs plain-greedy of the *same* checkpoint) | serving-vs-reference numerical nondeterminism on the *submitted* model | **argmax-preserving folds AND prune-clipping** |
| reference keying fix (**lawine #32, MERGED**) | infra correctness (resolves silent cross-submission collision) | n/a |
| **Served-gate reconciliation (kanna #38, MERGED)** | official gate = PPL + completion + modalities, **no token-identity** → spec leaderboard-legal; M=1 bar not over-conservative; fa2sw non-reproducible run-to-run | private-set TPS gap (→ kanna #44 probe) |

**→ Every HF-approval issue must attach BOTH `greedy_gate` AND `--check-same-path`. Neither catches an argmax-safe grader-conditional fold.**

---

## DRAFTER TRAIN↔SERVE GAP — ROOT CAUSE MEASURED (land #9 v2, 2026-06-13 ~19:28Z)

**The offline teacher-forced (tf) acceptance gate is NOT a faithful proxy for native HF assisted-generation acceptance** — and the cause is now a *measured fact*, not a hypothesis. Two independent drafter schedules (v0 tf, v1b free-running) each lifted tf +10–16% while native serving acceptance *regressed* −5–6% (stock 3.553 → v1b 3.341). tf and native are **anti-correlated** → rules out exposure bias.

**Measured root cause (land #9 v2):** instrumenting HF's real `SinglePositionMultiTokenCandidateGenerator` (`scripts/drafter/probe_serve_hidden.py`, on #9) shows serve feeds the draft's step-0 feature as `cat(embed(token@j), f_{j-1})` — the **EAGLE convention** (hidden from the *previous* position) — in 40/43 steady-state rounds. v0/v1b training (and the offline `propose_k` tf gate) fed `f_j` (same position). The L2 gap is decisive: **≈ 3 at the correct `f_{j-1}` vs ≈ 200–355 at the `f_j` trained on.** A single step-0 position off-by-one is the *entire* interface infidelity. Bonus: served hidden ≈ clean prefill `f_{j-1}` (L2≈3) ⇒ a pure index fix closes ~99% of the gap, no simulate-verify corpus needed. (HASS 2408.15766 / EAGLE-3 2503.01840 fix the *multi-step* hidden — which land's loop already did — but neither addresses step-0; that was the hole.)

**Programme implications:**
- Drafter-quality work must be gated on **native** accepted-tok/step (`heldout_native_accept_per_step`), NOT the tf proxy. **Propagated to fern #34** — if its EAGLE-3 export/loop also feeds `f_j` at step-0, the same bug eats its corpus-quality gains; asked fern to verify the served convention (land's probe) and switch to `f_{j-1}` before banking the retrain.
- The lever is **serve-faithful (HASS-style) training** — step-0 = `f_{j-1}` + full deep-step supervision. land #9 v2 (`--serve-faithful-hidden`, `--free-run-prob 0.0`) tests exactly this; `tf_sf` (offset-−1) is logged alongside `tf`/native as the falsifiable check (tf_sf↔native together ⇒ off-by-one was the whole story).
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

## Active assignments (cycle 29, ~22:50Z) — open2-* students

**CYCLE 29 SUMMARY:** Merged #53 (post-split-KV decode re-profile: drafter-forward is new #2 at 18.1%, 1445 µs/step) + #54 (AdaEDL: entropy AUC 0.857 >> accepthist 0.60, top1p cheapest signal, +5.5% TPS projected). Closed #9 (land KL-distill NEGATIVE: 3.4899 < stock 3.5532) + #47 (old-stark W8A8 stub). **Board post done** (Morgan-approved 22:31Z): `results/20260613-224956-731_senpai.md` — 481.53 TPS officially on the public board. Verifier intel: cmpatino-verifier has ONLY invalidated `need-for-speed` (TPS-capping cap440/cap432 stacks, 5.9%/6.6% drift) — NOT spec-decode stacks. Our uncapped linear MTP K=7 is a different mechanism; private-gap risk still tracked but less acute than the 12.4% chat-proxy estimate suggests.

**Revised decode lever ranking (post-#53 re-profile):**

| lever | fraction | status |
|---|---|---|
| Verify-body int4 GEMM | 53.2% | FLOOR — cannot improve |
| **Drafter-forward** | **18.1% (1445 µs/step)** | **NEW #2 — primary target** |
| Verify-attention | 7.6% (was 19.6%) | #43 MERGED, largely spent |
| lm_head | ~1% | Exhausted |

**All 8 open2-* students assigned (8/8, zero idle GPUs):**

| student | PR | track | status |
|---|---|---|---|
| open2-alphonse | **#55** | **Private-gap probe calibration** on split-KV `fa2sw_precache_kenyan`. Anchor on firfir-cast (7.2% known-invalid) → calibrated VALID/at-risk verdict. | Assigned |
| open2-askeladd | **#57** | **W8A8 INT8 drafter quant** — A10G IMMA 2x INT8 throughput for drafter-forward (18.1% block); 15-min single-layer gate first. | Assigned |
| open2-edward | **#62** | **HASS Top-K harmonized distillation** (arXiv 2408.15766) — replace CE with Top-K loss; 5k-step warmup gate tf_acc > 0.75. | Assigned |
| open2-fern | **#63** | **Drafter acceptance profiling by category** (mmlu_pro/gpqa/aime) — locate private-gap root cause by disaggregating E_accept. Fast diagnostic. | Assigned |
| open2-frieren | **#58** | **AdaEDL entropy-dynamic K in served stack** — deploy #54 entropy controller in production; measure real TPS vs static K=7. | Assigned |
| open2-nezuko | **#60** | **Private-gap probe recalibration with reasoning-heavy proxy** — 60/40 reasoning/chat vs chat-only (12.4% baseline); go/no-go for second board post. | Assigned |
| open2-tanjiro | **#61** | **EAGLE-3 multi-step hidden fidelity retraining** (j≥2 fix) — inject main-model hiddens at each draft step; gate native accept > 3.55 at 3k steps. | Assigned |
| open2-thorfinn | **#59** | **KV-cache pre-warm prefix** — batch warming prefix to eliminate cold-start KV misses contributing to private-gap. | Assigned |

**Orphaned old-student PRs still open:** #34 (fern/reasoning-corpus), #36 (ubel/int4-head awaiting terminal), #56 (lawine/max_num_batched_tokens). Monitor for terminal results; review on merit.

## Old active assignments record (cycle 28, ~21:55Z)

**Note:** Cycle 28 reviewed + **MERGED denken #51** (accepthist dynamic-K — decisive NEGATIVE + `--sim-K` argmax bugfix keeper; official bar UNCHANGED 126.378) and closed out **kanna #48** (logit bias, MERGED cycle-27) + **Issue #35** (lmhead12k bf16-launch retired, int4-head kept LOCAL). Reassigned the three freed students: **denken → #54** (drafter-entropy dynamic-K / AdaEDL — the *corrected* accepthist lever), **kanna → #55** (private-gap calibration on the split-KV frontier, pre-launch), **ubel → verify-GEMM M=8 roofline audit** (PR opens once #36 posts its final terminal marker + merges as the int4-head local keeper). Routed denken's split-KV small-M/short-ctx finding to wirbel #53. _(Cycle 27 recap: lawine #50 interlock MERGED → lawine #52 launch; wirbel #49 Sequoia MERGED, trees closed analytically → wirbel #53.)_

| student | PR | track | status |
|---|---|---|---|
| stark | **#47 (NEW, human)** | **W8A8 drafter precision probe (QSpec-style).** INT8 tensor-core draft forward + W4A16 Marlin verify; if W8A8 CUTLASS beats Marlin for gemma-4B shapes on sm_86, draft cost drops ~30% → +5–15% end-to-end with verify path unchanged (greedy-exact). 15-min single-layer wall-time gate first; fall to Token Recycling (A2) if Marlin already wins. | Assigned (morganmcg1 ~20:39Z) |
| kanna | **#55 (NEW, advisor)** | **Private-gap probe calibration (pre-launch).** Calibrate #44's `private_gap_probe.py` against firfir-cast (known-invalid 7.2%) → quantitative predictor; apply to the split-KV `fa2sw_precache_kenyan` stack (raw 12.4%) → VALID/at-risk verdict + CI *before* lawine #52 spends the launch quota. Step 0: relocate the inert #48 logit-bias hook out of `fa2sw_precache_kenyan/sitecustomize.py`. **#48 (logit bias) MERGED 21:42Z** — decisive negative, inference-time bias lane closed. | Assigned (advisor ~21:55Z) |
| wirbel | **#53 (NEW, advisor)** | **Post-split-KV decode re-profile → next-lever ID.** #30 composition profile is pre-split-KV; #43 cut verify-attention 4.38× so the bottleneck ranking shifted. Re-profile the served linear-MTP+split-KV stack (428.37), name the new #2 block, propose a ranked quantified next lever + cross-check the in-flight levers (stark #47 / denken #51 / land #9 / fern #34). LOCAL profiling, no launch. **#49 (Sequoia DP-tree) MERGED 21:32Z** (keeper; trees closed analytically). | Active |
| lawine | **#56 (NEW, advisor)** | **max_num_batched_tokens served-config A/B on the split-KV #1 stack.** #52's launch surfaced an unexplained `max_num_batched_tokens` warning; the knob is untested on the timed path post-split-KV (prior runtime-knob dead-ends were `max_num_seqs`/attn-backend/atomic-add, NOT this). Single-variable A/B {default, 512, 2048, 8192} on `fa2sw_precache_kenyan`: steady-state local TPS + PPL + 128/128, **split-KV-engaged guardrail at each arm**. Winner → one-line config change + projection (no HF launch); parity → hardened explicit setting that silences the warning. LOCAL served only. _(#52 launch DONE → NEW PUBLIC #1 481.53.)_ | Assigned (advisor ~22:25Z) |
| denken | **#54 (NEW, advisor)** | **Drafter-entropy dynamic-K (AdaEDL).** Entropy at draft time is a strictly stronger run-length predictor than acceptance history (r≈0.32, #51) — the *correct* read of the public top-3 (459). Log per-draft-position drafter entropy, re-run the #51 oracle/realizable bake-off with entropy as predictor; report fraction of the +16.1% oracle ceiling recovered; served A/B only if projection promising. **#51 (accepthist) MERGED 21:55Z** — decisive negative (+0.12%); acceptance-history lane closed, deployed K=7 confirmed near-optimal static. | Assigned (advisor ~21:55Z) |
| fern | **#34 (WIP)** | **Benchmark-matched reasoning corpus → EAGLE-3 retrain.** Self-distill greedy CoT under exact benchmark templates (MCQ `ANSWER: $LETTER`, step-by-step `ANSWER: $ANSWER`) on MMLU-Pro/GPQA/AIME; hard-dedup vs 128 eval ids; break the 0.73 acceptance plateau toward 0.78–0.85. Step-0 `f_{j-1}` convention check propagated from land #9 (else gains won't convert to native). | Active |
| ubel | **#36 → verify-GEMM (NEW, advisor)** | **#36 int4-head = LOCAL keeper** (133.3 / PPL 1.9713 / deterministic 128/128); bf16-launch arm retired, **Issue #35 CLOSED**. Awaiting ubel's final terminal marker → merge as the canonical lm_head-lane rung (no HF launch). **Next (PR opens on #36 merge): verify-body GEMM roofline audit at M=8** — is the #1 decode block (~53%, #30) roofline-bound or weight-under-reusing at the served M=8 verify batch? Complementary to wirbel #53. | Redirect posted; marker pending |
| land | **#9 (WIP)** | **Serve-faithful (HASS) wide KL-distilled drafter.** Root cause MEASURED: a single step-0 off-by-one (serve feeds `f_{j-1}`; v0/v1b trained `f_j`; L2≈3 vs 200–355) is the whole tf↔native gap. v2 (`--serve-faithful-hidden` + `--free-run-prob 0.0`; W&B `land-hass-v2-192212`) gated on native accept/step > stock 3.553. Path to **private-stable acceptance** (#1 programme risk) + reusable across the drafter ladder. | Active — v2 training |

**7/8 students busy; 1 gap.** **lawine** reassigned → **#56** (max_num_batched_tokens served-config A/B) after completing #52 (the official launch — NEW PUBLIC #1). **ubel** is the one transient gap: #36's experiment is complete (int4-head done) and its verify-GEMM follow-up branch is staged (`ubel/verify-gemm-m8-roofline`) — the PR opens the moment #36 posts its final terminal marker + merges (the one-WIP guard blocks a 2nd WIP PR until then). Redirect posted; monitoring for the marker.

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

1. **3D split-KV for M>1 verify (wirbel #43) → OFFICIAL LAUNCH (lawine #52) — BOTH MERGED ✓; NEW PUBLIC #1.** The single highest-leverage greedy-safe lever, now officially confirmed at frontier scale: local 428.37 TPS (+10.86%), attention op 4.38×, verify GPU ms −17.5%, PPL 2.3767 → **official a10g-small 481.53 TPS / PPL 2.3772 / 128/128** (Issue #46 human-approved one-shot, job `6a2dce05871c005b5352c0b9`). **New public #1** vs rock-ai 459.72 (+4.74%); the official bar moves 126.378 → **481.53**. **Remaining frontier work is now (a) the private-stability axis** (kanna #55 calibrating the gap probe — the 481.53 is public-only, private re-run not yet certified) **and (b) the decode-block levers** (verify-GEMM ubel, drafter-forward denken #54/wirbel #53, acceptance land #9/fern #34). This is the base stack for all in-flight cost-model work.
2. **Source-level batch-invariance (stark #23)** — THE unlock for strict M=1-greedy-valid spec decode. Only net-positive route in vLLM 0.22.0.
3. **Private-gap probe (kanna #44)** — #38 ANSWERED the served-gate question: leaderboard gate = PPL+completion+modalities, NO token-identity → drafter ladder already unlocked for frontier submissions; stark #23 is a hardening step, not a gate. The binding constraint is now the **private TPS re-run** (honest stacks lose 4–9%, >5%=INVALID). #44 builds a LOCAL early-warning probe (chat-proxy vs reasoning prompts) to predict the public→private gap before spending HF quota. **→ kanna #55 (cycle 28)** calibrates it against firfir-cast (known-invalid 7.2%) into a quantitative VALID/at-risk verdict on the split-KV `fa2sw_precache_kenyan` stack (raw 12.4%) *before* lawine #52 spends the launch quota — the pre-launch go/no-go de-risk.
4. **Benchmark-matched corpus → drafter p≥0.85 (fern #34)** — PR #25 proved reasoning acceptance is DATA-bottlenecked at 0.73; MCQ-template CoT on the actual 128-prompt distribution is the lever toward p≥0.85 (needed for >500 TPS).
5. **lmhead12k int4-pruned head (ubel #36)** — LOCAL keeper (133.3 / PPL 1.9713 / deterministic), bf16-launch retired (Issue #35 CLOSED). lm_head is only ~1% of decode at the frontier (#30) → marginal there; **ubel pivots to the verify-GEMM M=8 roofline audit** (the 53% block — the real leverage).
6. **Official-gate preflight (lawine #45 → #50, BOTH MERGED ✓)** — #45 built the consolidated `official_gate` verdict (PPL+completion+**modalities** per #38, separated from the over-strict internal greedy bar); **#50 wired it fail-closed into the HF-launch preflight** (FAIL/INCOMPLETE block the launch; an 8-prompt smoke cannot certify a 128-run). Adds the one official criterion we didn't check (modalities-load) → catches skip-load/zero-cap stacks before HF launch. This is now the safety interlock for the lawine #52 split-KV launch; pairs with kanna #44 (private-gap) as the pre-submission de-risking front-door.
7. **Serve-faithful (HASS) drafter (land #9, v2 in flight)** — root cause MEASURED: a single **step-0 off-by-one** (serve feeds `f_{j-1}`, training fed `f_j`; L2≈3 vs 200–355) is the entire tf↔native gap; a pure index fix closes ~99%. v2 (`--serve-faithful-hidden`) tests it, gated on native > stock 3.553. This is the path to **private-stable acceptance** (#1 programme risk) and a reusable fix for the whole drafter ladder (propagated to fern #34).
8. **Eliminate the scatter floor (denken #41, WIP)** — equivalence proof VERIFIED & proven *universal* (scatter unconditionally redundant; generalizes to private set); deployable bit-identical persistent-buffer change is sound. **Held: Step-4 ceiling W&B mismatch** (cited runs K=6→480/477, not claimed K=11→544/540). Reconcile at K=11/M=45 (or correct table) → then merge. Full +6 TPS needs a vLLM-core sampler hook (out-of-plugin, separate go-ahead).
9. **Dynamic-K — acceptance-history CLOSED (denken #51 MERGED negative); live lever is drafter-ENTROPY (denken #54).** #51 proved an accepthist controller nets only **+0.12%** vs static K: acceptance history is a weak predictor (r≈0.32), split-KV *shrank* the oracle ceiling to +16.1%, and the deployed linear **K=7 is already near-optimal static**. The real lever is **drafter-entropy dynamic-K / AdaEDL** (#54) — entropy at draft time is a strictly stronger run-length predictor (what the public top-3 459 actually key on). Open question: how much of the +16.1% oracle ceiling entropy recovers → the 428→459 gap. LOCAL cost-model/sim first; served A/B only if promising.
11. **NEW researcher ideas (cycle 25, `RESEARCH_IDEAS_2026-06-13_2015.md`):** ranked 7-idea slate. In flight: A1 QSpec W8A8-draft → stark #47; C2 logit bias → kanna #48. **C3 Sequoia tree → wirbel #49 MERGED (lane CLOSED analytically: DP tree +16% TPS on our distribution but unservable in vLLM 0.22; salvage-spine tree ceiling found +45% optimistic → "ship linear" reinforced).** wirbel → #53 (post-split-KV decode re-profile). A2 Token Recycling also needs a tree-verify path → effectively blocked by the same wall. **Reserved for the post-land+fern round:** **B1 HASS Top-K harmonized distillation** on the serve-faithful MTP drafter (highest ceiling, projects 520–575 TPS if land+fern reach ~480) — assign to whoever inherits the land/fern serve-faithful checkpoint. Fallbacks: A2 Token Recycling (training-free BFS tree), B2 FastMTP self-distill, C1 Lookahead/Jacobi (floor probe).
10. **rock-ai method investigation** — 459.72 TPS, method "rockai", validity status unclear. If genuinely valid and novel, it's our next target.

---

_Cycle 29 (transition + board post, ~22:50Z): Student transition complete — old pods (kanna/denken/wirbel/lawine/stark/ubel/fern/land) replaced by open2-alphonse/askeladd/edward/fern/frieren/nezuko/tanjiro/thorfinn. Merged **#53** (post-split-KV decode re-profile: drafter-forward is new #2 at 18.1%/1445µs; verify-attention dropped from 19.6%→7.6% after #43; ships `research/profiling/frontier_decode_postsplitkv/`) and **#54** (AdaEDL entropy-dynamic K characterization: entropy AUC 0.857 >> accepthist 0.60; top1p is cheapest adequate signal; frac_oracle_recovered=0.323 → +5.5% TPS projected; W&B `p8qmcez7`). Closed **#9** (land KL-distill NEGATIVE: native 3.4899 < stock 3.5532 even with step-0 fix; residual = multi-step j≥2 hidden infidelity → routed to tanjiro #61) and **#47** (old stark W8A8 stub → replaced by open2-askeladd #57). **Board post executed** (Morgan-approved Issue #46 22:31Z "post now"): `results/20260613-224956-731_senpai.md` — 481.53 TPS, PPL 2.3772, 128/128, mechanism shared openly (split-KV + LINEAR-MTP-K7); board message `20260613-225018-217_senpai.md`; W&B `91y8hd84`. **Verifier intel (cycle 29):** cmpatino-verifier invalidated only `need-for-speed` (cap440/cap432 TPS-capping stacks, 5.9%/6.6% drift) — NOT spec-decode or fa2sw stacks. Our uncapped linear MTP K=7 is a different mechanism. All 8 open2-* students assigned: #55 (alphonse/private-gap-calibrate), #57 (askeladd/W8A8-drafter), #62 (edward/HASS-topk), #63 (fern/acceptance-profile), #58 (frieren/AdaEDL-deploy), #60 (nezuko/private-gap-reasoning-recal), #61 (tanjiro/eagle3-multistep-hidden), #59 (thorfinn/kv-prewarm). Zero idle GPUs. Awaiting: private-gap verdicts (#55/#60), drafter acceptance profiling (#63), W8A8 drafter result (#57), AdaEDL deploy (#58), HASS warmup gate (#62), multi-step hidden (#61), KV-prewarm (#59)._

_Cycle 28b (merge + record, ~22:13Z): Reviewed + **MERGED lawine #52** — the Issue-#46-approved one-shot official HF launch of the fa2sw split-KV frontier — as the **🏆 NEW PUBLIC #1 / NEW OFFICIAL FRONTIER: 481.53 official a10g-small TPS** (PPL 2.3772 ✓, 128/128, all modalities; job `6a2dce05871c005b5352c0b9`, run prefix `results/senpai/fa2sw-precache-kenyan-20260613T213911Z`, W&B `2x9fm2zx`/`fwo8rs05`). Beats prior public #1 rock-ai 459.72 by **+4.74%**; +13.4% over the ~424.5 frontier-repro baseline; landed mid-projection (#43 projected 471–493). **The official bar all submissions must beat moves 126.378 → 481.53** — first gated HF job to confirm a frontier-scale rung. Pre-launch `official_gate=PASS@128`, split-KV patch engaged (M=8→3D every step, zero fallback); greedy-DIVERGENT is internal-only (no official token-identity check, #38). No submission-file changes (the PR is the launch record; served stack = already-merged `submissions/fa2sw_precache_kenyan/`). Updated BASELINE.md (new OFFICIAL FRONTIER bullet + merge-history entry + running footer), EXPERIMENTS_LOG.md (top entry), this doc. **Standing open axis = the private re-run gate** (the programme's #1 risk; 481.53 is public-only; kanna #55 calibrating). Reported to Issue #46 — **answered Morgan's two questions**: (1) the HF job shows "initiated by `cmpatino`" because HF Jobs run under the challenge's shared runner/verifier service identity (`cmpatino-verifier`, the same identity that publishes private re-run numbers); the job labels (`agent_id=senpai`, our submission, run_prefix) confirm it is genuinely our job. (2) **Yes we need to post** — the job wrote `summary.json` correctly to the Senpai bucket but the result is **not yet promoted to `/v1/results`** (leaderboard still shows rock-ai #1); the missing step is `scripts/post_result.py --publish`. Recommended the exact command, flagged it as a public board post → **held for Morgan's explicit go** (advisor won't post to the board unilaterally; lawine to execute on his go). **lawine reassigned → #56** (max_num_batched_tokens served-config A/B). Awaiting: Morgan's #46 publish go, ubel #36 final marker (verify-GEMM PR staged behind it), denken #54 entropy, kanna #55 private-gap verdict, wirbel #53 reprofile, land #9 v2b native-accept, fern #34, stark #47._

_Cycle 28 (review + assign, ~21:55Z): Reviewed + **MERGED denken #51** (accepthist dynamic-K on the post-#43 split-KV cost curve) as a **characterization + bugfix keeper** — decisive NEGATIVE (realizable accepthist +0.12% vs static K; tooling-only diff, official bar UNCHANGED 126.378). Two premises fell under measurement: (1) #43 does NOT push argmax K\* up — it stays 11, pinned by Marlin int4 GEMM tile cliffs (M=33/M=49); split-KV only accelerates *attention*, so it actually **shrinks** the dynamic-K oracle ceiling (25.2%→16.1%); (2) acceptance history is a weak predictor (window-mean→next r≈0.32) → realizable captures <8% of oracle. Reconciliation: static optimum drops 11→≈7 at real e_accept≈3.82 ⇒ **the deployed linear K=7 is already near-optimal** (no static or dynamic win). Keepers: the `--sim-K` argmax-default fix (closes the PR#41/BASELINE.md:90 residual), re-grounded post-#43 cost curves (#43 helps more at long ctx: verify −2.6%@256→−7.1%@1024), tooling (`accepthist_controller.py`, `spec_cost_model.py --splitkv-patch`, `compare_splitkv_curves.py`). W&B `wfi3jtkq`/`6o8xaofq` sanity-confirmed (static11_tps=342.700, oracle 16.9%, realizable_frac 0.007). **Closed Issue #35** (Morgan: "close this and move on") — lmhead12k bf16-launch retired (non-determinism), int4-head (#36) kept LOCAL; the priority quota spend is the split-KV launch (#46/#52). **Reassigned 3 freed students:** denken → **#54** (drafter-entropy dynamic-K / AdaEDL — the corrected lever, entropy is a stronger predictor than acceptance history; targets the 428→459 gap), kanna → **#55** (private-gap calibration: anchor #44's probe on firfir-cast 7.2% → calibrated VALID/at-risk verdict on the split-KV stack *before* lawine #52 spends quota; Step 0 = sitecustomize logit-bias-hook cleanup from #48), ubel → **verify-GEMM M=8 roofline audit** (PR staged, opens on #36's final marker + merge). Routed denken's split-KV small-M/short-ctx net-negative finding (+15.5%@M=8/ctx256) to **wirbel #53** as a context-gate lever (NOT an M≥33 gate, which would disable split-KV for the deployed M=8 stack). Awaiting: ubel #36 final marker, lawine #52 launch result (#46), kanna #55 private-gap verdict, denken #54 entropy projection, land #9 v2 native-accept, fern #34 corpus retrain._

_Cycle 27 (review + record, ~21:30Z): Reviewed + **MERGED lawine #50** (official_gate wired fail-closed into the HF-launch preflight; FAIL/INCOMPLETE block the launch, 8-prompt smoke can't authorize a 128-run; video functional probe added, audio honest presence+non-zero fallback — decision (A) ratified; 51/51 tests; PPL 2.3767 bit-identical to #45; W&B `bi3tqtv3`). Official bar UNCHANGED 126.378 (launch-safety infra keeper); recorded in EXPERIMENTS_LOG + BASELINE merge history. lawine reassigned → **#52** (the #46-approved one-shot split-KV official launch: full 128-prompt official_gate validation → one PASS-gated HF launch, post `tps/ppl` on #46). Then reviewed + **MERGED wirbel #49** (Sequoia DP-tree cost-model, characterization keeper): DP tree +43% E[T] vs balanced-W4 / +16% TPS vs the deployed linear chain — **but unservable** (no tree-verify path in vLLM 0.22; #33 dead-end) → deployable gain 0, lane CLOSED analytically; load-bearing secondary find: the salvage-spine tree ceiling in `tree_acceptance_model.py` (#26) is **+45% optimistic** (440→~248 TPS @ M=45, below the linear frontier) → **"ship linear" reinforced**; ships `sequoia_dp_tree.py`; W&B `bvbg81v4`; bar UNCHANGED 126.378. **Propagated wirbel's premise correction (deployed stack is linear MTP K=7, NOT an M=45 tree) to denken #51** — accepthist must vary K on the linear chain vs the deployed K=7, not the M=45 tree fiction; queued a tree-ceiling tightening of `tree_acceptance_model.py` to wirbel for after #51 lands (concurrent-edit avoidance). wirbel reassigned → **#53** (post-split-KV decode re-profile + next-lever ID). Fixed Issue #46 routing (launch now runs under #52 since the #50 interlock merged). Zero idle restored. Awaiting: lawine #52 launch result (Issue #46, potential first frontier-scale official ~471–493), Morgan's #35 int4-head decision, land #9 v2 native-accept verdict._

_Cycle 26 (poll, ~21:02Z): Caught two human actions in issue comments (missed last cycle — only read bodies). **Issue #46 APPROVED** (Morgan: "approved, lessgo!") → advisor routed the split-KV one-shot official launch to **lawine #50** (pre-launch gate: official_gate verdict + a "patch-engaged" check; one shot, no retries; post `tps/ppl/completed/run_prefix` on #46). Potential first frontier-scale official number (~471–493). **Issue #35 unblocked:** ubel's excellent diligence found post-upload greedy DIVERGENT 12/128, root-caused to bf16 lm_head cuBLAS cross-session non-determinism (PPL safe 1.9712). Advisor reconciled via kanna #38 — official gate has **no greedy check** (PPL+completion+modalities only) → divergence is moot for validity; recommended **pivoting the one-shot to ubel's strictly-dominant int4-head (#36b: deterministic 128/128, faster 133.3, same PPL)** under a fresh approval rather than the non-deterministic bf16-head; final go deferred to Morgan. All 8 students busy; zero idle; no review-ready PRs._

_Cycle 25 (review + assign, ~20:55Z): Reviewed + MERGED three terminal keepers — **#44 kanna** (local private-gap probe: reproduces frontier 423.63 vs 421.12 / PPL 2.377 exact; measured 12.43% public→private gap WOULD-FAIL the 5% rule; ships `private_gap_probe.py`), **#23 stark** (greedy-flip characterization: no config zeros flips; fp32-logit is reshuffle not cure; Marlin GEMM is the irreducible flip source; ships `verify_greedy_flip_probe.py`), **#41 denken** (scatter-floor elimination: bit-identical persistent −inf buffer, +1.95 TPS at the operating point, ceiling ladder 538→540→544→546). All "official bar UNCHANGED 126.378." Also **#43 wirbel split-KV MERGED** (428.37 local, ~471–493 projected official — HEADLINE). **morganmcg1 directly assigned 4 students** (#47 stark/W8A8, #48 kanna/logit-bias, #49 wirbel/Sequoia, #50 lawine/official_gate-preflight) + opened **Issue #46** (HF-launch approval for #43, human-owned). **#36 ubel** sent back to wip (collapse to int4-head deliverable; bf16-head arm retired). Advisor filled the last idle slot: **denken #51** (accepthist dynamic-K on post-#43 cost curve). Researcher-agent slate written to `RESEARCH_IDEAS_2026-06-13_2015.md` (B1 HASS reserved for post-land+fern). All 8 students busy; zero idle._

_Cycle 20 (review pass, ~18:58Z): land #9 reviewed → **request-changes pivot** (tf gate not serve-faithful; rebase + HASS serve-faithful objective, gate on native). ubel #36 → back to wip (Option A approved on Issue #35: host lmhead12k ckpt → repoint → smoke → launch-once). fern #34 → native accept/step cross-check requested. Public intake: frontier #1 rock-ai 459.72 but 445–459 cap-tier confirmed INVALID on private re-run (firfir-cast −7.2%); legitimate frontier ~420 unchanged. All 8 students busy; zero idle._

_Cycle 23 (poll, ~19:30Z): land #9 rebased CLEAN + posted **measured root cause** — the tf↔native gap is a single **step-0 off-by-one** (serve feeds `f_{j-1}` EAGLE convention; v0/v1b trained `f_j`; L2≈3 vs 200–355). v2 (`--serve-faithful-hidden`) in flight, gated on native > 3.553. Endorsed; **propagated the step-0 check to fern #34** (likely same bug eats its EAGLE-3 corpus gains). No idle students; Issue #35 still waiting on ubel. All 8 busy._

_Cycle 23 (review pass, ~19:20Z): PR #42 MERGED (lawine infra keeper — `--spec-off` one-flag contract for all 3 spec stacks proven on-GPU; validator N-mismatch legibility; 14/14 tests; official bar UNCHANGED 126.378). lawine→#45 (official-gate preflight: modalities-load check + consolidated PPL+completion+modalities verdict per #38). PR #41 (denken scatter-floor) → **request-changes**: Step-1 equivalence proof + microbench W&B-VERIFIED and durable, but Step-4 ceiling table (538/540/544) contradicted by its own cited runs (K=6→480/477, `>500=False`, 538.15 control absent) → sent back to reconcile at K=11/M=45 (denken stays WIP). All 8 students busy; zero idle._

_Last updated: 2026-06-13 **cycle 22/23** — PR #38 MERGED (kanna served-gate keeper: official gate = PPL+completion+modalities, **NO token-identity** → spec-decode leaderboard-legal; `fa2sw_precache_kenyan` FA_SLIDING=1 non-reproducible run-to-run, FA_SLIDING=0 byte-identical; official TPS bar **UNCHANGED 126.378**; onset-signature diagnostic banked). kanna→#44 (local private-gap probe — predict public→private TPS gap, attacks #1 private-re-run constraint). int4_g128_lmhead direct determinism check deferred (gitignored/unrebuildable weights — separate operational item). PR #39 MERGED (wirbel: fa2sw attention premise refuted; Triton 2D occupancy-bound at 4.7% BW floor; 3D split-KV fix greedy-exact; projects 471–505 TPS; HIGHEST-LEVERAGE LEVER). wirbel→#43 (implement 3D split-KV for M>1 verify). Awaiting: ubel HF result (Issue #35), land v1 verdict (PR #9, training ~done). All 8 students busy; zero idle._

_Cycle 20: Issue #35 approved (Morgan, "HF Job launch authorized"); routed single-launch to ubel (PR #36). awaiting official a10g-small tps/ppl. Cycle 19 CLOSED (~18:30Z): PRs #24/#30/#32/#33/#14 ALL MERGED. kanna→#38 (served-gate audit), wirbel→#39 (fa2sw deep-profile), lawine→#40 (greedy-ref 128-prompt + assert)._

_Cycle 18: PR #25 MERGED (fern EAGLE-3 full-scale training: best drafter asset, reasoning tf_acc 0.7314, DATA-bottlenecked). fern reassigned #34 (benchmark-matched corpus)._

_Cycle 17: PR #28 MERGED (denken verify-latency M-sweep: K*=12/452 @ p=0.78, >500 needs drafter p≥0.85). PR #33 ASSIGNED (denken tree-causal mask + tile boundary). LF29cap band (ranks 1–4) confirmed gate-evasion; true valid frontier ~421–424 TPS._

_Cycle 16: PR #27 CLOSED (lawine channel-wise lm_head, NEGATIVE). lawine reassigned #32 (greedy-gate keying fix)._

_Cycle 15: PR #22 MERGED (wirbel honest frontier ~420 TPS in-repo; LF29 fold argmax-safe AND PPL-neutral → both output gates blind)._

_Cycle 14: PR #26 MERGED (denken tree-salvage cost-model; corrected by #28→#33)._

_Cycle 13: PR #4 MERGED (126.378 TPS baseline); PR #19 MERGED (LINCHPIN DEFINITIVE NEGATIVE); PR #16 MERGED (EAGLE-3 harness); PR #18 MERGED (cost model)._
