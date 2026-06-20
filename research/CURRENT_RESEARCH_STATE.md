# SENPAI Research State — Fast Gemma Challenge

## ★★★★★★ CYCLE 62 — CURRENT STATE (2026-06-20 ~14:05Z) — BI0 SHIPPED @218; ★ HUMAN TARGET (#784 13:34Z): BREAK 250 TPS in-band; ★★ INT4HEAD MERGED (#788 ubel) — **+17.0% LOCAL (256.74) = BIGGEST SINGLE MEASURED LEVER; PROJECTS PAST 250 STANDALONE** (~255 official if local↔official tracks); FIRE-PREP PANEL (MMLU-Pro/AIME/GPQA) NOW RUNNING → HF APPROVAL ISSUE TO FOLLOW; #784 GATE = quality-in-5%-band (byte-identity NOT sacred); ★ SURGATTN (+6.69%) DECOMP: **wirbel #791** (quality-gate full identity-breaking variant — priority), **land #793** (byte-identical drafter-share), **stark #794** (whole byte-identical at split-KV combine); **kanna #771 SENT BACK** (fused-argmax +1.95% N≥5 confirm); drafter-dispatch/loopgraph/ngram CLOSED; denken pod WEDGED (stays down per #784 ops note — no re-ping) — authoritative

### ★ NEW GOVERNING HUMAN DIRECTIVE — Issue #784 (2026-06-20 11:24Z, morganmcg1)
**"Push hard on faster TPS within quality guardrails."** Operating target: optimize TPS aggressively; **maintain quality within 5% of the base model on AIME/MMLU/GPQA**; **byte-identical outputs are NOT sacred for this phase** — a non-byte-identical variant that keeps quality inside the 5% band is **worth testing and escalating**. Cadence: pick 2–4 variants, local speed smoke each, cheap quality evidence, keep winners + discard losers fast, **cap analysis per lane** (don't over-analyze a small hypothesis). Prefer local A10G profiling. **Do NOT launch HF benchmark/submission jobs without explicit human approval** (open an Approval-request issue first). → **I have relaxed every in-flight card's "greedy-token-identical MUST match" hard gate to "quality-in-5%-band"** (PPL ≤ 2.42 + 128/128 + all-4-modalities + AIME/MMLU/GPQA within 5%). ACK posted on #784.

### ★ HUMAN SUB-DIRECTIVE — #784 (2026-06-20 13:34Z, morganmcg1)
**"Keep pushing — can we push toward or break 250 TPS while keeping our quality gate?"** + two ops notes: **(1) IGNORE PPL-only competitors** (sparkgemma/ultra-gemma ~490–505 TPS are PPL-only-validated — PPL ~2.39, no AIME/MMLU/GPQA; PPL is an imperfect measure → calibration only, NEVER borrow their W192/CTK/noprecache configs); **(2) broken students stay down for a while → reassign to functional students** (denken's wedged pod is NOT getting fixed soon — stop blocking on #790, no more re-pings; keep the 7 functional students fully loaded, which they are). **My answer (posted #784 ~13:45Z):** 250 = +14.7% over 218.02, reachable by STACKING orthogonal quality-safe levers — surgattn-3D (~+6.7%*) × int8 lm_head (~+4–6%) × acceptance K/CENTROID (~+4–6%) [× fused-argmax +1.95%] → projects ~252–266 TPS. *surgattn is the biggest but **least-certain** (n=1, exceeds static ~2.6% attn ceiling) → being noise-certified + quality/identity-resolved by #791/#793/#794; if it de-rates, the path leans on lm_head + acceptance. No silver bullet; the stack. Will open an HF approval issue the moment a stacked config clears the local gate + 4-axis panel.

### ★ COMPETITIVE BAR (public board, cmpatino-verifier 11:18Z)
A competitor (`sparkgemma-s46b`) had **506.63 TPS re-run private at 490.72 (Δ 3.1%), PPL 2.3931 — VERIFIED VALID.** The frontier is being pushed by aggressive, *non*-byte-identical configs that sit inside the PPL≤2.42 bar. Our shipped bi0 = 218.02 → large headroom; the old byte-identity constraint was leaving TPS on the table (validates #784). NOTE: this is the public leaderboard scoreboard, not an in-scope branch — do NOT inspect/borrow sparkgemma's config; use the number only to calibrate urgency.

### What shipped (2026-06-20 09:32Z)
- `int4_mtp_bi0_surgattn` FIRED + OFFICIAL: **218.02 TPS / PPL 2.0058 / 128/128 VALID** (W&B `s63tb03x`, job `6a3656ef3093dba73ce2ac88`). Human-approved #769. +72.5% over int4_g128_lmhead@126.378. Config: int4 W4A16 + MTP K=6 + VLLM_BATCH_INVARIANT=0 + surgattn + prometheus guard.
- **★ bi0 quality panel NOW COMPLETE (all 4 Morgan axes):** MMLU-Pro **0.644** / GSM8K **0.867** / AIME **10/30** / GPQA-Diamond **0.4970** (#773 wirbel, W&B `kredc30c`, pooled 990 samples). All axes pass the ≤10% degradation bar. Reported to Morgan + dhruv-mishra on board `20260620-111347-905_senpai.md`. **The quality panel requirement for any 300+ candidate is: same bar (all 4 axes ≥90% of base).**

### Current research focus: reach a quality-safe 300+ via PRIVATE-STABLE levers, decomposed by the PROFILER

Morgan's question (board 08:19Z): "≤10% degradation vs base for any 300+ submission." **Status:** bi0 quality panel is complete and PASSES. dhruv-mishra's question ("separate validation other than perplexity?") answered on board 11:13Z — YES: full validity contract = PPL + 128/128 + greedy-identity + 4-modalities + cmpatino private Δ≤5%; PLUS our internal quality panel. The real target = PRIVATE-STABLE prompt-agnostic levers that beat 218 TPS with the same quality gates.

**★ PROFILER REALITY (BASELINE.md) drives the decomposition:** decode at conc=1 is **memory-bandwidth-bound — ~92% weight-GEMM, attn ~2.6%, sampling ~0.2%.** Only two lever FAMILIES have real headroom: **(1) the int4 W4A16 GEMM kernel** (numerics-neutral, greedy-identity) and **(2) drafter acceptance** (amortize the one big GEMM over more accepted tokens, greedy-verified).

**★ PERMANENTLY DEAD LEVERS (confirmed):**
- **int4 Marlin W4A16 kernel-SWAP (#781 ubel, CLOSED 11:55Z):** no numerics-equivalent faster W4A16 kernel exists on sm_86 — Marlin is the SOLE servable kernel (Cutlass/Machete need sm_90, AllSpark W8-only, Conch absent, Exllama rejects bf16); its `atomic_add` toggle is hw-gated off (Ampere no bf16 atomicAdd); no tunable tile/thread/split; CLI rejects quant-method mismatch. **★ The 58% M=1 HBM figure is a RED HERRING — deployed serving runs at M=7 (~79% of peak); spec already amortizes the M=1 dequant slack (13.29ms single-stream → 218.49 deployed = 2.9×).** Kernel-schedule lever closed; the body GEMM is at the wall where it runs. W&B `uaq6btet`.
- **FlashInfer (#779 stark, CLOSED):** architecturally incompatible with Gemma4-E4B on sm86 — head_dim=512 global-attn has no valid paged-prefill kernel in FlashInfer; crashes at warmup `NUM_MMA_D_QK=32→head_dim=512`. Not a config issue; permanent. W&B `6i58gvjk`.
- fp8-KV (#777 ubel, CLOSED): three-layer hardware wall on sm_86. Permanent.
- precache (#775/#778, CLOSED): prompt-replay gaming, fails program.md:325 + private-Δ gate. Permanent.
- osoi5-baked (#772 land, CLOSED): body-level quality collapse. Permanent.
- **drafter-weight-quant (#786 stark, CLOSED 12:35Z):** the bf16 MTP drafter is only **17.2% of GPU-busy + latency-bound at M=1** (weight read ~0.17ms/2.434ms, NOT BW-bound) — no "disproportionate bf16 BW share" to recover; AND vLLM 0.22.0 **silently no-ops** `--speculative-config quantization=` for `gemma4_mtp` (no packed weights to attach; proven 3 ways). Dead two ways. The int4 verifier (82.8%) is the budget. W&B group `bi0-drafter-gemm`.
- **ngram/prompt-lookup drafter (#782 land, CLOSED 13:40Z):** 89.91 vs 218.11 local TPS (2.4× slower) — ngram E[T] 1.99–2.17 vs MTP 3.34; suffix-matching misses on general chat, and num_spec 3→5 *lowers* accept_rate (miss-dominated). bi0's learned MTP head is the better acceptance lever; stacked ngram+MTP architecturally unsupported (single-`method` `SpeculativeConfig`). W&B `7m8eyv0f`/`rv1v3lxi`.
- **★ drafter-dispatch / launch-overhead family (#789 stark + #771-loopgraph kanna, CLOSED 13:40Z):** the 6× M=1 proposer passes are ALREADY PIECEWISE-captured (not eager; no size-1 graph realized — `adjust_cudagraph_sizes_for_spec_decode` rounds [1,2,4,8]→{7}); FULL/whole-loop proposer capture is hard-disabled in 0.22.0 (`llm_base_proposer.py:380`, no config knob); AND kanna's whole-loop LOOPGRAPH measured **+0.037% null**. Joint conclusion: the drafter `cpu 5.6 ≫ gpu 2.5` gap is **async-launch-queue depth hidden behind GPU compute, NOT recoverable wall-clock latency** — closed from both the config-capture (stark) and whole-loop-capture (kanna) angles. Verify-GEMM amortization via acceptance (fern #774 / lawine #792) is the live path. W&B `xn7opsy8`/`wjeykst8`.

#### Private-stable lever decomposition (Cycle 62 — all LOCAL-only, all quality-gated per #784):
| lever | profiler ceiling | quality risk | assigned | status |
|---|---|---|---|---|
| **★★ int4 W4A16 g32 lm_head GEMV** (1.342→0.378 GB/tok, 3.56×; per-token 2.777→0.750 ms; MTP does NOT amortize) | **LARGEST — +17.0% LOCAL (256.74 TPS)** | LOW (PPL identical, GSM8K −0.54%) | ubel → **fire-prep panel** | **MERGED #788 — BEST-LOCAL CANDIDATE; full 4-axis panel (MMLU-Pro/AIME/GPQA) RUNNING → HF approval issue to follow** |
| **★ surgattn +6.69% — QUALITY-GATE the full variant** (3D-on-all-M=1, 224.55 local; breaks identity 1.76% tok; PPL blind; n=1 above static attn ceiling) | **med** | **med–HIGH** (full 4-axis panel) | wirbel #791 | **IN-FLIGHT** (priority lane) |
| **★ surgattn +6.69% — DRAFTER-share byte-identical** (3D on 6× drafter proposer M=1 forwards only; proposals can't change emitted tokens → byte-exact by construction) | **med** | **none** (byte-identical target) | land #793 | **IN-FLIGHT** (forward-type attribution; complement of #791/#794) |
| **★ surgattn +6.69% — WHOLE byte-identical at the kernel** (fp32 partial-accum / deterministic reduction order at split-KV combine → quality-free if it holds) | **med–high** | **none** (byte-identical / tie-tolerant target) | stark #794 | **IN-FLIGHT** (kernel-numerics; complement of #791/#793) |
| MTP draft-acceptance tuning @ fixed K=6 (amortize verify GEMM over more accepts; incl. CENTROID_TOP_K) | **med–high** | low (greedy-safe primary) | lawine #792 | **IN-FLIGHT** |
| MTP K-depth sweep {0,2,4,6,8} (draft depth) | med | low | fern #774 | **IN-FLIGHT** |
| fused-sparse-argmax centroid kernel (byte-EXACT +1.95% @ ~1.8σ — loopgraph itself = +0.037% null) | low | none | kanna #771 | **SENT BACK** (fused-only N≥5 confirm, rebase needed) |
| ~~Marlin kernel-swap~~ / ~~locus~~ / ~~FlashInfer~~ / ~~fp8-KV~~ / ~~precache~~ / ~~osoi5~~ / ~~drafter-quant~~ / ~~verifier-M=7-graph~~ / ~~ngram (2.4× slower)~~ / ~~drafter-dispatch capture~~ | — | killed/dead | — | CLOSED |

### Key constraints
- **Quality gate (Morgan):** MMLU-Pro ≥ 0.572, GPQA ≥ 0.471, GSM8K ≥ 0.807, AIME ≥ 0.090. Greedy-identity to the bi0 control is the quality proof for numerics-neutral levers.
- **Full bi0 quality panel (reference):** MMLU-Pro 0.644 / GSM8K 0.867 / AIME 10/30 / GPQA-Diamond 0.4970. W&B `kredc30c` (GPQA), `s63tb03x` (TPS).
- **Serving fix:** prometheus guard ALONE. fastapi pin = INCOMPATIBLE with vLLM 0.22.0. Never use the pin.
- **No autonomous HF launch.** Open approval issue; fire only after human approves.
- **Private-stable mandate:** any fire MUST be prompt-agnostic — Δ TPS ≤ 5% private re-run gate. Precache / prompt-replay is permanently OFF (program.md:325).

### Fleet status (2026-06-20 ~14:05Z) — 7 healthy + 1 down (denken)
- **ubel:** **MERGED #788 (int4 g32 lm_head) → NOW ASSIGNED FIRE-PREP QUALITY PANEL** (MMLU-Pro/AIME/GPQA-Diamond on the merged int4 g32 config). The gating step before the HF approval issue.
- **fern #774:** MTP K-depth sweep, in-flight.
- **lawine #792:** MTP draft-acceptance @K=6 (CENTROID_TOP_K), in-flight.
- **wirbel #791 (PRIORITY):** surgattn full-3D quality re-gate. If passes → surgattn stacks cleanly on int4head. If fails → land #793 byte-identical drafter-share is the salvage.
- **land #793:** surgattn drafter-share byte-identical attribution (forward-type gating).
- **stark #794:** surgattn whole-variant byte-identical at split-KV combine (fp32 accum / reduction-order).
- **kanna #771:** SENT BACK — fused-sparse-argmax N≥5 confirmation (rebase needed, still CONFLICTING).
- **★ The +6.69% surgattn decomposition (3 complementary lanes):** wirbel #791 quality-gates the full identity-breaking variant; land #793 recovers the byte-identical DRAFTER-forward share; stark #794 asks whether the WHOLE +6.69% can be made byte-identical at the split-KV combine. If #794 succeeds → full +6.69% ships quality-free on top of int4head.
- **★ Pod infra:** denken WEDGED (stays down per #784 ops note — NOT re-pinging). Other 4 "worker unhealthy" flags (fern/lawine/kanna/wirbel) = stale-heartbeat false alarms (branches match + dirty>0). Reliable liveness = branch-match + commits, NOT heartbeat age.
- **MERGED this cycle:** **#788** (ubel int4 g32 lm_head — +17.0% local, 256.74 TPS, BEST-LOCAL QUALITY-SAFE CANDIDATE). **CLOSED prior cycles:** #789/#771-lg (drafter-dispatch family), #782 (ngram 2.4× slower), #781 (Marlin kernel), #776 (locus), #779 (FlashInfer), #786 (drafter-quant), #787 (verifier-graph), #777 (fp8-KV), #775/#778 (precache), #772 (osoi5).
- **★ CRITICAL PATH TO 250:** int4 g32 lm_head ALONE projects ~255 official (+17% local). **The single highest-value action is getting the official a10g number via an approved HF job.** Sequence: ubel fire-prep panel → HF approval issue → official A/B. If official confirms ≥250 → DONE (single lever). If official comes back 240–250 → stack surgattn/acceptance on top for the next rung. Either way, fire int4head ASAP. No HF launch without approval issue + human OK.

---

