# SENPAI Research State — Fast Gemma Challenge

## ★★★★★★ CYCLE 60 — CURRENT STATE (2026-06-20 ~13:20Z) — BI0 SHIPPED @218; #784 GATE = quality-in-5%-band (byte-identity NOT sacred); ★ LIVE +6.69% CANDIDATE: surgattn-OFF 3D-on-M=1 (224.55 local) → wirbel #791 quality re-gate (the priority lane); #785(surgattn-overhead→#779 answered)+#787(verifier-M=7-graph) CLOSED null; denken pod WEDGED → escalated #790 + MTP-accept reassigned to lawine #792; MARLIN/PRECACHE/FP8/FLASHINFER/LOCUS/drafter-quant/verifier-graph KILLED; all 8 assigned (denken pending pod-restart) — authoritative

### ★ NEW GOVERNING HUMAN DIRECTIVE — Issue #784 (2026-06-20 11:24Z, morganmcg1)
**"Push hard on faster TPS within quality guardrails."** Operating target: optimize TPS aggressively; **maintain quality within 5% of the base model on AIME/MMLU/GPQA**; **byte-identical outputs are NOT sacred for this phase** — a non-byte-identical variant that keeps quality inside the 5% band is **worth testing and escalating**. Cadence: pick 2–4 variants, local speed smoke each, cheap quality evidence, keep winners + discard losers fast, **cap analysis per lane** (don't over-analyze a small hypothesis). Prefer local A10G profiling. **Do NOT launch HF benchmark/submission jobs without explicit human approval** (open an Approval-request issue first). → **I have relaxed every in-flight card's "greedy-token-identical MUST match" hard gate to "quality-in-5%-band"** (PPL ≤ 2.42 + 128/128 + all-4-modalities + AIME/MMLU/GPQA within 5%). ACK posted on #784.

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

#### Private-stable lever decomposition (Cycle 60 — all LOCAL-only, all greedy-identity/quality-gated):
| lever | profiler ceiling | quality risk | assigned | status |
|---|---|---|---|---|
| **fewer-weight-bytes: int8/int4 lm_head GEMV** (1.342 GB/tok, NOT spec-amortized — the one un-amortized read) | **HIGH** | med (argmax flips → 5%-band gate) | ubel #788 | **NEW** (from #781 diagnostic) |
| ngram / prompt-lookup drafter (acceptance, lever b) | med–high | low (5%-band) | land #782 | in-flight |
| MTP draft-acceptance tuning @ fixed K=6 (amortize verify GEMM over more accepts; incl. CENTROID_TOP_K) | **med–high** | low (greedy-safe primary) | lawine #792 | **NEW** (reassigned from wedged denken #783) |
| MTP K-depth sweep {0,2,4,6,8} (draft depth) | med | low | fern #774 | **RUNNING** (GPU self-cleared >4min → green-lit guarded launch 12:05Z) |
| LOOPGRAPH + fused-argmax (drafter dispatch ~0.5%) | low | none | kanna #771 | port in-flight (capture-first gate) |
| **★ surgattn-OFF 3D-on-M=1: +6.69% TPS candidate** (224.55 vs 210.48, accept-neutral, prompt-agnostic — but breaks greedy identity 1.76% tok; PPL blind to it) | **med** | **med–HIGH** (needs full quality panel) | wirbel #791 | **NEW** (#785 answered #779: surgattn load-bearing; re-gate quality-first) |
| **drafter CUDA-graph capture** (6× M=1 proposer passes — eager→capture recovers ~16% GPU-busy launch latency) | **med–high** | none (byte-identical — capture is numerics-neutral) | stark #789 | **IN-FLIGHT** (from #786; #787 relayed: size-1 graph EXISTS → does proposer dispatch or run eager?) |
| ~~Marlin kernel-swap~~ / ~~locus-revert~~ / ~~FlashInfer~~ / ~~fp8-KV~~ / ~~precache~~ / ~~osoi5~~ / ~~drafter-weight-quant~~ / ~~verifier-M=7-graph (already FULL-captured)~~ | — | killed/dead | #781, #776, #779, #777, #775/#778, #772, #786, #787 | CLOSED |

### Key constraints
- **Quality gate (Morgan):** MMLU-Pro ≥ 0.572, GPQA ≥ 0.471, GSM8K ≥ 0.807, AIME ≥ 0.090. Greedy-identity to the bi0 control is the quality proof for numerics-neutral levers.
- **Full bi0 quality panel (reference):** MMLU-Pro 0.644 / GSM8K 0.867 / AIME 10/30 / GPQA-Diamond 0.4970. W&B `kredc30c` (GPQA), `s63tb03x` (TPS).
- **Serving fix:** prometheus guard ALONE. fastapi pin = INCOMPATIBLE with vLLM 0.22.0. Never use the pin.
- **No autonomous HF launch.** Open approval issue; fire only after human approves.
- **Private-stable mandate:** any fire MUST be prompt-agnostic — Δ TPS ≤ 5% private re-run gate. Precache / prompt-replay is permanently OFF (program.md:325).

### Fleet status (2026-06-20 ~13:20Z) — 7 healthy + 1 pending pod-restart (denken)
- **In-flight (healthy GPU):** fern #774 (K-sweep RUNNING, guarded-launch OK 12:17Z, ETA ~2h), kanna #771 (loopgraph port), ubel #788 (lm_head fewer-weight-bytes), land #782 (ngram drafter), stark #789 (drafter CUDA-graph capture — #787 relayed: size-1 graph EXISTS → does the proposer dispatch to it or run eager?), **wirbel #791 (NEW — surgattn-OFF 3D quality re-gate of the +6.69% candidate; the priority lane)**, **lawine #792 (NEW — MTP draft-acceptance @K=6, reassigned from wedged denken)**.
- **★ Pod infra:** the original 2-pod GPU leak (#780, fern+wirbel `20437 MiB @ 0% util` orphaned `EngineCore`) RESOLVED — both self-cleared/reaped, guarded-launched clean, both completed (fern running #774; wirbel #785 done). **NEW: a THIRD pod, denken, is WEDGED — escalated #790** (still on stale serve branch `denken/fire-bi0-surgattn-guarded`, zero student commits on `denken/bi0-mtp-accept`, no pickup 50min post-nudge; likely same `setsid`-detached-engine root cause). Advisor has no kubectl/reap — operator restart needed. Root-cause entrypoint-teardown fix still OPEN on #780.
- **CLOSED (dead-ends confirmed):** #781 (Marlin kernel-swap — no alt kernel sm_86), #776 (locus-revert — 211<218), #779 (FlashInfer — head_dim=512), #786 (drafter-weight-quant — latency-bound + vLLM no-ops the quant key), **#787 (lawine verifier-M=7-graph — already 100% FULL-captured, no headroom)**, #777 (fp8-KV hw-dead), #775/#778 (precache mirage), #772 (osoi5 collapse). **#785 (wirbel surgattn-overhead) ANSWERED #779 (surgattn load-bearing) — NOT dead: spawned the +6.69% re-gate #791.** #783 (denken MTP-accept) closed-no-result → reassigned lawine #792.
- **Next:** ★ **wirbel #791 is the priority** (a measured +6.69%/224.55 candidate — does it hold the 5% quality band? quality-first kill-gate). Watch fern #774 (~2h), lawine #792 / stark #789 / land #782 / kanna #771 / ubel #788 in-flight; denken pending pod-restart (#790 — give fresh card on recovery). **Gate per #784: quality-in-5%-band, NOT byte-identity.** No HF launch without an explicit human approval issue.

---

