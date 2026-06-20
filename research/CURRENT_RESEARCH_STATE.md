# SENPAI Research State — Fast Gemma Challenge

## ★★★★★★ CYCLE 60 — CURRENT STATE (2026-06-20 ~12:15Z) — BI0 SHIPPED @218 + QUALITY PANEL COMPLETE; #784 LOOSENS THE GATE (byte-identity NOT sacred → quality-in-5%-band); MARLIN-KERNEL/PRECACHE/FP8/FLASHINFER/LOCUS-REVERT KILLED; 2-pod GPU leak RESOLVED → fern+wirbel RUNNING; all 8 zero-idle — authoritative (historical cycles 56–59 pruned 12:15Z)

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

#### Private-stable lever decomposition (Cycle 60 — all LOCAL-only, all greedy-identity/quality-gated):
| lever | profiler ceiling | quality risk | assigned | status |
|---|---|---|---|---|
| **fewer-weight-bytes: int8/int4 lm_head GEMV** (1.342 GB/tok, NOT spec-amortized — the one un-amortized read) | **HIGH** | med (argmax flips → 5%-band gate) | ubel #788 | **NEW** (from #781 diagnostic) |
| ngram / prompt-lookup drafter (acceptance, lever b) | med–high | low (5%-band) | land #782 | in-flight |
| MTP draft-acceptance tuning @ fixed K=6 (draft quality) | med | low | denken #783 | in-flight (watch pickup) |
| MTP K-depth sweep {0,2,4,6,8} (draft depth) | med | low | fern #774 | **RUNNING** (GPU self-cleared >4min → green-lit guarded launch 12:05Z) |
| LOOPGRAPH + fused-argmax (drafter dispatch ~0.5%) | low | none | kanna #771 | port in-flight (capture-first gate) |
| surgattn 2D vs 3D attn overhead | low | low (5%-band; 3D now allowed if faster) | wirbel #785 | **RUNNING** (GPU self-cleared ~7min → green-lit 12:15Z; target fwds forced-2D both arms → identity expected, drafter-M=1-only delta) |
| MTP drafter GEMM format probe | low | none | stark #786 | in-flight (co-tenant 19.6 GiB) |
| CUDA graph coverage audit (verify-pass M=7 shape) | low | none | lawine #787 | in-flight |
| ~~Marlin kernel-swap~~ / ~~locus-revert~~ / ~~FlashInfer~~ / ~~fp8-KV~~ / ~~precache~~ / ~~osoi5~~ | — | killed/dead | #781, #776, #779, #777, #775/#778, #772 | CLOSED |

### Key constraints
- **Quality gate (Morgan):** MMLU-Pro ≥ 0.572, GPQA ≥ 0.471, GSM8K ≥ 0.807, AIME ≥ 0.090. Greedy-identity to the bi0 control is the quality proof for numerics-neutral levers.
- **Full bi0 quality panel (reference):** MMLU-Pro 0.644 / GSM8K 0.867 / AIME 10/30 / GPQA-Diamond 0.4970. W&B `kredc30c` (GPQA), `s63tb03x` (TPS).
- **Serving fix:** prometheus guard ALONE. fastapi pin = INCOMPATIBLE with vLLM 0.22.0. Never use the pin.
- **No autonomous HF launch.** Open approval issue; fire only after human approves.
- **Private-stable mandate:** any fire MUST be prompt-agnostic — Δ TPS ≤ 5% private re-run gate. Precache / prompt-replay is permanently OFF (program.md:325).

### Fleet status (2026-06-20 ~12:15Z) — all 8 assigned, zero-idle
- **In-flight (healthy GPU):** kanna #771 (loopgraph port), ubel #788 (lm_head fewer-weight-bytes — NEW from #781 diagnostic; pod healthy), land #782 (ngram), denken #783 (MTP-accept — pod was on old branch 08:27Z, watch for #783 pickup), stark #786 (drafter GEMM format — co-tenant 19.6 GiB, check nvidia-smi first), lawine #787 (CUDA graph audit).
- **★ 2-pod orphaned-context leak RESOLVED (GPU side) — #780:** fern #774 + wirbel #785 both hit the identical `20437 MiB @ 0% util` orphaned-`EngineCore` context (prior serve session, cross-namespace, no in-namespace PID). BOTH GPUs have since self-cleared/been-reaped and held `0 MiB stable >3 min` (fern 4min01s/25 samples 11:55–11:59Z; wirbel ~7min/3 samples 11:54–12:02Z) → I GREEN-LIT both with a guarded-launch protocol (final pre-serve `nvidia-smi` abort-on-resurface; init-OOM → clean up + hold + report, no retry-loop). **Root-cause still OPEN for operator:** student `serve.py` likely leaves `setsid`-detached vLLM engines alive across sessions → recurs fleet-wide; asked for an entrypoint teardown fix on #780.
- **CLOSED (dead-ends confirmed):** #781 (ubel Marlin kernel-swap — no servable alt kernel on sm_86; 58% M=1 BW is spec-amortized red herring), #776 (lawine locus-revert — attention locus necessary-but-insufficient in served CUDA graphs; 211<218), #779 (stark FlashInfer — head_dim=512 permanent), #773 (GPQA panel MERGED), #777 (fp8-KV hw-dead), #775/#778 (precache mirage), #772 (osoi5 collapse).
- **Next:** watch all 8 for results — fern #774 + wirbel #785 now RUNNING (guarded launches 12:05/12:15Z); land #782 / denken #783 / kanna #771 / lawine #787 / stark #786 / ubel #788 in-flight (denken #783 pickup unconfirmed at 12:15Z — W&B liveness check in progress). **Gate per #784: quality-in-5%-band, NOT byte-identity.** Run the full quality panel before proposing any fire; no HF launch without an explicit human approval issue.

---

