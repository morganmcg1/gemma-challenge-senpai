STUDENT denken:
SENPAI-RESULT: {"terminal":false,"status":"in_progress","pending_arms":true,"analysis_only":1,"official_tps":0,"no_hf_job":1,"fires":0,"wandb_run_ids":["dnik8s1d"],"primary_metric":{"name":"g32_assurance_P(clear_rel_bar)@base0.4667","value":0.5633},"test_metric":{"name":"base_bf16_greedy_smoke_acc","value":0.50}}

## Results — AIME gate-bar pin + ship-risk assurance

**Headline (premise-contradicting, needs your call):** the engine premise in your 09:19 revision does **not** hold for the path I tested. The surviving on-pod engine (`.venv` = **stock vLLM 0.22.0**) serves **bf16 base greedy AIME *faithfully*** — coherent step-by-step math, correct boxed answers — **not** the `0.1333` repetition-gibberish corruption kanna #699 reported. **The bar-pin may be UN-blocked.** I'm surfacing before spending the full n=60 sweep because (a) it overturns "no faithful engine," (b) it overlaps kanna #725's lane, and (c) you explicitly said "wait for kanna/stark before trusting any pinned number" — that instruction should be re-evaluated against this evidence.

I delivered all engine-INDEPENDENT parts you asked for, plus first-hand engine evidence.

### 1. Engine state — FAITHFUL on the surviving `.venv` (first-hand)

Your revision said: *"that engine CORRUPTS greedy AIME for both bodies even at cc=1/BI=1 (base 0.4667→0.1333) … there is currently NO clean greedy-AIME engine on the pod."* I verified the pod myself:

- `.venvs/vllm022` (the env kanna #699 flagged) and `/tmp/vllm0220-srv` are **both absent** on my pod. The **only** surviving vLLM env is `.venv` = clean **stock vLLM 0.22.0** (`transformers 5.9.0`, `torch 2.11.0+cu130`, CUDA OK with `CUDA_VISIBLE_DEVICES=0`).
- I served `submissions/bf16_base_aime` on that `.venv` (TRITON_ATTN backend, `VLLM_USE_FLASHINFER_SAMPLER=0` native sampler, full multimodal tower loaded) and ran **greedy `--temperature 0 --client-concurrency 1 --no-thinking --seed 1234`**.

**Smoke (n=4, gb2048):** maj@1 = **0.50 (2/4)**, `extract_fail_rate=0.0`. The completions are **flawless mathematical reasoning**, e.g. 2024-II-4 → full log-system derivation ending `"…m+n = 25+8 = 33. The final answer is 33."` (`stop`, ✓gold 33); 2024-I-4 → clean probability solution ending `"…m+n = 116."` (`stop`, ✓gold 116). **Both misses were pure `length` truncation at 2048** (sound reasoning cut off mid-computation), not gibberish — i.e. exactly the `max_tokens`-sensitivity this card is about.

**Probe (n=20, gb6144 — 7/20 harvested, run completing in background):** all 7 completions coherent; **4/7 correct (33, 23, 116, 809)**. Decisively, the **two problems that `length`-truncated to *wrong* answers at gb2048 (II-12, I-3) BOTH flip to *correct* at gb6144** (3→23 ✓, 5→809 ✓) — a direct, first-hand demonstration of the budget→truncation→accuracy mechanism this whole card rests on. (This partial is NOT the pinned denominator — n=7 is far too noisy; the real pin needs the full n=60 sweep, path A below.)

This is the OPPOSITE of `0.1333`. Cleanest reconciliation with kanna #699: the **corrupting** `.venvs/vllm022` is **gone**; the **surviving** `.venv` (stock 0.22.0) is **faithful** for the bf16 base path. (I have NOT tested the int4/int8 quant bodies — kanna's "int4→repetition gibberish" may still hold for the quant kernels, which are the fragile path; isolation bars me from cross-reading `lzbqp28p` to diagnose further.)

### 2. Bayesian-assurance machinery (the #719 CI-rescue, realized) — DONE

`research/validity/aime_gate_bar_shiprisk/bayes_assurance.py` — parameterized, engine-independent, self-contained. W&B run **`dnik8s1d`** (group `aime-gate-bar-pin-denken`; guard flags in `wandb.summary` under `guard/*`, linchpin under `linchpin/*`). For each config (k of n) it reports three lenses side by side: point margin, exact-binomial **Clopper-Pearson one-sided 95% LCB**, and **Bayesian assurance** `P(acc_draw ≥ 0.9·base_draw)` by MC (400k draws) over BOTH Beta posteriors (Jeffreys **and** uniform), propagating the bar's own n=60 uncertainty. This is the relative two-sample design that sidesteps the #719 n=1040 absolute-certification wall.

**Decision table (config k PROVISIONAL — see §3):**

| config | base | acc | bar=0.9·base | margin | CP-LCB95 | assur(Jeff) | assur(unif) |
|---|---|---|---|---|---|---|---|
| int4-body | 0.4667 (28/60) | 0.350 | 0.420 | −0.070 | 0.248 | 0.206 | 0.213 |
| **full-g32** | 0.4667 (28/60) | 0.433 | 0.420 | **+0.013** | 0.324 | **0.563** | 0.566 |
| int8-locus | 0.4667 (28/60) | 0.450 | 0.420 | +0.030 | 0.340 | 0.636 | 0.640 |
| int4-body | 0.50 (30/60) | 0.350 | 0.450 | −0.100 | 0.248 | 0.122 | 0.126 |
| **full-g32** | 0.50 (30/60) | 0.433 | 0.450 | **−0.017** | 0.324 | **0.425** | 0.431 |
| int8-locus | 0.50 (30/60) | 0.450 | 0.450 | +0.000 | 0.340 | 0.502 | 0.507 |

**What the assurance reveals that the point margin hides:** even where g32 "clears" on points (base 0.4667, margin +0.013), the posterior assurance is only **0.563** — a near-coin-flip — and the CP-LCB (0.324) sits far below the 0.420 bar. **At n=60, NO recovery config achieves decisive (≥0.8) assurance of clearing the relative gate at EITHER candidate base.** The point "passes" are statistically fragile because the bar itself is n=60-noisy. This is the #719 underpowered-instrument story, now quantified for the *relative* gate.

**Linchpin (robust to the n mapping — depends only on config acc):**
- **g32 fails the gate when base_acc > 0.4815** (knife-edge: base `k_b ≈ 28.9/60`, i.e. base **28/60 → g32 clears, 29/60 → g32 fails**).
- **int8 fails when base_acc > 0.50.**
- **int8-sole-lane window: base_acc ∈ (0.4815, 0.50].** Only in that narrow band is the int8 mandate-break the *sole* clearing recovery lane. Below it, g32 clears → int8 moot. Above 0.50, even int8 fails.

So the verdict **MOVES vs HOLDS** on whether the pinned base lands ≤0.4815 or >0.4815 — a one-problem swing in the n=60 base denominator.

### 3. Provenance audit (in-repo committed artifacts) — DONE

I audited every AIME result JSON committed to this branch under `research/downstream_quality_aime/`. **None of them is the bf16-vanilla base, full-g32, or int8-locus.** They are all old (2026-06-16, n=30/2024-only, **gb3072**, `int4_base_aime` + the surgical-357 ship), and all **truncation-crippled**: `base_greedy_aime.json` (int4) = 0.2667 with **18/30** `length`; `base_aime.json` (int4, sampled) = 0.40 with **26/30** `length`; the surgical-357 ship = **0.033** greedy. → The card's decisive scores (**g32 0.438**, **int8 0.450**) and base candidates (**0.4667**, **0.50**) have **no committed artifact**; their only provenance is W&B runs (fern #713 / fern #659 `nmjvtfov`) that **isolation bars me from auditing**, so I **cannot** confirm they reconciled base→0.4667 on a faithful engine. The config-k in §2 is therefore `round(acc·60)` and flagged PROVISIONAL; the assurance is k-sensitive (the linchpin thresholds are not).

### 4. Decision for you

The engine-independent deliverables are complete. The remaining CORE (pin the base denominator at gb6144/gb8192 with truncation logging, re-score g32) is the part you blocked on a faithful engine — and I have first-hand evidence a faithful engine **already exists** on the pod (`.venv`). **Two paths:**

- **(A) Authorize me to run the full pin now** on `.venv`: base bf16 + full-g32 greedy AIME on the **n=60** gate set (`--years 2024,2025`) at **gb6144 and gb8192**, truncation-logged, then drop the real k into the machinery for a decision-grade verdict. Wall-clock: ~50–55 min per budget at cc=1, so this needs its own run window (exceeds what's left in this one).
- **(B) Defer to kanna #725** if you want the pin on kanna's independently-reconstructed engine for cross-validation — but my evidence suggests the reconstruction may be unnecessary for the bf16 base.

I recommend **(A)**, optionally cross-checked against (B). Either way, the machinery is ready to convert the pinned number into the MOVE/HOLD verdict instantly.

### Guard flags / reproducibility
`analysis_only=1, official_tps=0, no_hf_job=1, fires=0`. No HF Job, no `train.py --launch`, no submission, locked 126.378 untouched. W&B: `dnik8s1d` (machinery). Smoke/probe JSON + driver logs under `research/validity/aime_gate_bar_shiprisk/`.

Smoke command (engine faithfulness):
```
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/downstream_quality_aime/aime_eval.py \
  --submission submissions/bf16_base_aime --server-python /workspace/senpai/target/.venv/bin/python \
  --serve-env MAX_MODEL_LEN=7168 --years 2024 --limit 20 --k 1 --temperature 0 --no-thinking \
  --max-tokens 6144 --client-concurrency 1 --seed 1234 --max-num-seqs 4 --save-text
```

**What happened:** the card's premise (engine corrupt → bar unmeasurable) is contradicted by first-hand evidence; the bar-pin appears un-blocked on the surviving `.venv`. I built the full relative-gate assurance instrument and showed the n=60 instrument is too weak to decisively clear ANY recovery config at either candidate base — the decision rests on a one-problem swing in the base denominator. **Suggested follow-up:** your call on path (A)/(B); if (A), I'll run the two-budget n=60 pin next window and finalize the MOVE/HOLD verdict.
