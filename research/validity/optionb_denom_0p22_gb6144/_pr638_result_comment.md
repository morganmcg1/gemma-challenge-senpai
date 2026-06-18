STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["u13z29hs","3hx5xqbo","9hxb88f5","pvmp7owq","uq5eqd1a","2epukh70"],"primary_metric":{"name":"gpqa_diamond_sampled_int4ar_0p22","value":0.498989898989899},"test_metric":{"name":"aime_int4ar_gb6144","value":0.35}}

## Results — Option-B denominator leg 3: int4-AR (live-rung body) panel @ gb6144

(Heartbeat for the advisor sign-of-life ping: **panel is DONE**, finished `2026-06-18T05:34:05Z`. gb6144 config + `min_tokens=8` EOS-guard confirmed in place. All 6 W&B runs are now in group `int4ar-livrung-denominator-0p22` — they posted late because the harness logs only at the end.)

**Server validity (gate for trusting the panel): PASS.** The server loaded the genuine live-rung int4 body, not bf16. Server log: `model=/workspace/gemma_build/int4_g128_lmhead`, `quantization=compressed-tensors`, `Using MarlinLinearKernel for CompressedTensorsWNA16`. Build log: deterministic minmax, **343** body modules quantized (rel_err mean 0.1021) + untied int4 g128 lm_head (`tie_word_embeddings=false`, packed `(262144, 320)`), rebuilt from the TRUE bf16 QAT master (`gemma-4-E4B-it-qat-q4_0-unquantized`) via the submission's own `submissions/int4_g128_lmhead/build_quant.py --group-size 128 --head-group-size 128`. Every engine flag is byte-identical to `serve_bf16_0p22.sh` (vLLM 0.22.0, `VLLM_BATCH_INVARIANT=1`, mml 8192, seqs 16, mnbt 2048, gpu-mem-util 0.90, `VLLM_SEED=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`, prometheus shim, `CUDA_VISIBLE_DEVICES=0`). **The only change vs your #628 is the model.**

### Three-way denominator table (every gate, same engine + same gb6144 budget + same eval code)

| gate | bf16 base (#628) | **int4-AR (this, live-rung body)** | int4+spec Option-B (#629) | int4-AR / bf16 | bar (0.9×bf16) | int4-AR clears? | Option-B clears? | int4-AR `fl` |
|---|---|---|---|---|---|---|---|---|
| MMLU-Pro | 0.7180 | **0.6680** | 0.664 | 0.930 | 0.6462 | ✅ | ✅ | 0.000 |
| GSM8K | 0.9280 | **0.9280** | 0.926 | 1.000 | 0.8352 | ✅ | ✅ | 0.000 |
| **GPQA-D sampled (BINDING)** | 0.5404\* | **0.4990** | 0.4652 | 0.923 | **0.4864** | ✅ (marginal) | ❌ | 0.0005 |
| GPQA-D greedy | 0.4899 | **0.5051** | 0.4444 | 1.031 | 0.4409 | ✅ | ✅ | 0.000 |
| **AIME** | 0.4667 | **0.3500** | 0.3667 | 0.750 | **0.4200** | ❌ | ❌ | 0.1667 |

`fl` = `finish_length_rate`. bf16 #628 `fl` was 0.000 ×4, AIME 0.1333. **Flag (as requested): int4-AR AIME `fl` = 0.1667 (10/60) is materially above bf16 0.1333 (8/60)** — int4 reasoning chains run longer and hit the 6144 cap more, which partly drives the AIME deficit (but even ignoring truncation, 0.350 is well under the 0.420 bar). All non-AIME legs match bf16's ~0 `fl`, so the denominators are clean.

\***Seed-count asymmetry (honest caveat):** the bf16 GPQA-sampled denominator `0.5404` is a **single seed** (n=198, sampling_seed 12345); int4-AR and Option-B are **10-seed n=1980** means. The bar `0.4864 = 0.9×0.5404` is therefore anchored on one bf16 seed that happens to land at the **top** of the int4-AR seed distribution (int4-AR's seed-0 = 0.5404 exactly). So int4-AR clearing the bar with its 10-seed *mean* (0.499) is **conservative** — a 10-seed bf16 mean would very likely be lower, lowering the bar.

**GPQA-D sampled detail (int4-AR):** acc 0.4990, n=1980, 988/1980 correct, Wilson95 **[0.4770, 0.5210]**, per-seed SE 0.0091, per-seed range [0.4495, 0.5404].

### Verdict — SPLIT BY AXIS (pre-registered label undersells the decisive finding)

The aggregator's pre-registered rule keys only on the binding GPQA-sampled axis and emits **`OPTIONB_HAS_SPEC_SPECIFIC_DEFICIT`** (int4-AR clears 0.9×bf16, Option-B does not). That is true on GPQA-sampled — but it **masks the result that actually answers #481:**

- **int4-AR clears 0.9×bf16 on 4/5 gates** (MMLU 0.93×, GSM8K 1.00×, GPQA-D sampled 0.923×, GPQA-D greedy 1.031×). It does **NOT** share the GPQA/MMLU/GSM8K deficit — the int4 body sits at ~bf16 level on those. **This refutes the PR's primary prediction** (int4-AR GPQA-sampled ≈ 0.46–0.475).
- **int4-AR FAILS only AIME: 0.3500 = 75.0% of bf16, well under the 0.420 bar — and *below* Option-B (0.3667).** So on AIME the deficit is the **int4 body itself**, not spec. ⇒ **The literal "≥90% of bf16" gate is ALREADY breached by the exact body the live submission ships — via AIME, regardless of spec.** That is the `INT4_AR_LIVE_RUNG_SHARES_DEFICIT` outcome the card was hunting, just on AIME instead of GPQA.

**Two cautions on the binding GPQA-sampled call** before reading `OPTIONB_HAS_SPEC_SPECIFIC_DEFICIT` literally:
1. **Marginal clearance:** the bar 0.4864 sits *inside* int4-AR's Wilson CI [0.4770, 0.5210]; int4-AR clears by only ~1.3σ of its own 10-seed SE.
2. **Tension with denken #626:** the int4-AR (0.499) vs Option-B (0.465) gap is +0.034 (~2σ). Attributing that to *spec* contradicts #626's spec≈AR graded-immaterial (net ΔGPQA −0.0101, CI ∋ 0). It is at least as likely seed/measurement noise across two separate 10-seed runs as a genuine spec-specific GPQA deficit.

**Net for the #481 gate-interpretation question:** reading "≥90% of bf16" **literally fails the int4 mandate the live submission already ships** (AIME 0.350 < 0.420). So the operative reading should be **"no regression vs the int4-AR config we ship"**, under which Option-B is graded-neutral (#626/#629). The nominal-binding GPQA-sampled axis points the other way (int4-AR clears, Option-B misses) but only marginally and in tension with #626 — it is not a clean stand-alone blocker.

### GPQA-D greedy cross-check vs lawine #627 (0.4444) — budget effect, not an artifact

int4-AR greedy @ gb6144 = **0.5051**; the cited lawine #627 figure (0.4444) was at the submission's **4096** budget → gap +0.0607. This is a **budget effect, not an int4/engine artifact**, confirmed from *my own data*: the **bf16 base at the same gb6144 budget is 0.4899**, i.e. int4-AR greedy ≈ bf16 greedy at matched budget (+0.015, ~3 questions, noise), and **both** sit well above 0.4444. The larger budget lifts greedy GPQA for bf16 and int4 alike (int4-AR `ctok_p95=3756`, `len@2048=0.41` — a real long tail truncated at 4096 but completed at 6144). No int4-specific divergence.

### Reproduce / exact commands
```bash
# (one-time) build the live-rung body locally from the QAT master (CPU-only, deterministic):
bash research/validity/optionb_denom_0p22_gb6144/build_int4ar.sh
# serve int4-AR on vLLM 0.22.0 @ gb6144 (engine flags identical to serve_bf16_0p22.sh):
bash research/validity/optionb_denom_0p22_gb6144/serve_int4ar_0p22.sh
# 5-gate panel (GPQA-sampled 10-seed, AIME, GPQA-greedy, MMLU-Pro, GSM8K), gb6144, min_tokens=8:
LIMIT=0 bash research/validity/optionb_denom_0p22_gb6144/run_panel_int4ar.sh
# three-way table + verdict + W&B:
.venv/bin/python research/validity/optionb_denom_0p22_gb6144/aggregate_int4ar.py --wandb
```

### Peak memory (A10G 24 GB, single GPU)
Model weights **9.88 GiB**; KV cache **8.39 GiB** (306,820 tokens, max concurrency 37.45× @ 8192 tok/req); CUDA-graph pool 0.08 GiB; `gpu-memory-utilization=0.90` (≈19.8 GiB reserved). No OOM; checkpoint 9.62 GiB on disk.

### W&B (group `int4ar-livrung-denominator-0p22`)
- `u13z29hs` ubel/int4ar-gpqa_sampled (BINDING / primary) · `3hx5xqbo` ubel/int4ar-aime (test)
- `9hxb88f5` ubel/int4ar-mmlu_pro · `pvmp7owq` ubel/int4ar-gsm8k · `uq5eqd1a` ubel/int4ar-gpqa_greedy
- `2epukh70` ubel/int4ar-VERDICT (`verdict=OPTIONB_HAS_SPEC_SPECIFIC_DEFICIT`, gpqa_sampled_under_bar=0, aime_under_bar=1)

### What happened — honest analysis
The card cleanly separated **which axis carries the int4 quality deficit**. It is **AIME, not GPQA/MMLU/GSM8K**. The int4 W4A16 body tracks bf16 within noise on every non-AIME gate (and *beats* bf16 on greedy GPQA at this budget), so "the int4 mandate body is uniformly ~82–86% of bf16" is false. The real story: a **localized AIME collapse** (0.350 vs 0.467, partly truncation-amplified) that the int4 body owns outright — Option-B's AIME (0.367) is no worse than the body it runs on. On the binding GPQA-sampled axis the int4 body is essentially at bf16 level and Option-B's miss (0.465) is within ~2σ of int4-AR and in tension with #626, so I would not read it as a hard spec-specific blocker. The decision-relevant takeaway is robust regardless of how you weight the axes: **the literal ≥90%-of-bf16 gate is already failed by the shipped int4 body (AIME), so a literal reading indicts the mandate itself.**

### Suggested follow-ups
1. **Re-measure bf16 GPQA-sampled at 10 seeds (n=1980)** to replace the single-seed 0.5404 denominator with a stable mean — the entire GPQA-sampled gate bar currently rests on one bf16 seed sitting at the top of the int4-AR seed distribution.
2. **AIME at a higher token budget (8192–12288) for bf16 + int4-AR** to split the truncation component (int4 `fl` 0.167 vs bf16 0.133) from genuine reasoning-quality loss, since the int4 AIME deficit is partly truncation-driven.
3. **Option-B int4+spec AIME at 10-seed sampled** to complete a clean 10-seed sampled head-to-head on AIME (mirrors the GPQA-sampled protocol) and confirm AIME is body-level, not spec-level.
