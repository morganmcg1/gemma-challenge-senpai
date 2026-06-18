STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["nijokiln","jz3ojbio","ui02lj1g"],"primary_metric":{"name":"int8_aime_maj1_greedy","value":0.4167},"test_metric":{"name":"int8_gpqa_diamond_maj1_greedy","value":0.5657},"verdict":"INT8_PARTIAL","int8_gpqa_greedy":0.5657,"pct_of_bf16_gpqa_int8":1.155,"int8_aime_greedy":0.4167,"pct_of_bf16_aime_int8":0.893}

## Results — the int4→int8→bf16 reasoning ladder is complete

**Verdict: `INT8_PARTIAL`.** Uniform int8 (W8A16 g128, body **and** lm_head) **clears GPQA-Diamond comfortably (115.5% of bf16)** but **misses AIME by a single item (89.3% of bf16, 25/60 vs 26/60 to clear)**. The AIME miss is a knife-edge point estimate (Wilson pct-CI [64%, 116%] spans both bf16 and the 0.42 bar), not a firm separation.

### GPQA-Diamond (n=198) — greedy maj@1, conc1 BI=1 gb6144, M=1 AR (no drafter)

| rung | acc (n) | Wilson CI | %-of-bf16 (0.4899) | 90% bar 0.4409 | %-of-bf16 (0.5404)¹ | bar 0.4864 | trunc | extract_fail | W&B |
|---|---|---|---|---|---|---|---|---|---|
| **int4** | 0.5000 (99/198) | [0.4310, 0.5690] | **102.1%** | ✅ | 92.5% | ✅ | 0% | 0 | `ui02lj1g` |
| **int8** | 0.5657 (112/198) | [0.4960, 0.6328] | **115.5%** | ✅ | 104.7% | ✅ | 0% | 0 | `nijokiln` |
| bf16 | 0.4899 (cited) | — | 100% (denom) | ✅ ref | — | — | — | — | `g3cig1xo` |

Per-rung Δ: int4→int8 **+0.0657**; int8→bf16 **−0.0758** (int8 *overshoots* bf16). Ordering **int8 > int4 > bf16** — **not bit-width-monotone**; all three rungs cluster within 0.076 and **all clear** both denominators' 90% bars.

¹ 0.5404 = the secondary GPQA denominator from your 06:34Z heartbeat. Per your 07:02Z resolution it matches no named run, so 0.4899 (`g3cig1xo`) is primary; 0.5404 is reported only as a robustness check. **GPQA clears under both denominators.**

### AIME (n=60: 2024 + 2025-I + 2025-II) — greedy maj@1, conc1 BI=1 gb6144, M=1 AR

| rung | acc (n) | Wilson CI | %-of-bf16 | pct-CI | 90% bar 0.4200 | trunc | extract_fail | W&B |
|---|---|---|---|---|---|---|---|---|
| **int4** | 0.4000 (24/60, cited) | — | **85.7%** | — | ❌ FAIL | (cited) | — | `dh0tbwpp` |
| **int8** | 0.4167 (25/60) | [0.3006, 0.5427] | **89.3%** | [64.4%, 116.3%] | ❌ FAIL by 1 item | 13.3% (8/60) | 0 | `jz3ojbio` |
| bf16 | 0.4667 (28/60, cited) | — | 100% (denom) | — | ✅ ref | — | — | `zoszxnb0` |

Per-rung Δ: int4→int8 **+0.0167 (+1 item)**; int8→bf16 **+0.0500 (+3 items)**. AIME **IS bit-width-monotone** (int4 < int8 < bf16). **int8 closes only 25% of the int4→bf16 gap** — doubling the bits (4→8) buys exactly one item. Clears at 26/60 = 0.4333 = 92.9% of bf16.

AIME per-year (int8): 2024 **15/30 (0.500)**, 2025-I **5/15 (0.333)**, 2025-II **5/15 (0.333)** — the new-set degradation, not the old set. The 8 truncated items (all hit the 6144 cap, finish_reason=length) are **all-wrong** reasoning-runaways; under the fixed gb6144 budget shared with the cited bf16/int4 endpoints this is apples-to-apples, but it means int8 loses ~13% of AIME to non-termination.

### Ladder reading — does spending bits recover the deficit?

- **GPQA-D is not the binding bar and is not bit-sensitive.** int4 (0.5000) and int8 (0.5657) both *exceed* bf16 (0.4899); the rungs don't order by bit-width. This is direct n=198 evidence that GPQA-Diamond greedy is **noise / model-limited**, not a precision-limited bar — consistent with the prior GPQA-knife-edge reads.
- **AIME is the binding bar and is bit-monotone, but recovery is weak.** int8 recovers only **¼ of the int4→bf16 gap** for 2× the bits. Extrapolating, the recovery knee sits **near bf16, not at int8** — which leans toward **most of the AIME reasoning loss being intrinsic to sub-bf16 precision**, only weakly bit-recoverable. int8 is not a free Reading-A-passing body on AIME; you'd need (near-)bf16 to clear, or a different lever (calibration / mixed-precision on the AIME-collapse layers).

### Honest cross-finding (flagging, not banking)

My **in-harness int4-GPQA greedy = 0.5000 = 102.1% of bf16** does **not reproduce the "~86%" int4-GPQA deficit** the hypothesis attributes to int4 (cited as kanna #643, which is outside my launch-isolation scope so I measured the endpoint myself). The likely cause is instrument difference: the ~86% figure appears to be a **sampled / Option-B-spec** number against a different denominator, whereas my ladder is **greedy M=1 AR, conc1, BI=1, byte-identical banked prompts**. In a self-consistent greedy ladder, **int4-GPQA clears** — so the GPQA half of the "int4 loses graduate reasoning" premise does not hold under greedy decode here. AIME is where the real int4 deficit lives (0.4000 = 85.7%).

### Apples-to-apples provenance

- **Bit-width-only ladder.** int8 = the bf16 base `google/gemma-4-E4B-it@fee6332c` quantized to W8A16 g128 (body + lm_head, 343 modules, body rel_err mean 0.0074 / max 0.0121); the int8 tensor set is **identical to the int4 skeleton (2765 tensors)** — only `num_bits` differs 4↔8. int4 = the live-rung `int4_g128_lmhead`.
- **Concurrency.** bf16 endpoints were measured at conc16; my int4/int8 cells at conc1. This is still apples-to-apples because **bf16 under `VLLM_BATCH_INVARIANT=1` is batch-invariant** (conc16 greedy argmax == conc1), whereas int4/int8 Marlin GEMM is *not* BI-covered, so quant cells **must** be pinned to conc1. (M=1 AR, no drafter, isolates the body's quality ceiling.)
- **Same instrument.** All measured cells reuse denken #637's byte-identical banked `prompt_token_ids` (same `prompt_sha256`) and `evalsets.score_item` verbatim. Clean reads: GPQA cells `extract_fail=0, trunc=0`; int8 AIME `extract_fail=0, trunc=13.3%`.

### Exact commands

```bash
cd target/research/validity/int8_bf16_reasoning_ladder
# 1. Build the int8 body (bit-width-only clone of the int4 g128 recipe):
python build_int8.py            # -> /workspace/gemma_build/int8_g128_lmhead (12.69 GB, W8A16 g128 body+lmhead)
# 2. Eval each (body, bar) cell — conc1 BI=1 gb6144 greedy, M=1 AR, idempotent 82-min windows:
VLLM_BATCH_INVARIANT=1 python eval_ladder.py --body int8 --evals gpqa,aime --mode full --soft-cap-min 82
VLLM_BATCH_INVARIANT=1 python eval_ladder.py --body int4 --evals gpqa      --mode full --soft-cap-min 82
# 3. Relog local summaries to W&B group int8-bf16-reasoning-ladder-fern (system python; eval venv lacks wandb):
WANDB_DIR=$PWD/results /usr/bin/python3 relog_wandb.py
```

### Run facts

- **Peak VRAM:** int8 serve ~18.9 GB, int4 serve **19.1 GB** (23 GiB A10G; int8 KV ~8.5 GiB free, 309,750 tokens).
- **W&B group `int8-bf16-reasoning-ladder-fern`** (3 measured cells): int8-GPQA [`nijokiln`], int8-AIME [`jz3ojbio`], int4-GPQA [`ui02lj1g`]. Cited endpoints: bf16 [`g3cig1xo`/`zoszxnb0`], int4-AIME [`dh0tbwpp`]. Every cell `analysis_only=true`, `official_tps=0`. **No HF Job, no submission, served `int4_g128_lmhead` @126.378 untouched.**
- **Decode speed (conc1 greedy):** int4 ~19.4 s/item, int8 ~28 s/item.

### What happened

The ladder cleanly separates the two failing bars. **GPQA-Diamond never had a real precision deficit under greedy decode** — int4, int8, and bf16 all land within noise of each other (0.50/0.57/0.49) and all clear, so spending bits there is moot. **AIME is the genuine reasoning bar**, it *is* bit-monotone, but int8 recovers only a quarter of the int4→bf16 gap — so a less-aggressive uniform quant (int8) is **not** a Reading-A-passing body: it still misses AIME, and the trend says the deficit is mostly intrinsic to sub-bf16 rather than cheaply bought back with bits. Net for the #481 gate-semantics call: **`INT8_PARTIAL`** — GPQA is decided (cleared by every rung, model-limited), AIME is the lone binding bar and uniform int8 doesn't clear it.

### Suggested follow-ups

1. **bf16-AIME is the knee, not int8.** If a Reading-A AIME pass is required, the cheapest precision lever is **mixed int4/int8 targeted at the AIME-collapse layers** (denken's collapse-locus probe) rather than uniform int8 — uniform int8 already showed the per-bit return is poor (¼ gap for 2× bits).
2. **AIME non-termination.** 13.3% of int8 AIME is lost to 6144-token runaways. A quick orthogonal check: re-score the 8 truncated items at a longer budget to see whether the int8 "miss by 1" is a reasoning deficit or a budget-cap artifact (would not change the apples-to-apples verdict, but tells us whether the knife-edge is recoverable by budget alone).
3. **Reconcile the int4-GPQA discrepancy** with kanna #643's ~86% when both are back in scope — almost certainly greedy-vs-sampled/Option-B, but worth a one-line confirmation so the panel's GPQA cell is unambiguous.
