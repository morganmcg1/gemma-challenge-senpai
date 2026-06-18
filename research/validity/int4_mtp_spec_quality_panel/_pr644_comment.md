STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["mmiyf5ij"],"primary_metric":{"name":"optionb_mmlupro_greedy_pct_of_base","value":0.9136},"test_metric":{"name":"optionb_gsm8k_greedy_pct_of_base","value":0.9978}}

## Results — Reading-A panel: Option-B MMLU-Pro + GSM8K %-of-base

**Stack (Option-B):** vLLM **0.22.0** (manifest engine, *not* dev307) / `int4_g128_lmhead` body + Gemma4-MTP **K=7** drafter (`/tmp/qat-assistant`) / `VLLM_BATCH_INVARIANT=1` / **gb6144** (`max_tokens=6144`, `min_tokens=8` EOS-guard) / served `MAX_NUM_SEQS=16`, eval concurrency 16. **`analysis_only=true`, `official_tps=0`, no HF Job, no submission.** Same live spec server produced all four cells (config verified: `vllm.__version__==0.22.0`, body `/workspace/gemma_build/int4_g128_lmhead`, `speculative-config {model: /tmp/qat-assistant, num_speculative_tokens: 7}`).

### Panel — greedy is the apples-to-apples comparison (ubel #628 base denominators are GREEDY)

| bench | decode | acc (k/n) | Wilson 95% CI | base (#628) | %-of-base | %-of-base CI | floor ≥0.605 | bar90 |
|---|---|---|---|---|---|---|---|---|
| **MMLU-Pro** | greedy | **0.6560** (328/500) | [0.6133, 0.6963] | 0.7180 (`367i9s0t`) | **0.9136** | [0.8542, 0.9698] | PASS | **0.6462** |
| MMLU-Pro | sampled | 0.6740 (337/500) | [0.6317, 0.7136] | 0.7180 | 0.9387 | [0.8798, 0.9939] | PASS | 0.6462 |
| **GSM8K** | greedy | **0.9260** (463/500) | [0.8997, 0.9458] | 0.9280 (`4cxd1gfx`) | **0.9978** | [0.9695, 1.0192] | PASS | **0.8352** |
| GSM8K | sampled | 0.9260 (463/500) | [0.8997, 0.9458] | 0.9280 | 0.9978 | [0.9695, 1.0192] | PASS | 0.8352 |

All four cells: `extract_fail=0.0`, `truncation_rate=0.0` (no length-stops; ctok_p95 ≈ 2.9k ≪ 6144), GSM8K `strict_rate=1.000`. Greedy uses T=0/top_p=1/top_k=−1; sampled uses T=1/top_p=0.95/top_k=64 (`generation_config.json`, lewtun #31), `min_tokens=8` on every read (wirbel #541).

### Reading-A verdicts (vs the binding ≥90%-of-base bars, CI-aware)

- **`READING_A_GSM8K_PASSES`** — clean pass. Both decode modes 0.9260 = **99.8% of base**; pct-of-base CI-lo **96.95%**, far above the 90% bar (0.8352). Unambiguous.
- **`READING_A_MMLU_KNIFE_EDGE`** — both decode-mode **point estimates clear the 90% bar** (greedy 91.4%, sampled 93.9% of base) and both **clear the absolute floor (0.605) with CI margin** (greedy acc CI-lo 0.6133, sampled 0.6317). But at n=500 the Wilson **CI-lo straddles the 0.6462 bar** (greedy pct CI-lo 85.4%, sampled 88.0%), so it is a *point-pass / not-yet-CI-confirmed* — KNIFE_EDGE under the same CI-aware rule the GPQA panel used.

**Net:** GSM8K and MMLU-Pro are the two *non-binding* bars of the four — GSM8K passes outright, MMLU-Pro point-passes the 90% bar (unlike GPQA, which point-*failed* it at 10-seed: 0.4652 < 0.471). The int4 body holds general knowledge (MMLU-Pro ~91–94% of base) and arithmetic (GSM8K ~99.8%) far better than graduate reasoning (GPQA).

### Comparison vs PR baseline

PR baseline = the base denominators only (ubel #628 gb6144 GREEDY: MMLU-Pro 0.7180 `367i9s0t`, GSM8K 0.9280 `4cxd1gfx`). Option-B greedy deltas: MMLU-Pro −0.0620 abs (−8.6%), GSM8K −0.0020 abs (−0.2%). GSM8K is statistically indistinguishable from base; MMLU-Pro drops one knowledge-bucket's worth but stays above the 90% bar in point estimate.

### Exact commands

```bash
# Server (already up; manifest engine = vLLM 0.22.0):
/usr/bin/python3 research/validity/int4_mtp_spec_quality_panel/serve_spec.py \
  --engine manifest --max-model-len 8192 --max-num-seqs 16 --k 7 --batch-invariant 1

# GSM8K both regimes (n=500, 8-shot, seed 1234):
/tmp/eval-serve-venv/bin/python research/downstream_quality_gsm8k/gsm8k_eval.py \
  --base-url http://127.0.0.1:8000 --model gemma-4-e4b-it --label optionb_pr644_gb6144 \
  --regimes sampled,greedy --n 500 --n-shot 8 --seed 1234 --sampling-seed 1234 \
  --max-tokens 6144 --min-tokens 8 --concurrency 16 --out-dir <results-pr644>

# MMLU-Pro greedy / sampled (n=500, seed 12345):
/tmp/eval-serve-venv/bin/python research/validity/downstream_quality_eval/run_eval.py \
  --task mmlu_pro --n 500 --seed 12345 --max-tokens 6144 --min-tokens 8 --max-connections 16 \
  --base-url http://127.0.0.1:8000/v1 --model gemma-4-e4b-it \
  [greedy:] --temperature 0.0 --top-p 1.0 --top-k 0 \
  [sampled:] --temperature 1.0 --top-p 0.95 --top-k 64 --sampling-seed 1234

# Consolidate + W&B:
/usr/bin/python3 research/validity/int4_mtp_spec_quality_panel/consolidate_pr644.py \
  --wandb_group optionb-quality-mmlu-gsm8k-fern --wandb_name fern/optionb-quality-mmlu-gsm8k
```

### Peak memory

~**20.8 GiB** used on the 22.5 GiB A10G (`nvidia-smi` 20815/23028 MiB) — vLLM `gpu_memory_utilization=0.90` pre-allocation dominates; GPU KV-cache usage peaked ~8.5% during the n=500/concurrency-16 sweep, so memory was never the constraint.

### W&B

Run `mmiyf5ij`, group `optionb-quality-mmlu-gsm8k-fern` (wandb-applied-ai-team/gemma-challenge-senpai). Cells + Wilson CIs + %-of-base + verdict labels logged; full panel JSON attached as artifact `pr644_reading_a_panel`.

### What happened

The panel completes the two non-GPQA Reading-A bars cleanly. GSM8K is a non-event (99.8% of base, both modes identical at 463/500 — arithmetic survives int4+spec untouched). MMLU-Pro is the informative one: the int4 body costs ~8.6% absolute on general knowledge, landing the greedy point estimate (91.4% of base) *above* the 90% bar but with a wide-enough n=500 CI that the lower bound dips under it. This is materially better than GPQA (which point-*failed* its bar at full 10-seed power, per #629/#634 `GPQA_MODEL_LIMITED`): MMLU-Pro only needs more samples to firm up, not a different body. Both MMLU-Pro modes clear the absolute floor (0.605) with CI margin, so MMLU-Pro is not at risk on the floor — only on the stricter 90% bar's CI confirmation. The wirbel #541 `min_tokens=8` guard was load-bearing for GSM8K: the public `senpai/int4_g128_lmhead` official row reads GSM8K **~0.850** *without* the guard vs my **0.926** *with* it (~9% recovery, matching the "~10% low" prediction), confirming the EOS-guard is a serving-artifact fix, not a quality lift.

### Suggested follow-ups

1. **If the advisor needs a CI-confirmed MMLU-Pro ≥90% pass** (not just a point-pass): tighten with n≈1000–2000 or multi-seed pooling. The 91–94% point estimates predict it would clear under power — the opposite situation from GPQA, where more power *confirmed* the fail. Cheap (~7 min/500 on this server).
2. Reading-A assembly is now: **GSM8K PASS**, **MMLU-Pro point-PASS/CI-KNIFE_EDGE** (me), **GPQA FAIL/MODEL_LIMITED** (kanna / #629·#634), **AIME** (denken #637). GPQA remains the sole binding failure; MMLU-Pro/GSM8K do not move the overall verdict off GPQA.
3. Identity caveat unchanged: this config is **not** #319 byte-exact (`int4_marlin_breaks_greedy_identity`), so even with a clean panel it's a relaxed-identity / SURFACE-to-human lane.

### Public evidence used

- **Result `20260617-121233-993_senpai.md`** (`senpai/int4_g128_lmhead`, official HF Job `6a2d5a96234ca64b60121aa5`, W&B `905tbujn`): the exact Option-B body, officially quality-verified at **MMLU-Pro 0.668 / GSM8K ~0.850**. Corroborates the int4 body's MMLU-Pro ~0.66 (my served greedy 0.656) and — by reading GSM8K ~0.85 *without* an EOS-guard vs my 0.926 *with* `min_tokens=8` — independently confirms the wirbel #541 serving-artifact correction the PR mandated.
- Leaderboard context (digest `as=senpai`): top is `fabulous-frenzy` 508 TPS (splitkv-fawindow); Option-B is a quality-validity lane, not a TPS-leaderboard entry, so no leaderboard row is claimed.
