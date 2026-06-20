STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["57izwrp6"],"primary_metric":{"name":"gpqa_diamond_pooled","value":0.503030303030303},"test_metric":{"name":"aime_n30_greedy","value":0.3}}

## Results — int4head fire-prep panel: **FIRE-WORTHY on all axes**

The merged int4 W4A16 **g32 lm_head** (`submissions/int4_mtp_bi0_int4head`, #788; body byte-identical to bi0, only `lm_head.weight` swapped bf16→int4 g32) **holds quality within band on every fire-gate axis**, and is at parity or *above* the bf16-head control (bi0) on each one. The +17.0% local-decode TPS lever (256.74 vs 219.34; PPL 2.0029) is quality-cleared for the official a10g A/B.

| axis | int4 g32 (this PR) | gate | bi0 ref | Δ vs bi0 | verdict |
|---|---|---|---|---|---|
| **GPQA-Diamond pooled** (5×198) | **0.5030** (498/990), Wilson95 [0.4719, 0.5341] | ≥ 0.4712 | 0.4970 (492/990) | **+0.0060** | ✅ PASS |
| **MMLU-Pro** n=250 @4096 (clean) | **0.6920** (173/250) | ≥ 0.572 | 0.644 | **+0.048** | ✅ PASS |
| MMLU-Pro n=250 @2048 (like-for-like) | 0.6040 (151/250) | — | 0.570 (n=100, #773) | +0.034 | ✅ (≥bi0) |
| **AIME greedy maj@1** n=30 | **9/30 = 0.300** | ≥ 0.090 (3/30) | base greedy 6/60 = 0.100 | **+0.200** | ✅ PASS (3.3×) |
| AIME sampled maj@8 n=30 (vs bi0) | **12/30 = 0.400** | — | bi0 maj@8 10/30 = 0.333 | **+0.067** | ✅ (≥bi0) |
| GSM8K greedy (#788, not re-run) | 0.915 | ≥ 0.807 | 0.867 | +0.048 | ✅ PASS |

**`fire_worthy: true` / `all_axes_present: true`** (W&B `57izwrp6`). Routing to **int4 g32** — no fallback to int8 needed.

### Task 1 — GPQA-Diamond (5 seeds × 198, max_tokens=6144, T=1/top_p=0.95/top_k=64, sampling_seed=0)

Seed-paired vs the same-day bi0 control panel (`research/validity/bi0_gpqa_panel/`, 10:16Z, identical harness/seeds/6144 cap, #773 W&B [`kredc30c`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/kredc30c)):

| seed | bi0 | int4 g32 | Δ |
|---|---|---|---|
| 12345 | 0.5303 | 0.5455 | +0.015 |
| 13579 | 0.4949 | 0.5101 | +0.015 |
| 23456 | 0.4899 | 0.5152 | +0.025 |
| 34567 | 0.5000 | 0.4848 | −0.015 |
| 45678 | 0.4697 | 0.4596 | −0.010 |
| **pooled** | **0.4970** | **0.5030** | **+0.006** |

- Gate is on **pooled** accuracy (PR #795): 0.5030 ≥ 0.4712 ✅. Even the **Wilson-95% lower bound (0.4719) clears the gate**.
- **Diagnostic — per-seed spread:** int4 g32 min seed = 0.4596 (s45678), which is below the 0.4712 bar. **So is bi0's min seed (0.4697, also s45678)** — i.e. a per-seed min gate would reject the bf16-head control itself, so it is not the gate. s45678 is simply the hardest shuffle for both arms; the int4↔bi0 gap there is −0.010 (2 items). The arms are statistically indistinguishable (pooled +6/990; 4/5 seeds within ±0.025).
- Clean run: 0 errors, 3/990 length-truncated, mean 2003 completion tokens. (bi0 had 1 scored-error/seed; int4 had 0.)
- I corrected my own aggregator (`aggregate_and_log.py`) which had ANDed a stricter per-seed-min clause into the GPQA verdict — that contradicts the PR's pooled gate and would have failed the bi0 control too. Min-seed + Wilson-lo are now reported as diagnostics; the gate is pooled. (This is my analysis script in `research/`, not the eval harness.)

### Task 2 — MMLU-Pro (n=250, 16-shot CoT, T=1/top_p=0.95/top_k=64, seed=12345)

- **@4096 (clean, PR-mandated): 0.6920** — above the 0.572 gate **and above bi0's 0.644 clean ref** (+0.048). Truncation only 2/250 (0.8%); 12.4% of completions would have been cut at 2048 but resolved by 4096.
- @2048 (like-for-like vs bi0 0.570): 0.6040, with 29/250 (11.6%) truncated — confirms the 2048 budget's truncation artifact the PR warned about, and int4 g32 still beats bi0's 2048 number.
- Harness = shared `research/validity/downstream_quality_eval/run_eval.py` (inspect_evals `mmlu_pro`), byte-identical to the bi0 #762 reference, so the comparison is apples-to-apples.

### Task 3 — AIME n=30 (years=2024)

- **PRIMARY (PR-mandated #580 protocol): greedy maj@1, temp=0, top_p=1, top_k=−1, max_tokens=3072, min_tokens=8, no-thinking, single-stream (concurrency=1).** Result **9/30 = 0.300**, gate ≥ 0.090 → clears **3.3×**. extract_fail = 0.
  - **Caveat — length truncation: 19/30** single greedy samples hit the 3072-token cap. This is inherent to the #580 tight cap on a long-CoT reasoning task (it depresses *both* base and variant); the base greedy ref under the same protocol is 6/60 = 0.100, and int4 g32 (0.300) exceeds it. maj@1 at n=30 is also a noisy single-draw statistic (cf. int4 AIME cross-session greedy nondeterminism), but the 3.3× gate margin absorbs that.
- **SUPPLEMENT (like-for-like vs bi0 #762): sampled maj@8, T=1/top_p=0.95/top_k=64, concurrency=16.** Result **12/30 = 0.400**, **above bi0's 10/30 = 0.333** (+0.067), mean pass-rate 0.2875, extract_fail = 0. This is the more robust AIME signal (8 samples/problem) and it favors int4 g32.

### Served config / provenance

- Model: `/workspace/gemma_build/bi0_int4head_g32` (10.5 GB), the exact #788 build — local validation reproduced **PPL 2.0029 / TPS 256.74** (`research/_int8head_smoke/prevalidate_int4_candidate/local_summary.json`), so the served head is the validated int4 g32 artifact.
- Serve: `submissions/int4_mtp_bi0_int4head/serve.py` → vLLM, `--max-model-len 12288 --gpu-memory-utilization 0.90 --max-num-seqs 16 --max-num-batched-tokens 512`, MTP K=6 drafter `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant`, surgattn force-2D, `VLLM_BATCH_INVARIANT=0`.

### Commands

```bash
nvidia-smi   # A10G, server already warm on :8020
# GPQA(5×198,6144) + MMLU-Pro(n=250 @4096 & @2048), then AIME(greedy maj@1 + sampled maj@8):
bash research/validity/int4head_firegate_panel/run_gpqa_mmlu.sh        # panel
bash research/validity/int4head_firegate_panel/chain_aime_after_panel.sh  # AIME (gated on PANEL DONE)
/usr/bin/python3 research/validity/int4head_firegate_panel/aggregate_and_log.py   # verdict + W&B
```

- **W&B run:** `57izwrp6` (group `bi0-int4head-firegate`, `wandb-applied-ai-team/gemma-challenge-senpai`).
- **Peak GPU memory:** ~20.3 GB / 23 GB on the single A10G (vLLM EngineCore, max-model-len 12288), steady through the panel.
- **LOCAL ONLY** — `analysis_only: true`, `official_tps: 0`. **No HF Job launched** (per operator rule; the official a10g A/B needs explicit human approval).

### What happened

The hypothesis held cleanly. Swapping only the lm_head from bf16 to int4 W4A16 g32 (PPL even *improves* 2.0058→2.0029) is **quality-neutral-to-positive on all three harder downstream panels**, mirroring the cheap-level GSM8K/PPL signals from #788. On every axis the int4 g32 head matches or beats the bf16-head bi0 control measured under an identical same-day harness. The single soft spot — GPQA seed 45678 dipping to 0.4596 — is shared by bi0 (0.4697) and disappears in the pooled estimate, whose Wilson lower bound still clears the gate.

### Suggested follow-ups

- **Open the HF approval issue for int4 g32** — this is the +17% lever ready to fire; it projects ~255 official TPS, breaking the #784 250-TPS target in a single lever, with the full quality panel now cleared.
- For the official A/B, GPQA-Diamond is the thinnest margin (pooled +0.006 over bi0, Wilson-lo 0.4719 just above the 0.4712 bar). If the advisor wants extra safety, a 10-seed GPQA pool would tighten the CI before spending submission quota — but it is not required by the gate.
- int8 channelwise (`int4_mtp_bi0_int8head`, +9.9% TPS, recon-err 0.010) remains the lower-risk fallback if the official GPQA A/B regresses unexpectedly; no int8 panel was needed here since int4 cleared every axis.
