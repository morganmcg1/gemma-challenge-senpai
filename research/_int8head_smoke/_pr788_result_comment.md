STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["9tcygwjf"],"primary_metric":{"name":"local_decode_tps_int4_head","value":256.74},"test_metric":{"name":"gsm8k_greedy_acc_int4_head","value":0.915}}

## Results — Fewer-weight-bytes: quantize the bf16 lm_head (int8 → int4) on bi0

**Both arms clear every candidate-worthiness gate. int4 W4A16 g32 is the bigger win (+17.0% local decode TPS, quality intact).** The hypothesis holds: the bf16 lm_head GEMV is a real, un-amortized per-accepted-token bandwidth cost, and shrinking it flows straight to decode TPS.

All three arms differ from each other by exactly **one tensor** — `lm_head.weight`. The int4 W4A16 body + embeddings + vision/audio towers are copied **byte-identical** (2762 tensors) from `google/gemma-4-E4B-it-qat-w4a16-ct`, so this isolates a single variable.

### Headline

| metric | bf16 control (bi0) | Arm A — int8 W8A16 ch | Arm B — int4 W4A16 g32 |
|---|---|---|---|
| lm_head kernel | cuBLAS `gemv2T` (bf16) | AllSpark (Ampere) | Marlin (same as body) |
| lm_head bytes/token | 1.342 GB | 0.671 GB (**2.00×**) | 0.378 GB (**3.56×**) |
| **local decode TPS** (spec ON, 128 prompts) | 219.34 | **241.09 (+9.9%)** | **256.74 (+17.0%)** |
| **per-token lm_head GEMV** (eager, M=1) | 2.777 ms | **1.430 ms (1.94×)** | **0.750 ms (3.70×)** |
| **PPL** (gate ≤ 2.42) | 2.0057 | 2.0051 | **2.0029** |
| decode completed | 128/128 | 128/128 | 128/128 |
| **GSM8K greedy** (paired n=200, seed 1234) | 0.9200 | 0.9300 (**+1.09%**) | 0.9150 (**−0.54%**) |
| all-4-modalities | ✓ | ✓ | ✓ |
| quant rel_err | — | 0.00966 | 0.06743 |

### Gates (candidate-worthy if ALL hold) — both arms PASS

| gate | int8 | int4 g32 |
|---|---|---|
| local decode TPS > bi0 control (219.34 > 218.02 official floor) | ✅ 241.09 | ✅ 256.74 |
| PPL ≤ 2.42 | ✅ 2.0051 | ✅ 2.0029 |
| 128/128 completed | ✅ | ✅ |
| all-4-modalities | ✅ | ✅ |
| downstream quality within 5% of base | ✅ +1.09% | ✅ −0.54% |

### Mechanism — per-token lm_head GEMV (reproduces PR #781's profiler)
Same harness / GPU / session, eager M=1 spec OFF, `scripts.local_validation.profile_decode --profile-mode op`:
- **bf16 2.777 → int8 1.430 → int4 0.750 ms/token**; bytes/token 1.342 / 0.672 / 0.378 GB.
- Speedups (1.0× / 1.94× / 3.70×) track the byte ratios → the GEMV is **bandwidth-bound** (~470–500 GB/s effective), exactly as the lever predicts.
- bf16 reproduces #781's 2.776 ms/token to 3 decimals. The int4 lm_head shares the body's Marlin template (no separate kernel entry); isolated by the `_C::marlin_gemm` op-count rising by **exactly +256** (= once per generated token) vs the byte-identical int8 body, Δ = 191.9 ms / 256.
- The lm_head fires once per **accepted** token (not per draft row), so MTP does **not** amortize it — confirmed: the saving flows to decode TPS under spec ON.

### Downstream quality reference
GSM8K 8-shot CoT, greedy, paired same-subset (seed 1234, n=200). Reference: merged-bi0 panel anchor **0.867** (sampled), official harness gate **≥0.807**. Paired per-item (int4 vs control): 182 both-correct, 15 neither, 2 bf16-only-correct, 1 int4-only-correct → 3/200 discordant (statistical tie). int8: 182 both, 12 neither, 2 bf16-only, 4 int8-only. All arms sit comfortably above 0.867 on this greedy subset.

### Greedy divergence vs bf16 control — EVIDENCE only (not a gate, per #784/#788)
- int8: 12/128 identical, frac_diverged 0.906, mean prefix-match 0.384.
- int4: 3/128 identical, frac_diverged 0.977, mean prefix-match 0.201.
- **Interpretation:** high token-level divergence is expected and is **quality-neutral**. Greedy autoregressive decode amplifies tiny argmax-margin perturbations over a 512-token rollout; the first flip forks the whole trajectory. The robust quality signals (PPL identity to ±0.1%, GSM8K paired agreement) show no degradation — int4 even has the lowest PPL. The bf16 lm_head GEMV is itself cross-session non-deterministic at the argmax (bf16 reduction-order), so a single control rollout is one noisy draw; byte-identity was never required for this lever.

### Peak memory (honest tradeoff)
Model-load weights: bf16 control **9.86 GiB** → int8 **10.49 GiB** → int4 **10.22 GiB**. Memory goes **up** slightly because quantizing the head requires untying it from `embed_tokens` (`tie_word_embeddings=false`): `embed_tokens` stays bf16 (1.342 GB) for the input gather **and** a separate quantized head is added. This is a storage cost, not a bandwidth cost — the lever is the per-token decode **read** (lm_head only), which shrinks 1.342→0.378 GB and drives the TPS win. Ample A10G headroom remains (KV cache 8.1 GiB available, ~22.5 GiB total).

### Exact commands
```bash
# Build (CPU, local only — no HF Job): int8 channelwise / int4 g32 lm_head, body byte-identical
python submissions/int4_mtp_bi0_int8head/build_lmhead_quant.py \
  --src <google/gemma-4-E4B-it-qat-w4a16-ct snapshot> \
  --out /workspace/gemma_build/bi0_int8head_ch  --num-bits 8 --head-group-size -1
python submissions/int4_mtp_bi0_int4head/build_lmhead_quant.py \
  --src <…w4a16-ct snapshot> \
  --out /workspace/gemma_build/bi0_int4head_g32 --num-bits 4 --head-group-size 32

# Prevalidate (local A10G; CUDA_VISIBLE_DEVICES=0, VLLM_USE_FLASHINFER_SAMPLER=0)
python scripts/local_prevalidate.py --submission submissions/int4_mtp_bi0_int4head \
  --venv-python <venv> --port 8013 --decode-num-prompts 128 --ppl-records 0 --output-dir <out>

# lm_head GEMV op-profile (eager, M=1, spec OFF)
python -m scripts.local_validation.profile_decode --model-id /workspace/gemma_build/bi0_int4head_g32 \
  --mode eager --profile-mode op --out-dir <out>

# Downstream quality (GSM8K 8-shot CoT greedy, paired)
python research/downstream_quality_gsm8k/gsm8k_eval.py --submission submissions/int4_mtp_bi0_int4head \
  --server-python <venv> --label int4head --regimes greedy --limit 200 --n-shot 8 --concurrency 32 --port 8023

# All-4-modalities (presence tier)
python -m scripts.local_validation.modalities_probe --submission submissions/int4_mtp_bi0_int4head
```

### W&B
Run [`9tcygwjf`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/9tcygwjf), group `bi0-lmhead-bytes` (`analysis_only=1`, `no_hf_job=1`). Full 3-arm table + all metrics logged.

### What happened
The lever works and is cleaner than expected. The bf16 lm_head GEMV (2.777 ms/token, ~31% of M=1 weight bytes) is genuinely un-amortized by MTP, so quantizing it converts directly into decode TPS: **int8 +9.9%, int4 +17.0%** (local, exploratory). Quality is preserved at both bit-widths — the QAT-trained int4 g32 recipe that the body already uses transfers cleanly to the head (PPL 2.0029, GSM8K within noise of base), so int4 carries no measurable quality penalty over int8 despite ~7× higher reconstruction error (0.067 vs 0.010). The GEMV speedups track byte ratios to within a few %, confirming the operation is purely bandwidth-bound and the model is sound. **Recommendation: int4 g32 is the candidate to promote.**

### ⚠️ Official confirmation needs an advisor-approved HF Job
These TPS numbers are **local A10G exploratory** (PPL + greedy token_ids + GSM8K are hardware-independent and carry over). Per the operator rules I have **not** launched any HF Job / submission. To get the official a10g-small served TPS for the int4 (or int8) arm, please approve an HF benchmark job — I'll open the `Approval request: HF job for …` issue with the exact command and artifact paths on your go-ahead. `model_id` is currently a local build path; it must be published to a Hub repo before any HF Job.

### Suggested follow-ups (not implemented — flagging only)
1. **Promote int4 g32** to an official a10g-small rung (advisor-approved HF Job) — biggest measured TPS lever here.
2. **int4 channelwise** (`--head-group-size -1`): even fewer bytes (no per-group scales) → ~4× vs 3.56×. Quality risk is higher (channelwise int4 on a 262k-row head); worth a cheap PPL/GSM8K screen if max TPS is wanted.
3. **MSE/clip-aware head scales:** the builder uses minmax. If int4 quality ever looks marginal at scale, an MSE-optimal clip (or GPTQ on the head only) would tighten rel_err at ~zero TPS cost. Not needed at g32 (quality already intact) — purely a hedge.
4. **Quantize `embed_tokens` too** to recover the untie storage cost — but it's a cheap gather (not bandwidth-bound at decode), so this is a memory optimization, not a TPS one; only worth it if head-storage memory becomes binding.
