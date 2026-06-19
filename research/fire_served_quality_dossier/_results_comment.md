STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["0eob690h"],"primary_metric":{"name":"mmlu_pro_acc","value":0.632},"test_metric":{"name":"gsm8k_acc","value":0.8667}}

## Results — Fire served-quality dossier (MMLU-Pro / GSM8K / AIME)

**Headline: the as-fired int4 + MTP-spec-dec fire retains ~100% of base reasoning across the panel (panel-mean 100.2% of base).** Speculative decoding does not degrade reasoning — consistent with the PPL 2.0055 ≤ 2.42 gate.

The dossier is two matched arms of the **same** submission `submissions/int4_mtp_batchinv`, served locally on the A10G through the real vLLM 0.22.0 api_server. The **only** difference between arms is the speculative drafter, so "base" is a perfectly-matched denominator that isolates the spec-dec drafter:

- **fire** — as-fired, drafter **ON** (`num_spec_tokens=6`, MTP `…q4_0-unquantized-assistant`).
- **base** — `SENPAI_REFERENCE_MODE=1` forces `num_speculative_tokens=0` → drafter **OFF**, plain int4 M=1 AR on the identical W4A16 / Marlin / vLLM stack (`speculative_config=None`).

### Panel (sampled T=1.0, top_p=0.95, top_k=64; min_tokens=8 EOS-guard; BI=1)

| Task | Metric | fire (drafter ON) | base (drafter OFF) | **% of base** |
|---|---|---|---|---|
| **MMLU-Pro** | acc | **0.6320** (158/250) | 0.6280 (157/250) | **100.64%** |
| **GSM8K** | acc | **0.8667** (260/300) | 0.8767 (263/300) | **98.86%** |
| **AIME 2024** | maj@8 | **0.3667** (11/30) | 0.3667 (11/30) | **100.00%** |
| AIME 2024 | mean pass-rate | 0.3125 | 0.3083 | 101.36% |
| | | | **panel mean** | **100.21%** |

All three tasks land within ±1.2% of the drafter-off base; MMLU-Pro and AIME-maj@8 are at-or-above base, GSM8K is 1.0 pp under (3 problems, within sampling noise at n=300). AIME maj@8 solves the **same 11/30** problems with and without the drafter.

### Denominator note (honesty flag)
The %-of-base above is **% of the drafter-OFF int4 base**, which is the correct denominator for the PR's question ("does the int4 **+ spec-dec** fire degrade reasoning?") — it isolates exactly the speculative drafter on a bit-identical stack. It is **not** "% of full-precision bf16 base". The int4 W4A16 quantization step is already baked into both arms. If the blog wants a "% of full-precision base" framing as well, that needs a third bf16-served arm (flagged as a follow-up, not in scope here).

### Protocol
- Decode: `generation_config.json` sampling (T=1.0, top_p=0.95, top_k=64) per lewtun #31 — the quality-monitoring axis, distinct from the greedy-identity gate.
- EOS-guard: `min_tokens=8` (#541) applied to **all** evals (empty_rate=0.0 on both MMLU-Pro arms; no first-token-EOS depression).
- `VLLM_BATCH_INVARIANT=1` + per-request seeds → each request's decode is batch-invariant, so `MAX_NUM_SEQS=16` (raised for eval tractability) leaves per-request outputs unchanged and the two arms matched.
- Native torch sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`): this box's CUDA toolkit ships no `curand.h` for the flashinfer JIT sampler; native top-k/top-p is numerically standard.
- `CUDA_VISIBLE_DEVICES=0` (container maps the A10G as NVML index 0).
- N: GSM8K 300, MMLU-Pro 250, AIME 2024 = 30 problems × k=8.

### Server proof (drafter ON vs OFF)
```
fire: speculative_config=SpeculativeConfig(method='mtp',
        model='…q4_0-unquantized-assistant', num_spec_tokens=6)  + "Loading drafter model…"
base: [serve] SENPAI_REFERENCE_MODE active: forcing num_speculative_tokens=0 (drafter OFF)
        Initializing … with config: … speculative_config=None
```

### Commands
```bash
cd /workspace/senpai/target
# fire arm (drafter ON)
python3 research/fire_served_quality_dossier/run_dossier.py --arm fire \
  --max-num-seqs 16 --mmlu-n 250 --gsm8k-n 300 --aime-years 2024 --tasks gsm8k,mmlu,aime
# base arm (drafter OFF, matched denominator)
python3 research/fire_served_quality_dossier/run_dossier.py --arm base \
  --max-num-seqs 16 --mmlu-n 250 --gsm8k-n 300 --aime-years 2024 --tasks gsm8k,mmlu,aime
# aggregate + log to W&B group fire_served_quality_dossier
python3 research/fire_served_quality_dossier/aggregate_dossier.py
```

### Run facts
- **W&B run id:** `0eob690h` (group `fire_served_quality_dossier`) — https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/0eob690h
- **GPU mem:** `gpu_memory_utilization=0.9` → vLLM reserves ~20.7 GB of the 23 GB A10G (int4 target + drafter weights + KV); one arm served at a time. The drafter is ~int4-of-a-small-assistant, so it fits comfortably under the 0.9 cap alongside the int4 target.
- **Wall:** ~17 min per arm (server load + 3 evals), single A10G, one arm at a time.
- **No HF Job; LOCAL served evaluation only.**

### What happened
The fire's spec-dec drafter is **lossless on reasoning** at the served, sampled, quality-monitoring axis: every panel task is within sampling noise of the drafter-off base, with two of three at-or-above. This is the expected result — MTP speculative decoding is a *speed* technique whose accepted tokens are verified against the target's own distribution, so it should not move task accuracy; the dossier confirms it empirically rather than relying on PPL alone. This pairs with the official TPS as the "high quality" evidence the #730 human / @cmpatino blog asked for.

### Suggested follow-ups
- **bf16 full-precision denominator arm** if the blog wants "% of full-precision base" (would quantify the int4 quantization cost separately from spec-dec; needs a second large model served, ~+17 min).
- **Tighter CIs** — n=300/250/30 give ~±3-6 pp per-task; if the blog wants error bars, bump GSM8K/MMLU-Pro n and add AIME 2025 (60 problems total).
- **MMLU-Pro full 12k** if a headline single-number "MMLU-Pro = X" is wanted rather than the n=250 screening subset.
