# Option-C: sub-int4 / mixed weight compression — TPS-vs-quality Pareto (PR #611)

**Mode:** LOCAL / `analysis_only=true` / **NO FIRE.** Any config that clears the gates is
non-byte-identical (relaxes #319), so it is *surfaced* to the human for the relax-#319
decision — never auto-fired. `official_tps=0` in all results.

## Hypothesis
The strict-#319 byte-identical AR frontier is closed at the A10G HBM ceiling (126.378 TPS,
9.85 GiB/token, M=1 bandwidth-bound). The only remaining lever to beat it is reading fewer
bytes/token = compressing body weights below int4. This card maps the option-C Pareto and
finds where it crosses the gates (PPL ≤ 2.42 + ≥90%-of-base 4-eval).

## Baseline to beat
- `int4_g128_lmhead` @ **126.378 official TPS**, PPL **2.019**, 9.85 GiB/token, greedy 128/128.
- 4-eval ≥90%-of-base bars: GSM8K ≥ 0.807 · MMLU-Pro ≥ 0.605 · GPQA-D ≥ 0.471 · AIME ≥ 0.090.

## Variants (Stage 0 feasibility, decreasing load-likelihood)
- **(a) 2:4-sparse int4** on MLP gate/up/down (126 modules); attn/PLE/embed/lm_head stay int4.
- **(b) mixed w4/w3** — MLP (or a fraction) → w3 g128; attn+embed+lm_head stay int4.
- **(c) uniform w3a16** — most aggressive; Pareto endpoint.
Stop a variant the moment it won't load/serve and record that as the finding.

## Environment (confirmed on pod)
- Engine: **vLLM dev307** `/tmp/senpai-venvs/5f4c623f772358a2` (vllm 0.22.1rc1.dev307+g3e8afdf78).
  0.22.0 craters MMLU on this int4 model (#547) — avoid for quality.
- Dense bf16 source: `/workspace/gemma_build/qat_unq` (15.9 GB, gemma4, no quant cfg). PRE-BUILT.
- int4 baseline body: `/workspace/gemma_build/int4_g128_lmhead` (10.3 GB). PRE-BUILT.
- Build tooling: `submissions/int4_g128_lmhead/build_quant.py` (compressed-tensors primitives),
  `official_quantized_modules.json` (343 lang modules: 126 MLP, 132 attn, 85 PLE/MatFormer).
- Quality harness: `research/validity/int4g128_quality_gate/` + `int4_mtp_spec_quality_panel/`.
- GPU: 1× A10G (sm_86, 24 GB), CUDA_VISIBLE_DEVICES must be 0 (inherited 1 → torch sees 0 GPUs).

## Phase 0 disk preflight — FINDING
Free at pickup: **132 GB** (nominal threshold 170 GB → under by 38 GB). PROCEEDING because:
1. The 15.9 GB dense source + 10.3 GB int4 body the 170 GB reserved for are **already built**
   (no 16 GB re-download / "163G" rebuild needed).
2. No concurrent eval-scratch writers in this single-student launch (GPU idle, 0 build procs).
3. Variants are built **one at a time** (~5–11 GB each) with intermediate cleanup, so peak
   added footprint stays well within 132 GB.
Disk delta flagged to advisor in the PR. Will abort + report if a build approaches ENOSPC.
