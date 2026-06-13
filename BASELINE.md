# BASELINE — Fast Gemma Challenge (advisor branch `approval-gated-8gpu-20260613`)

Primary metric: **`summary.json:tps` (output-token throughput, higher is better)**, measured
single-stream (max concurrency 1), output_len 512, on the fixed 128 public prompts,
on **a10g-small** via HF Jobs. Local AWS A10G numbers are **exploratory only**.

Validity gates (a submission is invalid if any fail):
- **PPL ≤ ~2.42** (reference 2.30 + 5%).
- **128/128** prompts completed.
- **Greedy decode token-identical** to plain greedy AR decode of the *submitted* checkpoint. **Reference must be served (spec-off API), not offline** — offline AR diverges on ~20% of prompts due to FP-reduction non-determinism (wirbel PR #8); an offline reference would falsely fail ~20% of valid served submissions.
- **All modalities loaded** (text/image/audio) — no text-only shortcut.

## Public frontier target (what we are reproducing, then beating)

Top **VALID** leaderboard entry as of 2026-06-13:
- **kenyan-duma `osoi5-feopt2-w20-e1-lmhead12k-fa2sw-precache` — 421.12 TPS / PPL 2.3774, 128/128** (job `6a2c7688871c005b5352b87a`).
- Other VALID repros at ~420.6–420.8 (frantic-penguin `fa2sw-fp`, agent-smith `fa2sw-v3`).
- The 3 entries above it (446–449 TPS: `ff-lf29cap432`, `mao-gemma-fast-cap433`, `pupa-lf29cap-repro`, all using a `DECODE_TPS_CAP`) are **PENDING / unverified** and look like decode-TPS-cap gaming — **not** our target. We target legitimately reproducing and beating the ~420 VALID frontier.

## The climb (intermediate milestones — our reproduction ladder)

| milestone | TPS (a10g-small) | PPL | lever |
|---|---|---|---|
| bf16 stock (`vllm_baseline`) | ~44.0 | ~2.30 | none (reference) |
| int4 QAT W4A16 (Marlin), as-is | ~95.4 | ~2.01 | 4× less weight bandwidth (dominant lever) |
| + untied int4 lm_head + full-body g128 | ~126.8 | ~2.02 | int4-Marlin **weight-byte floor** on Ampere |
| + MTP / QAT-drafter spec decode (K≈6) | ~273–286 | ~2.0–2.4 | amortize weight read over ~3.3 accepted tok/step |
| + lmhead12k sparse-verify + fa2sw + onegraph + precache | **~420** | **~2.377** | verify-cost + runtime + warmup levers (the frontier) |

Decode at conc=1 is **memory-bandwidth-bound** (profiler: ~92% weight-GEMM, attn ~2.6%, sampling ~0.2%).
Levers: (a) fewer weight-bytes/token, (b) more accepted tokens per weight read (better drafter),
(c) erase per-step overhead / cheaper 262k-vocab verification.

## Key risk for any near-frontier submission
The verifier re-runs on a **private** prompt set; top drafter stacks lose **4–9% TPS** on it
(prompt-distribution shift). Submissions die on the **5% TPS-reproduction gap, not on PPL**.
Private-stable acceptance (drafter trained on a wide distribution; prompt-content-invariant
verify paths) is a first-class objective, not an afterthought.

## Current local baseline in this repo
- **OFFICIAL BASELINE (a10g-small HF-Job confirmed) — `submissions/int4_g128_lmhead` (PR #4, lawine) — official a10g-small tps=126.378, ppl=2.019, 128/128 VALID** (job
  `6a2d5a96234ca64b60121aa5`, W&B `905tbujn`). int4 g128 + untied int4 lm_head re-quant, all modalities loaded, greedy-valid (GREEDY_IDENTICAL
  128/128 served-vs-served), same-path OK (gap 0.0000). **2.87× over bf16. 1.32× over PR #3 int4 base.**
  **The official bar all submissions must beat remains 126.38 TPS** until a gated HF job confirms a higher rung.
- **BEST-LOCAL RUNG (official a10g-small PENDING) — `submissions/lmhead12k_empirical` (PR #14, ubel) — 131.60 local single-stream / PPL 1.9712 / GREEDY_IDENTICAL 128/128 (self-consistency).** Top-12k bf16 lm_head prune; isolated single-variable lever **+34.8%**, local-to-local net over PR #4 **+2.7%**. Validated lever + auditable evidence, but local-only per the exploratory-only rule → official TPS + private-PPL await a gated HF job (approval issue opened). Does **not** displace the official 126.378 headline yet.
- Prior rung: `submissions/int4_qat` (PR #3, stark) — 95.463 TPS / PPL 2.0057 (int4 QAT W4A16 floor).
- `submissions/vllm_baseline` — bf16 stock vLLM 0.22.0 endpoint (**reference floor**). Prior HF smoke job
  `6a2c5fb77c68f455eff14260` (run prefix `results/senpai/vllm-baseline-20260612T193622Z`)
  reported **tps=44.018, completed=128** on a10g-small.
- **PPL-artifact resolution (priority #1, fern, PR #2) — RESOLVED 2026-06-13.**
  - Local PPL **confirmed 2.3012** over all 128 GT records (61,797 scored tokens) via the
    official `ppl_endpoint.py` against a local bf16 `serve.py` endpoint — within the ≤2.42 gate.
  - The prior job's missing `ppl_summary.json` was **not** disabled / OOM / unfetched: it was the
    **40-min HF Job wall-clock timeout**. Evidence (`job_status.json` timed_out@40m stage=RUNNING,
    `run_environment.json` ppl.enabled=true, `summary.json` duration_s=1488.8s) shows 11.9-min cold
    startup + 24.8-min benchmark left only ~6.5 min, so decode-capture (another ~24.8-min workload)
    and the PPL stage that runs *after* it never completed. At 44 TPS the bf16 baseline cannot fit
    startup+benchmark+decode+PPL inside 40 min; faster submissions will.
  - Reusable one-command local pre-validation harness: **`scripts/local_prevalidate.py`** (serves
    bf16, runs PPL + decode capture, prints `tps`/`ppl`/`completed`, no HF Jobs quota). Evidence
    artifacts under `research/local_validation/`.

## Merge history

### 2026-06-13 14:00 — PR #3: Reproduce int4 QAT W4A16 leader (~95 TPS) — base of the stack ⭐ NEW OFFICIAL BASE RUNG

- **Primary metric (tps):** **95.463** (official a10g-small, job `6a2d55c7234ca64b60121a6f`, run `results/senpai/int4-qat-20260613T130614Z`) — **2.17× over bf16 44.018**.
- **PPL (gate):** **2.0057** ≤ 2.42 ✓ (better than bf16's 2.30 same-path — Google's quality-matched QAT checkpoint).
- **completed:** 128/128 ✓ · **total_tps** 144.53 (diagnostic) · **duration_s** 686.5 · **job_status** COMPLETED ✓.
- **Validity:** all modalities loaded (vision/audio bf16 via QAT `ignore` list, no `--limit-mm-per-prompt`); greedy-valid within the same serve/job stack (no token-changing optimization added); cold-start fit the 40-min cap with ~3.5 min to spare (`ppl_summary.json` wrote 13:42:23Z).
- **W&B run:** N/A (serving-submission reproduction, no training). Official artifacts: `results/senpai/int4-qat-20260613T130614Z/{summary.json,ppl_summary.json,decode_outputs.jsonl,benchmark.jsonl,job_logs.txt}`.
- **Submission:** `submissions/int4_qat/` (`manifest.json` + `serve.py`), checkpoint `google/gemma-4-E4B-it-qat-w4a16-ct`, vLLM 0.22.0 / transformers 5.9.0 / `--dtype bfloat16`, Marlin int4 W4A16, CUDA graphs FULL_AND_PIECEWISE.
- **Reproduce (local exploratory):** `cd target/ && VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/local_prevalidate.py --submission submissions/int4_qat --decode-num-prompts 16` (local ≈ 95.99 TPS / 2.0055 PPL, <0.6% off official). **Official run is HF-Job + human approval only** (issue #11 approved).
- **Significance:** the foundation the entire ~420 frontier stack builds on. int4 W4A16 is confirmed the dominant single-stream lever on official hardware (memory-bandwidth-bound decode, ~4× less weight bandwidth). Next rung: int4 g128 + untied int4 lm_head (~127 TPS, lawine PR #4 in flight).

### 2026-06-13 08:40 — PR #2: Resolve PPL artifact path + validate bf16 baseline locally

- **Priority #1 resolved.** Root cause: 40-min HF Job wall-clock timeout (not OOM / disabled / unfetched).
- **Local PPL:** 2.3012 (128/128 GT records; within ≤2.42 gate) ✓
- **Local TPS (exploratory, A10G):** ~44.01 (16-prompt sample — not official a10g-small)
- **W&B run:** none (local validation + infra, no training)
- **New shared infra:** `scripts/local_prevalidate.py` — one-command local pre-validation for all future submissions.
- **Reproduce:** `cd target/ && VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/local_prevalidate.py --submission submissions/vllm_baseline --decode-num-prompts 16`
  (Env-var is a local-box workaround for broken FlashInfer JIT; not needed on official a10g-small image.)

### 2026-06-13 09:45 — PR #10: Offline suffix-run token-budget analysis for SAM-Decoding feasibility

- **Finding (GO on causal budget):** causal SAM-Decoding realized budget = **8.93%** free tokens at K>8 (K>4: 15.4%, K>6: 11.6%); clear **GO** for the Triton-kernel follow-up (threshold >3.6%). Robust across datasets (aime 10.74%, gpqa 9.23%, mmlu_pro 8.19%); greedy-safe by construction (zero PPL risk).
- **PR-spec proxy (`m(t)`):** 1.21% — *not* the decision metric. `m(t)` fires only on adjacent-period repetition; the exploitable structure is non-adjacent. Causal estimate cross-validated against brute-force O(n²) reference: 0 mismatches / 600 positions.
- **Caveat:** gain is *incremental* over the existing MTP/QAT drafter (~3.3 tok/step). Net headroom requires per-step acceptance trace from kanna's #5 — measuring SAM-drafter overlap de-risks the Triton build before GPU spend.
- **New shared infra:** `scripts/analyze_suffix_budget.py` — offline CPU-only suffix-budget analyzer; designed for extension to ingest a drafter acceptance trace for overlap quantification.
- **W&B run:** none (CPU-only offline analysis). 128/128 prompts captured (bf16, 43.94 TPS local).
- **Reproduce:** `cd target/ && python scripts/analyze_suffix_budget.py --input research/local_validation/vllm_baseline/decode_outputs_128.jsonl --output research/local_validation/suffix_budget/suffix_budget_analysis.json`

### 2026-06-13 10:30 — PR #13: SAM-Decoding drafter-overlap intersection analysis (de-risk Triton build)

- **New shared infra:** `scripts/analyze_suffix_budget.py --drafter-trace <file>` — extends PR #10 tooling with intersection logic. Computes `net_sam_beyond_drafter_frac` (SAM causal budget ∩ drafter acceptance = the decision metric for the Triton kernel GO/retire); 13/13 mock tests pass; no-drafter path byte-identical (regression-safe). Plus `research/sam_drafter_overlap/overlap_analysis_template.json` and `scripts/tests/test_drafter_overlap.py`.
- **Trace format (canonical):** `{"prompt_idx":0,"step":0,"accepted_token_ids":[...],"acceptance_len":N,"output_start":K}` — `output_start` is required for correct interleave alignment when spec tokens are interspersed with bonus tokens.
- **Net-headroom thresholds:** `net_frac > 0.03` → GO (open Triton kernel PR); `0.01–0.03` → marginal; `< 0.01` → retire SAM direction.
- **Caveat (fern):** real MTP drafter concentrates acceptances on predictable/repetitive spans — exactly where SAM runs live — so real overlap is likely higher than a uniform-random drafter, pushing real `net` lower than naive intuition. The base 8.93% budget is small. Brace for marginal/retire.
- **W&B run:** none (CPU-only tooling). Dev dep added: `pytest>=8` + `iniconfig` + `pluggy` (dev-only, no existing dep bumps).
- **Reproduce (smoke):** `cd target/ && uv run python -m pytest scripts/tests/test_drafter_overlap.py -v`
- **Reproduce (full analysis when trace lands):** `cd target/ && python scripts/analyze_suffix_budget.py --input research/local_validation/vllm_baseline/decode_outputs_128.jsonl --drafter-trace <trace.jsonl> --output research/sam_drafter_overlap/overlap_analysis.json`

### 2026-06-13 11:15 — PR #15: EAGLE-3 feature-export feasibility

- **Verdict: ACCESSIBLE → GO.** vLLM 0.22.0 + Gemma-4 E4B ship a complete EAGLE-3 feature-export path with **zero patching** — `Gemma4ForConditionalGeneration` implements `SupportsEagle3`; `Gemma4Model` is `EagleModelMixin`; aux layers `(2, 21, 39)` over the 42-layer E4B body; each `[T, 2560]` bf16, CUDA-graph safe (persistent buffers pre-allocated at capture). The drafter head arch also already exists (`models/llama_eagle3.py`, `v1/spec_decode/eagle.py`). Wire: `speculative_config{method:"eagle3", model:<draft>, eagle_aux_hidden_state_layer_ids:[2,21,39]}`.
- **Empirical probe:** `probe_result.json` confirms `supports_eagle3=True`, default_aux_layers=[2,21,39], 3 aux tensors shape [5,2560], no NaN, vision+audio towers intact; 15.3 GiB peak bf16 on A10G (fits).
- **Ceiling (literature):** ~480–550 TPS at accepted tok/step ~4–5+, vs current QAT-MTP ~2.2–3.3 tok/step. Serving validity still gated on kanna #5 linchpin (is int4 batched-verify spec greedy-valid?).
- **New shared infra:** `research/eagle3_feasibility/{feasibility_report.md, probe_eagle3_export.py, probe_result.json, probe.log}`
- **W&B run:** none (source audit + single model-load probe; no training).

### 2026-06-13 12:25 — PR #8: Local validation + profiling infra (greedy gate, PPL, profiler)

- **Infra shipped:** `scripts/local_validation/` — one-command `validate_submission`, served spec-off greedy reference generator (`gen_greedy_reference --spec-off`), local PPL runner, decode op-profiler. All future HF-Job approval issues should attach `validate_submission` output.
- **Critical methodological finding (greedy gate):** Offline AR reference diverges on 26/128 prompts (20.3%) from FP-reduction non-determinism. Greedy gate must compare **served-vs-served (spec-off)** — offline reference falsely fails ~20% of valid served submissions. `validate_submission` defaults to served anchor.
- **Profiler finding (int4 base, graph mode, 96.91 tok/s local):** lm_head vocab GEMV = **26.4% of de-duped decode GPU time** (262k-vocab bf16 GEMV). This is the largest addressable non-block, non-int4 target — directly confirms lmhead12k (ubel #14) as the top non-block, lowest-PPL-risk frontier lever. Weight-GEMM total 91.6%, attn 2.7%, norm/elementwise 3.8%, sampling 0.2%.
- **One-flag spec-off reference mode:** `gen_greedy_reference --mode served --spec-off` injects `SENPAI_REFERENCE_MODE=1` so drafter students get a canonical spec-off greedy reference on their own engine/kernels/quant before spending an HF-job slot.
- **Canonical greedy reference committed:** `research/greedy_reference/google__gemma-4-E4B-it/` (bf16 base, 128 prompts, served spec-off).
- **W&B run:** none (local infra + profiler, no training).
- **One-command validation:** `python -m scripts.local_validation.validate_submission --submission submissions/<dir> --server-python /tmp/server-venv/bin/python`

## Confirmed dead ends (do not re-spend on these)
sub-4-bit weight kernels (AWQ/GPTQ/AQLM/QuIP#/2:4-Sparse-Marlin/NVFP4) — no loadable Ampere
sm_86 kernel in vLLM 0.22; fp8 KV cache — rejected by A10G + Gemma4 attn; n-gram/prompt-lookup
spec decode — loses at conc=1; runtime knobs (attn-backend swap, max_num_seqs, MARLIN_USE_ATOMIC_ADD) —
parity/noise; body channel-wise quant — trades PPL for no TPS; widening draft centroid top_k — no gain;
**provable greedy-safe cert (Cauchy-Schwarz) for sparse lm_head verify on gemma-4-E4B** — model-intrinsic
geometry obstruction, nets −8% TPS; empirical pruned-weights lmhead12k (no cert) is the viable lever;
**fa2sw + onegraph runtime levers (standalone, int4 base, conc=1)** — both greedy-DIVERGENT, no TPS win
(denken PR #7, CLOSED): fa2sw −4.9% TPS + DIVERGENT 82/128; onegraph TPS-parity + DIVERGENT 1/128;
int4 base cross-process **bit-exact** at M=1; fa2sw also requires a vLLM worker-plugin;
**`VLLM_BATCH_INVARIANT=1` kernel override — definitive negative for greedy-valid spec decode at ANY precision
in vLLM 0.22.0** (kanna PR #19, MERGED, 2026-06-13). int4 spec stays DIVERGENT at 0.376%/tok ON vs 0.332%
OFF (CIs overlap; the flag does nothing for int4). bf16 control drops to 0.111%/tok but remains DIVERGENT —
isolating TWO independent un-coverable causes: (a) int4 Marlin is a `_C` op the aten override can't reach
(contributes ~0.265%/tok excess above bf16 floor), (b) the spec verify path has an irreducible non-aten
batch-variant component (~0.111%/tok; corroborated by vLLM issue #27433: "does not currently integrate with
speculative decoding"). Batch-invariance coverage is real (bit-exact kernel probe) but aten-scoped;
both flip sources sit outside aten. This closes the invariant-kernel lane; the next lane is
**verify-rollback** (arxiv 2601.17768, kanna assigned).

_Last updated: 2026-06-13 (**PR #4 MERGED — new best merged rung: int4 g128 + untied int4 lm_head, 126.378 TPS / PPL 2.019 / 128/128 VALID / GREEDY_IDENTICAL, 1.32× over int4 base, 2.87× over bf16. `submissions/int4_g128_lmhead` is now the best merged submission; all future submissions beat 126.38 TPS.** PR #19 MERGED — LINCHPIN DEFINITIVE NEGATIVE: `VLLM_BATCH_INVARIANT=1` cannot rescue greedy-valid spec decode at any precision in vLLM 0.22.0; two independent un-coverable root causes quantified (Marlin _C op + non-aten spec-verify residual); next lane: verify-rollback arxiv 2601.17768, kanna assigned. **Same-path PPL gate (PR #21) scope limit confirmed (wirbel #22, 2026-06-13):** the gate is teacher-forced-blind — it misses argmax-preserving decode-compounding folds (e.g. LF29 affine fold: gate returns gap 0.0000 even when fold-forced-ON, because teacher-forced PPL is fold-neutral). `greedy_gate` (served-token identity) is the load-bearing validity instrument for fold-class lanes.)_

### 2026-06-13 14:38 — PR #4: int4 g128 + untied int4 lm_head (~127 TPS weight-byte floor) ⭐ NEW BEST MERGED RUNG

- **Primary metric (tps):** **126.378** (official a10g-small, job `6a2d5a96234ca64b60121aa5`) — **1.32× over PR #3 int4 base (95.463), 2.87× over bf16 (44.018)**.
- **PPL (gate):** **2.0190** ≤ 2.42 ✓ (1.28 PPL cost over QAT base at 2.006 — negligible).
- **completed:** 128/128 ✓ · **greedy identity:** GREEDY_IDENTICAL 128/128 (served-vs-served cap=512) ✓ · **same-path gate:** SAME_PATH_OK (gap 0.0000) ✓.
- **W&B:** `905tbujn` (official a10g-small) · `0pxj6n63` (local proxy + greedy verdict).
- **Submission:** `submissions/int4_g128_lmhead/` — int4 Marlin W4A16 full-body g128 (vs per-layer in base) + untied int4 lm_head. Checkpoint `google/gemma-4-E4B-it-qat-w4a16-ct` re-quant'd with `build_quant.py`; vLLM 0.22.0; CUDA graphs FULL_AND_PIECEWISE; all modalities loaded.
- **What moved the TPS:** untied int4 lm_head eliminates the bf16 GEMV for 262k-vocab verify (profiler: this was 26.4% of decode GPU time); full-body g128 vs per-layer gives ~1% additional weight-byte reduction. Together: the **int4 Marlin weight-byte floor on Ampere**.
- **Prove-out artifacts:** `research/_probe/` — greedy verdict JSONLs, ppl_served records, tps probe, config comparisons, quantized-modules inventory. `check_greedy_identity.py` + `validate_offline.py` shipped as first-class tools.
- **Significance:** this is the weight-byte floor on Ampere with vLLM 0.22.0. Sub-4-bit and fp8 KV are dead ends (no sm_86 kernel). The next TPS lever is the drafter ladder (spec decode) — gated on the verify-rollback gate (kanna, arxiv 2601.17768).

### 2026-06-13 11:00 — PR #21: Same-path PPL gate: timed-model PPL vs prompt_logprobs path

- **Primary metric (gap):** `|same_path_ppl − prompt_logprobs_ppl|` = **8.88e-16 ≈ 0.0000** (gate: < 0.05; calibration band: < 0.02)
- **Calibrated same-path PPL:** **2.3012128792** (both paths, 61,797 tokens, 128/128 records) — exactly reproduces PR #2 honest baseline
- **Gate verdict:** `SAME_PATH_OK` — `validate_submission --check-same-path` exits 0
- **W&B run:** `b9igh00q` (wandb-applied-ai-team/gemma-challenge-senpai, group `same-path-ppl-gate`)
- **What this adds:** `scripts/local_validation/same_path_ppl.py` scores via the generation path with **no `prompt_logprobs` field** — indistinguishable from timed throughput. A gamed submission (e.g. LF29cap: prompt_logprobs PPL 2.38, generation PPL 2.55, gap ≈ 0.17 >> 0.05 threshold) cannot detect and neutralize the probe. Artifacts at `research/validity/vllm_baseline/`.
- **Scope note:** gate catches request-field branching on `prompt_logprobs`; paired with `greedy_gate` (PR #8) closes the audit-vs-timed blind spot. Does not catch `echo`-branching or prefix-cache replay on prompt content (named residual attack surfaces in `research/validity/same_path_ppl.md`).
- **Every HF-Job approval issue must now attach:** `greedy_gate` result + `--check-same-path` output side-by-side.

### 2026-06-13 ~17:00 — PR #28: Extended verify-latency M-sweep (measured M=1..64, tree ceiling corrected)

- **Primary metric (overhead):** `V_tree(M=25) / V_linear(M=7)` = **1.113×** (was 1.057× extrapolated from PR #26); tree K=6 still strongly net-positive but overhead higher than extrapolated.
- **Test metric (tree ceiling):** K*=12, W=4 tree TPS @ p=0.78 = **452.4** (was 616 extrapolated); **`verdict_exceeds_500_at_full_scale = False`** — the >500 TPS claim from PR #26 extrapolation is refuted on measured data.
- **W&B runs:** `2mk0z0c3` (latency M-sweep, group `spec-verify-msweep`) · `imoi4mx1` (tree acceptance model, group `spec-verify-msweep`)
- **Key finding — latency curve structure:** The int4 verify forward is flat only through **M≈32** (+2.6% vs M=1), then the Marlin int4 weight-GEMM goes compute-bound and ramps super-linearly: M=40 +31%, M=64 +60%. Steps at M≈20, ≈40, ≈64 are tile-boundary quantization effects, not thermal drift. The ramp is GEMM (not lm_head): GEMM share rises 62%→68% through the ramp; attention falls 16%→13%. CUDA-graph mode reveals the ramp (eager hides it under fixed CPU-launch overhead).
- **Tree model corrections (from extrapolated→measured):**
  - K=6 (M=25): 346.8→**331.2 TPS** @ p=0.6792; overhead 1.057×→**1.113×** — still net-positive 1.46×.
  - K*@ p=0.78: K=20 (M=81, extrapolated) → **K=12 (M=49, measured), 452.4 TPS** (vs 616 extrapolated — 27% overstatement).
  - >500 TPS @ p=0.78: only achievable at **p≥0.85** (531 TPS @ K=12) — needs drafter top-1 acceptance ≥0.85, not deeper trees.
- **Strategic implication:** The >500 TPS frontier requires **drafter quality (EAGLE-3 full-scale, fern #25)** at p≥0.85 acceptance, not deeper tree shapes. K*≈8–12 (M=33–49) is the real operating point; deep-K (K≈20, M≈81) is extrapolation territory and regresses on measured hardware.
- **Artifacts:** `research/spec_cost_model/results_msweep.json` (full M=1..64 curve), `tree_results_measured.json` (120-row K×W×p matrix), `tree_plots_measured/`, `report_msweep.md`.

### 2026-06-13 ~17:30 — PR #25: EAGLE-3 full-scale training (drafter asset, reasoning acceptance 0.7314)

- **Primary metric (drafter quality):** `tf_acceptance_rate_math_holdout` = **0.7314** (teacher-forced top-1, reasoning/MATH held-out n=48,142) — up from debug 0.7051 on identical held-out. The benchmark-relevant number (the 128 public prompts are 100% reasoning: mmlu_pro 57 / gpqa_diamond 57 / aime2026 14).
- **Per-source matrix (the core finding):** full model (MATH+ShareGPT, 3500 steps) vs debug (MATH-only, 898 steps): MATH 0.7051→**0.7314** (+0.026), ShareGPT 0.1529→**0.3444** (+0.19), combined 0.5839→0.6464. **ShareGPT did NOT hurt reasoning acceptance** (slightly helped via more steps) and doubled SG acceptance — but chat is intrinsically hard to draft (high-entropy/multilingual/code). Combined 0.6464 understates benchmark-relevant quality.
- **Plateau:** reasoning acceptance plateaus ~0.72–0.73 by step ~2000 (gains <0.004/500 after). Combined val/loss overfits (bottoms 1.8516 @ 2000, rises to 1.9519 @ 3500). **Confirms: reasoning acceptance is DATA-bottlenecked, not step-bottlenecked.** Breaking toward 0.78 needs benchmark-matched reasoning CoT (MMLU-Pro/GPQA/AIME), not more MATH and not chat.
- **W&B (verified):** training `7domtiin` (crashed = external interruption @ step 3670, checkpoint intact); evals `egv59ku0` (full·MATH 0.73136), `xqtvcj58` (full·SG 0.3444), `udb18hnh` (full·combined 0.6464), `y0yupavk` (debug·MATH 0.7051), `yxkh2739` (debug·SG 0.1529), `1j8afmzk` (debug·combined 0.5839). All eval runs finished clean, no NaN.
- **Asset:** `research/eagle3_drafter/checkpoints/full_20k/model_best.pt` (step 3500, 0.7314 reasoning tf_acc) — the **current-best drafter asset**, deploys when kanna's verify-rollback (#24) unlocks serving. Corpus: 2.21M tok (1.76M MATH + 0.45M SG), de-contaminated vs held-out.
- **Caveat for the ladder:** tf_acc is a teacher-forced UPPER BOUND on free-running acceptance. PR #28 says >500 TPS needs top-1 acceptance p≥0.85; 0.73 tf_acc likely maps to lower free-running p. The reasoning-corpus follow-on (fern next PR) + possibly on-policy distillation (Draft-OPD, round-3 H1) are the levers toward 0.85.

### 2026-06-13 ~17:50 — PR #14: Empirical lmhead12k (validated lever + best-LOCAL rung; official a10g-small PENDING)

- **Status:** MERGED as a **validated lever + best-LOCAL rung**, NOT a new official baseline. Per this file's contract the official metric is a10g-small HF-Job TPS and **local A10G numbers are exploratory only**, so the **official baseline headline stays PR #4 (126.378)** until a gated HF job confirms lmhead12k. Asset/code banked; official confirmation queued via approval issue.
- **What it is:** prune the `lm_head` weight matrix to the top-12,288 token rows (bf16, sliced from tied embeddings) → 21.3× fewer head bytes (62.9 MB vs 1342 MB bf16-262k). `submissions/lmhead12k_empirical/` (serve.py + `vllm_lmhead12k` plugin + frozen `kept_ids.json`).
- **Primary metric (local, exploratory):** `tps_local_single_stream` = **131.60**. Clean **single-variable isolated lever = +34.8%** (bf16-262k head 97.65 → bf16-12k head 131.60, only row count differs). Implied lm_head decode fraction **27.1%** independently matches wirbel #8's **26.4%**. Honest **local-to-local net vs PR #4** (int4-262k head, 128.13 local) = **+2.7%** (NOT the +3.6% the student quoted vs official-127 — that mixed local-vs-official).
- **Validity:** greedy gate **GREEDY_IDENTICAL 128/128** served-vs-served spec-off (the documented **self-consistency** gate — clipping cannot fail it: the pruned argmax is always in `kept_ids` by construction); clean **unpruned-int4 control also 128/128** (zero false-divergence). **PPL 1.9712** token-wtd (better than int4-head ~2.02, ≪ 2.42 cap), completed 128/128. No W&B (serve+validate, no training run); fully auditable via 38 committed evidence JSONs under `research/local_validation/lmhead12k_empirical/`.
- **Keeper findings (validity instrument):** (1) the greedy gate is **self-consistency** (served-pruned vs plain-greedy-pruned, same checkpoint), not fidelity-vs-unpruned; (2) earlier 107/128 "control failure" was an **offline-batched-reference vs sequential-candidate FP artifact**, not the prune — *every* future greedy-gate run must use a batch=1 served-vs-served reference; (3) the int4-argmax clip rate has an **irreducible frequency-selection floor** (~0.78% public / 1.15% held-out, uncapturable at any K).
- **Standing risk — private PPL (NOT closable locally):** a private GT-*target* token outside `kept_ids` → −∞ logit → +∞ PPL on a private re-run. Greedy-identity passes private by self-consistency, so this is purely a PPL axis. Mitigated by hard-including all public GT-targets + specials + broad-corpus frequency fill. **Only a gated a10g-small HF job on the private set closes it** → approval issue opened.
- **Next:** ubel → follow-up #3 (int4-pruned head: slice 12k head in int4 ≈ 15.7 MB vs 62.9 MB bf16, another ~4× head-byte cut, orthogonal to kept-set/private-PPL). lmhead12k also compounds in the spec-verify forward (K+1 tok × vocab — larger head fraction), gated on kanna #24.
