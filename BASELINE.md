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
- `submissions/vllm_baseline` — bf16 stock vLLM 0.22.0 endpoint. Prior HF smoke job
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
geometry obstruction (flat row norms, near-full-rank embedding: R_complement_max_norm=1.630 >> z_max/||h||≈0.59),
0%-fire on 16,384 real decode steps, nets −8% TPS (cert + full fallback > full alone); harness on
`ubel/vocab-prune-sparse-verify` branch; empirical pruned-weights lmhead12k (no cert) is the viable lever;
**fa2sw + onegraph runtime levers (standalone, int4 base, conc=1)** — both greedy-DIVERGENT, no TPS win
(denken PR #7, CLOSED): fa2sw −4.9% TPS + DIVERGENT 82/128 (FA2 numerics ≠ Triton → near-tie argmax flips;
mixed backend blocks full-graph capture); onegraph TPS-parity + DIVERGENT 1/128 (graph-capture knob perturbs
the numeric path, one near-tie flip); mechanism: ~92% weight-GEMM/BW-bound at conc=1, existing CUDA graph
already collapses the step — no per-step overhead to reclaim standalone; int4 base is cross-process
**bit-exact** (sha256 run#1==run#2, eager too); fa2sw also requires a **vLLM worker-plugin** (V1 spawns
a separate EngineCore process that a serve-process monkeypatch can't reach).

_Last updated: 2026-06-13 (PR #8 MERGED — local validation + profiling infra: served-vs-served greedy gate, validate_submission harness, lm_head=26% profiler confirmed; PR #15 EAGLE-3 ACCESSIBLE/GO; PR #7 CLOSED fa2sw/onegraph NEGATIVE — both greedy-DIVERGENT standalone on int4 base at conc=1; int4 base cross-process bit-exact in M=1 sequential regime. Linchpin: int4 batched-verify spec-decode structurally greedy-DIVERGENT in vLLM 0.22.0 — kanna #19 resolving via batch-invariant vLLM, gates rungs 4–5)._
