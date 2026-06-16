<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# TensorRT-LLM engine feasibility for `google/gemma-4-E4B-it` on A10G (sm_86)

**PR:** #502 · **Author:** fern · **Generated:** 2026-06-16T15:47:36.322448+00:00 · **W&B group:** `trtllm-engine-benchmark`

**LOCAL diagnostic. 0 GPU forward, 0 TPS, NO engine build completed, NO HF job, NO submission, NO served-file change.**

Reproduce: `cd target/ && .venv/bin/python research/trtllm_engine_benchmark/probe_trtllm_feasibility.py --self-test`

---

## Verdict: STRUCTURALLY BLOCKED -- engine does not build; M=1 TPS and M=1-vs-M=8 identity are not measurable

`feasibility_evidence_complete = 1` · `trtllm_build_succeeded = 0` · `trtllm_loads_checkpoint = 0` · TRT-LLM importable in serving env: `False`

## Ranked blockers (each LIVE-confirmed on the pod)

| rank | blocker | confirmed | evidence |
|---|---|---|---|
| 1 | B1_transformers_version_skew | YES | TRT-LLM 1.2.1 pins transformers==4.57.3 < gemma4-min 5.5.0; load under pinned -> ValueError (model type gemma4 unrecognized). |
| 2 | B2_head_dim_512_on_ampere | YES | full-attention global_head_dim=512 > Ampere FMHA cap 256; sliding head_dim=256 (mixed per-layer head_dim, no TRT-LLM path). |
| 3 | B3_ple_kvshare_perlayer_rope | YES | PLE vocab_per_layer=262144, num_kv_shared_layers=18, mixed_head_dim=True; not expressible in TRT-LLM decoder def. |
| 4 | B4_multimodal_contract | YES | serving contract forbids dropping modalities; TRT-LLM has no gemma4_audio (conformer/USM) path -> text-only engine non-compliant. |

### B1 reproduction (the first-impact blocker)

- Load source: `live_isolated_venv` (isolated transformers `4.57.3`).
- Loaded under pinned transformers: `False`.
- Error: `The checkpoint you are trying to load has model type `gemma4` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.`
- Upstream: NVIDIA/TensorRT-LLM#12764 (closed-unresolved (version skew)), TRT-LLM 1.2.0 -- same Failure B; Failure C is the catch-22 (`cannot import name AutoModelForVision2Seq` when upgrading bundled transformers).

## Architecture (live, current env)

- `model_type=gemma4` arch=`['Gemma4ForConditionalGeneration']`, env transformers `5.12.0`.
- text layers 42: 35 sliding(512) + 7 full (idxs [5, 11, 17, 23, 29, 35, 41]).
- head_dim 256 (sliding) / 512 (full) -> mixed_head_dim=True; Ampere FMHA cap 256.
- PLE hidden_per_layer=256, vocab_per_layer=262144; num_kv_shared_layers=18.
- multimodal: audio_config=True, vision_config=True.

## Counterfactual lane-closers (hold even though the build is blocked)

- **Determinism:** TRT-LLM deterministic mode = run-to-run reproducibility, NOT batch-size invariance (M=1 vs M=8). Byte identity across batch size is a strictly different, more expensive contract (per-token isolated accumulation). (`batch_invariant=False`; arXiv:2601.17768 'Enabling Determinism in LLM Inference'; TRT-LLM deterministic-reductions docs (run-to-run only).). So a clean TRT-LLM engine would NOT give M=1-vs-M=8 byte identity 'for free' -- same conclusion class as SGLang (denken #498).
- **Spec-dec:** EAGLE/Medusa/ReDrafter/MTP-draft require a working TRT-LLM model definition for the base model (blocked by B1/B3). Only Lookahead/NGram are model-def-agnostic, and neither matches the deployed MTP K=7 lane. (nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html).

## Honesty note

TPS, PPL, and the M=1-vs-M=8 identity census are reported as 0/NULL because the engine never builds -- B1 stops it before the build stage, so there is nothing to benchmark. This is a real, bankable NEGATIVE: it closes the TRT-LLM lane of the #481 engine-shopping zoom-out, complementing denken #498's SGLang close. The two counterfactual findings (determinism, spec-dec) mean even a hypothetical build would not have delivered the free byte-exact identity the lane was probing for.

## Public evidence used

- **denken #498** (`djwaqs7o`) -- SGLang/Flashinfer fast-but-NOT-byte-exact; the engine-independent -107 attention tax this card extends to a second alternative engine.
- **NVIDIA/TensorRT-LLM#12764** -- gemma4 runtime load failure (version skew), reproduced here.
- **Deployed vLLM baseline** -- 481.53 TPS reference ceiling / 399.75 byte-exact rung (PR #502 body); TRT-LLM cannot reach the start line to challenge either.
