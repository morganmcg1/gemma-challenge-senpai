<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Framework ZOOM-OUT: are the fixed-overhead floor + head byte-rate wall vLLM-specific or framework-universal? (`google/gemma-4-E4B-it`, A10G sm_86)

**PR:** #558 · **Author:** lawine · **Generated:** 2026-06-17T04:24:24.245402+00:00 · **W&B group:** `framework-zoomout-ceiling`

**LOCAL diagnostic. NO GPU model forward, NO serve, NO TPS on official prompts. NO HF job, NO submission, NO served-file change. `analysis_only=true`, `official_tps=0`.**

Reproduce: `cd target/ && .venv/bin/python research/framework_zoomout_ceiling/probe_framework_feasibility.py --self-test`

---

## Verdict: framework-robust-NO-FIRE — no alternate framework serves THIS w4a16/sm_86 checkpoint byte-identically, so neither ceiling term is framework-movable

`feasibility_evidence_complete = 1` · `framework_serves_byte_identical = 0` · `framework_moves_ceiling = 0` · `alt_framework_quality_safe_tps = 311.25` (= corrected #554 ceiling, unchanged)

## KEY OUTPUTS

| stage | metric | value |
|---|---|---|
| 1 | `framework_serves_byte_identical` | **0** (no) |
| 1 | `framework_tried` | SGLang (primary; FlashInfer/TRT-LLM fallbacks already closed #507/#502) |
| 1 | `framework_infeasible_reason` | `install-too-big-for-disk+downgrades-serve-torch+attention-identity-flip` |
| 1 | `argmax_identity_rate` | n/a — never served (standup install-blocked in box); predicted-flip per #507/#498 |
| 2 | `alt_framework_head_bw_GBs` | n/a — gated (no alt framework served) |
| 2 | `alt_framework_fixed_overhead_ms` | n/a — gated |
| 2 | `framework_moves_ceiling` | **0** (no) |
| 2 | `alt_framework_quality_safe_tps` | **311.25** (corrected #554 ceiling) |
| 3 | `lever_portable_to_vllm` | n/a — no non-vLLM lever found |
| 3 | verdict | **framework-robust-NO-FIRE** |

## Stage 1 — install-feasibility (this card, live 2026-06-17)

The most viable candidate (SGLang) cannot stand up in the time-box without breaking the serving stack or the disk:

| fact | value |
|---|---|
| pod serving torch (.venv) | `2.11.0+cu130` (vLLM `0.22.0`) |
| SGLang version resolved | `0.5.9` |
| → resolved torch (downgrade) | `2.9.1` (DOWNGRADES off the serve env) |
| → resolved sgl-kernel | `0.3.21` |
| → resolved attention backend | `flashinfer-python 0.6.3` (the family fern #507 measured to FLIP greedy identity on this checkpoint) |
| packages it would install | 164 |
| heavy-wheel download | **4.21 GB** (compressed; unpacked ≈1.7–2×) |
| free disk on pod | **3.33 GB** |
| install fits free disk? | **False** |

So a standup needs its OWN multi-GB isolated env (it downgrades torch 2.11.0+cu130 → 2.9.1 and pulls the full nvidia-cu12 stack), which does not fit the pod's free disk. This re-confirms denken #498's pod-uninstallable finding under the current env — without fighting the install past the box.

## Stage 1 — even with infinite disk, no identity-preserving attention path

SGLang's resolved decode attention is **FlashInfer** (`flashinfer-python 0.6.3`). On this checkpoint that backend is the exact one fern #507 measured to be **batch-variant / identity-flipping** (default split-KV: M=1-vs-M=8 byte identity 0.000). SGLang's **Triton** fallback does **not** support sliding-window attention (SGLang issue #6161, open). vLLM is byte-exact here precisely because it **forces** a Triton-with-sliding-window path for the heterogeneous head_dim (256 local / 512 global) — a path SGLang lacks. So even a hypothetical successful install would fail the #319 greedy-identity gate on the attention reduction order.

## Stage 2 — why neither ceiling term is framework-movable (cited, NOT re-derived)

- **Head byte-rate wall = 482.9 GB/s (80.5% of the 600.0 GB/s A10G HBM peak; denken #550).** Marlin is the ONLY w4a16 kernel on sm_86 (#550). The one alternate-framework GEMV that actually ran on the board — the public **llama.cpp** taskforce — was measured to **lose to Marlin at the M=8 verify shape** (@dixie-flatline). The wall is HBM bandwidth realized by the best-available kernel; no framework beats Marlin for w4a16 on Ampere, so none moves this term favorably.
- **Fixed-overhead floor = 0.573 ms (42 sequential SDPA launches; my #554).** denken #498 measured the 2D-attention tax to be **engine-independent** (the −107 tax bites in deployment regardless of engine). The 42-launch count is the heterogeneous-head-dim per-layer dispatch any engine must issue; vLLM already CUDA-graph-captures the propose/step loop (ONEGRAPH, +23% already in the deployed number). A different framework faces the same per-layer launch structure on the same hardware.

## On-branch corroboration (the framework wild-card, progressively closed)

| framework | PR | result |
|---|---|---|
| SGLang | #498 (`djwaqs7o`) | uninstallable on pod (`sglang_decode_tps`=None); FlashInfer-proxy census batch-VARIANT (identity 0.000); attention tax engine-independent |
| TensorRT-LLM | #502 | structurally blocked — engine never builds (build_succeeded=None, loads_checkpoint=None) |
| FlashInfer (standalone) | #507 | loads=1 but NOT free byte-exact (default_m_invariant=0); fixed_split costs 1.2–4.7× M=1; hd512 no path |
| llama.cpp (public taskforce) | — | 97.76 TPS, GGUF q4_0 (PPL 1.982) — a DIFFERENT quant, NOT byte-identical to the served w4a16 reference; GEMV loses to Marlin |

All four agree: on A10G sm_86, **vLLM + forced-Triton is the only stack that serves THIS w4a16 heterogeneous-head-dim checkpoint byte-identically.** The two ceiling terms are walls of the HARDWARE (HBM bandwidth + the per-layer launch structure of the het head_dim), not walls of vLLM. Morgan #481's framework wild-card closes from a sixth, orthogonal angle.

## Honesty / scope note

This card's Stage-1 answer (no alt framework serves byte-identically) was **already established on-branch** by denken #498 (SGLang) / fern #502 (TRT-LLM) / fern #507 (FlashInfer) under the **M-invariance / equivalence-frontier** lens. PR #558's framing that the slot is 'UNFILLED for cycles' is, strictly, inconsistent with that record. The genuinely **new** contribution here is (1) a fresh current-env install-feasibility measurement that re-confirms SGLang is uninstallable on the pod *today* (torch 2.11+cu130 → 2.9.1+cu12 downgrade + multi-GB env that does not fit free disk), and (2) re-framing the framework question against the **ceiling terms priced after those probes** (#554's 0.573 ms floor, #550's 482.9 GB/s wall) for the **base_fullhead quality-safe ship** — concluding both terms are framework-robust HARDWARE walls. No GPU forward was run because Stage 1 gates Stage 2 and the advisor's instruction is explicit: do not fight installation past the time-box.

## Public evidence used

- **llama.cpp taskforce** (`taskforces/llama-cpp/README.md`) — the only non-vLLM framework on the board: `llamacpp-inproc-v0` = 97.76 TPS / PPL 1.982, 128/128 VALID, GGUF q4_0 (a different quant → not byte-identical); @dixie-flatline's finding that llama.cpp-class GEMV kernels lose to Marlin at the M=8 verify shape.
- **Leaderboard digest** (`/v1/digest?as=senpai`, 2026-06-17) — top rows (508.6 / 505.9 / 489.6 …) are all vLLM-derived split-KV / fa_window stacks; zero alternate-framework entries above the llama.cpp 97.76 floor.
- **On-branch:** SGLang #498 (`djwaqs7o`), TRT-LLM #502 (`sxi590tz`), FlashInfer #507.
- **Cited ceilings (not re-derived):** #554 (`fi8vr1nb`) 0.573 ms floor / 311.25 corrected ceiling; #550 (`5aobahij`) 482.9 GB/s head wall / Marlin-only-on-sm_86; #507 FlashInfer identity-flip prior; Morgan #481 ZOOM-OUT directive.
