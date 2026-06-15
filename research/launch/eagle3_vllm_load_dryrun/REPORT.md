<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Static vLLM load dry-run — which of #328's 4 residual caveats close at 0 GPU?

**PR:** #338 · **Author:** ubel · **Issue:** #319 · **Generated:** 2026-06-15T10:23:06.957269+00:00
· **W&B group:** `eagle3-load-dryrun`

**0-GPU static load-readiness audit. NO model forward, NO checkpoint load, NO publish,**
**NO bucket write, NO manifest/served-file change, NO HF job, NO submission, NO GPU.**

Reproduce: `cd target/ && .venv/bin/python research/launch/eagle3_vllm_load_dryrun/eagle3_vllm_load_dryrun.py --self-test`

---

## Verdict: 🟢 GREEN — C1/C3/C4 CLOSED at 0 GPU; only C2 (GPU numerics) remains

`vllm_load_dryrun_self_test_passes = 1` · `caveats_closed_at_0gpu = 3` (of 4; C2 is GPU-only by construction)

Audited fork: **vLLM `0.22.1rc1.dev307+g3e8afdf78`** (`Eagle3LlamaForCausalLM` in `model_executor/models/llama_eagle3.py`), located via `filesystem`. This is the
same fork that serves the deployed frontier; the candidate must load under it. The audit
reads the **real fork source** (registry dict, `stacked_params_mapping`, the `d2t` logic) as
data — it does not construct the model or run a forward.

## #328 caveat ledger (deliverable 5)

| # | caveat | status | evidence |
|---|---|---|---|
| C1 | config-field survival + class registration | ✅ CLOSED-AT-0-GPU | registry resolution + 1:1 param-manifest map + AutoConfig field survival |
| C2 | fp32->bf16 inference numerics + live served greedy-identity | 🖥️ REQUIRES-GPU-SMOKE | irreducible: needs a live GPU forward + served greedy-token-identity (#192 HARD gate); a CPU static audit cannot exercise it |
| C3 | absent-d2t -> identity-map default | ✅ CLOSED-AT-0-GPU | zero-init + skip-on-absent + identity scatter, draft_vocab==target_vocab |
| C4 | vLLM-fork version / schema pin | ✅ CLOSED-AT-0-GPU | fork _version.py pin + config schema parse + arch resolution |

**Residual set the human's §4 A10G smoke must still cover:**
- C2: fp32->bf16 inference numerics + live served greedy-identity

C2 stays residual by construction: the fp32→bf16 forward numerics and the served
greedy-token-identity contract (#192 HARD gate) cannot be exercised without a live GPU
forward. Everything else needed to *load* the head is verified statically below.

---

## Step 1 — registration / class resolution (C1)

- `architectures[0]` = `Eagle3LlamaForCausalLM` resolves through the fork's
  `_SPECULATIVE_DECODING_MODELS` to `['llama_eagle3', 'Eagle3LlamaForCausalLM']` (`registry_resolves=True`).
- Class defined in fork source: `True`; 7 registry aliases route to it.

## Step 2 — param-manifest 1:1 (C1)

- Real fork `stacked_params_mapping` == #333 converter port: `True`.
- `15`/`15` published keys map 1:1 onto `15` vLLM load-targets (`0` unexpected, `0` shape-mismatch, `0` missing).
- Dry-run input: `ondisk_#333_candidate_header` (matches analytic manifest: `True`).

## Step 3 — absent-`d2t` → identity (C3)

- zero-init default: `llama_eagle3.py:292 (Eagle3LlamaForCausalLM.__init__)`
- skip-when-absent: `llama_eagle3.py:400 (Eagle3LlamaForCausalLM.load_weights)`
- identity scatter: `llama_eagle3.py:354 (Eagle3LlamaForCausalLM.compute_logits)`
- `draft_vocab == target_vocab == 262144` → the identity scatter is
  full-coverage (no silent token remap). Absent `d2t` is therefore safe, not a blocker.

## Step 4 — version / schema pin (C4)

- Fork version (from `_version.py`): `0.22.1rc1.dev307+g3e8afdf78` (token `0.22.1rc1` present: `True`).
- Converted config schema parses via `stub` and `Eagle3LlamaForCausalLM` resolves under this fork.
- Fork-required config fields not emitted by the converter: `[]` (empty = none).

---

## Honesty note

This is a STATIC load-readiness audit, not a runtime check. It does not touch emission,
PPL, or served greedy-identity (those are the human's §4 GPU smoke = C2). It validates
against the real installed fork source by reading its load contract as data; it never
constructs the model, loads a checkpoint, or runs a forward. Publishing the artifact and
running the §4 smoke stay HUMAN-owned.

## Public evidence used

- **ubel #328 / `eagle3_ckpt_publish_readiness/REPORT.md` (`27y5xxce`)** — the §5 C1-C4
  'verify-at-smoke' caveats this card statically closes; this card operationalizes that §5.
- **ubel #333 / `eagle3_safetensors_converter` (`quzi85y0`)** — the converter whose published
  manifest + config are reused here as the dry-run input (single source of truth).
- **vLLM `0.22.1rc1.dev307+g3e8afdf78` `llama_eagle3.py`** — the `Eagle3LlamaForCausalLM` / `LlamaModel` load
  contract read as the authoritative target (registry, `stacked_params_mapping`, `d2t` logic).
