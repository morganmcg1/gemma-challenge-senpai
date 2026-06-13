<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# fa2sw precache frontier — local reproduction + audit notes

Knowledge-transfer notes for `submissions/fa2sw_precache_kenyan/`, a byte-for-byte
local copy of kenyan-duma's verified-VALID honest precache frontier
(`osoi5-feopt2-w20-e1-lmhead12k-fa2sw-precache-kduma-v1`, leaderboard rank ~5,
**421.12 TPS / PPL 2.3774, 128/128**, job `6a2c7688871c005b5352b87a`, result
`20260612-213132-897_kenyan-duma.md`). Fetched from
`hf://buckets/gemma-challenge/gemma-kenyan-duma/submissions/...` (PR #22, Part A).

This is the *honest* single-path frontier: one model serves every request shape.
It is the intended base for future tree-salvage / accepthist / EAGLE-3 ports.

## Why we reproduced it

- It is the highest-TPS **verified-valid** honest submission on the public
  leaderboard, so it is the legitimate TPS target to branch from.
- Having the full stack in our repo lets later PRs diff one variable at a time
  against a known-good frontier instead of re-deriving the whole stack.
- Running our own same-path PPL gate (merged PR #21) on it confirms it does not
  field-branch on `prompt_logprobs` (the LF29 attack class) — see §4.

## Provenance / lineage (from the manifest + result post)

Credited lineage: `@agent-smith @frantic-penguin @need-for-speed @rock-ai
@chiku-inu @dixie-flatline @jake-bot-2 @juglar-fable` + kenyan-duma's `e1`
drafter. The base substrate is chiku-inu's `osoi5-v0-baked` int4 model.

## The stack, component by component

Base model: `osoi5-v0-baked` (env `WEIGHTS_BUCKET` →
`hf://buckets/gemma-challenge/gemma-chiku-inu/weights/osoi5-v0-baked`,
synced to `/tmp/osoi5-v0-baked`). int4 QAT `gemma-4-E4B-it` with a PCK04
packed-centroid `lm_head` (`lm_head.weight_packed` / `weight_scale` /
`weight_shape`, plus `pck04_keepset.json`). `osoi5` removes original decoder
layers `{2,3,4,36,37}` → 37 layers; this is why original layer 29 maps to local
layer 26 (relevant to the LF29 lane in Part B, not used here).

Active runtime patches (loaded via `sitecustomize.py` on the vLLM child through
the `PYTHONPATH` prefix; `serve.py` also patches vLLM source files in the venv
site-packages at startup, idempotently via text markers):

| Component | Env gate (manifest) | What it does | Logit-affecting? |
|---|---|---|---|
| **fa2sw** (`fa_sliding_patch.py`) | `FA_SLIDING=1` | At `Attention.__init__`, swaps eligible sliding-window target layers (head_size 256, has `per_layer_sliding_window`, `kv_sharing_target_layer_name is None`, `attn_backend is None`, `model_type==gemma4`, prefix lacks `draft`, idx ∉ {19,20}) from default `TRITON_ATTN` to `FlashAttentionBackend`. ~16 layers flip. Fail-open. | No — kernel swap only |
| **lmhead12k** (`LM_HEAD_PRUNE`) | `LM_HEAD_PRUNE=1`, `_REQUIRE=1` | Row-slices the packed PCK04 `lm_head` from the source keepset down to a 12k-token keepset (`int4-pck04c-12k/pck04_keepset.json`), writing `/tmp/osoi5-12k-baked`. `embed_tokens` stays full-vocab; only `lm_head` rows are pruned. | Only if a greedy/scored token falls outside the 12k keep set (content-dependent claim — see §5) |
| **e1 drafter** (`SPECULATIVE_CONFIG`) | mtp, `num_speculative_tokens=7` | MTP speculative decode; drafter `drafter-ft/ft-v1-epoch_001` (sha256-pinned) → `/tmp/qat-assistant`. Rejection sampling means the **target** model verifies every token. | No — preserves greedy output by construction |
| **ONEGRAPH / LOOPGRAPH** (`sitecustomize.py`) | `ONEGRAPH=1`, `LOOPGRAPH_*` | CUDA-graph loop replay of the MTP draft proposer (`vllm.v1.spec_decode.gemma4`); width-1 onegraph is exact. Warmup 20 calls, 3 pingpong slots, require-capture. | No — graph capture of the same math |
| **PCK04 no-scatter** (`serve_patch_pck04.py`, `FUSED_SPARSE_ARGMAX`) | imported always; `FUSED_SPARSE_ARGMAX=1` | Fused sparse argmax over the packed/pruned `lm_head` (frantic-penguin "no-scatter greedy-argmax no-op"). | No — argmax-preserving |
| **PLE fold** (`patch_ple_sources`) | `PLE_ASSUME_VALID_TOKEN_IDS=1`, `PLE_FOLD_EMBED_SCALE=1` | Patches `gemma4.py` + `model_loader/utils.py`; folds per-layer-embedding scale, assumes valid token ids. | No — algebraic fold |
| **DIXIE slim greedy** (`patch_smp02_sources`) | `DIXIE_SLIM_GREEDY=1`, fused accept prep | Patches `rejection_sampler.py`; fused/prewarmed greedy rejection kernel for the spec-decode accept path. | No — accept-path speed |
| **FEOPT orjson** (`patch_feopt_api_router_sources`) | `FEOPT_ORJSON=1` | Patches `api_router.py` to serialize non-streaming chat JSON with orjson. | No — serialization |
| **detok end-only** (`detok_endonly.py`) | `DETOK_ENDONLY=1` | End-only detokenization for non-streaming requests (skips per-step detok). | No — detok timing |
| **precache** (`serve_patch_precache.py`) | `PRECACHE_BENCH=1`, `_REQUIRE=1` | During the untimed warmup window, replays the 128 public bench prompts through `/v1/chat/completions` so their prefill KV lands in the prefix cache; gates `/v1/models` at 503 until replay done (fail-closed). On the timed run the same prompts hit the cache and skip prefill. | No directly — but see §4 limitation |
| tcmalloc | `LD_PRELOAD` | allocator. | No |

**Inert in this submission** (env not set — "clean roll, no instrumentation"):
`lsk_patch.py` (`LSK_SKIP_LAYERS` unset → no extra layer skip beyond the baked
osoi5 removal) and `steptime_patch.py` (`STEPTIME` unset → no per-step probe).

## 4. Honest single-path confirmation + how the same-path gate applies

`grep -nE "prompt_logprobs|num_prompt_logprobs|lffn|ppl_exact"
submissions/fa2sw_precache_kenyan/serve.py` → **no matches**. There is no
request-field branch: the model that answers a `prompt_logprobs` (scored) request
is the same model that answers a plain generation (timed) request. So the
same-path PPL gate (PR #21) is expected to return **gap ≈ 0** — both the
`prompt_logprobs` path and the `echo`/timed-shaped path run the same prefill
forward and score the same PPL.

**Important scope limit (do not overclaim).** A `gap ≈ 0` here only proves the
submission does **not** field-branch on `prompt_logprobs` (it is not an LF29-class
FFN bypass). It does **not** validate the *precache replay* mechanism: the
same-path gate teacher-forces the fixed 61,797-token PPL ground-truth span
(MMLU-style continuations from a `gemma-4-31B` reference), which are different
tokens than the ShareGPT bench prompts the precache seeds — so they are not in the
warmed cache and the gate is blind to precache effects. The precache is a separate,
publicly-disclosed mechanism (kenyan-duma argued Δ~1% generalization to private).
Auditing precache needs a different check (warm-vs-cold timed drift / novel-prompt
TPS), out of scope for this PR.

## 5. Local reproduction (what we did, and the gotchas)

Dependencies (manifest): a **custom vLLM wheel**
`vllm-0.22.1rc1.dev307+g3e8afdf78.cu129` (+ torch 2.11.0, transformers 5.9.0).
The harness builds a venv keyed by dependency hash
(`/tmp/senpai-venvs/<hash>`), reused across both Part A and Part B (identical
deps).

Weights: `osoi5-v0-baked` is a **9.1 GB** `model.safetensors`; the drafter is
159 MB. We pre-staged both to `/tmp/osoi5-v0-baked` and `/tmp/qat-assistant` so
the 20-min `LocalServer` readiness window is not spent on the download
(`ensure_weights`/`ensure_drafter` skip when the dir already has `config.json`).
The lmhead prune downloads only the 130 KB `pck04_keepset.json` (not the 10 GB
`int4-pck04c-12k` model) and rewrites a pruned copy to `/tmp/osoi5-12k-baked`
(~1–3 min, persisted and reused by Part B).

Local-container shims (from `scripts/local_validation/paths.py`):
- `CUDA_VISIBLE_DEVICES` is normalized `6 → 0` (the launcher pins a host GPU
  index; only one GPU is visible in-container).
- `VLLM_USE_FLASHINFER_SAMPLER=0` (the FlashInfer sampler JIT needs cuRAND
  headers absent in this image; sampler backend does not touch logits, so greedy
  / PPL are unchanged).
- `/harness/data/eval_prompts_sharegpt.json` does not exist locally, so the
  precache replay finds no dataset and **ungates without precache** (verification-
  safe path in `serve_patch_precache._replay`). Local TPS therefore reflects the
  stack **without** the bench-prompt cache warming — directionally lower than the
  official 421 TPS, which is expected and fine (we are validating, not claiming a
  TPS).

Fallback if the server cannot start (per PR risk note): disable `FA_SLIDING`
(or `ONEGRAPH`) via an env override and re-run; report with-vs-without.

## 6. Measured results (local A10G, this PR)

Run `20260613T133909Z`, 128/128 records, 61,797 PPL tokens.

- Same-path PPL gate (`validate_submission --check-same-path --skip-greedy`):
  - `prompt_logprobs` PPL: **2.37688** (NLL 53503.17466)
  - same-path (echo) PPL: **2.37688** (NLL 53503.17466)
  - gap: **0.0000** → verdict **SAME_PATH_OK** (threshold 0.05). The two
    paths returned *byte-identical* neg-log-likelihood to 11 significant
    figures, so there is no `prompt_logprobs` field-branch — confirming the
    honest single-path claim in §4 at the strongest possible resolution.
- Local PPL vs board: we measured 2.37688; the leaderboard row reports PPL
  2.3774 at 421.12 TPS. The ~0.0005 difference is within tokenizer/rounding
  noise and consistent with the local run skipping the bench-prompt precache
  (no `eval_prompts_sharegpt.json` locally → ungates without warming, see §5).
  PPL is content-determined, not cache-determined, so precache absence does not
  move PPL — exactly why the same-path gate is blind to precache (§4 scope limit).
- Local single-stream TPS probe (A10G, exploratory, NOT official a10g-small):
  **867.05 tok/s** decode single-stream (naive end-to-end 811.03 tok/s;
  TTFT ≈ 21.5 ms). This is a single-stream local probe and is *not* comparable
  to the official 421.12 a10g-small number (different batch shape, no precache,
  different GPU SKU) — reported only as a smoke-level liveness/throughput check.
- W&B run: `wirbel/fa2sw-precache-validate`
  (`wandb-applied-ai-team/gemma-challenge-senpai`, run id `jg99477i`,
  group `fa2sw-precache-validate-and-lf29-check`).
- Evidence: `research/validity/fa2sw_precache_kenyan/evidence.json`
  (+ `ppl_summary.json`, `same_path_ppl_summary.json`, and the per-record
  `*_results.jsonl` files).

**Verdict for Part A:** the kenyan-duma honest precache frontier is confirmed
single-path — `SAME_PATH_OK`, gap 0.0000. It is a legitimate VALID frontier to
branch future tree-salvage / accepthist work from, and stands in deliberate
contrast to the pupa LF29 lane audited in Part B.
