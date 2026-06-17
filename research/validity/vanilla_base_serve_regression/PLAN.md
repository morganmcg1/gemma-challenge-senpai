# Vanilla-base serve regression — root-cause + recover a healthy quality denominator (PR #557)

**Analysis-only, NO FIRE.** `analysis_only=true`, `official_tps=0`. No HF Job, no
`train.py --launch`, no `/v1/jobs:run`, no submission, no served-file change. Local serve + inference only.
`--wandb_group vanilla-base-serve-regression`.

## The open question (#542 follow-up)
My #542 (`92pcnx6a`) found the **fresh vanilla base on the current `dev307` build is
broken at every batch width**: MMLU-Pro 0.432 / GPQA-D 0.3131, with 36.6% / 33.8% of
completions hitting `max_tokens` in repetition loops (never committing to `ANSWER: X`).
The seqs1==seqs16==seqs32 control proved `degeneration_is_batch_artifact=false`. The
program's quality denominator (Morgan #515 "≥90% of vanilla base") therefore rests on the
**documented ubel #511 anchor (0.668 / 0.470)** that no fresh base reproduces. This card
root-causes that regression and tries to recover a reproducible non-degenerate base.

## Root cause (found by reading the dev307 build + the #542 base log; CONFIRMED empirically)
**It is NOT an attention-backend problem.** Reading the dev307 source
(`vllm-0.22.1rc1.dev307+g3e8afdf78`): `gemma4.py::get_per_layer_inputs` looks up the
Per-Layer Embeddings but **no longer multiplies them by `embed_scale_per_layer` (= sqrt(256)
= 16.0)** at runtime — the "Challenge fast path" removed the multiply and folded it into a
**load-time fold gated behind env `PLE_FOLD_EMBED_SCALE=1`** (`model_loader/utils.py:130`). A
plain vanilla serve never sets that env var, so every decoder layer receives **16× too small**
per-layer embeddings → subtly corrupted long-CoT greedy decode → repetition loops →
max_tokens truncation → MMLU 0.432 / GPQA 0.3131.

**The attention backend is a red herring.** dev307's `config.py:100` forces TRITON_ATTN for the
heterogeneous head dims (`head_dim=256` sliding vs `global_head_dim=512` full) — but EVERY arm
below logs `Forcing TRITON_ATTN` / `Using AttentionBackendEnum.TRITON_ATTN`, so attention is
held constant. The ONLY arm whose server log carries `Folding Gemma4 PLE embed_scale_per_layer`
→ `Folded Gemma4 PLE embed scale 16.0 into weight` is `ple_fold`, and it is the only one that
recovers. The surgical 2D/`FA_SLIDING` attention path (Stage 3) does NOT touch the PLE scale and
stays broken (0.414 / 0.242, trunc 37.6% / 41.9%) — **disproving the attention hypothesis.**

## Design
Build = `vllm-0.22.1rc1.dev307+g3e8afdf78` for EVERY arm (`/tmp/senpai-venvs/5f4c623f772358a2`).
Stock ckpt = `/tmp/gemma4-e4b-qat-w4a16-ct` (native 262k head, int4 W4A16, `speculative_config=None`).
Client harness = ubel #511 `run_eval.py` at greedy temp=0, `--max-connections 32`, byte-identical
seeded item set (`prompt_sha` asserted vs #542 → `n_prompt_mismatch=0`). All arms keep the same
forced-TRITON attention; only the named lever moves.

**Stage 1 — root cause (PRIMARY).** Isolate the PLE embed-scale fold as the cause:
- (banked) `triton_default` = #542 broken base (0.432 / 0.3131, 36.6%/33.8% trunc), NO fold line.
- `surgical_attn` (FA_SLIDING + 2D) — same forced TRITON, NO fold → STILL broken (attention is
  not the cause). Doubles as Stage 3.
- `ple_fold` (PLE_FOLD_EMBED_SCALE=1) — same forced TRITON + the embed-scale fold → recovers.
- `global_fa` control (`VLLM_ATTENTION_BACKEND=FLASH_ATTN`) — expect load failure (FA rejects the
  512-dim full layers), proving a *stock global* attention override is not the fix either.
`regression_root_caused`, `regression_cause` (class = `ple_embed_scale_fold_gated_off_on_vanilla_serve`).

**Stage 2 — recover the denominator (PRIMARY).** The `ple_fold` arm IS the recovery: a plain
vanilla serve (`speculative_config=None`, no MTP/2D/split-KV/onegraph) + ONLY the embed-scale
fold. Full MMLU-Pro n=500 + GPQA-Diamond n=198. Truncation must collapse vs the broken base and
accuracy recover toward the anchor. `recovered_vanilla_base_mmlu/gpqa`,
`recovered_base_reproduces_anchor` (within CI of 0.668 / 0.470),
`strict_cilb_passes_vs_healthy_fresh_base`, `healthy_fresh_base_recoverable`. Inspect completions
(`failmodes.py`) to confirm the same items the broken serve truncated now converge — not just the aggregate.

**Stage 3 — surgical-attn-only ablation (SECONDARY).** vanilla base + `FA_SLIDING=1` +
`SURGICAL_ATTN_USE_3D_OFF=1` ONLY (NO MTP, split-KV, onegraph, PLE fold, head prune). Same two
axes. `surgical_attn_only_mmlu/gpqa`, `surgical_attn_alone_fixes_longcot`. RESULT: broken
(0.414 / 0.242) — the attention lever alone does NOT fix long-CoT; quality and the PLE-fold lever
are the coupled pair, attention is separable and not load-bearing for correctness.

## Anchors / floors
- ubel #511 documented base: MMLU-Pro **0.668** / GPQA-D **0.470** (the denominator to reproduce).
- #542 base_fullhead: MMLU-Pro 0.636 (95.2% of anchor, z=-1.06, indistinguishable) / GPQA-D 0.4697
  (99.9%); truncation 9.6% / 17.2%.
- Morgan #515 floors: MMLU-Pro ≥ 0.601, GPQA-D ≥ 0.423.
- #542 broken fresh base: MMLU-Pro 0.432 / GPQA-D 0.3131; truncation 36.6% / 33.8%.

`primary_metric = recovered_vanilla_base_mmlu`.
