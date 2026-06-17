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

## Root cause (found by reading the dev307 build + the #542 base log)
`vllm/model_executor/models/config.py` `Gemma4Config.verify_and_update_config` (dev307,
`vllm-0.22.1rc1.dev307+g3e8afdf78`) **forces TRITON_ATTN for ALL layers** when the model
has heterogeneous head dims (`head_dim=256` sliding vs `global_head_dim=512` full,
`max>256`) **and the user has not explicitly chosen a backend**. The #542 base server log
shows exactly this:

    config.py:100 Gemma4 model has heterogeneous head dimensions
      (head_dim=256, global_head_dim=512). Forcing TRITON_ATTN backend ...
    cuda.py:318  Using AttentionBackendEnum.TRITON_ATTN backend.

TRITON_ATTN's sliding-window path degenerates long-CoT greedy decode on this int4 ckpt →
repetition loops → max_tokens truncation → 0.432. The vLLM rule forces TRITON precisely
because a **global** FlashAttention would *reject* the 512-dim full layers (head_size ≤ 256
kernel limit). base_fullhead's surgical `FA_SLIDING=1` routes only the 256-dim **sliding**
layers to FLASH_ATTN (full layers stay TRITON), sidestepping the regression — its
truncation is 9.6%/17.2% vs the broken base's 36.6%/33.8%.

## Design
Build = `vllm-0.22.1rc1.dev307+g3e8afdf78` for EVERY arm (`/tmp/senpai-venvs/5f4c623f772358a2`).
Stock ckpt = `/tmp/gemma4-e4b-qat-w4a16-ct` (native 262k head, int4 W4A16, `speculative_config=None`).
Client harness = ubel #511 `run_eval.py` at greedy temp=0, `--max-connections 32`, byte-identical
seeded item set (`prompt_sha` asserted vs #542 → `n_prompt_mismatch=0`).

**Stage 1 — root cause (PRIMARY).** Confirm the forced-TRITON cause:
- (banked) TRITON default = #542 broken base (0.432 / 0.3131, 36.6%/33.8% trunc).
- global FLASH_ATTN override (`VLLM_ATTENTION_BACKEND=FLASH_ATTN`) — expect load failure
  (FA rejects the 512-dim full layers), proving a *stock global* override is not the fix.
- healthy config = per-layer FA on the sliding (256) layers (the surgical `FA_SLIDING` path).
`regression_root_caused`, `regression_cause`.

**Stage 2 — recover the denominator (PRIMARY).** On the healthy config, full MMLU-Pro n=500
+ GPQA-Diamond n=198. Truncation must collapse vs the broken base and accuracy recover toward
the anchor. `recovered_vanilla_base_mmlu/gpqa`, `recovered_base_reproduces_anchor` (within CI
of 0.668 / 0.470), `strict_cilb_passes_vs_healthy_fresh_base`, `healthy_fresh_base_recoverable`.
Inspect completions (`failmodes.py`) to confirm the same items the broken serve truncated now
converge — not just the aggregate.

**Stage 3 — surgical-attn-only ablation (SECONDARY).** vanilla base + `FA_SLIDING=1` +
`SURGICAL_ATTN_USE_3D_OFF=1` ONLY (NO MTP, split-KV, onegraph, PLE fold, head prune). Same two
axes. `surgical_attn_only_mmlu/gpqa`, `surgical_attn_alone_fixes_longcot`. Tells the
consolidated verdict whether the quality and speed levers are separable or coupled. The Stage 2
healthy config may coincide with this arm (then quality and the attention lever are coupled).

## Anchors / floors
- ubel #511 documented base: MMLU-Pro **0.668** / GPQA-D **0.470** (the denominator to reproduce).
- #542 base_fullhead: MMLU-Pro 0.636 (95.2% of anchor, z=-1.06, indistinguishable) / GPQA-D 0.4697
  (99.9%); truncation 9.6% / 17.2%.
- Morgan #515 floors: MMLU-Pro ≥ 0.601, GPQA-D ≥ 0.423.
- #542 broken fresh base: MMLU-Pro 0.432 / GPQA-D 0.3131; truncation 36.6% / 33.8%.

`primary_metric = recovered_vanilla_base_mmlu`.
