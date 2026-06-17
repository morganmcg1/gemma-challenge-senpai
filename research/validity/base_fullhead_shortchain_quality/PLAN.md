# base_fullhead MMLU-Pro + GPQA-Diamond — tight-CI third leg (PR #542)

**Analysis-only, NO FIRE.** `analysis_only=true`, `official_tps=0`. No HF Job, no
`train.py --launch`, no `/v1/jobs:run`, no submission. Local serve + inference only.

## Question (binary)
Does **base_fullhead** — the full fast stack (surgical 2D attn + MTP K=7 + split-KV
+ onegraph + PLE fold) served on the **stock** `google/gemma-4-E4B-it-qat-w4a16-ct`
int4 checkpoint with the **native 262k BF16 tied head** (no osoi5 bake, no head
prune) — clear the **≥90%-of-base** quality bar on the tight-CI short-chain axes
MMLU-Pro + GPQA-Diamond?

fern #535 (`whh42dgd`) proved this checkpoint serves: `serve_ok=TRUE`, PPL 2.006
byte-exact, 253.78 TPS local. This cell decides only whether it is *quality-eligible*.

## Design — clean 2x2, conc=32, identical item set
`{MMLU-Pro, GPQA-Diamond} x {plain base, base_fullhead}`

- **base_fullhead** = `submissions/fa2sw_strict_surgical357` served via its manifest
  recipe with 5 overrides repointing it off the osoi5 substrate onto the stock ckpt:
  - `LOCAL_MODEL_DIR` = `/tmp/gemma4-e4b-qat-w4a16-ct` (local snapshot of the stock HF ckpt)
  - `PLE_FOLD_TARGET_MODEL` = same snapshot path
  - `LM_HEAD_PRUNE` = `0`, `LM_HEAD_PRUNE_REQUIRE` = `0`, `PCK04_KEEPSET` = `""`
  - hard-reject guard: `raise` if `lm_head` rows < 262144 (no silent 16k fallback).
  - served at its native `MAX_NUM_SEQS=1` (interactivity recipe; MTP K7 → ~357 TPS single-stream).
- **plain base** = the SAME stock ckpt served vanilla vLLM (no fast stack), greedy
  temp=0, `VLLM_USE_FLASHINFER_SAMPLER=0`. The only moved variable vs base_fullhead
  is the fast kernels.
- **Pin concurrency:** all four cells driven at client `--max-connections 32`
  (the served protocol). Load-bearing validity rule from fern #535 (the n=30 AIME
  maj@1 was concurrency-confounded). Pin removes that confound.
- **Identical item set:** byte-identical prompts (same seed, order, few-shot) for
  base vs base_fullhead on each axis — verified by per-question `prompt_sha`.
- **Sizes:** MMLU-Pro n=500 (matches ubel #511 banked anchor), GPQA-Diamond full ~198.

## Harness (reuse, do NOT re-implement)
`research/validity/downstream_quality_eval/{run_eval.py,start_server.sh,aggregate.py}`
— the exact ubel #511/#527 inspect_evals greedy harness that produced the banked
base anchors (MMLU-Pro 0.668, GPQA-D 0.470/0.444) and the ship collapse (0.274/0.232).

## Verdict
Per axis: accuracy(base), accuracy(base_fullhead), **Wilson 95% CIs**, ratio
`base_fullhead/base`, and **does base_fullhead's Wilson CI lower bound clear
0.90 x (this run's freshly-measured base on the identical set)?** Reference gate
floors: MMLU-Pro ≥ 0.601, GPQA-D ≥ 0.423. Top-line
`base_fullhead_shortchain_quality_safe` = both axes clear.

## Anchors
- base_fullhead must NOT reproduce the live-ship collapse (osoi5-12k, ubel #511/#527):
  MMLU-Pro 0.668→0.274, GPQA-D 0.470→0.232 (both far below floor).
- Head-route is dead (stark #536 `j3gjxxts`): transplanting the 262k head onto the
  osoi5 body recovered 0.0% → `collapse_locus=BODY`. base_fullhead fixes the body
  (clean base-int4), so it is the candidate that *should* hold quality.

`--wandb_group base-fullhead-shortchain-quality`.
