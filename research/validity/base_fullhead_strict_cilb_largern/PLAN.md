# base_fullhead quality under the strict Wilson CI-lb lens at larger n (PR #564, #557 follow-up)

**Analysis-only, NO FIRE.** `analysis_only=true`, `official_tps=0`. No HF Job, no
`train.py --launch`, no `/v1/jobs:run`, no submission, no served-file change. Local serve +
inference only. `--wandb_group base-fullhead-strict-cilb-largern`. Engine = dev307 ONLY.

## The question this settles (#557 follow-up)
My #557 (`yw6vwk1w`) recovered a healthy quality denominator (ple_fold vanilla base:
MMLU-Pro **0.662** / GPQA-D **0.4848**, reproducing the ubel #511 anchor). Against that
now-higher live denominator, base_fullhead's **STRICT Wilson CI-lower-bound** MISSES the
≥90% line at n=500/198:
- MMLU: base_fullhead CI-lb **0.593** vs 0.90×0.662 = **0.596** — near-miss by 0.003.
- GPQA: base_fullhead CI-lb **0.401** vs 0.90×0.4848 = **0.436** — miss by 0.035.

The #542 (`92pcnx6a`) two-proportion z-GO already passed (base_fullhead statistically
indistinguishable from the denominator). The two lenses disagree ONLY because of Wilson CI
*width* at n=500/198 (a Wilson lb sits ~0.04 below the point at n=500). **The clean question:
does the strict-CI-lb gate flip TRUE once both CIs are tightened by larger n?**

## Why this is non-obvious
Larger n tightens the CI symmetrically. IF the points genuinely hold at 0.636/0.662 (96.1% of
the denominator), the MMLU CI-lb should rise above 0.596 — the headroom is point−gate = 0.636−0.596
= 0.040, and the n=500 half-width is ≈0.042, so the flip is expected around **n≈554+** UNLESS the
point itself drifts at the larger draw (sampling-set composition) or the denominator drifts up.
Only the run settles it.

**Structural asymmetry (a reported finding):** GPQA-Diamond is the FULL 198-item set in #542/#557
and here — its Wilson width is fixed at the dataset ceiling and CANNOT be tightened by larger n
(clearing the GPQA gate would need n≈830, impossible on a 198-item benchmark). So the larger-n
lever is **MMLU-Pro only** (12032-item pool). GPQA's verdict therefore turns on Stage 3: is the
GPQA point gap (3 items: 93 vs 96 / 198) a real surgical-attention decrement or sampling noise?

## Design
Build = `vllm-0.22.1rc1.dev307+g3e8afdf78` for BOTH arms (`/tmp/senpai-venvs/5f4c623f772358a2`).
Stock ckpt = `/tmp/gemma4-e4b-qat-w4a16-ct` (native 262k head, int4 W4A16). Client harness = ubel
#511 `run_eval.py`, greedy temp=0, `--max-connections 32`, seqs=16, byte-identical seeded item set
(`prompt_sha` asserted across arms → `n_prompt_mismatch=0`). The ONLY moving variable between arms
is the attention reduction path; both carry the PLE embed-scale fold (the #557 correctness gate).

- **base_fullhead** = `FA_SLIDING=1` + `SURGICAL_ATTN_USE_3D_OFF=1` + `PLE_FOLD_EMBED_SCALE=1`.
  The submission's exact surgical 2D order-preserving attention (the #557 `surgical_attn` arm) PLUS
  the fold — the healthy base_fullhead quality config. (#542 measured 0.636/0.4697 with this
  attention under the full MTP/split-KV/onegraph speed stack at seqs=1; that stack is
  greedy-identity-preserving and the surgical attention is batch-invariant, so seqs=16 no-MTP
  reproduces the same greedy tokens while staying fast enough for larger n. The larger-n
  base_fullhead point being consistent with ~0.636 is itself the serve-equivalence check.)
- **ple_fold** = `PLE_FOLD_EMBED_SCALE=1` only — the #557 recovered healthy denominator. MUST set
  the fold or it collapses to the broken 0.43/0.31 base (#557 root cause).

## Stages (all PRIMARY)
- **Stage 1 — re-measure both arms at larger MMLU-Pro n + full GPQA-198.** Report each point +
  Wilson 95% CI + `n_used`. Confirm byte-identical item sets (`n_prompt_mismatch=0`).
- **Stage 2 — strict-CI-lb gate at larger n.** `strict_cilb_mmlu_passes_largern`,
  `strict_cilb_gpqa_passes_largern`, `strict_cilb_passes_largern` (both), each margin
  (`base_fullhead_cilb − 0.90×ple_fold_point`) and the delta vs the #557 n=500/198 margins.
- **Stage 3 — paired surgical-vs-vanilla mechanism.** On the EXACT same items, paired:
  `point_gap_mmlu`/`point_gap_gpqa` (= ple_fold − base_fullhead), McNemar/sign-test p
  (`point_gap_is_significant`), `point_gap_mechanism` (`surgical_attention_cost` vs
  `sampling_noise`). If significant, which item class flips.

## Anchors / gates
- #557 recovered ple_fold denominator: MMLU-Pro **0.662** / GPQA-D **0.4848**.
- #542 base_fullhead: MMLU-Pro **0.636** ([0.593, 0.677]) / GPQA-D **0.4697** ([0.401, 0.539]).
- Morgan #515 gate: base_fullhead ≥ 0.90 × vanilla base. z-GO (#542) stands; this hardens the
  strict-CI-lb lens. #319 strict greedy-identity and the TPS benchmark are SEPARATE and unaffected.
- kanna #563 owns the lewtun-#31 SAMPLING-protocol axis; THIS card is the GREEDY-protocol CI-width
  axis (the protocol the program record + the strict-CI-lb finding were measured on).

`primary_metric = strict_cilb_passes_largern` (or the binding margin).
