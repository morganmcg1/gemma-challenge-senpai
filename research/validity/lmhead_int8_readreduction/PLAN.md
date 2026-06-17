# PR #593 — Cheaper identity-safe lm_head: int8/top-K read-cut vs bf16 252.69?

**Branch:** `stark/lmhead-int8-readreduction` · **W&B group:** `lmhead-int8-readreduction`
**Mode:** LOCAL diagnostic card — `analysis_only=true`, `official_tps=0`, **NO FIRE.**

## Hypothesis
`base_fullhead`'s dominant single-stream decode cost is the full 262k-row bf16
lm_head HBM read (~0.5 GB/token-step). The 12k head-prune is fast (375.857) but
collapses quality. Un-probed middle: can we get the SAME greedy argmax from a
CHEAPER head READ without pruning vocab — via (a) int8-per-channel lm_head (half
bytes, full 262k vocab), or (b) top-K candidate gather / hierarchical argmax that
reads only a candidate slice? Under the OPERATIVE/#407 identity (argmax-class vs
served int4 `base_fullhead`), an argmax-preserving cheaper head read would be a
NEW, un-closed speed lever on the bottleneck. NOT the strict-byte-identity census
(#556) — int8 changes logits but may PRESERVE the argmax.

## Anchors (authoritative, post-grounding 2026-06-17)
- `base_fullhead` = stock int4_g32 QAT body + full native 262k bf16 lm_head.
  Anchor **252.69 TPS / PPL 2.0057** (wirbel #553, run `83jiwjr9`). spec-OFF,
  greedy temp=0, MAX_NUM_SEQS=1, `min_tokens=8` EOS-guard (#541),
  `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Ship to beat: 375.857 official (12k prune, quality-collapsed). Decode floor 311.27 (#569).
- Identity: operative/#407 int4-referenced (wirbel #585, run `2u44yaa1`).
- Quality gates (if re-cert needed downstream): MMLU-Pro ≥0.605, GPQA-D ≥0.471, GSM8K ≥0.807, AIME ≥0.090.

## Plan
1. **int8 head numerical argmax-preservation probe (primary, cheap):** real int4-body
   decode hidden states → bf16 head (ref argmax) vs int8-per-channel head (test argmax).
   Match rate vs operative threshold. Make-or-break; de-risks before serving.
2. **Head-read TPS probe:** per-token head-read cost bf16 vs int8 (batch=1 HBM-bound GEMV);
   serve locally if warranted → TPS vs 252.69.
3. **top-K gather probe (secondary):** can argmax be recovered from a reduced HBM read at
   operative reliability? Be explicit about what is actually read.
4. Combine → `operative_safe_head_read_reduction_exists` + best `safe_head_tps`.
5. Log argmax-match + TPS + verdict bools to W&B.

## Verdicts (to fill)
- `int8_head_identity_safe`: TBD
- `int8_head_tps_gain`: TBD
- `operative_safe_head_read_reduction_exists`: TBD
- `safe_head_tps`: TBD
