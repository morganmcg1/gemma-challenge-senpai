# PR #536 — lm_head transplant: compatibility check (Step 1)

`analysis_only=true`, `official_tps=0`, NO HF Job, NO `--launch`. Local A10G only.

## Dim contract (measured from the actual checkpoints)

| tensor | osoi5-v0-baked (body) | base `google/gemma-4-E4B-it-qat-w4a16-ct` |
|---|---|---|
| `lm_head` | `weight_packed` I32 `[16384, 320]` (int4 channelwise, in=2560) | `lm_head.weight` **BF16 `[262144, 2560]`** (unquantized) |
| `lm_head.weight_scale` | F16 `[16384, 1]` (channel strategy) | — (none; head is BF16) |
| `embed_tokens.weight` | BF16 `[262144, 2560]` | BF16 `[262144, 2560]` |
| `hidden_size` | 2560 | 2560 |
| `tie_word_embeddings` | **False** | **True** |
| lm_head quant in config | `group_0_lmhead` int4 channel | `lm_head` in `ignore` (BF16) |

## Verdict

- **`head_input_dim_compatible = TRUE`.** The osoi5 body emits a standard **2560-dim** final hidden
  (embed_tokens `[262144,2560]`, hidden_size 2560, lm_head in-dim = 320 int32 × 8 = 2560). The base
  head input dim is 2560. No `<2560` bottleneck was inserted at bake → the body→head interface is
  NOT co-adapted → **transplant is dimensionally feasible.**

## Correction to the PR premise (load-bearing)

The PR body assumed the base head is **int4-packed** `[262144, 320]`. It is not. The base
`...qat-w4a16-ct` keeps `lm_head` **unquantized BF16 `[262144, 2560]`** (lm_head is in the quant
`ignore` list) and **tied** to `embed_tokens` (`tie_word_embeddings: True`). Verified byte-identical:
base `lm_head.weight` == base `embed_tokens.weight` across head/mid/tail 1 MB windows.

Consequence for the transplant:
- The transplanted head is **BF16 full-precision**, not int4. This is the *strongest possible* head
  (full vocab + full precision), so it is the cleanest probe of the head-vs-body question: if even
  this head fails to recover AIME on the osoi5 body, the BODY is definitively implicated.
- It does mean the transplant moves TWO things vs the osoi5-16k row: vocab 16k→262k AND head
  precision int4→BF16. For the HEAD/BODY verdict this is fine (both live "in the head"); only a
  follow-up needs to separate prune-vs-quant within the head.
- The fast-ship implication is softened: a re-bake-free fast ship would still need a *fast* 262k head
  (full-vocab argmax tax, no fused-sparse path) — the transplant TPS is a lower bound, not a ship number.
