# PR #794 — Byte-identical split-KV: can the surgattn +6.69% ship quality-free?

Investigation of WHY the 3D split-KV path is faster on M=1 forwards and whether
the speedup is separable from the greedy-identity break. Base: shipped bi0
`submissions/int4_mtp_bi0_surgattn`. LOCAL ONLY.

## Mechanism (from the bi0 + vLLM 0.22.0 TRITON_ATTN source)

`vllm/v1/attention/ops/triton_unified_attention.py`:
- `unified_attention(...)` launch gate (L918-932): `use_3d = True` only when all
  segm buffers are allocated AND `max_seqlen_q == 1` AND `num_seqs <= seq_threshold_3D`
  AND not batch-invariant. So **only M=1 forwards can take 3D**; the M=7 spec-verify
  (`max_seqlen_q>1`) is ALWAYS 2D.
- bi0's `vllm_force2d_attn_patch.py` nulls the three `softmax_segm_*` kwargs on
  every call → forces `use_3d=False` everywhere → M=1 decode byte-matches M=7 verify
  → greedy identity. surgattn-OFF = let M=1 take 3D = wirbel #785's +6.69% arm.
- The 3D occupancy win: 2D grid = `(num_q_blocks, num_kv_heads)` (tiny on M=1,
  underfills SMs). 3D grid = `(num_q_blocks, num_kv_heads, NUM_PAR_SOFTMAX_SEGMENTS=16)`
  → 16× more blocks → the KV-length reduction is split across 16 segments (program_id(2)),
  recovering SM occupancy. `reduce_segments` (L645-732) then combines the partials.

## KEY FINDING #1 — the split-KV partials are ALREADY fp32 (config (a) is a no-op)

`triton_attn.py:185-204` allocates ALL THREE scratch buffers as `torch.float32`:
- `softmax_segm_output`  (seq_threshold_3D, num_heads_q, 16, headdim_padded)  float32
- `softmax_segm_max`     (..., 16)                                            float32
- `softmax_segm_expsum`  (..., 16)                                            float32

The combine `reduce_segments` is fully fp32: loads fp32 partials, rescales by fp32
`exp(segm_max - overall_max)`, sums via `tl.sum(..., axis=0)` in fp32, divides in fp32.
The ONLY narrowing cast (fp32 acc → bf16 output) happens at the FINAL store — and the
2D path does the identical final cast. **There is no bf16/fp16 accumulator anywhere in
the split-KV path.** => PR config (a) "force an fp32 accumulator for the combine" is
already in effect; it cannot reduce divergence further. (To be confirmed at runtime.)

## KEY FINDING #2 — the divergence is pure reduction REASSOCIATION (two sources)

With fp32 partials, 3D still ≠ 2D bit-for-bit because the reduction is re-associated:
1. **Cross-segment combine order.** 2D accumulates ALL tiles into one running
   (m, l, acc) sequentially. 3D computes 16 independent partials then merges them
   (`exp(m_s-overall_max)`-weighted sum). `((a+b)+c)+d` vs `(a+b)+(c+d)` → different
   fp32 rounding. This is INHERENT to splitting the KV reduction = inherent to the
   occupancy win. You cannot parallelize the reduction without re-associating it.
2. **Tile-size mismatch (confounder).** 2D uses `TILE_SIZE_PREFILL`; 3D uses
   `TILE_SIZE_DECODE` (L962 vs L965). For the global-attention layers these are
   32 vs 16 → different online-softmax rescale boundaries even within a segment.

## Working hypothesis (to be tested, not yet concluded)

The +6.69% appears mathematically inseparable from the reassociation: the occupancy
win IS the parallel (re-associated) reduction, and the partials are already fp32 so
there is no precision lever to pull. config (a) = no-op; config (b) "deterministic
combine order" cannot reproduce the 2D sequential association without serializing the
combine (= removing the split = removing the win). Expected outcome: CLEAN NULL that
hands wirbel #791 the quality-gate decision. MUST confirm with:
- runtime assertion that 3D is taken on M=1 and buffers are fp32,
- microbenchmark: 2D vs 3D output diff on identical M=1 inputs (expect fp32-epsilon),
  + bf16-buffer control showing fp32 is already the best accumulator,
- end-to-end greedy-compare + TPS (N>=5) reproducing wirbel's anchor.

## Baselines (PR body)
- bi0 control (2D forced): official TPS 218.02, PPL 2.0058, 128/128. W&B s63tb03x.
- wirbel #785 3D anchor: local TPS 224.55 (+6.69%), 6.25% prompts / 1.76% tok
  divergence, PPL 2.0057, 128/128. W&B ak4k3wt4.
