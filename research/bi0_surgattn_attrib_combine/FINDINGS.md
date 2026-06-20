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

## VERDICT — CONFIRMED CLEAN NULL (W&B kcojzw20)

The +5.35% IS mathematically inseparable from the reassociation. The occupancy win
IS the parallel (re-associated) reduction; partials are already fp32 so there is no
precision lever; deterministic combine order requires serializing (= removing the
split = removing the win). Confirmed on three independent legs:

**Microbench (isolated M=1 attention, 8 layer/ctx configs):**
- 3D kernel is 1.38x-9.13x faster than 2D (2D bandwidth-starved ~20-37 GB/s; 3D
  recovers occupancy 49-337 GB/s). All 8 configs selected the 3D path.
- config (a) no-op CONFIRMED: forced-bf16 partials diverge from 2D MORE than fp32 in
  8/8 configs -> fp32 is already the optimal (and in-use) accumulator.
- pure reassociation CONFIRMED: 2D-vs-3D differ in fp32 (pre-bf16-cast) at >=99.9%
  of elements in 7/8 configs, max abs only 2.63e-5. NOT a quality regression: 3D is
  as-close-or-closer to SDPA than 2D in 6/8 (equally-valid roundings).

**E2E (within-lineage 2D control, 128x512):**
- decode-wall TPS 218.30 (2D, == bi0 official 218.02) -> 229.97 (3D) = +5.35%
  (reconciles wirbel #785's +6.69%). Probe TPS (256-tok) +1.33% @ 9.0sigma — a
  short-context underestimate (3D win grows with KV length).
- PPL bit-identical (2.0055, NLL equal; prefill teacher-forcing never hits M=1).
- greedy 3D-vs-2D: 124/128 identical, 4 divergent, onsets [373,382,472,509]
  (73-99% into the 512-tok output = deep near-tie flips).

**Determinism control (the clincher):** same-config run-to-run noise band is EXACTLY
0 (2d_a-vs-2d_b = 0 divergent; 3d_a-vs-3d_b = 0). The 4/128 is identical across all
four cross-config pairings -> purely the 2D-vs-3D reassociation, not session noise.
Since the band bi0 clears is literally 0, the reproducible 4/128 is neither
byte-identical nor tie-tolerant-equivalent. Byte-identical route is DEAD; hands
wirbel #791 the quality-gate decision.

## Methodology note — local spec-off reference looks stale (NOT a 2D-vs-3D finding)
Official `greedy_gate` vs the cached local spec-off reference reads ~99/128 divergent
for BOTH spec-on arms equally (2d:99, 3d:100; token-0 matches 127/128; onsets spread
0-481). Since both spec-on arms are mutually bit-reproducible yet both diverge equally
from that one Jun-19 reference, this is cross-job spec-on-vs-spec-off int4
nondeterminism against a stale reference (consistent with the gate being within-job),
not specific to either arm. Regenerate within-job before trusting it. Not a bi0 bug.

## Baselines (PR body)
- bi0 control (2D forced): official TPS 218.02, PPL 2.0058, 128/128. W&B s63tb03x.
- wirbel #785 3D anchor: local TPS 224.55 (+6.69%), 6.25% prompts / 1.76% tok
  divergence, PPL 2.0057, 128/128. W&B ak4k3wt4.
