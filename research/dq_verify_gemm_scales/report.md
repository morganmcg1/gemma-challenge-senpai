# PR #104 — Double-quant verify-GEMM FP16 scales to INT8: build-or-kill

**Verdict: KILL.** The deployed verify-GEMM FP16 group-scales do **not** double-quantize
to INT8 bit-exactly: only **13.1%** of scales round-trip bit-for-bit (gate: >98%).
The 86.9% FP16 sparse-exception set brings even the most byte-favorable lossless
hybrid back to **−1.27%** vs the original FP16 buffer (i.e. slightly *larger*) —
versus the ~+47% scale-byte saving the lever needs. There is no greedy-lossless
byte saving here. Phase 2 (GPU bandwidth microbench) is **not run** (gated on GREEN).

W&B: `6or2w3ee` (group `dq-verify-gemm-scales`). Repro:
`python scripts/profiler/dq_scale_roundtrip.py` (CPU-only; reads only the
`weight_scale` tensors, never the packed weights or the token stream).

## Deployed checkpoint facts (premise check)

- Deployed frontier `fa2sw_precache_kenyan` → PLE-folded `/tmp/osoi5-v0-baked`,
  compressed-tensors **pack-quantized, num_bits=4, symmetric, group_size=128**,
  `weight_scale` dtype **F16**. The PR's g=128 framing is **correct for the
  deployed checkpoint**. (The `group_size=32` note in roofline #68 refers to the
  *un-folded base* `google/gemma-4-E4B-it-qat-w4a16-ct` — a different artifact.)
- core7 verify-GEMM body scales (q/k/v/o/gate/up/down across 37 layers):
  **26,849,280 scales = 53.70 MB FP16** (54.84 MB incl. Gemma-3n altup/laurel/PLE
  projections). Int4 body weight bytes = **1754.7 MB**, so scales are **3.06%** of
  the verify-GEMM weight stream.
- **Correction to the PR baseline:** the PR states "FP16 g=128 scale tensor ≈ 26 MB
  ≈ 0.78% of weight bytes." Measured truth is **53.7 MB ≈ 3.06%** of the int4 body
  stream (~2× the MB, ~4× the %). The lever *target* is bigger than the PR thought —
  but that is moot because the lever yields no bit-exact saving.

## Method

Asymmetric INT8 double-quant per QLoRA-style secondary block (offset=min,
step=(max−min)/255, FP32 secondary scale/offset), dequant→FP16, compared on raw
FP16 bit patterns (`uint16` view, not `allclose`). Asymmetric (full 256 levels
over the actual range) is the most bit-exact-favorable uniform scheme; FP32
secondary makes the per-block overhead negligible (0.03 B/scale @ block 256).
Lossless hybrid byte cost credited the **cheapest** correct encoding (INT8 only
for bit-exact scales + FP16 only for exceptions + 1 presence-bit/scale bitmap).

## Results

| secondary block | bit-exact frac | exception frac | net saving (% of scale bytes) |
|---|---|---|---|
| 256 (QLoRA) | **0.1309** | 0.869 | −1.27% |
| 128 | 0.1633 | 0.837 | −1.21% |
| 64 | 0.2078 | 0.792 | −2.11% |
| 32 | 0.2726 | 0.727 | −5.12% |
| 16 | 0.3769 | 0.623 | −12.40% |

- **Within-block dynamic range is the killer:** median (max−min)/min = **2.60**
  (p90 = 5.13). **0%** of secondary blocks fall under the bit-exact-guarantee
  threshold of 255/1024 ≈ 0.249. Smaller blocks raise bit-exactness (→37.7% @ 16)
  but never reach GREEN, and net saving stays **negative** at every block size
  (overhead + still-high exception rate dominate).
- **Per-role:** uniformly 10–16% bit-exact (gate 0.110, down 0.149, q 0.147,
  v 0.160, o 0.116…). Nothing is salvageable by exempting a role.
- **BF16-storage variant:** only **6.87%** of the FP16 scales are exactly
  BF16-representable → storing scales as BF16 is itself lossy for 93% of scales →
  fails greedy-identity *independently of* double-quant. (The 80.45% INT8-on-BF16
  bit-exact figure is moot because the BF16 base is already lossy.)
- Achievable TPS lift ≈ **−0.02%** (best-encoding net byte saving is marginally
  negative → no lift; you simply would not deploy it).

## Root cause (why no 8-bit scheme can work here)

FP16 carries 10 mantissa bits. The per-secondary-block scale range spans ~1.4–1.9
octaves (median max/min ≈ 3.6). Any 8-bit-per-scale code — uniform INT8 **or**
non-uniform FP8-E4M3 (3 mantissa bits) — discards ≥2 bits relative to FP16 and
cannot reconstruct the bulk of scales bit-exactly. Only a code with **>9 bits**
(e.g. a palette of the distinct values) or keeping FP16 is bit-exact. Double-quant
to INT8 is therefore fundamentally incapable of the greedy-lossless saving on this
checkpoint; this is an information-theoretic barrier, not a tuning issue.

## Suggested follow-ups (not implemented — scope discipline)

1. **Lossless scale palette / LUT (the lever that actually exists).** core7 scales
   take only **1,009 distinct FP16 values globally** (per-tensor median **427**).
   A 10-bit global (or 9-bit per-tensor) index into a palette of the *exact*
   original values is **bit-exact by construction** and saves **~37.5%** of scale
   bytes (≈20 MB; ~0.6% of the verify-GEMM stream → est ~0.3% TPS at the 53.2%
   verify-GEMM block). Cost: a non-power-of-two index unpack folded into the
   dequant. Worth a dedicated build-or-kill against the Marlin scale path.
2. **Log-domain / mantissa-aware secondary quant** would track FP16's structure
   better than uniform INT8, but is still 8-bit < 10-bit mantissa, so it cannot be
   bit-exact for wide blocks — expect it to fail the same gate.
3. Either path must be re-checked against whichever verify-GEMM kernel wins (Marlin
   vs ubel #84 SplitK), since it changes the scale *format* the kernel reads.
