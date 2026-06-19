# PR #721 — Kernel-authoring viability: custom int4 M=1 GEMV vs Marlin wall

**LOCAL-ONLY.** `analysis_only=1`, `official_tps=0`, `no_hf_job=1`, `fires=0`.
One pod A10G (sm_86). NO HF Job, NO `--launch`, NO submission, NO served-file change.

## Question
The #655 human greenlight lifted the kernel-authoring gate. Can a purpose-built
**non-Marlin int4 g128 M=1 (conc=1) decode GEMV** in Triton beat Marlin's realized
M=1 decode-step latency on this A10G, or is the **90.7% HBM read-peak wall**
(denken #676, `vwiqwzvk`) fundamental to *any* kernel on this hardware?

## Anchors
- Locked AR submission: `submissions/int4_g128_lmhead` — official a10g tps **126.378**,
  PPL 2.019, GREEDY_IDENTICAL 128/128, W&B `905tbujn`. Full int4-g128 body + int4
  lm_head; vLLM repacks to **Marlin** at load. **+10 bar = 136.378.**
- denken #676 (`vwiqwzvk`): byte-identical int4 GEMV walled at ~0 headroom =
  **90.7% of A10G HBM read-peak**. The wall this kernel tries to beat.
- stark #433 was a pinned-K **attention** split (−5.82% was an FA2 num_splits
  artifact on `kernel_unified_attention`), NOT an int4 weight GEMV. The relevant
  GEMV prior art is `research/m1_gemv_microbench.py` (#506, bf16) + stark #448
  (int4-GEMM audit: Marlin is the unique sm_86 int4-g128 GEMM). [flag to advisor]

## Physics framing
At M=1 conc=1, decode is ~92% weight-GEMM and memory-bandwidth-bound. The int4-g128
weight bytes are FIXED (same weights). A custom kernel beats Marlin only by reading
the *same* bytes at a higher fraction of HBM read-peak. Marlin is at 90.7%; the
practical streaming ceiling on this pod is ~517.6 GB/s (ubel #450 STREAM read peak)
vs 600 spec. Headroom above 90.7% is structurally small — but MEASURE it, don't assume.

## Method
1. Extract a real int4-g128 weight matrix from the locked checkpoint (lm_head
   262144x2560 is the dominant single GEMV; also body shapes).
2. Build the Marlin reference GEMV at M=1 (vLLM/compressed-tensors path) — latency +
   achieved HBM%.
3. Author a custom Triton int4-g128 M=1 GEMV (dequant-in-register, K-sequential,
   occupancy-tuned for N-tiling). Verify numerics vs dequant->bf16 matmul.
4. Head-to-head M=1 latency, implied forward-TPS delta, achieved HBM read-peak %.
5. #319 self-consistency: served-fast greedy with the kernel == plain-AR greedy of
   the same weights, 128/128.

## Verdicts
- `KERNEL_HEADROOM_REALIZABLE` — custom kernel beats Marlin M=1 latency AND preserves
  #319 identity. Report speedup + new implied AR-lane TPS.
- `KERNEL_HBM_WALLED` — cannot beat Marlin / the 90.7% read-peak wall. The
  byte-identical int4 GEMV is bandwidth-fundamental, not a Marlin artifact.
