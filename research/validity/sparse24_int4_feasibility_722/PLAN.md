# PR #722 — 2:4 sparse-int4 feasibility (stark)

**Verdict space:** SPARSE_INT4_SPEED_LEVER / SPARSE_INT4_NO_SPEEDUP /
SPARSE_INT4_IDENTITY_BROKEN / SPARSE_INT4_KERNEL_ABSENT.

**Anchor:** `submissions/int4_g128_lmhead` — a10g-small tps=126.378, PPL 2.019,
GREEDY_IDENTICAL 128/128, W&B `905tbujn`. +10 bar = 136.378.

**Hypothesis:** 2:4 structured sparsity is the one HW-accelerated fewer-bytes
M=1 lever left after #721 closed the same-bytes kernel lane (Marlin at 86–99.3%
HBM read-peak, 517.4 GB/s). 2:4 stores ~2.65 eff bits/wt vs 4 dense → −34%
weight bytes → projected ~1.3–1.5× decode TPS (→ ~165–190) if the byte cut
realizes at the bandwidth-bound decode wall.

## Screen order (first GO/NO-GO = servability)
1. **Servability** — can int4-g128 + 2:4 build & serve on A10G sm_86 in vLLM?
   Sparse-Marlin (IST-DASLab) / compressed-tensors `sparse-24`+`w4a16`
   (CompressedTensors24). No kernel → SPARSE_INT4_KERNEL_ABSENT, stop.
2. **M=1 decode TPS** vs 126.378 + fraction of 517 GB/s read-peak. Bar =
   faster than 126.378. (reuse #721 bench harness)
3. **#319 self-consistency at BI=1** — served-fast greedy == own plain-AR
   greedy, 128/128. + kanna #699 batch-invariance health check (cc=1 only).
4. **Coarse quality** — PPL + small greedy AIME subset vs int4-body
   (AIME 0.347, PPL 2.019). Magnitude, not gate.

## Hard constraints
LOCAL A10G sm_86 only. analysis_only=1, official_tps=0, no_hf_job=1, fires=0.
NO HF Job / `/v1/jobs:run` / `train.py --launch`. Locked 126.378 untouched.
2:4 is a mandate-question lever (like int8 #655) → LOCAL measure needs no
approval; a SHIP would → surface to human if it's a real lever.

## Env note
Fresh pod: `.venvs/vllm022` (py3.11) must be rebuilt. CUDA_VISIBLE_DEVICES=0
for torch (inherited =1 → torch sees 0 GPUs).
