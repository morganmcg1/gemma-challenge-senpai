# PR #448 — Is the served int4 verify-GEMM the fastest config for sm_86 M=8?

**stark · group `kernel-tiling-sweep` · LOCAL A10G (sm_86) microbench · NO served-file change, NO kernel BUILD, NO HF submission.**

## Question
The verify body int4 W4A16 GEMM is the DOMINANT share of verify (denken #441 `7rb089z3`:
T_body = 16935.9µs of T_verify = 18270.6µs → **92.7% of verify**, attention only 6.90%). Is the
served Marlin kernel+config the FASTEST AVAILABLE for the real sm_86 / M=8 / head_dim=256 /
GQA-8:2 shape — or is there selectable headroom (Marlin config knob, or an ALTERNATIVE int4 GEMM
already compiled into the vLLM-0.22 wheel: Machete, marlin_gemm, awq) with **selection + config only, no build**?

## Real served shapes (from `/tmp/osoi5-v0-baked/config.json`)
hidden=2560, intermediate=10240, Q heads=8, KV heads=2, head_dim=256, layers=37 (30 sliding + 7 full),
int4 W4A16 **group_size=128 symmetric** (compressed-tensors pack-quantized), lm_head channel-wise int4.
Fused GEMM shapes vLLM serves (per decode layer, M=8):
- `qkv_proj`  K=2560  N=3072  (×37)
- `o_proj`    K=2048  N=2560  (×37)
- `gate_up`   K=2560  N=20480 (×37)
- `down`      K=10240 N=2560  (×37)
- `lm_head`   K=2560  N≈12000 (×1, pruned; minor — #441 lmhead 74.75µs)

## Plan
1. **Identify dispatch** — confirm the compressed-tensors W4A16 path routes to `gptq_marlin_gemm`
   on sm_86; enumerate the selectable surface (`use_fp32_reduce`, `use_atomic_add`, group_size,
   alternative ops `machete_mm` / `marlin_gemm` / `awq`).
2. **Microbench served default** at M=8 for each shape — primary timing is **CUDA-graph replay**
   (mirrors served ONEGRAPH amortized-launch path), eager as cross-check. Cross-check the summed
   T_body against #441's 16936µs to validate shapes/counts.
3. **Sweep selectable surface**: (a) Marlin config knobs; (b) alternative int4 kernels already in
   the wheel. Report best kernel+config latency vs served default. Expectation/prior to test:
   **Machete is Hopper sm_90a-only** (kanna #132 `n_subbit_servable_in_wheel=0`) → empirically
   verify it errors / returns no schedules on sm_86.
4. **Honest end-to-end map** — apply the best COMPUTE-part speedup via the Amdahl frontier
   (`f_body`=0.7624, base 467.14 realized / bar 481.53 deployed), NOT the raw isolated-op Δ.
   Avoids the pinned-K (#433 +13.998→−5.82) / cb3 (#437 +15.60→0.0) trap.
5. **Byte-exact gate** — candidate-vs-default GEMM on the SAME stack/size path; bit-compare
   outputs (a reduction-order change like `use_fp32_reduce=False` that flips bits = identity FAIL).
6. **Verdict** — is there **>+2 TPS honest end-to-end headroom** in int4-GEMM selection, or is
   served Marlin already optimal for this shape? If the only faster path is a source/kernel BUILD,
   flag NO-GO scope precisely (like stark #440), do not build.

## Anchors
- Realized equivalence frontier **467.14** (denken #423 `5a6zq2yz`).
- Deployed incumbent **481.53** / PPL 2.3772 (PR #52 `2x9fm2zx`, non-equivalent).
- Body GEMM is **M-invariant** weight-read bound (#441/#437: +0.47% M=1→8); Marlin ~64% peak-copy
  HBM eff at M=8 (#437 `marlin_hbm_eff=0.6418`, peak-copy 482.29 GB/s).
- PPL ≤ 2.42 anchor; strict byte-exact greedy identity gate.
</content>
