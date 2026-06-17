### Same-recipe head-to-head (surgical357 substrate, only the checkpoint differs)

| frame | base_fullhead (int4_g32 + bf16 262k head) | int4_g128_lmhead (int4_g128 + int4 head) | faster | Δ (×σ_hw) |
|---|---|---|---|---|
| **spec-OFF AR M=1** | 83.54 | 106.66 | int4_g128 | +23.12 (+4.8σ) |
| **spec-ON MTP-K7** | 254.00 (E[T]=3.851) | 337.32 (E[T]=4.255) | int4_g128 | +83.32 (+17.1σ) |

_Official-projected (×τ=1.03524): base AR 86.5, base spec 263.0, int4_g128 AR 110.4, int4_g128 spec 349.2._

### lm_head byte-read decomposition

| head | bytes | read @501GB/s |
|---|---|---|
| bf16 262k (base_fullhead) | 1.342 GB | 2.679 ms |
| int4-g128 262k (int4_g128_lmhead) | 0.346 GB (25.8%) | 0.691 ms |
| **savings** | **0.996 GB** | **1.988 ms/step** |

- `int4_head_read_savings_tps` (AR frame, surgical357 base) = **+16.64 TPS**
- measured AR-frame delta (head + g128 body scales) = **+23.12 TPS**; body-scale residual = +6.47 TPS

### Official 126.378 sits in the **spec-OFF AR (M=1)** frame
- int4_g128_lmhead submission manifest carries NO SPECULATIVE_CONFIG (plain serve.py, vllm 0.22.0) -> no drafter -> cannot be a spec-ON number
- 126.378 is bracketed in the AR band: just ABOVE our surgical357 int4_g128 AR floor (~110.4 official-proj) and FAR BELOW any spec rate (int4_g128 spec-ON ~349, base spec ~263). >2x below spec => AR, not spec.
- **Quantitative repro (no extra run):** surgical357-refmode AR overhead = base 83.54 / clean 97.01 = ×0.861. Apply to int4_g128 surgical AR 106.66 → implied clean AR 123.85 local → ×τ = **128.22 official** (vs 126.378, err 1.84; reproduces: True).
- surgical357 is the faster stack in the SPEC frame but carries M=1-AR overhead from its verify-oriented kernels; int4_g128's OWN plain serve.py is faster for pure AR, so 126.378 sits ABOVE our surgical357 AR floor (g_off*tau). Applying base_fullhead's measured surgical-refmode AR overhead factor to int4_g128's surgical AR recovers its clean-serve AR, which x_tau reproduces the official 126.378 -> 126.378 IS the AR-frame number.