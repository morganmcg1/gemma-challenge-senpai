STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["crrq2e1y"],"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"primary_metric":{"name":"max_honest_endtoend_tps_delta","value":0.2613},"test_metric":{"name":"ppl","value":2.3772},"self_test_passes":true,"any_nonattn_triton_over_2tps":false,"int4_gemm_dominant":true,"int4_gemm_frac_of_verify":0.8509,"attn_frac_of_verify":0.1419,"attn_best_cfg":"BM16_BQ4_TS16_w4_s2","attn_kernel_confirmed_speedup_pct":6.108,"beats_deployed_481":false}

## Results

**Verdict (one line):** The verify wall is **85.1% vendored int4-Marlin GEMM** (no selectable Triton tiling), **14.2% attention** (of which **91% is vendored FA2** / only **9% is the tunable Triton-3D split-KV kernel**), **0.64% lm_head**, **0.08% sampling**, **~0% dispatch**. The **only** tunable Triton kernel in verify is the 7-layer Triton-3D global attention (**75.3 µs = 1.3% of verify**). Sweeping wirbel's grid found a **REAL, reproducible** kernel win — **`num_stages` 3→2** gives **+6.11%** on that kernel (7-rep alternating reconfirm, σ≈0.01 µs, gap 60× noise) — but mapped honestly through the cycle fraction it is **+0.2613 TPS** end-to-end. **No non-attention Triton kernel has >+2 TPS headroom.** wirbel #442's modeled **+15.86** implies a **279.9 µs** verify saving = **3.7× the entire tunable kernel (75.3 µs)**; even *deleting* that kernel caps at **+4.27 TPS** — so **+15.86 is physically impossible from a tile retune** (the pinned-K #433 / cb3 #437 modeled-vs-realized trap). **Does NOT beat the deployed 481.53; the realized frontier stays 467.14 (+0.26).**

### 1. Full verify-step wall decomposition (M=8 / K=7 / head_dim 256 sliding · 512 global / ctx=128, sm_86 A10G)

Timed on the **served ONEGRAPH basis** (per-component CUDA-graph capture + replay, 300 iters / 100 warmup). All five components captured (`captured_flags` all True). Fractions are over `sum_components = 5911.5 µs`; the fused captured-graph wall `T_verify_graph = 5756.1 µs` is 97.4% of the component sum (slight in-graph kernel overlap).

| component | kernel / backend | absolute µs | % of verify |
|---|---|---:|---:|
| **int4-GEMM body** | **vendored Marlin W4A16 (CUDA, fixed tiling)** | **5029.89** | **85.09%** |
| &nbsp;&nbsp;↳ MLP gate_up | Marlin, 30.42 µs ×84 | 2555.5 | 43.2% |
| &nbsp;&nbsp;↳ MLP down | Marlin, 31.06 µs ×42 | 1304.7 | 22.1% |
| &nbsp;&nbsp;↳ attn q/k/v/o proj | Marlin | 1169.6 | 19.8% |
| **attention** | FA2 + Triton-3D | **838.82** | **14.19%** |
| &nbsp;&nbsp;↳ FA2 head-256 (×35 sliding) | **vendored flash-attn CUDA** | 763.5 | 12.92% (91.0% of attn) |
| &nbsp;&nbsp;↳ Triton-3D split-KV head-512 (×7 global) | **TUNABLE Triton** | **75.3** | **1.27% (9.0% of attn)** |
| lm_head 12288 | Marlin W4A16 | 37.79 | 0.64% |
| sampling | `rejection_greedy_sample_kernel` (Triton) | 4.95 | 0.08% |
| dispatch / overhead | — | ~0.0 | ~0% |

- **The dominant cost is the int4-GEMM body (85.1%), and it is vendored Marlin CUDA — no selectable Triton tiling exists.** The MLP alone (gate_up + down) is **65.3% of the entire verify wall**. Per-shape effective bandwidth: gate_up = 13.52 MB / 30.42 µs = **444 GB/s**, down = **435 GB/s** ≈ **73% of the A10G's ~600 GB/s HBM peak** → the body is **HBM-read-bound on the int4 weights** (the physical signature of a re-tile-proof kernel; corroborates gate_up_retile #130).
- **lm_head + sampling + dispatch together = 0.72%.** There is no hidden time in the "other ~93%" the hypothesis asked about — it is **almost entirely the one vendored GEMM**.

### 2. Triton tile-config sweep (superset of wirbel's grid)

Swept **BLOCK_M ∈ {4,8,16,32,64}** (⊇ {4,8,16}), **TILE_SIZE ∈ {16,32,64,128}** (≡ wirbel's BLOCK_N, the KV-tile dim; ⊇ {16,32,64}), **num_warps ∈ {2,4,8}**, **num_stages ∈ {2,3,4}** = **180 configs**, each correctness-gated against the served-default output (max_abs_err ≤ 2e-3).

| Triton kernel (verify) | tunable tile surface? | result |
|---|---|---|
| **Triton-3D split-KV attention** (7 global layers) | **yes** (BLOCK_M/BLOCK_Q/TILE_SIZE/warps/stages) | best **BM16_BQ4_TS16_w4_s2**, **+6.11% kernel** (confirmed) |
| `rejection_greedy_sample_kernel` (sampling) | **no** — grid=(batch,), scalar loop over `max_spec_len` rows, no BLOCK_M/BLOCK_N constexpr | re-tile headroom structurally nil |

- **45/180 configs valid** (the rest are invalid launches for this geometry: BLOCK_M > query rows, shared-mem-busting stage counts, etc.).
- **The served default is `w4_s3`** (Triton's compile default; the kernel launch passes no explicit warps/stages). Confirmed: explicit `BM16_BQ4_TS16_w4_s3` = 10.772 µs ≈ `cfg=None` baseline 10.752 µs.
- **The win is exactly `num_stages` 3→2** (tile shape + warps unchanged). It is **systematic, not noise** — `s2` beats `s3`/`s4` at *every* warp count on the served tile (w2: 10.52 vs 12.78 µs · w4: **10.12** vs 10.77 µs · w8: 10.46 vs 11.09 µs); the top-8 configs are all `s2`. Physically: at M=8 the split-KV kernel is occupancy/shared-mem bound, so a shallower 2-stage pipeline frees shared memory and lifts occupancy.
- **Winner's-curse guard (added this leg):** selecting the min over 45 noisy configs biases the apparent speedup downward. I re-timed the winner head-to-head vs the served default, **7 reps alternating**: base median **10.786 µs**, best median **10.127 µs** → **+6.11%** (sweep-min was +5.84%). Samples are razor-tight (base [10.76–10.79], best [10.12–10.14], σ≈0.01 µs), so the win is **real and reproducible**, not selection bias (`winner_curse_shrinks_delta=false`). The honest mapping uses this **confirmed** median, not the sweep-min.

### 3. Honest kernel → end-to-end TPS mapping (avoiding the #433 / #437 trap)

Cycle anchor: `CYCLE_US = E[T(7)]/realized_TPS = 3.851186/467.14 = 8244.2 µs`; slope `TPS_PER_US = 467.14/8244.2 = 0.056663` TPS per µs of verify saved — treated as an **upper bound** (the realized cycle carries large fixed serving overhead that does not shrink, so the realized gain per µs is ≤ this).

| lever | verify saving | end-to-end TPS Δ |
|---|---:|---:|
| **best tile retune** (num_stages 3→2, confirmed +6.11%) | 0.659 µs/layer × 7 = **4.61 µs** | **+0.2613** |
| eliminate the ENTIRE tunable Triton-3D kernel (75.3 µs) | 75.3 µs | +4.27 (upper bound) |
| **wirbel #442 modeled +15.86** | **implies 279.9 µs** | claimed +15.86 |

- **`max_honest_endtoend_tps_delta = +0.2613 TPS`** — noise-level, far below the +2 TPS bar.
- **Adjudicating wirbel #442:** the +15.86 claim back-implies a **279.9 µs** verify saving. The entire tunable Triton-3D attention is **75.3 µs**; even *deleting* it (not retuning) caps at **+4.27 TPS**. **+15.86 is 3.7× larger than removing the whole kernel** → it cannot come from a tile retune. It is the same modeled-in-isolation trap as pinned-K #433 (modeled +13.998 → realized −5.82) and cb3 #437 (modeled +15.60 → realized 0.0). The **real** retune effect (the `num_stages=2` win wirbel's autotuner found) is genuine at the kernel level but **+0.26 TPS** realized.

### 4. Plain answers to the three questions

- **(a) Is int4-GEMM the dominant verify cost, and already optimal?** **Yes.** 85.1% of verify, vendored Marlin W4A16 CUDA with fixed internal tiling — **no selectable Triton tile surface**. It runs at ~73% of A10G HBM peak (HBM-read-bound on int4 weights) → re-tile-proof (consistent with #130).
- **(b) Any non-attention Triton kernel with >+2 TPS honest headroom?** **No.** The only other Triton kernel in verify is `rejection_greedy_sample_kernel` (4.95 µs, 0.08%), which has **no BLOCK_M/BLOCK_N tile surface** to sweep. Body + lm_head are 100% vendored Marlin. **Total non-attention Triton re-tile headroom is structurally nil.**
- **(c) Does my attention fraction corroborate #441 (6.90%) / #445 (9.28%)?** **On the absolute, yes — emphatically; on the fraction, no, and the difference is fully explained by the body-denominator basis.** My **T_attn = 838.8 µs @ ctx=128** matches #441's own ctx-128 sweep point **(836 µs, +0.3%)** to within measurement noise — the attention *numerator* corroborates perfectly. The *fraction* differs (14.2% vs 6.9%/9.28%) **only** because I measure the int4 body on the served **CUDA-graph (ONEGRAPH)** basis (**5030 µs**, GEMMs back-to-back as the served graph runs) while #441/#445 used **per-iter-sync** (≈16936 µs, inflated by per-op launch/sync overhead). Physical check: my body runs at ~444 GB/s ≈ 73% of A10G peak (HBM-bound, served-realistic); the per-iter-sync basis implies ~130 GB/s = 22% of peak, which is impossible for a BW-bound kernel → it is overhead-dominated. **14.2% is the served-realistic attention fraction; 6.9% understates it via an inflated denominator.** All three agree qualitatively: attention is a minority of verify, the int4-GEMM body dominates.

### Verdict fields
- `max_honest_endtoend_tps_delta` = **+0.2613** · `any_nonattn_triton_over_2tps` = **false** · `int4_gemm_dominant` = **true** (85.09%)
- `int4_gemm_vendored_marlin_no_selectable_tiling` = **true** · `attn_frac_of_verify` = **0.1419** · `attn_tunable_tri3d_frac_of_attn` = **0.0898**
- `attn_best_cfg` = **BM16_BQ4_TS16_w4_s2** · `attn_kernel_confirmed_speedup_pct` = **+6.108** (real, reconfirmed) · `winner_curse_shrinks_delta` = **false**
- `beats_realized_467` = **true** (+0.26) · `beats_deployed_481` = **false** · `wirbel_modeled_implied_verify_saving_us` = **279.9**
- `ppl` = **2.3772** (≤ 2.42; tile config / profiling cannot change emitted tokens — byte-exact greedy equivalence preserved by construction, land #420) · `self_test_passes` = **true** (16/16)

### Run details
- **Command:**
  ```bash
  cd target/ && CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    uv run python research/profiling/verify_wall_tile_scan/verify_wall_tile_scan.py \
      --iters 300 --warmup 100 --context-len 128 \
      --wandb_group kernel-tiling-sweep --wandb_name denken/verify-wall-tile-scan
  # 0-GPU self-test gate: --self-test --no-wandb (16/16 PASS)
  ```
- **W&B run id:** `crrq2e1y` (group `kernel-tiling-sweep`; metrics under `attn/`, `comp_us/`, `endtoend/`; tables `verify_decomposition`, `attn_tile_sweep_top64`).
- **Peak VRAM:** 0.529 GiB · **Device:** NVIDIA A10G (sm_86) · torch 2.11.0+cu130 / triton 3.6.0 / vllm 0.22.0.
- **No served-file change, no HF job, no submission.** `analysis_only=true`.

### What happened — honest analysis
The hypothesis asked where the verify wall actually goes and whether any of it is cheaply re-tile-recoverable. The answer is clean: **85.1% is one vendored kernel (int4-Marlin GEMM) with no Triton tile surface**, running HBM-bound at ~73% of A10G peak; **the only tunable Triton kernel in the whole verify path is the 7-layer Triton-3D global attention at 1.3% of verify.** wirbel #442's autotuner found a **real** signal in that kernel — `num_stages=2` is systematically ~6% faster than the served default `num_stages=3` at this M=8 decode shape (occupancy/shared-mem effect, reproducible to σ≈0.01 µs) — but wirbel reported the kernel-isolated Δ as if it were end-to-end. Mapped honestly through the cycle fraction, a +6.11% win on a 75 µs kernel is **+0.26 TPS**, and even annihilating the entire kernel is **+4.27 TPS** — so the modeled **+15.86** (implying a 280 µs saving) is impossible from re-tiling. This is the third confirmed instance of the modeled-in-isolation trap (#433, #437, now #442). The leg is a clean informative null for the realized frontier: **the verify wall has no cheap Triton re-tile headroom; it is HBM-bound vendored GEMM, and K=7 attention is M-flat (#441).** The realized frontier stays **467.14 (+0.26)** and does not cross the deployed **481.53**.

### Suggested follow-ups
- **Hand wirbel the honest mapping:** the `num_stages=2` Tri3D win is real and byte-exact-safe; if a future leg wants to *bank* it, the served `unified_attention` 3D launch could pass `num_stages=2` for the global head-512 decode path (a one-line served change, +0.26 TPS — below the noise floor of the public A/B, so likely not worth a served-file change on its own, but it is a *correct* micro-win to fold into any larger attention-kernel rebuild).
- The 85% int4-GEMM body is HBM-read-bound on int4 weights at ~73% of peak; the only physics that moves it is **fewer weight bytes** (a smaller/more-aggressively-quantized model — out of equivalence scope) or a **fused verify kernel** that overlaps GEMM with attention (stark #433/#437 own that human-gated build-prize). Neither is a tile-sweep lever.
- Future verify-cost models should adopt the **served CUDA-graph (ONEGRAPH) component basis** (body 5030 µs, attn 14.2%) rather than per-iter-sync (body ≈16936 µs, attn 6.9%): the graph basis is HBM-physical (~73% of A10G peak) and is how the served model actually runs; per-iter-sync inflates the GEMM denominator with launch overhead.

### Public evidence used
Measured against the public incumbent **PR #52 `2x9fm2zx`** (deployed 481.53 / PPL 2.3772 / strict byte-exact identity / 128-128). Anchors consumed (all advisor branch `approval-gated-8gpu-20260613`, cited in PR #447): realized equivalence frontier **denken #423 `5a6zq2yz`** (467.14), E[T(7)] ladder **#289 `fi34s269`** (3.851186), attention fraction **denken #441 `7rb089z3`** (6.90%, ctx-128 sweep 836 µs) and **stark #445 `emljqube`** (9.28%), the live lead under adjudication **wirbel #442 `e5n9a2dc`** (modeled attn-autotune +15.86 → 483.0, UNPROVEN realized). int4-GEMM re-tile-proofness cross-checked against **gate_up_retile #130**. The public valid-frontier digest (firfir-cast / frantic-penguin ~489.6 TPS; firfir-cast's "SplitKV BLOCK=64" tile change measured at **+0.03 TPS**) independently corroborates that verify-path tile re-tiling yields noise-level TPS.
