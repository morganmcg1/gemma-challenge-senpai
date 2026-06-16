STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["7rb089z3"],"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"t_attn_frac_of_verify":0.068960,"k_opt":7,"tps_at_kopt":467.14,"kopt_beats_k7":false,"kopt_lift_tps":0.0,"kopt_beats_deployed_481":false,"ppl":2.3772,"self_test_passes":true,"primary_metric":{"name":"tps_at_kopt","value":467.14},"test_metric":{"name":"self_test_passes","value":1}}

## Results

**Verdict (one line):** verify-attention is **6.90%** of verify wall time and **~M-INVARIANT** (T_attn(M=8)/T_attn(M=4)=**1.005**; deployed 3D split-KV + FA2 are KV-reduction/KV-load bound, not query-row bound) → **K_opt=7**: VALIDATES the deployed K=7 with hard data — the M-invariant verify model behind K=7 was correct. **Does NOT beat the deployed 481.53** (the realized equivalence frontier stays 467.14). **kopt_lift_tps = +0.00.**

The hypothesis's load-bearing premise — *"the served attention kernel M-SCALES, attending M=1+K tokens is NOT M-invariant"* — is **refuted by direct measurement**. Attention does ride on the KV cache, but the heavy term is the KV reduction (M-invariant), not the per-query-row work.

### 1. T_verify decomposition (A10G sm_86, CUDA-graph attn / CUDA-event GEMM, 500 iters / 200 warmup)

| M=1+K | T_attn (µs) | T_body (µs) | T_lmhead (µs) | T_verify (µs) |
|------:|------------:|------------:|--------------:|--------------:|
| 1     | 1243.33     | 16857.1     | 72.70         | 18173.12      |
| 4 (K3)| 1253.22     | 16890.9     | 73.58         | 18217.69      |
| 5 (K4)| 1254.63     | 16902.1     | 73.87         | 18230.65      |
| 6 (K5)| 1254.79     | 16913.4     | 74.17         | 18242.36      |
| 7 (K6)| 1255.55     | 16924.7     | 74.46         | 18254.68      |
| 8 (K7)| 1259.95     | 16935.9     | 74.75         | 18270.64      |

- **`t_attn_frac_of_verify` (M=8) = 6.90%.** This is **>5%**, so per the instructions I did *not* stop at the byte-fraction gate — I ran the full K-sweep. But the 5% gate was a proxy for *"is attention big enough that its M-scaling could move K?"* The real determinant is whether attention **M-scales**, and the direct measurement says it does not.
- **`attn_mscaling` = T_attn(8)/T_attn(4) = 1.005** → flat. **`verify_mscaling` = T_verify(8)/T_verify(4) = 1.003** → flat. Body GEMM is M-invariant as cb3 #437 found (int4-Marlin weight-read reused across rows: 16857→16936 µs, +0.47% over M=1→8).
- **cb3 cross-check:** cb3 #437 carried attention as a *banked* constant (F_ATTN imported from #388's M=1, never re-measured with the served backends). As a fraction of verify-forward that banked value is **10.80%**. My **direct** measurement with the actual served FA2+Triton-3D backends is **6.90%** → the direct measurement **refines the banked constant DOWN** (FA2 over the 35 sliding layers is cheaper than the coarse banked Triton estimate). K=7 is validated with *more* margin than cb3 assumed, not less.

### Why attention is M-invariant (per-backend, M=1 → M=8)

| backend (layers) | M=1 µs | M=8 µs | Δ |
|---|---:|---:|---:|
| FA2 head-256, 8Q/2KV (×35 sliding, FA_SLIDING=1) | 33.26 | 33.42 | **+0.5%** (dead flat) |
| Triton-3D split-KV head-512, 8Q/2KV (×7 global, SPLITKV_VERIFY=1) | 11.32 | 12.90 | +13.9% (but 1.6µs×7 absolute) |

The bulk 35-layer FA2 term is **flat** — paged decode at small M is pure KV-load. All of the tiny attention M-drift lives in the 7 global Triton-3D layers, and even there it is 1.6 µs/layer. **Context-length sweep confirms the binding axis** (T_attn at M=8): ctx 128 → **836 µs**, ctx 256 → **1260 µs**, ctx 512 → **2118 µs** — attention scales with **KV length** (≈linear), not with **M**. That is the physical signature of a KV-reduction-bound kernel: the M query rows ride along on one KV load.

### 2–3. T_cycle(K) → TPS(K) (E[T(K)] banked #289 `fi34s269`, anchored TPS(7)=467.14)

| K | E[T(K)] | TPS amort | TPS floor | TPS realized (static_k) |
|--:|--------:|----------:|----------:|------------------------:|
| 3 | 2.7224  | 332.69    | 338.38    | 398.67 |
| 4 | 3.0838  | 376.16    | 380.96    | 426.83 |
| 5 | 3.3855  | 412.23    | 415.71    | 450.99 |
| 6 | 3.6377  | 442.13    | 443.99    | 460.38 |
| 7 | 3.8512  | **467.14**| **467.14**| **467.14** |

- `T_draft` cost bracketed two ways: **in-graph amortized** = F_DRAFT·STEP/K = 20.9 µs/pass (the regime the realized STEP_US=1218 lives in) and **standalone bf16 floor** = 101.2 µs/pass (#254). Measured drafter attention/pass = 38.2 µs (4 layers, 4Q/2KV).
- **k_opt = 7 in all three independent models** (amort forward, floor forward, realized wall-clock) — triple agreement. Because T_verify is M-flat, T_cycle(K) is dominated by the M-invariant body+lmhead; dropping K saves almost nothing in verify latency while strictly losing E[T] (monotone), so TPS is monotone increasing in K up to K=7.

### Reconciliation with the realized wall-clock (static_k_wallclock_ab)

The forward-only PR model omits the large fixed serving overhead (~5.8× forward time, the static_k finding). That overhead does **not** shrink when draft passes drop, so the realized curve is even *steeper* in K's favor (realized TPS(3)=398.67 vs amort 332.69). Both the forward-only model (which is an *upper bound* on how low K_opt could go) and the realized A/B agree on **k_opt=7** → the null is doubly robust.

### Verdict fields
- `t_attn_frac_of_verify` = **0.0690** · `k_opt` = **7** · `tps_at_kopt` = **467.14**
- `kopt_beats_k7` = **false** · `kopt_lift_tps` = **+0.00** · `kopt_beats_deployed_481` = **false**
- `ppl` = **2.3772** (≤2.42; a K change is teacher-forced PPL-neutral — verify is the byte-exact arbiter, land #420 `qe4qagc1`) · `self_test_passes` = **true** (13/13)

### Run details
- **Command:**
  ```bash
  cd target/ && CUDA_VISIBLE_DEVICES=0 python research/validity/verify_attn_mscaling_kopt/verify_attn_mscaling_kopt.py \
    --iters 500 --warmup 200 --context-len 256 \
    --wandb_group verify-attn-kopt --wandb_name denken/verify-attn-mscaling-kopt
  # 0-GPU self-test gate: --self-test --no-wandb (13/13 PASS)
  ```
- **W&B run id:** `7rb089z3` (metrics under `summary/` prefix; tables `tps_vs_k`, `attn_mscaling`)
- **Peak VRAM:** 0.47 GiB · **Device:** NVIDIA A10G (sm_86, 80 SM)
- **No served-file change, no HF job, no submission.** `analysis_only=true`.

### What happened — honest analysis
The single primary untested assumption in the TPS model (does verify-attention M-scale?) is now **directly measured and falsified**. The served verify attention is the deployed FA2 (35 sliding head-256) + Triton-3D split-KV (7 global head-512), and **both are KV-bound at the M=1+K decode widths** — the M=1+K query rows ride along on one KV load, so widening the verify batch from M=1 to M=8 costs +1.3% of attention (+0.5% on the bulk FA2 term). Attention is only 6.9% of verify to begin with (less than cb3's banked 10.8%), and that 6.9% is flat. With T_verify M-flat, there is no latency to recover by shrinking K, while E[T] falls monotonically — so K=7 is the TPS-maximizing K. This is the clean informative-null the PR framed: it **validates the deployed K=7 with hard data**. It does not lift the realized frontier (stays 467.14) and does not cross the deployed 481.53.

### Suggested follow-ups
- The realized frontier (467.14) and the deployed incumbent (481.53) gap is **not** a K-axis phenomenon — K is exhausted. The remaining realized levers are the fixed serving overhead (static_k: ~5.8× forward) and a human-gated fused verify kernel (stark #433/#437 own that build-prize); neither is in this leg's boundary.
- If a future leg wants to *raise* the frontier via acceptance (not latency), that is the demand axis (land #436/#439, E[T] ceiling 520.95 @ λ=1) — a higher a₁ (the position-1 cliff, #289) would raise E[T(7)] and is orthogonal to the K choice measured here.
- The direct attention measurement (6.90% of verify, M-flat) is a better constant than cb3's banked 10.80% M=1 import — future verify-cost models should adopt the measured value.

### Public evidence
Measured against the public incumbent **PR #52 `2x9fm2zx`** (deployed 481.53 / PPL 2.3772 / identity 0.9966 / 128-128). Anchors consumed: realized equivalence frontier **denken #423 `5a6zq2yz`** (467.14), E[T] ladder **#289 `fi34s269`** (E[T]=3.851, a₁=0.7293), verify-BW λ=1 wall **land #436 `nvsbctji`** (520.95). Banked verify decomposition cross-checked against **stark cb3 #437** (F_ATTN=0.0951).
