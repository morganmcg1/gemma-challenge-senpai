# PR #676 — int4-GEMV HBM-roofline ceiling (bounds the kernel axis)

**denken · group `gemv-hbm-roofline-denken` · run `vwiqwzvk` · LOCAL A10G
(sm_86) microbench · `analysis_only` · `official_tps=0` · `fires=false` · NO
served change / NO HF Job.**

## TL;DR verdict

**`GEMV_AT_HBM_WALL`.** The PR's back-of-envelope "~50% of spec peak" is a
**measurement artifact, not a 2× opportunity**. Measured at matched shapes in
isolation, the **dominant 86% of per-token bytes (gate_up + down + lm_head) run
at 469.3 GB/s = 90.7% of the empirical read-peak (517.7 GB/s)** — at the wall.
The GEMV is **bandwidth-bound** (M-invariant: M8/M1 = 1.020) and the **int4→bf16
dequant is fully hidden under the HBM read** (dequant-ALU tax on the dominant
shapes = **−2.1%**: int4 is *faster* than the bf16-weight control because it
moves 4× fewer weight bytes). The apparent ~50%-of-spec comes from three stacked
subtleties — (a) spec 600 is unreachable (empirical peak 517.7 = 86% of spec),
(b) more bytes move than the envelope (2.38 GB exact vs ~2.1 GB; 42 layers +
full-vocab head), (c) the #674 in-loop "matmul" bucket over-counts the GEMV by
**17%** (fused triton epilogues + per-layer launch misattributed to marlin).
**Byte-identically realizable headroom = 0.** Physics ceiling if *every* shape
magically hit read-peak is only **1.162×** and is **unrealizable** under #319. →
**lawine #675 is bounded (`ALREADY_OPTIMAL` near-certain); refocus speed effort
on spec-dec (kanna #673).**

## Deliverable 1 — roofline anchors (empirical peak + per-token bytes)

- **Empirical achievable HBM peak** (co-measured this A10G, 512 M-elem bf16):
  STREAM **read 517.7 GB/s**, copy 480.2 GB/s, cudaMemcpy DtoD 480.2 GB/s. The
  **600 GB/s datasheet is unreachable** — STREAM read itself is only **86.3%** of
  it. Read-peak is the correct denominator (a weight-GEMV is read-dominated).
  Independently corroborated: stark #602 read-peak 517.9, gemm_roofline 517.6.
- **Per-token bytes moved** (exact, from the served safetensors header,
  `/workspace/gemma_build/int4_g128_lmhead/model.safetensors`):
  - **body weight+scale = 2.034 GB** (int4-packed + g128 fp scales), summed over
    the **real** layer counts: 42 decode layers, k/v only on **24/42**
    (KV-sharing), gate/up/down ×42, PLE input-gate + projection ×42.
  - **lm_head = 0.346 GB** (full vocab 262144 × 2560, int4 + g128 scales).
  - **full GEMV path = 2.380 GB/token.** Activations at M=1 are 0.6 MB
    (0.02% — weight-read dominates; KV read is in the attn 11% leg, not the GEMV).
  - **The PR envelope (~2.1 GB) undercounts by ~13%** — it assumed 37 layers and
    omitted full-vocab head traffic. Using exact bytes shifts the floor up.

  | component | layers | MB/token | % of GEMV bytes |
  |---|---|---|---|
  | gate_up_proj | 42 | 1135.4 | 47.7% |
  | down_proj | 42 | 567.7 | 23.8% |
  | lm_head | 1 | 346.0 | 14.5% |
  | qkv_proj | 42 (k/v ×24) | 170.3 | 7.2% |
  | o_proj | 42 | 132.5 | 5.6% |
  | ple_proj | 42 | 28.4 | 1.2% |

## Deliverable 2 — achieved bandwidth + % of empirical peak

Two methods, reconciled (this reconciliation **is** the answer to the PR's
"~50%?" question):

- **In-loop bucket** (#674's 6920 µs/tok "matmul" ÷ 2.380 GB): **344.0 GB/s =
  66.4% of read-peak / 57.3% of spec.** This is the number behind the PR's
  ~50%-of-spec envelope. It is an **upper bound on time**, not the kernel BW.
- **Isolated kernel** (sum of per-shape marlin GEMV time at matched M=1 shapes,
  L2-cold, CUDA-graph replayed): full path **5735 µs ⇒ 415.0 GB/s = 80.2% of
  read-peak / 69.2% of spec**; body-only **407.2 GB/s = 78.7%**.
- **Gap = 1184 µs = 17.1% of the #674 bucket** — fused triton epilogues
  (rms_norm/gelu) name-matching "marlin_gemm" + per-layer launch/interleave
  overhead folded into the in-loop matmul bucket. Removing it, the GEMV is far
  closer to the wall than the in-loop bucket implies.
- **Headline (decisive): the dominant 86.1% of bytes = 469.3 GB/s = 90.7% of
  read-peak** — at the wall. The aggregate 80.2% is dragged down only by small
  under-saturated projections (see Deliverable 3).

  | shape | int4 µs (M=1) | GB/s | % read-peak | % spec |
  |---|---|---|---|---|
  | gate_up_proj | 56.4 | 480.4 | **92.8%** | 80.1% |
  | lm_head | 741.2 | 467.6 | **90.3%** | 77.9% |
  | down_proj | 30.1 | 449.7 | **86.9%** | 75.0% |
  | qkv_proj | 13.5 | 300.1 | 58.0% | 50.0% |
  | o_proj | 9.3 | 292.2 | 56.4% | 48.7% |
  | ple_proj | 4.2 | 81.4 | 15.7% | 13.6% |

- **`gemv_achieved_bw_gbps = 415.0`**, **`gemv_pct_of_hbm_peak = 80.2%`** (vs
  empirical read-peak; **69.2% vs 600 spec**). Dominant-byte
  `dominant_pct_of_read_peak = 90.7%`.

## Deliverable 3 — BW-vs-ALU decomposition (the crux: bandwidth, not dequant)

Two independent reads both say **bandwidth-bound, dequant hidden**:

- **bf16-weight GEMV control** (`F.linear`, same shapes, no dequant) isolates the
  dequant-ALU tax. On the **dominant shapes the tax is −2.1%** — int4 is
  *faster*, because at the wall what matters is bytes read and int4 moves 4×
  fewer weight bytes; the dequant ALU runs entirely under the HBM read.

  | shape | int4 GB/s | bf16 GB/s | dequant-ALU tax |
  |---|---|---|---|
  | gate_up_proj | 480.4 | 463.0 | **−3.7%** |
  | down_proj | 449.7 | 440.1 | **−2.2%** |
  | lm_head | 467.6 | 483.2 | +3.2% |
  | qkv_proj | 300.1 | 419.2 | +39.7% |
  | o_proj | 292.2 | 356.7 | +22.1% |
  | ple_proj | 81.4 | 206.9 | +154% |

  The big positive taxes are **only on the small projections** — and those are
  **occupancy/wave-quantization limited at M=1, not byte-identically
  recoverable**: the bf16 control *also* under-saturates them (qkv 81%, o 69%,
  ple 40% of read-peak), so even deleting the dequant entirely leaves them below
  peak. Recovering them needs a retile (split-K / BLOCK_K / num_warps) that
  changes the K-reduction order → **breaks greedy byte-identity (#319)** (stark
  #602 proved exactly this: fp32-reduce-off bit-flips qkv/o/down). They are 14%
  of bytes and structurally unreachable here.
- **M-invariance** (gate_up, M∈{1,2,4,8}): 480.7 → 480.0 → 478.6 → 477.0 GB/s,
  **M8/M1 = 1.020 (flat)**. A latency-bound kernel with a free tiling fix would
  get *faster* per-token as M rises; flat ⇒ already **bandwidth-bound at M=1**.
  Body time only rises past the A10G compute knee (M≈52), the textbook
  memory→compute roofline transition. AR decode lives at M=1, deep in the flat
  bandwidth plateau.

→ The dominant GEMV is **memory-bandwidth-bound at ~91% of achievable peak**, not
dequant-ALU-bound. There is no "int4 dequant is the bottleneck" lever.

## Deliverable 4 — official-equiv TPS ceiling + byte-identical realizability

- **Physics floor** (full GEMV at read-peak): 2.380 GB ÷ 517.7 GB/s = **4598 µs**
  vs measured 5735 µs ⇒ nominal save 1137 µs. **Speedup if every shape hit
  read-peak = 1.162×** (basis-independent).
- **Official-equiv ceiling.** Applying the speedup to the **live** anchor:
  126.378 × 1.162 = **146.9 TPS (live-basis)**. The literal stark-tax basis
  (0.870 × local 142.83) = **124.3 TPS** — but note 0.870 × local-now (122.87) =
  106.9 < live 126.378, i.e. **the 0.870 tax is conservative/breaks down at this
  operating point** (live official exceeds local), so the live-basis 146.9 is the
  meaningful upper bound and 124.3 is the conservative literal. The **speedup
  factor 1.162× is the robust number**; the absolute basis is ambiguous by the
  known tax gap.
- **Realizability: ZERO byte-identically.** The 1.162× is **unrealizable** — it
  assumes (a) closing the 600→518 silicon gap (impossible), (b) the dominant
  shapes going 91%→100% (already at the wall; the residual is unreachable
  silicon), and (c) the small shapes going to peak via a retile that **breaks
  #319 greedy identity**. `headroom_is_byte_identical = False`. **The operational
  ceiling = the current 126.378.** No byte-identical body-GEMV speed lever exists
  for the strict-AR lane.

## Why this bounds lawine #675

lawine #675 owns *which byte-identical kernel/config is empirically fastest*. This
card supplies the **ceiling that sweep is searching under**: the active marlin
GEMV is already at **90.7% of achievable read-peak on the 86% of bytes that
matter**, M-invariant, with dequant hidden. Any byte-identical kernel variant can
only chase the remaining unreachable residual (unreachable silicon + occupancy on
14% of bytes that even bf16 can't saturate + identity-breaking retiles). So
lawine #675 is **bounded — `ALREADY_OPTIMAL` near-certain** — and the AR-speed leg
is fundamentally capped. The +10-over-126.378 hunt belongs to the spec-dec axis
(kanna #673), not the kernel axis. Converges with stark #602 (body 84.2% of
read-peak, no byte-identical lever) and lawine #591 (0.0 reclaimable from a fresh
M=1 bandwidth axis).

## Anchors / reproduction

- rung to beat: `int4_g128_lmhead` @ **126.378** official TPS, PPL 2.019, 128/128
  (strict AR). +10 ⇒ ≥136.4. PPL/greedy untouched (`analysis_only`, no served
  change).
- matched AR anchor: #674 `decode_overhead_graph_audit` matmul **6919.9 µs/tok =
  85%**, local **122.87 TPS** (BI=1, within ±5% of g128_AR 126.94); local→official
  tax 0.870 (stark).
- empirical peak co-measured here: read **517.7** / copy 480.2 / memcpy 480.2 GB/s
  (spec 600 unreachable). Priors: stark #602 `ymss9maz` body 436.3 GB/s = 84.2%;
  gemm_roofline read-peak 517.6.
- cmd: `CUDA_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=1 uv run python
  research/speed/gemv_hbm_roofline_ceiling/gemv_hbm_roofline_ceiling.py`
- W&B `vwiqwzvk` (group `gemv-hbm-roofline-denken`); summary scalars
  `analysis_only=1`, `official_tps=0`, `fires=0`, `verdict=GEMV_AT_HBM_WALL`,
  `gemv_pct_of_hbm_peak=80.17`, `dominant_pct_of_read_peak=90.66`,
  `dominant_dequant_alu_tax_frac=-0.0211`, `m_invariance_m8_over_m1=1.020`,
  `ceiling_speedup_if_kernel_hit_peak=1.162`, `self_test_passes=1`.
- peak VRAM well under the 22.06 GiB card (~2.5 GB working set); self-test 8/8 PASS.
```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["vwiqwzvk"],"primary_metric":{"name":"dominant_pct_of_read_peak","value":90.66},"test_metric":{"name":"gemv_pct_of_hbm_peak","value":80.17}}
```
