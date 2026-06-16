STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["w41knrqd"],"primary_metric":{"name":"multistream_hideable_us","value":287.29},"test_metric":{"name":"multistream_strict_tps_ceiling","value":474.44}}

## Results

**Verdict: `overlap_is_real = True`. There IS real hideable overlap — a gated multi-stream served change could push the strict frontier from 457.55 toward ~474 TPS while staying byte-exact. 457.5 is NOT the hard single-stream ceiling.** The strict verify-attention is SM/latency-bound, not DRAM-BW-bound, so it does not compete with the body GEMM for bandwidth — a concurrent second stream hides **64% of it**.

This is the direct two-stream micro-probe (instruction 4, strongest evidence), run on the **exact stark #472 kernels**: the real 37-layer self-built g=128 int4-Marlin `body_gemm` (`apply_gptq_marlin_linear` → `ops.marlin_gemm`, the deployed GEMM) and the 7 served-Triton full-attn hd=512 reductions (`iso_strict` = natural M=8 2D order-preserving == `VLLM_BATCH_INVARIANT=1`; `iso_perm` = deployed 3D split-KV). Each CUDA-graph captured (private pools), timed solo vs concurrent on two CUDA streams (fork/join, paired per round, N=21).

### Headline (L=640, deployed KV)
| quantity | value |
|---|---|
| **`multistream_hideable_us` (PRIMARY)** | **287.3 µs (71% of the +401.9 µs/cycle strict tax)** |
| **`multistream_strict_tps_ceiling` (TEST)** | **474.44 TPS** (σ band 474.37–474.50) |
| gain vs single-stream 457.55 | **+16.89 TPS** (clears σ_hw 4.864 by 3.5×; clears the 467.14 composed ceiling) |
| `gemm_body_us` | 4125.9 µs (#450 anchor 4152.96, guard ✓) |
| `strict_attn_us` | 556.9 µs (serial tax 422.9 µs reproduces #466 422.9, guard ✓) |
| `overlap_fraction_strict` | **0.6449** (64% of the strict attention hidden under the GEMM) |
| residual strict tax under 2 streams | 114.6 µs (σ 1.05) — down from 401.9 µs single-stream |
| `gemm_dram_bw_util` | 0.85 (440 GB/s / 518 read-peak) |
| `gemm_sm_occupancy` (AI/ridge proxy, ncu N/A) | 0.145 → **~86% SM compute-headroom** |
| `attn_is_sm_bound` | **True** (35 GB/s = 6.7% of peak — not bus-competing) |
| `ppl` | 2.3772 (pinned by construction; profiling cannot move tokens) |
| strict byte-identity / flips | **1.0000 / 0** (launch contract held); permissive 0.0000 (reproduces non-equivalence) |
| peak VRAM | 2.51 GiB |

### Why it overlaps — complementary resources (instruction 2)
- **Body GEMM is DRAM-BW-bound:** M=8 → arithmetic intensity 30 ≪ ridge 208 FLOP/byte; achieved 440 GB/s = 85% of measured read-peak, ~86% of the SMs idle-waiting on memory.
- **Strict verify-attention is SM/latency-bound:** moves only ~19 MB (Q/K/V at M=8, hd=512, 7 layers) → 35 GB/s = **6.7% of peak**. It fills the GEMM's idle SM cycles instead of fighting for the bus.
- **Methodology smoking gun:** two BW-bound GEMMs on two streams reach only **1.14× speedup** (they serialize on the bus), yet GEMM‖attention hides 64% — so the overlap is *real complementary-resource concurrency*, not a measurement artifact. (`symmetric_gemm_serializes` guard ✓.)

### Two ceilings bracketed (honest)
- **474.44 TPS** (PRIMARY, conservative): residual strict-over-permissive tax when *both* arms use the 2nd stream — `(body‖strict − body) − (body‖perm − body)` = 114.6 µs. Cancels the common attention baseline + fork/join overhead.
- **477.58 TPS** (secondary, vs the literal deployed permissive-*serial* incumbent): strict exposes 197.5 µs vs deployed perm-serial 134.0 µs → added 63.5 µs.

Both clear 467. **Neither reaches deployed 481.53** (`reaches_deployed_481=False`) — the 7 attentions are not *fully* hidden (64%, not 100%), so a residual tax remains.

### KV-length sweep (overlap fraction is stable; ceiling rises as the tax shrinks)
| L | body µs | iso_strict µs | overlap_frac | `multistream_hideable_us` | ceiling TPS | sym speedup |
|---|---|---|---|---|---|---|
| 128 | 4127.1 | 159.5 | 0.62 | 390.0 | **480.79** | 1.14 |
| 384 | 4128.4 | 360.1 | 0.64 | 338.5 | **477.58** | 1.14 |
| 640 | 4125.9 | 556.9 | 0.64 | 287.3 | **474.44** | 1.14 |

### Comparison to baseline
- Strict single-stream frontier (the bar, #472 `wfggu51k`): **457.55** → two-stream **ceiling 474.44** (+16.9, clears σ_hw 4.864).
- Composed ceiling 467.14 (#423, refuted as single-stream realizable): **cleared** by the two-stream ceiling.
- Deployed non-equivalent 481.53 (#52): not reached; multi-stream stays byte-exact (the strict reductions) and lands ~7 TPS below.

### Self-test: PASS (23/23)
All guards green incl. `iso_tax_reproduces_466`, `body_gemm_anchor_matches_450`, `symmetric_gemm_serializes`, `ms_added_le_iso_tax`, `strict_byte_exact`, `strict_zero_flips`, `permissive_reproduces_nonequiv`, `strict_captured_survives` (the M=8 strict reduction captures+replays — no M=1 collapse).

### Exact command
```
cd target/ && CUDA_VISIBLE_DEVICES=0 \
  /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/speed/multistream_overlap_probe/multistream_overlap_probe.py --self-test --no-wandb
# W&B (repo .venv — GPU tool-venv has no usable wandb):
cd target/ && .venv/bin/python \
  research/speed/multistream_overlap_probe/wandb_log.py \
  --json research/speed/multistream_overlap_probe/multistream_overlap_probe.json \
  --wandb_group equivalence-escalation-anchors --wandb_name lawine/multistream-overlap-probe
```
- Device: NVIDIA A10G sm_86, torch 2.11.0+cu130. `analysis_only=true`, `official_tps=0`, `no_served_file_change=true`, `no_kernel_rebuild=true`. No HF job, no submission, no `--launch`.
- W&B: `w41knrqd` (group `equivalence-escalation-anchors`).

### What happened
The hypothesis holds: the strict tax is hideable because it is the *right kind* of work. The deployed ONEGRAPH serializes body-GEMM → attention on one stream, so the strict reduction's +401.9 µs sits fully on the critical path. But the GEMM leaves ~86% SM headroom (it's memory-stalled), and the strict attention is a low-bandwidth, latency-bound kernel — exactly the co-runner that fills those idle SM cycles. The direct two-stream measurement confirms 64% of it disappears under the GEMM, and the symmetric GEMM‖GEMM control (1.14×, bus-serialized) proves the concurrency is real rather than a clock/timing illusion. Byte-exactness is untouched (strict identity 1.0000, 0 flips) because we only change *where/when* the reductions run, not the reduction order.

**Important caveat (what the probe does and does NOT show):** this measures *resource feasibility* — the bus and SMs have complementary slack for the two kernels to overlap, giving a **ceiling** of ~474 TPS. It does **not** model the per-layer data dependency a real served schedule must respect: layer L's attention output feeds layer L's `o_proj`/MLP, so a faithful multi-stream graph would overlap each layer's attention with *subsequent* layers' GEMMs (software-pipelined), and the realizable fraction is bounded by that dependency structure, not just by resource headroom. So 474 is an upper bound on a gated served implementation, not a drop-in number.

### Suggested follow-ups
1. **Scope a gated two-stream served graph** (the natural next PR): put the 7 full-attn strict reductions on a side CUDA stream with correct cross-stream event barriers (verify must consume each layer's attention output before that layer's `o_proj`), software-pipelined across layers. Measure the *dependency-constrained* realized fraction vs this 474 ceiling. This requires a served-file change → human-gated, out of scope here.
- 2. **Stress the overlap at higher verify width / batch:** at M=8 the GEMM is deeply BW-bound; if a future drafter raises M, the GEMM gains arithmetic intensity (less SM headroom) and the overlap fraction would shrink — worth bracketing M∈{8,16} so the scoping PR knows the operating envelope.
- 3. **Per-layer interleave probe:** instead of one body-block ‖ one attention-block, capture a per-layer `[GEMM(L) ‖ attn(L-1)]` pipelined graph to directly estimate the dependency-bounded realizable overlap (closer to a served schedule than this resource-ceiling probe).
