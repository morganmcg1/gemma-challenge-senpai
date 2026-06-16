STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["ys7nwwi2","8lz3pbfb","xgwfhi6q","qe8rbacs"],"primary_metric":{"name":"warm_median_tps","value":442.47},"test_metric":{"name":"ppl","value":2.376981}}

## Results — byte-exact split-KV NUM_SEGMENTS occupancy sweep

**Verdict: FLAT. NUM_SEGMENTS is NOT the 442→457 lever.** The occupancy-optimal byte-exact
segment count is **S=64 → 442.47 warm-median TPS** — i.e. exactly #519's value (+0.12 vs the
442.35 baseline, well under σ_hw=4.864). Raising S only makes it *slower*, monotonically. The
realization gap is structurally not a segment-count effect (mechanism below).

### KEY OUTPUTS (PR-required)
- `best_byteexact_s` = **64**
- `best_s_warm_median_tps` = **442.47**
- `best_s_vs_519_delta` = **+0.12** (vs 442.35; vs in-session S=64: 0.00)
- `best_s_vs_surgical357_delta` = **+85.41** local / **+66.61** vs official ship 375.857
- `all_swept_configs_byteexact` = **true** (self-det 1.0, mechanism_valid, 0 flips, every S)
- `all_swept_configs_ppl_pass` = **true** (PPL 2.3770 ≤ 2.42, every S)
- `reaches_457` = **false** | `beats_519_by_sigma` = **false** (σ_hw=4.864)

### Per-S table (full 128×512, seed 1, 3 decodes, warm = median of rounds 1–2)
| S | coverage (64·S) | max KV | active seg | warm-med TPS | PPL | self-det | 128/128 | byte-exact | peak MiB | GATE |
|----:|----:|----:|----:|----:|----:|:----:|:----:|:----:|----:|:----:|
| **64** | 4096 | 2939 | 46 | **442.47** | 2.376981 | 1.0 | ✓ | ✓ | 21655 | **PASS (best)** |
| 128 | 8192 | 2939 | 46 | 390.60 | 2.376981 | 1.0 | ✓ | ✓ | 21453 | PASS |
| 256 | 16384 | 2939 | 46 | 288.63 | 2.376979 | 1.0 | ✓ | ✓ | 21099 | PASS |

All three configs pass every hard gate (self-det = 1.0, PPL ≤ 2.42, 128/128, byte-exact
mechanism). They are all valid — S=64 is simply the fastest, and it is also the smallest
runnable S for this workload.

### Why the suggested {48, 96} were dropped (single-variable, FIXED_TPS=4 held)
Two hard kernel bounds leave only powers of 2 ≥ 46, i.e. **{64, 128, 256}**:
1. **Power-of-2:** the byte-exact patch sets `triton_attn.NUM_PAR_SOFTMAX_SEGMENTS = S`, and the
   stock `reduce_segments` kernel does `tl.arange(0, NUM_PAR_SOFTMAX_SEGMENTS)`. Triton requires
   an arange range to be a power of 2 → S=48 / S=96 raise `ValueError` ~80s into engine init
   (confirmed by an S=48 smoke: server exits code 1 before readiness).
2. **Coverage floor:** `coverage = FIXED_TPS·S·TILE_SIZE = 64·S` keys must cover the longest
   decode KV. Measured from #519's decode artifacts: **max KV = 2939** (prompt 2427 + 512,
   id `gpqa_diamond-1d37a7a51d`), mean 784. Safe floor `S ≥ ⌈2939/64⌉ = 46`; S=32 (cov 2048)
   under-covers and would silently drop the context tail.

So the smallest valid S is **64 — already #519's value** — and the sweep can only explore
*upward*. I added a fail-fast power-of-2/coverage pre-check to the harness so an invalid S is
rejected before an 80s server boot.

### Mechanism: why upward is flat-to-slower (byte-exactness preserved throughout)
Byte-exactness comes from pinning `tiles_per_segment = T` (FIXED_TPS=4), **not** from S. Each
parallel-softmax segment therefore covers a FIXED 64-key span at a fixed absolute key position
(M-invariant reduction order). The 3D grid is `(q_blocks, kv_heads, S)`, and segment `segm_idx`
early-exits when `segm_idx·64 ≥ seq_len`. So the number of **active** (real-work) segments is
`⌈seq_len/64⌉` — pinned by seq_len and T, **independent of S** (mean ~13, max 46 here).
`reduce_segments` masks to that same active set, so the result is **byte-identical for every
S ≥ 46** (confirmed: PPL/self-det identical across S; the tiny S=256 PPL drift in the 6th decimal
is the round-to-even float noise of a wider masked reduction, not a flip — 0 token flips). Raising
S only adds **idle early-exit grid blocks** and a wider scratch/reduction, which is pure overhead
(FlashDecoding SM-fill optimum is ~SMs/(batch·heads) ≪ 64). Hence TPS falls 442→390→288 as S
goes 64→128→256. Peak mem is ~flat (21.7→21.1 GiB, absorbed by the 0.90 pre-reservation).

### Command
```bash
cd target/ && .venv/bin/python -m research.validity.splitkv_numseg_sweep.sweep_numseg \
    --segments 64 128 256 --wandb-group splitkv-numseg-sweep
```
LOCAL only: `analysis_only=true`, `official_tps=0`, **no HF Job / no `--launch` / no submission**
(challenge PAUSED). Peak GPU mem: **21655 MiB** (S=64). Single A10G.

### W&B (group `splitkv-numseg-sweep`)
- S=64: [`ys7nwwi2`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/ys7nwwi2) ·
  S=128: [`8lz3pbfb`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/8lz3pbfb) ·
  S=256: [`xgwfhi6q`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/xgwfhi6q)
- summary: [`qe8rbacs`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/qe8rbacs)
  (per-S table + sweep_summary artifact)

### What happened
The hypothesis (the 442→457 residual is occupancy left on the table at S=64) is **refuted as a
NUM_SEGMENTS effect**. Because byte-exactness pins `tiles_per_segment`, the active-segment count
that does the FlashDecoding split is fixed by the workload KV length, not by S — so S cannot
re-tune occupancy without breaking the byte-exact invariant. S=64 is already at (in fact below)
the SM-fill optimum, and it is simultaneously the coverage floor, so there is no faster byte-exact
S to find. This is #481 priority-#1 datapoint #2: the realization gap lives elsewhere.

### Suggested follow-ups
- The remaining 442→~457 must come from a knob **orthogonal to segment count**: e.g. the cold→warm
  delta is only ~3 TPS, so it's steady-state decode throughput, not capture/warmup. Candidates that
  keep the byte-exact reduction order: TILE_SIZE / block-M of the decode kernel, the reduce
  dtype/epilogue, or CUDA-graph/loopgraph capture coverage of the split-KV path. Each is a separate
  single-variable PR.
- If the frontier map's ~457 assumed a *different* FIXED_TPS (active-segment granularity), a
  **FIXED_TPS** sweep (T ∈ {2,4,8}, re-checking byte-exactness per T since T changes the reduction
  order) is the natural sibling experiment — but that moves the byte-exact invariant, so it is NOT
  single-variable here and needs its own identity re-cert.
