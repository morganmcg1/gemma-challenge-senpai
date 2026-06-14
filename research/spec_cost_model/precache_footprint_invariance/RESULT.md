# PRECACHE_BENCH tree-footprint calibration-invariance (PR #169)

**Verdict: `bus_ratio_tree_invariant = 1`. K_cal = 125.268 transfers cleanly linear → M=32 tree. The 537.8 / 510.6 projections stand, unmoved (`official_shift_tps = 0.0` both topologies).**

This closes the last unverified link in the denominator→official compose chain feeding the
launch-readiness packet: the only term that was *asserted* invariant (the +6.019% local→official
multiplier, named as "a hardware bus ratio held invariant by `PRECACHE_BENCH=1`" in my #148) is now
*measured* invariant across the linear-stack footprint and the M=32 tree's 20.47 GB resident working
set. **Scope is a LOCAL single-A10G proxy — the TRUE multiplier needs the official environment and
cannot be closed without a launch. This banks the analysis; it does not authorize a shot.**

## What was measured

The +6.019% multiplier is a *throughput ratio* (481.53 / 454.1937). It rides the achievable WARMED
HBM transfer rate of the timed decode read. If the byte-identical decode-step read transfers at the
same GB/s whether the small linear stack or the large M=32 tree footprint is resident, the
multiplier's invariance assumption survives the footprint and K_cal transfers.

The rig holds **per-step geometry fixed** (the same `chunk_bytes` contiguous bfloat16 read — the
model-faithful dtype, 86% of the A10G's ~600 GB/s roofline) and varies **only the resident
footprint** by rotating over more allocated chunks. Per repeat: quiesce → cold ramp window
(the `PRECACHE_BENCH=0` analogue) → warmup replay (the `PRECACHE_BENCH=1` analogue, drives the
identical kernel to steady state) → warmed steady-state window. Local boost clocks are pinned
(1710/6251 persistent — the "#148 1710 pin"), which controls clock-ramp *out* of the measurement so
the residual bus_ratio is the footprint-relevant residency/launch transient. Clock-ramp is
footprint-independent by construction anyway.

- **Load-bearing gate:** `d_bw` — the shift in achievable warmed bandwidth (what the throughput
  multiplier rides). Invariance ⇔ `|d_bw| ≤ 0.787%` (#148's one-sided transfer band).
- **Corroboration:** `d_ratio` — the warm/cold transient delta. Clock-pinned-small and noisier, so
  it is *not* the gate, but a within-band `d_ratio` strengthens the conclusion.

## Footprint sweep (7 repeats each, A10G, idle clocks 1695/6251 MHz)

| footprint (GB) | warmed_bw (GB/s) | % roofline | bus_ratio | cold_first |
|---:|---:|---:|---:|---:|
| 4.00 (linear anchor) | 513.6 ± 0.1 | 86% | 1.037 ± 0.003 | 4.0 ms |
| 6.00 | 513.6 ± 0.1 | 86% | 1.037 ± 0.002 | 4.0 ms |
| 10.00 | 513.6 ± 0.1 | 86% | 1.037 ± 0.002 | 4.0 ms |
| 14.00 | 513.6 ± 0.1 | 86% | 1.036 ± 0.001 | 4.0 ms |
| 18.00 | 513.6 ± 0.1 | 86% | 1.036 ± 0.001 | 4.0 ms |
| **20.47 (M=32 tree, lawine #153)** | **513.6 ± 0.0** | **86%** | **1.037 ± 0.001** | 4.0 ms |

Warmed bandwidth is flat to the timer quantum across a 5×-larger footprint.

## Deliverable 1 — bus behaviour at the two footprints

| metric | linear | M=32 tree | delta |
|---|---:|---:|---:|
| warmed_bw (GB/s) | 513.575 | 513.575 | **`d_bw = +0.0000%`** |
| bus_ratio | `bus_ratio_linear = 1.0368` | `bus_ratio_tree_m32 = 1.0366` | `d_ratio = −0.0254%` |

Both deltas are far inside #148's **0.787%** band. The warmed bandwidth medians are bit-identical
(`d_bw` exactly 0), so the multiplier's load-bearing throughput term does not move with the footprint.

## Deliverable 2 — propagation into K_cal and the projection

`d_bw = 0` ⇒ `k_cal_factor = bw_tree / bw_lin = 1.0` ⇒ **`k_cal_tree_corrected = 125.268`** (= K_cal
unchanged). With K_cal unmoved, every projection that rides it is unmoved:

| topology | central (K_cal=125.268) | corrected | `official_shift_tps` |
|---|---:|---:|---:|
| both-bugs @ roofline step | 537.84 | 537.84 | **0.0** |
| descent-only @ roofline step | 522.29 | 522.29 | **0.0** |

**Plain statement:** `PRECACHE_BENCH=1` holds the bus ratio invariant across the linear → M=32-tree
footprint. **There is no footprint tax on the calibration.** K_cal = 125.268 stands; the 537.8 / 510.6
projections (and 535.4 @ overlap / 525.5 / 522.3 — all K_cal-riders) stand unchanged.

## Deliverable 3 — `PRECACHE_BENCH=1` is the invariance mechanism

Cross-check at the tree footprint with the warmup replay disabled (the `PRECACHE_BENCH=0` analogue):
the un-warmed first-step read pays the cold residency/launch transient.

- **`precache_off_divergence_pct = 3.53%`** (single-shot, the first decode step off a cold working set).
- Amortized over a 512-step decode window: **0.007%**.

So the multiplier *does* diverge without the warmup — confirming `PRECACHE_BENCH=1` is the named,
load-bearing launch dependency that MUST be retained in the tree submission manifest. The single-shot
3.53% / amortized 0.007% split cross-validates #148 Leg B (the precache ramp is a one-time entry cost,
negligible across the bench's output length).

## Deliverable 4 — self-validation (PRIMARY)

**`precache_footprint_self_test_passes = 1` (11/11 PASS). `bus_ratio_tree_invariant = 1`. All metrics NaN-clean (`metrics_nan_clean = 1`).**

The rig reproduces, from the consumed #148 anchors (not re-derived):

1. K_cal == 481.53/3.844 → 125.26795
2. multiplier == 481.53/454.1937 → 1.060187 (+6.019%)
3. K_cal == (local wall / E[T]_lin) · multiplier
4. both-bugs 537.8446 @ roofline reproduced
5. descent 522.2888 reproduced
6. both-bugs 535.4377 @ overlap step reproduced
7. band ordering K_lo (124.282) < K_cal (125.268)
8. band width reproduces 0.787046%
9. invariant ⇒ corrected both-bugs reproduces 537.8446
10. invariant ⇒ `k_cal_tree_corrected` reproduces 125.268 within band
11. all measured + propagated values finite

## Deliverable 5 — hand-off

`bus_ratio_tree_invariant = 1` confirms K_cal = 125.268 transfers and the 537.8 / 510.6 projections
stand — the last unverified link in the denominator→official compose feeding the pinned
launch-readiness packet is now measured, not assumed. It also hardens the **`PRECACHE_BENCH=1`
MUST-RETAIN** flag in land #71's tree manifest with a measured divergence number (single-shot 3.53%,
amortized 0.007%). **Does NOT authorize a launch — the true local→official multiplier still needs the
official environment.**

## Honest scope

This is the same bounded-not-closed scope stark #156 carries on the private axis. I measured the
LOCAL footprint-sensitivity of the warmed HBM transfer as a faithful proxy for whether the
multiplier's invariance assumption survives the tree footprint. The physical basis for invariance is
sound (GDDR6 refresh/ECC overhead is constant, not occupancy-dependent; `cudaMalloc` never spills to
host; clock-ramp is footprint-independent and pinned out here), and the measurement confirms it. But
the *absolute* +6.019% multiplier is a property of the official serving environment and can only be
closed by a launch, which this PR does not request.

## Reproduce

```
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/profiler/precache_footprint_invariance.py \
  --linear-gb 4.0 --tree-gb 20.47235584 --sweep-gb 6 10 14 18 \
  --chunk-gb 2.0 --repeats 7 --warmup-iters 200 --ramp-iters 96 --timed-iters 64 \
  --quiesce-s 2.0 --window-steps 512 \
  --wandb-group precache-bench-tree-footprint-invariance \
  --wandb-name ubel/precache-bench-tree-footprint-invariance \
  --output research/spec_cost_model/precache_footprint_invariance/precache_footprint_invariance.json
```

- **W&B run:** `0czdgugp` (group `precache-bench-tree-footprint-invariance`, project `gemma-challenge-senpai`)
- **JSON:** `precache_footprint_invariance.json` (this directory)
- **Peak GPU:** ~19.1 GiB resident at the 20.47 GB tree footprint (A10G, 22.06 GiB usable); runtime 148.7 s
- **Anchors consumed (not re-derived):** `research/kcal_tree_transfer/kcal_tree_transfer_band.json` (#148 K_cal band + multiplier decomposition + projection cells)
