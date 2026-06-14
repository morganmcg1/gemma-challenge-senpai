# PR #130 — gate_up tile-shape re-tiling: can a re-tile break denken #117's CTA-saturation SplitK ceiling?

**Verdict: 🔴 RED — `gate_up` is re-tile-proof at our shapes; denken #117's CTA-saturation
ceiling is confirmed HARD.**
No tile shape beats the deployed int4 W4A16 Marlin `gate_up` GEMM. The realized per-step
speedup is **0.0%** (M=8) / **+0.07%** (M=32, noise). The binding constraint is the
**physical GDDR6 read wall (~83.6% of datasheet), which saturates at ONE wave** — so the
entire SplitK / extra-CTA mechanism the re-tile would unlock manufactures essentially no
bandwidth. Marlin already runs at **79.4%** of datasheet, i.e. **95% of the achievable
streaming wall**, leaving only **+5.26%** of headroom that NO compute kernel reaches (the
best re-tiled Triton kernel tops out at 71.5% HBM and is *slower*; every exposed Marlin
knob is within ±0.07%).

- **Primary** `gate_up_retile_per_step_speedup_pct` = **0.0%** (M=8); +0.069% (M=32).
- **Test** `gate_up_retile_projected_official_tps` = **492.77** (tree-free, stacked on the
  #123 491.8 build-complete ceiling — *unchanged*, because the re-tile adds nothing).
- W&B `ryftxgom` (group `gate-up-retile`). Repro:
  `CUDA_VISIBLE_DEVICES=0 python scripts/profiler/gate_up_retile.py`.
- LOCAL, 1×A10G, micro-bench + roofline. Synthetic value-independent Marlin weights
  (Marlin time depends only on M,in,out,group); launch-free CUDA-graph-replay timing (the
  deployed serve mechanism). **No HF Job, no submission, no served-file/token-stream
  change.** Greedy identity untouched by construction (isolated GEMM/stream timing).

## Units & method (load-bearing)

The re-tile lever, if it worked, would reduce the **per-GEMM wall-time** of `gate_up`. That
flows into the ship model as a reduction of the gate_up step slice: gate_up = **54% of
verify** (#117), verify = **53% of M=8 decode** (#105) → gate_up = **0.286 step units**.
A realized per-step speedup δ scales tree-free TPS by `1/(1 − 0.286·δ)` and stacks on the
#123 built ceiling by `step₀/(step₀ − 0.286·δ)`. τ=1.06019 (#99) is folded into K_cal.

**Group reconciliation (resolves the g=128/g=32 ambiguity):** denken #68/#117 report the
deployed gate_up at **62.9 µs = 475 GB/s / 79%**. That byte count (29.9 MB) matches **only
group_size=32** (weight 26.2 MB + scales 3.28 MB); g=128 reads 27.4 MB → 436 GB/s at the
same time. Our warm g=32 M=8 measures **62.65 µs / 477 GB/s / 79.4% / AI=28.1**, reproducing
#117 to the digit → **the deployed checkpoint is g=32**, and that is the baseline the
re-tile must beat. (We bench g=128 too; the weight bytes dominate, so the wall conclusion is
group-robust.)

**Timing hygiene (one subtle artifact, caught and fixed):** the very first timed config on a
cold A10G reads ~8% slow (62.9 µs vs warm 57.8 µs for *the same* config) — a power/cache ramp
transient, **not** clock scaling (SM clocks are pinned at 1710 MHz throughout; 8 consecutive
warm `graph_time` calls land at 57.79–57.80 µs, <0.1% spread). Measuring Part A cold and the
Marlin-knob sweep warm would manufacture a **phantom +7.6% "speedup"** out of pure drift. A
sustained burn-in before any timing removes it, and knob deltas are measured against the
**in-loop default** (same warm regime) to isolate the knob. After the fix, every knob is
within ±0.07%.

## Step 1 — the deployed gate_up tiling (the bar) — confirms denken #117

| group | M | t (µs) | GB/s | %HBM | AI | CTAs | waves | saturated |
|---|---|---|---|---|---|---|---|---|
| **32 (deployed)** | **8** | **62.65** | **477** | **79.4%** | **28.1** | **160** | **2.00** | **YES** |
| 32 (deployed) | 32 | 67.21 | 461 | 76.8% | 108.4 | 160 | 2.00 | **YES** |
| 128 | 8 | 57.83 | 474 | 79.0% | 30.6 | 160 | 2.00 | YES |
| 128 | 32 | 62.46 | 456 | 76.1% | 117.7 | 160 | 2.00 | YES |

denken #117's characterization is **reproduced exactly**: at M=8 the GEMM tiles N=20480 into
128-wide column blocks → **160 CTAs = exactly 2.00 full waves** on 80 SMs, AI≈28 (3.8× below
the A10G ridge of 107), 79.4% HBM. **Critically, M=32 is ALSO 160 CTAs / 2.00 waves** (the
M-tile granularity is 16, so M=8 and M=32 both emit 1–2 M-tiles × 160 N-tiles and stay
2-wave-saturated). The tree's M=32 verify width is **equally CTA-saturated** — so a re-tile
that helped the tree would have to come from the same place a tree-free M=8 re-tile does.

Note the **AI cliff with M**: at M=32 AI jumps to ~108 (g=32) / ~118 (g=128) — *at or past
the ridge*. So lever (b) "raise AI toward the ridge" is structurally unavailable: AI is fixed
by M and the weight-byte volume (not by the tile), and the only way to raise it — more M — is
already past the ridge at M=32, where %HBM has *fallen* to 76.8% and latency *risen* to 67 µs.
There is no AI headroom to harvest at either verify width.

## Step 2 — the re-tile sweep + the physical wall

### Part B — the kernel-agnostic HBM streaming ceiling (the decisive test)

A pure read-stream of the 26.2 MB gate_up weight volume, swept over CTA/wave count:

| grid | waves | GB/s | %HBM |
|---|---|---|---|
| 40 | 0.50 | 432 | 72.0% |
| 80 | 1.00 | 494 | **82.3%** |
| 120 | 1.50 | 499 | 83.2% |
| 160 | 2.00 | 498 | 83.1% |
| 240 | 3.00 | 500 | 83.3% |
| 320 | 4.00 | 499 | 83.1% |
| 640 | 8.00 | **502** | **83.6%** |
| 1280 | 16.00 | 499 | 83.1% |

**Bandwidth saturates at ONE wave** (82.3% at 1 wave → 83.6% at 8 waves: +1.3 pp over an 8×
CTA increase). This is the whole ballgame: the SplitK / extra-CTA mechanism that a re-tile
would unlock adds occupancy *past* 1 wave — but past 1 wave there is **no bandwidth left to
add**. The physical wall is **83.6% of datasheet (502 GB/s)** — GDDR6 refresh/ECC/page-
conflict efficiency, corroborated by `torch.sum` (469 GB/s) and `copy_` (476 GB/s, read+write)
landing in the same band. Marlin's deployed **79.4%** is **95% of this achievable wall**;
total exploitable headroom is **+5.26%**, and it is *not* a re-tile/occupancy lever (the
stream already saturates at 1 wave) — it is the residual DRAM-efficiency gap that no tiling
touches.

### Part C — tunable Triton W4A16 tile sweep (the re-tile MECHANISM, 192 configs/M)

Swept {BLOCK_M∈[16,32,64], BLOCK_N∈[64,128,256,512], SPLIT_K∈[1,2,4,8], num_warps∈[4,8],
num_stages∈[3,4]} (BLOCK_K pinned to the group for scale correctness). All configs
bit-correct vs an fp32 dequant reference (rel_err ~3e-4). Best:

| M | best shape | t (µs) | %HBM | CTAs | vs Marlin |
|---|---|---|---|---|---|
| 8 | BM16 BN128 SK1 w8 s3 | 63.9 | **71.5%** | 160 | **−2.1% (slower)** |
| 32 | BM16 BN256 SK1 w4 s4 | 108.1 | 44.0% | 160 | −61% (slower) |

Even the best re-tiled shape — including every smaller-N / higher-CTA / SplitK config — reaches
only **71.5% HBM**, *below* Marlin's 79.4% and far below the 83.6% wall. More CTAs and explicit
SPLIT_K do **not** lift achieved bandwidth above the deployed tiling; the dequant compute
overhead of a non-Marlin kernel dominates. The mechanism is directly falsified.

### Part D — exposed Marlin knobs (no recompile, byte-identical path)

The only `gate_up` tiling controls reachable from Python (workspace `max_blocks_per_sm`,
`use_atomic_add`, `use_fp32_reduce`) all land within **±0.07%** of the default at both M=8 and
M=32. They size/route the SplitK *reduction*, not the GEMM tiling (`thread_n=128` is baked into
Marlin's kernel selection at M≤16, and at N=20480 `use_atomic_add` is force-disabled
internally) — so they cannot move the 160-CTA tiling, and they don't.

## Step 3 — projection through the decode budget

| path | base | **realized (δ=0)** | optimistic ceiling (δ=5.26%, the unreachable wall) |
|---|---|---|---|
| tree-free alone (on 481.53) | 481.53 | **481.53** | 488.88 |
| tree-free stacked on #123 491.8 | 492.77 | **492.77** | 500.40 |
| tree M=32 (on 538.0, #125) | 538.0 | **538.11** | — |

**Realized: nothing moves.** tree-free stays at the #123 **492.77** ceiling; the tree gains
**+0.11 TPS** (noise), nowhere near the +5 GREEN bar. Even the **physically-unreachable**
optimistic ceiling — granting every byte the 83.6% streaming wall that *no* compute kernel
achieves — lifts tree-free-alone only to **488.9** (<500) and the stacked path to **500.4**
(barely 500, and only at a wall Marlin misses by 4.2 pp). There is no realistic path to 500
through a gate_up re-tile.

## Verdict + hand-off

**🔴 RED.** Trigger: every tile shape is ≤ the deployed Marlin tiling within noise (Triton
slower, Marlin knobs ±0.07%) AND the streaming ceiling shows the BW headroom is non-exploitable
(saturates at 1 wave; only +5.26% residual DRAM gap, unreachable). `gate_up` is re-tile-proof
at our shapes. **The single named ceiling-breaker for denken #117's SplitK wall is spent.**

- **→ denken #117 / the SplitK lever:** your CTA-saturation ceiling is now **mechanistically
  confirmed from the kernel side**. The 160-CTA/2-wave saturation isn't just an occupancy
  story — the streaming roofline shows BW saturates at **1 wave**, so even a tiling that broke
  the 2-wave integer cliff would find no bandwidth past it. The verify-GEMM is at **95% of the
  physical GDDR6 wall**; there is no re-tile dividend. SplitK stays capped at your 1.56% net.
- **→ the 500 push (theykk target):** the verify-GEMM bandwidth lane is **closed**. Both
  un-capped candidates named in #117 (re-tile (a) wave-remainder, (b) AI-to-ridge) are
  measured dead — (a) because BW saturates at 1 wave, (b) because AI is fixed by M and already
  past the ridge at M=32. 500 must come from **τ** (lawine #116 anchor) or the **tree**
  (land #71), not from cheaper verify bytes.
- **→ the tree (fern #125 / land #71):** a faster gate_up will **not** raise the 538 supply —
  the M=32 verify GEMM is equally 2-wave-saturated at the same 76.8% HBM, and the re-tile
  delivers +0.07% (noise). The tree's TPS is gated by E[T]/τ realization, not verify-GEMM
  wall-time.

## Public-evidence corroboration

denken #117's field cross-check already noted public SplitK-class submissions realize only
**+0.6–1.7% TPS** over the 481.53 frontier (implied s≈1.1–3.3%) — consistent with the
verify-GEMM being at its physical wall. This report supplies the **kernel-side mechanism**:
the wall is the 1-wave GDDR6 streaming saturation, and Marlin sits at 95% of it. The
competitive field has not realized a gate_up re-tile dividend because there is none to realize.

## Bottom line

The MEASURED roofline gap is a *peak-vs-achieved* gap, not a *re-tile-harvestable* gap: it
lives in the residual ~4 pp DRAM-efficiency band that saturates at one wave and that no tiling
(Marlin knob or hand-rolled Triton re-tile, at any CTA count or SplitK depth) can cross. This
is a valuable negative — it **retires the last un-capped verify-GEMM lever** and routes the 500
push decisively to τ (lawine #116) or the tree (land #71).
