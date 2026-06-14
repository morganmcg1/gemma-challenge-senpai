# PR #117 — SplitK realization ceiling: can SplitK physically reach #109's 14.34% corner?

**Verdict: 🔴 RED — the 500 corner (and 540-margin) is genuinely τ/tree-gated.**
The MEASURED +29.8% achieved-vs-roofline HBM gap is **NOT** physically convertible to
SplitK wall-time at M=8. The realizable ceiling is **3.20% gross** (net-after-overhead
**1.56%**; band to **7.81%** at the optimistic 88%-GDDR6 wall) — **below ubel #108's
realistic central 8.5%**, far below #109's **14.34%** conservative corner, and below
even the **5.49%** corner at *perfect* τ=1.0. **SplitK cannot close the corner alone.**
The binding constraint is the **HBM PRACTICAL roofline**, not the compute floor and not
the 29.7% datasheet gap ceiling.

- **Primary metric** `splitk_realization_ceiling_pct` = **3.20%** (net 1.56%, band-high
  7.81%); **binding constraint = HBM-practical-roofline**.
- **Test metric** `splitk_headroom_to_corner` = **−11.15%** (band-high −6.54%) →
  `corner_reachable_on_splitk_alone = False`.
- W&B `z9eaoxj5` (group `splitk-realization-ceiling`). Repro:
  `python scripts/profiler/splitk_realization_ceiling.py [--wandb]`.
- LOCAL, CPU-only, analytic. Extends denken #68 `verify_gemm_roofline.json` (MEASURED
  Marlin W4A16) + #105/#109 compose ship model. No GPU/vLLM/HF Job/submission/kernel
  build. Greedy identity untouched.

## Units (load-bearing — pinned against the #105/#109 model)

SplitK `s` is a **bandwidth-headroom fraction**: the #105/#109 compose model applies
`vg → vg/(1+s)`, so `s` is the fractional INCREASE in achieved aggregate verify-GEMM
bandwidth. #109's corner **14.34%**, ubel's **8.5%/12%**, the gap ceiling **29.7%**, and
this report's ceiling are ALL in these `s` units. Check: `s=29.7% ↔ BW 462→600 GB/s ↔
22.9% verify-GEMM wall-time cut = 1 − 0.771` (the PR's hard ceiling).

## Step 1 — the SplitK wall-time CEILING at M=8, and why it's there

SplitK's only lever on a BW-bound GEMM is **occupancy**: at M=8 a small-N GEMM emits
`N/tile_n` output tiles in 1 M-tile, starving the 80 SMs; splitting the K reduction ×g
fills them, raising memory-level parallelism and achieved BW (arxiv:2402.00025, the
directly-analogous W4A16-SplitK reference). The MEASURED #68 per-GEMM occupancy at M=8
shows **why most of the verify-GEMM has no SplitK headroom left**:

| role | out (N) | CTAs (N/128) | saturated ≥80 SM? | %HBM | SplitK | time-share |
|---|---|---|---|---|---|---|
| mlp.gate_up_proj | 20480 | **160** | **YES** | 79.2% | 1× (frozen) | **54.2%** |
| mlp.down_proj | 2560 | 20 | no | 74.2% | 4× | 29.0% |
| self_attn.qkv_proj | 3072 | 24 | no | 78.5% | 4× | 6.9% |
| self_attn.o_proj | 2560 | 20 | no | 68.8% | 4× | 5.3% |
| self_attn.qkv_proj | 6144 | 48 | no | 65.9% | 2× | 3.3% |
| self_attn.o_proj | 4096 | 20 | no | 103.6% | 1× (already past wall) | 1.4% |

**The dominant GEMM is already at the wall.** `gate_up` is **54.2% of verify time** and
is **CTA-saturated** (160 CTAs = 2 full waves on 80 SMs). SplitK gives it **~0 BW** — it
already has full occupancy; splitting K only adds reduction overhead. It tops out at
**79.2% of datasheet** (475 GB/s) — that 79.2% IS the in-situ DRAM-efficiency wall every
other GEMM is being driven toward. Only the occupancy-limited laggards (`down` + the
attention projections, 20–48 CTAs) have headroom, and only up to that same wall.

So the ceiling = lift the ~46% laggard slice from its current 65–78% achieved BW up to
the DRAM-efficiency wall, with `gate_up` frozen. Across three wall assumptions:

| scenario | DRAM-eff wall | agg util | **s_gross** | s_net | compute@BW |
|---|---|---|---|---|---|
| **measured** (gate_up's 79.2%) | 79.2% | 77.1→78.3% | **3.20%** | 1.56% | 20.5% |
| practical (88% GDDR6) | 88.0% | 77.1→81.8% | **7.81%** | 6.19% | 21.4% |
| datasheet (100%, UNREACHABLE) | 100.0% | 77.1→86.1% | **13.25%** | 11.68% | 22.5% |

**Binding constraint = HBM PRACTICAL roofline.** The aggregate verify-GEMM is *already*
at 77.1% of datasheet; the realizable wall is gate_up's measured **79.2%** (driven by
GDDR6 refresh/ECC/page-conflict, not by tuning — GPU-STREAM puts sustained GDDR6 at
~80%). The primary ceiling lifts laggards to that MEASURED wall → **3.20% gross**. Even
granting laggards the optimistic 88%-GDDR6 practical wall caps at **7.81%**.

**The compute floor never binds** (the PR's candidate #2 ceiling). AI=28 is 3.8× below
the A10G ridge (107); at M=8 compute util is 20.2%, and even at **100% BW** it rises only
to **22.5%** — there is no compute wall anywhere near the corner. The cap is purely HBM.

**Reduction overhead** (candidate #3) is the gross→net haircut: the g partial-sum tiles
add write+read+launch traffic. At M=8 the output is byte-tiny vs the weight, so it's
L2/launch-dominated; we model the byte term explicitly → net **1.56%** (measured wall).
Net is an UPPER bound on what survives the overhead.

**Why the 29.7% datasheet gap ceiling is doubly unreachable:** #109's
`SPLITK_CEILING = 1/0.771 − 1 = 29.7%` assumes the verify-GEMM reaches **100% of
datasheet** — which (a) GDDR6 physically can't sustain, and (b) even if it could, gate_up
is frozen so the aggregate caps at **13.25%**, not 29.7%.

## Step 2 — ceiling vs the corner and ubel's band

`splitk_headroom_to_corner = ceiling − 14.34% = 3.20 − 14.34 = **−11.15%**` (band-high
`7.81 − 14.34 = −6.54%`). **Negative at every wall assumption ⇒ the corner is
unreachable on SplitK alone.**

The corner ladder (#109 ship model, min SplitK to clear 500 at each τ) shows the ceiling
falls below the corner at *every* τ, **including perfect τ=1.0**:

| τ | corner SplitK | ceiling clears it? |
|---|---|---|
| 0.96 (floor) | 14.34% | no (−11.15) |
| 0.99 (lawine #116 mechanism floor) | 7.57% | no (−4.37; band-high +0.23 only at 88%-GDDR6) |
| 1.00 (perfect) | 5.49% | no (−2.29; band-high +2.31 only at 88%-GDDR6) |

**ubel's 12% nominal-high is far ABOVE the physical ceiling, not below it.** This is the
opposite of the GREEN precondition ("ubel's 12% sits below the ceiling with impl
headroom"). The honest reading of ubel #108's band against this roofline:

- ubel **central 8.5%** sits right at our **band-high 7.81%** → achievable only if every
  liftable laggard reaches the *optimistic* 88%-GDDR6 wall (above gate_up's measured
  79.2%). Against the MEASURED wall, central 8.5% is already ~2.6× the 3.20% ceiling.
- ubel **nominal-high 12%** exceeds the practical-88 band-high (7.81%) by ~4pp and is
  reachable only in the **physically-unreachable** datasheet-100 scenario (gross 13.25% /
  net 11.68%, requiring 100% datasheet BW with gate_up still frozen). In practice it is
  **unreachable**.

So the roofline brackets ubel's **low end (5%)**, not its central/high. ubel's projected
band is optimistic relative to the M=8 occupancy distribution: there simply isn't enough
liftable, sub-saturated verify-GEMM time to manufacture 8.5–12% aggregate BW headroom.

## Step 3 — verdict + fleet hand-off

**🔴 RED.** Trigger: `ceiling (3.20%, band-high 7.81%) < ubel's realistic central 8.5%`.
(The second RED trigger — compute floor binds before 14.34% — is **not** active; the
floor never binds. The active cause is the HBM practical roofline.) Independent of how
well ubel #108 implements the kernel, SplitK at M=8 physically tops out **below** the
corner at every τ. **540-margin and the 500 corner are genuinely τ/tree-gated.**

**TPS cross-check + field corroboration.** Composed through #109's ship model, the corner
TPS *at* the 3.20% ceiling is **474.6 (τ=0.96) / 489.4 (τ=0.99) / 494.3 (τ=1.0)** — all
**below 500**; SplitK alone never crosses the bar. Public SplitK-class submissions realize
only **+0.6–1.7% TPS** over the 481.53 frontier → implied `s ≈ 1.1–3.3%`, which
**straddles our 3.20% measured-wall ceiling** — an independent, field-side confirmation
that SplitK is already at its physical wall. The competitive field has not realized a
SplitK lift anywhere near the corner because **there isn't one to realize.**

### Hand-off

- **→ ubel #108 (the SplitK BUILD):** **stop tuning SplitK past ~7.8%** (the optimistic
  band-high; the measured-wall expectation is ~3.2% gross / 1.6% net). Your central 8.5%
  / high 12% band is optimistic vs the M=8 occupancy roofline — `gate_up` (54% of verify
  time) is CTA-saturated and frozen, so the aggregate is dominated by a term SplitK can't
  move. Target the band-high realistically; do **not** spend effort chasing the corner on
  SplitK — it isn't there.
- **→ lawine #116 (τ-derivation):** τ is **not merely co-required, it is the gate.** Since
  SplitK can't reach even the τ=1.0 corner (5.49%), the corner closes **only** via τ
  (your roofline) or the tree. Your τ-anchor approval ask is **justified**: it is the
  decisive measurement, not a nice-to-have. (This is the AMBER→RED distinction: AMBER
  would mean SplitK + τ co-required; RED means SplitK is spent and τ/tree carry it.)
- **→ denken #109 (the ship gate):** the conservative corner stands as **τ/tree-gated**.
  The AMBER hold in #109 was correct and is now **mechanistically explained**: the
  straddle-500 at ubel 8.5% wasn't a tuning gap SplitK could close — it's a physical wall.
  The official shot remains best spent as the **lawine #116 τ-anchor**, OR the tree
  (land #71) re-enters for a comfortable 500. SplitK is no longer a candidate to retire
  the τ assumption.

## Bottom line

The MEASURED +29.8% HBM gap is a *roofline* gap (achieved-vs-theoretical-peak), **not**
a *wall-time* gap SplitK can harvest: most of it lives in `gate_up`, which is already
CTA-saturated, and the rest is gated by the ~79–88% GDDR6 DRAM-efficiency wall. SplitK's
physical ceiling is **3.2% (band to 7.8%)**, leaving an **11.15pp shortfall** to the
14.34% corner. This is a valuable negative: it **stops wasted SplitK tuning past ~8%**
and **justifies the lawine #116 τ-anchor** (or land #71 tree) as the only physical paths
to the corner.
