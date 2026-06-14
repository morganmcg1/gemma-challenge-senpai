# Tree-free 500-path projection instrument + τ-bound (PR #112, lawine)

**Gate: 🟡 AMBER** — instrument ARMED; the ≥500 call turns on the τ floor → denken #109
needs **one official SplitK anchor** to confirm τ≥0.99 (the bandwidth mechanism predicts it).

- Primary `tree_free_projection_armed` = **True** (null-lever self-check = 481.530000, bit-exact)
- Test `tau_band_local` = **[0.99, 1.00]**  ·  recommendation = `ONE_OFFICIAL_SPLITK_ANCHOR`
- W&B `hcrvdf31` (group `tree-free-projection-harden`) · artifact `tree_free_projection_harden_results.json`

## What the instrument is
`scripts/profiler/local_official_projection.py` now imports denken #105's
`tree_free_500_ceiling.py` as the **single source of truth** for the decode-budget slice
model, and maps a measured SplitK% (ubel #108) + additive levers (LK #95, wirbel #110
palette) → a 3-corner projected-official band vs 500. The #99 multiplier CI
`[1.05999, 1.06038]` enters as a **relative rescale `mult/mult_central`** (1.0 at the
central anchor → central stays bit-exact on 481.53; ±0.018% at the CI edges). τ is denken's
residual realization factor.

One command:
```bash
.venv/bin/python scripts/profiler/local_official_projection.py \
  --tree-free --splitk-frac <s> --splitk-lo <lo> --splitk-hi <hi> \
  --wandb --wandb-group tree-free-projection-harden \
  --out research/spec_cost_model/tree_free_projection_harden_results.json
```

**Consistency vs denken:** my central-corner SplitK-for-500 = **5.43%** vs denken's 4.44%.
The 0.99% gap is exactly the **de-credited double-quant** — #104 was info-theoretically
KILLed, so wirbel #110 palette central is banked at 0 (unrealized), not the dead +0.5% byte
lever. The instrument is deliberately more conservative than denken's literal central.

## Why τ can't be pinned from committed local data (Step 2)
Exactly ONE matched (official, local) pair exists on the frontier — the deployed anchor,
which *defines* τ=1.00. No committed config gives a second matched pair in the same meter:

- **Meter confound (the blocker).** Identical deployed stack, three committed meters:
  steady 428.37 / wall_tps 454.09 / windowed-steady 459.83 → implied multipliers
  1.124 / 1.060 / 1.047. **Meter choice alone swings the multiplier 7.14%**, swamping any
  cross-precision config signal → precision rungs cannot bound τ.
- **Within-meter (matched) stability.** K-sweep 5–9 + MBT-sweep 512–8192 move the wall_tps
  denominator by only **0.056%** → transfer is NOT config-sensitive within a matched meter,
  but these configs have no official counterpart (bound *local* stability, not τ).
- **Band = mechanism + physical ceiling.** τ≤1.00 is a hard ceiling (a bandwidth-utilisation
  lever can't over-realize officially). Both A10G boxes are sm_86 / GDDR6 ~600 GB/s and the
  verify-GEMM is bandwidth-bound (SM-clock-insensitive) → fractional SplitK speedup transfers
  ≈1:1 → **τ ∈ [0.99, 1.00]**.

## Decision surface for denken #109 (conservative corner = mult-low × τ-low × SplitK-low)

| τ | SplitK-for-500 | ≤ ubel nominal-high (12%)? |
|---|---|---|
| 1.00 | 5.49% | ✅ |
| 0.99 (mechanism floor) | 7.57% | ✅ |
| 0.98 | 9.74% | ✅ |
| 0.97 | 12.00% | ✅ (edge) |
| 0.96 (generic floor) | 14.34% | ❌ (> 12%, still < 29.7% gap ceiling) |

**GO/HOLD off ubel #108's number:**
- `s ≥ 14.34%` → clears at conservative corner even at the generic τ=0.96 → **GO, no anchor.**
- `s ∈ [7.57%, 14.34%)` → clears only to the mechanism floor τ=0.99 → **GO needs the one
  official SplitK τ-anchor** (a single approval-gated run; confirms τ≥0.99, flips this GREEN).
- `s < 5.49%` → conservative corner < 500 at every τ in band → **HOLD / add a lever.**

The named de-risk is NOT a full multiplier re-anchor — just a one-point τ confirmation on the
new SplitK kernel class.
