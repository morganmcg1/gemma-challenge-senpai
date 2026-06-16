STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["sb1n4aa6"],"primary_metric":{"name":"decision_flip_tps_threshold","value":12.2346},"test_metric":{"name":"ppl","value":2.3772}}

## Results

CPU-only analytic — the **decision surface** that turns my #458 cost-ledger's `GRADED_DECISION_PENDING` *rule* into numeric *thresholds*, so the recommendation falls out automatically the instant stark #452 reports. The analytic complement to #458 (states the rule) and a direct input to fern #357 (composes the one-screen packet): **I build the decision math, fern presents it.** **No HF job, no submission, no served-file change.** Round-trips committed `#457` + `#458` + `directive4_correct_bar` JSONs; re-derives nothing (every banked source number round-trips at **0.0**).

`decision_surface_self_test_passes = True` (10/10 conditions). W&B run [`sb1n4aa6`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/sb1n4aa6), group `equivalence-escalation-anchors`.

### (1a) TPS-gain decision-flip thresholds — largest human bar B for a **CI-clean** GO

CI-clean GO at confidence `k` ⟺ `(gain − k·σ_hw) ≥ B`. The supremum admissible bar is therefore `gain − k·σ_hw` (σ_hw = 4.8153, single-σ convention matching #457's sigma-envelope; deployed anchor treated fixed).

| k (σ_hw) | relax-realistic (+17.05) max CI-clean bar B | ceiling (+29.34) max CI-clean bar B |
|:--|---:|---:|
| 0 (point) | **+17.0499** | +29.3424 |
| **1 (headline)** | **+12.2346** ← `decision_flip_tps_threshold` | **+24.5271** |
| 2 | +7.4193 | +19.7118 |

- **`decision_flip_tps_threshold = +12.2346`** (PRIMARY) — the bar B above which relax-realistic *stops* being a CI-clean GO at 1σ_hw.
- **Cross-check:** the ceiling CI-clean bar at k=1 is **+24.5271 ≡ #457's banked `headroom_deployed_to_ceiling_lcb_tps` (24.527)** — a hard validation that my CI arithmetic is the *same* convention as the parent card (`ceiling_ci_clean_k1_equals_banked_lcb` selftest passes).
- **`relax_realistic_ci_clean_of_threshold = True`** against the strongest concrete strict alternative — the best byte-exact (greedy-safe) lever to date, **+0.26 TPS**. i.e. relax-realistic beats the best greedy-*safe* option by a CI-clean margin (12.23 ≥ 0.26), so going greedy-unsafe is *worth it on TPS* for any non-aggressive bar.
- **Takeaway:** the TPS clause is **not where this decision is tight** — relax clears every plausible bar up to +12.23 (realistic) / +24.53 (ceiling) CI-clean. It only flips NO-GO if the human sets an aggressive bar.

### (1b) PPL decision boundary (hard gate)

- **`max_admissible_relax_ppl = 2.42`** — relax is PPL-admissible **iff** stark #452's measured PPL ≤ 2.42.
- Margin from the deployed anchor 2.3772 = **0.0428** (~**1.77%** of the gate). Razor-thin.

### (1c) Flip-count framing — Δflips for ΔTPS, **orthogonal** to the gate

| relax flips N | Δflips vs deployed 3-flip status quo | buys (realistic) | gate depends on N? |
|---:|---:|:--|:--:|
| 3 | 0 | +17.05 TPS | **No** |
| 4 | +1 | +17.05 TPS | **No** |
| 10 | +7 | +17.05 TPS | **No** |
| 20 | +17 | +17.05 TPS | **No** |

`decision_is_N_invariant = True`. flip-COUNT (equivalence severity) and PPL (quality) are **orthogonal** — the gate is PPL ≤ 2.42 **AND same-KIND**, never the count N. N is reported for transparency only. (This is #458's orthogonality discipline carried into the surface — conflating count with quality is the trap that sank four isolated levers.)

### (2) Sensitivity / tornado — which pending stark-#452 input most swings the recommendation

Ranked by how close the deployed nominal sits to each axis's decision-flip boundary (normalized; closer = more sensitive):

| rank | pending input | norm. dist-to-flip | can flip decision? | swing |
|:--:|:--|---:|:--:|:--|
| **1** | **measured PPL** | **0.0177** | **yes** | **FULL (GO↔NO-GO)** — hard gate, 0.0428 margin |
| 2 | measured TPS gain | 0.7023 | conditional | flips only if gain craters < ~+5 or bar aggressive (> +12.23) |
| 3 | measured flip-count | ∞ (orthogonal) | **no** | none on the count — only the KIND gates |

- **`most_sensitive_pending_input = "ppl"`.** The decision turns almost entirely on stark #452's **PPL** read: it's a hard gate with a thin 0.0428 margin, and it's *unpredictable* — flips can be PPL-neutral OR PPL-breaching, so only the measurement decides.
- **Justified region:** relax is justified in `{measured PPL ≤ 2.42}` ∩ `{break is same accumulation-order KIND}` ∩ `{measured gain − k·σ_hw ≥ human bar B}`. A half-space below the PPL gate, intersected with the same-KIND set and the CI-clean TPS half-space. **N does not bound the region.**

### (3) Pre-wired stark #452 → recommendation (the one-number-swap)

When stark #452 reports `(TPS, PPL, flip-count)`, each value resolves exactly one clause, and a single function collapses them:

```python
recommend(measured_gain_tps, measured_ppl, break_same_kind, human_bar_tps, k=1):
    GO            iff PPL ≤ 2.42 AND same_kind AND (gain − k·σ_hw ≥ B)
    CI-AMBIGUOUS  if  PPL ≤ 2.42 AND same_kind AND (B ≤ gain < B + k·σ_hw)   # clears bar only within hw noise
    NO-GO         if  PPL > 2.42  OR  new-kind  OR  point gain < B
```

| stark #452 reports | resolves | rule |
|:--|:--|:--|
| measured TPS (⇒ gain) | clause-1 | CI-clean GO iff `gain − k·σ_hw ≥ B`; CI-AMBIGUOUS in the noise band |
| measured PPL | clause-2 | admissible iff `PPL ≤ 2.42` |
| measured flip **KIND** | clause-3 | GO if accumulation-order near-ties; NO-GO if a new quality-destroying mode |

Worked corners (verdict the instant stark #452 lands):

| gain | PPL | kind | bar B | → verdict |
|:--|:--|:--|:--|:--:|
| +17.05 | 2.3772 (neutral) | same | +0.26 | **GO** |
| +17.05 | 2.43 (breach) | same | +0.26 | **NO-GO** |
| +17.05 | 2.42 (at gate) | same | +0.26 | **GO** |
| +17.05 | 2.3772 | **new** | +0.26 | **NO-GO** |
| +17.05 | 2.3772 | same | +15 | **CI-AMBIGUOUS** |
| +17.05 | 2.3772 | same | +20 | **NO-GO** |
| +29.34 (ceiling) | 2.3772 | same | +20 | **GO** |

**Live status:** the TPS clause is *already* a CI-clean GO for any bar ≤ +12.23. The live recommendation is **GO-PENDING-PPL-AND-KIND** — it resolves to **GO** the instant stark #452 reads PPL ≤ 2.42 with same-kind flips (at any bar ≤ +12.23), and to **NO-GO** if PPL > 2.42 or the break is a new kind — regardless of N.

### Baseline comparison

| quantity | baseline (PR body) | this card | match |
|:--|:--|:--|:--:|
| deployed / strict / relax / ceiling | 481.53 / 467.14 / 498.58 / 510.87±4.82 | identical (round-trip resid **0.0**) | ✓ |
| TPS gains | +17.05 / +29.34 | +17.05 / +29.34 | ✓ |
| deployed_ppl_gate_margin | 0.0428 | 0.0428 | ✓ |
| ceiling LCB (1σ_hw) | 24.527 (#457) | ceiling CI-clean k=1 = **24.5271** | ✓ (cross-check) |
| σ_hw / PPL gate | 4.8153 / ≤2.42 | 4.8153 / 2.42 | ✓ |

Adds **0 TPS**; greedy/PPL untouched (PPL anchor **2.3772**, `official_tps = 0`, `analysis_only = True`, `no_served_file_change = True`).

### Command

```bash
cd target/
python3 research/validity/relax_decision_surface/relax_decision_surface.py \
  --wandb_name "land/relax-decision-surface" --wandb_group "equivalence-escalation-anchors"
# self-test only (CPU): add --self-test  → "self-test PASS"
```

### Peak memory

**12.11 MiB** (CPU-only; no GPU, no vLLM, no model load).

### What happened

Worked cleanly. The packet now has the missing decision **math**, not just the rule. Three findings worth flagging:

1. **The TPS clause is already settled.** Relax-realistic +17.05 is a CI-clean (1σ_hw) GO for any human bar up to **+12.23**; the ceiling up to **+24.53** (which exactly reproduces #457's banked headroom LCB — a free cross-check). Both dwarf the best byte-exact strict lever (+0.26), so on TPS alone the relax move clears any non-aggressive bar.
2. **The decision is a PPL decision.** The tornado is unambiguous: **PPL is the single most decision-swinging input** — a hard gate with only 0.0428 of margin, and unpredictable (flips can be neutral or breaching). TPS-gain is robustly GO; flip-count is **orthogonal** and never gates on the count.
3. **It's now a one-number-swap.** `recommend(gain, ppl, same_kind, bar, k)` collapses stark #452's three numbers + the human bar into GO / NO-GO / CI-AMBIGUOUS. The slots are wired; the human only ever has to (a) set the bar B and (b) read stark #452's PPL + flip-kind.

### Public evidence used

Leaderboard digest (`as=senpai`, fetched 2026-06-16) frames the relevance: the **valid** frontier sits at ~489.66 TPS (firfir-cast `hayai-ctk48-mwfix-v1`, verification=valid), with **pending** entries pushing to **508.63** (fabulous-frenzy `ff-splitkv-frantic-fawindow`) — i.e. real submissions now bracket exactly the relax-prize band this surface arbitrates (realistic **498.58** / ceiling **510.87**). The rank-1 `f64-max` row (1.79e308 TPS) is a non-physical sentinel and was ignored. The decision surface tells the human precisely when crossing from the valid-strict frontier into that greedy-unsafe band is justified. (No board write; this is a CPU-only reconciliation of committed internal banked JSONs — `#457`, `#458`, `directive4_correct_bar` — with no HF job, submission, or served-file change.)

### Suggested follow-ups

- **stark #452** remains the one blocker: its measured (TPS, PPL, flip-kind) drops straight into `recommend()` and resolves the live `GO-PENDING-PPL-AND-KIND` to a final verdict with zero further derivation.
- For **fern #357**: this surface + #458 (rule) + #457 (prize axis) share the exact anchors (round-trip resid 0.0) and compose without re-derivation — fern reads `decision_flip_tps_threshold`, the tornado order, and `recommend()` straight into the one-screen GO/NO-GO.
- The only remaining human input is the **TPS bar B**; everything else is measured (#457/#458) or scheduled (stark #452). If the human names B now, the surface returns the live verdict immediately (modulo PPL/kind).
