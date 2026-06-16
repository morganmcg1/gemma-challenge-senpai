STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["uhhyec0q"],"primary_metric":{"name":"relax_realistic_tps_gain_over_deployed","value":17.05},"test_metric":{"name":"ppl","value":2.3772}}

## Results

CPU-only analytic reconciliation — the COST sibling to my #457 ceiling card. Builds the **equivalence-severity axis + PPL-gate budget + status-quo-relative decision rule** as one integrand for fern's #357 capstone. **No HF job, no submission, no served-file change.** Round-trips committed `#457` + `directive4_correct_bar` JSONs; re-derives nothing (every banked source number round-trips at **0.0**).

`cost_ledger_self_test_passes = True` (10/10 conditions). W&B run [`uhhyec0q`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/uhhyec0q), group `equivalence-escalation-anchors`.

### (1) Equivalence-severity axis — one axis, monotone in TPS, identity monotone-degrading

| TPS | greedy-identity | flips | PPL | greedy-safe | point | source |
|---:|---:|---:|---:|:--:|:--|:--|
| 467.14 | 1.0000 | 0 | reference | **YES** | strict frontier | denken #423 `5a6zq2yz` |
| 481.53 | 0.9966 | 3 | 2.3772 (measured) | no | **deployed** | PR #52 `2x9fm2zx` |
| 498.58 | **SLOT** (≤0.9966) | **SLOT** (≥3) | **SLOT** (unmeasured) | no | relax-realistic | ubel #450 `c5oyb7gv` |
| 510.87 ± 4.82 | n/a | n/a | n/a | — | unified physical ceiling | land #457 `h0uggl9i` |
| 520.95 | n/a | n/a | n/a | — | spec over-optimistic **UB** | land #436 `nvsbctji` (via #457) |

- `axis_monotone_in_tps = True` (467.14 < 481.53 < 498.58 < 510.87 < 520.95).
- `identity_monotone_degrading = True` (1.0 → 0.9966 → ≤0.9966 across operating points).
- **`deployed_off_strict_frontier = True`** — the reframe: the deployed 481.53 is **already** off the strict frontier (identity 0.9966, 3 flips). The relax slots (identity/flips/PPL) are stark #452's measurement, consumed here as parameterized **UNMEASURED** placeholders — never mistaken for a measurement (`relax_slots_marked_unmeasured` selftest passes).
- The ceiling (510.87) and spec-UB (520.95) are physical-limit **markers**, not operating points → identity/flips/PPL are n/a there (the UB is explicitly labelled "not an operating point").

### (2) PPL-gate headroom ledger (gate PPL ≤ 2.42)

| point | PPL status | ppl_gate_margin | admissible |
|:--|:--|---:|:--|
| strict frontier | reference-by-construction (identity 1.0 ⇒ greedy-reference stream) | ≥ 0.0428 | **True** (by construction) |
| deployed | measured 2.3772 | **0.0428** | **True** |
| relax-realistic | **UNMEASURED** (stark #452 quality run) | n/a | **unknown — pending measurement** |

- **`deployed_ppl_gate_margin = 0.0428`** (2.42 − 2.3772). This is the available PPL budget from the deployed anchor; the relax-prize is PPL-admissible **only if** a measured quality run reads PPL ≤ 2.42.
- **`flip_and_ppl_costs_kept_orthogonal = True`.** Flip-COUNT and PPL are *different, orthogonal* costs: a flip is a token-ID divergence (equivalence cost); PPL is a quality cost. Flips can be **PPL-neutral** (reduction-order near-ties, like the deployed 3) or **PPL-breaching**. You cannot infer PPL-admissibility from the flip count — that conflation is the trap that sank four modeled-in-isolation levers, so the ledger keeps them in separate columns and never derives one from the other.

### (3) Status-quo-relative decision rule (fern #357 verbatim)

> relax is justified iff **(TPS gain over deployed ≥ human-set threshold)** AND **(measured PPL ≤ 2.42)** AND **(the break is the SAME KIND already deployed — accumulation-order flips, not a new failure mode)**.

| clause | status today |
|:--|:--|
| TPS gain ≥ threshold | **QUANTIFIED**: `relax_realistic_tps_gain_over_deployed = +17.05`, `relax_ceiling_tps_gain_over_deployed = +29.34` (re-cited from #457, not re-derived). Threshold is the human's to set. |
| measured PPL ≤ 2.42 | **PENDING**: relax PPL UNMEASURED (stark #452). Budget 0.0428 from the deployed anchor. |
| same-kind break | **PROVISIONAL same-kind**: relax = FP-reassociating split-K re-tiling (accumulation-order) — same family as the deployed reduction-order near-tie flips, not a new failure mode. Confirm via stark #452's flip characterization. |

`GRADED_DECISION_PENDING` (1/3 clauses fully resolved). The deployed 3-flip / identity-0.9966 point is the **status-quo the human implicitly already accepted** when PR #52 was deployed, so the relax decision is **graded** ("3 flips → N flips for +17..29 TPS, does quality survive?"), **not** the binary "pristine-strict vs dirty-relax" the packet first framed.

### Baseline comparison

| quantity | baseline (PR body) | this card | match |
|:--|:--|:--|:--:|
| deployed | 481.53 / id 0.9966 / 3 flips / PPL 2.3772 | identical (round-trip resid 0.0) | ✓ |
| strict frontier | 467.14 / id 1.0 | identical | ✓ |
| relax realistic / ceiling | 498.58 / 510.87 | identical | ✓ |
| deployed_ppl_gate_margin | 0.0428 | 0.0428 | ✓ |
| TPS gains | +17.05 / +29.34 | +17.05 / +29.34 | ✓ |

Adds **0 TPS**; greedy/PPL untouched (PPL anchor **2.3772**, `official_tps = 0`, `analysis_only = True`, `no_served_file_change = True`).

### Command

```bash
cd target/
python3 research/validity/relax_decision_cost_ledger/relax_decision_cost_ledger.py \
  --wandb_name "land/relax-decision-cost-ledger" --wandb_group "equivalence-escalation-anchors"
# self-test only (CPU): add --self-test  → "self-test PASS"
```

### Peak memory

**12.12 MiB** (CPU-only; no GPU, no vLLM, no model load).

### What happened

Worked cleanly. The packet's COST leg now has a dedicated owner. The single most decision-useful output is **`deployed_off_strict_frontier = True`**: because the deployed incumbent already ships 3 reduction-order flips at identity 0.9966, the human's #407 fork is **not** pristine-vs-dirty — it is a graded "how many more flips, and does PPL survive" question against a status quo the human already accepted. The ledger keeps the two costs the packet kept conflating — **flip-count (equivalence)** vs **PPL (quality)** — strictly orthogonal, and parks the relax identity/flips/PPL as clearly-marked UNMEASURED slots for stark #452 so a placeholder can never be read as a measurement. TPS-gain side of the decision rule is fully quantified now (+17.05 / +29.34); the other two clauses are explicitly pending stark #452's quality + flip-kind read.

One reconciliation worth flagging: directive4's earlier strict reading was identity **0.9989** (stark #412, 1 residual bitwise-tie flip @ prompt 90). I used the PR-given **1.0** (denken #423 `5a6zq2yz`), which supersedes it now that the residual tie was canonicalized (stark #429 → denken #423). Noted in the axis row so the lineage is auditable.

### Public evidence used

None required — this is a CPU-only reconciliation of **committed internal banked JSONs** (`#457` unified_absolute_ceiling, `directive4_correct_bar` shared_baselines) with no HF job, submission, or served-file change, so no public-board/leaderboard intake was needed. Source runs cited inline above.

### Suggested follow-ups

- **stark #452** is the one blocker to a fully-resolved decision rule: it fills the three relax slots (identity, flip-count, **measured PPL**) and confirms the flips are accumulation-order near-ties (same-kind). Once it lands, this ledger's clause-2 and clause-3 flip from PENDING → resolved with one number swap (the slots are already wired).
- For fern's capstone: this card + #457 are the matched **cost axis** and **prize axis**; they share the exact same anchors (round-trip resid 0.0), so they compose without re-derivation.
- The only remaining human input is the **TPS-gain threshold** in clause-1 — everything else is either measured or scheduled (stark #452).
