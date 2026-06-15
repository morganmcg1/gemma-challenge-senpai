STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["3u2urqzj"],"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"go_nogo_verdict":"HOLD-for-conjunction","selrec_excluded":true,"primary_metric":{"name":"blanket_strict_stack_tps","value":482.74},"test_metric":{"name":"deploy_gonogo_self_test_passes","value":1.0}}

## Results

**Analysis-only / 0-TPS card.** No build, no served-file change, no submission, no HF job. This re-anchors the #419 GO/NO-GO deploy decision surface off the **refuted selrec leg** onto the realizable **blanket-strict + cb3** stack, exposes the honest two-conjunct contingency, and pre-computes the verdict on current best estimates. Shipping remains human-approval-gated; this card produces the decision surface that would justify flagging one, it does not perform one.

### The re-anchored realizable stack

| quantity | value | source |
|---|---|---|
| **`blanket_strict_stack_tps` (PRIMARY)** | **482.74** | blanket-strict 467.14 + cb3 +15.60 |
| `margin_over_deployed_tps` | **+1.21** (knife-edge) | 482.74 − 481.53 |
| blanket-strict base | 467.14 ± 0.16 | stark #412 `blanket_strict_measured_tps` |
| cb3 supply (M=8, k*=229) | +15.60 | kanna #403 `m8_lift_at_kstar` (PPL-safe, equiv-neutral) |
| deployed non-strict #1 (ship-breakeven) | 481.53 / PPL 2.3772 / 128-128 | #52 `2x9fm2zx` (NO identity guarantee) |
| ~~selrec~~ (REFUTED, excluded) | 384.11 realizable / identity 0.9853 / 97.42 tax | stark #412 |

The deployed 481.53 is **non-strict** — it gives no identity guarantee. The strict stack's value proposition is `+1.21 TPS` **AND** a byte-identity guarantee the deployed config lacks. selrec is no longer the strict leg: #412 measured it at 384.11 realizable (the ~2.6-TPS model is fused-kernel-only) and identity-**degrading** (0.9853, 13 flips). The fastest *realizable* strictly-equivalent config is **blanket-strict** (`fastest_realizable_strictly_equivalent_config = blanket_strict`).

### The decision = a conjunction of TWO pending measurements

```
GO iff (measured_margin_tps > 0)     # kanna #416: measured(blanket-strict+cb3) > 481.53   [PENDING]
       AND (identity_value == 1.0)    # stark #421: canonical tie-break closes prompt-90 flip [PENDING]
       AND (ppl <= 2.42)              # cb3 PPL-safe, 2.3772 unchanged (OK by construction)
       AND (completed == 128)         # full public run (OK by construction)
```

**CURRENT VERDICT: `HOLD-for-conjunction`** — both decision inputs are pending:
- `conjunct_margin_green = False` — kanna #416 has not measured the combined stack yet (modeled +1.21).
- `conjunct_identity_green = False` — blanket-strict identity is **0.9989** (`residual_flip_count = 1`, one bitwise-tie flip @ prompt 90); 1.0 needs stark #421's canonical tie-break.

**What each input must reach for GO**
- `measured_margin_tps > 0` ⇒ the realized cb3 lift must clear **14.39 TPS** (a haircut **< 7.8%** on the modeled +15.60).
- `identity_value == 1.0` ⇒ stark #421's canonical tie-break closes the single prompt-90 flip.

Scenario grid (the decision surface): `modeled_margin + identity_closed → GO`; `worst-case-haircut(14.9%) + identity_closed → NO-GO (480.42, margin −1.11)`; `modeled_margin + identity_open → HOLD`; `margin_at_breakeven + identity_closed → NO-GO` (strict `>`).

### Binding contingency (ranked: **margin → identity**)

**Binding = `margin`.** *What kills this deploy:* if kanna #416 measures the cb3-over-blanket-strict additivity haircut above **~7.8%** (well inside ubel #410's ≤14.9% bound), the strict stack lands at or below the deployed 481.53, the +1.21 evaporates, and there is then **no TPS reason to ship the strict config at all** — a byte-identity guarantee with zero speed upside is a NO-GO.

- **Margin** is a *continuous* measurement risk: the failure region (7.8%, 14.9%] is ~48% of #410's admissible haircut band — roughly a coin-flip.
- **Identity** is the *lesser* risk: one *discrete* canonical tie-break at a single **true bitwise tie**. #412 proves precision cannot close it (`identity_1p0_unreachable_by_precision=True`), but a tie-break can. Note #405 (merged) showed a **global** lowest-id rule is RED (introduces 14 new flips, because the M=1 AR reference is not uniformly lowest-id) — so #421 must canonicalize **only true ties**, the de-risked successor; closable-by-construction.

### The exact deploy config a GO entails (BLANKET-STRICT, not selrec)

`deploy_config`: **PIN blanket-strict verify reduction (flag `STRICT_VERIFY_REDUCTION=1`) + ADD cb3 supply (k*=229). Selects BLANKET-STRICT, NOT selrec. Whole stack additive + reversible; human-gated.** `selrec_excluded = True`.

- **Attention/verify pin** — `submissions/fa2sw_treeverify_kenyan/{splitkv_verify_patch.py, fa_sliding_patch.py, manifest.json}`: pin the served reduction to the **high-precision (blanket-strict) reduction everywhere** — the strict reference path. NOT the deployed non-strict fast reduction, NOT the refuted selrec eps-near-tie kernel. No new kernel: blanket-strict reuses the high-precision reduction the verify already supports, applied unconditionally. Flag `STRICT_VERIFY_REDUCTION` ON=blanket-strict (ship), OFF=today's-served (rollback); `selects_selrec=False`.
- **cb3 supply** — the 6 ADDITIVE files / 0 in-place edits from the #417 ledger (cb3 QTIP/QuIP# kernel wheel + quant patch + manifest fork + serve fork + sitecustomize fork + cb3-baked checkpoint bucket). Orthogonal subsystem (body-GEMM quant) → stacks additively in one combined submission + one checkpoint.

### Deliverables (W&B run `3u2urqzj`, group `blanket-strict-deploy`)

| field | value |
|---|---|
| `blanket_strict_stack_tps` (PRIMARY) | 482.74 |
| `margin_over_deployed_tps` | +1.21 |
| `go_nogo_verdict` | HOLD-for-conjunction |
| `conjunct_margin_green` | False (kanna #416 pending) |
| `conjunct_identity_green` | False (stark #421 pending) |
| `identity_conjunct_value` | 0.9989 |
| `residual_flip_count` | 1 (@ prompt 90) |
| `binding_contingency` | margin |
| `deploy_config` | blanket-strict pin + cb3 (NOT selrec) |
| `selrec_excluded` | True |
| `deploy_gonogo_self_test_passes` | True (83/83 checks) |

W&B markers: `analysis_only=True`, `no_hf_job=True`, `no_served_file_change=True`, `official_tps=0`.

### Rigor

- **Self-test: 83/83 PASS.** Predicate is a true AND-gate (any bad conjunct ⇒ not GO; missing margin ⇒ HOLD); scenario grid validated; flag confirmed to NOT select selrec.
- **Pinned-import cross-check: 15/15 byte-exact** against the merged advisor-branch JSON — stark #412 (`blanket_strict_measured_tps=467.14`, `selective_recompute_measured_tps=384.11`, `served_identity_after_selective=0.9853`, `fastest_realizable_strictly_equivalent_config=blanket_strict`, `selective_beats_blanket=False`, `identity_1p0_unreachable_by_precision=True`), kanna #403 (`m8_lift_at_kstar=15.60`, `k_star=229`), ubel #410 (`delta_demand_tps_frac_of_lift=0.149`, `supply_demand_additive=True`), lawine #419 (deployed_tps/ppl_cap/cb3_kstar). No constant re-derived.

### Public / merged evidence used

All inputs are merged artifacts on `approval-gated-8gpu-20260613`: #412 (selrec refutation + blanket-strict base + identity), #403 (cb3 supply k*=229), #410 (supply×demand additivity ≤14.9% haircut), #417 (deploy-surface ledger / cb3 6-file surface), #419 (prior GO/NO-GO predicate + verify CI), #405 (global tie-break RED — informs the identity-conjunct risk). Pending decision inputs (named in the assignment): kanna #416 (measured margin), stark #421 (canonical tie-break identity).

### Command

```bash
.venv/bin/python research/validity/blanket_strict_deploy_gonogo_reanchor/blanket_strict_deploy_gonogo_reanchor.py \
  --wandb_name "lawine/blanket-strict-deploy-gonogo-reanchor" --wandb_group "blanket-strict-deploy"
# self-test: python3 .../blanket_strict_deploy_gonogo_reanchor.py --self-test  (83/83 PASS)
```

Peak memory: N/A (0 GPU compute — pure static analysis). Human-readable GO/NO-GO checklist emitted to `research/validity/blanket_strict_deploy_gonogo_reanchor/GO_NOGO_CHECKLIST.md`.

### What happened — honest analysis

The re-anchor **shrinks the case for shipping the strict config from comfortable to knife-edge.** #419 priced the strict stack on the modeled selrec leg at a +11.55 margin over deployed; once stark #412 refuted selrec (realizable 384.11, identity-degrading), the *realizable* strict leg is blanket-strict at 467.14, and the whole upside now rides on cb3's +15.60 paying for the ~14.4-TPS blanket-strict verify tax with only **+1.21 left over**. That +1.21 is genuinely fragile: it survives only if #410's additivity haircut stays below ~7.8%, and #410 admits up to 14.9% — so roughly half the admissible band erases it. The identity conjunct is the safer of the two (a single true-bitwise-tie flip, closable by #421's canonical tie-break), but it is still a real pending input and #405 shows a naive global tie-break backfires.

**Honest bottom line: HOLD.** A GO requires BOTH (i) kanna #416 to measure the combined stack strictly above 481.53, and (ii) stark #421 to drive identity to exactly 1.0. The binding risk is the **margin** — if #416 comes back ≤481.53 the strict config has no speed justification and the right call is to keep the deployed non-strict 481.53 (or pursue a larger supply lever before re-pricing). The deploy this card would flag is unambiguously **blanket-strict + cb3**, never selrec.

### Suggested follow-ups

1. **Resolve the binding conjunct first.** Prioritize kanna #416's *combined* (blanket-strict + cb3) end-to-end measurement over the identity work — it is the variable most likely to flip the verdict, and a sub-481.53 reading would moot the tie-break entirely.
2. **De-risk the knife-edge before committing quota.** Because the margin is ~half-a-band from flipping, consider whether a larger equivalence-neutral supply lever (or a higher-confidence cb3 k* with a wider PPL margin) could lift the stack off the knife-edge so a GO is robust rather than +1.2.
3. **Confirm #421 canonicalizes true ties only.** Given #405's global-rule failure, verify stark #421 applies the canonical tie-break strictly at `m1_self_gap=0.0` positions so it cannot reintroduce flips elsewhere.
4. If both conjuncts land GREEN, this card's `GO_NOGO_CHECKLIST.md` is the ready human-approval doc — pair it with the #319 3-tier identity-verify CI (~41.8 GPU-min) before any flag to the human.
