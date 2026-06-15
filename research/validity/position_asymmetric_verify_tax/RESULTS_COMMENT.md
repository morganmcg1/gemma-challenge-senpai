STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["uc7jg6vs"],"no_hf_job":true,"official_tps":0.0,"analysis_only":true,"no_served_file_change":true,"can_shave_m8_tax_below_2p6":false,"shave_is_byte_identity_safe":false,"shaved_equiv_tax_tps":2.6,"equiv_tps_at_shaved_tax":478.93,"equiv_tps_gain_vs_uniform478p93":0.0,"flip_position_granularity":"per_position","can_shave_robust_across_position_models":true,"equiv_tax_at_m8_used":2.6,"position_asymmetric_verify_tax_self_test_passes":true,"primary_metric":{"name":"equiv_tps_at_shaved_tax","value":478.93},"test_metric":{"name":"position_asymmetric_verify_tax_self_test_passes","value":1.0}}

## Results

**Headline — the shave is REFUTED by a HARD strict-equivalence gate. `can_shave_m8_tax_below_2p6 = False`, robustly. The M=8 equiv-tax stays at the full 2.6 TPS; `equiv_tps_at_shaved_tax = 481.53 − 2.6 = 478.93` (== #413 `equiv_tps(7)`).** Per-position asymmetric verify precision cannot beat uniform precision under strict byte-identity, for two measured, independent reasons: (1) the near-tie population at gap ≤ eps*=0.125 **blankets all 7 readable chain positions** (counts [6,7,7,1,7,5,7], 40 near-ties, 37 currently-correct knife-edge), so **no chain row is free to leave fast**; and (2) the down-precision perturbation ceiling (0.125) **equals** eps* (0.125) — a knife edge with **zero proof margin** — so no row is *provably* flip-safe even in principle. This pins **2.6 as the irreducible per-position floor** for this lever (fern #357 rollup). It resolves my own #413 suggested-follow-up #2.

### Headline fields (PR deliverable)

| field | value |
|---|---|
| `position_asymmetric_verify_tax_self_test_passes` (**PRIMARY**) | **True** (54/54 checks, ≥20 required) |
| `can_shave_m8_tax_below_2p6` | **False** |
| `shave_is_byte_identity_safe` | **False** |
| `shaved_equiv_tax_tps` | **2.6** (uniform precision forced; protect 8/8 rows) |
| `equiv_tps_at_shaved_tax` | **478.93 TPS** (= 481.53 − 2.6) |
| `equiv_tps_gain_vs_uniform478p93` | **+0.000 TPS** |
| `flip_position_granularity` | **per_position** (#405 `j` field, used directly — no bounding model needed) |
| `can_shave_robust_across_position_models` | **True** |
| `equiv_tax_at_m8_used` | 2.6 (#397; one-line calibratable, stark #412 supersedes) |
| scope | `analysis_only`=True, `no_hf_job`=True, `no_served_file_change`=True, `official_tps`=0 |

### Per-position structure (reach weights + near-tie census, from the #289 ladder and #405 j6h228xy)

| j (chain pos) | w_p = P(reached) | accept mass | near-ties @ eps* | flip? |
|--:|--:|--:|--:|:--:|
| 1 | 1.00000 | 0.72925 | 6 | no |
| 2 | 0.72925 | 0.55391 | 7 | no |
| **3** | 0.55391 | 0.43924 | 7 | **FLIP** (×2: prompts 18, 118) |
| 4 | 0.43924 | 0.36141 | 1 | no |
| 5 | 0.36141 | 0.30173 | 7 | no |
| 6 | 0.30173 | 0.25218 | 5 | no |
| **7** | **0.25218** | 0.21347 | 7 | **FLIP** (prompt 11) |

- **Consistency self-test (instruction 2):** Σ accept mass = 2.851186 = E[accepted] (#289) to residual **0.00e+00** (< 1e-6). ✓
- **Flip-position map is `per_position`** (instruction 3): the #405 artifact records each flip's chain position via the `j` field, so I use it directly (no uniform/concentrated bounding model required). The 3 served flips are at **(prompt 11, j=7), (18, j=3), (118, j=3)** → distinct positions **{3, 7}**.
- **One flip is at j=7, the LOWEST-reach position (w₇=0.252), and it WAS served.** This is a measured counterexample to the hypothesis's marginal-reach mechanism ("late positions rarely the emitted-token source → a flip there rarely matters"). It matters: it's one of the 3 served flips.

### The HARD byte-identity safety gate (instruction 5 — the decider)

A shave is valid only if leaving a chain row **fast** provably keeps all 882 emitted sequences byte-identical. A down-precisioned row carries a worst-case reduction-order perturbation; it is provably safe **only if** (i) it hosts **no** near-tie at gap ≤ eps* on any eval prompt **AND** (ii) the perturbation is **strictly below** the margin. Both fail:

| quantity | value | reading |
|---|---|---|
| down-precision perturbation ceiling | **0.125** (1 bf16-ULP final cast; #87/#381/#405) | the max \|Δlogit\| a left-fast row carries |
| near-tie margin eps* | **0.125** (= 16 bf16-ULP at the flip magnitude; #405) | the 3 flips sit at gap == 0.125 exactly |
| `perturb_max ≥ eps*` (knife edge) | **True** | (ii) fails — **no proof margin** at any row |
| thinnest global gap (#87, 65,536 pos) | **0.03125** < 0.125 | sub-perturbation near-ties demonstrably exist |
| near-ties populate all 7 rows | **True** (min 1 at j=4) | (i) fails — **no row hosts zero near-ties** |
| **sparable rows** | **[ ] (none)** | nothing can be left fast → protect 8/8 |

⇒ `shave_is_byte_identity_safe = False` → `can_shave_m8_tax_below_2p6 = False`. Uniform precision is forced; `shaved_equiv_tax_tps = 2.6`, `equiv_tps_at_shaved_tax = 478.93`.

**[GATE-OFF / NOT ACHIEVABLE, for completeness]** If only the 2 observed-flip rows {3,7} needed protection (2/8): `tax = 2.6·2/8 = 0.65` → `equiv_tps = 480.88`. The strict gate forbids this; the **+1.95 TPS** gap is exactly the price of strict byte-identity over the optimistic expected-value read. The asymmetric idea looks attractive *in expectation* and is *forbidden* under strict equivalence.

### Robustness across position models (instruction 3)

Because granularity is `per_position`, the verdict is **measured, not modeled**. I still ran the two distributional null models the PR asks for, fit to the measured near-tie total (40):
- **Uniform model:** near-ties at all 7 rows → no sparable row → `can_shave = False` (and the proof-margin failure forbids a shave regardless).
- **Concentrated-late model** (weighted by 1−w_p): predicts the highest-reach row j=1 is near-tie-free → would *naively* permit sparing j=1 → `can_shave = True`. **But the measurement refutes it: j=1 hosts 6 near-ties.** The premise "near-ties concentrate at low-margin late positions" is empirically false — the measured distribution [6,7,7,1,7,5,7] is ~uniform across early and late rows.

⇒ `can_shave_robust_across_position_models = True`: the only model that would have permitted a shave is directly refuted by the per-position data.

### Greedy identity (exact by construction, PPL unchanged)

The linear-chain spec verify emits the target's argmax token at every position (the drafter only *proposes*), so the emitted token is the target greedy token regardless of how the verify reduction is precisioned → **PPL unchanged 2.3772 ≤ 2.42**. The equiv-tax is purely the cost of making the M=8 *batched* verify byte-identical to the M=1 sequential reference (removing the #381/#405 reduction-order flips); this card asks only *how to allocate that precision across positions*, never changing which token is emitted. The tree dimension (M>K+1) is closed negligible by my #409 (+1.33 TPS, β-fragile); scope is the linear chain only.

### Reproduce (0-GPU, stdlib-only)

```bash
cd target/ && .venv/bin/python -m research.validity.position_asymmetric_verify_tax.position_asymmetric_verify_tax --self-test
cd target/ && .venv/bin/python -m research.validity.position_asymmetric_verify_tax.position_asymmetric_verify_tax \
  --wandb_group position-asymmetric-verify-tax --wandb_name denken/position-asymmetric-verify-tax
# calibrate to stark #412's measured M=8 tax (one-line equivalent): append --equiv-tax-m8 <value>
```

- **Peak memory:** 13.7 MiB (pure-CPU, no GPU, no HF Job, no submission, no served-file change)
- **W&B run:** `uc7jg6vs` (entity wandb-applied-ai-team, project gemma-challenge-senpai; 102 summary keys, `summary/` prefix)
- **Self-test:** **54/54** checks pass — provenance (anchors byte-exact from merged #413 + the #405 artifact, with a runtime cross-check that pinned constants match the raw `arm_heuristic_result.json`), reach-weight consistency, flip-map granularity, the safety gate, the verdict numbers, the gate-off optimistic sanity, robustness, PPL, calibration knob, numeric hygiene.

### What happened — honest analysis

The hypothesis (asymmetric per-position precision shaves the M=8 tax below 2.6 by exploiting flip concentration) **does not survive the strict byte-identity gate**, and the per-position data is unusually clean about *why*. The optimistic intuition has two parts and the data kills both:

1. **"Flips concentrate by chain position."** *Partly true but irrelevant.* The 3 served flips do concentrate (at {3, 7}). But the safety gate isn't about the 3 *observed* flips — it's about the **37 other currently-correct near-ties** that a down-precision could *newly* flip. Those 37 (plus the 3) sit at gap ≤ eps* = 0.125 and **blanket all 7 readable rows**. To leave any row fast you must prove it hosts no flippable near-tie; every row hosts 5–7 (j=4 the lone exception at 1). No row qualifies.
2. **"Late positions rarely reach the output, so a flip there is cheap."** *Refuted by a measured counterexample.* This is an expected-value argument; strict byte-identity is not an expectation — it forbids a single served flip. One of the 3 served flips is at **j=7, the lowest-reach position (w₇=0.252)**, and it was served. "Rarely" ≠ "never."

The deeper reason is a **zero-margin knife edge**: the deployed atomic-off / fp32-reduce regime caps the batched-verify reduction-order divergence at exactly ±1 bf16-ULP (0.125), which is *exactly* the near-tie gap at which flips occur (the 3 flips have gap = 0.125). So the perturbation a down-precisioned row would carry is exactly large enough to flip a knife-edge near-tie — there is no slack to prove safety, and the global lm_head margin map (#87) confirms gaps as thin as 0.03125 < 0.125 exist in the population. This is the same wall #405 hit one layer up (the *resolution* of a flagged near-tie needs the precise value, not a free shortcut); #418 shows the wall also blocks **spatial** (per-position) precision allocation, not just the resolution rule. Net: **the only correct equivalent fix remains the uniform #397 selective recompute at 2.6 TPS; the asymmetric refinement reclaims nothing.** When stark #412 lands its *measured* M=8 tax, swap it into `EQUIV_TAX_AT_M8` (one line) and `equiv_tps_at_shaved_tax` updates directly — the verdict (can_shave=False) is tax-level-invariant because it rests on the per-position blanket + the knife edge, not on the magnitude 2.6.

### Suggested follow-ups

1. **Treat 2.6 (→ #412's measured value) as a hard floor for the equivalent path and stop pursuing precision-allocation shaves** (spatial per-position here; logit-/id-level in #405; resolution-rule in #397 follow-ups). All three are now closed: the residual is a *value-precision* phenomenon at a zero-margin knife edge, immune to flagging, id-ordering, and spatial sparsity. Any identity-1.0 lever must lower the *uniform* recompute cost itself (faster high-precision reduction), not skip work.
2. **The only remaining tax-reduction lever is making the high-precision reduction itself cheaper** (e.g., a faster fp32-accumulate verify path that is still byte-identical, or reducing the flagged-step *fraction* 23.6% rather than the per-step cost). That is an equiv_tax-*engineering* question on the served path (human-approval-gated), distinct from this analytic closure.
3. **If a future drafter/kernel narrows the reduction-order divergence below 1 bf16-ULP** (e.g., a verify that casts to bf16 only after a wider-than-final-cast comparison, giving perturb_max < eps*), re-run this card: a *strict* proof margin would reopen the per-position shave for any row that is then provably near-tie-free. Under today's deployed atomic-off / fp32-reduce regime the margin is exactly zero, so the door is shut.

### Public evidence used

Human re-scope **#407** (maximize fastest strictly-equivalent TPS, forget 500). Banked byte-exact: my merged **#413** (`se8mf9ax`) for MU_P=481.53 (#52 `2x9fm2zx`), BASE_467 (#393 `0q7ynumg`), EQUIV_TAX_AT_M8=2.6 (#397) + selective band [476,479], the #289 ladder + E[accepted]=2.851 (`fi34s269`), and `equiv_tps(7)`=478.93. The per-position flip + near-tie census is read byte-exactly from stark **#405** (`argmax_tiebreak_zero_cost_semantic/arm_heuristic_result.json`, run `j6h228xy`): 3/882 served flips @ prompts 11/18/118 with per-row `j` ∈ {7,3,3}, 40 near-ties @ eps*=0.125 distributed [6,7,7,1,7,5,7], 37 currently-correct. Perturbation ceiling 0.125 + thinnest gap 0.03125 from kanna **#87** verify-argmax-margin map (`875cujdk`, 65,536 positions). Identity tax anchor #397 (stark **#412** measuring → supersedes via the `EQUIV_TAX_AT_M8` one-liner). Tree dimension closed by my #409 (+1.33 TPS, β-fragile). Nothing re-derived; the only new modelling is the per-position reach weights, the position-targeted tax model, and the byte-identity safety gate.
