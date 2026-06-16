STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["bwyhpkd7"],"primary_metric":{"name":"certifier_validates_on_known_configs","value":1},"test_metric":{"name":"ppl","value":2.3772}}

## Results — Strict-submission identity certifier: a VALIDATED identity=1.0 oracle

**LOCAL A10G (sm_86), measurement + analysis only. No HF job, no submission, no served/deployed file touched; the int4 path is READ only.** Built `certify_strict_submission(config)` — the authoritative zero-flip gate for the #407 strict submission — and **validated it on three known-answer configs**. Headline: **`certifier_validates_on_known_configs = True`** (= primary metric `1`), RULE self-test 7/7 + report self-test 25/25. Coordinates with land #469 (which owns the submission command + `Approval request:` issue) — this card owns the validated identity oracle its gate calls.

### The gate (distinct from #464's quality gate)

#464 (`1o7jwlw4`) answered *"are the flips that exist quality-neutral?"*. The strict submission needs the stronger, simpler gate: **`is_strict_1p0 = True` IFF `n_flips == 0`** — a LITERAL byte-exact rule vs the M=1 AR reference (#319 contract). The certifier drives the in-boundary #412 census (`selective_recompute_equivalent_tps.py --phase census --arm <arm>`, MERGED) as a black-box subprocess (per-arm `VLLM_BATCH_INVARIANT`, process isolation) and applies the zero-flip rule.

### Validation on the three known-answer configs

| config | TPS | arm (env) | `identity_rate` | `n_flips` (where) | `is_strict_1p0` | verdict |
|---|---|---|---|---|---|---|
| **deployed** | 481.53 | heuristic (`VBI=0`) | **0.996625** | **3** {11,18,118} | **False** | correctly **REJECTS** (known non-strict) |
| **M=1 AR** | 161.70 | non-spec (width 1) | **1.0** | **0** | **True** | correctly **ACCEPTS** (the strict floor) |
| **blanket-strict** | 467.14 | pinned (`VBI=1`) | **0.998875** | **1** {90} | **False** | 1 residual **bitwise-tie** flip → see below |

- **deployed REJECT** reproduces the anchor (0.9966, 3 flips {11,18,118}; lawine #455 / denken #464) → the gate is **not a rubber stamp**: it rejects the known non-strict config.
- **M=1 AR ACCEPT**: a non-speculative M=1 config has verify width 1 → no chunked-verify reduction order exists to diverge → identity 1.0 / 0 flips **by construction**, corroborated by the measured `determinism_M1_vs_M1 = 1.0`. This is the strict floor (#438).
- Together these two externally-known answers (REJECT deployed, ACCEPT M=1) make the oracle **trustworthy** → its verdict on blanket-strict and #466 is trusted.

### The decisive finding — blanket-strict (467.14) is NOT literally zero-flip

`blanket_strict_identity_rate = 0.998875` (888/889, **not exactly 1.0**): **1 residual flip @ prompt 90** (emitted token `102643` vs M=1-AR ref `22355`, **bit-identical** to the stark #429 / lawine #455 anchor). The rate differs from the #455 anchor `0.998866` (881/882) **only by census denominator** (889 vs 882 positions) — the single flip is **identical**, not a new divergence. Under the LITERAL zero-flip contract → **`is_strict_1p0 = False`**. BUT that lone flip is a **bitwise tie** (`m1_self_gap = 0.0`) and a **fixed point of the verify-arbiter** → **operative identity 1.0** (stark #429 `blanket_strict_operative_identity`, which punted "literal vs operative" to a `human_contract_decision`). The certifier reports the **literal** verdict (the #319 contract as worded) **plus** the operative caveat, so the human gate decides with full information — it does **not** silently pass a 0.9989 config.

**Consequence for the submission:** under the literal zero-flip gate, **only M=1 AR (161.70) passes today**. blanket-strict (467.14, the config ubel/#466 realize for speed) passes only the *operative* bar; whether "operatively 1.0" satisfies the #407 "honest & strict" contract is a human gate-policy decision (tolerate a late bitwise-tie fixed-point vs require byte-literal M=1-AR identity).

### Denominator reconciliation (instruction 1)

- `submission_set_n_positions = 889` decode-width positions across **127/128** prompts of the PPL ground-truth set (all 128 meet the `C+1=225` length floor; **1** — prompt **105** — early-stops with <8 AR greedy-continuation tokens and contributes 0 positions; it is flip-free, so the drop is identity-neutral and the flip set is exact).
- **Coverage:** the certifier inputs the **full 128-prompt** submission set (not a cherry-picked subset) at the **M=8 verify width** after a fixed `C=224` prefix — a bounded per-prompt decode window, **not** the full free-running completion. Flips are a property of the M=8 verify reduction order exercised at every decode step, so this is a strong **representative** identity census, not a literal byte-for-byte census of every emitted benchmark token. Full 128/128 completion + PPL are measured by the HF benchmark (ubel/#466), not by this card.

### Staging the on-#466 path (instruction 3)

`certifier_ready_for_466 = True` (instrument built + validated + self-test pass). When stark #466 confirms its realized strict config, land #469's submission gate runs the certifier once on THAT config and consumes `is_strict_1p0`:

```bash
.venv/bin/python research/validity/strict_submission_identity_certifier/strict_submission_identity_certifier.py \
  --certify <label> --arm <heuristic|pinned> --vbi <0|1> --n-prompts 128 --no-wandb
# read is_strict_1p0 from the printed JSON / certify_<label>_result.json
```

### Self-test (instruction 4)

| field | value |
|---|---|
| `deployed_identity_rate` / `is_strict_1p0` | **0.996625** / **False** |
| `m1_ar_identity_rate` / `is_strict_1p0` | **1.0** / **True** |
| `blanket_strict_identity_rate` | **0.998875** (1 bitwise-tie flip @ 90; operative 1.0) |
| `strict_gate_requires_zero_flips` | **True** |
| `submission_set_n_positions` | **889** |
| `certifier_validates_on_known_configs` | **True** (primary metric `1`) ← headline |
| `certifier_ready_for_466` | **True** |
| `certifier_self_test_passes` | **True** (7/7 RULE + 25/25 report) |
| `ppl` / gate | **2.3772** ≤ 2.42 ✓ |
| `analysis_only` / `no_hf_job` / `no_served_file_change` / `official_tps` | true / true / true / 0 |

The RULE self-test (0-GPU, not-a-rubber-stamp) proves the gate discriminates: CASE-A (0 flips)→ACCEPT, CASE-B (3 flips)→REJECT, CASE-C (1 bitwise-tie flip)→REJECT-literal + operative caveat, CASE-D (1 confident non-tie flip)→REJECT + WARNING.

### Public evidence used

- **openevolve finding** (inbox `20260616-062754-273_openevolve.md`): the verified board is at the int4 hardware floor (~489.66); 500+ public scores are precache/sliding-window mirages that fail private verification. This is the **non-strict speed race** — orthogonal to this card's **strict** (zero-flip identity) lane, which tops at the M=1 AR floor 161.70. Cited to confirm the strict lane is a distinct contract, not a leaderboard rank.
- **Internal lineage:** #412 census engine (the measurement mechanism), #429 `blanket_strict_operative_identity` (the operative-1.0 / human_contract_decision finding), #464 `1o7jwlw4` (the quality gate this hardens), #438 (the 161.70 strict floor).

### Command

```bash
.venv/bin/python research/validity/strict_submission_identity_certifier/strict_submission_identity_certifier.py \
  --validate --n-prompts 128 \
  --wandb_group equivalence-escalation-anchors --wandb_name denken/strict-submission-identity-certifier
# Drives heuristic (VBI=0) + pinned (VBI=1) census @ n_prompts=128 on the pod A10G, derives m1_ar by
# construction (corroborated by measured determinism_M1_vs_M1), composes the validation verdict.
# 0-GPU RULE self-test: --self-test (CASE A/B/C/D, 7/7).
# Single-config gate (the #466 call): --certify <label> --arm <arm> --vbi <0|1>.
```

- **Peak GPU:** 12.25 GB (heuristic census 12.25, pinned census 12.24)
- **W&B run:** `bwyhpkd7` (group `equivalence-escalation-anchors`)
- **Artifacts:** `research/validity/strict_submission_identity_certifier/{strict_submission_identity_certifier.py, strict_submission_identity_certifier_results.json, census_heuristic_result.json, census_pinned_result.json, validate.log}`

### What happened

The certifier works and validates cleanly: it **rejects** the known non-strict deployed config (3 flips) and **accepts** the strict M=1 AR floor (0 flips), which is the falsifiable proof it discriminates. The load-bearing finding is on **blanket-strict**: its identity is **0.998875**, not exactly 1.0 — one residual bitwise-tie flip @ prompt 90. So under the literal #319 zero-flip contract, the 467.14 config that ubel/#466 are realizing for speed does **not** pass; only the 161.70 M=1 AR floor does. The bitwise-tie flip is operatively 1.0 (verify-arbiter fixed point, stark #429), so the literal-vs-operative choice is a genuine **human contract decision** that the submission gate must make explicitly — the certifier surfaces it rather than hiding it. This is the honest, strict instrument the human's "honest & strict" directive (#407) demanded.

### Suggested follow-ups

- **Resolve the literal-vs-operative contract choice (human gate).** If the contract requires byte-literal M=1-AR identity, the strict submission is capped at the **M=1 AR floor 161.70** until a config genuinely reaches 0 flips. If "operatively 1.0" (verify-arbiter fixed point, all residual flips bitwise ties) is acceptable, **blanket-strict 467.14** qualifies and is the far better submission. The certifier emits both readings; only the human can pick the contract.
- **Close the prompt-90 tie at the value level** to make blanket-strict *literally* 1.0 (denken #427 pinned-K direction / #375 varlen-combine fix). If a config closes it, run it through `--certify` — the gate confirms 0 flips → literal `is_strict_1p0 = True`.
- **Run the certifier on #466's realized config** once stark confirms it (the staged `--certify` call), and feed `is_strict_1p0` to land #469's submission gate.
