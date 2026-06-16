STUDENT lawine:
SENPAI-RESULT {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["vjhef1du"],"analysis_only":true,"no_served_file_change":true,"official_tps":0,"canonical_blanket_strict_tps":467.14,"directive4_banked_win_tps":15.60,"directive4_bar_is_feasible_set_max":true,"leaderboard_beat_margin_tps":1.21,"bar_B_haircut_flip_threshold_pct":7.76,"ships_if_bar_B_negative":true,"self_test_passes":true,"primary_metric":{"name":"directive4_banked_win_tps","value":15.60},"test_metric":{"name":"self_test_passes","value":1.0}}

## Results

**0-GPU decision-ledger / reconciliation. No served-file change, no HF job, no submission. Pure static analysis + 80-check self-test, like #425.** New additive file tree `research/validity/directive4_correct_bar/`; #425's `GO_NOGO_CHECKLIST.md` left in place. Baseline **481.53 TPS / PPL 2.3772 / 128-128** (#52 `2x9fm2zx`) UNCHANGED.

### Verdict — yes, 482.74 is a banked +15.60 equivalent win against the *correct* Directive-#4 bar

Under #407's literal objective (*"improve on the TPS that is the fastest that still respects this equivalence"*) the max is taken over the **feasible set** — configs that *respect* the equivalence — only. The deployed **481.53** is the **non-strict** fast path (3/882 M=8 reduction-order flips, identity **0.9966**, #405 `j6h228xy`), so it does **not** respect the equivalence and is **excluded from the max**. It is therefore the *wrong* bar. The mis-anchoring on 481.53 is what made #425's "+1.21 knife-edge" look binding.

### 1. Feasible-set ladder (strictly-equivalent configs, lowest → highest)

| # | Config | TPS | Identity | In feasible set | W&B |
|---|--------|-----|----------|-----------------|-----|
| 1 | pure-AR-greedy (M=1 reference) | `unmeasured` in-cycle (~165.44 **official est**, #196) | **1.0** (by definition) | ✅ | #196 |
| 2 | blanket-strict | **467.14** (local-measured, σ=0.16) | 0.9989 | ✅ | #412 `dnjvqbtf` |
| 3 | blanket-strict + cb3 (k\*=229) | **482.74** (= 467.14 + 15.60) | 0.9989 | ✅ | #425 `3u2urqzj` + #403 `iv9i2wks` |
| — | ~~deployed fast path~~ **(EXCLUDED)** | 481.53 | 0.9966 | ❌ | #52 `2x9fm2zx` |

The M=1 reference TPS is marked `unmeasured` in-cycle: the only banked number (165.44, #196 `nonspec_official_tps_est`) is an **official-hardware estimate**, not regime-consistent with the local 467.14/482.74 — so I do not fabricate a local figure. Its role here is the identity floor (1.0 by definition).

### 2. Blanket-strict base reconciliation → `canonical_blanket_strict_tps = 467.14`

- **467.14** (#412 `dnjvqbtf`): a **direct** end-to-end local A10G measurement of the blanket-strict *served* stack (σ=0.16, within-config identity 0.9989).
- **467.48** (#393 `0q7ynumg`): a **decode-η back-projection** (`deployed_tps_decode_eta`=467.475 = deployed base × η_attn 3.01%) — an analytic projection of the idealized strict path, not a measurement.
- **Canonical = 467.14.** It is the direct measurement; the cb3 +15.60 lift (#403) was measured on *this* base, so the stack 482.74 = 467.14 + 15.60 is internally consistent. The two agree to **+0.34 TPS (0.073%, ~2.1σ)** — a confirming cross-check at the 0.1% level. The projection is marginally optimistic, so 467.14 is also the conservative choice.

### 3. The two bars (stated explicitly)

**Bar A — Directive-#4 (fastest prior *feasible*): 467.14.**
`directive4_banked_win_tps = 482.74 − 467.14 = +15.60 (+3.34%)`. **No margin contingency:** Bar A's margin *is* the realized cb3 lift, so even under ubel #410's worst-case **14.9%** additivity haircut the realized lift is **+13.28** (stack 480.42) → Bar A stays **+13.28**, strictly positive across the *entire* [0, 14.9%] haircut band. **No kanna #416 measurement can make Bar A negative.**

**Bar B — leaderboard-beat (illegal incumbent): 481.53.**
`leaderboard_beat_margin_tps = +1.21` (modeled, contingent on kanna #416 budget-exact). The cb3 lift must clear 481.53 − 467.14 = **14.39 TPS**, so Bar B flips **negative above a ~7.76% haircut** (`bar_B_haircut_flip_threshold_pct`), well inside #410's 14.9% bound. Worst-case Bar B margin **−1.11**. This is the knife-edge #425 surfaced — but it is a **secondary, stricter, leaderboard-cosmetic** bar.

### 4. Ship-policy under Bar B negative → `ships_if_bar_B_negative = True`

If kanna #416 measures 482.74 < 481.53, the stack **still ships as the fastest *feasible* config** under #407: 481.53 is excluded from the max (identity 0.9966), and even in the worst-haircut world the cb3 stack (480.42) still strictly dominates the next feasible config (blanket-strict 467.14). Bar B is a public-leaderboard bonus, **not a #407 gate**.

**Conditionality (honest):** this rests on the *current* operative-identity partition (481.53 → infeasible). **Stark #429** (operative-identity resolution, in flight; successor to #421's canonical tolerance tie-break on the self-referential gate #414 `bq7xkfcv`) is the **separate** pending input that fixes the partition. **#429 can only help:** (i) if it leaves 481.53 infeasible → Bar A holds, +15.60 banked; (ii) if it canonicalizes the fast path's bitwise-tie flips and *promotes* 481.53 into the feasible set → the bar rises to 481.53 but the same cb3 +15.60 stacks on the faster base (fast + cb3 ≈ 497), so the supply win is **preserved/enlarged**. Either way the cb3 +15.60 is banked; #429 only selects *which* strict base it stacks on.

### 5. Superseding GO card (additive)

New `research/validity/directive4_correct_bar/GO_NOGO_CHECKLIST.md` **leads with the banked Directive-#4 win**, demotes Bar B to a cosmetic line, and keeps **identity as a separate row citing stark #429** (not re-derived here; cb3 is equivalence-neutral, adds 0 flips, so the residual is the same single blanket-strict base flip @ prompt 90, #412). **Current verdict: HOLD-for-IDENTITY-ONLY** — a strict collapse of #425's 2-conjunct HOLD (margin + identity) to a **1-conjunct HOLD** (identity only), because the margin conjunct was mis-anchored on the wrong bar.

### Comparison against the PR baseline

| Quantity | PR body | This card | Match |
|---|---|---|---|
| canonical blanket-strict | 467.14 (#425) vs 467.48 (#393) | **467.14** (reconciled, conservative) | ✅ |
| directive4 banked win | +15.60 (+3.34%) | **+15.60 (+3.34%)** | ✅ |
| 481.53 excluded from feasible max | yes | `directive4_bar_is_feasible_set_max=True` | ✅ |
| leaderboard-beat margin (Bar B) | +1.21 modeled | **+1.21**, flips <0 above 7.76% haircut | ✅ |
| ships if Bar B negative | yes | `ships_if_bar_B_negative=True` | ✅ |

### Command

```bash
cd target/
# self-test (0 GPU, no wandb):
python research/validity/directive4_correct_bar/directive4_correct_bar.py --self-test
# full card + wandb (run under the repo .venv that has wandb):
.venv/bin/python research/validity/directive4_correct_bar/directive4_correct_bar.py \
  --wandb_name "lawine/directive4-correct-equivalent-bar" --wandb_group "directive4-correct-bar"
```

- **Self-test:** 80/80 PASS (arithmetic 482.74 = 467.14 + 15.60; both bars; feasibility partition; additive-file + no-served-file hygiene; 12 pinned-import cross-checks against merged #412/#403/#410/#393/#196 JSON).
- **Peak memory:** 0 GPU (CPU-only static analysis; no torch/vLLM/model load).
- **W&B run:** `vjhef1du` — https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/vjhef1du

### Public evidence used

- **Issue #407** (human directive, 2026-06-15 21:13:17Z) — the literal re-scope text that defines the feasible-set objective.
- **Deployed incumbent** #52 `2x9fm2zx` (481.53 / 0.9966) — the leaderboard row this card argues is *infeasible* under #407.
- Merged advisor-branch runs cross-checked byte-exactly: #412 `dnjvqbtf`, #393 `0q7ynumg`, #403 `iv9i2wks`, #410 `7rzf74q5`, #196 `floor_report.json`, #405 `j6h228xy`.

### What happened

The hypothesis holds. The +1.21 "knife-edge" that #425 made binding was an artifact of pricing against **481.53**, which #407 *excludes from the feasible set*. Re-priced against the correct bar (the fastest config that actually respects the equivalence, blanket-strict **467.14**), the realizable strict stack **482.74** is a clean **+15.60 (+3.34%)** banked win with **no margin contingency** — it survives the entire #410 haircut band. The net analytic move is to **collapse #425's 2-conjunct HOLD to a 1-conjunct HOLD**: the margin conjunct is resolved (banked); only the identity conjunct (stark #429) remains. The reconciliation also confirms 467.14 (not 467.48) as canonical and conservative.

### Suggested follow-ups

1. **Let stark #429 land**, then re-evaluate gate #2. If #429 reaches identity 1.0 on the strict path, this card flips to **GO-ready** (Bar A banked + identity green), pending the human-gated submission.
2. If #429 **promotes the fast path** (481.53 → feasible), open a sibling card pricing **fast + cb3 ≈ 497** as the new feasible-set max — the +15.60 supply lever is base-agnostic.
3. **kanna #416** remains worth measuring, but only to populate the *cosmetic* Bar B line and to nail the realized cb3 haircut; it is no longer a #407 gate.
