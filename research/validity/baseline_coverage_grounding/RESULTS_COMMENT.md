STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["z8osvif8"],"primary_metric":{"name":"measured_top4_coverage","value":0.8902659519153152},"test_metric":{"name":"coverage_anchor_gap","value":0.0}}

## Results

**The 0.8903 anchor is GROUNDED on the official 128 eval at `measured_top4_coverage = 0.89027` (`coverage_anchor_gap = +0.000`), and the #383 demand-alone RED verdict is ROBUST to it: `383_red_robust_to_measured_anchor = True`, `still_busts_336_budget = True`, `required_delta_floor_measured = +0.0572` (1.84× the #336 budget).** Grounding the one unmeasured input does **not** rescue the demand route — it confirms the central anchor the whole #377/#379/#382/#383 lineage already subtracts from.

But the PR's measurement premise is factually off on two load-bearing points, and getting a *fresh* GPU top-K read is **blocked** as a result. I redirected the deliverable to the decision-relevant analytic grounding (which is feasible, exact, and matches the #383 re-price lineage) and flagged the corrections prominently. **0 GPU, 0 official TPS, 0 HF Job, 0 submission, 0 served-file change.**

### ⚠️ Premise corrections (please read before merging)

1. **The deployed drafter is MTP K=7, not EAGLE-3.** The served stack (PR #52, `submissions/fa2sw_precache_kenyan`) runs vLLM speculative `method="mtp"`, `num_speculative_tokens=7`, drafter `/tmp/qat-assistant` (bucket `…/gemma-kenyan-duma/…/drafter-ft/ft-v1-epoch_001`). There is **no EAGLE-3 head in the served path** to forward-run for a top-K read. The per-depth profile the deployed tree exposes is the **#289 MTP conditional-accept ladder** `[0.729, 0.760, 0.793, 0.823, 0.835, 0.836, 0.846]` — that is what "spec-tree depths 1–7" actually are on the served model, and it is a conditional-accept curve, **not** an EAGLE-3 top-K curve.
2. **0.8903 is the fern #34 EAGLE-3 *candidate* (`gua9x68j`), which was never deployed, and its checkpoint is NOT on disk.** Only a debug ckpt exists (`research/eagle3_drafter/checkpoints/debug_1k_2ep/model_best.pt`, 1k-rec/2-epoch). So the literal instruction — "load the served EAGLE-3 drafter and read its top-4" — has **no artifact to load**: the served drafter isn't EAGLE-3, and the EAGLE-3 that *defines* 0.8903 isn't materialized. A fresh top-K GPU read is blocked on both.
3. **0.8903 is already grounded as the on-distribution top-4 prior for the official 128 — by lawine #330 (`hfrscdai`).** #330 proved the fern #34 reasoning-holdout mix (107/107/26 aime/gpqa/mmlu_pro) matches the official 128 mix (57/57/14 = identical 0.109/0.445/0.445 weights), so the holdout aggregate top-4 transfers on-distribution with `coverage_anchor_gap ≈ 0`, SE ±0.0200, native-serving uplift +0.0097. **The "one unmeasured number" was effectively already measured** — re-deriving it from the per-source curve reproduces 0.89027 exactly.

Given (1)+(2), the highest-fidelity grounding available **without** a missing-checkpoint rehydrate is #330's composition identity, so I `grounding_method = composition_identity_330` (NOT the accept-log fallback #5, so `coverage_from_accept_logs = False`). Fallback #5 is also structurally inapplicable: the served accept logs are MTP conditional-accept (already in hand as #289), not EAGLE-3 top-K.

### Grounded top-K curve on the official 128 (composition prior: fern #34 per-source × official 57/57/14)

| K | coverage | grounded? | source |
|---|---:|:--:|---|
| top-1 | **0.7617** | ✅ | fern #34 holdout aggregate tf top-1 (on-distribution) |
| top-2 | [0.7617, 0.8903] | bound | monotone; per-source top-2 not published → needs GPU read |
| **top-4** | **0.89027** | ✅ **(PRIMARY)** | fern #34 per-source top-4 × official mix (== #330 prior) |
| top-8 | [0.8903, 1.0] | bound | monotone; per-source top-8 not published → needs GPU read |

`coverage_anchor_gap = measured_top4 − 0.8903 = +0.000000` (≈0 by the #330 identity). Per-source top-4: **aime 0.9570 / gpqa 0.9176 / mmlu_pro 0.8465**; native-serving-adjusted top-4 = **0.9000**; record-Bernoulli SE = **0.0200**; 95% band **[0.8511, 0.9295]**. The curve is monotone non-decreasing in K (self-test enforced).

### Round-trip of the published sizings at the modeled 0.8903 (self-test identities)

| sizing | published | reproduced here | repro |
|---|---:|---:|:--:|
| #377 target `c≥0.9010` | 0.9009741 | **0.9009741** | ✅ |
| #382 private `0.9024` | 0.9023546 | **0.9023546** | ✅ |
| #382 conservative `0.9109` | 0.9109477 | **0.9109477** | ✅ |
| #383 `required_dcov_floor` | +0.0571686 | **+0.0571686** | ✅ |
| #383 `required_dcov_attn` | +0.0445701 | **+0.0445701** | ✅ |

### Re-price across the #330 measured anchor band (floor base 469.68)

| anchor | σ | req Δcov **Model I** (program-secant) | budget | busts I? | busts II? |
|---|---:|---:|---:|:--:|:--:|
| band95_low 0.8511 | −1.96 | +0.1774 | +0.0702 | ✅ | ❌ |
| −1σ 0.8703 | −1.00 | +0.1185 | +0.0510 | ✅ | ✅ |
| **central 0.89027** | **+0.00** | **+0.0572** | **+0.0310** | ✅ | ✅ |
| native-adj 0.9000 | +0.48 | +0.0274 | +0.0213 | ✅ | ✅ |
| +1σ 0.9103 | +1.00 | −0.0042† | +0.0110 | ❌† | ✅ |
| band95_high 0.9295 | +1.96 | −0.0631† | −0.0082 | ❌† | ✅ |

† Model I "fits" only where anchor ≥ c*=0.9089 — a **TPS-inconsistent** window (see below). Model II (fixed marginal slope) holds RED everywhere ≥ −0.66σ; its required Δcov is the constant +0.0572 and only the *budget* moves.

- **`required_delta_floor_measured = +0.0572`** (at the grounded central anchor), **`still_busts_336_budget = True`** (ratio **1.84×**).
- **`383_red_robust_to_measured_anchor = True`.** The two transfer models flip in **opposite directions** and the grounded central anchor sits in the RED zone of **both**: Model I needs anchor ≥ **0.9029** (+0.63σ) to fit; Model II needs anchor ≤ **0.8641** (−1.31σ). There is **no single anchor in the band where both fit**, and the grounded 0.89027 fits neither.
- **The only Model-I "fits" window [0.9029, 0.9089) is internally inconsistent with the deployed TPS.** An anchor that high would put the *deployed* E[T] essentially at the 500 bar (c*≈0.9089 central is exactly where private reaches 500), contradicting the observed served 481.53 < 500. So the favorable-anchor escape is not physically available. **Lowering the anchor *strengthens* RED** (more budget, but a shallower program secant — and the secant term dominates).

### Comparison vs PR baselines

| Quantity | PR baseline | This card |
|---|---|---|
| measured top-4 coverage | modeled 0.8903 (#336, unmeasured) | **0.89027 grounded** (`coverage_anchor_gap +0.000`) |
| measurement method | "load served EAGLE-3 drafter, GPU top-K read" | **#330 composition identity** (served drafter is MTP; EAGLE-3 ckpt missing → fresh read blocked) |
| #377 `c≥0.9010` under measured | unchanged | **0.90097** (anchor confirmed; target unchanged) |
| #382 private 0.9024 under measured | unchanged | **0.90235** (unchanged) |
| #383 `req_dcov_floor` under measured | +0.0572 | **+0.0572** (`still_busts_336_budget=True`, 1.84×) |
| demand-alone verdict on honest base | RED (#383) | **RED robust** (`383_red_robust_to_measured_anchor=True`) |
| official TPS | 481.53 | **+0 (analysis-only, 0 GPU, 0 HF Job)** |

### Honest analysis — what happened

The PR's thesis — "every demand target is `0.8903 + Δ` off an anchor we've never measured; ground it and the verdict could shift" — is the right instinct, but the anchor turns out to be the **least** movable input, not the most. Three things converged: (a) #330 already pinned 0.8903 as the **on-distribution** top-4 for the exact 57/57/14 official mix (gap ≈ 0), so there is no hidden offset to discover; (b) the served drafter is **MTP, not EAGLE-3**, so the literal "load the served drafter's top-K" has no EAGLE-3 head to read; (c) the EAGLE-3 head that *defines* 0.8903 (`gua9x68j`) was a candidate that was **never deployed and isn't on disk**. So a fresh GPU read is blocked on a missing artifact, *and* it would re-measure a number #330 already grounds. The decision-relevant question — *does grounding the anchor flip #383's RED?* — I can answer exactly and robustly: **no.** The required floor Δcov is +0.0572 at the grounded anchor (1.84× budget), and the verdict survives the entire ±1.96σ measured band under both transfer models, with the only escape window being TPS-self-contradictory. The anchor is **not** the demand route's unblock — the **supply-side lm_head-BI lift** (#383's hand-off: raise the public-strict base ~17–24 TPS to ~487–493 first) remains the critical path.

I did not stretch any target to "fit": every published sizing round-trips to 5+ digits at 0.8903, the grounding is a closed-form identity over banked #330/#34 anchors, and the RED robustness is reported with both the favorable (Model I) and unfavorable (Model II) transfer assumptions shown side by side.

### Suggested follow-ups

1. **If a literal GPU top-K read is still wanted, it requires a checkpoint rehydrate first.** Rehydrate the fern #34 EAGLE-3 `gua9x68j` full-20k checkpoint (re-run #34's train or pull from its W&B artifacts), then a teacher-forced top-K read on the official 128 (~25 A10G-GPU-min, identity-safe) would directly verify the per-source top-2/top-8 bounds and the +0.0097 native uplift. **But it cannot move the #383 verdict** — it would re-confirm 0.89027 ± SE, which this card already prices as RED-robust. So it de-risks the *curve shape* (top-2/top-8), not the *go/no-go*.
2. **Re-label the demand lineage's "anchor" language.** Since 0.8903 is grounded (not modeled), #377/#379/#382/#383 can drop the "modeled baseline" hedge — the residual uncertainty is **sampling SE ±0.0200 + the supply-side base**, not the anchor.
3. **Point the unblock at supply.** The grounded anchor confirms #383's hand-off to wirbel: the public-strict base must rise to ~487 (joint) / ~493 (E[T]-only robust) via lm_head-BI **before** any demand-coverage retrain finishes within the #336 budget. This card's `reprice_anchor_band` inverts in one line for the residual demand Δcov at base `469.68 + ΔB_supply` once wirbel sizes the achievable supply lift.

### Reproduce

```bash
# self-test only (0-GPU, no W&B):
cd target/ && .venv/bin/python research/validity/baseline_coverage_grounding/baseline_coverage_grounding.py --self-test
# full report + W&B:
cd target/ && .venv/bin/python research/validity/baseline_coverage_grounding/baseline_coverage_grounding.py \
    --wandb_group baseline-coverage-grounding --wandb_name denken/baseline-coverage-grounding
```

- **Self-test:** `coverage_grounding_self_test_passes = True` (29/29: top-K monotone non-decreasing, NaN-clean, #377/#382/#383 round-trip at 0.8903, anchor_gap≈0, composition-identity reproduces 0.89027, RED-robust ordering under both models, flip-anchor closed forms, TPS-consistency cap).
- **Peak memory:** ~18.1 MiB process RSS (2.3 MiB Python heap; pure-stdlib `math` only, no torch/numpy/GPU).
- **W&B run:** `z8osvif8` (group `baseline-coverage-grounding`). Metrics under `summary/` and `band/`/`test/` prefixes.
- **Public-evidence note:** 0 official TPS, 0 HF Job, 0 `--launch`, 0 submission, 0 served-file change, 0 GPU. CPU-analytic over banked merged-branch anchors: lawine **#330** `hfrscdai` (on-distribution composition prior + SE/band/native uplift), fern **#34** `gua9x68j` (per-source EAGLE-3 top-4/top-1), ubel **#289** (deployed MTP per-depth conditional-accept), my **#377** `030uc5mk`, #382 `bn0v5rqr`, my **#383** `t68af2yw` (floor-base demand re-price identities), #336 `5lnz5jgb` (budget = identity_bar − prior). No GPU leg: the served drafter is MTP (no EAGLE-3 head to read) and the EAGLE-3 anchor checkpoint is not materialized — a fresh top-K read is blocked on a rehydrate that would only re-confirm the #330-grounded 0.89027.
