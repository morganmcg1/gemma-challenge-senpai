STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["ec7i3z5t"],"no_hf_job":true,"official_tps":0.0,"analysis_only":true,"no_served_file_change":true,"primary_metric":{"name":"mtp_acceptance_headroom_self_test_passes","value":1},"test_metric":{"name":"frac_of_28p58_gap_covered","value":0.0}}

## Results

**Verdict: NEGATIVE — there is NO cheap deployable acceptance-headroom lever. For the deployed LINEAR MTP K=7 chain (M=8, top-1 per position), every no-retrain/no-served-kernel lever is a rank-order no-op or a forbidden kernel rebuild. `realized_dcov_best_lever = 0`, `realized_tps_lift_best_lever = +0.00 TPS`, `frac_of_28p58_gap_covered = 0%`, `dcov_lever_feeds_demand_route = False`. The demand route's d-cov is real and FITS the #336 budget, but must be SUPPLIED by a drafter retrain or a tree verify — not by a draft-head tweak.**

Pure-CPU analytic card + seeded numpy Monte-Carlo (0 GPU, 0 official TPS, 0 HF Job, no served-file change, no drafter load). Baseline **481.53 TPS / PPL 2.3772** and corrected strict base **471.42 (#390)** UNCHANGED. W&B `ec7i3z5t` (group `mtp-drafter-acceptance-headroom`). Self-test **39/39** (≥20 required). New file: `research/validity/mtp_drafter_acceptance_headroom/`.

### Step 1 — realized acceptance, reused from #289/#387 (deliverables 1–3)

The deployed scheme is **linear MTP K=7, M=8 verify** = 1 bonus + 7 **top-1** draft positions (PR #52 / #387). The realized per-position conditional ladder (#289 `fi34s269`):

`a_1..a_7 = [0.7293, 0.7596, 0.7930, 0.8228, 0.8349, 0.8358, 0.8465]` → **E[accepted] = 2.851, E[T] = 3.851**.

Coverage anchors (#387 `z8osvif8`, on-distribution for the official 128): **top-1 = 0.7617**, **top-4 = 0.8903** (round-trips from per-source × official mix to 1e-9). The head-ceiling gap `coverage_ceiling_gap = 1 − 0.8903 = 0.1097` is an **UPPER bound** — a direct top-8/top-16 read is blocked (#387 `direct_gpu_topk_read_blocked_on`: deployed=MTP identity + missing checkpoint), so the true ceiling ∈ [0.890, 1.0] and the gap ∈ [0, 0.1097].

### Step 2 — the deployable-lever sweep (deliverables 4–8): all cheap levers are no-ops

I built a seeded numpy ensemble calibrated to reproduce the banked anchors (MC top-1 = 0.7618, top-4 = 0.8898 ✓) and **measured** the realized Δ(coverage) of each lever:

| lever | deployable? | realized Δ(top-4 cov) | why |
|---|---|---|---|
| **draft-head temperature** `z→z/T` | ✅ | **0.0000** (MC exact) | monotone → preserves argmax AND the top-K **set** → coverage invariant |
| **affine calibration** `z→a·z+b` (a>0) | ✅ | **0.0000** (MC exact) | monotone → rank-membership statistic cannot move |
| **top-K width (tree verify)** | ❌ kernel change | 0.0000 (locked **+0.1286**) | verify >1 cand/pos → verify-M change → CUDA-graph rebuild (#390 counts rebuilds) — EXCLUDED by *no-served-kernel-change*; the +0.1286 is the tree/EAGLE-3 prize |
| **per-class logit bias** | ❌ retrain | 0.0000 | rank-CHANGING (control fires) but a *fitted* bias = (micro)retrain, overfits public 128, private-unstable — EXCLUDED |

MC: monotone levers (8 temperatures × {0.25…10} + 9 affine cells) give **max\|Δcov\| = 0.00e+00** (machine-exact). The per-class control (random N(0,2) bias) gives \|Δtop-4\| = 0.5929 — **the detector fires**, proving the 0.0s are a real invariance, not a broken harness. The whole monotone-lever family is a no-op because **a monotonic logit transform preserves rank order**, and both top-1 acceptance and top-4 coverage are rank-membership statistics.

⇒ `best_deployable_dcov_lever = "none_deployable"`, `realized_dcov_best_lever = 0`, `realized_tps_lift_best_lever = +0.00`, `frac_of_28p58_gap_covered = 0%`, `dcov_lever_feeds_demand_route = False`.

### Step 3 — map to TPS on the 471.42 base, and where the d-cov must come FROM

Using the program demand secant `S_central = 7.9126` (E[T]/cov, #387/#383) and `TPS = (1+E[accepted])/T_step` on the 471.42 base (T_step fixed — no kernel change): **968.57 TPS per unit Δcov**.

- d-cov needed to close the **28.58** gap = **+0.0295** — *within* the #336 budget (+0.031), **95%** of it.
- the full #336 budget (+0.031) → **+30.06 TPS** (closes the gap, `full_336_budget_closes_gap = True`).
- the best **deployable** lever supplies **0** of that 0.0295.

So denken #396's question — does the required d-cov *fit the budget*? — is **YES** (+0.0295 ≤ +0.031). But the complementary question this card was meant to answer — is there a **concrete deployable lever** that *supplies* it? — is **NO**. The d-cov must be manufactured by a **drafter retrain** (raise the a_j ladder itself) or a **tree verify** (harvest the locked top-1→top-4 +0.1286 coverage). Both are priced by other lanes (coverage-retrain / EAGLE-3 / tree-488); neither is a free draft-head tweak.

### PPL (the gate to keep in mind)

Every draft-side lever (temperature, calibration, even tree width) preserves greedy identity — speculative decode emits the **target model's** greedy token exactly — so **PPL is unchanged at 2.3772 ≤ 2.42 for all of them**. The binding constraint for this lever family is **deployability (kernel rebuild) + private transfer**, never PPL. A "coverage lever that degrades draft quality" cannot exist here, because draft quality changes *speed*, not *output*.

### Deliverables (W&B `summary/`)

`deployed_mtp_acceptance_ladder_measured` = #289 ladder; `top4_coverage_measured = 0.8903`; `coverage_ceiling_gap = 0.1097`; `best_deployable_dcov_lever = none_deployable`; `realized_dcov_best_lever = 0`; `realized_tps_lift_best_lever = 0`; `frac_of_28p58_gap_covered = 0`; `dcov_lever_feeds_demand_route = False`; `mtp_acceptance_headroom_self_test_passes = True` (**PRIMARY**, 39/39). Flags `analysis_only / no_hf_job / no_served_file_change = True`, `official_tps = 0`.

### Command

```
cd target/ && .venv/bin/python -m research.validity.mtp_drafter_acceptance_headroom.mtp_drafter_acceptance_headroom --self-test
cd target/ && .venv/bin/python -m research.validity.mtp_drafter_acceptance_headroom.mtp_drafter_acceptance_headroom \
  --wandb_group mtp-drafter-acceptance-headroom --wandb_name ubel/mtp-drafter-acceptance-headroom
```

Peak memory: ~0.2 GB (numpy MC over a 20000×64 float64 ensemble; no torch/GPU). W&B run `ec7i3z5t`.

### What happened

The hypothesis proposed that a draft-head temperature / top-K-width / calibration adjustment could raise top-4 coverage 0.890→X without a retrain or kernel change. It **cannot**, and the reason is structural: the deployed drafter proposes by **argmax** and the verify keeps the longest **top-1**-matching prefix, so coverage is a rank-membership statistic — and the cheap levers (temperature, affine calibration) are **monotonic**, which provably leaves rank order (hence the entire top-K set) untouched. The Monte-Carlo measures this as an exact zero across the whole sweep, while a rank-changing control (per-class bias) moves it by 0.59 — so the zero is the physics, not a bug. The only levers that *can* move coverage are exactly the ones the constraint set forbids: a **tree verify** (top-K width > 1 = a served-kernel rebuild) to harvest the locked +0.1286 top-1→top-4 headroom, or a **drafter retrain** (move the a_j ladder / head distribution).

The decision-critical payload for denken #396: the +0.0295 d-cov that closes the 28.58 gap **does fit** the +0.031 budget, but there is **no deployable lever that supplies it** — the budget is fundable only from retrain/tree work, so the demand route should be costed against those lanes, not against a phantom "free" draft-head knob.

### Suggested follow-ups

- **Measure the locked top-8/top-16 ceiling.** `coverage_ceiling_gap` is currently the [0, 0.1097] upper bound because #387's direct GPU top-K read is blocked (deployed=MTP identity + missing checkpoint). Loading the deployed MTP drafter and reading per-position top-8 coverage on the official 128 (a local-profiling GPU card, *measuring* not deploying a tree) would pin how much of the +0.1286 a depth-1 tree could actually realize — i.e., size the tree prize precisely. This is the natural GPU companion to this analytic card.
- **Cost the demand d-cov against the retrain lane, not a lever.** Since +0.0295 must come from a retrain, the operative question becomes the coverage-retrain deliverability (does a no-retrain-budget run lift the a_j ladder / head coverage by +0.0295 with private transfer?), which the coverage-retrain / EAGLE-3 cards already price. This card closes the "cheap lever" branch so the demand route is evaluated only on the real supply.
- **Tree-width vs verify-cost trade.** The locked +0.1286 is gross coverage; a depth-1 top-K tree pays a verify-M cost that this card holds out of scope. Composing the tree's Δ(E[accepted]) against its step-time tax (the #390 kernel-rebuild + wider-M roofline) would convert the locked coverage into a *net* TPS number — the actual tree go/no-go.

### Public evidence used

0-submission internal analytic card; **no public leaderboard method reproduced**. Checked the shared digest (`?as=senpai`): the public a10g frontier is **~489.6 TPS** (`frantic-penguin` `osoi5-…-skv64`, **valid**, PPL 2.3774; `hayai-agent`/`openevolve` ~489 pending) — all **split-KV MTP K=7** stacks sitting just under the **500** bar this card's demand route targets, confirming the 28.58 gap is the live frontier question. Grounded in the internal frontier (PR #52, 481.53 / `2x9fm2zx`), the #289 ladder (`fi34s269`), #387 coverage (`z8osvif8`), and the #390 corrected base (`5y64zbjz`), per the PR baseline table.
