STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["zvkyeyqp"],"primary_metric":{"name":"two_path_portfolio_self_test_passes","value":1},"test_metric":{"name":"combined_reach500_prob","value":0.6290074}}

## Results

**Headline: the fleet has TWO independent routes to 500, and under stated priors `combined_reach500_prob = 0.629` — Path-A (land #245 tree build) alone is P_A = 0.28, so Path-B (the 5-lever linear portfolio, P_B = 0.485) adds a `second_shot_margin = 0.349` and DOES materially de-risk the all-in-on-land's-build posture. BUT both routes are PROJECTIONS, not facts (`reach500_A_is_measured = False`, `path_b_go_returned = False`): `readiness_verdict = NOT-READY` is carried forward from #238 UNCHANGED until the FIRST measured PPL-valid wall_tps ≥ 500 from EITHER path.**

CPU-only **bank-the-analysis** integration (PRIMARY = self-test). **0 TPS, BASELINE stays 481.53, authorizes nothing. NOT a launch. NOT open2.** Imports #238/#241/#244/#249 VERBATIM (loaded from committed result JSON, **19/19 headline scalars round-trip at 0.0 error**); the 5 in-flight Path-B screens are priced as a PORTFOLIO PRIOR over the speed-levers researcher doc's projected-gain ranges (NOT their unreturned measured results). The only new object is the COMBINED reach.

### The two-route decision table

| route | gate | current status | reach-prob | what collapses the prior to a fact |
|---|---|---|---|---|
| **PATH-A** (tree build) | measured E[T]_both ≥ **4.3305** ∧ measured λ̂ q[2..9] ≥ **0.9780113** | PRIOR (UNMEASURED): `treeverify_served_gain_MEASURED_realized = 0.0` | **P_A = 0.280** | land #245's ONE build run → a MEASURED end-to-end ≥500 tree-decode artifact |
| **PATH-B** (linear-lever portfolio) | stacked linear gain ≥ **3.8357%** (481.53→500 on `official = K_cal·(E[T]/step)·τ`) | PRIOR (NO GO returned): 5 in-flight screens | **P_B = 0.485** | each screen's GO/NO-GO + a kernel build clearing PPL-valid wall_tps ≥ 500 |
| **COMBINED** = 1−(1−P_A)(1−P_B) | FIRST measured PPL-valid wall_tps ≥ 500 from EITHER route | PRIOR: neither route measured → NOT-READY | **0.629** | land #245's build (A) OR any Path-B screen GO + kernel build ≥ 500 (B) |

### Step 1 — Path-A reach (the build route), P_A = 0.280

`P_A = p_build_lands · p_ET_clears · p_lambda_clears`, with land #245/#71's projection (E[T]_both = 4.512, min-λ = 0.983) as the **central** estimate and the external-fleet stall rate on the tree-decode build (the long pole) as the **adverse** anchor. The two gate-clears are measured INDEPENDENTLY (denken #241 operative framing: λ̂ from q[2..9], deep-tail-protected, does not move with the E[T]-shortfall δ), so they multiply.

| sub-prior | adverse | **central** | optimistic | basis |
|---|---|---|---|---|
| `p_build_lands` | 0.30 | **0.50** | 0.70 | a genuine coin-flip — every external fleet stalled on this exact build (#238 "the long pole") |
| `p_ET_clears` | 0.65 | **0.80** | 0.90 | the **4.02%** shortfall tolerance (denken #241 `delta_max_tps500`) gives the E[T] gate room below the 4.512 projection |
| `p_lambda_clears` | 0.55 | **0.70** | 0.85 | the **thin 0.005** λ margin (0.983 − 0.9780) makes this the tighter of the two gates |
| **P_A** | **0.107** | **0.280** | **0.535** | |

`reach500_A_is_measured = False` because `treeverify_served_gain_MEASURED_realized = 0.0` — there is NO live tree win yet; P_A is a prior, not a delivered probability.

### Step 2 — Path-B reach (the linear-lever portfolio), P_B = 0.485 [prior sweep 0.186 … 0.732]

`P_B = P(stacked linear gain ≥ 3.8357%)`, MC over GO×Uniform projected-gain priors (N = 400k, seed 20260614, MC se = 0.0008, stability spread = 0.0003). **Revealed-difficulty prior:** the served stack has PLATEAUED at 481.53 after extensive tuning — if any single lever were an easy deployable greedy-identical ≥+3.84% win it would already be banked, so every `p_go` sits well below the literature's nominal success rate, and a "GO" is the *conjunction* {deployable on this A10G single-stream stack ∧ greedy-IDENTICAL ∧ PPL-valid ∧ clears the screen's stop condition}.

| screen | lever (doc rank) | step component | p_go | projected gain | clears ALONE if it fires? |
|---|---|---|---|---|---|
| stark **#247** | OPT-Tree adaptive topology (T-1) | E[T] | 0.28 | +0.5–3.5% | **no** (even at hi) — capped by #244 topology-exhausted |
| lawine **#246** | FlashInfer + CUDAGraph (K-1) | SYS | 0.40 | +2–8% | yes at hi, no at lo |
| kanna **#248** | int3 draft / QSpec (Q-1) | DRAFT | 0.30 | +1–6% | yes at hi, no at lo |
| ubel **#250** | n-gram / REST draft (N-1) | DRAFT | 0.20 | +0–5% | yes at hi, no at lo |
| wirbel **#251** | activation-recycle (HBM-capped) | SYS | 0.25 | +1–5% | yes at hi, no at lo |

**Stacking / overlap assumption (stated):** levers are grouped by the step component they target. WITHIN a group they do NOT stack (you cannot remove the same step time twice): SYS `overlap_retain = 0.25` (FlashInfer + activation-recycle are complementary but mostly redundant on the GPU-bound step); DRAFT `overlap_retain = 0.0` (int3-draft and n-gram are **mutually exclusive draft sources** — deploy one, not both). ACROSS groups the E[T]-axis composes **multiplicatively** with the step-axis `(1+a_ET)·(1+b_step)−1`, and the two step-lowering groups carry an Amdahl efficiency `η_step = 0.85`.

**Why P_B ≈ a coin-flip:** if all 5 levers fire at central gain → **+10.0%** (clears with margin). But the GO-weighted *expected* stacked gain is **+3.33%** — it lands *just below* the 3.836% bar. So reaching 500 needs a slightly-better-than-expected draw (a couple of levers firing toward their high ends), which is ~48% of the MC mass. The binding Path-B uncertainty is the GO probabilities, not the gain magnitude.

### Step 3 — the combined reach

`combined_reach500_prob = 1 − (1 − 0.280)(1 − 0.485) = 0.629` (prior range **[0.274, 0.876]** as P_A and P_B sweep jointly). `second_shot_margin = combined − P_A = 0.349 ≥ 0.10` → **Path-B materially de-risks** the all-in-on-land's-build posture (it more than doubles the reach probability over the tree build alone).

### Self-test (PRIMARY) — `two_path_portfolio_self_test_passes = True`

- **(a)** `combined` round-trips `1−(1−P_A)(1−P_B)` and obeys the probability bounds `max(P_A,P_B)=0.485 ≤ 0.629 ≤ 0.765=P_A+P_B`, all in [0,1] ✓
- **(b)** P_A and P_B EACH carry an explicit stated prior + the measurement that collapses it ✓
- **(c)** `reach500_A_is_measured=False` ∧ `path_b_go_returned=False` ⇒ `readiness_verdict` stays **NOT-READY** (the card cannot read a prior as a delivered win — same discipline as #238's `treeverify_realized==0` gate) ✓
- **(d)** the +3.8357% Path-B threshold round-trips 481.53→500: `481.53·(1+thr)=500.0` exactly, and the composition is LINEAR — a g-fractional gain in (E[T]/step) is exactly g in official TPS (slope-invariant) ✓
- **(e)** the Path-A gates match the imported #249 (λ = 0.9780112973731208) and #241 (E[T] = 4.330527243789328) **EXACTLY**; 19/19 provenance scalars round-trip at 0.0 error ✓
- **(f)** NaN-clean ✓

### Comparison vs the PR baseline anchors

| anchor | PR body | this card |
|---|---|---|
| Path-A E[T] floor (operative / self-insured) | 4.3305 / 4.4890 | imported #241 verbatim, round-trip 0.0 |
| Path-A λ̂ build bar | 0.9780 (op) / 0.9808 (def) | imported #249 operative 0.9780112973731208, round-trip 0.0 |
| land projection / measured | E[T]_both 4.512, min-λ 0.983 / treeverify_realized 0.0 | imported #238 verbatim, round-trip 0.0 |
| Path-B threshold (481.53→500) | +3.835% | computed 3.8357% (= 500/481.53 − 1) via the linear composition |
| readiness | NOT-READY (#238) | carried forward UNCHANGED |

### Reproduce

```
cd target/ && CUDA_VISIBLE_DEVICES="" python \
  research/validity/two_path_500_portfolio/two_path_500_portfolio.py \
  --self-test --wandb_group launch-readiness-integration \
  --wandb_name fern/two-path-500-portfolio
```

- **W&B run:** `zvkyeyqp` (wandb-applied-ai-team/gemma-challenge-senpai, group `launch-readiness-integration`)
- **Peak memory:** 87.2 MiB (CPU-only; numpy 400k-trial MC), **elapsed 0.60 s**
- **summary.json fields:** N/A — no benchmark/draw this leg (0 TPS; `tps`/`ppl`/`completed`/`run_prefix` unchanged from the served baseline 481.53 TPS, PPL 2.3772, 128/128, PR #52).

### What happened — honest analysis

It worked, and the integration sharpens the launch posture rather than softening it. The decisive structural finding: **the two routes are genuinely independent** (land's tree build raises E[T] via a new served topology; the Path-B levers move the *existing* 481.53 linear stack), so pricing them as `1−(1−P_A)(1−P_B)` is the right composition, and Path-B's `second_shot_margin = 0.349` is real — the team is NOT all-eggs-in-one-build.

The honest core is that **both numbers are PRIORS, and the card refuses to read either as a fact.** P_A = 0.28 is the tree build conditioned on land's projection with the external-fleet stall as the adverse anchor; P_B = 0.485 is a near-coin-flip because the GO-weighted expected linear stack (+3.33%) lands just *below* the +3.836% bar. Two cross-checks keep me honest: (i) my own banked **#244** certifies the verify-tree topology is exhausted (`topology_lift_max = 0`), which is exactly why I priored stark #247 (OPT-Tree, a topology lever) DOWN to `p_go = 0.28` and capped its gain below the threshold — it is the *only* lever that cannot clear alone even at its high end; (ii) the **revealed-difficulty** argument — the stack's plateau at 481.53 is Bayesian evidence against easy linear gains, so all `p_go` sit below literature success rates. Under the pessimistic prior P_B drops to 0.186 (combined → 0.274); under optimistic it rises to 0.732 (combined → 0.876). The prior dependence is the dominant uncertainty, and it is reported, not hidden.

What does NOT move: `readiness_verdict = NOT-READY`. Neither route has a measured PPL-valid wall_tps ≥ 500, so the launch stays single-blocked on a MEASURED win from EITHER path. The card structures the decision and prices the combined reach — it does not claim 500.

### Hand-off (one sentence)

*The team has two independent routes to 500 — Path-A (land #245 tree build, gated E[T]_both ≥ 4.3305 ∧ λ̂ ≥ 0.9780) and Path-B (the 5-lever linear portfolio, reach iff stacked gain ≥ 3.8357%) — with `combined_reach500_prob = 0.629` under stated priors; both are currently PROJECTIONS (no measured ≥500), so readiness stays NOT-READY until the first measured PPL-valid wall_tps ≥ 500 from EITHER path, but Path-B DOES materially de-risk the all-in-on-land's-build posture (second-shot margin 0.349).* (Consumers: human team #124 publish-first + land #245.)

### Public evidence used

- **fern #238** (`launch_decision_card`, `xioud4hv`): readiness NOT-READY, n_green_gates=2, `treeverify_served_gain_MEASURED_realized=0.0`, land projection E[T]_both 4.512 / min-λ 0.983 — round-tripped.
- **denken #241** (`measured_et_shortfall`, `hqewf1d6`): E[T] floor 4.3305 (op) / 4.4890 (self-insured), `delta_max_tps500=0.04022`, λ margin 0.00502, the LINEAR composition K_cal=125.268 / step=1.2182, λ=1 ceiling 520.95 — round-tripped.
- **fern #249** (`build_lambda_bar`, `on4u78ul`): operative build-λ gate 0.9780112973731208 — round-tripped.
- **wirbel #244** (`ceiling_gap_topology_headroom`, `sgjvbzu3`): compliant-PRIVATE-500 lane TOPOLOGY-DEAD (`topology_lift_max=0`, reopener=coverage/λ→1) — the banked evidence that priors stark #247 (OPT-Tree) DOWN.
- **speed-levers researcher doc** (`research/RESEARCH_IDEAS_2026-06-14_speed-levers.md`): the Ranked Priority Table supplying each Path-B lever's projected-gain range and greedy-ID risk (T-1 OPT-Tree, K-1 FlashInfer+CUDAGraph, Q-1 int3-draft, N-1 n-gram, activation-recycle).

### Suggested follow-ups

- **First measured ≥500 from EITHER path collapses the headline to a fact:** wire `combined_reach500_prob` to flip — set the corresponding `reach500_*_is_measured`/`path_b_go_returned` True and re-read readiness. The card is built so a single GO from land #245 OR any of the 5 screens is the trigger.
- **Tighten the Path-B prior as screens return:** each GO/NO-GO verdict replaces a `p_go` prior with a measured 0/1 and collapses one factor of the MC — the highest-information screen is lawine #246 (FlashInfer, the only lever that can clear alone at its central-to-high range and the strongest single contributor to P_B).
- **If land #245's build lands but misses a gate,** denken #241's shortfall table adjudicates directly (E[T] within 4.02% of 4.512 → TPS500 clears; λ̂ ≥ 0.9780 independent) — no re-derivation needed.
- **Do NOT spend node budget widening the verify tree** (wirbel #244): the only Path-A topology headroom is per-step adaptation (OPT-Tree), already priored low here.
