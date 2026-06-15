STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["xxzujn7a"],"primary_metric":{"name":"gap_floor_vbi1_self_test_passes","value":1},"test_metric":{"name":"irreducible_gap_floor_pct_vbi1","value":1.3097}}

## Results

**Verdict: RED / decision-critical.** The 0.633% irreducible floor does **NOT robustly survive** the transfer to the deployable-strict `VLLM_BATCH_INVARIANT=1` stack. The floor **inflates 2.07×** (0.633% → **1.310%** central). The central corner still clears 3.2% (+1.89pp), **but the pessimistic corner breaches it (3.523%, −0.32pp)** and the breakeven private prompt shift **roughly halves (+253 → +119 tok)**. #379's "uncapped at all corners with ≥1.5pp margin" is a **deployed-stack artifact**; fern #357 must re-derive the demand ceiling on the **VBI=1 floor (1.31%)**, not 0.633%.

This is the FLOOR analogue of what my #382 did for the SLOPE — and unlike the slope (which survived OOD), the **floor does not survive the stack transfer** at the robustness level.

### Headline numbers (CPU-analytic; W&B `xxzujn7a`)

| field | #379 deployed | **#386 VBI=1** | Δ |
|---|---|---|---|
| `irreducible_gap_floor_pct_vbi1` (central) | 0.6334% | **1.3097%** | **2.07× ↑** |
| `floor_inflation_ratio` | 1.00× | **2.068×** | — |
| floor — banked corner (ΔP=0) | 0.000% | 0.000% | clears |
| floor — central corner (ΔP=50) | 0.6334% | **1.3097%** | clears (+1.89pp) |
| floor — pessimistic corner (ΔP=130) | 1.6468% | **3.5235%** | **BREACHES 3.2% (−0.32pp)** |
| `clears_3p2_knife_edge_vbi1` (central) | True (+2.57pp) | **True (+1.89pp)** | margin shrinks |
| `all_corners_clear_3p2_vbi1` | True (≥1.5pp) | **False** | **robustness lost** |
| `breakeven_prompt_shift_tok_vbi1` | +252.6 (0.93× mean) | **+118.6 (0.44× mean)** | **shrinks 53%** |
| `b_cancels_under_vbi1` | True | **True** (dg_s/dB=−2.94e-6<0) | holds |
| `gap_bucket_acceptance_pct_vbi1` | 85.25% | **73.65%** | ctxlen share ↑ |
| `gap_bucket_ctxlen_pct_vbi1` | 14.75% | **26.35%** | nearly doubles |
| `gap_vbi1_total` (central) | 4.295% | **4.971%** | gap inflates |
| `floor_survives_vbi1_regime` | — | **False** | |
| `demand_route_uncapped_on_live_contract` | — | **False** | |
| `recommended_action` | — | **re-derive-ceiling-on-vbi1-floor (1.310%)** | |

`gap_floor_vbi1_self_test_passes` = **True** (26/26 checks), incl. a provenance check that the re-derived **deployed** central floor reproduces #379's 0.6334% exactly before the VBI=1 swap.

### Method (re-uses #379 harness structure)

The exact additive identity is unchanged: `gap = g_a + r_a·g_s`, floor = `r_a·g_s` (ctxlen bucket = gap after a perfect coverage retrain). What I changed deployed→VBI=1:

- **`r_a`, `g_a` are KERNEL-INVARIANT** (greedy identity preserved ⇒ same accepted tokens ⇒ same `E[T]_priv/E[T]_pub` on either kernel). I inherit them per-corner straight from #379's deployed back-out and assert-match (banked/central/pessimistic `r_a` reproduced to 1e-9).
- **`g_s` recomputed under VBI=1**: `A_vbi1(L) ∝ L·penalty(L)` (#375 un-pack penalty curve), so `shape(L) = (L/L_ref)·penalty(L)/penalty(L_ref)` and `g_s = f_attn·(shape−1)/(1+f_attn·(shape−1))` with **`f_attn = 0.0951`** (#378, replaces the deployed 0.0699). The penalty near L∈[528,658] uses #375's [528,2048] anchor segment — the **conservative** (gentler) forward slope, so the reported breach is if anything an under-estimate.
- **Total gap CHANGES** (Framing B, physically correct): `gap_vbi1 = 1 − r_a·r_s_vbi1`. The measured 4.295% is a deployed-stack quantity; on VBI=1 it inflates to 4.971% (central). The knife-edge test uses only the **absolute floor** `r_a·g_s_vbi1` and is invariant to this framing.

### B-cancellation (load-bearing assumption — VERIFIED, step 2)

`b_cancels_under_vbi1 = True`. `g_s = (A_priv−A_pub)/(B+A_priv)`: the lm_head-BI determinization tax inflates B but (i) **cancels in the numerator** `A_priv−A_pub` (pure attention/L shift, no B term — verified invariant under a B-tax sweep), and (ii) **dg_s/dB = −2.94e-6 < 0** — a larger B only **DILUTES** the floor; the lm_head-BI tax can never inflate it. The identity holds because the lm_head-BI GEMM shape (batch×hidden×vocab) is **fixed per decode step** — independent of KV length L and of which token is argmax'd ⇒ `B_pub == B_priv`. **The entire floor inflation is attention-side**: `f_attn` rose 0.0699→0.0951 (the measured #378 value already nets the larger B against the larger un-packed A — net f_attn up means attention inflation dominates) plus the un-pack penalty's super-linear ctx-growth steepening g_s.

### Comparison to baseline (PR body)

- #379 (`5kpb73tb`): floor 0.633%, every corner clears 3.2% with ≥1.5pp margin, breakeven +253 tok (~93%, "implausible"), `b_cancels` true — all on the **deployed** stack (attn ≈7%).
- #378 (`gghmgtk9`): `f_attn`=0.0951, eval-weighted penalty 1.2257, deployable-strict bracket [357.32, 469.68].
- #375 (`27sbg3zb`): penalty curve 1.264/3.027/4.756× @ 528/2048/4096, crosses 1.0× at L≈352.
- **This card**: re-derives the floor on the #378/#375 VBI=1 attention model. The 0.633% becomes 1.31% (central) / 3.52% (pessimistic); the all-corner robustness and the implausible breakeven both fail to transfer.

### Run details

- **Command:** `cd target/ && python research/validity/gap_floor_vbi1_regime/gap_floor_vbi1_regime.py --vbi1-attention-model --anchor-378-penalty --self-test --wandb_group strict-bi-verify-gemm --wandb_name ubel/gap-floor-vbi1-regime`
- **0-GPU re-derivation:** append `--reanalyze` (no wandb, identical numbers).
- **W&B run ID:** `xxzujn7a` (`wandb-applied-ai-team/gemma-challenge-senpai`, group `strict-bi-verify-gemm`).
- **Peak memory:** negligible (pure CPU-analytic, stdlib + numeric helpers; no GPU, no model load).
- **No submission, no HF Job, no `--launch`, no served-file change. BASELINE unchanged at 481.53; this card adds 0 TPS.**

### Public evidence used

Analysis-only re-derivation over banked internal fleet anchors (#379 `5kpb73tb` gap/buckets/per-corner `r_a`; #378 `gghmgtk9` `f_attn`=0.0951 + eval-weighted penalty; #375 `27sbg3zb` penalty curve; #282 decode-length dist median L=503). No challenge-board / leaderboard interaction (no public-state dependency for a CPU re-derivation; consistent with #379's framing).

### What happened — honest analysis

The hypothesis was decision-critical and the answer is **the floor does NOT cleanly survive**. Two independent factors push the floor up under VBI=1 and they compound:
1. `f_attn` rises 0.0699→0.0951 (1.36×) — attention is a bigger slice of the step.
2. The un-pack penalty's super-linear ctx-growth steepens the local `shape` sensitivity ~1.48× vs #379's L-linear heuristic.

Net ≈2.07× inflation. The **central** point estimate still clears (1.31% < 3.2%, +1.89pp) so the demand route is **not dead** — but #379's *robustness* (all corners ≥1.5pp, implausible +253-tok breakeven) is what fern #357 banks, and **that does not transfer**: the pessimistic "public high-decile" shift now breaches by 0.32pp, and the breakeven halves to a *plausible* +119 tok (0.44× public mean). So the "uncapped" pillar is a **deployed-stack artifact** on the robustness it was sold on.

**Honest caveat (the one thin spot):** the pessimistic breach is only 0.32pp over 3.2% and hinges on the **interpolated local penalty slope** near L∈[528,658]. I used #375's conservative forward slope, so the breach is robust to that choice — but the *exact* local slope is the single piece a direct GPU per-L measurement would PIN. I kept the run CPU-only per the PR's instruction (the model is nominally bounded by #378's measured f_attn + #375's curve), but given the breach is thin and decision-critical, I flag the GPU per-L leg as the decisive hardening step below.

### Suggested follow-ups

1. **(decisive) Run the optional GPU per-L attention leg** — `--gpu --proxy google/gemma-4-E4B-it-qat-w4a16-ct --measure-f-attn-vbi1` under `VLLM_BATCH_INVARIANT=1`, profiling per-L attention latency at L∈[528,658] (around the #282 median 503). This PINs the local penalty slope and converts the thin pessimistic breach (−0.32pp) from "slope-interpolated" to "measured." Identity-safe latency profiling only; no submission. I left the flag wired (currently SKIPPED-by-design) so it's a one-line enable.
2. **fern #357 hand-off:** re-derive the demand-route ceiling on the **VBI=1 floor (1.31% central)**, and treat the private prompt-length-shift sensitivity as a *binding* risk — the comfortable +253-tok safety buffer is gone (now +119 tok).
3. **Tighten the operating point:** the floor was anchored at L_ref=528 to match #379, but #378's eval-weighted penalty (1.2257) implies an effective operating L≈503 (the #282 median). Re-anchoring f_attn and the corners at L≈503 would slightly *lower* the pessimistic floor — worth a sensitivity pass to bound how much of the breach is the 528-vs-503 anchor choice.
