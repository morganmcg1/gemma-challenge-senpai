# SENPAI Research Results

## 2026-06-15 07:21 — PR #310: Does E[T]=6.11 clear PRIVATE 500 under the per-position model? — 🟢 the ×0.804 "402 NO-GO" (#305) was a HIDDEN DOUBLE-COUNT: the 4.966 "public floor" already carries one 0.804 haircut → it is the scalar-PRIVATE-500 floor, not public-500 → #305 applied 0.804 twice. On ONE honest public base (622.08 @ E[T]=6.11) the per-position model clears PRIVATE 500 at 586.08 TPS (+17.2%); tree-recovered 0.955→594 CLEARS; raw width-1 0.7797→485 misses; break-even ρ_priv=0.8038 — bank-the-analysis (0 TPS)

- **Branch:** `fern/eagle3-private-perposition-reconcile` · **Student:** fern · merged 07:21:50Z by advisor (8gpu launch; CPU-only reconciliation of #305's conservative scalar vs #300's per-position model — NO training, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`2u3kcnv5`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/2u3kcnv5) (group `eagle3-private-perposition`, finished, NaN-clean, self-test 8/8). Independently W&B-verified by advisor (ALL MATCH; ρ_priv_e3 correctly carried from lawine #300). Merge commit `023f8fd`.
- **Primary:** `private_perposition_reconcile_self_test_passes=1` (8/8). **Test:** `private_tps_at_611_perposition=586.08`.
- **Key finding:** #305's ×0.804 PRIVATE NO-GO (402 central) was a hidden DOUBLE-COUNT — the 4.966 "public floor" already embeds one 0.804 private haircut (it is the scalar-PRIVATE-500 floor, not the public-500 floor), so #305 applied the 0.804 scalar twice. Re-grounding on one honest public base (622.08 @ E[T]=6.11): the per-position private model (lawine #300's ρ_priv_e3=0.9421, c₁-held + c_deep=0.97135) projects PRIVATE **586.08 TPS (+17.2%)** → CLEARS 500. Tree-recovered ratio 0.955 → 594 (clears); raw width-1 ratio 0.7797 → 485 (misses); break-even at **ρ_priv=0.8038**. PRIVATE-500 is GO under the credible per-position convention, NO-GO only under the doubly-conservative scalar.
- **Conclusion — build-economics matrix axis (g) PRIVATE-BAR reconciled at CLEARS under the per-position model; resolves the #305 YELLOW.** The decisive correction to the GO-card's binding axis: the sub-500 private projection was an artifact of double-applying the 0.804 scalar. Remaining caveat: ρ_priv_e3=0.9421 is MODELED from the LINEAR spine, not measured on the {2,21,39} fusion head — the fusion's true deep-private tax could erode the headroom toward 0.8038. Bounding that worst case is reseated to **fern #318** (deep-private-tax lower bound). Pairs with fern #305 (GO-card conservative bookend), lawine #300 (ρ_priv_e3), denken #308 (a1-trainability).

## 2026-06-15 07:21 — PR #308: EAGLE-3 a1-cliff — trainable target or intrinsic floor? — 🟢🟡 against the #309-relaxed 0.7731 bar (M=8 tree-salvage) the IN-REPO {2,21,39} EAGLE-3 head (fern #34 gua9x68j) already measures native step-1 a1=0.7714 (+0.0017, inside the 0.0097 native-tf spread), salvage_net_positive=True → trainability FLIPS GREEN-YELLOW (the 0.9213 uniform bar is NOT required — banked RED reference only). YELLOW: additive-upper step regime 6.86 re-inflates demand to 0.8717>0.80; cov_W measured on the LINEAR spine may not transfer to fusion — bank-the-analysis (0 TPS)

- **Branch:** `denken/eagle3-a1-cliff-trainability` · **Student:** denken · merged 07:21:31Z by advisor (8gpu launch; analytic + literature-only, reads native acceptance off the EXISTING in-repo head — NO drafter build, NO training, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`5axqa6oa`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/5axqa6oa) (group `eagle3-a1-trainability`, finished, NaN-clean, self-test 47/47). Independently W&B-verified by advisor (ALL MATCH). Merge commit `02abf1e`.
- **Primary:** `a1_cliff_trainability_self_test_passes=1` (47/47). **Test:** `a1_required_for_611=0.9213` (the uniform-bar banked RED reference).
- **Key finding:** against denken #304's uniform 0.9213 demand the answer would be RED (+26% top-1 over deployed = intrinsic-floor risk) — but that demand was relaxed by lawine #309's M=8 tree-salvage to raw **a1≈0.7731**. Against THAT bar the in-repo {2,21,39} EAGLE-3 head (fern #34, `gua9x68j`) measures native step-1 **a1=0.7714** — essentially AT the bar (+0.0017, inside the 0.0097 native-vs-tf spread), `salvage_net_positive=True`. A fresh retrain hitting ≥0.7731 is plausible, not a cliff → trainability is **GREEN-YELLOW**. YELLOW residuals: (1) the additive-upper step regime (6.86) re-inflates the demand to 0.8717 > 0.80; (2) cov_W is measured on the LINEAR spine and may not transfer to a fusion draft's rejection geometry (→ lawine #316). The 0.9213 uniform bar is retained only as a banked RED bookend.
- **Conclusion — build-economics matrix DRAFT-SIDE TRAINABILITY flips to GREEN-YELLOW: the a1-cliff is NOT an intrinsic floor under the relaxed tree-salvage demand.** Together with fern #310 (PRIVATE-500 clears), the two GREEN pillars move the EAGLE-3 GO/NO-GO to **GREEN-pending-build** — the residual is the human-gated build/measure spend, escalated as **issue #319**. Pairs with denken #304 (0.9213 uniform demand), lawine #309 (0.7731 relaxed bar), fern #34 (in-repo head gua9x68j).

## 2026-06-15 07:07 — PR #313: Pre-register the read-only probe that flips #309's YELLOW — 🟢 the whole #309 YELLOW collapses to ONE measurable bar: fusion frac_true_beyond_top4 < 0.3935 → GREEN (salvaged a1 demand <0.80); RED (#304's 0.9213) is UNREACHABLE for any positive rank-2+ transfer; CPU dry-run reproduces #79's cov4=0.6532 → harness checkpoint-ready — bank-the-analysis (0 TPS)

- **Branch:** `lawine/eagle3-fusion-rankcov-probe` · **Student:** lawine · merged 07:07:34Z by advisor (senpai; CPU-only pre-registration over #79/#309/#304 banked records — NO GPU, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`sw492nih`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/sw492nih) (group `eagle3-rankcov-probe`, finished, NaN-clean, self-test 13/13). Independently W&B-verified by advisor (ALL MATCH: fusion_rankcov_probe_self_test_passes=1, frac_beyond_top4_threshold_for_green=0.3935, probe_reproduces_linear_cov4=1, cov4_threshold_for_green=0.6065, verdict_linear_W4=GREEN). Merge commit `acbd278`.
- **Primary:** `fusion_rankcov_probe_self_test_passes=1` (13/13). **Test:** `frac_beyond_top4_threshold_for_green=0.3935`.
- **Key finding:** reproduces #309's salvage operator + inverse to 0.00e+00, then solves the W-invariant decision threshold `cov_W*=(T−d)/(1−d)`. At the trainable band edge d=0.80: cov4*=0.6065, frac*=0.3935. Falsifiable rule (primary W=4): measure fusion `frac_true_beyond_top4` — <0.3935 → GREEN (salvaged raw-a1 demand <0.80); the RED zone (#304's 0.9213) is UNREACHABLE for any positive rank-2+ transfer (returns only at literal cov4=0). The deployed LINEAR spine sits at frac=0.3468 → GREEN with +0.0467 headroom (cov4 may degrade 7.1% rel before YELLOW). CPU dry-run pushes #79's banked pooled histogram through the REAL analyze() → reproduces cov₂/cov₃/cov₄=0.4165/0.5715/0.6532 to 0.00e+00 (`probe_reproduces_linear_cov4=1`) → harness checkpoint-ready.
- **Conclusion — build-economics matrix TREE-SALVAGE sub-axis: #309's YELLOW reduced to a single cheap measurable bar.** The fusion `cov_W` remains UNMEASURED (no checkpoint) — this PRE-REGISTERS the test, does not run it. The transfer-function from that bar to E[T]/TPS (the YELLOW-band cost + the worst fusion rank-coverage that still clears E[T]=6.11) is reseated to **lawine #316**. Pairs with lawine #309 (salvage operator), wirbel #79 (rank-coverage z6wi4z4v), denken #304 (0.9213 demand).

## 2026-06-15 07:07 — PR #312: Price the served-file loopgraph rewrite EAGLE-3 requires — 🟡 the config-only EAGLE-3 swap REGRESSES to ~402 TPS (−16.5% vs 481.53): the onegraph loopgraph that PRODUCES 481.53 goes inert under method:eagle3 → stock eager EagleProposer; recovering frontier runtime costs 2 moderate-rewrites + 1 correctness-rederivation reopening all 4 gates + the #272 guard co-edit — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/eagle3-loopgraph-rewrite-cost` · **Student:** wirbel · merged 07:07:25Z by advisor (senpai; read-only static scoping of sitecustomize.py T5/T6/T7 + banked eager-step decomposition — NO training, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`9b1arani`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/9b1arani) (group `eagle3-rewrite-cost`, finished, NaN-clean, self-test 13/13). Independently W&B-verified by advisor (ALL MATCH: loopgraph_rewrite_cost_self_test_passes=1, eager_fallback_tps_floor=402.1, rewrite_reopens_greedy_identity=1, floor_band [302.3,471.0], n_touchpoints=3, guard_272_coedit_required=1). Merge commit `bfb5838`.
- **Primary:** `loopgraph_rewrite_cost_self_test_passes=1` (13/13). **Test:** `eager_fallback_tps_floor=402.1`.
- **Key finding:** the onegraph loopgraph is the ENGINE of 481.53, not a transparent optimization: keyed to `Gemma4Proposer`/`gemma4_mtp.get_top_tokens` whose "width-1 is exact" correctness rests on MTP being Q-only/KV-shared — FALSE for EAGLE-3's own-KV Llama draft. Under method:eagle3 all 3 touchpoints (T5 fused sparse-argmax, T6 loopgraph proposer key, T7 KV-bearing chain) go INERT → served path drops to stock eager EagleProposer at **~402.1 TPS** (band [302.3, 471.0]; directly measured from drafter eager 2859µs / step 13893µs at iso-MTP-acceptance), a **−16.5% regression**. Recovering frontier runtime = 0 mechanical + 2 moderate-rewrite (T5/T6) + 1 correctness-rederivation (T7, KV-bearing draft chain), reopening ALL 4 gates (greedy-identity/PPL≤2.42/boot-500/TPS), plus the near-zero #272 `_guard_included_router` boot-500 guard co-edit (frontier 0 / sibling 2 at lines 2680/2696).
- **Conclusion — build-economics matrix DEPLOYMENT-COST axis priced: ~80 TPS of deployment debt before the build's numerator gain even counts.** A config-only EAGLE-3 swap is net-NEGATIVE at deploy (regresses below baseline); the 481.53 frontier is recoverable only by the T5/T6/T7 rewrite. So the EAGLE-3 numerator gain (#295/#304) must clear 481.53−402=~80 TPS of deployment debt before the rewrite lands. CRITICAL: the eager floor is at iso-MTP-acceptance — whether a higher EAGLE E[T] lifts the eager path to ≥500 (SKIPPING the rewrite entirely) is reseated to **wirbel #314**; the draft-side dispatch correctness of the rewrite to **ubel #315**. Pairs with wirbel #307 (integration YELLOW), fern #305 (GO-card).

## 2026-06-15 07:07 — PR #311: EAGLE-3 #101 risk — capture-SIZE dispatch, not VRAM — 🟢 the deployed frontier does NOT pin cudagraph_capture_sizes (inherits ceiling 16); verify-side dispatchable widths = {8,16}: deployed M=8 SAFE, M=16 boundary-SAFE, M=32 falls to #101's IndexError CRASH — and the blocker is the DISPATCH LIST (+0 bytes), fixed by adding [24,32], NOT VRAM (#306's +12 MiB) — bank-the-analysis (0 TPS)

- **Branch:** `ubel/eagle3-capture-dispatch-risk` · **Student:** ubel · merged 07:07:16Z by advisor (senpai; read-only static analysis of the served capture config + per-width dispatch arithmetic — NO GPU, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`os01ttw9`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/os01ttw9) (group `eagle3-dispatch-risk`, finished, NaN-clean, self-test 7/7 a–g). Independently W&B-verified by advisor (ALL MATCH: capture_dispatch_risk_self_test_passes=1, max_safe_tree_width_under_deployed_list=16, deployed_m8_dispatch_safe=1, m32_blocker_is_dispatch_not_memory=1, xcheck_306_max_abs_err=0, mitigating_sizes_to_add=[24,32]). Merge commit `8b3a061`.
- **Primary:** `capture_dispatch_risk_self_test_passes=1` (7/7). **Test:** `max_safe_tree_width_under_deployed_list=16`.
- **Key finding:** the deployed frontier `fa2sw_precache_kenyan` does NOT pin `cudagraph_capture_sizes` (no `--compilation-config`/`--cuda-graph-sizes`/`--enforce-eager` in serve.py) → inherits vLLM's default ceiling `max_cudagraph_capture_size=16` (banked #101/#306). Verify-side dispatchable widths = the (1+K)=8-multiples within it = **{8,16}**: deployed **M=8 SAFE** (prewarm (1,8), serve.py:487), **M=16 boundary-SAFE**, **M=32 → #101's IndexError CRASH** (32>16). The M=32 blocker is the DISPATCH LIST (+0 bytes), fixed by ADDING `[24,32]` to the list — NOT by freeing memory (#306 priced M=32 VRAM at trivial +12 MiB; xcheck vs y1lji0c6 `max_abs_err=0`). Draft side (K=7) is a manual ONEGRAPH capture (`CUDAGraphMode.NONE`) → safe by construction (not list-dispatched). Honest dual-boundary caveat carried: the public board attributes the in-serve size-29 crash to a star_gqa attention KERNEL (separate axis from this cudagraph-dispatch axis) — both must clear for any width past M=8.
- **Conclusion — build-economics matrix axis (k) dispatch sub-axis CLOSED: deployed M=8 dispatch-safe; the #101 risk is a config-list correctness property orthogonal to VRAM.** Completes ubel #306's runtime-VRAM finding. Whether the #312 loopgraph rewrite reintroduces a DRAFT-side dispatch risk (the EAGLE KV-bearing draft chain vs the MTP ONEGRAPH that is safe-by-construction) is reseated to **ubel #315**. Pairs with ubel #306 (runtime VRAM y1lji0c6), lawine #101 (size-29 crash), wirbel #312 (rewrite scope).

## 2026-06-15 06:31 — PR #305: EAGLE-3 GO/NO-GO decision card — 🟡 every banked constant round-trips ≤1e-6 and the corrected-target invariant reproduces public=500 to 5.7e-14, but under the CONSERVATIVE scalar ×0.804 PRIVATE is the binding axis: sub-500 across the entire #295 bracket (402 central, 451 top), private-cross at public E[T]=7.601; P(private≥500)=3.9% indep / 0% coupled — bank-the-analysis (0 TPS)

- **Branch:** `fern/eagle3-go-card` · **Student:** fern · merged 06:31:52Z by advisor (senpai; CPU-only synthesis of the priced axes — NO training, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`m4nmtdl9`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/m4nmtdl9) (group `eagle3-go-card`, finished, NaN-clean, self-test 7/7). Independently W&B-verified by advisor (ALL MATCH: eagle3_go_card_self_test_passes=True, p_private_clears_500=0.038575, go_card_tornado_top_axis=step_multiplier, et_private_cross_500=7.601). Merge commit `366bd67`.
- **Primary:** `eagle3_go_card_self_test_passes=1` (7/7). **Test:** `p_private_clears_500=0.0386`.
- **Key finding:** 25 banked imports verified ≤1e-6 (official 481.53, λ1 520.953, K_cal 125.268, E[T] 3.844, corrected 6.1112 [5.3636,6.8588], private_factor 0.804, companion 0.998, read ratio −2.0177, VRAM 20.10/3.90, build 107.47 GPU-hr); corrected-target invariant reproduces public=500 to 5.7e-14. Decision-relevant result: under the conservative scalar ×0.804, PRIVATE is the binding axis — at E[T]=6.11 private projects **402.0**, at bracket-top 6.8588 only **451.2**, both sub-500; private-500 crossing needs public **E[T]=7.601** (above the bracket, into the K+1=8 ceiling). Tornado headline axis `step_multiplier` (Δprivate 99.85), E[T]-bracket a near-tie (correlated view of the same regime), but the largest INDEPENDENT private lever is `private_factor`. P(private≥500)=3.9% independent / 0% honest-coupled.
- **Conclusion — build-economics matrix GO-card rollup: every COST axis is GO/feasible, but the conservative-scalar private projection is the YELLOW.** Public clearance at 6.11 is true by construction; the open question is the OOD private factor. NB this card prices the conservative bookend — lawine #300/#309's per-position model (rho_priv_e3=0.9421, private-500 needs public E[T]≈4.19) is precisely the dominant `private_factor` lever and the credible path to private≥500; reconciling the two private conventions at E[T]=6.11 is reseated to **fern #310**. Conditional on the in-flight reachability lane (kanna #294 / denken #308 / lawine #313). Pairs with all priced axes (a)–(k).

## 2026-06-15 06:31 — PR #307: EAGLE-3 served-path integration readiness — 🟡 NOT a config drop-in (swap_is_config_only=0): the onegraph loopgraph that PRODUCES 481.53 is MTP-specific and goes INERT under method:eagle3 — 3 served-file touchpoints (T5/T6/T7 in the boot-fragile sitecustomize.py) need rewrite + re-validation; readiness_blocks_go=1 — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/eagle3-integration-readiness` · **Student:** wirbel · merged 06:31:49Z by advisor (senpai; read-only static analysis of the served path — NO training, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`88eh8twv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/88eh8twv) (group `eagle3-integration-readiness`, finished, self-test 13/13). Independently W&B-verified by advisor (ALL MATCH: eagle3_integration_readiness_self_test_passes=1, swap_is_config_only=0, readiness_blocks_go=1). Merge commit `269c2cb`.
- **Primary:** `eagle3_integration_readiness_self_test_passes=1`. **Test:** `swap_is_config_only=0`.
- **Key finding:** the active frontier `fa2sw_precache_kenyan` runs MTP (stock upstream vLLM source-patched at import — no vendored fork). 6/9 integration touchpoints ARE config-only (stock vLLM exposes {2,21,39} aux hidden states, 0h work — feasibility PR #15). But the SPEED substrate is the blocker: the onegraph loopgraph that makes this the 481.53 frontier is keyed to `Gemma4Proposer`/`gemma4_mtp.get_top_tokens` and its "width-1 is exact" correctness rests on MTP being KV-shared/Q-only — FALSE for EAGLE-3's own-KV Llama draft layer. Under method:eagle3 those 3 patches (T5 sparse-argmax, T6 loopgraph proposer key, T7 `_build_static_buffers`/`_capture_graph` correctness) go INERT → frontier runtime lost unless rewritten for fused `[1,7680]` KV-bearing chain + greedy-identity/PPL/boot re-validated. Boot cross-check: EAGLE-3 does NOT re-open #272 (touchpoints live in spec-decode/model-exec, not the prometheus route path), BUT the rewrite reopens the same sitecustomize.py that is itself MISSING the `_guard_included_router` guard (0 matches) — port it in the same edit.
- **Conclusion — build-economics matrix INTEGRATION-READINESS YELLOW: deployment is a served-file change to a boot-fragile file, not a like-for-like swap.** Even if the economics say GO, the human decision must see that the 481.53-producing onegraph must be re-derived for the EAGLE proposer. The rewrite COST is unpriced — reseated to **wirbel #312** (scope T5/T6/T7 + the EAGLE-on-eager fallback floor + the #272 co-edit). Pairs with ubel #306 (runtime VRAM), fern #305 (GO-card).

## 2026-06-15 06:31 — PR #309: M=8 verify-tree relaxes the a1-cliff demand — 🟢🟡 the tree salvages rank-2+ so the EAGLE-3 drafter needs raw a1≈0.7731, not denken #304's 0.9213 — a +5.9% lift over the deployed linear (not +26%); robust across W=2..4, reproduces #300's c1=1.0 to 6dp; YELLOW pending fusion-draft rank-coverage transfer — bank-the-analysis (0 TPS)

- **Branch:** `lawine/eagle3-tree-salvage-a1` · **Student:** lawine · merged 06:31:47Z by advisor (senpai; CPU-only inversion of #300's salvage operator over wirbel #79's measured rank-coverage — NO training, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`7tkn4d9x`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/7tkn4d9x) (group `eagle3-tree-salvage-a1`, finished, NaN-clean, self-test 12/12). Independently W&B-verified by advisor (ALL MATCH: tree_salvage_a1_self_test_passes=1, a1_draft_required_after_tree_salvage=0.7731, tree_recovered_c1_at_a1_073=0.9064, rho_priv_e3=0.9421). Merge commit `9387adc`.
- **Primary:** `tree_salvage_a1_self_test_passes=1` (12/12). **Test:** `a1_draft_required_after_tree_salvage=0.7731`.
- **Key finding:** the salvage operator `c1_eff(a1)=a1+(1−a1)·cov_W` (cov_W = wirbel #79's MEASURED cumulative rank-coverage, run z6wi4z4v: cov₂=0.4165, cov₃=0.5715, cov₄=0.6532) reproduces #300's linear-spine c1=1.0 to 6dp as the anchor. At the deployed raw a1=0.73 the tree lifts effective a1 to **0.9064** (98.4% of #304's 0.9213). Inverting for the demand: to net E[T]=6.11 the fusion drafter needs raw **a1≈0.7731** (primary W=4) — `salvage_relaxes_304_demand_by=0.1482`, robust across the bracket (W=2 still 0.865<0.92). This moves the trainability target from RED (+26% top-1 over deployed) to GREEN-YELLOW (+5.9%). Collapses to #304's 0.9213 EXACTLY only at literal zero rank-2+ transfer.
- **Conclusion — build-economics matrix TREE-SALVAGE relaxes the draft-side demand: the verify-tree buys 0.148 in raw a1.** The decisive reconciliation of #304 (draft-level a1-break DEMANDED) and #300 (tree recovers c1 on the linear spine): inverting #300's own operator shows the fusion draft needs raw a1≈0.77, not 0.92. YELLOW caveat: cov_W is measured on the LINEAR spine; a {2,21,39}-fusion draft's distinct rejection geometry could lower cov_W (if frac_true_beyond_top4 > the linear 0.347). Pre-registering the cheap read-only probe that flips YELLOW→GREEN/RED is reseated to **lawine #313**; the relaxed 0.77 target feeds **denken #308**. Pairs with denken #304 (0.9213 demand), lawine #300 (c1 recovery), wirbel #79 (rank-coverage).

## 2026-06-15 06:31 — PR #306: EAGLE-3 runtime VRAM survives ONEGRAPH capture — 🟢 the build's 3.90 GiB headroom survives capture: runtime peak 20.158 GiB (resident 20.10 + transient 0.058) fits ≤24-hard with 3.84 GiB to spare; lawine's #101 crash is a capture-SIZE dispatch IndexError, NOT an OOM, and the deployed M=8 spine clears the size-16 list — bank-the-analysis (0 TPS)

- **Branch:** `ubel/eagle3-capture-peak` · **Student:** ubel · merged 06:31:44Z by advisor (senpai; LOCAL transient-VRAM proxy stacked on #299's resident anchor — NO training, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`y1lji0c6`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/y1lji0c6) (group `eagle3-capture-peak`, finished, NaN-clean, self-test 7/7). Independently W&B-verified by advisor (ALL MATCH: eagle3_capture_peak_self_test_passes=1, eagle3_build_peak_gb=20.158). Merge commit `d8fb99f`.
- **Primary:** `eagle3_capture_peak_self_test_passes=1` (7/7). **Test:** `eagle3_build_peak_gb=20.158`.
- **Key finding:** the feared mode ("capture-time peak eats the 3.90 GiB headroom → OOM launch-blocker") is REFUTED. Runtime build peak **20.158 GiB** (resident 20.10 + transient 0.058) fits the 24-hard ceiling with **3.84 GiB** headroom (1.90 vs device-visible 22.058). The capture pool is the dominant transient (~42 MiB) but only 1.1% of headroom; the 262k-vocab tree-verify logit buffer is bf16-native MB-scale (4 MiB at M=8, 16 MiB even at M=32 — no fp32 upcast, PyTorch #123911). Crucially, lawine's #101 size-29 crash is re-diagnosed (vLLM #29091/PR#23679) as a capture-SIZE-list DISPATCH IndexError (`max_cudagraph_capture_size=16`), NOT an allocation failure: VRAM clears M=32 trivially (+12 MiB) but the dispatch list does not. Deployed M=8 (K+M=15<16) clears.
- **Conclusion — build-economics matrix axis (k) RUNTIME-VRAM CLOSED at GREEN: the VRAM≤24 clause holds at runtime, not just at rest.** Completes ubel #299's resident-only finding. The genuine #101-class risk is orthogonal to memory (a capture-size dispatch-list correctness property) — pricing it against the deployed `cudagraph_capture_sizes` is reseated to **ubel #311**. Launch-hygiene flag carried: a serving process inheriting `CUBLAS_WORKSPACE_CONFIG=:4096:8` inflates the pool ~4→24 MiB/handle (still sub-GiB) — pin unset at launch. Pairs with ubel #299 (resident 20.10), wirbel #295 (step 6.11).

## 2026-06-15 06:02 — PR #304: Does hitting 6.11 require breaking the a1 cliff? — 🔴-for-the-drafter the heavier {2,21,39}-fusion step DEMANDS breaking the position-1 acceptance cliff (a1: 0.7292 → ~0.9213, ×1.263); kanna #289's spec yields E[T]=4.9196 and falls 1.19 SHORT of the 6.11 target — even the physically-impossible perfect-tail corner caps cliff-kept E[T] at 6.11. Axis (j) NUMERATOR-REACHABILITY CLOSED — bank-the-analysis (0 TPS)

- **Branch:** `denken/spec-reaches-611-target` · **Student:** denken · merged 06:02:47Z by advisor (senpai; CPU-analytic chain-product inversion of kanna #289's per-position spec — NO training, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`dtf1ouml`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/dtf1ouml) (group `spec-reaches-611-target`, finished, NaN-clean, self-test 13/13). Independently W&B-verified by advisor (ALL MATCH: spec_reaches_611_self_test_passes=1, heavier_step_demands_a1_break=1, a1_required_for_611=0.9213, spec289_et=4.9196, kanna289_spec_reaches_611=0/False, spec_shortfall_vs_611=1.1916). Merge commit `2b9cbb61`.
- **Primary:** `spec_reaches_611_self_test_passes=1` (13/13). **Test:** `heavier_step_demands_a1_break=1`.
- **Key finding:** inverts the chain-product `E[T] = 1 + Σ_{k=1}^{7} Π_{j≤k} a_j` to find the a_k profile required to hit wirbel #295's step-corrected **6.11** target. kanna #289's spec (a₁≥0.73, j≥2→0.91) yields E[T]=**4.9196** — **1.1916 short** of 6.11. The binding constraint is **position 1**: even at the perfect-tail corner (a_{j≥2}=1.0), an a₁ held at the deployed 0.7292 cliff caps E[T] at exactly 6.11 < 6.1112. To reach 6.11 the drafter needs **a₁ ≈ 0.9213** (×1.263 over deployed, vs low-quartile 0.6550 from #297) — a strictly harder, near-uniform-high drafter, INCLUDING position 1.
- **Conclusion — build-economics matrix axis (j) NUMERATOR-REACHABILITY CLOSED: the 6.12 numerator is reachable only by a drafter that BREAKS the a1 cliff.** This is the decisive tightening of the human GO/NO-GO: every cost axis (step/VRAM/build-cost/companion/private) is priced GO/feasible, but they all assume a drafter hitting the required a_k EXISTS — and #304 shows that drafter must do something the deployed linear drafter does NOT (lift a₁ from 0.73 to 0.92). The open question splits cleanly into two orthogonal follow-ups: **draft-side** trainability (is a₁≈0.92 inside the published EAGLE-3 envelope or an intrinsic floor? → denken #308) and **verify-side** salvage (does the M=8 tree recover a₁ for a fusion draft the way lawine #300 showed it does for the linear spine, relaxing this demand? → lawine #309). (NB: the earlier-banked entries refer to this lane as "wirbel #303" — that placeholder PR was de-conflicted/closed since denken #304 owns the same chain-product inversion; wirbel reseated to the orthogonal integration-readiness axis #307.) Pairs with wirbel #295 (6.11 step), kanna #289 (per-position a_k), lawine #300 (private c1 recovery).

## 2026-06-15 06:02 — PR #300: Private-bar EAGLE-3 — 🟢 a public-E[T] build clears the PRIVATE ≥500 bar; the card reproduces the organizer-verified 460.85 private TPS exactly (resid 0.00, Δ4.29%) and prices the private collapse as position-1 HELD (c1=1.0 via M=8 tree-salvage), residual c_deep=0.97135 on j≥2 — bank-the-analysis (0 TPS)

- **Branch:** `lawine/private-bar-eagle3` · **Student:** lawine · merged 06:02:50Z by advisor (senpai; CPU-analytic private-bar decomposition calibrated to ground truth — NO model forward, NO training, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`8t5q6sr0`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/8t5q6sr0) (group `private-bar-eagle3`, finished, NaN-clean, self-test 13/13). Independently W&B-verified by advisor (ALL MATCH: private_bar_eagle3_self_test_passes=1, rho_priv_e3=0.9421, private_tps_deployed_anchor=460.85, private_tps_resid_vs_460p85=0.00, delta_pct_deployed=0.0429). Merge commit `ffb515bd`.
- **Primary:** `private_bar_eagle3_self_test_passes=1` (13/13 + nan-clean). **Test:** `rho_priv_e3=0.9421`.
- **Key finding:** the deployed a_k collapse model reproduces the organizer-verified **460.85** private TPS exactly (resid **0.00**, Δ **4.29%** vs 481.53 — matches the stated 4.3%), so the decomposition is sound before pricing the build. The crux: the **0.804 aggregate is the RAW width-1 spine ratio** (ubel #258: 3.0898/3.8445), but the deployed-verified 460.85 = E[T]_priv **3.6790** is the **tree-recovered ratio 0.955** — the M=8 verify-tree salvages the rank-2+ matches the width-1 spine rejects, **recovering the position-1 cliff** (c₁=1.0). So the deployed-effective private collapse is **position-1 HELD, residual c_deep=0.97135 on j≥2**, and the EAGLE-3 public-E[T] build's private projection clears ≥500 under this calibrated collapse (`rho_priv_e3=0.9421`).
- **Conclusion — build-economics matrix axis (g) PRIVATE-BAR CLOSED at CLEARS.** The build does not have a hidden private-side death: the private gap is dominated by a j=1 discrimination loss that the deployed M=8 tree already salvages, and deep j≥2 conditionals hold (ubel #258 showed deep j actually improves OOD). This sets up the key reconciliation with denken #304: #304's a₁-cliff-break demand is a DRAFT-level requirement, while #300 shows the deployed M=8 TREE already recovers c₁ on the linear drafter — so does that salvage transfer to a fusion draft and relax the demand? Reseated to **lawine #309** (verify-side) alongside **denken #308** (draft-side). Pairs with ubel #258 (rank-2+ OOD), ubel #263 (private rank-2 collapse), wirbel #295 (6.11 step).

## 2026-06-15 05:51 — PR #299: EAGLE-3 build VRAM budget — 🟢 the {2,21,39}-fusion drafter + hidden-state retention FITS the 24GB lane at 20.10 GB resident (3.90 GiB headroom); dominant term is extra_kv, drafter weights negligible — bank-the-analysis (0 TPS)

- **Branch:** `ubel/eagle3-vram-budget` · **Student:** ubel · merged 05:51:31Z by advisor (senpai; LOCAL bottom-up VRAM accounting + live-config spot-check on the A10G — NO training, NO checkpoint, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`jnoss7id`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/jnoss7id) (group `eagle3-vram-budget`, finished, NaN-clean, self-test 10/10 a–k). Independently W&B-verified by advisor (ALL MATCH: self_test=1, eagle3_build_fits_24gb=1, eagle3_build_resident_gb=20.10, headroom_vs_24_hard_gib=3.90, headroom_vs_23_usable_gib=2.90, dominant_memory_term=extra_kv, drafter_weights_gib=0.037, extra_kv_hold_capacity_gib=0.719, nontorch_cuda_context_gib=0.950, net_memory_delta_gb=0.80, spot_check_rel_err=0.0022). Merge commit `1aa53e3e`.
- **Primary:** `eagle3_vram_budget_self_test_passes=1` (10/10 a–k). **Test:** `eagle3_build_fits_24gb=1` (also `fits_23_usable=1`).
- **Key finding:** the human-gated EAGLE-3 build (the {2,21,39}-fusion 1-Llama-layer 2560-dim drafter + the 3-layer hidden-state retention for fusion) lands at **20.10 GB resident** → **3.90 GiB headroom** vs the 24 GB hard ceiling (2.90 GiB vs the 23 GB usable line). The dominant memory term is **`extra_kv`** (0.719 GiB of KV-hold capacity), NOT the drafter weights (0.037 GiB — negligible) — net delta over the deployed system is only **+0.80 GB**. Live-config spot-check rel-err **0.22%** (config read from live, dims match banked). Honest caveat carried: `s2_exceeds_device_visible=1` — the full deployed model exceeds the device-visible window (expected, explicitly tracked via `selftest_k_honest_caveats_carried`).
- **Conclusion — build-economics matrix axis (e) VRAM CLOSED at FITS (at rest).** The build is memory-feasible on the a10g-small 24 GB lane with ~3.9 GiB to spare; weights are not the constraint, KV-hold is. This is the RESIDENT budget — the runtime PEAK (ONEGRAPH CUDA-graph capture + M-wide tree-verify logit transient over the 262k vocab) is the separate question reseated to **ubel #306** (does the 3.90 GiB headroom survive capture, given lawine #101's size-29 graph-capture crash precedent?). Pairs with denken #301 (cost 107.47 GPU-hr GO), wirbel #295 (step 6.11), lawine #300 (private bar): the build is now priced feasible on BOTH dollars and memory.

## 2026-06-15 05:51 — PR #302: Read-cut as a build companion — 🟢 the #287 8.43% read-cut FITS the build's PPL headroom (cost 0.0203 < 0.0428) but REGRESSES on the wall (ratio −2.0177 ≈ stark #273's −2.018) → BUILD ALONE; the read-cut is CLOSED as standalone AND companion — bank-the-analysis (0 TPS)

- **Branch:** `fern/read-cut-build-companion` · **Student:** fern · merged 05:51:10Z by advisor (senpai; LOCAL composition/realization analysis re-using stark #273's banked realization behavior — NO training, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`8jewx2ur`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/8jewx2ur) (group `read-cut-build-companion`, finished, NaN-clean, self-test 10/10). Independently W&B-verified by advisor (ALL MATCH: self_test=1, read_cut_is_free_build_companion=0, read_cut_wall_realized_tps_credit=−33.43, read_cut_realization_ratio=−2.0177, read_cut_ppl_cost=0.02032, ppl_headroom=0.04280, headroom_residual_after_read_cut=0.02248, read_cut_fits_ppl_headroom=1, realized_uplift_pct=−6.687, realization_classification=regresses, composed_build_credit_diluted_pct=2.666). Merge commit `ab8493c`.
- **Primary:** `read_cut_build_companion_self_test_passes=1` (10/10). **Test:** `read_cut_is_free_build_companion=0` (verdict: NOT a free companion).
- **Key finding:** the read-side analog of lawine's SAM drafter-companion question. fern's #287 (MERGED) closed the read-cut as a STANDALONE lever (8.43% PPL-safe body-read cut, but re-priced on denken #283's 38% read-fraction it does not clear 500). #302 asks the orthogonal portfolio question: is it a FREE COMPANION to the EAGLE-3 build? On PPL it FITS — the 8.43% cut costs only **0.0203** PPL, inside the build's **0.0428** headroom (residual 0.0225 left). But on the WALL it does NOT realize: composing the read-credit on the deployed ONEGRAPH K=7 graph and discounting by stark #273's MEASURED static-K realization behavior gives `read_cut_wall_realized_tps_credit=−33.43 TPS`, `read_cut_realization_ratio=−2.0177` — a **sign-flip** that closely matches #273's static-K −2.018 (`realized_uplift_pct=−6.69%`, classification **`regresses`**). The composed build-credit dilutes to 2.666% and goes negative on realization. Recommendation banked: **BUILD ALONE**.
- **Conclusion — build-economics matrix axis (i) READ COMPANION CLOSED; the read-cut is dead on BOTH doors.** Body-side deviations from the deployed ONEGRAPH K=7 graph regress on the wall exactly as stark #273's precedent predicted (#273 was load-bearing here). The read-cut is not free standalone (#287) and not free as a companion (#302) — the EAGLE-3 build should not bundle it. This narrows the portfolio: the build's TPS must come from the NUMERATOR (acceptance via the trained drafter), not from stacking body-side read levers. Pairs with lawine #296 (SAM companion, also sub-additive) — both companion levers priced and both shrink under the build.

## 2026-06-15 05:37 — PR #301: EAGLE-3 build-cost card — 🟢 the GATED drafter costs ~107 A10G-GPU-hr (~13.4 h wall on 8×A10G) to BUILD — GO-ON-COST under the ≤200 GPU-hr lane (headroom 92.5); the 262k-vocab lm_head GEMM DOMINATES (68.2 hr, 78% of train), NOT the 124.5M draft net — bank-the-analysis (0 TPS)

- **Branch:** `denken/eagle3-build-cost` · **Student:** denken · merged 05:37:16Z by advisor (senpai; LOCAL bottom-up FLOP/GPU-hour synthesis on the A10G — NO training, NO checkpoint, NO served-file change; analysis-only; BASELINE 481.53, 0 TPS). W&B [`b4zg7b6c`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/b4zg7b6c) (group `eagle3-build-cost`, finished, NaN-clean, peak 13.7 MiB, self-test 11/11 a–k). Independently W&B-verified by advisor (ALL MATCH: self_test=1, total_build_gpu_hours=107.466, capture=20.317, drafter_train=87.148, lm_head=68.174, wall_8xa10g=13.433, headroom=92.534, go_nogo=GO, lm_head_dominates_drafter=1, drafter_term_divergence_x=4.593). Merge commit `4e944c1`.
- **Primary:** `eagle3_build_cost_self_test_passes=1` (11/11 a–k). **Test:** `total_build_gpu_hours=107.47` (verdict `build_cost_feasible_under_budget=true`, GO-ON-COST).
- **Key finding:** bottom-up at A10G sm_86 (125 TFLOP/s × MFU 0.35 = 43.75 TFLOP/s eff), N=4e8 train-tokens (532K entries × ~752 tok/entry), 10 epochs: **capture-pass 20.32** GPU-hr (target fwd 2·4.0B·4e8) + **drafter-train 87.15** GPU-hr = **107.47** total → **13.43 h wall** on 8×A10G (DP, ideal scaling). The drafter-train splits **core 18.97** + **frozen 262k-vocab lm_head GEMM 68.17** — the lm_head DOMINATES the 124.5M draft net **3.59×/token** and inflates the total **2.74×** over the PR-literal core-only accounting (39.29 GPU-hr), divergence flagged >2×. Sensitivity grid stays feasible through 4e8×20ep (194.6 GPU-hr) and only breaks at 8e8×10ep (214.9 > 200). Honest caveat carried: ORDER-OF-MAGNITUDE estimate, NOT a paper quote — the paper reports entries not tokens, gives no epoch/MFU, and reports NO train cost, so this bottom-up IS the anchor.
- **Conclusion — the human EAGLE-3 GO/NO-GO now has its SPEND axis, and it is GO.** ~107 A10G-GPU-hr (~13.4 h on 8×A10G) to build the gated drafter at N=4e8 × 10ep, comfortably under the ≤200 GPU-hr lane (headroom 92.5). Cost is no longer the missing input. Dominant term is the lm_head GEMM, not the draft net — so any build-cost reduction lever (smaller vocab head, tied/factored head, fewer epochs) attacks the lm_head first. Build-economics matrix item **(h) build-cost CLOSED at GO**. Pairs with wirbel #290 (acceptance ceiling), wirbel #295/#293 (step-cost denominator 6.11), kanna #294 (Phase-1 viability), ubel #299 (VRAM), lawine #300 (private bar): the build's economics are now bounded on cost, step, companion, and per-position shape — the live residual is numerator-REACHABILITY (wirbel #303: can a trained EAGLE-3 actually deliver E[T]≈6.12?).

## 2026-06-15 05:27 — PR #295: EAGLE-3 fusion-drafter step profile — 🟢 the measured random-init draft-step cost VALIDATES #293's conservative 6.1245 (regime bracket [5.36, 6.86] central 6.11, multiplier ~2.95× vs assumed 3×) rather than collapsing it; the step-cost DENOMINATOR is now a measured point, axis (a) CLOSED — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/eagle3-step-profile` · **Student:** wirbel · merged 05:26Z by advisor (senpai; LOCAL random-init EAGLE-3 fusion-drafter forward profiling on the A10G — NO training, NO checkpoint, NO served-file change — + CPU-analytic correction; BASELINE 481.53, 0 TPS). W&B [`c334qaqu`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/c334qaqu) (group `eagle3-step-profile`, finished, NaN-clean, self-test 8/8 a–h). Independently W&B-verified by advisor (ALL MATCH: self_test=1, corrected_central=6.1112, eagle3_corrected_target_measured=5.3636, eagle3_draft_wall_us=1233.23, measured_multiplier_faithful=1.7447, per_step_multiplier_faithful=1.7091, bracket_straddles_293=1, bracket_central_validates_293=1, bracket_within_window=1, byte_ratio×BW-frac=1.7456, gpu_peak_mem_gib=1.906).
- **Primary:** `eagle3_step_profile_self_test_passes=1` (8/8 a–h). **Test:** `eagle3_corrected_target_central=6.1112`.
- **Key finding:** finalizes wirbel #293's MODELED correction by MEASURING the EAGLE-3 fusion-draft step. Profiled (CUDA-graph, bf16, batch=1, ctx=528, K=7) the faithful {2,21,39}-fusion drafter (2560-dim, 1 Llama layer + fc 7680→2560 + head) vs the deployed linear K=7 chain (706.86µs anchor, denken #278). Raw standalone multiplier **1.745×** — but that is a **dispatch-compressed LOWER bound**: the 256-dim linear runs at only **11.5%** of A10G BW (launch/dispatch-bound at batch=1) vs the faithful's **59.9%**; the deployed ONEGRAPH+INT4 regime removes that asymmetry and RAISES the true ratio. Self-consistency: `byte_ratio 9.116 × (BW_lin/BW_eag 68.8/359.4) = 1.746 == measured 1.745` ✓. Two honest anchorings bracket the corrected target: MULTIPLICATIVE (lower, 1.745×) → 5.364; ADDITIVE (upper, 4.16×) → 6.859; **regime-central (~2.95×, ≈INT4-corrected) → 6.111** — straddling #293's 6.1245 within 0.012.
- **Conclusion — axis (a) of the build-economics matrix CLOSED; 6.12 stands as the central sizing bar.** The measured ~2.95× confirms #293's modeled 3× was essentially dead-on; the corrected BUILT-raise target is now a MEASURED point (central 6.11, bracket [5.36, 6.86]) INSIDE the 8.0 feasibility window across the whole physical range (only the unphysical pure-BW extreme 9.92 exits). The naive optimistic read ("1.745× → 5.36, EAGLE-3 is cheap") was correctly NOT headlined. This is the step-cost DENOMINATOR for kanna #294's Phase-1 viability gate; it bounds STEP COST, NOT achieved E[T] — the numerator-REACHABILITY question (can a trained EAGLE-3 deliver E[T]≈6.12?) is the separate lane reseated to **wirbel #303**. Student follow-up #2 (in-ONEGRAPH profiling to pin the multiplicative anchor) is the cleanest residual-uncertainty collapse if [5.36, 6.86] ever needs tightening. Pairs with lawine #296 (residual-2.22 at 6.1245), kanna #289 (per-position shape the build must hit at this target).

## 2026-06-15 05:25 — PR #287: Read-reduction PPL Pareto — 🟢 the max PPL-safe body-read cut is 8.43% but re-priced on denken #283's MEASURED 38% read-fraction it does NOT clear 500 (verdict FLIPS True→False); the READ-side door to 500 is CLOSED — bank-the-analysis (0 TPS)

- **Branch:** `fern/read-reduction-ppl-pareto` · **Student:** fern · merged 05:25:34Z by advisor (senpai; LOCAL per-layer int3/int4 sensitivity-ranked fake-quant PPL Pareto + TPS re-attribution, no served-file change; BASELINE 481.53, 0 TPS). W&B [`17en3hus`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/17en3hus) (REVISED run; group `read-reduction-ppl-pareto`, finished, NaN-clean, self-test 10/10) — supersedes the optimistic first pass [`uc2mqt82`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/uc2mqt82). Independently W&B-verified by advisor (all MATCH: self_test=1, max_ppl_safe_read_reduction_pct=8.431155, read_reduction_lever_clears_500=False, best config `mixed_int3_demote16L` projected_deployed_ppl=2.397521 ≤ 2.42, all 10 selftest sub-keys=1). Merge commit `35d75bfb`.
- **Primary:** `read_reduction_ppl_pareto_self_test_passes=1` (10/10). **Test:** `max_ppl_safe_read_reduction_pct=8.431` (verdict `read_reduction_lever_clears_500=False`).
- **Key finding:** the maximum PPL-safe body-read reduction is **8.43%** (sensitivity-ranked mixed int4/int3 demotion of the 16 least-sensitive layers, config `mixed_int3_demote16L`, projected deployed PPL **2.3975** ≤ 2.42 with 0.0203 of the 0.0428 gate headroom; 2:4 structured sparsity correctly ruled PPL-CATASTROPHIC and excluded). On the OPTIMISTIC full-read proxy the lever appeared to clear 500 (first pass `uc2mqt82`, verdict True) — but after the advisor send-back asked for honest TPS re-attribution on **denken #283's MEASURED 38% read-fraction** (reads are only 38% of the honest wall, not the full step), the verdict **FLIPS to False**: an 8.43% cut of a 38%-of-wall read share is far too small to reach 500. This is the advisor→student iteration working as intended — the re-priced run is the bankable one.
- **Conclusion — the READ-side door to 500 is CLOSED (standalone):** read-reduction is a small, PPL-bounded body-side lever that does NOT clear 500 alone, reinforcing the step-side closure (denken #291/#283, free ceiling 487.7) and the host-side closure (ubel #284, 0.50%) from the read denominator — **the sole >500 path remains the human-gated EAGLE-3 E[T]-raise BUILD**. The 8.43% PPL-safe cut + per-layer int3 sensitivity ranking is bankable as the read-reduction frontier. Reseated fern #302 (is the #287 read-cut a FREE body-side COMPANION to the build — does the 8.43% cut stack as TPS within the 0.0428 PPL headroom, or under-realize/regress on the ONEGRAPH K=7 graph like stark #273's static-K did? the read-side analog of lawine's SAM drafter-companion). Orthogonal to denken #283 (read floor) + the EAGLE-3 build legs.

## 2026-06-15 05:11 — PR #297: Tail-resolved per-position — 🟢 the hard tail does NOT shift the cliff, it DEEPENS a_1 at the SAME position-1 cliff ⇒ the EAGLE-3 per-position build target is UNIFORM (lift j≥2 to ~0.91), not prompt-adaptive — bank-the-analysis (0 TPS)

- **Branch:** `denken/tail-resolved-per-position` · **Student:** denken · merged 05:11Z by advisor (senpai; LOCAL per-prompt per-position acceptance remeasure on the DEPLOYED linear drafter, conc=1 MAX_NUM_SEQS=1, checkpoint unmodified, no served-file change; BASELINE 481.53, 0 TPS). W&B [`vo2ir6ca`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/vo2ir6ca) (group `tail-resolved-per-position`, finished, NaN-clean, self-test 16/16). Independently W&B-verified by advisor (all MATCH: self_test=True, tail_cliff_shifts=False, cliff_position_low=cliff_position_top=1, eagle3_target_is_prompt_adaptive=False, a1_low=0.6550, a1_top=0.8291). Peak GPU 18.94 GB, decode wall 245.9s.
- **Primary:** `tail_resolved_per_position_self_test_passes=1` (16/16). **Test:** `tail_cliff_shifts=0` (False).
- **Key finding:** RESOLVES kanna #289's shape-transfer caveat (`per_prompt_per_position_banked: False`) with a DIRECT per-quartile remeasure off a fresh 128×7 per-prompt acceptance matrix. Per-prompt snapshotting is EXACT (whole-run reconcile resid 0.00, prefix-invariant resid 0.00; whole-run a_k reproduces #289 to **3.3e-06**). The cliff is **prompt-INVARIANT at position 1** (LOW & TOP quartiles both cliff at 1; 124/128 individual prompts cliff at 1, 4 at 2, none deeper). The hard (bottom-E[T]) tail's deficit is a **UNIFORM vertical down-shift** of the whole a_k chain ("a1-deepen": LOW a_1=0.6550 vs TOP a_1=0.8291, gap 0.174; token_loss[1] 2.415 vs 1.196 — DEEPER at the same position, not shifted later). LOW pooled E[T]=3.1319 reconstructs #289's shape-transfer 3.093 (resid +0.039); TOP 5.0085 reconstructs 5.052 (resid −0.043).
- **Conclusion — the EAGLE-3 per-position target is UNIFORM:** because the deepening is a vertical shift (not a horizontal cliff move), one fixed per-position target (lift j≥2 to ~0.91) covers both tails; no prompt-adaptive depth needed, and the binding constraint is raising a_1 on the bottom-E[T] tail. Cleanest possible input for the build spec. Feeds **kanna #294** (a SINGLE a_2 threshold, no prompt-conditioning; the GO bar is best evaluated against the LOW quartile's a_1=0.6550 headroom, not whole-run 0.7293). Step-side denominator already closed by denken #291 (487.7). Reseated denken #301 → EAGLE-3 build-cost card (GPU-hour SPEND axis). Orthogonal to the other EAGLE-3 build legs.

## 2026-06-15 05:10 — PR #296: SAM × EAGLE-3 companion stacking — 🟢 the +2–4% SAM companion SUB-ADDITIVELY shrinks to [0.79,1.59]% under a better drafter (r_overlap 0.603), RAISING the honest residual EAGLE-3 must cover alone 0.902→0.998 — bank-the-analysis (0 TPS)

- **Branch:** `lawine/sam-eagle3-stacking` · **Student:** lawine · merged 05:10Z by advisor (senpai; CPU analytic over banked W&B numbers — loads #292/#289/#290/#293 reports, no re-measure, no served-file change; BASELINE 481.53, 0 TPS). W&B [`15ilrhrg`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/15ilrhrg) (group `sam-eagle3-stacking`, finished, NaN-clean, self-test 7/7 a–g). Independently W&B-verified by advisor (all MATCH: self_test=True, sam_marginal_under_eagle3_pct=1.5865, true_residual_for_eagle3=0.9979, r_overlap=0.6034, sam_companion_survives_eagle3=True).
- **Primary:** `sam_eagle3_stacking_self_test_passes=1`. **Test:** `sam_marginal_under_eagle3_pct=1.5865` (upper; 0.793 lower).
- **Key finding:** de-optimises lawine #292's clean-additive SAM residual (0.902). On a kanna-#289-profile EAGLE-3 drafter the +2–4% SAM companion **sub-additively shrinks to [0.79, 1.59]%** (keeps ~40%). `r_overlap=0.6034 = 0.4118 (a_k-lift: EAGLE-3 raises abar 0.80311→0.88418, miss-reduction 0.4118) + 0.1916 (correlation-tilt: SAM hits concentrate where the drafter already accepts, pearson +0.32576)`. The honest residual EAGLE-3 must cover ALONE rises from 0.902 → **0.9979 E[T]** (vs 4.9029; +10.6%); vs 6.1245 → 2.2196. Mechanism: SAM is mutually-exclusive per step (~14% retrieval-active / ~86% neural fallback), so a stronger neural drafter absorbs the shared substrate — corroborated by SAM-Decoding's own +3.28–11.13% over-EAGLE-2 vs +84–129% over-greedy.
- **Conclusion — #292's additive residual was OPTIMISTIC; SAM is a real-but-weaker companion on a better drafter.** Decision-relevant: at the kanna #289 target profile EAGLE-3 ALONE clears the 4.9029 step-banked target (E[T]=4.9156) — SAM is a redundant bonus there, not load-bearing; but against the honest 6.1245 step-overhead-corrected target (wirbel #293) NEITHER EAGLE-3 alone nor EAGLE-3+shrunk-SAM clears (residual 2.22; SAM covers ~3%). So the human EAGLE-3 GO/NO-GO should price against **6.1245**, where SAM is NOT the lever. Reseated lawine #300 → private-bar EAGLE-3 target (does the PUBLIC build clear the binding PRIVATE ≥500 gate after ubel #263's 0.804 rank-2+ collapse?). Pairs with wirbel #293 (step cost) + denken #297 (per-position target) to price the full honest cost of the gated raise.

## 2026-06-15 05:03 — PR #284: Decode-loop host overhead — 🟢 the decode loop is 99.5% GPU-bound (host/serving overhead 0.50%); the host front is CLOSED — bank-the-analysis (0 TPS)

- **Branch:** `ubel/decode-host-overhead` · **Student:** ubel · merged 05:03:13Z by advisor (senpai; LOCAL STEPTIME host-to-host decode-wall profiling + CUDA-event GPU-busy + first-party kernel/sampling trace, no served-file change; BASELINE 481.53, 0 TPS). W&B [`u58fxtu6`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/u58fxtu6) (group `decode-host-overhead`, finished, NaN-clean, self-test 6/6). Independently W&B-verified by advisor (all MATCH: self_test=1, host_overhead_frac=0.0049894, host_overhead_clears_materiality=0/False, recoverable_host_overhead_tps=+0.499, 0 NaN keys). Merge commit `ee8d7121`.
- **Primary:** `decode_host_overhead_self_test_passes=1` (a–f). **Test:** `host_overhead_frac=0.004989` (0.50%).
- **Key finding:** directly measuring the per-step decode wall (**8017µs** host-to-host p50) and subtracting the **DEPLOYED** model-forward GPU-busy (**7977µs**, CUDA-event — verify 6532 + drafter 1445) leaves **40µs (0.50%)** host/serving overhead — an order of magnitude below fern #274's inferred ~40% (φ band [12.5%, 73.5%]). Wall-identity holds: 8017µs × 16154 steps = 129.51s ≈ decode_wall_total 129.41s (resid 0.076%). The decode loop is **99.5% GPU-bound**; every host lever (ONEGRAPH step-fusion, FUSED_SPARSE_ARGMAX on-GPU sampling, DETOK_ENDONLY, FASTRENDER+orjson) is ALREADY deployed; the 40µs residual is irreducible inter-graph dispatch. **Makes the denken #278 over-credit concrete:** subtracting the M=1 micro-built model-forward (5673.6µs) manufactures a PHANTOM **29.2%** host overhead because the micro-bench under-counts deployed M=8 in-stack GPU-busy by **2303µs of REAL GPU work** (verify M=8−M=1 +1565µs; drafter +738µs) — that artifact is exactly what put fern #274 near ~40%. First-party sampling measured at **2.72%** of GPU-busy (replaces a hardcoded 2.92% estimate).
- **Conclusion — the host/serving front is CLOSED:** recoverable host overhead = **+0.50 TPS** (composition-honest, denken-bridge-discounted), well below the 9.63 TPS (2%) materiality gate (`host_overhead_clears_materiality=False`). Closes the per-step decode-wall coverage — prefill (#275, 2.85%) + model-forward (#278) + host (#284, 0.50%) + read floor (#283) all reconcile to the same wall, explaining the 481–490 TPS leaderboard saturation as a 99.5%-GPU-bound stack with no host slack to convert into TPS. Reinforces "the sole >500 path is the human-gated E[T]-raise BUILD" from the host side. **3 script bug-fixes** (wandb namespace-shadow import; kernel-pass `_IncludedRouter` prometheus guard = the Issue #272 boot-500 guard; measured-not-hardcoded sampling 2.72%) — ubel's validity script only, no served change. Reseated ubel #299 (EAGLE-3 build VRAM budget — the build's memory-feasibility axis). Orthogonal to denken #283 (read floor) + the EAGLE-3 build legs.

## 2026-06-15 05:02 — PR #273: Static-K wall-clock A/B — 🟢 measured local wall-clock REFUTES the static-K composition gain (K4-vs-K7 = −8.63%, realization ratio NEGATIVE); deployed K=7 stands — bank-the-analysis (0 TPS)

- **Branch:** `stark/static-k-wallclock-ab` · **Student:** stark · merged 05:02:27Z by advisor (senpai; LOCAL wall-clock A/B on the real served stack, 128×512 single-stream greedy, 2 seeds, no served-file change; BASELINE 481.53, 0 TPS). W&B [`51bdsbpw`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/51bdsbpw) (verdict aggregator) + 6 per-K/per-seed arms (`1gumbffx` k3, `a2459s21` k4·s1, `ba66548a` k5·s1, `gnvplti3` k6·s1, `7ucfac4h` k4·s2, `6mlmt3y7` k5·s2), group `static-k-wallclock-ab`. Independently W&B-verified by advisor (all MATCH: self_test=1, gain_k4_vs_k7=−8.629%, K4 median 414.475 TPS, K5 437.942, realization ratio K4 −2.018 / K5 −0.864, 0 NaN). Merge commit `54fadfa6`.
- **Primary:** `static_k_wallclock_ab_self_test_passes=1`. **Test:** `measured_local_wall_tps_gain_k4_vs_k7_pct=−8.63`.
- **Key finding:** the #266 static-K composition predicted **+4.28%** (K=4) / +4.00% (K=5) net gains; measured local wall-clock = **−8.63%/−8.79%** (K=4, two seeds) and **−3.46%/−3.74%** (K=5) — realization ratio **K4 = −2.02, K5 = −0.86, both NEGATIVE** (not merely <1, a SIGN FLIP). The composition over-credited the per-K draft-pass saving (the denken #278 bridge class); on the deployed ONEGRAPH/precache stack, any K≠7 falls off the K=7-optimized CUDA graph and REGRESSES. Outcome (b).
- **Conclusion:** the static-K re-opt lever is CLOSED — K=7 is a hard local optimum on the served stack, and the composition-to-wall realization gap is not just lossy but NEGATIVE. Confirms the step-side is saturated at K=7 from the draft-shape axis. **Method note:** stark's realization-ratio (measured Δ / composed Δ) is the load-bearing tool — it caught a sign-flip a careful composition missed; reused next to wall-audit the banked 487.7 free ceiling (does the composed step ceiling realize on the host-to-host wall, or over-credit like static-K?). Reseated stark #298 (does the banked 487.7 free ceiling realize on the wall-clock?). Orthogonal to the EAGLE-3 build legs.

## 2026-06-15 04:40 — PR #291: Verify-compute hideability (honest kernel-addressable floor) — 🟢 the step-side is DEFINITIVELY CLOSED: honest floor lands ON 487.7 (only 4.8% of verify-above-read compute is overlap-hideable); #283's optimistic all-hides 746.9 was never realizable — bank-the-analysis (0 TPS)

- **Branch:** `denken/verify-compute-hideability` · **Student:** denken · merged 04:40:10Z by advisor (Morgan; LOCAL GPU verify-kernel CUDA-event profiling + CPU analytic, no served-file change; BASELINE 481.53, 0 TPS). W&B [`3myn1fzl`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/3myn1fzl) (canonical, fresh GPU probe) + [`myttnvah`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/myttnvah) (analytic replicate, identical floor/fraction). Advisor (Morgan) W&B-verified bit-identical: self_test=1, tps_kernel_floor_honest=487.7289, hideable_frac=0.0482, tps_all_compute_hides=746.91, free_lane=0, nan_clean=1.
- **Primary:** `verify_compute_hideability_self_test_passes=1` (7/7 a–g). **Test:** `tps_kernel_floor_honest=487.7289`.
- **Key finding:** refines denken's own #283 optimistic `tps_kernel_floor=746.9` (which assumed ALL 2104.6µs verify-above-read compute hides). Measured: of the 2104.6µs, only **4.8% (101.5µs wall = 15.48µs normalized × φ_WS) is greedy-SAFE overlap-hideable** — and it is EXACTLY the one bit-identical lever (SDPA num_stages 3→2) wirbel #285 already found; the other **95.2% (2003µs) is exposed/serial** (irreducible non-body memory + near-roofline int4 MLP/GEMV that is only retunable by reassociation = greedy-UNSAFE). So the honest kernel-addressable floor lands **ON 487.7289 TPS** (`coincides_with_free_lossless=True`, Δ<0.05 vs wirbel #285), NOT above it. **`free_lane_to_500_exists=FALSE`** (reaching 500 needs a 2.9× larger hideable lever than the only greedy-safe one — no candidate exists). The 487.7↔746.9 gap was a **basis artifact** (the composition-compression ratio φ_WS = W/S = 7982.9/1218.2 = **6.5530** maps normalized-step ↔ honest-wall TPS, both round-trip 481.53) compounded by #283's optimistic all-hides assumption — `basis_reconciled=True`. Verify-side bridge≈1.0 carried (kanna #286), NOT the draft-side 0.21. GPU-grounded: fresh `--gpu-probe` read floor 3412.99µs/515.76 GB/s/86.0% nominal; verify-above-read CUDA-event resid 0.0µs vs imported.
- **Conclusion — the step-side is DEFINITIVELY CLOSED at 487.7:** denken's #283 746.9 was never realizable; 487.7 is the true FREE (non-build, greedy-safe) step ceiling, BELOW 500. This is the honest measured replacement for the optimistic all-hides floor and the capstone of the step-side audit — it reinforces fern #281 (Path-A closed) from the denominator side: there is NO free lane to ≥500, so the human-gated E[T]-raise BUILD is the sole >500 path. Reseated denken #297 → Tail-resolved per-position (does the hard-prompt acceptance cliff SHIFT? — the per-prompt per-position remeasure kanna #289 flagged as a caveat). Orthogonal to ubel #284 (the host/other 2109µs = the OTHER 26% block).

## 2026-06-15 04:34 — PR #292: SAM-Decoding retrieval acceptance-lift — 🟢 the answer to "can we reach 500 without a training gate?" is NO: SAM is a +2–4% UNGATED COMPANION, residual 0.902 E[T] for EAGLE-3 — bank-the-analysis (0 TPS)

- **Branch:** `lawine/sam-decoding-acceptance-lift` · **Student:** lawine · merged 04:34:49Z by advisor (senpai; CPU analytic over banked W&B numbers + a 128-prompt suffix-recurrence measurement, no served-file change; BASELINE 481.53, 0 TPS). W&B [`3sqnkveo`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/3sqnkveo) (group `sam-decoding-acceptance-lift`, finished, NaN-clean). Independently W&B-verified by advisor (all metrics MATCH: self_test=True, hit_rate_n3=0.16124, longest_suffix_hit=0.2037, lift band [+2%,+4%], lifted E[T] [3.921,3.998], residual 0.902, pearson 0.32576, per-prompt join 128/128).
- **Primary:** `sam_decoding_acceptance_lift_self_test_passes=1`. **Test:** `sam_et_lift_pct_upper=4.0`.
- **Key finding:** SAM-Decoding (Hu et al. 2024, arXiv:2411.10666 — suffix-automaton retrieval drafting) is a **+2–4% UNGATED COMPANION, NOT a standalone path to 500**. Measured prompt-side suffix-recurrence hit_rate(n=3)=**0.16124** (longest-suffix 0.2037); applying the literature lift band gives lifted E[T] ∈ **[3.921, 3.998]** — far below the 4.90 step-banked target (wirbel #290) — leaving a **residual 0.902 E[T]** that only a gated drafter (EAGLE-3) can cover. **Decisive low-tail finding:** retrieval lands on the prompts that are ALREADY fast — high-E[T] decile hit 0.170 vs low-E[T] decile 0.007, `pearson(E[T],hit_rate)=+0.32576` — so SAM's hits are largely REDUNDANT with what the deployed linear K=7 drafter already accepts; it does NOT rescue the slow tail where the headroom lives. Greedy-safe (emission = verify argmax), PPL-pinned, training-free but a served-change (still gated). Student made a sound analytical deviation (anchored on the literature [+2%,+4%] band, treating the naive prompt-only 14.5% recurrence as a diagnostic CEILING that double-counts linear-drafter wins) — advisor AFFIRMED (do NOT revert; the +0.326 correlation is the proof of redundancy).
- **Conclusion — there is NO training-free standalone path to 500:** SAM banks as a free companion lever on TOP of a gated raise, not instead of it. The residual 0.902 E[T] hands the load-bearing build decision to EAGLE-3. Reseated lawine #296 → SAM × EAGLE-3 companion-stacking additivity (does the +2–4% companion SURVIVE under a better drafter, or does EAGLE-3 absorb the same recurrence substrate, shrinking SAM a second time and RAISING the honest residual above 0.902?). Pairs with wirbel #293's EAGLE-3 step-overhead — together they price the full honest cost of the gated raise.

## 2026-06-15 04:31 — PR #293: EAGLE-3 drafter step-overhead — 🟢 the heavier fusion drafter RAISES the BUILT-raise target 4.9029→6.1245 (eats the free lever 19.4×); window holds but TIGHT — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/eagle3-step-overhead` · **Student:** wirbel · merged 04:31:27Z by advisor (Morgan; CPU analytic over banked W&B numbers, no served-file change; BASELINE 481.53, 0 TPS). Merge commit `d1866c5`. W&B [`abhoog1x`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/abhoog1x) (group `eagle3-step-overhead`, finished, NaN-clean, self-test 8/8).
- **Primary:** `eagle3_step_overhead_self_test_passes=1` (8/8). **Test:** `eagle3_corrected_target=6.1245`.
- **Key finding:** re-banks wirbel #290's step-banked target 4.9029 against the HEAVIER EAGLE-3 fusion drafter. Model `eagle3_draft = m_fuse × linear_draft` (g_draft_frac=0.12459, linear_draft=149.84µs, verify_step=1052.88µs UNCHANGED by the drafter swap — EAGLE-3 changes the DRAFT forward, not verify); at L_fuse=3/m_fuse=3 the eagle3_step rises to 1502.4µs, lifting the corrected public-E[T] target to **6.1245** (band [5.80, 6.12]). That **eats the 0.0631 free lossless lever 19.4×** (`eagle3_step_eats_free_lever=True`, Δtarget +1.2217) and pushes the target **1.16 ABOVE fern #281's 4.966**. The feasibility window still holds at all m_fuse∈{2,3,4,6} but is TIGHT at m_fuse=6 (7.957 < E_T_max 8.0, 99% of the cap→ceiling headroom eaten). **HONEST CAVEAT (student-flagged):** `m_fuse × linear_draft` is a CONSERVATIVE UPPER model — it treats the L_fuse=3 feature fusion as 3 FULL draft forwards, but EAGLE-3's drafter is ONE forward ingesting a fused feature (the {2,21,39} hidden states combined by a single input-side projection); the true marginal is the fusion projection, NOT 3 forwards, so the architecturally-honest target is likely closer to ~5.0. Needs an A10G profile to pin.
- **Conclusion — de-optimism on the gated raise:** the EAGLE-3 retrain's true E[T] bar is HIGHER than the linear-family 4.90 once you charge for the drafter's own step cost — but how much higher (alarming 6.12 vs plausible ~5.0) hinges on the fusion-cost model. Reseated wirbel #295 (Morgan) → EAGLE-3 fusion-drafter step profile (collapse the 6.12 band to the architecturally-honest value). Pairs with lawine #292's SAM companion floor: #293 prices the gated raise's STEP cost, #292 prices what a free companion can shave off the E[T] it must reach.

## 2026-06-15 04:11 — PR #289: Per-position acceptance decay — 🟢 cliff at POSITION 1 (45.7% of E[T] loss); feasibility asymmetry (deep-lift feasible, a_1-only ceiling-bound) ⇒ BUILT raise requires non-linear drafter — bank-the-analysis (0 TPS)

- **Branch:** `kanna/per-position-acceptance-decay` · **Student:** kanna · merged 04:11:48Z by advisor (senpai; CPU analytic over banked W&B numbers, no served-file change; BASELINE 481.53, 0 TPS). W&B [`fi34s269`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/fi34s269) (group `per-position-acceptance-decay`, finished, NaN-clean, self-test 12/12). Independently W&B-verified by advisor (all metrics MATCH: self_test=True, acceptance_cliff_position=1, a_k array to 4 sig fig, E_T_decomposed=3.8512, both feasibility booleans True, position-1 loss fraction 45.7%). Merge commit `a1e9cb5d`.
- **Primary:** `per_position_acceptance_decay_self_test_passes=1` (12/12). **Test:** `acceptance_cliff_position=1`.
- **Key finding:** decomposed deployed E[T]=3.844 into the per-position conditional-acceptance chain via lawine #282's whole-run survival counter: E[T] = 1 + Σ G(m) = 1 + 48684/17075 = **3.85119** (reproduces the 3.844 anchor to +0.0072 / 0.18% noise). Conditional `a_k=[0.729,0.760,0.793,0.823,0.835,0.836,0.846]` — the acceptance cliff is at **POSITION 1** by all three measures (max token-loss, min conditional a_k, max absolute survival drop — they agree): position 1 alone forfeits **1.895 expected tokens = 45.7%** of the total 4.149 loss. Conditional acceptance RISES down the chain (0.729→0.846 = **survivorship**: contexts that clear the first token are the "easy" ones); the well-known marginal "j≥2 collapse" (G(k) 0.73→0.21) is the COMPOUNDING consequence of the position-1 bottleneck, not independent deep-position failure. The first draft token is both the weakest link AND the highest-leverage (a_1 multiplies all 7 survival terms). Round-trips: `cumprod(a_k)=G(1..7)` exactly; Σ token-loss = 4.149 = 8−E[T].
- **Key finding 2 — feasibility asymmetry pricing fern #281's 4.966:** lifting the **deep positions alone is FEASIBLE** (a_2..a_7→0.914 flat reaches the target; equivalently a +0.0797/position uniform lift), but lifting the **first token alone is NOT** — even perfect a_1=1 gives only E[T]=**4.910 < 4.966** (ceiling-bound). `feasibility_asymmetry_deep_yes_a1_no=True`. The deployed E[T]=3.851 sits AT the linear cap (denken #119: 3.8445, within noise) → linear path saturated → **`built_raise_requires_nonlinear_drafter=True`**. Reconciles independently: honest-500 floor 3.9914 / private 0.804 = 4.964 ≈ fern's 4.966 (resid −0.002). Tail structure: low-quartile pooled E[T]=3.093 (α=0.694) vs top 5.052 (α=0.863); `low_tail_cliff_position = top_tail_cliff_position = 1` under the whole-run shape-transfer assumption (honest caveat: a genuine tail-specific cliff-SHIFT would need a per-prompt per-position remeasure — lawine #282 banked per-prompt means only).
- **Conclusion — the BUILT-raise target now has an exact per-position spec:** lift j≥2 conditional acceptance to ≈0.91 while keeping a_1≥0.73 — the recoverable headroom lives in the DEEP positions (j≥2), the first token is ceiling-bound. Mechanism-matched to EAGLE-3's multi-layer hidden-state fidelity (attacks deep-position OOD acceptance collapse, ubel #263's priv/pub 0.804). The WITHIN-CHAIN acceptance-SHAPE complement to wirbel #290's transfer-factor budget (localizes WHERE the 1.0584 E[T] budget lives) and the numerator counterpart to denken #283's open verify front. Reseated kanna #294 (Morgan) → EAGLE-3 Phase-1 viability gate (the cheap-proxy GO threshold).

## 2026-06-15 04:04 — PR #290: Step-banked BUILT-raise target + EAGLE-3 feasibility bracket — 🟢 honest target 4.9029 (free lever banked); budget 1.0584 E[T] off the linear family; feasibility window holds — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/eagle3-feasibility-bracket` · **Student:** wirbel · merged 04:04:40Z by advisor (senpai; CPU analytic over banked W&B numbers, no served-file change; BASELINE 481.53, 0 TPS). W&B [`ub3kpsso`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/ub3kpsso) (group `eagle3-feasibility-bracket`, finished, NaN-clean, self-test 9/9). Independently W&B-verified by advisor (all 6 checks MATCH: self_test=1, step_banked_target=4.90288, recoverable_budget=1.05838, finished, 9/9 selftest keys, nan_clean).
- **Primary:** `eagle3_feasibility_bracket_self_test_passes=1` (9/9). **Test:** `step_banked_built_raise_target=4.9029`.
- **Key finding:** banking wirbel's own #285 free lossless step lever (new_step 1202.717µs, bridge=1.0 basis-honest per kanna #286) into fern #281's 4.966 public-E[T] floor RELAXES the BUILT-raise target to **4.9029** (= 4.966 × 1202.717/1218.2), shaving **Δtarget=0.0631** off what EAGLE-3 must deliver. The load-bearing STRUCTURAL finding: the deployed linear drafter sits AT denken #119's linear cap (E[T]=3.844 vs cap 3.8445, headroom 0.0005) → the entire **`eagle3_recoverable_budget_et=1.0584`** E[T] budget is OFF the linear family (no linear-chain retrain at any capacity recovers any of it). The target is reachable in principle (`built_raise_target_within_feasibility_window=True`: 3.8445 < 4.9029 < E_T_max 8.0; only 25.5% of the cap→ceiling headroom) but ONLY by a structurally non-linear drafter (`budget_requires_nonlinear_drafter=True`). EAGLE-3's multi-layer hidden-state fusion (target layers {2,21,39}) is mechanism-matched: it attacks the j≥2 OOD acceptance collapse (ubel #263, priv/pub 0.804) that BOUNDS the linear cap.
- **Conclusion — the honest step-banked aggregate target is settled:** 4.9029 public E[T] (budget +1.0584 beyond the linear cap), inside the feasibility window but recoverable only by a non-linear drafter — a NECESSARY-condition de-risk for the human-gated EAGLE-3 retrain; `eagle3_sufficiency_is_build_gated=True` (does NOT claim EAGLE-3 reaches the target). Pairs with kanna #289's per-position a_k cliff (WHERE the budget lives) for full budget localization; confirms denken #283's open verify front (binding constraint = E[T], not read/kernel). Reseated wirbel #293 (advisor) → EAGLE-3 drafter step-overhead (re-bank the 4.9029 target for the HEAVIER fusion drafter's draft-step; does it eat the 0.0631 free-lever relaxation?).

## 2026-06-15 03:58 — PR #288: PPL local→official transfer (τ_ppl) — 🟢 τ_ppl=1.000218, safe local PPL bar 2.4185; the transfer trinity (τ_lo/τ_acc/τ_ppl) is COMPLETE — bank-the-analysis (0 TPS)

- **Branch:** `lawine/ppl-local-official-transfer` · **Student:** lawine · merged 03:58:09Z by advisor (senpai; PPL-harness measurement on the official-corpus mirror, no served-file change; BASELINE 481.53, 0 TPS). W&B [`i1e5054m`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/i1e5054m) (group `ppl-local-official-transfer`, finished, NaN-clean, self-test passes). Independently W&B-verified by advisor (all values MATCH exactly).
- **Primary:** `ppl_local_official_transfer_self_test_passes=1`. **Test:** `tau_ppl=1.000217619920667`.
- **Key finding:** τ_ppl = **1.000217619920667** (±0.000210) — local int4 PPL is a near-perfect (offset <0.022%) proxy for official PPL. `local_ppl_int4_deployed=2.376682786480556` (measured on the official-corpus mirror, NOT a synthetic proxy) vs official 2.3772 → the PPL gate transfers 1:1. Safe local PPL bar = **2.41846** (build to local PPL ≤ 2.4185 to clear the official 2.42 gate under worst-case τ_ppl jitter); `gate_meaningful_local_ppl_headroom=0.04177` (97.6% of the official 0.0428 headroom is preserved locally). Unlike τ_lo=1.0352 (TPS is a rate with a systematic +3.5% clock offset), PPL is a corpus likelihood → no systematic offset, hence τ_ppl≈1.
- **Conclusion — the local→official transfer TRINITY is COMPLETE:** τ_lo (TPS), τ_acc=1.0 (acceptance, lawine #276), and now τ_ppl=1.000218 (PPL) — all three local→official transfer factors are pinned. Any future built lever (EAGLE-3, SAM-Decoding) can be screened LOCALLY with known-safe bars (local λ̂≥0.9855, local PPL≤2.4185) and trusted to transfer to official. Arms fern #287's read-reduction PPL Pareto (the safe-PPL ceiling for body-read cuts) and lawine's own #292 SAM-Decoding "zero PPL risk" claim. Reseated lawine #292 (advisor) → SAM-Decoding retrieval acceptance-lift (the one training-free forward E[T] lever).

## 2026-06-15 03:49 — PR #283: HBM-bound TPS ceiling — 🟢 the system is NOT read-bound (read=38% of honest wall); verify front OPEN; build is the sole >500 path — bank-the-analysis (0 TPS)

- **Branch:** `denken/hbm-bound-tps-ceiling` · **Student:** denken · merged 03:49:45Z by advisor (senpai; LOCAL GPU read-bench + CPU analytic, no served-file change; BASELINE 481.53, 0 TPS). W&B [`vmxuwxm0`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/vmxuwxm0) (group `hbm-bound-ceiling`, finished, NaN-clean, self-test 6/6). Independently W&B-verified by advisor (all 9 checks MATCH). Merge commit `b8fcd79`.
- **Primary:** `hbm_bound_ceiling_self_test_passes=1` (6/6). **Test:** `official_tps_as_frac_of_hbm_ceiling=0.3805`.
- **Key finding — the HBM-bound premise REFUTED:** the HBM-bandwidth-bound ceiling is **1265.64 TPS** (official), so deployed 481.53 sits at **0.3805** of it. Unit-consistent computation overturns #278's "floor (2933.8µs) > 1218.2µs step ⇒ HBM-bound" reading: the **1218.2µs step is a ~6.5× compressed NORMALIZED composition unit**, and the honest per-step wall is `1/K_cal = 7982.9µs`. On that honest wall the int4 body read (3037.2µs official) is only **38.0%** — NOT the dominant cost. The 62% non-read slack = draft 731.8µs (9%) + **verify-compute-above-read 2104.6µs (26%, kernel-addressable → kanna #280)** + host/other 2109.3µs (26% → ubel #284). `kernel_timing_clears_500=True` — bare-read-floor ceiling 1265.6 and even the all-kernel-addressable-hides floor `tps_kernel_floor=746.9` both ≫ 500. Levers: `et_for_500_at_read_floor=1.519` (ceiling already >500 ⇒ tokens-per-read non-binding ON THE CEILING; the DEPLOYED POINT reaches 500 at **E[T]≥3.9914**, independently reproducing fern #281 to resid 0.00044); `read_reduction_pct_for_500_at_fixed_et=−153%` (NEGATIVE ⇒ bytes-per-read non-binding, ceiling already >500). GPU read-bench grounding: measured **513 GB/s = 85.5%** of nominal 600 on a 1.76 GB reduction → even at achievable BW the ceiling is 1082 TPS (≫500), verdict robust. **Bug-fix accepted:** completed the documented reproduce venv with the already-declared `wandb==0.27.2` (pyproject/uv.lock dep, not new) + hardened `_maybe_log_wandb` import order so the gitignored `./wandb` output dir can't shadow the package — tooling fix, no served-file change.
- **Conclusion — the verify-side front is OPEN, not read-capped:** "floor > normalized-step" measured composition COMPRESSION (re-proving #278's 4.818× step-shaving over-credit), NOT wall read-boundedness. Body-read-reduction (fern #287 / lawine #288) is the WRONG lever for the deployed point (ceiling already >500). The path to a MEASURED ≥500 build = **E[T]-raise (deployed point→500 at E[T]≥3.991, human-gated drafter) + closing the 62% non-read slack** (hideable verify compute kanna #280 + host overhead ubel #284) — not read-floor-limited. Refines/reframes denken's own #278 (the bridge stays correct; the "HBM-bound" label was the normalized-step-as-wall confusion). Reseated denken #291 (advisor) → verify-compute overlap-hideability (measure how much of the 2104.6µs actually hides; reconcile the 487.7↔746.9 floors into one honest kernel-addressable floor).

## 2026-06-15 03:40 — PR #285: Lossless micro-lever envelope — 🟢 the FREE step-side ceiling is 487.7 TPS; lossless axis CLOSED — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/lossless-micro-lever-envelope` · **Student:** wirbel · merged 03:40:43Z by advisor (morganmcg1; LOCAL GPU micro-profiling, ONEGRAPH-faithful, no served-file change; BASELINE 481.53, 0 TPS). W&B [`97b57hhe`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/97b57hhe) (group `lossless-micro-lever-envelope`, finished, NaN-clean, self-test 7/7). Independently W&B-verified by advisor (all 10 checks MATCH).
- **Primary:** `lossless_micro_lever_envelope_self_test_passes=1` (7/7). **Test:** `envelope_tps_gain_pct=1.2873`.
- **Key finding:** the TOTAL greedy-safe lossless step-shaving envelope = **15.483µs** (SDPA 15.48 + lm_head 0.00 + norms 0.00) → **+1.29% → 487.729 TPS**; `envelope_clears_500=False`; residual gap to 500 = **+2.516% / 29.52µs**. The four-lever stack collapses to ONE incremental free lever — **SDPA num_stages 3→2** (global h512 ×1.0194 + sliding h256 ×1.0885); lm_head (0.66µs fused-epilogue ceiling, FUSED_SPARSE_ARGMAX already on-GPU) and norms (ONEGRAPH + vLLM fused add+rmsnorm; kanna #280 in-graph remainder 15.44µs has no foldable kernel) are `already_captured` in the deployed baseline. Bit-identical **0/128, maxdiff=0.0**, ppl=2.3772, nan_clean. `sliding_only_captures_pct=87.04%` (lower-blast-radius variant). `basis_flag.bridge=1.0` (verify-side, deployed M=8 — NO discount), independently confirmed by kanna #286.
- **Conclusion — the FREE step-side ceiling is 487.7 TPS; lossless axis CLOSED as a research lane:** even stacking every greedy-safe bit-identical micro-lever, the composed envelope cannot move 481.53 → 500. The residual 29.5µs gap is structurally OFF the lossless step axis — it lives on the E[T]-raise axis (fern #281), which trades acceptance, not float order. Accepting wirbel's follow-up: step-shaving CLOSED, all remaining 481→500 effort routes to E[T]/coverage. Reseated wirbel #290 → step-banked BUILT-raise target + EAGLE-3 feasibility bracket.

## 2026-06-15 03:35 — PR #286: Bridge basis-honesty lever card — 🟢 the bridge is DRAFT-SIDE-SPECIFIC (0.21); verify-side=1.0; composed basis-honest stack 493.64 (still 6.36 short) — bank-the-analysis (0 TPS)

- **Branch:** `kanna/bridge-repricing-lever-card` · **Student:** kanna · merged 03:35:32Z by advisor (senpai; CPU analytic over banked W&B numbers, no served-file change; BASELINE 481.53, 0 TPS). W&B [`0k4azmjo`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/0k4azmjo) (group `bridge-repricing-lever-card`, finished, NaN-clean, self-test passes). Independently W&B-verified by advisor (all 6 checks MATCH).
- **Primary:** `bridge_repricing_lever_card_self_test_passes=1`. **Test:** `best_basis_honest_step_tps=487.758`.
- **Key finding:** the composition bridge (denken #278) is **DRAFT-SIDE-SPECIFIC**. draft-side bridge = **0.2147** (4.66× over-credit — a batch=1 WALL draft saving subtracted from the batch-amortized normalized step over-credits 4.66×); verify-side bridge = **1.0** (deployed M=8 IS the normalized basis — no discount). Best single basis-honest lever = verify SDPA num_stages #279 = **487.758**; full composed disjoint stack (SDPA + lm_head + norms) = **493.637**, `composed_step_stack_crosses_500=0`, `gap_to_500=6.36`.
- **Conclusion — step-side closed at BOTH raw AND basis-honest level:** resolves denken #278's open bridge ambiguity — the 0.21 over-credit applies ONLY to draft-side batch=1 wall savings; every verify-side lever (SDPA, lm_head, norms) sits at bridge=1.0, so wirbel #285's lossless envelope needs NO discount (raw == basis-honest). The composed basis-honest stack tops out at 493.64, still 6.36 short of 500. This is the E[T]-numerator complement to fern #281's capstone: the step-side denominator is settled; the sole forward >500 path is a BUILT public-E[T] raise to ≥4.97 (denken #119 linear cap 3.8445). Reseated kanna #289 → per-position acceptance decay (the BUILT-raise target profile).

## 2026-06-15 03:20 — PR #282: E[T] per-prompt distribution (128 prompts) — ⚪ moderate variance, public gap +0.140, headroom is in the low tail (BUILT raise only) — bank-the-analysis (0 TPS)

- **Branch:** `lawine/et-prompt-distribution` · **Student:** lawine · merged 03:20:51Z by advisor (morganmcg1; CPU analytic over banked run logs, no served-file change; BASELINE 481.53, 0 TPS). W&B [`2j0e8xgg`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/2j0e8xgg) (group `et-prompt-distribution`, finished, NaN-clean, self-test 8/8).
- **Primary:** `et_prompt_distribution_self_test_passes=1` (8/8). **Test:** `et_std=0.8013`, `et_gap=0.1403`.
- **Key finding:** per-prompt E[T] on 128/128 prompts (0 non-finite): token-weighted aggregate **et_mean=3.8512** (reproduces kanna's 3.844 to +0.0072), et_arith_mean=3.9982, et_median=3.9088, percentiles p5/p25/p75/p95 = 2.930/3.415/4.360/5.582, **et_std=0.8013** (moderate variance, single hump ~3.9, left tail to 2.5 / right tail to 6.4). et_min = prompt **idx 85** (2.535), et_max = prompt **idx 37** (6.370). **The PUBLIC E[T] gap to 500 is +0.140 (need 3.991) — the smallest gap of any axis** — and the headroom is demonstrably present (top-quartile prompts already run at E[T]≥4.36). Honesty caveats (load-bearing): (1) the linear price K_cal·E[T] OVERSTATES above E[T]≈4.16 (crosses the 520.95 λ=1 served ceiling there) — et_p75_tps=546/et_ceiling_tps=699 are linear-unit headroom, NOT deliverable rate; the physically-bounded counterfactual (bottom-quartile→median lift) = **515.93 TPS** (clears 500 by ~16, but needs the WHOLE low tail lifted to median). (2) **No free prompt-side lever** — low-E[T] prompts don't cluster by length/code; the benchmark prompts are fixed.
- **Conclusion — the E[T] axis is characterized:** moderate-variance, smallest gap of any axis (+0.140 public), but the only realizable path to 500 is a **BUILT public-E[T] raise** (better acceptance on the low-E[T] tail). DOVETAILS with fern #281: there is no free lever; the raise must be built. Reseated lawine #288 (Morgan) → PPL local→official transfer.

## 2026-06-15 03:11 — PR #281: E[T]-raise feasibility — 🟢 CAPSTONE: Path-A CLOSED on all three axes; only re-open is a BUILT public-E[T] raise to ≥4.97 — bank-the-analysis (0 TPS)

- **Branch:** `fern/et-raise-feasibility` · **Student:** fern · merged 03:11:39Z by advisor (morganmcg1; composition-priced feasibility integration, no served-file change; BASELINE 481.53, 0 TPS). W&B [`10necg21`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/10necg21) (group `et-raise-feasibility`, finished, NaN-clean across 41 keys, self-test 7/7).
- **Primary:** `et_raise_feasibility_self_test_passes=1` (7/7). **Test:** `path_a_fully_closed=1`.
- **Key finding — THE CAPSTONE CLOSURE:** **Path-A is CLOSED on all three axes** (`path_a_fully_closed=True`). No realizable `(E[T]_real, M, step-shave)` point reaches the honest-500 floor — the deployed 481.53 frontier cannot reach 500 by any speculative-decoding lever under measured constraints. `path_a_axes_closed = {draft_cut: True (fern #274), width: True (denken #271 M*=32=479.6), et_raise: True}`. Numbers: `min_et_real_needed` (at max measured step-shave 5.58%, φ_hi) = **3.8342**; `best_achievable_et_real` (private-0.804-degraded, M=8) = **3.0898**; gap = **−0.7444 → CLOSED**; `go_region_exists=False` (0 realizable cells; 3 counterfactual only @ priv=1.0). Public E[T] needed @ deployed step (priv 0.804) = **4.966** (measured public ceiling 3.844 → gap **+1.12**). Canonical private factor going forward: **0.804** (ubel #263 `2khp8gzs`, swept also 0.65/0.73 — verdict holds at every value).
- **Conclusion — the analytic Path-A is fully closed:** the ONLY re-open fern leaves is a **BUILT public-E[T] raise to ≥4.97 at the deployed step — acceptance-per-candidate, NOT width** (denken #119: the LINEAR drafter caps at 3.8445 at perfect capacity, so the +1.12 raise is unreachable by the current drafter → requires a structurally non-linear/feature-conditioned drafter). This is the plateau-protocol pivot: kernel/tree/step/draft-cut levers are exhausted; the sole >500 path is a trained better drafter (researcher-agent RANK 2 EAGLE-3 multi-layer hidden-state fidelity is the prime candidate — attacks the j≥2 OOD acceptance collapse directly). land #245's tree build (Morgan banking) is the only vehicle to deliver+measure such a built raise. Reseated fern #287 (Morgan) → read-reduction PPL pareto.

## 2026-06-15 03:01 — PR #280: Verify-step component roofline — ⚪ verify is HBM-bound (MLP 66%); only num_stages=2 SDPA +1.185% is greedy-safe — bank-the-analysis (0 TPS)

- **Branch:** `kanna/verify-step-component-roofline` · **Student:** kanna · merged 03:01Z by advisor (senpai; LOCAL GPU micro-profiling, no served-file change; BASELINE 481.53, 0 TPS). W&B [`sdrerk5h`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/sdrerk5h) (group `verify-step-roofline`, finished, NaN-clean, self-test passes).
- **Primary:** `verify_step_component_roofline_self_test_passes=1`. **Test:** `max_verify_side_step_shaving_tps_gain_pct=1.185`.
- **Key finding:** verify forward (88% of the linear step) decomposes at M=8 into **MLP 66.1%** (gate_up 43.0% @ 71.2% BW + down 23.1% @ 66.5% BW) / **SDPA 14.5%** @ 34.9% BW / io+attn 17.0% / lm_head 2.4% @ 83.4% BW. The M≥8 batching lifts the int4-Marlin GEMMs to **59–71% BW → approaching-roofline-intrinsic** — the MLP slack is reassociation-gated (greedy-UNSAFE, touches bf16 K-reduction order). The ONLY greedy-safe verify-side lever is num_stages=2 on the 14.5%-share SDPA = **+1.185% composition** (corroborates wirbel #279 from the component side). lm_head efficiency falls 83.4% (M=8) → 60.5% (M=32), vocab output-write bound — a fused/tiled epilogue is a clean lossless follow-up (handed to wirbel #285).
- **Conclusion — verify decomposition CLOSED:** the verify forward is HBM-bound MLP-dominated; no greedy-safe verify lever beyond the SDPA num_stages tune clears anything material. Confirms the step-side has no standalone >500 lever on the verify path either. Reseated kanna #286 → bridge basis-honesty re-pricing card (classify every banked step lever draft-side vs verify-side, re-price to basis-honest TPS).

## 2026-06-15 03:00 — PR #279: Verify SDPA linear-deploy num_stages transfer — ⚪ +1.29% (487.8 TPS), CLOSES last standalone step candidate — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/verify-sdpa-linear-deploy` · **Student:** wirbel · merged 03:00Z by advisor (senpai; LOCAL GPU micro-profiling, no served-file change; BASELINE 481.53, 0 TPS). W&B [`xme9snkv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/xme9snkv) (group `verify-sdpa-linear-deploy`, finished, NaN-clean, self-test 9/9).
- **Primary:** `verify_sdpa_linear_deploy_self_test_passes=1` (9/9). **Test:** `projected_tps_gain_pct=1.29` (487.8 TPS).
- **Key finding:** verify SDPA num_stages=3→2 → **+1.29% realistic-ctx (487.8 TPS)**; does NOT clear 500 even at inflated ctx=2048 (497.7). **Premise correction:** the served config is **MAX_NUM_SEQS=1 (NOT 8 concurrent) + SPLITKV_VERIFY**, which routes the M=8 verify to the **3D split-KV TILE=16** path (not #270's 2D TILE=32). Global head-512 collapses 1.192×→1.018× (already near-roofline); sliding head-256 retains 1.093×. **Bit-identical 0/128 (maxdiff=0.0)** — greedy-safe composable micro-lever. The 3D slack lives in the 14 sliding head-256 layers, not global.
- **Conclusion — last standalone >500 step candidate CLOSED:** verify SDPA tune is real (+1.29%) but insufficient standalone; banked as a greedy-safe micro-lever to stack. The step-side is now fully mapped: no standalone step lever (draft OR verify) crosses 500. Reseated wirbel #285 → lossless micro-lever envelope (compose all greedy-safe bit-identical step levers — SDPA + sliding-only + lm_head epilogue + norm folds).

## 2026-06-15 02:59 — PR #278: Deployed linear-step decomposition — 🟢 the 1218µs step is NORMALIZED (not a wall sum); composition over-credits draft savings 4.82× — bank-the-analysis (0 TPS)

- **Branch:** `denken/linear-step-decomposition` · **Student:** denken · merged 02:59Z by advisor (morganmcg1; ONEGRAPH-faithful CUDA-event measurement, no served-file change; BASELINE 481.53, 0 TPS). W&B [`bu44n30q`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/bu44n30q) (group `linear-step-decomposition`, finished, NaN-clean, self-test 6/6).
- **Primary:** `linear_step_decomposition_self_test_passes=1` (6/6). **Test:** `composition_is_honest=0`, `step_overhead_us=−4455.4`, `bridge=0.2147`.
- **Key finding:** the deployed M=1 linear verify = **4966.8µs** (CUDA-event, int4-body HBM-bound) — ~10× the assumed 511µs residual, 4.08× the whole 1218.2µs step. `step − draft − verify = −4455.4µs` (NEGATIVE/unphysical) → **the 1218.2µs step is a NORMALIZED (batch-amortized) composition unit, NOT a wall draft+verify+overhead sum.** HBM floor: one int4 forward MUST read 1.76 GB = 2934µs > whole step (`floor_exceeds_whole_step=True`). **bridge = step_norm/step_wall = 1218.2/5673.6 = 0.2147**; batch=8 verify 9036.7µs → per-seq 1129.6µs ≈ step_norm (b1→b8-per-seq amortization 4.40× ≈ 1/bridge=4.66×). **Consequence:** a batch=1 WALL draft saving subtracted directly from the normalized step **over-credits by 4.82×**. Re-prices kanna #269: +4.39% composition → **+0.91% basis-honest**.
- **Conclusion — composition honesty for the LINEAR path priced:** the LINEAR step is a normalized unit (unlike fern #274's tree-wall φ=0.603 mechanism — a different discount on a different path). All batch=1 wall draft savings carry the bridge≈0.21 over-credit; verify-side deployed-M=8 savings may carry bridge≈1.0 (open — handed to kanna #286). Reseated denken #283 (Morgan) → HBM intrinsic ceiling.

## 2026-06-15 02:59 — PR #275: Prefill / TPS-denominator slack — ⚪ prefill 2.85% of wall, denominator CLOSED — bank-the-analysis (0 TPS)

- **Branch:** `ubel/prefill-denominator-probe` · **Student:** ubel · merged 02:59Z by advisor (morganmcg1; LOCAL GPU measurement, no served-file change; BASELINE 481.53, 0 TPS). W&B [`s26cb1tv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/s26cb1tv) (group `prefill-denominator-probe`, finished, NaN-clean, self-test passes).
- **Primary:** `prefill_denominator_self_test_passes=1`. **Test:** `prefill_wall_share_pct=2.849`.
- **Key finding:** prefill = **2.849%** of wall at the official 512-token output point; the TPS denominator is CLOSED. Precache banks 1.65pp (82% prefix-cache hit on the 46-token preamble). The MTP drafter contributes **0 marginal prefill**. Recoverable band [0%, 2.85%] — even perfect prefill elimination yields <2.85% and the precache already captures most of it.
- **Conclusion — prefill denominator CLOSED:** prefill is not a material >500 lever; the wall is decode-dominated (97.15%). Reseated ubel #284 (Morgan) → decode-loop host overhead.

## 2026-06-15 02:33 — PR #276: Acceptance local→official transfer (τ_acc) — 🟢 τ_acc=1.0, local λ̂ = official proxy 1:1, safe bar=0.9855 — bank-the-analysis (0 TPS)

- **Branch:** `lawine/local-official-tps-transfer-acceptance` · **Student:** lawine · merged 02:33:02Z by advisor (senpai; acceptance-harness measurement, no served-file change; BASELINE 481.53, 0 TPS). W&B [`vcgtsl1c`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/vcgtsl1c) (group `acceptance-local-official-transfer`, finished, NaN-clean, all 7 sub-gates pass).
- **Primary:** `acceptance_local_official_transfer_self_test_passes=1` (7/7). **Test:** `tau_acc=1.0`.
- **Key finding:** τ_acc=1.0 (central) with ±0.0075 envelope — local λ̂ is a **kernel-invariant proxy** for official λ̂. Unlike TPS (a rate, systematic +3.5% hardware/clock offset → τ_lo=1.0352), λ̂ is a **probability** (per-token argmax agreement), so there is NO systematic clock-induced offset. Per-M divergence table: M1↔M8 (served linear) = 0.007292 (reproduces lawine #232 EXACTLY); M8↔M16 = 0.007200; M1↔M16 = 0.007537. Safe local bar = **0.985548** (build to local λ̂ ≥ 0.9855 to clear official 0.9780 under worst-case ±0.0075 jitter). **Architectural refine:** int4-Marlin body GEMMs are **bit-exact across M=1/8/16** (max_abs_diff=0.0, zero argmax batch-variance) — the entire ~0.73% jitter originates in bf16 lm_head + TRITON_ATTN accumulation. The launch consequence is favorable: local acceptance screening transfers 1:1 to official. Composition round-trip: K_cal(125.268) × E_T(3.844) = 481.53 resid=0 confirmed.
- **Conclusion — acceptance leg of the launch-gate transfer CLOSED:** parallel to #267's TPS leg (τ_lo=1.0352), this closes the acceptance leg at τ_acc=1.0. A local build must screen λ̂ ≥ 0.9855 (vs official bar 0.9780) to clear the acceptance gate. Reseated lawine #282 → E[T] per-prompt distribution on 128 competition prompts (E[T] variance and upside characterization).

## 2026-06-15 02:22 — PR #277: Draft io_projection roofline — 🟢 INTRINSIC-M=1, NULL; draft decomposition COMPLETE — bank-the-analysis (0 TPS)

- **Branch:** `kanna/draft-io-projection-roofline` · **Student:** kanna · merged 02:22:47Z by advisor (morganmcg1; LOCAL GPU micro-profiling + CPU analytic, no served-file change; BASELINE 481.53, 0 TPS). W&B [`ahw089yi`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/ahw089yi) (group `draft-io-projection-roofline`, finished, NaN-clean, self-test passes).
- **Primary:** `draft_io_projection_roofline_self_test_passes=1`. **Test:** `projected_tps_gain_pct=0.0`.
- **Key finding:** io_projection is **NOT q/k/v/o** (as the PR body assumed based on #264's labeling) — it is EXACTLY the **2 residual-stream GEMVs** (`pre_projection [256,5120]` at pass start, `post_projection [2560,256]` at pass end), both single GEMVs with NO companion kernel. Measured: `io_projection_measured_us=14.04µs` (recovers #264's 13.86µs to 1.3%); `io_bw_utilization=46.7%` (vs 80.9% for the 128 MiB anchor GEMV that saturates HBM). At 46.7%, the io GEMVs are **INTRINSIC-M=1**: the single-warp M=1 GEMV physically cannot fill the HBM bus (same regime as the MLP GEMVs at 41%). `recoverable_io_us=0.06µs → recoverable_material=False`. **CLOSES the draft-pass decomposition:** MLP (#269) + attention (#270) + io (this) = **93.27µs = 95.2%** of the 98µs GEMV chain — **ALL intrinsic-M=1**. The ONLY recoverable draft slack anywhere is the MLP's GeluAndMul fold (+7.32µs/pass, +4.39% ceiling, φ-discounted by fern #274 → honest +2.65%). The architectural relay from wirbel #270: kanna #264's "28.5µs attention" IS the q/o-proj GEMV, NOT the autotunable SDPA (~3.3µs real SDPA is immaterial on the draft path). The "io_projection vs q/k/v/o" premise was a misrecollection — io_projection is the backbone↔draft interface, not the attention projection chain.
- **Conclusion — draft decomposition CLOSED:** the bf16 draft floor is essentially ALL intrinsic-M=1. No recoverable draft slack beyond the MLP's GeluAndMul fold (which is φ-discounted to +2.65% honest by fern #274, below 500 standalone). The draft axis is closed as an independent 500 lever. Reseated kanna #280 → verify-step component roofline (decompose verify_us(M) to pre-price wirbel #279's SDPA speedup ceiling).

## 2026-06-15 02:20 — PR #274: Composition-overhead honesty (φ-correction) — 🟢 φ=0.603, ALL draft-pass-cut levers below 500, honest static-K=4=493.96 — bank-the-analysis (0 TPS)

- **Branch:** `fern/composition-overhead-honesty` · **Student:** fern · merged 02:20:13Z by advisor (senpai; CPU analytic + LOCAL GPU step measurement, no served-file change; BASELINE 481.53, 0 TPS). W&B [`brnmnl60`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/brnmnl60) (group `composition-overhead-honesty`, finished, NaN-clean, self-test passes).
- **Primary:** `composition_overhead_honesty_self_test_passes=1`. **Test:** `honest_corrected_static_k4_tps=493.957`.
- **Key finding:** **φ=0.603** (measured fraction of wall step that is variable/GPU-forward; the composition formula over-credits draft-pass cuts by 1/φ ≈ 1.66×). Honest TPS table: static-K=4: composition 502.12 → **honest 493.96 (below 500)**; static-K=5: 500.79 → **honest 493.15 (below 500)**; static-K=3/6: even worse. **ALL draft-pass-cut levers fall below 500 under honest composition.** The `any_draft_lever_clears_500_under_honest_composition` metric logged as two split keys: `any_draft_lever_clears_500_measured_gd=1` (refers to tree-width E[T]-raising at 539.50 honest under fern's model — but independently refuted by denken #271's empirical M*=32=479.6 TPS); `any_draft_lever_clears_500_assumed_gd=0`. Mechanistic origin of φ: the measured wall step (~9-10ms at conc=1) is dominated by fixed CPU/scheduling overhead (the bulk of the ~8ms wall step at conc=1 is NOT model-forward) — model-forward is only a fraction, so draft-pass cuts only save the model-forward fraction, not the fixed overhead. Advisory note: the φ=0.603 is specific to the tree-path wall step scenario; the LINEAR path (step=1218.2µs, of which draft=706.9µs = 58%) may have a different φ, potentially closer to 1.0 — denken #278 is measuring this directly.
- **Conclusion — the decisive analytic gate:** combined with denken #271 (g_d=0.0191, tree WIDTH=479.6<500), fern #274 closes the draft-STEP axis definitively. The ONE remaining >500 candidate is wirbel #279 (verify SDPA tune on the deployed LINEAR batch=8 path — a VERIFY-side lever NOT subject to φ-discount since it doesn't touch draft-pass count). Reseated fern #281 → E[T]-raise feasibility: three-axis Path-A closure verdict.

## 2026-06-15 02:10 — PR #271: Deployed g_d measurement (step-basis reconciliation) — 🟢 `deployed_gd=0.0191`, M*=32=479.6 TPS < 500, tree WIDTH definitively CLOSED — bank-the-analysis (0 TPS)

- **Branch:** `denken/gd-step-basis-reconcile` · **Student:** denken · merged ~02:10Z by advisor (morganmcg1; ONEGRAPH-faithful CUDA graph captures, no served-file change; BASELINE 481.53, 0 TPS). W&B [`9mlmaen3`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/9mlmaen3) (group `gd-step-basis-reconcile`, finished, NaN-clean, self-test passes).
- **Primary:** `gd_step_basis_reconcile_self_test_passes=1`. **Test:** `deployed_gd=0.0191`, `tps_at_Mstar32_deployed=479.57`.
- **Key finding:** **deployed_gd=0.0191 CI[0.01905,0.01907]** (vs the assumed 0.168 in the #268 fork). This collapses the #268 g_d fork to the **central NO-GO**: M*=32 → TPS=**479.6 < 500**, `width_clears_500_deployed=False`. Additional physics bound: HBM floor independently caps g_d_max=0.0348 (the assumed 0.168 is physically impossible — 4.8× faster than the HBM floor). Key decomposition: `draft_k7_chain_us_graphed=706.9µs`, `verify8_us=5357.8µs`, `verify32_us=6032.2µs`. The g_d pivot for 500 TPS is g_d=0.04018 — nearly 2× the HBM floor cap. Results files: `research/validity/gd_step_basis_reconcile/deployed_gd_measurement.json` + `gd_step_basis_reconcile_results.json`.
- **Conclusion — tree WIDTH axis definitively CLOSED:** no tree topology (shallow/full M*=32) can clear 500 on the deployed hardware. This is the INDEPENDENT EMPIRICAL closure of land #245's tree build: a live measurement cannot exceed 479.6 TPS at M*=32 given the measured g_d. Combined with the HBM floor argument (g_d_max=0.0348), the tree-width axis is physically AND empirically closed. The two remaining candidates after this closure: (1) draft-pass cuts (fern #274 will φ-discount these); (2) verify-SDPA tune on the linear path (wirbel #279). Reseated denken #278 → deployed linear-step decomposition (is the 511µs residual mostly M=1 verify or fixed overhead? — prices composition honesty for the LINEAR path).

## 2026-06-15 02:10 — PR #270: Draft/tree attention Triton autotune — 🟢 Draft SDPA NULL (~3.3µs immaterial); verify SDPA MISTUNED (1.097×/1.090×/1.092× at M=8/16/32, bit-identical) — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/draft-attn-autotune` · **Student:** wirbel · merged ~02:10Z by advisor (morganmcg1; LOCAL Triton patch + CUDA-event timing, no served-file change; BASELINE 481.53, 0 TPS). W&B [`iwwcmvez`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/iwwcmvez) (group `draft-attn-autotune`, finished, NaN-clean, self-test passes).
- **Primary:** `draft_attn_autotune_self_test_passes=1`. **Test (tree-verify):** `sdpa_kernel_speedup_M8=1.097`, `sdpa_kernel_speedup_M16=1.090`, `sdpa_kernel_speedup_M32=1.092` (bit-identical, maxdiff=0.0).
- **Key finding — TWO decisive results:**
  (1) **Draft SDPA = NULL**: the real in-graph draft SDPA is only **~3.3µs** (immaterial — essentially free). **ARCHITECTURAL CLARIFICATION**: kanna #264's "28.5µs attention" bucket is NOT the autotunable SDPA — it is the **q/o-proj GEMV** (a static ONEGRAPH-embedded linear layer). The actual Triton SDPA in the draft pass runs in ~3.3µs because M=1 flash attention on tiny queries degenerates to a nearly free operation.
  (2) **Verify SDPA MISTUNED**: the deployed Triton SDPA is bare `@triton.jit` (default `num_stages=3`, NOT shape-specialized, NOT autotuned). Applying `num_stages=3→2` gives **1.097×/1.090×/1.092×** speedup at M=8/16/32 tree-verify shapes — **bit-identical (maxdiff=0.0)**. `attn_is_shape_specialized=False`. BUT: these measurements were for the TREE-VERIFY path (M=8/16/32 tree candidates per sequence). The tree path is now closed (denken #271: M*=32=479.6 TPS<500).
- **Conclusion — ONE SURVIVING LEVER:** the verify SDPA mistuning (1.097× at M=8) ALSO applies to the DEPLOYED LINEAR batch=8 path (the target model SDPA processes 8 concurrent sequences simultaneously = same M=8 dimension as the tree measurement). This is a VERIFY-SIDE lever NOT subject to fern #274's φ-discount (which specifically discounts DRAFT-pass cuts) and NOT closed by the tree closure (the linear path is live). wirbel #279 measures whether this 1.097× tune transfers to the deployed LINEAR batch=8 path with full-verify-pass speedup and bit-identity over 128 prompts. Reseated wirbel #279 → verify SDPA linear deploy.

## 2026-06-15 01:57 — PR #269: Draft MLP roofline — 🟢 ABOVE roofline but INTRINSIC-M=1; GeluAndMul fold +4.39% composition ceiling (crosses 500 standalone) — bank-the-analysis (0 TPS)

- **Branch:** `kanna/draft-mlp-roofline` · **Student:** kanna · merged 01:56:55Z by advisor (morganmcg1; analysis-only, no served-file change, BASELINE 481.53, 0 TPS). W&B [`epl52mkq`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/epl52mkq) (group `draft-mlp-roofline`, finished, NaN-clean, all self-tests pass) — W&B-verified independently (all 11 metrics confirmed exactly).
- **Primary:** `draft_mlp_roofline_self_test_passes=1` (a–f all pass). **Test:** `projected_tps_gain_pct=4.392` (band +4.4..+5.9%).
- **Key finding:** draft MLP (bf16, 4 layers, hidden=256, intermediate=2048, gated, NOT quantized) measures **50.83µs/pass** (2.42× above its 20.97µs HBM-bandwidth roofline @ 600 GB/s) — BUT the gap is **INTRINSIC-M=1**: the single-warp GEMV hits only **41% of peak HBM BW** (vs 81% for a large 128 MiB reference GEMV that IS saturating the bus), so the 20.97µs roofline is physically unreachable at M=1 without M≥16 batching. gate+up already fused in the served Gemma4MLP (3.6µs saving, not a free lever). The ONLY recoverable slack: the **separate GeluAndMul kernel** (does ~0 real work — 12 KB HBM, purely launch-bound) → fold as gate_up GEMV epilogue (CUTLASS EVT) → **+7.32µs/pass recoverable** → `step_after=1166.9µs` → **+4.39% composition ceiling (band +4.4..+5.9% → 502.7–510.1 TPS, crosses 500 standalone)**. Greedy/PPL-safe: LOSSLESS (epilogue fold has no reduction-order change; L2 megakernel must re-verify FP-reassociation). Over-credit caveat: this is a COMPOSITION projection — stark #273 empirically tests whether draft-step cuts survive wall-clock.
- **Conclusion — MLP is intrinsic-M=1 with one separable-launch lever:** the 51.7% dominant draft term hides **modest** recoverable headroom (the activation kernel only, NOT the roofline gap). Refutes the naive "2.42× → ~20% headroom" read. **Reseated kanna → #277** (draft io_projection roofline — the multi-launch q/k/v/o path may carry more fold-able slack than the MLP's single activation kernel, as #264 found the io/embedding path overhead-bound at M=1).

## 2026-06-15 01:38 — PR #267: Local→official TPS transfer — 🟢 `tau_lo=1.03524` (stable, ~85% hardware/clock); local bar for official 500 = 482.98 — bank-the-analysis (0 TPS)

- **Branch:** `lawine/local-official-tps-transfer` · **Student:** lawine · merged 01:38:09Z by advisor (senpai; LOCAL re-serve + telescoping decomposition matched to the official `summary.json:tps` boundary, no served-file change; BASELINE 481.53, 0 TPS). W&B [`nzqnd154`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/nzqnd154) (group `local-official-tps-transfer`, finished, NaN-clean, self-test pass) — advisor W&B-verified (`tau_lo=1.03524`, `additive_gap_tps=16.389`, `tau_lo_stable=1`, `local_bar_for_official_500=482.982`, `transfer_calibration_analysis_only=1`).
- **Primary:** `local_official_tps_transfer_self_test_passes=1`. **Test:** `tau_lo=1.03524`.
- **Key finding:** the deployed linear submission reads **465.14 local vs 481.53 official** — a systematic **+16.39 TPS (+3.40%)** offset; transfer factor `tau_lo=1.03524`, **STABLE** (spread 0.135% across E[T] 3.5–4.5), offset ÷ run-to-run noise ≈ 100× (a systematic MEAN offset, NOT σ_hw variance). Telescoping decomposition: **85.2% HARDWARE/CLOCK** (local A10G accept-cycle ~3% slower + sglang aggregation) + **14.8% harness** (dominated by first-few-request warm-up inclusion, which official sglang discards — NOT steady-state tokenization). Independent cross-check (kanna #217): local accept-cycle **8.221 ms** vs official **7.983 ms** = +2.99%, reproducing the hardware leg to 4 decimals.
- **Conclusion — closes the TPS leg of the launch-gate transfer:** a LOCAL build must read **≥482.98 TPS** to clear official 500, and local profiling is a trustworthy **×1.0352** proxy. Every local screen is now mappable to official — immediately load-bearing for **stark #273** (wall-clock K A/B local→official) and **land #245** (live built-step TPS). Reseat lawine #276 (advisor) → **acceptance local→official transfer** (the E[T]/λ̂ COMPANION to tau_lo — does local λ̂ carry a τ_acc from int4-Marlin batch variance, or is it a kernel-invariant proxy for the official 0.9780 bar?).

## 2026-06-15 01:32 — PR #263: Private rank-2+ probe (OOD tree-recovery) — 🟢 DECISIVE NULL, tree DEGRADES OOD, `private_rank2plus_coverage=0.4768` — bank-the-analysis (0 TPS)

- **Branch:** `ubel/private-rank2plus-probe` · **Student:** ubel · merged 01:32:33Z by morganmcg1 (advisor; rank-only analysis-only forward over private proxy, `--no-logits` #79 path, no served-file change; BASELINE 481.53, 0 TPS). W&B [`he7glotf`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/he7glotf) (private, group `private-rank-probe`) + [`pvt1rprm`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/pvt1rprm) (public repro) — Morgan W&B-verified (priv cov 0.47678, beyond-top4 0.52322, partition=1.00000 exact, ρ [0.2782,0.1738,0.1226], n_div 17,677, align_bad 0; public repro 0.65409 resid 0.0011), self-test 6/6 (a–f).
- **Primary:** `private_rank2plus_probe_self_test_passes=1`. **Test:** `private_rank2plus_coverage=0.4768`.
- **Key finding — the width>1 tree DEGRADES out-of-distribution:** private rank-2+ draft coverage **0.4768** vs public **0.6532** (Δ **−0.1764**, −27% rel); beyond-top-4 **0.5232** (the true target-argmax falls past the width-4 tree >half the time, vs public 0.3468). Spine also degrades: private top-1 accept 0.5975 vs public 0.7335. The branch-salvage ρ-vector **collapses ~34%** across all three lanes — private **[0.2782, 0.1738, 0.1226]** vs public [0.4165, 0.2655, 0.1908]. The tree recovers only **73% of the intrinsic private E[T]-gap** (private E[T] stays depressed at **3.6406**, not the 3.8445 it would reach if coverage transferred). **Decisive linkage:** the earlier `tree_private_acceptance_gap` (ytxfi6zk) 505.46 "clears 500" projection held branch-salvage ρ at **public** values — this probe measures that exact assumption and refutes it, re-pricing toward the raw-proxy ~450 regime.
- **Conclusion — the THIRD side of tonight's Path-A bound:** step side (fern #262: no tree clears 500 at the grounded built step, caps ~407–420), **private-E[T] side (this PR: tree E[T] recovery degrades OOD)**, verify side (openevolve: in-serve dense-mask tree verify does NOT recover λ either). Three independent angles agree Path-A does not reach 500 at the grounded numbers; the private 500-clear is **structurally at-risk even with the tree**. land #245 must re-price the descent-walk topology with the **measured private ρ** (the 505.46 figure is an upper bound). 🐛 **Bug flag (separate, human-owned):** the active 481.53 frontier `submissions/fa2sw_precache_kenyan/sitecustomize.py` LACKS `_guard_included_router` (sibling `treeverify` has it) — latent prometheus `_IncludedRouter` boot-500 risk that actually manifested on the local container; Morgan escalating as a GH issue for the human team (validated byte-identical port #177/`bjtwr9jn`). Reseat ubel #275 (advisor) → **prefill / TPS-denominator probe** (the untouched orthogonal Path-B lever — is the unmeasured prefill phase recoverable slack?).

## 2026-06-15 01:27 — PR #266: Adaptive-K onegraph realizability — ⚪ adaptive-K UNREALIZABLE; only static-K=4 survives (composition-level), `best_realizable_net_tps_gain_pct=4.2765` — bank-the-analysis (0 TPS)

- **Branch:** `stark/adaptive-k-realizability` · **Student:** stark · merged 01:27:49Z by advisor (senpai; LOCAL launch/host-read profiling + composition projection, no served-file change; BASELINE 481.53, 0 TPS, NO launch trigger). W&B [`cpjafa3h`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/cpjafa3h) (group `adaptive-k-realizability`, finished, NaN-clean, self-test 6/6) — advisor W&B-verified.
- **Primary:** `adaptive_k_realizability_self_test_passes=1` (6/6). **Test:** `best_realizable_net_tps_gain_pct=4.2765` (static-K=4 composition).
- **Key finding — runtime adaptive-K is UNREALIZABLE on the ONEGRAPH stack:** #256's +13.2% UB needs a runtime early-exit, but the realizable cost is **L=276.6µs/pass launch + h=161µs/pass host-read = 437.6µs** vs the **53.43µs/pass break-even budget** (8.2× over) ⇒ realizable **s/g_d = −5.38** (onegraph-break) / **−1.35** (multi-graph), BOTH net-harmful (the static-graph launch-amortization g_d=0.168 bakes in is worth MORE than the whole per-pass saving). **ONLY static-K nets positive:** K=4→**502.12 (+4.28%)**, K=5→500.79 (+4.00%), K=3/K=6 miss, K=7→481.53 (deployed); greedy-identical by construction; multigraph VRAM fits (20.96 GB). **CRITICAL CAVEATS (stark's own):** (1) the composition OVER-CREDITS draft-pass cuts — model-forward ~1.2182 ms is only a fraction of the ~8 ms wall step at conc=1, so static-K=4's +4.28% is composition-level, NOT wall-clock; (2) **deployed-K=7 is the standing evidence the composition over-credits** (if K were tuned under true wall-clock the frontier would already be at K=4/5). **static-K=4 is a CANDIDATE, NOT a measured ≥500 — no launch trigger.** Self-test (b) transparency: spec assumed s/g_d∈[0,1]; the measurement refuted it (<0); stark gated PRIMARY on the real s/g_d≤1 invariant and surfaced `b_s_over_gd_within_optimistic_01=False` separately — **advisor endorsed** (keep PRIMARY=1; the negative s/g_d IS the headline).
- **Conclusion — adaptive-K dead, static-K=4 the only survivor (composition-level only):** the next move is to TEST the composition-honesty question from both sides. Reseats (zero-idle): stark #273 (advisor) → **static-K wall-clock A/B** (does K=4/5 actually beat deployed K=7 in REAL local TPS? — the empirical composition-honesty test; stark expects it to shrink/invert); fern #274 (advisor) → **composition-overhead honesty** (analytic: decompose the wall step into fixed-overhead + model-forward, derive the φ-correction, re-price every draft-side "clears 500" claim — the analytic twin of stark's A/B).

## 2026-06-15 01:27 — PR #262: Fidelity-safe shallow tree (spine+ρ₂), RE-GROUNDED — 🟢 NULL, no tree clears 500 at the grounded step, `shallow_tree_implied_tps=406.59` — bank-the-analysis (0 TPS)

- **Branch:** `fern/fidelity-safe-shallow-tree` · **Student:** fern · resubmitted after send-back + bank-merged 01:27:27Z by advisor (senpai; analytic step-conditioned implied-TPS, no GPU/served-file change; BASELINE 481.53, 0 TPS). W&B [`u94y569g`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/u94y569g) (group `fidelity-safe-shallow-tree`, finished, NaN-clean, self-test 11/11 a–k) — advisor W&B-verified (self_test=1, shallow 406.586, full-tree 419.969, bracket 490.760/406.586/382.454, E[T]_needed 5.288, metrics_nan_clean=1).
- **Primary:** `fidelity_safe_shallow_tree_self_test_passes=1` (11/11). **Test:** `shallow_tree_implied_tps=406.59` (grounded central; was the anchor-inflated 506.27 pre-regrounding).
- **Key finding — the honest re-grounding (the STEP side of the Path-A double-bound):** the retired 1.085 ms anchor is struck; the step-conditioned bracket is {grounded-opt 1.1186→**490.76**, grounded-CENTRAL 1.3458→**406.59**, grounded-pess 1.4294→**382.45**} — **all three miss 500**. Inverted break-even: clearing 500 at the grounded central step needs **E[T]≥5.288** (+0.99 over the shallow tree's 4.300, +0.78 over the full tree's 4.512). The full 16-node tree at the grounded central step = **419.97**, reproducing denken #257's `implied_tps_at_ET4512` to resid 0.0. λ̂=0.9780113 has **zero headroom** (clears the 0.9780 bar by 1.1e-5).
- **Conclusion — Path-A (tree) does NOT reach 500 at the grounded step:** the binding constraint has FLIPPED from E[T]/fidelity to the **built STEP**; no tree route clears 500 unless the built step drops below ~1.10 ms. This is the FIRST of tonight's three independent Path-A bounds (step side here; private-E[T] side ubel #263; verify side openevolve). The +0.99 E[T] must come from cheaper drafting (the K_cal·(E[T]/step) denominator), not relaxed acceptance (λ̂ is maxed). Reseat fern #274 (Morgan) → composition-overhead honesty (does the composition over-credit draft-step cuts?). land #245's live built step is the arbiter.

## 2026-06-15 01:17 — PR #268: Verify-GEMM M-aware TPS(M) → M* — 🟢 `M*=32` (λ-ceiling, not interior); g_d fork SPLITS the verdict — bank-the-analysis (0 TPS)

- **Branch:** `denken/verify-gemm-m-aware-tps` · **Student:** denken · merged 01:17:09Z by advisor (senpai; CPU analytic over the #257 measured roofline, no GPU/served-file change; BASELINE 481.53, 0 TPS). W&B [`zgyiialr`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/zgyiialr) (group `verify-gemm-m-aware-tps`, finished, NaN-clean, self-test 5/5, 13.59 MiB) — advisor W&B-verified.
- **Primary:** `verify_gemm_m_aware_tps_self_test_passes=1` (5/5). **Test:** `best_realizable_tps_at_m_star=480.0` (central g_d) / `m_star=32`.
- **Key finding — M*=32 is the λ-valid ceiling (depth≤9), NOT an interior width below 32:** TPS(M) is monotone-nondecreasing across M∈[1..32] for BOTH step edges (E[T](M) gain outpaces the verify(M) growth — M16→M32: verify +10.6% but E[T] +14.3% ⇒ no interior max). The unconstrained optimistic argmax sits at M≈48 but is **λ-EXCLUDED** (unmeasured depth). **The g_d fork SPLITS the go/no-go:** central (measured g_d=0.0195, step M32=**1.34584 ms**) max **480.0 NO-GO**; optimistic (assumed g_d=0.168, step M32=**1.11859 ms**) max **577.6 GO**. The #241 floor E[T]=4.3305 never clears central and barely misses 485.0 optimistic. Step provenance byte-exact to #257.
- **Conclusion — the g_d step-basis fork is THE binding Path-A decision:** central NO-GO (480.0) vs optimistic GO (577.6) hinges entirely on which g_d the deployed step uses. Hand-off to land #245: build M*=32 (E[T]=5.1573 ≥ floor, λ̂=0.9827 ≥ bar, step∈[1.1186,1.3458] ms). Reseat denken #271 (advisor) → **g_d step-basis reconciliation** (measure the deployed-path g_d directly to collapse the fork — the single highest-leverage open lever, surfaced by both #257 and this PR).

## 2026-06-15 01:14 — PR #264: Draft-head vocab roofline — ⚪ NULL, vocab-head IMMATERIAL (~5% of pass), `projected_tps_gain_pct=0.0` — bank-the-analysis (0 TPS)

- **Branch:** `kanna/draft-head-vocab-roofline` · **Student:** kanna · bank-merged 01:14:13Z by advisor (senpai; LOCAL GPU micro-profiling + CPU analytic, no served-file change; BASELINE 481.53, 0 TPS). W&B [`95x7qv6h`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/95x7qv6h) (group `draft-head-vocab`, finished, NaN-clean, self-test 6/6) — advisor W&B-verified (self_test=True, projected_tps_gain_pct=0, head_share_pct=5.0085%, VRAM 0.349 GiB).
- **Primary:** `draft_head_vocab_roofline_self_test_passes=1`. **Test:** `projected_tps_gain_pct=0.0`.
- **Key finding — the premise was factually false for THIS proposer:** the deployed `Gemma4AssistantForCausalLM` head is NOT a dense hidden×256k GEMV — it is a **centroid-routed sparse gather** (`Gemma4AssistantMaskedEmbedder`: 256→2048 centroids → top-64 → gather 8192 active tokens/pass, tied embedding). A dense 256k head measured at M=1 = **276.9µs = 2.7× the whole 101.2µs pass** ⇒ physically absent. The vocab-projection that IS there (centroids sampler 256→2048) is **5.0%** of the pass — below the PR's own ≥10% materiality gate ⇒ correct NULL. **Draft-pass decomposition (the load-bearing by-product):** MLP **50.7µs/51.7%**, attention **28.5µs/29.1%**, io_projection 13.9µs/14.1%, vocab-head 4.9µs/5.0% (GEMV chain 98.0µs, resid 3.2% vs the 101.2 anchor). The one live-looking number (+7.18% from a dense top-32768 swap) is a SERVED head-SWAP (replace the trained centroid router), E[T]-unpriced and non-increasing ⇒ correctly NOT banked, handed to denken.
- **Conclusion:** the draft vocab-head lever is DEAD (immaterial, < 10%) — orthogonal to and as dead as weight-quant (int3 #248 / int4 #254). The decomposition relocates the recoverable draft step-mass to **MLP (51.7%) + attention (29.1%) = 80.8%**. **Reseats (zero-idle):** kanna → **#269** (draft MLP roofline — is the 50.7µs MLP at its HBM-bandwidth floor or overhead-recoverable via fusion/launch-erasure?); wirbel → **#270** (draft/tree attention Triton autotune — bit-identical kernel-config slack on the 28.5µs attn term at M∈{1,8,16,32}, the lever lawine #246's forced-TRITON_ATTN finding opens). [wirbel #265 verify+accept epilogue capture was Morgan-merged 01:09Z, freeing that seat.]

## 2026-06-15 01:09 — PR #265: Verify+accept epilogue capture — ⚪ verify-side epilogue IS separately launched but gain is a λ=1 ceiling, `projected_tps_gain_pct=3.8357` — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/verify-accept-epilogue-capture` · **Student:** wirbel · merged 01:09:28Z by advisor (senpai; LOCAL served-loop-graph launch manifest read, no GPU/served-file change; BASELINE 481.53, 0 TPS). W&B [`12mh3ov7`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/12mh3ov7) (group `verify-accept-epilogue-capture`, finished, NaN-clean, self-test 7/7) — advisor W&B-verified.
- **Primary:** `verify_accept_epilogue_separately_launched=True` / self-test 7/7. **Test:** `projected_tps_gain_pct=3.8357` (λ=1 CEILING, NOT realizable — realizable = λ×ceiling).
- **Key finding — the COMPLEMENT of #261's captured draft side:** the VERIFY+accept epilogue **IS separately launched** (EAGER), unlike the draft argmax+embed epilogue (#261, already inside ONEGRAPH). The eager epilogue (mid **45 µs**) accounts for **33.8%** of the 133.2 µs served-vs-built gap; the remaining ~66% is verify-forward compute-bound + CPU/Python (→ handed to denken #257's built-step roofline). `capture_greedy_safe=True` (0/0/0 token divergence over 2000). Clarifies #261's 56/70/84 µs are **14-launch TOTALS** (per-launch 4/5/6 µs).
- **Conclusion:** verify-side epilogue is a real separate launch, but the headline +3.8357% is a **λ=1 ceiling, not a measured gain** — and the bulk of the served-vs-built gap is verify-forward compute, which #257 then grounded (DIVERGE, step [1.12,1.43] ms). Reseat wirbel #270 (Morgan) → **draft/tree attention Triton autotune** (bit-identical kernel-config slack on the 28.5 µs attn term from kanna #264's decomposition).

## 2026-06-15 00:47 — PR #257: Built-step roofline grounding — 🟢 `DIVERGE`, `step_built_measured_proj_ms=1.3458` (Δ+0.261 vs 1.085 analytic) — bank-the-analysis (0 TPS)

- **Branch:** `denken/built-step-roofline-grounding` · **Student:** denken · merged 00:47:19Z by morganmcg1 (advisor; LOCAL forward-pass roofline, no GPU train/served-file change; BASELINE 481.53, 0 TPS). W&B [`h1gj2ved`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/h1gj2ved) (group `built-step-roofline-grounding`, finished, NaN-clean, self_test 5/5). Terminal marker present.
- **Primary:** `built_step_roofline_grounding_self_test_passes=1` (5/5). **Test:** `step_built_measured_proj_ms=1.3458` (anchor 1.085 #252 → **DIVERGE +0.261**); peak VRAM 13.15 GB.
- **Key finding — verdict `DIVERGE` (high-value terminal, NOT a fail):** the forward-pass roofline does NOT support the analytic 1.085 ms built tree step. Measured `verify_us(M)` = {M8:5.164, M16:5.405, M32:5.980} ms ⇒ **verify(M32)/verify(M8)=1.158** (verify forward GROWS ~16% from served M=8 → tree M=32; knee at M=16; M=8 still weight-bound 1.010×, M=32 compute-bound 1.134×). `g_d_measured=0.0195` (vs 0.168 assumed; ~9× below) — but that is an isolated forward-pass ratio; under ONEGRAPH=1 (embed+argmax every iter, one static replay) the deployed-stack effective step could sit anywhere in **[1.119, 1.346] ms**.
- **Conclusion:** the **1.085 ms anchor is RETIRED**; advisor pre-registered **[1.12, 1.43] ms** as the comparison band on #245, and land #245's LIVE measured build step is the only true arbiter. Strategic consequence: with the built-step *advantage* refuted (built ≳ served 1.2182, not 1.085), Path-A to 500 now leans on **E[T], not a cheaper step** — needs E[T] ≳ 5.37 at the high-end step vs the 4.512 projection ⇒ **Path-A knife-edge**. Reseat denken #268 (Morgan) → verify-GEMM M-aware TPS(M)→M* (does any tree width clear 500 under the measured step roofline?).

## 2026-06-15 00:40 — PR #246: K-1 FlashInfer + CUDAGraph draft speedup — 🔴 CLOSED, both legs REFUTED (`self_test_passes=False`) — dead-end, 0 TPS

- **Branch:** `lawine/k1-flashinfer-cudagraph` · **Student:** lawine · **CLOSED** 00:40:00Z by morganmcg1 (advisor; clean dead-end with self_test=False — cleaner than merge). W&B [`0qc5lk4y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/0qc5lk4y) (group `k1-flashinfer-cudagraph`, finished, NaN-clean). LOCAL only, no served-file change.
- **Test:** local control **465.14 TPS** / eager 378.018 TPS; PPL 2.3767, VRAM 20.894 GB, CV 0.00777%. `self_test_passes=FALSE` (hypothesis-truth checks wired into the self-test; experiment-validity sub-checks PASS).
- **Key finding — both legs dead:** (1) **FlashInfer UNREACHABLE** — vLLM force-pins `TRITON_ATTN` at config load for Gemma-4 (heterogeneous head dims: local head_dim=256 / global 512); the FlashInfer kernels never initialize. (2) **CUDAGraph ALREADY DEPLOYED** (ONEGRAPH=1, +23.047% already inside the 481.53 baseline) ⇒ 0% net new; AND toggling ONEGRAPH on/off is NOT bit-identical (1218/65536 tok diverge, 9/128 prompts, FP reassociation) ⇒ greedy-identity risk.
- **Conclusion:** dead-on-arrival. **De-risk for land #245:** the linear path captures cleanly at batch (1+7)·1=8 — **NO size-29 crash** on the linear stack. Remaining systems lever surfaced: backport `CUDAGraphMode.FULL` (vLLM PR #21367, ~2–5% ceiling, large scope). lawine's clean local **465.14 anchor** (vs 481.53 official) → reseat lawine #267 → local→official TPS transfer (what local TPS clears the official 500 gate?).

## 2026-06-15 00:39 — PR #256: Adaptive-K early-exit drafting — 🟢 `+13.209%` projected (UPPER BOUND, onegraph-gated) — bank-the-analysis (0 TPS); the most promising TPS direction in many cycles

- **Branch:** `stark/adaptive-k-early-exit` · **Student:** stark · merged 00:39:28Z by morganmcg1 (advisor; LOCAL fuzz + analytic projection, no served-file change; BASELINE 481.53, 0 TPS, NO launch trigger). W&B [`d2yiv9jw`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/d2yiv9jw) (group `adaptive-k-early-exit`, finished, NaN-clean). Terminal marker present.
- **Primary:** `adaptive_k_self_test_passes=1` (token-identical over 102,400 fuzzed steps). **Test:** `projected_tps_at_theta_star=545.137` (+13.209%); `headline_is_onegraph_upper_bound=1`.
- **Key finding:** confidence-gate the MTP draft — if MTP top-1 confidence c_i < θ, stop proposing pass i+1..7. **Greedy-safe BY CONSTRUCTION** (verify criterion untouched). θ*=0.54: E[T] 3.844→3.360, mean_K 7→4.047, projected 481.53→**545.137**. PPL pinned 2.3772. The +13.2% is an **ONEGRAPH=1 UPPER BOUND** (the static graph replays all 7 passes; a runtime early-exit is incompatible). Saving-survival: break-even s/g_d=0.221, g_d=0.168/skipped pass ⇒ **SIGN robust** (survives losing 78% of g_d), **MAGNITUDE sensitive**.
- **Conclusion:** the best TPS lever in many cycles, but the headline is an **analytic projection, NOT a measured ≥500** (no launch). Conditional GO. Reseat stark #266 (advisor) → adaptive-K onegraph realizability (pin s/g_d — does any onegraph realization clear 500 with greedy-identity vs the DEPLOYED served path?).

## 2026-06-15 00:31 — PR #258: Private E[T]-gap decomposition — 🟢 PROVABLE NEGATIVE `INTRINSIC-WEAKNESS`, `recoverable_fraction=0.0` — bank-the-analysis (0 TPS)

- **Branch:** `ubel/private-et-gap-decomp` · **Student:** ubel · merged 00:30:55Z by advisor (CPU/cached-proxy decomposition, no served-file change; BASELINE 481.53, 0 TPS). W&B [`2khp8gzs`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/2khp8gzs) (group `private-et-gap-decomp`, finished, NaN-clean) — advisor W&B-verified (cond_drop_d1=0.1296 = 0.729−0.599, consistent).
- **Primary:** `private_et_gap_decomp_self_test_passes=1`. **Test:** `recoverable_fraction=0.0`.
- **Key finding — Argmax-invariance theorem:** the 0.75-E[T] public→private draft shortfall (pub 3.8445 → priv 3.0898) is a **DISCRIMINATION loss, not miscalibration**. Under width-1 greedy-exact verify, accept = 1[argmax(draft)==argmax(target)]; any strictly-monotone transform (temperature/threshold/rescale) preserves argmax ⇒ **NO greedy-safe calibration knob can flip a reject to accept** ⇒ calibration-recoverable share is EXACTLY 0. Axis 1: shallow collapse (d1 cond_drop 0.130) WITH deep IMPROVEMENT (d5–d7 better on private). Axis 2: universal across all 6 categories.
- **Conclusion:** the **entire calibration/temperature lever class is CLOSED** by theorem. The only greedy-safe E[T]-lifter is a width>1 tree (already priced via Path-A). No calibration/temperature screen warranted. Reseat ubel #263 (advisor) → private rank-2+ probe (does the width>1 tree recover the OOD E[T]-gap? — the ONE surviving lever).

## 2026-06-15 00:24 — PR #260: Plan-B serve-overhead screen — ⚪ `LIVE-BUT-TINY`, `projected_tps_gain_pct=0.0616%` (~+0.30 TPS) — bank-the-analysis (0 TPS)

- **Branch:** `kanna/serve-overhead-instrumentation` · **Student:** kanna · merged 00:24:29Z by advisor (LOCAL served-stack introspection, no served-file change; BASELINE 481.53, 0 TPS). W&B [`icnlsyer`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/icnlsyer) (group `serve-overhead-instrumentation`, finished, NaN-clean).
- **Primary:** `serve_overhead_self_test_passes=1` (version_pinned_match=True). **Test:** `projected_tps_gain_pct=0.0616` (band [0.018%, 0.185%], ~+0.30 TPS).
- **Key finding:** the prometheus `Instrumentator` IS attached on the hot `/v1/completions` path (live-read from the pinned served wheel), BUT the official bench is non-streaming (`--disable-stream`, `sglang.bench_serving --backend vllm-chat`, max_concurrency=1, output_len=512) ⇒ the middleware amortizes over a ~162 ms / 512-tok window ⇒ standalone gain only **+0.062%**. `vidraft_transfers_to_our_bench=0` (VIDRAFT's 0.90% claim does NOT transfer to our non-streaming bench). Greedy-safe (pure τ term, stacks additively).
- **Conclusion:** LIVE but a tie-breaker only (~+0.30 TPS). Reseat kanna #264 (advisor) → draft-head vocab roofline (a BIGGER orthogonal lever: is the 101 µs bf16 draft floor's 256k vocab head recoverable via restricted-vocab propose?).

## 2026-06-15 00:24 — PR #259: E[T] branch-interior sensitivity — 🟢 `PATH-A FRAGILE-TO-FULL-DISCOUNT`, breakeven `x*=0.1427` — bank-the-analysis (0 TPS)

- **Branch:** `fern/et-branch-interior-sensitivity` · **Student:** fern · merged 00:24:26Z by advisor (CPU renewal-reward decomposition, no served-file change; BASELINE 481.53, 0 TPS). W&B [`1j099vrm`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/1j099vrm) (group `et-branch-interior-sensitivity`, finished, NaN-clean) — advisor W&B-verified (breakeven x* in prose only, non-blocking; primary self-test + test metric exact).
- **Primary:** `et_branch_interior_sensitivity_self_test_passes=1`. **Test:** `confirmed_et=4.3003` (vs floor 4.3305).
- **Key finding:** E[T]=4.512 decomposes EXACTLY (residual 8.9e-16, additive renewal-reward over the committed topo74 parent array) into **E_spine=4.0244 (89.19%, real-KV-confirmed)** + **E_branchhit_ρ₂=0.2760 (6.12%, real-KV-confirmed)** + **E_branch_interior=0.2117 (4.69%, scratch-SUSPECT, nodes [6,9,10])**. Full discount (interior removed) = `confirmed_et=4.3003` → **MISSES the 4.3305 floor by −0.0302**; fidelity discount (×0.599) = 4.4271 → clears +0.0966. **Breakeven x*=0.142663** (real-path branch-interior fidelity must clear ~14.27% of projection; scratch's 0.599 is 4.2× that).
- **Conclusion:** Path-A is **FRAGILE to a full branch-interior discount** and the **branch-interior is LOAD-BEARING** — land #245's real-integration verify is make-or-break. Reseat fern #262 (advisor) → fidelity-safe shallow tree (price the spine+ρ₂ 13-node sub-tree at its OWN cheaper step — a potential THIRD route to 500 sidestepping the fidelity wall).

## 2026-06-15 00:22 — PR #261: Plan-B Rank-2 draft argmax+embed fusion screen — ⚪ `ALREADY-CUDAGRAPH-CAPTURED`, no separate launch tax — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/draft-argmax-embed-fusion` · **Student:** wirbel · merged 00:21:58Z by advisor (LOCAL served-loop-graph manifest read, no served-file change; BASELINE 481.53, 0 TPS). W&B [`egaz6m2f`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/egaz6m2f) (group `draft-argmax-embed-fusion`, finished, NaN-clean).
- **Primary:** `draft_argmax_embed_fusion_self_test_passes=1`. **Test:** `realizable_bound_pct≈0` (draft argmax+embed epilogue already inside the ONEGRAPH static replay).
- **Key finding:** the rank-2 draft argmax+embed epilogue is **ALREADY captured inside the ONEGRAPH=1 static graph** — there is no separate kernel launch to fuse/recover on the draft side. The counterfactual launch-tax band (lo=56 / mid=70 / hi=84 µs) is against a path the deployed stack already captures ⇒ 0 net realizable.
- **Conclusion:** draft-side launch-tax NOT separately recoverable (mirrors #246's "CUDAGraph already deployed"). Reseat wirbel #265 (advisor) → verify+accept epilogue capture (the COMPLEMENT: is the VERIFY-side launch tax separately launched / recoverable?).

## 2026-06-15 00:00 — PR #255: Greedy-verify lm_head/argmax epilogue screen — ⚪ already fused, `realizable_bound_pct=0.0538%` — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/greedy-verify-epilogue-screen` · **Student:** wirbel · merged 00:00:12Z by morganmcg1 (advisor; LOCAL served-stack introspection screen, no GPU run/served-file change; BASELINE 481.53, 0 TPS). W&B [`q8qism2z`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/q8qism2z) (group `greedy-verify-epilogue-screen`, finished, NaN-clean). Terminal marker present.
- **Primary:** `greedy_verify_epilogue_screen_self_test_passes=1`. **Test:** `projected_tps_gain_pct=0.00` (`realizable_bound_pct=0.0538%`).
- **Key finding — verdict `GREEDY-VERIFY-EPILOGUE-ALREADY-FUSED-NO-GO`:** the served verify epilogue (lm_head + argmax) is **already fused on-GPU**. The served path returns `SamplerOutput(sampled_token_ids=<GPU>, logprobs_tensors=None)` with the lm_head pruned to **K=12288** — it does NOT materialize+d2h the full M=8 logit tensor (the bad path the screen tested for). `gain_matd2h_pct=15.97%` is a counterfactual against a path the served stack already avoids; the realizable gain is `0.0538%` ≈ 0. Greedy-safety: bf16→fp32 monotonic upcast, argmax bit-identical. **Banked (re-confirmed):** `verify_us=559.83`, `drafter_us=658.37`, `verify_share=0.4596`.
- **Conclusion:** the verify epilogue is already optimal — no lever. Another linear-path micro-lever closed. Reseat wirbel #261 (Morgan) → Rank-2 draft argmax+embed fusion screen (is the draft launch tax recoverable?).

## 2026-06-15 00:00 — PR #254: Q-1' int4-draft roofline — ⚪ draft-quant DEAD on Ampere, `net_tps_gain_pct_int4_draft=−27.44%` — bank-the-analysis (0 TPS)

- **Branch:** `kanna/int4-draft-roofline` · **Student:** kanna · merged 00:00:10Z by morganmcg1 (advisor; LOCAL profiling + cached E[T], no served-file change; BASELINE 481.53, 0 TPS). W&B [`zav6nr8y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/zav6nr8y) (group `int4-draft-roofline`, finished, NaN-clean). Terminal marker present.
- **Primary:** `int4_draft_roofline_self_test_passes=1` (SAFETY GREEN — greedy+PPL pinned by construction). **Test:** `net_tps_gain_pct_int4_draft=−27.44%` (SPEED NO-GO); `draft_pass_us_int4_marlin=165.79`.
- **Key finding:** the int4 **W4A16 Marlin** draft pass = **165.79 µs vs bf16 101.2 µs** → `draft_pass_speedup_int4_vs_bf16=0.605` (Marlin is **SLOWER** than bf16 at M=1); `step_speedup_int4=0.726`. The Marlin kernel (`ops.marlin_gemm`, the deployed verify-body kernel family) is tile-optimized for large GEMMs, not the M=1 GEMV-shaped draft pass; the dequant overhead dominates. `net_tps_ceiling_bandwidth_bound=75.01` (even the BW-bound ceiling is small); with E[T] loss `−68.98%`.
- **Conclusion:** int4-draft NO-GO, **worse than int3** (#248: −13.68%). The #248 PIVOT hypothesis ("int4 has a real sm_86 kernel") is TRUE (kernel exists) but it **loses to bf16 at M=1**. ⇒ **draft-side weight-quantization (int3 AND int4) is now decisively DEAD on Ampere at M=1** — the bf16 draft's 101.2 µs/pass is the floor; draft speedups must come from launch-overhead erasure (lawine #246 CUDAGraph / wirbel #261 argmax+embed fusion) or fewer forwards (stark #256 adaptive-K), NOT quant. Reseat kanna #260 (Morgan) → serve-overhead screen.

## 2026-06-14 23:56 — PR #245 (cycle-1): Live tree-decode build — ⚪ BLOCKED / foundation-reframing: scratch-reconstruction verify forward only ~60% faithful (a real bug, NOT #192) — NOT a launch, BASELINE 481.53, 0 TPS

- **Branch:** `land/live-tree-decode-build` · **Student:** land · cycle-1 BLOCKED (terminal=false, pending_arms=true), sent back to wip by advisor with GO direction on the real `execute_model` integration. Local 1×A10G only, no HF Job/submission/served-file change; committed path stayed linear K=7 (probes observational) → PPL/greedy-identity unmoved.
- **Test:** `treeverify_served_gain_MEASURED_realized=0.0` (no live commit built); `tps_local_treeverify_served_linear_control=381.01` (same-session linear K=7 control, PPL 2.377). `live_treedecode_self_test_passes`: NOT PASSED (a: live commits > linear not built; b PPL ✓, c greedy-id ✓, d NaN-clean ✓, e VRAM≤24 ✓).
- **Key finding:** the M=8 clean-room control (batch-matched M=8, qq_bias OFF) reproduces the deployed verify argmax only **0.599** (0.633 excl. depth-0) over 7,644 rows — a faithful reconstruction should be ~1.0. Mismatch signature (43% confident gap≥1.0, 26% rank-1/2, 5.5% razor-ties) is the OPPOSITE of #192 int4-Marlin batch-variance ⇒ a **genuine scratch-reconstruction forward bug, worst at root, NOT #192** (uniform ~0.55–0.68 degrade across depths 1–6 ⇒ genuine forward bug by sitecustomize.py:777-778's own criterion). Consequences: (i) the real-KV SPINE ladder (mean_accepted_len 2.57–3.69, top1 0.69–0.78, λ q[2..9] all clear 0.9780, branch-hit ρ₂≈0.4165) SURVIVES — real-KV-derived, so the gate-2 λ̂ evidence is intact; (ii) the deep branch-INTERIOR contribution to E[T]=4.512 is now UNVERIFIED (scratch-derived) and may be optimistic; (iii) eager penalty brutal (381→57 TPS) ⇒ CUDA-graph capture existential; size-29 crash diagnosed (`max_cudagraph_capture_size=16`; M=16 verify sits at the boundary, capturable).
- **Advisor direction (GO, sent back to wip):** pivot to the real `execute_model` integration (bypasses the scratch reconstruction via real embed+PLE + real scheduler split-KV metadata). FIDELITY-FIRST gate — prove the real-path verify reproduces deployed argmax ~1.0 (the scratch was 0.60) BEFORE any capture; build the tree as sequential width-1 capturable forwards; capture the M=16 verify; gate unchanged (`treeverify_served_gain_MEASURED_realized>0` at PPL-valid wall_tps≥500, greedy-identity preserved on the committed path). **fern → reseated to #259** to price whether Path-A survives the branch-interior discount (does E[T]_lower still clear the 4.3305 floor?). **Empirically confirms human #124's "gate = built artifact, the analysis probe cannot substitute."**

## 2026-06-14 23:48 — PR #253: Two-path-to-500 portfolio card — ⚪ NOT-READY, `combined_reach500_prob=0.629` — bank-the-analysis (0 TPS)

- **Branch:** `fern/two-path-500-portfolio` · **Student:** fern · merged 23:48:04Z by morganmcg1 (advisor; CPU-only portfolio integration, no HF Job/submission/served-file change; BASELINE 481.53, 0 TPS, authorizes nothing). W&B [`zvkyeyqp`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/zvkyeyqp) (group `two-path-500-portfolio`, finished, NaN-clean) — advisor W&B-verified (self-test True, all provenance scalars round-trip).
- **Primary:** `two_path_portfolio_self_test_passes=1` (6/6). **Test:** `combined_reach500_prob=0.629`.
- **Key finding:** `0.629 = 1 − (1−P_A)(1−P_B)` with **P_A=0.280** (tree build) + **P_B=0.485** (5-lever linear portfolio); `second_shot_margin=0.349`; `readiness_not_ready=1`. 19/19 provenance scalars round-trip at 0.0 (every input prior traces to its source PR/run). P_B ≈ coin-flip (GO-weighted +3.33%, just below the +3.836% bar). Two independent shots at 500 (tree OR linear) materially beat either alone, but neither path is GO yet.
- **Conclusion:** Path-B priors are **STALE** — 3 of the 5 priced screens have since returned NO-GO (kanna #248 int3, ubel #250 n-gram, wirbel #251 activation-recycle). Reseat (Morgan → fern #259, E[T] branch-interior sensitivity — does Path-A survive land #245's scratch-verify finding). Portfolio **re-price deferred to batch on the next screen-return wave** (kanna #254 int4-draft / stark #256 adaptive-K / wirbel #255 verify-epilogue / lawine #246 K-1 + land #245 real-path anchor + #259 output), not per-screen.

## 2026-06-14 23:47 — PR #250: N-1 n-gram draft source — ⚪ clean NEGATIVE, `ngram_beats_trained_anywhere=False` — bank-the-analysis (0 TPS)

- **Branch:** `ubel/ngram-draft-matchrate` · **Student:** ubel · merged 23:47:34Z by morganmcg1 (advisor; CPU/cached-proxy diagnostic, no GPU hybrid built, no served-file change; BASELINE 481.53, 0 TPS). W&B [`2clig5mg`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/2clig5mg) + [`firpmfuz`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/firpmfuz) (group `ngram-draft-matchrate`, finished, NaN-clean) — advisor W&B-verified (gating metrics clean).
- **Primary:** `ngram_draft_matchrate_self_test_passes=1` (4/4). **Test:** `e_t_ngram_hybrid=4.214` (loose bound).
- **Key finding:** standalone `e_t_ngram=1.370` (hit 0.470, correct 0.558) — RED match-rate gate ⇒ no GPU hybrid built. Transfer probe: private chat proxies ARE ~2× more n-gram-predictable (gen-only E[T] 2.0–2.5) but the **trained draft ALSO wins there** (3.5–3.9) ⇒ `ngram_beats_trained_anywhere=False` (margin −1.36 to −2.57 everywhere). **N-1 (researcher rank 13) CONFIRMED DEAD.** Minor: `e_t_ngram` was never logged to the W&B summary (prose only) — non-blocking, gating metrics clean.
- **Conclusion + GOLD pointer:** the sharpest by-product isn't the n-gram — it's that the trained draft's own **PRIVATE E[T] (≈3.09 ShareGPT chat) sits ~0.75 below its PUBLIC E[T] (3.844)**; that gap is exactly where the private re-draw validity risk (operative λ̂≥0.9780, `P_invalid=0.05`) lives. Reseat ubel #258 → public→private E[T]-gap decomposition (recoverable greedy-safe calibration mismatch vs intrinsic draft weakness).

## 2026-06-14 23:34 — PR #247: T-1 OPT-Tree per-step DP — 🟢 confirms static-optimality, `e_t_gain_vs_static=+0.0235` (below 0.05 stop) — bank-the-analysis (0 TPS)

- **Branch:** `stark/opttree-perstep-dp` · **Student:** stark · merged 23:34:47Z by morganmcg1 (advisor; CPU-only online-DP self-test, no GPU/served-file change; BASELINE 481.53, 0 TPS). W&B [`nwkjfyqd`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/nwkjfyqd) (group `opttree-perstep-dp`, finished, NaN-clean) — Morgan-merged, advisor result-comment-read.
- **Primary:** `opttree_dp_self_test_passes=1`. **Test:** `e_t_gain_vs_static=+0.0235` E[T] (= +2.4 TPS at M=32).
- **Key finding:** a SECOND, ONLINE method confirms wirbel #244's static-optimality. The +0.0235 win is **purely variance-driven** (low-dev half +0.0013 vs high-dev half +0.0456 = 35× concentration); **at the deployed M=8 the DP collapses 100% to the linear chain** (no per-step shape to exploit). DP latency 21.2µs/step, greedy-safe by construction (verifier argmax stays authoritative).
- **Conclusion:** topology is now **DOUBLY exhausted** — wirbel #244 (static ρ-optimal) + stark #247 (online per-step DP) both say the deployed depth-9/branch-3 tree is already optimal at M=8. The launch hinges on the live build, not topology. Reseat stark #256 → adaptive-K early-exit drafting.

## 2026-06-14 23:34 — PR #252: Composition step re-anchor — 🟢 `anchor_verdict=DISTINCT-STEPS-BOTH-VALID`, 520.95 CONFIRMED, 4.3305 SURVIVES — bank-the-analysis (0 TPS)

- **Branch:** `denken/composition-step-reanchor` · **Student:** denken · merged 23:34:44Z by morganmcg1 (advisor; CPU-only analytic re-anchor, no GPU/served-file change; BASELINE 481.53, 0 TPS). W&B [`a7llo7o7`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/a7llo7o7) (group `composition-step-reanchor`, finished, NaN-clean) — Morgan-merged, advisor result-comment-read.
- **Primary:** `composition_step_reanchor_self_test_passes=1`. **Test:** `reanchored_tps_at_ET4512=520.9527`.
- **Key finding — resolves the load-bearing #241 anchor flag:** the served step (**1.2182 ms**, linear MTP K=7 — the 481.53 path, realized E[T]≈4.6828) and the built step (**1.084953 ms**, tree decode — projected E[T]=4.512) are **TWO DIFFERENT per-step costs for TWO DIFFERENT paths**, not a contradiction. The 520.95 GO-anchor is CONFIRMED under the BUILT step; the #241 `E_T_meas_floor=4.3305` SURVIVES (slope-invariant, moves 0.0). The 463.97 "inconsistency" #241 flagged was a path-mismatch artifact (pricing the BUILT E[T]=4.512 at the SERVED step). Edge decomposition: `520.95/481.53 = 1.0819 = 1.1228 (step) × 0.9635 (E[T])` — the build's win is a step-cost win, partly offset by its E[T] being ~3.6% BELOW the served path's (so the anchor is CONSERVATIVE on E[T]). **The launch TPS gate reads against the BUILT 1.085 step.**
- **Conclusion + caveat:** the re-anchor is a self-consistency round-trip; whether land's LIVE build actually measures ~1.085 ms/step stays **land #245's to measure**. Reseat denken #257 → built-step roofline grounding (empirically bracket the 1.085 ms tree step).

## 2026-06-14 23:33 — PR #251: Plan-B Rank-4 activation-recycle screen — ⚪ NULL/NO-GO, premise structurally void, `projected_tps_gain_pct=0.00` — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/planb-activation-recycle-screen` · **Student:** wirbel · merged 23:33:31Z by morganmcg1 (advisor; CPU-only architecture-introspection screen, no GPU/served-file change; BASELINE 481.53, 0 TPS). W&B [`zofotw9f`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/zofotw9f) (group `winners-curse-redraw-budget`, finished, NaN-clean) — advisor W&B-verified.
- **Primary:** `activation_recycle_screen_self_test_passes=1`. **Test:** `projected_tps_gain_pct=0.00`; `verifier_recomputes_L1_8=False`.
- **Key finding:** the premise is **structurally void**. The served Gemma-4 MTP drafter is a **4-layer/256-dim** assistant that consumes the target's 2560-dim hidden state AND **shares the target KV cache** (`num_kv_shared=16`) → it **never computes the target's L1-8** (target = 37-layer/2560-dim, full L1-8 compute). There is no L1-8 activation to recycle from drafter to verifier. `scenarioA_gain_pct=11.03%` is a false-premise counterfactual; `greedy_ppl_identity_breaks_if_built=1`. **Banked by-product (reused downstream):** `verify_us≈559.83`, `drafter_us≈658.37`, `verify_share=0.4596`.
- **Conclusion:** Plan-B Rank-4 activation-recycle DEAD. Reseat wirbel #255 → greedy-verify lm_head/argmax epilogue screen (does verify materialize+d2h the full M=8 logit tensor?).

## 2026-06-14 23:28 — PR #248: Q-1 QSpec int3 draft — ⚪ DOUBLE-DEAD on Ampere, `net_tps_gain_pct_int3_draft=−13.68%` — bank-the-analysis (0 TPS)

- **Branch:** `kanna/qspec-int3-draft` · **Student:** kanna · merged 23:28:20Z by morganmcg1 (advisor; LOCAL profiling + cached E[T], no served-file change; BASELINE 481.53, 0 TPS). W&B [`vsujhev0`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/vsujhev0) (group `qspec-int3-draft`, finished, NaN-clean) — advisor W&B-verified.
- **Primary:** `qspec_int3_draft_self_test_passes=1` (SAFETY GREEN — zero greedy-identity risk by construction). **Test:** `net_tps_gain_pct_int3_draft=−13.68%` (SPEED NO-GO).
- **Key finding — double-dead:** (1) **no fused W3A16 kernel on sm_86** → the dequant→bf16 path is 3.6× SLOWER than the bf16 draft (363.78 vs 101.15 µs/pass); (2) int3 is a **weak proposer** (E[T] 3.30→1.40). TWO load-bearing corrections to the fleet's mental model: **(a) the deployed draft is bf16, NOT int4** (49/49 BF16 tensors; proposer = `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant` at `/tmp/qat-assistant`); **(b) QSpec is arXiv:2410.11305**.
- **Conclusion + PIVOT:** int3-draft dead, but the correction is gold — the **bf16 draft pass = 101.2 µs × K=7 ≈ 708 µs ≈ 58% of the step** ⇒ draft-side speedups are HIGH-leverage, and **int4 (W4A16 Marlin) DOES have a real sm_86 kernel** where int3 didn't. Reseat kanna #254 → int4-draft roofline (does W4A16 Marlin beat the bf16 101.2 µs/pass draft at M=1?).

## 2026-06-14 23:14 — PR #249: Build-λ̂ target reconciliation — 🟢 two numbers not one, `build_lambda_operative_gate=0.9780` — bank-the-analysis (0 TPS)

- **Branch:** `fern/build-lambda-bar` · **Student:** fern · merged 23:14:25Z by morganmcg1 (parallel advisor; CPU-only analytic reconciliation, no HF Job/submission/served-file change; BASELINE 481.53, 0 TPS, authorizes nothing). W&B [`on4u78ul`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/on4u78ul) (group `build-lambda-bar-reconciliation`, finished, NaN-clean) — Morgan-verified (self-test True, both gates exact, no NaN).
- **Primary:** `lambda_bar_reconciliation_self_test_passes=1`. **Test:** `build_lambda_operative_gate=0.9780112973731208` (+ `build_lambda_defended_target=0.9807516141069097`).
- **Key finding — the [0.978, 0.986] band was never one number; the four floating constraints split 2-2 across TWO RISK AXES.** (i) VALIDITY (is λ̂ a high-enough LCB?): P95 bar 0.978011 + #243 worst-case-vertex floor 0.978413 (mean=500 @ adverse NLS vertex) — these agree to **4.0e-4**. (ii) DRAW-RISK (does a σ=7.391 draw land ≥500?): #239 integrated-5% 0.980752 div-informed / 0.9861 uniform. #239/#243 imported VERBATIM, round-tripped to committed JSON at **0.0 error** (≤1e-6). **OPERATIVE gate λ̂≥0.9780** (P95 validity, MUST clear, residual P_invalid=0.05); **DEFENDED target λ̂≥0.9808** (divergence-informed 5% draw-risk, SHOULD clear — advisory because under #124 publish-first the draw-below-500 risk 0.0589 at the gate is ACCEPTED post-hoc).
- **Conclusion:** the launch λ̂ bar is SETTLED at two numbers. Advisor + Morgan both AGREED with fern's judgment call: keep the operative gate at the P95 **0.9780**, do NOT raise to 0.978413 (the +4.0e-4 gap is below the resolution at which a distributional-95%-LCB and a point-mean-500 floor can be distinguished — immaterial to any build target). Folds into the launch card (now fern #253 two-path portfolio) row (iii). The TPS side of the same gate → denken #252 (composition step re-anchor).

## 2026-06-14 23:10 — PR #241: Projection→measurement E[T] shortfall tolerance — 🟢 pre-registered build floor, `E_T_meas_floor=4.3305` — bank-the-analysis (0 TPS)

- **Branch:** `denken/measured-et-shortfall` · **Student:** denken · merged 23:10:34Z by morganmcg1 (parallel advisor; CPU-only analytic, ~0.05 s, 12.1 MiB, no GPU). W&B [`hqewf1d6`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/hqewf1d6) (group `issue192-reading-calibration`, finished, NaN-clean) — advisor W&B-verified (all 6 sub-flags a–f green, `E_T_meas_floor` exact match, zero NaN/Inf).
- **Primary:** `measured_et_shortfall_self_test_passes=1`. **Test:** `E_T_meas_floor=4.330527243789328`.
- **Key finding:** the composition `official=K_cal·(E[T]/step)·τ` is LINEAR through the origin in E[T], so `delta_max_tps500 = 1 − 500/TPS_0` is **SLOPE-INVARIANT** (depends only on the δ=0 anchor TPS_0=520.95; step/τ cancel in the ratio) → **E_T_meas_floor=4.3305 (4.02% shortfall)** for the TPS500 milestone. The acceptance gate λ̂≥0.9780 is measured INDEPENDENTLY and is deep-tail-protected (#230: λ_deep floor 0.78749 == imported 0.7875 budget, resid 0.0; 39× the binding min-λ headroom) → **TPS500 BINDS** in the operative framing. Conservative uniform-adverse coupling floor 4.4890 (0.51%). Pre-registers land #245's one build run: **GO iff measured E[T]_both ≥ 4.3305 AND measured min-λ q[2..9] ≥ 0.9780, binding gate = TPS500.**
- **Conclusion + load-bearing flag:** clean shortfall band for land's critical-path build. denken transparently flagged that the BANKED composition step 1.2182 maps E[T]=4.512 → 463.97 TPS (a MISS at δ=0!) while land's GO-read needs effective step ~1.085 — a real anchor inconsistency the whole packet rests on → reseated to denken #252 to resolve.

## 2026-06-14 23:06 — PR #244: Ceiling-gap topology headroom — 🟢 compliant-private topology DEAD, `topology_lift_needed=+7.53 TPS` — bank-the-analysis (0 TPS)

- **Branch:** `wirbel/ceiling-gap-topology-headroom` · **Student:** wirbel · merged 23:06:49Z by morganmcg1 (parallel advisor; CPU-only, 33.3 MiB, 0 GPU, 0 TPS). W&B [`sgjvbzu3`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/sgjvbzu3) (group `winners-curse-redraw-budget`).
- **Primary:** `ceiling_gap_topology_headroom_self_test_passes=1` (6/6 a–f). **Test:** `topology_lift_needed=0.07324 E[T]` (= +7.53 TPS, +1.45%).
- **Key finding — VERDICT `COMPLIANT-PRIVATE-500-TOPOLOGY-DEAD-REOPENABLE-ONLY-BY-COVERAGE`:** NO reachable verify-tree topology lifts the operative int4-spec ceiling 520.95 above stark #226's 528.48 private bar. The deployed **depth-9 max-branch-3 tree is ALREADY both width- and depth-optimal at the TPS objective** (`topology_lift_max=+0.000 TPS`, best reachable = the DEPLOYED tree). The 528.48 compliant-PRIVATE lane reopens **ONLY** along the coverage/determinism axis (λ→1; realized_frac 0.9707→0.9848, +1.40pp coverage recovery), NOT topology. Arithmetic correction: the PR's "15.71 TPS gap" framing was wrong — the real gap is **+7.53 TPS** (`topology_lift_needed`). Public 500 lane unaffected (both ceilings clear public 500).
- **Conclusion:** closes the compliant-PRIVATE-500-via-topology question (dead; coverage λ→1 is the only reopener) — affirms fern #238's "compliant-private lane INFEASIBLE (520.95<528.48)" and **bounds stark #247's T-1 OPT-Tree upside**: the best STATIC tree is already deployed, so online per-step DP can only win via per-step confidence VARIANCE a fixed tree leaves on the table (expect the low end of the +3–10% range, or ~0 if draft confidence is step-stationary).

## 2026-06-14 22:45 — PR #238: #124 launch-decision GO/NO-GO card v2 — ⚪ NOT-READY / NO-GO, `n_green_gates=2` — bank-the-analysis (0 TPS)

- **Branch:** `fern/launch-decision-card` · **Student:** fern · merged ~22:45Z by advisor (CPU-only integration of the banked packet; no HF Job/submission/served-file change; BASELINE 481.53, 0 TPS, authorizes nothing). W&B [`xioud4hv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/xioud4hv) (group `launch-readiness-integration`, finished, NaN-clean) — advisor-verified.
- **Primary:** `launch_decision_card_self_test_passes=1` (all 6 PASS). **Test:** `n_green_gates=2`.
- **Key finding:** the self-test pins `readiness_verdict=NOT-READY` as long as `treeverify_served_gain_MEASURED_realized==0.0`, so the card mechanically refuses to read the +18.3%/E[T]=4.512/~520 projection as a delivered win. Only 2 gates unconditionally GREEN (PPL, 128/128); READINESS is the top-line RED — a deliberate correction of #231's "5-of-6 GREEN." Compliant-private lane INFEASIBLE (520.95<528.48). Single launch blocker localized to land #245's MEASURED ≥500 build.
- **Conclusion:** launch-validity packet COMPLETE and clean; sole blocker is a measured build. Carries the [0.978, 0.986] build-λ band (from #239/#243) forward to fern #249.

## 2026-06-14 22:45 — PR #239: f_priv-distribution integrated private-draw risk — 🟢 one number not two, `integrated_risk_at_speed_gate=0.1347` — bank-the-analysis (0 TPS)

- **Branch:** `kanna/fpriv-distribution-risk` · **Student:** kanna · merged ~22:45Z by advisor. W&B [`vbk7lq8z`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/vbk7lq8z) (group `winners-curse-redraw-budget`, finished, NaN-clean) — advisor-verified.
- **Primary:** `fpriv_distribution_risk_self_test_passes=1` (7/7). **Test:** `integrated_risk_at_speed_gate=0.13466`.
- **Key finding:** integrating a distribution on f_priv∈[0.957054 grounded, 0.969107 assumed] through #237's σ_draw=7.391 model (bookends reproduce <1e-9) collapses #237's 4× two-point spread (0.0583/0.2394) into ONE number: 0.1347 uniform / 0.1046 divergence-informed, band [0.0635, 0.2263]. **Load-bearing:** `lambda_integrated_risk5=0.9861` (uniform) / 0.9808 (divergence-informed) vs #237's point 0.9700 — f_priv uncertainty RAISES the 5%-draw-risk build-λ ABOVE the point estimate and above the 0.9780 P95 bar. Complementary to stark #243 (stark owns the worst-case BLEND vertex; kanna owns the draw-risk distribution).

## 2026-06-14 22:45 — PR #243: NLS worst-case f_priv under measured 0.73% divergence — 🟢 un-straddles the publish-first breakeven, `fpriv_worstcase_under_measured_div=0.96895` — bank-the-analysis (0 TPS)

- **Branch:** `stark/fpriv-worstcase-measured-div` · **Student:** stark · merged ~22:45Z by advisor. W&B [`3ml0shkm`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/3ml0shkm) (group `issue192-reading-calibration`, finished, NaN-clean) — advisor-verified.
- **Primary:** `fpriv_worstcase_measured_div_self_test_passes=1`. **Test:** `fpriv_worstcase_under_measured_div=0.96895`.
- **Key finding:** the corrected worst-case f_priv 0.96895 sits +0.009170 ABOVE the publish-first breakeven 0.959780 → the realizable band no longer straddles it (un-straddle threshold d*=0.4339; the measured 0.73% clears it ~59×). Corrected publish-first λ_floor=0.978413 (≈ central 0.978044, vs ∅/unreachable at the old grounded floor): the publish-first POINT gate is reachable at the worst realizable vertex once the corrected int4 physics is in. With kanna #239, brackets the build-λ target at [0.978 (worst-case vertex), 0.986 (5%-risk)].

## 2026-06-14 22:45 — PR #242: int4 divergence M-sensitivity — 🟢 PLATEAU at M=16, `projected_divergence_at_M16=0.0073` — bank-the-analysis (0 TPS)

- **Branch:** `lawine/int4-divergence-m-sensitivity` · **Student:** lawine · merged ~22:45Z by advisor. W&B [`itrpyg25`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/itrpyg25) (group `issue192-reading-calibration`, finished, 38 keys + artifact, NaN-clean) — advisor-verified.
- **Primary:** `int4_divergence_m_sensitivity_self_test_passes=1` (5/5). **Test:** `projected_divergence_at_M16=0.0072917` (~0.73%).
- **Key finding — mechanism correction (load-bearing for land #245's M=16 build):** all four int4-Marlin BODY GEMMs (qkv/o/gate_up/down) are bit-exact across M∈{1,8} (`max_abs_diff=0.0`) → the int4 body split-K is M-INVARIANT, contributes ZERO batch-width divergence. The residual 0.73% is the bf16 tied lm_head + bf16 attention/norm being batch-variant. For M=16: int4 body stays bit-exact; only the bf16 lm_head's per-row K-reduction can move, and that order is set by the hidden-dim K-partition (shared across rows), NOT by M → holds M=8→M=16 unless M=16 crosses a tiling boundary. Projection biases toward PLATEAU (~0.73%) with bounded upper tail. Corrects the PR's "int4 split-K=f(M)" framing — the divergence locus is bf16, not int4. Unblocks stark #243's re-pricing.

## 2026-06-14 22:46 — PR #240: TPS-vs-private-risk exchange rate — 🟢 NEGATIVE result: speed/safety co-monotone in build-λ, `tps_per_pct_risk_at_speed_gate=−0.6431` — bank-the-analysis (0 TPS)

- **Branch:** `ubel/tps-risk-exchange-rate` · **Student:** ubel · merged ~22:46Z by advisor. W&B [`t2nrnf2m`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/t2nrnf2m) + [`cl6poy6t`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/cl6poy6t) (group `issue192-reading-calibration`, finished, anchors 513.557@gate/520.953@λ=1 resid 0.0, NaN-clean) — advisor-verified.
- **Primary:** `tps_risk_exchange_rate_self_test_passes=1` (5/5). **Test:** `tps_per_pct_risk_at_speed_gate=−0.6431` (assumed) / −0.2443 (grounded).
- **Key finding:** along the build-λ axis TPS(λ) and private-clearance(λ) are CO-MONOTONE (both rise with λ) → the exchange rate is NEGATIVE everywhere (−0.19 TPS/pp floor → −4.93 λ=1; −0.64 at gate); there is NO speed-for-risk trade to optimize. Dropping λ loses 2.374 TPS to buy +2.87pp risk — strictly dominated. ubel correctly recorded `pr_premise_tps_decreasing_in_lambda_holds=False` and made PRIMARY mean "the composition leg is correct" (round-trips bit-for-bit), surfacing the premise refutation as a separate flag — the right self-test-hygiene pattern (adopted as standard; design Q answered: keep the encoding, do NOT flip to FAIL). **Reframes #124:** not "is the speed worth the risk?" — speed and safety point the same direction; the real tension is build-λ vs landing difficulty (land #245).

## 2026-06-14 22:36 — PR #71: Live tree-verify serving path (deploy wirbel #49's +16% E[T]) — ⚪ TERMINAL / BANK-THE-FOUNDATION (option B), NOT a launchable win: `treeverify_served_gain_MEASURED_realized`=0.0 — the served stack is STILL linear MTP K=7 (the ~1,400 lines of "tree" code are env-gated observational probes that never touch committed tokens, `sitecustomize.py:1590-1604`), so the +18.3% / E[T]_both=4.512 / ~520 TPS is an UNMEASURED analytic projection, never realized end-to-end; banked as viability + 4 validated components + a LABELLED-unmeasured projection + the greedy_exact cert; BASELINE STAYS 481.53, claims 0 TPS — MERGED by parallel advisor (bank-the-foundation)

- **Branch:** `land/tree-verify-serving-path` · **Student:** land · merged 22:36:05Z by morganmcg1 (parallel advisor); terminal landed 22:28Z (zero-quota kernel-level self-tests on the warm `/tmp/server-venv`, no full-model load ⇒ `wandb_run_ids:[]` by design, the honest record not a gap; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, served path = linear MTP K=7 untouched, no launch, authorizes nothing).
- **Primary:** `treeverify_foundation_selftest_pass=1.0` (4/4: E[T]-core/tree_spec; descend-walk accept twin branch-hit 0.4182≈ρ₂ 0.4165; 3c fused KV-relocate bit-exact 8.3µs/step; leg-1→leg-2 seam sync-free). **Test (honest headline):** `treeverify_served_gain_MEASURED_realized=0.0`.
- **Key finding — the launch-validity analytic packet is COMPLETE but the build it prices DOES NOT EXIST yet (the honest reset).** land self-caught the over-optimistic signal: MEASURED vs PROJECTED were conflated; the served gain is 0.0, the deep q[8..9] λ is a scratch-M=16 forward (noisier, deflation-caveated). Every banked analytic leg (fern #231, stark #191/#226/#233, wirbel #235, denken #236, ubel #234, kanna #237) prices the TARGET build (operative ceiling 520.95, publish-first floor λ 0.9780, private worst-case 528.48, PPL third-gate headroom 0.0428) — that pricing stays banked as the ACCEPTANCE SPEC; the reset clarifies the missing half: the build that meets the spec is the long pole where every external fleet team stalled.
- **Conclusion — (B): bank the foundation as terminal; the live tree-decode build is THE critical-path follow-up and the entire team's launch hinges on it.** The launch gate = the first `treeverify_served_gain_MEASURED_realized>0` at valid-PPL representative wall_tps. land → reseated by the parallel advisor → **#245 (live tree-decode build — commit tree-accepted tokens at measured wall_tps; scope: CUDA-graph capture past the size-29 crash + live 3c KV-relocate wired into the decode commit).** Build-unblock recipes relayed from the Plan-B researcher sweep: Rank-5 multi-shape CUDA-graph suite keyed by M (size-29) + Rank-3 pinned-memory async KV-relocate (3c).

## 2026-06-14 22:25 — PR #237: Publish-first accepted-risk curve — how much single-draw risk did #124 accept? — 🟢 GREEN / `accepted_risk_at_speed_gate`=0.0583 (assumed f_priv) / 0.2394 (grounded #224): the #124 publish-first GO accepts ~6% single-draw private-miss risk under the assumed f_priv=0.969 but ~24% under the grounded #52 f_priv=0.957054 — highly f_priv-sensitive; `lambda_risk5`=0.9700 (the λ capping private-miss at 5%) sits 0.008 BELOW stark #191's 0.9780 bar, i.e. the P95 bar OVER-protects (carries public sampling σ the private draw doesn't see) — MERGED by advisor (bank-the-analysis)

- **Branch:** `kanna/publish-first-accepted-risk` · **Student:** kanna · merged ~22:2xZ (advisor drove merge) (LOCAL CPU-only analytic over banked #228/#217/#224 + σ_draw 7.391; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `8x7i38jh` (group `winners-curse-redraw-budget`, finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace).
- **Primary:** `publish_first_accepted_risk_self_test_passes=1`. **Test:** `accepted_risk_at_speed_gate=0.0583` (assumed) / 0.2394 (grounded).
- **Key finding — quantifies the #124 accepted-risk band.** `P(private draw clears 500 | built λ)` across the publish-first band [0.9138, 0.9780]: at the operative public speed gate λ=0.9675 the single-draw private-MISS risk is **5.83%** under assumed f_priv=0.969 but **23.94%** under the grounded #52 f_priv=0.957054. `lambda_risk5`=0.9700 caps private-miss at 5% and sits 0.008 below stark #191's LCB bar 0.9780 — confirming the P95 bar over-protects by carrying public sampling σ.
- **Conclusion — the #124 publish-first GO is a ~6–24% single-draw private-miss bet depending on f_priv grounding; the grounded read is the conservative one.** Hand-off → fern: carry the accepted-risk row [assumed 5.83% / grounded 23.94%] at the 0.9675 gate. kanna → reseated → #239 (f_priv-distribution integrated risk band).

## 2026-06-14 22:24 — PR #236: PPL public-gate headroom — is PPL≤2.42 a THIRD binding public gate under the lossy int4 verify? — 🟢 GREEN / `ppl_is_binding_public_gate=False`, `ppl_headroom_at_build_bar`=0.0428: the served PPL is pinned at 2.3772 (int4-greedy stream) and is **λ-INVARIANT** (the verify accept/reject changes WHICH tokens commit but not the int4-greedy decode distribution the scorer measures), so the milestone's PPL≤2.42 condition is NOT a third binding public gate — the only binding public gate stays TPS≥500 — MERGED by advisor (bank-the-analysis)

- **Branch:** `denken/ppl-public-gate-headroom` · **Student:** denken · merged ~22:2xZ (advisor drove merge) (LOCAL CPU-only analytic over banked frontier #52 served-PPL + composition; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B group `issue192-reading-calibration` (finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace).
- **Primary:** `ppl_public_gate_headroom_self_test_passes=1`. **Test:** `ppl_headroom_at_build_bar=0.0428`.
- **Key finding — PPL is not a binding public gate; the headroom is small but the gate is λ-invariant so it never binds at the build.** Frontier #52 serves PPL 2.3772 vs the 2.42 milestone ceiling → headroom only **0.0428** — narrow, but the build's verify layer is greedy-spec over the SAME int4 decode stream, so served PPL does not move with λ (the tree changes commit ORDER/COUNT, not the per-token int4-greedy distribution). ⇒ `ppl_is_binding_public_gate=False`.
- **Conclusion — the publish-first public milestone is single-gated on TPS≥500; PPL≤2.42 and 128/128 are satisfied-by-construction on the int4-greedy stream.** Hand-off → fern: drop PPL from the binding-gate set (carry as a satisfied precondition, headroom 0.0428). denken → reseated → #241 (projection→measurement E[T] shortfall tolerance — pre-registers land #71's one build run pass/fail vs the 4.512 projection).

## 2026-06-14 22:20 — PR #235: Two-ceiling reconcile — is the operative compliant ceiling 536.66 or 520.95? — 🟢 GREEN / **INFEASIBLE FLIP**: `operative_compliant_ceiling`=520.95 (the int4-SPEC E[T] ceiling, NOT the reach-DP 536.66) < stark #226's private bar 528.48 ⇒ the compliant-PRIVATE-500 lane is INFEASIBLE even with a perfect kernel, a **15.71 TPS gap that is 100% topology/coverage, NOT kernel** — MERGED by parallel advisor (bank-the-analysis)

- **Branch:** `wirbel/two-ceiling-reconcile` · **Student:** wirbel · merged ~22:1xZ by morganmcg1 (parallel advisor) (LOCAL CPU-only analytic over banked #199/#216/#226 + the int4-spec E[T] core; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `w6a34f51` (group `winners-curse-redraw-budget`, finished, NaN-clean).
- **Primary:** `two_ceiling_reconcile_self_test_passes=1`. **Test:** `operative_compliant_ceiling=520.95`.
- **Key finding — the operative compliant ceiling is the int4-spec 520.95, not the reach-DP 536.66, and that FLIPS stark #226's feasibility verdict to DEAD.** The 536.66 is a reach-DP E[T] upper bound; the OPERATIVE compliant ceiling is the int4-spec E[T] at the current verify topology = **520.95** (λ=1). Since stark #226's worst-case private bar is **528.48**, the compliant-private-500 lane is INFEASIBLE by **15.71 TPS** — and the gap is 100% topology/coverage (branch-width/depth/ρ), NOT the kernel (a perfect batch-invariant kernel still lands at 520.95). Closes the lane AT THE CURRENT TOPOLOGY.
- **Conclusion — the compliant-PRIVATE-500 lane is DEAD at the current topology; closing it needs a topology change, not a kernel.** Hand-off → fern + land #71: under strict-#192 compliance, a perfect kernel still misses the private bar by 15.71 TPS ⇒ land's build should target publish-first (public 500), not the compliant-private point, unless a reachable topology lifts 520.95 above 528.48. wirbel → reseated by parallel advisor → #244 (topology headroom — price the +15.71 E[T] lift; is any reachable branch-width/depth/ρ change enough, or is the lane certified DEAD?).

## 2026-06-14 22:16 — PR #234: Publish-first public margin — the σ→LCB public GO margin at the #124 floor — 🟢 GREEN / public margin map: 0.0 @ the publish-first floor → +2.367 @ the 0.9780 bar; `λ_public_gate`=max(0.9675, `lambda_floor_publish_first`)=0.9675 (the speed gate binds, not the private-mean floor 0.9138) — MERGED by parallel advisor (bank-the-analysis)

- **Branch:** `ubel/publish-first-public-margin` · **Student:** ubel · merged ~22:1xZ by morganmcg1 (parallel advisor) (LOCAL CPU-only analytic over banked #229/#228 + E[T](λ)→composition; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `izpjgncc` (group `issue192-reading-calibration`, finished, NaN-clean).
- **Primary:** `publish_first_public_margin_self_test_passes=1`. **Test:** public-margin map (0.0 @ floor, +2.367 @ 0.9780).
- **Key finding — completes #229's flagged cross: the operative publish-first public gate is the speed gate 0.9675, not the private-mean floor.** Tabulating the σ→LCB public GO margin across the band, the margin is 0.0 at the publish-first speed floor and rises to **+2.367** at the 0.9780 validity bar (matching ubel #229's worst-case). `λ_public_gate`=max(0.9675, `lambda_floor_publish_first`=0.9138)=**0.9675** ⇒ the public speed sub-gate binds the launch, consistent with the nesting 0.9138 < 0.9675 < 0.9780.
- **Conclusion — fern reads the public launch gate as 0.9675 with a 0.0→+2.367 margin ramp to the 0.9780 bar.** Hand-off → fern + ubel #240. ubel → reseated → #240 (TPS↔risk exchange rate — compose this TPS(λ) with kanna #237's risk(λ) into dTPS/drisk).

## 2026-06-14 22:12 — PR #233: Publish-first f_priv-breakeven — does the empirical calibration tail flip the #124 point-estimate gate? — 🟢 GREEN / `f_priv_breakeven_publish_first`=0.9598 (the realizable worst-case [0.957 grounded, 0.969 assumed] STRADDLES it); `d(λ_floor)/d(f_priv)`=−2.35 — MERGED by parallel advisor (bank-the-analysis)

- **Branch:** `stark/publish-first-fpriv-breakeven` · **Student:** stark · merged ~22:1xZ by morganmcg1 (parallel advisor) (LOCAL CPU-only analytic over banked #226/#224/#217 + #52 paired draw; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `pszvrf2a` (group `issue192-reading-calibration`, finished, NaN-clean).
- **Primary:** `publish_first_fpriv_breakeven_self_test_passes=1`. **Test:** `f_priv_breakeven_publish_first=0.9598`.
- **Key finding — the publish-first point-estimate GO straddles the f_priv breakeven.** At the λ=1 ceiling, private mean = 504.86 (GO) at central f_priv=0.969 vs 498.58 (NO-GO) at the empirical-floor f_priv=0.957 → break-even **f_priv≈0.9598**, which the realizable worst-case band [0.957, 0.969] STRADDLES; `d(λ_floor)/d(f_priv)`=−2.35 (the non-Latin-script maximizing vertex). So whether the publish-first POINT estimate clears 500-private at the ceiling depends on which f_priv calibration holds.
- **Conclusion — the #124 point-estimate private gate is f_priv-calibration-sensitive at the breakeven 0.9598.** Hand-off → fern + kanna. stark → reseated → #243 (re-price the worst-case f_priv blend under lawine #232's MEASURED 0.73% near-greedy divergence — does the breakeven stop straddling once the int4 decode-drop weight collapses?).

## 2026-06-14 22:08 — PR #232: int4 token-identity at deployed M — the M=8 divergence measurement — 🟢 GREEN / `int4_token_identity_M1_vs_M8`=0.9927 (i.e. **0.73% divergence, near-greedy** at the deployed M=8), an ORDER-OF-MAGNITUDE correction to kanna #114's M=1 56.08%; root cause = the int4 Marlin split-K reduction order is a function of batch width M — MERGED by advisor (bank-the-analysis)

- **Branch:** `lawine/int4-token-identity-deployed-m` · **Student:** lawine · merged ~22:0xZ (advisor drove merge) (LOCAL GPU model-loading + local inference probe — the allowed smoke-test class; READ the served path, did not modify it; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `nxwv6pam` (group `issue192-reading-calibration`, finished, NaN-clean). **Advisor-verified independently**.
- **Primary:** `int4_token_identity_self_test_passes=1`. **Test:** `int4_token_identity_M1_vs_M8=0.9927`.
- **Key finding — the deployed int4 stream is NEAR-GREEDY (0.73% divergence), not 56% — the 56.08% was an M=1 native-spec artifact.** Measuring per-token argmax identity between M=1 and the deployed M=8 batch width gives **0.9927** (0.73% divergence). Root cause pinned: split-K partitions the K-reduction across M-dependent tiles, so the float accumulation order — and argmax-flip probability near ties — is a function of M; at the deployed M=8 the reduction order is far more stable than at kanna #114's M=1 (56.08%). This is the divergence WEIGHT every downstream validity leg consumes.
- **Conclusion — the deployed int4-greedy stream is near-greedy; the strict-#192 exposure is ~0.73%, not 56%, at the served M.** Hand-off → denken #236 (PPL), stark #243 (f_priv worst-case), fern, land #71 (the tree verify runs at M=16, not M=8 — the M-axis projection is lawine's next leg). lawine → reseated → #242 (project divergence(M) to the build's M=16 via the split-K reduction-order mechanism + a confirming-probe flag).

## 2026-06-14 22:04 — PR #231: Launch-readiness GO-card — pre-register the decision on the one unknown — 🟢 GREEN / READINESS=NOT-YET: the integrated #124 publish-first GO/NO-GO card folds every banked launch axis into one pre-registered decision keyed on the single remaining unknown (land #71's measured build), and headlines `launch_authorized=False` because no ≥500 built checkpoint exists yet — MERGED by parallel advisor (bank-the-analysis)

- **Branch:** `fern/launch-readiness-gocard` · **Student:** fern · merged ~22:0xZ by morganmcg1 (parallel advisor) (LOCAL CPU-only integration over the banked launch packet; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B group `launch-readiness-integration` (finished, NaN-clean).
- **Primary:** `launch_readiness_gocard_self_test_passes=1`. **Test:** `n_green_gates` (the count of pre-resolved gates; the build gate stays RED).
- **Key finding — the launch decision is fully pre-registered except the build.** The card consumes the whole banked packet (binding bar 0.9780, public trigger 512.41/514.63, single validity gate #229, private bar interval [528.48, 535.14], one-run gate-2 rule #225/#230) and reduces the GO/NO-GO to ONE pending input — land #71's measured ≥500 build + its λ̂ tail. Headline `launch_authorized=False`: every analytic gate is green or pre-resolved, but the PHYSICAL build gate is unmet ⇒ READINESS=NOT-YET.
- **Conclusion — the integrator is armed; the launch is a one-call decision the moment land #71 measures, but cannot fire on a projection.** Hand-off → the human + land #71. fern → reseated → #238 (refresh the integrated GO/NO-GO card to fold in the post-#124 packet #232–#237 + carry the #235 INFEASIBLE flip; headline READINESS=NOT-YET).

## 2026-06-14 21:43 — PR #230: Depth-resolved gate-2 confirmation power — does ONE run carry the deep tail? — 🟢 GREEN / WHOLE_DEPTH_PROFILE_ONE_RUN_CONFIRMABLE but RAZOR-THIN: decomposing #225's aggregate gate-2 ASN by tree depth shows the comfortable 58× one-run margin is mostly the cheap spine — the sparse deep tail q[8..9] (λ_ref 0.7875 budget, 56× the spine's Bernoulli variance, 9% reach mass) needs `n_confirm_deeptail`=10,508 accept positions vs 13,366 available, a **1.27× headroom** (46× thinner than the aggregate's 58×), so the deep tail is the BINDING constraint on one-run gate-2 confirmation — MERGED by advisor (bank-the-analysis)

- **Branch:** `denken/gate2-depth-resolved-power` · **Student:** denken · merged ~21:43Z (advisor drove merge) (LOCAL CPU-only analytic decomposition of #225's aggregate ASN by tree depth; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `bo706b7n` (group `issue192-reading-calibration`, finished, 30.2 MiB, NaN-clean). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `gate2_depth_resolved_power_self_test_passes`=1 (6/6), TEST `n_confirm_deeptail`=10507.88, `aggregate_roundtrip_n_confirm`=1124.7629 (resid 0).
- **Primary:** `gate2_depth_resolved_power_self_test_passes=1` (6/6). **Test:** `n_confirm_deeptail=10507.88`.
- **Key finding — the aggregate's 58× one-run margin is mostly the cheap spine; the deep tail is doubly disadvantaged.** Depth-resolved `n_confirm(d) = n_confirm_agg·σ_d²/σ̄²`: spine q[2..7] (λ_ref 0.997, σ²=0.00299) each needs only ≈188 positions; deep tail q[8..9] (λ_ref 0.7875 budget = worst-case σ²=0.16735, **56×** the spine variance) each needs **10,508**. The deep tail is also sparse — q[8..9] carries only **9.08%** of the q[2..9] reach mass (#208/#203 β-extended reach-weights) → only 13,366 of 65,536 q-positions reach depth 8. The two effects nearly cancel: 13,366 available vs 10,508 needed = **1.27× one-run headroom** (vs the aggregate's 58×). Occ-weighted round-trip `Σ occ_norm(d)·n_confirm(d)`=1124.7629 reproduces #225 exactly (resid 0). Honest bookend: under DOUBLE-Deff (ACF×ICC #190, Deff_icc=4.41) deflation the ratio drops to 0.288 → NOT one-run confirmable — the matched single-Deff accounting (ACF once) is the intended read.
- **Conclusion — sharpens (does not flip) #225's one-run gate-2 confirmability; the deep tail is the binding constraint.** Hand-off → fern #185: carry `n_confirm_deeptail`=10,508 as the deep-tail-specific confirmation-run budget alongside #225's aggregate 1,125; land #71's per-step reach-DP occupancy dump is the tightening follow-up (resolves the 1.27× vs 1.03× base-norm / 2.25× independent-reads sensitivity), and the deep-tail λ̂ VALUE stays land #71's to measure (variance sized at the 0.7875 budget worst-case ⇒ a higher measured λ ⇒ lower variance ⇒ MORE confirmable). denken → reseated → #236 (PPL public-gate headroom: is PPL≤2.42 a third binding public gate under the lossy int4 verify?).

## 2026-06-14 21:43 — PR #228: Publish-first λ-floor — the built-λ where the private mean reaches 500 — 🟢 GREEN / `lambda_floor_publish_first`=0.913827 (CENTRAL reading, ADVISOR-CONFIRMED): the publish-first POINT-ESTIMATE floor (where the central private projection 535.433·f_priv reaches 500) reproduces stark #191's `lambda_star_central_private` to **8.88e-16** (machine epsilon), 0.0642 below the 0.9780 P95 both-bugs bar — so the band [0.9138, 0.9780) is the publish-first GO / P95-private HOLD region = exactly the single-draw risk the human accepted in #124 — MERGED by parallel advisor (bank-the-analysis)

- **Branch:** `kanna/publish-first-lambda-floor` · **Student:** kanna · merged 21:43:44Z by morganmcg1 (parallel advisor) (LOCAL CPU-only analytic inversion over banked #191/#217/#224 composition; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `352ifoi8` (group `winners-curse-redraw-budget`, finished, 28.7 MiB, NaN-clean). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `publish_first_lambda_floor_self_test_passes`=1 (8/8), TEST `lambda_floor_publish_first`=0.91382706, `lambda_gap_pe_vs_p95`=0.06418, `lambda_floor_central_xcheck_191_resid`=8.88e-16.
- **Primary:** `publish_first_lambda_floor_self_test_passes=1` (8/8). **Test:** `lambda_floor_publish_first=0.913827`.
- **Key finding — the publish-first POINT-ESTIMATE floor is 0.9138, and it equals #191's central private bar to machine epsilon.** Inverting the central private-mean curve `private_mean = K_cal·(E[T](λ)/step)·τ·f_priv` (central public anchor 535.433, f_priv 0.969107) for `private_mean=500` gives **0.913827** == stark #191's already-banked `lambda_star_central_private` (resid 8.88e-16) — an independent re-derivation off the live #183/#175/#184 reach-DP landing exactly on the banked central bar. **Convention call (advisor-confirmed on review):** the CENTRAL reading is the correct #124 publish-first gate — publish-first IS the *less*-conservative point-estimate launch, so the floor uses the central public estimate (535.433), NOT the conservative public-LCB ceiling (520.953·f_priv→floor 0.97783, which collapses onto the 0.9780 P95 bar — erasing the accepted-risk band — because 520.953·f_priv ≡ #191's `private_lcb_lambda1` by construction). f_priv sensitivity `dλ/df_priv`=−2.44: under the grounded f_priv 0.957054 (#224) the floor rises to 0.9433 but the central mean still clears at λ=1 (512.44≥500).
- **Conclusion — the publish-first band endpoints, and the three launch λ-thresholds nest cleanly.** Reconciliation for fern #185 + ubel #234: the 0.9138 floor sits BELOW ubel #229's `lambda_speed_clears`=0.9675, so under #124 the PUBLIC speed sub-gate (0.9675) binds the LAUNCH, not the private-mean floor — **0.9138 (private mean=500) < 0.9675 (public speed clears) < 0.9780 (P95 private valid).** fern carries 0.9138 as the publish-first private-mean row; the operative launch gate is the public 0.9675. kanna → reseated → #237 (publish-first accepted-risk curve: P(private draw clears 500 | built λ) — how much risk did #124 actually accept?).

## 2026-06-14 21:28 — PR #227: Valid-verify cluster capstone — can the Blackwell node unlock 500? — 🟢 GREEN / THE VALID-VERIFY MENU HAS COLLAPSED TO ONE SURVIVOR (`n_surviving_valid_500_paths=1`): the custom int4 batch-invariant verify kernel (lane-a), DOUBLE-gated on (a) the kernel BUILD landing near its 0.9455%-of-step split-K floor AND (b) land #71 λ ≥ 0.8572; every alternative is OUT — fp16 306.44, MarginGate 227.13 best-case, no-spec 165.44, off-the-shelf VLLM_BATCH_INVARIANT +51.78% — and a Blackwell draft-training run helps ONLY gate-b (λ), NEVER gate-a (the kernel build), so it is NOT the cluster unlock — MERGED by parallel advisor (bank-the-analysis)

- **Branch:** `wirbel/valid-verify-cluster-capstone` · **Student:** wirbel · merged 21:28:31Z by morganmcg1 (parallel advisor) (LOCAL CPU-only consolidation of the banked valid-verify ledger crossed against Issue #211's Blackwell offer; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `o674wmna` (group `valid-verify-cluster-capstone`, finished, NaN-clean). Terminal marker: PRIMARY `valid_verify_cluster_capstone_self_test_passes`=1, TEST `n_surviving_valid_500_paths`=1.
- **Primary:** `valid_verify_cluster_capstone_self_test_passes=1`. **Test:** `n_surviving_valid_500_paths=1`.
- **Key finding — exactly ONE valid >500 verify path survives, and the cluster cannot build it.** The collapsed menu: **lane-a int4 BI kernel** (split-K reduction-order fix ⇒ verify-M argmax == AR-M=1 ⇒ greedy-valid; λ=1 ceiling 520.95, floor-adj ≈516.1) — **the only survivor**, double-gated [(a) kernel build near the 0.9455% floor AND (b) land #71 λ≥0.8572]; **lane-b fp16** 306.44 (valid-premise, but <500 ∀ physical M_step≥1.3, no draft lifts it — lawine #221); **MarginGate** 227.13 best-case (sound skip ≤ 0.4392 from #114's 56.08% flip vs demand ≥0.9706 — 0.53-wide unclosable gap); **lane-c no-spec int4 AR** 165.44 (token-identity 1.0, structural compliant floor 66.9% below 500); off-the-shelf `VLLM_BATCH_INVARIANT=1` +51.78% (lane-a's whole-model foil, ~55× the verify-GEMM-only cost — NOT a 5th survivor). The Blackwell cluster's draft-training raises acceptance λ (gate-b) but leaves the verify GEMM (gate-a) untouched — so it cannot move the stack from invalid→valid.
- **Conclusion — operationalizes the #211 reframe: hold the Blackwell node; both surviving levers gate on CHEAP diagnostics + land #71's λ, not on draft quality.** The single mapped valid->500 route is the custom int4 BI verify kernel, whose buildability is the (human-gated) ~1–2 day GPU microbench (#211 diagnostic) — NOT a full-node draft-training run. Relayed to fern #185 (valid-verify menu = int4-kernel-only, double-gated) + Issue #211 (the cluster is not the unlock; the verify kernel is). wirbel → reseated by parallel advisor → #235 (two-ceiling reconcile: reach-DP 536.66 vs int4-spec 520.95).

## 2026-06-14 21:28 — PR #229: Speed margin at the validity bar — does speed clear at λ=0.9780? — 🟢 GREEN / `VALIDITY_BINDS_SPEED_ALWAYS_CLEARS`: re-scoring ubel #222's +2.367 worst-case margin AT the marginal validity λ=0.9780 (where E[T] and therefore speed are LOWEST) reproduces mu_pub 515.924 and the +2.367 BIT-EXACTLY, and speed clears at ALL THREE #218 σ corners — there is NO [0.9780, X) validity-passes-speed-fails band, so fern carries ONE gate (validity) — MERGED by parallel advisor (bank-the-analysis)

- **Branch:** `ubel/speed-margin-at-validity-bar` · **Student:** ubel · merged 21:28:29Z by morganmcg1 (parallel advisor) (LOCAL CPU-only analytic stress-test over banked #222/#218 + E[T](λ); no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `bz2b3fw8` (group `winners-curse-redraw-budget`, finished, NaN-clean). Terminal marker: PRIMARY `speed_margin_at_validity_bar_self_test_passes`=1 (6/6), TEST `speed_margin_at_validity_bar_worstcase`=+2.3666.
- **Primary:** `speed_margin_at_validity_bar_self_test_passes=1` (6/6). **Test:** `speed_margin_at_validity_bar_worstcase=+2.3666`.
- **Key finding — ubel #222's +2.367 was ALREADY scored at the marginal validity λ, so the gate ordering is fully hardened.** At the precise coupled validity bar 0.9779783 (nominal 0.9780), mu_pub_speed = 520.953·E[T](λ)/E[T](1) = **515.924** (gap −5.029 below the ceiling, the speed cost of landing at the marginal λ instead of the ceiling), which still sits **+2.367 above** the worst-σ trigger 513.557; `margin_drop_from_222_to_validity_bar`=0.0 (the #222 map imported directly, reproduced bit-exactly). Speed clears at all three #218 σ corners (tight 512.519 / central 512.735 / worst 513.557). Round-trips #204's 520.953 at λ=1. **No validity-passes-speed-fails band exists.**
- **Conclusion — the robustness capstone of ubel's launch-σ lane (#204→#207→#218→#222→#229): the launch is SINGLE-gated on validity.** Once land #71's built λ̂ clears the 0.9780 validity bar, the speed trigger is automatically satisfied (with +2.367 worst-case headroom) — fern #185 reads the VALIDITY gate alone, no second binding region. Relayed to fern #185 (one gate; speed auto-clears at the validity bar). ubel → reseated by parallel advisor → #234.

## 2026-06-14 21:25 — PR #226: Private-bar worst-case hardening — f_priv over realizable domain blends — 🟢 GREEN (null-spread) / THE REALIZABLE-BLEND AXIS IS EXHAUSTED, THE BINDING UNCERTAINTY MIGRATES TO DECODE-DROP CALIBRATION: re-pointing the #208 worst-case-blend LP from the λ-acceptance axis to the f_priv (private-TPS-drop) axis finds the f_priv-minimizing vertex is the SAME non-Latin-script vertex that maximizes the λ-deficit, so f_priv_worstcase=0.969107 == kanna #217's central and private_bar_worstcase=528.48 adds ZERO blend-spread; the lane is private-FEASIBLE vs the 536.66 central ceiling (−8.18) but INFEASIBLE vs the 525.73 LCB (+2.75) — MERGED by advisor (bank-the-analysis)

- **Branch:** `stark/private-fpriv-worstcase` · **Student:** stark · merged 21:25:19Z (advisor drove merge) (LOCAL CPU-only analytic worst-case LP over banked #217/#208/#198/#199/#202 + PR #52 paired draw; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `tzcc5xuq` (group `private-drop-shape-robustness`, finished, peak 46.9 MiB, NaN-clean, 53 summary keys). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `private_fpriv_worstcase_self_test_passes`=1 (a–f), TEST `private_bar_worstcase`=528.4836, `f_priv_worstcase`=0.969107, `compliant_lane_private_feasible`=1 (vs central 536.66), `compliant_lane_private_feasible_vs_lcb`=0 (vs LCB 525.73), `f_priv_breakeven_central_536`=0.95434, `lp_linear_vertex_optimal`=1, `feasible_vs_central_stable_over_beta`=1.
- **Primary:** `private_fpriv_worstcase_self_test_passes=1`. **Test:** `private_bar_worstcase=528.4836`.
- **Key finding — the realizable BLEND adds ZERO private-bar spread (we were already sitting on the worst vertex), and the genuinely load-bearing uncertainty is the decode-drop CALIBRATION.** Non-Latin-script (NLS) is BOTH the λ-deficit-maximizing AND the f_priv-minimizing vertex, so kanna #217's "central" f_priv=0.969107 *is* the adverse-vertex value; the worst-case-over-blends f_priv collapses onto it and `private_bar_worstcase`=528.48 round-trips the central bar to 13 digits (200k-pt Dirichlet + vertex-argmax LP, `lp_linear_vertex_optimal`=1, NLS the binding vertex, runner-up `native_code` at 0.0164pp). The binding risk migrates to the decode-drop CALIBRATION: the lone empirical hard paired draw (#52: f_priv=0.95705 → bar **535.14**) sits ≈1.5× OUTSIDE the realizable simplex (scale s=1.517) and only +0.0027 above the central break-even 0.95434; the scale sweep only crosses the 536.66 ceiling at s=1.633. Negative coupling (#198) does NOT widen the bar (the drop is smallest at the operating λ<1, so λ=1 is the conservative point).
- **Conclusion — closes the private-worst-case lane (#176→#198→#208→#215→#226); the private bar is an INTERVAL, not a point.** Feasibility: `private_bar_worstcase` 528.48 stays −8.18 below wirbel #199's compliant-spec ceiling 536.66 → compliant-verify lane private-FEASIBLE at the worst realizable blend vs the central ceiling, but MISSES the conservative LCB ceiling 525.73 by +2.75 → private-INFEASIBLE at P95 if the LCB is the gate. Hand-off → fern #185: carry the private bar as the **interval [central 528.48, empirical-floor 535.14]**, FEASIBLE vs 536.66 central / INFEASIBLE vs 525.73 LCB. Complements kanna #224 (central f_priv + physical-ceiling reachability). stark → reseated by advisor → #233 (publish-first f_priv-breakeven: does the empirical calibration tail flip the #124 point-estimate gate? — at the ceiling 504.86 GO at central 0.969 vs 498.58 NO-GO at empirical-floor 0.957, break-even f_priv≈0.9598 the interval straddles).

## 2026-06-14 21:15 — PR #221: FP16-verify cost + validity — measure the valid-path penalty locally — 🔴 RED / LANE-B CONFIRMED DEAD ON SPEED + ITS VALID-BY-CONSTRUCTION PREMISE EMPIRICALLY QUALIFIED: the measured fp16/int4 per-step ratio M_step=1.766 (inside stark #220's swept [1.3,2.3] band) gives fp16verify_tps_at_lambda1=294.99 ≪ 500 (no draft rescues it), AND a LOCAL token-identity probe shows fp16/bf16-verify is NOT perfectly greedy-valid-by-construction — `fp16_token_identity_M1_vs_M8=0.9894` (98.94%, ~1.06% RESIDUAL_BF16_BATCHVAR) — so the "cuBLAS ⇒ valid-by-construction" claim is qualified (OUT verdict unchanged, in fact reinforced) — MERGED by advisor (bank-the-analysis)

- **Branch:** `lawine/fp16-verify-valid-cost` · **Student:** lawine · merged 21:15:22Z (advisor drove merge) (LOCAL GPU model-loading + local inference profiling — the allowed-work class, same as a smoke test; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched — READ the served path, did not modify it; no launch, authorizes nothing). W&B `6m40u2bg` (finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `fp16_verify_cost_self_test_passes`=1 (8/8 sub-tests), TEST `m_step_fp16_int4`=1.766.
- **Primary:** `fp16_verify_cost_self_test_passes=1` (8/8). **Test:** `m_step_fp16_int4=1.766`.
- **Key finding — TWO results, one speed (settles stark #220) and one validity (QUALIFIES stark #220).** (1) SPEED: the measured fp16/int4 per-step ratio M_step=**1.766** lands inside stark #220's swept band [1.3, 2.3], pinning `fp16verify_tps_at_lambda1`=**294.99** (= 520.95/1.766) ≪ 500 — so no draft can rescue fp16-verify; #220's structural OUT verdict is now nailed to ONE measured M_step, not a swept assumption. (2) VALIDITY (the honest correction): the local M1-vs-M8 token-identity probe returns `fp16_validity_premise_confirmed=False`, `fp16_token_identity_M1_vs_M8`=**0.9894** (98.94%, premise `RESIDUAL_BF16_BATCHVAR`) — bf16/fp16-verify is NOT perfectly greedy-valid-by-construction; there is a residual **~1.06%** batch-width divergence (far below int4-Marlin's 56.08%, but non-zero). Within-dtype/within-batch determinism holds (tier-2).
- **Conclusion — lane-b stays OUT of the valid-verify menu (now collapsed to the int4 batch-invariant kernel alone, wirbel #216/#223), and its validity basis must be annotated honestly.** Hand-off → fern #185 (annotate lane-b's validity as "residual bf16 batch-variance 1.06%", NOT "valid-by-construction"; OUT on speed regardless) + stark #220 (premise qualified, verdict reinforced) + Issue #211 (fp16 is not the cluster unlock — speed AND a small residual validity gap). This harness (`6m40u2bg`) is reused by lawine's reseat #232 to measure the CLEAN deployed-M=8 int4 divergence (disambiguating #114's native-spec-vs-M1 56.08%). lawine → reseated by advisor → #232 (int4 deployed-M=8 token-identity probe).

## 2026-06-14 21:07 — PR #185: Launch-trigger calculator — one-call GO/NO-GO + filled approval block (re-run folding kanna #217) — 🟢 GREEN / THE LAUNCH-PACKET INTEGRATOR, BANKED AT THE #217 SNAPSHOT: 18/18 self-test, binding_bar=0.9780 (private #191 dominates), public σ→LCB trigger 512.41 central / 514.63 worst (ubel #204/#207), two-flag public-trigger-PASS vs private-bar-at-P95 structure (λ=1 ceiling 520.95 clears public but MISSES private 528.48 by +7.53), launch_authorized=False (HOLD on 3 hard gates) — MERGED by advisor (bank-the-analysis)

- **Branch:** `fern/launch-trigger-calculator` · **Student:** fern · merged 21:07:31Z (advisor drove merge) (LOCAL CPU-only analytic composition over the full typed launch-CI ledger; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, `launch_authorized=False` — authorizes nothing). W&B `cw4naa0t` (group `launch-trigger-calculator`, finished, peak 75.41 MiB, NaN-clean). **Advisor-verified independently** (`summary/` namespace): PRIMARY `launch_trigger_calculator_self_test_passes`=True, TEST `both_bugs_go_at_lambda_star`=True (binding_bar/launch_authorized are descriptive values the 18/18 self-test asserts internally).
- **Primary:** `launch_trigger_calculator_self_test_passes=True` (18/18, a–r). **Test:** `both_bugs_go_at_lambda_star=True`.
- **Key finding — `launch_decision(measured_tuple)` is one call that takes land #71's measured tuple and emits a human-ready GO/NO-GO + pre-filled (un-filed) `Approval request: HF job` block.** This snapshot consumes launch-readiness legs up through kanna #217: binding_bar=0.9780 (private #191 dominates public-iid 0.9052 / public-ICC 0.9513); σ→LCB public GO trigger 512.41 central / 514.63 worst (ubel #204/#207, clean-1σ unit rebase); the load-bearing #217 finding wired as TWO separate flags — FLAG-1 public-trigger (λ=1 ceiling 520.95 clears it +8.54/+6.32) vs FLAG-2 private-bar-at-P95 (528.48 = mu_safe_fresh/f_priv@0.969; ceiling MISSES it by +7.53, private clear only 0.744<0.95 even at λ=1). descent-only is the instructive NO-GO (clears the #183 build-gate but misses the #179 launch-projection — the two-LCB divergence).
- **Conclusion — the launch packet the human's Approval request will read; it gates the launch, does NOT trigger it.** `launch_authorized=False` persists (3 hard gates: land #71 build · measured λ̂≥0.9780 q[2..9] direct · #192 ruling). Snapshot consumed up through #217 ONLY — fern flagged #222/#223/#224/#225 as merged-but-unconsumed. Reseat attempted (integrator re-run folding the 4 post-#217 legs) but parallel advisor had already reseated fern → #231 (launch-readiness GO-card); fern covered, my orphan branch deleted.

## 2026-06-14 21:02 — PR #225: Gate-2 confirmation runbook — confirm measured λ̂_built ≥ 0.9780 both-bugs from ONE served run's q[2..9] data — 🟢 GREEN / GATE-2 IS CONFIRMABLE IN ONE RUN: a Wald SPRT certifies λ̂_built ≥ 0.97798 in n_confirm ≈ 1,125 decode steps (measured-ACF), which FITS one served run's ~65,536 q[2..9] positions; decision rule λ̂_LCB≥0.97798⇒PASS / [0.857,0.978)⇒HOLD / <0.857⇒NO-GO — MERGED by parallel advisor (bank-the-analysis)

- **Branch:** `denken/gate2-sprt-runbook` · **Student:** denken · merged 21:02:03Z by morganmcg1 (parallel advisor) (LOCAL CPU-only analytic SPRT/ASN consolidation over banked #205/#212/#191; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `851z7itj` (finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `gate2_confirmation_self_test_passes`=1, `n_confirm_measured_acf`=1124.763.
- **Primary:** `gate2_confirmation_self_test_passes=1`. **Test:** `n_confirm_measured_acf=1124.763`.
- **Key finding — gate-2 (the measured-λ̂ build confirmation) is operationally CHEAP and fits one served run.** The Wald SPRT over q[2..9] accept data confirms λ̂_built ≥ 0.97798 both-bugs (stark #191/#208 worst-case bar) in ~1,125 decode steps at the measured-ACF realism (denken #212's data-grounded point; 405 IID-floor / 1,788 flat-loose envelope ends). One served run yields ~65,536 q[2..9] positions ≫ 1,125, so a SINGLE run confirms the build. Decision rule for fern's gate-2 read: **λ̂_LCB ≥ 0.97798 ⇒ PASS · [0.857, 0.978) ⇒ HOLD (kernel-feasible but below private bar) · < 0.857 ⇒ NO-GO**.
- **Conclusion — consolidates denken's liveprobe-cost lane (#205 SPRT → #212 AR(1)-ASN → #225 runbook) into the operational gate-2 procedure.** Arms fern #185 with the PASS/HOLD/NO-GO flag and chiku-inu's build bench (gate-2 reads off land #71's served q[2..9] ladder). denken → reseated by parallel advisor → #230 (gate-2 depth-resolved power).

## 2026-06-14 21:02 — PR #224: Private-bar reachability — ground f_priv and answer whether 500-private is reachable at the λ=1 ceiling — 🟢 GREEN / f_priv=0.969 WAS OPTIMISTIC; UNDER THE GROUNDED 0.957 THE GAP WIDENS AND 500-PRIVATE IS UNREACHABLE AT THE PHYSICAL CEILING: the private build target rises to mu_ceiling_needed=535.14 (private mean @ λ=1 ceiling 498.58 < 500), so the kernel-ceiling route must target ~528–535, not 500 — MERGED by parallel advisor (bank-the-analysis)

- **Branch:** `kanna/private-bar-reachability` · **Student:** kanna · merged 21:01:58Z by morganmcg1 (parallel advisor) (LOCAL CPU-only analytic grounding over banked #217/#202/#191 + PR #52 paired draw; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `1081oc84` (finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `private_bar_reachability_self_test_passes`=1, `mu_ceiling_needed`=535.1394.
- **Primary:** `private_bar_reachability_self_test_passes=1`. **Test:** `mu_ceiling_needed=535.1394`.
- **Key finding — the assumed f_priv=0.969107 (kanna #217's basis for the 528.48 private bar) was OPTIMISTIC; the frontier's one hard paired draw grounds f_priv=0.957054 (#52: 481.53 public / 460.85 private), which is WORSE.** Under the grounded value the private build target rises to **535.14** and the gap to the physical λ=1 ceiling 520.95 WIDENS: at the ceiling the private mean is 498.58 < 500. So **500-PRIVATE is UNREACHABLE at the physical ceiling even at full self-KV recovery (λ=1)** — the compliant kernel-ceiling route must target ~528–535, not 500. This is the central-grounding complement to stark #226 (worst-case f_priv over domain blends).
- **Conclusion — sharpens fern #185's private-bar-at-P95 row from the optimistic 528.48 to the grounded 535.14 and confirms the launch HOLD is not a thin miss but a structural ceiling-vs-private gap.** Relayed to fern #185 (re-price the private bar row to 535.14 central, carry [528.48 optimistic, 535.14 grounded, worst-case TBD per stark #226]). kanna → reseated by parallel advisor → #228 (publish-first λ floor).

## 2026-06-14 21:01 — PR #223: MarginGate compliant-500 budget — price the researcher's top-ranked valid-verify path against the #213 budget — 🔴 RED / MarginGate STRUCTURALLY MISSES THE COMPLIANT-500 BUDGET: a sound margin gate's provable-stable skip rate must be ≥ 0.9706 (λ=1) but skip ⊆ non-flip ⇒ skip ≤ 1−flip_rate = 0.4392 ≪ 0.9706; Hybrid+DVR also misses (rollback burden = flip_rate, constant in skip) — the lowest-overhead compliant route stays the custom batch-invariant int4 kernel (#216) — MERGED by parallel advisor (bank-the-analysis)

- **Branch:** `wirbel/margingate-budget` · **Student:** wirbel · merged 21:01:54Z by morganmcg1 (parallel advisor) (LOCAL CPU-only analytic budget pricing over banked #213/#216/#114; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). Implemented from the PR-body spec (the RESEARCH_IDEAS_VALIDVERIFY file had not yet landed on its branch — faithful, not a blocker). W&B `54dtull1` (finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `margingate_budget_self_test_passes`=1, `skip_rate_min_at_lambda1`=0.970608.
- **Primary:** `margingate_budget_self_test_passes=1`. **Test:** `skip_rate_min_at_lambda1=0.970608`.
- **Key finding — MarginGate (researcher Rank-1, arxiv 2605.30218, P=0.45) is REFUTED by a structural argument, not a tuning miss.** To clear the 7.33%-overhead budget at λ=1 the provable-stable skip rate must be ≥ **0.970608**. But a SOUND margin gate (using the worst-case split-K perturbation ε_max) can only skip positions whose argmax cannot flip under ANY valid reduction order — and that provably-stable set is a SUBSET of the non-flip set, so skip ≤ 1 − flip_rate = 1 − 0.5608 = **0.4392 ≪ 0.9706**. The MarginGate+DVR hybrid also misses: the DVR rollback burden equals the flip_rate, which is constant in the skip rate, so no skip threshold buys it back.
- **Conclusion — closes the MarginGate/DVR analytic branch and pins the compliant-500 lane to ONE surviving route.** Under strict #192 the only compliant >500 path is the custom batch-invariant int4 verify kernel (wirbel #216, off-the-shelf 31.4% dies / custom floor 0.95%, λ_min=0.8572), conditioned on an UNMEASURED <7.33% kernel microbenchmark. Relayed to fern #185 (MarginGate/DVR-hybrid REFUTED; #216-kernel-only compliant route). wirbel → reseated by parallel advisor → #227 (valid-verify cluster capstone).

## 2026-06-14 21:01 — PR #222: Binding gate — does clearing the validity bar λ̂=0.9780 auto-clear the 513.557 speed trigger? — 🟢 GREEN / VALIDITY BINDS: at the validity bar λ̂=0.9780 the build shows public μ_pub = 515.924 ≥ 513.557 worst-case speed trigger (margin +2.367), so land #71 has a SINGLE launch target (0.9780) and fern reads the VALIDITY gate alone — MERGED by parallel advisor (bank-the-analysis)

- **Branch:** `ubel/binding-gate` · **Student:** ubel · merged 21:01:49Z by morganmcg1 (parallel advisor) (LOCAL CPU-only analytic gate-comparison over banked #218/#191/#183; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `yw7i2ece` (finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `binding_gate_self_test_passes`=1, `mu_pub_at_validity_bar`=515.9241.
- **Primary:** `binding_gate_self_test_passes=1`. **Test:** `mu_pub_at_validity_bar=515.9241`.
- **Key finding — the validity gate DOMINATES the speed trigger, so the build has one target, not two.** At the binding validity bar λ̂=0.9780 (private both-bugs, stark #191/#208 worst-case) the build's public mean is μ_pub = **515.924 TPS**, which already clears ubel #218's worst-case speed trigger 513.557 by **+2.367**. So a build that clears the validity bar has automatically cleared the speed trigger — the σ→LCB speed trigger (512.41/514.63) is dominated, not a separate binding test. land #71's single launch target is λ̂_built ≥ 0.9780.
- **Conclusion — simplifies fern #185's gate logic: read the VALIDITY gate alone.** Capstone of ubel's launch-σ lane (#204→#207→#218→#222). Relayed to fern #185 (single 0.9780 gate; speed trigger auto-satisfied). ubel → reseated by parallel advisor → #229 (speed-margin at validity bar).

## 2026-06-14 20:37 — PR #220: fp16-verify valid-path ceiling — can any draft clear 500 without an int4 kernel? — 🔴 RED / FP16-VERIFY IS VALID BUT A DEAD 500-PATH: greedy-valid-by-construction (cuBLAS has no M-dependent split-K ⇒ batch-invariant, no #114/#192 divergence, no kernel) but the draft-INDEPENDENT λ=1 ceiling 520.95/M_step is <500 at every physical M_step (crossover M_step*=1.0419 unreachable since fp16 is strictly slower ⇒ M_step>1); λ=1 cap 400.73 even at the optimistic M_step=1.3 — MERGED (bank-the-analysis)

- **Branch:** `stark/fp16-verify-ceiling` · **Student:** stark · merged 20:37Z (advisor drove merge) (LOCAL CPU-only analytic ceiling map over banked #204/#175/#184 + swept M_step; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `pqjnybbf` (group `fp16-verify-valid-ceiling`, finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `fp16_verify_ceiling_self_test_passes`=1 (a–f), `fp16verify_ceiling_at_lambda1`=306.44 (M_step=1.7), `mstep_crossover_ceiling_500`=1.041905, `fp16_verify_valid_by_construction`=1, ceiling {1.3→400.73, 1.7→306.44, 2.3→226.50}, `all_mstep_ceilings_below_500`=1, `lambda_min_fp16verify_clears_500`=∅ (unreachable sentinel, key-absent, passes selftest_d — not a NaN).
- **Primary:** `fp16_verify_ceiling_self_test_passes=1`. **Test:** `fp16verify_ceiling_at_lambda1=306.44`.
- **Key finding — fp16/bf16-verify is VALID-by-construction but its ceiling is structurally below 500.** cuBLAS fp16/bf16 GEMM has no M-dependent split-K ⇒ AR-M=1 and verify-M=K+1 produce identical argmax ⇒ greedy-valid with no kernel work. But the speed cost is fatal: λ=1 ceiling = 520.95/M_step, and fp16 verify is strictly slower than int4 Marlin (M_step>1 always), so the ceiling never reaches 500. The crossover M_step*=1.0419 is physically unreachable (swept band [1.3, 2.3]); the λ=1 column (E[T] saturated at the tree max, draft-independent) is the "no draft saves it" cap and sits at 400.73 even at the optimistic M_step=1.3. Self-test round-trips: M_step=1.0 reproduces the int4-spec ceiling 520.95 at λ=1 EXACTLY (a); ceiling·M_step=520.95 at every M_step (e); ceiling↓ in M_step (b), TPS↑ in λ (c).
- **Conclusion — narrows the Issue #211 valid-500 menu: lane-b is OUT.** fp16-verify is valid but capped <500, so no Blackwell draft can rescue it (confirms the human's own #211 framing — fp16's ceiling, not the draft, is the bottleneck). The surviving valid-500 routes are lane-a (wirbel #216 int4 batch-invariant kernel, double-gated) and the newly-surfaced MarginGate path (wirbel #223). Distinct from FP8-on-A10G deadness (fp16/bf16 IS valid here, just slow; FP8 isn't available on sm_86). Relayed to fern #185 (DROP lane-b from the valid-path menu) + lawine #221 (still pins the exact M_step + empirically confirms the fp16 batch-width token-identity premise, but the ceiling verdict is settled NO regardless of M_step) + Issue #211 (fp16 is not the cluster-unlock). stark → reseat.

## 2026-06-14 20:32 — PR #216: Compliant-kernel feasibility — is a custom batch-invariant int4 verify buildable under the budget? — 🟢 GREEN / LANE-A IS REAL BUT DOUBLY-CONDITIONAL: the off-the-shelf `VLLM_BATCH_INVARIANT=1` scoped to the verify GEMM is +31.41% (clears at NO physical λ), but a CUSTOM kernel fixing only the int4-Marlin split-K reduction order has a first-principles floor ~0.95% — inside the λ=1 budget (7.33%) — feasible iff (a) land #71 builds λ≥0.857 AND (b) the kernel lands near its floor — MERGED (bank-the-analysis), the pivotal Issue #211 verify-kernel gate

- **Branch:** `wirbel/kernel-feasibility` · **Student:** wirbel · merged this turn (advisor drove merge) (LOCAL CPU-only analytic feasibility bound over banked #213/#199/#184/#175; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `pc8g6s04` (group `compliant-spec-et-ceiling`, finished, NaN-clean). **Advisor-verified independently** (two subagents, `summary/` namespace, all match): PRIMARY `kernel_feasibility_self_test_passes`=1, `lambda_min_kernel_feasible`=0.8572, `verify_gemm_cost_share_of_step`=0.6066, off-the-shelf-scoped overhead 31.41%, custom-kernel floor 0.9455%, band [0.95%, 31.4%] (54.8× wide); round-trips #213 endpoints (budget@λ=1=7.332%, λ_crit=0.8345).
- **Primary:** `kernel_feasibility_self_test_passes=1`. **Test:** `lambda_min_kernel_feasible=0.8572`.
- **Key finding — the compliant batch-invariant int4 verify kernel (lane-a) is a REAL 500-path but doubly conditional, and it's a buildability PRIOR, not a proof.** Three nested results: (1) the off-the-shelf `VLLM_BATCH_INVARIANT=1` is the WRONG tool — even scoped to just the verify GEMM (not whole-model) it costs 31.41% (clears at NO physical λ; the whole-model tax is ~55× larger than needed); (2) a CUSTOM kernel fixing ONLY the int4-Marlin split-K reduction order to be M-invariant has a first-principles floor ~0.95% — comfortably inside the λ=1 budget 7.33% — so the gap between "off-the-shelf dead" and "custom feasible" is 54.8× and the question is purely *where in [0.95%, 31.4%] a real kernel lands*; (3) feasibility is doubly conditional — needs BOTH (a) land #71 builds self-KV recovery λ≥0.857 (below that even a FREE kernel misses 500, per #213's λ_crit) AND (b) the kernel lands near its 0.95% floor. #71's interim spine λ=0.997 clears comfortably at the floor; its pessimistic liveprobe λ̂=0.342 fails outright.
- **Decisive remaining uncertainty + the cheap path.** The 0.95% floor is *estimated*, not measured — the single number that converts the prior to a proof is a GPU microbenchmark of the verify GEMM at M∈{1..8}, fixed-vs-M-adaptive split-K schedule (~1–2 days, NOT a full-node run). This ARMs the human's #211 Blackwell decision: the verify-path question is gated on a cheap diagnostic, not on draft quality.
- **Conclusion — capstone of wirbel's compliant-spec lane (#199 ceiling → #213 budget-curve → #216 feasibility).** Pins lane-a as conditionally-real and identifies the cheap measurement that decides it. Relayed to the human (#211 Blackwell call, with a fresh researcher finding — MarginGate, a HIGHER-ranked valid-verify path the fleet had not priced) + fern #185. wirbel → **#223 (MarginGate budget — price the researcher's top-ranked valid-verify path, arxiv 2605.30218, P=0.45, against the #213 budget: derive `skip_rate_min(λ)`, the provable-stable-margin skip rate MarginGate must beat to clear 7.33%@λ=1; arms the `verify_flip_probe` GPU diagnostic)**.

## 2026-06-14 20:31 — PR #217: Launch-trigger reconcile — resolve 512.4 (N=1) vs 528.5 (best-of-N) into one pinned trigger — 🟢 GREEN / THE TENSION RESOLVES BY AXIS, NOT BY N — AND THE λ=1 CEILING DOES NOT CLEAR THE PRIVATE BAR: the unified T(N)=T_base+σ_sel·E[Z_(N:N)] reproduces #204's 512.41/514.63 at N=1 EXACTLY (N*=1, best-of-N HARMFUL); 512.41 is the PUBLIC-confidence GO trigger, 528.48 is the N-independent PRIVATE build target, and the physical ceiling 520.95 clears the public trigger but MISSES the private bar by +7.53 TPS (private clear only 0.744 at the ceiling) — MERGED (bank-the-analysis)

- **Branch:** `kanna/trigger-reconcile` · **Student:** kanna · merged this turn (advisor drove merge) (LOCAL CPU-only analytic reconcile over banked #204/#210/#207/#202/#191; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `vgovdrjc` (group `winners-curse-redraw-budget`, finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `trigger_reconcile_self_test_passes`=1, `n_star_launch`=1, both #204 anchors reproduced at N=1 (512.41 central / 514.63 worst), `lambda1_ceiling_clears_private_bar`=0, `private_bar_minus_ceiling`=7.5308 (528.48−520.95), private clear at ceiling 0.744 (mean 504.86, P95 LCB 492.70<500), `f_priv`=0.969107.
- **Primary:** `trigger_reconcile_self_test_passes=1`. **Test:** `private_bar_minus_ceiling=7.5308`.
- **Key finding — the 512.41-vs-528.48 tension is NOT a contradiction to be averaged; it is two DIFFERENT axes, and the second one does not clear at the physical ceiling.** The unified order-statistic trigger `T(N)=T_base+σ_sel·E[Z_(N:N)]` reproduces #204's 512.41 central / 514.63 worst-case at N=1 EXACTLY, and `n_star_launch=1` (best-of-N is HARMFUL — it raises the *seen* trigger for ZERO private-mean gain, consistent with kanna #210's flat-in-N private clear). Reading the two numbers correctly: **512.41** = the N=1 PUBLIC GO trigger (95%-confidence the *public* mean ≥ 500); **528.48** = the N-independent PRIVATE build target (= mu_safe_fresh/f_priv, f_priv=0.969107 — the public mean needed to clear *500-PRIVATE* at P95). **The λ=1 ceiling 520.95 clears the public trigger but MISSES the private bar by +7.53 TPS** (`lambda1_ceiling_clears_private_bar=0`; at the ceiling the private mean is 504.86, P95 LCB 492.70 < 500, private clear only **0.744**, not ≥0.95).
- **Honest self-correction (endorsed).** kanna caught that the PR's literal `528.48 = T_base + 23.61` was a premise error: 23.61 is a *composite* vs the #202 frozen-public bar 504.873, NOT the order-statistic winner's-curse tax (that's only 5.66 frozen / 8.60 fresh at N=5). The student corrected this transparently rather than reverse-fitting.
- **Conclusion — pins the launch trigger AND surfaces a NEW load-bearing private-bar row fern must carry separately.** A GO firing on the public reading (512.41, N=1) answers the PUBLIC-confidence question; it does NOT certify the PRIVATE bar at P95. Relayed to fern #185 (carry public-trigger-PASS vs private-bar-P95 as TWO separate flags; the private-bar row 528.48 > ceiling 520.95 is provisional under f_priv=0.969 and WIDENS to ~534 if f_priv≈0.957, the frontier's one hard paired draw 481.53→460.85). kanna → **#224 (private-bar reachability — ground f_priv [0.969 assumed vs #52-observed 0.957] and answer: what closes the +7.53 ceiling-vs-528.48 gap; is 500-private reachable at the physical λ=1 ceiling at all?)**.

## 2026-06-14 20:30 — PR #219: Issue #192 enforcement-reading calibration — convert kanna #114's 56.08% into per-reading pass-fractions A/B/C — 🟢 GREEN / THE #192 RULING MENU IS PRICED: strict per-sequence token-identity (reading A) passes only 0.125 (16/128) — 88% of served sequences are NOT token-identical and the NO-GO is ROBUST to maximal clustering (cap 0.439<0.5) — while the auto-scorer's actual PPL-only check (reading C) passes 100% — MERGED (bank-the-analysis)

- **Branch:** `denken/issue192-reading-calibration` · **Student:** denken · merged this turn (advisor drove merge) (LOCAL CPU-only analytic conversion over kanna #114's banked per-sequence split; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `0unwptbz` (finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `issue192_calibration_self_test_passes`=1, `strict_a_pass_fraction`=0.125 (16/128 OBSERVED), `strict_a_robust_to_clustering`=1 (max-clustering cap 1−p=0.439<0.5), `ppl_only_pass_fraction`=1.0 (served 2.3772≤2.42), per-token-θ CDF {θ=0→0.125, θ=0.05→0.141, θ=0.5→0.383, θ=1→1.0}, `applies_to_frontier_and_tree`=1.
- **Primary:** `issue192_calibration_self_test_passes=1`. **Test:** `strict_a_pass_fraction=0.125`.
- **Key finding — kanna #114's single 56.08% per-token divergence number becomes the actual menu the human's #192 ruling picks from, using #114's banked PER-SEQUENCE split directly (so strict-A is OBSERVED, not modeled).** Three readings: **(A) strict per-sequence token-identity** (all output tokens == M=1 greedy AR): pass = **0.125 (16/128)** — 88% of served sequences are NOT token-identical; this sits ~182 orders of magnitude ABOVE the iid floor (1−p)^512=1.14e-183 because flips arrive in a cascade (onset median 120/512, not iid per-token), and even the model-free maximal-clustering cap (1−p = 0.439) keeps the pass < 0.5 ⇒ `strict_a_robust_to_clustering=1`, NO-GO-under-strict-A is robust. **(B) per-token-θ** (per-sequence flip fraction ≤ θ): empirical CDF θ=0→0.125, θ=0.05→0.141, θ=0.5→0.383, θ=1→1.0. **(C) PPL-only** (the auto-scorer's ACTUAL check, PPL≤2.42): **100%** (served 2.3772). This is a FRONTIER-WIDE exposure — both the 481.53 frontier and the land #71 tree ride the same int4-Marlin spec basis (`applies_to_frontier_and_tree=1`).
- **Conclusion — arms the human's #192 ruling with the decision-menu, not a recommendation.** byteshark's strict-A board read → only 12.5% compliant today → the only strict-A-survivable >500 routes are the compliant-verify paths (MarginGate / custom batch-invariant kernel #216 / fp16-verify), all of which wirbel #223 / lawine #221 / stark #220 are now pricing. The ruling is the human's; this leg priced the menu. Posted to the message board (chiku-inu/openevolve/byteshark) alongside the DVR/MarginGate nuance. denken → **#225 (gate-2 confirmation runbook — consolidate the #205 SPRT + #212 AR(1)-ASN into the operational procedure that CONFIRMS measured λ̂_built ≥ 0.9780 both-bugs from ONE served run's q[2..9] accept data [65,536 positions]; decision rule λ̂_LCB≥0.9780⇒PASS, [0.857,0.978)⇒HOLD kernel-feasible-but-below-private-bar, <0.857⇒NO-GO)**.

## 2026-06-14 20:05 — PR #218: Inter-leg ρ grounding — validate the worst-case combined-σ launch trigger — 🟢 GREEN / FLAT ρ=+0.3 IS CONSERVATIVE: grounding the three launch-noise legs from physical sources (accept⊥hw≈0, hw⊥private≈0, only accept↔private mild +0.30) gives worst-case trigger 513.557 vs #204's flat-ρ 514.635 — recovers +1.077 TPS of λ=1 margin (worst-case margin +7.395) — MERGED by Morgan (bank-the-analysis)

- **Branch:** `ubel/launch-sigma-175-reconcile`→`interleg-rho` · **Student:** ubel · merged 20:05:12Z by Morgan (commit 6c19cfe) (LOCAL CPU-only analytic ρ-grounding over banked #204/#188/#190/#176/#191; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `0ug7vd7d` (group `launch-sigma-unit-rebase`, finished, NaN-clean). Logged from ubel's terminal result — self-test round-trips both #204 anchors machine-exact (err 0.0).
- **Primary:** `interleg_rho_self_test_passes=1` (6/6). **Test:** `go_trigger_grounded_worstcase=513.557` (vs #204 flat-ρ 514.635, −1.077 TPS).
- **Key finding — #204's flat worst-case ρ=+0.3-on-all-pairs is conservative; the grounded ρ-matrix is strictly less correlated.** From physical sources: ρ(accept,hw)≈0 [0,0,0.10] (acceptance is greedy-identical across pods — a model/content property; hardware timing is an orthogonal allocation/thermal draw; only a weak 2nd-order "more accepts→fewer steps→less CLT timing-averaging" coupling, bounded ≤0.10); ρ(accept,private)=+0.30 [0.10,0.30,0.50] — **the one real coupling** (an adverse private re-grade lowers acceptance → lowers TPS through the SAME channel; not +1 because σ_accept also carries domain-independent KV-recovery variance #190 and σ_private carries non-acceptance PPL-margin #176/#191); ρ(hw,private)≈0 [0,0,0.05] (pod ⊥ domain). Grounded combined σ band **[7.6113, 8.2423]** ⊂ #204's [7.5448, 8.8972]; GO trigger band [512.519 tight, 512.735 central, 513.557 worst]; λ=1 margin [+8.433, +7.395]. PSD at every corner (min eig ∈ [0.497, 1.0]).
- **Conclusion — capstone of ubel's launch-σ lane (#204→#207→#218).** `flat_03_is_conservative=True`, recovering +1.077 TPS of λ=1 margin (worst-case margin +7.395 vs #204's +6.318). The accept↔private coupling keeps grounded σ +0.0665 TPS above the pure-independent 7.5448 floor — the residual carried honestly, not assumed away. Relayed to fern #185 (carry the grounded [512.519, 513.557] trigger band, accept↔private as the only real correlation). ubel → **#222 (binding gate — does clearing the validity bar λ̂=0.9780 auto-clear the 513.557 speed trigger? which gate binds for land #71's single build-target)**.

## 2026-06-14 20:03 — PR #209: Frozen-vs-fresh regime — empirically pin the local re-benchmark variance (σ_hw or σ_draw?) — 🟢 GREEN / LOCAL HARNESS IS FROZEN: token-identity 1.0 across 8 fresh reloads + σ_reload 0.064 official-TPS (~87× below the σ_sample a FRESH regime would inject) ⇒ fern #185's conservative `mu_bar_frozen_p95`=504.87 default is empirically-confirmed; FRESH 499.08 NOT admissible from local evidence — MERGED (bank-the-analysis)

- **Branch:** `lawine/frozen-regime-local-pin` · **Student:** lawine · merged 20:03:17Z (advisor drove merge; commit 8a1ca22) (LOCAL fresh-reload profiling on the served stack; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `njx7n0gs` (group `frozen-regime-local-pin`, finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `frozen_regime_pin_self_test_passes`=1, `local_harness_is_frozen`=1, `token_identity_rate_across_reloads`=1.0, `sigma_reload_walltps`=0.0602, `f_resample_local`=0.0, `regime_local_code`=0 (FROZEN), `mean_reload_official`=481.56, `no_official_draw`=1.
- **Primary:** `frozen_regime_pin_self_test_passes=1`. **Test:** `local_harness_is_frozen=1`.
- **Key finding — the LOCAL re-benchmark harness is decisively FROZEN, on two independent legs.** (i) Byte-identical tokens across all 8 fresh LocalServer reloads of the deployed `fa2sw_precache_kenyan` stack (`token_identity_rate_across_reloads`=1.0, 0 divergent reloads, per-prompt SHA256 re-verified) — the token-identity contract self-check (fixed 128 prompts + deterministic greedy ⇒ no prompt/sampling resample). (ii) Run-to-run wall_tps σ_reload=0.0602 wall-TPS (CV 0.0133%) → 0.0638 official via the #180 bridge — ~87× BELOW the σ_sample=5.564 a FRESH regime would inject in quadrature. `f_resample_local`=clip((σ_reload²−σ_hw²)/σ_sample²,0,1)=0.0 → FRESH ruled out decisively. This empirically resolves kanna #202's load-bearing FROZEN-vs-FRESH regime assumption (the one fern #185 carries as a conservative default).
- **Honest caveat (credited).** σ_reload landed ~76× *below* σ_hw=4.864 — NOT "≈ σ_hw" — because the local A10G ran LOCKED clocks (sm_clock 1710 MHz, min=max every reload), so local timing noise sits far below the official a10g-small's unlocked/queue/thermal band that #188's σ_hw measured. lawine correctly scored self-test leg (c) as the one-sided frozen-consistency check (σ_reload not inflated *above* the σ_hw band) and reported the strict two-sided "reproduces σ_hw within CI" separately as False with the locked-clock explanation — the locked-clock result *strengthens* FROZEN (essentially zero run-to-run variance ⇒ no room for a hidden resample term). The right local→official error bar is the bridge-transfer band, not reload noise.
- **Scoping clarification (advisor, recorded on the PR).** The 0-divergence determinism leg shows the served-spec config is internally deterministic *run-to-run* — it does NOT show the config is greedy-*valid*. #114/#192's 56% divergence is between the served-spec output and the pure-AR-greedy *reference* (M=1 AR vs M=K+1 verify split-K float-order gap *within* each run), a different axis than cross-reload identity. So this is a clean FROZEN/determinism result, orthogonal to (NOT a resolution of) the #192 verify-kernel validity question.
- **Conclusion — confirms fern #185's conservative default.** `mu_bar_frozen_p95`=504.87 is the empirically-correct launch-bar default; FRESH 499.08 is not admissible from local evidence. The OFFICIAL-scorer regime stays kanna #202's human-gated two-official-draw pin, but the local result is strong corroboration under the shared token-identity contract. Relayed to fern #185 + kanna #206. lawine → **#221 (fp16-verify cost+validity — MEASURE the fp16/int4 step multiplier M_step + the fp16 batch-width token-identity locally; the empirical half of the Issue #211 valid-path question, complements stark #220)**.

## 2026-06-14 20:03 — PR #215: Deep-tail build-bar budget — min q[8..9] acceptance to clear the certified 0.9780 — 🟢 GREEN / DEEP-TAIL BUDGET 0.7875: given land #71's measured shallow spine λ(q[2..7])=0.997, the depth-aggregate clears the #208-certified bar iff reach-weighted deep-tail q[8..9] ≥ 0.7875 — NOT a free pass (shallow-only λ̂=0.9065 MISSES; coherent #193 β-proj 0.685 misses by 0.102, flips at β_crit=0.846) — MERGED (bank-the-analysis)

- **Branch:** `stark/deeptail-bar-budget` · **Student:** stark · merged 20:03:15Z (advisor drove merge; commit 920cb5d) (LOCAL CPU-only depth-decomposition over banked #208/#203/#191/#193; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `ccff87tb` (group `deeptail-bar-budget`, finished, NaN-clean, peak 12.0 MiB CPU). **Advisor-verified independently** (`summary/` namespace, all match): PRIMARY `deeptail_bar_budget_self_test_passes`=1 (a–e), `min_deeptail_lambda_q8q9_clears_bar`=0.787487, `w_mass_shallow_q2q7`=0.90921, `w_mass_deeptail_q8q9`=0.09079, `lambda_hat_shallow_only`=0.906482, `d_lambdahat_d_deeptail`=0.09079, `spine_value_where_deeptail_budget_hits_zero`=1.07564, `budget_vs_spine` {0.990→0.8576, 0.995→0.8075, 0.997→0.7875, 0.999→0.7675, 1.000→0.7574}.
- **Primary:** `deeptail_bar_budget_self_test_passes=1` (a–e, round-trips #208 threshold to 1e-9). **Test:** `min_deeptail_lambda_q8q9_clears_bar=0.787487`.
- **Key finding — the scalar #208 bar is now an actionable per-depth build target, and the deep tail is NOT a free pass.** Holding land #71's posted shallow-mid spine λ(q[2..7])=0.997, the depth-aggregate λ̂=Σ w_d·λ_d clears the certified 0.977978 bar iff the reach-weighted deep-tail over q[8..9] ≥ **0.7875**. Two-sided honesty: (1) the deep tail carries only ~9.1% of the q[2..9] reach mass (shallow/deep ratio 10.01×), so the budget sits far below the measured spine — *looks* like slack; (2) BUT with the deep tail collapsed to 0 the aggregate is `lambda_hat_shallow_only`=0.9065 — *below* the bar — so the deep tail genuinely carries build risk. The coherent #193 β-from-spine mechanism projection (continue β-decay from the 0.997 spine) gives deep-tail ≈ 0.685 → **MISSES the 0.7875 budget by 0.102 in λ**, with the GO/NO-GO flipping at **β_crit=0.846** — squarely inside #193's β construction range [0.616, 0.950] (primary β=0.765 misses). The launch genuinely hinges on land #71's *unmeasured* q[8..9].
- **Robustness + advisor call.** `budget_vs_spine` slope = −W_shallow/W_deep = −10.01 (each +0.001 spine buys down the deep-tail budget 0.010); `spine_value_where_deeptail_budget_hits_zero`=1.0756 (>1) ⇒ the deep tail is NEVER irrelevant — even a perfect spine 1.0 still needs deep tail ≥ 0.7574. **Advisor adopted the q[2..9] / head-excluded reading (budget 0.7875)** as the canonical deep-tail GO threshold — apples-to-apples with land #71's `lambda_spine_min_q2_q7` marker (the depth-1 head is the separately-anchored liveprobe λ̂₁, denken #193/#205 lane); the head-inclusive 0.7225 stays on record as stark's documented sensitivity arm.
- **Conclusion — capstone-of-the-capstone of the private-validity lane (#176→#191→#198→#203→#208→#215).** Converts the certified bar into the precise instruction for land #71: MEASURE q[8..9] (and its decay β) — don't infer it from the strong shallow spine. Relayed to fern #185 (carries 0.7875 as the binding deep-tail build target; drop per slope −10.01/unit if the spine firms above 0.997) + land #71 (the measurement to make; clears iff β ≥ 0.846). stark → **#220 (fp16-verify valid-path ceiling map — can ANY draft clear 500 without an int4 kernel? the analytic half of the Issue #211 valid-path question, complements lawine #221)**.

## 2026-06-14 19:49 — PR #212: AR(1)-corrected ASN — tighten the SPRT liveprobe realism band — 🟢 GREEN / AR-CORRECTED, #205's FLAT ×4.41 CONFIRMED CONSERVATIVE: folding #190's *decaying* within-prompt ACF into the SPRT partial-sum variance tightens the realism band 1.59–2.66× — E[N]_nogo drops from flat-loose 1,788 to 672 (AR(1) optimistic) / 1,125 (measured-ACF, data-grounded); realized (0.05,0.95) and bar 0.9780 untouched — MERGED (bank-the-analysis)

- **Branch:** `denken/sprt-ar-asn` · **Student:** denken · merged 19:49:48Z (advisor drove merge; commit 4d71e9b) (LOCAL CPU-only AR(1)/ACF correction over banked #205/#190/#197; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `b70053sw` (group `sprt-liveprobe-budget`, finished, 39 keys NaN-clean, peak 28.7 MiB CPU). **Advisor-verified independently** (`summary/` namespace, 8/8 match): PRIMARY `sprt_ar_self_test_passes`=1 (sub-tests a–f), `expected_n_nogo_ar`=672.34, `rho_lag1_190`=0.2583, `deff_flat_441`=4.4106 (↔#205), `deff_ar_asymptote`=1.6966, `expected_n_nearbar_ar`=24734.6, `expected_n_worstcase_ar`=40460.9, `flat_441_is_conservative`=1, `nan_clean`=1.
- **Primary:** `sprt_ar_self_test_passes=1` (6/6). **Test:** `expected_n_nogo_ar=672.34`.
- **Key finding — #205's flat ×Deff=4.41 over-counts the correlation length of a DECAYING ACF.** The running-LLR-sum variance is the AR-corrected partial sum `Var(Σ_n)=σ²·[n+2·Σ(n−k)ρ^k]`, not `n·σ²·Deff_flat`. Evaluated at #190's within-prompt cluster horizon, the effective design effect drops from the flat-exchangeable 4.4106 to measured-ACF 2.7743 (1.59× tighter) / AR(1) ρ^k 1.6584 (2.66× tighter, asymptote 1.6966). Since the AR correction rescales per-trial INFORMATION not drift, the ASN scales by the common multiplier → E[N]_nogo: IID floor 405 → AR(1) 672 → measured-ACF 1,125 → flat-loose 1,788 (near-bar/worst-case scale identically). NO double-count: the `flat_441` column reproduces #205's own banked `expected_n_sprt_nogo_realistic_icc`=1788.17 bit-for-bit, so the table inflates from the IID floor. The **75× collapse vs #197's fixed-N (30,455) is Deff-INVARIANT** (fixed-N reference and SPRT E[N] scale by the SAME cluster Deff → savings_ratio=75.12 under every Deff model). Realized (α,power)=(0.05,0.95) and bar 0.9780 untouched.
- **Honesty call (endorsed).** Student headlined the **data-grounded measured-ACF 1,125** as the realism point, NOT the rosier AR(1) 672 — because ρ(2)=0.168 ≫ ρ(1)²=0.067, so the empirical ACF decays SLOWER than pure AR(1) (the truth sits between AR(1) and flat, closer to measured-ACF). Also caught + corrected the PR's own hand-off template: 405 is the **IID-floor/tight** end (zero-correlation), not loose — the loose/conservative end is the flat 1,788; band orientation is [405 tight … 1,788 loose] with AR/measured-ACF as the interior. `flat_441_is_conservative=True`.
- **Conclusion — capstone of denken's liveprobe-budget lane (#197 fixed-N → #205 SPRT → #212 AR-correction).** The decaying-ACF correction SHARPENS the absolute realism band below the conservative flat 4.41 while leaving the 75× headline and the NO-GO-is-cheap conclusion intact; never reverses. Orthogonal to #192. Relayed to fern #185 (carry the measurement-cost row [405 IID → 672 AR → 1,125 realistic → 1,788 flat-loose], realistic point 1,125). denken → **#219 (#192 enforcement-reading calibration — convert kanna #114's 56.08% per-token divergence into per-reading pass-fractions A/B/C; the within-sequence strict-A pass-fraction IS denken's #190/#212 ICC machinery re-pointed at the #114 flip process)**.

## 2026-06-14 19:38 — PR #207: Launch-σ #175-reading reconcile — does robust-YES survive the larger sampling half-width? — 🟢 GREEN / ROBUST-YES SURVIVES: the two #175 readings are the SAME finite-sample TPS CI at DIFFERENT bench sizes (not √D-apart, not different axes); launch-correct = the smaller full-generation 5.178 = #204's basis, trigger 512.41/514.63 STANDS, λ=1 clears — MERGED (bank-the-analysis)

- **Branch:** `ubel/launch-sigma-175-reconcile` · **Student:** ubel · merged 19:38:44Z by Morgan (CPU-only analytic reconcile over banked #204/#175/#187/#190; no HF Job/submission/served-file change/official draw; BASELINE 481.53, 0 TPS, greedy/PPL untouched, no launch). W&B `17vi7fda` (group `launch-sigma-unit-rebase`, finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace): PRIMARY `reconcile_175_self_test_passes`=1 (5/5), `robust_yes_survives`=1, `lambda1_ceiling`=520.95, `trigger_central_hout`=512.41, `trigger_central_175sampling`=520.98, `ratio_175_readings`=2.106, `acceptance_1sigma_175sampling`=5.60.
- **Primary:** `reconcile_175_self_test_passes=1`. **Test:** `lambda1_clears_under_conservative_reading=0` (mechanically false but the 10.906 sub-bench reading is the WRONG quantity — flagged by ubel; this is a clean reconcile, not a regression).
- **Key finding — the larger #175 reading was a bench-size artifact, not a real axis.** The two readings (h_out 5.178 @ B=65536 vs #175-sampling 10.906 @ B=16384) are the SAME finite-sample TPS CI `HW=z·slope·σ_L/√N_steps` at different benchmark token budgets. The ~2.106 ratio decomposes to σ_L op-point ×1.0319 · bench-size √N ×2.0411 (√D≈2.10 match is coincidence to 0.286%); #175's own fixed-λ=1 readings (10.906@16384, 5.4531@65536) are exactly √4 apart — pure bench size. Reading 10.906 as 5.178·√D would DOUBLE-COUNT the design effect (√D is #190's ICC inflation applied ON TOP of the iid 10.906 to reach 22.905). The launch-correct quantity is the smaller full-generation 5.178 = #204's exact basis. h_out is already input/output de-duped via #187 (overlap 0.893).
- **Conclusion — the robust-YES from #204 survives the larger-half-width scare.** Under the launch-correct reading #204's GO trigger STANDS: 512.41 central / 514.63 worst-case, λ=1 ceiling 520.95 clears (+8.54 / +6.32). Pairs with kanna #210 (which raises the SAME trigger via the winner's-curse private correction — reconciled by kanna #217). Relayed to fern #185. ubel → **#218 (inter-leg ρ grounding — validate the worst-case combined-σ trigger; is #204's flat ρ=+0.3 conservative?)**.

## 2026-06-14 19:38 — PR #210: Winner's-curse re-draw — does best-of-N clear the binding PRIVATE bar? — 🟢 GREEN / NO, BUILD HIGHER: best-of-N is selection on non-replicating noise; private conditional clear is FLAT in N; to clear 500-private at P≥0.95 under best-of-5 the public build must reach 528.48 (+23.61 winner's-curse tax) — MERGED (bank-the-analysis)

- **Branch:** `kanna/winners-curse-redraw-budget` · **Student:** kanna · merged 19:38:42Z by Morgan (CPU-only analytic selection model over banked #194/#202/#200/#191; no HF Job/submission/served-file change/official draw; BASELINE 481.53, 0 TPS, takes no draws, authorizes no shot count, no launch). W&B `hwvv7nn1` (group `winners-curse-redraw-budget`, finished, NaN-clean, 260 numbers checked). **Advisor-verified independently** (`summary/` namespace): PRIMARY `winners_curse_self_test_passes`=1 (a–g), `delta_mu_winners_curse`=23.61, `n_star_private`=1, `private_clear_flat_in_n`=1, `p_private_clear_at_mu512p2`=0.312, `freeze_robust_512_survives_private`=0, `mu_bar_frozen_public_202`=504.87.
- **Primary:** `winners_curse_self_test_passes=1`. **Test:** `mu_bar_private_corrected=528.48`.
- **Key finding — best-of-N raises the PUBLIC number you SEE, never the checkpoint's true PRIVATE mean.** The launch trigger fires on max_i X_i, which overstates the replicable mean by σ_sel·E[Z_(N:N)] — luck that does NOT carry to the fresh private grade (Y ⊥ trigger). So P(Y≥500 | trigger) is EXACTLY flat in N; against the binding 500-PRIVATE bar (#191) the only lever is build higher, not re-draw more. To clear 500-private at P≥0.95 under a best-of-5 trigger the public build must reach μ_pub=528.48 — a +23.61 TPS winner's-curse tax over #202's public-only frozen bar 504.87. In FROZEN the ENTIRE best-of-N gain (#202: moves only σ_hw) is hardware luck that evaporates on a fresh private re-bench. Regime-invariant on the private column (FROZEN σ_sel=σ_hw=4.864 / FRESH σ_sel=σ_draw=7.391 both bound). The public−private GAP grows with N (0.79 → 0.88 at μ=505) — the silent over-optimism a reader of the public max would buy.
- **Conclusion — confirms N*=1 and tightens the build target.** At μ=512.2 (≈#204's trigger) the private clear is only 0.312, so #204's trigger does NOT survive the winner's-curse-corrected PRIVATE bar IF best-of-N is used — but the tax vanishes at N=1, reconciling with ubel #207 (kanna #217 pins this: T(1)=512.4 = #204, T(5)=528.5). Relayed to fern #185 (carry N*=1 + the 528.48 best-of-5 penalty). kanna → **#217 (trigger reconcile — resolve 512.4 N=1 vs 528.5 best-of-N into one pinned launch trigger for fern)**.

## 2026-06-14 19:38 — PR #213: Kernel-overhead budget vs λ — the compliant-spec 500 margin curve — 🟢 GREEN / BUDGET-OPENS-ONLY-ABOVE-LAMBDA-CRIT: the compliant-spec kernel-overhead budget opens from ≤0 at λ̂=0.342 to 7.33% at λ=1; zero-overhead first clears 500 at λ_crit=0.8345 (both-bugs); off-the-shelf #122 +51.78% clears at NO physical λ — MERGED (bank-the-analysis)

- **Branch:** `wirbel/kernel-budget-lambda` · **Student:** wirbel · merged 19:38:40Z by Morgan (CPU-only analytic E[T](λ)→TPS→budget curve over banked #199/#184/#193/#175/#169; no HF Job/submission/served-file change/official draw; BASELINE 481.53, 0 TPS, greedy/PPL untouched, no launch). W&B `5o7zcj8s` (group `compliant-spec-et-ceiling`, finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace): PRIMARY `kernel_budget_lambda_self_test_passes`=1 (6/6), `overhead_budget_at_lambda_1_both_bugs`=7.332% (round-trips #199's 536.66), `descent`=4.123%, `overhead_budget_at_lambda_hat_0342`=−16.74% (≤0, round-trips #199's floor-miss), `lambda_crit_both_bugs`=0.8345, `lambda_crit_descent`=0.9067, `off_the_shelf_122_clears_at_physical_lambda`=0.
- **Primary:** `kernel_budget_lambda_self_test_passes=1`. **Test:** `lambda_crit_clears_500_zero_overhead=0.8345`.
- **Key finding — the compliant 500-lane has a hard λ floor, and off-the-shelf never suffices.** Converted #199's binary ceiling/floor into the actionable `max_kernel_overhead_pct(λ)` curve (linear floor-spine→ceiling-spine blend `t(λ)=(λ−λ̂)/(1−λ̂)` through the same #175/#184 reach-DP, round-trips both #199 endpoints bit-exactly). Two deliverables: (1) **λ_crit=0.8345 both-bugs / 0.9067 descent** — below it NO overhead budget exists, even a FREE kernel misses 500; so land #71 must build λ above 0.8345 before kernel-dev matters. (2) **kanna #122's off-the-shelf +51.78% clears at NO physical λ** — stronger than ">1": the budget tops out at 50.10% (both-bugs) even at the probability-saturation wall, so `VLLM_BATCH_INVARIANT=1` exceeds even the theoretical-max budget. τ=0.9924 conservative corner shifts budgets −0.7–0.8pp (logged).
- **Conclusion — pins the compliant-spec build target.** The only compliant 500-lane needs BOTH λ_achieved > 0.8345 AND a CUSTOM batch-invariant verify kernel under `max_kernel_overhead_pct(λ_achieved)`. Carries #199's three optimisms (rank-1 coverage 0.7304, λ-realism, zero-overhead) as a noted band. Relayed to land #71 + fern #185 + issue #192. wirbel → **#216 (kernel feasibility — is a custom batch-invariant int4 verify buildable under the budget? bound it between the FP floor and #122's off-the-shelf ceiling — the bold "is lane-a real" question)**.

## 2026-06-14 19:26 — PR #208: Multi-vertex realizability — is 0.9780 the worst-case over all blends? — 🟢 GREEN / 0.9780 STANDS: non-Latin-script IS the maximizing vertex over ALL realizable domain blends (optimum_exceeds_nls=False, resid 0.0 vs #203) — the last argued-from-construction assumption now closed by explicit optimization — MERGED (bank-the-analysis)

- **Branch:** `stark/multivertex-realizability` · **Student:** stark · merged 19:26:59Z (advisor drove merge) (CPU-only analytic realizability LP over banked #176/#198/#203; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch, authorizes nothing). W&B `wi4gxxx8` (group `private-drop-shape-robustness`, finished, NaN-clean, peak 46.82 MiB CPU). **Advisor-verified independently** (`summary/` namespace): PRIMARY=1 (all six sub-tests a–f=1), `both_bugs_bar_worstcase_blend`=0.9779783, `optimum_exceeds_nls`=0, `nogo_robust_worstcase_blend`=1, `bar_0978_stands`=1, `nan_clean`=1.
- **Primary:** `multivertex_self_test_passes=1` (6/6). **Test:** `both_bugs_bar_worstcase_blend=0.977978` (resid 0.0 vs #203 single-axis).
- **Key finding — 0.9780 is the TRUE worst-case private go-bar over ALL realizable domain blends, proven not argued.** The reach-weighted deficit Σ w·δ is LINEAR in the blend weights → the worst realizable blend is always a pure vertex (single axis), confirmed by a 200k-pt Dirichlet interior sweep (sweep max 2.334pp < vertex max 2.349pp). Over #176's six decode-drop-realizable axes at #203's reach-weights, non-Latin-script (NLS) is the unique argmax: it pairs the most front-loaded SHAPE with the largest realizable deficit MASS (Σδ=0.04169 at the 4.3% decode-drop calibration) → `max_weighted_deficit_pp`=2.349, runner-up code 2.331 (margin **+0.018pp**), bar 0.977978. β-stable (NLS argmax across β∈[0.6165,0.9496], worst-case bar band **[0.977978, 0.978015]**). NO-GO survives the true worst case (floor→bar gap **0.636** in λ; realistic floor λ̂=0.342 ≪ 0.978).
- **Methodology call (endorsed).** Stark rejected the PR's literal fixed-mass Σδ=0.04169 constraint as a degenerate counterfactual — every #176 axis was calibrated to decode-drop=GT-4.3%, NOT to a common Σδ, so forcing equal mass scales axes off-calibration into non-realizable deficits (math ×4.98, long-context sign-flip). Headlined the physically faithful natural-mass polytope, kept fixed-mass as a flagged SECONDARY arm (its only "winners" are super-NLS shapes that go private-UNREACHABLE — a strictly STRONGER NO-GO, per #203's c_crit=−1.672). Both framings agree: no realizable blend has a finite bar > 0.9780. Followed the PR's own "state the constraint set explicitly" instruction.
- **Conclusion — closes the private-validity lane (#176→#191→#198→#203→#208).** The last open "argued-from-construction" assumption (NLS is worst over all blends, not just the single measured axis) is now closed by explicit LP optimization. Only residual scope: the polytope is #176's six MEASURED axes — a genuinely new organizer domain could add a vertex (LP ingests it directly), and the one unmeasurable input stays an organizer tree-stack re-run on the real private set. Relayed to fern #185 (binding bar 0.9780, robustness band [0.977978, 0.978015]) + land #71 (per-rung `q_adv[d]/q_pub[d]` co-log → fern reads the ACTUAL measured blend and replaces worst-case with the exact bar). stark → **#214 (reseat — next private-validity / launch-readiness lever)**.

## 2026-06-14 19:06 — PR #199: Compliant-spec E[T] ceiling — can batch-invariant verify clear 500? — 🟢 GREEN / YES but NOT FREE: a token-identical batch-invariant int4 verify CAN clear 500 (ceiling 536.66, LCB 525.73) only if the compliant kernel holds ≤7.33% overhead — MERGED (bank-the-analysis)

- **Branch:** `wirbel/compliant-spec-et-ceiling` · **Student:** wirbel · merged 19:06:38Z by Morgan (CPU-only analytic E[T]→TPS projection over #184/#175/#169 + the pinned composition; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch). W&B `wdyqnx3g` (group `compliant-spec-et-ceiling`, finished, NaN-clean, 31 scalars). **Advisor-verified independently** (`summary/` namespace, 9/9 match): PRIMARY=1, ceiling 536.659, floor 416.307, et_ceiling 5.21888 / et_floor 4.04848, clears_500=1, LCB 525.729, rank1_coverage 0.730444, max_kernel_overhead 7.332% — artifact type `validity` (no TPS benchmark output).
- **Primary:** `compliant_spec_et_self_test_passes=1` (5/5). **Test:** `compliant_spec_tps_ceiling=536.66`.
- **Key finding — the #192 lane-a answer: a compliant spec 500-lane EXISTS, but only behind a hard kernel.** A token-identical batch-invariant int4 verify can clear 500 (both-bugs zero-overhead ceiling 536.66, LCB 525.73>500), but the path is BRACKETED not point-measured: floor 416.31 (MISS) → ceiling 536.66 (CLEAR), and the verdict hinges on a kernel-dev budget — the compliant verify may inflate per-step cost by ≤ ~7.33% (both-bugs) / ~4.12% (descent-only) and still clear. The only off-the-shelf datum, kanna #122's `VLLM_BATCH_INVARIANT=1` (+51.78%), blows that budget ~7× (and isn't token-correct). Three optimisms inflate the ceiling (rank-1 coverage 0.7304 over-counts the true compliant accept; λ=1 vs realistic λ̂=0.342; zero overhead). openevolve oracle cross-check reproduces the depth-1 anchors (522/538 vs 520.6/536.7).
- **Conclusion — pairs with lawine #196 to complete the #192 picture.** Lane-b (#196, empirical, spec OFF): no compliant non-spec 500-lane (floors 165). Lane-a (this, analytic, spec ON + batch-invariant verify): a compliant spec 500-lane EXISTS behind a <~7.3% kernel. Under strict #192 the ONLY compliant 500-route is this batch-invariant verify kernel — the deployed batch-VARIANT spec stack (kanna #114, 56% divergence) and dropping speculation are both out. Relayed to #192 + fern #185. wirbel → **#213 (kernel-overhead budget vs λ — his own follow-up #3, the `max_kernel_overhead_pct(λ)` curve so land #71 reads the budget at its achieved λ)**.

## 2026-06-14 19:06 — PR #205: SPRT liveprobe budget — expected-N early-stop vs #197's fixed-N 30k — 🟢 GREEN / ~75× COLLAPSE: realistic sequential cost to certify the likely NO-GO is E[N]=405 (vs fixed-N 30,455), Deff-invariant — MERGED (bank-the-analysis)

- **Branch:** `denken/sprt-liveprobe-budget` · **Student:** denken · merged 19:06:36Z by Morgan (CPU-only analytic SPRT/ASN over banked #197/#190/#191; no HF Job/submission/served-file change/official draw; BASELINE 481.53, 0 TPS, no launch). W&B `eijqklu2` (group `sprt-liveprobe-budget`, finished, NaN-clean, 44 scalars). **Advisor-verified independently** (`summary/` namespace, 10/11 match 3+ sig figs): PRIMARY=1, A/B=±2.9444, E[N]_nogo=405.42, nearbar=14,915.06, worstcase=24,398.04, FSS anchor `n_fixed_z95_197`=30,455.40, realized (α,power)=(0.05,0.95), β_crit 0.96488, bar 0.97801. (Logged `expected_n_sprt_nogo`=405.42; the prose's 405.27 = `asn_beta_0p765`=405.274 — a 0.04% nit, the 75.12× ratio holds either way.)
- **Primary:** `sprt_budget_self_test_passes=1` (6/6). **Test:** `expected_n_sprt_nogo=405`.
- **Key finding — the fixed-N 30k is a worst case, not the realistic cost.** Because the grounded NO-GO truth sits far below the bar in λ-equivalent space (β=0.765 → λ_eq=0.539 ≪ bar 0.978, private_LCB 419.6≪500), the per-trial Wald LLR drifts hard to the NO-GO boundary → SPRT concludes in ~405 trials (a build clearly below the bar is rejected cheaply). Shallow-heavy Neyman weighting (#197) means depths {2,3,4} carry 65.3% of the decisive info — a {2,3,4} screen rejects most clear-NO-GO builds before any depth-7–9 probing. Only a build genuinely AT the bar is expensive (ASN peak 24,398 ≤ 30k). The no-early-stop limit round-trips #197's 30,455 exactly. The 75.12× saving is Deff-invariant (same ratio under either FSS anchor); realistic-ICC path E[N]=1,788 also logged.
- **Conclusion:** prices the certification COST, not the verdict (bar/NO-GO unchanged). Relayed to fern #185 (carry the (E[N], OC) tuple — nogo 405 / nearbar 14,915 / worstcase 24,398, realized (0.05,0.95) — as the measurement-cost row replacing #197's fixed-N). denken → **#212 (AR(1)-corrected ASN — his own follow-up #2, fold #190's decaying ACF into the partial-sum variance to tighten the conservative ×4.41 band)**.

## 2026-06-14 19:02 — PR #206: Frozen-regime cost crossover — re-price #200 build-vs-redraw under #202's frozen bar — 🟢 GREEN / build-higher DOMINATES 7.25× wider: under frozen the crossover shifts hard to building higher; build-to-512.2/N=1 is the regime-invariant minimax hedge — MERGED (bank-the-analysis)

- **Branch:** `kanna/frozen-cost-crossover` · **Student:** kanna · merged 19:02:52Z (advisor drove merge) (CPU-only analytic re-pricing over banked #200/#202/#194/#188; no HF Job/submission/served-file change/official draw; BASELINE 481.53, 0 TPS, no launch, authorizes no shot count). W&B `gk6053y7` (group `frozen-cost-crossover`, finished, NaN-clean, 59 scalars). **Advisor-verified independently** (`summary/` namespace): PRIMARY=1, test 2.38544, anchors `crossover_fixed_frozen`=0.41921 / `crossover_sequential_frozen`=3.39395 / `f_where_redraw_competitive`=0.84553 (= #202 `frozen_fraction_breakeven` 0.846, `aligns_with_202_breakeven`=True) — the `*_tps` fields are analytic inputs, no benchmark artifact.
- **Primary:** `frozen_cost_self_test_passes=1`. **Test:** `build_higher_dominates_below_b=2.3854`.
- **Key finding — freeze shifts the build-vs-redraw crossover hard toward build-higher.** Under frozen sampling, redrawing at the bar beats only σ_hw (43% variance), so forcing P≥0.95 by re-drawing needs N=30 shots (vs 5 fresh) — collapsing the per-shot crossover slope (`c*_fixed` 3.039→0.419·b, ÷7.25) and widening the build-higher-dominates region 7.25× (`build_higher_dominates_below_b` 0.329→2.385). The deliverable is the regime-INVARIANCE of build-to-μ=512.2/N=1: a single draw shares the same marginal σ_draw in both regimes, so N=1 is the minimax-regret hedge against the unpinned harness regime (same cost + same P≥0.95 clear either way), whereas best-of-N-at-bar's worst-case regret blows to 29 shots' GPU-$ at b→0 under frozen. Partial-freeze c*(f) bridges monotone f=0→1; the f=0.846 break-even reproduces #202 exactly.
- **Conclusion:** fern #185's budget row defaults to build-higher/N=1; budget best-of-N only if the regime is empirically confirmed FRESH (which lawine #209 now pins locally). kanna → **#210 (winner's-curse — does best-of-N even help against the binding PRIVATE bar #191, or is the selected public max non-replicating luck)**.

## 2026-06-14 18:54 — PR #196: Compliant-lane floor — non-spec int4 greedy-exact serve TPS — 🟢 GREEN / STRUCTURAL_GAP_SPEC_EXISTENTIAL: the #192-compliant non-spec M=1 AR floors at 165.44 official TPS (−66.9% vs 500); no compliant non-spec 500-lane — speculation buys +316 TPS — MERGED (bank-the-analysis)

- **Branch:** `lawine/compliant-lane-floor` · **Student:** lawine · merged 18:54:07Z by Morgan (LOCAL repeated-reload wall-tps measurement, non-spec int4 M=1 AR token-identical to plain greedy AR by construction; no HF Job/submission/served-file change/official draw; BASELINE 481.53, 0 TPS, no launch). W&B `y4tavh9p`+`ekds1cy5` (advisor-verified, 10/10 metrics match, NaN-clean 81+36 scalars).
- **Primary:** `nonspec_floor_self_test_passes=1` (5/5). **Test:** `nonspec_official_tps_est=165.44`.
- **Key finding — there is NO compliant non-spec 500-lane, and the gap is structural.** The #192-compliant serve (non-spec int4 M=1 AR, token-identical to plain greedy AR by construction: `nonspec_token_identity_rate=1.0`, 0 divergences / 65,536 tokens × 3 fresh reloads, PPL 2.37656 ≤ 2.42, 128/128) floors at 156.05 wall_tps → 165.44 official-comparable (σ_hw band [160.6, 170.3]) — −334.6 TPS / −66.9% below 500. The non-compliant speculation in the deployed 481.53 stack therefore buys +316.1 TPS (191%) over the best provably-compliant serve. Clean one-lever manifest diff (`SPECULATIVE_CONFIG` off, int4/precache/split-KV/lm_head byte-identical, serve log confirms `speculative_config=None` + Marlin int4 ON); wall→official bridge recovers the 481.53 anchor to −0.024%; CV 0.0085% over N=3.
- **Conclusion — pairs with wirbel #199 to price strict #192 enforcement.** Strict literal token-identity kills the entire 316-TPS speculation premium → the only compliant 500-route becomes wirbel #199's batch-invariant int4 verify kernel (the spec path is existential for the target). Relayed to fern #185 (compliant fallback ceiling ~165 official TPS) + issue #192. lawine → **#209 (FRESH/FROZEN local regime-pin — extend this reload harness to settle kanna #202/#206's load-bearing regime question)**. Logging nit: the verdict string `STRUCTURAL_GAP_SPEC_EXISTENTIAL` was not logged as a W&B scalar (every number that justifies it is).

## 2026-06-14 18:48 — PR #203: Private-bar shape-robustness — is 0.9780 deficit-shape-invariant? — 🟢 GREEN / SHAPE-SENSITIVE-IN-VALUE but NO-GO-ROBUST: bar tracks reach-weighted deficit Σw·δ, non-Latin-script is the realizable worst case (0.9780), deeper shapes LOOSEN it, NO-GO survives every shape — MERGED (bank-the-analysis)

- **Branch:** `stark/private-drop-shape-robustness` · **Student:** stark · merged 18:48:57Z (CPU-only analytic synthesis over banked #176/#191/#198 curves + #193's λ(depth) mechanism; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch authorized). W&B `hexhagf6` (group `private-drop-shape-robustness`, finished, NaN-clean — `nan_clean=1`). **Advisor-verified independently** (`summary/` namespace; PRIMARY=1 with all 5 legs `selftest_a..e`=1; `worstcase_delta_vs_198=0.0` EXACT, `worstcase_delta_vs_191=−3.3e−5`, `nogo_robust_all_shapes=1`, `smallest_floor_to_bar_gap=0.538`).
- **Primary:** `shape_robustness_self_test_passes=1` (a–e). **Test:** `both_bugs_bar_worstcase_shape=0.9779783`.
- **Key finding:** closes the single-shape assumption #198 carried. The both-bugs private bar is shape-**SENSITIVE in value** (a monotone function of the reach-weighted deficit `Σ_d w_d·δ_d`; the tree reach-weights `w_d_at_bar=[0.413,0.336,0.271,0.240,0.172,0.103,0.090]` fall **4.6×** shallow→deep, set by #193's β-decay) but **0.9780 is the worst-case over REALIZABLE adverse shapes**: #176's non-Latin-script vertex (c=−1.0, Σw·δ=2.349pp) already MAXIMIZES Σw·δ — the exact quantity #176's adversarial-vertex search was built to maximize — so the single measured shape is the realizable worst case, not an arbitrary sample (anchor reproduces #198's coupled bar 0.977978 to machine zero). Flatter/deeper shapes redistribute deficit OFF the high-weight shallow rungs → **LOOSEN** the bar (0.945 flat, 0.880 deepest). The only way to tighten past 0.978 is a counterfactual MORE front-loaded than non-Latin-script, which saturates at the full-recovery ceiling (`c_crit=−1.672`) then goes private-**UNREACHABLE** — a STRONGER NO-GO, not a higher finite bar.
- **Conclusion:** REFUTES the #198 worry in the safe direction — a DEEP private deficit does NOT raise the bar (it lowers it). The realistic-floor NO-GO (λ̂₁=0.342 ≪ bar) survives **every** shape (tightest gap **0.538 in λ** at the deepest shape) → `nogo_robust_all_shapes=True`, shape assumption is no longer load-bearing for the NO-GO. Relayed to fern #185 (use 0.9780 if land #71's measured deficit is shallow-or-flat Σw·δ≤2.349pp, else the bar rises toward UNREACHABLE) + land #71 (co-log per-rung `q_adv[d]/q_pub[d]`, d=1..9, riding denken #197/#205's recovery-λ ladder — one measurement, two readouts). Honest residual: non-Latin-script-is-worst is argued from #176's single-axis construction, not yet proven over all domain BLENDS. Orthogonal to #192. stark → **#208 (multi-vertex realizability — an LP over #176's 6 banked per-axis deficits to prove 0.9780 is the worst case over ALL blends, + β-robustness of the reach-weights)**.

## 2026-06-14 18:42 — PR #204: Launch-σ clean-1σ unit rebase — does λ=1 clear 500 at P95 centrally? — 🟢 GREEN / YES, central AND worst-case: #201's knife-edge was a UNITS BUG (acceptance leg was a z=1.96 half-width); clean trigger 512.41/514.63 both below the λ=1 ceiling — MERGED (bank-the-analysis)

- **Branch:** `ubel/launch-sigma-unit-rebase` · **Student:** ubel · merged 18:41Z (CPU-only analytic re-basing over ubel's OWN banked #201 curve + #194/#195/#190 σ sources; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch authorized). W&B `m7vwuus2` (group `launch-sigma-unit-rebase`, finished, NaN-clean). **Advisor-verified independently** (`summary/` namespace, 35 scalars): all 16 reported values match, 6/6 self-test booleans (`self_test_a..f`)=1, both anchors machine-zero (`anchor_err_195_dedup=0.0`, `anchor_err_194_breakeven=0.0`), zero NaN/Inf.
- **Primary:** `unit_rebase_self_test_passes=1` (6/6: convention A≡B, single-leg textbook P95, idempotent-on-1σ, #194 break-even survives, worst-case≥central, NaN-clean). **Test:** `mu_clears_500_clean_central=512.4101`.
- **Key finding — #201's GO trigger was a UNITS BUG, not physics.** The dominant acceptance leg (11.170 TPS) was a z=1.96 two-sided **half-width** (traced footing-preserving: #195's #175-sampling source string → #187 de-dup `h_in`⊕`h_out`→5.31870 → #190's dimensionless √D=2.100 ratio = 11.17004 HW), but #201 mixed it into the quadrature beside the 1σ σ_hw/σ_priv legs and THEN applied z₁=1.645 in `LCB=μ−z₁·σ` — double-counting z on the variance-dominant leg. Clean fix divides by z₂ (11.170→**5.6991 1σ**) ⇒ combined σ **12.2153→7.5448** central / **13.7956→8.8972** worst-case. Cross-checked two ways: convention-A (all-1σ-then-×z, #194 basis) ≡ convention-B (all-half-width, #190 basis) give the **identical** LCB (err 0.0, central and at the ρ=+0.3 corner; hypot is 1-homogeneous) — #201's bug was a *third, inconsistent* basis.
- **The resolved verdict (deliverable):** clean GO trigger **μ ≥ 512.41 central / 514.63 worst-case**, BOTH below the λ=1 ceiling 520.95 ⇒ **λ=1 clears 500 at P95 CENTRALLY (+8.54) AND worst-case (+6.32)**. #201's central↔worst-case straddle of the ceiling (central +0.86, worst-case NOT clearing) was the units artifact. The σ-inflation MECHANISM (de-dup × realistic-ICC, 7.26→~12) is unchanged and robust; only the final unit footing was wrong.
- **Honest self-correction (good):** `rebase_direction_matches_prediction=0` — ubel's #201 scoping predicted `σ_hw·(z₁−1)=+3.14` (more conservative); actual **−7.68** (less). Mechanism ubel correctly named: the heuristic anchored on the *small* σ_hw leg moving up, but the mis-based leg is the *dominant* acceptance leg moving down. Wrong leg, wrong sign, ~2.4× wrong magnitude — reported transparently.
- **The one residual (ubel flagged, → reseat):** the clean acceptance magnitude (de-dup 5.31870 HW → 2.7137 1σ) traces to #187's `h_out`=5.178, but #195/#190 carry a LARGER #175-sampling half-width 10.906 — two readings of the SAME #175 CI differing by ~√D. If 10.906 is the launch-correct iid half-width, the de-duped acceptance magnitude (hence the trigger) moves UP. The footing fix is correct regardless; this is a *which-banked-quantity* audit ⇒ **ubel reseated → #207 (launch-σ #175-reading reconcile — does the robust-YES survive the larger reading?)**.
- **Conclusion:** the launch-σ question resolves from #201's PROVISIONAL knife-edge to a robust YES at λ=1 (central + worst-case both clear), modulo the #175-reading audit (#207). **fern #185** wires the clean trigger 512.41/514.63 (retires #201's 520.09/522.69); **land #71** co-log (n=385) now *tightens* a YES rather than rescuing a NO. **Launch still HELD** on the three hard gates (land #71 build · measured λ̂ ≥ 0.9780 q[2..9] direct · issue #192 ruling — no human reply yet); this leg authorizes no draw, no launch. ubel launch-σ lane (#148/#169/#181/#188/#195/#201/#204→#207).

## 2026-06-14 18:30 — PR #202: Frozen-sampling re-draw budget — does best-of-N beat down all of σ_draw or only σ_hw? — 🟢 GREEN / FROZEN regime beats down ONLY σ_hw (66% of scatter): N=5@bar gives P=0.81 not 0.97, conservative bar 504.87, but μ=512.2 stays freeze-robust at N=1 — MERGED (bank-the-analysis, parallel advisor)

- **Branch:** `kanna/frozen-sampling-redraw-budget` · **Student:** kanna · merged 18:30:50Z by the parallel advisor (CPU-only pure-Python analytic over #194/#200's banked σ-decomposition; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch authorized — a human still approves the spend AND confirms the harness behavior). W&B `533jd6l1` (group `frozen-sampling-redraw-budget`, finished, NaN-clean — `metrics_nan_clean=1`, 431 numbers checked; reproduces #194's banked `frozen_probe` to **0.0 abs err**).
- **Primary:** `frozen_budget_self_test_passes=1` (a–f). **Test:** `mu_bar_frozen_p95=504.87`.
- **Key finding:** stress-tests #194's load-bearing FRESH assumption (its own §5d flag). `σ_draw²=σ_sample²+σ_hw²` (5.564²+4.864²=7.391²). **FRESH** (re-benchmark re-randomizes the 128 prompts ⇒ best-of-N beats down the FULL σ_draw — the #194 premise). **FROZEN** (re-benchmark RE-USES the fixed 128 prompts under deterministic greedy ⇒ the per-checkpoint sampling bias `b~N(0,σ_sample²)` is COMMON across shots, only HF-Job timing `ε_hw~N(0,σ_hw²)` re-draws ⇒ best-of-N beats down ONLY **σ_hw=4.864 = 65.8% of one-σ / 43.3% of variance**). At the μ=500 bar FROZEN best-of-5 gives **P=0.810** (NOT fresh's 0.969) → #194's N=5-at-bar does NOT reach P≥0.95 under freeze. To restore P≥0.95 at N=5 the build rises to `mu_bar_frozen_p95`=**504.87** (freeze-tax **+5.79 TPS** vs the fresh-N=5 bar 499.08). The σ_sample-governed operationally-safe ceiling is Φ((μ−500)/σ_sample) (a reported-max N→∞→1 only via a one-in-N hardware-lucky allocation that won't replicate / fails private re-bench). Partial-freeze breakeven `frozen_fraction_breakeven`=**0.846** (84.6% of σ_sample must re-randomize for N=5 to hold @bar). Sequential @bar: E[shots] **2.34 frozen** vs 1.94 fresh, exhaust-without-clear **19.0% frozen** vs 3.1% fresh; uncapped quota 30 frozen vs 5 fresh (frozen-bad checkpoints rescued only by HW luck).
- **One PR expectation INVERTED (reported honestly):** freeze does NOT "raise the bar above 512.2". Because **N=1 is regime-invariant** and μ_safe=512.157 already clears at P≥0.95 with one shot, `delta_mu_frozen`=**−7.28** (the frozen N=5 bar 504.87 sits BELOW the fresh N=1 safe point) → `n_shots_frozen_at_512`=**1** (=#194's N=1). The freeze penalty is confined to **low-μ, best-of-N-reliant** plans.
- **Conclusion:** the challenge's contract (fixed 128 prompts + greedy token-IDENTITY ⇒ same tokens every run ⇒ σ_sample cannot re-randomize, only HW timing does) leans **FROZEN**, so FROZEN is the **conservative default** for budgeting until the human confirms the harness re-draw behavior (which regime applies stays the harness-owner's open question, like #192 enforcement). Net guidance for the `Approval request: HF job`: EITHER confirm the harness re-randomizes prompts before trusting N=5 at μ=500, OR build clear of the FROZEN bar (μ≥504.9 for N=5, or μ≥512.2 for the freeze-robust N=1) rather than leaning on best-of-N against a frozen bias. fern #185 carries the conservative `mu_bar_frozen_p95`=504.87 (not the fresh 499.08) as the multi-shot row; cheapest decisive pin = two official re-draws of one checkpoint (differ ≫σ_hw ⇒ FRESH; agree within ~σ_hw ⇒ FROZEN) — human-gated. Orthogonal to #192. kanna draw-budget lane (#159/#188/#194/#200/#202).

## 2026-06-14 18:28 — PR #197: Liveprobe depth-budget — which depths × N for a decisive private GO/NO-GO — 🟢 GREEN / Neyman shallow-heavy budget 30,455 trials @λ=1, full-ladder REQUIRED, depth-1-only a FALSE GO (85 TPS), mechanism CANNOT clear private bar at β=0.765 — MERGED (bank-the-analysis)

- **Branch:** `denken/liveprobe-depth-budget` · **Student:** denken · merged 18:28:59Z (CPU-only analytic synthesis over banked #193/#187/#191/#183 mechanism; no HF Job/submission/served-file change/official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch authorized). W&B `wqr94io4` (group `liveprobe-depth-budget`, finished, NaN-clean — `nan_clean=1`). Advisor-verified independently (`summary/` namespace; PRIMARY=1 with all 5 legs `selftest_a..e`=1; decision scalars all match — `min_depths_for_decisive_int=9`, `false_go_risk_depth1_only=1`, `mechanism_can_clear_private_bar=0`, `depth1_plus_2_suffices=0`, `depth1_overstatement_tps=85.21`, efficiency 1.4337).
- **Primary:** `depth_budget_self_test_passes=1` (5/5). **Test:** `total_trials_for_decisive_private=30455.40` (@λ=1, decisive margin 0.022).
- **Key finding:** the concrete liveprobe spec land #71's harness was missing. (1) **Neyman allocation on the E[T] functional** (not on a degenerate aggregate λ̂, which would dump all trials on one depth): per-depth physics weights `a_d=∂E[T]/∂q_d` fall **15×** across the ladder → the budget is **shallow-heavy**, `N_d[1..9]=[0, 7873, 6428, 5581, 4268, 2982, 1837, 1044, 442]` (depth-1 pinned 0 — deployed; depth-2 gets **18×** depth-9), total ≈**30,455** to decisively certify best-case λ=1 against the private bar 0.9780 (cost scales quadratically near the bar: λ=0.98 → 3.8M trials). (2) **Full ladder is MANDATORY** — a 2-depth β-fit leaves β unidentified (CI [0.0153, 38.3]); `min_depths_for_decisive`=full-ladder, land must probe depths 2..9 DIRECTLY, no shortcut. (3) **Depth-1-only is a FALSE GO worth 85 TPS** — at λ̂₁=1.0 the naive-flat read claims a private GO (504.9≥500) while the true β-decayed mechanism is a hard NO-GO (419.6). (4) **Structural cross-cut `mechanism_can_clear_private_bar=False`:** at the grounded β=0.7651 (denken #193), even *perfect* depth-1 recovery yields private_LCB **419.6 ≪ 500** — so NO real build clears the private bar; the 30k budget really sizes a **β≈1 confirmation across the full ladder**, not a point-λ̂ check (`β_crit_depth1_sufficient=0.9649`).
- **Conclusion:** structurally **negative for the launch** — exactly the rigorous validity finding we bank: the private GO hinges on **confirming β≈1 across the ladder** (no salvage staleness), not on any point λ̂, and the depth-1-only false GO (85 TPS) is the silent mis-certification the harness must avoid. Relayed to land #71 (measure q[2..9] directly, Neyman shallow-heavy per the budget; a depth-1-only "clear" is a false GO) + fern #185 (consume the per-depth budget + full-ladder requirement + decisive margin ≤0.022 in λ; the GO hinges on β≈1 not a point λ̂). Capstone of denken's measurement-design lane (#178→#183→#187→#193→#197). Orthogonal to #192. denken → **#205 (SPRT liveprobe budget — the sequential/expected-N analog: an early-stopping Wald test should certify the likely NO-GO far below the fixed-N 30k)**.

## 2026-06-14 18:20 — PR #201: Launch-σ closure — fold #195 de-dup × #190 realistic-ICC into ONE combined σ→LCB curve fern imports — 🟢 GREEN / σ 7.26→12.22 (√D=2.10), GO trigger μ≥520.09c/522.69wc lands ON the λ=1 ceiling 520.95 → P95-unreachable@λ=1 worst-case; EXACT trigger PROVISIONAL pending clean-1σ re-base (#204) — MERGED (bank-the-analysis, parallel advisor)

- **Branch:** `ubel/launch-sigma-closure` · **Student:** ubel · merged 18:20:37Z by the parallel advisor (CPU-only analytic over banked #195/#190/#187/#188/#176; no HF Job/submission/served-file change; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch authorized). W&B `spau6tch` (group `launch-sigma-closure`, finished, NaN-clean — 20 keys). Advisor-verified independently (`summary/` namespace; PRIMARY=1; both anchors **byte-exact**: #195's 7.2617 @ICC=0 err 0.0, #194's 512.157 via σ→LCB err 0.0; all decision scalars match).
- **Primary:** `launch_sigma_closure_self_test_passes=1` (7 legs). **Test:** `combined_sigma_launch_central=12.2153` (worst-case 13.7956).
- **Key finding:** de-dup (#195, σ IDENTITY — collapse the ρ=0.945 #175×#187 double-count into ONE acceptance axis at iid 5.32) and ICC (#190, σ MAGNITUDE — ×√D) are ORTHOGONAL corrections to the same acceptance axis. `D=1+(24.58−1)·0.1446=4.4106` (=#190 design-effect exact), √D=2.100, acceptance 5.32→**11.17**. Combined launch σ = **12.215 central / 13.796 worst-case** (ρ(*,hw)=+0.3, PSD min-eig 0.672) — REPLACES #195's iid 7.26/17.04. Realistic ICC erodes **~8 TPS** of headroom: the P95 GO trigger moves from #194's iid 512.16 up to `mu_clears_500_central=520.09` / `_worstcase=522.69`, landing ON the λ=1 ceiling (520.95). `lambda1_clears_500_central=1` (margin +0.86 TPS) but `_worstcase=0` (misses by 1.74) → P95-UNREACHABLE@λ=1 in the worst-case ρ(*,hw) corner. land #71 co-log (n=385 cross-device allocations) retires the [−0.3,+0.3] ρ(*,hw) band.
- **⚠️ Convention flag (DECISION-CRITICAL, provisional):** ubel flagged the inherited axis vector mixes units — the acceptance leg is a **95% half-width**, σ_hw/σ_private are **1σ**, combined then ×z again in `LCB=μ−z·σ`. At this knife-edge that is NOT cosmetic: a consistent clean-1σ footing could move the trigger **~3–6 TPS** (direction hinges on whether #187's 5.32 is a 1σ or a half-width) and **FLIP `lambda1_clears_500`**. The "unreachable@λ=1" verdict + the exact trigger are **PROVISIONAL** until the clean-1σ re-base (#204). **fern #185: import the de-dup×ICC reconciliation + ICC-erodes-8-TPS + the ρ(*,hw)-gated STRUCTURE now; do NOT hard-wire the exact trigger yet.**
- **Conclusion:** the launch GO/NO-GO is genuinely gated on the one unmeasured quantity ρ(*,hw) (between-device hardware↔acceptance coupling), which land #71's co-log retires. Orthogonal to #192. ubel → **#204 (clean-1σ launch-σ re-base — does λ=1 clear 500 centrally? resolve #187's 5.32 provenance; anchor BOTH conventions A/B)** — Morgan-reseated, parallel-converged with the advisor design.

## 2026-06-14 18:11 — PR #198: λ-dependent private drop — does the adverse-domain drop couple with recovery λ, moving #191's fixed-drop bar? — 🟢 GREEN / directional prior REFUTED but #191 VALIDATED (coupling NEGATIVE: shallow-concentrated deficit compounds with depth → drop FALLS at low λ → 0.9780 conservative) — MERGED (bank-the-analysis, parallel advisor)

- **Branch:** `stark/lambda-dependent-private-drop` · **Student:** stark · merged 18:11:02Z by the parallel advisor (CPU-only analytic over banked #176/#191/#193; no HF Job/submission/served-file change; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch authorized). W&B `llo1bzn3` (group `lambda-dependent-private-drop`, finished, NaN-clean — 31 keys, run self-reports `nan_clean=1`). Advisor-verified independently (`summary/` namespace; PRIMARY=1 with all 5 sub-checks a–e=1; TEST coupled bar 0.9779783 matches to 6 sig figs; drop@floor 2.2935% ≤ drop@bar 2.3489% ≤ drop@λ=1 2.350%).
- **Primary:** `lambda_private_drop_self_test_passes=1` (a–e). **Test:** `both_bugs_lambda_star_lcb_private_coupled=0.977978` (vs #191 fixed-drop 0.978011 → Δ −3.3e−5 in λ, NEGLIGIBLE).
- **Key finding:** closes the λ-independence assumption #191 carried (it composed #176's adverse drop as a CONSTANT 2.35% across all λ). The PR predicted a POSITIVE coupling (harder-to-draft adverse tokens → lower λ → #193 amplifies at depth → drop RISES at low λ → stricter bar → more-robust NO-GO). The mechanism does the OPPOSITE: #176's per-rung deficit is **shallow-concentrated** (δ=[+4.41, +1.95, +0.98, −0.33, −0.57, −0.91, −1.36]%); shallow positive deficits **compound multiplicatively** along the accepted chain → a deeper (high-λ) tree accumulates MORE total drop, the shallow realistic-floor tree (λ̂₁=0.342) accumulates LESS → drop is **smallest at the floor** (`coupling_sign=negative`). stark proved this is the driver (clip r≤1 keeps coupling negative). β-robust: across #193's β range [0.6165, 1.0] the coupled bar is within **±3.3e−5** of 0.9780 and drop@floor ≤ 2.350%.
- **Conclusion:** #191 used the FULL-recovery drop (the *largest*), so its fixed-drop composition is a **conservative upper bound** — **0.9780 both-bugs STANDS**, descent-only stays UNREACHABLE, both-bugs required, realistic-floor NO-GO **UNCHANGED** (floor misses by 0.636 in λ either way; `private_nogo_more_robust_under_coupling=False` but `nogo_verdict_unchanged=True`). The private-validity axis gets MORE robust, not less. Relayed to fern #185 (keep 0.9780 as the private-validity row). Honest scope: conservatism rests on the shallow-concentrated non-Latin-script shape (the single measured shape) — stark's follow-up #2: if land #71's measured q[2..9] deficit is DEEP-concentrated the sign could differ → fern must read the SHAPE. Orthogonal to #192. stark → **#203 (deficit-shape robustness — is 0.9780 shape-invariant + a co-log spec for land #71)**.

## 2026-06-14 18:02 — PR #200: Cost-aware re-draw budget — pricing #194's N\*(μ) curve in GPU-$ — 🟢 GREEN / cost-min N=5 at the bar (c-invariant); sequential early-stop pays only 1.94 shots; build-higher iff reaching μ=512.2 < 4 shots — MERGED (bank-the-analysis)

- **Branch:** `kanna/cost-aware-redraw-budget` · **Student:** kanna · merged 18:02Z (CPU-only pure-Python cost model over #194's banked curve; NO official draw, authorizes NO spend — a human still approves N; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched). W&B `n3alx7ca` (group `cost-aware-redraw-budget`, finished, NaN-clean). Advisor-verified independently (`summary/` namespace; all 7 reported scalars match — crossover 3.039268, sequential_at_bar 1.9375, mu_safe 512.157, σ_draw 7.391; self-test reproduces #194's best-of-N P verbatim, max abs err 0.0).
- **Primary:** `cost_budget_self_test_passes=1` (6/6). **Test:** `cost_optimal_n_at_bar=5`.
- **Key finding:** adds the cost layer #194 was missing. (1) **At a fixed build μ, cost does NOT change N** — always the fewest feasible shots (N=5 at the bar, c-invariant); cost only bites across the (μ,N) frontier where build substitutes for shots. (2) **Build-vs-redraw is scale-free:** build to μ=512.2/N=1 iff reaching it costs **< 4 official shots' GPU-$** (fixed-N crossover `c*=3.039·b`, Δμ=12.157) — sidesteps the unknown land #71 build cost. (3) **Sequential early-stop is the dominant lever, not building higher:** best-of-5 at the bar pays only **1.94 shots** on average (~61% under the naive 5, since ~half clear on shot 1), raising the build-higher crossover to **12.97·b** — under stop-on-first-clear the cheap path is *stay at the bar, pay ~2 shots* unless shots are ≳13× the build cost.
- **Conclusion:** the human's `Approval request: HF job` should authorize the cheaper of {build-higher μ≥512.2 → N=1} vs {build-at-bar μ=500 → N_max=5, pay ~1.94 sequential}; the realistic budget row is **1.94 shots**, not 5. Relayed to fern #185 (multi-shot budget annotation). Honest scope: build cost `b` swept (headline scale-free so it survives the gap); FROZEN-sampling (#194 §5d) is cost-independent and still load-bearing; #192 separate. kanna → **#202 (frozen-sampling re-draw budget — does best-of-N beat down all of σ_draw or only σ_hw? pin the conservative-regime build bar)**.

## 2026-06-14 17:50 — PR #195: Cross-axis CI covariance — is fern #185's quadrature-independence valid? — 🟢 GREEN / quadrature INVALID (ρ=+0.945 A1×A2 double-count), de-dup→7.26 not inflate→15.26, hw-coupling UNMEASURED→worst-case 17.04 — MERGED (bank-the-analysis, parallel advisor)

- **Branch:** `ubel/ci-axis-covariance` · **Student:** ubel · merged 17:50:51Z by the parallel advisor (CPU-only analytic over banked co-logged traces; no HF Job/submission/served-file change; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no launch authorized). W&B `3658ncbe` (group `ci-axis-covariance`, finished, NaN-clean). Advisor-verified independently (`summary/` namespace; `combined_sigma_corrected=15.2585`, quadrature 12.536 / dedup 7.262 / worstcase 17.037, PSD min-eig 0.781 — all confirmed).
- **Primary:** `ci_covariance_self_test_passes=1` (6/6). **Test:** `combined_sigma_corrected=15.2585 TPS` (`quadrature_valid=0`).
- **Key finding:** pins the **ADDITIVE** CI-quadrature law — the twin of ubel's multiplicative `official=K_cal·(E[T]/step)·τ` pins (#148/#169/#181). fern #185's 4-axis quadrature `σ=√(Σσ_i²)` is **INVALID**, but NOT from the hypothesized hardware coupling: the violation is a **double-count inside the sampling block** — ρ(sampling #175 ±10.9, input-λ̂ #187 ±3.71)=**+0.945** (Fisher-z CI [0.923,0.961]); the OUTPUT-side accepted-length scatter and the INPUT-side λ̂ CI are two views of the SAME accept draw (denken #187's `overlap_fraction=0.893`=ρ²). Mechanically the +2ρσσ term inflates σ to **15.26** (vs quadrature 12.54 → LCB too optimistic by −3.49 TPS); the **physically-correct fix is to de-duplicate A1+A2 into ONE acceptance axis** at #187's overlap-corrected 5.32 → 3 independent axes → **7.26 TPS** (smaller than quadrature). The hardware↔acceptance coupling fern actually feared is **UNMEASURED**: co-log is within-device only (σ_within 0.056, 87× below the dominant σ_between 4.864; ρ_within=−0.50 but multiplies the tiny within-σ), so the launch-relevant **BETWEEN-device** ρ(\*,hw) is carried honestly as a bounded **[−0.3,+0.3]** → worst-case **17.04**, pending land #71's served draw. ICC=1 corner (wirbel #190): σ_sampling→54.9, combined→57.8 (scenario, not central).
- **Conclusion:** corrects a load-bearing assumption under the launch GO/NO-GO — fern #185 must **NOT** stack A1+A2 in quadrature; consume the **de-dup acceptance axis** (denken #187's 5.32) and carry **worst-case 17.04** until land #71 co-logs per-allocation acceptance⊕TPS. Relayed to fern #185 (Morgan's 17:52Z send-back folds this into fern's in-flight re-run) + land #71 (the co-log spec). Orthogonal to the #192 greedy-identity gate. ubel → **#201 (launch-σ closure — fold this de-dup + wirbel #190's realistic ICC into the single combined σ→LCB curve fern imports)**.

## 2026-06-14 17:31 — PR #194: Official re-draw budget — how many shots N for P(clear 500)≥0.95 — 🟢 GREEN / N=5 at the bar, N=1 at μ≥512.2, σ- AND ICC-invariant at μ=500 — MERGED (bank-the-analysis)

- **Branch:** `kanna/oneshot-redraw-budget` · **Student:** kanna · merged 17:31Z (CPU-only pure-Python budget model; NO official draws taken, authorizes none; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched). W&B `mxm5q63j` (group `oneshot-redraw-budget`, finished, 28 metrics finite, NaN-clean). Advisor-verified independently (`summary/` namespace).
- **Primary:** `redraw_budget_self_test_passes=1`. **Test:** `n_shots_for_p95_at_bar=5`.
- **Key finding:** best-of-N official re-draws exploit which uncertainty re-randomizes per draw (iid sampling ±10.9 = fresh 128 prompts/run; σ_hw=4.864 = fresh/allocation). N*(μ): **N=5 at μ=500** (σ- AND ICC-invariant — P(single≥500)=0.5 sits exactly at the bar regardless of scatter; `n_shots_for_p95_icc=5` confirms), **N=1 at μ=520.95** (P 0.998), **break-even μ=512.16**. The one real risk is the **FROZEN-sampling regime** (NOT ICC): if the official harness re-uses the fixed 128 prompts under deterministic greedy, ε_sample is a fixed bias and best-of-N saturates ~0.81 at μ=500 even at N=5 — hinges on whether an official re-submission re-benchmarks fresh prompts (open Q for the harness owner, like #192 enforcement).
- **Conclusion:** the human's `Approval request: HF job` budgets N=5 shots if land #71 projects μ≈500, dropping to N=1 once μ≥512.2 — the multi-shot complement to fern #185's single-shot GO/NO-GO. kanna → **#200 (cost-aware re-draw budget — expected-cost-minimizing shot count)**.

## 2026-06-14 17:26 — PR #190: Realistic within-prompt ICC / N_eff — pin the launch-CI half-width — 🟢 GREEN / realistic ICC=0.145 inflates the half-width 2.1× to ±22.9, public bar → 0.9513, both-bugs stays GO — MERGED (bank-the-analysis)

- **Branch:** `wirbel/icc-neff-launch-ci` · **Student:** wirbel · merged 17:26Z (CPU-only analytic; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched, no served-file change). W&B `fva6o4ug` (group `icc-neff-launch-ci`, finished, 24 numeric metrics finite, NaN-clean). Advisor-verified independently (`summary/` namespace; 7-row self_test_checks table carries the ICC=0/ICC=1 exact reproductions).
- **Primary:** `icc_neff_self_test_passes=1` (7/7). **Test:** `lambda_star_lcb_realistic_icc=0.9513` (`bar_shift_from_icc=+0.0461`).
- **Key finding:** the within-prompt sampling scatter is NOT iid — `icc_hat=0.1446`. With m̄=24.58, `Deff=1+(m̄−1)·ICC=4.41` turns #175's iid ±10.9 into **±22.9 TPS** (between iid ±10.9 and #184 worst ±54.9). Both-bugs stays **GO** (LCB 510.6 §4 / 508.5 P≥0.9), flipping only at ICC=0.373 = **2.6× realistic**; descent-only NOT robust (LCB 495, breaks for any ICC>0.067, across the whole realistic range). Public build bar rises to **0.9513** (vs iid 0.9052), still reachable. Caveat: ICC estimated at liveprobe λ̂=0.342, transported as dimensionless to λ=1 (2.6× breakpoint absorbs ≤2× underestimate); retire with land #71 served traces.
- **Conclusion:** fern #185 consumes ±22.9 (not iid ±10.9) and public bar 0.9513 (not 0.9052); the BINDING bar stays private 0.9780 (#191). The WITHIN-axis of the launch CI. wirbel → **#199 (compliant-spec E[T] ceiling, parallel-advisor assignment)**.

## 2026-06-14 17:23 — PR #193: Salvage-staleness λ(depth) mechanism vs flat-depth transfer — 🟢 GREEN / geometric staleness λ_d=λ̂₁·β^(d−1), β=0.765 — MISSES-BOTH robust physics, depth-1 probe NOT sufficient — MERGED (bank-the-analysis, parallel advisor)

- **Branch:** `denken/salvage-staleness-lambda-depth` · **Student:** denken · merged 17:23Z by the parallel advisor (CPU-only decision/measurement merge; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched). W&B `2clxvlr8` (`summary/` namespace, finished, NaN-clean).
- **Primary:** `lambda_depth_profile_self_test_passes=1` (5/5). **Test:** `both_bugs_mechanism_floor_tps=396.72` (`beta_primary=0.7651`, `misses_both_robust_to_mechanism=1`, `beta_crit_depth1_sufficient=0.9649`).
- **Key finding:** replaces the load-bearing FLAT depth-1→depth>0 λ-transfer with a mechanism-derived geometric staleness law `λ_d=λ̂₁·β^(d−1)`, β grounded in wirbel #135's measured salvage ladder (β=0.7651, inside #178's guessed [0.7,0.9]). (1) MISSES-BOTH is robust physics not a flat-transfer artifact — flat (β=1) is the OPTIMISTIC plateau and already misses at λ̂₁=0.342 (416.3<500); every grounded β∈[0.616,0.950] only WIDENS the miss (416→397). (2) **land #71's depth-1 probe is necessary but NOT sufficient** — under any per-step staleness β<0.965 the bar is UNREACHABLE even at perfect depth-1 recovery λ̂₁=1.0, so land must measure the q[2..9] ladder DIRECTLY, not infer from depth-1. (3) physically explains #187's depth-9-dominant variance. Orthogonal to #192 greedy-identity.
- **Conclusion:** with stark #191 (private 0.978) this DOUBLY hardens the NO-GO-at-realistic-floor posture and sharpens land #71's build requirement (measure the deep ladder directly — relayed to land #71). denken → **#197 (liveprobe depth-budget, parallel-advisor assignment)**.

## 2026-06-14 17:21 — PR #191: Private-side build bar — does the adverse-skew private drop demand a stricter λ than public 0.9052 — 🟢 GREEN / private λ*_LCB=0.9780 BINDS stricter than public, descent private-UNREACHABLE — MERGED (bank-the-analysis)

- **Branch:** `stark/private-build-bar` · **Student:** stark · merged 17:21Z (CPU-only analytic over banked σ's; no official draw; BASELINE unchanged 481.53, 0 TPS, greedy/PPL untouched). W&B `jeclr39w` (group `private-build-bar`, finished, 32 summary metrics finite, NaN-clean). Advisor-verified independently (`summary/` namespace).
- **Primary:** `private_build_bar_self_test_passes=1`. **Test:** `lambda_star_lcb_private=0.9780` (both-bugs).
- **Key finding:** composing #176's adverse-skew private drop through #183's finite-sample LCB map → both-bugs private bar **0.9780**, vs public 0.9052 (#183) — **+0.0728 stricter**, so `binding_bar = private = 0.9780`. Two flips: (1) **`both_bugs_required_at_private_bar=True`** — FLIPS #176's central-based `both_bugs_required_private=False`; the +4.15 margin #176 banked vs the private CENTRAL evaporates against the finite-sample LCB (13.99-TPS seam). (2) **descent-only private-UNREACHABLE** — its private LCB tops out at 490.16<500 even at λ=1 (`descent_unreachable=1`); lawine #180's descent "GO" was PUBLIC-only. `valid_at_bar=True` (4.295% drop ≤5% DQ), `private_lcb_at_public_bar=484.55`.
- **Conclusion:** the binding build target for land #71 moves from λ̂≥0.9052 (public) to **λ̂≥0.9780 both-bugs (private-binding)**; descent-only drops out as a private-viable launch (relayed to land #71 + fern #185). stark → **#198 (λ-dependent private-drop mechanism, parallel-advisor assignment)**.

## 2026-06-14 17:11 — PR #180: Realize #154's argmax-only decode — output-neutral step realization — 🟢 GREEN / output-neutral EXACT, realized step 1.2160 (only +0.176%, 16% of #154's projection), descent-only flips to a THIN GO 500.85/0.913 — MERGED (bank-the-analysis)

- **Branch:** `lawine/argmax-decode-step-realization` · **Student:** lawine · merged 17:11Z (advisor-driven merge — LOCAL single-A10G serve A/B only, throwaway copy `submissions/fa2sw_argmax_decode/`, `manifest.json` byte-identical; no HF Job/submission/committed served-file change; BASELINE unchanged 481.53, adds 0 TPS; greedy/PPL untouched, no launch authorized). W&B `kbn064b0` (group `argmax-decode-step-realization`, finished, NaN-clean). Advisor-verified independently: all 8 gating metrics present under the `prop/` namespace, run finished.
- **Primary:** `argmax_decode_step_self_test_passes=1` (all 5 legs). **Test:** `descent_only_lcb_with_argmax_decode=500.847` (P 0.913, +0.85 TPS).
- **Key finding:** the argmax-only decode patch (skip the `[M, full_vocab]` scatter on the spec-verify step, direct argmax) is **output-neutral EXACTLY** — `token_identity_rate=1.0` (**384/384** completion token-ids byte-identical, 128 prompts × 3 A/B runs) with PPL parity to 6 digits (`ppl_argmax=ppl_control=2.376683`, Δppl=0), 128/128 (greedy=argmax by construction — Issue #124). But the **step saving did NOT materialize**: `step_realized_argmax=1.2160`, only **+0.176%** vs the shipped 1.2182 — **just ~16% of #154's projected 1.106%** (to 1.2047). The deployed precache/star-attn-overlap stack already hides ~84% of #154's modeled 97.55µs scatter saving, so it frees only +0.176% wall time. Linear control reproduces the 1.2182 anchor to 0.0014% and recovers 481.53 official to 0.030% (validated wall_tps→official bridge). That +0.176% is still 25× the descent-only break-even (0.0070%), so it **flips descent-only from fern #174's MISS (499.97 / P 0.8994) to a GO** — but a THIN one: **LCB 500.85 / P 0.913, margin +0.85 TPS** (vs fern's projected robust 505.55 / 0.963 at the full 1.2047). both-bugs (514.88 / P 0.996) stays the more robust first shot. Key engineering fix: the skip gate is `M == 8` **exact** (= K_spec+1 under MAX_NUM_SEQS=1), NOT `M <= 8` — a `<=` gate corrupted the M=1 prefill-sample token (raw pruned column index, no keepset remap), faking a −9% slowdown (E[accept] 3.85→2.58); the exact-M gate fixed both identity and the wall delta.
- **Critical scope note (Issue #192):** the token-identity proven here is **decode-step identity** (patched-spec vs unpatched-spec, argmax over logits) — **orthogonal to** and does **NOT** resolve the #192 greedy-identity gate (the int4 Marlin **spec-verify GEMM batch-variance** → 56.08% divergence from *plain greedy AR*, kanna #114 `9q5yy9l1`). The descent-only "GO" here is a **TPS-composition verdict only**; the descent-only **spec** build remains gated by the unresolved #192 token-identity question.
- **Conclusion:** makes the BUILD choice explicit and quantified for fern #179 / land #71 — *simpler descent-only + this 1 output-neutral decode patch at a thin +0.85 margin* vs *robust both-bugs (514.88)*. Refutes #154's 1.2047 as a SHIPPED step (confirms lawine #168's CONDITIONAL tag). Residual for land #71: the `M == 8` gate is exact — under the BUILT tree decode loop the verify width may differ, so re-key it to the tree's actual verify width or the skip silently no-ops (safe, but forfeits the +0.176%). Exact patch diff handed off at `research/validity/argmax_decode_step/decode_patch_for_land71.diff`. lawine → **#196 (compliant-lane floor)** — measure the non-speculative int4 greedy-exact serve TPS (the #192 fallback, token-identical to plain greedy AR by construction): how far below 500 is our best *provably-compliant* serve, and how much is the speculation actually buying?

## 2026-06-14 17:00 — PR #189: Executable fail-closed MUST-RETAIN submission gate — 🟢 GREEN / verify_submission_gate catches the 85%-cost row-1 host-loop NO-GO — MERGED (bank-the-analysis)

- **Branch:** `ubel/executable-submission-gate` · **Student:** ubel · merged 17:00:41Z (advisor-driven merge — CPU-only static gate, no HF Job/submission/served-file change; BASELINE unchanged 481.53, adds 0 TPS; greedy/PPL untouched, no launch authorized — it BLOCKS/CLEARS the packaging precondition, a human still approves). W&B `pqpb8ugk` (group `executable-submission-gate`, finished, 16 finite metrics, NaN-clean). Advisor-verified: 34/34 fixtures pass.
- **Primary:** `submission_gate_self_test_passes=1` (**34/34** fixtures (a)–(g)). **Test:** `gate_catches_row1_host_loop=1` (host-loop → NO-GO, row 1 binding, 444.92 TPS = 85.17% caught).
- **Key finding:** converts ubel #186's **static** MUST-RETAIN manifest (a document) into an **executable, fail-closed** gate (an enforcement): `verify_submission_gate(build_env, build_introspection) → {packaging_verdict: GO|NO-GO, failing_rows[], per_row_assertions[], validity_class_failures[], binding_failure}`. Imports #186's JSON as the source of rows+costs (does NOT re-derive), walks all 22 flags, asserts each of the 19 MUST-RETAIN rows present/correct + the TRAP (`LSK_SKIP_LAYERS`) UNSET, and is **fail-closed**: any MUST-RETAIN FAIL, present TRAP, or missing/unparseable introspection → **NO-GO, never silent-pass** — naming the failing row + its banked cost-of-omission. Row decomposition: **3 structural** (relocate / accept-walk / decode) + **2 env-json** (num_speculative_tokens / temperature) + **16 env** + **1 trap**. The headline (row-1 binding): the host-loop fixture is correctly NO-GO with row 1 binding + the **444.92 TPS / 85.17%** cost attached — the 85%-cost relocate-host-loop regression is now impossible to ship silently. The 5 double-load-bearing rows route to a `validity_class_failures` bucket (the `validity_seam`) so the gate merges with denken's output-validity preflight into one pass/fail surface (gate asserts flag presence/shape; output correctness stays denken's lane).
- **Conclusion:** the operational safety net for the irreversible official shot — run `verify_submission_gate` against land #71's assembled build BEFORE any `Approval request: HF job`; a NO-GO names the exact failing flag + cost (esp. row 1). Feeds fern #185's ledger as the **`packaging-gate: GO`** precondition row (the static-enforcement twin of fern's numerical GO/NO-GO; authorizes nothing). ubel → **#195 (cross-axis CI covariance)** — is fern #185's 4-axis quadrature-independence assumption valid, or is there a positive cross-axis covariance that makes the combined σ larger (and the launch LCB too optimistic)?

## 2026-06-14 17:00 — PR #188: One-shot launch-draw hardware bound (σ_oneshot decomposition) — 🟢 GREEN / σ_hw=4.86 is ALREADY the one-shot cross-allocation draw, launch bound NOT wider — MERGED (bank-the-analysis)

- **Branch:** `kanna/oneshot-hw-bound` · **Student:** kanna · merged 17:00:38Z (advisor-driven merge — CPU-only hardware-σ decomposition, no HF Job/submission/served-file change; BASELINE unchanged 481.53, adds 0 TPS; greedy/PPL untouched, no launch authorized). W&B `pp1r5orx` (group `oneshot-hw-bound`, finished, 23 finite metrics, NaN-clean). Advisor-verified: 6/6 self-test checks pass, `sigma_oneshot` matches to full precision.
- **Primary:** `sigma_hw_decomposition_self_test_passes=1` (6/6). **Test:** `sigma_oneshot=4.8645 TPS` ( = `sigma_hw`; reconstruction gap +0.0e0).
- **Key finding:** the one-shot launch bound is **NOT wider** — `sigma_hw=4.86` is ALREADY the between-device cross-allocation draw, not a within-run std; the PR's premise ("maybe 4.86 was measured run-to-run on one warmed device, so a fresh launch draw is wider") is **refuted by #159's own construction.** `sigma_within=0.0111% (0.056 TPS)` = #159's fresh noise floor, n=12 fresh-server restarts on ONE pinned A10G (1710 MHz, 57–58 °C); `sigma_between=0.9623% (4.864 TPS)` = frantic-penguin's same-submission 3-draw across the HF a10g-small POOL (3 independent allocations). `sigma_oneshot = √(within²+between²) = 4.864 == #159 sigma_hw exactly`; ratio `sigma_between/sigma_within ≈ 86.6×` → **cross-allocation dominated** (packet already flags `cross_allocation_dominated: true`). Conservative if anything (n=9 frontier CV 0.555% upper bound puts pure-hw σ below 0.962%; on-pod SM clock holds 1710 MHz across all 24 runs, no throttle). τ-floor (#181) cross-check `consistent` — no double-count (τ-floor = clock **mean** compute-exposed corner; σ_clock = clock **variance**; σ_BW orthogonal/BW-dominated; τ ⟂ K_cal per #181's orthogonality).
- **Conclusion:** de-risks the **HARDWARE axis** of the launch CI — retires "is 4.86 the right σ for a SINGLE launch draw?" (YES, it is the cross-allocation draw the one-shot launch faces, and conservative). fern #185's single-shot quadrature consumes the correct σ_hw=4.86 with no inflation; the launch bound does NOT widen on the hardware leg. Complements sampling-ICC (wirbel #190), input-λ̂ (denken #187), private-drop (stark #191). kanna → **#194 (official re-draw budget)** — N\*(μ) shots for P(clear 500)≥0.95, exploiting that the dominant sampling scatter (±10.9) re-randomizes per official run while σ_hw is per-allocation.

## 2026-06-14 16:53 — PR #187: Margin-aware λ̂_built measurement-CI (input-side resolvability gate) — 🟢 GREEN / λ̂ halfwidth ±0.017, on-bar builds unresolvable, don't double-count #175 — MERGED (bank-the-analysis)

- **Branch:** `denken/lambda-built-ci` · **Student:** denken · merged 16:53:55Z (advisor-driven merge — CPU-only analytic synthesis, no HF Job/submission/served-file change; BASELINE unchanged 481.53, adds 0 TPS; greedy/PPL untouched, no launch authorized). W&B `tloghme9` (group `lambda-built-ci`, finished, NaN-clean, 27.1 MiB CPU). Advisor-verified independently: 6/6 self-test sub-checks pass, halfwidth/overlap match the PR claim to 5 digits.
- **Primary:** `lambda_built_ci_self_test_passes=1` (6/6). **Test:** `lambda_built_halfwidth=±0.017140` (WLS/MLE, both-bugs @ λ̂=0.905, 128×512) → CI **[0.8881, 0.9224]**.
- **Key finding:** prices the **INPUT-side** of the GO gate — the sampling CI on land #71's *measured* λ̂_built — the dual of wirbel #175's output-side TPS CI. Measurement model `q̂_d ~ Binomial(n_d, q_d)`, `n_d = N_steps·S(d)` survival-thinned off #175's pmf → `n_ladder=[11585,9970,7752,6682,5115,4194,3164,2313]` (depth-9 gets 5× fewer trials than depth-2, so **depth-9 dominates the variance** → WLS inverse-variance pooling is the right estimator, and land #71 must report `n_d` per depth for an auditable CI). **Resolvability gate (the deliverable):** `N_resolve = N0·(hw_ref/margin)²` — a confidently-good **true-λ=0.93 build resolves decisively at N≥62 prompts** (default 128 already suffices); a build **on** the bar (0.905) needs ~717k prompts → **effectively unresolvable** — a point λ̂ within ±0.017 of 0.9052 is an indecisive GO. **Double-count audit:** INPUT-CI (±3.71 TPS via the λ̂ route) and #175's OUTPUT-CI (±5.18 TPS via the L̄ route) are linear functionals of the *same* δq̂_d on a shared bench → **partial-overlap, `overlap_fraction=0.8929`** (ρ=0.945); naive quadrature 6.37 TPS overstates vs the corrected 5.32 TPS. VIF sensitivity 1.0→±0.0171 / 1.5→±0.0210 / 2.0→±0.0242.
- **Conclusion:** makes land #71's eventual GO/NO-GO *decisive, not a coin-flip on the bar* — **aim comfortably clear of 0.9052, not nominally over** — and hands fern #185 a concrete guardrail: compose INPUT⊕OUTPUT CIs by **subtracting the 89% overlap on a shared bench**, NOT naive quadrature (the ubel #181 double-count discipline, applied to variance). Honest scope: prices noise on the gate INPUT; does NOT move the #183 bar or authorize a launch. denken → #193 (**salvage-staleness λ(depth) mechanism profile** — the depth-1→depth>0 flat-λ transfer that #178/#183/#187 ALL inherit is the last un-pinned axis of the bar now that wirbel #184 fixed its height; ground λ(depth) in the `d−1`-step-stale salvage-KV physics and re-run MISSES-BOTH).

## 2026-06-14 16:46 — PR #176: Adverse domain-skew private certificate — descent-only survives the worst realistic skew — 🟢 GREEN / private cert PASSES, τ-low 504.15 (+4.15 margin), both-bugs-not-required HARDENED — MERGED (bank-the-analysis)

- **Branch:** `stark/descent-private-adverse-skew` · **Student:** stark · merged 16:46:07Z (CPU-only analytic + local-A10G private probes, precache=off/bench=private/no official draw — no HF Job/submission/served-file change; BASELINE unchanged 481.53, adds 0 TPS; greedy/PPL untouched, no launch authorized). W&B `uzl7ixll` (group `descent-private-adverse-skew`, finished, 31 finite metrics, NaN-clean). Advisor-verified: 4/4 sub-tests pass.
- **Primary:** `adverse_skew_stress_self_test_passes=1` (4/4). **Test:** `descent_only_taulow_tps_adverse_corner=504.15` (central 507.99).
- **Key finding:** the adverse-skew private certificate PASSES — descent-only survives the worst realistic private domain skew, clearing 500 at BOTH τ corners (**+4.15 TPS margin at τ-low**). #164's `both_bugs_required_private=False` is **HARDENED, not flipped** — both-bugs is still NOT a hard private dependency even under adverse skew. Adverse vertex = **pure non-Latin-script** (multilingual, W_hard 0.290), tree drop **2.300% descent / 2.350% both** — confirmed worst over a 2491-direction simplex scan (singles + pair-edges + triple-faces + 1500 Dirichlet interior; `adverse_vs_worst_single_pp=+0.000`). **`cap_binding_at_optimum=False`** → the corner is interior to the diversity cap, so the certificate is insensitive to the cap choice. Honest envelope vs #164: widened descent CI worst-case **2.301%** (vs #164's 3-axis mid 2.04%, +0.26pp). Stressed 6 axes (3 #164 byte-identical + 3 NEW hard tails: non-Latin-script / math-notation / long-context), each on the deployed `fa2sw_precache_kenyan` stack (128/128, PPL 2.377, greedy untouched), each passing the ≤0.5pp GT-4.3% calibration gate. Bug-fix kept (non-fatal): wrapped `log_report_to_wandb` in try/except so a broken submission-venv wandb can't fail an already-complete probe — confined to `scripts/validity/private_gap_probe.py`, no served/eval code touched.
- **Conclusion:** de-risks the **PRIVATE axis** of the launch go/no-go — descent-only is private-robust even under adverse domain skew, so the 460.85 private-verified VALID headroom holds and both-bugs carries no extra private burden. Complements the public/finite-sample CI work (wirbel #190, kanna #188). stark → **#191 (private-side build bar)** — compose #176's adverse-skew drop through #183's forward map: does the private drop demand a STRICTER λ than the public 0.9052, and is the build VALID-at-the-bar (≤5% drop)?

## 2026-06-14 16:41 — PR #184: λ-robust verify-tree topology — can a topology lower the recovery bar below #83's 0.838? — 🔴 BANK-NEGATIVE / front-loading REFUTED, #83 confirmed the build target — MERGED (bank-the-analysis)

- **Branch:** `wirbel/lambda-robust-topology` · **Student:** wirbel · merged 16:41:13Z (CPU-only analytic synthesis — no HF Job/submission/served-file change; BASELINE unchanged 481.53, adds 0 TPS; greedy/PPL untouched, no launch authorized). W&B `7uek36mx` (group `lambda-robust-topology`, finished, 45 finite metrics, NaN-clean). Advisor-verified: 7/7 conditions pass.
- **Primary:** `lambda_robust_topology_self_test_passes=1` (7/7). **Test:** `lambda_robust_topology_lambda_bar=0.8345` (min LCB-recovery bar over the 182-tree feasible family — but only at depth-14, past the measured horizon).
- **Key finding:** `BANK-NEGATIVE-ONLY-DEEP-EXTRAP-BEATS-83`. No 32-node max-branch-3 topology lowers the self-KV-recovery bar below #83's 0.838 **within the defensible (≤depth-9) horizon** — at the as-built depth-9 horizon #83 IS the min-λ_bar optimum (0.9052 LCB / 0.838 central), the gap `[λ_robust, 0.838]` is **empty**. Front-loading is strictly *worse*: **min-λ-bar ≡ max-E[T]@λ=1** (corr **−0.95**) — the SAME lever #83 already optimized, because the both-bugs floor `q_floor` *rises back* after depth-2. **No free lunch:** `max E[T]@λ=0 = 3.705 < bar 4.862` for EVERY topology → **topology alone can NEVER clear 500 at the floor.** Item-4 N_eff bonus (closes #175's iid caveat): #175's clear-500 (LCB 521) **survives moderate within-prompt clustering (ICC ≤ ~0.4) but FLIPS under heavy correlation** — at ICC=1 (24.6 steps/prompt fully correlated → N_eff=128) the half-width inflates **3.8× (±10.9 → ±54.9)** and even #83 at full recovery lands LCB **480.5 < 500**.
- **Conclusion:** confirms **#83 (depth-9, max-branch-3) is the build target** (correctly optimized; depth-11 a weak fallback only under recovery-curve extrapolation, and its worst-case N_eff LCB also fails 492.4). Strengthens the core finding: **the binding lever is land #71's measured self-KV recovery ladder q[2..9], NOT topology.** The N_eff caveat operationalized into wirbel → **#190 (realistic within-prompt ICC/N_eff)** — pin the launch CI's sampling-correlation axis between iid ±10.9 and worst ±54.9; relayed to fern #185 to fold ICC into the launch CI.

## 2026-06-14 16:36 — PR #186: Submission MUST-RETAIN manifest — flag-by-flag packaging de-risk — 🟢 GREEN / one dropped flag = 85% TPS, no submit-time warning — MERGED (bank-the-analysis)

- **Branch:** `ubel/tree-submission-must-retain-manifest` · **Student:** ubel · merged 16:36:44Z (CPU-only manifest consolidation + reproduction-gate — no HF Job/submission/served-file change; BASELINE unchanged 481.53, adds 0 TPS by design; greedy/PPL untouched, no launch authorized). W&B `u9kje7sn` (group `tree-submission-must-retain-manifest`, finished, 18 finite metrics, NaN-clean). Advisor-verified: 25/25 self-test conditions pass.
- **Primary:** `manifest_self_test_passes=1` (**25/25** — re-loads each banked source JSON at runtime and reproduces the cost from raw fields, so the consolidation is provably faithful, not a drifted re-summary). **Test:** `binding_packaging_cost_pct=85.17%` (of realizable descent TPS).
- **Key finding:** one dropped flag costs 85% of projected throughput with no submit-time warning. 22 served-surface flags → **19 MUST-RETAIN, 5 double-load-bearing, 8 priced.** The binding risk is **row 1: `relocate_salvaged_kv` must be vectorized/device, NOT a host Python loop** — reverting to a 37-layer host loop is 1571× per-call and collapses descent **516→77 TPS (−85%)**, an order of magnitude above the next flag (PRECACHE_BENCH, 3.5%). The 5 double-load-bearing items break **validity** if dropped (decode argmax-only is greedy-exact ONLY if the full scatter+LP stays on the prefill PPL path; `temperature=0.0`/conc=1/ctx4096/bf16/int4-pck04). `LSK_SKIP_LAYERS` is a **TRAP** that must remain UNSET. Capturability rows (`ONEGRAPH`/`DIXIE_*`) price Δ=0 but are MUST-RETAIN (losing capture re-prices rows 1/3/4).
- **Conclusion:** the operational/packaging de-risk twin of fern #185's numerical GO/NO-GO — hands land #71 a flag-by-flag verify checklist for the `Approval request: HF job` (verify the as-submitted build realizes the projected stack BEFORE the irreversible shot). Orthogonal to the self-KV λ gate. Self-contained bug-fix to the new script only (wandb_log KeyError fixed, re-ran `u9kje7sn`); no served/eval code touched. ubel → **#189 (executable fail-closed submission gate)** — convert this static manifest into `verify_submission_gate(build_env, introspection)→GO/NO-GO` that auto-catches the 85%-cost row-1 host-loop regression before approval.

## 2026-06-14 16:15 — PR #183: Margin-aware λ-acceptance card (finite-sample-LCB build bar) — 🟢 GREEN / build bar λ≥0.9052 both-bugs (Δ+0.067 stricter than #178's point) — MERGED (bank-the-analysis)

- **Branch:** `denken/lambda-acceptance-card` · **Student:** denken · merged 16:15:53Z (CPU-only analytic synthesis — no HF Job/submission/served-file change; BASELINE unchanged 481.53, adds 0 TPS; greedy/PPL untouched, no launch authorized). W&B `82uisrez` (group `lambda-acceptance-card`, finished, NaN-clean). Advisor-verified: 6/6 self-test conditions pass.
- **Primary:** `lambda_acceptance_card_self_test_passes=1` (6/6). **Test:** `both_bugs_lambda_star_lcb=0.905229` (full finite-sample LCB clears 500, τ=1).
- **Key finding:** the margin-aware build bar is uniformly ~6.6 points of recovery STRICTER than #178's central point estimate. both-bugs λ\*_LCB=**0.9052** (vs #178 point 0.8384, Δλ **+0.0668**); descent-only **0.9750** (τ=1); τ=0.9924 floor → 0.9234 both / 0.9926 descent. **The punchline:** at #178's own point λ=0.838 the finite-sample LCB is only **486.2 — a 14-TPS MISS**; building to the point bar fails the finite-sample test. Forward map (both-bugs, τ=1, `card_is_monotone=True`): λ=0.342→404, 0.838→486.2, **0.9052→500.0**, 1.0→520.95. **Provenance lock:** σ_L(λ) read off wirbel #175's pmf on the same spine #178's `et_backward` consumes — reproduces #175's published numerator bounds to 1e-15 (the 2e-4 TPS resid = K_cal `125.268` vs canonical `125.26795` rounding gap, ≪ 0.5-TPS tol).
- **Conclusion:** THE gate fern #185's launch-trigger calculator consumes and the number land #71's built kernel must clear — **measured λ̂_built ≥ 0.9052 both-bugs**, NOT 0.838. Folds wirbel #175 ±10.9 ⊕ kanna #159 σ_hw=4.86 into the build bar. Relayed to land #71 (the 0.9052 build target) + fern #185 (wire the real card, drop the #178 fallback). denken → #187 (**λ̂_built measurement-CI** — price the INPUT-side finite-sample noise on the measured `q[2..9]` ladder: how many prompts does land #71 need so its implied λ̂ decisively resolves the 0.9052 bar at 95%, the dual of #175's output-side TPS CI).

## 2026-06-14 16:15 — PR #177: Launch-boot de-risk — darwin _IncludedRouter startup-500 guard — 🟢 GREEN / proven OUTPUT-NEUTRAL (token-id 1.0, PPL byte-identical) → land #71 banks the guard — MERGED (bank-the-analysis)

- **Branch:** `kanna/included-router-boot-validation` · **Student:** kanna · merged 16:15:39Z (local serve output-neutrality proof — no HF Job/submission/served-file change; BASELINE unchanged 481.53, adds 0 TPS; throwaway patched copy only, no launch authorized). W&B `bjtwr9jn` (launch-boot de-risk, finished, NaN-clean). Advisor-verified: 7/7 sub-checks pass.
- **Primary:** `included_router_fix_self_test_passes=1` (7/7). **Test:** `token_identity_rate=1.000000` (128/128 completion-token-ids byte-identical).
- **Key finding:** the darwin `_IncludedRouter` startup-500 guard is OUTPUT-NEUTRAL on the deployed `fa2sw_precache_kenyan` stack — completion-token-ids identical 128/128, PPL byte-identical (2.376976138392039 both sides), TPS within 0.02% (459.969→460.066, noise). The guard unit-neutralizes the real `prometheus_fastapi_instrumentator.routing._get_route_name` AttributeError on a pathless matched route (darwin's exact mechanism) AND is a byte-verbatim no-op on a normal `/v1/models` route. On THIS local A10G image the 500 did NOT reproduce (local fastapi 0.136.3 mounts no `_IncludedRouter`; honest `startup_500_reproduced=False`) — but darwin reproduced it 3× on the fresh HF runner image. **Zero-cost no-op insurance:** the fix where the runner crashes, a verified no-op where it doesn't (HTTP-metrics middleware ONLY; never touches greedy/PPL/token-ids; also no-ops when the instrumentator is absent).
- **Conclusion:** closes the launch-BOOT dependency (one of the 3 launch blockers in fern #179's ledger — build / boot / PRECACHE). land #71 should INCLUDE the guard — exact diff posted (append to `submissions/fa2sw_precache_kenyan/sitecustomize.py` after the PRECACHE_BENCH block, line 1293); merged harness `scripts/validity/included_router_boot_selftest.py`. Relayed the confirmed diff to land #71. kanna → #188 (**one-shot launch-draw hardware bound** — decompose σ_hw=4.86 into within-run vs between-device/thermal, reconcile against ubel #181's 0.9924 τ-floor on the same clock-residual axis: is 4.86 the right σ for a SINGLE A10G launch draw?).

## 2026-06-14 16:09 — PR #181: Pin the τ overlap/coverage-efficiency factor — 🟡 KNIFE-EDGE / last composition factor closed, τ=1.0 band [0.9924,1.0], floor-confirmed (no free margin) — MERGED (bank-the-analysis)

- **Branch:** `ubel/tau-overlap-efficiency-pin` · **Student:** ubel · merged 16:09:53Z (CPU-only analytic synthesis — no HF Job/submission/served-file change; BASELINE unchanged 481.53, adds 0 TPS; greedy/PPL untouched, no launch authorized). W&B `j65cvgj4` (group `tau-overlap-efficiency-pin`, finished, NaN-clean). Advisor-verified: a–e all pass.
- **Primary:** `tau_efficiency_self_test_passes=1` (a–e). **Test:** `tau_descent=1.0`, band **[0.9924, 1.0]** (`tau_both_bugs` identical — topology-invariant, same M=32 wide-verify GEMM).
- **Key finding:** closes the LAST unpinned factor in `official=K_cal·(E[T]/step)·τ`. τ is the local→official **roofline transfer multiplier** (M=32 wide-verify decode-step fidelity, A10G 1710 MHz local → official free-clock); the **E[T] numerator cancels exactly** (greedy on identical weights). Floor **0.9924318649** = lawine #126 `mild_throttle_full_exposure` corner = ubel #148 K_cal Leg A (clock-exposure) — the dominant calibration-leg downside, sourced from the roofline. **No double-count** (`tau_no_double_count=True`): vs K_cal (bus/BW +6.019% #169, footprint-invariant) — τ is the incremental compute-exposed fraction at the clock ratio; vs step (#168 +0.447% launch-idle is clock-independent time → cancels in τ); vs E[T] (rank-coverage ρ lives entirely in the E[T] DP, not τ — the "rank-coverage efficiency" name is a misnomer, τ is purely the overlap/clock-exposure channel). **Reproduces fern #174's descent-only LCB `499.965` to machine precision (Δ +0.000)** — #174 already used τ=1.0 central with [0.9924,1.0] in the calibration CI (per fern #155 convention), so there was NO hidden margin to hand back. `descent_only_clears_500_pinned_tau=False`.
- **Conclusion:** **τ is NOT the lever.** Even a perfect τ=1.0 ceiling clears 500 only in the 3-term framing (500.96); fold in kanna #159 σ_hw (4.86 TPS) → 499.49 < 500. The knife-edge is **sampling + σ_hw bound**, not τ-bound — the only path to free τ margin is tightening the floor above 0.99258, which needs a real served M=32-tree official-clock measurement we cannot take from one pod (→ land #71's eventual HF job). Composition is now **FULLY pinned** (K_cal=125.268 #148/#169, step 1.2182 #168, E[T] descent 5.0564 / both-bugs 5.2070, τ=1.0 [0.9924,1.0] — this leg); **both-bugs (LCB 514.9) stays the robust first shot**, descent-only stays the knife-edge MISS (499.97). Launch remains gated solely on land #71's measured self-KV λ (denken #178 realistic floor λ̂=0.342 misses 500). ubel → #186 (**submission MUST-RETAIN manifest** — flag-by-flag packaging de-risk: consolidate the cost-of-omission of every load-bearing serving flag so the as-submitted build faithfully carries the pinned stack).

## 2026-06-14 16:04 — PR #179: Launch-packet refresh — both-bugs PRIMARY GO — 🟢 GREEN / one current go/no-go artifact, both-bugs LCB 514.88 — MERGED (bank-the-analysis)

- **Branch:** `fern/launch-packet-refresh-bothbugs` · **Student:** fern · merged 16:03:58Z (CPU-only analytic consolidation — no HF Job/submission/served-file change, no issue filed, **no launch authorized**; BASELINE unchanged 481.53). W&B `d71gvk5i` (group `launch-packet-refresh-bothbugs`, finished, NaN-clean, CPU-only). Advisor-verified: 5/5 self-test checks pass.
- **Primary:** `launch_packet_refresh_self_test_passes=1` (5/5). **Test:** `both_bugs_launch_lcb_tps=514.88`.
- **Key finding:** consolidated #167→#174 into ONE current go/no-go artifact with **both-bugs as PRIMARY GO** — GO at all 3 step framings (LCB **514.88** shipped 1.2182 / 517.18 roofline / 520.65 scatter-LP); descent-only the knife-edge MISS at the shipped step only (499.97). σ_hw two-axis: best-of-2 retires the hardware axis (P=0.9829) without subtracting from the projection LCB; naive-fold sensitivity invariant (both-bugs GO either way, 513.4/99.2%). Dependency ledger: **5 LANDED / 4 IN-FLIGHT / 1 PENDING / 1 PENDING-BUILD**. Carries the #172 BOUNDED-NOT-ROBUST caveat (central numerator is the optimistic full-recovery value; the GO requires ≥91% deep-spine recovery).
- **Conclusion:** the canonical static launch packet — recommendation **both-bugs GO gated on (build + boot-fix + PRECACHE + human approval)**, correct and conservative (does NOT authorize launch). **Used to answer the human launch question (issue #182):** the answer is HOLD — the build (land #71) is unbuilt (draft WIP, `terminal:false`), the make-or-break self-KV λ is unmeasured, and the darwin boot-fix is still in validation (kanna #177). Fold-in for the next snapshot (now merged since assembly): denken #178 (λ̂=0.342 → both miss 404/416) + wirbel #175 (±10.9 finite-sample). fern → #185 (**launch-trigger calculator** — operationalize this geometry into a one-call *verified* GO/NO-GO + filled `Approval request: HF job` block from land #71's measured tuple).

## 2026-06-14 15:47 — PR #178: Realistic self-KV E[T] floor (graded recovery curve) — 🟡 AMBER / `REALISTIC-FLOOR-MISSES-BOTH` at liveprobe λ̂=0.342 — MERGED (bank-the-analysis)

- **Branch:** `denken/realistic-selfkv-floor` · **Student:** denken · merged ~15:47Z (CPU-only analytic synthesis — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `zjdc7hhh` (group `descent-realistic-selfkv-floor`, finished, NaN-clean, 12.1 MiB). Advisor-verified: 4/4 self-test conditions pass; deliverable is sound (AMBER flags the cautionary FINDING, not the analysis).
- **Primary:** `realistic_selfkv_floor_self_test_passes=1` (4/4). **Test:** `descent_only_realistic_floor_E_T=3.9294` (constant-λ at liveprobe λ̂=0.342).
- **Key finding:** `REALISTIC-FLOOR-MISSES-BOTH`. Converts #172's binary fixable/unfixable into a graded `E[T](λ)` anchored to the openevolve liveprobe (`λ̂_1=(0.6927−0.674)/(0.7287−0.674)=0.342`). At λ̂: descent-only E[T]=3.9294 → **404 TPS (−96)** and both-bugs 4.0485 → **416 TPS (−84)** — **both miss 500**. The binding constraint for BOTH paths is **depth>0 self-KV recovery λ, NOT BUG-1** (the depth-1 BUG-1 fix buys only ~+12 TPS at the floor). Clear-500 thresholds λ*=0.909 (descent) / 0.838 (both-bugs) at τ=1; realistic λ̂=0.342 sits far below. Endpoints reproduce #172 exactly (λ=1→5.0564 resid 0.0; λ=0→3.5346 resid 9e-16). Geometric-decay band (γ=0.7–0.9) all miss too — robust to the deeper-depth assumption.
- **Conclusion:** the load-bearing temper for the packet — fern #174's "both-bugs robust GO" robustness lives at **λ=1**; at the realistic anchor both miss, so the 500 case now rests ENTIRELY on land #71's built kernel demonstrating measured **λ≥~0.84** (the one pre-build measured point is far below the bar). Honest scope: the liveprobe is one BUG-1-present depth-1 point; the depth-1→depth>0 carry is modelled (constant-λ primary, geometric band). Relayed to land #71 (the gating build) + fern #179 (packet reframe: GO conditional on the BUILT λ). denken → #183 (**margin-aware λ-acceptance card** — fold wirbel #175's ±10.9 + kanna #159 σ_hw into the finite-sample-LCB build bar (stricter than 0.838) + the per-depth `q[2..9]` ladder land #71 tests against).

## 2026-06-14 15:47 — PR #175: E[T] second moment — finite-sample TPS CI + distribution gate — 🟢 GREEN / single-draw scatter ±10.9 TPS dominates input band 4.5× — MERGED (bank-the-analysis)

- **Branch:** `wirbel/et-second-moment` · **Student:** wirbel · merged ~15:47Z (CPU-only analytic — no HF Job/submission/served-file change; BASELINE unchanged 481.53; assigned by parallel advisor, reviewed/merged here). W&B `zh1accmi` (group `et-second-moment-tps-ci`, finished, NaN-clean, 33.2 MB). Advisor-verified: 18/18 checks pass.
- **Primary:** `et_second_moment_self_test_passes=1` (18/18). **Test:** `tps_finite_sample_ci_halfwidth=±10.906 TPS` (both-bugs, B=16384).
- **Key finding:** the second moment is a free read off #160's `reach[]` DP — pmf certified exact by two independent enumerations (max_abs_diff 0.0 / 2.8e-17) + 2M-trial MC (5.6e-4). Headline: finite-sample single-draw scatter **±10.906 TPS DOMINATES lawine #168's input band (±2.4) by 4.5×** — single-shot TPS uncertainty is **sampling-limited, not input-limited**. At λ=1 both topologies clear at the 95% LCB: both-bugs [524.5, 546.3] (+24.5), descent-only [509.1, 530.8] (+9.1). σ_L=3.0354 (both-bugs) / 3.0593 (descent). Caveat: iid-CLT lower bound — positive step serial-correlation would widen.
- **Conclusion:** the finite-sample NUMERATOR leg of the launch's total single-shot CI (composes in quadrature with kanna #159 σ_hw=4.86 denominator → ±11.9 total). Conditioned on λ=1 — composes with denken #178's λ-grading as "is the ceiling reached (λ)?" × "single-draw scatter around it (this ±10.9)?". Also hands land #71 a distribution-shape build gate (complements #170's mean gate). Relayed to fern #179 (the σ-quadrature line). wirbel → #184 (**λ-robust verify-tree topology** — try to LOWER the build's λ-bar below 0.838 by front-loading acceptance into λ-insensitive shallow depths + correlation-refined N_eff that closes this iid caveat).

## 2026-06-14 15:20 — PR #173: Descent-walk step cost (salvage-descend accept-prep) — 🟢 GREEN / descending kernel is step-NEUTRAL, descent gain not eroded — MERGED (bank-the-analysis)

- **Branch:** `lawine/descent-walk-step-neutrality` · **Student:** lawine · merged ~15:20Z (local A10G analytic profiling — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `r13idrlx` (group `descent-walk-step-neutrality`, finished, NaN-clean, A10G, ~105s). Advisor-verified: 4/4 self-test legs pass.
- **Primary:** `descent_walk_step_self_test_passes=1` (4/4: linear control reproduces 1.2182 anchor to 0.064%, sign-flip, both clear 500, NaN-clean). **Test:** `descent_walk_step_delta_pct=0.1022%` (faithful all-mismatch worst case).
- **Key finding:** when land #71 swaps the strictly-linear accept-prep kernel for the DESCENDING (salvage-descend) one, the per-step cost is step-NEUTRAL in operative use — realistic marginal **+1.96µs (+0.020%, sub-floor)**, adversarial all-mismatch worst +9.94µs (+0.1022%, AMBER), naive O(depth) ceiling +14.42µs (+0.1482%). Step anchor stays **1.2182** for quoting (`descent_kernel_step_pinned=1.21944` worst). Crucially descent gain NOT eroded: descent-only **519.18** / both-bugs **534.64** at the worst ceiling — clear 500 by ~18/~35 TPS.
- **Conclusion:** upgrades ubel #163's unmeasured "+0 net by design" to a measured ≤0.15% worst-case bound — the descending build's denominator is safe. Follow-ups (pin 1.2182 in the packet, re-run vs real Triton symbol once land #71 assembles, confirm early-terminate collapses the worst case) folded into fern #179 + land #71. lawine → #180 (**argmax-only decode step-realization** — realize ubel #154's conditional 1.2047 output-neutrally to restore descent-only to GO).

## 2026-06-14 15:20 — PR #169: PRECACHE_BENCH tree-footprint calibration-invariance — 🟢 GREEN / K_cal=125.268 INVARIANT under M=32 tree, zero BW drift — MERGED (bank-the-analysis)

- **Branch:** `ubel/precache-bench-tree-footprint-invariance` · **Student:** ubel · merged ~15:20Z (local A10G profiling — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `0czdgugp` (group `precache-bench-tree-footprint`, finished, NaN-clean, ~149s). Advisor-verified: 11/11 self-tests pass.
- **Primary:** `precache_footprint_self_test_passes=1` (11/11). **Test:** `bus_ratio_tree_invariant=1`.
- **Key finding:** the composition constant `K_cal=125.268` (established under the LINEAR fa2sw frontier) is INVARIANT under the M=32 tree footprint with `PRECACHE_BENCH=1` — `k_cal_tree_corrected=125.268` (factor 1.0, zero drift). Load-bearing gate: warmed bandwidth **513.57 GB/s is byte-identical across the full 4.0→20.5 GB footprint sweep** (`warmed_bw_delta_pct=+0.0000%`); bus_ratio linear 1.0368 vs tree-M32 1.0366 (−0.025%, inside the 0.787% band); `official_shift_tps=0.0` for both descent and both-bugs.
- **Conclusion:** K_cal carries linear→tree with ZERO recalibration — both tree projections still clear 500 (corrected). precache-off divergence 3.53% single-shot / 0.007% amortized(512) reconfirms `PRECACHE_BENCH=1` as the named launch dependency. Pins the denominator's calibration constant for the launch packet. ubel → #181 (**τ overlap/coverage-efficiency pin** — close the last unpinned composition factor; quantify any free margin for fern #174's knife-edge).

## 2026-06-14 15:17 — PR #174: Launch verdict at the conservative launch-realized step — 🟢 GREEN / verdict FLIPS descent-only→both-bugs at shipped step 1.2182 — MERGED (bank-the-analysis)

- **Branch:** `fern/conservative-step-launch-verdict` · **Student:** fern · merged ~15:17Z (pure-analytic CPU-only synthesis — no HF Job/submission/served-file change, **no launch authorized**; BASELINE unchanged 481.53). W&B `s2vihqh1` (group `conservative-step-launch-verdict`, finished, NaN-clean, CPU-only). Advisor-verified: 5/5 self-test checks pass.
- **Primary:** `conservative_step_verdict_self_test_passes=1` (5/5). **Test:** `descent_only_p_clear500_at_conservative_step=0.8994`.
- **Key finding:** the binding verdict FLIP. At the SHIPPED step **1.2182** under the full quadrature (kanna σ_hw #159 + input bands), descent-only's P≥0.9 LCB = **499.97 TPS — a knife-edge MISS by 0.035 TPS** (P=0.8994). So descent-only-first flips to **both-bugs** as the robust GO: both-bugs LCB **514.88**, P=0.9959, GO at all three step framings. The 0.035 miss is exactly the **+3.96 TPS LCB** that shipping #154's argmax-only decode (step 1.2047) would restore to descent-only.
- **Conclusion:** correctly defers — "the refreshed packet remains a pre-filled draft, it does NOT authorize a launch"; human-approved `Approval request: HF job` still required (no overclaim). Supersedes the descent-only-first recommendation in fern #167. Drives fern → #179 (**packet refresh** enthroning both-bugs as primary GO + σ_hw-composed CI + denken #172 E[T]-floor caveat) and lawine #180 (the argmax-decode restoration test).

## 2026-06-14 15:17 — PR #172: Descent-E[T] model audit (independent re-derivation) — 🟢 GREEN / 5.0564 triple-confirmed, but adversarial self-KV floor 3.5346 FAILS 500 — MERGED (bank-the-analysis)

- **Branch:** `denken/descent-et-dp-audit` · **Student:** denken · merged ~15:17Z (pure-analytic CPU-only — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `gh8pa4f3` (group `descent-et-dp-audit`, finished, NaN-clean, CPU-only, 25.7 MiB). Advisor-verified: 4/4 self-test conditions pass (cross-method M1≡M2 to 2.7e-15).
- **Primary:** `descent_et_audit_self_test_passes=1`. **Test:** `descent_only_E_T_lower_bound=3.5346`.
- **Key finding:** the descent central **E[T]=5.0564 is now independently triple-confirmed** (backward renewal-reward DP + brute-force path enumeration + imported #135 DP, all agreeing to ~1e-15) — NOT a DP artifact. But the honest caveat is the value-add: the **adversarial self-KV-starvation floor (cause #2, 100% depth>0 starvation) is E[T]=3.5346 → ~363 TPS, FAILS 500 by ~137**. So 5.0564 is OPTIMISTIC not a floor — the 520 projection rests on cause #2 being a FIXABLE build defect (clear-500 needs ≥91% deep-spine spread recovery, λ*≈0.908/0.890).
- **Conclusion:** the load-bearing E[T] caveat for the launch packet — pairs with the depth>0 self-KV plumbing land #71 builds, and with openevolve's liveprobe. Drives denken → #178 (**realistic self-KV E[T] floor** — convert this binary fixable/unfixable into a graded recovery curve E[T](λ) anchored to openevolve's liveprobe, with an explicit clear-500 verdict at the realistic λ̂).

## 2026-06-14 15:17 — PR #159: Hardware-variance envelope σ_hw — 🟢 GREEN / σ_hw=4.86 TPS, cross-allocation-dominated; descent-only single-draw P(clear500)=0.791 — MERGED (bank-the-analysis)

- **Branch:** `kanna/hardware-variance-envelope` · **Student:** kanna · merged ~15:17Z (local A10G cold-server runs — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `i415ucg5` / `1wqmfps0` / `12u7su0b` (group `hardware-variance-envelope`, all finished, NaN-clean). Advisor-verified: 8/8 internal checks pass.
- **Primary:** `hw_variance_envelope_self_test_passes=1` (8/8). **Test:** `sigma_hw_pct=0.9624%` (= 4.86 TPS).
- **Key finding:** σ_hw — the missing DENOMINATOR quadrature leg — is **4.86 TPS** and cross-allocation-dominated (σ_within=0.011% immaterial over 12 fresh cold-server runs vs σ_cross CV 0.96% from frantic-penguin's 3 same-submission draws). 95% band under σ_hw alone **[495.9, 515.0]** straddles 500. Descent-only single-draw **P(clear 500)=0.791** under full quadrature (~21% hardware-scatter fail rate) — NOT the ~0.88 assumed; **best-of-2** official draws restores P≥0.90 on the hardware axis.
- **Conclusion:** composes in quadrature with wirbel #175's finite-sample numerator term for the launch's total single-shot TPS CI, and is the load-bearing input that turns fern #174's knife-edge descent-only verdict (LCB 499.97). Re-draw budget (best-of-2) now armed for the packet. kanna → #177 (**darwin `_IncludedRouter` launch-boot validation** — reuses the same local cold-serve rig to prove the startup-500 fix output-neutral).

## 2026-06-14 15:05 — PR #164: Tree native private-drop directly measured (3 organizer-faithful proxies) — 🟢 GREEN / descent-only IS private-safe, drop 2.04% CI [1.87,2.21] — MERGED (bank-the-analysis)

- **Branch:** `stark/descent-vs-bothbugs-private-decision` · **Student:** stark · merged ~15:05Z (analytic propagation + local sglang proxy scoring — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `5hz3dfrq` (group, finished, NaN-clean). Advisor-verified primary+test.
- **Primary:** `native_proxies_reproduce_flagship_4p3=1.0` (all 3 proxies within ±0.001pp of GT-4.2946%). **Test:** `tree_private_drop_pct_native_ci=2.04%` (descent-only CI mid, band [1.87, 2.21]).
- **Key finding:** replacing #156's single-shape interpolation (1.80%) with 3 independent organizer-faithful proxies (code/casual/sharegpt) count-pooled to the GT-4.3% decode-linear anchor lands the native drop at **2.04% mid (+0.24pp)** — and descent-only still clears 500 at every proxy: central band **[508.5, 510.2]**, worst conservative τ-low corner **504.6** (margin +4.6). `both_bugs_required_private=False` — the spine is NOT a hard private dependency. The +0.24pp is almost entirely cross-domain shape independence (pooling-vs-interpolation on the identical component is only +0.07pp).
- **Conclusion:** descent-only is private-safe, directly measured. Honest limit: the 3-proxy band is a CONSTRUCTION-variance band, not a sampling CI over the real private set — which drove stark → #176 (**adverse domain-skew private stress** — widen to 5–6 calibrated axes, find the worst realistic skew vertex, test whether descent-only's τ-low survives below 500). Binding input for fern #174's packet (replaces the single 1.80% point with the [1.87, 2.21] band).

## 2026-06-14 15:05 — PR #170: Descent over-acceptance signature — 🟢 GREEN / joint (E[T],v) trustworthy region, magnitude complement to #158 — MERGED (bank-the-analysis)

- **Branch:** `wirbel/descent-overaccept-signature` · **Student:** wirbel · merged 15:05:04Z (pure-analytic CPU-only — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `ne7p642c` (group `descent-overaccept-signature`, finished, NaN-clean, 12/12 checks, CPU-only). Advisor-verified via wandb-query: 7/7 checks pass (primary=1, TEST=1.0, `max_et_inflation_at_v0_{descent,both}=0`, `v_at_denken158_point=0.08050847=19/236` exact, `matches_detector=True`, finished).
- **Primary:** `overaccept_signature_self_test_passes=1` (12/12). **Test:** `et_inflation_at_unit_overaccept=1.0` (δ(ε=1)=+1.0 E[T], both topologies — a node count).
- **Key finding:** the joint **(E[T], v) trustworthy acceptance region** land #71's measured tuple must fall in. Over-acceptance commits ε extra nodes/step past the greedy boundary; each is +1 committed token (→+1 E[T]) AND one greedy violation → `E[T](ε)=E[T]*+ε`, `v(ε)=ε/(E[T]*+ε)`, over-accept locus `E[T]=E[T]*/(1−v)`. **Degenerate at v=0:** greedy-exact pins a UNIQUE E[T] (`max_et_inflation_at_v0=0`) ⇒ any E[T] > ceiling REQUIRES v>0 — an inflated E[T] read is over-acceptance, not headroom. Cross-check vs denken #158's binary detector is an **exact count-identity**: `v_at_denken158_point=19/236=0.080508=1−exactness(0.919491)`, `matches_detector=True` — the continuous v(ε) and the empirical binary detector are one quantity at the same operating point. v_tol noise-floor (1/65536) buys <1e-4 E[T] inflation ⇒ any meaningful E[T] above ceiling is over-accept.
- **Conclusion:** the **magnitude complement** to denken #158 — together they bound BUG-2's binding build-risk (the carrier wirbel #165 named) from BOTH the binary (#158: "any violation?") and the magnitude (#170: "is the E[T] inflation explained by violation?") side. Hands land #71 the predicate `land_tuple_in_trustworthy_region(E_T, v, E_T_star=5.2070)` (strict v_tol=0 ⇒ trustworthy ⇔ v=0 AND E[T]≤ceiling); 3 regions TRUSTWORTHY / OVER-ACCEPT-BUG-2 (on locus) / ANOMALOUS (E[T]>ceiling but v≈0 ⇒ DP ceiling itself conservative, investigate not alarm). Stops an inflated E[T] from being mis-read as acceptance headroom before the irreversible shot; does NOT change the clear-500 bar. wirbel → #175 (**E[T] second moment** — finite-benchmark TPS sampling CI + land #71 distributional readout gate: the sampling-uncertainty leg, the DP's 2nd moment to complement this 1st-moment trustworthiness gate).

## 2026-06-14 14:45 — PR #167: Pinned-operating-point launch decision + readiness packet — 🟢 GREEN / descent-only-first GO (96.3%) at pinned drop; decision-geometry CAPSTONE — MERGED (bank-the-analysis)

- **Branch:** `fern/pinned-launch-decision-packet` · **Student:** fern · merged ~14:45Z (pure-analytic CPU-only synthesis — no HF Job/submission/served-file change, **no issue filed, no launch authorized**; BASELINE unchanged 481.53). W&B `l3pdlh22` (group `pinned-launch-decision-packet`, finished, NaN-clean, CPU-only). Advisor-verified: all 4 self-test assertions pass.
- **Primary:** `launch_packet_self_test_passes=1`. **Test:** `descent_only_p_clear500_at_pinned_drop=0.9630`.
- **Key finding:** instantiates the #142/#145/#149/#155/#162 arc at stark #156's pinned drop (1.80% desc / 1.86% both) + realistic bar (E[T]≥4.809). **descent-only → 519.6, P(clear-500)=96.3%, LCB(P≥0.9)=505.6, GO** (BUG-1 deferred: pinned 1.80%≪#162's ~6% binding threshold); both-bugs → 534.8, 99.9%, GO (deferrable insurance). descent-only pinned **exceeds #162's 511.1 by +8.5 TPS** (and at the easier drop). 4 falsifier assertions pass: oracle 2.621→NO-GO, both-bugs→GO, descent-only op-point-specific (GO@1.80% / NO-GO@9% ceiling=494.8 reproducing #162), PENDING/BANKED sets match. Packet = verbatim `Approval request: HF job` block, 6 BANKED / 4 PENDING.
- **Conclusion:** the decision-geometry **capstone** — a pre-filled approval-request draft parameterized on land #71's pending tuple; does NOT authorize a launch. **Two refresh items landed same cycle (correctly flagged PENDING at assembly):** lawine #168 ruled the headline step 1.2047 CONDITIONAL → launch-realized **1.2182** (descent-only tightens to ~513.9 private / LCB≈500, GO-but-tight; both-bugs stays comfortable ~528/LCB~520); denken #166 banked the PPL stamp → PENDING shrinks 4→2 (kanna #159 σ_hw, land #71 tuple). fern → #174 (**conservative-step launch verdict** — re-instantiate at 1.2182, settle descent-only-vs-both-bugs first shot, fold in #168/#166 + new legs #172/#173).

## 2026-06-14 14:39 — PR #168: Step-anchor stack reconciliation — 🟢 GREEN / 4 anchors → ONE launch-realized step 1.2182 (±2.4 TPS) — MERGED (bank-the-analysis)

- **Branch:** `lawine/step-anchor-reconciliation` · **Student:** lawine · merged ~14:39Z (pure-analytic CPU-only synthesis of #136/#154/#161 — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `oti5l4sb` (group `launch-step-reconciliation`, finished, NaN-clean, CPU-only). Advisor-verified: all 4 checks pass.
- **Primary:** `step_reconciliation_self_test_passes=1`. **Test:** `launch_realized_step_both_bugs=1.2182`.
- **Key finding:** the 4 step anchors collapse to **ONE launch-realized step = 1.2182** for both descent-only and both-bugs, ±2.4 TPS roofline↔overlap band. roofline 1.2127 ⟷ overlap 1.2182 are **SUBSTITUTES** (same physical step; the +0.447% = real exposed eager star-attn launch idle 43.3µs/step that survives GEMM overlap — under `PRECACHE_BENCH=1` the served fa2sw stack PAYS it → 1.2182 = served reality; 1.2127 = optimistic edge a fully-graphed-attn build, not shipped, would recover). #161 depth-1 spine adds exactly 0 → both-bugs step == descent step. **#154's 1.2047 held CONDITIONAL** (needs the unshipped argmax-only decode build) → does NOT lower the launch step. Propagation: descent-only 5.0564 → **official 519.96** / both-bugs 5.2070 → **535.44** @ realized (522.29 / 537.84 roofline edge).
- **Conclusion:** closes the last step-DENOMINATOR unknown. Reconciles cleanly with **ubel #163** (1.2182 = shipped-reality-today vs 1.2086 = realizable-if-#154-ships) — fern's packet quotes 1.2182 conservative with 1.2086/4.824 as realizable upside. lawine → #173 (**descent-walk step-neutrality** — does land #71's ACTUAL salvage-descend kernel hold 1.2182, the descent analog of #161's spine measurement).

## 2026-06-14 14:39 — PR #166: Tree-path PPL-margin bound — 🟢 GREEN / PPL gate is M-invariant, worst-case 2.4134≤2.42 — MERGED (bank-the-analysis)

- **Branch:** `denken/tree-path-ppl-margin-bound` · **Student:** denken · merged 14:38:54Z (pure-analytic CPU-only, peak 12.13 MiB — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `z4l8ljd7` (group `tree-path-ppl-margin-bound`, finished, NaN-clean). Advisor-verified: self-tests pass.
- **Primary:** `ppl_margin_bound_self_test_passes=1`. **Test:** `tree_path_ppl_worst_case=2.4134` ≤ 2.42 (structural 2.37667, margin 0.0433).
- **Key finding:** the PPL gate is **M-INVARIANT** — the scorer's PPL is teacher-forced **prefill** (`prompt_logprobs`, `max_tokens:1`), which the M=32 verify **decode** batch never enters → scored tree-path PPL ≡ scored linear PPL = 2.37667, untouched by M=32. The conservative transplant (pretend decode jitter lands on *every* prefill-scored token) still clears at all 3 frequency models — binding extreme 2.4134 — because int4-Marlin batch-variance is mean-zero, argmax-preserving (kanna #87: M=32 max|Δlogit|=0.25, **0 flips**/65536) and symmetric logit noise moves PPL only at 2nd order (PSD softmax Hessian). Break-even ε=0.275–0.351 vs measured 0.25; only the unphysical 2ε/token model breaches.
- **Conclusion:** the PPL-side validity stamp for the launch packet (joins denken #158 greedy-exact → BOTH validity contracts now stamped on the tree path). Honest caveat: bounds the M=32 verify-batch dim only; a change to **prefill chunk geometry** would touch scored PPL and needs a separate audit. Correctly classified the openevolve localizer as a TPS-lane issue orthogonal to the PPL gate. denken → #172 (**descent-E[T] DP audit** — the NUMERATOR twin of this PPL bound: re-derive 5.0564 + conservative lower bound).

## 2026-06-14 14:33 — PR #163: Descent-path host-residency sweep — 🟢 GREEN / residual host ops = 0, field swept clean for land #71 — MERGED (bank-the-analysis)

- **Branch:** `ubel/descent-path-host-residency-sweep` · **Student:** ubel · merged 14:33:59Z (LOCAL single-A10G profiling + subprocess-isolated capture probe — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `dmcskhwi` (group `descent-path-host-residency-sweep`, finished, NaN-clean, 10/10 self-tests). Advisor-verified: PRIMARY pass, both #154/#157 anchors re-discovered, capturability probe 4/4 consistent.
- **Primary:** `host_residency_sweep_self_test_passes=1`. **Test:** `descent_path_residual_host_ops_count=0`; `measured_step_anchor=1.2182`; `net_descent_step_pinned=1.2086`.
- **Key finding:** enumerated all 12 ops on the timed decode window (6 host-resident) and accounted for every one — 2 are ubel's own #154/#157 anchors, 3 are lawine #147's consumed accept-walk sync surface (design sync-free, GPU-hidden), 1 is the structurally-unavoidable terminal output-token sync already inside the 1.2182 anchor. **Residual host ops beyond the two known anchors = 0** → no hidden host-loop landmine survives the sweep. The distinctive leg is the empirical capturability probe (a host Python loop need not register as a sync yet still breaks capture): 4/4 subprocess-isolated cases match the taxonomy with the exact predicted CUDA errors (`host_loop_relocate` → "Cannot copy between CPU and CUDA tensors during CUDA graph capture"; `sync_bound_accept_walk` → "AcceleratorError ... during capture"). Two false alarms correctly cleared: `kv_commit_blocktable_update` (host-bound only if `accept_len` reads to host — keep it a device scalar = zero-copy paged relocate) and `terminal_output_token_ids_cpu` (the one unavoidable stream sync, already in the anchor).
- **Conclusion:** the de-risk green light land #71 needed — `residual=0` means no host-residency landmine beyond #154/#157. Net-step bonus: the realizable build (descent + vectorized relocate +35.3µs + #154 scatter+LP −111.9µs) lands at **1.2086 units — BELOW the 1.2182 anchor** → the clear-500 bar FALLS to 4.824 and the descent cushion RISES 0.178→0.216 E[T]; descent 522 / both-bugs 540 both clear 500 (only the host-loop relocate variant, 32.54 bar / 77 TPS, blows the budget — exactly the landmine #157 already designed out). ubel → #169 (**PRECACHE_BENCH tree-footprint calibration-invariance** — measure whether ubel #148's K_cal=125.268 / +6.019% bus-ratio multiplier holds at the M=32 tree's 20.47 GB footprint vs the linear stack it was calibrated on).

## 2026-06-14 14:33 — PR #165: Shared index-map coherence — 🟢 GREEN / ONE corrected map fixes BOTH bugs (super-additive coupling) — MERGED (bank-the-analysis)

- **Branch:** `wirbel/shared-index-map-coherence` · **Student:** wirbel · merged 14:33:55Z (pure-analytic CPU-only — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `laxllfjl` (group `shared-index-map-coherence`, finished, NaN-clean, 13/13 `check/*`). Advisor-verified: PRIMARY pass, anchors reproduced (neither-fixed 2.621, descent-only 5.0564, both-fixed 5.2070).
- **Primary:** `index_map_coherence_self_test_passes=True`. **Test:** `composed_fix_E_T=5.206954`; `shared_index_map_fixes_both_bugs=True`; `composed_fix_greedy_identity_safe=True`.
- **Key finding:** traces BOTH bugs to the SAME dereference — `_dixie_fused_accept_prep_kernel` (`sitecustomize.py:921`) reads BUG-1 (spine root, pos==0, :945) and BUG-2 (descent walk, :942–951) through the **same** `target_argmax_ptr` at the **same** index base `start_idx+pos`, filled by the **one** upstream `target_logits_indices` gather; the kernel holds NO second map and its `draft==target_argmax` test is already correct. So ONE corrected map (slot-0 own rank-1 row → BUG-1 f→0; descent-ordered node layout → BUG-2) fixes both, and `composed_fix_E_T=5.2070` is computed ONCE = the both-bugs ceiling. The super-additivity is the proof of coupling: the false-independent additive model = 5.1818, the true single-map composition = 5.2070, the **+0.0252 being the coupling a higher spine feeds into the descending branches**.
- **Conclusion:** turns denken #133's "maybe one map, maybe two" into a one-fix build directive — land #71 builds ONE unified fix, not two; the both-bugs private-safe topology rides along with the descent build at near-zero marginal cost (one contract, one validation, lower risk). Binding build-risk carrier remains BUG-2 (the linear→descending structural change, ~19× BUG-1's E[T] lever); BUG-1's slot-0 re-point is a trivial single-index rider. **Converges with land #71's LIVE Component 3** (`comp3_index_map_checks_pass=17`) — resolves analytically what land is confirming empirically. Greedy-safe (changes the upstream gather only; denken #158's GREEDY_EXACT certificate transfers, `--audit-kernel-symbol` armed for the assembled kernel). wirbel → #170 (**descent over-acceptance signature** — the joint (E[T], greedy-violation) acceptance region land's measured tuple must fall in, bounding the BUG-2 over-acceptance risk wirbel #165 itself named).

## 2026-06-14 14:22 — PR #162: Tightened private-safe 500-frontier + land #71 min-recovery build gate — 🟢 GREEN / (λ_min,μ_min)=(0.881,0.735); BUG-1 is CONDITIONAL insurance — MERGED (bank-the-analysis)

- **Branch:** `fern/tightened-private-500-frontier` · **Student:** fern · merged 14:22:18Z (pure-analytic CPU-only synthesis — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `0il5xhji` (group `tightened-private-500-frontier`, finished, NaN-clean, CPU-only). Advisor-verified: all 4 assertions pass, green-area reproduces #149's 0.0300 exact.
- **Primary:** `tightened_frontier_self_test_passes=1`. **Test:** `lambda_mu_min_private_safe=[0.8809, 0.7353]` (λ_min@μ1=0.8809, μ_min@λ1=0.7353).
- **Key finding:** INVERTS #149's joint frontier into land #71's concrete build gate. At ubel #154's realistic bar (E[T]=4.809) + GT drop, land's descent kernel must realize **(λ_min=0.8809 spread @ full width, μ_min=0.7353 width @ full spread)** for P(clear-500)≥0.5 (P≥0.9 LCB: 0.9465/0.8764). λ_min(μ) trade-curve unreachable for μ≲0.73; μ=0.90→0.926; μ=1.00→0.881. Lower bar widens public green 3.00%→3.99% (+0.99pp) but the private haircut shrinks private-safe to 1.72% (−1.28pp vs #149) — net the private constraint dominates. Self-test: reproduces stark's 9.88% breakeven by construction (4a), anchors land RED/GREEN/INDETERMINATE under the updated bar (4c), gate intercepts P=0.5 (4d). Axis mapping is #149-faithful (λ=deep-spine spread q[2:], μ=branch width ρ_cond — both BUG-2 facets; depth-1 q₁=BUG-1 held at ρ-opt) — advisor-confirmed the only mapping that reproduces 9.88%.
- **Conclusion:** the decisive hand-off — **BUG-1 (wirbel #160 spine) is a CONDITIONAL requirement, not unconditional.** At the GT operating point BUG-2-descent-alone (depth-1 UNfixed) already clears at proj 511.1 (P=0.845); BUG-1 becomes mandatory only above ~6–7% private drop (at the 9% ceiling the BUG-1-unfixed corner falls to 494.8). **This converges cleanly with stark #156 (merged same cycle), which PINNED the realized drop at 1.80%/1.86% ≪ 6%** → at the pinned operating point land #71's BUG-2 descent is the *unconditional* gate; wirbel #160's spine is *deferrable insurance*. Non-blocking NOTE acknowledged: #149's committed frontier JSON carries a pre-#142-τ-floor-fix `tau_band.low` (0.9983→0.99243); fern correctly did NOT mutate the merged file and its τ_central=1.0 numbers are unaffected (logged for whoever next refreshes the #149 artifact). fern → #167 (**pinned-operating-point launch decision + readiness packet** — instantiate the GO/NO-GO at stark #156's pinned drop + assemble the verbatim `Approval request: HF job` projection/validity block).

## 2026-06-14 14:22 — PR #161: Both-bugs accept-prep step cost — 🟢 GREEN / step-neutral, 537.8 hardened assumed→measured — MERGED (bank-the-analysis)

- **Branch:** `lawine/both-bugs-step-cost` · **Student:** lawine · merged 14:22:20Z (LOCAL A10G profiling, peak 0.258 GB / 49 s, paired 5-round — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `2heov0f4` (group `both-bugs-step-cost`, finished, NaN-clean). Advisor-verified: self-test step 1.21792, both pinned-officials match.
- **Primary:** `step_cost_self_test_passes=1` (reproduced 1.2179 vs 1.2182 anchor, +0.023%). **Test:** `both_bugs_step_delta_pct=+0.0000%`; `both_bugs_official_pinned_roofline=537.84` (drop 0.0 TPS) / `overlap=535.43`.
- **Key finding:** the depth-1 spine fix (BUG-1) is **step-neutral** vs descent-only — paired device-busy marginal **−0.031µs** (all 5 rounds negative, sign flips run-to-run ⇒ true marginal IS zero). Physics-first: BUG-1 is **upstream plumbing** (corrects `target_logits_indices` so the root verify-row compares against the drafter's rank-1 token; denken #133: 96% plumbing, c_intrinsic=0.0), UPSTREAM of `_dixie_fused_accept_prep_kernel`. The served kernel is byte-identical (`sitecustomize.py:921`); the fix changes the VALUES (more depth-1 matches → E[T] 5.0564→5.2070), not the op-count. Accept-prep is 0.0195% of the 9150µs step, GPU-hidden. Even an explicit kernel-resident `accept_prep_depth1_spine` worst-case variant came in at −0.016µs (nil) → an in-kernel spine is step-safe too.
- **Conclusion:** closes the **last step-denominator unknown** — the both-bugs 537.8 official is now *measured*, not assumed (joins #152/#153 tree-width closure + wirbel #160's spine spec). Combined with #162 (BUG-1 deferrable at the pinned drop) + stark #156 (pinned 1.80%): the descent-only shot clears at the measured step, and adding the spine is a free-on-step upgrade for the 9%-band-ceiling margin. Honest methodology: first pass mis-flagged RED gating on statistical within-CI (device-busy is so repeatable a sub-100ns marginal sits outside its tiny ci95 0.005µs yet is physically nil); corrected to lead with PRACTICAL significance (|step delta|<0.10% ≈ 0.5 TPS), retaining `marginal_within_ci` as a diagnostic not a gate — advisor-confirmed the right call. lawine → #168 (**step-anchor stack reconciliation** — collapse roofline 1.2127 / overlap 1.2182 / scatter-LP 1.2047 / both-bugs-neutral into the single launch-realized step for fern's packet).

## 2026-06-14 14:08 — PR #158: Descent greedy-exactness differential harness (per-token accepted==argmax contract gate) — 🟢 GREEN / 2nd validity leg, armed for land #71 — MERGED (bank-the-analysis)

- **Branch:** `denken/descent-greedy-exact-harness` · **Student:** denken · merged 14:08:30Z (pure-analytic CPU-only — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `opbbrnce` (group `descent-greedy-exact-harness`, finished, NaN-clean 23 metrics, CPU-only). Advisor-verified: all 4 selftest sub-checks=1, metrics bit-match log.
- **Primary:** `greedy_exact_harness_self_test_passes=1`. **Test:** `linear_stack_exactness_rate=1.0`; `bug2_exactness_rate=0.9194` (19 violations CAUGHT); `known_good_ppl=2.37666` (margin 0.0433 under 2.42).
- **Key finding:** imports `sitecustomize._get_fused_accept_prep_kernel` — the EXACT kernel `serve.py:429` invokes — and runs a 35-case battery asserting each committed token == in-step argmax of the reference logits. Known-good linear stack passes at 1.0; the BUG-2 salvage-no-descend kernel is caught at 0.9194 with **every violation localized** (e.g. req=7 pos=0 committed=3000 ref_argmax=3777). `--audit-kernel-symbol module:func` arms it for land #71's kernel: correct linear→descending exits 0 (GREEDY_EXACT); silent non-argmax commit exits 1 (VIOLATION). Design note (reconciled): per-step in-step-argmax IS the contract reference; the literal completion-sha256 spec-vs-AR comparison is DIVERGENT (0.6169 / 118-of-128) = documented int4-Marlin batch-variance (Issue #124, NOT a contract violation).
- **Conclusion:** the 2nd VALIDITY leg (does-it-honor-the-contract) complementing #150 (does-it-score) — the per-token catcher the scorer's no-token-check + #150's aggregate-PPL gate cannot see. land #71 gets a pre-merge contract gate to run against its own kernel before submitting. denken → #166 (**tree-path PPL-margin bound** — the aggregate-PPL complement: bound the M=32 batched-verify worst-case PPL vs 2.42 under int4-Marlin batch-variance, the dimension #150/#158 assume but don't bound).

## 2026-06-14 14:08 — PR #160: Depth-1 spine (BUG-1) build spec + both-bugs E[T] — 🟢 GREEN / buildable f→0 input-contract fix — MERGED (bank-the-analysis)

- **Branch:** `wirbel/depth1-spine-build-spec` · **Student:** wirbel · merged 14:08:28Z (pure-analytic CPU-only — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `x8vffgbs` (group `depth1-spine-build-spec`, finished, NaN-clean 29 metrics, CPU-only). Advisor-verified: 10/10 checks pass, metrics bit-match log.
- **Primary:** `spine_spec_self_test_passes=True`. **Test:** `both_bugs_E_T_specced=5.2070`; `descent_only_E_T=5.0564`; `idealization_gap=0`; `spine_fix_greedy_identity_safe=True`.
- **Key finding:** the depth-1 accept gap (oracle 0.679 vs target q₁=0.7287) is **one input-contract bug, not kernel arithmetic**: the spine-root verify slot (pos=0) reads a rank-2-contaminated logits row because `target_logits_indices` indexes the wrong rank. Contamination model `q₁(f)=(1−f)·q_true + f·ρ₂` (q_true=0.7287, ρ₂=0.4165 branch-hit floor); spec'd fix is **f→0** = index the spine root's own logits row. Annotated against `sitecustomize.py:942-951`.
- **Conclusion:** the buildable BUG-1 half of the two-bug recovery for land #71. Banked, it closes descent-only (5.0564 → ~522) → both-bugs (5.2070 → ~535–538, clears 500 AND 530). Combined with land #71's descent kernel (BUG-2), the spec says the realized spine reaches the both-bugs anchor; does NOT itself authorize a launch. Flagged coordination: denken #133's shared-index-map hypothesis — may the SAME corrected `target_logits_indices` fix BOTH bugs? wirbel → #165 (**shared index-map coherence** — answer that flag: ONE unified correction vs TWO independent fixes for land's build).

## 2026-06-14 13:56 — PR #157: relocate_salvaged_kv host-loop audit — 🔴 LIVE-LANDMINE / descent-path build-blocker — MERGED (bank-the-analysis) [parallel-advisor merged]

- **Branch:** `ubel/salvage-kv-relocation-audit` · **Student:** ubel · merged 13:56:38Z by parallel advisor (LOCAL A10G profiling + analysis — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `rh8ysitz`. Verified NaN-clean: 11/11 self-test checks, `equivalence_rate=1.0` (bit-exact, max_abs_err=0.0).
- **Primary:** `salvage_kv_audit_self_test_passes=1` (11/11). **Test:** `recoverable_step_pct_salvage_kv=569.9%`. **VERDICT: LIVE-LANDMINE / build-blocker.**
- **Key finding:** the `relocate_salvaged_kv` host-loop runs **145.2 ms/call → 55.46 ms/step → descent 77 TPS**; the gpu_vectorized path is **92.4 µs/call → 35.3 µs/step → descent 516 TPS** (paged_slotmap ~517 TPS) — a **1571× speedup**, all three bit-exact (equivalence_rate=1.0). Implied salvage frac 0.342. This is EXACTLY the op land #71's descent fix arms — if land's descending kernel keeps the host-loop relocation, the descent path collapses to 77 TPS regardless of E[T].
- **Conclusion:** the SECOND host-side step tax (stacks on ubel #154's decode-path scatter+LP avoidance). A live landmine that MUST be designed out of land #71's descent build (vectorized GPU or paged slotmap), not a dead fallback. Build-blocker hand-off to land #71: descent salvage relocation must be vectorized-GPU/paged, never host-loop. [ubel reassigned by parallel advisor → #163 descent-path host-residency/graph-capture sweep.]

## 2026-06-14 13:56 — PR #156: Tree private-drop reconcile — pin the native drop vs flagship GT-4.3% — 🟢 GREEN / descent-only IS private-safe at pinned 1.80% — MERGED (bank-the-analysis) [parallel-advisor merged]

- **Branch:** `stark/tree-private-drop-reconcile` · **Student:** stark · merged 13:56:36Z by parallel advisor (sglang vllm-chat scored reconcile — no HF Job/leaderboard official spend; BASELINE unchanged 481.53). W&B `6wtn6790` / `t2tlqzxc`. Verified NaN-clean.
- **Primary:** `harness_pin_reproduces_flagship_4p3=True` (calibrated linear 4.29% vs GT 4.29%). **Test:** `tree_private_drop_pct_pinned=1.80%` (descent-only) / 1.86% (both-bugs) → tree 510.6 TPS descent-only.
- **Key finding:** the pinned protocol (sglang vllm-chat scored, anchored to GT-4.3%) measures the ACTUAL tree private drop at **1.80% (descent) / 1.86% (both-bugs)** — far below stark #151's feared 5.89%. This **REFUTES #151's central fear**: descent-only is ACTUALLY private-safe at the pinned drop (510.6 TPS). The earlier 19.6%/11.3% values were harness/proxy artifacts (decode_outputs.py under-reads; hard-proxy ~2.5× real drop). The descent walk is ~0.42× as private-fragile as linear. At the 9%/10.68% band-ceiling anchors descent-only fails (499.9/495.9 TPS) — only both-bugs clears there.
- **Conclusion:** PINS the private-drop value that fern #162's tightened-frontier `--private-drop` parameter consumes, and largely rehabilitates descent-only as private-safe (cushions #149's tight green corner). Strengthens #151's both-bugs safety; supersedes the proxy-inflated drops. [stark reassigned by parallel advisor → #164 native private-drop decomposition.]

## 2026-06-14 13:40 — PR #155: Approval-projection consolidator (7th/capstone projection-spine instrument + τ-floor fix) — 🟢 GREEN / one-call GO/NO-GO, validity-line structurally complete — MERGED (bank-the-analysis)

- **Branch:** `fern/approval-projection-consolidator` · **Student:** fern · merged 13:40:49Z (pure-analytic CPU-only synthesis — no HF Job/submission/served-file change; BASELINE unchanged 481.53). W&B `gd1ok9zd` (group `approval-projection-consolidator`, finished ~1 s CPU-only). Advisor-verified: `consolidator_self_test_passes=1`, `p_clear_500_at_oracle=0.0`, `self_test_anchors_ok=1`, `self_test_bitmatch_ok=1`, NaN-clean.
- **Primary:** `consolidator_self_test_passes=1`. **Test:** `p_clear_500_at_oracle=0.0` (expected ≈0 for the as-built oracle).
- **Key finding:** ONE entry point `consolidate(E_T, branch_hit, λ, μ, step, ppl, τ) → {proj_tps, ci, p_clear_500, validity_gate, binding_leg}` that UNIONS the six projection-spine legs (#142 gate · wirbel #146 sampling-CI · ubel #148 calibration · #149 decision-geometry · denken #150 validity contract · #136/#147 measured-step) into a single quadrature-propagated verdict with the **binding leg named on every call**. Load-bearing correctness fix: caught + fixed a ~0.6% τ-optimism in its OWN #142 gate (τ-low floor SplitK-class 0.9983 → tree-class 0.9924318649123313, now **bound to ubel #148's `scale_floor`** — one source of truth, can't drift again); central column untouched, only the conservative corner tightens (conservative E[T]-to-clear-500 rises 4.8707→4.8995, supply ceiling at τ-low 536.93→533.77). 3 bracketing anchors land exactly: oracle 2.621→269.5 robust-RED p=0 NO-GO / both-bugs 5.207→535.4 robust-GREEN p=0.999 GO / boundary 4.862→500 INDETERMINATE p=0.5 HOLD. Bit-match proof (faithful union, not re-derivation): wirbel #146 CIs + ubel #148 K_cal band [124.282,125.268] + #149 green-area 0.0300 all exact.
- **Conclusion:** the **decisive behavior** is the partial-recovery veto — a tuple (E_T=4.90, λ=0.6, μ=1.0, ppl=2.40) returns central proj 503.9 (naïvely "GO" on the scalar CI) but `p_clear_500≈3e-6` with `binding_leg=decision_geometry`, because (0.6,1.0) never reaches #149's ~3% green corner; the joint frontier OVERRIDES the optimistic scalar CI — exactly the trap a hand-chained under-pressure synthesis walks into right before the irreversible shot. This is the **CAPSTONE**: the launch evidence-line is now structurally complete (6 instruments + the consolidator). State PENDING/ARMED/PENDING (`land_measured_pending=1`) — it IS the verbatim projection block of the future `Approval request: HF job` issue, awaiting land #71's measured tuple `(E[T], branch_hit ρ₂, spread_λ, width_μ, step, ppl, boots, completed)` that #142/#146/#150 already consume. Adds 0 TPS; does NOT authorize a launch. fern → #162 (**tightened private-safe 500-frontier + land #71 min-recovery build gate** — fold ubel #154's lower bar + stark #151's private-stability into #149's frontier, INVERT to land's (λ_min,μ_min) build target).

## 2026-06-14 13:31 — PR #153: Verify-step(M) cost curve — is the depth-9 step flat in M (free tree-growth)? — 🟢 GREEN / KNEE_AT_32, tree-growth is NOT free — MERGED (bank-the-analysis) [parallel-advisor merged]

- **Branch:** `lawine/verify-step-m-curve` · **Student:** lawine · merged 13:30:59Z by parallel advisor (LOCAL A10G real-int4-Marlin timing, 20.47 GB / 189 s, median-of-3 — no HF Job/submission; BASELINE unchanged 481.53). W&B `ma0qlpas` (group `verify-step-m-cost-curve`). Verified NaN-clean: all 4 validation flags green (`m32_reproduces_1p2182=1`, `m8_graphed_reproduces_1p0=1`, `r_attn_matches_107=1`, `r_gemm_matches_107=1`, `verdict_knee_at_32=1`).
- **Primary:** `verify_step_flat_M_ceiling=32`. **Test:** `step_M128_rel_increase_pct=+122.05%`.
- **Key finding:** step(M) is NOT flat — it's an int4-Marlin 16-row-tile STAIRCASE. M=32 = exactly 2 Marlin tiles = the largest tree that amortizes a single HBM weight-read wave on the A10G (GA102 sm_86); the 33rd candidate row opens a 3rd tile = fresh weight re-read = +30% GEMM. Net step +19.1% at M=48, +122% at M=128. M≤32 is weight-read-bound/flat (GEMM rises only +17.9% across M=8→32 — the "fp32-free" BW regime); M>32 is wave-bound/linear. Drafter-fill is NOT the binding term (grows 1.24× M32→128 vs the verify-GEMM's 3.05×; its step-share FALLS 20.1%→9.4%). M=32 reproduces lawine #136's 1.2182 anchor (1.2187, Δ0.04% — independent 3rd measurement).
- **Conclusion:** "a bigger draft tree is nearly free" is FALSIFIED for M_crit>32. This is the COST curve under wirbel #152's topology DP: #152 says M=32 is E[T]-optimal, this says M=32 is the largest single-wave tree — the two meet exactly, tree-width is closed from both value and cost sides. Corollary for the DP: within a tile band step is flat, so M∈{33..47} are dominated by M=48; only band-tops {48,64,96,128} are worth scoring (and all LOSE official TPS). Three reads now converge on where the recoverable step budget lives: this (NOT tree width) + ubel #154 (IS the scatter+LP wrapper, +4.3–5.6 TPS) + kanna #138 (NOT the sparse-argmax tile). lawine → #161 (both-bugs accept-prep step cost — does the depth-1 spine fit the step budget).

## 2026-06-14 13:30 — PR #152: Topology re-opt — does re-allocating the M=32 build array against the measured ladder clear 530? — 🟢 GREEN / NO (M=32 already near-optimal), de-risks pinned topology — MERGED (bank-the-analysis) [parallel-advisor merged]

- **Branch:** `wirbel/topology-m-reopt` · **Student:** wirbel · merged 13:30:55Z by parallel advisor (CPU-only analytic DP, 34.8 MB / ~18 s — no HF Job/submission/GPU/kernel-build; greedy untouched; BASELINE unchanged 481.53). W&B `f2hxitrk` (group `topology-m-reopt`). Verified NaN-clean: `topology_reopt_official_tps=523.02`, `topology_reopt_clears_530=0`, `topology_dp_self_test_passes=1`.
- **Primary:** `topology_reopt_official_tps=523.0` (optimal in-scope). **Test:** `topology_reopt_clears_530=0`.
- **Key finding:** re-allocating the M=32 build array against the MEASURED oracle ladder does NOT clear 530. The deployed M=32 pinned shape (wirbel #83) is already within +0.14% of the greedy-M32 optimum; re-opt buys only +0.0071 E[T] (+0.73 TPS) — a wash. Marginal-node curve confirms M=32 is the wall: last-kept node #31 (marginal 0.0262) ≈ first-dropped #32 (0.0262); the descent-only E[T] curve is FLAT at the optimum and every larger rung (M∈{48,64,96,128}) LOSES official TPS (the Marlin int4 step staircase overwhelms sub-0.03 E[T]/node gains). Self-test reproduces merged anchors bit-for-bit (descent-only E[T] 5.0564 ✓, both-bugs 5.2070 ✓; MC×DP |Δ|=0.0006).
- **Conclusion:** DE-RISKS the pinned topology for launch — the build team does NOT need to re-shape the tree; the 530 gap is an ACCEPTANCE (depth-1 spine / bug-1) problem, not a tree-allocation problem. The only large-TPS lever on the topology side is the BUG-1 depth-1 spine fix (both-bugs → 537.8 clears 530), and it does so at the already-pinned M=32 topology — no rebuild risk. Converges with lawine #153 (merged alongside): #152 (value side) says M=32 is E[T]-optimal, #153 (cost side) says M=32 is the largest single-wave tree — "grow the tree for free" closed from both sides. wirbel → #160 (depth-1 spine bug-1 build spec + both-bugs E[T]).

## 2026-06-14 13:21 — PR #154: Step-denominator reduction audit (decode-path scatter + LP avoidance) — 🟢 GREEN / +4.3–5.6 TPS, lowers clear-500 bar to 4.808–4.820 — MERGED (bank-the-analysis)

- **Branch:** `ubel/step-denominator-reduction` · **Student:** ubel · merged 13:21:25Z (LOCAL A10G profiling + analysis — no HF Job/submission/served-file change; greedy/PPL untouched; BASELINE unchanged 481.53). W&B `zioer4bm` (group `step-denominator-reduction`). Advisor-verified: every reported value bit-matches the log, NaN-clean.
- **Primary:** `step_reduction_audit_self_test_passes=1` (8/8 checks). **Test:** `recoverable_step_pct=0.857%` (conservative) / `1.108%` (realistic) — tree M=32 @ clear-500 bar E[T]=4.862.
- **Key finding:** the FIRST non-drafter, non-quant `step_time` (denominator) lever to clear greedy-safety since the GEMM-bandwidth lanes closed. Leg 1 = decode-path `[M,262144]` scatter + LogitsProcessor-wrapper avoidance (argmax-only, greedy-token-identical, researcher RANK-1, equivalence_rate=1.0): reproduces denken #144's M=8 anchors (Marlin GEMM 38.27 µs, scatter `index_copy_` 8.15 µs, full `compute_logits` 135.82 µs ⇒ LP-share ≈89 µs) and extends to M=32 — **97.5 µs/step (linear) → 111.9 µs (tree)** avoidable. Net **+4.3–5.6 TPS at the bar**, and — load-bearing — **lowers the clear-500 bar from E[T]≥4.862 to 4.808–4.820** (Δ≈0.04–0.05 in E[T]). Cross-confirmed by kanna #138 (closed same cycle): the sparse-argmax tile is OFF the K7 critical path (block16 8.485 ms ≈ block64 8.486 ms) ⇒ the recoverable budget lives in the scatter+LP wrapper exactly where this audit went, not the argmax kernel — two independent reads converge.
- **Conclusion:** widens #149's tight ~3% green corner (a partial-(λ,μ) landing that was RED at bar 4.862 may be GREEN at 4.808) and cushions stark #151's AMBER private-margin (a lower bar buys back private-drop tolerance) — multiplicative insurance on land #71's descent, not a substitute. Build hand-off: argmax-only decode path is greedy-safe → carry in the tree manifest, credited +4.3–5.6 TPS. ubel → #157 (**`relocate_salvaged_kv` host-loop audit** — the SECOND host-side step tax, on the salvage/descent path: chiku-inu CPU p90 335 ms vs GPU p50 19.18 ms — live landmine or dead fallback? stacks on #154).

## 2026-06-14 13:21 — PR #150: Local tree-submission validity preflight (READY/NOT-READY) — 🟢 GREEN / 4/4 self-tests pass; validity leg complete — MERGED (bank-the-analysis)

- **Branch:** `denken/tree-submission-preflight` · **Student:** denken · merged 13:21:21Z (LOCAL A10G + CPU — no HF Job/leaderboard spend/served-file change; BASELINE unchanged 481.53). W&B `9ptw7jxb` (group `tree-submission-preflight`). Advisor-verified: all 4 gate sub-flags + `known_good_ready=1`, NaN-clean.
- **Primary:** `harness_self_test_passes=1` (4/4). **Test:** `live_preflight_ready=1` (logged to W&B as `known_good_ready=1` — naming nit, same substance; flagged to student).
- **Key finding:** validates a fully-assembled submission against the scorer's THREE hard validity gates → one READY/NOT-READY verdict, locally. Self-test: known-good linear stack → READY (A∧B∧C); injected boot-fault → NOT-READY naming **Gate A**; over-cap PPL → **Gate B**; under-count completion → **Gate C**. Answers a DIFFERENT question from the TPS legs: not "will it clear 500" but "will it SCORE AT ALL" (boots ∧ PPL≤2.42 ∧ 128/128).
- **Conclusion:** completes the VALIDITY leg — the 6th/final projection-spine instrument (joins #142 scalar gate + #146 sampling CI + #147 sync-audit + #148 calibration + #149 decision-geometry). fern #155's consolidator gates its whole verdict on this: NOT-READY ⇒ NO-GO irrespective of TPS. Status: ARMED/PENDING land #71 (READY confirmed on the known-good linear served stack; the tree drop-in fires the instant land's stack assembles). denken → #158 (**descent greedy-EXACTNESS differential harness** — per-token `accepted==target_argmax` contract-correctness, the BUG-2-class catcher the scorer's no-token-check + #150's aggregate-PPL gate cannot see).

## 2026-06-14 13:21 — PR #138: K-sweep re-characterization with block64 — ⚪ NULL / K*=7 unchanged; block64 is K-neutral — CLOSED (clean negative)

- **Branch:** `kanna/k-sweep-block64-reopt` · **Student:** kanna · closed 13:21Z (LOCAL A10G profiling, N=3 median — no HF Job/submission; BASELINE unchanged 481.53). W&B group `k-sweep-block64-reopt`: K6 `6765tk64` / K7 `acd7vk07` / K8 `a19f3hyb` / K9 `ez25rlh4` / summary `urwmw2yi` / topk `2hi1o1wg`.
- **Primary:** `k_optimal_wall_tps_block64=454.045` (K7). **Test:** `k_star_block64=7`. Anchor K7-block16=454.190 reproduces lawine #90 (454.338, +0.03%).
- **Key finding:** hypothesis refuted — block64 frees ZERO step-time budget at K7 (block16 8.485 ms ≈ block64 8.486 ms), so the sparse-argmax tile is OFF the critical path and K* cannot shift. K8/K9 stay −13/−16 TPS (lawine #90 cliff intact); CENTROID_TOP_K=64 stays optimal (topk128 −3.9 TPS, no accept gain). block64 is greedy-token-identical (128/128) and K-neutral — a safe no-regression manifest flag, but **0 standalone TPS** on this stack.
- **Conclusion:** closed dead-end (no K-shift lever), two useful banks: (1) de-risks the block64 manifest choice for land #71 (greedy-safe + K-neutral confirmed — carry it, credit 0 TPS; the #137 +0.085% is within noise); (2) kanna's own follow-up ("profile where the K7 step-time goes — verify GEMM/KV/sampler, not the argmax tile") INDEPENDENTLY confirms ubel #154's finding that the recoverable step% lives in the scatter+LP wrapper. kanna → #159 (**hardware-variance envelope σ_hw** — the missing 4th quadrature term in fern #155: within-allocation clock/thermal/cold-start + cross-draw bound; answers stark #151's "is 505 safely >500?").

## 2026-06-14 13:05 — PR #151: Tree private-acceptance gap — 🟢 GREEN / descent-only NOT private-robust (tol 5.89%), both-bugs IS (9.88%) — MERGED (bank-the-analysis)

- **Branch:** `stark/tree-private-acceptance-gap` · **Student:** stark · merged ~13:05Z (LOCAL single-A10G profiling ≈20.7 GiB + CPU analysis — no HF Job/submission/served-file change; greedy/PPL untouched; BASELINE unchanged 481.53). W&B `box0yfh9` (descent-walk E[T] propagation) / `ytxfi6zk` (private-ladder calibration), group `tree-private-acceptance-gap`. Advisor-verified: box0yfh9 `selftest/passes=True`, NaN-clean.
- **Primary:** `tree_private_tps_proj=505.46` (descent-only, on the private-proxy ladder). **Test:** `tree_private_clears_500=1` (true, but knife-edge — inside ±1% of 500). Drop-tolerance: descent-only **5.891%**, both-bugs **9.880%**; tps_band [449.6, 506.3].
- **Key finding:** the FIRST quantification of BASELINE.md's #1 documented launch risk ("top drafter stacks lose 4–9% TPS on the private set; submissions die on the 5% reproduction gap, not on PPL") against the **tree** path. The tree's E[T] is *somewhat* robust to the spine haircut (branch-rescue absorbs ~35% of the linear drop), BUT the as-built descent-only projection clears 500 publicly by only ~20 TPS, and at the organizer's measured GT-4.3% private drop it lands at **505.5 — inside the ±1% precache/harness uncertainty of the 500 line**. Its drop-tolerance is **5.89%**, so it survives the *measured* flagship drop but **fails across 6–9% of the documented 4–9% band**. The **both-bugs (bug-1 depth-1 spine) fix lifts tolerance to 9.88%**, covering the whole band with ≥+20 TPS at GT-4.3%. The biggest private deficit is the depth-1 conditional (−17.8%).
- **Conclusion:** materially changes launch sequencing — **do NOT launch descent-only expecting a safe private clear; the private-stable shot is the both-bugs topology** (descent + bug-1 depth-1 spine), or a drafter with stronger private depth-1. This **elevates the bug-1 spine fix from "the 522→538 margin" to a private-stability requirement** for land #71. The private-stability leg of the launch evidence-line (joins #142/#145/#146/#147/#148/#149 + denken #150 validity + fern #155 consolidator). stark → #156 (**pin the tree's TRUE private drop** — reconcile the 2–4× harness gap 4.3%/11.3%/19.6% → which proxy matches the organizer's private re-run, and measure the tree-specific drop under it; the 505.46 verdict assumes the linear stack's 4.3%, but the tree's deeper drafter dependence may drop more).

## 2026-06-14 12:55 — PR #149: Joint (spread × width) clears-500 frontier map — 🟢 GREEN / self-test passes; only ~3% of the recovery square clears 500 — MERGED (bank-the-analysis)

- **Branch:** `fern/joint-spread-width-500-frontier` · **Student:** fern · merged 12:55:35Z (LOCAL CPU-only analytic — no GPU/vLLM/HF Job/submission; BASELINE unchanged 481.53). W&B `7q19axht` (group `joint-spread-width-500-frontier`).
- **Primary:** `joint_frontier_self_test_passes=1` (reproduces every #145 anchor to 0.01%: corner (1,1)→537.84, (λ=0,μ=1)→376.26; λ-intercept 0.90 @ μ=1; μ-intercept 0.70 @ λ=1). **Test:** `green_region_area_fraction=0.0300` (3.00% @ measured step / 3.37% @ roofline).
- **Key finding:** `--joint-frontier` mode added to #145's `scripts/profiler/deep_spine_width_spread_decomp.py` (654 insertions, 0 deletions — #145 default flow byte-identical) lifts the two 1-D recovery slices into the full (λ spread × μ width) ∈ [0,1]² decision surface. The decisive result: **only ~3% of the recovery square clears 500** — the GREEN region is a tight top-right corner, NOT a forgiving band. ⇒ **partial recovery of both facets lands mostly RED**; land's descent must push BOTH spread and width near-fully into the corner, it cannot half-fix one and coast. No contradiction with "descent-only → 522 ✅" (#134): that 522 is the descent *fully* working (a specific high-(λ,μ) point, depth-1 left as the 522→538 margin); #149 characterizes the *partial-recovery interior*, which is mostly sub-500. Also surfaced the **`relocate_salvaged_kv` host-bound Python loop** (chiku-inu `20260614-111022-934`: cpu p90 335 ms vs gpu p50 19.18 ms) as the live-step confirmation.
- **Conclusion:** the 500 verdict is **corner-sensitive** — land's measured (λ,μ) landing point, not just his headline E[T], is what the gate must read. Decision-geometry leg of the launch evidence-line complete (pairs with #142 scalar gate + #145 1-D slices + #146 sampling CI + #148 calibration + #150 validity). fern → #155 (approval-issue projection-CI **consolidator** — rolls #136/#146/#148/#149/#150 into one P(clear-500) + GO/NO-GO, AND fixes the τ-floor bug ubel #148 flagged in fern's own #142 gate: SplitK-class 0.9983 → tree-class 0.9924).

## 2026-06-14 12:40 — PR #148: K_cal tree-transfer validation — 🟢 GREEN / K_cal transfers to the tree (band 0.787% one-sided↓) — MERGED (bank-the-analysis)

- **Branch:** `ubel/kcal-tree-transfer-validation` · **Student:** ubel · merged 12:40:28Z (LOCAL CPU-only analytic, ~29 MB RSS — no GPU/vLLM/model load/HF Job/submission; BASELINE unchanged 481.53). W&B `y8ihyogv` (group `kcal-tree-transfer-validation`).
- **Primary:** `kcal_decomp_self_test_passes=1` (9/9 checks: K_cal=125.267950 exact; multiplier 1.0601865 pooled / 1.05985 locked; decomposition closes abs_err 0.0; reproduces 522.29/537.84). **Test:** `kcal_tree_transfer_band_width_pct=0.787`.
- **Key finding:** the calibration leg of the launch evidence-line, now quantified. The local→official **+6.019% multiplier decomposes** into: prompt-set/output_len/warmup/concurrency = **0%** (tree-invariant); scorer prefill/TTFT amortization ≈ **0%** (neutralized — the deployed `PRECACHE_BENCH=1` replays the 128 bench prompts during untimed warmup → the *timed* window is pure-decode for both linear and tree); residual **+6.019% = GPU clock/thermal/power bus ratio** ("the bus is the wall"), a hardware/scorer-class property held invariant across linear↔tree. So the only genuinely tree-sensitive leg is the small **clock-exposure** term (M=32 tree's compute-exposed verify-GEMM/tree-mask fraction transfers at the *clock* ratio not the *bus* ratio; bounded by #126's tree-class τ floor 0.9924). Band **K_cal ∈ [124.282, 125.268], one-sided↓, width 0.787%** — far inside the PR's own "3% drift → ±15 TPS flips GREEN→RED" tripwire. Propagated: 522→[518.2, 522.3], 538→[533.6, 537.8] (both lower edges clear 500 at K_lo). In quadrature with wirbel #146's sampling leg, the 522 GREEN survives until the **sampling leg alone exceeds 4.19%** → calibration is **not** the binding 500-boundary constraint.
- **Two findings (flagged, not unilaterally edited):** (1) doc arithmetic — "1.06019 = 481.53/454.338" is actually 1.05985; canonical 1.06019 uses the 9-run pooled mean 454.1937; 0.032% apart, both reproduced (harmless). (2) fern #142's committed gate carries the **SplitK-class τ floor 0.9983** for a *tree* projection where the tree-correct floor is #126's **0.9924** (~0.6% optimism on the tau-low corner) — flagged to fold into the decision-geometry lane before the gate is quoted in the approval issue; non-blocking (clears 500 with margin even at the tighter floor).
- **Conclusion:** clean de-risk — K_cal is ~tree-invariant. **CRITICAL hand-off → land #71 / build-team: the tree submission MUST retain `PRECACHE_BENCH=1`** (replay 128 bench prompts in untimed warmup, gate `/v1/models` 503 until done), else Leg B reactivates as tree-sensitive↓ and the timed window stops being pure-decode — the whole 481.53→tree calibration transfer is conditional on it. Now a named launch-preflight check (pairs with denken #150's READY/NOT-READY harness). Calibration leg joins sampling (#146) + measured-step (#136) + sync-audit (#147) + validity-preflight (#150) as the projection-CI spine of the eventual `Approval request: HF job`. ubel → #154 (step-denominator reduction audit — decode-path scatter avoidance + CUDA-graph launch overhead; ubel now owns K_cal so the step-% propagation rides a de-risked constant).

## 2026-06-14 12:32 — PR #147: Live re-bench + sync-audit harness for land's descent kernel — 🟢 GREEN / harness self-test passes; sync-free constraint now verifiable — MERGED (bank-the-analysis)

- **Branch:** `lawine/live-rebench-sync-audit` · **Student:** lawine · merged 12:32:59Z (LOCAL A10G — extends #143's profiler, no served-file change/HF Job/submission; BASELINE unchanged 481.53). W&B `25wdfi8x` / `nkcsrovn` (group `tree-submission-preflight`-adjacent sync-audit).
- **Primary:** `harness_self_test_passes=1`. **Test:** `live_audit_ready=1`.
- **Key finding:** extends #143's `scripts/profiler/salvage_walk_overhead.py` with `--trace` / `--self-test` / `--print-schema` (the #143 `run()` path untouched, regression-GREEN). The self-test classifies both regimes correctly: **(a) #143 sync-free model** → **0** non-terminal host-syncs, +0.357% (GPU-hidden, 22.8× isolation→interleaved collapse), bar **4.880** (vs 4.881 target, Δ0.001) → **PASS**; **(b) synthetic sync-bound** → **4.241 syncs/step**, +1.922%, bar **4.956** → **FAIL** with all 4 injected call-sites named. Confirms the terminal `output_token_ids.cpu()` is GPU-hidden + already in the 1.2182 #136 anchor (land does NOT need to fuse it — the right boundary to draw).
- **Conclusion:** makes the sync-free build constraint **verifiable the instant land #71's descent kernel exists** — drop in `--trace <launch-trace>` → emits `live_syncs_per_step` + measured clear-500 bar + PASS/FAIL with offending sites. The measurement leg of the launch evidence-line (pairs with #136 denominator anchor + #143 sync constraint). First thing the build-team runs against the real trace. lawine → #153 (verify-step(M) cost curve — is the depth-9 step flat in M / free tree-growth headroom; parallel-advisor assigned).

## 2026-06-14 12:30 — PR #146: Measured-500-gate confidence envelope (CI + required-N for E[T]) — 🟢 GREEN / gate CI self-test passes; required_n=5 — MERGED (bank-the-analysis)

- **Branch:** `wirbel/measured-gate-confidence-envelope` · **Student:** wirbel · merged 12:30:37Z (LOCAL CPU-only analytic — no GPU/vLLM/HF Job/submission/kernel build; BASELINE unchanged 481.53). W&B `1vj5nwz3` (group `measured-gate-confidence-envelope`).
- **Primary:** `gate_ci_self_test_passes=1` (RED ∧ GREEN ∧ borderline-INDET ∧ fern-points-reproduced). **Test:** `required_n_for_robust_500_verdict=5` (oracle point E[T]=2.621, 99%).
- **Key finding:** statistical layer `scripts/profiler/m16_gate_confidence_envelope.py` **wraps** fern #142's point gate (imports `measured_m16_to_official` verbatim, no duplication) and adds bootstrap sampling-uncertainty CIs. Self-test classifies the three anchors that bracket the 500 question: as-built oracle 2.621 → 269.5 TPS, CI99 **[253.1, 286.1] robust-RED**; ρ-optimal ceiling 5.207 → 535.4, **[506.8, 563.6] robust-GREEN**; clear-500 boundary 4.862 → 500.0, **[471.7, 528.1] INDETERMINATE**; fern #142 point anchors reproduced (270.7/537.8 ≈ 271/538). Quantifies how many measured verify-steps land #71 must log for a **statistically robust** 500 verdict: **required_n=5** at the oracle operating point (99%).
- **Conclusion:** upgrades the point gate to a confidence-aware decision instrument — the projection now carries a CI, and the borderline region around 500 is explicitly flagged INDETERMINATE (a measured E[T] near 4.862 needs ≥5 steps to resolve GREEN/RED at 99%). The sampling leg of the launch evidence-line (pairs with ubel #148 calibration leg via quadrature; the 522 GREEN survives until the sampling ½-width alone exceeds 4.19%). wirbel → #152 (topology re-opt against the measured ladder — clears 530?; parallel-advisor assigned).

## 2026-06-14 12:18 — PR #137: Block64 argmax-reclaim — ⚪ NULL / +0.085% (sub-threshold) + scorer-variance reframing — CLOSED (decision recorded)

- **Branch:** `stark/block64-argmax-reclaim` · **Student:** stark · closed 12:18Z (LOCAL A10G paired wall_tps A/B — research-only artifacts under `research/block64_argmax_reclaim/` + `research/walltps_ab/block64*`, no served-file change, no HF Job/submission; BASELINE unchanged 481.53).
- **Primary:** block64 ΔTPS ≈ **+0.085% local ≈ +0.41 official** (below the 0.10% materiality threshold) → NULL lever. **Decision: Option 2 — adopt block64 as config, fold into land #71's tree-launch manifest (frontier-parity, provably greedy/PPL-safe, zero-cost); NO dedicated launch.** Option 3 (spend a launch to confirm the null) rejected — the one shot is reserved for the tree.
- **Key finding (the valuable one):** the "8-TPS gap to #1" is **best-of-N official-scorer variance (~1.9%), NOT a real deficit** — frantic-penguin's own 3-draw spread (489.63 / 483.80 / 480.41) brackets our 481.53. We are at frontier *parity* on the linear stack; the gap to #1 is noise. This reframes the "catch #1" instinct: 500 is a **tree-path E[T] story** (land #71's descent), not a linear micro-opt story.
- **Conclusion:** block64 sub-threshold (correct not to chase standalone); scorer-variance reframing banked as the takeaway. block64 handed to land's manifest as a free config line. stark → reassigned to a fresh higher-leverage lever (Plateau Protocol — researcher-agent sweep in flight).

## 2026-06-14 12:17 — PR #144: lm_head verify-candidate shortcut (audit + prune verify GEMM cols) — 🔴 RED / verify GEMM is full-12288 (LIVE) but candidate-prune is net-slower + greedy-unsafe — MERGED (bank-the-analysis)

- **Branch:** `denken/lmhead-verify-candidate` · **Student:** denken · merged 12:17:44Z (LOCAL A10G profiling — audit + isolated microbench, no HF Job/submission/PPL change; BASELINE unchanged 481.53). W&B `8vgm3mx0` (group `lmhead-verify-candidate-shortcut`).
- **Primary/Test:** `official_tps_projection=481.53` (unchanged).
- **Key finding:** Step-0 audit settled the open question — the verify lm_head GEMM is genuinely **full-12288 (LIVE)**: the 8192-sparse `FUSED_SPARSE_ARGMAX` patches the **drafter** (`Gemma4MTPMaskedEmbedder.get_top_tokens`), NOT the verifier (`Gemma4ForCausalLM.compute_logits`) — a real distinction from the #121-style MOOT case, nailed by a 7-point code-citation chain. Step-1 NO-GO on two independent grounds: **(A) net-slower** — the optimistic candidate gather-GEMM (80.67 µs) is **2.1× the dense int4 Marlin read (38.27 µs)** at M=8 (gathered bf16 embeddings 10.8 MB > int4 weight 15.7 MB), same physics as the closed GEMM-BW lane #117/#130/#108; **(B) greedy-unsafe** — argmax over a 263-candidate set can't certify the true full-vocab argmax without the full projection → breaks exact-greedy + PPL≤2.42. The lm_head GEMM is also only 0.46–0.74% of the decode step.
- **Conclusion:** verifier-side active-vocab is the wrong place; the drafter is the only safe home (already restricted at 8192). Lane closed. denken's follow-up (decode-path `[M,262144]` scatter avoidance — the accept kernel only needs `target_argmax` during decode; full materialization is prefill/PPL-only) is correctness-safe but a fraction-of-a-fraction (<2% ceiling) — not staffed. denken → #150 (tree-submission local preflight harness — the validity leg of the launch evidence-line).

## 2026-06-14 12:15 — PR #145: Deep-spine width-vs-spread decomp (the 537.8-vs-376 watched risk) — 🟢 GREEN / decomp self-test passes; deep-spine-spread IS the 161.6-TPS swing — MERGED (bank-the-analysis)

- **Branch:** `fern/deep-spine-width-spread` · **Student:** fern · merged 12:14:54Z (LOCAL CPU-only analytic — no GPU/vLLM/HF Job/submission; BASELINE unchanged 481.53). W&B `nd51qpxf` (group `deep-spine-width-spread-decomp`).
- **Primary:** `decomp_self_test_passes=1` (reproduces both banked anchors to 0.01%: FULL 5.207→537.84 vs 537.8; WIDTH-ONLY 3.643→376.26 vs 376.3). **Test:** `width_vs_spread_band_armed=1`.
- **Key finding:** `scripts/profiler/deep_spine_width_spread_decomp.py` splits the realized tree E[T] into three TPS-attributable facets; on the both-bugs-fixed topology the **161.6-TPS watched swing is ~entirely facet (c) deep-spine-spread** (nested ΔTPS +160.9 / Shapley +151.1, ~3–4× the next facet; branch-width +96–103; depth-1 +13–16, both attributions additive). Spread-recovery map: **λ≥0.90 clears 500** (λ=0.80→492.3 fails, λ=0.90→512.9 clears) with width fully restored; width-recovery μ≥0.70 clears 500. The live gate decomposes a sub-GREEN ladder to a BINDING facet ("SPREAD FAILURE: fix the deep-spine descent" vs "WIDTH FAILURE: fix the rank≥2 re-seed").
- **Naming reconciliation (advisor-confirmed):** WIDTH = rank≥2 branch re-seed; SPREAD = deep rank-1 spine (depths 2–9) — the mapping that reproduces both anchors + matches wirbel #135's width(C−A)/spread(B−A) + the facet names. The PR's step-2 parentheticals were the only swapped gloss (fern's `naming_note` disambiguates for land); advisor confirmed the anchors+hand-off reading, do NOT re-pin to the literal step-2 reading.
- **Conclusion:** the binary land #71 must resolve is whether the descent lifts the deeper rank-1 spine toward the ρ-opt rising ladder (≥90% recovery to clear 500). Decision-geometry leg of the launch evidence-line; pairs with #134 (matrix) + #142 (scalar gate). fern → #149 (joint spread×width 2-D clears-500 frontier — upgrades the two 1-D slices to the full decision surface for partial-both-facet recovery).

## 2026-06-14 12:06 — PR #140: Marlin group-size scale-BW — 🔴 RED / no servable coarser group passes PPL — MERGED (bank-the-analysis)

- **Branch:** `ubel/marlin-groupsize-scalebw` · **Student:** ubel · merged 12:06Z (LOCAL A10G ~12 GB, ~2.5 min/scan — no HF Job/submission/quota; BASELINE unchanged 481.53). W&B `r5z3apii` / `2s1zck87` / `fckn7cdk` (group `marlin-groupsize-scalebw`).
- **Primary:** `groupsize_scalebw_official_tps_proj=481.53` (unchanged). **Test:** `best_ppl_passing_groupsize=128`; `groupsize_clears_500=0`.
- **Key finding:** the coarser-group scale-byte lever is closed on two independent gates. **Servability:** pinned vLLM-0.22 Marlin exposes `MARLIN_SUPPORTED_GROUP_SIZES=[-1,32,64,128]` → **g=256 is UNSERVABLE** (max group 128), killing the only +0.3–0.8% upside branch outright. **PPL:** the only coarser servable group, per-channel **g=-1**, costs **+0.122 PPL → cap-comparable 2.503 > 2.42** — a 3× overshoot of the 0.039 headroom above the deployed g=128 anchor (2.3812), robust across two head configs (+0.122 int4 / +0.121 bf16-tied). Cap-comparability handled by pinning g=128 to the offline anchor and carrying the pipeline-invariant coarsening delta (offset +0.2722). g=-1 *would* cut 95.7% of scale bytes → 484.75 IF it passed — sensitivity-only/moot since it fails the cap.
- **Conclusion:** best stays g=128 → official unchanged 481.53. The scale-byte slice this targeted is **already harvested losslessly by the banked palette #110** (+0.3% TPS, bit-exact) — no residual value. Lane retired (conditional re-open only if a future wheel bump dispatches g=256, whose delta ≈ +0.06 *might* fit the headroom). Banked the PPL-scan scripts + an `apply_body` OOM fix (in-place module copy, ≤50 MB temp vs a ~7 GB 2× body duplicate). ubel → #148 (K_cal tree-transfer validation — the calibration leg of the launch evidence-line).

## 2026-06-14 11:59 — PR #143: Salvage-walk Python-overhead probe — 🟢 GREEN / last un-measured step component is GPU-hidden (if sync-free) — MERGED (bank-the-analysis)

- **Branch:** `lawine/salvage-walk-overhead` · **Student:** lawine · merged 11:59Z (LOCAL A10G, peak 0.034 GB, 70 s — no model/HF/quota/submission; BASELINE unchanged 481.53). W&B `uowx93d9` (group `salvage-walk-overhead`).
- **Primary:** `salvage_walk_step_overhead_pct=0.392` (sync-free interleaved — a measurement-floor upper bound; true marginal ≈ 0). **Test:** `salvage_walk_gpu_hidden=1` (19× isolation→interleaved collapse, inflation < 1%, bar under ceiling).
- **Key finding:** the drafter + salvage-walk Python control flow — the one component #136 could not measure — **pipelines behind the per-step GEMM exactly like the eager attn idle.** Isolation 728 µs → interleaved **38.1 µs** (19× collapse) when ~8.5 ms Marlin GEMM is in flight → **+0.39%** step (sync-free) vs **+2.20%** for a naive per-node `.item()` walk (sync-bound). Faithful descent model validated against the oracle (E[T] 2.602 vs 2.621, −0.7%; salvage 35.5% vs 38.2%; full-reach 3.9% vs 3.6%). Operative clear-500 bar: sync-free **4.881** / sync-bound 4.970 / both under the 5.207 ceiling (anchor = #136 step 1.2182 → 4.862). Per-sync cost ~41–51 µs (corroborates 10–50 µs literature).
- **Conclusion:** the step denominator's last unknown is closed — **≈0 tax if land #71 stays sync-free, +2.2% if not.** Deliverable = the **sync-free build rule** for land: resolve `accept_len` as a device scalar (match-mask → cumprod → argmax-first-mismatch), gather accepted tokens by device index, no per-branch `.item()`/`.cpu()`/`bool(tensor)`; the one unavoidable terminal stream-sync is already in the 1.2182 anchor (`fused_kernel_required=False`). This is the vLLM-v1 `RejectionSampler` "zero CPU-GPU sync" pattern (PR #14930). fern #142's ~4.86 gate confirmed safe. lawine → #147 (live re-bench + sync-audit harness — her follow-up #1, the drop-in tool to verify land honored the rule on his real kernel).

## 2026-06-14 11:55 — PR #141: fp8 KV-cache BW lever — 🔴 RED (servability) / fp8 KV undispatchable on a10g-small sm_86 — CLOSED (bank-the-lane)

- **Branch:** `wirbel/fp8-kv-cache-bw` · **Student:** wirbel · closed 11:55Z (LOCAL own-A10G servability probe — a minimal env-gated `KV_CACHE_DTYPE`→`--kv-cache-dtype` passthrough was added to serve.py then **REVERTED**, deployed submission byte-identical; no HF Job; BASELINE unchanged 481.53). W&B `zif6pueq` (group `fp8-kv-cache-bw`, job_type `servability`).
- **Primary:** `fp8_kv_official_tps_proj=481.53` (unchanged — lever cannot dispatch). **Test:** `fp8_kv_servable_a10g=0` (hard servability RED).
- **Key finding:** both fp8 KV arms crash at engine-core KV init, for two independent space-bracketing reasons. **e4m3 (`fp8e4nv`) is hardware-impossible on sm_86** (Inductor: "type fp8e4nv not supported in this architecture"; needs sm_89+ Ada/Hopper) — dispositive, since the official scorer is also a10g-small sm_86. **e5m2 is software-blocked** by the over-broad compressed-tensors guard (vLLM Issue #39137) for our int4 W4A16 checkpoint ("fp8_e5m2 kv-cache is not supported with fp8 checkpoints"). Even if e5m2's guard were bypassed it would hit the FA_SLIDING `NotImplementedError` (vLLM PR #14221) for a negligible 512-ctx payoff against ~1.8% PPL headroom. bf16-KV control unchanged (PPL 2.3772, wall_tps 454.338). Public evidence: no leaderboard entry ships fp8 KV — consistent with non-dispatchability on the target hardware, not merely unexplored.
- **Conclusion:** banks the **KV-read BW stream** as the one un-attackable memory stream after weights are floored at int4 — a clean lane-closure, not a tuning miss. `research/fp8_kv_cache_bw/CLOSURE.md` committed on-branch; analysis mirrored here. **DO NOT re-propose `kv_cache_dtype=fp8` on a10g-small — it is hardware-closed.** Added to closed-lanes (Theme 4). wirbel → #146 (measured-500-gate confidence envelope).

## 2026-06-14 11:43 — PR #142: Measured-M16 → official 500-shot go/no-go gate — 🟢 GREEN / gate ARMED + self-validated (bit-matches #134 matrix) — MERGED (bank-the-analysis)

- **Branch:** `fern/measured-m16-500-gate` · **Student:** fern · merged 11:43Z (LOCAL CPU-only analytic — no GPU/vLLM/HF Job/submission/kernel build; BASELINE unchanged 481.53). W&B `mjynhw39` (group `m16-measured-500-gate`).
- **Primary:** `gate_self_test_passes=1` (reproduces both bracketing anchors within ±2%). **Test:** `gate_ready_for_measured_build=1`.
- **Key finding:** builds `scripts/profiler/m16_measured_500_gate.py` — a one-call `measured_m16_to_official(accept_length, branch_hit, step_time, tau)` that converts land #71's measured M=16 descent-walk readout into a single official-TPS GO/NO-GO. Self-tests the two anchors that bracket the whole 500 question (as-built 270.73 vs ~271 RED, rel-err 0.10%; both-bugs-fixed 537.84 vs ~538 GREEN, 0.03%) AND bit-matches the merged #134 4-cell recovery matrix (270.73/282.99/522.29/537.83). Separates the TPS verdict (GREEN/AMBER/RED vs 500/530) from the GO/NO-GO by wiring validity preconditions in as gate-gates: PPL≤2.42, **tok/step>3.844 HARD-ABORT floor** (the linear-MTP floor — the tree adds nothing below it), branch-hit≈ρ₂=0.4165, greedy-IDs captured. GO bracket E[T]∈[4.841, 5.207] at the roofline step. **Roofline-pending** (uses the 1.2127 #125 W* step, not yet lawine #136's measured 1.2182 — flagged on every output).
- **Conclusion:** the sanctioned single-entry 500 decision instrument, armed for land #71's number; **produces the decision input ONLY — does NOT authorize a launch** (official shot stays human-gated). Refinement now unblocked: #136 merged → live readout should pass `--measured-step 1.2182` (nudges the clear-500 bar 4.841→4.862). fern → #145 (deep-spine width-vs-spread decomposition — the watched 537.8-vs-376 risk).

## 2026-06-14 11:43 — PR #133: Root-cause the 13.1pp depth-1 deficit — 🟢 GREEN / fp32 GPU-confirmed NOT the fix; deficit is FIXABLE build-plumbing — MERGED (bank-the-analysis)

- **Branch:** `denken/depth1-rootcause` · **Student:** denken · merged 11:43Z (GPU-direct measurement, single A10G ~18.3 GB, int4 base, batch=1 — no served-file change, no HF Job, no kernel build, no submission; BASELINE unchanged 481.53). W&B `k2dhcvbn` (group `depth1-rootcause`).
- **Primary:** `depth1_logit_star_relerr=0.009094` (amplifies 9.1× off the 1e-3 attn relerr, but stays under the 1.5e-2 re-open line). **Test:** `drafter_spine_depth1_mismatch=0` (no STRUCTURAL mismatch — depth-1 root is causal-context/tree-mask-invariant).
- **Key finding (three high-value results):** (1) **fp32 is NOT the depth-1 fix** — NET fp32-recovery ~0pp; the argmax flips it induces are exact-tie reshuffles (eps=1e-6 flips the SAME 0.52%), so fp32 cannot fix them. GPU-confirms #128's analytic ≤1.4pp ⇒ the fp32 lane is closed for good, **do NOT spend the tree-488-pw-fp32-v0 quota on an fp32-only build.** (2) The **13.07pp depth-1 deficit (0.598→0.7287) is build-plumbing, ~96% (≈12.5pp)** — a wrong-rank/index spine extraction in land #71's verify path (41.9% rank-2 contamination via `target_logits_indices` reproduces 0.598 exactly using ρ_marginal[2]=0.4165); bf16 precision ≤0.52pp (4%, NET ~0 recoverable); intrinsic/structural 0.0pp (RED ruled out — depth-1 is recoverable to 0.7287 → E[T]=5.207 → 537.84). Verifier = the in-bounds canonical int4 `google/gemma-4-E4B-it-qat-w4a16-ct` (margin median 5.125 independently reproduces kanna #87's 4.875). (3) **Shared-index-map hypothesis for land #71:** the SAME `target_logits_indices` class plausibly corrupts BOTH the depth-1 spine AND the BUG-2 descent traversal (E[T]=2.10 ≪ ρ-opt 4.81) — one corrected index map may address both bugs (needs land's build to pinpoint the exact line).
- **Conclusion:** the BUG-1 secondary lever is now de-risked and handed to land as a concrete build fix (root verify-row must compare against the drafter's rank-1, not rank-2); fp32 lane closed (quota saved). Demotes BUG-1 to the 522→538 margin (consistent with #134/#135). denken → #144 (lm_head verify-candidate shortcut — served micro-lever).

## 2026-06-14 11:24 — PR #136: Measured step-anchor for the depth-9 verify step + selective-root-row re-price — 🟡 AMBER / denominator FIRM at ~roofline (+0.45%); root-row clears 530 at measured step — MERGED (bank-the-analysis)

- **Branch:** `lawine/fp32-step-anchor` · **Student:** lawine · merged 11:24Z (LOCAL A10G roofline + isolation/interleaved micro-bench, 0.258 GB peak / 22 s — no HF Job, no submission, no kernel build; BASELINE unchanged 481.53). W&B `dzyf345a` (group `fp32-step-anchor`).
- **Primary:** `measured_depth9_step_time=1.2182` (+0.45% vs the 1.2127 roofline). **Test:** `rootrow_clears_530_at_measured_step=1`.
- **Key finding (load-bearing methodology catch):** the eager star-attn launch idle (37 `attn_py_calls/step`) is **hidden behind per-layer GEMM GPU work** — the step is GPU-bound and the hot path is sync-free, so the CPU pipelines launches ahead. Isolation bench (90.3 µs/call → 3.34 ms/step, +34.5%) vs interleaved-with-filler-GEMM (1.17 µs/call → 43 µs/step, +0.45%) disagree ~80×; **an isolation-only measurement would have reported a misleading RED.** Operative clear-500 bar moves only **4.841 → 4.862** (graphed verify recovers 4.841 exactly). Selective root-row recipe clears 530 (5.169 bar, 0.038 E[T] + 2.7× idle-budget margin); full fp32 upcast still cannot (consistent with #131). Realized-official cross: oracle E[T]=2.621 → 269.5 (≈271 ✓), chiku-inu E[T]=2.07 → 212.9.
- **Conclusion:** the step-time **denominator** every fleet 500-verdict divides by is firm at ~roofline; the binding lever is confirmed to be the **numerator** (BUG-2 descent / BUG-1 spine), not the step. AMBER only because openevolve's full-step `wall_tps` had not yet landed (board request `20260614-111141-880` posted; re-run flips AMBER→GREEN). The one un-measured remainder — the drafter+salvage-walk Python control flow — is now lawine → #143 (salvage-walk Python-overhead probe).

## 2026-06-14 11:10 — PR #135: BUG-2 salvage-descent root-cause — 🟢 GREEN / descent (BUG-2) is the DOMINANT ceiling, 19.3× BUG-1 — MERGED (bank-the-analysis)

- **Branch:** `wirbel/bug2-salvage-descent` · **Student:** wirbel · merged 11:10Z (LOCAL E[T]-DP decomposition — no GPU run, no HF launch; BASELINE unchanged 481.53). W&B `2n3bhhfz`.
- **Primary:** `bug2_et_recovery=2.4203` (descent-only fix → E[T] +2.42). **Test:** `bug2_is_dominant_ceiling=1`.
- **Key finding (independent of fern #134's method, same verdict):** E[T]-DP decomposition of the measured oracle ladder. Descent-only fix → **E[T] 5.041** (clears the 4.841 bar by itself); spine-only fix → 2.746 (fails). **BUG-2 / BUG-1 = 19.3×.** Step-1 reconstructs the oracle's E[T]=2.621 from the measured per-position ladder with residual 0, and pins the 391/1024 salvages at +0.077 (2.9% of E[T]) — they fire but do not descend.
- **Conclusion:** the descending accept walk is the whole 500-game; the depth-1 spine (0.679 vs 0.7287) is only the secondary margin. Converges exactly with fern #134's official-TPS matrix (522 descent-only). Hand-off: land #71 builds the descent walk; denken #133 owns the (now-demoted) BUG-1 spine.

## 2026-06-14 11:10 — PR #134: Live oracle readout — measured E[T]=2.621 → official-TPS go/no-go + bug-fix recovery matrix — 🟢 GREEN / tree LIVES iff BOTH bugs fixed — MERGED (bank-the-analysis)

- **Branch:** `fern/oracle-live-readout` · **Student:** fern · merged 11:10Z (LOCAL analytic over the openevolve oracle readout — no GPU run, no HF launch; BASELINE unchanged 481.53). W&B `3mp2vtup`.
- **Primary:** `measured_official_tps_as_built=270.73` (the live oracle's as-built tree fails 500 by a wide margin). **Test:** `tree_clears_500_at_both_bugs_fixed=1`.
- **Key finding (official-TPS recovery matrix, depth-9 step 1.2127):** as-built 270.7 ❌ · BUG-1-spine-only fix 283 ❌ · **BUG-2-descent-only fix 522.3 ✅ (E[T] 5.056)** · both-bugs-fixed 537.8 ✅. **Descent-only clears 500 even with the depth-1 spine left broken.**
- **Conclusion:** double-confirms (with wirbel #135's independent E[T]-DP) that the descending accept walk is the decisive 500-lever; the spine is only the 522→538 margin. Feeds the measured-M16→official gate (fern → #142 measured-M16 → official 500-shot go/no-go gate).

## 2026-06-14 10:50 — PR #132: Q-Palette sub-4-bit weights — 🔴 CLOSED (sub-4-bit architecturally impossible on sm_86/vLLM-0.22; Step-1 gate kill; kanna → #138 K-sweep block64)

- **Branch:** `kanna/qpalette-sub4bit` · **Student:** kanna · CLOSED terminal, ~10:50Z (LOCAL CPU code-inspection + literature scan — no GPU, no HF launch; BASELINE unchanged 481.53). W&B `g8dgvmkd` (state=finished, primary `qpalette_projected_official_tps=481.53`, test `qpalette_servable_and_clears_500=0`).
- **Hypothesis:** Sub-4-bit weight quantization (Q-Palette, 3.0–3.5 avg bits, 15–25% fewer weight bytes) would translate bandwidth savings to TPS uplift on the BW-bound path, independent of Issue #124 greedy-identity ruling.
- **Primary:** `qpalette_projected_official_tps=481.53` (unchanged — Step-1 kill before any PTQ). **Test:** `qpalette_servable_and_clears_500=0`.
- **Key finding:** All 10 W-only GEMM kernels in the pinned vLLM-0.22 wheel have minimum 4 bits. Q-Palette targets sm_89 (Ada), Machete is sm_90a (Hopper), FLUTE/QTIP/AQLM/QuIP# have no vLLM-0.22 serving path. `n_subbit_servable_in_wheel=0`. The int4-Marlin 4-bit floor is hardware-hard for sm_86 decode.
- **Conclusion:** Sub-4-bit weight lane definitively closed for sm_86/vLLM-0.22. Staged-gate Step-1 kill was correct. kanna → #138 (K-sweep re-characterization with block64).

## 2026-06-14 10:50 — PR #108: SplitK W4A16 verify-GEMM kernel — 🔴 CLOSED (gate_up M=8=0.0% speedup; triple-confirmed CTA-saturation wall; ubel → #139 cudagraph fix)

- **Branch:** `ubel/splitk-restart` · **Student:** ubel · CLOSED (non-terminal marker, ~10:50Z; LOCAL micro-bench only — no kernel integrated, no HF launch; BASELINE unchanged 481.53). W&B `l9m0o6wc` (state=finished, primary `splitk_verify_gemm_m8_speedup_pct=54.07` [Triton-vs-Triton artifact], test `gate_up_m8_best_speedup_pct=0.0`).
- **Hypothesis:** SplitK decomposition of the M=8 W4A16 Marlin verify-GEMM could recover the ~23% HBM bandwidth gap for a lossless TPS gain.
- **Primary:** `splitk_verify_gemm_m8_speedup_pct=54.07` (Triton-vs-Triton, not decision-relevant). **Test:** `gate_up_m8_best_speedup_pct=0.0` (the binding metric).
- **Key finding:** gate_up (54% of verify time) gets 0.0% speedup from SplitK. Marlin's software pipelining already extracts the pipeline headroom. The CTA-saturation wall (83.6% achievable HBM, Marlin at 79.4% = 95%) leaves zero occupancy headroom. Three independent probes: denken #117 (roofline cap 3.20%/1.56% net), wirbel #130 (re-tiling 0%), ubel #108 (direct SplitK 0%). Note: `splitk_greedy_identical=1` — 0 argmax flips, numerics clean; the lever is dead but not unsafe.
- **Conclusion:** GEMM-bandwidth lane permanently closed. Student asked for direction (a) bank negative or (b) re-home. Advisor: bank as triple-confirmation + close + redirect. ubel → #139 (tree cudagraph crash fix — kernel expertise needed there most).

## 2026-06-14 10:32 — PR #121: QuantSpec drafter-KV premise-check — 🔴 CLOSED/BANKED (drafter Q-only; zero KV bytes; QuantSpec moot for entire MTP frontier; stark → #137 block64 reclaim)

- **Branch:** `stark/quantspec-drafter-kv-2` · **Student:** stark · MERGED terminal, 10:32Z (LOCAL CPU code-inspection — no GPU, no HF launch; BASELINE unchanged 481.53). W&B `zglt88kf` (state=finished, primary `quantspec_drafter_kv_net_wall_tps_pct=0.0`, test `drafter_kv_separate_bool=0`).
- **Hypothesis:** Does the deployed Gemma4 MTP drafter maintain SEPARATE KV cache (lever live) or share the verify path's KV (moot)?
- **Key finding:** `Gemma4MTPAttention` builds only `q_proj/o_proj/q_norm` — no `k_proj`, no `v_proj`. The drafter allocates zero KV-cache bytes and reads K/V from the verify model's shared page pool. Covers `int4_mtp_batchinv`, `fa2sw_precache_kenyan`, `lf29cap444_pupa_check`. `drafter_kv_separate_bool=0`. QuantSpec drafter-KV permanently retired.
- **Conclusion:** Lever moot. Banked with 5 code-citation evidence fields in W&B. stark → #137 (FUSED_SPARSE_ARGMAX_BLOCK 16→64 reclaim — the highest-value single-line fix available).

## 2026-06-14 10:25 — PR #131: fp32 star-attn step-time tax — 🟡 AMBER / full fp32 clears 500 (514 TPS central), misses 530; selective root-row-only fp32 = 563 TPS 🟢 — MERGED

- **Branch:** `lawine/fp32-star-attn-tax` · **Student:** lawine · merged ~10:25Z (LOCAL A10G step-cost bench; BASELINE unchanged 481.53). W&B `tksrxyk5` (state=finished, primary `fp32_tree_official_tps_central=514.4`, test `fp32_tree_clears_500=1`).
- **Primary:** `fp32_tree_official_tps_central=514.4`. **Test:** `fp32_tree_clears_500=1`.
- **Key results table:** full-fp32 M=32 step tax = +9.7% central (compute-exposed, AI=128 > bf16 ridge 117); tree-free with full fp32 → 514 TPS (clears 500, misses 530; 530 break-even E[T]=5.365 > ceiling 5.207 = physically unreachable). **Selective root-row-only fp32 (1/32 rows, depth-1 fix):** 563 TPS central, 554 conservative floor — GREEN for 530.
- **Conclusion:** Binding constraint is E[T] numerator (oracle measured 2.621, far below needed 4.841), not the step-time denominator priced here. Selective fp32 (root-row only) is the right build target: 2× cheaper than full fp32 AND clears 530. lawine → #136 (step-cost anchor to oracle measured wall_tps).

## 2026-06-14 10:25 — PR #130: gate_up tile-shape re-tiling — 🔴 RED / ALL verify-GEMM-bandwidth levers PERMANENTLY CLOSED — MERGED

- **Branch:** `wirbel/gate-up-retile` · **Student:** wirbel · merged ~10:25Z (LOCAL A10G 192-config Triton sweep; BASELINE unchanged 481.53). W&B `ryftxgom` (state=finished, primary `gate_up_retile_per_step_speedup_pct=0.0`, test `gate_up_retile_projected_official_tps=492.77`).
- **Primary:** `gate_up_retile_per_step_speedup_pct=0.0`. **Test:** `gate_up_retile_projected_official_tps=492.77`.
- **Key finding:** A10G HBM saturates at ONE wave (83.6% of datasheet); Marlin at 79.4% = 95% of achievable. 192-config Triton sweep: every smaller-N/higher-CTA/SplitK shape slower than Marlin. Zero occupancy headroom at gate_up (160 CTAs = 2 full waves on 80 SMs). Even the streaming ceiling lifts tree-free-alone only to 488.9 (< 500). Cold-vs-warm artifact (+7.6% phantom) identified and suppressed. **This closes ALL three verify-GEMM-bandwidth probes** (denken #117 roofline + denken #113 LUT + wirbel #130 re-tile = triple convergence on the same 1-wave wall).
- **Conclusion:** The GEMM-bandwidth lane is permanently closed. wirbel → #135 (BUG-2 salvage-descent root-cause, Morgan assignment).

## 2026-06-14 10:25 — PR #129: Oracle-readout harness — 🟡 AMBER / harness armed; operative bar=E[T]=4.841; placeholder TPS=216.9 (awaiting live oracle) — MERGED

- **Branch:** `fern/oracle-readout-harness` · **Student:** fern · merged ~10:25Z (LOCAL CPU analytic; BASELINE unchanged 481.53). W&B `09ge5wmp` (state=finished, primary `oracle_accept_length_to_clear_500=4.841`, test `measured_official_tps=216.91`).
- **Primary:** `oracle_accept_length_to_clear_500=4.841`. **Test:** `measured_official_tps=216.91` (placeholder: bf16-bug tree at E[T]=2.10).
- **Key finding:** Harness maps oracle numbers → measured official TPS + 500 go/no-go, bit-exact self-test (reproduces 481.53). **Key correction:** operative clear-500 bar = **E[T]=4.841** (not 4.624 topology-floor; deeper trees are more expensive per step — bar rises with depth). The test TPS=216.91 is the as-built bf16-bug placeholder; the live oracle number (openevolve E[T]=2.621) gives ~271 TPS, far below 500.
- **Conclusion:** Harness armed and ready. AMBER because live oracle run was not yet available at submission. fern → #134 (live oracle readout with actual E[T]=2.621, Morgan assignment).

## 2026-06-14 10:09 — PR #128: fp32 star-verify cross-check — does QK+PV upcast recover the 13pp depth-1 deficit? 🔴 RED (terminal) — MERGED (pre-run numeric cross-check: fp32 closes only ~0.7–1.4pp of the 13.1pp deficit → fp32 is NOT the depth-1 silver bullet; SAVES the scarce quota run + redirects the build to the real cause; BASELINE unchanged 481.53)

- **Branch:** `denken/fp32-star-verify-crosscheck` · **Student:** denken · merged 10:09Z (LOCAL CPU analytic — no HF Job, no submission, no kernel build; ~57MB peak). W&B `nswm8p6c`.
- **Hypothesis:** chiku-inu's static trace localized the tree's 13pp depth-1 deficit (built 0.598 vs correct 0.7287) to the bf16 star VERIFY FORWARD; their fix is a QK+PV→fp32 upcast. Does the bf16→fp32 upcast QUANTITATIVELY recover the 13pp, or only part — answered BEFORE chiku-inu spends a scarce quota run.
- **Primary:** `bf16_depth1_flip_frac_predicted = 0.00693` (0.69% Gaussian; worst-case model-independent bound 1.38%) vs the **0.131 (13.1pp)** deficit. **Test:** `fp32_recovers_depth1 = 0`.
- **The killer number:** convolving chiku-inu's MEASURED bf16 star relerr (~1e-3) with kanna #87's banked 65,536-position argmax-margin map (median 4.875; **98.6% of positions provably flip-proof**) → a 1e-3 perturbation flips at most **1.38%** of root-row argmaxes, NOT 13%. bf16 explains 5.3% (Gaussian) / 10.6% (worst-case) of the deficit. To BE the 13pp deficit the logit-level relerr would have to be **15–71× larger** (1.5–7%) than measured — a bf16 attention does not carry that. fp32 (relerr 1e-6) correctly zeroes the bf16 contributor — but that contributor is only ~0.7–1.4pp. fp32 residual flip-frac ≈ 0 (kanna #87 direct: 0/65,536 fp32-regime flips).
- **Step-3 forward (reproduces fern #125 exactly: E[T](0.7287)=5.207→537.84, step_time(W*)=1.2127, K_cal=125.268):** even granting fp32 its full flip-frac AND a fully ρ-optimal topology, predicted recovery 0.605–0.612 → official **499–501** (straddles 500, not a confident ≥500). And the build's realized **E[T]=2.10 ≪ 4.81** (ρ-optimal at the build's OWN q1=0.598) → a large INDEPENDENT realization/descent gap (BUG-2) dwarfs the depth-1 deficit.
- **Conclusion / consequence:** CORRECTS the advisor's earlier "fp32 is the last lever" framing. fp32 is worth folding in as a correctness fix (zeroes a real ~1pp contributor, matches the kanna #87 greedy regime) but is NOT the depth-1 fix. Real ~11.7pp cause is elsewhere — denken's prime suspect = **drafter-spine mismatch** (does the tree's depth-1 spine token == the linear-chain token defining 0.7287?) + index-mapping, re-examined under LIVE tree masking; cheap decisive check = measure the LOGIT-level relerr directly. Relayed to chiku-inu/land/openevolve (board `20260614-101703-365`); denken reassigned **#133** (BUG-1 residual-cause hunt), kanna **#134** (BUG-2 descent gap).

## 2026-06-14 10:05 — PR #122: Batch-invariant verify kernel — restore spec==own-AR, at what TPS cost? 🔴 RED (terminal) — MERGED (#114 follow-up: HARDENS #114 — divergence is STRUCTURAL in the int4 Marlin GEMM, no cheap local fix → the Issue #124 human ruling is now load-bearing for the whole spec-decode 500 lane; BASELINE unchanged 481.53)

- **Branch:** `kanna/batch-invariant-verify-probe` · **Student:** kanna · merged 10:05Z (LOCAL 1-GPU interlock — no HF Job, no submission, no quota; ≈21.6 GiB, no OOM). W&B `n5bypf5h`.
- **Hypothesis:** #114 proved the deployed spec stack diverges from its OWN M=1 AR by 56.08% of tokens (M=K+1 batched verify reduces in a different float order → near-tie argmax flips cascade). Can a batch-invariant verify (M-independent reduction order) restore spec==own-AR (0 divergence), and at what TPS cost?
- **Primary:** `batch_invariant_self_divergence_tokens = 38387` (58.57% — UP from 36751, target 0 → **FAIL**). **Test:** `batch_invariant_tps_cost_pct = 51.78` (target <2% → **FAIL**).
- **Mechanism (the load-bearing finding):** `VLLM_BATCH_INVARIANT=1` only patches aten ops + attention — which kanna proved were ALREADY invariant on fa2sw (TRITON_ATTN single-segment, fa_sliding 0-fire, splitkv auto-gated-off, FUSED_SPARSE_ARGMAX per-row invariant by construction). By elimination the sole M-variance source is the **int4 Marlin weight GEMM** (`ops.marlin_gemm` — a custom CUDA op OUTSIDE the aten dispatcher; split-K geometry chosen internally as f(M); NO num_splits knob; NO batch-invariant Marlin anywhere in the pinned wheel). The 51.78% TPS cost is pure loss (forcing M=1 decode off the 4.14× 3D split-KV onto single-segment 2D) for ZERO validity benefit. Both reloads bit-identical (structural, not #38 wobble); same prompts diverge (Jaccard 0.829).
- **Conclusion / consequence:** converts #114's "the stack diverges" into "the divergence is STRUCTURAL and has no cheap local route to 0" — reaching 0 would need a new fixed-split-K int4 Marlin/Machete CUDA kernel (not in the wheel) or dequant-to-bf16 (catastrophic TPS). HARDENS #114; no spec lever (ubel SplitK, land tree) can get an honest greedy-identity pass locally → the **Issue #124 human ruling is the load-bearing decision for the entire spec-decode 500 lane.** Key tree clarification: chiku-inu's fp32 "greedy-EXACT by construction" is exactness w.r.t. the tree's OWN M=32 verify, NOT spec==M=1-AR (the int4 Marlin GEMM is M-variant at M=32 too) → the tree still rides on #124. Banks the corrected `greedy_determinism.py` docstring + the interlock harness `--config batch_invariant` extension. kanna reassigned **#134**.

## 2026-06-14 09:34 — PR #125: Tree E[T] realization ceiling — can the tree PHYSICALLY clear 500? 🟢 GREEN (terminal) — MERGED (SUPPLY-side complement to #123's demand: the tree physically realizes E[T]=5.207 → official ~538 at W*=M=32/depth-9/max-branch-3, clearing 500 with +38 margin and EXCEEDING #123's demand 4.624 by ~+0.59 E[T]; binding side is now BUILD FIDELITY, not physics; BASELINE unchanged 481.53)

- **Branch:** `fern/tree-et-realization-ceiling` · **Student:** fern · merged 09:34Z (LOCAL CPU analytic roofline — no HF Job, no submission, no kernel build; BASELINE unchanged). W&B `cgtb24xz`.
- **Hypothesis:** compute the maximum E[T] the tree can physically REALIZE (supply) net of the real M=32 wide-verify costs (Marlin M=33 tile-cliff, lawine #107 step-ratio, wirbel #98 tree-mask attention, drafter tree-expansion), so land #71 builds the physically-optimal tree and we know whether supply clears the demand #123 sets.
- **Primary metric:** `tree_et_realization_ceiling = 5.207` (E[T] at the realization optimum W*). **Test:** `tree_clears_500_physically = 1` (central +37.8, conservative corner +36.9, both ≥500).
- **Realization optimum:** W* = **M=32 / depth 9 / max-branch 3** → official ≈ **537.8** central (band [536.9 conservative … 566.7 optimistic]). Binding constraint: **Marlin M=33 tile-cliff** (width — M=33 jumps gemm_cost_mult 1.098→1.284 = +14.6%/step; even generously granting a 33-node tree the M=32 ceiling E[T] crashes official to 467) + **acceptance-saturation** (depth — official peaks d9=538, F_tree saturates q∞≈0.847 while drafter cost grows linearly). M=32 is the hard width ceiling; the optimum sits ON the flat-Marlin plateau (no cliff).
- **Novel finding — the measured-attention supply haircut:** pricing the MEASURED 1.83× tree-mask attention tax (lawine #107, correcting denken #85's optimistic 1.06× that the #100 compose still carried) pulls realized official 569 → 538 (−31 TPS). A LEVEL shift, not an optimum-location shift (the tax depends on M not depth, so W* stays M=32/d9) — so the verdict is unchanged: the tree clears 500 at every corner.
- **Conclusion / consequence:** the SUPPLY side answers DEMAND. Supply (5.207) > #123's demand (clear-500 4.624, overtake-tree-free 4.727) → with tree-free capped at 491.8 (#123) and the tree physically able to realize 538, **the binding side is now BUILD FIDELITY (land #71 at E[T]=2.10 → target ∈ [4.624, 5.207]), NOT supply physics.** Handoff to land #71: build the M=32/depth-9/max-branch-3 ρ-optimal tree (parent array in `rho_optimal_topology_results.json → handoff_land71.build_target_M32_parent`), do NOT exceed M=32 total verify nodes. Closes the tree-triangulation supply leg (denken #123 demand + fern #125 supply both done; lawine #126 tree-τ still in flight).

## 2026-06-14 09:34 — PR #118: 2:4 structured sparsity on verify-GEMM — PPL-gated build-or-kill 🔴 KILL (terminal) — MERGED (banks a reusable offline-PPL harness + an independent CONFIRMATION of the kanna #96→#114 self-referential-gate reframe, despite the negative lever verdict; BASELINE unchanged 481.53)

- **Branch:** `wirbel/maskllm-2to4-ppl-gate` · **Student:** wirbel · merged 09:34Z (LOCAL 1-GPU offline PPL + byte model — no HF Job, no submission; BASELINE unchanged). W&B `zpbsuy26` (SparseGPT) / `nuunqupv` (magnitude) / `8y2rtxnv` (flipdiag).
- **Hypothesis:** 2:4 structured sparsity on the verify-GEMM weights (the highest-upside untested byte lever) is greedy-safe-by-construction under the self-referential gate (#96→#114), so the only binding numerics gate is PPL ≤ 2.42 — does a one-shot 2:4 mask pass it?
- **Primary metric:** `ppl_2to4_best = 7.507` (min over recipes, faithful int4 re-quant — **3.1× over the 2.42 gate**). **Test:** `maskllm_projected_official_tps = 484.2` (safe-subset central; trivial +0.6%).
- **Verdict: KILL the 2:4 lever for this checkpoint.** Global core-7 2:4 fails PPL badly under both recipes: magnitude-2:4 = 25.918 (+988%), SparseGPT-2:4 = 7.507 (+215%, the strongest one-shot recipe, still 3.1× over). The +1.6% gate headroom is so tight that even SparseGPT's per-layer error compounds past it by the 3rd of 37 layers → PPL-safe subset is trivial (2 layers / 5.2% of verify bytes → ~+0.6% TPS), not worth a Sparse-Marlin build. The lever's *physics* is real (full-core7 ceiling ~539 official WOULD clear 500) but it is numerically gated out.
- **Two banked assets (why MERGED not closed):** (1) an **offline PPL harness faithful to served #52** (no-mask anchor 2.3812 vs served 2.3772 = +0.17%, directly comparable to the 2.42 cap) — reusable for any future weight-perturbation lever. (2) An **independent confirmation of the self-referential-gate reframe**: a PPL-passing 2-layer config still flips 5.95% of greedy tokens vs dense (low-margin tie swaps; kanna #87 dense median margin 4.875), so an argmax-vs-dense gate would WRONGLY reject a valid checkpoint → PPL (not greedy-identity-vs-dense) is the correct binding gate. Global 2:4 flips 58.0%.
- **Conclusion / consequence:** one-shot 2:4 is conclusively dead at 4B; the only reclaim route is LEARNED masks (MaskLLM, Gumbel-softmax over a training corpus — training-scale, out of the zero/light-GPU lane). 2:4 retired for this checkpoint. The live byte-lever story stays wirbel #110 (9-bit scale-palette, lossless, 43% scale-byte saving) to stack onto ubel #108 SplitK post-500.

## 2026-06-14 09:11 — PR #123: Re-price the tree-free-500 path after #117 — is the tree now mandatory? 🔴 RED (terminal) — MERGED 09:34Z (rebased clean; result deterministic, faithful to #117)

- **Branch:** `denken/tree-free-500-reprice` · **Student:** denken · terminal 09:11:09Z (LOCAL CPU compose model — no HF Job, no submission; BASELINE unchanged 481.53). W&B `0yv2nw9s`.
- **Hypothesis:** re-price the tree-free 500-path with #117's physical SplitK ceiling (central 1.56% net, band 1.6–7.8%) substituted for #105's assumed ≥4.44% / #109's assumed ubel 8.5%, pushed through the #100/#105/#109 compose × #99 multiplier × #116 τ-band, stacking the surviving cheap levers (palette #110, LK #95; dq #104 DEAD) — does tree-free still clear 500, or is land #71's tree now MANDATORY?
- **Primary metric:** `tree_free_500_ceiling_at_splitk_wall = 491.8` (central; band [489.5 conservative corner, 527.3 optimistic band-high]; 495.9 at the 3.20% gross wall). `clears_500_central = False` (gap 8.2). **Test:** `tree_required_et_to_clear_500 = 4.624` (bare tree; 4.555 with cheap levers; ~57% up #101's recoverable band [3.844, 5.207]).
- **Composed cheap-stack lever table (central, #117 wall, τ=1.0):** frontier 481.5 → +SplitK #117 net 1.56% = 485.5 (+3.95) → +palette #110 0.3% = 486.9 (+1.45) → +LK #95 1.0% = 491.8 (+4.87). Even the LK-high upside total = 499.6, still <500. Tree-free needs SplitK ≥ 4.84% (central levers) / 5.84% (corner) to clear 500; #117 delivers 1.56% → MISS. Self-check reproduces #117's cross-check to the decimal (474.6 / 489.4 / 494.3 at τ=0.96/0.99/1.00) — only SplitK→#117-ceiling and τ→#116-band were swapped vs #105/#109, so the miss is cleanly attributable to the SplitK ceiling, not a τ artifact (the #116 0.9983 floor actually HELPED the cheap path).
- **Conclusion / consequence:** **the tree (land #71) flips from bounded-UPSIDE (#106 AMBER, optional) to REQUIRED-for-500.** No cheap-lever combination clears 500 at the #117 central wall; closing 500 now requires the tree's E[T] numerator (≥4.624, well above the as-built 2.097). ⇒ **"Fixing the #101 tok/step=2.10 tree build defect is now the single highest-leverage 500-path action in the fleet"** (denken's conclusion) — the tree is on the critical path, not insurance. NB the composed cheap stack STILL projects **491.8** (> our 481.53 and > competitor 489.63) — a real frontier gain, just not the 500-closer. Field corroboration: public SplitK/argmax-block class (byteshark 484.62, need-for-speed 488.07) realizes only +0.6–1.7%, none clears 500. Validity (kanna #114 RED + Issue #124) is a SEPARATE gate on top — but SplitK (0-flip) / palette (bit-exact) / LK (prediction-only) / tree (greedy-exact) are all greedy-lossless, so the re-price does not move it.

## 2026-06-14 08:57 — PR #119: Definitive drafter-E[T] ceiling closure — decompose q0=0.729 into intrinsic-vs-capacity + price the cost crossover 🟡 AMBER (terminal) — MERGED by parallel advisor (closes the FIXED-COST drafter-quality lane: capacity-perfect E[T]=3.8445 ties the frontier, below clear-500's 4.62; the one escape — relax drafter cost — is tree-dominated ⇒ **past-530 is provably TREE-ONLY**; BASELINE unchanged 481.53)

- **Branch:** `fern/drafter-et-ceiling-decompose` · **Student:** fern · merged 08:57:20Z (LOCAL — no HF Job, no submission; BASELINE unchanged 481.53). W&B `ljfxajh6`.
- **Hypothesis:** decompose the q0=0.729 draft-position-1 reject mass into verifier-intrinsic-irreducible vs drafter-capacity-recoverable, then price the drafter-cost crossover (E[T] uplift vs step-time penalty via #100) — is past-530 PROVABLY tree-only at ANY drafter cost?
- **Primary metric:** `drafter_et_ceiling_capacity_perfect = 3.8445` (→ 481.59 official, ties frontier). **Test:** `et_per_drafter_cost_crossover = 1.0`.
- **Conclusion / consequence:** even a **capacity-perfect** fixed-cost drafter caps E[T] at 3.8445 → 481.59 (ties 481.53, well below the clear-500 break-even 4.62); openevolve A10G-oracle parity ~3.83 across CE / recipe-sweeps / faithful-vLLM-hidden / DeepSeek-MTP-KL distillation pins fixed-capacity recovery at ~0. The only escape (relax the drafter cost budget) is **tree-dominated** — even the optimistic m≈2 corner only TIES the tree (568). ⇒ **the drafter-quality E[T] escape past ~530 is definitively CLOSED; past-530 is TREE-ONLY** (the build-blocked land #71 tree at E[T]=2.10 is the only live E[T] lever). Fleet action: commit the tree as the past-530 path. Reinforces the cycle-43 reversal — with SplitK capped (#117) and tree the sole past-530 lever, the tree moves from insurance toward mandatory.

## 2026-06-14 08:57 — PR #120: Lockstep meter — collapse the 7.14% cross-meter spread so the one scarce official anchor banks a clean 2nd matched pair 🟢 GREEN (terminal) — MERGED by parallel advisor (pins the official-shot meter = `wall_tps`, methodologically identical to the official `output_throughput`; the 7.14% spread was pure definition-mismatch, collapses to 0.10%; BASELINE unchanged 481.53)

- **Branch:** `lawine/lockstep-meter` · **Student:** lawine · merged 08:57:18Z (LOCAL — no HF Job, no submission; BASELINE unchanged 481.53). W&B `t9wjejgv`.
- **Hypothesis:** methodology-align the local meters to the official HF-Jobs TPS definition to collapse the 7.14% cross-meter spread (steady 428.37 / wall_tps 454.09 / windowed-steady 459.83), so the one scarce official shot is captured in lockstep on a bias-free meter.
- **Primary metric:** `residual_spread_after_alignment_pct = 0.10` (≤1%). **Test:** `lockstep_meter_matches_official_methodology = 1`.
- **Conclusion / consequence:** the lockstep meter = `wall_tps` (= num_completion_tokens / decode_duration_s) is methodologically identical to the official `output_throughput = Σ(output_lens)/dur_s`. The 7.14% cross-meter spread was pure definition-mismatch (5.55% PPL-phase leak + 1.59% cold-start on the unweighted-mean estimator) → collapses to the wall_tps floor 0.10%. Self-check 454.338 × τ=1.06019 → 481.68 vs anchor 481.53 (0.032%). Finalizes #116's pre-registered capture spec: capture wall_tps RAW (cold-included, N=3 median, decode-only) in lockstep with any eventual official shot — so the scarce approval-gated run banks a clean, bias-free 2nd matched (official,local) pair.

## 2026-06-14 08:50 — PR #114: Self-referential greedy gate — confirm SplitK/tree pass the OFFICIAL gate by construction + rebuild the pre-quota interlock 🔴 RED-escalate (terminal) — MERGED (the by-construction claim is REFUTED: the deployed 481.53 spec stack diverges from its OWN M=1 AR by 56.08% of tokens; official scorer runs NO token-identity check → contract real but UNENFORCED; frontier-validity ruling escalated as Issue #124, official shot ON HOLD; BASELINE unchanged 481.53)

- **Branch:** `kanna/self-referential-greedy-gate` · **Student:** kanna · merged 08:50:00Z (LOCAL 1 GPU A/B interlock — no HF Job, no submission, greedy untouched; BASELINE unchanged 481.53). W&B `9q5yy9l1`.
- **Hypothesis:** the OFFICIAL greedy gate is self-referential per checkpoint (program.md 27-28; #52-int4 "passed") ⇒ SplitK/tree are greedy-safe by the speculative acceptance rule, kernel-agnostic; rebuild the pre-quota interlock to composed-spec==composed-plain-AR.
- **Primary metric:** `self_referential_divergent_runs = 2`. **Test:** `composed_self_consistency_divergence = 0.5608` (36751/65536 tokens, 112/128 prompts).
- **Two-part finding:** (1) the reframe's FIRST half holds — the greedy *reference* IS self-referential: the submission's own M=1 autoregressive trajectory on its own quant/kernels (mechanism-proven from `sitecustomize.py:945-951` + `SENPAI_REFERENCE_MODE`). (2) BUT the SECOND half ("any deterministic verify kernel is greedy-safe by construction") is **REFUTED**: the deployed speculative stack diverges from its **own M=1 AR by 56.08% of tokens**, deterministically and reproducibly (spec-ON reload-vs-reload = 0 divergence; spec-OFF reload-vs-reload = 0 divergence ⇒ pure structural spec-ON↔spec-OFF delta, not run-to-run wobble, not an env confound). **Mechanism:** the M=K+1 batched-verify GEMM reduces in a different float order than M=1 sequential decode, so at near-tie positions `argmax(verify) ≠ argmax(decode)`, and one early flip cascades the whole sequence (onset median ~120/512 tokens).
- **Enforcement finding:** the official scorer runs **no token-identity check** — the "#52 passed 128/128" anchor is a *completion count* (`result["completed"]` in `speed_benchmark/hf_bucket_single_job.py`), and `grep -rn greedy_identity speed_benchmark/` is **EMPTY**. The program.md 27-28 contract is real but **unenforced by automation**.
- **Conclusion / consequence:** "SplitK/tree greedy-safe by construction" is RETIRED — every spec-decode lever on the 500-roadmap (ubel #108 SplitK, land #71 tree) inherits the same batch-non-invariant verify divergence; a #71×SplitK frontier is no safer than the already-deployed stack. The deployed **481.53 itself** is 56% greedy-divergent from its own AR → if the contract binds strictly, the current submission is technically non-compliant too. An honest greedy GREEN needs either a **batch-invariant verify kernel** (kanna #122, in flight — does divergence→0, at what TPS cost?) OR a **human contract exception** (Issue #124, escalated; options A binds-strictly / B served-greedy≠AR-acceptable / C PPL-bounded-middle). **HOLDING any approval-gated official shot** (denken #109's first official run, used as a self-consistency check, would return RED on the current stack). Reassigned kanna → **#122** (batch-invariant verify probe).

## 2026-06-14 08:50 — PR #117: SplitK realization-ceiling roofline — can SplitK physically reach #109's 14.34% corner, or is 540-margin τ/tree-gated? 🔴 RED (terminal) — MERGED (SplitK physically caps at 3.20% gross / 1.56% net; the dominant `gate_up` verify-GEMM is CTA-saturated and frozen → SplitK alone CANNOT clear 500, the corner is genuinely τ/tree-gated; ubel #108 RETARGETED to `gate_up` re-tiling; BASELINE unchanged 481.53)

- **Branch:** `denken/splitk-realization-ceiling` · **Student:** denken · merged 08:49:58Z (LOCAL CPU analytic roofline — no HF Job, no submission, greedy untouched; BASELINE unchanged 481.53). W&B `z9eaoxj5`.
- **Hypothesis:** can SplitK *physically* realize denken #109's 14.34% conservative corner (from #68's +29.8% HBM-utilisation gap), or is the 540-margin τ/tree-gated regardless of ubel's implementation quality? Tells ubel #108 how far SplitK can be pushed.
- **Primary metric:** `splitk_realization_ceiling_pct = 3.199` (gross; **1.56% net**). **Test:** `splitk_headroom_to_corner = -11.15` (pp short of the 14.34% corner). Band-high 7.81% only at an optimistic 88%-GDDR6 wall.
- **Mechanism (why the ceiling is hard):** the dominant verify GEMM `gate_up` is **54% of verify time** and **CTA-saturated** — 160 CTAs = exactly 2 full waves on 80 SMs → SplitK gives it ~0 extra bandwidth, only reduction overhead. Binding regime = **HBM-practical-roofline** (operational AI ≈ 28 FLOP/byte, 3.8× below the A10G compute ridge of 107), NOT the compute floor. Corner TPS at the 3.20% ceiling = **474.6 / 489.4 / 494.3** at τ = 0.96 / 0.99 / 1.00 — **ALL < 500**. Field cross-check: public SplitK-class kernels report +0.6–1.7% → realized s ≈ 1.1–3.3%, consistent with the 3.2% ceiling.
- **Conclusion / consequence:** **SplitK alone CANNOT clear 500 → the corner is genuinely τ/tree-gated**, falsifying both #105's "tree-free 500 @ SplitK≥4.44%" precondition and #109's "ubel central 8.5%" assumption. The single ceiling-breaker is **`gate_up` tile-shape sensitivity** — a smaller N-tile under-fills CTAs and could re-open headroom; that is the ONLY lever left on SplitK, so ubel #108 is **RETARGETED** from a dead corner-chase to `gate_up` re-tiling. denken #123 re-prices the tree-free-500 path against the 3.2% wall (is land #71's tree now MANDATORY, not insurance?). NB: greedy validity (kanna #114 RED + Issue #124) sits on top of ALL this TPS math — even a tree-free 500 needs the validity ruling.

## 2026-06-14 08:25 — PR #116: τ-endgame roofline — derive the bandwidth-lever local→official transfer to ship tree-free 500 without the scarce official anchor 🟢 GREEN (terminal) — MERGED (roofline DERIVES τ=[0.9983,1.00], replacing #112's *asserted* 0.99 floor → tree-free 500 ships on theory + ubel's SplitK% ALONE; the scarce official anchor becomes OPTIONAL confirmation; BASELINE unchanged 481.53)

- **Branch:** `lawine/tau-endgame` · **Student:** lawine · merged 08:25:13Z (LOCAL CPU roofline analytic — no HF Job, no submission, greedy untouched; BASELINE unchanged 481.53). W&B `l7hk8s80` (finished, no NaN over 28 keys; advisor-verified).
- **Hypothesis:** can a first-principles bandwidth-lever roofline *derive* the local→official transfer τ for a verify-GEMM HBM-traffic reduction (the SplitK class) and tighten it below #112's *asserted* [0.99,1.00] — enough to ship tree-free 500 on theory + ubel's SplitK% alone, without spending the scarce, approval-gated official τ-anchor?
- **Primary metric:** `tau_roofline_central = 1.0` (derived band **[0.9983, 1.00]**). **Test:** `tree_free_ship_gate_without_official_anchor = True`. vs #112 asserted 0.99 → **+0.83pp** tighter; vs generic 0.96 → +3.83pp.
- **Mechanism (why the band is tight, not asserted):** τ = τ_eff·τ_mix and → 1 as the lever size s → 0 ⇒ any deviation is **2nd-order in s** (structural). Under denken #97's "the bus is the wall" the ~32% small-kernel tail is 97.83% GPU-busy → BW-bound like the verify-GEMM ⇒ **τ_mix = 1.0 EXACTLY** (HBM bandwidth cancels in the local/official ratio: time ∝ bytes/BW). ε: admissible tail transfer ∈ [1.00, 1.216] → **|ε| ≤ 0.364%** across ubel's 5–12% SplitK CI (closed form `|ε| = s·φ_vg·|m_tail/m_vg − 1|`); adversarial floor (bw-carries @ s=12%) = 0.9983; over-realize ceiling 1.0036 capped at 1.0. Stress **ROBUST:** verify-GEMM AI ≈ 32 FLOP/byte sits 1.6–6.5× left of the sm_86 ridge (BW-bound on the official box too, far left of the M=33 tile cliff); the 77.1% HBM-util gap is wave-quantization set by SM count → architecture-invariant. One un-pinnable residual = split-K reduction-sync *absolute*-BW sensitivity (≤1.26% rel τ_eff haircut), absorbed by ubel-central 8.5%.
- **Conclusion / consequence:** conservative-corner ship threshold at the derived floor = **5.84%** (vs 5.49% @ τ=1.00) → ubel-central SplitK 8.5% clears 500 with **+2.66pp margin** ⇒ tree-free 500 is shippable on theory + ubel's SplitK% ALONE; the scarce official anchor becomes **optional, maximally-informative confirmation** (banks the long-missing 2nd matched (official,local) pair + pre-prices denken #113 LUT-GEMM, same HBM-traffic class ⇒ same τ_eff). Live 500 risk re-pinned to **SplitK DELIVERY at the low CI edge** (ubel-LOW 5% fails at *any* τ — ubel #108's kernel-delivery question, NOT a transfer-factor one). Retires denken #109's generic τ=0.96 fallback for this lever class. Reassigned lawine → **#120** (lockstep-meter: methodology-align the local meters to the official HF-Jobs TPS definition to collapse the 7.14% cross-meter spread, so the one scarce official shot banks a clean, bias-free 2nd matched pair).

## 2026-06-14 08:07 — PR #115: Hydra sequential MTP heads — headroom to break the E[T]=3.844 ceiling? 🔴 KILL — MERGED (the Hydra premise is ARCHITECTURALLY VOID — the deployed drafter is ALREADY recurrent, not Medusa independent heads → sequential conditioning cannot move E[T]; the binding constraint is drafter CAPACITY at draft-position-1, q0=0.729, not conditioning)

- **Branch:** `fern/hydra-sequential-heads` · **Student:** fern · merged 08:07:49Z (LOCAL analytic + A10G oracle corroboration — no HF Job, no submission, greedy untouched; BASELINE unchanged 481.53)
- **Hypothesis:** does conditioning MTP head k+1 on head k's emitted draft token (Hydra-style sequential vs Medusa independent heads) lift deep ρ₃/ρ₄ and break the linear E[T]=3.844 floor toward #106's 4.45/4.62/4.7 milestones?
- **Primary metric:** `independence_attributable_reject_frac = 0.0` (BY CONSTRUCTION — drafter already recurrent). **Test:** `et_ceiling_sequential_conditioning = 3.844` (conditioning cannot move E[T]). W&B `ucp8iotk`.
- **Why KILL (file:line proof):** `gemma4_mtp.py:463` cat(embed(token), prev_hidden); `llm_base_proposer.py:574` step-k input = step-(k−1) draft token; backbone hidden fed forward. The deployed `Gemma4MTP` (via `Gemma4Proposer`) is ALREADY a recurrent sequential module — Hydra's "add sequential conditioning" is a no-op because it is already present. Under temp=0 (accept ⇔ draft==target-argmax) every accepted prefix token is target-correct → the recurrent head is already conditioned on exactly the token a Hydra head would add.
- **The binding constraint (banked):** 34.5% of ALL chain rejections are at **draft position 1** — fed the real verified token + real target hidden (oracle conditioning) yet accepting only **q0=0.729**. That miss is drafter CAPACITY + genuine model uncertainty (256-d, 4-layer, KV-shared Q-only head) — structurally immune to conditioning. openevolve's A10G oracle independently corroborates: every retrained drafter (CE, recipe sweeps, itaca's DeepSeek-MTP KL-distillation) lands at **parity ~3.83** ("at the architecture's acceptance ceiling"). Three independent lines converge.
- **Strategic consequence (banked fleet-wide):** the drafter-quality E[T] escape past ~530 is now **CLOSED** — past-530 is genuinely **TREE-ONLY** (the tree is the only live E[T] lever, build-blocked at tok/step=2.10 per denken #101, NOT conditioning-blocked). Sharpens allocation: 500 closes tree-free on SplitK+τ (denken #117 + lawine #116); 530→556 requires the tree build. Reassigned fern → #119 (definitive drafter-E[T] ceiling closure: decompose q0=0.729 into verifier-intrinsic-irreducible vs drafter-capacity-recoverable + price the cost crossover — to convert this near-closure into a fleet-committable certainty).

## 2026-06-14 08:07 — PR #110: Lossless scale-palette/LUT byte-lever — bit-exact 9-bit index into distinct FP16 scales 🟢 banked (terminal) — MERGED (43.0% scale-byte saving, palette_bit_identical=1.0 by construction over all 26.8M scales; Phase-2 gate shows scales ARE BW-critical-path ~80% un-overlapped → standalone Marlin fork correctly REJECTED, banked as a lossless composable artifact for the post-SplitK compose pass)

- **Branch:** `wirbel/scale-palette-lut` · **Student:** wirbel · merged 08:07:46Z (LOCAL analytic + bare-tensor build probe — no HF Job, greedy untouched by construction; BASELINE unchanged 481.53)
- **Hypothesis:** the int4 verify-GEMM ships ~26.8M FP16 group-scales; if they cluster into a small palette, replace each 16-bit scale with a short index into a per-tensor codebook → fewer scale-bytes on the BW-bound critical path → free TPS.
- **Primary metric:** `scale_byte_saving_pct = 42.996%` (9-bit per-tensor palette over 1,009 distinct FP16 scales; bit-exact). **Test:** `palette_bit_identical = 1.0` (every one of 26.8M scales reconstructs exactly). W&B `6hpco94j` + `83puhkbe`.
- **Build decision (c):** REJECT the standalone Marlin scale-load fork — Phase-2 BW-critical-path gate confirms scales are LIVE (~80% un-overlapped, consistent with denken #85), so the ≤0.5% upper-bound saving does NOT justify a net-negative-risk standalone kernel fork. Bank Phase 1 (the bit-exact palette) + Phase 2 (the critical-path gate) as a lossless composable artifact; defer composition INTO ubel's SplitK kernel to the post-500 compose pass (protect the critical path).
- **Commentary:** correct triage of a thin lever — a real but ≤0.5% byte saving that earns its place as a banked, zero-risk compose-later artifact rather than a risky standalone fork now. Reinforces the cycle-41 fresh-margin narrowing: palette is thin, LUT is dead (denken #113), so the 500-corner closes on exactly two surviving cheap levers (SplitK ceiling denken #117 + τ lawine #116). [wirbel seat subsequently claimed by parallel advisor #118 (2:4 sparsity).]

## 2026-06-14 07:54 — PR #113: LUT/GANQ W4A16 GEMM feasibility at M=8 — does it beat Marlin int4 + give #109's straddling corner its missing ≥500 margin? 🔴 RED / KILL — MERGED (0% best-case, −24.7% realistic; a COMPUTE lever cannot move a BANDWIDTH-bound GEMM; INT8-TC substrate doesn't even serve M≤16 on A10G; LUT does NOT stack with SplitK — do NOT pivot ubel #108 off SplitK)

- **Branch:** `denken/lut-gemm-feasibility` · **Student:** denken · merged ~07:54Z (LOCAL CPU + bare-tensor INT8-TC probe, <1 GiB — no model load, no HF Job, greedy untouched by construction; BASELINE unchanged 481.53)
- **Hypothesis:** does INT8-TC LUT-GEMM (GANQ-style) beat Marlin int4 at M=8 on sm_86 and give denken #109's straddling conservative corner the missing ≥500 margin (alone or additive to SplitK)? SIZING not build.
- **Primary metric:** `lut_gemm_m8_speedup_vs_marlin_pct = 0.0%` (best-case iso-bytes; −24.7% realistic BCQ B=4, −62.2% per-group codebook). **Test:** `lut_gemm_ppl_projected = 2.3777` (≤2.42 holds, but MOOT — speed ceiling ≤0). W&B `htk6wnof`.
- **Why RED (the load-bearing finding):** verify-GEMM time = bytes / achieved-BW. denken #68 MEASURED M=8 Marlin at **77.1% HBM / 20.2% compute → BW-bound**. LUT only buys compute, and that compute floor is fully hidden under memory stalls → zeroing it moves a BW-bound time by ~0. **The +29.8% verify-GEMM headroom is a bandwidth-UTILISATION ceiling owned by SplitK (utilisation lever), not LUT (compute lever) — wrong tool for the regime.**
- **Two banked sub-findings:** (1) **INT8-TC doesn't serve M=8** — Ampere IMMA is m16n8k32; measured `torch._int_mm` *refuses* M≤16 on the A10G; at M=32 int8 gate_up=203.6µs vs Marlin int4 ~67µs (~3× slower, 2× bytes). (2) **LUT does NOT stack with SplitK** (same BW slice; SplitK+LUT combined = 487.0, LUT +0.00) → **do NOT pivot ubel #108 from SplitK to LUT.**
- **Commentary:** literature pass corroborates — no published sm_86/sm_80 LUT W4A16 beats Marlin at M=4–16; GANQ's 2.57× is RTX-4090 (sm_89) at **M=1**; T-MAC CPU-only. The researcher-agent's "+12–22% LUT" was an M=1 GEMV-latency win that does NOT transfer to M=8 BW-bound verify (caught honestly). **Net:** the missing corner-500 margin must come from higher SplitK realization (ubel #108), palette (wirbel #110, ~0.2–0.5%), τ (lawine #116), or the tree (land #71) — NOT LUT. Lane CLOSED. Reassigned denken → #117 (SplitK realization-ceiling roofline: can SplitK *physically* reach #109's 14.34% corner, or is 540-margin τ/tree-gated?).

## 2026-06-14 07:46 — PR #112: Harden the tree-free-500 projection instrument + bound τ from local data (zero-lag SplitK%→official-vs-500) 🟡 AMBER — MERGED (instrument ARMED bit-exact on 481.53; τ-band [0.99,1.00] is a mechanism inference — the data path is blocked by only ONE matched official/local pair + a 7.14% cross-meter spread; recommendation ONE_OFFICIAL_SPLITK_ANCHOR — converges with denken #109 + fern #111)

- **Branch:** `lawine/tree-free-projection-harden` · **Student:** lawine · merged ~07:46Z (LOCAL CPU-analytic, ~150 MiB RSS — no HF Job, no GPU, greedy untouched; BASELINE unchanged 481.53)
- **Hypothesis:** harden #99's projection into a calibrated zero-lag instrument that maps a measured SplitK% (ubel #108) → projected official-vs-500 at the conservative corner, and bound τ (the realization factor) as tightly as committed local data allows — so denken #109's ship decision reads data, not assumptions.
- **Primary metric:** `tree_free_projection_armed = True` (null-lever self-check = 481.530000, residual 0.00e+00%, bit-exact). **Test:** `tau_band_local = [0.99, 1.00]` + recommendation `ONE_OFFICIAL_SPLITK_ANCHOR`. W&B `hcrvdf31` (group `tree-free-projection-harden`).

| ubel SplitK s | gate |
|---|---|
| s ≥ 14.34% | **GO, no official anchor** (clears conservative corner even at generic τ=0.96) |
| s ∈ [7.57%, 14.34%) | **GO requires the one official SplitK τ-anchor** (clears only to mechanism floor τ=0.99) |
| s < 5.49% | **HOLD** / needs another lever (LK, palette) |

- **Step 1 (instrument armed):** imported denken #105's `tree_free_500_ceiling.py` as the single source of truth → projection harness + ceiling model cannot drift; #99 multiplier CI enters as a relative rescale (central stays bit-exact on 481.53). One command maps SplitK% + additive levers → 3-corner official band.
- **Step 2 (the decisive τ finding):** τ for a *kernel swap* can't be pinned from committed local data — NOT because the transfer is unstable (stable to 0.056% within a matched meter) but because there is exactly ONE matched (official, local) pair (the deployed #52 anchor, which *defines* τ=1.00) and the cross-meter spread is 7.14% (steady 428.37 / wall_tps 454.09 / windowed-steady 459.83), which drowns the cross-precision signal. The band [0.99,1.00] is a mechanism inference (bandwidth-lever transfers ~1:1 on sm_86/GDDR6) + a hard physical ceiling τ≤1.00.
- **Commentary:** independent cross-check of denken #109 — lawine's generic-floor (τ=0.96) conservative corner = **14.34%**, landing *exactly* on denken #109's published corner (two harnesses, same number); lawine central 5.43% vs denken 4.44% differs by precisely the de-credited double-quant (#104 KILLed → palette banked central=0), so the gap is explained not noise. **Fleet convergence:** three independent lines now agree the one approval-gated official run should BE the SplitK τ-anchor (doing double duty with kanna #114's greedy self-consistency) — lawine #112 (`ONE_OFFICIAL_SPLITK_ANCHOR`), denken #109 (`reanchor=YES`), fern #111 (verdict + "3× cheaper" both collapse to the τ-path). Reassigned lawine → #116 (τ endgame: *derive* τ from a first-principles bandwidth-lever roofline to tighten the band below [0.99,1.00] + consolidate the fleet τ verdict into one pre-registered ship protocol).

## 2026-06-14 07:24 — PR #111: Settle crossover at landed C=518.1 + post-500 lever-ROI climb 🟢 GREEN allocation map (+ 🔴 ceiling-flag) — MERGED (τ→1.00 is the #1 buildable lever ROI 20.1; cheap non-tree stack caps at ~530, 540→556 tree-gated; both the crossover verdict AND denken's "3× cheaper" claim collapse to the τ-realization path — resolved by denken #109 τ-reanchor=YES)

- **Branch:** `fern/climb-roi` (settle-crossover) · **Student:** fern · merged ~07:24Z (LOCAL CPU-analytic, ~32 MiB / 0.12s — no HF Job, greedy untouched; BASELINE unchanged 481.53)
- **Hypothesis:** settle the tree-vs-tree-free crossover headline at denken #105's LANDED ceiling C=518.1, then rank the 500→556 climb levers by official-TPS-per-build-effort (test denken #105's "τ 3× cheaper" claim).
- **Primary metric:** `post500_top_lever_roi_tau_localcal = 20.05` → top lever = **τ→1.00**. **Test:** `climb_to_ceiling_tps_at_realistic_stack = 519.49` (τ + SplitK→12%). W&B `v3465t8u`.

| rank | lever | ΔTPS | effort | ROI |
|---|---|--:|---|--:|
| 1 | τ → 1.00 | +20.1 | S (local-cal) | 20.1 |
| 2 | tree-recovery → 4.7 | +37.9 | L (build-blocked) | 9.5 |
| 3 | SplitK 4.44→12% | +17.5 | M | 8.7 |
| 5 | LK re-rank → 1.024 | +6.7 | M | 3.3 |
| 6 | scale-palette byte | +2.9 | S | 2.9 |

- **Step 1 (crossover settled):** at landed C=518.1 → AMBER (tree=upside, recover E[T]≥4.79); band spans GREEN(496.8)→RED(540.8) so the AMBER rests on confidence in the tree-free CENTRAL (which denken #109 pins).
- **Commentary:** denken #105's "τ 3× cheaper" is CONDITIONAL — CONFIRMED under local-cal (τ ROI 20.1 = 2.1–6.9× others), BREAKS under official-anchor (τ ROI 10.0 = 1.15× SplitK→12%). The elegant collapse: both the gate verdict AND the 3× claim hinge on ONE unknown — the τ-realization path — RESOLVED by denken #109 (τ-reanchor=YES → τ is "M" effort, co-leads SplitK→12%). Banked fleet order: τ-anchor official run + SplitK→12% → ~519–530, no tree dependency. **RED-flag:** cheap non-tree levers cap at ~530 (full stack τ+SplitK→12%+LK+byte = 529.9); 540→556 is tree-gated but the tree is build-blocked (E[T]=2.10) → the non-tree escape is a better DRAFTER (E[T] via conditioning) — parallel advisor assigned fern #115 (Hydra heads, break E[T]=3.844).

## 2026-06-14 07:21 — PR #109: Tree-free-500 ship-readiness — min SplitK for a CONFIDENT (conservative-corner) ship + does pinning τ need an official re-anchor? 🟡 AMBER — MERGED (corner SplitK 14.34% vs #105 central 4.44%; at ubel ~8.5% the projection STRADDLES 500: 487→507 across τ band; τ-reanchor=YES — the official shot should BE the τ-anchor)

- **Branch:** `denken/tree-free-ship-readiness` · **Student:** denken · merged 07:21Z (LOCAL CPU-analytic decision doc — no HF Job, no served change, greedy untouched; BASELINE unchanged 481.53)
- **Hypothesis:** turn #105's central GREEN (tree-free clears 500 at SplitK 4.44%) into a SHIP gate — the minimum SplitK at the CONSERVATIVE CORNER (τ-floor 0.96 × multiplier-CI-low × levers-low), and whether pinning τ forces a scarce approval-gated official re-anchor.
- **Primary metric:** `min_splitk_for_confident_ship_pct = 14.34%` (corner, margin 0; +1%→16.67%, +2%→19.05%) vs #105 central 4.44%. **Test:** `tau_official_reanchor_required = YES` (τ_required @ ubel-central 8.5% = 0.986, above floor 0.96). W&B `pyjib2k8`.

| SplitK % | τ=0.96 | τ=0.98 | τ=1.00 |
|---|--:|--:|--:|
| 4.44 (#105 central) | 477.5 HOLD | 487.5 HOLD | 497.4 HOLD |
| 8.50 (ubel central) | 487.0 HOLD | 497.1 HOLD | 507.3 GO |
| 14.00 | 499.3 HOLD | 509.7 GO | 520.1 GO |

- **Commentary:** two honest corrections moved the bar UP vs #105: (1) byte-lever = wirbel PALETTE not INT8 double-quant (wirbel #104 KILL) → corner contribution 0, central SplitK-for-500 4.44%→4.84%; (2) multiplier CI factored with no double-count (official-side risk carried ONCE, in τ). At ubel's plausible SplitK ~8.5% the projection STRADDLES 500 (487→507 across the τ band) → cannot ship on the projection alone. Decision: the one approval-gated official run should BE the τ-anchor of the SplitK-built submission (converts τ from assumed-[0.96,1.0] to measured), NOT a blind ship. NOT RED (reaches 500 at τ≈1.0 for SplitK≥6.5%); simply lacks conservative-corner margin until SplitK→~14% OR palette/LK realize. Converges with kanna #96 (the same official run also validates greedy self-consistency). → denken reassigned to LUT-GEMM feasibility (#113), the fresh kernel margin the corner needs.

## 2026-06-14 07:21 — PR #96: Network-wide greedy-compounding gate — do per-layer ≤1-ULP perturbations compound to flip argmax on the composed frontier? 🔴 RED (cross-kernel) → REFRAMED GREEN-for-official-gate — MERGED (971/65,536 near-tie flips Marlin-AR vs SplitK-AR; but the OFFICIAL gate is SELF-REFERENTIAL per checkpoint → SplitK/tree greedy-safe by the acceptance rule)

- **Branch:** `kanna/greedy-compounding-gate` · **Student:** kanna · merged 07:21Z (LOCAL single-A10G measurement — no HF Job, no served change, greedy untouched; BASELINE unchanged 481.53)
- **Hypothesis:** close #87's named residual — do the network-wide ≤1-ULP reduction-order perturbations of the composed land#71×ubel#84 frontier compound across ~37 layers to flip the greedy argmax vs the deployed stack?
- **Primary metric:** `compounded_argmax_flip_count_realistic = 971/65,536 (1.482%)` (RED vs ~0.1% threshold). **Test:** `compounded_argmax_flip_count_adversarial = 2783 (4.247%)`. W&B `bre5n6ip`.
- **Mechanism:** every flip is a near-tie — 866 (68%) EXACT bf16 ties, 100% within 8 ULP of a tie; upstream-only==full-frontier (971==971) so cleanly attributed to network-wide-h compounding (lm_head adds 0). The DEPLOYED baseline is already this fragile: 964 decode/prefill-wobble positions; **bs=1 vs bs=32 decode alone moves 62% of greedy tokens.** kanna also CAUGHT that #87's cross-tab was positionally invalid (38% trajectory agreement, bs mismatch) + added a trajectory-alignment guard.
- **ADVISOR REFRAME (load-bearing):** the RED is correct for the question asked (Marlin-AR vs SplitK-AR cross-kernel) but that is NOT the official gate. program.md 27-28 = "token-identical to plain greedy AR **for the submitted checkpoint**" → SELF-REFERENTIAL per submission. Proof: #52-int4 PASSED the official 128/128 greedy gate; a canonical-reference gate would reject any quantized submission (int4 noise ≫ near-ties), yet quantization (311) + greedy-preserving speculation (314) are ALLOWED. So submission-spec == submission's-OWN-plain-AR by the acceptance rule (emit==argmax(verify_logits)), kernel-agnostic → **SplitK/tree greedy-safe for the official shot by construction**, independent of the 971 cross-kernel flips. Surviving deliverables: (1) the corrected pre-quota gate is composed SELF-determinism + composed-spec==composed-plain-AR (NOT vs-baseline byte-identity — over-strict, false-REDs on near-ties); (2) decode-path-pinning is mandatory (bs alone moves 62%). → kanna reassigned (#114) to confirm the self-referential gate rigorously + rebuild the interlock to the correct comparison + bound decode-pin invariants.

## 2026-06-14 07:13 — PR #107: Tree-step denominator measurement — pin the REAL M=8→M=32 verify-step ratio 🟢 GREEN — MERGED (measured verify-forward floor 1.237×; whole-step bracket [1.145,1.156] CONFIRMS fern's 1.16×; break-even 4.614 holds vs 4.624; GEMM NOT flat — Marlin 16-row tile staircase 1.169× — but offset by attention-as-modeled 1.83× → nets to fern's 1.158)

- **Branch:** `lawine/tree-step-denominator` · **Student:** lawine · merged 07:14Z (LOCAL A10G microbench — no HF Job, timing only, greedy untouched; BASELINE unchanged 481.53)
- **Hypothesis:** convert the load-bearing 1.16× M=8→M=32 tree-step denominator (under fern #102's break-even + the 569 projection) from a back-solved model assumption into a MEASURED number with a CI, on the GEMM+causal-attn floor (no star-attn tree-mask kernel needed).
- **Primary metric:** `measured_M32_M8_step_ratio = 1.2370` [1.2268, 1.2472], CV 0.944% (median N=5, verify-FORWARD floor = GEMM+attn). **Test:** `corrected_breakeven_ET = 4.614` [4.568, 4.614] vs fern #102's 4.624 (Δ −0.21%, holds). W&B `tbhywbmw`.

| component | M=8 | M=32 | ratio |
|---|--:|--:|--:|
| **verify-forward floor (GEMM+attn)** | 5588 µs | 6910 µs | **1.2370** |
| GEMM (int4 W4A16 Marlin, ×42 layers) | 5013 µs | 5856 µs | 1.1686 |
| attention (deployed fp32 split-KV `unified_attention`) | 575 µs | 1054 µs | 1.8325 |

- **Verdict — 🟢 GREEN, fern's 1.16× whole-step denominator CONFIRMED by direct measurement.** Mapping the verify-forward floor (1.237×, on the ~61% GEMM+attn share) to the WHOLE step via two transparent maps gives bracket **[1.1446 lumped, 1.1560 budget-share]** vs fern's modeled **1.1584** → confirmed (not corrected). The component nuance is the real find: **GEMM is NOT flat within M≤32** (r=1.169, the 16-row Marlin tile staircase: M=8 fills 1 tile, M=32 fills 2 → +17%; still ≪ the +29% M=33 cliff denken #68 flagged) — this *corrects* denken #68's "~1.0× flat" piece — **but attention is exactly wirbel #98's 1.83×**, and in the budget-share map the GEMM-not-flat surplus and the as-modeled attention move opposite to the model and cancel, so the whole-step lands on 1.158. Break-even 4.614 sits comfortably between beat-linear 4.45 and the 5.207 ceiling → tree go/no-go unchanged.
- **The one open denominator RISK (lawine's caveat, carried forward):** the measured floor holds the drafter+host remainder (~39% of the M=8 step) FLAT — fern's M-invariance assumption. Two residuals sit ON TOP and are unmeasurable until the build: (1) the star-attn **tree-mask** kernel delta (causal attn is its lower bound); (2) **drafter tree-expansion** (a 32-node tree may issue more drafter passes than the linear 8-chain). If tree-expansion adds real M-variant cost, the ratio rises above 1.16 and the break-even tightens — this is what land #71's build must resolve and what the harness stays armed for.
- **Bug fixed (banked):** the first timed CUDA-graph kernel in a cold process runs at A10G BASE clock (~2× slow — no-warmup GEMM M=8 = 10013 µs vs warm 5013 µs); added a mandatory sustained warmup (clock ramp + Triton JIT both widths) before timed repeats. Protects every future microbench on this rig.
- **Banks:** `scripts/profiler/tree_step_denominator.py` + W&B `tbhywbmw`. **Next:** lawine → **#112 (harden the tree-free-500 projection + bound τ)** — denken #105 made tree-free the primary 500-path; arm the zero-lag SplitK%→official-vs-500 meter + bound τ from local data for denken #109's ship decision.

## 2026-06-14 06:55 — PR #106: Tree-vs-tree-free crossover + build-milestone ladder — at what realized E[T] does the tree overtake tree-free? 🟡 AMBER — MERGED (crossover@C=500 = 4.624 = #102 break-even verbatim; with denken #105's landed C=518.1 the tree is UPSIDE not critical-path; corner-matched recovery gate E[T]≥~4.7; CPU-only, greedy untouched)

- **Branch:** `fern/tree-vs-treefree-crossover` · **Student:** fern · merged 06:56Z (analysis-only, BASELINE unchanged 481.53)
- **Hypothesis:** generalize #102's break-even from target=500 to target=C (denken #105's tree-free ceiling); find the crossover E[T]ₓ(C) where `tree_official(E[T]) = C`, plus a build-milestone ladder official(E[T]) for ship-gates.
- **Primary metric:** `tree_vs_treefree_crossover_ET = 4.727` (corner-matched central). **Test:** `build_milestone_ladder_clear500_ET = 4.624`. W&B `1qkiheqb` (CPU-only ~27 MiB/~1s; reuses #102 `breakeven_raw_et` verbatim, rescale error 0.0).

| denken #105 ceiling C | crossover E[T]ₓ | verdict | meaning |
|---|--:|---|---|
| C < 500 | < 4.624 | 🟢 GREEN | tree-free can't hit 500 → tree CRITICAL-path |
| **500 ≤ C < 540.7** | 4.624–5.000 | 🟡 **AMBER** | tree-free clears 500 → tree is **UPSIDE** |
| C ≥ 540.7 | ≥ 5.000 | 🔴 RED | tree barely beats tree-free → pivot |
| C ≥ 563.1 | > 5.207 | 🔴 deep-RED | tree never beats tree-free → pivot+escalate |

- **Verdict — 🟡 AMBER, settled by denken #105's landed C=518.1.** 500 ≤ 518.1 < 540.7 → the tree is **bounded UPSIDE, not the critical path**. Corner-matched crossover (same optimism lifts both sides) collapses to a tight **4.834 / 4.727 / 4.737** → whatever corner reality picks, **the build must recover to E[T] ≈ 4.7–4.8 to overtake tree-free** (alone; ~4.52 with splitk+lk also built). Milestone ladder: beat-linear 4.45 → clear-500 4.62 → overtake-tree-free ~4.7; ~10.8 official TPS per +0.1 accept_length. Tree's recoverable official band [416, 563] central — overtakes tree-free only in the upper part of denken #101's [3.844, 5.207] band (the floor 3.844 never clears any plausible C). Caveat: at the conservative ceiling corner (496.8 < 500) the verdict flips GREEN (tree critical) → the AMBER call rests on the tree-free *central*, which denken #109 ship-readiness pins.
- **Jointly with denken #101 + #105:** converts the tree from a single point of failure into bounded upside on a hard recovery gate (E[T] ≥ ~4.7).
- **Banks:** `scripts/profiler/tree_vs_treefree_crossover.py` + `tree_vs_treefree_crossover_results.json` + report. **Next:** fern → **#111 (settle headline at C=518.1 + post-500 lever-ROI climb)** — rank the 500→556 levers by official-TPS-per-build-effort.

## 2026-06-14 06:52 — PR #105: Tree-free 500-path ceiling — does the build-complete stack clear 500 with NO tree, at what SplitK threshold? 🟢 GREEN — MERGED (tree-free C=518.1 central [496.8,540.8]; SplitK-for-500 = 4.44% < ubel's +5% floor; ceiling 556; the tree is now INSURANCE; binding gate moves SplitK→τ)

- **Branch:** `denken/tree-free-500-ceiling` · **Student:** denken · merged 06:53Z (analysis-only, BASELINE unchanged 481.53)
- **Hypothesis:** can SplitK #84 + LK #95 + double-quant #104 clear 500 with NO tree (which is build-blocked per #101), and at what SplitK threshold? — the go/no-go deciding whether the tree is critical-path or insurance.
- **Primary metric:** `tree_free_max_official_tps = 518.1` [496.8, 540.8] (ubel-high SplitK 12%). **Test:** `splitk_threshold_for_500 = 0.0444` (4.44% central, below ubel's +5% floor). W&B `0kiktnqt` (CPU-only; composes merged levers via fern #100 model, K_cal=125.268).
- **Verdict — 🟢 GREEN, 500 is reachable TREE-FREE.** The build-complete linear stack clears 500 at SplitK ≥ **4.44%** (central) — below the +5% floor ubel #84 already targeted; SplitK 8.5%→509.9, 12%→518.1; full ceiling **556.0 [533.2, 581.1]** at 29.7% gap-close. **Strategic consequence: the tree (land #71) is now INSURANCE/UPSIDE (500→556-581), not critical-path** — the #101 build defect no longer blocks 500. **The binding gate moves from SplitK to τ** (realization factor [0.96,1.00]): τ→1.00 is ~3× cheaper than any other margin lever. SplitK × double-quant netting is orthogonal (multiply, no double-count).
- **Critical-path handoff:** land ubel #84 SplitK to ~8.5% (→ 509.9 central) + pin lawine #99's τ. **Next:** denken → **#109 (tree-free-500 ship-readiness)** — min SplitK for a *confident* (conservative-corner) ship + whether pinning τ needs an approval-gated official re-anchor or ships on lawine #99 local calibration.
- **Banks:** `scripts/profiler/tree_free_500_ceiling.py` + `tree_free_500_ceiling_results.json` + report.

## 2026-06-14 06:52 — PR #104: Double-quant verify-GEMM scales 🔴 KILL (banked) — MERGED (bit-exact frac 13.1% « 98% gate; info-theoretic dead: FP16 10 mantissa bits vs ~1.4-1.9-octave scale spread; corrected byte estimate 53.70 MB = 3.06% of int4 body; successor = lossless scale-palette/LUT)

- **Branch:** `wirbel/double-quant-verify-gemm-scales` · **Student:** wirbel · merged 06:52Z (analysis-only, BASELINE unchanged 481.53)
- **Hypothesis:** double-quantize the verify-GEMM int4 scales (quant-the-scales) for a greedy-lossless byte-saving → wall_tps lift.
- **Primary metric:** `dq_scale_roundtrip_bitexact_frac = 0.1309` (gate >0.98 → **FAILED**). **Test:** `dq_tps_lift_est_pct = -0.02%`. W&B `6or2w3ee` (CPU-only byte/precision analysis).
- **Verdict — 🔴 KILL, info-theoretic (not tunable).** FP16 carries 10 mantissa bits; the per-group scale spread is ~1.4–1.9 octaves, so any 8-bit re-code of the scales loses ≥2 bits → only 13.1% of scales round-trip bit-exact « the 98% greedy-safety gate. Not fixable by a better codebook — it is an information bound. **Banked value outlives the negative:** (1) corrected byte accounting — core-7 verify-GEMM scales = **53.70 MB = 3.06%** of the 1754.7 MB int4 body (**g=128** confirmed for the folded osoi5-v0-baked weights; the earlier g=32 was the *unfolded* base); (2) the **successor that survives** — a lossless **scale-palette/LUT**: the scales take only **1,009 distinct FP16 values globally** (per-tensor median 427) → a 10-bit global / 9-bit per-tensor index into a palette of the *exact* values is **bit-exact by construction**, ~37.5% scale-byte saving (~20 MB), ~0.3% TPS.
- **Banks:** `scripts/profiler/dq_scale_roundtrip.py` + `dq_scale_roundtrip.json` + report. **Next:** wirbel → **#110 (lossless scale-palette/LUT)** — build-or-kill: are scale bytes on the BW-critical path or already hidden by Marlin?

## 2026-06-14 06:33 — PR #102: Tree E[T] break-even / margin-of-safety — what MIN accept_length clears 500? 🟡 AMBER — MERGED (break-even E[T]*=4.624 tree-alone; the M=32 step is ~1.16× heavier so the tree needs E[T]≥4.45 just to TIE linear 481.53, not 3.844; byteshark's 2.10 is a regression; no lever stack pulls break-even <4.0)

- **Branch:** `fern/tree-et-breakeven` · **Student:** fern · merged ~06:33Z (analysis-only, BASELINE unchanged — official bar UNCHANGED 481.53)
- **Hypothesis:** invert fern #100's forward model (`official_TPS = K_cal·E[T]/step_time·τ`) — solve `official=500` for the threshold accept_length E[T]*, tree-alone + per compounding lever stack; place byteshark's 2.097 + denken #101's recoverable band on that axis.
- **Primary metric:** `breakeven_ET_tree_alone = 4.624` [4.481 opt, 5.026 cons]. **Test:** `ET_recovery_needed_from_2p10 = 2.527` (tree-alone, central). W&B `l12ikxea` (CPU-only; imports #100's `lever_composition.py` verbatim, max |direct−rescale| = 1.8e-15 machine-zero; reproduces #100's 563.1 at E[T]=5.207 exactly).

| break-even ladder (raw accept_length to clear 500) | cons | central | opt | recovery from 2.097 |
|---|--:|--:|--:|--:|
| **tree alone** | 5.026 | **4.624** | 4.481 | **2.527** |
| tree+splitk #84 | 4.922 | 4.458 | 4.254 | 2.361 |
| tree+lk #95 | 5.001 | 4.587 | 4.376 | 2.490 |
| tree+lk+splitk | 4.897 | 4.422 | 4.155 | 2.325 |
| full stack (+persist #97) | 4.897 | 4.339 | 3.648 | 2.242 |

- **Verdict — 🟡 AMBER, must-recover-most-of-the-way.** The load-bearing reframe: fern separated the M=32 *denominator* widening (a step-time fact) from the accept-length *numerator* (free variable), and showed the binding floor is NOT denken #101's structural 3.844 (accept-length units) but **4.45 in OFFICIAL-TPS units** — because the ~1.16× heavier M=32 step means a 'merely correct' tree at 3.844 is still a TPS *regression* (415 official). The tree needs E[T] ≥ **4.45 to TIE 481.53** (the abort line), ≥ **4.624 to clear 500 alone**, and **no lever stack pulls the central break-even under 4.0** (SplitK is the most useful, −0.17; LK barely moves it −0.04). byteshark's as-built **2.097 is a regression** (~227 official, <½ the linear frontier). Critical recoverable threshold (central): <4.34 → no path to 500 with any stack (escalate); [4.34, 4.62) → needs compounding levers; ≥4.62 → tree alone clears. Conservative corner: break-even 5.026 ≈ the 5.207 ceiling → margin only +0.181 even at full recovery.
- **Reframes #100's GREEN:** #100 ('tree clears 500 with margin') is true *given* E[T]=5.207, but in accept-length units the margin is thin (0.58 central / 0.18 cons below ceiling). byteshark's 2.097 falsifies the 5.207 assumption, so the binding variable is the realized E[T] — GREEN→AMBER not because the model changed but because the build's accept-length collapsed.
- **Composes with denken #101:** denken says 2.10→≥3.844 recovery is structurally guaranteed once the spine+salvage defects are fixed; fern says 3.844→4.62 then needs ~75% of the branch premium. Build job precisely bounded: fix defect (→3.844 floor), land full-depth traversal (→toward 5.207). **Abort line E[T]<4.45.**
- **Banks:** `scripts/profiler/tree_et_breakeven.py` + `research/spec_cost_model/tree_et_breakeven_results.json` + report. **Next:** fern → **#106 (tree-vs-tree-free crossover + build-milestone ladder)** — at what realized E[T] does the tree overtake denken #105's tree-free ceiling; partial-recovery curve for build ship-gates.

## 2026-06-14 06:33 — PR #99: Local→official projection calibration + tree-A/B harness 🟢 GREEN — MERGED (multiplier 1.06019 config-stable to 0.056%; closed-loop self-check reproduces 454.338 AND maps back onto 481.53 within 0.014%; zero-lag build-agnostic projection harness armed; current-spec tree projects 569 [552,587])

- **Branch:** `lawine/projcal-tree-harness` · **Student:** lawine · merged ~06:33Z (calibration-only; live dry-run was a tree=OFF self-null on the frontier — BASELINE unchanged 481.53)
- **Hypothesis:** pin the local-wall_tps→official-TPS multiplier + ready the tree-A/B harness so land #71's build is a zero-lag ≥500 decision.
- **Primary metric:** `local_to_official_multiplier = 1.06019` [1.05999, 1.06038] (±0.018%). **Test:** `linear_chain_wall_tps_reproduced = 454.258` (Δ0.018% < 0.10% MDE). W&B `zcfjgog9`.
- **Verdict — 🟢 GREEN.** Multiplier is a pure hardware/environment transfer factor (`wall_tps` is *definitionally* the official `output_throughput`), config-stable to **0.056%** across 5 committed sessions (454.085–454.338). Closed-loop self-check PASSES end-to-end: the live meter reproduces 454.338 within MDE AND the multiplier maps it back onto the 481.53 anchor (both arms within 0.014%). The projection harness (`local_official_projection.py`, 12 unit tests, build-agnostic) maps any candidate's measured wall_tps → projected-official band in one command. Current-spec tree (E[T]=5.207, +18.2%) → **569.3 official [552.1, 586.6]**, clears 500 by +10.4% at the conservative low edge; gate robust unless the official anchor is ~12% below its private-VERIFIED value.
- **Advisor scoping note (recorded with merge):** the 569 GREEN is conditional on E[T]=5.207 — the projection MATH + harness are sound and armed, but the binding variable (realized tree E[T]) is contested by byteshark's 2.097 + fern #102's 4.624 break-even. GREEN = 'instrument calibrated and ready,' not 'tree will score 569.' Honest caveat from lawine: cross-*precision* invariance (int4 vs bf16) not cleanly testable from repo (older rungs metered with the 16-prompt meter) — but NOT the binding axis for the tree (same int4/split-KV precision as anchor, only widens drafter M=8→M=32).
- **Banks:** `scripts/profiler/local_official_projection.py` + `paired_tps_ab.py` projection layer + `scripts/tests/test_local_official_projection.py` (12 tests). **Next:** lawine → **#107 (tree-step denominator measurement)** — pin the real M=8→M=32 verify-step wall-time ratio (the load-bearing 1.16× under fern #102 + this 569 projection), measurable now without land's star-attn kernel.

## 2026-06-14 06:25 — PR #101: Tree accept-length reconciliation — why is the as-built tok/step=2.10 vs analytical 5.207? 🟢 GREEN — MERGED (2.10 is a FIXABLE BUILD DEFECT, not acceptance collapse; ≥56.1% provably build-defect, 100% fixable, (D)-ceiling 0%; ~568 survives; tree marked BUILD-BLOCKED/re-measure-pending)

- **Branch:** `denken/tree-accept-reconciliation` · **Student:** denken · merged ~06:24Z (analysis-only, BASELINE unchanged — official bar UNCHANGED 481.53)
- **Hypothesis:** byteshark's first real tree build (`tree-v2-merge-eager-v1`) delivered tok/step=2.097 — BELOW even linear-MTP 3.844, far below analytical 5.207. Back out implied per-position ρ̂ from the accept hist, compare to wirbel #79's ρ-ladder, classify the defect (A truncated walk / B draft-quality collapse / C eager artifact / D model-optimistic), and hand fern an honest E[T] band.
- **Primary metric:** `tree_accept_length_gap_explained_pct = 100.0` (fixable). **Test:** `implied_tree_rho_hat_depth1 = -0.481` (negative ⇒ impossible for a real tree ⇒ build corrupts the spine). W&B `c5nbdjic` (CPU-only, no GPU; E[T] anchors reproduce wirbel 5.207 to 4.6e-5).

| reference (tok/step, bonus incl) | E[T] | note |
|---|--:|---|
| model full tree (M=32) | 5.2070 | wirbel, \|Δ\|=4.6e-5 |
| model spine-only (rank-1, depth-9) | 4.1773 | |
| **deployed linear-MTP (measured)** | **3.8441** | hard floor a correct tree MUST clear |
| **as-built (byteshark eager-diag)** | **2.102** | 40.4% of model |

- **Verdict — 🟢 GREEN, fixable build defect.** The decisive observation: 2.10 sits 1.74 tok BELOW a *measured* floor (linear 3.844) that the *same drafter+verifier* already achieve — and the tree spine IS the linear chain, so a correct tree cannot accept less. ⇒ not an acceptance question at all. Gap = 5.207−2.102 = 3.105 tok decomposes as: **(A) sub-linear collapse 3.844→2.102 = 1.742 tok = 56.1% = PROVABLE build defect (dominant)**; spine depth-9 ext 0.333 tok = 10.7% (unrealized premium); branch premium 1.030 tok = 33.2% (unrealized premium). **(B) draft-quality** bounded SMALL (salvage fires 0.358/step = 1.07× model-expected 0.336 → drafter IS producing correct rank-2+ candidates); **(C) eager-mode** ~0% of accept-length (overhead/TPS axis, not numerics); **(D) optimistic model = 0%** (fern #92 realized 5.208 vs independent 5.207, +0.025%).
- **Mechanism (the build defect):** (1) depth-1 spine continuation collapses to **0.598 vs the required q[1]=0.7287** (depth-1 is drafter-identical to linear, no tree-attn → a correct build MUST hit 0.7287) ⇒ verify/dispatch corrupts the spine before any branch logic; (2) salvage walk recovers the immediate rescue node but does NOT descend its sub-path — full-tree reach **1.10% as-built vs ~60.8% model**. wirbel #79's ρ-ladder is sound and **un-exercised** (build corrupts below its own rank-1 floor).
- **Corrected band handed to fern #100/#102:** compose against **[3.844 floor, 5.207 ceiling]**, central 5.14–5.21 IF full-depth traversal + fp32 star-attn land; tree marked **BUILD-BLOCKED / re-measure-pending**. Do NOT plug 2.10 (defect artifact) or 5.207 (unrealized) directly. JSON `step3_gate.corrected_ET_band_for_fern100` is machine-readable.
- **Build hand-off (relayed to land #71 / byteshark / chiku-inu / openevolve, board 20260614-062536):** (1) fastest localizer = assert depth-1 == 0.7287; (2) make the salvage walk descend the recovered sub-tree (full-reach ≫1.1%); (3) re-measure accept_length on the **fp32** star-attn path (NOT relerr eager — wirbel #93 is a *separate* greedy blocker).
- **Banks:** `scripts/profiler/tree_accept_reconciliation.py` + `research/spec_cost_model/tree_accept_reconciliation_results.json` + `report_tree_accept_reconciliation.md`. **Strategic consequence:** the tree is no longer a single point of failure for 500 — denken reassigned to **#105 (tree-free 500-path ceiling)**: can SplitK #84 + LK #95 + double-quant #104 clear 500 with NO tree, and at what SplitK threshold? (go/no-go: tree critical-path vs insurance). **Next (queued, denken on request):** re-run this exact reconciliation once byteshark posts a per-position branch-hit histogram from the fixed build → converts 5.207 into a realized band + bounds the deep-tail (B).

## 2026-06-14 06:12 — PR #98: fp32 star-attn cost gate — does the #93-mandated fp32 accumulation erode the tree's +18.2%? 🟢 GREEN — MERGED (fp32 star-attn is ~free: conservative +0.34% M=32 / +0.01% M=8, realized NEGATIVE; haircut 0.404pp → tree net +19.41%; the #93 fp32 constraint is NOT load-bearing)

- **Branch:** `wirbel/fp32-starattn-cost-gate` · **Student:** wirbel · merged ~06:12Z (analysis-only, BASELINE unchanged — official bar UNCHANGED 481.53)
- **Hypothesis:** #93 (RED) mandated fp32 star-attn accumulation (bf16-relerr-1e-3 flips 0.59% greedy tokens). This gate PRICES that mandate: does fp32 erode the tree's +18.2%, or is a tail-only-fp32 hybrid needed?
- **Primary metric:** `fp32_starattn_tree_gain_haircut_pp = 0.404`. **Test:** `fp32_starattn_cost_pct_M32_conservative = 0.339`. W&B `r8xckc7s` (primary) + `jbooswq1` (bit-identical replicate); advisor-verified (both finished, no NaN, verdict_green=1, Δ<0.001pp).

| layout | realized cost% | conservative cost% | attn % of decode |
|---|--:|--:|--:|
| M=8 frontier | −0.305 | +0.010 | 6.43% |
| M=32 tree-verify | −0.572 | +0.339 | 8.12% |

- **Verdict — GREEN. fp32 star-attn is ~free; the #93 constraint is NOT load-bearing.** Priced the only BW-relevant channel (per-segment softmax partial buffer) on the real deployed 3D split-KV `unified_attention` kernel: fp32 vs bf16 partials, everything else identical (KV stays bf16 → read bytes flat in M). Conservative worst-case **+0.339% M=32 / +0.010% M=8** (≤1% → GREEN); realized NEGATIVE. **Mechanism = A10G 6MB L2-residency:** fp32 partials fit L2 for every layer-type except full-M=32 (spills by 2.4MB = the ONLY place fp32 costs wall-time, +5.6µs/op × 7 full layers). Haircut on denken #85's net = **0.404pp → +19.82%→+19.41%** (576.96→575.02 official-projected, −1.94 TPS). The bf16 reduction that would "save" 0.34% is exactly the #93-unsafe path (flips 0.59% greedy tokens) → **fp32 is both SAFE and FREE.** Tail-only-fp32 hybrid NOT needed (#93's 0.537% margin map stays banked if #71's real kernel ever measures >3%).
- **Secondary finding (orthogonal, routed to denken #101 / land #71):** real-kernel M=32 split-KV attention = **1.83× M=8** (8.12% of step) vs denken #85's SDPA-proxy 1.06× — a small attention-BASELINE haircut (NOT fp32) for denken #101 to fold in with the real paged kernel. denken #85's KV-bytes-flat-in-M claim still holds; the wall-time doesn't at the conc=1 floor (the 32-row Q·K·softmax·V compute becomes visible above the tiny KV-read). Caveat: measured on vLLM `unified_attention` (causal), not #71's custom star-attention tree-mask kernel — worth re-confirming with the real paged kernel.
- **Banks:** `scripts/profiler/star_attn_fp32_cost.py` + `research/star_attn_gate/fp32_cost_results.json`. **Next:** wirbel → **#104** (double-quant verify-GEMM scales — INT8 scale-of-scales, greedy-lossless ~0.4–1.1% byte lever; CPU round-trip build-or-kill first).

## 2026-06-14 06:05 — PR #100: Lever-composition economics — composed official-TPS landscape + minimal lever ordering to clear 500 🟢 GREEN — MERGED (tree-sufficient: tree alone clears 500 at the conservative corner; composition is order-independent; min_levers=1) — verdict conditional on E[T]≈5.207, which byteshark's empirical 2.10 now contests

- **Branch:** `fern/lever-composition-economics` · **Student:** fern · merged ~06:08Z (analysis-only, BASELINE unchanged)
- **Hypothesis:** the in-flight 500-path levers are priced in isolation. Compose tree #71 × SplitK #84 × persistent-kernel #97 × LK #95 into one official-TPS landscape + the minimal lever ordering that clears 500, accounting for compounding vs anti-compounding.
- **Primary metric:** `composed_official_tps = 600.0` (full stack, band [531.6, 713.7]). **Test:** `min_levers_to_clear_500 = 1` (`['tree']`).

| lever stack | conservative | central | optimistic | clears 500? |
|---|--:|--:|--:|---|
| frontier | 462.3 | 481.5 | 481.5 | no |
| + LK #95 | 466.9 | 486.3 | 493.1 | no |
| + persist #97 | 462.3 | 496.4 | 553.5 | no |
| + SplitK #84 | 474.2 | 502.4 | 510.5 | central only |
| **+ tree #71** | **518.0** | **563.1** | **581.0** | **YES (even conservative)** |
| full stack | 531.6 | 600.0 | 713.7 | yes |

- **Verdict — GREEN (tree-sufficient), with a load-bearing caveat.** Model: `official_TPS = K_cal·(E[T]/step_time)·τ`, `K_cal = 481.53/3.844 = 125.268` (frontier reproduces exactly); step decomposed into ABS slices (verify-GEMM 0.53 / drafter 0.07 / attn 0.08 / other 0.32). Numerator levers (tree, LK) multiply E[T]; denominator levers (SplitK, persist) subtract an **absolute** saving from their slice. **Key structural finding: the final step is ORDER-INDEPENDENT** — the "lever ordering" is purely a relative-attribution artefact, which collapses the sequencing question. `min_levers_to_clear_500 = 1 (['tree'])` at BOTH conservative (518.0) and central (563.1) → the tree alone is sufficient, every other lever is insurance/upside. **CAVEAT (the reason this is not an operational green-light):** the entire result assumes the tree realizes E[T]≈5.207 (+18.2%). byteshark's first empirical build delivers **tok/step=2.097** — outside the conservative [558,581] band. So GREEN holds *conditional on the build working as analyzed*, a condition now empirically open. **denken #101** diagnoses the recoverable E[T]; **fern #102** (break-even inverse) computes the required E[T]* — their intersection is the real go/no-go.
- **W&B:** `ncseu3ar`. **Next:** fern → #102 (tree E[T] break-even / margin-of-safety — invert this model for the minimum accept_length that clears 500, alone + per lever stack; place byteshark's 2.10 + denken #101's recoverable band on that axis).

## 2026-06-14 05:48 — PR #97: Persistent-kernel overhead gate — is the ~32% "other" GPU-idle (megakernel-reclaimable) or GPU-busy? 🟡 AMBER — MERGED → persistent-kernel/megakernel LANE CLOSED (only 2.17% reclaimable GPU-idle; #65's GPU-bound finding EXTENDS to the megakernel objective)

- **Branch:** `denken/persistent-kernel-overhead-gate` · **Student:** denken · merged ~05:59Z (analysis-only, BASELINE unchanged)
- **Hypothesis:** the parallel-advisor LEVER 1 prices a persistent-kernel/megakernel at +8–15% by assuming the ~32% "other/overhead" is reclaimable GPU-idle (launch latency, host round-trips, inter-kernel bubbles). Tested against denken's own #65 (decode 99.41% GPU-bound). Is the 32% GPU-idle (reclaimable, GREEN) or GPU-busy small-kernel-tail/bus-spillover (#65 extended, CLOSE)?
- **Primary metric:** `persistent_kernel_reclaimable_pct = 2.17%` (GPU-idle ceiling). **Test:** `decode_gpu_idle_fraction = 0.0217` (2.173% ± 0.024% across 39 cycles).

| idle bucket (a+b+c = reclaimable) | % of decode wall |
|---|---|
| (a) kernel-launch / API overhead | 0.53% |
| (b) host-device sync / Python round-trip | 0.33% |
| (c) inter-kernel GPU-idle bubble | 1.31% |
| **total GPU-idle (reclaimable ceiling)** | **2.17%** |
| (d) GPU-BUSY real kernels (the other ~29.8pp of "32%") | **93% of the bucket — NOT reclaimable** |

- **Verdict — AMBER (2.17% < 3% GREEN) → CLOSE the persistent-kernel/megakernel lane.** A trace-direct timeline of the deployed frontier decode step (CUDA graphs ON, conc=1, committed #43 post-split-KV trace) shows the GPU is **97.83% busy / 2.17% idle** in steady decode (1049 kernels per 8.16 ms cycle). Of the coarse "~32% other", only **2.17pp is GPU-idle**; **29.8pp (93%) is GPU-busy** under-counted attention + drafter + norm/sampling/lm_head/elementwise — all real kernels a megakernel *reorders* but **cannot remove** (the bus is the wall, #94). LEVER 1's "+8–15% reclaimable scheduling idle" premise is **refuted**; **#65's 99.41%-GPU-bound finding EXTENDS** from launch-overhead to the full megakernel objective (a sharper closure: not just CUDA-graph-immune but megakernel-immune). Even the 2.17% is mostly intra-graph bubble CUDA-graphs already minimized. Re-labels the 32% bucket correctly: it is the GPU-busy small-kernel tail + bus-bound spillover, not idle slack.
- **W&B:** `gro3qa0d`. **Next:** denken → #101 (tree accept-length reconciliation — why the as-built tree gives `tok/step=2.10` vs analytical E[T]=5.207; the #1 lever's first empirical number).

## 2026-06-14 05:43 — PR #95: Drafter loss-objective gate — is the MTP draft head acceptance-optimal or only likelihood-optimal? (LK-Loss headroom) 🟡 AMBER — MERGED (LK headroom is +1.0–2.4% E[T] under greedy, NOT the +8% headline; re-rank channel rigorously CLOSED; prediction channel untested; banks the measured acceptance profile)

- **Branch:** `fern/drafter-accept-objective` · **Student:** fern · merged ~05:43Z (analysis-only, BASELINE unchanged)
- **Hypothesis:** the MTP draft head is trained likelihood-optimal; an acceptance-aware (LK-Loss / rank-calibrated) objective claims +8–10% E[T]. Does that headroom survive our GREEDY (T=0) verify, and is it a re-ranking win (free, re-order existing draft logits) or a prediction win (needs a trained head)?
- **Primary metric:** `lk_implied_ET_headroom_pct = +2.4%` (greedy ceiling). **Test:** `measured_drafter_top1_accept = 0.7287`.

| channel | headroom under greedy | status |
|---|---|---|
| LK paper headline (T=1 / sampling) | +8–10% E[T] | NOT our regime |
| re-ranking (re-order existing draft logits) | **0.0%** | rigorously CLOSED — drafter argmax already acceptance-ordered (rank-1 best by +0.6 margin) |
| prediction-improvement (trained head) | **+1.0–2.4%** | only surviving channel (EAGLE-3 +2.4 / Medusa +1.0 / MLP-spec +1.2); #80 likelihood-only never tested it |

- **Verdict — AMBER → SIZE, DON'T TRANSFER.** The +8–10% is the paper's sampling-regime figure; under our greedy verify it collapses 3–8× to **+1.0–2.4% E[T]** (~486–493 official). The **re-ranking channel is rigorously CLOSED**: the drafter's argmax is already acceptance-ordered (rank-1 is the best candidate by +0.6 acceptance margin → 0.0% from re-weighting existing logits). Only the **prediction-improvement channel** (a head trained on the acceptance objective — which #80's likelihood-only EAGLE training never isolated) remains live. Positive selection confirmed: P(accept | position k) rises 0.7287→0.8473 across the chain; measured top-1 accept 0.7287 reconciles E[T]=3.8445 to 3e-4. **Banks the measured per-position acceptance profile.** Student rec: do NOT transfer the +8% headline; do NOT full-launch unsized; size the prediction channel with a cheap LoRA/projection probe first; do NOT close the lane. **LK LoRA/projection probe QUEUED** (a separate, approval-gated run if it needs real quota).
- **W&B:** `8kzjyzxb`. **Next:** fern → #100 (lever-composition economics — compose tree #71 × SplitK #84 × persistent-kernel #97 × LK #95 into the official-TPS landscape + minimal lever ordering to clear 500).

## 2026-06-14 05:31 — PR #93: Star-attention greedy-equivalence gate — does the tree-mask numerical path preserve greedy argmax? 🔴 MERGED — RED (relerr-1e-3 star-attention is NOT greedy-safe; land #71 MUST run fp32 accumulation before quota; banks the reusable attention-side flip-gate)

- **Branch:** `wirbel/star-attn-greedy-gate` · **Student:** wirbel · merged ~05:31Z (analysis-only, BASELINE unchanged; official bar 481.53)
- **Hypothesis:** land #71 / chiku-inu implement the tree mask via a triton star-attention path (per-row prefix + rank-1 self, no dense mask; vLLM force-overrides FlexAttention→TRITON_ATTN on gemma-4-E4B), validated externally only to relerr ~1e-3. But the greedy-identity contract (program.md 27-28) is bit-for-bit. Does a ~1e-3 relative attention-output perturbation flip the top1-vs-top2 decision at close-margin positions? Attention-side twin of kanna #87's now-GREEN GEMM gate — the LAST un-audited half of land #71's pre-quota numerical surface.
- **Primary metric:** `greedy_flip_rate_at_1e3 = 0.005927` (0.59%, RED). **Test:** `min_greedy_margin_p1 = 0.001965` (1st-pct relative final-logit top1−top2 margin).

| metric | value | read |
|---|---|---|
| `greedy_flip_rate_at_1e3` (primary, bf16/deployed argmax) | **0.59%** | RED (>0 = bit-for-bit contract violation) |
| noise floor (clean-vs-clean, bf16 & fp32) | **0.0** | control — every flip is perturbation-attributable, not nondeterminism |
| fp32-propagation share of flips | **82%** | TRUE top-1 changes → needs fp32 ACCUM, not just fp32 readout |
| bf16-tie-readout share | 18% | a higher-precision argmax removes only these |
| eps sweep 1e-4 / 1e-3 / 1e-2 (bf16) | 0.597 / 0.593 / 0.734% | FLAT 1e-4→1e-3 — near-tie tail governs, not eps magnitude |
| near-tie population (rel-margin <1e-2 / <1e-3) | 4.77% / 0.537% | the flip-risk set |
| eps_first_flip | 1e-4 (smallest tested) | even realized relerr 3.3e-5 flips 0.485% |

- **Verdict — RED.** A star-attention-magnitude (relerr 1e-3) perturbation flips 0.59% of greedy tokens; noise floor provably 0/65,536, so it's a real effect. Decisive decomposition: 82% are genuine fp32-propagation flips (the TRUE top-1 token changes) → a higher-precision argmax readout removes only ~18%; **land #71 needs fp32 accumulation on the star-attention reduction (softmax·V / o_proj), not merely a higher-precision readout.** Flip rate is FLAT across eps=1e-4→1e-3 because the margin distribution has a thin near-tie tail (~0.5% of positions <1e-3 rel-margin) that any surviving perturbation tips → land needs ~bit-exactness (≲1e-6) at near-tie positions; fp32 accum typically reaches ~1e-6. Flipped positions: median fp32 margin 0.075, with a rare large-margin tail (max 3.74) from cross-position residual-stream propagation (the reason a purely local margin-gated salvage may under-recall).
- **Routing to land #71 (Morgan, 05:30Z — HARD PRE-QUOTA CONSTRAINT):** star-attention path MUST accumulate in fp32 (or be proven bit-exact) before ANY quota spend; re-verify with this gate's script against the kernel's *measured* relerr to confirm flip_rate→0. Discovering this post-launch would have burned the approved HF Job. Salvage option (wirbel follow-up #3): margin-gated fallback recomputing only the ~4.8% near-tie positions in fp32 — folded into wirbel #98's cost-erosion gate.
- **Banks:** `scripts/profiler/star_attn_greedy_gate.py` (reusable attention-side flip-gate). CPU/eager teacher-forcing of the deployed int4 checkpoint (bf16 argmax reproduced exactly per serve.py:410; cross-kernel caveat quantified: eager-vs-vLLM disagrees at 1.48% of positions — itself near-tie fragility, which reinforces the finding); 15.54 GB single-GPU, no HF launch.
- **W&B:** `ut6a94qa` (group `star-attn-greedy-gate`, 428s, 5 seeds). Advisor-verified independently (wandb-query): all substantive metrics (primary/test/noise-floor/full eps-sweep/margin-distribution) match to 6+ sig figs; `verdict_red=1`; `step1/n_positions=65536`; run `finished` (`peak_gpu_mem_gb` unlogged + `_runtime=0` are cosmetic W&B artifacts).
- **Pre-quota numerics surface:** GEMM ✅ GREEN (kanna #87) · attention 🔴 RED→needs-fp32 (this) · network-wide compounding 🔄 (kanna #96). **wirbel → #98** (fp32 star-attn cost-erosion gate: does the #93-mandated fp32 accum erode the tree's +18.2%?).

## 2026-06-14 05:31 — PR #90: MTP draft-length K sweep — empirical wall_tps confirmation of K=7 optimality ✅ MERGED — GREEN/CONFIRM (clean inverted-U, K=7 optimal; locks 454.338 linear-chain reference for land #71; retires the ±4.4% fragile-estimator caveat)

- **Branch:** `lawine/mtp-k-sweep-wall-tps` · **Student:** lawine · merged ~05:31Z (config A/B, BASELINE unchanged; official bar 481.53)
- **Hypothesis:** the deployed linear MTP K=7 was confirmed "near-optimal static" only on the OLD fragile estimator (±4.4% floor); never directly verified with the new robust `wall_tps` runner (lawine #82, CV 0.007%, MDE ≥0.1% N=3). A direct K sweep closes it empirically — confirm K=7 or find a free serve-config win. First "real lever" test of the #82 paired-A/B runner.
- **Primary metric:** `mtp_k_optimal_wall_tps = 454.338` (best K=7). **Test:** `mtp_k7_confirmed_optimal_bool = 1`.

| K | median wall_tps | Δ% vs K=7 | verdict | E[accept] tok/step |
|---|---|---|---|---|
| 5 | 438.412 | −3.505% | REAL | 3.4902 |
| 6 | 451.047 | −0.724% | REAL | 3.7160 |
| **7 (ref)** | **454.338** | — | REF | **3.8555** |
| 8 | 440.282 | −3.094% | REAL | 3.9720 |
| 9 | 440.784 | −2.983% | REAL | 4.0794 |

- **Verdict — K=7 CONFIRMED OPTIMAL.** Clean inverted-U; every non-K7 arm clears the REAL bar (|Δ|≥0.10% N=3) by 7–35×. E[accept] rises monotonically with K (3.49→4.08) with shrinking increments — wall_tps peaks at K=7 because past it the marginal acceptance gain no longer repays the per-step drafter+verify cost (exactly denken #51's analytic K*≈7, now on the robust meter → the ±4.4% caveat is retired). Asymmetric curve: the K7→K8 step-cost jump (+0.54ms) is anomalously large vs ~0.23–0.25ms/K elsewhere → hypothesis: LOOPGRAPH/ONEGRAPH capture sized for M=8 (K=7); K≥8 (M≥9) falls outside the captured bucket and pays a re-pad penalty → K=7 is the *engineered* sweet spot the whole capture+precache stack is tuned around. No free config win.
- **Locks:** **454.338 wall_tps = the linear-chain reference** for land #71's tree-verify gain measurement (paired, CV 0.001%; E[T]=3.8555 matches deployed 3.844 +0.3%). Don't re-derive it.
- **Banks:** `research/walltps_ab/{run_k_sweep.sh,analyze_k_sweep.py,mtp_k{5,6,8,9}/...}`. Single-variable (only num_speculative_tokens via SPECULATIVE_CONFIG env; no served-file change to fa2sw_precache_kenyan); shared fresh K=7 baseline byte-identical across reuse arms; completed 128/128 every run; PPL/greedy untouched (serve-config knob, verifier argmax unchanged). Local-only, no quota.
- **W&B:** K6 `vz5whvxs` (holds shared K7 baseline) · K5 `7ven5w5b` · K8 `bvms4yto` · K9 `ela8jaqt`. Advisor-verified independently (wandb-query): all per-K wall_tps Δ + E[accept] match (3 sig figs); K7 baseline byte-identical CV 0.001%; all 4 runs `finished`, no NaN. **lawine → #99** (local→official projection calibration + tree-A/B harness readiness for land #71).

## 2026-06-14 05:22 — PR #94: Draft-verify overlap gate — can the drafter be hidden behind verify on a BW-bound A10G? ✅ MERGED — AMBER / OVERLAP LANE CLOSED (single-GPU conc=1; banks the reusable A10G dual-stream contention probe + `bus_contention_factor=0.506`)

- **Branch:** `denken/draft-verify-overlap-gate` · **Student:** denken · merged ~05:22Z (analysis-only, BASELINE unchanged)
- **Hypothesis:** at conc=1 the decode loop is serial drafter(N)→verify(N)→drafter(N+1)…; Saguaro/AMUSD-style secondary-stream overlap runs drafter(N+1) concurrently with verify(N). #75 priced the drafter block at 15.5% → naive fully-hidden ceiling ~+18% TPS. Gate: does it survive the A10G single-HBM-bus contention?
- **Primary metric:** `bandwidth_limited_overlap_ceiling_pct = 4.22%` wall / 4.41% TPS (AMBER). **Test:** `drafter_verify_step_time_ratio r = 0.183` (M=8), 0.178 (M=32).

| metric | value | read |
|---|---|---|
| `drafter_verify_step_time_ratio` (r, M=8) | **0.183** | drafter cheap → timing gate stays OPEN (CLOSE iff r>0.85) |
| naive compute-limited ceiling | +15.46% wall / **+18.29% TPS** | reproduces the PR's +18% projection exactly |
| verify solo HBM | **491 GB/s (82% peak)** | one stream nearly saturates the bus |
| two concurrent verify | **1.97× one verify** | fully serialized (symmetric speedup 1.01×) |
| `bus_contention_factor` | **0.506** | ~full serialization on the shared bus |
| `drafter_overlap_efficiency` | **0.273** | only 27% of the drafter hides |
| **bandwidth-limited ceiling** | **+4.22%** wall / +4.41% TPS | 0.273 × 15.46% — AMBER |
| realized after accept-boundary haircut | **+1.16 / +2.09 / +2.86%** (1/2/3-path) | official 487.1 / 491.6 / 495.3 |

- **Verdict — AMBER → LANE CLOSED.** The timing gate does NOT close it (r=0.183 ≪ 0.85: the drafter is ~5.5× cheaper than verify, so a *compute-limited* world would hide it almost fully). **The A10G's single HBM bus is the wall:** verify alone pulls 82% of HBM peak, two memory-bound streams serialize (1.01× symmetric speedup, contention 0.506), combined bandwidth ≈ single-stream (498 GB/s, not the ~982 GB/s additive overlap needs). So the naive +18% collapses ~4× to **+4.22% bandwidth-limited**, then to **+1.2-2.9% realized** after the serial accept-boundary haircut (zero-accept P=0.271). Saguaro/AMUSD's "free drafter" relies on a **separate device with its own HBM**; the premise does not transfer to one A10G at conc=1. Not worth a dual-stream scheduler + speculative continuation tree + rollback for sub-3%, and it **fights the tree** for the same non-GEMM slack (#85).
- **Banks:** `scripts/profiler/dual_stream_hbm_contention.py` (reusable A10G dual-stream probe), `scripts/profiler/draft_verify_overlap_gate.py`, and the reusable **`bus_contention_factor=0.506`** A10G constant — any future "overlap two memory-bound kernels at conc=1" idea should assume ~full serialization. CPU gate + ~4.7 GB GPU probe; greedy token-identity preserved by construction (overlap reorders the GPU timeline only). Re-run regime: a compute-bound decode (much higher concurrency / larger M) returns the bus headroom.
- **W&B:** `1127zef4`. **Next:** denken → #97 (persistent-kernel overhead-reclamation gate — is the ~32% "other" GPU-idle or GPU-busy?).

## 2026-06-14 05:17 — PR #87: Verify-GEMM argmax-margin greedy-safety gate ✅ MERGED — GREEN (both verify-GEMM levers clear the FP-numerics gate; lm_head-isolated; banks the 65,536-position margin map)

- **Branch:** `kanna/verify-gemm-argmax-margin` · **Student:** kanna · merged ~05:17Z
- **Hypothesis:** the "lossless by construction" claim for ubel #84 SplitK + land #71 M-widen is unverified at the FP-numerics level — a reduction-order/tiling change could flip the greedy argmax (kanna's own #73 atomic-add control proved ~36% flips out-of-regime). Map the deployed verify's top-2 logit margin, emulate the SplitK K-partition + M=16/32, count argmax flips. GREEN = 0 flips → protects both levers from an FP-nonassoc disqualification BEFORE quota.
- **Primary metric:** `verify_gemm_argmax_flip_count_splitk = 0`. **Test:** `verify_gemm_min_top2_margin_ulp = 0` (min-positive 0.5 ULP, median 39 ULP).

| metric | value | read |
|---|---|---|
| `verify_gemm_argmax_flip_count_splitk` (primary) | **0** | GREEN — SplitK S∈{2,4,8} isolated reduction order, ≤1 bf16-ULP, split-INDEPENDENT |
| M-widen M=16 / M=32 (real Marlin) | **0 / 0** | M=16 bit-identical to M=8; M=32 ≤0.25 logit |
| provably flip-proof (margin > 2·max\|Δ\|) | **98.13%** | 64,310/65,536; residual 1,226 → 0 measured flips |
| exact bf16 ties | 907 (1.38%) | deterministic lowest-index tie-break (greedy-safe) |
| SplitK-vs-real-M=8 disagreements | 186 (0.28%) | = emu fidelity gap (FP32-emu vs FP16-Marlin-MAC), split-independent — NOT a lever indictment |
| positions audited | 65,536 | 128 prompts × 512 tok, official config |

- **W&B:** `875cujdk` (~45min GPU decode + ~3min CPU analysis, peak ~19.5GB/23GB A10G). Advisor-verified independently: all 7 numeric metrics + the logged artifact match reported values to exact; run `finished`, `verdict: GREEN`. Honest residual flags (`comfortable_headroom: false`, 1.87% measurement-dependent, 907 tie-bounded) all logged transparently.
- **Analysis/conclusions:** The #73 mechanism (reduction-order swap → argmax flip) does NOT trigger for the claimed-lossless swaps: margins are wide where they matter (median 39 ULP, 98.13% provably flip-proof) and at the genuinely thin positions the in-regime perturbation (≤1 bf16-ULP) is too small to flip — 0/907 ties broken even where the swap reached a full ULP. Mechanism: the n=12,288 vocab head forces vLLM Marlin into `use_fp32_reduce=True` + atomic-add OFF (`should_use_atomic_add_reduce()` False for n≥2048), so the only lossy step is the single final FP32→bf16 cast → reduction-order change capped at ±1 bf16-ULP. **Hand-off:** land #71 M-widen DIRECT GREEN (M=16 literally bit-exact), proceed to quota; ubel #84 SplitK GREEN with residual = 907 ties to confirm under the real kernel (margin map handed over). **Honest scope:** audits the lm_head projection (the GEMM feeding argmax); upstream network-wide compounding bounded by the per-layer ≤1-ULP regime argument + ultimately the official 128/128 gate → kanna #96 closes that residual directly. **ONE of two pre-quota numerics gates CLEARED** (wirbel #93 attention-side remains). Banks the 65,536-position margin map as a standing safety contract for any future verify-GEMM kernel change. kanna → #96 (network-wide compounding gate).

## 2026-06-14 05:07 — PR #92: Tree E[T] independence-gap ✅ MERGED — GREEN / DE-RISKED (realized tree E[T] matches independent model +0.025%; last analytical assumption in the 500-path confirmed under real correlated draws)

- **Branch:** `fern/tree-et-independence-gap` · **Student:** fern · merged ~05:07Z by morganmcg1
- **Hypothesis:** wirbel #83's +18.2% E[T] (~568 official) and fern #91's topology confirm both assume chain-rule independence (per-rank AND per-position) of drafter acceptance. Real top-W emissions are correlated (wirbel #86: confident drafter → higher ρ₂, r=−0.97). If correlation lowers realized tree E[T] below 5.207, land #71's projection is inflated; if it matches, the last untested assumption is de-risked.
- **Primary metric:** `ET_independence_gap_pct = +0.0247%`. **Test:** `realized_tree_ET = 5.20824`.

| metric | value | read |
|---|---|---|
| `ET_independence_gap_pct` (primary) | **+0.0247%** | GREEN if \|·\|≤3% — independence holds |
| `realized_tree_ET` (B, MTP regime mixture) | **5.20824** | == #86 ET_uniform_global to 8.9e-16 |
| `independent_tree_ET` (A, analytic) | 5.20695 | wirbel #83 ≈5.207 / fern #91 5.20695 ✓ reproduced |
| `independent_tree_ET` (A, MC 5×400k) | 5.20559 ± 0.00248 | fern #91 MC 5.2056 ✓ reproduced |
| conservative analytic \|gap\| bound | +2.267% | < 3% (GREEN even at the bound) |
| EAGLE-3 real-vs-shuffle (cross-position xcheck) | −1.78% | within ±3% |
| `land71_official_proj_recalibrated` | 568.1 | ~unchanged |

- **W&B:** `r9pq2qon` (CPU-only, 52s, 33.4 MiB RSS). Advisor-verified independently: all 5 numeric metrics + both tables (`et_estimators` 6 rows, `realized_per_regime` 5 bins) match reported values to full FP precision; run `finished` clean (the `_runtime=1s` is a known W&B CPU-only artifact, not a crash).
- **Analysis/conclusions:** Independence is **confirmed**, but the deeper reason is that draws are strongly correlated and the correlation is E[T]-neutral across all four channels: (1) spine cross-depth survivorship — ZERO gap by construction (depth-dependent q[d] ≡ ∏ chain-rule survivorship); (2) rescue depth-dependence — #79 ρ₂-by-depth flat (slope +0.0032/depth), pooling justified; (3) within-step confidence↔rescue (wirbel #86's headline r=−0.97) — REALIZED via a freq-weighted entropy-regime mixture over 13,491 steps → 5.20824 = pooled to machine precision (the pooled ladder IS the freq-weighted mean of regime ladders; E[T] near-linear → only a +0.025% Jensen residual, correlation marginally HELPS); (4) branch-continuation/cross-position — unmeasurable on the deployed linear MTP chain (cancels in the gap), cross-checked on the independent #80 EAGLE-3 trace via real-order-vs-shuffle bootstrap = −1.78% (mechanism: rank-1 runs 17× over-dispersed vs geometric → a few long runs spill past the spine-depth cap). No fresh GPU capture: a full-joint tree accept/reject capture is **structurally impossible** on a linear MTP chain (a rank-2 branch's own continuation is never drafted), so the committed ≥13k-step captures the PR pointed to are the correct data. **For land #71:** ~568–569 (denken #85 net ~576) STANDS; carry ±2–3% band (≈558–581). The only true test of channel 4 is land's first tree `accept_length` run (prior: the −1.78% EAGLE-3 number). **Analytical tree-economics lane now fully saturated** (#88 RED, #86 RED, #91 confirm, #92 confirm). fern → #95 (LK-Loss headroom gate).

## 2026-06-14 04:44 — PR #89: Prompt-lookup × MTP first-reject overlap ✅ MERGED — DROP / LANE CLOSED (realized +1.67% gross below +2% build bar; structural positive-correlation ceiling; prompt-lookup-augment lane CLOSED)

- **Branch:** `denken/promptlookup-augment-overlap` · **Student:** denken · merged ~04:44Z
- **Hypothesis:** prompt-lookup (n-gram draft) hits MTP first-reject misses (m=0 steps, 27% of decode) → complementary, not redundant; augmenting MTP with free PLD tokens at m=0 steps is the highest-value cheap augment behind land #71's fork.
- **Primary metric:** `promptlookup_realized_augment_tps_pct = +1.67%` [CI +1.36, +2.02]. **Test:** `promptlookup_mtp_firstreject_overlap_frac = 0.0354`.

| metric | value | read |
|---|---|---|
| `promptlookup_realized_augment_tps_pct` | **+1.67%** [+1.36, +2.02] | BELOW +2% build bar (gross, no fork cost subtracted) |
| oracle upper bound | +2.38% | hard ceiling (best-match pick) |
| `corr_q_vs_m` | **+0.354** | POSITIVE → redundant, not complementary |
| `promptlookup_mtp_firstreject_overlap_frac` | **0.0354** | 3.54% of m=0 misses get PLD accept-extending hit |
| `share_extra_from_m0` | 0.389 | below #81's independence assumption (54.5%) |
| independence UB (this trace) | +7.87% | realized only 21% of it |

- **Verdict: DROP.** Structural limiter: PLD hits POSITIVELY correlate with MTP acceptance (corr=+0.354) — redundant, not complementary. Oracle best-case caps at +2.38%. No realistic path to ≥+2% build-worthy net. **Prompt-lookup-augment lane CLOSED. Do NOT queue behind land #71.**
- **W&B:** `tz2oaemz`. denken → cycle-39 reassignment (persistent-kernel scheduling).

## 2026-06-14 04:27 — PR #86: Entropy–branching correlation ✅ MERGED — STRONG SIGNAL, NON-ACTIONABLE / LANE CLOSED (r=−0.9688, sign-reversed; oracle ceiling +0.27% E[T] / +0.33pp TPS; entropy-branching lane CLOSED)

- **Branch:** `wirbel/entropy-branching-correlation` · **Student:** wirbel · merged by morganmcg1 ~04:27Z
- **Hypothesis:** drafter uncertainty (token-level entropy) predicts rank-2 branching value — high-entropy steps benefit from deeper branching → entropy-gated dynamic tree delivers free E[T] over a static topology.
- **Primary metric:** `rho2_entropy_correlation_r = −0.9688`. **Test:** `entropy_gated_tree_E_T_gain_pct = 0.273`.

| metric | value | read |
|---|---|---|
| `rho2_entropy_correlation_r` | **−0.9688** | ONE OF STRONGEST SIGNALS IN PROGRAMME — but SIGN-REVERSED |
| sign direction | drafter CONFIDENCE (low entropy) → HIGHER ρ₂ | anti-direction from hypothesis |
| `entropy_gated_tree_E_T_gain_pct` | **+0.273%** | oracle ceiling on dynamic entropy-gated tree |
| TPS equivalent | ~+0.33pp | BELOW cost of forfeiting onegraph CUDA graph |
| pooled ρ₂ | 0.4172 | matches #79's 0.4165 ✓ (acceptance model self-consistent) |
| data collected | 13,491 first-reject steps | 128 prompts × 512 tok, greedy, seed 1, conc 1 |

- **Verdict: NON-ACTIONABLE.** drafter CONFIDENCE (low entropy) predicts HIGHER ρ₂ — so high-entropy steps (where branching would theoretically help most) are precisely where MTP acceptance is LOWEST. Signal is purely within-step; a static depth-indexed tree captures none of it (consistent with #83's flat per-depth ρ₂). Oracle ceiling +0.27% E[T] / +0.33pp TPS < cost of forfeiting onegraph CUDA graph. **Lane closes on actionability, not on null correlation.** The strongest r in the programme closes the weakest lever.
- **W&B:** `59u7qcwa` / `79u01jm8`. wirbel → cycle-39 reassignment (double-quant verify-GEMM scales).

## 2026-06-14 04:26 — PR #91: Tree topology E[T] — max-branch-3 vs max-branch-4 ✅ MERGED — CONFIRMED (+0.9614%, all three estimators agree, validates acceptance model; land #71: build max-branch-3)

- **Branch:** `fern/tree-topology-et-comparison` · **Student:** fern · merged by morganmcg1 ~04:26Z
- **Hypothesis:** wirbel #83's DP-optimal max-branch-3 topology buys +0.96% E[T] / +1.13pp TPS over max-branch-4 (both depth-9, M=32). This analytic prediction is directly measurable by MC simulation using fern's #88 harness.
- **Primary metric:** `topology_et_delta_pct = +0.9614%`. **Test:** `topology_et_confirmed = 1`.

| estimator | E[T] mb3 | E[T] mb4 | delta_pct | SE |
|---|---|---|---|---|
| **analytic** (exact, `score_tree_depthrank`) | 5.206954 | 5.157273 | **+0.9633%** | 0 |
| **independent MC** (2M trials/topo, 5 seeds) | 5.20559 | 5.15602 | **+0.9614%** | ±0.073pp |
| **CRN paired** (6M trials, 3 seeds) | 5.20790 | 5.15859 | **+0.9560%** | ±0.003pp |
| wirbel #83 predicted | — | — | +0.9633% | — |

- CRN 95% CI = **[+0.950%, +0.962%]** — entirely above the +0.8% CONFIRMED threshold; gap to wirbel's analytic 0.002pp. #88 Leg A reproduced bit-for-bit (engine integrity). 0 greedy violations on both topologies. `score_tree_depthrank` ≈ MC to ~1e-3 tok → future tree-topology DP results trusted analytically without per-candidate MC. **Build recommendation for land #71: max-branch-3 array** `[-1,0,0,0,1,1,1,2,3,4,4,5,7,9,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,28,29]` (depth-9, spine widths [3,3,2,2,1,1,1,1,1]). Mechanism: only ~23.8% of decode steps show any topology difference; mb4 wastes a node on rank-4 (marginal ≈0.022, ρ₄=0.1908) while mb3 reallocates to rank-2 breadth (ρ₂=0.4165 ≫ ρ₄).
- **W&B:** `exkahicq` (CPU-only, 33.3 MiB RSS, 155s). fern → cycle-39 reassignment (LK-loss draft head).

## 2026-06-14 04:07 — PR #88: Traversal Verification E[T] gate ✅ MERGED — RED / AXIS CLOSED (provably zero under greedy; standard root-to-leaf confirmed; land #71 keeps existing acceptance rule)

- **Branch:** `fern/traversal-verify-et` · **Student:** fern · merged ~04:07Z
- **Status:** MERGED as decisive RED characterization keeper (CPU analytical + MC simulation; no GPU training, no served-file change; official bar UNCHANGED 481.53). Banks `scripts/profiler/traversal_verify_et.py` + `research/spec_cost_model/traversal_verify_et_results.json`.
- **Hypothesis:** Traversal Verification (NeurIPS 2025 OpenReview 8nOMhDFpkU) — leaf-to-root tree acceptance — recovers sibling-subtree mass from wirbel #83's salvage oracle (rho2=0.4165), potentially delivering a free, provably-lossless E[T] uplift on land #71's M=32 tree.
- **Primary metric:** `traversal_et_uplift_pct = 0.000`. **Test:** `traversal_greedy_violation_count = 0`.

| Leg | Regime | E[T] root→leaf | E[T] leaf→root | uplift % | greedy viol. |
|---|---|---|---|---|---|
| **A** physical M=32, 400k MC | **greedy (T=0)** | **5.2140** | **5.2140** | **+0.000** | **0** |
| **B** sampling-proxy contrast | sampling proxy | 4.4324 | 4.6348 | +4.567 | 26,984 |
| **C** real #80 ranks, 1868 steps | greedy | 3.3330 | 3.3330 | +0.000 | 0 |
| **D** exhaustive all trees n≤6 | greedy-valid | — | — | 0.000 | 0 / 872 trees |

- **Verdict: RED.** Traversal Verification is **provably zero under greedy decode** — structural, not a measurement artefact. Under temperature 0, the target argmax at each position is a single token, so at most one child can match at any tree node. The consistent paths form a unique chain; both walks return the same chain → E[T] uplift 0, greedy violations 0, for any tree/corpus. wirbel's rho2=0.4165 salvage mass is **fully realized by root-to-leaf** (it is the value of the tree topology over the linear chain, not incremental headroom for a different acceptance rule). Leg B confirms the mechanism exists in sampling regimes (+4.57%) but is vacuous at T=0. **Acceptance-rule axis CLOSED. land #71: keep standard root-to-leaf verification.**
- **W&B:** `yiwl2jfj`. fern → #91 (tree-topology E[T] comparison: max-branch-3 vs max-branch-4, using this harness).

## 2026-06-14 03:53 — PR #82: Operationalized wall_tps: paired-A/B runner + re-baseline + #56 re-screen ✅ MERGED (infra keeper: canonical A/B entrypoint + locked re-baseline 454.09 + confirmed deployed MBT=512 optimal)

- **Branch:** `lawine/walltps-ab-runner` · **Student:** lawine · merged ~03:53Z
- **Status:** MERGED as infrastructure keeper (no served-file change; official bar UNCHANGED 481.53). Banks `scripts/profiler/paired_tps_ab.py` as the canonical one-command paired-`wall_tps` A/B entrypoint.
- **Hypothesis:** the #72 `wall_tps` protocol (CV 0.035%, MDE ≥0.1% N=3) is proven but not yet operationalized into a reusable tool. A one-command paired-A/B runner lets every lever-builder (land #71, ubel #84, stark #78, kanna #87) decide on the same robust metric without re-implementing the harness.
- **Primary metric:** `deployed_local_wall_tps = 454.085` (locks re-baseline). **Test:** `paired_ab_self_null_gain_pct = 0.030` (NULL ✓ = unbiased).

| arm | median wall_tps | Δ vs deployed 512 | verdict |
|---|---|---|---|
| **A=B self-null (baseline check)** | 454.09 | +0.030% | NULL ✓ (unbiased) |
| MBT 512→2048 (real-change validation) | 450.83 | −0.716% | REAL ✓ (~9× MDE, sensitive) |
| MBT 512→4096 | 453.54 | −0.120% | REAL (small regression) |
| MBT 512→8192 | 453.06 | −0.226% | REAL (small regression) |

- **Conclusions:** (1) Runner validated — unbiased (self-null NULL +0.030%) + sensitive (real change REAL at ~9× MDE). (2) Re-baseline locked at **454.09 wall_tps** (CV 0.007%, confirms #72's N=12 454.12; retires fragile 428.37). (3) **#56 re-screen: no hidden win** — deployed MBT=512 is at/near optimum; all increases are small REAL regressions (E[accept] drifts 3.853→3.879 but scheduling overhead dominates at conc=1). (4) `paired_tps_ab.py` supports `--candidate-env` serve-time overrides, `--reuse-baseline-from` for multi-arm re-screens, structured `paired_ab.json` + W&B logging. First real lever job (topology A/B, re-opt max-branch-3 vs #74, ~+1.13pp) queued behind land #71. lawine → #90 (MTP K sweep).
- **W&B (group walltps-ab-runner):** selfnull `2mq96qz1` · detok_off `dorrmq8l` · mbt2048 `xmwqvtmk` · mbt4096 `5ny0egab` · mbt8192 `pvg56gnm`.

## 2026-06-14 03:29 — PR #85: Tree-verify non-GEMM overhead audit at M=32 ✅ MERGED (decisive GO: non-GEMM tree machinery 2.597% decode, ~8× smaller than the +21.8% GEMM gain → net +19.82% survives; no O(M²); attention amortizes 1.06×; hands land #71 a per-op cost-budget oracle)

- **Branch:** `denken/tree-overhead-audit` · **Student:** denken · merged by morganmcg1 ~03:28Z
- **Status:** MERGED as research artifact (local profiling, no GPU-train, no served-file change → BASELINE official bar UNCHANGED 481.53; net +19.82% is a PROJECTION off the frontier, not a measured submission). Banks `scripts/profiler/tree_nongemm_overhead.py` + `research/spec_cost_model/report_tree_nongemm_overhead.md` + `tree_nongemm_overhead.json`.
- **Hypothesis:** the +21.8% M=32 tree-verify re-price (wirbel #79/#83) prices the GEMM SAVINGS but not the tree's NON-GEMM systems OVERHEAD (mask construct, scatter/gather, M-row sampler-prep, valid_counts scheduling). Auditing it in isolation either confirms the net gain holds or reveals erosion before land #71 spends an approval-gated launch.
- **Primary metric:** `tree_overhead_nongemm_pct_decode = 2.597` (% decode at M=32 static, W&B `f0c8mb39`). **Test:** `net_tree_gain_after_overhead_pct = 19.82`.

| quantity (M=32 DP-tree vs M=8 linear, 11.6ms step) | value |
|---|--:|
| non-GEMM overhead (static, mask precomputed) | 2.597% (301µs) vs +21.8% GEMM → ~8× smaller |
| Δ vs M=8 linear (the slice eroding the gross) | +1.65pp (192µs) |
| verify-side ONLY (excl. drafter M-row sampler) | 0.512% (59µs) |
| net tree gain after overhead (static) | +19.82% (gross 21.8 − 1.98pp) |
| attention M=32/M=8 (split-KV 3D FlashDecoding) | 1.06× (≪4×, KV read shared) |
| only [M,M] op (ancestor mask) scaling exp | 0.16 (≈flat); 0/step precomputed static |

- **Conclusions:** (1) Tree non-GEMM machinery is ~8× smaller than the GEMM gain it unlocks → net **+19.82%** (≈576 official projected, 3-base). (2) **NO O(M²)** anywhere — ancestor mask exp 0.16 ≈flat, refutes the byteshark O(M²)-mask risk; the two M-growing ops (drafter M-row sampler, full-vocab verify-argmax) are ≈O(M) linear. (3) **Attention amortizes 1.06×** — #43 split-KV routes all M≤64 verify rows to 3D FlashDecoding, shared-prefix KV read once → attention stays at floor (closes the #69-at-M=32 question). (4) Hands land #71 a **per-op cost-budget oracle** (expected µs + 1.5× ceiling) — PERFORMANCE half of its debug gate, pairs with wirbel #83's ≈0.41 salvage (correctness half): op over budget = byteshark layout bug, caught pre-launch. (5) **Side-finding:** denken's roofline (weight-bandwidth-bound, free-to-M≤32) + this audit (KV shared, mask ~0) RULE OUT the tile-scheduler / KV-layout / fused-mask kernel levers at M≤32 → **SplitK (ubel #84) is the only live verify-GEMM kernel lever** (a useful cohort negative). (6) Methodology: eager 327µs → graph-basis 37µs (8.8× launch artifact) — the #77 lesson. (7) denken → #89 (prompt-lookup × first-reject overlap build-or-kill).

## 2026-06-14 03:25 — PR #80: Multi-step (HASS) drafter training — break the K=1 chain-collapse ceiling ✅ MERGED (thesis CONFIRMED +57.8% native accept/step, but E[T]=2.23 ≪ MTP 3.844 → bank-and-close; EAGLE-3 single-layer drafter-training lane CLOSED)

- **Branch:** `fern/eagle3-multistep-hass` · **Student:** fern · merged by advisor ~03:13Z
- **Status:** MERGED as research artifact (offline training/eval, no served-file change → BASELINE official bar UNCHANGED 481.53; confirmed BELOW frontier). Banks the serve-faithful HASS unroll machinery (`train_eagle3.py --unroll_steps`, detached depth unroll, native-acceptance sim).
- **Hypothesis:** the K=1 teacher-forced regime (not the corpus) was the binding native-acceptance ceiling on the EAGLE-3 drafter; HASS multi-step (J=3) unroll — feeding the draft its own rolled-forward hidden — lets per-step acceptance sustain past step-1 instead of collapsing.
- **Primary metric:** `native_accept_per_step_bench_holdout = 1.2294` (HASS J=3, K=8, W&B `at46onde`). **Test:** `..._34ckpt = 0.7792` (#34 K=1 baseline, W&B `bsu901oj`; training `pkcmx1zl`).

| metric (240-rec holdout, K=8, shared harness) | #34 (K=1) | HASS J=3 | Δ |
|---|--:|--:|--:|
| native accept/step (primary) | 0.7792 | 1.2294 | +57.8% |
| E[T] = tok/target-forward | 1.7792 | 2.2294 | +0.4502 |
| step-2 conditional accept | 0.8% | 32.4% | 38.6× |
| tf top-1 (secondary) | 0.7617 | 0.7475 | −1.9% |

- **Conclusions:** (1) Thesis CONFIRMED decisively — HASS unroll lifts step-2 conditional accept 0.8%→32.4% (the own-hidden hand-off the chain died at) and native accept/step +57.8%; the draft genuinely learned to condition on its own rolled-forward hidden. (2) **NOT a frontier candidate** — E[T]=2.23 = 58% of MTP's 3.844, ~1.6 tok short; the ceiling is architectural (single-layer head capacity), not a training schedule. (3) openevolve independently found every retrained MTP head lands at parity too → the parity finding generalizes across head families. (4) Measurement handled right — gated on a serve-faithful sim with a large margin to the 3.844 bar, not a fragile HF proxy (the openevolve over-report caveat). (5) **EAGLE-3 single-layer drafter-training lane CLOSED** per fern's own recommendation; HASS machinery banked as a drop-in if head capacity is ever raised. (6) fern → #88 (Traversal Verification E[T] gate).

## 2026-06-14 03:24 — PR #73: Greedy-identity — is the frontier stack bit-exact or distributional? ✅ MERGED (refutes the premise: deployed stack is BIT-EXACT run-to-run AND satisfies the contract at ~489 local TPS; determinism is ENGINEERED — atomic-add-off is load-bearing; + analyze_determinism.py bugfix)

- **Branch:** `kanna/greedy-determinism` · **Student:** kanna · merged by advisor ~03:13Z
- **Status:** MERGED as research artifact (local measurement, served stack UNCHANGED → BASELINE official bar UNCHANGED 481.53). Banks `scripts/validity/greedy_determinism.py` + `analyze_determinism.py` (with bugfix) + captures.
- **Hypothesis:** the deployed stack is run-to-run token-nondeterministic, so greedy-identity can only be a distributional property (the #66 contract-foundation question).
- **Primary metric:** `greedy_identity_verdict = 0` (bit-exact, W&B `lr1ornnl` N=10 / `45y7ui1o` N=7). **Test:** `fa_sliding0_tps_cost_pct = 0.03` (FA_SLIDING=0 is a no-op, +0.03% within noise).

| config | N | mean byte-id | official greedy_gate | verdict |
|---|--:|--:|---|---|
| **default (deployed)** | 10 | **1.0** | GREEDY_IDENTICAL (0/65536 div) | bit-exact |
| splitkv_off | 3 | 1.0 | GREEDY_IDENTICAL | stable |
| **atomic_on (positive control)** | 7 | **0.8214** | DIVERGENT (35.7% tok) | breaks bit-exactness |

- **Conclusions:** (1) Premise REFUTED — the deployed spec-ON M=1 greedy frontier is BIT-EXACT run-to-run (N=10 fresh reloads, official GREEDY_IDENTICAL, 0/65,536 divergent tokens, flip hazard 0.0) AND satisfies the contract at ~489 local TPS — a third option the PR's dichotomy excluded. (2) **Determinism is ENGINEERED, not luck** — the atomic-add positive control (forcing VLLM_MARLIN_USE_ATOMIC_ADD=1 flips ~36% tokens) proves keeping it OFF is load-bearing; FA2 sliding is inert (flips 0 layers), #43 split-KV is frozen inside the captured graph. (3) The prior "non-reproducible" numbers (BASELINE line 49, lawine #56) were PROXY configs (spec-OFF, plain int4), not the deployed stack. (4) Churn cannot move the gates — PPL invariant to 12 digits (teacher-forced), private TPS gate stable (~0.2% spread). (5) **Bugfix:** analyze_determinism.py had a false hardcoded "atomic-add hardware-gated OFF" prior contradicted by its own data → replaced with data-driven branching + cluster_signatures() + atomic_add_breaks_determinism field. (6) The atomic-add control PROVES the kernel-swap→argmax-flip mechanism → kanna's own follow-up #3 (margin map) = the **argmax-margin gate**, reassigned to her as **#87**.

## 2026-06-14 03:15 — PR #83: Re-optimize M=32 tree topology with measured rho ladder + salvage oracle ✅ MERGED (decisive positive: max-branch-3 optimal, +1.13pp over #74; salvage oracle delivered; headline corrected to +18.2% / ~569 official; wirbel acceptance-cost-model axis CLOSED)

- **Branch:** `wirbel/rho-optimal-topology` · **Student:** wirbel · merged by advisor ~03:15Z
- **Status:** MERGED as research artifact (CPU-only analytic, no GPU, no served-file change -> BASELINE official bar UNCHANGED 481.53). Banks `scripts/profiler/rho_optimal_topology.py` + `research/spec_cost_model/report_rho_optimal_topology.md` + `rho_optimal_topology_results.json`.
- **Hypothesis:** re-running the Sequoia/DP tree optimization with the measured DECLINING rho ladder (rho2=0.4165 >> rho3=0.2655 > rho4=0.1908) instead of #74's borrowed flat rho=0.565 yields the true-rho-optimal M=32 topology (parent arrays land #71 should build) + the expected per-position salvage curve for land's debug-gate oracle.
- **Primary metric:** `measured_rho_optimal_M32_gain_pct = 0.1817` (+18.17% drafter-aware re-priced gain, W&B `6tghbnjn`). **Test:** `expected_pooled_branch_hit_salvage = 0.4165` (rho2 debug-gate target).
- **Results:**

| topology | E[T] | max-branch | gain (drafter-aware) | wall_tps proj |
|---|---|---|---|---|
| #74 (flat rho=0.565) | 5.157 | 4 | +17.04% | 531.5 |
| **#83 measured-rho-optimal** | **5.207** | **3** | **+18.17%** | **536.6** |
| delta re-opt | +0.96% | -1 (rank-4 dropped) | +1.13pp | +5.1 |

3 bases: **+18.2% relative / wall_tps x454.1 -> 536.6 / official x481.53 -> ~569 projected**. MC cross-check: 400k-trial sim E[T]=5.214 vs analytic 5.207 (|err|=0.007). Anchor: F_linear(8)=3.84445 == measured 3.8441.

- **Salvage oracle (land #71 per-position debug gate):**

| spine pos | width | E[salvage rank-2] |
|---|---|---|
| 1 | 3 | 0.397 |
| 2 | 3 | 0.431 |
| 3 | 2 | 0.413 |
| 4 | 2 | 0.428 |
| 5-9 | 1 | 0 |

Universal rank-2 gate = rho2 0.4165. A correct width-2 branch at any divergence reads ~0.41; byteshark broken tree-v2 read 0.033 (12x discrepancy = layout-bug signature).

- **Cost-model deviation:** g_d=0.168 drafter-depth term added (MTP runs `depth` sequential passes, 15.5-18.1% of step). Under the SAME M-only cost as #79, re-opt reads +23.3% vs +22.2% for #74 -> topology improved; the headline drop (from +21.8% to +18.2%) is an honesty correction for depth cost. The re-opt DELTA (+0.96%/+1.13pp) is cost-model-independent (both depth-9).
- **Conclusions:** (1) Declining rho ladder makes max-branch-3 optimal (width-4 buys +0.00pp under measured rho); (2) beyond-width-4 never pays (rank-5 leaf marginal 0.0179 < least placed node 0.0272); (3) salvage oracle delivered to land #71 (universal rank-2 gate = 0.4165); (4) headline corrected to +18.2% (~569 official) -- still well above 500 target and 488.07 competitor; (5) +1.13pp topology delta should be confirmed by lawine #82 A/B before banking; (6) wirbel's acceptance-cost-model axis (#49->74->76->79->83) is CLOSED for the tree build.

## 2026-06-14 02:54 — PR #81: Prompt-lookup/n-gram hybrid drafter — free accepted tokens on top of MTP (Step-0 gate) ✅ MERGED (decisive Step-0 gate: CLEARS at 0.4066 extra-accept, but realistic +1-3% & not buildable in stock vLLM → do-not-build-now, queued behind land #71 fork)

- **Branch:** `denken/prompt-lookup-drafter` · **Student:** denken · merged by morganmcg1 02:54Z
- **Status:** MERGED as research artifact (CPU-only, no-GPU, no-HF-Job, zero served-file change → BASELINE official bar UNCHANGED 481.53). Banks `scripts/analyze_prompt_lookup.py` + the verdict.
- **Hypothesis:** prompt-lookup/n-gram (PLD) gives training-free accepted tokens from self-repetition; AUGMENT mode (PLD on top of MTP, not replace) could add free tokens where MTP misses. Step-0 gate: self-ngram ≥~0.3 extra accept tok/step on the 128 public prompts → build; below → kill.
- **Primary metric:** `promptlookup_extra_accept_tokens_per_step = 0.4066` (W&B `ed46yvkz`). **Test:** `promptlookup_augment_tps_uplift_pct_independence_ub = 10.64`.

| metric | value | notes |
|---|--:|---|
| extra accept tok/step (vLLM-faithful) | **0.4066** | clears the ≥0.3 gate |
| TPS uplift (independence UB) | **+10.6%** | upper bound (q–m independent) |
| oracle best-occurrence UB | 0.494/step / +12.9% | absolute ceiling |
| n=2 ngram hits: generated vs prompt | 0.316 vs 0.207 | reasoning DOES self-repeat |
| extra tokens from MTP-full-miss steps (m=0) | 54.5% | the most valuable free tokens |
| realistic uplift (q–m correlated) | **+1-3%** | conservation-constrained full-span, a_H 0.90-0.95 |
| REPLACE mode E[T] | 1.45-1.51 | ngram-only LOSES to MTP 3.84 |

**Conclusions:**
1. **Gate CLEARS but the lever is modest + blocked.** Extra-accept 0.4066/step (>0.3) and reasoning self-repeats (generated n-gram 0.316 > prompt 0.207), refuting "reasoning is generated not copied → PLD won't fire." 54.5% of free tokens come from steps where MTP fully missed (most valuable kind).
2. **+10.6% is an INDEPENDENCE upper bound; realistic = +1-3%.** Under realistic positive q–m correlation (PLD fires on predictable spans where MTP already wins), the conservation-constrained sweep pulls it to +1-3%, → 0 at perfect correlation. True gain needs the q–m correlation pinned.
3. **Decisive blocker is COMPOSABILITY, not magnitude.** vLLM 0.22.0 `SpeculativeConfig.method` is single-choice — mtp XOR ngram. Only stock mode is REPLACE (ngram-only E[T]=1.45-1.51 LOSES to MTP 3.84). AUGMENT (the valuable mode) needs a fork of the spec-decode proposer loop + composition with land #71's tree-verify → do-not-build-now.
4. **denken reassigned → #85 (M=32 tree-verify non-GEMM overhead audit).** Prompt-lookup queued behind land #71's proposer-loop fork (revisit + pin q–m correlation after tree-verify lands). denken's GPU goes to de-risking the headline: the tree-overhead performance oracle complementing wirbel #83's salvage correctness oracle.

## 2026-06-14 02:46 — PR #36: int4-quantize the pruned 12k lm_head 🔒 CLOSED-BANKED (clean terminal LOCAL rung; lm_head lever exhausted; ubel → #84 SplitK verify-GEMM)

- **Branch:** `ubel/...` (int4 lm_head) · **Student:** ubel
- **Status:** CLOSED-BANKED (terminal local rung, NOT a frontier advance → no BASELINE change; 481.53 official stays). Result preserved here + on branch (recoverable). PR was never un-drafted and carried only a `terminal:false, pending_arms:true` marker (the `terminal:true` text in-thread was the advisor's template with `[...]` placeholder run-ids).
- **Hypothesis:** int4-quantize the pruned-12k lm_head (4× head-byte cut) on the PR #14 bf16-12k rung (131.60 local) for a cross-session-deterministic, contract-safe TPS bump.
- **Result:** **133.299 local TPS (+1.3% over 131.60), PPL 1.9713** (drift +0.0001), **GREEDY_IDENTICAL 128/128**, head 62.9MB→16.22MB, cross-session deterministic (real int4 Marlin GEMV bit-exact). 4th datapoint on the lm_head-bytes↔TPS bandwidth model, landed on projection (133.3 predicted).

| metric | value | notes |
|---|--:|---|
| local TPS (lmhead12k rung) | **133.299** | +1.3% over 131.60 bf16-12k rung |
| PPL (128rec) | **1.9713** | drift +0.0001 vs bf16-12k |
| greedy identity | **128/128** | int4 Marlin GEMV bit-exact, cross-session deterministic |
| head bytes | 62.9MB → 16.22MB | 4× cut |

**Conclusions:**
1. **Clean, validated work — but the lm_head lever is EXHAUSTED.** lm_head is only ~1% of decode (wirbel #30), so even a 4× byte cut is a negligible full-stack lever; the 133.3 rung is sub-frontier vs the 481.53 fa2sw split-KV frontier (PR #52).
2. **Cannot compose into the frontier:** the pruned-12k vocab would break full-vocab greedy-identity if dropped into the 481.53 stack. Issue #35 (Morgan-closed: "close this and move on") retired the lmhead12k LAUNCH lane → no official-benchmark path remains for this rung.
3. **Banked, not merged:** no valid terminal marker + sub-frontier + lever exhausted → closed with the record preserved rather than adding a sub-frontier submission to the tree.
4. **ubel reassigned → #84 (SplitK W4A16 verify-GEMM):** promote ubel's int4-Marlin-kernel experience to the #1 decode block — close denken #68's 23% HBM gap on the verify-GEMM (53% of decode, 77.1% HBM at M=8) via SplitK K-decomposition (arXiv:2402.00025). Lossless/greedy-safe, composes with land #71, ~+5-12% wall_tps ceiling.

## 2026-06-14 02:50 — PR #79: Pin rank-2+ drafter coverage (ρ) — the last borrowed input to the +18.7% tree gain ✅ MERGED (decisive measurement positive: ρ₂=0.4165/ρ₃=0.2655/ρ₄=0.1908; cov₂₋₄=0.6532 > borrowed 0.565 — gain was CONSERVATIVE; byteshark cross-val PASS; full max-branch-4 justified; ~586 official)

- **Branch:** `wirbel/rank-coverage` · **Student:** wirbel
- **Status:** MERGED as research artifact (cost-model measurement; no served-file change → BASELINE official bar UNCHANGED 481.53). Lands `scripts/profiler/rank_coverage.py`, `scripts/profiler/rankprobe_patch.py`, `scripts/profiler/treeshape_measured_accept.py`, `research/rank_coverage/rank_coverage_results.json` + pr79_report.md, `research/accept_calibration/treeshape_measured_results.json`.
- **Hypothesis:** The +18.7% M=32 tree-verify gain (wirbel #74, borrowed flat ρ=0.565 from EAGLE-3) has one open parameter: **ρ = rank-2+ drafter coverage** (P(target in drafter top-k | rank-1 rejected at first divergence)). If ρ < 0.565 the gain is overstated; if ρ > 0.565 it was conservative. Measuring locally from the deployed kenyan-duma MTP drafter's top-4 outputs vs the verifier's greedy path pins the last borrowed input, and cross-validating against byteshark's official-stack ρ₂=0.4130 (BLOCK=64) confirms the measurement is a drafter intrinsic, not a config artifact.
- **Primary metric:** `drafter_rank2_coverage = 0.4165` (ρ₂, W&B `z6wi4z4v` + `6wr8r2y0`). **Test:** `reprice_M32_proj_tps = 521.64` (M=32 re-priced on measured ρ ladder, old local base).

| metric | value | notes |
|---|--:|---|
| ρ₂ (rank-2 coverage) | **0.4165** | 12,869/30,874 first-reject divergence events where target in top-2 |
| ρ₃ (rank-3 coverage) | **0.2655** | declining ladder (not flat) |
| ρ₄ (rank-4 coverage) | **0.1908** | above cost threshold ~0.10 → width-4 still pays |
| cov₂₋₄ (aggregate top-4) | **0.6532** | > borrowed EAGLE-3 0.565 → gain was CONSERVATIVE |
| beyond-top-4 hard miss | **34.7%** | rank-5+ below GEMM-row cost → max-branch-4 confirmed optimal |
| top-1 q0 (cross-check #76) | **0.7335** | vs #76's 0.7287 (Δ0.0048) ✓ |
| per-depth ρ₂ | flat 0.397–0.445 | no depth trend across depths 1–7 |
| align_bad (greedy preserved) | **0** | 16,524 records; greedy-identity INTACT |
| M=32 re-price (old local base) | **+21.8% / 521.6 TPS** | ≈586 official (481.53×1.218) |
| byteshark cross-val Δ ρ₂ | 0.85% (0.4165 vs 0.4130) | PASS — drafter property, not block-size artifact |
| byteshark cross-val Δ cov₂₋₄ | 1.16% (0.6532 vs 0.6609) | PASS |
| byteshark cross-val Δ mean_emit | 1.96% (3.844 vs 3.921) | PASS |

**Conclusions:**
1. **ρ₂=0.4165 measured (replaces borrowed EAGLE-3 0.565).** The declining ladder ρ₂>ρ₃>ρ₄ reflects kenyan-duma MTP drafter's actual per-rank distribution; the borrowed scalar (35% above ρ₂) was wrong but the aggregate cov₂₋₄=0.6532 > 0.565 means the gain estimate was CONSERVATIVE, not overstated.
2. **Full max-branch-4 is justified.** ρ₂=0.4165 >> ρ₃=0.2655 > ρ₄=0.1908 >> ~0.10 threshold; 34.7% hard-miss beyond top-4 is below GEMM-row cost → M=32 max-branch-4 confirmed optimal per #68's roofline. wirbel #74's parent arrays stand; land #71 should build.
3. **M=32 re-price: +21.8% central → ~586 official.** 481.53×1.218≈586 TPS; well past 500-TPS target, clears need-for-speed 488.07 by ~20%.
4. **byteshark cross-validation PASS at all 3 metrics (<2%).** ρ₂ 0.85% / cov₂₋₄ 1.16% / mean_emit 1.96% between local BLOCK=16 and byteshark official-stack BLOCK=64 — coverage ladder is a drafter intrinsic.
5. **Per-depth ρ₂ flat (0.397–0.445, depths 1–7)** — depth-flat DP model used in #74 is justified; no depth correction needed for the topology.
6. wirbel reassigned → **#83 (rho-optimal-topology)**: re-run Sequoia DP with measured declining ρ (ρ₂=0.4165/ρ₃=0.2655/ρ₄=0.1908) to verify #74 is ρ-robust or find a better topology, and produce per-position expected-salvage oracle for land #71's debug gate (target ≈0.41 pooled vs byteshark's broken 3.3%).

## 2026-06-14 02:12 — PR #72: TPS measurement protocol — tighten the ±4.4% noise floor ✅ MERGED (decisive measurement positive: ±4.4% was estimator artifact; wall_tps CV 0.035% / MDE 0.2% N=1; wandb_logging bug fixed)

- **Branch:** `lawine/tps-noise-floor` · **Student:** lawine
- **Status:** MERGED as a research artifact + bug fix (measurement harness + `scripts/wandb_logging.py` 1-line fix; no served-file change → BASELINE official bar UNCHANGED 481.53). Lands `research/tps_noise_floor/` harness (N=12 run data, analysis scripts, PROTOCOL.md) + 1-line fix to `scripts/wandb_logging.py` (group kwarg was hardcoded → `--wandb_group` was a no-op for ALL callers).
- **Hypothesis:** The ±4.4% same-config TPS noise floor (from #56) is larger than our experiment deltas → the team can't tell a real <5% gain from noise. A hardened measurement protocol would shrink the effective noise floor and give a defensible MDE for every future TPS A/B.
- **Primary metric:** `tps_noise_floor_cv = 0.035` (wall_tps CV = **0.035%**, W&B `n07jrhxl`). **Test:** `tps_mde_pct_wall_paired_n1 = 0.095` (MDE = **0.095%** at N=1 paired on wall_tps).

| metric (A10G, deployed fa2sw_precache_kenyan, N=12 fresh, 128×512, conc=1) | mean TPS | CV | notes |
|---|--:|--:|---|
| `steady_gen_tps_mean` (old metric, fragile) | 449.06 | **0.33%** | unweighted interval-mean; cold 1st interval drags it |
| **`wall_tps` = tokens / decode_duration_s** | **454.12** | **0.035%** | = official `output_throughput` defn; **NEW STANDARD** |
| windowed steady (drop W=3 cold intervals) | 459.83 | 0.05% | robust interval-meter variant |
| `e_accept_exact` | 3.855 | 0.07% | near-deterministic greedy acceptance |
| **#56 reproduced on wall_tps** | 429.04→448.01 (+4.42%) | → **454.30→454.35 (+0.01%)** | throughput never moved; estimator did |

**Variance decomposition:**
- **(a) Warmup/cold-start — dominant for raw estimator.** First interval 29% below steady; dropping W≥1 collapses raw CV 0.33%→0.07%.
- **(b) Steady jitter — small.** windowed CV 0.05%, wall CV 0.035% (irreducible floor).
- **(c) Thermal/clock drift — ZERO.** A10G SM clock pinned 1710 MHz; temp flat ~53°C; TPS~time slope −0.006 tps/run (n.s.).
- **(d) Token nondeterminism — negligible.** E[accept] CV 0.07%; not a meaningful TPS-noise source.

**Recommended protocol (copy-paste):** decide every local TPS A/B on `wall_tps`, median N=3 fresh decode-only runs; MDE: **≥0.2% real at N=1; ≥0.1% at N=3**. Any delta ≥0.2% is real. Full protocol: `research/tps_noise_floor/PROTOCOL.md`.

**Conclusions:**
1. **The ±4.4% noise floor was an estimator artifact** — wall_tps was identical (+0.01%) across both #56 runs. The sub-5% wins concern is SOLVED by changing the metric, not by adding runs.
2. **New canonical local A/B metric: wall_tps ≈ 454** (replaces fragile "428.37 steady"; the 428.37 headline was the fragile metric's low point-estimate). Official bar unchanged 481.53.
3. **MDE 0.2% at N=1 / 0.1% at N=3**: land #71 tree-verify (+21-23% projected) clears by >100×; stark #78 GEMM fusion (~+2.6%) and denken #81 prompt-lookup are well above the floor. Sub-5% wins are now reliably detectable.
4. **wandb_logging.py bug fixed** (`group=group or agent`): `--wandb_group` was silently a no-op for ALL callers before this merge. All prior runs that used `--wandb_group` logged to the wrong W&B group.
5. lawine reassigned → **#82 (walltps-ab-runner)**: operationalize the protocol as a reusable `paired_tps_ab.py` runner for the whole team + re-baseline local frontier + retrospective re-screen of prior within-noise A/Bs.

## 2026-06-14 01:49 — PR #77: Drafter non-GEMM profile — map the real ~70% binding drafter cost ✅ MERGED (decisive FAIL-FAST negative: no contract-safe non-GEMM drafter lever; drafter axis fully harvested → land tree-verify #71)

- **Branch:** `denken/drafter-nongemm-profile` · **Student:** denken
- **Status:** MERGED as a research artifact (audit-only, zero served-file change → no BASELINE.md change; frontier bar UNCHANGED 481.53). Lands a reusable profiler `scripts/profiler/drafter_nongemm_profile.py` + `research/spec_cost_model/{drafter_nongemm_profile.json,report_drafter_nongemm_profile.md}`.
- **Hypothesis:** #75 showed ~70% of the drafter forward is NON-GEMM. Profile that non-GEMM mass per sub-op to find a contract-safe lever (the drafter's last remaining headroom before we commit to the tree-verify build).
- **Primary metric:** `drafter_nongemm_binding_subblock_pct_of_decode = 1.54` (W&B `q9p4vetv`). **Test:** `realistically_addressable_drafter_tps_headroom_pct = 0.0`.

| drafter non-GEMM sub-block (A10G, deployed M=1×K=7, conc=1) | cost | % decode step |
|---|--:|--:|
| **binding sub-block: `centroid_sampler_fused`** | **178 µs/step** | **1.54%** |
| attention | 61 µs/step | 0.53% (memory floor) |
| rest (fused glue + dispatch) | — | no hotspot |
| "gather only candidate set" optimization | already DEPLOYED (8192/262144 = 3.1%, 31× cheaper) | — |
| **realistically addressable headroom** | **~0%** | — |

**Conclusions:**
1. **Decisive FAIL-FAST negative.** There is no contract-safe non-GEMM drafter lever — the binding sub-block is 1.54% of the decode step, attention is at its memory floor, and the obvious win ("gather only the candidate set") is already deployed. Addressable headroom ≈ 0%.
2. **The drafter axis is now fully harvested** across #75 (forward roofline, int4-drafter refuted) + #77 (non-GEMM profile). No remaining drafter-side TPS lever survives the roofline.
3. denken's own verdict: *"Do not build a drafter non-GEMM optimization — land tree-verify (#71), the #1 lever."* Banks the third reusable profiler onto the advisor branch.
4. denken reassigned → **prompt-lookup / ngram hybrid drafter Step-0 viability gate** (orthogonal free-tokens lever; hedges the tree-verify timeline).

## 2026-06-14 01:38 — PR #34: Benchmark-matched reasoning corpus — break the 0.73 drafter plateau ✅ MERGED (decisive corpus positive: native 0.7792, +81% rel; BUT K=1 drafter not deployable vs MTP 3.844 — not a TPS win)

- **Branch:** `fern/bench-reasoning-corpus` · **Student:** fern
- **Status:** MERGED as a research artifact (corpus + EAGLE-3 training/eval pipeline; the deployed MTP serving stack is unchanged → no BASELINE.md change; frontier bar UNCHANGED 481.53). Lands `scripts/drafter/{gen_eagle3_corpus,train_eagle3,eval_eagle3}.py` + `research/eagle3_drafter/arch_notes.md`.
- **Hypothesis:** the 0.73 drafter-acceptance plateau is a CORPUS-distribution problem, not a capacity ceiling. Train EAGLE-3 on a benchmark-matched reasoning corpus (aime / gpqa / mmlu_pro) → break 0.73 on tf_acc and native accept/step.
- **Primary metric:** `native_accept_per_step_bench_holdout = 0.7792` (vs #25 ckpt 0.4315, **+81% rel**) — W&B `56ksyxgw` / `gua9x68j` / `cjhjnsff`. **Test:** `native_accept_per_step_bench_holdout_25ckpt = 0.4315`.

| quantity (EAGLE-3 drafter, 240-rec / 159k-tok bench holdout, feature_shift=1) | value |
|---|--:|
| **native accept/step (bench corpus ckpt)** | **0.7792** (vs #25 ckpt 0.4315, +81% rel) |
| teacher-forced acc (same holdout) | 0.7617 vs 0.4709 |
| per-source: aime / gpqa / mmlu_pro | 0.8426 / 0.8033 / 0.7006 |
| chain past step 1 (K=1 regime) | **collapses → ~1.78 tok/step** |
| deployed MTP K=7 E[T] (reference) | **3.844 tok/step** |

**Conclusions:**
1. **The corpus lever DECISIVELY breaks the 0.73 plateau** on both teacher-forced (0.7617) and native (0.7792) — benchmark-matched reasoning data is the right distribution. Strong, reproducible drafter-quality result.
2. **BUT this is NOT a TPS frontier win as-is:** the EAGLE-3 drafter is K=1-regime and its chain collapses past step 1 (~1.78 tok/step), **far below** the deployed MTP K=7 chain's E[T]=3.844. Step-0 acceptance is excellent; multi-step chaining is the gap.
3. **Residual ceiling = the K=1 training regime, not the corpus.** Merged to bank the proven corpus + training/eval pipeline.
4. fern reassigned → **#80 (multi-step / HASS drafter training)**: train the drafter on its own hidden states for steps 2..K to lift the chain past step 1 toward MTP's E[T], on the proven benchmark corpus.

## 2026-06-14 01:30 — PR #76: Calibrate deployed-chain acceptance to pin the tree-verify gain band ✅ MERGED (decisive positive: top-1=0.729, E[T]=3.844; M=32 +18.7%/≈508 TPS empirically anchored; wirbel→#79 ρ probe)

- **Branch:** `wirbel/acceptance-calibration` · **Student:** wirbel
- **Status:** MERGED as a research artifact (measurement-only, zero served-file change → no BASELINE.md TPS change; frontier bar UNCHANGED 481.53). Lands reusable harnesses: `scripts/profiler/accept_calibration.py`, `scripts/profiler/treeshape_measured_accept.py`; `research/accept_calibration/*.json`.
- **Hypothesis:** Pin the deployed chain's real per-rank acceptance to resolve the #49 vs #68 discrepancy (0.6792 vs E[accept]≈3.8-implied 0.775) and re-price #74's M=32/M=16 DP trees with measured acceptance. De-risk land #71 (tree-verify build).
- **Primary metric:** `deployed_chain_mean_tokens_per_step = 3.8441` (W&B `5m17r52s` / `zfzxl0np`, group `acceptance-calibration`).

| quantity (A10G, deployed MTP K=7, conc=1, 128 prompts × 512 tok) | value |
|---|--:|
| **top-1 acceptance (rank-1)** | **0.729** |
| conditional acceptance depth-1→7 | 0.729 → 0.847 (rising with depth) |
| **measured E[T] (tok/step)** | **3.844** (primary) / 3.849 (Prometheus, Δ0.005) |
| draft acceptance rate (E[T]−1)/7 | 0.406 |
| **M=32 tree re-price** | **+18.7%** (≈508 local TPS) vs +20.1% modeled (−1.4pp) |
| **M=16 tree re-price** | **+11.5%** vs +13.1% modeled (−1.6pp) |
| M=32 still dominates M=16? | Yes |
| Fail-fast triggered? | No — tree gain not marginal |

**Reconciliation (decisive):** #49's 0.6792 was an EAGLE-3 drafter scalar (wrong drafter — deployed MTP kenyan-duma has higher top-1). #68's back-solve top-1≈0.775 overstated because real acceptance profile RISES with depth (0.729→0.847); constant-p forced to hit E[T]=3.84 sits above the true top-1. **Authoritative: top-1=0.729, E[T]=3.844** (not a real discrepancy — two estimators applied to the same chain, plus one wrong-drafter scalar).

**Conclusions:**
1. **M=32 +18.7% / ≈508 local TPS empirically anchored.** #74 projection confirmed to −1.4pp; tree not marginal; M=32 dominates M=16.
2. **Dominant uncertainty shifted** from top-1 (resolved) to **ρ = rank-2+ drafter coverage** (P(target == drafter rank-2/3/4 | rank-1 missed)). Linear chain can't expose ρ; borrowed EAGLE-3 ρ=0.565 → credible band **+11…+25%, central +18.7%**.
3. wirbel reassigned → **#79 (rank-2+ drafter coverage probe)** — measures ρ locally, cross-validates byteshark's official-stack rank-2 conditional.
4. land #71: proceed with M=32 build. Expected official projection: ~481.53 × 1.187 ≈ **571 TPS** (>>500 target). Remaining risk = ρ; wirbel #79 + byteshark resolve it.

## 2026-06-14 00:47 — PR #75: Drafter-forward roofline — is the 15.5% block bandwidth-bound? ✅ MERGED (decisive negative: refutes int4-drafter-for-TPS; the drafter's #2-block headroom is non-GEMM, not weight bytes)

- **Branch:** `denken/drafter-forward-roofline` · **Student:** denken
- **Status:** MERGED as a research artifact — the **sibling roofline to #68** (verify-GEMM). Audit-only, zero served-file change → no BASELINE.md change. Frontier bar **UNCHANGED 481.53.** Lands a reusable profiler (`scripts/profiler/drafter_forward_roofline.py`) + the drafter decode-composition cost report.
- **Hypothesis:** stark #70 was building int4 drafter weights on an **unaudited premise** — that the K=7 MTP drafter forward is weight-bandwidth-bound at the deployed M=1×K=7. A Step-0 roofline (#68 method, FP16-ceiling) validates or refutes that premise *before* stark spends the build.
- **Primary metric:** `drafter_forward_pct_hbm_peak_at_M1K7 = 47.17%` (W&B `uknpbk94`, finished; primary verified exact).

| quantity (A10G, drafter bf16, deployed M=1×K=7) | value |
|---|--:|
| **`drafter_forward_pct_hbm_peak_at_M1K7`** | **47.2%** (7-pass GEMM chain, launch-free onegraph) |
| arithmetic intensity at M=1 | 1.0 FLOP/byte (ridge 86.8 → 86× below) → memory-bound *regime* |
| achieved compute at M=1 | 0.45% of FP16 peak (52.1 TFLOPS realizable ceiling) |
| most-repeated GEMVs (sliding-attn q/o, 6 of 19/pass) | **19% HBM** → **latency/launch-floored, not bandwidth-saturated** |
| 7-pass drafter GEMM chain (deployed graph) | **566 µs/step = 4.88% of the 11.6 ms decode step** |
| drafter forward total (#69 budget, the #2 decode block) | 1798–2100 µs = 15.5–18.1% |
| → **non-GEMM** drafter (centroid sampler + 262k masked-embed gather + SDPA + sampling) | **~69–73% of the drafter** (untouched by int4) |

- **int4-drafter-for-TPS ceiling (stark #70 cross-check):** hard ceiling (every drafter GEMM → 0 µs) = **+5.13%**; int4 bandwidth-scaling = +3.62% (optimistic); **realistic +1.5…+3%**. Premise-implied naive ("3.5× faster 15.5–18.1% block") = +12.5…+14.9% → **overstated ~3–5× vs ceiling, ~4–8× vs realistic.** The premise is right about the *regime* (AI≈1, memory-bound) but wrong that the block is a *saturated* bandwidth wall (47%, not 75–100%) — and int4 touches only 4.88% of decode.
- **Onegraph:** drafter runs **inside blake's `onegraph` (CUDA-graphed), launch-free** — it does NOT pay #68's ~55 µs/call eager floor (eager chain 2859 µs vs graph 566 µs; the 2.3 ms gap is already-harvested launch overhead, not free headroom). int4 working set 6.5 MB still > 6 MB A10G L2 → weights still spill every pass; int4 does not make them L2-resident.
- **Pass-count lever (feasibility only): INFEASIBLE with unchanged outputs.** MTP is autoregressive (pass *i* consumes pass *i−1*'s token) → no single wider GEMM yields the identical 7-token chain; L2-residency needs <6 MB (int4 6.5 MB still spills); K<7 changes accept behavior (fern #34's axis).
- **Conclusions / actions taken:**
  1. **stark #70 CLOSED** — int4-drafter-weights-for-TPS refuted (≤+3% realistic, +5.13% ceiling; not the double-digit win the framing implied). stark reassigned to a higher-value orthogonal BUILD lever (prompt-lookup/ngram free draft tokens).
  2. **Also informs open2-askeladd #57** (W8A8 int8 drafter) — same drafter-quant-for-TPS premise; flagged cross-board (saves their quota, byteshark-negatives ethos).
  3. **The real drafter lever is the ~70% non-GEMM**, not the weights — denken reassigned to a per-op decomposition of the non-GEMM (centroid sparse sampler / 262k masked-embed gather / SDPA / sampling) using his #75 reconstructed-module harness, to find the fattest reducible/fusable op. Secondary: drafter kernel-fusion (lift the 47% chain / 19% GEMVs off the launch floor) — contract-safe, larger than int4.
  4. **Verify-GEMM (53%, #68: free to widen to M≤32) remains the higher-value block** (land #71 tree-verify = the 500-path); the drafter is the #2 block but with little weight-byte headroom.

## 2026-06-14 00:40 — PR #74: TPS-optimal tree-shape under denken #68's measured M≤32 verify-cost curve ✅ MERGED (the concrete build target for land #71)

- **Branch:** `wirbel/tree-shape-cost-model` · **Student:** wirbel
- **Status:** MERGED as the **canonical TPS-optimal tree-shape verdict** — same artifact class as #68 (a decisive, MC-validated research deliverable; audit-only, zero served-file change → no BASELINE.md change). Frontier bar **UNCHANGED 481.53.** Converts #68's real cost curve + wirbel's own #49 acceptance model into a concrete build target for **land #71**.
- **Hypothesis:** #49's DP-optimal tree (+16% TPS) assumed a simple verify cost; re-solving the DP against #68's *measured* non-uniform V(M) curve (cheap tile-tops M=16/32, expensive M=24, hard M=33 cliff) yields the actual TPS-optimal (shape, M) — the exact topology land #71 should build.

| operating point (real #68 cost, g=0.532, measured p, geom) | E[T] | step mult vs M=8 | proj local TPS | vs deployed linear (428.37) |
|---|--:|--:|--:|--:|
| deployed **linear K=7 / M=8** (anchor) | 2.976 | 1.000 | **428.37** | — |
| linear own-optimum (M=16, saturated) | 3.111 | 1.034 | 433.1 | +1.1% |
| **DP tree M=16** (Marlin tile-1 top — Step-1 build) | **3.481** | **1.034** | **484.7** | **+13.1%** |
| **DP tree M=32** (tile-2 top — PRIMARY) | — | **1.098** | **514.32** | **+20.1%** |

- **Headline:** the TPS-optimal tree is the **M=32 DP tree → ~514 local TPS, +20.1%** over the deployed linear K=7 chain; cheaper secondary at the M=16 tile-1 top → **~485 TPS, +13.1%.** Primary metric `treeshape_opt_proj_tps_gain_real_costcurve = +0.2007` (M=32).
- **Three canonical takeaways:** (1) **The optimum did NOT shift from #49** — the deep-spine DP tree at M=32 survives the real-cost refinement (projection even ticks *up* +19.0%→+20.1%, because measured M=32 mult 1.098 < modeled 1.108). Reassurance, not a pivot. (2) **"Build to a tile-top, never mid-tile"** — M=16/M=32 sit at the cheap Marlin tile tops (9 µs/row marginal); **M=24 is strictly dominated.** (3) **Shape/budget separation:** verify cost depends only on node budget M, not tree shape (the GEMM processes all M rows regardless) → the tree designer optimizes acceptance freely under a hard M≤32 row budget.
- **Build targets handed to land #71** (advisor comment, 00:40:49Z): **(1) Step-1 = M=16 DP tree** `parent=[-1,0,0,0,1,1,2,4,4,5,6,7,11,12,13,14]` (16 nodes, depth 8, 4 rank-2+ branches) — build FIRST to validate measured acceptance + greedy identity on the real tree-verify path; **(2) Primary = M=32 DP tree** (32 nodes, depth 9, 9 rank-2+ branch points, max branch 4, bushy crown).
- **Validation:** brute-force n≤7 == DP; MC 400k max rel-err 0.11%; robust **+16.5–21.7%** across pricing / GEMM-share / rank-decay / base-acceptance variants. W&B `p1yyrwpr`. Local cost-model study, **no HF Job**, lossless by construction.
- **One open number (→ wirbel #76):** the projection brackets +18% (if rank-1=0.6792, #49) vs +20% (if top-1≈0.775, implied by deployed E[accept]≈3.8). These disagree materially → **wirbel reassigned to #76** to pin the deployed chain's real per-rank served acceptance, turning "+18–20% modeled" into one defensible number before land #71 spends any submission quota.
- **Artifacts:** `research/spec_cost_model/report_treeshape_real_cost.md`, `treeshape_real_cost_results.json`, `scripts/profiler/treeshape_real_cost.py`.

## 2026-06-14 00:15 — PR #68: Verify-GEMM M=8 roofline audit — is the 53% block free to widen? ✅ MERGED (GREEN — greenlights the 500-path)

- **Branch:** `denken/verify-gemm-m8-roofline` · **Student:** denken · merged to advisor branch (commit `f2ec624`).
- **Status:** MERGED as a **characterization keeper** (reusable roofline harness + cost curve; #49/#51-class positive verdict, not a baseline-beater → no BASELINE.md change). Frontier bar **UNCHANGED 481.53.** This is the audit the entire **tree-verify thread (land #71)** was gated on — verdict is decisive GREEN.
- **Hypothesis:** at the deployed M=8 verify, is the dominant 53.2% int4-Marlin verify-GEMM block (#30) compute/tile-bound (irreducible) or weight-bandwidth-bound (free headroom to widen M for multi-candidate/tree verify)?

| quantity (A10G, int4 W4A16 Marlin, M=8) | value |
|---|--:|
| achieved HBM bandwidth | **462 GB/s = 77.1% of 600 GB/s peak** → **BANDWIDTH-BOUND** |
| achieved compute | 13.0 TFLOP/s = **20.2% of FP16 peak** (measured 64.3 TFLOPS) |
| arithmetic intensity @ M=8 | **28 FLOP/byte vs ridge 107** (3.8× below) |
| widen M=8→16 (top-2 tree) | **+6.4% verify-GEMM** (+2.7% step), marginal **9 µs/row** |
| widen M=8→32 (top-4 tree) | **+18.4% verify-GEMM** (~+7.7% step), aggregate **~37 µs/extra row** |
| M=24 (avoid) | +16.9%, marginal **64 µs/row** (expensive) |
| **M=33 — HARD TILE CLIFF** | **+53.3%** (Marlin 16-row M-tile boundary; reproduces #51's M=33/49 cliffs) |

- **Verdict:** "**at M=8 the int4 W4A16 Marlin verify-GEMM is unambiguously WEIGHT-BANDWIDTH-BOUND, not compute/tile-bound. Free verification headroom EXISTS and is bounded by the Marlin M=33 tile cliff.**" ~80% of the verify-GEMM is pure weight-movement that only 8 rows consume → ~4× under-utilised per-weight-read amortization. **Free window M ∈ [8, 32]: up to 4× more candidate positions at ~37 µs/row, hard ceiling M=33.** Break-even for the downstream tree: an M=8→32 batch (+898 µs) is net-positive TPS if it adds **> ~0.43 accepted tokens/step** — a low bar for a width-2…4 tree at the drafter's *existing* depth (adds verify rows **without** adding sequential drafter forwards).
- **Premise correction (strengthens the conclusion):** Marlin W4A16 **dequantizes int4→FP16 on-chip** and runs FP16×FP16 tensor-core MACs (arXiv 2408.11743 §3); 4-bit is a *weight-storage* format that cuts HBM traffic 4×, never an int4 compute path. So the compute ceiling is the **FP16 peak (64.3 TFLOPS)**, not int4 (~280 TOPS) — using int4 would have *understated* utilisation 4× and falsely implied more headroom. The "completely free to widen" impression from earlier eager timing was a ~55 µs/call launch-overhead floor; launch-free CUDA-graph timing reveals the true ~37 µs/row.
- **Empirical complement (the shape of the headroom):** public **byteshark linear K=8 probe = VALID but SLOWER, 470.84 < 481.53** (verify M=9, deep inside the cheap M≤32 GEMM regime) — yet it *loses* TPS because **linear** K-widening adds a sequential drafter forward (drafter 15.5% of decode) at low marginal accept probability. **Takeaway: the GEMM headroom is real but linear chains can't spend it — it must go to multi-candidate/TREE verify (parallel candidates at fixed depth).** Exactly the lever this audit greenlights.
- **W&B (group `verify-gemm-m8-audit`):** `av8a5wh8` (launch-free CUDA-graph, primary) · `av98bjsw` (eager cross-check showing the launch floor). Local A10G only, **no HF Job** (read-only GEMM microbench, lossless by construction → PPL 2.3772 / greedy-identity 128/128 definitionally unchanged).
- **Artifacts (merged, now canonical team assets):** `scripts/profiler/verify_gemm_roofline.py`, `research/spec_cost_model/verify_gemm_roofline.json`, `research/spec_cost_model/report_verify_gemm_roofline.md`.
- **Follow-ups / propagation:** (1) **land #71** builds the tree-verify serving path sized to **M ≤ 32**, snapping total verify rows to ≤32 (never cross M=33) — handed the exact M-budget. (2) **wirbel → #74** re-solves the #49 Sequoia DP-optimal tree under this *measured* non-uniform V(M) curve (pack candidates into the cheap-marginal M=16/M=32, avoid M=24, hard M≤32) → exact build topology for land #71. (3) **denken → #75** drafter-forward roofline (the last unaudited decode block; validates/refutes stark #70's int4-drafter bandwidth-bound premise). Does NOT change drafter K or the AdaEDL/#54 dynamic-K lane (scope guard).

## 2026-06-14 00:15 — PR #69: Attention split-KV roofline audit — is the #2 block (19.6%) at the floor? ✗ CLOSED (NEGATIVE — attention is irreducible)

- **Branch:** `wirbel/splitkv-nseg-roofline` · **Student:** wirbel
- **Status:** CLOSED as the **third clean Step-0 systems negative** (with #65 CUDA-graph, #67 norm-fusion) — a keeper-in-the-record that sharpens the lever map. No code/served change → frontier bar **UNCHANGED 481.53.** Excellent fail-fast discipline.
- **Hypothesis:** the split-KV verify-attention is a custom (non-Inductor-fused) kernel ⇒ may carry hand-tunable headroom at the served M=8.

| quantity (deployed M=8, post-#43) | value |
|---|--:|
| attention % of GPU-busy | **7.6%** (was 19.6% pre-#43) → already the **#3 block, not #2** |
| attention µs/step | **605** (was 1836; #43 cut it 3.03×) |
| achieved BW vs peak | **20.0%** (96.6 GB/s vs 482 GB/s copy) — memory-**LATENCY**-bound, not BW-bound |
| occupancy @ n_seg=16 | **96 CTAs ≥ 80 SMs → saturated** (no occupancy bump available) |
| n_seg sweep {1…64} × ctx | **deployed n_seg=16 is exactly optimal at served-dominant shapes** (sliding ctx256 43.8% of cycles, full ctx512/1024 all 1.00×) |
| oracle best-vs-deployed ceiling | **+0.126% TPS** — and un-CUDA-graph-able (n_seg is a onegraph capture-shape constexpr) |
| free attention→0 ceiling (hypothetical) | only **+8.2% TPS** (de-prioritised) |

- **Verdict:** "**BW-bound? Occupancy YES, bandwidth NO — it's the irreducible conc=1 latency floor. Residual lossless headroom ≈ +0.13% TPS (oracle, un-CUDA-graph-able). No fix worth prototyping.**" At conc=1 each layer reads one sequence's KV (sliding 0.25–1 MB, full 2.2 MB) — far below the working set needed to hide HBM latency on 80 SMs → 20% of peak is the **floor, not slack** (BW *rises* monotonically with read size = the latency-bound signature). 80% peak only exists at large batch, which this single-stream submission never sees.
- **Two premise corrections banked:** (1) **Attention is already the #3 block at 7.6%, not #2 at 19.6%** — the 19.6% is the stale #30 *pre*-split-KV number; **#43 already harvested this block** (wirbel's own `r0ahjs45` re-profile). The PR chased a number #43 had already taken. (2) The served kernel is **100% stock vLLM-native Triton `unified_attention`** (3D split-KV/FlashDecoding) — not a custom submission kernel we own, and not Inductor-fused; the fa2sw FA2 router is **INERT** (0 FlashAttention kernels in the served trace; vLLM forces TRITON_ATTN for the heterogeneous sliding-256/full-512 head_dims).
- **W&B:** `rajcg6an` (group `attention-splitkv-audit`). Local A10G only, **no HF Job** (read-only op-microbench; served stack untouched → PPL 2.3772 / 128/128 definitionally unchanged).
- **Artifacts:** `research/profiling/splitkv_nseg/` (`nseg_sweep.py`, `aggregate.py`, `FINDING.md`, `breakdown.md`).
- **MAP UPDATE (load-bearing):** with **#65 (CUDA-graph), #67 (norm/elementwise), #69 (attention)** the decode **SYSTEMS layer is confirmed fully harvested.** Combined with **#68 (verify-GEMM bandwidth-bound, free to widen M≤32)**, the open frontier is now unambiguously **ALGORITHMIC** — verify **width** (→ land #71 tree-verify) and **acceptance/tokens-per-step**. With verify-GEMM (#68), attention (#69) and drafter (incoming #75/stark #70) roofline-mapped, all three big decode blocks are characterised.
- **Follow-ups:** **wirbel → #74** (he authored the #49 tree cost model → the right owner to find the TPS-optimal tree-shape under denken #68's real V(M) curve, feeding land #71). Flagged-not-implemented: **fa2sw dead-config cleanup** (the inert FA2 sliding router — pure simplification, no perf/PPL change, in the submission name); de-prioritised cross-layer KV read-coalescing (YOCO/CLA — would break the lossless gate).

## 2026-06-14 00:10 — PR #56: max_num_batched_tokens served A/B on the split-KV #1 stack ✗ CLOSED (parity characterization keeper — NOT a winner, NOT a regression)

- **Branch:** `lawine/maxbatchtok-served-ab` · **Student:** lawine
- **Status:** CLOSED as a parity/characterization keeper. No served-file change (research-only A/B harness + bugfix only). Frontier bar **UNCHANGED at 481.53**. Disposition: the knob is conclusively closed (parity + invalid-above-512); lawine's own "Suggested follow-ups: None on this knob."
- **Hypothesis:** sweeping `MAX_NUM_BATCHED_TOKENS` (512/2048/4096/8192) on the deployed `fa2sw_precache_kenyan` stack yields a decode-TPS gain and/or silences the #52 spec-decode launch warning.

| `max_num_batched_tokens` | steady TPS (n=14) | Δ vs control | PPL | completion | valid? |
|---|--:|--:|--:|---|---|
| **512 (control / deployed)** | **448.01** | — | **2.3767** | 128/128 | ✅ |
| 2048 | 445.92 | −0.47% | OOM | 128 decode, PPL crash | ❌ |
| 4096 | 453.40 | +1.20% | OOM | 128 decode, PPL crash | ❌ |
| 8192 | 449.56 | +0.35% | OOM | 128 decode, PPL crash | ❌ |

- **Analysis:** clean NEGATIVE (parity), with two extra teeth. (1) **No decode-TPS leverage** — at conc=1 / `max_num_seqs=1` each decode step verifies only M=8 tokens (far below any mbt), so the knob governs only prefill chunking; every inter-arm delta (≤+1.2%) is *inside* the control's own +4.4% run-to-run swing (429.04 vs 448.01 same-config). (2) **512 is the only PPL-passing value** — mbt≥2048 OOMs the `prompt_logprobs` log_softmax (+1.34 GiB) on the validity pass (decode completes 128/128, the gate crashes); footprint grows monotonically 20.92→21.02 GiB. Caveat: local A10G 22.06 GiB vs official a10g-small ~24 GiB, so the OOM might not reproduce officially — but there's no TPS upside regardless. (3) **#52 warning is benign AND structurally un-silenceable** — `vllm.py:1597` silences only at mbt≥8192, `scheduler.py:281` only at mbt≤4096; the regions never overlap, so some warning always fires; the only spec-decode-silencing value (8192) OOMs PPL. **Net: the deployed `MAX_NUM_BATCHED_TOKENS=512` is decode-optimal and the only gate-passing value — validated, no change.** Useful invariant banked: the **validity pass, not decode, is the memory-tight phase** at 0.90 util (matters for any future activation-growing change, e.g. tree-verify wider-M / land #71).
- **W&B (group `maxbatchtok-served-ab`):** 512→`3756geng` · 2048→`3vvsjm10` · 4096→`q28zoru2` · 8192→`k76d5d0a`. No HF job (local served A/B only).
- **Bug fix (kept on branch, cherry-pickable):** made the research-only `maxbatchtok_ab.py` harness's wandb-log + PPL pass non-fatal so a PPL OOM is captured as data (`engine_oom=true`) rather than discarding a completed arm. No served files touched.
- **Follow-ups:** none on this knob (closed). The frontier lever remains **(b) more accepted tokens per weight read** → tokens-per-step: **land #71 tree-verify serving path** (deploys wirbel #49's +16%), **denken #68 verify-GEMM roofline**, **lawine #72 noise-floor protocol** (needed to detect sub-5% wins). Queued idea (no idle seat): **ngram/prompt-lookup hybrid drafter** (training-free copy-span tokens-per-step).

---

## 2026-06-13 22:13 — PR #52: fa2sw split-KV — Issue-#46-approved one-shot HF launch ✓ MERGED ⭐ NEW PUBLIC #1 / NEW OFFICIAL FRONTIER (481.53 official TPS)

- **Branch:** `lawine/fa2sw-splitkv-official-launch` · **Student:** lawine
- **Status:** MERGED as the **new official frontier baseline.** First gated HF job to confirm a rung above the 126.378 AR floor on the spec-decode frontier → **the official bar all submissions must beat moves 126.378 → 481.53 TPS.** Human-approved launch (Issue #46, Morgan: "approved, lessgo!"); no submission-file changes (the PR is the launch record — served stack is the already-merged `submissions/fa2sw_precache_kenyan/` with #43 split-KV).
- **Hypothesis:** the locally-validated fa2sw split-KV stack (linear MTP K=7 + #43 3D split-KV, 428.37 local steady-state) reproduces on official a10g-small hardware above the prior public #1 (rock-ai 459.72), gated on the #50 fail-closed `official_gate` PASS@128 preflight.

| metric | value | gate |
|---|--:|---|
| **Official TPS (a10g-small)** | **481.53** | **NEW PUBLIC #1** (vs rock-ai 459.72, +4.74%; +13.4% over ~424.5 repro baseline) |
| PPL | 2.3772 | ≤ 2.42 ✓ |
| completed | 128/128 | ✓ |
| modalities | text+image+audio | all loaded ✓ |
| official_gate (preflight) | PASS@128 | split-KV patch engaged, zero 2D fallback ✓ |

- **Analysis:** clean reproduction — landed mid-projection (PR #43 projected 471–493). Pre-launch `official_gate=PASS@128` with the split-KV patch **engaged** (M=8 verify → 3D FlashDecoding every step, zero fallback, backend TRITON_ATTN). Greedy-identity DIVERGENT is an internal signal only (the official gate has no token-identity check, kanna #38) → spec decode is leaderboard-legal. **Standing risk UNCHANGED (the programme's #1):** the private re-run gate — kanna #44 probe predicts ~12.4% public→private on a pure-chat proxy (WOULD-FAIL >5%); the 481.53 is the **public** number; private stability is a separate open axis (kanna #55 calibrating on this exact frontier).
- **W&B:** `2x9fm2zx`, `fwo8rs05` (official launch; job `6a2dce05871c005b5352c0b9` COMPLETED, run prefix `results/senpai/fa2sw-precache-kenyan-20260613T213911Z`, `ppl_summary.json` 61,797 tokens). Leaderboard row pending organizer re-sync.
- **Follow-ups:** (a) report the new #1 to Issue #46 (done); (b) `max_num_batched_tokens` warning A/B — separate PR, touches the timed path; (c) #50 audio functional-probe polish (local tooling); (d) the open frontier lever stays the **private-stable acceptance** axis (kanna #55) + the verify-GEMM/drafter-forward decode blocks (ubel verify-GEMM, denken #54 entropy-K, wirbel #53 reprofile).

---

## 2026-06-13 21:55 — PR #51: accepthist dynamic-K on post-#43 split-KV cost curve ✓ MERGED (characterization + bugfix keeper — decisive negative, official bar UNCHANGED)

- **Branch:** `denken/accepthist-dynamic-k` · **Student:** denken
- **Status:** MERGED as a characterization + bugfix keeper. Official TPS bar **UNCHANGED at 126.378** (primary `projected_dynamic_k_tps_costmodel_post43_ctx512`=343.1 is a cost-model projection, **+0.12% vs static K=11**=342.7 = noise; not a served number, not comparable to the 428 served baseline).
- **Hypothesis:** dynamic draft length via acceptance history (`accepthist`) beats static K\*; #43 split-KV flattened cost(K) so argmax K\* should shift up; the public top-3 VALID (459) all use accepthist. **Premise corrected (wirbel #49, propagated to #51):** the deployed stack is **LINEAR MTP K=7 (M=8 verify), not an M=45 tree** → K varies on the linear chain; tree cost-model (540) is not the baseline.

| post-#43 ctx512 policy | TPS | vs K=11 | mean_K (sd) |
|---|--:|--:|---|
| static K=11 (= best static) | 342.7 | — | 11 (0) |
| **clairvoyant ORACLE** | 400.6 | **+16.9%** | — |
| best AIMD | 300.5 | −12.3% | 6.6 (3.6) |
| best window-mean linear | 328.5 | −4.1% | 10.1 (3.5) |
| **best realizable (LUT)** | **343.1** | **+0.12%** | captures **0.7%** of oracle |

- **Analysis:** decisive NEGATIVE on the headline. Two premises fail under measurement: (1) **#43 does NOT push K\* up — stays 11 on every curve/ctx** because the operating point is pinned by **Marlin int4 GEMM tile cliffs (M=33 +2.0ms, M=49 +2.9ms)**, and split-KV only accelerates *attention*, leaving the cliffs (hence argmax) put; (2) **acceptance history is too weak a predictor** (window-mean→next r≈0.32; lag-1 autocorr +0.16) → realizable control captures **<8%** of the real +16.1% oracle ceiling → net ≈0. **Split-KV *shrinks* the dynamic-K headroom** (oracle 25.2%→16.1%): flattening attention makes the unchanged GEMM staircase relatively more dominant — opposite of the hypothesis. **Reconciliation:** static optimum drops 11→**≈7** at the real e_accept≈3.82 → **the deployed linear K=7 is already near-optimal statically** (no static re-tune win either). **Keepers:** the `--sim-K` argmax-default fix (closes the PR#41/BASELINE.md:90 residual — every run now prints its `ARGMAX OPERATING POINT`); the re-grounded post-#43 cost curves (**#43 helps *more* at long ctx: verify −2.6%@256 → −7.1%@1024**); tooling `accepthist_controller.py` + `spec_cost_model.py --splitkv-patch` (redirect counter `total_redirected=106260` proves the patch fired) + `compare_splitkv_curves.py`. Tooling-only diff — no served-submission change. PPL 2.377 preserved by construction (greedy-exact; valid per #38).
- **W&B:** `wfi3jtkq` (sim; `splitkv_ctx512_static11_tps`=342.700, `splitkv_ctx512_oracle_gain_vs11_pct`=16.901, `realizable_frac_of_oracle`=0.007 — all confirmed), `6o8xaofq` (cost curve), group `accepthist-dynamic-k`. CPU sim + GPU cost curve (~21.6 GB A10G).
- **Follow-ups:** (a) **drafter-ENTROPY dynamic-K (AdaEDL, denken's suggestion 1) → denken #54** — entropy at draft time is a strictly stronger predictor than acceptance history; the *correct* read of the public top-3. (b) split-KV **net-negative at M=8/short-ctx** (+15.5%@ctx256) → **context-gate** the redirect (NOT M≥33) → routed to **wirbel #53**. (c) spine-E→DP tightening of `tree_acceptance_model.py` now **unblocked** (#51 landed) → queued to wirbel, rebased on #51.

---

## 2026-06-13 21:42 — PR #48: Token-frequency logit bias on the drafter ✓ MERGED (characterization keeper — decisive negative, official bar UNCHANGED)

- **Branch:** `kanna/token-freq-logit-bias` · **Student:** kanna
- **Status:** MERGED as a characterization keeper. Official TPS bar **UNCHANGED at 126.378** (decisive negative; primary `tps`=463.49 is the best biased arm, *below* the in-screen bias=0 baseline 471.35).
- **Hypothesis:** a static unigram logit bias on the drafter (boost top-K frequent output tokens) raises drafter acceptance without touching the verifier → +1–3% acceptance → +5–15 TPS, greedy-exact. **Forced deviations (both more favorable to the claim):** no `train.py --local-only --env` → reused the #44 LocalServer + `sglang.bench_serving` harness (fresh server/arm, one changed var); drafter is the centroid-sparse MTP head (not dense [B,262144]) → sparse-candidate bias table + drafter-only re-rank, bias=0 bypasses the hook (byte-identical to leaderboard, stays on the fused kernel).

| bias (K=500, n=32) | E_accept | ΔE_acc | TPS | ΔTPS% | per-step lat |
|---|--:|--:|--:|--:|--:|
| **0.0** (fused, =leaderboard) | 3.95587 | — | **471.35** | — | **8.29 ms** |
| 0.5 (grid optimum) | 3.97793 | **+0.56%** | 463.49 | **−1.67%** | 8.48 ms |
| 1.0 | 3.95160 | −0.11% | 461.15 | −2.16% | 8.47 ms |
| 2.0 | 3.87126 | −2.14% | 451.94 | −4.12% | 8.47 ms |

- **Analysis:** decisive NEGATIVE for TPS. TPS ≈ E_accept / latency moves in opposite directions: acceptance best-case +0.56% (b=0.5; *reverses* at higher bias — the FT'd MTP head already encodes the unigram marginal, so an external prior pulls it off the verifier's conditional argmax, consistent with #25's plateau ~0.73), while leaving the fused Triton sparse-argmax kernel costs a constant **+2.2%/step** (bias-independent = implementation cost), ~4× the gain. Full (K×bias) grid bounded: optimum K=500/b=0.5 = +0.56%. Even a zero-cost *fused* version ceilings at **~474 TPS (+2.6)** → "don't pursue." PPL 2.3767 unchanged by construction. Strategic read (with #49): cheap inference-time tricks are exhausted; the real acceptance lever is drafter DATA quality (land #9 / fern #34), not re-ranking.
- **W&B:** `96pn3c43` / `rrp0xc6e` (K=500 ×2, bit-identical E_accept) / `rggrg6r6` (K=100) / `l32wjlig` (K=1000). Ships `scripts/validity/drafter_bias_screen.py` (reusable drafter-tweak A/B harness) + `build_freq_bias_tokens.py`.
- **Cleanup queued → kanna:** relocate/inert the bias hook out of the about-to-launch frontier submission `fa2sw_precache_kenyan/sitecustomize.py` (Step 0 of kanna's next PR). **Reassigned → kanna:** private-gap calibration (#44 follow-up) — quantify the split-KV stack's private-re-run risk before the launch lands.

---

## 2026-06-13 21:32 — PR #49: Sequoia DP-optimal draft tree (cost-model study) ✓ MERGED (characterization keeper, official bar UNCHANGED)

- **Branch:** `wirbel/sequoia-dp-tree` · **Student:** wirbel
- **Status:** MERGED as a characterization keeper. Official TPS bar **UNCHANGED at 126.378** (primary_metric `dp_vs_linear_tps_gain_own_opt_costmodel`=1.1677 is a cost-model ratio, not throughput; the lane has no servable path).
- **Hypothesis:** a Sequoia (arXiv 2402.12374) DP-optimal draft tree beats the fixed/balanced tree by +3–15% E[T] on our measured acceptance, composing with the merged split-KV verify (#43). **Premise corrected by wirbel:** the deployed `fa2sw_precache_kenyan` drafter is **linear MTP K=7 (M=8 verify), not a width-4 tree**; vLLM 0.22 has no tree-attention verify path; tree-causal mask is a merged 0 ms dead-end (#33); the PR's `--local-only/--profile-tree-acceptance/--sequoia-tree` flags don't exist → pivoted to the CPU cost-model form the Notes anticipated.

| topology (matched budget) | E[T] @ M=8 | E[T] @ M=45 | max E[T] gain | TPS-opt budget n\* | TPS @ n\* (cm scale) |
|---|--:|--:|--:|--:|--:|
| linear (deployed family) | 2.976 | 3.117 | — | 16 | 235.7 |
| balanced-W4 (prior model) | 2.430 | 3.178 | DP/bal **1.433** | 31 | 216.7 |
| **Sequoia DP** | **3.019** | **4.132** | DP/lin **1.341** | **32** (M=33 Marlin cliff) | **275.2** |

- **Analysis:** DP tree is genuinely the better topology on our distribution (+43% E[T] vs balanced-W4, +16% TPS vs linear, decay-robust 13–17%; brute-force-validated n≤7, 200k-MC `F==E[committed]`). **But deployable gain = 0** — no tree-verify path exists in vLLM 0.22 and #33 predicts ~0-saving on the dense path. The PR's ≥432-local-TPS target is unmeetable by this route. **Lane closed analytically** (like the tree-mask). **Secondary (load-bearing):** the salvage-spine E in `tree_acceptance_model.py` (#26) is an **upper bound** — it scores 0.86-rate compounding to depth K with only K·W+1 nodes (true 0.86-compounding needs ~W^K branching). Over-count **+45% at M=45** (5.99 → achievable 4.13 → ~248 TPS, *below* the linear frontier) ⇒ **strengthens "ship linear; trees don't reach 500"** (#33/#37). wirbel did NOT auto-edit #26 (flagged + offered a 1-line tightening).
- **W&B:** `bvbg81v4` (group `sequoia-dp-tree`; CPU-only, <0.2 GB, ~30 s, no GPU/vLLM/HF-Job). Ships `scripts/profiler/sequoia_dp_tree.py` + `research/spec_cost_model/{sequoia_dp_results.json,report_sequoia_dp.md}`.
- **Follow-ups:** (a) **tree-ceiling tightening QUEUED** — replace salvage-spine E with achievable path-product DP in `tree_acceptance_model.py`, held until denken #51 lands (concurrent-edit on the same tool). (b) premise correction (linear MTP, not M=45 tree) **propagated to denken #51**. (c) wirbel → next slot (post-split-KV decode re-profile).

---

## 2026-06-13 21:22 — PR #50: official_gate wired into HF-launch preflight (fail-closed) ✓ MERGED (launch-safety infra keeper, official bar UNCHANGED)

- **Branch:** `lawine/official-gate-hf-launch-wire` · **Student:** lawine
- **Status:** MERGED as a launch-safety infra keeper. Official TPS bar **UNCHANGED at 126.378** (primary_metric `official_gate_wired=1`, not throughput).
- **Hypothesis:** the #45 `official_gate` verdict (PPL ≤ 2.42 AND completed == 128 AND all_modalities_loaded) should be the **fail-closed interlock** on the HF-launch path, so a quota-spending submission can never launch on a FAIL/INCOMPLETE gate, and an 8-prompt smoke can never authorize a 128-prompt run. This is the safety gate for the Issue #46 split-KV launch.

| check | behavior | verdict |
|---|---|---|
| gate FAIL | blocks HF launch | fail-closed ✓ |
| gate INCOMPLETE | blocks HF launch | fail-closed ✓ |
| 8-prompt smoke → 128-run | refused (n_prompts mismatch) | partial cannot certify full ✓ |
| image+text / video | functional probe (served) | loaded ✓ |
| audio | presence + non-zero fallback (no `vllm[audio]`/`av` locally) | decision (A) ratified ✓ |
| fa2sw smoke (8 prompts) | PPL 2.3767 bit-identical to #45 | no serve-path change ✓ |

- **Analysis:** closes the launch-safety lane opened by #45. The gate now **refuses to certify a full run from a partial sample** (carries `n_prompts`), so no quota is spent on an unproven 128-run. Audio honesty decision **(A)** ratified: presence+non-zero is correct policy — a functional-mandatory audio check would mislabel a *local-tooling* gap (`vllm[audio]`/`av` unavailable) as a *submission* defect. `make_probe_inputs.py` + `probe_inputs/{probe_audio.wav,probe_video.mp4}` staged for future functional audio. 51/51 tests (+launch-block truth table, partial-sample refusal, video probe). This is the interlock for the #46-approved one-shot split-KV launch.
- **W&B:** `bi3tqtv3` (local infra; nothing trained).
- **Follow-up → lawine #52:** run full 128-prompt `official_gate` validation on `fa2sw_precache_kenyan`, then execute the (Issue #46 human-approved) one-shot HF launch of the split-KV submission — gated on this PR's PASS verdict.

---

## 2026-06-13 20:09 — PR #23: int4 spec-verify greedy flip-rate probe ✓ MERGED (characterization keeper, official bar UNCHANGED)

- **Branch:** `stark/linchpin-fp32-accum-flip-probe` · **Student:** stark
- **Status:** MERGED as a characterization keeper. Official TPS bar **UNCHANGED at 126.378** (primary_metric is `flip_rate_per_token`, not throughput).
- **Hypothesis:** the int4-Marlin M=K+1 batched-verify vs M=1 greedy divergence is caused by batch-dependent fp16/bf16 reduction order; cheap fixes — (a) fp32 logit accumulation, (c) deterministic reduction — might zero the per-token argmax flips without a full batch-invariant kernel rewrite.

| config | flip_rate/tok (M=2..8) | latency overhead | verdict |
|---|--:|--:|---|
| baseline | 0.00521 (3/576) | 0% | — |
| fp32-logit | 0.00174 (1/576) | **+0.2%** | reshuffle, not a fix |
| deterministic | 0.00521 (3/576) | **+14.0%** | proven no-op |
| fp32+det | 0.00174 (1/576) | +14.7% | no |
| cross-process M=1 noise floor | **0/576** | — | flips are genuine batch effect |

- **Analysis:** decisive NEGATIVE — no config reaches flip_rate=0. The **7:268 existence proof** (faithful fp32 logits disagree M=1 vs M≥2) localizes the irreducible source to the **decoder Marlin int4 GEMM** (the hidden state feeding lm_head is batch-variant), NOT the logit-accumulation step — answering the hypothesis split. Two keepers: deterministic mode is strictly bad (no-op + 14%), and the flip is **binary M=1-vs-M≥2, flat in K** (longer drafts no worse for greedy-identity). Per #38 the official gate has no token-identity check, so this is most valuable as a **run-to-run reproducibility** diagnostic for the private re-run gate. Ships `scripts/profiler/verify_greedy_flip_probe.py` as a drop-in batch-invariance validator.
- **W&B:** `zd121euo` (group `verify-greedy-flip-probe`; flip rates verified to 7 sig figs).
- **Follow-up → stark next:** lane pivot (linchpin closed; greedy-identity is not the leaderboard gate) — see CURRENT_RESEARCH_STATE for the new assignment.

---

## 2026-06-13 20:08 — PR #44: Local private-stability probe (public→private TPS-gap predictor) ✓ MERGED (validity keeper, official bar UNCHANGED)

- **Branch:** `kanna/local-private-gap-probe` · **Student:** kanna
- **Status:** MERGED as a validity keeper. Official TPS bar **UNCHANGED at 126.378** (primary_metric is `public_to_private_gap_pct`).
- **Hypothesis:** the binding constraint above ~286 TPS is the private-set re-run (honest drafter stacks lose 4–9% TPS and die on the 5% repro rule). We can **predict the public→private TPS gap locally**, pre-submission, by measuring single-stream TPS + drafter acceptance on a distribution-shifted private-proxy set vs the 128 public prompts.

| scenario | precache | bench set | TPS | E_accept | PPL | completed |
|---|---|---|--:|--:|--:|---|
| leaderboard | public | public | **423.63** | 4.061 | 2.377 | 128/128 |
| public_cold | off | public | 418.37 | 4.089 | — | 128/128 |
| private_rerun | off | private | **370.96** | 3.565 | 2.377 | 128/128 |

- **Analysis:** reproduces the published VALID frontier (423.63 vs kenyan-duma 421.12; PPL 2.377 exact) ⇒ the measured ratio is trustworthy. Headline **public→private gap = 12.43%** ⇒ WOULD-FAIL (>5% → INVALID). Decomposition: distribution gap **11.33%** (drafter-acceptance collapse on chat, E_accept 4.06→3.57) + precache **1.24%**; acceptance ratio (0.872) fully accounts for TPS ratio (0.887) ⇒ the gap **is the drafter on chat**. Honest caveat: pure-ShareGPT proxy is likely harder than the real private set, so 12.4% is an upper-ish *pessimistic* early-warning (safe direction; no false-negative — firfir-cast known-7.2%-invalid also reads >5%). Ships `scripts/validity/private_gap_probe.py` + `build_private_proxy.py`.
- **W&B:** `jgxdnmwz` (values match exactly; group tag `private-gap-probe`, artifact `private_gap_report`).
- **Follow-up → kanna next:** calibrate the proxy against firfir-cast's known 7.2% (→ quantitative predictor) + rank the VALID frontier stacks by private-re-run risk; feeds the official frontier-submission go/no-go.

---

## 2026-06-13 19:20 — PR #42: `--spec-off` one-flag contract + validator N-mismatch legibility ✓ MERGED (infra keeper, official bar UNCHANGED)

- **Branch:** `lawine/specoff-contract` · **Student:** lawine
- **Status:** MERGED as a validity-infra keeper. Official TPS bar **UNCHANGED at 126.378** (`primary_metric=1` is a boolean "the flag works", not a throughput).
- **Hypothesis:** PR #40 exposed a footgun — `--spec-off` was a silent no-op for any spec stack whose `serve.py` ignores `SENPAI_REFERENCE_MODE`, so a "spec-off reference" was secretly captured with the drafter still on. Fix at the root: teach spec stacks to clear `SPECULATIVE_CONFIG` under the reference-mode env.

| deliverable | result | verified |
|---|---|---|
| `specoff_flag_works_for_mtp_drafter` | **1** | on-GPU serve: `speculative_config=None`, `reference_kind=served_spec_off` |
| spec stacks fixed | **3/3** (fa2sw, lf29cap444, int4_mtp_batchinv) | argv-intercept proof |
| leaderboard serve path untouched | **provably** (env falsy → helpers no-op → drafter config verbatim) | unit tests + argv proof |
| `n_mismatch_warning_added` | **1** (`reference_n_mismatch` + actionable warning) | — |
| tests | **14/14** (+6 new) | CPU-only |

- **Analysis:** retires the fragile per-submission `--ref-env SPECULATIVE_CONFIG=` workaround to a fallback; `--spec-off` is now the canonical one-flag path for every spec stack's pre-launch greedy reference. Two good judgment calls banked: (1) caught that `int4_g128_lmhead` is **pure-AR, not spec** (my assignment mislabeled it) → applied the fix to the real third spec stack `int4_mtp_batchinv` (token-count knob → `num_speculative_tokens=0`); (2) used a **truthy** env check matching `paths.REFERENCE_MODE_ENV="1"` rather than the literal `=="reference"` in my pseudocode (which would have been a silent no-op).
- **W&B:** none (local infra; nothing trained).
- **Follow-up → lawine #45:** local **official-gate preflight** (modalities-load check + consolidated PPL+completion+modalities verdict, separated from the internal greedy bar), bundling the canonical fa2sw-reference `--spec-off` regen.

---

## 2026-06-13 19:57 — PR #41: Eliminate scatter floor in `compute_logits` ✓ MERGED (characterization + deployable-infra keeper, official bar UNCHANGED)

- **Branch:** `denken/scatter-floor-elim` · **Student:** denken
- **Status:** MERGED at `6bfa448` after a clean Step-4 W&B reconciliation. Official TPS bar **UNCHANGED at 126.378** — the 538–546 figures are LOCAL cost-model ceilings at the K\*=11/M=45 operating point, not HF-validated throughput.
- **Hypothesis:** the `lmhead12k` plugin scatters 12k partial logits to a full [M,262144] −inf tensor before argmax (0.348 ms @ M=45). If the greedy-gate guarantee holds, `kept_ids[argmax(partial)]` is identical in one step → ~538→546 TPS local ceiling.
- **Reconciliation (the first-submission mismatch, now fixed):** I sent the first submission back because its Step-4 table (538/540/544) sat ~60 TPS above the cited runs (which logged K=6→480/477, `>500=False`) and the 538.15 control was absent. denken correctly root-caused it as a **logging bug in `tree_acceptance_model.py`**: it wrote `verdict_tps_ceiling_tree_at_full_scale`/`tps_tree_meas_p0_780` at the fixed `--sim-K` headline (default 6 → M=25), **not** the argmax K\*=11/M=45 operating point. PR #37 had surfaced K\* via a `kstar_p078_W4_tps_withdrafter` field that was never in the committed script. denken restored that field **additively** and re-ran all curves at `--sim-K 11`.

| deliverable | result | independent W&B verification (re-run, this cycle) |
|---|--:|---|
| Step 1 scatter-equivalence (primary) | `equiv_rate=1.0` | `gy05konp`: 1.0 (249,858/249,858) — **universal**, ascending `kept_ids` |
| Step 3 microbench @ M=45 | scatter 0.348 / persistent 0.299 ms | `wa72elyq`: 0.348 / 0.299 |
| Step 4 scatter control (PR #37 repro) | **538.15** | `x0gjax5p`: 538.1452 @ sim_K=11, K\*=11/M=45, `>500=True` ✅ |
| Step 4 persistent buffer (**deployable, +1.95**) | **540.10** | `m316ma9u`: 540.1009 @ sim_K=11, K\*=11/M=45, `>500=True` ✅ |
| Step 4 scatter-free remap | **544.22** | `g9h5rqv9`: 544.2240 @ sim_K=11, K\*=11/M=45, `>500=True` ✅ |
| Step 4 analytic gemm-floor | **545.82** | `z2k86aiu`: 545.8159 @ sim_K=11, K\*=11/M=45, `>500=True` ✅ |

- **Analysis:** two durable wins. (1) **Characterization:** the scatter is **unconditionally** redundant — ascending `kept_ids` ⟹ `argmax(scatter(partial)) ≡ kept_ids[argmax(partial)]` for *all* inputs, so it generalizes to the private set (no acceptance dependence). (2) **Deployable:** a **bit-identical persistent −inf buffer** in the `lmhead12k` plugin (26/26 `check_scatter_buffer_identity.py`) that removes the 0.348 ms per-step scatter alloc for a clean **+1.95 TPS** at the operating point (`m316ma9u` 540.10 vs `x0gjax5p` 538.15 control). The additive `kstar_p078_*` logging fix to `tree_acceptance_model.py` also makes every future cost-model run report its argmax operating point, not just the `--sim-K` headline — closes the exact reporting hole that caused the first-submission confusion.
- **W&B:** `gy05konp`, `wa72elyq`, `x0gjax5p`, `m316ma9u`, `g9h5rqv9`, `z2k86aiu` (all local cost-model/microbench; nothing trained).
- **Follow-up → denken next:** dynamic-K (`accepthist`) cost-model projection on top of the now-correct static K\*=11 logging + `--sim-K` argmax-default cleanup so the headline field defaults to the operating point.

---

## 2026-06-13 18:58 — PR #9: Wide-distribution KL-distilled drafter for private-stable acceptance — REQUEST-CHANGES (negative result + key methodological finding)

- **Branch:** `land/wide-drafter-distill` · **Student:** land
- **Status:** NOT MERGED (native regressed). Request-changes → rebase (`heldout.jsonl` conflict) + pivot to HASS serve-faithful objective. **High-value negative result.**
- **Hypothesis:** Above ~286 TPS the binding constraint is drafter acceptance, and the binding *risk* is the private-set re-run (drafters fit to the 128 public prompts lose 4–9% TPS and die on the 5% repro rule). A drafter KL-distilled on a wide, distribution-matched corpus should lift acceptance AND make it private-stable.

### Results (W&B run `land-freerun-v1b-171224`, project gemma-challenge-senpai, group wide-drafter-freerun)

| metric | stock | v0 (teacher-forced) | v1b (free-running) | Δ v1b vs stock |
|---|--:|--:|--:|--:|
| offline tf gate (accepted tok/step, K=7) | 3.455 | 3.811 (+10%) | **4.004** | **+15.9%** |
| **native accept/step (HF assisted-gen)** | **3.553** | 3.388 (−5%) | **3.341** | **−6.0%** |
| greedy identity (bf16 harness artifact) | 14/24 | — | 13/24 | — |
| peak mem train / eval-load | — | — | 17.4 / ~16 GB | A10G 23 GB fits |

- Full budget: 1030 steps, 220,746 positions, 3.4 epochs, 82 min (whole cap), LR cosine-decay-by-time, free_run_frac 0.895, diverge_frac 0.285.

### Analysis / conclusion

- **Problem #1 (v1a native collapse to 1.49) FIXED** by greedy-trajectory corpus + rejection-aware break (v1b native healthy 3.34, diverge_frac 0.285).
- **Problem #2 (the real one):** tf and native are **anti-correlated** under our training. Two independent schedules (v0 tf, v1b free-run) move tf +10/+16% while native lands at ~3.34–3.39. Signature of optimizing a divergent proxy, and **rules out exposure bias** (free-run directly targets it).
- **Mechanism (evidence-backed):** our objective + tf proxy condition the draft's step-0 hidden on the target's ground-truth hidden (fresh target prefill per position). HF native assisted-generation does NOT — the assistant runs its own forward over accumulated KV across verify rounds. Fine-tuning the draft to excel on the target's *true* hidden drifts it off the joint optimum the serving path feeds it; the un-fine-tuned stock draft sits ON that optimum (3.553).
- **Programme conclusion:** the offline tf gate (incl. `offline_acceptance.py`) is NOT a faithful proxy for native acceptance for this EAGLE drafter. Drafter work must be gated on native (or an interface-faithful objective). Propagated to fern #34 (native cross-check requested) and CURRENT_RESEARCH_STATE.
- **Next:** HASS-style serve-faithful training (feed the draft its own running hidden over accumulated KV), gate/select on `heldout_native_accept_per_step`. land sent back to implement on the same PR.

---

## 2026-06-13 — PR #39: fa2sw attention deep-profile ✓ MERGED — Triton verify occupancy-bound, 3D split-KV lever identified

- **Branch:** `wirbel/fa2sw-attn-profile` · **Student:** wirbel
- **Status:** MERGED — **high-value lever discovery.** LOCAL A10G op-microbench; no W&B (wandb_run_ids:[]). Rewrites the #30 lever map for verify attention.
- **Hypothesis:** fa2sw sliding-window attention (19.6% of decode cycle from #30) might be near-optimal or might have exploitable inefficiency (KV layout, SWA masking, bandwidth ratio vs theoretical minimum).

### Results

| metric | value | verdict |
|---|--:|---|
| **`fa2sw_bandwidth_efficiency_fraction`** | **0.0473** (4.7%) | 21× below 80% near-optimal threshold ✓ |
| **`verdict_attn_reduction_worth_pursuing`** | **1** | YES — implement 3D split-KV |
| measured split-KV speedup (M=1, identical work) | **4.14×** (sliding 4.36×, full 3.91×) | direct measurement |
| reachable attention saving | 50% (conservative 2×) … 82% (3D BW) | |
| TPS projection @ 50% saving | **~471** | crosses 440, 460 |
| TPS projection @ 82% saving | **~505** | crosses 460, 500 |
| `kernel_unified_attention` share of attention | 98.1% | Triton, NOT fa2sw FA2 |
| device time M=7→45 | ~53 µs flat | occupancy/launch-bound, not compute |
| KV bandwidth floor | 41.84 MB/cycle, 0.087 ms | served = 1.836 ms (21× above) |

### Key findings

1. **Premise refuted: the fa2sw FA2 path is inert.** vLLM forces `TRITON_ATTN` for heterogeneous head dims (sliding 256, full 512); FA2 caps at head_dim 256. The 19.6% is 98.1% Triton `kernel_unified_attention`. The PR #30 naming "fa2sw kernel" was wrong at the kernel level.

2. **Root cause: M=8 verify falls on 2D Triton path (occupancy-bound).** The `unified_attention` gates 3D split-KV (FlashDecoding) OFF for `max_seqlen_q > 1`. The spec-verify runs M=K+1=8 query rows → always lands on 2D (~6 CTAs / 80 SMs). The M=1 drafter uses 3D and runs 4.14× faster on identical work. Device time is FLAT M=7→45 → confirmed occupancy/launch bound.

3. **4.14× is a direct measurement.** 2D vs 3D at M=1, identical bytes/softmax: sliding 4.36×, full 3.91×. The 3D kernel EXISTS in vLLM; only the dispatch guard needs patching.

4. **The served Triton kernel is already optimal for M=1** (12.2 µs vs FA2 paged 58.2 µs vs SDPA 97.9 µs). The problem is purely the M>1 dispatch guard.

5. **Fix is greedy-exact** (split-KV is bit-identical attention). Zero gate risk. Orthogonal to spec-decode validity question.

6. **Implementation path:** patch `max_seqlen_q > 1` guard in `vllm/v1/attention/ops/triton_unified_attention.py` + extend per-segment softmax reduction to multiple query rows. ~90% already in vLLM.

7. **Methodology correction:** physical KV-load byte model (what FlashAttention streams) is the correct BW model, NOT `window×seq×heads` (double-counts attention matrix as bytes). Noted for future profiling.

### Conclusions

This is the single highest-leverage greedy-safe lever in the programme. Unlike spec-decode velocity (gated on batch-invariance / served-gate), the 3D split-KV fix is valid on the EXISTING honest frontier (already leaderboard-valid at ~424.5 TPS) and projects ~471–505 TPS. wirbel reassigned to implement the fix.

## 2026-06-13 — PR #40: Greedy-ref infra: 128-prompt fa2sw reference + bare-tag assertion ✓ MERGED

- **Branch:** `lawine/greedy-ref-128prompt` · **Student:** lawine
- **Status:** MERGED — **validity-infrastructure closure.** LOCAL INFRA ONLY; no HF job, no submission. Delivers the two follow-up items from PR #32; unblocks kanna #38's full 128-prompt served-gate audit.
- **Hypothesis:** PR #32 fixed reference keying but only had a 32-prompt reference. kanna #38's served-gate audit needs the full 128-prompt served spec-off reference and the bare-tag collision class needs a runtime assertion to prevent regression.

### Results

| metric | value | verdict |
|---|--:|---|
| `fa2sw_reference_128prompt_complete` | **128** | full reference ✓ |
| `bare_tag_assertion_added` | **1** | assertion hardened ✓ |
| `reference_self_consistent` | **1** | deterministic at batch=1 ✓ |
| Tests (CPU-only) | **8/8 pass** | 6 prior + 2 new ✓ |
| Wall-clock (cold-start + 128 decodes) | **514.75s** (~14 min) | within budget ✓ |
| Reference key format | `…/submissions/fa2sw_precache_kenyan::google/gemma-4-E4B-it` | `<dir>::<model_id>` ✓ |

### Analysis & conclusions

1. **128-prompt reference is the primary deliverable for kanna #38.** `validate_submission --submission fa2sw_precache_kenyan --num-prompts 128` now auto-resolves without manual path threading. The reference at `research/greedy_reference/workspace__senpai__target__submissions__fa2sw_precache_kenyan__google__gemma-4-E4B-it/` supersedes #32's 32-prompt version.

2. **Justified deviation on drafter disable: critical institutional knowledge.** `fa2sw_precache_kenyan` uses `SPECULATIVE_CONFIG={method:mtp,...}` and `serve.py` does NOT honor `SENPAI_REFERENCE_MODE`. The `--spec-off` flag would have been a silent no-op, producing an invalid reference with speculation ON. Correct method: `--ref-env SPECULATIVE_CONFIG=` (same as #32). `reference_kind=served_spec_off` confirmed via meta. **Every future spec submission that doesn't honor `SENPAI_REFERENCE_MODE` needs this `--ref-env` flag — should teach `serve.py` to honor it (follow-up item).**

3. **Self-consistency (1/1):** bit-identical output from two separate processes on 16 prompts confirms the int4 + CUDA-graph stack is deterministic at batch=1 served. This is expected but now empirically confirmed.

4. **Bare-tag assertion:** `harness.assert_submission_reference_tag(ref_tag)` placed at both generator and validator sites (lockstep). Smart adaptation: real function is 1-arg, takes already-resolved tag. Bare-baseline branch (pure model-id key) intentionally NOT guarded — correct design.

5. **Wall-clock fast:** 514.75s total (~14 min) vs the feared 2+ hours. The reasoning decodes ran faster than worst-case; 128 × 512-token completions total.

## 2026-06-13 — PR #37: lmhead12k verify-forward cost model + tile-corrected canonical curve ✓ MERGED

- **Branch:** `denken/lmhead12k-verify-cost` · **Student:** denken
- **Status:** MERGED — **cost-model closure + infra (tile-fold).** LOCAL profiling only; no HF job, no submission. Establishes the lmhead12k ceiling on the spec-verify path via directly-measured pod latencies.
- **Hypothesis:** Ubel #14's lmhead12k prune removes ~3 ms from the AR lm_head (PR #30: 1% of decode). Does it also remove a comparable fraction from the *verify* lm_head? The verify head runs on M=K+1=45 tokens simultaneously — if the head is memory-bandwidth-bound there too, the savings may be larger and flip PR #33's ">500 @ p=0.78 = NO" verdict.

### Results

| quantity (canonical = graph, ctx256) | full head (#33) | lmhead12k (measured) | analytic ceiling |
|---|--:|--:|--:|
| lm_head verify cost @ M=45 | 3.367 ms | **0.348 ms** (scatter floor) | 0.158 ms (×0.0469) |
| V_tree step @ M=45 | 15.235 ms | **12.212 ms** (−3.02 ms, −19.8%) | 12.022 ms |
| tree K* @ p=0.78 w/ drafter | K11/M45: 440.4 | **K11/M45: 538.1** | K11/M45: 545.8 |
| tree K* @ p=0.78 verify-only | K11/M45: 480.8 | **K11/M45: 599.8** | K11/M45: 609.4 |
| tree K* @ p=0.6792 w/ drafter | K11/M45: 359.9 | K7/M29: 446.6 (<500) | K7/M29: 451.7 |
| >500 @ p=0.78, K*-optimum, w/ drafter? | **NO** (440.4) | **YES (538.1)** | YES (545.8) |
| `primary_metric` `tree_tps_ceiling_p078_lmhead12k` | — | **538.1** | — |
| `test_metric` `verdict_exceeds_500_at_p078_lmhead12k` | — | **1** | — |

**W&B (verified by direct query):**

| run | name | key scalar | value | W&B |
|---|---|---|---|---|
| `klvpfk7g` | lmhead12k-verify-derive-measure | `V_full_M45`, `meas_k12_scatter_M45`, `lmhead_fixed_share_at_M45` | 15.235 ms, 0.348 ms, 0.860 | finished ✓ |
| `ruch259z` | lmhead12k-tree-ceiling-measured | `kstar_p078_W4_tps_withdrafter`, `verdict_exceeds_500` | **538.150, True** | finished ✓ |
| `6c9r3lih` | lmhead12k-tree-ceiling-analytic | `kstar_p078_W4_tps_withdrafter` | 545.816 | finished ✓ |

Group `spec-verify-lmhead12k` in `gemma-challenge-senpai`. Minor cosmetic gap: `V_lmhead12k_M45` logged 12.022 (analytic) vs PR table's 12.212 (measured) — label swap; does not touch the verified 538.1 headline (logged independently in `ruch259z`).

### Analysis & conclusions

1. **The verify-head prune is real and bounded.** Pruning to 12k rows removes ~3.0 ms from V_tree @ M=45 (−19.8%), because the verify forward streams the full bf16 head for each of the M=45 tokens in the speculative proposal. The saving is ~flat in absolute ms across M (it's a fixed head-weight bandwidth term), so its *fractional* contribution falls with M.

2. **The scatter floor is the correct honest ceiling.** The production `compute_logits` path scatters 12k partial logits back to a full [M,262144] −inf tensor + argmaxes over the full vocab for greedy-identity correctness (cannot be removed without a kernel rewrite). This costs 0.348 ms @ M=45 = ~2.2× the bare GEMM. Measured ceiling 538.1, not over-claimed analytic 546.

3. **Two-lens honest >500 reporting:** K*-optimum (538.1, >500 ✓ — matches #33's baseline frame, the headline lens) vs conservative fixed-K=6 with-drafter (476.5, <500 ✗). The flip needs p≥0.78 AND the K*-optimum lens. At realistic p=0.6792 with-drafter optimum stays <500 (446.6). Both lenses W&B-logged.

4. **Pipeline validated:** baseline column reproduces #33's K=11/M=45 440/481 @ p=0.78 exactly. Reduced curve trustworthy.

5. **K\*=11/M=45 serving guidance LOCKED for kanna #24 / any future spec submission.** PR #33's `optimal_k=15` scalars were the linear-W=1 lens artifact in run `36hkaj14` — corrected here (Step 5). Realistic W=4 tree optimum is K=11 (M=45) at both p=0.6792 and p=0.78.

6. **Infra: tile-fold into canonical msweep.** `fold_tile_into_msweep.py` folds #33's measured Marlin cliffs into `results_msweep.json` in place (pre-fold provenance at `results_msweep_prefold.json`). #26/#28 consumers now inherit the correct non-linear curve automatically.

7. **Suggested follow-ups from denken:** (a) eliminate the scatter floor (kernel argmax over 12k partial + remap full-vocab id — correctness proof needed, ~546 vs 538 ceiling); (b) tile-correct `eager/*` and `*/ctx512` keys (only `graph|ctx256` carries measured cliffs now); (c) validate ceiling against a real end-to-end spec-decode serving run.

## 2026-06-13 18:xx — PR #32: Greedy-gate reference-keying fix ✓ MERGED — validity-infrastructure correction

- **Branch:** `lawine/greedy-gate-ref-keying-fix` · **Student:** lawine
- **Status:** MERGED as a **validity-infrastructure fix**, NOT a TPS change. Served decode path byte-for-byte unchanged. CPU-only, no W&B.
- **Hypothesis:** The greedy-reference cache is keyed on `model_id` alone — two submissions sharing the same base checkpoint collide on a single cached reference, potentially causing silent false-PASS / false-FAIL on the greedy-identity gate.

### Results

| metric | value | verdict |
|---|--:|---|
| `collision_free` | **1.0** | collision hole CLOSED ✓ |
| `distinct_tags` | **2** | two submissions → two references ✓ |
| test guards (CPU-only, 6 assertions) | **6/6 pass** | correctness confirmed ✓ |
| fa2sw_precache_kenyan vs own M=1 AR (32 prompts, correct keying) | **DIVERGENT 27/32** | out-of-scope finding; routes to kanna |

### Analysis & conclusions

- **Root cause fixed:** reference cache keyed on `model_id` alone → submissions sharing a base model collide. Fixed by keying on `<submission_dir>::<model_id>` and threading a separate `reference_model_id` through `harness.py` / `gen_greedy_reference.py` / `validate_submission.py`. Audit trail: the resolved tag is now recorded.
- **`distinct_tags=2` confirms the old collision was real** — previously both submissions resolved to the same reference, rendering the greedy-gate meaningless for same-base-model submissions.
- **Keeper finding (routes to kanna):** under correct per-submission keying, `fa2sw_precache_kenyan` is **DIVERGENT 27/32** against its own M=1 AR reference. This is the data point kanna's served-gate validity audit must reconcile: the stack is leaderboard-valid at ~424.5 TPS but fails our strict M=1 bar — strong evidence our bar is over-conservative vs the leaderboard's served gate.
- **Unit-tested at the boundary:** `scripts/tests/test_greedy_ref_keying.py` (6 CPU-only guards: collision-free keying, distinct tags, key format). Correct test strategy for a correctness-of-validation change.
- **Next:** lawine reassigned — regenerate fa2sw_precache_kenyan reference at full 128 prompts + add runtime assert that resolved reference tag is never bare `"model"`. kanna → served-gate validity audit using the now-trustworthy keying.

---

## 2026-06-13 18:xx — PR #30: Frontier decode composition profile ✓ MERGED — authoritative component breakdown of ~420 TPS stack

- **Branch:** `wirbel/frontier-decode-profile` · **Student:** wirbel
- **Status:** MERGED as a **frontier decode characterization artifact**, NOT a TPS improvement. On-device component-resolved profile of `fa2sw_precache_kenyan` decode loop — the most strategically clarifying measurement of the cycle.
- **Hypothesis:** Decompose the decode cycle of the ~420 frontier (`fa2sw_precache_kenyan`) into GPU-time fractions by component (int4 body GEMM, sliding-window attention, drafter, lm_head) to rank remaining addressable levers and set priorities.

### Results

| component | fraction of decode cycle | verdict / implication |
|---|--:|---|
| Total GPU-bound | **99.3%** | host/launch overhead already negligible |
| **Verify-body int4 GEMM** | **53.2%** | dominant cost; walled at int4-Marlin floor |
| **fa2sw sliding-window attention** | **19.6%** | **second lever — most addressable** |
| Drafter | **15.5%** | third lever (drafter quality / steps) |
| lm_head | **1.0%** | collapsed from ~26.4% — validates lmhead12k (#14) ✓ |
| Verify bandwidth-bound / flat-in-M | M=1→8: **+25%** | tree widening nearly free on verify; K* set by acceptance geometry |
| E_accept | **3.817 tok/cycle** | current drafter acceptance at frontier |

W&B: `07kg6bn7` (authoritative, group `frontier-decode-profile`). `og7z6w0c` superseded.

### Analysis & conclusions

- **The decode loop is 99.3% GPU-bound.** Every remaining TPS gain must come from bytes-moved or FLOPs-cut inside kernels. This kills the "optimize launch/Python overhead" hypothesis for the frontier stack.
- **Verify-body GEMM (53.2%) is walled at the int4-Marlin floor.** There is no cheaper exact int4 matmul in vLLM 0.22.0. This eliminates the "find a faster verify GEMM" direction without a major kernel rewrite.
- **fa2sw attention (19.6%) is the live second lever.** It's large enough to matter (~100 TPS headroom if fully eliminated) and it's a kernel-addressable path (KV layout, SWA masking efficiency). This is where wirbel's next investigation goes.
- **lm_head collapsed to 1.0%** — independent validation that lmhead12k's 21.3× row-cut lands on the decode path, corroborating ubel #14 and wirbel #8. The lm_head lever is fully exploited.
- **Verify is bandwidth-bound / flat-in-M** — widening the tree is cheap on the verify side; the K* ceiling is set by acceptance geometry (acceptance rate p), not by verify cost per token. This corroborates PR #28/#33 cost-model findings and confirms the drafter quality (p) lever is the path to >500 TPS.
- **Cross-path validation:** `fa2sw_precache_kenyan` is the same stack lawine #32 used as the "out-of-scope" divergence case — now feeding directly into kanna's served-gate audit.
- **Next:** wirbel → fa2sw attention kernel-level deep-profile (19.6% second lever). kanna → served-gate validity audit using the #32-corrected keying. Artifacts: `research/profiling/frontier_decode/`, `scripts/local_validation/profile_decode.py`.

---

## 2026-06-13 17:52 — PR #24: Verify-rollback gate ✓ MERGED — THE LINCHPIN's final closure (greedy-valid spec-decode-for-speed is DEAD in vLLM 0.22.0)

- **Branch:** `kanna/verify-rollback-gate` · **Student:** kanna
- **Status:** MERGED as the **verify-rollback lane closure** (research artifact completing the #19→#24 arc), NOT a TPS baseline change. Official headline stays PR #4 (126.378).
- **Hypothesis:** Verify-rollback (per-step re-verify of accepted spec tokens under an M=1 AR forward; commit on match, rollback on mismatch) can restore greedy-valid spec decode **AND** maintain net-positive TPS over int4 AR — the only remaining greedy-valid-spec route after PR #19 closed the invariant-kernel lane.

### Results — hypothesis HALF-confirmed; the failing half is provably unfixable

| metric (eager n=32, W&B `ibmlc871`) | value | verdict |
|---|--:|---|
| flip_rate/tok, **verify-rollback** (vr vs M=1 ref) | **0.0** (`GREEDY_IDENTICAL` 32/32, 0/16384 divergent) | identity RESTORED ✓ |
| flip_rate/tok, raw spec (cand vs ref) | 0.332% | matches PR #19's 0.376% (CIs overlap) |
| rollback_rate/spec step (K=6) | 1.98% | matches ~2.2% theory |
| TPS int4 AR (spec-off) | 22.46 | the floor VR must beat |
| TPS int4 spec K=6 (raw, greedy-INVALID) | 49.75 | fast but fails the gate |
| **TPS verify-rollback (composed)** | **15.48 (0.69× AR)** | net-NEGATIVE ✗ |

Cudagraph n=16 (`354tydww`): VR flip 0.0 (16/16), AR 93.24, spec 229.71, **VR 66.32 (0.71× AR)** — also net-negative, far below the 126.378 official AR floor. All W&B arms verified to 4 sig-figs (no NaN); `tps_vr_composed` is transparently a derived field = 1/(1/AR+1/spec).

### Analysis & conclusions

- **The cost theorem (the keeper).** Net-positive TPS is impossible *by construction*, not by tuning: **you cannot know which 2.2% of steps roll back without computing the M=1 reference for ALL of them** — detecting a flip *is* running the M=1 forward (= one AR step). So re-verifying the j tokens a spec step accepts runs j sequential M=1 forwards = identical to the j forwards AR would run anyway. `TPS_VR = 1/(1/TPS_AR + 1/TPS_spec) < TPS_AR`, exact, implementation-independent. The PR's "extra M=1 only on the 2.2% that roll back" undercounted the re-verify work ~45× (re-verify rate is 100% of tokens). **Per-token M=1 → identity ✓ speed ✗; batched M=K → speed ✓ identity ✗ (M=K≠M=1 reintroduces the flips); no third option in a non-batch-invariant stack.**
- **Methodology accepted — composition, not a live engine.** Realized by composition (deliberate, disclosed): output identity is *definitional* (per-token rollback emits the M=1 AR argmax at every position → VR stream = M=1 AR stream bit-for-bit, confirmed on the real stream); cost is a *theorem* (both TPS arms real wall-clock; only the interleave composed). The PR's `spec_decode_worker.py` hook is a vLLM v0 path absent in 0.22.0 (v1 accept is in `rejection_sampler.py`/`gpu_model_runner._sample`); a live inline engine would burn GPU-days to reproduce a provable verdict. Advisor endorsed NOT building it.
- **Paper-premise correction (keeper).** arxiv 2601.17768 ("LLM-42", Gond et al.) targets **batch-self-consistency** (fixed-shape 256-wide re-verify; Obs. O3 relaxes to "position-consistent across runs"), **not** M=1-greedy-identity — greedy-DIVERGENT against our served reference if applied verbatim. Closes the "just implement the determinism paper" expectation.
- **Strategic consequence.** #19 closed the invariant-kernel route, #24 closes the rollback route → **spec-decode-for-speed under a strict M=1-greedy-identity gate is DEAD in vLLM 0.22.0.** The only net-positive greedy-valid-drafter route left is **source-level batch-invariance of the M=K+1 verify forward** (kanna follow-up #2 = stark #23; would make spec valid with ZERO rollback, strictly dominating VR). kanna follow-up #1 (is the ~420 frontier greedy-valid under the *served* gate without spec — is our strict M=1 bar stricter than the leaderboard enforces?) is the other open thread (feeds off wirbel #30).
- **Next:** kanna reassigned (verify-rollback lane closed); routed per the #30 frontier picture. Artifacts: `research/verify_rollback/{paper_notes.md,verify_rollback_patch.py,run_vr_arm.py,arms/}`.

---

## 2026-06-13 17:40 — PR #33: Tree-causal mask (dead) + Marlin tile-boundary correction ✓ MERGED — cost-model closure (NOT a TPS change)

- **Branch:** `denken/tree-causal-mask-verify-cost` · **Student:** denken
- **Status:** MERGED as a **LOCAL cost-model closure / profiler-infrastructure landing**, NOT a leaderboard/baseline change. Official headline stays PR #4 (126.378 a10g-small); best-LOCAL rung stays PR #14 (131.60 local). Directly refines PR #28's verify-latency curve.
- **Hypothesis:** A sparse tree-causal attention mask (each node attends only to its ancestors) cuts the attention term of the int4 verify forward at tree shapes K=6/8/12, potentially shifting the K=12 ceiling from PR #28's 452 toward 470–490 (or across 500) @ p=0.78. Secondary: the GEMM ramp steps at M≈20/40 are Marlin tile-boundary effects; a fine M-sweep finds the "free" plateau tree shapes.

### Results

| quantity (graph, ctx256, p=0.78, W=4) | PR #28 dense baseline | this PR (tree-masked + tile-corrected) |
|---|--:|--:|
| tree-mask saving M=25/33/49 — **production SDPA** | — | **0.000 / 0.000 / 0.000 ms** |
| tree-mask saving M=25/33/49 — FLOP-ideal ceiling | — | 0.076 / 0.104 / 0.175 ms (≤1.1% of step) |
| Marlin cliff Δ at M=17 / 33 / 49 | (interpolated, hidden) | **+0.772 / +2.176 / +2.869 ms** |
| **V_tree(M=49)** direct | 15.28 ms (interp) | **18.13 ms** (interp under-stated 2.68 ms / 17%) |
| tree K\* @ p=0.78 (drafter / verify-only) | K=12 (M=49): 452.4 / 493.4 (artifact) | **K=11 (M=45): 440.4 / 480.8** |
| **K12 tree TPS @ p=0.78** (primary metric, variant B) | 452.4 (artifact) | **393.9** |
| **verdict_exceeds_500 @ p=0.78** (test metric) | FALSE | **FALSE** (max 440 / 481) |

**W&B runs:** `k56d6cxe` (tree-mask), `36hkaj14` (tile boundary), `aid45far` (tree model), group `spec-verify-tree-mask` — all finished. Advisor sub-agent verified the tile deltas, M=49=18.134 ms, and `verdict_exceeds_500_at_full_scale_withdrafter=False` to logged precision.

### Analysis & conclusions

- **Finding 1 — tree-causal mask is DEAD for this model/hardware.** On the production dense-SDPA + topology-mask path (SpecInfer Eq.4 / EAGLE / Medusa / vLLM) the saving is **exactly 0 by construction** — a tree mask changes *which* scores are masked, not *how many* are computed. Even the unrealizable FLOP-ideal kernel saves ≤0.18 ms (≤1.1% of the step); FlexAttention is *negative* (the whole M≤49 tree fits one 128×128 block → partial-block overhead, pytorch #133562). Attention is only ~2.6% of the int4 verify step; the GEMM ramp dominates and is sparsity-invariant. Added to BASELINE.md dead-ends.
- **Finding 2 (the keeper) — Marlin tile-boundary cost-model bug-fix.** Step jumps at M=17/33/49 land *exactly* where `thread_m_blocks = ceil(M/16)` predicts (Marlin arXiv:2408.11743), and they are large (+2.18, +2.87 ms). PR #28's `LatencyCurve` linearly interpolated across them, **under-stating M=49 by 2.68 ms (17%)**. The corrected curve carries directly-measured boundaries — protects every future drafter-ladder TPS projection.
- **Net on the programme:** >500 TPS @ p=0.78 stays **FALSE — now firmer**; the only reading that approached 500 (variant-C 499.1) *was* the interpolation artifact this PR removes. **Serving guidance for kanna #24: target the M=45 (K=11) tmb=3 plateau, avoid M=17/33/49** — same accepted length, ~12% cheaper verify, no code change beyond tree shape.
- **One open reconciliation (non-blocking, flagged on PR):** report optimum K\*=11 (M=45) vs W&B-logged `optimal_k_*=15` (range-cap; likely the optimistic-accept scenarios — p=0.85 pushes K deeper to 511/558); `tps_tree_meas_p0_780=377.1` matches the K=6 sim exactly. denken to confirm scenario keying before the M=45 guidance is locked.
- **Suggested follow-ups (from denken):** (1) tell kanna #24 to target M=45 not M=49; (2) don't pursue tree-mask kernels here; (3) re-measure the M=45 plateau with a real per-position accept trace once a drafter lands; (4) fold the tile curve back into canonical `results_msweep.json`.
- **Next:** denken → fresh local profiling/cost-model assignment (incl. folding the tile correction into the canonical curve + the highest-value next decode-cost question).

---

- **Branch:** `ubel/empirical-lmhead12k` · **Student:** ubel
- **Status:** MERGED as a **validated lever + best-LOCAL rung**, NOT a new official baseline. Official a10g-small TPS + private-PPL await a gated HF job (approval issue opened). Official baseline headline stays PR #4 126.378.
- **Hypothesis:** Pruning the `lm_head` weight matrix to the top-12,288 most-frequent token rows (bf16, sliced from tied embeddings) cuts the lm_head GEMV bandwidth ~21× and yields a measurable single-stream TPS gain on the int4 base; it passes the official greedy-identity gate empirically (the pruned model is self-consistent) even though it is not adversarially safe.

### Results

| metric | unpruned control (bf16-262k head) | pruned (bf16-12k head) | delta | verdict |
|---|---|---|---|---|
| **tps_local_single_stream** (isolated, single-variable) | 97.65 | **131.60** | **+34.8%** | lm_head prune is real & standalone-positive |
| implied lm_head decode fraction | — | **27.1%** | matches wirbel #8's 26.4% | two independent measurements agree |
| local-to-local net vs PR #4 (int4-262k head, 128.13 local) | — | 131.60 | **+2.7%** | honest cross-config net (student's +3.6% mixed local-vs-official) |
| served_ppl (token-wtd) | — | **1.9712** | better than int4-head ~2.02 | ≤ 2.42 cap ✓ |
| greedy gate (served-vs-served, spec-off) | **GREEDY_IDENTICAL 128/128** | **GREEDY_IDENTICAL 128/128** | 0 divergent | valid (self-consistency) ✓ |
| completed | 128/128 | 128/128 | — | ✓ |

**W&B runs:** NONE (`wandb_run_ids: []`) — serve+validate experiment, no training run. Fully auditable via **38 committed evidence JSONs** under `research/local_validation/lmhead12k_empirical/` (`stage1_evidence/evidence.json`, `greedy_report.json`, `ppl_summary.json`, `control_int4_served/control_result.json`, `clip_floor_ksweep.json`, plus `vllm_baseline_128/` control). Advisor confirmed the marker progression (blocked_local_gpu → greedy_identity_divergent → running_corrected_gate → terminal) and the evidence-file backing; merge preflight passed.

### Analysis & conclusions

- **The lever is real and standalone-positive.** +34.8% isolated single-variable (only head row count differs) with an implied 27.1% lm_head decode-bandwidth fraction that independently matches wirbel #8's 26.4% profiler split. lmhead12k is **rung 5 of the BASELINE.md ladder** ("lmhead12k sparse-verify … the frontier"); this is the first in-repo standalone confirmation.
- **Three keeper validity findings** (sharpen our instrument): (1) the greedy gate is **self-consistency** (served-pruned vs plain-greedy-pruned, *same* checkpoint) — clipping cannot fail it by construction; the PRUNE-EFFECT (pruned-vs-*unpruned*) A/B measures fidelity to a model the gate never tests, not the gate. (2) The earlier 107/128 unpruned "control failure" was an **offline-batched-reference (batch≈128) vs strictly-sequential-candidate (batch=1) FP-reduction artifact** — *every* future greedy-gate run must use a batch=1 served-vs-served reference (wirbel #8's warning, larger here). (3) The int4-argmax clip rate has an **irreducible frequency-selection floor** (~0.78% public / 1.15% held-out) because some argmax tokens appear in *no* selection corpus — "held-out clip ~0" is unreachable by selection, and per finding (1) it isn't the gate anyway.
- **Honest framing:** per BASELINE.md, local A10G is exploratory-only; the official metric is a10g-small HF-Job TPS. So this merges as a validated lever/best-local rung, not a new official baseline. The +2.7% local net over PR #4 is plausible-but-unconfirmed officially (and the head dtypes differ: bf16-12k vs int4-262k).
- **Standing residual risk — private PPL** (not closable locally): a private GT-*target* token outside `kept_ids` → −∞ → +∞ PPL on the private re-run. Greedy-identity passes private by self-consistency, so this is purely a PPL axis. Only a gated a10g-small HF job on the private set closes it.
- **Next:** ubel → follow-up #3 (int4-pruned head, another ~4× head-byte cut, orthogonal to the kept-set). Also compounds in the spec-verify forward (gated on kanna #24). HF-approval issue opened for the official confirmation.

---

## 2026-06-13 17:30 — PR #25: EAGLE-3 full-scale training ✓ MERGED — keeper (drafter asset, reasoning acceptance 0.7314; DATA-bottlenecked)

- **Branch:** `fern/eagle3-full-scale-training` · **Student:** fern
- **Status:** MERGED as a research keeper (drafter asset). No TPS-baseline change — baseline stays PR #4 126.378 TPS (the drafter cannot deploy until kanna's verify-rollback #24 unlocks greedy-valid serving). The asset is the current-best drafter checkpoint, banked for the moment serving is unlocked.
- **Hypothesis:** Training the EAGLE-3 drafter at full scale (20k-step budget, benchmark-distribution data) past the PR #16 harness debug head (tf_acc 0.6816) pushes teacher-forced top-1 acceptance toward 0.78 — the level PR #28 says is needed to approach >500 TPS. Reframed mid-run: full MATH+ShareGPT as a per-source-decomposed arm to isolate whether chat data helps or hurts reasoning acceptance.

### Results

| metric | debug (MATH-only, 898 steps) | full (MATH+SG, 3500 steps) | delta | verdict |
|---|---|---|---|---|
| **tf_acceptance_rate, MATH holdout (n=48,142)** | 0.7051 | **0.7314** | **+0.026** | the benchmark-relevant number (128 public prompts are 100% reasoning) |
| tf_acceptance_rate, ShareGPT holdout | 0.1529 | **0.3444** | +0.19 | chat doubled but intrinsically hard to draft (high-entropy/multilingual/code) |
| tf_acceptance_rate, combined holdout | 0.5839 | 0.6464 | +0.063 | combined understates benchmark-relevant quality |
| val_loss, MATH holdout (final) | — | **1.2876** | — | reasoning fit |
| val_loss, combined (overfit signature) | — | 1.8516@2000 → 1.9519@3500 | +0.10 | overfits after ~2000 steps |

**W&B runs:** `7domtiin` (training — "crashed" = external interruption @ step 3670, `model_best.pt` step 3500 checkpoint intact) · evals `egv59ku0` (full·MATH 0.73136) · `xqtvcj58` (full·SG 0.3444) · `udb18hnh` (full·combined 0.6464) · `y0yupavk` (debug·MATH 0.7051) · `yxkh2739` (debug·SG 0.1529) · `1j8afmzk` (debug·combined 0.5839). All six eval runs finished clean; advisor independently verified headline 0.73136 and all per-source numbers to 4 s.f., no NaN. Training "crashed" status is an external interruption, not a divergence — checkpoint and eval lineage are intact.

### Analysis & conclusions

- **Reasoning acceptance is DATA-bottlenecked, not step-bottlenecked.** MATH-holdout tf_acc plateaus ~0.72–0.73 by step ~2000 (gains <0.004 per 500 steps thereafter), and combined val/loss *overfits* after step 2000 (1.8516→1.9519). More steps on this corpus will not break 0.73. The lever is **benchmark-matched reasoning CoT** (MMLU-Pro / GPQA / AIME-math), not more MATH and not more chat.
- **ShareGPT did not hurt reasoning** — it slightly *helped* MATH acceptance (0.7051→0.7314, via more total steps) while doubling its own acceptance (0.15→0.34). So mixing chat is safe, but chat is intrinsically low-acceptance (the combined 0.6464 is dragged down by the hard SG tail and understates the benchmark-relevant figure, since the 128 public prompts are 100% reasoning: mmlu_pro 57 / gpqa_diamond 57 / aime2026 14).
- **Ceiling caveat (PR #28 linkage):** tf_acc is a *teacher-forced UPPER BOUND* on free-running acceptance. PR #28 established >500 TPS needs free-running top-1 p≥0.85; 0.73 tf_acc maps to something lower free-running. So this asset, while the best drafter we have, is not yet the >500 TPS key — it sets up the next two levers.
- **Asset banked:** `research/eagle3_drafter/checkpoints/full_20k/model_best.pt` (step 3500, 0.7314 reasoning tf_acc). Corpus 2.21M tok (1.76M MATH + 0.45M SG), de-contaminated vs the 128 eval ids. Deploys the moment verify-rollback (#24) unlocks greedy-valid serving.
- **Student's flagged next step (correct):** a benchmark-matched reasoning corpus distilled from the served target on MMLU-Pro/GPQA/AIME. That is fern's next assignment — the corpus that should break the 0.73 plateau toward 0.78. On-policy distillation (Draft-OPD, round-3 H1) is the follow-on lever if static-corpus distillation plateaus below 0.85.

---

## 2026-06-13 17:00 — PR #28: Extended verify-latency M-sweep ✓ MERGED — keeper (ceiling corrected, extrapolation killed)

- **Branch:** `denken/verify-latency-msweep` · **Student:** denken
- **Status:** MERGED as a research keeper. Replaces the only extrapolated input in the PR #26 tree-salvage cost model with measured data. No TPS-baseline change — baseline stays PR #4 126.378 TPS.
- **Hypothesis:** The int4 verify forward stays bandwidth-bound and ~flat in M well beyond M=16, so extrapolating the PR #18 curve to M=25 (K=6 tree) and M=41 (K=10 tree) is safe, and the >500 TPS @ p=0.78 claim from PR #26 holds on measured data.

### Results

| metric | PR #26 extrapolated | PR #28 measured | verdict |
|---|---|---|---|
| V_tree(M=25) / V_lin(M=7) — K=6 tree overhead | 1.057× | **1.113×** | higher than extrapolated but ≪ 4× naive fear |
| K=6 tree TPS @ p=0.6792 | 346.8 | **331.2** (−4.5%) | net-positive 1.46×, holds |
| Tree K* @ p=0.78 | K=20 (M=81): **616 TPS** (extrapolated) | **K=12 (M=49): 452.4 TPS** | **30% overstatement** — interior optimum found |
| >500 TPS @ p=0.78? | YES (extrapolated K≈10) | **NO — max 452/493 TPS** | ceiling refuted at debug-head acceptance |
| Knee M* | ≥16 (edge of old sweep) | **M≈24** (ramp starts M≈20) | step-structure from tile quantization |

**W&B runs:** `2mk0z0c3` (latency M-sweep, group `spec-verify-msweep`) · `imoi4mx1` (tree acceptance model, group `spec-verify-msweep`). Both finished; all cited numbers verified vs W&B artifacts (60-row cost table, 120-row tree table).

### Analysis & conclusions

- **The hypothesis is partially refuted — and that's the finding.** The verify forward IS flat through M≈32 (+2.6%), so the K=6 moderate tree (M=25) extrapolation was essentially sound (1.057→1.113×). But beyond M≈32 the int4 Marlin W4A16 GEMM goes compute-bound and ramps: M=40 +31%, M=64 +60% over M=1. Discrete steps at M≈20, 32, 64 are Marlin tile-boundary quantization effects.
- **The ramp is GEMM, not lm_head.** The forward GEMM share rises 62%→68% through the ramp; lm_head grows smoothly (2.86→3.57 ms). CUDA-graph mode exposes the ramp (eager masks it with fixed CPU-launch overhead).
- **The REAL interior optimum is K*≈8–12** (not K=20). At p=0.78: K=8 (M=33) gives 429.3 TPS → peaks at K=12 (M=49): 452.4 TPS → then declines as ramp outpaces saturating acceptance.
- **>500 TPS requires drafter quality, not deeper trees.** Only at p≥0.85 (top-1 acceptance ≥0.85) does the K=12 tree clear 500 (531 TPS). The debug-head acceptance regime (p≈0.68) caps at ~366–406 TPS (K*=8). **This re-anchors the entire team's focus on fern #25 (EAGLE-3 full-scale training) as the ceiling-setter.**
- **Dense-M upper-bound caveat** (reported by student): the profiler times a dense/full-causal M-token forward (upper bound). The true tree-causal-masked cost is cheaper only in the attention term (16%→13% of the ramp), so the GEMM-dominated correction is sub-2 ms at M≈49 — tight upper bound.
- **Strategic re-anchor:** K*≈8–12, not K≈20. The next steps are (a) tree-causal mask measurement to tighten the dense-M upper bound, (b) EAGLE-3 training to push p toward 0.85, (c) kanna's verify-rollback to unlock serving.

---

## 2026-06-13 16:20 — PR #27: int4 channel-wise lm_head sweep ✗ CLOSED — confirmed NEGATIVE (g128 stays the floor)

- **Branch:** `lawine/int4-channel-lmhead-sweep` · **Student:** lawine
- **Status:** CLOSED as a clean, fully-characterized NEGATIVE. No TPS-baseline change — baseline stays PR #4 126.378 TPS. The channel submission dir stays on the student branch (dead-end; not merged).
- **Hypothesis:** channel-wise (`group_size=-1`) int4 lm_head gives +~1 TPS over g128 (PR #4) because per-output-channel dequantization requires a simpler scale lookup in the Marlin GEMV kernel; PPL cost small (lm_head error affects low-confidence vocab tail). Single-variable change: one line in `submissions/int4_g128_lmhead/build_quant.py`.

### Results

| metric | g128 control (PR #4) | channel-wise (g=-1) | delta | verdict |
|---|---|---|---|---|
| local TPS (A10G, 128 prompts) | **128.13** | 127.74 | **−0.39** | NO GAIN — within noise |
| local PPL (128 prompts / 61,797 tok) | **2.0188** | **2.0212** | +0.0024 | ≤ 2.42 cap ✓ |
| greedy identity (self spec-off) | GREEDY_IDENTICAL 128/128 | **GREEDY_IDENTICAL 128/128** | — | valid ✓ (0 divergent / 65,536 tok) |
| same-path PPL gate | SAME_PATH_OK (gap 0.0) | **SAME_PATH_OK (gap 0.0)** | — | honest ✓ |
| completed | 128/128 | **128/128** | — | ✓ |
| Marlin g=-1 support | — | confirmed (no g=32 fallback needed) | — | — |

**W&B runs:** `gtlruguu` (channel prevalidate, TPS 127.74/PPL 2.0213) · `a0xtk79t` (g128-ctrl prevalidate, TPS 128.13/PPL 2.0188) · `c9qy6rcq` (channel validation, same_path_gap 0/SAME_PATH_OK/128/128). All three in `gemma-challenge-senpai` or `wandb-applied-ai-team/senpai`; all finished; independently verified by advisor to >3 sig figs, no NaN.

### Analysis & conclusions

- **The TPS gain did not materialize.** The lm_head is a single GEMV per decode step over a tiny fraction of total decode traffic; the scale-lookup simplification for g=-1 vs g128 is sub-noise at the whole-model level. The PPL moved +0.0024 (well under +0.011 projection and far under the 2.42 cap), and the greedy self-gate is byte-exact 128/128 — the coarser head did NOT flip any near-tie argmax.
- **Net verdict:** channel-wise is SAFE but POINTLESS as a speed lever. **lm_head quant granularity is not a TPS knob.** A head-side TPS lever must come from a smaller effective vocab at decode (the lmhead12k direction), not from g128→channel.
- **HF approval issue:** correctly NOT opened by lawine (no improvement to confirm). Correct protocol.
- **The real deliverable:** lawine's **bug flag** — a **silent-correctness hazard on the greedy-gate auto-reference resolution** (`harness.py:84-92` manifest `env.MODEL_ID="model"` copied into serve env before `setdefault` → `srv.model_id` stays the relative literal `"model"` → `reference_for("model")` keys shared `greedy_reference/model/` tag → NO_REFERENCE AND every `env.MODEL_ID="model"` submission collides on the same tag → silent wrong-reference verdict risk). The actual GREEDY_IDENTICAL was confirmed offline via `--reference` flag (sound). **lawine reassigned to harness fix → PR #32**.

---

## 2026-06-13 15:49 — PR #22: Honest fa2sw-precache frontier in-repo + LF29 dual-gate-blind finding ✓ MERGED — keeper (asset + validity)

- **Branch:** `wirbel/fa2sw-precache-validate-and-lf29-check` · **Student:** wirbel
- **Status:** MERGED as a research keeper (plain squash; no TPS-baseline change — baseline stays PR #4 126.378 TPS). Two deliverables: (A) the honest ~420 TPS frontier stack is now an in-repo VALID base; (B) a validity finding about our own tooling.
- **Hypothesis (two-part):** (A) reproduce kenyan-duma's honest precache frontier locally; it should pass the same-path PPL gate (gap ≈ 0). (B) the pupa-lf29cap444 lane is a grader-conditional FFN bypass → same-path PPL gate should return gap ≈ 0.17 → FAIL.

### Results

| part | gate | result | verdict |
|---|---|---|---|
| **A** — kenyan-duma honest frontier | same-path PPL (`same_path_ppl.py`) | gap **0.0000**, both paths PPL **2.37688**, bit-identical NLL (11 sig figs) | `SAME_PATH_OK` — confirmed single-path honest ✓ |
| **B** — pupa-lf29cap444 | same-path PPL (teacher-forced) | gap **0.0000**, PPL **2.37794** (NOT the predicted 0.17) | `SAME_PATH_OK` — gate is **blind** to this fold |
| **B** — pupa-lf29cap444 | greedy identity (fold-on vs exact-FFN, spec-off AR, 65,536 tok) | **0 flips / 128 prompts identical**, `flip_rate_per_token=0` | `GREEDY_IDENTICAL` — fold is argmax-safe |
| W&B | `jg99477i` (Part A), `tju905db` (Part B same-path), `gz5b064e` (greedy gate) | all 3 finished; metrics verified vs logged summary (5+ sig figs) | no fabrication |

### Analysis & conclusions

- **Part A asset:** `submissions/fa2sw_precache_kenyan/` (serve.py + patches, no weights — synced at runtime) is now an in-repo VALID base for future TPS work (tree-salvage, accepthist, EAGLE-3 can branch from the real frontier stack). Mechanism documented component-by-component in `research/validity/fa2sw_precache_notes.md`. Local exploratory TPS 867 tok/s (NOT official a10g-small — liveness only).
- **The headline finding — both output gates are BLIND to the LF29 fold class.** The pupa LF29 lane keys layer-29 FFN on `num_prompt_logprobs` (exact FFN when PPL is graded; cheap affine fold for timed decode) — confirmed in `serve.py:411-415`. But the deployed fold is **both teacher-forced-PPL-neutral AND argmax-safe**: same-path PPL gap 0.0000 (forcing the fold ON every request gives 2.3767, marginally *below* exact-FFN 2.3779) and greedy flip_rate 0/65,536. **Neither same-path PPL nor greedy_gate can detect this lane.** The only detector is **static mechanism inspection** of the grader-conditional branch. This corrects the prior research-state assumption that `greedy_gate` is the load-bearing detector for fold-class lanes — it is also clean here. BASELINE.md's "every HF-approval issue requires `--check-same-path` output" reads PASS even for this invalid lane.
- **The 2.55 mystery:** neither output gate reproduces frantic-penguin/itaca's community 2.55. Since greedy text is byte-identical to exact-FFN (0 flips ⇒ no prefix divergence ⇒ no error compounding), free-running greedy PPL on pupa's deployed weights is ≈2.378. The 2.55 is most likely a **reconstructed** fold (R²≈0.80, not pupa's weights) or a non-greedy regime — needs the external frantic-penguin method to settle.
- **Intellectual honesty:** wirbel falsified their own hypothesis (predicted gap 0.17 / flip>0; measured 0/0), reported faithfully, and held the board post for human approval (Issue #29). Excellent diligence.
- **Scope-limit doc kept:** `research/validity/same_path_ppl.md` now permanently documents that same-path PPL + greedy_gate are blind to argmax-preserving / decode-compounding folds; mechanism inspection is load-bearing.
- **Follow-ups:** (1) wirbel reassigned → **PR #30** (frontier decode-step profile on the new in-repo `fa2sw_precache_kenyan` base — find the next TPS lever beyond 421). (2) **Issue #29** opened (board post to evals taskforce) — HELD, human-gated; advisor verified the W&B evidence but is NOT approving publication. (3) Suggested team direction: a static mechanism-scanner for grader-conditional request-field branching — the only detector for this fold class.

---

## 2026-06-13 15:20 — PR #26: Tree-salvage acceptance model (width-4 tree vs linear K) ✓ MERGED — keeper (cost model)

- **Branch:** `denken/tree-salvage-acceptance-model` · **Student:** denken
- **Status:** MERGED as a research keeper (no served checkpoint / no TPS-baseline change; baseline stays PR #4 126.378 TPS). Plain squash-merge. `scripts/profiler/tree_acceptance_model.py` + extended `eval_eagle3.py` (top-k + trace) now canonical.
- **Hypothesis:** width-4 tree decoding raises E[accepted tok/invoke] substantially over linear K=6 for our EAGLE-3 head, and the acceptance gain outweighs the tree-verify overhead → realistic TPS ceiling >500 at full-scale acceptance.

### Results

| metric | value | note |
|---|---|---|
| top-1 acc | 0.6792 | reproduces PR #16 tf_acc 0.6816 (within 0.4%) |
| top-4 acc | 0.8605 | hypothesis ≥0.82 ✓ |
| **rescue_rate (width-4)** | **0.5651** | **beats fableous 0.431 by +0.134** — our head is more tree-salvageable |
| E_accept tree4 / linear (empirical) | **1.5923** | primary metric; i.i.d. model agrees (1.60) |
| **measured tree-verify overhead** | **1.06×** | M=25 forward ≈ as cheap as M=7 (PR #18 flat-in-M); NOT the feared 4× |
| K=6 tree TPS @ p=0.6792 | 346.8 (+53% vs linear 227.3) | verify V=12.05ms **extrapolated** at M=25 |
| full-scale ceiling @ p=0.78, K=6 | **393 TPS** (w/ drafter) | `verdict_exceeds_500_at_full_scale = False` at K=6 |
| >500 TPS @ p=0.78 | only at K≈10 (M≈41, **extrapolated**) | beyond PR #18 measured M≤16 |
| W&B | eval `8idbwjk1`, cost-model `zlzti9h0` (group `tree-salvage-acceptance-model`) | all metrics independently verified vs logged summary |

### Analysis & conclusions

- **Tree-salvage is real and net-positive on this hardware.** The decisive fact is the **1.06× measured verify overhead**, not the acceptance gain alone: under a 4×/additive verify model the tree is net-negative; under PR #18's measured bandwidth-bound (flat-in-M) curve it's +53%. The tree-salvage case **depends on the int4-verify-flat-in-M finding** — a clean, physically-grounded refutation of the naive "4× tree cost" framing.
- **Validates the acceptance lever for kanna's verify-rollback path (#24).** With overhead ~1.06× and E gain ~1.6×, width-4 tree at K≈6–8 is the concrete config to prototype once spec decode is greedy-valid.
- **Honest limits (denken flagged all):** (1) the >500 @ full-scale is conditional — needs p→0.78 AND deep K≈10 where M≈41 is **extrapolated** beyond PR #18's measured M≤16; (2) empirical trace is slightly *sub*-geometric (0.96× i.i.d.) — the "easy-span" positive correlation hypothesized did NOT appear on this head+MATH set, though the tree/linear ratio is preserved so the gain conclusion is robust; (3) D=1.4ms is fableous's *linear* drafter cost — a width-4 tree drafter expands K·W nodes so may cost more (verify-only vs +drafter band brackets it).
- **Checkpoint-provenance catch (excellent diligence):** the PR-named `debug_1k/` is a 28-step underfit (tf_acc 0.2484); the real 0.6816 head is `debug_1k_2ep/` (898 steps), confirmed against W&B `30bgs1rs`. denken evaluated the correct head on held-out `debug_1k_eval_corpus.pt` and staged canonical paths. **Note for fern #25 / future drafter work: use `debug_1k_2ep/`, not `debug_1k/`.**
- **Follow-up assigned → denken PR #28:** extend the PR #18 verify sweep to M∈{20,24,28,32,40,48,64} to replace the M=25/M=41 extrapolation with measured latency — the only soft spot in the >500 projection.

---

## 2026-06-13 14:38 — PR #4: int4 g128 + untied int4 lm_head (~127 TPS) ✓ MERGED — new leaderboard baseline rung

- **Branch:** `lawine/int4-g128-lmhead` · **Student:** lawine
- **Status:** MERGED — new best merged rung. `submissions/int4_g128_lmhead` is now the best merged submission. All future submissions beat 126.38 TPS.
- **Hypothesis:** untied int4 lm_head (eliminating the bf16 GEMV for 262k-vocab verify = 26.4% of decode GPU time per PR #8 profiler) + full-body g128 granularity (slight additional weight-byte reduction vs per-layer) → reaches the int4 Marlin weight-byte floor on Ampere.

### Results

| metric | value (official a10g-small) | vs PR #3 base |
|---|---|---|
| tps / output_tps | **126.378** | 1.32× (**+32%**) |
| ppl (served) | **2.019** | ≤ 2.42 ✓ |
| completed | **128 / 128** ✓ | — |
| greedy identity | **GREEDY_IDENTICAL 128/128** (served-vs-served cap=512) ✓ | — |
| same-path gate | **SAME_PATH_OK (gap 0.0000)** ✓ | — |
| job | `6a2d5a96234ca64b60121aa5` | — |
| W&B | `905tbujn` (official a10g-small) · `0pxj6n63` (local proxy + greedy) | — |

**Overall: 2.87× over bf16 (44.018 TPS), 1.32× over PR #3 int4 base.**

### Analysis & conclusions

- **Confirms lmhead profiler finding** (PR #8): 26.4% of decode GPU time was the 262k-vocab bf16 GEMV. Untied int4 lm_head eliminates it, explaining the +32% TPS gain. This is the exact profiler prediction.
- **This is the weight-byte floor.** Sub-4-bit (no sm_86 kernel) and fp8 KV (no A10G support) are dead ends. No further weight-bandwidth reduction is achievable in vLLM 0.22.0 on Ampere. Every remaining TPS lever is either (a) the drafter ladder (spec decode, gated on kanna verify-rollback), (b) lmhead12k (ubel #14, cheaper verify), or (c) runtime/warmup (precache, onegraph — the frontier stack).
- **Greedy validity methodology confirmed:** served-vs-served (spec-off) via `check_greedy_identity.py` passes cleanly (GREEDY_IDENTICAL 128/128). This is the gold-standard test.
- **lawine confirmed official PPL artifact** present on the HF job result — closing the near-cap timing question from last cycle.

---

## 2026-06-13 14:38 — PR #19: Batch-invariant vLLM spec decode ✓ MERGED — LINCHPIN DEFINITIVE NEGATIVE

- **Branch:** `kanna/batch-invariant-vllm-spec` · **Student:** kanna
- **Status:** MERGED — definitive negative. Closes the invariant-kernel lane. Next lane: verify-rollback (kanna PR #24).
- **Hypothesis:** `VLLM_BATCH_INVARIANT=1` (aten-override batch-invariant kernels) makes the M=K+1 verify forward bit-match the M=1 AR forward → greedy-identical spec decode.

### Results

| arm | INV | target GEMM | flip/tok | 95% CI | identical/32 | W&B |
|---|---|---|---|---|---|---|
| int4 ON (decisive) | 1 | Marlin `_C` (un-covered) | **0.376%** | [0.234, 0.518]% | 5/32 | `hz8jkc5h` |
| int4 OFF (control) | 0 | Marlin `_C` (un-covered) | 0.332% | [0.205, 0.460]% | 6/32 | `8wne15eh` |
| bf16 ON (discriminator) | 1 | aten linear (covered) | **0.111%** | [0.057, 0.166]% | 16/32 | `z0mclftv` |
| bf16 OFF (PR #5 ref) | 0 | aten linear | 0.72% | — | — | — |

**Primary metric:** int4_mtp_batchinv_greedy_flip_rate_per_token = **0.00376** (0.376%) — NOT zero. **Verdict: DIVERGENT, invariant-kernel lane CLOSED.**

### Analysis & conclusions

The bf16 control arm is the key insight. By removing int4 Marlin (using aten-covered bf16 GEMM) while keeping INV=1, we isolate TWO independent un-coverable root causes:

- **(a) int4 Marlin `_C` op:** contributes ~0.265%/tok excess above bf16 floor. The Marlin custom op is outside aten's scope; batch-invariance cannot intercept it. This was the main prior hypothesis (Marlin was "plausibly already M-invariant") — REFUTED.
- **(b) Spec verify path non-aten residual:** bf16 ON (full aten coverage, zero Marlin) is STILL divergent at 0.111%/tok. An irreducible non-aten component in the spec verify forward (attention-metadata build, rejection-sampler logits compare, or a fused step) remains batch-variant. Corroborated by vLLM issue #27433: "batch-invariance does not currently integrate with speculative decoding."
- **Consistency check:** 0.265% (a) + 0.111% (b) ≈ 0.376% (observed int4 ON). The two sources are independent and additive.
- **Implication:** neither int4 nor bf16 target drafter ladders are rescuable by `VLLM_BATCH_INVARIANT`. The invariant-kernel lane is closed for greedy-valid spec decode at ANY precision in vLLM 0.22.0.
- **Next lane:** verify-rollback (arxiv 2601.17768) — re-verify accepted tokens under fixed-shape M=1 reduction after each spec step; commit consistent / roll back violators. This targets both causes: (a) is dodged (rollback uses M=1 AR path, no Marlin batch-size dependency on committed path), (b) is caught and corrected by the re-verify. Assigned to kanna PR #24.

---

## 2026-06-13 14:38 — PR #16: EAGLE-3 draft-head training harness ✓ MERGED — keeper research artifact

- **Branch:** `fern/eagle3-training-pipeline` · **Student:** fern
- **Status:** MERGED — keeper (training harness + asset). No leaderboard TPS improvement; infrastructure needed for the drafter ladder.
- **Hypothesis:** An EAGLE-3 draft head trained via offline distillation from Gemma-4 E4B (using aux hidden states from layers 2, 21, 39) can achieve teacher-forced acceptance ≥ 3.5 tok/step on a held-out STEM corpus at debug scale.

### Results

| metric | value | note |
|---|---|---|
| tf_acceptance_rate_debug_1k | **0.6816** | at 1k steps, 200 MATH train samples |
| final_val_loss_debug_1k | 1.3372 | still converging |
| W&B | `30bgs1rs` (group `eagle3-drafter-training`) | |

**Verdict:** pipeline confirmed functional. 0.6816 is in the "0.50–0.70 → schedule full run" range.

### Analysis & conclusions

- **Harness architecture:** faithful PyTorch reimplementation of vLLM's Eagle3DraftHead with vLLM-matching weight names/shapes (deployable checkpoint). Llama decoder layers (not Gemma), RoPE/RMSNorm/GQA/SwiGLU. feature_shift=1 vLLM-faithful alignment. Chunked 262k-way CE to avoid OOM.
- **Corpus:** EleutherAI/hendrycks_math (allenai/MATH 404s), 200 train samples, 52,751 tokens.
- **Key finding:** no public Gemma-4 E4B EAGLE-3 checkpoint exists (thoughtworks/Gemma-4-31B-Eagle3 is shape-incompatible) → trained from scratch.
- **Next:** full-scale training (2000 MATH + 500 ShareGPT samples, 20k steps, targeting tf_acc ≥ 0.78) assigned to fern PR #25. Serving is gated on kanna's verify-rollback PR #24.

---

## 2026-06-13 14:38 — PR #18: int4 decode-step cost model vs K ✓ MERGED — keeper research artifact

- **Branch:** `denken/spec-verify-cost-model` · **Student:** denken
- **Status:** MERGED — keeper (analytical cost model). No leaderboard TPS improvement; foundational analysis for drafter-ladder decisions.
- **Hypothesis:** characterize the ideal TPS ceiling of int4 spec decode as a function of K (draft count) and acceptance probability p.

### Results

| metric | value | note |
|---|---|---|
| tps_ceiling_ideal_at_kstar | **1,269.5 TPS** | at K*=15, acceptance p=0.7 |
| optimal_k_geom_p0.7 | **K*=15** | geometric acceptance, 40% of weight-GEMM time is verify |
| W&B | `pvj0qogp` (group `spec-cost-model`) | |

### Analysis & conclusions

- **The sky is high:** 1,269.5 TPS ideal ceiling (at p=0.7, optimal K) confirms the drafter ladder has massive headroom. Even at p=0.5, the ceiling is > 600 TPS.
- **K=6 is suboptimal:** at p=0.7, ideal K*=15. The current MTP drafter at K=6 leaves TPS on the table even at full acceptance. Higher acceptance rate raises K* — tree decoding (fableous: width-4 rescues 43.1% of linear misses) could change the optimal strategy.
- **Feeds verify-rollback net-value:** the cost model now establishes the ceiling. denken's next assignment (PR #26) extends it to tree decoding.
- **Dropped dependency in rebase:** no functional issue — the cost model files (research/spec_cost_model/ + scripts/profiler/spec_cost_model.py) are self-contained; the dropped dependency was an unmerged PR-specific hook that was correctly removed.

---

## 2026-06-13 14:15 — PR #22: Honest precache frontier + LF29cap same-path validity (SENT BACK, WIP)

- **Branch:** `wirbel/fa2sw-precache-validate-and-lf29-check` · **Student:** wirbel
- **Status:** NON-TERMINAL (pending_arms=true). Sent back for greedy_gate on pupa-lf29cap444 + terminal marker.
- **Hypothesis:** (A) reproduce kenyan-duma honest precache frontier (PPL ~2.377); (B) test whether pupa-lf29cap444 fails the same-path PPL gate (gap ~0.17).

### Part A results (PASS — clean asset)

| metric | value |
|---|---|
| same_path_ppl_gap (fa2sw_precache) | **0.0000** (SAME_PATH_OK, exit 0) |
| same_path_ppl | **2.37688** |
| NLL equality | byte-identical to 11 sig figs — single-path confirmed |
| W&B | `jg99477i` |

Part A confirmed: kenyan-duma honest precache frontier is single-path at the strongest possible resolution. Clean VALID base for tree-salvage / accepthist / EAGLE-3 branching.

### Part B results (UNEXPECTED finding — important tooling insight)

| metric | predicted | measured | verdict |
|---|---|---|---|
| same_path_ppl_gap (pupa-lf29cap444) | ~0.17 / FAIL | **0.0000 / SAME_PATH_OK** | gate is BLIND to this class |
| fold-forced same_path_ppl | — | 2.3767 (−0.0013 vs exact) | fold is teacher-forced-neutral |
| W&B | — | `tju905db` | |

**Critical finding (structural — affects all future validity work):** the same-path PPL gate (merged PR #21) is **teacher-forced-blind** — it cannot detect argmax-preserving / decode-compounding folds. The LF29 affine fold (ridge approximation of layer-29 FFN, R²≈0.80) is teacher-forced-neutral because each token is scored on the ground-truth prefix; the fold's cost is in free-running decode where argmax flips compound. Two independent mechanisms: (1) teacher-forced scoring is fold-neutral by construction; (2) `echo+logprobs` is coupled to `prompt_logprobs` in vLLM (`completion/protocol.py:276-277`), tripping the same bypass exemption. **→ `greedy_gate` (served-token identity) is the load-bearing validity instrument for fold-class lanes.** The same-path gate catches logit-level path splits (request-field branching on `prompt_logprobs`).

This corrects the BASELINE.md scope statement: "every future HF-approval issue must attach `--check-same-path` output" still holds for logit-path split detection, but greedy_gate is ALSO required for fold-class lanes. The `research/validity/same_path_ppl.md` scope-limit update (wirbel PR #22) will land when the PR merges.

### Next steps (pending)

wirbel authorized to run greedy_gate on pupa-lf29cap444 (local, spec-off served-vs-served). Expected: flip_rate > 0 (the fold changes decode-path argmax where the approximation crosses a decision boundary). Board post held for human approval. Terminal marker expected once greedy_gate completes.

---

## 2026-06-13 14:00 — PR #3: Reproduce int4 QAT W4A16 leader (~95 TPS) ✓ MERGED — first official int4 base rung

- **Branch:** `stark/int4-qat-w4a16` · **Student:** stark
- **Status:** MERGED — new official base rung of the reproduction ladder. `submissions/int4_qat` is now the best merged submission.
- **Hypothesis:** int4 W4A16 (Marlin) is the dominant single-stream speed lever (decode is memory-bandwidth-bound; quartering text-linear weight bytes bf16→int4 lifts ~44→~95 TPS). Google's QAT checkpoint keeps PPL *below* the bf16 reference (~2.01 vs 2.30), so faster AND safely inside the 2.42 cap.

### Results

| metric | value (official a10g-small) |
|---|---|
| tps / output_tps | **95.463** (2.17× over bf16 44.018) |
| ppl | **2.0057** (≤ 2.42 cap ✓; better than bf16 2.30) |
| completed | **128 / 128** ✓ |
| total_tps | 144.53 (diagnostic) |
| duration_s | 686.5 · job_status COMPLETED ✓ |
| greedy identity | valid within same serve/job stack (no token-changing optimization added) |
| job / run | `6a2d55c7234ca64b60121a6f` / `results/senpai/int4-qat-20260613T130614Z` |

**W&B run:** N/A (serving-submission reproduction, no training). Official artifacts under `results/senpai/int4-qat-20260613T130614Z/`. Local proxy ≈ 95.99 TPS / 2.0055 PPL (<0.6% off official).

### Analysis & conclusions

- int4 W4A16 confirmed as the **dominant single-stream lever on official hardware**: ~4× less weight bandwidth, the foundation the entire ~420 frontier stack builds on. Base rung is now an official, valid, merged result.
- **Cold-start/40-min-cap did NOT bite** for a submission this fast: `ppl_summary.json` wrote 13:42:23Z, ~3.5 min before the cap. PPL is cheap (one forward pass) and benchmark+decode run ~2.2× faster than bf16. (Slower stack rungs later will tighten this margin — keep the watch.)
- All modalities loaded (vision/audio bf16 via QAT `ignore` list, no `--limit-mm-per-prompt`). No text-only shortcut.
- **Next rung already landed:** lawine PR #4 (int4 g128 + untied int4 lm_head) reports official **126.378 TPS / PPL 2.019 / GREEDY_IDENTICAL 128/128**, +32% on this base — merging once rebased onto this commit + official-ppl artifact confirmed.

## 2026-06-13 13:00 — PR #21: Same-path PPL gate ✓ MERGED

- **Branch:** `wirbel/same-path-ppl-gate` · **Student:** wirbel
- **Status:** MERGED — validity tooling protecting all future HF submissions. No TPS change.
- **Hypothesis:** for an honest single-path submission, timed-generation-path PPL equals prompt_logprobs-path PPL; a non-zero gap (>0.05) reveals grader-conditional branching on `bool(num_prompt_logprobs)`.

### Results

| metric | value |
|---|---|
| `same_path_ppl` (echo/no `prompt_logprobs`) | **2.3012128792** |
| `prompt_logprobs_ppl` (official path) | **2.3012128792** |
| `\|gap\|` | **8.88e-16 ≈ 0.0000** |
| gate verdict | **SAME_PATH_OK** (exit 0) |
| GT records | 128/128 |
| scored tokens | 61,797/61,797 |

**W&B run:** `b9igh00q` (wandb-applied-ai-team/gemma-challenge-senpai, group `same-path-ppl-gate`, finished, all values verified).

### What was built

- `scripts/local_validation/same_path_ppl.py` — scores reference continuations via the generation path with **no `prompt_logprobs` field** in the request (indistinguishable from timed throughput). Uses `echo:true` + `logprobs:1` to read per-token logprobs without triggering the branch a gamed submission would key on.
- `--check-same-path` flag wired into `validate_submission.py` — non-zero exit if `|gap| > 0.05`.
- Calibration artifacts at `research/validity/vllm_baseline/` (both `*_summary.json` + `*_results.jsonl`).
- Documentation at `research/validity/same_path_ppl.md` with honest-vs-gamed reference points.

### Why this matters (public context)

The LF29cap lane (pupa-agent 459 TPS / need-for-speed 457 TPS, cmpatino-verifier "VERIFIED VALID") was confirmed grader-conditional by frantic-penguin (`20260613-090759-237`): `lffn_ppl_exact_active = (LFFN_PPL_EXACT==1 and bool(num_prompt_logprobs))` — `prompt_logprobs` grader gets exact FFN (PPL 2.378), decode gets cheap affine fold (same-path PPL 2.5499, > 2.42 cap). PPL 2.3779 identical across ALL LF29cap verifier re-runs (smoking gun: frozen artifact). frantic-penguin escalated to cmpatino-verifier + evals taskforce. Our gate cleanly separates honest (gap ≈ 0) from gamed (gap ≈ 0.17).

**Required from now on:** every HF-approval issue must attach both `greedy_gate` verdict + `--check-same-path` output.

### Critical scope note

Gate catches request-field branching on `prompt_logprobs`. Does NOT catch `echo`-branching or prefix-cache replay keyed on public-prompt content. Named residual attack surfaces in `research/validity/same_path_ppl.md`.

### Advisory action

- PR comments addressed (advisor guided probe design: no `prompt_logprobs` in request).
- wirbel assigned next task (#22): reproduce kenyan-duma honest precache frontier locally + apply gate to LF29cap lane + publish to evals taskforce.

---

## 2026-06-13 12:55 — PR #14: Empirical lmhead12k ↩ REVIEW → request-changes (int4-argmax re-selection)

- **Branch:** `ubel/empirical-lmhead12k` · **Student:** ubel
- **Status:** WIP (non-terminal; `greedy_identity_divergent_pending_decision`). Reviewed, requested changes, sent back. NOT merged (greedy-invalid), NOT closed (alive + crisp fix).
- **Hypothesis:** pruning the 262k lm_head to a ~12,288 kept-vocab set cuts lm_head GEMV bandwidth (~5–8% TPS over the int4 base) while preserving PPL ≤ 2.42 and greedy identity.
- **Results (local A10G, exploratory):**

| metric | pruned lmhead12k (12,288) | bf16 stock | gate | verdict |
|---|---|---|---|---|
| TPS (single-stream) | 128.23 | 43.95 | higher | ≈ int4 base (prune delta unmeasured — no unpruned-int4 control) |
| served PPL | 1.9767 | 2.3012 | ≤ 2.42 | ✓ (but blind — see below) |
| completed | 128/128 | 128/128 | =128 | ✓ |
| greedy-identity | **DIVERGENT** | (ref) | required | ✗ **invalid** |

- W&B: none logged (local serve/validate, no training). Artifacts: `research/local_validation/lmhead12k_empirical/{greedy_identity_summary,greedy_prune_effect_int4full_vs_pruned,select_analysis,*_summary}.json`.
- **Root-cause finding (valuable, non-obvious):** `kept_ids` was selected from the **bf16** model's argmax, but the served model is **int4**. int4 quantization moves ~1.33% of greedy-argmax decisions (874/65,536; 114/128 prompts) to tokens bf16 never emits → pruning clips them → near-tied survivors flip across numeric paths → DIVERGENT. Clean offline-eager A/B (int4full vs pruned) confirms the prune itself diverges (10/128), independent of serving config. **The kept set covers the wrong model.**
- **PPL is blind to greedy clips:** the −inf scatter on 250k pruned rows shrinks the softmax denominator, *inflating* every kept token's logprob (PPL 1.98 < bf16 2.30). Teacher-forced PPL cannot see a greedy argmax clip. Reinforces why same-path/greedy gates (not PPL) are the validity backstop.
- **Decision & rationale (request changes):** fix = re-select `kept_ids` from the **int4** model's argmax over a **broad corpus** (not public-128-specific), sized so the int4-argmax-outside-kept clip rate is ~0 on public AND a held-out split. Report the **held-out clip rate** = private greedy-identity failure rate (the lmhead12k analog of private TPS drift). Re-run the gate **served-vs-served** (wirbel #8), not offline-eager (avoids ~20% false divergence). Cheap add: serve an unpruned-int4 control to isolate the prune's conc=1 TPS delta — if ~neutral, lmhead12k's value lives in the spec-decode verify forward (gated kanna #19), not standalone. Drafter-independent rung; GPU now available.

## 2026-06-13 (cycle 9) — PR #16: EAGLE-3 draft-head training pipeline ↩ INTERIM REVIEW → sent back (option c)

- **Branch:** `fern/eagle3-training-pipeline` · **Student:** fern
- **Status:** WIP (not terminal). Reviewed an interim/blocking-question update; steered, did not merge or close.
- **Hypothesis:** an EAGLE-3 head distilled from Gemma-4 E4B aux states `(2,21,39)` can reach offline teacher-forced acceptance well above the QAT-MTP baseline, and the training pipeline is functional + CUDA-graph-compatible.
- **What landed (Steps 1–4, validated):** faithful plain-PyTorch `Eagle3DraftHead` (vLLM-matching weight names/shapes; the vLLM head is inference-only/no-autograd), from-scratch (no compatible public Gemma-4 EAGLE-3 ckpt), frozen tied embed/lm_head init, chunked 262k-way CE (avoids `[N,262144]` fp32 OOM), `feature_shift=1` vLLM-faithful alignment. Corpus: `EleutherAI/hendrycks_math` (allenai/MATH 404s), 200 train + 20 held-out, 52,751 tokens. Peak GPU **11.2 GB**.

### Interim result (accidentally cap-constrained 2-epoch run)

| epoch | step | held-out tf_acceptance | held-out loss |
|---|---|---|---|
| ~0.5 | 7 | 0.066 | 5.68 |
| ~1.0 | 14 | 0.192 | 4.64 |
| ~1.5 | 21 | 0.236 | 4.19 |
| **~2.0** | **28** | **0.248** | **4.10** |

Train loss 12.97→3.72, train acc 0→0.295. W&B `rxxd8yen` (group `eagle3-drafter-training`).

### Decision & rationale

fern flagged a **binding conflict**: the PR's "1000 steps" over the 200-sample corpus = ~71 epochs, violating the live launch's accidental `SENPAI_MAX_EPOCHS=2` bound. fern correctly **refused to override** the bound and ran the max-compliant 2-epoch run. The held-out acceptance is monotone and **still climbing steeply at the cap** (chance ≈ 4e-6) — viability is demonstrated, but 0.248@28-steps is too weak to anchor the full-scale go/no-go.

**Revised steer:** the pod cap has been raised to `SENPAI_MAX_EPOCHS=9999`, so the student should run the intended 1000-step debug training directly, using a corpus broad enough to avoid a public-slice memorization artifact. Terminalize with a defensible `tf_acceptance_rate_debug_1k`. Serving/full-scale remain gated on (a) this number and (b) the int4 spec greedy-identity linchpin (#19). EAGLE-3 is the highest-ceiling drafter (lit. ~480–550 TPS) and is deployable on the public VALID frontier's drafter (`e1`) spec path independent of the int4 linchpin.

## 2026-06-13 (cycle 8) — PR #7: fa2sw + onegraph runtime levers ✗ CLOSED (negative)

- **Branch:** `denken/fa2sw-onegraph`
- **Student:** denken
- **Status:** CLOSED — rigorous, well-isolated NEGATIVE. Both runtime levers are dead ends standalone on the int4 base at conc=1. Knowledge preserved here and in BASELINE.md "Confirmed dead ends."
- **Hypothesis:** fa2sw (route 35× hd-256 sliding-window local layers to FlashAttention-2) + onegraph (`cudagraph_mode=FULL`) erase per-step overhead at conc=1, enabling a TPS gain over the int4 base without drafter or lmhead changes.

### Results

| variant | TPS (local, conc=1) | Δ vs base | greedy (official verifier, 128-prompt) |
|---|---|---|---|
| base (int4 QAT W4A16) | **96.89 ±0.01** | — | REFERENCE |
| fa2sw only | 92.11 ±0.02 | **−4.9%** | **DIVERGENT** 82/128 (12,075 tok) |
| onegraph only | 96.82 ±0.00 | ~0% (parity) | **DIVERGENT** 1/128 (59 tok, @idx 197) |
| both | 92.12 ±0.00 | **−4.9%** | **DIVERGENT** 82/128 (11,767 tok) |

**W&B run:** `57bb3a6s` — ablation matrix table + per-variant metrics.

### Analysis

Both levers **fail the strict zero-tolerance greedy gate**, so neither can ship standalone regardless of TPS:
- **fa2sw:** FA2 sliding-window numerics ≠ Triton → near-tie argmax flips on 82/128 prompts. The mixed FA2+Triton backend also *blocks* a single full-graph capture, producing the −4.9% TPS regression.
- **onegraph:** A pure graph-capture knob (`cudagraph_mode=FULL`) still perturbs the numeric path (one near-tie argmax flip) — confirms the "different numeric path even from a pure graph-capture knob" warning.
- **fa2sw dominates** — `both` == fa2sw's divergence set; onegraph's addition doesn't expand the failure set.

**Root cause of no TPS win:** Decode at conc=1 is **~92% weight-GEMM / bandwidth-bound** (attn ≈2.6%, sampling ≈0.2%). The existing CUDA graph already collapses the decode step into one launch. There is **no per-step overhead left to reclaim** standalone at conc=1. This closes the "per-step overhead gap" hypothesis for these two levers.

**Determinism control (bonus finding — 4th int4 greedy-determinism reconciliation data point):**
Int4 base is **cross-process bit-exact** (sha256 `base_clean`==`base_clean2`, also deterministic in eager mode). The divergences above are a real mechanism, not run noise. This is the clearest data point yet: int4 base greedy **IS gate-valid in M=1 sequential prefix-cache-OFF**, narrowing the linchpin to the *spec M=K+1 batched-verify path* specifically.

**fa2sw serving caveat:** fa2sw cannot be served via a serve-process monkeypatch — vLLM V1 spawns a separate EngineCore process; a real fa2sw serve path requires a **vLLM worker-plugin** entry point. Moot since it's invalid, but prevents wasted re-discovery.

### Suggested follow-up (from denken, evaluated by advisor)
fa2sw layered *on top of the MTP drafter* (where attention share under spec verify may be higher) — valid direction but drafter-gated (kanna #5 linchpin). Assigned denken the hardware-grounded TPS ceiling curve instead (PR #18: decode-step cost model vs K), which directly quantifies when attention-share rises enough for fa2sw to matter.

---

## 2026-06-13 11:15 — PR #15: EAGLE-3 feature-export feasibility ✓ MERGED

- **Branch:** `fern/eagle3-feature-export-feasibility`
- **Student:** fern
- **Status:** MERGED — binary feasibility verdict: ACCESSIBLE → GO. Research report + reusable probe script. No TPS change; foundational prerequisite for the highest-ceiling drafter path.
- **Hypothesis:** Multi-layer intermediate hidden states from Gemma-4 E4B ARE accessible from vLLM 0.22.0's model executor (either natively or via a minimal model-class override).

### Results

| field | value |
|---|---|
| `eagle3_hiddens_accessible` | **1 (yes, natively)** |
| Access mechanism | Built-in `SupportsEagle3` interface — zero patching |
| Model-class override effort | **0 hours** (already implemented) |
| Aux layers (default) | `(2, 21, 39)` over the 42-layer E4B body |
| Aux shape/dtype | `[num_tokens, 2560]` bf16 per layer |
| CUDA-graph compatible | **Yes** (persistent buffers pre-allocated at capture) |
| Drafter head arch | Already exists: `llama_eagle3.py`, `v1/spec_decode/eagle.py` |
| W&B run | None (source audit + single model-load probe) |

**Empirical probe (PR #15 `probe_result.json`):** `supports_eagle3=True`, `default_aux_layers=[2,21,39]`, 3 tensors `[5,2560]` no NaN; vision+audio towers intact; 15.3 GiB peak bf16 on A10G.

**Key vLLM source refs (vLLM 0.22.0):**
- `model_executor/models/interfaces.py:1285-1392` — `EagleModelMixin` + `SupportsEagle3` Protocol
- `gemma4_mm.py:917-923` — `Gemma4ForConditionalGeneration implements SupportsEagle3`
- `gemma4.py:958` — `Gemma4Model is EagleModelMixin` (42 layers)
- `v1/worker/gpu_model_runner.py:4861-4987` — concatenates 3 aux layers `dim=-1` (that's the EAGLE-3 multi-layer fusion)
- `v1/worker/gpu/cudagraph_utils.py:382-395` — persistent aux buffers for CUDA-graph safe capture

**Serving-validity gate:** greedy-identity of EAGLE-3 spec decode on int4 is gated on kanna #5 linchpin (int4 batched-verify greedy-validity).

### New shared infra
`research/eagle3_feasibility/{feasibility_report.md, probe_eagle3_export.py, probe_result.json, probe.log}`

### Recommendation → GO
Full EAGLE-3 drafter head training assigned to fern (PR #16). Literature projects **480–550 TPS** at ~4–5+ accepted tok/step. Serving run gated on kanna #5 linchpin.

---

## 2026-06-13 10:45 — PR #13: SAM-Decoding drafter-overlap intersection analysis ✓ MERGED

- **Student:** fern
- **Status:** MERGED — CPU-only infra extension to `analyze_suffix_budget.py`. No TPS change; shared tooling for net-headroom decision.
- **What was built:** `--drafter-trace <file>` extension; `drafter_overlap` block with `net_sam_beyond_drafter_frac` (the GO/marginal/retire decision number); 13/13 mock tests pass; no-drafter path byte-identical (regression-safe). Canonical trace format (`output_start` for spec interleave alignment). `research/sam_drafter_overlap/overlap_analysis_template.json`. Dev dep `pytest>=8` added.
- **Metrics:** `sam_causal_frac_gt_k8_base_reproduced=0.0893` (PR #10 anchor), `mock_tests_passed=13`.
- **Net-headroom thresholds:** `net_frac > 3%` → Triton kernel GO; `1–3%` → marginal; `< 1%` → retire SAM.
- **Caveat (fern):** real MTP drafter concentrates acceptances on predictable/repetitive spans — exactly where SAM runs live — so real overlap likely HIGHER → real net LOWER than naive intuition. Base 8.93% is small; brace for marginal/retire.
- **Next:** tool ready; trace landing depends on kanna's linchpin outcome (PR #5 → real acceptance trace gated on greedy-validity resolution).
- **Reproduce:** `cd target/ && uv run python -m pytest scripts/tests/test_drafter_overlap.py -v`

## 2026-06-13 10:45 — PR #14: Empirical lmhead12k (pruned-weights top-12k vocab) — IN PROGRESS (non-terminal, blocked)

- **Student:** ubel
- **Status:** NON-TERMINAL (`terminal=false`, `status=blocked_local_gpu`) — sent back to WIP with advisor answers. GPU void on pod (intermittent); int4 base checkpoint not on node. Implementation complete (CPU feasibility done, GPU steps pending).
- **Key findings (change the plan):**
  1. **12k underspecified:** 128 benchmark prompts have only 7,338 unique tokens — can't frequency-fill to 12,288 from the benchmark alone. Tight kept set = 7,584 (34.6× bandwidth). Must use a general corpus to reach 12,288 faithfully.
  2. **Hard-include public GT tokens is NECESSARY:** official PPL scorer (`ppl_endpoint.py:163-183`) does NOT floor −∞ for out-of-vocab tokens → GT target token outside kept vocab → −∞/missing → gate fail. The tight set is intrinsically public-tailored; would fail private PPL re-run. General-12,288 cut is required for private validity.
  3. **Only 31/128 decode captures available locally** (fern's 128-capture gitignored, not on scratch bucket); greedy-identity proven on 31 only.
- **Serving design (correct):** custom vLLM model class `Gemma3ForCausalLMLMHead12k` — scatters kept-row logits into full 262,144 (−∞ on pruned) inside `compute_logits` (VOCABTRIM-style); `LogitsProcessor` path insufficient (V1 reads `prompt_logprobs` before logits processors).
- **Advisor answers:** self-build int4+g128 base via path-(a) (prune bf16 → quantize, deterministic from public source, no cross-node dep); build general-12,288 cut from broad STEM corpus; regenerate full 128 decode capture; report both bandwidth numbers.
- **Note: DRAFTER-INDEPENDENT** — not affected by kanna's spec-decode linchpin. Building block toward ~420 regardless of linchpin outcome.

## 2026-06-13 10:30 — PR #5: int4 + MTP/QAT drafter spec-decode ({8,4} engine fix + greedy-validity finding) — REQUEST CHANGES (→ WIP)

- **Branch:** `kanna/int4-mtp-drafter`
- **Student:** kanna
- **Status:** REQUEST CHANGES — terminal SENPAI-RESULT but submission **INVALID** (greedy DIVERGENT). Sent back to WIP for a decisive precision-localization experiment. The `{8,4}` backport + wandb-scraper fix are keepers on the branch.
- **Hypothesis:** int4 W4A16 target + QAT-MTP drafter spec-decode reaches ~285 TPS greedy-identical once the vLLM 0.22.0 `{8,4}` attention-group blocker is fixed.

### Results (local A10G, exploratory; W&B group `int4-mtp-drafter`)

| K | mean accepted tok/step | exploratory TPS (A10G) | PPL | greedy | W&B run |
|---|---|---|---|---|---|
| 5 | 2.151 | 164.45 | 2.0064 | DIVERGENT | zbt1fras |
| 6 | 2.197 | 163.87 | 2.0064 | DIVERGENT | 7vnkis8z |
| 7 | 2.188 | 160.28 | 2.0064 | DIVERGENT | 0fa5c8fx |

W&B cross-check (advisor): tps/ppl/accept match the PR verbatim; `greedy_identical=0` boolean = DIVERGENT confirmed; the malformed `spec/accept_rate_posN` values are the pre-fix scraper bug kanna disclosed and fixed.

### Engineering win — `{8,4}` blocker SOLVED
Backported upstream vLLM PR #43543 / commit `dede691c9536` ("split attention groups by `num_heads_q` for spec-decode drafts") as a fork/spawn-safe runtime monkeypatch (`vllm_attn_group_patch.py` + `sitecustomize.py`). Serves cleanly eager + cudagraph. (The PR-cited commit `3e8afdf7` is WRONG — that's a Cohere2MoE fix; the real fix is #43543.)

### CRITICAL FINDING — int4 spec-decode is structurally greedy-DIVERGENT in vLLM 0.22.0
At temp=0 vLLM's rejection sampler emits `argmax(target_logits)` from the **batched M=K+1 verify forward**; plain AR (the reference) emits `argmax` from the **M=1 decode forward**. int4 Marlin accumulation is batch-shape-dependent → logits differ in the last bits → ~0.33%/token argmax flips on near-ties → compounds to DIVERGENT over 512 tokens (6/32 prompts identical). Structural for any K≥1; no batch-invariant/deterministic knob exists in 0.22.0 (kanna grep-confirmed). K0-vs-K0 control is IDENTICAL → divergence is 100% the spec verify path.

### Advisor verification of the gate mechanics (this cycle)
- Read the official verifier (`gemma_greedy_identity_verifier_flowian-powers/greedy_identity.py`): **strict bit-exact**, full `completion_token_ids`, zero tolerance — any 1 flipped token → DIVERGENT.
- Traced the harness (`speed_benchmark/scripts/{hf_bucket_single_job,decode_outputs}.py`): it generates ONLY the candidate decode (128×512, seed 1, temp 0, ignore_eos); the **reference is organizer-held** = "plain greedy decode of the submitted checkpoint" = int4 M=1 AR — exactly what kanna compared against. **kanna's DIVERGENT is very likely the official verdict.** Refutes her hypothesis (c) "audit is lenient."

### LINCHPIN question (gates rungs 4–5 / the path to 420)
If int4+vLLM-spec cannot be greedy-valid in 0.22.0, how is the ~420 frontier VALID? Remaining hypotheses: **(a)** higher-precision target (fewer near-tie flips, but can't hit 420 at int4 bandwidth) or **(b)** batch-invariant kernels in a newer vLLM (only if the harness honors manifest `python_packages`). **Next experiment (assigned to kanna):** hold the spec stack fixed, vary target precision (int4 vs bf16 vs fp8), measure greedy flip-rate per arm — localizes the divergence and decides whether the drafter ladder is salvageable. Plus: definitively confirm whether a10g-small honors the manifest vLLM version.

### Secondary
Acceptance underdelivers: 2.20 tok/step (vs ~3.3 target) — strong pos0 (87%) but steep decay caps speedup ~2.2× (~270 effective TPS). Real-prompt corroboration: K6 340.9s vs K0 730.2s = 2.14×.

## 2026-06-13 10:30 — PR #9: Wide-distribution KL-distilled drafter (private-stable acceptance) — REQUEST CHANGES (→ WIP)

- **Branch:** `land/wide-drafter-distill`
- **Student:** land
- **Status:** REQUEST CHANGES — tf-gate PASSES but native serving regressed; sent back for v1 (free-running schedule). Drafter infra + deduped corpus are keepers on the branch.
- **Hypothesis:** A wide, distribution-matched (4-dist) KL-distilled drafter lifts acceptance uniformly — including the chat/private-proxy floor — improving private-set stability over the reasoning-skewed stock drafter.

### Results (offline acceptance, held-out shard; committed JSONs `research/wide_drafter/eval/{stock,wide}.json`)

| metric | stock | wide (v0) | Δ |
|---|---|---|---|
| tf accepted-tok/step (the gate), overall | 3.455 | 3.811 | **+0.356 (+10.3%)** |
| tf — chat (private proxy) | 2.753 | 3.052 | **+0.299 (+10.9%)** |
| native `generate(assistant_model=)` overall | 3.553 | 3.388 | **−0.165 (−4.6%)** |

W&B run `eqqdeodf` (group `wide-drafter-distill`). **Reporting gap (advisor W&B check):** the cited run logged only `train/*` loss curves — the acceptance numbers live in committed JSONs + reproduce commands, NOT in W&B. v1 must log the heldout eval to W&B.

### Analysis
- Width corpus works on the metric it optimizes: +10.3% tf, **uniform incl. chat/private-proxy floor (+10.9%)** — the target signal. Dedup proof: zero overlap with the 128 public prompts.
- **Native regressed −4.6%, uniformly** — train↔serve schedule mismatch (teacher-forced training vs free-running serving) + undertraining (0.87 epoch, 40 of 90 budget-min unused, losses still falling). Correctly diagnosed by land.

### Next (v1, assigned to land)
Change ONE variable: **free-running / scheduled-sampling (EAGLE-3-style) unroll** to close the exposure-bias gap; same ~5k corpus + recipe; full ~82-min budget; primary = `heldout_native_accept_per_step` (beat stock 3.553); log eval to W&B. Optional 2nd arm: narrow-corpus contrast to isolate the width variable.

### Infra/methodology notes
- `scripts/drafter/offline_eval.py` is the correct EAGLE-aware acceptance tool (the reference `shared_resources/.../offline_acceptance.py` mis-measures EAGLE drafters as standalone CausalLM — flagged to wirbel #8).
- `google/gemma-4-E4B-it-assistant` is the correct control; `Tonykip/...` baseline didn't resolve (fine). hf_xet wedge → `HF_HUB_DISABLE_XET=1`.
- Coupling: converting acceptance → served TPS depends on int4 spec being greedy-valid (kanna #5's linchpin question).

## 2026-06-13 10:00 — PR #6: Greedy-safe vocab-prune / top-k sparse-verify (verify-cost lever) ✗ CLOSED (negative)

- **Branch:** `ubel/vocab-prune-sparse-verify`
- **Student:** ubel
- **Status:** CLOSED — confirmed dead end (provable Cauchy-Schwarz certificate, 0%-fire on Gemma4 geometry). Option A authorized: empirical lmhead12k (new PR incoming).
- **Hypothesis:** A Cauchy-Schwarz sufficient certificate determines per decode step whether the greedy
  argmax is within the top-K kept set — allowing the step to skip the full 262k GEMM if certified,
  with a greedy-safe adversarial fallback when not.

### Results (measured on A10G, K=12000, 64 prompts × 256 tokens = 16,384 decode steps)

| metric | value | verdict |
|---|---|---|
| Certificate fire rate | **0.0%** (0 / 16,384 steps) | dead end |
| Fallback rate | **100%** | always pays full 262k GEMM |
| Isolated lm_head GEMM speedup (12k vs 262k kept) | **20.1×** | ceiling for the empirical approach |
| Effective speedup with cert overhead | **0.92×** (−8% slower) | provable lever LOSES |
| TPS (net) | null (slower than baseline) | — |
| PPL (128/128 GT records, 61,797 tokens) | 2.304 | ≤ 2.42 ✓ |
| Greedy identity (128 public prompts) | GREEDY_IDENTICAL (trivially — 100% fallback) | ✓ |
| Adversarial fallback (rare-token test) | PASS (cert correctly refuses → full GEMM emits true argmax) | ✓ |
| Unit tests | 7/7 PASS | ✓ |
| W&B run | none | — |

### Root cause — model-intrinsic geometry obstruction

`R_complement_max_norm = 1.630` vs real `z_max/||h|| ≈ 0.59` → the Cauchy–Schwarz sufficient
condition **provably cannot fire** on real Gemma4 hidden states. The model has flat row norms, tiny
kept-vs-pruned margins, and a near-full-rank embedding. No kept-set construction rescues the cert
on this lm_head. The **Cauchy-Schwarz provable-greedy-cert family is a confirmed dead end on
`gemma-4-E4B-it`**.

### Key program finding

The frontier's `lmhead12k` (kenyan-duma, 421.12 TPS VALID) is the **empirical prune**: compute
only top-12k logits, emit the kept-argmax, **no per-step certificate**. It captures the ~20×
isolated GEMM speedup. It is NOT adversarially safe — the rare-token case diverges (ubel measured
this: id 258090 outside 12k → kept-only emits 188798). It passes the official greedy-identity
check because benchmark prompts apparently do not generate rare tokens. The empirical approach is
what the leaderboard rewards; the provable approach cannot compete on this geometry.

**On this lm_head: provable safety OR TPS win — not both.**

### Decision

- Provable greedy-safe cert (Cauchy-Schwarz) on Gemma4: **DEAD END**. Added to BASELINE.md.
- **Option A authorized:** build the pruned-weights empirical `lmhead12k` checkpoint (top-12k
  rows of the int4+g128 lm_head), serve it, measure TPS/PPL/greedy-identity + rare-token divergence
  rate. New PR for ubel: `empirical-lmhead12k`.

---

## 2026-06-13 09:45 — PR #10: Offline suffix-run token-budget analysis for SAM-Decoding feasibility ✓ MERGED

- **Branch:** `fern/sam-decoding-offline-analysis`
- **Student:** fern
- **Status:** MERGED (`c8dfdb3`) — analysis deliverable + shared infra (`scripts/analyze_suffix_budget.py`).
- **Hypothesis:** The SAM-Decoding paper (arXiv 2411.10666) claims a 3.6–3.9% verbatim-suffix-run
  budget on reasoning prompts. Confirm on our 128 benchmark prompts; produce a go/no-go for the
  Triton in-graph suffix-match kernel (Rank 5 from round-2 research).

### Results

| budget definition | K>4 | K>6 | **K>8** | K>10 | verdict (K>8) |
|---|---|---|---|---|---|
| `m(t)` (PR spec; adjacent-only, non-causal) | 1.47% | 1.37% | **1.21%** | 1.14% | no-go (flawed proxy) |
| **Causal SAM realized** (actionable, greedy-safe) | 15.37% | 11.60% | **8.93%** | 7.16% | **GO** |
| ↳ causal decode-steps-saved (TPS-correct) | 13.74% | 10.66% | **8.35%** | 6.77% | — |
| LPF forward-oracle (loose upper ref) | 30.56% | 21.37% | 16.21% | 12.42% | — |

**Per-dataset causal K>8:** aime2026 10.74% | gpqa_diamond 9.23% | mmlu_pro 8.19% (uniform 8–11%).

SENPAI-RESULT: `{"terminal":true,"status":"complete","frac_tokens_gt_k8":0.0121,"causal_sam_realized_frac_gt_k8":0.0893}`

**Decision metric:** causal_sam_realized_frac_gt_k8 = **8.93%** → **GO** (>3.6% threshold).
`frac_tokens_gt_k8` (0.0121) is the literal PR-spec `m(t)` value — documented but *not* the decision metric.

### Key points

- **`m(t)` is a flawed proxy:** fires only on adjacent-period repetition (the s tokens immediately before t
  reappearing at t). Only 127 such runs across all 128 prompts (~1/prompt). The exploitable structure is
  non-adjacent — prompt re-quotes, formula restatements, repeated option text — which `m(t)` cannot see.
- **Causal estimate validated:** cross-checked against brute-force O(n²) causal reference: 0 mismatches
  over 600 positions. Robust to nondeterminism: 10.51% (PR #2's 16-prompt capture) vs 10.49% (this
  run's first 16 prompts) — Δ0.02pp.
- **Greedy-safe:** SAM-Decoding verifies each drafted token against live target logits → greedy-safe by
  construction → zero PPL risk.
- **Critical caveat:** the ~420 TPS frontier already runs an MTP/QAT model-drafter (~3.3 tok/step).
  SAM adds to it; the incremental gain = causal budget MINUS drafter-accepted positions. Net headroom
  can only be measured by intersecting causal suffix runs with the drafter's per-step acceptance trace
  (needs kanna's #5 to serve). This is the de-risking step before the Triton kernel build.

### New shared infra

`scripts/analyze_suffix_budget.py` — offline CPU-only suffix-budget analyzer. Designed for extension
with a `--drafter-trace` flag to intersect causal suffix runs with a drafter acceptance trace and
output the net incremental headroom.

**W&B run:** none (CPU-only offline analysis). 128/128 prompts captured (bf16, 43.94 TPS local).
**Artifacts:** `research/local_validation/suffix_budget/suffix_budget_analysis.json` (committed).

### Next steps

- **fern** extends `analyze_suffix_budget.py` with drafter-overlap intersection + synthetic mock-trace
  validation (non-blocked, CPU-only). Once kanna's #5 drafter serves and emits an acceptance trace,
  the net-headroom number is one command away.
- If net_headroom > 3%: assign Triton in-graph suffix-match kernel PR.
- If net_headroom < 1%: SAM direction adds near-nothing to the drafter stack — retire.

---

## 2026-06-13 09:30 — PR #4: int4 g128 + untied int4 lm_head re-quant (~127 TPS weight floor) [IN PROGRESS — awaiting HF Job]

- **Branch:** `lawine/int4-g128-lmhead`
- **Student:** lawine
- **Status:** WIP — local evidence complete; **awaiting human approval of HF Job (GitHub issue #12)**
  before posting terminal SENPAI-RESULT with official a10g-small numbers. Held at the int4 (PR #3)
  rung deliberately: the ladder is confirmed bottom-up and, per BASELINE.md, local A10G numbers are
  exploratory only — no merge to a confirmed TPS rung without the official a10g-small score.
- **Hypothesis:** Re-quantizing the QAT base (`gemma-4-E4B-it-qat-q4_0-unquantized`) to group_size=128
  across all 343 body modules plus an **untied int4 `lm_head`** (`embed_tokens` kept bf16) hits the
  int4-Marlin Ampere **weight-byte floor**, lifting single-stream TPS from the ~95 int4 base to ~127
  with PPL essentially unchanged (~2.02). This is the last "fewer weight-bytes/token" lever before
  sub-4-bit (a confirmed sm_86 dead end).

### Local Results (exploratory, A10G — NOT official a10g-small)

| metric | value | gate | pass? |
|---|---|---|---|
| Local PPL (served, 128/128 GT records, 61797 tokens) | **2.0190** | ≤ 2.42 | ✓ |
| Offline fake-quant PPL | 2.0197 | ≤ 2.42 | ✓ |
| Local TPS (exploratory, A10G, single-stream) | **127.99** | — | on target ~126.8 (+33% over int4 base ~96) |
| Greedy identity (official served-vs-served, standard cap=512 config) | **GREEDY_IDENTICAL** 128/128 prompts, 16384/16384 tok, 0 divergent | byte-exact | ✓ |
| Quantized modules | 343 body @ g128 + untied int4 lm_head = 344 total, 9.62 GiB on disk | — | ✓ |
| compressed_tensors version | 0.15.0.1 (vLLM 0.22.0's shipped version) | — | ✓ (see note) |
| All modalities | vision/audio loaded | — | ✓ |
| W&B run | `0pxj6n63` (`wandb-applied-ai-team/senpai-v1`, finished) | — | ✓ corroborates tps 127.99 / ppl 2.019 / GREEDY_IDENTICAL, logged verbatim |

### Key points

- **TPS lever:** 127.99 local = +33% over the int4 base (~96 local) and +0.9% above the ~126.8 public
  ladder target — confirms the int4-Marlin weight-byte floor on Ampere. group_size 128 + untied int4
  `lm_head` is the last weight-bytes/token reduction available (sub-4-bit AWQ/GPTQ/etc. have no
  loadable sm_86 kernel in vLLM 0.22 — confirmed dead end in BASELINE.md). lawine's track is at its
  natural floor; the next lever above this rung is the drafter (kanna #5 / land #9), not more quant.
- **Greedy identity (same resolution as stark's PR #3):** the official gate is served-vs-served at a
  SHARED config. lawine proved **GREEDY_IDENTICAL 128/128 at the standard cap=512 config**; spurious
  divergence only appears under cross-config (no-cap reference vs cap=512 candidate). Not a blocker.
- **Version note:** the PR body states compressed_tensors==0.10.2 but lawine actually built against
  **0.15.0.1** — the version vLLM 0.22.0 ships. 0.15.0.1 is the correct/required choice; 0.10.2 is
  incompatible with vLLM 0.22.0. Acknowledged on the PR; the built checkpoint is the valid artifact.
- **PPL-metric note (reusable):** the scored gate metric is the token-weighted `served_ppl=2.0190`
  (`exp(Σnll/Σtok)` over all 61,797 tokens). The W&B run also logs an unweighted per-record mean
  `served_mean_record_ppl=2.1787`, which runs higher because short records weigh equally — it is
  informational only, not the contract metric, and both are under the 2.42 gate.

### Next Steps

- Human approves GitHub issue #12 → lawine runs
  `python train.py --submission submissions/int4_g128_lmhead --name int4-g128-lmhead --launch --wait`
- Official a10g-small TPS/PPL confirmed → lawine posts terminal SENPAI-RESULT to PR #4
- Advisor merges PR #4 → updates ladder (int4 g128/lmhead weight-floor rung officially confirmed, ~127)
- lawine's weight-quant track is then complete → pivot lawine to a fresh frontier lever next round

---

## 2026-06-13 09:00 — PR #3: Reproduce int4 QAT W4A16 leader (~95 TPS) [IN PROGRESS — awaiting HF Job]

- **Branch:** `stark/int4-qat-w4a16`
- **Student:** stark
- **Status:** WIP — local evidence complete; awaiting human approval of HF Job (GitHub issue #11)
  before posting terminal SENPAI-RESULT with official a10g-small numbers.
- **Hypothesis:** Stock vLLM 0.22.0 Marlin int4 W4A16 endpoint on `google/gemma-4-E4B-it-qat-w4a16-ct`
  reproduces the ~95.4 TPS / PPL ~2.01 VALID leader. The dominant lever: int4 weight quantization
  reduces bandwidth by ~4×, lifting TPS from 44 → ~95 with better PPL (QAT-trained).

### Local Results (exploratory, A10G — NOT official a10g-small)

| metric | value | gate | pass? |
|---|---|---|---|
| Local PPL (128/128 GT records) | **2.0055** | ≤ 2.42 | ✓ |
| Local TPS (exploratory, A10G, 32 prompts) | **95.99** | — | on target ~95.4 |
| Marlin kernel | `MarlinLinearKernel for CompressedTensorsWNA16` | — | ✓ confirmed |
| All modalities | vision/audio encoder cache initialized | — | ✓ |
| CUDA graphs | `FULL_AND_PIECEWISE`, no eager fallback | — | ✓ |
| Peak GPU memory | ~21.1 GiB / 23 GiB | — | no OOM |
| W&B run | none (serving task, no training) | — | — |

### Key Finding — Greedy-Identity Nondeterminism

stark discovered that the int4+vLLM endpoint is **run-to-run nondeterministic** for greedy decode
at output_len=512: Marlin split-K GEMM / Triton-attn FP non-associativity introduces ~1 ULP noise
at near-tie logit positions, cascading to token-flip divergences at a handful of hotspots (idx 83,
104 consistently). Cross-path comparison (HF bf16 dense GEMM vs vLLM Marlin int4) always diverges
— different arithmetic paths.

**Advisor ruling:** NOT a blocker. The as-is stock int4 Marlin leader (~95.4 TPS, same stack) is
VALID on the official leaderboard. This submission IS that stack. Within-stack greedy identity
(same vLLM endpoint, same job run) is consistent; the official harness compares decode_outputs.jsonl
generated from the same serving instance. Determinism study deferred — not needed for this rung.

### Next Steps

- Human approves GitHub issue #11 → stark runs `python train.py --submission submissions/int4_qat --name int4-qat --launch --wait`
- Official a10g-small TPS/PPL confirmed → stark posts terminal SENPAI-RESULT to PR #3
- Advisor merges PR #3 → updates ladder (int4 rung officially confirmed)

---

## 2026-06-13 08:40 — PR #2: Resolve PPL artifact path + validate bf16 baseline locally

- **Branch:** `fern/vllm-baseline-ppl-resolution`
- **Student:** fern
- **Hypothesis:** Before spending HF Jobs quota on speed work, definitively explain why the prior
  bf16 smoke job (`6a2c5fb77c68f455eff14260`) produced `tps=44.018` but no confirmed
  `ppl_summary.json`. Prove the PPL and decode contracts against a local endpoint, deliver a
  reusable one-command local pre-validation harness, and confirm the `MAX_NUM_BATCHED_TOKENS=512`
  OOM-safety hypothesis on the longest GT context (2431 tokens). Research priority #1.

### Results

| metric | value | gate | pass? |
|---|---|---|---|
| Local PPL (128/128 GT records) | **2.3012** | ≤ 2.42 | ✓ |
| GT records completed | 128/128 | 128/128 | ✓ |
| PPL contract (`prompt_logprobs` on integer-ID prompt) | proven | — | ✓ |
| Decode contract (`choices[0].token_ids` len 512) | proven | — | ✓ |
| OOM safety (longest ctx=2431 tokens at `MAX_NUM_BATCHED_TOKENS=512`) | +560 MiB transient (< 0.5 GiB budget) | no OOM | ✓ |
| Root cause of missing artifact | 40-min HF Job timeout | — | identified |
| W&B run | none (local validation task) | — | — |

### Root Cause — Definitive

The 40-min HF Job wall-clock cap killed the job before PPL ever started. Timeline:

| stage | duration | cumulative | status |
|---|---|---|---|
| Cold startup (model load + torch.compile + CUDA-graph capture) | 11.9 min | 11.9 min | completed |
| Benchmark stage (128 prompts, decode, tps measurement) | 24.8 min | 36.7 min | completed |
| Decode capture (same 128×512 workload) | ~24.8 min est. | 61.5 min | **killed @ 40 min** |
| PPL stage (runs *after* decode) | n/a | n/a | **never reached** |

Evidence from preserved artifacts (`research/local_validation/prior_job_6a2c5fb77c68f455eff14260/`):
- `job_status.json` → `status:timed_out`, `stage:RUNNING`, `timeout_minutes:40` → rules out OOM (clean wall-clock stop)
- `run_environment.json` → `ppl.enabled:true` → rules out disabled
- `summary.json` → `duration_s:1488.8` (benchmark alone = 24.8 min) → rules out unfetched

**Implication:** at 44 TPS the bf16 baseline cannot fit startup+benchmark+decode+PPL in 40 min. All
faster submissions (≥95 TPS) will fit comfortably. The local harness (below) provides a timeout-free
gate.

### OOM-Safety Confirmation

Longest GT record (`gpqa_diamond-1d37a7a51d`, ctx=2431, tgt=512, combined=2943 tokens): HTTP 200 +
valid `prompt_logprobs` (len 2943). Peak GPU: 21009 MiB (+560 MiB transient). Theoretical chunked
bound: 512 positions × 262,144 vocab × 4B = 0.50 GiB. Confirms `MAX_NUM_BATCHED_TOKENS=512`
chunked prefill bounds the `log_softmax` peak as predicted in DATASET_ANALYSIS.md.

### New Shared Infrastructure

`scripts/local_prevalidate.py` — one-command local pre-validation gate:
```bash
cd target/ && VLLM_USE_FLASHINFER_SAMPLER=0 \
  python scripts/local_prevalidate.py --submission submissions/vllm_baseline --decode-num-prompts 16
# → SENPAI-LOCAL tps=44.0056 ppl=2.3012 completed=128
```

**All students should run this against their submission before opening an HF Job approval issue.**

### Local-Environment Note

FlashInfer JIT is broken on this node (CUDA 13.2 nvcc vs. vendored libcudacxx). Workaround:
`VLLM_USE_FLASHINFER_SAMPLER=0`. Numerically identical for greedy decode (argmax) and PPL
(logits/log_softmax). Not needed on official a10g-small image.

### Analysis & Conclusions

**Verdict: merge (infra + priority-1 resolution).** Not a TPS improvement but delivers essential
shared infrastructure and closes the highest-priority uncertainty blocking all future submissions.

- The bf16 baseline is correct: PPL ≈ 2.30 exactly matches the reference. The prior smoke job was
  not defective — it just ran out of time.
- The local pre-validation harness (`scripts/local_prevalidate.py`) is now a team-wide gate. Every
  student should PPL-validate locally before requesting an HF Job.
- The OOM-safety analysis confirms DATASET_ANALYSIS.md's `MAX_NUM_BATCHED_TOKENS=512` recipe is
  correct; the longest GT context (2431 tokens) fits within the GPU memory budget.
- The 40-min timeout root cause is important baseline knowledge: the benchmark + decode stages
  together consume ~24.8 + 24.8 = ~49.6 min at 44 TPS, plus ~12 min cold startup ≈ 61.5 min
  total. Any future a10g-small bf16 confirmation needs the timeout cap raised, or the decode
  prompt count reduced. Fast submissions (≥95 TPS) automatically fit in 40 min.

### Suggested follow-up (fern's own note, endorsed)

- Wire `local_prevalidate.py` into the pre-submission checklist (all students: run it locally;
  only request an HF Job once it passes). ← **Done — see "New Shared Infrastructure" above.**
- For an a10g-small bf16 confirmation, fern will open a separate `Approval request: HF job for
  vllm-baseline` issue — not done in this PR (local-only by instruction).

_PR #2 merged to `approval-gated-8gpu-20260613` as squash commit `dd17c17`._
