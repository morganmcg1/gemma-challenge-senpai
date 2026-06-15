# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-15 ~06:48Z (cycle 52q)
- **Advisor branch:** `approval-gated-8gpu-20260613`

## 🆕 Cycle-52q Snapshot — 4 EAGLE-3 economics cards banked; the drafter-trainability blocker may be FLIPPING GREEN

**This turn (06:31, Morgan-merged, all 0-TPS bank-the-analysis): #306 VRAM runtime-peak 20.158 GiB GREEN (axis e/k fully closed, 3.84 GiB headroom) · #307 integration NOT config-only (the onegraph loopgraph is MTP-keyed, goes INERT under method:eagle3; 3 served-file touchpoints T5/T6/T7 need rewrite) · #309 M=8 tree-salvage RELAXES the required raw a1 from 0.9213→0.7731 (GREEN-YELLOW) · #305 GO card: under the conservative scalar ×0.804 the PRIVATE bar binds and projects sub-500 (P=3.9%).**

**★ THE PIVOTAL SYNTHESIS (#309 × #308):** denken #308 declared the a1 target `out-of-reach` (RED) — but against the SUPERSEDED 0.9213 bar. lawine #309's M=8 salvage moved the bar to **0.7731**, which is **+0.0017** from the demonstrated in-repo {2,21,39} EAGLE-3 head (fern #34 `gua9x68j` native step-1 = **0.7714**), INSIDE the published 0.77–0.80 envelope. **The central drafter-trainability blocker may FLIP GREEN.** denken #308 SENT BACK (advisor 06:43Z) to re-issue the verdict against 0.7731, reconcile its own 0.7714, and close the salvage COST loop (the M=8 heavier verify must be NET positive after its verify-row cost).

**★ THE GO/NO-GO REDUCES TO ONE CRUX:** is the conservative ×0.804 private scalar justified (#305 → sub-500), or does it double-count the private drop already inside #300's projection (ρ_priv 0.9421 → CLEARS)? → fern #310 reconciles. Hidden cost: does method:eagle3 retain the 481.53 onegraph base (wirbel #312)? **Plateau hedge — researcher-agent RETURNED (06:48Z):** of 5 fresh non-EAGLE-3 ideas, only ONE survives the advisor filter as a genuine >500 lever: **🥇 Lookahead Decoding** (training-free, ICML 2024 arXiv 2402.02057) — Jacobi-trajectory n-gram drafting that targets **E[T] directly** and is **greedy-EXACT** (emission = verify argmax preserved, unlike relaxed acceptance), no retrain, vLLM integration exists. This is the strongest plateau "bigger swing" and the **prime candidate for the next idle slot** (open question: does Gemma-4's 128-prompt output have enough n-gram regularity, and does the Jacobi step survive the bridge=0.2147 normalization?). FILTERED OUT: relaxed acceptance (PR #66) is **greedy-UNSAFE** → gated on the human greedy-identity decision (Issues #124/#192), NOT merely the PPL gate the agent assumed; drafter-quant / CUDAGraph.FULL-drafter / SplitK-verify-GEMM (#3/#4/#5) are all **step-side, which is DEFINITIVELY CLOSED <500** (487.7 ceiling) and draft savings are bridge-discounted 4.82× → deprioritized. No idle slot to assign Lookahead this turn (8/8 occupied); banked as TOP next-direction.

---

## 🆕 Cycle-52k Snapshot — PATH-A ANALYTICALLY CLOSED (fern #281 capstone); the sole >500 path is a BUILT drafter raise

**★ THE CAPSTONE TURN: fern #281 (`10necg21`, MERGED) closed Path-A on ALL THREE axes (`path_a_fully_closed=True`) — draft-cut (fern #274), tree-width (denken #271, M*=32=479.6), AND E[T]-raise. No realizable `(E[T]_real, M, step-shave)` cell reaches the honest-500 floor; the deployed 481.53 frontier cannot reach 500 by ANY speculative-decoding lever under the measured constraints. The step-side is fully closed too: the 1218.2µs step is a NORMALIZED unit (denken #278), verify is HBM-bound MLP-dominated (kanna #280), verify SDPA insufficient (wirbel #279, 487.8), prefill 2.85% (ubel #275), draft 95% intrinsic-M=1 (kanna #277).**

**★ THE SOLE RE-OPEN (fern #281): a BUILT public-E[T] raise to ≥4.97 at the deployed step — acceptance-per-candidate, NOT width.** denken #119 proved the LINEAR drafter caps at E[T]=3.8445 even at perfect capacity, so the +1.12 raise from 3.844→4.966 is UNREACHABLE by tuning the current drafter — it requires a structurally non-linear / feature-conditioned drafter. This is the **PLATEAU-PROTOCOL pivot**: all kernel/tree/step/draft-cut levers are exhausted; the only >500 path is a **trained better drafter** (greedy-SAFE by construction — emission = verify argmax). The researcher-agent's **RANK 2 = EAGLE-3 multi-layer hidden-state fidelity** (attacks the j≥2 OOD acceptance collapse directly; `SupportsEagle3` interface ready in the vLLM fork, PR #15) is the prime candidate. A full EAGLE-3 retrain is a TRAINING run → **human-approval-gated** (route via `Approval request: HF job`); a Phase-1 architecture-adaptation viability gate (2h single GPU) is the cheap precursor.

---

## FOUR Decisive Closures This Cycle (52k)

### 1. The linear step is NORMALIZED, not a wall sum (denken #278 — `bu44n30q`)
- Deployed M=1 linear verify = **4966.8µs** (CUDA-event, int4-body HBM-bound) — 4.08× the whole 1218.2µs step.
- `step − draft − verify = −4455.4µs` (unphysical) → the **1218.2µs step is a batch-amortized normalized unit**.
- HBM floor: one int4 forward MUST read 1.76 GB = 2934µs > whole step.
- **bridge = step_norm/step_wall = 0.2147**; a batch=1 WALL draft saving over-credits by **4.82×**. kanna #269 +4.39% → **+0.91% basis-honest**.

### 2. Verify forward is HBM-bound MLP-dominated (kanna #280 — `sdrerk5h`)
- M=8 decomposition: MLP **66.1%** (gate_up 43% @ 71.2% BW + down 23% @ 66.5% BW) / SDPA 14.5% @ 34.9% BW / io+attn 17% / lm_head 2.4%.
- M≥8 batching lifts int4 GEMMs to **59–71% BW → approaching-roofline-intrinsic**; MLP slack reassociation-gated (greedy-UNSAFE).
- Only greedy-safe verify lever = num_stages=2 SDPA = **+1.185%**.

### 3. Verify SDPA tune insufficient standalone (wirbel #279 — `xme9snkv`)
- num_stages=3→2 → **+1.29% (487.8 TPS)**; does NOT clear 500 even at inflated ctx=2048 (497.7).
- Premise correction: served config is **MAX_NUM_SEQS=1 + SPLITKV_VERIFY** → M=8 verify routes to **3D split-KV TILE=16** (global head-512 collapses to 1.018×; sliding head-256 retains 1.093×).
- **Bit-identical 0/128 (maxdiff=0.0)** — banked as a greedy-safe composable micro-lever.

### 4. Prefill denominator CLOSED (ubel #275 — `s26cb1tv`)
- Prefill = **2.849%** of wall at the official 512-token point; precache banks 1.65pp (82% prefix-cache hit). MTP drafter = 0 marginal prefill.
- Wall is decode-dominated (97.15%). Prefill is not a material >500 lever.

---

## Current BASELINE

```
481.53 TPS  (approval-gated-8gpu-20260613, PR #52, fa2sw_precache_kenyan)
PPL 2.3772 · 128/128 completion · λ=1 ceiling 520.95
```

Official target: **500 TPS**. Gap: **−18.47 TPS (−3.835%)**. Private-verified 460.85 (Δ4.3%≤5%).

### Composition anchors (grounded, frozen)
```
official = K_cal · (E[T]/step) · τ = 125.268 · (3.844/1218.2) · 1.218 = 481.53 TPS
K_cal = 125.268 · step = 1218.2µs (NORMALIZED unit) · E[T] = 3.844 (K=7 linear) · τ = 1.218
τ_lo  = 1.03524   (local→official TPS transfer, lawine #267)
τ_acc = 1.0       (local→official acceptance transfer, lawine #276)
τ_ppl = 1.000218  (local→official PPL transfer, lawine #288 — trinity COMPLETE; safe local PPL bar 2.4185)
bridge = 0.2147   (batch=1 wall draft → normalized step over-credit 4.82×, denken #278)
φ_tree = 0.603    (tree-path wall-step fixed-overhead discount, fern #274 — DIFFERENT mechanism)
g_d (deployed) = 0.0191  (tree width, denken #271 → M*=32 = 479.6 < 500)
draft_k7 = 706.9µs · verify(M=8) = 5348µs · safe local λ̂ bar = 0.9855
E[T] floor for honest 500 = 3.9914 (fern #274)
```

---

## Active Roster (cycle 52q, 8/8 GPUs — build-economics matrix FULLY CLOSED on cost/fit/validity; live residual = drafter TRAINABILITY + the unpriced deployment loopgraph-rewrite COST)

| Student | PR | Hypothesis | Owner | Status |
|---------|-----|-----------|-------|--------|
| denken  | #308 | EAGLE-3 a1-cliff trainability — RE-EVAL vs the #309-relaxed **0.7731** bar (is 0.92 inside the published EAGLE-3 first-token envelope or an INTRINSIC pos-1 floor? reconcile own in-repo 0.7714 + close the M=8-salvage COST loop) | me | 🔄 WIP (SENT BACK 06:43Z — original RED was vs the SUPERSEDED 0.9213 bar) |
| fern    | #310 | Does E[T]=6.11 clear PRIVATE 500 under the per-position model (not the scalar ×0.804)? (reconciles fern #305's conservative-scalar "private sub-500" with lawine #300's per-position rho_priv_e3=0.9421 / private-500-needs-E[T]≈4.19 — the GO/NO-GO literally flips on which private model) | Morgan | 🔄 WIP (reseat; #305 MERGED 06:31 → GO-card rollup, private binding under ×0.804) |
| ubel    | #311 | EAGLE-3 #101 launch risk is capture-SIZE dispatch, not VRAM — price it (audit deployed `cudagraph_capture_sizes` vs M={8,16,32} tree widths; deployed M=8 clears size-16, M=32 re-enters the IndexError regime) | Morgan | 🔄 WIP (reseat; #306 MERGED 06:31 → runtime VRAM peak 20.158/3.84 headroom, axis (k) CLOSED at GREEN) |
| wirbel  | #312 | Price the served-file loopgraph rewrite EAGLE-3 requires (#307 found swap_is_config_only=0; scope T5/T6/T7 in sitecustomize.py + the EAGLE-on-eager fallback TPS floor + the #272 guard co-edit — the unpriced DEPLOYMENT-cost axis) | Morgan | 🔄 WIP (reseat; #307 MERGED 06:31 → integration is a served-file change, axis INTEGRATION-READINESS CLOSED at YELLOW) |
| lawine  | #313 | Pre-register the read-only rank-coverage probe that flips #309's YELLOW (adapt wirbel #79's RANKPROBE for a fusion draft; derive the exact frac_true_beyond_top4 threshold that keeps the relaxed a1 demand <0.92; dry-run reproduces #79's cov₄=0.6532) | Morgan | 🔄 WIP (reseat; #309 MERGED 06:31 → tree relaxes a1 demand 0.92→0.7731, axis TREE-SALVAGE CLOSED) |
| kanna   | #294 | EAGLE-3 Phase-1 viability gate (the cheap-proxy GO threshold before the human-gated retrain) | Morgan | 🔄 WIP |
| stark   | #298 | Free-ceiling wall-clock realization (does the banked 487.7 free step ceiling REALIZE on the host-to-host wall, or over-credit like static-K?) | me | 🔄 WIP |
| land    | #245 | Tree fidelity build — Morgan banking (latest marker `terminal:false, pending_arms:true` → NOT mergeable; left alone) | Morgan | 🟡 banking (non-terminal) |

*(Roster shared with the parallel open2 advisor — re-survey live PR state before every assignment/merge.)*

---

## Portfolio Plateau Map (exhausted/closed levers)

### Step-side: DEFINITIVELY CLOSED (cycle 52k)
- **Tree WIDTH (M\*)**: g_d=0.0191 → M\*=32 → 479.6 TPS. Empirically + HBM-floor closed (denken #271).
- **Draft-pass-cut (all K, φ≤1)**: static-K=4 composed-honest = 493.96 TPS (fern #274) — but MEASURED local wall-clock REFUTES it (stark #273, MERGED 05:02, `51bdsbpw`: K4-vs-K7 = **−8.63%**, realization ratio **NEGATIVE** — any K≠7 falls off the ONEGRAPH K=7 graph and regresses); **deployed K=7 stands**. Closed at both the composed AND the measured level.
- **Draft decomposition**: MLP+attn+io = 95.2% intrinsic-M=1; only GeluAndMul fold recoverable (+2.65% honest). Closed (kanna #277/#269, wirbel #270).
- **Linear step normalization**: 1218.2µs is a normalized unit; batch=1 wall draft savings over-credit 4.82× (denken #278).
- **Verify forward**: HBM-bound MLP 66%; int4 GEMMs approaching-roofline; only +1.185% greedy-safe SDPA (kanna #280).
- **Verify SDPA tune**: +1.29% (487.8), insufficient standalone (wirbel #279).
- **Verify-compute hideability**: only **4.8%** of the 2104.6µs verify-above-read compute overlap-hides greedy-safe → kernel-addressable floor **487.7 < 500**; `free_lane_to_500_exists=0` (the MEASURED-fraction floor replacing #283's optimistic 746.9 all-hides ceiling; coincides exactly with wirbel #285's 487.729). The free non-build step lane is measured-CLOSED (denken #291, MERGED 04:40, `3myn1fzl`).
- **Prefill denominator**: 2.85% of wall, decode-dominated (ubel #275).
- **Host/serving overhead**: decode loop is **99.5% GPU-bound** — host/serving = **0.50%** (40µs of the 8017µs wall), an order of magnitude below fern #274's inferred ~40%; the denken #278 M=1 micro-built subtraction manufactures a phantom 29.2% (under-counts deployed M=8 GPU-busy by 2303µs of REAL work). Recoverable +0.50 TPS < 9.63 materiality gate. Host front CLOSED (ubel #284, MERGED 05:03, `u58fxtu6`).
- **GEMM-bandwidth**: PERMANENTLY CLOSED — HBM 1-wave saturation wall 83.6%, 0.0% speedup at any tile shape (PR #130/#117/#108).
- **int4-Marlin body GEMMs**: bit-exact across M=1/8/16, already deployed. Closed.
- **ONEGRAPH/CUDAGraph / TRITON_ATTN pin**: already deployed in 481.53. Closed.
- **τ_acc**: 1.0 ± 0.0075, local=official. Closed (lawine #276).
- **τ_ppl**: 1.000218 ± 0.000210, local int4 PPL = official proxy. Safe local PPL bar 2.4185. Closed (lawine #288) — the local→official transfer TRINITY (τ_lo/τ_acc/τ_ppl) is COMPLETE.

### Step-side consolidation — MERGED (banks credit, does NOT cross 500 alone)
- **Lossless micro-lever envelope (wirbel #285, MERGED 03:40, `97b57hhe`)**: total greedy-safe bit-identical step-shaving = **15.48µs → +1.29% → 487.7 TPS** (`envelope_clears_500=False`). The four-lever stack collapses to ONE incremental lever (SDPA num_stages 3→2); lm_head (0.66µs fused ceiling, FUSED_SPARSE_ARGMAX on-GPU) + norms (ONEGRAPH+vLLM fused add+rmsnorm) `already_captured`. The **FREE step-side ceiling is 487.7 TPS**; residual gap +2.52% lives off the step axis.
- **Bridge basis-honesty card (kanna #286, MERGED 03:35, `0k4azmjo`)**: the bridge is **DRAFT-SIDE-SPECIFIC** — draft-side 0.2147 (4.66× over-credit), verify-side **1.0** (no discount). Best single basis-honest lever = verify SDPA 487.758; composed disjoint stack = **493.637** (still **6.36 short** of 500). Confirms wirbel #285's verify-side envelope needs no discount → **step-side closed at BOTH raw and basis-honest level.**

### THE ANALYTIC PATH-A IS CLOSED (fern #281 capstone) — sole re-open is a BUILT drafter raise
- **fern #281 verdict:** Path-A CLOSED on all three axes; `go_region_exists=False`; no realizable `(E[T]_real, M, step-shave)` cell reaches 500. The analytic frontier is settled at 481.53.
- **lawine #282 corroboration:** the E[T] gap is +0.140 public (smallest of any axis); headroom is real (top-quartile prompts at E[T]≥4.36; bottom-quartile→median lift = 515.93) but there is NO free prompt-side lever — it must be BUILT.
- **The sole re-open:** a BUILT public-E[T] raise to ≥4.97 (acceptance-per-candidate, NOT width). denken #119: the linear drafter caps at 3.8445 at perfect capacity → the +1.12 raise REQUIRES a structurally non-linear / feature-conditioned drafter. Greedy-SAFE by construction (emission = verify argmax).
- **Prime build candidate:** EAGLE-3 multi-layer hidden-state fidelity (researcher RANK 2) — fuses target layers {2,21,39} into the drafter at every step, directly attacking the j≥2 OOD acceptance collapse (ubel #263). `SupportsEagle3` ready in the vLLM fork (PR #15). Companion: PARD-2 CAT loss (RANK 4, same training run). Additive: SAM-Decoding suffix-automaton retrieval (RANK 3, +2-4%, zero PPL risk).
- **Gate:** full retrain is a TRAINING run → human-approval-gated. Cheap precursor = Phase-1 architecture-adaptation viability (2h single GPU, no submission).

---

## Strategic Posture (cycle 52k)

**Resolved this cycle:** fern #281 closed Path-A analytically; lawine #282 confirmed no free prompt-side lever. The analytic exploration is complete — 481.53 is the analytic frontier.

**Emergent theme (cycle 52l) — PRICING THE HONEST COST OF THE GATED RAISE (the build-economics matrix):** with Path-A closed and the BUILT EAGLE-3 raise the sole >500 path, the fleet has pivoted from "is there a path?" to a coordinated MATRIX pricing every facet the human GO/NO-GO must weigh — each a distinct, non-colliding 0-TPS analytic axis:
- **(a) STEP cost — CLOSED** — wirbel #293→#295 (both MERGED): the heavier fusion drafter RAISES the E[T] bar 4.9029→6.1245; #295 MEASURED the random-init {2,21,39} fusion step at **~2.95×** the linear draft (vs #293's modeled 3×) → corrected target central **6.11**, bracket [5.36, 6.86], VALIDATING 6.1245 rather than collapsing it (the naive standalone 1.745× is a dispatch-compressed LOWER bound — the 256-dim linear runs at 11.5% BW vs faithful 59.9%; deployed ONEGRAPH+INT4 raises the ratio). 6.12 stands as the central sizing bar, inside the 8.0 window.
- **(b) COMPANION floor** — lawine #292→#296 (both MERGED): SAM shaves only the ungated +2–4%, and that companion SUB-ADDITIVELY shrinks to [0.79,1.59]% under a better drafter (r_overlap 0.6034), RAISING the honest residual EAGLE-3 must cover ALONE 0.902→0.998; against the honest 6.1245 target SAM covers only ~3% — NOT the lever.
- **(c) PER-POSITION target** — kanna #289→denken #297 (both MERGED): the EAGLE-3 spec is UNIFORM (lift j≥2→0.91, keep a_1≥0.73); the cliff is prompt-invariant at position 1 (a1-deepen on the hard tail), so NO prompt-adaptive depth — and the binding constraint is the LOW-quartile a_1=0.6550.
- **(d) FEASIBILITY gate** — kanna #294 (WIP): the cheap-proxy GO threshold (a SINGLE a_2 per #297, no prompt-conditioning).
- **(e) VRAM fit — CLOSED at FITS** — ubel #299 (MERGED): the {2,21,39}-fusion drafter + hidden-state retention lands at **20.10 GB resident → 3.90 GiB headroom** vs 24 GB hard (2.90 vs 23 usable); dominant term `extra_kv` (0.719 GiB), drafter weights negligible (0.037 GiB), net delta +0.80 GB. Memory-feasible AT REST; the runtime capture/verify PEAK is the separate axis (k) reseated to ubel #306.
- **(f) FREE-CEILING realization** — stark #298 (WIP): does the banked 487.7 step ceiling realize on the wall, or over-credit like static-K (#273)?
- **(g) PRIVATE-bar target — CLOSED at CLEARS** — lawine #300 (MERGED 06:02, `8t5q6sr0`): the public-E[T] build clears the binding PRIVATE ≥500 gate. The card reproduces the organizer-verified **460.85** private TPS EXACTLY (resid 0.00, Δ4.29%); the private collapse is **position-1 HELD (c1=1.0)** — the M=8 tree salvages the rank-2+ matches the width-1 spine rejects — residual c_deep=0.97135 on j≥2, `rho_priv_e3=0.9421`. No hidden private-side death. Sets up the #304 reconciliation: the tree already recovers c1 on the LINEAR drafter — does it transfer to a fusion draft? → reseated **lawine #309**.
- **(h) BUILD COST — CLOSED at GO** — denken #301 (MERGED): the gated drafter costs **~107 A10G-GPU-hr** (capture 20.3 + drafter-train 87.1, of which the frozen 262k-vocab lm_head GEMM is **68.2 = 78%**, dominating the 124.5M draft net 3.59×/token) → **13.4 h wall** on 8×A10G (DP) at N=4e8 tok × 10ep, MFU 0.35 — **GO** under the ≤200 GPU-hr lane (headroom 92.5; feasible through 4e8×20ep, breaks only at 8e8×10ep). ORDER-OF-MAGNITUDE estimate (paper reports entries not tokens, no epoch/MFU, no train cost) — this bottom-up IS the anchor. Cost is no longer the missing input; any build-cost lever (smaller/tied/factored head, fewer epochs) attacks the lm_head first.
- **(i) READ COMPANION — CLOSED (dead on both doors)** — fern #302 (MERGED): the #287 8.43% read-cut FITS the build's 0.0428 PPL headroom (cost 0.0203) but REGRESSES on the wall (`read_cut_realization_ratio=−2.0177` ≈ #273's static-K −2.018, `realized_uplift=−6.69%`, classification `regresses`) → **BUILD ALONE**. The read-cut is not free standalone (#287) and not free as a companion (#302); body-side deviations from the deployed ONEGRAPH K=7 graph regress exactly as #273 predicted. The build's TPS must come from the NUMERATOR, not from stacking body-side read levers.
- **(k) RUNTIME VRAM PEAK — CLOSED at GREEN** — ubel #306 (MERGED 06:31, `y1lji0c6`): the runtime refinement of (e) holds — runtime build peak **20.158 GiB** (resident 20.10 + transient 0.058) fits ≤24-hard with **3.84 GiB** headroom (1.90 vs device-visible 22.058). The capture pool is the dominant transient (~42 MiB, 1.1% of headroom); the 262k-vocab tree-verify logit buffer is bf16-native MB-scale (4 MiB@M=8, 16 MiB@M=32 — no fp32 upcast). DECISIVE re-diagnosis: lawine's #101 size-29 crash is a capture-SIZE-list DISPATCH `IndexError` (`max_cudagraph_capture_size=16`), **NOT an OOM** — VRAM clears M=32 trivially (+12 MiB) but the dispatch list does not; deployed M=8 (K+M=15<16) clears. "Fits at rest" AND "fits at runtime." The genuine #101-class risk (orthogonal to memory) → reseated **ubel #311** (price `cudagraph_capture_sizes` vs tree width).
- **(l) INTEGRATION-READINESS — CLOSED at YELLOW** — wirbel #307 (MERGED 06:31, `88eh8twv`): wiring the {2,21,39}-fusion drafter into the served runner is **NOT a config drop-in** (`swap_is_config_only=0`, `readiness_blocks_go=1`). 6/9 touchpoints ARE config-only (stock vLLM exposes the aux hidden states, 0h — feasibility PR #15), but the onegraph loopgraph that PRODUCES 481.53 is MTP-specific (keyed to `Gemma4Proposer`; "width-1 exact" false for EAGLE's own-KV draft) → under method:eagle3 the 3 speed patches (T5/T6/T7 in `sitecustomize.py`) go INERT; frontier runtime lost unless rewritten + re-validated. EAGLE-3 does NOT re-open #272 (different layer), but the rewrite reopens the same unguarded sitecustomize.py → port `_guard_included_router` in the same edit. The rewrite COST is unpriced → reseated **wirbel #312**.
- **(m) TREE-SALVAGE (draft-side demand relaxation) — CLOSED** — lawine #309 (MERGED 06:31, `7tkn4d9x`): inverting #300's salvage operator `c1_eff=a1+(1−a1)·cov_W` over wirbel #79's MEASURED rank-coverage (cov₄=0.6532), the M=8 verify-tree **relaxes denken #304's drafter demand from raw a1→0.9213 (RED, +26%) to raw a1→0.7731 (GREEN-YELLOW, +5.9%)** — `salvage_relaxes_304_demand_by=0.1482`, robust across W=2..4 (W=2 still 0.865<0.92), reproduces #300's c1=1.0 to 6dp, collapses to 0.9213 only at literal zero rank-2+ transfer. The reconciliation of #304 (draft-level a1-break) and #300 (tree recovers c1): the fusion draft needs raw a1≈0.77, not 0.92. YELLOW: cov_W measured on the LINEAR spine; fusion rejection geometry could lower it (if frac_true_beyond_top4 > 0.347). Pre-registering the cheap probe that flips YELLOW→GREEN/RED → reseated **lawine #313**; relaxed 0.77 target feeds denken #308.
- **(j) NUMERATOR REACHABILITY — CLOSED** — denken #304 (MERGED 06:02, `dtf1ouml`): inverting the chain-product `E[T]=1+Σ Π a_j`, hitting #295's 6.11 target CANNOT be done with the position-1 cliff held — kanna #289's spec (a_1=0.73, j≥2=0.91) yields only E[T]=**4.9196** (1.1916 short), and even the perfect-tail corner caps cliff-kept E[T] at exactly 6.11<6.1112. The 6.12 numerator is reachable ONLY by a drafter that lifts **a_1: 0.7292→~0.9213** (×1.263, vs LOW-q 0.6550). **The decisive tightening: every COST axis is priced GO/feasible, but they all assume a drafter hitting the required a_k EXISTS — and #304 shows that drafter must do what the deployed LINEAR drafter CANNOT (break the a_1 cliff).** (wirbel #303 was the de-conflicted duplicate of this exact inversion — closed; wirbel reseated to integration-readiness #307.) The reachability question now splits into two orthogonal **TRAINABILITY** follow-ups → **denken #308** (DRAFT-side: is a_1≈0.92 inside the published EAGLE-3 first-token envelope, or an intrinsic floor?) + **lawine #309** (VERIFY-side: does the M=8 tree-salvage that recovers c1=1.0 for the linear spine (#300) transfer to a fusion draft and relax the demand below 0.92?).

The matrix de-risks the human build decision on ALL of achievability, step cost, fit, validity, dollar cost, and free-companion stacking before any training spend.

**ROLLUP — MERGED (fern #305, `m4nmtdl9`):** with axes (a) step 6.11, (b) companion 0.998, (c) per-position uniform, (e) VRAM 20.10 GB/3.90 GiB, (g) private CLEARS (ρ_priv 0.9421), (h) cost 107.47 GPU-hr GO, (i) read-companion CLOSED, (j) numerator-reachability CLOSED, (k) runtime-VRAM GREEN, (l) integration YELLOW, (m) tree-salvage relaxes a1→0.77 all banked, fern's consolidated GO/NO-GO **decision card** lands: 25 imports verified ≤1e-6, corrected-target invariant reproduces public=500 to 5.7e-14. **The decision-relevant headline: under the CONSERVATIVE scalar ×0.804, PRIVATE is the binding axis** — at E[T]=6.11 private projects only 402.0, at bracket-top 6.8588 only 451.2, both sub-500; private-500 crossing needs public E[T]=7.601 (above the bracket). P(private≥500)=3.9% indep / 0% coupled; the dominant INDEPENDENT private lever is `private_factor`. **★ THE LIVE CRUX: the GO/NO-GO flips on the private model.** The scalar ×0.804 is fern #305's conservative bookend, but lawine #300/#309's per-position model (rho_priv_e3=0.9421; private-500 needs public E[T]≈4.19 ≪ the 6.11 target) is exactly that dominant lever and the credible path to private≥500. Reconciling the two private conventions at E[T]=6.11 — does the build clear PRIVATE 500 under the per-position tax? — is reseated to **fern #310** (the single most decision-relevant analysis remaining). The matrix is now FULLY CLOSED on cost/fit/validity; the live residual is (1) the private-model reconciliation (fern #310), (2) the a1-cliff drafter TRAINABILITY now relaxed to ~0.77 ((d) kanna #294 + denken #308 draft-side + lawine #313 verify-probe), and (3) the unpriced deployment loopgraph-rewrite COST (wirbel #312). Not a launch recommendation.

**Step-side consolidation — DONE this cycle (the step-side credit a built raise stacks on, now closed at the basis-honest level):**
- wirbel #285 (lossless envelope, MERGED) + kanna #286 (bridge basis-honesty, MERGED): the FREE step ceiling is 487.7, the composed basis-honest stack is 493.64 — both <500. The step-side denominator is settled at both raw and basis-honest level.
- denken #283 (MERGED 03:49, `vmxuwxm0`): the HBM-bound ceiling = **1265.6 TPS**; deployed 481.53 is only **38% of the honest 1/K_cal=7982.9µs wall** — the system is **NOT read-bound** (REFUTES the "floor>step ⇒ HBM-bound" reading of #278; that gap was composition-COMPRESSION, re-proving the 4.82× over-credit). The 62% non-read slack = draft 9% + **verify-compute 26%** + **host 26%** (ubel #284, Morgan, in flight).
- **denken #291 (MERGED 04:40, `3myn1fzl`+`myttnvah`) CLOSES the verify-side front #283 opened:** the honest kernel-addressable floor lands **ON 487.7289 TPS** — only **4.8% (101.5µs) of the 2104.6µs verify-above-read compute is greedy-SAFE overlap-hideable** (exactly the one SDPA num_stages lever wirbel #285 already found); the other 95.2% is exposed/serial. #283's optimistic all-hides 746.9 **was never realizable** (over-credited ~259 TPS); the 487.7↔746.9 gap was a basis artifact (φ_WS = W/S = 6.5530 composition compression). **`free_lane_to_500_exists=FALSE`** — there is NO free non-build step lane to ≥500. The step-side is now **DEFINITIVELY CLOSED at the FREE ceiling 487.7** at both the normalized and honest-wall basis. Reframes the path conclusively: **E[T]-raise BUILD is the sole >500 lever** — the denominator side is fully audited and shut.
- land #245 (Morgan banking): tree-fidelity proof (scratch-KV bug +0.235, tree-causal mask +0.088, tree-vs-linear delta ≈0) — the durable result; full live-integration build is OFF the critical path (g_d settled it).

**THE PIVOT — BUILT public-E[T] raise (Plateau-Protocol bigger swing):**
1. **Phase-1 viability (cheap, in-bounds):** EAGLE-3 architecture-adaptation sanity (2h single GPU, `SupportsEagle3` load + run for Gemma-4, no retrain, no submission). De-risk the interface before spending training.
2. **Pre-build target (analytic):** **wirbel #290 (MERGED 04:04, `ub3kpsso`)** settled the aggregate honest step-banked target at **4.9029** public E[T] — budget **+1.0584** beyond denken #119's linear cap 3.8445 (which the deployed drafter sits AT, zero linear headroom), inside the feasibility window (4.9029 < E_T_max 8.0; 25.5% of cap→ceiling headroom), recoverable ONLY by a structurally non-linear drafter; `eagle3_sufficiency_is_build_gated`. **kanna #289 (MERGED 04:11, `fi34s269`)** decomposed E[T]=3.844 into the per-position a_k profile: the acceptance cliff is at **POSITION 1** (forfeits 1.895 tokens = 45.7% of the loss; conditional acceptance RISES with depth = survivorship) and the BUILT-raise target now has an exact per-position spec — **lift j≥2 conditional acceptance to ≈0.91 while keeping a_1≥0.73** (deep-position lift is feasible, a_1-only is ceiling-bound at E[T]=4.910<4.966 ⇒ `built_raise_requires_nonlinear_drafter`), localizing WHERE wirbel #290's 1.0584 budget lives. **kanna #294 (reseat, Morgan)** → EAGLE-3 Phase-1 viability gate (the cheap-proxy GO threshold). **denken #297 (MERGED 05:11, `vo2ir6ca`)** RESOLVED #289's shape-transfer caveat by direct per-quartile remeasure (fresh 128×7 per-prompt matrix, reconcile resid 0.00): the cliff is prompt-INVARIANT at position 1 (both quartiles; 124/128 prompts), the hard tail merely DEEPENS a_1 (LOW a_1=0.6550 vs TOP 0.8291, "a1-deepen" vertical shift) ⇒ the EAGLE-3 per-position target is **UNIFORM** (lift j≥2→0.91), NOT prompt-adaptive; the binding constraint is the LOW-quartile a_1=0.6550. Reseated denken #301 → EAGLE-3 build-cost card (GPU-hour SPEND axis). **wirbel #293 (MERGED 04:31, `abhoog1x`)** re-banked the 4.9029 target against the HEAVIER EAGLE-3 fusion drafter's draft-step overhead: under `eagle3_draft = m_fuse × linear_draft` the corrected target RISES to **6.1245** (band [5.80, 6.12] at L_fuse=3), eating the 0.0631 free lossless lever **19.4×** and landing 1.16 ABOVE fern #281's 4.966 — the window holds at all m_fuse∈{2,3,4,6} but is TIGHT at m_fuse=6 (7.957<8.0). **HONEST CAVEAT (student-flagged): `m_fuse×linear_draft` is a CONSERVATIVE UPPER model** (treats fusion as m_fuse full forwards); EAGLE-3's drafter is ONE forward ingesting a fused feature, so the architecturally-honest target is likely ~5.0. **wirbel #295 (MERGED 05:26, `c334qaqu`)** MEASURED the random-init {2,21,39} fusion draft step at **~2.95×** the linear draft (regime-corrected from the dispatch-compressed standalone 1.745× — the 256-dim linear runs at 11.5% BW vs faithful 59.9%, and ONEGRAPH+INT4 raises the ratio; byte-ratio self-consistency 9.116×68.8/359.4=1.746) → corrected target central **6.11**, bracket [5.36, 6.86] — VALIDATING #293's 6.1245 rather than collapsing it; **axis (a) step-cost CLOSED**, 6.12 stands as the central sizing bar inside the 8.0 window. Reseated wirbel → integration-readiness #307 (the #303 numerator-reachability placeholder was de-conflicted/closed — denken #304 MERGED owns that inversion: 6.12 needs a_1≈0.92 vs deployed 0.73, axis (j) CLOSED).
3. **Ungated forward companion — VERDICT IN (lawine #292, MERGED 04:34, `3sqnkveo`):** SAM-Decoding suffix-automaton retrieval is a **+2–4% UNGATED COMPANION, NOT a standalone path to 500**. Measured prompt suffix-recurrence hit_rate(n=3)=**0.16124** → lifted E[T] ∈ [3.921, 3.998] ≪ the 4.90 step-banked target, leaving a **residual 0.902 E[T]** only a gated drafter (EAGLE-3) can cover. **Decisive low-tail finding:** retrieval lands on the ALREADY-fast prompts (high-E[T] decile hit 0.170 vs low 0.007, `pearson +0.32576`) → largely REDUNDANT with the deployed linear drafter; it does NOT rescue the slow tail. **There is NO training-free standalone path to 500** — SAM banks as a free companion ON TOP of a gated raise. Greedy-safe (emission = verify argmax), PPL-pinned (lawine's own #288 trinity → local bar 2.4185). **lawine #296 (MERGED 05:10, `15ilrhrg`):** the +2–4% companion SUB-ADDITIVELY shrinks to [0.79,1.59]% under a kanna-#289-profile EAGLE-3 drafter (keeps ~40%; r_overlap 0.6034 = 0.4118 a_k-lift + 0.1916 corr-tilt — the +0.326 redundancy CONFIRMED), RAISING the honest residual EAGLE-3 must cover ALONE 0.902→**0.998** (vs 4.9029; vs 6.1245 → 2.22, SAM covers ~3%). The human GO/NO-GO should price against 6.1245, where SAM is NOT the lever. Reseated lawine #300 → private-bar EAGLE-3 target (the build-VALIDITY axis).
4. **Full EAGLE-3 retrain (human-approval-gated):** route via `Approval request: HF job`. Companion PARD-2 CAT loss (same run). Phase-1 architecture-adaptation viability (2h single GPU) is the cheap precursor.
5. **Composition:** any built E[T] raise stacks multiplicatively on the lossless step envelope (wirbel #285) — `official = K_cal·(E[T]/step)·τ`, E[T]-independent step levers compose cleanly.

**Launch posture:** NEVER launch unilaterally. Route via `Approval request: HF job`. Publish-first (#124), human green-light required. All cycle-52l deliverables are bank-the-analysis (0 TPS, baseline unchanged at 481.53).

**Launch-readiness flag (Issue #272, advisor-acknowledged 05:1xZ):** the active 481.53 submission `fa2sw_precache_kenyan/sitecustomize.py` is MISSING `_guard_included_router` (0 matches vs the sibling `fa2sw_treeverify_kenyan`'s 6) — risks a vLLM 0.22.1rc1 + prometheus `_IncludedRouter` boot-500 → `/v1/models` 500 → 0 records scored on a FRESH runner (ubel reproduced it live, `he7glotf`). Does NOT affect the already-recorded 481.53 (it booted on its scoring run), but ANY re-launch — including a future ≥500 — needs the guard ported first. Remediation = port the 6-match guard (defensive, zero perf/numerics change) + a fresh-runner boot smoke-test; approval-gated (served-file change) — advisor asked the humans on #272 whether to open the `Approval request: HF job for fa2sw_precache_kenyan boot-guard` issue. On the launch checklist; advisor will NOT touch the served file until approved.

**Greedy-identity (Issue #124, advisor closing note 05:1xZ):** the human call stands — publish 500+ first, greedy-identity is accepted-risk for organisers to adjudicate. stark #273 corroborates the divergence is INTRINSIC (deployed K=7 itself ~59% divergent from its own AR, same order as kanna #114's 56%; no K recovers identity). The kicker: the BINDING live gate is PPL≤2.42, which is K-invariant + teacher-forced (never invokes the drafter) — so the token-identity divergence does NOT touch the quality gate that actually binds. The accepted-risk is narrowly scoped to contract-interpretation, with no exposure on delivered quality.

---

## Recent Human Researcher Directives

- (None new this cycle — operating under standing directives.)
- Standing: maximize single-stream TPS on Fast Gemma Challenge; 500 TPS is the gate (PPL≤2.42, 128/128); compound every improvement; zero idle GPUs.

---

## Key Reference: TPS Composition

```
official = K_cal · (E[T] / step) · τ = 125.268 · (3.844 / 1218.2) · 1.218 = 481.53 TPS

  K_cal = 125.268    (calibration constant)
  E[T]  = 3.844      (expected accepted tokens, K=7 linear MTP, M=8 verify)
  step  = 1218.2µs   (NORMALIZED/batch-amortized composition unit — NOT a wall sum, denken #278)
  τ     = 1.218

To reach 500: need +3.835% (E[T] floor 3.9914). With step fully closed, the ONLY
multiplicand that can move is E[T]. Bridge-discount all batch=1 wall draft savings
by 0.2147; verify-side deployed-M=8 savings carry bridge≈1.0 (kanna #286 confirming).
```
