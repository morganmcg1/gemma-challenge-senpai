# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-15 ~05:05Z (cycle 52k)
- **Advisor branch:** `approval-gated-8gpu-20260613`

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

## Active Roster (cycle 52k, 8/8 GPUs — step-side consolidation + E[T]-axis verdict)

| Student | PR | Hypothesis | Owner | Status |
|---------|-----|-----------|-------|--------|
| wirbel  | #295 | EAGLE-3 fusion-drafter step profile (collapse the alarming 6.12 corrected-target band to the architecturally-honest ~5.0; A10G-profile the single-forward fusion cost) | Morgan | 🔄 WIP (reseat; #293 MERGED 04:31 → m_fuse×linear_draft model raises target 4.9029→6.1245, eats the free lever 19.4×; conservative-UPPER caveat) |
| kanna   | #294 | EAGLE-3 Phase-1 viability gate (the cheap-proxy GO threshold before the human-gated retrain) | Morgan | 🔄 WIP (reseat; #289 MERGED 04:11 → acceptance cliff at POSITION 1 = 45.7% of E[T] loss, feasibility asymmetry: deep-lift feasible / a_1-only ceiling-bound ⇒ BUILT raise requires non-linear drafter) |
| fern    | #287 | Read-reduction PPL pareto | Morgan | 🔄 WIP (SENT BACK by advisor — headline over-credited the deployed point as 100% read-bound; denken #283 measures body-read=38% of honest wall ⇒ re-price required read-reduction on the 38% fraction, reconcile the −153% sign conflict) |
| lawine  | #296 | SAM × EAGLE-3 companion-stacking additivity (does the +2–4% SAM companion SURVIVE a better drafter, or does EAGLE-3 absorb the same recurrence substrate, raising the honest residual above 0.902?) | me | 🔄 WIP (reseat; #292 MERGED 04:34 → SAM = +2–4% ungated companion, residual 0.902 E[T], low-tail redundancy pearson +0.326) |
| denken  | #297 | Tail-resolved per-position (does the hard-prompt acceptance cliff SHIFT? — the per-prompt per-position remeasure kanna #289 flagged) | me | 🔄 WIP (reseat; #291 MERGED 04:40 → honest kernel floor lands ON 487.7, only 4.8% of verify-above-read overlap-hideable, #283's 746.9 never realizable, free lane to ≥500 does NOT exist) |
| ubel    | #299 | EAGLE-3 build VRAM budget (does the {2,21,39} fusion drafter + hidden-state retention fit ≤24GB?) | me | 🔄 WIP (reseat; #284 MERGED 05:03 → decode loop 99.5% GPU-bound, host overhead 0.50%, host front CLOSED; the denken #278 M=1 over-credit made concrete = phantom 29.2%) |
| stark   | #298 | Free-ceiling wall-clock realization (does the banked 487.7 free step ceiling REALIZE on the host-to-host wall, or over-credit like static-K?) | me | 🔄 WIP (reseat; #273 MERGED 05:02 → static-K composition REFUTED, K4-vs-K7=−8.63% realization ratio NEGATIVE, deployed K=7 stands — the realization-ratio method reused to wall-audit 487.7) |
| land    | #245 | Tree fidelity build — Morgan banking Cycles 1-4 (terminal pending), will reseat non-tree | Morgan | 🟡 banking |

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

**Emergent theme (late cycle 52k) — PRICING THE HONEST COST OF THE GATED RAISE:** with Path-A closed and the BUILT EAGLE-3 raise the sole >500 path, the decision-critical analytics have pivoted from "is there a path?" to "what does the gated raise actually have to clear?". Two axes now priced: (a) **STEP cost** — wirbel #293 showed the heavier fusion drafter RAISES the E[T] bar 4.9029→6.1245 (conservative-upper; #295 tightening toward the architecturally-honest ~5.0); (b) **COMPANION floor** — lawine #292 showed SAM-Decoding shaves only the ungated +2–4%, leaving a 0.902 residual the gate must cover (#296 testing whether that residual is even HONEST under a better drafter). Together with kanna #289's per-position acceptance spec (lift j≥2→0.91) and kanna #294's Phase-1 GO threshold, the cycle is de-risking the human build decision before any training spend.

**Step-side consolidation — DONE this cycle (the step-side credit a built raise stacks on, now closed at the basis-honest level):**
- wirbel #285 (lossless envelope, MERGED) + kanna #286 (bridge basis-honesty, MERGED): the FREE step ceiling is 487.7, the composed basis-honest stack is 493.64 — both <500. The step-side denominator is settled at both raw and basis-honest level.
- denken #283 (MERGED 03:49, `vmxuwxm0`): the HBM-bound ceiling = **1265.6 TPS**; deployed 481.53 is only **38% of the honest 1/K_cal=7982.9µs wall** — the system is **NOT read-bound** (REFUTES the "floor>step ⇒ HBM-bound" reading of #278; that gap was composition-COMPRESSION, re-proving the 4.82× over-credit). The 62% non-read slack = draft 9% + **verify-compute 26%** + **host 26%** (ubel #284, Morgan, in flight).
- **denken #291 (MERGED 04:40, `3myn1fzl`+`myttnvah`) CLOSES the verify-side front #283 opened:** the honest kernel-addressable floor lands **ON 487.7289 TPS** — only **4.8% (101.5µs) of the 2104.6µs verify-above-read compute is greedy-SAFE overlap-hideable** (exactly the one SDPA num_stages lever wirbel #285 already found); the other 95.2% is exposed/serial. #283's optimistic all-hides 746.9 **was never realizable** (over-credited ~259 TPS); the 487.7↔746.9 gap was a basis artifact (φ_WS = W/S = 6.5530 composition compression). **`free_lane_to_500_exists=FALSE`** — there is NO free non-build step lane to ≥500. The step-side is now **DEFINITIVELY CLOSED at the FREE ceiling 487.7** at both the normalized and honest-wall basis. Reframes the path conclusively: **E[T]-raise BUILD is the sole >500 lever** — the denominator side is fully audited and shut.
- land #245 (Morgan banking): tree-fidelity proof (scratch-KV bug +0.235, tree-causal mask +0.088, tree-vs-linear delta ≈0) — the durable result; full live-integration build is OFF the critical path (g_d settled it).

**THE PIVOT — BUILT public-E[T] raise (Plateau-Protocol bigger swing):**
1. **Phase-1 viability (cheap, in-bounds):** EAGLE-3 architecture-adaptation sanity (2h single GPU, `SupportsEagle3` load + run for Gemma-4, no retrain, no submission). De-risk the interface before spending training.
2. **Pre-build target (analytic):** **wirbel #290 (MERGED 04:04, `ub3kpsso`)** settled the aggregate honest step-banked target at **4.9029** public E[T] — budget **+1.0584** beyond denken #119's linear cap 3.8445 (which the deployed drafter sits AT, zero linear headroom), inside the feasibility window (4.9029 < E_T_max 8.0; 25.5% of cap→ceiling headroom), recoverable ONLY by a structurally non-linear drafter; `eagle3_sufficiency_is_build_gated`. **kanna #289 (MERGED 04:11, `fi34s269`)** decomposed E[T]=3.844 into the per-position a_k profile: the acceptance cliff is at **POSITION 1** (forfeits 1.895 tokens = 45.7% of the loss; conditional acceptance RISES with depth = survivorship) and the BUILT-raise target now has an exact per-position spec — **lift j≥2 conditional acceptance to ≈0.91 while keeping a_1≥0.73** (deep-position lift is feasible, a_1-only is ceiling-bound at E[T]=4.910<4.966 ⇒ `built_raise_requires_nonlinear_drafter`), localizing WHERE wirbel #290's 1.0584 budget lives. **kanna #294 (reseat, Morgan)** → EAGLE-3 Phase-1 viability gate (the cheap-proxy GO threshold). **denken #297 (reseat, me)** resolves #289's shape-transfer caveat — directly remeasures the per-quartile per-position profile (re-running lawine #282's harness) to decide whether the EAGLE-3 per-position target is UNIFORM (lift j≥2→0.91 across prompts) or PROMPT-ADAPTIVE (the cliff shifts deeper for the hard tail), the within-chain NUMERATOR complement to wirbel's step-side denominator work. **wirbel #293 (MERGED 04:31, `abhoog1x`)** re-banked the 4.9029 target against the HEAVIER EAGLE-3 fusion drafter's draft-step overhead: under `eagle3_draft = m_fuse × linear_draft` the corrected target RISES to **6.1245** (band [5.80, 6.12] at L_fuse=3), eating the 0.0631 free lossless lever **19.4×** and landing 1.16 ABOVE fern #281's 4.966 — the window holds at all m_fuse∈{2,3,4,6} but is TIGHT at m_fuse=6 (7.957<8.0). **HONEST CAVEAT (student-flagged): `m_fuse×linear_draft` is a CONSERVATIVE UPPER model** (treats fusion as m_fuse full forwards); EAGLE-3's drafter is ONE forward ingesting a fused feature, so the architecturally-honest target is likely ~5.0. **wirbel #295 (reseat, Morgan)** → EAGLE-3 fusion-drafter step profile (collapse the 6.12 band to the honest single-forward fusion cost via A10G profile).
3. **Ungated forward companion — VERDICT IN (lawine #292, MERGED 04:34, `3sqnkveo`):** SAM-Decoding suffix-automaton retrieval is a **+2–4% UNGATED COMPANION, NOT a standalone path to 500**. Measured prompt suffix-recurrence hit_rate(n=3)=**0.16124** → lifted E[T] ∈ [3.921, 3.998] ≪ the 4.90 step-banked target, leaving a **residual 0.902 E[T]** only a gated drafter (EAGLE-3) can cover. **Decisive low-tail finding:** retrieval lands on the ALREADY-fast prompts (high-E[T] decile hit 0.170 vs low 0.007, `pearson +0.32576`) → largely REDUNDANT with the deployed linear drafter; it does NOT rescue the slow tail. **There is NO training-free standalone path to 500** — SAM banks as a free companion ON TOP of a gated raise. Greedy-safe (emission = verify argmax), PPL-pinned (lawine's own #288 trinity → local bar 2.4185). **lawine #296 (reseat, me)** → SAM × EAGLE-3 companion-stacking additivity: does the +2–4% companion SURVIVE under a better drafter, or does EAGLE-3 absorb the same recurrence substrate (the +0.326 corr says it will), shrinking SAM a SECOND time and RAISING the honest residual above the optimistic 0.902?
4. **Full EAGLE-3 retrain (human-approval-gated):** route via `Approval request: HF job`. Companion PARD-2 CAT loss (same run). Phase-1 architecture-adaptation viability (2h single GPU) is the cheap precursor.
5. **Composition:** any built E[T] raise stacks multiplicatively on the lossless step envelope (wirbel #285) — `official = K_cal·(E[T]/step)·τ`, E[T]-independent step levers compose cleanly.

**Launch posture:** NEVER launch unilaterally. Route via `Approval request: HF job`. Publish-first (#124), human green-light required. All cycle-52k deliverables are bank-the-analysis (0 TPS, baseline unchanged at 481.53).

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
