# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-15 ~03:42Z (cycle 52k)
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
| wirbel  | #290 | Step-banked BUILT-raise target + EAGLE-3 feasibility bracket (banks #285 envelope into fern #281's floor) | me | 🔄 WIP (reseat; #285 MERGED 03:40 → free step ceiling 487.7) |
| kanna   | #289 | Per-position acceptance decay (the BUILT-raise a_k target profile to E[T]≥4.97) | me | 🔄 WIP (reseat; #286 MERGED 03:35 → bridge draft-0.21/verify-1.0, stack 493.64) |
| fern    | #287 | Read-reduction PPL pareto | Morgan | 🔄 WIP |
| lawine  | #288 | PPL local→official transfer (τ_ppl): safe local bar for gate | Morgan | 🔄 WIP |
| denken  | #283 | HBM intrinsic ceiling (the physics floor under the normalized step) | Morgan | 🔄 WIP |
| ubel    | #284 | Decode-loop host overhead (CPU/scheduling fraction of wall) | Morgan | 🔄 WIP |
| stark   | #273 | Static-K wall-clock (ongoing from 52i) | me | 🔄 WIP |
| land    | #245 | Tree fidelity build — Morgan banking Cycles 1-4 (terminal pending), will reseat non-tree | Morgan | 🟡 banking |

*(Roster shared with the parallel open2 advisor — re-survey live PR state before every assignment/merge.)*

---

## Portfolio Plateau Map (exhausted/closed levers)

### Step-side: DEFINITIVELY CLOSED (cycle 52k)
- **Tree WIDTH (M\*)**: g_d=0.0191 → M\*=32 → 479.6 TPS. Empirically + HBM-floor closed (denken #271).
- **Draft-pass-cut (all K, φ≤1)**: static-K=4 honest = 493.96 TPS. Closed (fern #274).
- **Draft decomposition**: MLP+attn+io = 95.2% intrinsic-M=1; only GeluAndMul fold recoverable (+2.65% honest). Closed (kanna #277/#269, wirbel #270).
- **Linear step normalization**: 1218.2µs is a normalized unit; batch=1 wall draft savings over-credit 4.82× (denken #278).
- **Verify forward**: HBM-bound MLP 66%; int4 GEMMs approaching-roofline; only +1.185% greedy-safe SDPA (kanna #280).
- **Verify SDPA tune**: +1.29% (487.8), insufficient standalone (wirbel #279).
- **Prefill denominator**: 2.85% of wall, decode-dominated (ubel #275).
- **GEMM-bandwidth**: PERMANENTLY CLOSED — HBM 1-wave saturation wall 83.6%, 0.0% speedup at any tile shape (PR #130/#117/#108).
- **int4-Marlin body GEMMs**: bit-exact across M=1/8/16, already deployed. Closed.
- **ONEGRAPH/CUDAGraph / TRITON_ATTN pin**: already deployed in 481.53. Closed.
- **τ_acc**: 1.0 ± 0.0075, local=official. Closed (lawine #276).

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

**Step-side consolidation — DONE this cycle (the step-side credit a built raise stacks on, now closed at the basis-honest level):**
- wirbel #285 (lossless envelope, MERGED) + kanna #286 (bridge basis-honesty, MERGED): the FREE step ceiling is 487.7, the composed basis-honest stack is 493.64 — both <500. The step-side denominator is settled at both raw and basis-honest level.
- denken #283 / ubel #284 (Morgan): HBM floor + host overhead — the physics under the normalized step (in flight).
- land #245 (Morgan banking): tree-fidelity proof (scratch-KV bug +0.235, tree-causal mask +0.088, tree-vs-linear delta ≈0) — the durable result; full live-integration build is OFF the critical path (g_d settled it).

**THE PIVOT — BUILT public-E[T] raise (Plateau-Protocol bigger swing):**
1. **Phase-1 viability (cheap, in-bounds):** EAGLE-3 architecture-adaptation sanity (2h single GPU, `SupportsEagle3` load + run for Gemma-4, no retrain, no submission). De-risk the interface before spending training.
2. **Pre-build target (analytic, NOW ASSIGNED):** **kanna #289** decomposes E[T]=3.844 into the per-position acceptance profile `a_k`, locates the acceptance cliff (which draft positions EAGLE-3 must fix), and prices the per-position lift to 4.97. **wirbel #290** (the aggregate complement) banks his own #285 lossless envelope into fern #281's floor (target relaxes 4.97→~4.90 at bridge=1.0) and brackets EAGLE-3's recoverable budget (target − denken #119's linear cap 3.8445) inside the feasibility window — a necessary-condition de-risk for the gated retrain.
3. **Full EAGLE-3 retrain (human-approval-gated):** route via `Approval request: HF job`. Companion PARD-2 CAT loss (same run). Additive SAM-Decoding retrieval (+2-4%, zero PPL risk).
4. **Composition:** any built E[T] raise stacks multiplicatively on the lossless step envelope (wirbel #285) — `official = K_cal·(E[T]/step)·τ`, E[T]-independent step levers compose cleanly.

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
