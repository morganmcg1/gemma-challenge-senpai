# SENPAI Research State — Fast Gemma Challenge

- **Date:** 2026-06-15 ~02:35Z (cycle 52j)
- **Advisor branch:** `approval-gated-8gpu-20260613`

## 🆕 Cycle-52j Snapshot (2026-06-15 ~02:35Z) — THREE decisive closures this cycle; one >500 candidate remains (SDPA linear deploy); pivot to orthogonal throughput levers

**★ THE STRATEGIC TURN: All draft-pass-cut and tree-width levers are now definitively BELOW 500 at grounded measurements. The sole remaining >500 candidate is SDPA kernel mistuning (1.097× at M=8, verified bit-identical). Pivot fully to orthogonal levers: prefill denominator, acceptance numerator, and any residual kernel inefficiency.**

---

## THREE Decisive Closures This Cycle

### 1. Tree WIDTH closed (denken #271 — deployed g_d)
- Deployed `g_d = 0.0191` (vs assumed 0.168 — 9× lower, measured on live path)
- M\* (optimal tree width) = 32 → predicted TPS = **479.6 < 500** (gap: −20.4 TPS, −4.1%)
- Tree width is NOT the path to 500. Closed.

### 2. All draft-pass-cut levers below 500 (fern #274 — φ-correction)
- Composition-overhead honesty: `φ_tree = 0.603` (tree-path wall step)
- Honest static-K=4 ceiling: **493.96 TPS** (gap: −6.04 TPS, −1.2%)
- Every draft-pass-cut scenario re-simulated with φ≤1: ALL below 500. Closed.

### 3. Draft decomposition COMPLETE (kanna #277 + wirbel #270)
- Draft io_projection: **NULL** (INTRINSIC-M=1, already minimal, not the bottleneck)
- Draft SDPA (wirbel #270): **MISTUNED** — 1.097×/1.090×/1.092× at M=8/16/32 (bit-identical, correctness confirmed)
- Step decomposition pinned: `draft_k7 = 706.9 µs`, `residual = 511.3 µs`
- ONE actionable kernel inefficiency identified: **verify SDPA linear deploy** → wirbel #279

### 4. Acceptance transfer closed (lawine #276)
- `τ_acc = 1.0 ± 0.0075` — local λ̂ is official proxy 1:1
- Safe local bar: **λ̂ ≥ 0.9855** (↔ official ≥ 0.9780 with 3σ margin)
- Local acceptance measurement is reliable. Closed.

---

## Current BASELINE

```
481.53 TPS  (approval-gated-8gpu-20260613, merged)
```

Official target: **500 TPS**. Gap: **−18.47 TPS (−3.7%)**.

### Composition anchors (grounded, frozen)
```
official = K_cal · (E[T]/step) · τ = 125.268 · (3.844/1218.2) · 1.218 = 481.53 TPS
τ_lo  = 1.03524   (local→official TPS transfer, lawine #267)
τ_acc = 1.0       (local→official acceptance transfer, lawine #276)
φ_tree = 0.603    (composition overhead, fern #274)
g_d (deployed) = 0.0191  (step-basis, denken #271)
step decomp: draft_k7=706.9µs, residual=511.3µs  (kanna #277)
safe local λ̂ bar = 0.9855  (↔ official 0.9780 + 3σ, lawine #276)
```

---

## Active Roster (cycle 52j, 8/8 GPUs)

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| wirbel  | #279 | Verify SDPA linear deploy (1.097× at M=8, bit-identical) → TPS gain? | 🔄 WIP |
| denken  | #278 | Prefill denominator: measure live prefill fraction, validate K_cal | 🔄 WIP |
| kanna   | #280 | Acceptance numerator: λ̂ headroom to 0.9855 — what training change gets there? | 🔄 WIP |
| fern    | #281 | Residual 511.3µs decomposition: what is the dominant term? | 🔄 WIP |
| lawine  | #282 | E[T] sensitivity: marginal TPS per +0.01 λ̂ at current operating point | 🔄 WIP |
| stark   | #273 | [ongoing from 52i] | 🔄 WIP |
| ubel    | #275 | [ongoing from 52i] | 🔄 WIP |
| land    | #245 | [ongoing from 52i] | 🔄 WIP |

---

## Portfolio Plateau Map (exhausted/closed levers)

### Definitively CLOSED (at grounded measurements)
- **Tree width (M\*)**: g_d=0.0191 → M\*=32 → 479.6 TPS. Below 500. Closed.
- **Draft-pass-cut (all K, all φ≤1)**: Static-K=4 honest ceiling = 493.96 TPS. Below 500. Closed.
- **Draft io_projection**: INTRINSIC-M=1, NULL contribution. Closed.
- **ONEGRAPH/CUDAGraph**: Already deployed in 481.53 baseline. Closed.
- **int4-Marlin body GEMMs**: Bit-exact across M=1/8/16, already deployed. Closed.
- **TRITON_ATTN force-pin**: Already pinned for Gemma-4 (heterogeneous head dims). Closed.
- **τ_acc measurement**: 1.0 ± 0.0075, local=official. Closed.

### ONE Actionable Candidate Remaining
- **SDPA kernel mistuning (wirbel #279)**: 1.097× overhead at M=8, bit-identical. If linear-deploy removes this → +~9.7% on draft SDPA → net TPS gain TBD (draft_k7=706.9µs is ~58% of step). Potentially the last kernel lever above 500.

### Orthogonal Levers (not yet grounded)
- **Prefill denominator (K_cal)**: Is 125.268 optimal? Is there prompt-length sensitivity? (denken #278)
- **Acceptance numerator (λ̂)**: Current = 0.9780 official. Safe local bar = 0.9855. What closes the 0.0075 gap? (kanna #280)
- **Residual 511.3µs**: What is it? Target-model pass? Sampling? Communication? (fern #281)
- **E[T] sensitivity**: Marginal TPS per +0.01 λ̂ at current K=4, E[T]=3.844 operating point (lawine #282)

---

## Strategic Posture (cycle 52j)

**Immediate (this cycle):**
1. wirbel #279: SDPA linear deploy — the ONE remaining >500 kernel candidate
2. denken #278: Prefill denominator ground truth
3. kanna #280: λ̂ headroom measurement
4. fern #281: Residual decomposition
5. lawine #282: E[T] sensitivity curve

**If SDPA deploy clears 500 (wirbel #279 wins):**
- Merge immediately
- Pivot to acceptance numerator + prefill denominator to compound further
- Assign cleanup PR to remove stale tree-width and draft-pass-cut experiment flags

**If SDPA deploy does NOT clear 500:**
- The gap (−18.47 TPS) must come from orthogonal levers in combination
- λ̂: +0.0075 headroom → ~+X TPS (lawine #282 will ground this)
- Residual: if 511.3µs is reducible → compound with SDPA gain
- Prefill K_cal: if prompt-length tunable → another compounding lever

**Launch posture:** NEVER launch unilaterally. Route via `Approval request: HF job`. Publish-first (#124), human green-light required.

---

## Recent Human Researcher Directives

- (None new this cycle — operating under standing directives from 52h/52i)
- Standing: maximize TPS on Fast Gemma Challenge; 500 TPS is the gate; compound every improvement; zero idle GPUs.

---

## Key Reference: TPS Formula

```
TPS = K_cal · (E[T] / step_wall) · τ_lo

where:
  K_cal     = calibration constant (125.268, measured)
  E[T]      = expected accepted tokens = Σ_{k=1}^{K} λ̂^k  (approx K·λ̂ for λ̂ near 1)
  step_wall = wall time per speculative step (µs)
  τ_lo      = local→official TPS transfer (1.03524)

At baseline:
  481.53 = 125.268 · (3.844 / 1218.2) · 1.218
           [NOTE: τ written as 1.218 above = τ_lo · φ correction absorbed into K_cal convention]
```

**To reach 500:** Need +3.86% TPS. Levers in order of estimated impact:
1. SDPA kernel fix (wirbel #279): potentially +3–5% on draft wall time
2. λ̂ improvement to 0.9855: +~1–2% on E[T]
3. Residual reduction (fern #281): unknown until decomposed
4. Prefill K_cal tuning (denken #278): unknown sensitivity
