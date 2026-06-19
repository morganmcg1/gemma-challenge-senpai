STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["fzvf5d64"],"primary_metric":{"name":"strict_clean_routeb_net_official_equiv_tps","value":85.62},"test_metric":{"name":"strict_clean_beats_126378","value":0}}

## Results

**Verdict: NO. There is no strict-byte-exact spec config that beats 126.378 buildable in-scope right now.** Route (b) tree-verify-at-M=1, costed honestly, caps at **~85.6 official-equiv (K=5)** — it *fails* the 126.378 bar by **~40.8 TPS**. Even the physically-impossible overlap ceiling only *ties* the bar. The only route that keeps the speedup *and* is byte-exact is route (a)'s M-invariant int4 GEMV kernel, which is absent / out-of-scope to build this cycle. **The #730 τ=0.3 fire stays a tolerance gamble; strict-byte-exact tolerance-dependence is irreducible without route (a).**

Analysis-only (`analysis_only=1`, `official_tps=0`, no HF Job, no fire). All inputs are on this branch (#728 sweep, PR-616 flip analysis, BASELINE #122). Cost model: `research/strict_clean_spec_route/cost_model.py` → `report.json`.

### 1. Locus confirmed: with BI=1 pinned, the residual strict-divergence is the int4 Marlin GEMM M-dependent tie-break — NOT attention

| quantity | value | source (on-branch) |
|---|---|---|
| pure-M flip rate (m8 vs m1, **BI=1 both**, teacher-forced same tokens, prefill) | **0.432%** (283/65536), CI95 [0.374, 0.494]% | PR-616 `flip_report.json` |
| fraction of flips that are near-ties (gap ≤ 0.5 nat) | **100%** (gap p99 = 0.25 nat ≤ 0.3; median ≈ 0) | PR-616 |
| M=1-vs-M=1 determinism floor | **0.0** | PR-616 + #728 `ar_determinism_control` (AR2 vs AR1 GREEDY_IDENTICAL, 0/65536) |
| deployed spec-vs-own-AR seq_exact (K=5 / K=6) | 0.172 / 0.203 (106/128, 102/128 divergent) | #728 `report.json` |
| confident genuine flips at τ=0.3 (all K) | **0** · PPL exact **2.0189** | #728 |

**Reconciling land #680's "break is ATTENTION not GEMM":** that was a no-BI / non-single-segment read. Per BASELINE.md #122 (on this branch), under the #728 config the attention path is *already* batch-invariant (TRITON_ATTN single-segment; `fa_sliding` 0-fire; `splitkv` auto-gated-off), so `VLLM_BATCH_INVARIANT=1` changes attention divergence by ~0. By elimination the **sole** M-dependent reduction left is the int4 Marlin GEMM (`marlin_gemm` picks split-K geometry = f(M), no exposed `num_splits`/`max_par` knob; no batch-invariant Marlin in the pinned `vllm==0.22.0` wheel). M=1 decode and M=K+1 verify reduce the K axis in different float order → low-bit logit deltas → near-tie argmax flips. **100% int4 grid-ties (gap ≤ 0.3 nat), 0 confident genuine flips, PPL exact — benign but strict-DIVERGENT.**

### 2. Route (b) tree-verify-at-M=1 cost — the headline

**Why the *cheap* version of route (b) is not byte-exact (the crux).** The target's K/V projections are int4 Marlin GEMMs too. So the M=K+1 verify writes a KV-cache whose floats differ from the M=1 AR KV-cache **even at positions whose argmax is M-invariant** (non-tie), and that drift compounds through attention. PR-616's 0.432% is measured over teacher-forced **identical-token** prefixes, so it already bakes in this compounding KV drift — it is not just the final lm_head tie. Consequence: you **cannot** certify byte-exactness by re-decoding only the tie positions. A selective re-decode runs on the drifted M=K+1 KV and is therefore *not* AR-exact at exactly the tie positions it is meant to repair. So selective re-decode is **still τ-tolerance-dependent — it buys nothing over #730's gamble.**

**The only byte-exact construction rebuilds the M=1 KV for the whole accepted block → a full M=1 re-decode.** Per accepted block of length `L` (accepted drafts + 1 bonus):

```
cost/block = [drafter K-steps + verify(M=K+1)]   (measured spec machinery = L / spec_tps)
           + [full M=1 re-decode of L tokens]     (== an AR decode of L tokens = L / ar_tps)
emit/block = L
=> route_b_tps = 1 / (1/spec_tps + 1/ar_tps)      (L cancels; serial; UPPER bound)
```

`E[extra M=1 passes per accepted block] = E[L] ≈ 3.3 tok` (K=6, BASELINE line 28) — i.e. the **whole** block is re-paid; the headline is L-invariant. This is < `ar_tps` for any finite `spec_tps`, so **route (b) can never beat AR (126.378), regardless of K or acceptance.**

| K | spec wall (local) | speedup | **STRICT route (b) net official-equiv** (×1.192 anchored / ×1.0352 floor) | beats 126.378? |
|---|---|---|---|---|
| 5 | 222.69 | 2.100× | **85.62** / 74.36 | ❌ (−40.76) |
| 6 | 218.61 | 2.062× | **85.10** / 73.91 | ❌ (−41.27) |

- **Overlap-optimistic ceiling** (drafter+verify fully hidden — physically impossible at conc=1 single-GPU HBM-bound): `= ar_tps = 126.378 official` → only **ties**, never beats.
- **Non-strict selective contrast** (shown only to size what strictness gives up; NOT byte-exact): K=5 ≈ [243, 263] official, K=6 ≈ [239, 258]. This is the speedup you'd keep *if* you tolerated ties — i.e. exactly #730.

### 3. Route (a) M-invariant int4 GEMV — secondary feasibility

- **Buildable in-scope? NO.** Needs a *new* fixed-split-K int4 Marlin/Machete CUDA kernel. BASELINE #122: `marlin_gemm` split-K = f(M), no exposed knob, no batch-invariant Marlin in the pinned wheel; corroborates stark #722 `SPARSE_INT4_KERNEL_ABSENT` and the land #506 M=1 BI-GEMV prior.
- **Overhead if built:** at conc=1 the verify is HBM-bound (weight-load dominates), so pinning the M=1 reduction order keeps the verify ≈ 1 AR-pass-time → the spec speedup is *preserved* and byte-exact (official-equiv stays ~226–265). The **build** is the blocker, not the runtime cost.
- **Verdict: GO-on-value, NO-GO-on-in-scope-buildability this cycle.**

### 4. Verdict

There is **no** strict-clean spec route that beats 126.378 buildable now:
- **Route (b)** (kernel-agnostic): strict byte-exactness forces a full M=1 re-decode → `1/(1/spec+1/ar) ≈ 85.6` official, fails by ~41. The cheap selective variant isn't byte-exact (drifted KV at ties) → still tolerance-dependent. **No free lunch: strictness costs the *entire* spec speedup.**
- **Route (a)** (M-invariant GEMV): would keep the speedup *and* be byte-exact (~226–265 official), but the kernel is absent and out-of-scope to build this cycle.

⇒ The #730 un-rescued K=6 fire remains a **τ=0.3 self-consistency gamble** (depends on the organizer honoring the tolerance ruling), not a strict-clean fire above the wall. Tolerance-dependence is irreducible without route (a)'s kernel.

### Reproduce / evidence

```bash
cd target/ && python3 research/strict_clean_spec_route/cost_model.py
```
- Artifact: `research/strict_clean_spec_route/{cost_model.py,report.json}`
- W&B run: **`fzvf5d64`** (`denken/strict-clean-spec-route`, group `denken-strict-clean-spec-route`; metrics under `headline/`, `locus/`, `k5/`, `k6/`)
- Peak VRAM: N/A (analysis-only, no GPU job; CPU arithmetic over #728's already-measured anchors)
- Public evidence used: locked anchor `int4_g128_lmhead` 126.378 (BASELINE rung, PR #4); #728 sweep anchors (this branch); PR-616 flip analysis (this branch); BASELINE #122 batch-invariant-verify probe (this branch).

### What happened — honest analysis

The hypothesis hoped route (b) might keep "enough" spec speedup to clear 126.378 while being byte-exact. It can't, for a structural reason: byte-exactness to AR requires the M=1 KV trajectory, and the int4 K/V projections are themselves M-dependent Marlin GEMMs, so the verify's KV cache is float-drifted at *every* position (not only ties). You therefore cannot patch ties cheaply — the whole block must be re-decoded at M=1, which costs an entire AR pass per token. Route (b)'s strict ceiling is thus *AR minus the drafter+verify tax* (≈ 85 official), strictly below the 126.378 AR-equiv anchor. The model is L-invariant and an upper bound (early in-block divergence only shortens emitted runs), so the NO-GO is robust. The strict-clean fire would require route (a)'s M-invariant GEMV — a from-scratch CUDA kernel that prior work (BASELINE #122, stark #722) places out-of-scope.

### Suggested follow-ups

1. **Re-scope route (a) as a training/kernel request** if a strict-clean fire is worth a build: a fixed-split-K int4 GEMV that reproduces the M=1 reduction order would keep ~226–265 official *and* be byte-exact. Estimate the kernel effort against the marginal value over the #730 τ=0.3 path before committing.
2. **Quantify the organizer-audit risk on #730 directly** instead: the only thing route (b) was trying to buy is independence from the τ=0.3 tolerance ruling. A short note pinning how the official scorer treats greedy-identity (it runs TPS+PPL+128/128, no token-identity check per BASELINE #124) may de-risk #730 more cheaply than any kernel.
3. **Measure the true tie-rate `r_tie`** (gap ≤ 0.3 nat per position, not just the flip rate) on a cheap teacher-forced pass — it bounds the selective-route cost exactly, in case a future *tolerance-allowed* fast path wants it.
