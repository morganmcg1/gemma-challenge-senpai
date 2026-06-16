STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["3oudivg1"],"primary_metric":{"name":"strict_private_delta_pct","value":4.6894},"test_metric":{"name":"private_validity_safe","value":1}}

## Results

**SAFE — with a flag.** Strict #474's predicted public→private speed Δ is **4.689%** (central), which **clears the organizer's 5% re-run gate**, and PPL **2.3772 ≤ 2.42** clears by construction (greedy byte-identity). But the speed margin is **thinner than the deployed stack's** (0.311pp vs 0.705pp) and a sufficiently large private prompt-length shift (**+89.5 tok**) would breach. PPL is unconditionally safe; **speed is the live gate**, and it passes under every realistic assumption.

| PR metric | value |
|---|---|
| **`strict_private_delta_pct`** (PRIMARY, central) | **4.689%** |
| `strict_private_predicted_tps` | 440.15 TPS |
| `private_validity_breach_prob` (realistic σ_hw) | **0.002%** |
| `private_validity_breach_prob_conservative` (1% convention) | 38.3% |
| `local_to_public_bias` | 0.9660 |
| `private_trajectory_delta_pct` (deployed ctxlen bucket) | 0.633% |
| **`private_validity_safe`** | **True** |
| `ppl` (carry) | 2.3772 ≤ 2.42 (margin 0.0428) |

### The model — one load-bearing cancellation

Strict shares the MTP‑K7 drafter, the body GEMM/lm_head/framework tax **B**, the global-layer attention, and therefore *both* the acceptance behaviour and the global-layer ctxlen growth with the deployed stack. It differs **only** by pinning the 7 sliding-window local reductions to FULL attention for byte-identity (stark #472/#475 `added_us(L)`). That gives an exact additive identity:

```
strict_gap   = 1 − (1 − deployed_gap)·(1 − local_pinned)
local_pinned = 1 − kvw_strict(private_traj)/kvw_strict(public_traj)
```

The deployed **acceptance** (3.661%) and **ctxlen** (0.633%) buckets **cancel out of `local_pinned`** — they are already inside `deployed_gap`. Strict inherits the full deployed 4.295% gap *plus* only the extra L‑sensitivity from running the local reductions full instead of windowed. Verified in self-tests: `local_pinned(ΔP=0) = 0` exactly, and `kvw_strict(public_traj)` reproduces stark #475's banked **461.8049**.

### Corners (private prompt-shift ΔP from ubel #379)

| Corner | ΔP (tok) | `local_pinned` | **`strict_gap`** | predicted private TPS | clears 5%? |
|---|---:|---:|---:|---:|:---:|
| banked (pure‑ρ) | +0 | 0.000% | 4.295% | 441.97 | ✅ |
| **central** | **+50** | **0.413%** | **4.689%** | **440.15** | ✅ |
| pessimistic | +130 | 1.068% | 5.317% | 437.25 | ❌ |
| **breakeven to 5%** | **+89.5** | — | 5.000% | — | — |

The banked corner reproduces the deployed gap exactly (the cancellation), confirming strict adds **nothing** to the gap at zero trajectory shift. The whole risk is the *systematic* private-prompt-length shift, and the breakeven (**+89.5 tok**, a +17% bump over the ~528-tok public mean trajectory L) sits between #379's central +50 and pessimistic +130.

### P(breach) = P(systematic + session-noise > 5%) — and why σ_hw decides it

The single private re-run draws TPS ~ N(μ_priv, σ_hw); breach iff the draw < 0.95·strict_public. The systematic shift sets μ_priv; the session noise σ_hw is the band on top (this **STACKS** with kanna #478's noise band — no double-count, because μ carries the deterministic gap and σ carries only fresh single-draw noise).

| shift | σ_hw = 0.349 (empirical, lawine #467) | σ_hw = 4.815 (1% convention) |
|---|---:|---:|
| central (+50) | **0.002%** | 38.3% |
| pessimistic (+130) | ~100% | 62% |

lawine #467 *measured* the between-run served-TPS σ at **0.349 TPS (0.073%)** over 10 fresh clock-locked runs — the "1% convention" (4.815 TPS) overstates it **13.8×**. Under the measured σ, the central-shift breach prob is negligible (0.002%). The convention number is logged only as the conservative band-top; it is the (refuted) worst case.

### Calibration (instruction 1) — the premise correction

The PR baseline framed strict 461.80 as a "raw local, pre-calibration" number to be multiplied by `local_to_public_bias`. **It is already official-anchored.** stark #475 builds `kvw_strict(L) = 481.53 · CYC/(CYC + added(L))`: the local pod enters *only* through the hardware-clock-**invariant** ratio `CYC/(CYC+added)` (both µs, the clock cancels), and the absolute scale is the **official** deployed 481.53 anchor. Re-multiplying by 0.9660 would **double-count** the transfer. I log the deployed bias (0.9660 = local 465.14 / official 481.53) for context and bound the residual tax-fraction-transfer risk by τ_lo's 0.13% stability. Self-test `bias_does_not_remultiply_strict`: `tps_from_added(0) = 481.53` exactly. ✅

### Honest analysis — what happened

PPL is not the gate — strict is byte-identical greedy (denken #471/#476), so its private PPL is the deployed private PPL (2.3777 measured), and the 2.3772 anchor clears 2.42 unconditionally. **Speed is the live gate**, and the verdict is *clears, but watch the margin*: strict's 0.311pp headroom is < half the deployed stack's 0.705pp, purely because pinning the local layers to full attention steepens the per‑L tax. The decomposition is robust because the verdict rides on the **cancellation**: strict adds gap *only* through `local_pinned`, which is a small (0.4–1.1%) roofline-bounded term — attention is a thin slice of the 7667‑µs cycle. The only modeled degree of freedom is the private trajectory shift ΔP, and the breakeven (+89.5 tok) is well above the same-methodology #379 central +50.

One honesty caveat: the breakeven is *closer* than the deployed gap's own breakeven (+253 tok to 5% for deployed alone). A private split whose prompts run ~+90 tokens longer on average than public would push strict over. That is implausible for a held-out split drawn the same way, but it is not the comfortable 5× headroom the PPL side enjoys — hence **SAFE with a FLAG**, not "SAFE, ignore."

**Independent corroboration (public-evidence bracket):** an openevolve board finding (20260616-062754-273, @senpai-mentioned) reports the 5% gate is *actively invalidating* high-public entries (a w256+precache run 508.04→470.95 = 7.3% INVALID; a verified 489.66→~470 = 3.9% valid; "honest private decode ~470"). Max verifiable public ≈ private/0.95 ≈ 495.7. Strict's 461.80 public sits **below** that mirage line (it has no sliding-window/precache public-only boost), so it is in the *honest* regime — bracketing `strict_gap` from **below**, while this card's full-attention per‑L model is the conservative **upper** bound. The two agree strict is not a windowing mirage.

### Comparison vs. PR baselines

| Quantity | PR baseline | This card |
|---|---|---|
| deployed gap | 4.295% (cmpatino-verifier) | 4.2946% reconstructed (resid <1e‑4) |
| strict public TPS | 461.80 (stark #475) | 461.8049 reproduced (kvw harmonic) |
| strict private Δ | *the question* | **4.689%** central (clears 5%) |
| 5% gate | clears? | ✅ central + banked; ❌ only beyond +89.5 tok |
| PPL | 2.3772 | 2.3772 ≤ 2.42 (margin 0.0428), greedy-invariant |
| official TPS | 481.53 (unchanged) | **+0 (analysis-only)** |

### Suggested follow-ups

1. **Pin ΔP with one cheap measurement** (de-risks the only modeled DOF): if private-VALID prompt token-length stats become available (or are measurable on the int4‑ct proxy over a private-like slice), drop the real ΔP into `local_pinned` to replace the +50 central. Even the +130 pessimistic only *breaches* by 0.32pp, so this is confirmation, not a gate — but it converts the FLAG to a hard SAFE/UNSAFE.
2. **Hand kanna #478 the stacked model:** μ_priv = strict_pub·(1 − strict_gap_systematic) and σ_hw ∈ [0.349, 4.815] are the inputs for a joint systematic⊕noise breach band; my `breach_grid` is the σ‑sweep slice.
3. **One real cross-session σ_hw** would collapse the 0.002%↔38% band: the official re-run is cross-node/cross-session (unmeasured); lawine #467's 0.349 is same-session. A 2–3 run cross-session probe would pin the true σ and finalize the breach prob.

### Reproduce

```bash
cd target/ && .venv/bin/python research/validity/private_rerun_validity/private_rerun_validity.py \
    --wandb_group equivalence-escalation-anchors --wandb_name denken/private-rerun-validity-474
```

- **Self-test:** `self_test_passes` = **True** (18/18: kvw reproduces banked 461.80, `local_pinned(0)=0` exact, monotone in shift, deployed gap reconstructs 4.295%, strict_gap ≥ deployed_gap, central clears 5%, banked == deployed, breakeven between central/pessimistic, pessimistic breaches, strict_pub below openevolve mirage line, bias in unit band, bias does not re-multiply strict, PPL clears with margin, breach monotone in σ and in shift, threshold = 95% of pub, σ_empirical < convention, NaN-clean).
- **Peak memory:** pure-stdlib CPU-analytic (no torch/numpy in the core path; harmonic over 128×512 trajectory points).
- **W&B run:** `3oudivg1` (group `equivalence-escalation-anchors`).
- **Public-evidence note:** 0 official TPS, 0 HF Job, 0 `--launch`, 0 submission, 0 served-file change, 0 kernel rebuild. CPU-analytic over banked W&B anchors (stark #475 strict per‑L tax + trajectory, stark #472 whole-cycle, ubel #379 deployed gap decomposition, systems local→official transfer, lawine #467 empirical σ_hw). No GPU used.
