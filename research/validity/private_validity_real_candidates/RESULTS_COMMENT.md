STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["shcdordv"],"primary_metric":{"name":"floorlock_strict_private_delta_pct","value":0.6334},"test_metric":{"name":"floorlock_private_validity_safe","value":1}}

## Results

**A safe ship exists — but it is the SLOW one, and the PR's premise needs one correction.** Of the two real #474 fire candidates, **floor-lock 161.70 is private-safe** (Δ **0.633%**, 4.37pp headroom) and **global-flag 234.47 is NOT** (Δ **4.295%**, only 0.71pp headroom, **36.7%** one-shot breach). The reason flips the PR's framing: byte-exact *output* does **not** imply ~0% private speed-Δ. Floor-lock (M=1 AR) has **no drafter**, so it sheds the deployed stack's 3.661% **acceptance** bucket; global-flag is **still speculative** (E_accept≈3.87) on the same MTP-K7 drafter, so it **re-inherits** that acceptance gap. The acceptance gap is a *speed* effect (drafter accepts fewer tokens on the shifted private prompt distribution), and byte-exactness — which only fixes the output tokens / PPL — does nothing for it.

### Verdict table (PR ask 3)

| Candidate | public TPS | predicted **private** TPS | **Δ %** | safe (<5%)? | headroom | one-shot breach |
|---|---:|---:|---:|:---:|---:|---:|
| **floor-lock** (M=1 AR, literal-1.0) | 161.70 | **160.68** | **0.633%** | ✅ **YES** | **4.37pp** | 0.0008%¹ |
| **global-flag** (blanket BI, spec-alive) | 234.47 | **224.40** | **4.295%** | ❌ **NO** | 0.71pp | **36.7%**² |
| *— global-flag under PR's byte-exact premise* | 234.47 | *232.98* | *0.633%* | *✅* | *4.37pp* | *1.8%* |
| deployed reference (#52) | 481.53 | 460.85 | 4.295% | ✅ | 0.71pp | — |

¹ floor-lock breach under the **physical fractional** σ_hw (multiplicative clock/thermal noise). Under the **conservative absolute** σ_hw 4.864 it is 7.33% — see the σ-fraction section; that tail is the *only* flag on floor-lock. ² global-flag breach under σ_oneshot 4.876 abs; 24.3% even under fractional σ.

**Safest ship: floor-lock 161.70.** It is the only candidate whose systematic Δ sits comfortably below the gate.

### The one load-bearing distinction — acceptance is a SPEED gap, not a quality gap

The deployed public→private Δ decomposes (ubel #379, `5kpb73tb`) as:

```
deployed_gap 4.295%  =  3.661% ACCEPTANCE (drafter)  +  0.633% ctxlen (global-layer KV)
```

The **acceptance** bucket is the MTP-K7 drafter accepting fewer tokens per verify step on the private prompt distribution (E[T] 3.844→~3.66). That is a property of *the drafter × the prompt distribution* — **not** of the verifier's output bytes. So:

- **Floor-lock** = M=1 autoregressive, **no drafter at all** → acceptance bucket is structurally **0** → Δ = ctxlen only = **0.633%**. (Self-test `floorlock_nonspec_zero_acceptance`.)
- **Global-flag** = blanket `VLLM_BATCH_INVARIANT=1`, but the spec decode is **still live** (ubel #470 `ugqnytji` measured E_accept≈3.87, did *not* collapse to the AR floor) → **re-inherits the full 3.661% acceptance gap** → Δ = **4.295%**, the same as deployed. (Self-test `globalflag_inherits_acceptance`.)

Byte-exactness via the BI pin guarantees the *output tokens* match a reference (→ PPL 2.3770 ≤ 2.42), which is why the PR reasoned "~0% quality-driven Δ." That is true for **quality**. But the private *speed* Δ is driven by draft acceptance on the shifted distribution, which the BI pin leaves **untouched**. Hence global-flag carries the deployed speed gap in full.

### σ_hw-fraction effect (PR ask 2) — quantified at zero systematic gap

Absolute σ_hw is a fixed TPS band; as a *fraction* of a slower config it grows, so even a perfectly byte-exact (Δ=0) config's one-shot breach inflates purely from being slow:

| config | public TPS | σ_between **/TPS** | breach @ Δ=0, **abs** σ | breach @ Δ=0, **frac** σ | gate distance |
|---|---:|---:|---:|---:|---:|
| deployed | 481.53 | **1.01%** | 3.7e-7 | 4.0e-7 | 4.95σ |
| global-flag | 234.47 | **2.07%** | 0.80% | 4.0e-7 | 2.41σ |
| floor-lock | 161.70 | **3.01%** | **4.82%** | 4.0e-7 | **1.66σ** |

So the answer to "does either slow candidate's Δ-% inflate toward the gate from σ alone?" is: **floor-lock's does — to 4.82% breach under absolute σ, even with a near-zero systematic Δ**, purely because 4.864 TPS is 3.01% of 161.70. Under the **physical fractional** σ (noise scales with TPS) it stays at ~4e-7 for every config (self-test `fractional_sigma_is_config_invariant`). This is why floor-lock ships with a **flag, not a fail**: its systematic Δ is tiny, and the only path to >5% is the conservative-worst-case assumption that hardware noise is absolute rather than multiplicative.

### Breach probability under empirical σ_hw (PR ask 1)

P(single private draw < 0.95·public) with μ_priv = public·(1−Δ_systematic), σ from lawine #467 (`sigma_within` 0.349 same-session, `sigma_between` 4.864, `sigma_oneshot` 4.876 = √(between²+within²)):

| Candidate | μ_priv | breach (σ_oneshot 4.876, abs) | breach (fractional, physical) |
|---|---:|---:|---:|
| floor-lock | 160.68 | 7.38% (conservative tail) | **0.0008%** |
| global-flag | 224.40 | **36.7%** | **24.3%** |

Floor-lock's breach is ~0 under the physical model and a bounded ~7% tail under the conservative one. Global-flag breaches **24–37%** of the time — the thin 0.71pp systematic headroom (from the re-inherited acceptance gap) leaves a single noisy draw little room, *and* its 2.07% σ-fraction widens the band. It is a coin-flippy gamble on a one-shot gate, not a safe ship.

### Honest analysis — what happened, and the premise correction

The PR framed both byte-exact candidates as having "~0% quality-driven private delta … driven purely by σ_hw." **That holds for floor-lock but breaks for global-flag**, and the break is the whole result: speculative byte-exact ≠ speed-safe. The deployed 4.295% gap is 85% acceptance, and any config that keeps the drafter keeps that 85%. Only dropping to M=1 AR (floor-lock) removes it — at the cost of dropping to 161.70 TPS. So there is a genuine **speed↔private-safety tension**: the fast byte-exact candidate (global-flag) is the *unsafe* one; the safe candidate (floor-lock) is 31% slower than global-flag and ~3× below the deployed flagship.

The verdict rides on the #379 decomposition (acceptance vs ctxlen) and on E_accept≈3.87 staying alive under the BI pin (ubel #470) — both banked. The only modeled choice is absolute-vs-fractional σ_hw, and it changes *only floor-lock's tail* (7% vs ~0), never the floor-lock systematic-safe / global-flag systematic-thin verdict. Self-tests `globalflag_spechonest_not_safe` and `floorlock_safest_systematic` pin it.

One honesty caveat in the other direction: global-flag's full-local Triton attention (BI pin runs single-segment local attention) could grow KV slightly *more* than the deployed M=8 path on long private prompts, so its 0.633% ctxlen term is a **conservative-low** central — global-flag's true Δ could be ≥4.295%, never below. That only strengthens "global-flag not safe." It does not touch floor-lock (M=1 attention is *smaller* than M=8, so floor-lock's roofline ctxlen 0.107% ≤ the 0.633% deployed bound I conservatively used; self-test `floorlock_ctxlen_roofline_below_conservative`).

### Public evidence used

- **ubel #379** (`5kpb73tb`) gap decomposition: 4.295% = 3.661% acceptance + 0.633% ctxlen. `BASELINE.md` — *"top drafter stacks lose 4–9% TPS on the private set (prompt-distribution shift)"* — corroborates that the acceptance bucket is a **drafter** property that any spec config inherits.
- **denken #476** (`p68oo5tj`) literal-1.0 reachability: literal byte-exact 1.0 is reachable **only** at the M=1 AR floor **161.70**.
- **ubel #470** (`ugqnytji`) BI-pin cross-check: blanket `VLLM_BATCH_INVARIANT=1` realizes **234.47** official, spec **alive** E_accept≈3.87 (does **not** collapse to 161.70), PPL 2.3770 ≤ 2.42.
- **lawine #467** σ reconciliation: σ_within 0.349, σ_between 4.864, σ_oneshot 4.876.
- Deployed pair 481.53→460.85 (Δ 4.3%), organizer cmpatino-verifier / PR #52 (`2x9fm2zx`), `BASELINE.md`.

### Comparison vs PR baselines

| Quantity | PR baseline | This card |
|---|---|---|
| basis config | defunct composed-457 | **real fire candidates** (floor-lock 161.70, global-flag 234.47) |
| #480 strict Δ | 4.689% (on the 457 that doesn't realize) | superseded |
| floor-lock Δ | *the question* | **0.633%** → **SAFE** (4.37pp) |
| global-flag Δ | *the question* (PR premise: ~0%) | **4.295%** → **NOT SAFE** (0.71pp) — re-inherits acceptance |
| σ_hw-fraction at Δ=0 | *the question* | 1.01%→3.01% of TPS; floor-lock breach 0%→**4.82%** (abs only) |
| PPL | — | 2.3772 / 2.3770 ≤ 2.42, both byte-exact |
| official TPS | 481.53 (unchanged) | **+0 (analysis-only)** |

### Suggested follow-ups

1. **If the human wants the faster ship**, global-flag's only path to private-safe is to *kill the acceptance gap* — i.e. shrink the public→private E[T] drop. That is a drafter/distribution problem, not a kernel-pinning one; the BI pin cannot fix it. Worth a separate card on whether any served config gets byte-exact output *and* AR-free speed without the acceptance penalty (likely none exists — that is the floor-lock/global-flag fork).
2. **One real cross-session σ_hw on a slow config** would collapse the absolute-vs-fractional ambiguity that is floor-lock's only flag: measure between-session σ at ~160 TPS directly. If it scales to ~1.6 TPS (fractional), floor-lock's breach is ~0 and the flag is removed; if it stays ~4.86 (absolute), the 7% tail stands.
3. **Pin global-flag's ctxlen term** (its full-local Triton attention KV growth on long private prompts) to confirm Δ ≥ 4.295% rather than guessing — confirmation only, since it can only make global-flag *less* safe.

### Reproduce

```bash
cd target/ && .venv/bin/python research/validity/private_validity_real_candidates/private_validity_real_candidates.py \
    --wandb_group private-validity-real-candidates --wandb_name denken/private-validity-real-candidates
```

- **Self-test:** `self_test_passes` = **True** (23/23: deployed gap reconstructs 4.295%; acceptance+ctxlen=gap; acceptance is the majority bucket; floor-lock non-spec zero-acceptance; floor-lock ctxlen roofline ≤ conservative bound; floor-lock Δ well below gate; floor-lock safe under fractional σ; floor-lock σ-fraction is largest; global-flag is speculative; global-flag inherits acceptance; global-flag Δ near gate; global-flag thin headroom; global-flag breach material; global-flag byte-exact-premise would be safe; global-flag spec-honest NOT safe; σ-fraction inflates for slow configs; fractional σ config-invariant; σ_oneshot reconstructs between⊕within; floor-lock safest systematic; PPL clears both; threshold = 95% of public; candidates below deployed flagship; NaN-clean).
- **Peak memory:** pure-stdlib CPU-analytic (no torch/numpy/GPU).
- **W&B run:** [`shcdordv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/shcdordv) (group `private-validity-real-candidates`).
- **Constraints honored:** 0 official TPS, 0 HF Job, 0 `--launch`, 0 submission, 0 served-file change, 0 kernel rebuild, 0 GPU. CPU-analytic over banked W&B anchors only.
