<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PPL public-gate headroom — is PPL≤2.42 a third binding public gate? (PR #236)

**denken · `denken/ppl-public-gate` · W&B `hodnu1w1` · BANK-THE-ANALYSIS (adds 0 TPS, no draw, no launch)**

## The question

The public 500-milestone is a **conjunction of three conditions**:

> **TPS ≥ 500  AND  PPL ≤ 2.42  AND  128/128 complete.**

Every launch-readiness leg this cycle (#217 composition, #222 binding-gate, #229 speed-margin,
#218 interleg-σ, #228 publish-first λ-floor) prices only the **TPS / λ** axes. The third
condition **PPL ≤ 2.42** has been carried as a static "served 2.3772, fine" fact and never
priced as a function of the build's **acceptance aggressiveness** under the lossy int4 verify.
Frontier #52 serves **2.3772** — a margin of only **0.0428 (1.77%)** below the cap — and kanna
#114 measured the int4-Marlin spec-verify diverging **56.08%** per-token from plain greedy AR.
**Crux:** as the ≥500 build pushes acceptance higher (deeper/wider tree, higher λ) to gain TPS,
does served PPL **drift** toward the int4-standalone PPL and cross the cap — i.e. is PPL a
**hidden third public gate**?

## The load-bearing physics — why the drift premise is FALSE on the λ axis

`program.md` L27-28 sets the contract: *"Greedy decode must remain token-identical to plain
greedy autoregressive decode **for the submitted checkpoint**."* The submitted checkpoint is the
int4 model, so **the served stream IS the int4 model's greedy stream, exactly** — and a correct
(greedy-identity-preserving) speculative / tree decode changes only **throughput** (tokens per
verify call), never the served-token distribution. That is the defining guarantee of speculative
decoding: it is output-equivalent to running the verify model alone, just faster.

Consequence — the served PPL is a property of the **verify model** (int4), invariant to the
draft acceptance λ / tree depth:

```
ppl_served(λ) = int4_standalone_ppl = 2.3772   for ALL λ        (output-equivalence)
d(ppl_served)/d(λ) = 0
ppl_headroom_at_build_bar = 2.42 − 2.3772 = 0.0428              (λ-invariant)
```

The divergent-accept fraction `f_div` is the int4-vs-target divergence **on the served (=int4)
stream** = 0.5608, and is **itself λ-invariant** (λ does not change which model verifies). The
build's λ/tree aggressiveness cannot push `f_div` up — **only a change of the verify model**
(coarser quantization) could.

## The mixture (the PR's requested decomposition) — fully pinned

In log-PPL (= average NLL) space the served stream is a token-fraction mixture:

```
ln(ppl_served) = (1 − f_div)·ln(ppl_agree) + f_div·ln(ppl_div)
```

| term | value | source |
|---|---|---|
| `ppl_agree` (f=0, agreeing tokens ≈ reference) | **2.30476** | `program.md`: cap = reference + 5% → cap/1.05 |
| `f_div` (operating divergent-accept fraction) | **0.5608** | kanna #114 (λ-invariant) |
| `ppl_div` (f=1, divergent-token sub-PPL) | **2.4355** | solved so the mixture round-trips #52 |
| `ppl_served` (the served **average**) | **2.3772** | frontier #52 (round-trip resid **4.4e-16**) |

`int4_standalone_ppl_implied = 2.3772` (**pinned**, resid 0 — the served stream *is* the
int4-standalone stream). Note the divergent **sub-component** `ppl_div = 2.4355` sits *above*
the cap — but it is only 56% of the stream, so the served **average** is 2.3772. The served
average would cross 2.42 only at **`f_div* = 0.8841`** — a **+0.3234** increase the build
**cannot** make via λ (output-equivalence). Drift driver `d(ppl)/d(f_div) = 0.1312`.

## The λ projection (the deliverable) — speed RISES, PPL FLAT

| λ | E[T] | speed (TPS) | ppl_served | headroom | f_div | clears cap |
|---|---|---|---|---|---|---|
| **#52** (map-implied λ≈0.815) | 4.824 | 481.53 | **2.3772** | **0.0428** | 0.5608 | ✅ |
| 0.9500 | 5.106 | 509.67 | **2.3772** | **0.0428** | 0.5608 | ✅ |
| **0.9780 (build bar)** | 5.169 | 515.92 | **2.3772** | **0.0428** | 0.5608 | ✅ |
| 0.9970 | 5.212 | 520.26 | **2.3772** | **0.0428** | 0.5608 | ✅ |
| 1.0 (ceiling) | 5.219 | 520.95 | **2.3772** | **0.0428** | 0.5608 | ✅ |

Across the grid the public **speed climbs +8.2%** (481.53→520.95 TPS) while **ppl_served stays
pinned at 2.3772** and headroom stays 0.0428 — **PPL is decoupled from the build's acceptance
axis.** `ppl_headroom_at_build_bar = 0.0428` (TEST), `d(ppl)/d(λ) = 0`,
**`ppl_is_binding_public_gate = False`** (HEADLINE).

## Robustness + framing (secondary)

- **Divergence sweep [0.50, 0.60]** around #114's 0.5608 (and lawine #232's rate when it lands,
  if inside this band): the served PPL is the **measured anchor** 2.3772 → **invariant** to the
  divergence rate; only the `f_div` decomposition shifts (`f_div*` ∈ [0.788, 0.945], margin ∈
  [0.288, 0.345], always positive). **Verdict stable.** lawine #232 has not landed a banked rate,
  so this leg uses #114's 0.5608.
- **Publish-first framing (#124):** PPL ≤ 2.42 is a condition of the **public milestone itself** —
  a **launch gate** read at submission, not a post-hoc private-bar defence. A breach would fail the
  public milestone directly even with TPS ≥ 500 and 128/128. Pricing it is load-bearing; the result
  is that the int4 verify keeps it pinned 0.0428 under cap, λ-invariant.
- **Honest bookend:** the worst the served average can reach by adding divergence is the
  int4-standalone PPL **2.3772 itself** (the served stream is already the full int4 stream) — the
  ceiling the drift can never exceed, 0.0428 under cap.

## Self-test (PRIMARY) — `ppl_public_gate_self_test_passes = True`

(a) mixture calibration round-trips #52's 2.3772 (resid 4.4e-16) and int4-standalone is pinned
(resid 0); (b) `ppl_margin_frontier = 0.0428` within 1e-4; (c) ppl_served monotone ↑ in `f_div`
and **flat in λ**; (d) the binding bool is consistent with `ppl_served(0.9780) > 2.42`;
(e) headroom monotone non-increasing (flat) in λ; (f) NaN-clean; (g) decoupling — speed strictly
rises while PPL is flat across the same grid. All seven pass.

## Hand-off (fern #231 + land #71 + human #124 packet)

> the public milestone's third condition PPL≤2.42 has headroom **0.0428** at the λ=0.9780 build
> bar (drift tolerance **+0.3234** in divergent-accept-fraction from #52's 2.3772 anchor under the
> 56% int4 divergence), and it is **λ-invariant** (output-equivalence: the served stream is the
> int4 greedy stream, so the build's acceptance aggressiveness cannot drift PPL) — so PPL is **NOT
> a binding public gate** on the launch-σ axis; land #71's served run reports the scorer's PPL
> natively, so it is **one-run-confirmable** but has comfortable slack (a verify-model change, not
> λ, is the only thing that moves it); fern #231 carries `ppl_headroom_at_build_bar = 0.0428` as
> the third public-gate row alongside the TPS≥500 trigger.

## Reproduce

```bash
cd target/ && CUDA_VISIBLE_DEVICES="" \
  python research/validity/ppl_public_gate/ppl_public_gate.py \
    --self-test --wandb_group issue192-reading-calibration --wandb_name denken/ppl-public-gate
```

CPU-only, peak ≈ 27 MiB, < 1 s. Imports: frontier #52 served PPL 2.3772, kanna #114 (`9q5yy9l1`)
divergence 0.5608, #222 `binding_gate` E[T](λ)→μ_pub map + the 0.9780 build bar, stark #191 build
bar. BASELINE 481.53 untouched. NOT a launch. NOT open2.
