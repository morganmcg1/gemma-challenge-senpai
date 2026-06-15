STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["i08xlqbg"],"primary_metric":{"name":"token_identity_rate","value":0.99609375},"test_metric":{"name":"margin_identity_self_test_passes","value":1}}

## Results

**Verdict: RED — the margin gate does NOT yield a sub-blanket identity lever.** The required headline fields: `token_identity_rate = 0.996094` (PRIMARY, corroborates #361), `selective_eta_at_identity_1 = 44.60%` (provable τ) / `17.02%` (in-sample τ100), `selective_beats_blanket = False`, `selective_clears_500_budget = False`, W&B run **`i08xlqbg`**. Both etas exceed the **9.841%** blanket, the **4.02%** >500 budget, and the **0.97%** probe-gated floor (wirbel #362). **The margin gate fails not on recall (it catches 100% of flips) but on PRECISION**: flips are a 0.39% needle buried in a ~17% natural low-margin haystack, so any threshold that catches every flip also flags 17–45% of all positions — and at a full-M=1-forward per flagged position that is **4.5× more expensive than the blanket it was meant to beat.**

### Primary
| metric | value | meaning |
|---|---|---|
| **`token_identity_rate`** (PRIMARY) | **0.996094** | 8160/8192 positions match plain greedy AR (M=1) |
| `verify_divergence` | 0.003906 | **32/8192** positions flip (16 prompts × 512 new) |
| `per_sequence_strict_pass_fraction` | **0.25** | only **4/16** sequences are fully byte-exact |
| **`margin_identity_self_test_passes`** (TEST) | **True** | all 20 self-test checks pass (0 fails) |

**First divergence — bit-for-bit identical to #361's fingerprint:** prompt 0 (`mmlu_pro-000c2031fb`), generated offset **75** / abs pos **331**, AR token **616** (logprob −1.090 in #361) vs M=8 verify token **2086**, `margin_m8 = 0.4375` nats. The harness reproduces #361's exact divergence locus on real silicon.

### The mechanism: high recall, catastrophic precision (this is the whole result)
The margin **is** a separable signal — but separability is not enough when the positive class is this rare.

| quantity | value | reading |
|---|---|---|
| `flip_margin_auc` | **0.9721** | flips ARE concentrated at low margin (separable) |
| max flip `margin_m8` | **1.125 nats** | every one of the 32 flips sits below 1.125 nats |
| flip base rate | **0.39%** | 32 flips / 8192 positions |
| `frac_positions < τ100` (1.125 nats) | **17.02%** | …but **17%** of ALL positions are also sub-1.125-nat (natural near-ties) |
| **`flip_recall_at_tau100`** | **1.0** | the gate flags every flip ✅ (recall is NOT the problem) |
| precision @ τ100 | **2.3%** | 32 flips / 1394 flagged → **44× over-flag** |
| precision @ τ_robust | **0.88%** | 32 flips / 3654 flagged → **114× over-flag** |

The advisor's stated worry was that a *logit-level* margin might miss the *hidden-state-driven* flips (low recall). It doesn't — **recall is exactly 1.0 at every threshold** (τ99/τ100/τ_robust). The failure mode is the opposite and more fundamental: because flips are a 0.39% base rate, even an AUC-0.97 detector cannot achieve usable precision. To catch the rare flips you must accept the entire low-margin background, and in natural-language decoding that background is ~17% of positions.

### Selective eta — RED under every threshold (full-M=1 recompute, per wirbel #362)
Each flagged position needs a **full M=1 forward** (lm_head-only does NOT restore identity — wirbel #362 `5k3px8p1`: the divergence is injected by bf16 attention upstream and propagates to the final hidden state, `token_identity_rate_lmhead_only=0.9948<1.0`). The margin probe itself is **free** (read off the M=8 logits we already have), so `selective_eta = frac_below_τ × 1.0`.

| τ choice | frac flagged = **selective_eta** | identity after repair | guarantee |
|---|---|---|---|
| **τ_robust = 4.375 nats** (provable: top-4 spread bound) | **44.60%** | **1.000000** | out-of-sample safe |
| τ100 = 1.125 nats (in-sample max flip margin) | 17.02% | 1.000000 | in-sample only |
| τ99 = 1.125 nats | 16.59% | ~1.0 | in-sample only |
| — reference floors — | | | |
| blanket batch-invariant GEMM (wirbel #360) | **9.841%** | 1.0 | — |
| >500 kernel budget (λ=1 ceiling) | **4.021%** | — | — |
| probe-gated floor (wirbel #362, perfect detector) | **0.97%** | 1.0 | — |
| lm_head-only floor (NOT identity-valid, #362) | 18.31% | 0.9948 ✗ | — |

- `selective_beats_blanket = False` — 44.60% (provable) and 17.02% (empirical) both **>** 9.841%. The provable gate is **4.5× more expensive** than the blanket.
- `selective_clears_500_budget = False`; `selective_beats_probe_gated_362 = False`.
- The provable τ_robust is set by the **worst-case** M8−M1 perturbation spread (top-4 = 4.375 nats, full-K = 6.375 nats), ~3.9× the max flip margin (1.125) — but even discarding the safety margin and using the in-sample τ100, eta is still 17% > blanket.

### Determinism controls (signal, not noise)
- `determinism_ref_gen = 1.000000` (greedy-AR int4 bit-exact within session) and `determinism_verify_geometry = 1.000000` (M=8 re-forward reproduces bit-exactly) ⇒ the 0.39% divergence and the 32 flip positions are **deterministic**, the genuine M=8-vs-M=1 reduction-geometry effect, not jitter.

### Corroborates the deployed strict-failure ladder
| anchor | divergence | identity |
|---|---|---|
| lawine #196 strict floor (M=1 int4 AR) | 0.00% | **1.0** |
| **this card** (16-prompt M=8 verify geometry) | **0.39%** | 0.996094 |
| ubel #361 (8-prompt M=8 verify) | 0.46% | 0.995361 |
| lawine #232 (deployed M=8 batch-replica) | 0.73% | 0.992708 |

My 0.39% sits just under #361's 0.46% (Δ = −0.0007, within the 1% reconcile tol; `corroborates_361_identity` ✅) — same mechanism, larger sample.

### Secondary (LOCAL RELATIVE — ~7× off official per #245; 0 official TPS)
- REF greedy-AR decode: **19.4 tok/s** (51.5 ms/step) local-relative.
- `f_lmhead = 0.4106` (bandwidth model: full bf16 lm_head 1.34 GB vs int4 body 1.93 GB over 42 layers) — even if lm_head-only recompute *were* identity-valid (it is not, #362), it would cost 18.31%, still > blanket.
- lm_head full-vocab GEMV = 2778.7 µs, body 1-layer = 421.8 µs (local-relative timing).
- Peak GPU: **11.91 GB** (A10G, single sequence).

### Baseline comparison (per PR)
- Official frontier **481.53 TPS / PPL 2.3772 (PR #52)** — **UNCHANGED**; this is a local strict-mechanism screen, **0 official TPS**, no served-file change, no HF job, no submission. ✅
- λ=1 ceiling **520.953**; blanket caps strict at 520.953·(1−0.09841)=469.68<500; this margin lever would need eta<4.02% to clear >500 with the ceiling and instead costs 44.6%.

### Command
```
cd target/ && python research/validity/margin_localized_identity/margin_localized_identity.py \
    --n-prompts 16 --n-new 512 --det-prompts 2 --topk 32 --dh-iters 200 \
    --wandb_group margin-localized-identity --wandb_name ubel/margin-localized-identity-eta
```
GPU phases run as isolated subprocesses (CUDA_VISIBLE_DEVICES=0) on the on-target pod A10G; int4 substrate is the deployed `gemma-4-E4B-it-qat-w4a16-ct` snapshot. **W&B run `i08xlqbg`** (group `margin-localized-identity`), τ-sweep frontier logged as a Table.

### What happened
The card cleanly **closes lever (b) for the logit-margin detector.** The premise held in part — flips really are low-margin (AUC 0.972, all ≤1.125 nats) and the gate restores identity to 1.0 with perfect recall — but the lever still dies on **precision/base rate**: at 0.39% flips against a ~17% natural low-margin background, the smallest threshold that catches every flip flags 17% of positions (44.6% for the provable worst-case-spread τ). At a full-M=1 forward per flagged position (the corrected cost — lm_head-only is identity-invalid, #362), that is **4.5× the 9.841% blanket and 17–46× the 0.97% probe-gated floor.** This is the measured answer to the advisor's sharpened question: *frac_below_τ × full-M=1-cost = 44.6%/17.0%, far above 4.02%, with recall=1.0 — so the binding constraint is precision, not recall.* Bank the RED: the blanket batch-invariant GEMM (or a sub-int4 ceiling-lift) remains the floor; a logit-margin gate is not the cheap+correct detector wirbel #362 needs.

### Suggested follow-ups
1. **A precision-first (not separability-first) detector.** The margin is a sufficient *necessary*-condition signal but its precision is base-rate-limited. To approach wirbel's 0.97% floor a detector must flag ~0.49% (the true divergent rate), i.e. ~35× tighter than the margin. Candidate: a signal that reads the **hidden-state perturbation directly** (e.g. an M=8-vs-M=1 residual-stream disagreement probe at a mid layer), since the divergence is hidden-driven — a logit-level proxy is structurally precision-capped.
2. **Two-stage gate.** Use margin<τ as a cheap *pre-filter* (recall 1.0 at 17%) then a second cheap test on the 17% to recover precision before paying the full M=1 forward — only wins if stage-2 is much cheaper than a forward.
3. **Speculative-accept reframing.** Since recall=1.0, margin<τ could instead gate *which positions to trust without verify* (the complement) — a throughput lever rather than an identity lever; out of scope here but the data supports it.
4. **Tighten the spread bound** with top-2-only restriction when flips are 2-way (here flip_target_m1_rank≤2): t2=3.875 vs t4=4.375 — a 0.5-nat reduction, still far from closing the 9.84% gap, confirming the bound's looseness is not the binding issue (precision is).

### Repro / bug notes
- Run with an interpreter that has both `vllm` and `wandb` (the serving venv works). GPU phases self-isolate; `--relog-wandb` re-logs an existing `_results.json` without re-running GPU.
- The provable robustness bound restricts the M8−M1 perturbation spread to the **top-T M1 candidates** (T∈{2,4,8}, auto-selected by the observed max flip-target M1-rank = 2 → top-4 here); the full-top-K spread is tail-contaminated (6.375 nats) and reported for reference only.

_Public-evidence note: all anchors cited (frontier 481.53/PPL 2.3772 #52; strict floor identity=1.0 #196; deployed-M8 0.73% #232; M=8 break 0.46% #361; blanket 9.841% / probe-gated 0.97% / lm_head-only-invalid wirbel #360+#362; λ=1 ceiling 520.953) are reused, not re-derived. This card adds 0 official TPS and changes no served file._
