STUDENT ubel:
SENPAI-RESULT: {"terminal": true, "status": "complete", "pending_arms": false, "wandb_run_ids": ["plqx2sxn"], "analysis_only": true, "no_hf_job": true, "no_served_file_change": true, "official_tps": 0.0, "headline_rung": "cb3-ldlq", "cb3_real_rung_delta_demand_tps": 107.7311, "band_lo": 0.0, "band_hi": 107.7311, "band_width_tightened_from_15p74": 107.7311, "tightened_below_int8_anchor": false, "cross_term_now_negligible": false, "cb3_real_frac_of_demand_lift": 1.0202, "cb3_rung_kv_l2_relative": 0.047778, "cb3_rung_body_argmax_flip": 0.111955, "cb3_fakequant_ppl_matches_careful_int4": true, "cb3_rung_sits_between_int8_and_crude_int4": true, "cb3_datafree_upper_bound_dtps": 261.6239, "verdict_zone": "additivity_not_certified_destructive_not_excluded", "primary_metric": {"name": "cb3_real_rung_delta_demand_tps", "value": 107.7311}, "test_metric": {"name": "cb3_real_rung_self_test_passes", "value": 1.0}}

## Results — real cb3 RHT+VQ rung: the int8 stand-in was too gentle; the band WIDENS, the fake-quant tightening lever fails

**Verdict: NEGATIVE for the tightening goal — a clean result, not a harness bug.** The faithful cb3 rung *does* land between int8 and crude-int4 on the drafter-read tensors (the PR's bracket claim holds), but it sits **mid-bracket, not near the int8 edge**. Propagating its body-argmax flip through the #402 secant (962.27) gives a demand-TPS upper bound of **107.73 TPS ≈ 1.02× the entire +105.59 demand lift** — vs the int8-anchored 15.74. The band goes **[0, 15.74] → [0, 107.73]: it widens, not tightens.** `cb3_real_rung_self_test_passes = True (32/32 invariants)`.

### Headline (W&B `plqx2sxn` / `ubel/cb3-real-rung-perturbation`)

| field | value |
|---|---|
| `cb3_real_rung_delta_demand_tps` **(PRIMARY, faithful cb3-ldlq)** | **107.73 TPS** |
| `cb3_real_rung_self_test_passes` (harness validity) | **True — 32/32 invariants** |
| `cb3_real_rung_kv_l2_relative` | 0.04778 (int8 0.00714 ‖ crude-int4 0.07342) |
| `cb3_real_rung_body_argmax_flip` | 0.1120 (int8 0.0164 ‖ crude-int4 0.2047) |
| `cb3_real_rung_hidden_l2_relative` | 0.3647 |
| `cb3_fakequant_ppl_matches_careful_int4` | **True** (tf-PPL 23.86 < fp16 25.73 ≪ crude-int4 30.35) |
| `cb3_rung_sits_between_int8_and_crude_int4` | **True** |
| `band_lo` / `band_hi` | 0.0 / **107.73** |
| `band_width_tightened_from_15p74` | 107.73 — `tightened_below_int8_anchor = False` (Δ vs int8 +91.99) |
| `cross_term_now_negligible` | **False** (5% bar = 5.28 TPS; frac of lift = 1.020) |
| `verdict_zone` | `additivity_not_certified_destructive_not_excluded` |
| data-free literal recipe (no LDLQ), strict UB | KV L2-rel 0.1021, flip 0.2719 → **261.62 TPS** |

### The monotone curve (128 deployed prompts, vs bf16)

| rung | KV L2-rel | body-argmax flip | body tf-PPL |
|---|---|---|---|
| fp16 | 0.00000 | — | 25.734 |
| int8 RTN g128 — *the #410 stand-in* | 0.00714 | 0.0164 | 25.672 |
| **cb3-ldlq — FAITHFUL (RHT+VQ+LDLQ)** | **0.04778** | **0.1120** | **23.856** |
| crude-int4 RTN g128 | 0.07342 | 0.2047 | 30.349 |
| cb3-real — data-free UB (RHT+VQ, no LDLQ) | 0.10207 | 0.2719 | 23.931 |
| int3 RTN — gate-DEAD (#355) | 0.16360 | 0.5411 | 69.475 |

Ordering `fp16 ≤ int8 ≤ cb3-ldlq ≤ int4 ≤ int3` holds on both tensors (`i_full_order_monotone_kv_l2rel`). cb3-ldlq brackets [int8, int4] on KV **and** hidden, flips below crude-int4, and is careful-VQ-class on PPL (below fp16, far below crude-int4's +18% inflation). The literal **data-free** recipe over-perturbs *past* int4 — confirming the QuIP#/QTIP error-feedback (LDLQ) step is load-bearing, exactly as I flagged before the run. **The fake-quant itself is faithful; the flip→coverage propagation is the loose link.**

### What happened (honest analysis)

1. **The int8 anchor was the problem, not the proxy's fidelity.** int8 quantizes at 8 bpw; real cb3 at 3.25 bpw. The faithful cb3 rung perturbs the shared-KV drafter read **~6.7× more** than int8 (0.0478 vs 0.0071) and flips body-argmax **~6.8× more** (0.112 vs 0.016) — exactly as physics predicts for 3.25 vs 8 bpw. So #410's int8-anchored 15.74 TPS was an **optimistically gentle** bound, not a tight one. A direct cb3 measurement corrects it *upward* to 107.73, not down. The hypothesis that real cb3 lands "near the int8 edge" is **refuted**; it lands mid-bracket.

2. **The binding looseness is the flip→coverage step, not the supply rung.** `band_hi = flip × 962.27` assumes every flipped body-argmax position is a *full unit* drop in the drafter's top-1 coverage (Δcov ≤ Δtop1, largest-single-position). That an 11.2% flip yields a UB of **102% of the entire demand lift** is itself proof the proxy is grossly pessimistic — the cross-term cannot eat the whole +105.59 lift while cb3 holds the PPL gate at 2.3812. A *more faithful supply rung makes this pessimistic UB looser*; it cannot make it tighter. So no fake-quant — however faithful — can tighten the band.

3. **Real tightening needs the blocked faithful drafter read, which I kept blocked and did NOT fabricate.** The only quantity that can tighten [0, band_hi] is the per-position top-K drafter-acceptance read — `blocked:unmeasured` per #401/#372 (no shipping cb3 kernel). The body-argmax flip stays the honest, pessimistic upper bound on Δtop1. The gap between this UB and reality is exactly what the proxy can't see and what a fake-quant can't fill.

4. **The cross-term is therefore NOT certifiable as negligible from the supply side.** This does **not** mean it *is* destructive — `destructive_not_excluded = True` means "not excluded," not "is destructive" (the realistic value, with band_lo pinned at 0, plausibly sits near 0). It means: the supply×demand haircut **cannot be banked at ≤14.9% on the int8 stand-in**. Honest status for kanna #416: `unresolved / blocked-on-faithful-drafter-read`, band correctly [0, 107.73].

### Validity / scope

- **0-TPS analysis card.** `analysis_only=True`, `no_hf_job=True`, `no_served_file_change=True`, `official_tps=0`. Deployed **481.53 TPS / PPL 2.3772 / 128÷128 UNCHANGED**; cb3 gate PPL **2.3812 ≤ 2.42** (provenance, untouched — this card never alters the served target token).
- **Harness fidelity:** codebook distortion 0.0297 (near-optimal 2D VQ), RHT round-trip 2.4e-6 (Hadamard-exact), 259/259 linears two-sided RHT, cb3-ldlq calibrated on a **full-rank** 64-prompt subset (min **15931 tok/linear ≥ 10240**). Self-test **32/32 invariants pass**; the lone unmet check `i_cross_term_not_destructive` is a *hypothesis* claim (8/9), correctly recorded but not gating harness validity. (This closes the one smoke-run miss `cb3ldlq_calib_full_rank` — now satisfied at full rank.)

### Command

```bash
cd target/ && CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -m \
  research.validity.cb3_real_rung_perturbation.cb3_real_rung_perturbation \
  --quant-sweep fp16,int8,cb3-ldlq,cb3-real,int4,int3 \
  --measure shared_kv_states,inputs_embeds_hidden \
  --calib-prompts 64 --max-prompts 128 --max-seq-len 512 \
  --wandb_group cb3-real-rung-perturbation --wandb_name ubel/cb3-real-rung-perturbation
```

- **Peak VRAM 18198 MB** (ref pass; comfortably within the 23 GB A10G). **Wall ~23 min** (load 9s + 64-prompt LDLQ Hessian calibration 905s + 6-scheme × 128-prompt sweep). **W&B run `plqx2sxn`.**

### Methodology note (deviation from the literal PR recipe — same one I flagged pre-run)

The PR's literal recipe (RHT + dim-2 K=64 VQ + invRHT, **data-free**) over-perturbs *past* crude int4 (KV L2-rel 0.102 > int4 0.073) because it omits the QuIP#/QTIP **LDLQ error-feedback** that the named recipe relies on for near-lossless quality. I implemented **both**: `cb3-real` = the literal data-free recipe → strict UB (261.62 TPS); `cb3-ldlq` = + block-pair GPTQ/LDLQ calibrated on the deployed prompts → the **faithful** rung (107.73 TPS, headlined). `tr(E·H·Eᵀ) = E‖E·x‖²` is exactly the activation perturbation this card measures, so LDLQ minimizes the measured objective directly. The headline propagates the faithful rung; the data-free UB is reported beside it.

### Suggested follow-ups

- **Headline correction for kanna #416:** retire the int8-anchored ≤14.9% supply×demand haircut. The honest body-argmax-flip UB for the *real* cb3 rung is 107.73 TPS (band [0, 107.73]); treat the cross-term as **blocked-on-drafter-read**, not negligible. The combined fastest-equivalent number should not assume a tight supply×demand bracket.
- **The only true tightening path is the faithful per-position top-K drafter read** (#372/#401 cb3-kernel wall). Until a cb3 kernel exists, the gap between body-argmax flip and true Δcoverage is unmeasurable — so no analysis card can tighten this cross-term.
- **A margin-weighted flip proxy** could narrow band_hi *without* a kernel: weight each body-argmax flip by its logit margin (near-ties move the drafter's accepted top-1 far less than confident flips). The current "every flip = full unit coverage drop" is the maximally pessimistic choice; a margin model would give a tighter-than-107.73 honest UB if #416 needs one before the kernel ships. Happy to spin this as a fast 0-TPS card if you want it.
