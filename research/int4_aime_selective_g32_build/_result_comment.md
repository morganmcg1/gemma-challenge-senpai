STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["nqk9izab"],"primary_metric":{"name":"selective_g32_aime_compliant","value":0.3467},"test_metric":{"name":"selective_recovery_fraction","value":0.520}}

## Results — `SELECTIVE_G32_PARTIAL`

The 48-module activation-critical subset (#700 `vjhzcvmu`: 40 PLE-gate + 3 q + 3 k + 2 v, **1.353% body params**, ~80% proxy impact-energy) carries **real** AIME signal — it recovers ~52% of the uniform-g32 gain, materially above the g128 floor (control separation `P>0`=0.984) — **but it does NOT project a clear of the 0.420 gate**. The #700 first-order INPUT-activation proxy is **directionally validated, quantitatively over-optimistic**.

### 3-arm AIME table — #31 gate basis, bf16-fake-quant serve, 5-seed pooled (n=300), Wilson z=1.96

| arm | pooled | n_corr | Wilson95 | per-seed | int4-Marlin anchor |
|---|---|---|---|---|---|
| **full_g128** (served recipe) | **0.3033** | 91/300 | [0.2541, 0.3576] | 0.250 / 0.283 / 0.333 / 0.367 / 0.283 | 0.347 |
| **selective-g32-on-48** (test arm) | **0.3467** | 104/300 | [0.2951, 0.4022] | 0.367 / 0.350 / 0.283 / 0.383 / 0.350 | — |
| **full_g32** (uniform ceiling) | **0.3867** | 116/300 | [0.3333, 0.4429] | 0.400 / 0.300 / 0.417 / 0.417 / 0.400 | 0.438 |

- **PRIMARY** `selective_g32_aime_compliant` = **0.3467** (Wilson-hi 0.4022)
- **TEST** `selective_recovery_fraction` = **0.520** (point); MC-propagated median 0.517, 95% CI **[−0.93, 2.08]** (denominator is only ~1.5× the per-arm Wilson half-width → the ratio is genuinely noisy, as pre-flagged)
- `selective_int4scale_projection` = **0.3943** [MC 0.262, 0.536]; **P(int4-scale proj ≥ 0.420) = 0.247**
- control separation g32−g128 = **0.0831** [0.0073, 0.1586]; `rf_clear_threshold` = 0.802

### Verdict is robust to the criterion debate (resolves my 08:01 open question)

The saturation concern I flagged (full_g32 ceiling Wilson-hi 0.443 ≥ 0.420) **does not bind the final verdict**, because the selective arm fails BOTH criteria:
- **Original absolute** PR trigger: selective Wilson-hi **0.4022 < 0.420** → does not clear.
- **Scale-fair int4 projection** (your 06:35 steer): 0.3943 < 0.420, rf 0.520 < 0.802 → does not clear.

Both agree → **PARTIAL**. No adjudication of the headline-criterion question is needed for this result.

### Controls validate the instrument (your linchpin test passes)

Both controls reproduce their int4-Marlin anchors within a **uniform ~0.044–0.05 bf16-serve offset** (full_g128 0.3033 vs 0.347; full_g32 0.3867 vs 0.438). The offset is uniform, not a quant bug → the serving path is faithful, the g128 AIME gap is a **real recipe property**, and `recovery_fraction` is computed on the measured controls (self-correcting for the offset), so it is unaffected. `aggregate.py` controls: `full_g128_reproduces_ref=True`, `full_g32_reproduces_ref=True`, `selective_between_controls=True`.

### What happened — the proxy over-promises, and recovery is diffuse

The #700 proxy claimed the 48-module subset carries **~80%** of the activation-weighted impact-energy. Fixing exactly those 48 modules to g32 recovers only **52%** of the realized g32 gain → proxy efficiency **0.52 / 0.80 ≈ 0.65**. The missing ~48% of recovery is spread across the **other 295 body modules** that the proxy assigned only ~20% of impact-energy. So the realized AIME-critical error is **more diffuse than the first-order INPUT-activation proxy predicted** — the localization is real (1.35% of params buying 52% of recovery is hugely concentrated) but it is not the *whole* AIME-critical set.

**Minimum wider-subset footprint to clear (bound, not a measurement).** Linear-chord interpolation between the two measured points (1.353% → rf 0.520) and (100% → rf 1.0) puts `rf=0.802` at **f ≈ 59% body params** → byte-law TPS ≈ **122.0** (a ~4.3 TPS hit vs the 126.378 anchor). Because the recovery curve is strongly concave (front-loaded), 59% is an **upper bound** — the true clearing footprint is smaller but cannot be pinned from two points. **Takeaway: a clearing subset is NOT speed-free at the 48-module footprint; it costs materially more than the 48-module subset's projected ~0.10 TPS.**

### Reproduce / command

```bash
cd target/
# 3-arm sweep: build dense bf16 fake-quant body per arm, serve once (BI=1, generation_config sampling), 5-seed #31-gate AIME, stop, rm build
bash research/int4_aime_selective_g32_build/run_sweep.sh          # arms=full_g128 full_g32 selective, seeds=0..4
/usr/bin/python3 research/int4_aime_selective_g32_build/aggregate.py     # pool -> verdict + MC recovery_fraction CI
/usr/bin/python3 research/int4_aime_selective_g32_build/log_wandb.py     # resume run nqk9izab, log table + artifact
```
- AIME protocol matched to lawine #693 / #650 ref: years 2024,2025-I,2025-II (n=60), k=1 sampled (T=1.0 top_p=0.95 top_k=64), max_tokens=12288, min_tokens=8, no-thinking, 5-seed pooled (300), Wilson z=1.96.
- Build = g32 dequant on the 48-module subset + g128 dequant on the other 295 body modules, lm_head held g128 (build log: `g32-direct=48 / fake_quant_g128=295`); all three arms share the identical bf16-dense serve path.

### Guards / ops

- **`analysis_only=1`, `official_tps=0`, `no_hf_job=1`, `fires=0`** (W&B summary scalars). No HF Job / `/v1/jobs:run` / `train.py --launch` / submission / leaderboard / served-file change. Locked `int4_g128_lmhead`@126.378 untouched. Cannot be a fire (126.275 < 126.378).
- **W&B run:** https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/nqk9izab (group `int4-aime-selective-g32-build-ubel`, resumed; verdict + 3-arm table + `selective_g32_subset48` manifest artifact logged). State: finished.
- **Peak memory:** ~20 GiB resident (model 16.37 GiB + KV 2.33 GiB + CUDA-graph 0.11 GiB; `GPU_MEMORY_UTILIZATION=0.92` on the 23 GiB A10G). GPU drained to 0 MiB post-sweep.
- **Disk:** held; bounded floor 1.1 G at the final build, freed to ~21 G after `rm -rf fq_selective` (KEEP_BUILD=0). No ENOSPC.

### Suggested follow-ups

1. **Intermediate-subset sweep (cheap, same instrument):** re-run the fake-quant arm at top-96 / top-160 / top-256 modules (by impact_sq/param) to pin the recovery curve and find the *minimum* footprint that projects `rf ≥ 0.802`. This converts the 59% upper bound into a measured clearing footprint + its exact byte-law TPS cost — the single most decision-useful next step.
2. **Faithful-serve build is justified ONLY if (1) finds a clearing subset within an acceptable speed budget.** That step needs `qat_unq` fetched + mixed-grid serialization (disk freed, re-opens the pruned-lm_head assert) — do not pay that cost until the fake-quant sweep proves a clearing footprint exists.
3. **Second-order localization proxy:** the first-order INPUT-activation proxy's 0.65 efficiency is the specific failure mode. An OUTPUT-sensitivity / Hessian-diag / GPTQ-error proxy may localize the diffuse residual better and shrink the clearing footprint.
4. **If no speed-cheap clearing subset exists,** the diffuse-recovery finding corroborates closing the recipe axis on quality and narrowing to the eval-protocol family (kanna #699 budget axis) + precision-allocation (fern #659), exactly the PR's PARTIAL branch.

This arm is denken #709's pre-registered falsification test: the activation-localized selective-g32 fix is **falsified as a standalone speed-flat clear** (P(clear)=0.25), but the subset is confirmed to carry real, concentrated AIME signal — the recovery program continues toward a wider-subset / better-proxy footprint, not a 48-module ship.
