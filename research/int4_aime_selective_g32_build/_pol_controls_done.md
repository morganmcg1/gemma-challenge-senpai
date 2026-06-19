STUDENT ubel: proof-of-life — both controls complete, selective (decision) arm now running.

**Run:** https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/nqk9izab (group `int4-aime-selective-g32-build-ubel`; verdict resumes into this same run). Disk holding ~20G, no trouble. No HF Job / submission — `analysis_only=1`, `official_tps=0`, `no_hf_job=1`, `fires=0`.

**Controls done (both 5-seed pooled, #31 gate basis, n=300, Wilson z=1.96):**

| arm | pooled | n_corr | Wilson95 | per-seed | int4 anchor |
|---|---|---|---|---|---|
| full_g128 | **0.3033** | 91/300 | [0.2541, 0.3576] | 0.25 / 0.283 / 0.333 / 0.367 / 0.283 | 0.347 |
| full_g32 | **0.3867** | 116/300 | [0.3333, 0.4429] | 0.40 / 0.30 / 0.417 / 0.417 / 0.40 | 0.438 |

**Linchpin passes (your control-fidelity test).** full_g32 = 0.387 [0.333, 0.443] reproduces the int4 anchor 0.438 *within the ~0.05 bf16-serve offset* (full_g128 = 0.303 sits the same ~0.044 below its 0.347 anchor → the offset is **uniform**, not a quant bug). So the serving path is faithful and the g128 AIME gap is a **real recipe property**, exactly as you predicted. `recovery_fraction` self-corrects against these measured controls.

**Scale-fair clearing target.** Measured control separation g32−g128 = **0.0833** (only ~1.5× the per-arm Wilson half-width ~0.05 → `recovery_fraction` will be noisy; I'll report its propagated uncertainty, not just the point estimate). For an int4-scale projected clear, `rf ≥ 0.8022` → the selective arm must measure **pooled ≥ 0.370** (≈ the full_g32 ceiling). The #700 proxy claims 79.5% impact-energy in the subset, so this lands right on the knife's edge by construction — a clean falsification test.

**Selective arm live.** READY 07:21:05Z, EVAL seed 0/5 running (build confirmed `g32-direct=48 / fake_quant_g128=295`, lm_head held g128, subset = #700 `vjhzcvmu` 40 PLE-gate + 3q + 3k + 2v). 5 seeds × ~22 min → **ETA ~09:11Z**. I'll aggregate {full_g128 / selective / full_g32} with propagated `recovery_fraction` CI and post the verdict on BOTH the absolute 0.420 bar and the int4-scale projection.
