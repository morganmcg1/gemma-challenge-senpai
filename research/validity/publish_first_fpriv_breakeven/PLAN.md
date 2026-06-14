# PR #233 — Publish-first f_priv-breakeven

**Bank-the-analysis (PRIMARY = self-test; adds 0 TPS). CPU-only. No GPU/vLLM/HF/submission/draw.**

The f_priv-axis companion to kanna #228's λ-axis publish-first floor. Prices the
f_priv-axis of the human's #124 publish-first POINT-estimate launch gate.

## The model (the private MEAN, not the bar)

    private_mean(λ, f_priv) = K_cal · (E[T](λ) / step_int4) · τ · f_priv
                            = ceiling(λ) · f_priv,      ceiling(1) = 520.953

This is kanna #224's object (the achieved private mean = ceiling × f_priv), DISTINCT
from my #226 private *bar* (the build target = mu_safe_fresh / f_priv).

## Imported (NOT re-derived)

- K_cal = 125.268 (#148/#169), step_int4 = 1.2182 (#168), τ = 0.9924318649123313 (#181)
- int4-spec physical ceiling = 520.9527323111674 (#204) = K_cal·(E[T](1)/step)·τ
- f_priv central = 0.969106920637722 (#217, my #226 realizable worst-case)
- f_priv empirical-floor = 0.957054 (#52 lone hard paired draw)
- kanna #224 anchors at λ=1: 504.86 (f_priv=0.969) / 498.58 (f_priv=0.957)
- E[T](λ) reach-DP (#175/#184) imported unchanged
- kanna #228 lambda_floor_publish_first (central) imported if banked, else computed here

## Deliverables

1. `private_mean_vs_fpriv_at_ceiling` over f_priv ∈ {0.957054, 0.95977, 0.969107}
2. `f_priv_breakeven_publish_first` = 500 / 520.953 ≈ 0.95977; `publish_first_at_ceiling_verdict_flips`
3. `lambda_floor_central`, `lambda_floor_empirical_floor` (∅ sentinel), `dlambda_floor_dfpriv`
4. Honest band (calibration-tail vs realizable worst-case; P95 stays 0.9780)
5. PRIMARY `publish_first_fpriv_breakeven_self_test_passes` + TEST `f_priv_breakeven_publish_first`

## Reproduce

    cd target/ && CUDA_VISIBLE_DEVICES="" python \
      research/validity/publish_first_fpriv_breakeven/publish_first_fpriv_breakeven.py \
      --self-test --wandb_group winners-curse-redraw-budget \
      --wandb_name stark/publish-first-fpriv-breakeven
