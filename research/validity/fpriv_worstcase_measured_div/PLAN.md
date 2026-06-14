# PR #243 — NLS worst-case f_priv under the corrected 0.73% divergence

**Bank-the-analysis (PRIMARY = self-test; adds 0 TPS). CPU-only. No GPU/vLLM/HF/submission/draw.
BASELINE stays 481.53.** `--wandb_group issue192-reading-calibration`.

Re-prices my #233 worst-case-vertex f_priv as a divergence-weighted BLEND, swapping the OLD
int4-divergence weight (kanna #114 M=1 56.08%) for lawine #232's MEASURED near-greedy 0.73%.

## The frame (the blend)

    f_priv_wc(d) = (1 - d)·f_clean + d·f_int4div

- `d`          int4 divergence fraction (the weight on the adverse int4-divergent decode-drop)
- `f_clean`    the clean-decode f_priv at the binding NLS vertex = 0.969107 (#226/#217)
- `f_int4div`  the fully-int4-divergent decode-drop, PINNED by the round-trip so that at the OLD
               d=0.5608 the blend reproduces #233's grounded floor 0.957054 (calibration round-trip).

## Imported (NOT re-derived)

- #233 (`pszvrf2a`): `f_priv_breakeven_publish_first=0.9597799742440889`, realizable worst-case band
  [0.957054, 0.969107], `lambda_floor_central=0.97804`, `d(λ_floor)/d(f_priv)=-2.3535`.
- #226 (`tzcc5xuq`): NLS (`native_multilingual`) is the f_priv-minimizing vertex, f_clean=0.969107;
  runner-up `native_code` 0.969269; six realizable axes.
- lawine #232 (`nxwv6pam`): deployed M=8 divergence **0.73%** (0.9927 identity) — the CORRECTED weight.
- kanna #114 (`9q5yy9l1`): M=1 **56.08%** — the OLD weight #233's band implicitly used.
- #224/#52 grounded f_priv=0.957054; CEIL_INT4=520.9527 (#204); TARGET=500.

## Deliverables

1. The blend frame; solve f_int4div from the d=0.5608 round-trip (reproduces [0.957,0.969]).
2. Re-price `fpriv_worstcase_under_measured_div = f_priv_wc(0.0073)` + corrected `lambda_floor` via
   the -2.3535 sensitivity. Straddle verdict: does it still straddle breakeven 0.9598 or move ABOVE?
3. NLS vertex confirmation under the corrected divergence; one table d ∈ {0.0073, 0.10, 0.30, 0.5608}
   × (f_priv_wc, implied λ_floor, straddles-breakeven bool).
4. Self-test (PRIMARY): (a) d=0.5608 round-trips the band; (b) f_priv_wc ↑ as d↓; (c) corrected >
   old worst-case; (d) straddle verdict stated; (e) NLS vertex confirmed; (f) NaN-clean.
5. Hand-off to kanna's f_priv-band + fern's card + #124.

PRIMARY `fpriv_worstcase_measured_div_self_test_passes` + TEST `fpriv_worstcase_under_measured_div`.

## Reproduce

    cd target/ && CUDA_VISIBLE_DEVICES="" python \
      research/validity/fpriv_worstcase_measured_div/fpriv_worstcase_measured_div.py \
      --self-test --wandb_group issue192-reading-calibration \
      --wandb_name stark/fpriv-worstcase-measured-div
