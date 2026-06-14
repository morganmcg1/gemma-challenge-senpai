# PR #226 — Private-bar worst-case hardening: f_priv over realizable domain blends

**Lane:** capstone of stark private-worst-case lane (#176 → #198 → #208 → #215 → this).
**Scope:** LOCAL CPU-only analytic worst-case LP over banked private legs. No GPU / vLLM / HF
Job / submission / served-file change / official draw. BASELINE stays 481.53. Bank-the-analysis
(primary = self-test, adds 0 TPS). `--wandb_group private-drop-shape-robustness`.

## The mechanism (imported, not re-derived)

- **f_priv model:** `f_priv(blend) = (1 - drop(blend)) * tau_low`, `tau_low = 0.9924318649123313`.
  - Reproduces kanna #217 `f_priv = 0.969107` at the NLS full-recovery drop `drop = 2.3502816%`
    (my #198 `llo1bzn3` `drop_both_176`): `(1 - 0.023502816) * 0.9924318649 = 0.969107`.
  - `drop(blend)` = both-bugs private TPS drop via #198 tree-DP (`et_of_spine`) on the blend's
    per-rung deficit `delta_d(blend) = Σ_axis p_axis · delta_d^(axis)` (linear in blend, #176 shapes).
- **private bar:** `private_bar(f_priv) = mu_safe_fresh / f_priv`,
  `mu_safe_fresh = private_bar_217 * f_priv_217 = 528.48 * 0.969107` (round-trips kanna #217's 528.48).
- **#52 observed:** `f_priv_obs = 460.85 / 481.53 = 0.95705` (arithmetic round-trip; worse than central).

## Six realizable axes (#176/#208, decode-drop-realizable, natural mass)

NLS (`native_multilingual`) is the λ-deficit-maximizing vertex (reach-weighted deficit 2.3490pp).
Re-point the #208 LP at the f_priv (private-TPS-drop) axis: is NLS *also* the f_priv-minimizing vertex?

## Deliverables

1. f_priv model (formula + both anchors).
2. f_priv_worstcase = min over realizable simplex (vertex argmax + Dirichlet interior sweep #208 check).
3. private_bar_worstcase = mu_safe_fresh / f_priv_worstcase vs (a) 528.48 central, (b) wirbel #199
   compliant-spec ceiling 536.66/LCB 525.73; f_priv_breakeven at the 536.66 ceiling;
   compliant_lane_private_feasible (bool).
4. Robustness: #176 reach-weights + decode-drop calibration bands; #198 NEGATIVE-coupling caveat.
5. Self-test (PRIMARY): a–f round-trips; `private_fpriv_worstcase_self_test_passes`.
6. Hand-off to fern #185 + kanna #224.

## Reproduce

```
cd target/ && CUDA_VISIBLE_DEVICES="" python research/validity/private_fpriv_worstcase/private_fpriv_worstcase.py \
  --self-test --wandb_group private-drop-shape-robustness --wandb_name stark/private-fpriv-worstcase
```
