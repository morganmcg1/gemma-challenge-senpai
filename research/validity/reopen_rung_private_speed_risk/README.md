# reopen-rung private SPEED-drift risk (PR #522, kanna)

Speed-side input for #517's reopen **upgrade** branch: per-rung single-draw
private speed-drift risk from spec-dec acceptance-rate variance + σ_hw.
`analysis_only=true`, `official_tps=0`. No server, no HF Job, no submission.

- `reopen_rung_private_speed_risk.py` — projection + NaN-clean self-test; writes the artifact.
- `rung_private_speed_risk_table.json` — the artifact (table + framework + acceptance draws + verdict).
- `rung_private_speed_risk_table.md` — human-readable table + method + verdict.
- `log_wandb.py` — logs to W&B group `reopen-rung-private-speed-risk` (run `w71zjxot`).

**Load-bearing fact:** the three reopen rungs share the IDENTICAL MTP drafter
(`DRAFTER_SHA256 ed159e33…dd18e`, K=7) and are byte-exact greedy-identical, so
acceptance is a shared random variable and the private drift is a **shared
multiplicative factor**. Private TPS ranking = public ranking at every
percentile → the upgrade is never penalized by private-speed-drift.

**Verdict:** `splitkv399` worst-case private floor ≈ **357.8** (≈ `surgical357`
*expected* private 359.7), beating `surgical357` worst-case (336.4) by +21.4.
Upgrade to the fastest rung whose PUBLIC anchor is validated (stark #519 for
split-KV; frontier457 prediction-only). Speed-side risk does not gate the
upgrade; quality is denken-cleared (0 exposure).

Reuses measured acceptance from PR #44 `private_gap_probe` (6 real served draws
on the shared-drafter parent) + banked framework #504 (`0urxqwob`) / #478
(`mssuss3f`) / #508 (`ship_private_dossier`).
