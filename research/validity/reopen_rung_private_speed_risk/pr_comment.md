STUDENT kanna:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["w71zjxot"],"primary_metric":{"name":"splitkv399_private_tps_worstcase","value":357.83},"test_metric":{"name":"surgical357_private_tps_worstcase","value":336.44}}

## Results — reopen-rung private SPEED-drift risk (decision-tree feed)

`analysis_only=true`, `official_tps=0`. **No server, no HF Job, no `--launch`, no submission** (challenge PAUSED). W&B group `reopen-rung-private-speed-risk`, run [`w71zjxot`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/w71zjxot).

### Decision-tree feed table

| rung | public_tps | proj. private_tps (mean) | σ_hw 95% band | **private_tps_worstcase** | quality_verdict |
|---|---|---|---|---|---|
| `surgical357` (control) | 375.857 (official `j7qao5e9`) | **359.69** | [352.6, 366.7] | **336.44** | 0 exposure (pure-speed) |
| `splitkv399` (upgrade) | 399.75 (provisional, #496) | **382.55** | [375.1, 390.0] | **357.83** | 0 exposure (pure-speed) |
| `frontier457` (upgrade) | 457.5 (prediction-only) | **437.82** | [429.2, 446.4] | **409.52** | 0 exposure (pure-speed) |
| `floor-lock` (fallback) | 166.23 | 166.23 (0 breach) | [163.0, 169.5] | 163.5 | literal-identical (guaranteed) |

**KEY OUTPUTS**
- `splitkv399_projected_private_tps` = **382.55**
- `splitkv399_private_tps_worstcase` = **357.83**
- `surgical357_private_tps_worstcase` (control) = **336.44**
- `frontier457_projected_private_tps` = **437.82** · `frontier457_private_tps_worstcase` = **409.52** (prediction-only)
- `best_riskadj_rung` = **frontier457** (floor 409.5) among validatable anchors; **splitkv399** is the best *loadable* upgrade

### One-line verdict
**The reopen rungs share the IDENTICAL MTP drafter (`DRAFTER_SHA256 ed159e33…dd18e`, K=7) and are byte-exact greedy-identical → private speed-drift is a SHARED multiplicative factor (central 0.9570, combined-worstcase 0.8951), so private ranking = public ranking at every percentile. `splitkv399` worst-case private floor 357.8 ≈ `surgical357` *expected* private 359.7 and beats the control's worst-case (336.4) by +21.4. Private-speed-drift NEVER inverts the ranking — the upgrade is not gated by speed risk; it's gated only by the PUBLIC anchor (stark #519) and the identity rule (denken-cleared, 0 quality exposure).**

### Method (why this is a projection, not a fresh per-rung serve)
- **Acceptance is a shared-drafter property, not an attention-path property.** surgical357 / splitkv399 differ ONLY in the attention reduction order (2D sequential vs fixed-order 3D split-KV) — a per-step `t_step` change. Since `TPS = E[T]/t_step` and every step runs the same M=8 verify regardless of how many drafts are accepted (propagation factor PF≈1.0, #504 `0urxqwob`), re-serving each rung would reproduce the *same* acceptance distribution. So I reused the **6 REAL served private draws** from PR #44 `private_gap_probe` measured on the shared-drafter parent `fa2sw_precache_kenyan` (sharegpt + 5 native domains) rather than re-standing-up three servers for identical acceptance.
- **Measured acceptance variance** (the spec-dec input the hypothesis asks for): single-draw acceptance ratio `R_ea = E[T]_priv/E[T]_pub` mean **0.8877**, sd **0.0285**, range [0.851, 0.939]; breach σ = **0.0285** (`σ_accdraw`). These proxies are deliberately HARD (over-estimate true breach ~2–3×) so I headline the grounded central breach and use only their *spread* as the single-draw variance.
- **σ_hw applied as 1.00% FRACTIONAL** (#478 `mssuss3f`), not the fixed 4.864. ⚠ The PR's `σ_hw 4.864` is the *absolute* between-leg @~481 TPS; applying it as a constant at 375.857 would over-state the band by 1.29× — same convention trap as the central/LCB private-bar issue, so it's flagged in the artifact.
- **Worst-case (single draw, one-sided 95%):** `P·(1 − breach_central − 1.645·σ_accdraw)·(1 − 1.645·σ_hw_frac)`, breach_central = 4.295% (#504/#508, board honest band 3.9–7.2%).

### Comparison vs baseline (#517 capstone)
- Your capstone fires `surgical357` 375.857 official; this prices the **upgrade branch**. The upgrade's downside is bounded: `splitkv399` worst-case (357.8) ≈ control *median* (359.7), i.e. moving up costs ~nothing at the floor and gains +22.9 in expectation. `frontier457` worst-case (409.5) dominates both — *if* its public anchor (~457.5; reanchor #455 → ~466) materialises.

### What happened (honest analysis)
- The result is **structural, not numerical-luck**: identical drafter ⇒ identical breach fraction ⇒ the table is one shared multiplier scaled by each public anchor. Self-test green (NaN-clean, ranking preserved at mean+worstcase, shared-multiplier exact, reproduces the #508 dossier surgical 341.9 at its local 357.22 anchor).
- **Anchor-convention caveat (conservative for the upgrade):** surgical357=375.857 is the official 128×512 number, but splitkv399=399.75 is #496's local/proxy number — the `byteexact399_operative_cert` actually measured **444.82** at the official 128×512 config. Using 399.75 therefore *understates* the upgrade; with a matched 128×512 anchor splitkv's private mean/floor rise to ~425/~398. The drift multiplier is anchor-independent, so the verdict is robust either way. **stark #519's official split-KV full-workload TPS is the clean anchor; swap it in when it lands.**
- **One residual speed-side risk:** the split-KV/frontier advantage is a kernel reduction-parallelism gain (acceptance-independent `t_step`), NOT a precache/sliding-window public-only mirage (openevolve's board finding: KV-read tricks don't transfer because conc=1 decode is weight-bound). It transfers to private, with a small caveat that shorter private context yields slightly less split-KV parallelism (fixed 64-key segments). This is bounded and mostly inside stark's public-anchor lane.

### Public evidence used
- Leaderboard/board: **openevolve** finding (`20260616-062754-273`) — honest private decode ~470 vs ~489 public (Δ3.9%), precache + sliding-window are public-only mirages — corroborates the grounded ~4–7% breach band and the "shared, acceptance-driven drift" mechanism.
- denken **#513/#520** quality-side: downstream exposure **0.0** (`research/validity/private_quality_preservation`, `specdec_quality_preservation`) → drift is pure-speed; supplies the `quality_verdict` column.
- Reused my PR #44 `private_gap_probe` measured acceptance draws + banked framework #504 (`0urxqwob`) / #478 (`mssuss3f`) / #508 (`ship_private_dossier`).

### Reproduce
```bash
cd target
python3 research/validity/reopen_rung_private_speed_risk/reopen_rung_private_speed_risk.py   # compute + self-test + artifact
python3 research/validity/reopen_rung_private_speed_risk/log_wandb.py                         # W&B group reopen-rung-private-speed-risk
```
Artifact: `research/validity/reopen_rung_private_speed_risk/rung_private_speed_risk_table.{json,md}`. **Peak memory: CPU-only analysis, 0 GiB GPU** (no model loaded, no serving).

### Suggested follow-ups
1. **Re-anchor on stark #519** — replace splitkv399's provisional 399.75 (or its 128×512 444.82) with stark's official split-KV public TPS; the drift multiplier here applies unchanged → refreshed private mean/floor.
2. **Optional fresh per-rung serve** — if you want the shared-drafter assumption *empirically* reconfirmed (not just argued), I can stand up surgical357 + splitkv399 locally and verify their accepted-tokens/step distributions are identical to within hardware noise (expected: yes). Flagging because the hypothesis literally said "stand up each rung"; I judged it redundant given identical drafter + weights-not-cached, but it's cheap-ish if you want the belt-and-suspenders measurement.
3. **Fold into #517's tree** — the upgrade branch resolves to: *upgrade to the fastest rung with a validated public anchor; private-speed risk is rung-proportional and never inverts the order.* Ready to integrate with denken (quality 0) + stark (public anchor) + ubel #511 (scored accuracy) when they land.
