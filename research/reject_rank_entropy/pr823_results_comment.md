STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["ud7dqzxq"],"decision":"clean-negative","margin_dominates_uniform":false,"recommended_variant":"uniform_topk_no_margin_gate","primary_metric":{"name":"projected_best_uniform_tps","value":415.82},"test_metric":{"name":"best_margin_retention_pct_same_k","value":25.5}}

## Results

**Clean negative: a margin gate does NOT dominate uniform top-k. Ship plain uniform-k (no margin gate) to stark #816.** Margin (`dp/ap`) is the *right* classifier — it cleanly isolates the rank-2 near-misses where entropy (#820) could not — but that very separation proves uniform-k's speed lives in the LOW-margin (genuine verifier-disagreement) region. Removing the genuine-error relaxes removes almost all the speed; they are coupled. 0-submission, local-only offline projection over my own #820 capture (no re-serve, **0 new GPU**, no kernel change), exactly as instructed.

| question | answer |
|---|---|
| Does margin separate rank-2 near-misses from high-rank scatter (where entropy didn't)? | **YES** — high-margin (m≥0.5) rejects are **88.2%** rank≤2; low-margin **24.7%**. Entropy stayed ~32% and *inverted*. |
| Does a margin gate recover most of uniform-k's speed while removing the genuine-error relaxes? | **NO** — requiring even margin≥0.5 keeps only **9–26%** of uniform-k's gain (TPS 416→267 at k=16). |
| Is there a quality-safe `(k, margin)` that keeps ≥90% of the gain? | **NONE** — best retention at any margin>0 is **26%** (k=2, m=0.5). |
| What to hand stark #816? | **Plain uniform top-k, no margin gate.** Its quality cost is intrinsic and must be *measured* on the task gates, not gated away. |

### Method + faithfulness (reuses #820 capture `0r80mau9` — 0 GPU)
New offline twin `scripts/profiler/reject_rank_margin_project.py` **imports #820's validated pure helpers** (oracle fit, strict-stats, uniform projection, block-replay) so the method is byte-identical and directly comparable. It reads the per-draft-position JSONL #820 already wrote (19,538 blocks / 117,228 positions, vocab 262,144; per position: `acc, rk, H, dp, ap`). Margin gate = drop-in addition to the accept test:

```
accept(d) = strict-argmax (acc==1)  OR  ( rank(d) ≤ k  AND  dp/ap ≥ margin )
```

so `margin=0` reduces **exactly** to uniform top-k. Self-consistency reproduces #820 to the digit:

| | strict E_accept | r | uniform k=2 / k=5 / k=16 relaxes |
|---|---|---|---|
| #820 baseline | 3.379 | 0.397 | 7,793 / 18,360 / 28,419 |
| this twin | **3.3800** | **0.3967** | **7,793 / 18,360 / 28,419** ✓ |

Oracle TPS curve (stark, same anchors as #820): `TPS = 43.60 + 61.37·L`, R²=0.9989; strict E_accept→251.0 TPS (≈ the ~253 baseline). Within-block replay is exactly faithful (one teacher-forced pass fixes rank/`dp`/`ap` at each depth regardless of accept decisions); truncation cascades are honored (refusing an early low-margin reject truncates the rest of the block).

### Step 1 — margin SEPARATES where entropy did NOT (the decisive contrast)
`margin = dp/ap ∈ (0,1]` at the 40,638 reject positions: median **0.030**, mean 0.160; even at **rank-2 rejects** the margin is mostly small (median **0.287**, max **0.883** — no rank-2 reject is a true near-tie). High-margin rejects are rare: **12.1%** of rejects have m≥0.5, **3.4%** ≥0.8, **1.0%** ≥0.95.

**Margin-conditioned reject-rank CDF** (high = near-tie):

| margin thr | subset | ≤2 | ≤5 | ≤16 | n |
|---|---|---|---|---|---|
| 0.50 | **high (m≥thr)** | **0.882** | 0.999 | 1.000 | 4,901 |
| 0.50 | low (m<thr) | 0.247 | 0.566 | 0.799 | 35,737 |
| 0.80 | **high** | **0.977** | 1.000 | 1.000 | 1,381 |
| 0.95 | **high** | **1.000** | 1.000 | 1.000 | 420 |

**Side-by-side, the #820 entropy table (REFUTED — does not separate / inverts):**

| H thr (nats) | high-H ≤2 | low-H ≤2 |
|---|---|---|
| 1.0 | 0.254 | **0.395** |
| 2.0 | 0.151 | **0.348** |

→ Margin monotonically isolates rank-2 (P(rank=2 \| margin decile) climbs **0.141 → 0.967** across deciles); entropy is flat-to-inverted. **#820's follow-up #2 was correct: margin is the right signal.** But "right signal" here delivers bad news, not a lever (Step 2).

### Step 2 — the (k, margin) grid: near-ties are too rare to carry speed

| k | margin | TPS_gated | Δ% vs ~253 | retain% (same-k gain) | risky low-m relaxes dropped | safe high-m kept |
|---|---|---|---|---|---|---|
| 2 | 0.00 (=uniform) | 303.4 | +19.9 | 100 | 0 | 7,793 |
| 2 | 0.50 | 265.9 | +5.1 | **26** | 5,322 | 2,471 |
| 2 | 0.80 | 255.4 | +0.9 | 5 | 7,024 | 769 |
| 5 | 0.00 | 363.9 | +43.8 | 100 | 0 | 18,360 |
| 5 | 0.50 | 267.4 | +5.7 | **13** | 14,834 | 3,526 |
| 16 | 0.00 | 415.8 | +64.4 | 100 | 0 | 28,419 |
| 16 | 0.50 | 267.4 | +5.7 | **9** | 24,217 | 4,202 |
| 16 | 0.80 | 255.5 | +1.0 | 2 | 27,245 | 1,174 |
| 16 | 0.95 | 252.4 | -0.2 | -0 | 28,045 | 374 |

**Quality proxy — uniform-k's relaxes are riskier than entropy suggested.** At k=16, margin 0.5: **85.2%** of uniform-k's relaxes are low-margin (genuine-error risk) vs #820's 52.2% low-*entropy*. And the two signals disagree: of those low-margin relaxes, **44%** (10,635 of 24,217) are *high*-entropy — exactly the ones #820's entropy gate would have *kept* as "safe." Margin re-labels them correctly as risky.

### Step 3 — verdict: clean negative
No `(k, margin>0)` keeps ≥90% of uniform-k's gain (best is **26%**, k=2/m=0.5). Two compounding reasons: (1) high-margin near-ties are only ~12% of rejects, and (2) cascade truncation — a low-margin reject typically appears *early*, so refusing it also kills downstream high-margin relaxes (at k=16/m=0.5 only 14.8% of relaxes are high-margin, yet retention is 9%). The only truly "free" relaxes are the **420 exact-logit-tie rejects** (margin=1.0, byte-level ties; 0.36% of positions) — negligible speed.

**→ Ship plain uniform top-k to stark #816. The relaxation lever's quality cost is intrinsic (coupled to its speed) and must be MEASURED on the quality floors (PPL≤2.42, AIME≥0.090, MMLU-Pro≥0.572, GPQA≥0.471, GSM8K≥0.807, 128/128), not gated away by a cheap probabilistic threshold.** k is the speed↔quality dial the panel sets (no free knee, per #820).

### Repro
```bash
# 0-GPU offline projection over #820's existing capture (re-runnable, no re-serve):
python -m scripts.profiler.reject_rank_margin_project \
  --in-dir research/reject_rank_entropy/int4head \
  --wandb-group bi0-margin-gate-accept
```
- **Peak GPU: 0 (no new GPU).** Pure offline projection over #820's JSONL. (The underlying GPU capture in #820 peaked ~19.7 GiB on the 23 GiB A10G.)
- **W&B:** run [`ud7dqzxq`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/ud7dqzxq), group `bi0-margin-gate-accept` (full projection JSON uploaded as artifact).
- **Artifacts:** `research/reject_rank_entropy/int4head/{reject_rank_margin_projection.md,reject_rank_margin_projection.json}`; script `scripts/profiler/reject_rank_margin_project.py`.

### What happened
The hypothesis was that margin would do what entropy couldn't — separate benign near-ties from genuine errors — and thereby recover most of uniform-k's speed while removing the risky relaxes. **The first half is confirmed and the second half is refuted, and the two together are the finding.** Margin *is* the correct signal (high margin ⟺ rank-2 near-tie, by construction; entropy was flat/inverted). But applying it reveals that uniform-k's +20–64% speed comes overwhelmingly (85% at the k=16 ship-ceiling) from positions where the verifier genuinely prefers a *different* token by a real probability margin — i.e. the speed and the quality risk are the *same* events. There is no quality-safe subset of meaningful size: the near-ties (m≥0.5) are 12% of rejects and, after truncation, worth ≤26% of the gain. This **doubly de-risks the team's fork**: #820 ruled out entropy gating; #823 rules out the better signal too — *no cheap probabilistic accept-gate (entropy or margin) makes uniform-k quality-safe.* The honest next step is a served quality measurement of uniform-k, not another gate.

### Suggested follow-ups
1. **stark #816: measure, don't gate.** Run uniform top-k through the task gates at k=2 first (smallest exposure: 7,793 non-argmax emissions), climb k while the gates hold. The ship-k is whatever the panel tolerates — this projection cannot answer it and no margin gate shortcuts it.
2. **If quality fails even at k=2**, the lever is dead on this drafter and the gain must come from a *better drafter* (raising the strict accept rate r≈0.397), consistent with the K=6 depth-exhaustion finding (#774) — not from relaxing the accept test.
3. **Do not re-attempt margin/entropy/threshold accept-gates** on this verifier — both signals are now mapped (entropy wrong, margin right-but-inseparable). This matches the public board: paxenos-gemma-boom closed a `MarginGate` lane (274–288 TPS) and CGD found "drafter margin is not a safe verify-skip oracle" (p_false_adj 43% vs ≤0.5% gate). Different mechanisms, same conclusion.

### Public evidence used
- **Leaderboard frontier ≈ 513.8 TPS** (digest `as=senpai`, 2026-06-20: rank-1 `w160-ctk42-noprecache-gemma-slayer-lean`) vs shipped bi0 official 218.02 (`s63tb03x`) — the +20–64% acceptance lever is materially relevant, which is why de-risking *how* to ship it (this PR) matters.
- **MarginGate is a known closed lane:** paxenos-gemma-boom closing summary (`20260617-130946-677_paxenos-gemma-boom.md`) lists "MarginGate (274–288 TPS)" and CGD's "drafter margin is not a safe verify-skip oracle … p_false_adj 43%." My result (verifier-side accept-relaxation margin, not their drafter-side verify-skip) independently lands the same conclusion from the other direction.
- **Quality surface = task gates, not greedy-identity:** dixie-flatline / fabulous-frenzy thread consensus ("the task gates are the only valid quality surface; greedy/byte identity is the legality surface") — directly supports follow-up #1 (measure uniform-k on the gates).
- Baselines from the PR body: int4head `7ntx4nrn` (E_accept 3.379 / r 0.397), oracle `zc76n7xz`; my #820 capture `0r80mau9`.
