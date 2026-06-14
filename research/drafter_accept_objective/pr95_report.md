STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["8kzjyzxb"],"primary_metric":{"name":"lk_implied_ET_headroom_pct","value":2.4},"test_metric":{"name":"measured_drafter_top1_accept","value":0.7287}}

## Results — AMBER: corrected greedy LK headroom is +1.0–2.4% E[T] (NOT +8-10%); re-ranking channel closed, prediction channel untested — size with a cheap probe

**Headline:** the PR's "+8-10% E[T]" is the LK-Losses paper's **T=1 / sampling** figure. Under our **greedy (T=0)** verify the paper's OWN tables give a far smaller gain: **+2.4% for EAGLE-3, +1.0% for Medusa, +1.2% for MLP-speculator** — a 3-8× collapse. The **re-ranking channel is rigorously closed** (the head's argmax is already acceptance-ordered → 0.0% from re-weighting candidates), so any LK gain must come from the **prediction-improvement channel**, which #80 (likelihood-only) never tested. Net realizable greedy headroom: **+1.0…+2.4% E[T] → ~486–493 TPS** (vs the naive +8% → 520 the PR assumed). **AMBER:** marginal, real, and worth **sizing with a cheap LoRA/projection-layer probe** before any full GPU fine-tune — do **not** transfer the +8% headline, do **not** full-launch unsized, do **not** close the lane.

CPU-only analytical gate from committed data — **no forward pass needed**, no GPU, no served-file change, no HF launch. Greedy identity holds by construction (verify step untouched).

### Step 1 — measured per-position acceptance profile P(accept | k), k=1..7

Deployed-chain conditional acceptance `q[k] = P(accept k | reached k)` under greedy (accept = verifier argmax == draft top-1), from #76 accept-calibration (16,700 drafts), cross-checked vs wirbel #79 rank-coverage (16,524 records):

| k | P(accept\|k) #76 | #79 xcheck | cumulative C[k] |
|---|---|---|---|
| 1 | **0.7287** | 0.7335 | 0.7287 |
| 2 | 0.7590 | 0.7655 | 0.5531 |
| 3 | 0.7925 | 0.7966 | 0.4383 |
| 4 | 0.8217 | 0.8260 | 0.3602 |
| 5 | 0.8343 | 0.8408 | 0.3005 |
| 6 | 0.8353 | 0.8379 | 0.2510 |
| 7 | 0.8473 | 0.8496 | 0.2126 |

- **Shape: RISING**, +0.1185 from k=1 (0.7287) to k=7 (0.8473) — a positive selection effect (positions that survive deeper are "easy" contexts), not geometric/constant.
- **E[T] reconciliation:** `E[T] = 1 + Σ_k Π_{j≤k} q_j = 3.8445` via fern #88/#91 `score_tree_depthrank(build_linear(8), pvecs)` (closed form == engine == 3.8445), |err| vs reported 3.844 = **3.2e-4**. Reconciles exactly with E[T]=3.844, top-1 q=0.7287, ρ ladder [0.4165, 0.2655, 0.1908]. ✓

### Step 2 — LK-optimal E[T] ceiling vs #80's closure

**(a) Re-ranking channel = +0.000% (committed-data rigorous).** The only thing an acceptance loss can do *without* changing the drafter's weights is re-order its own top-W candidates by acceptance. Is rank-1 already the highest-acceptance token at every depth?

| k | P(rank1=true\|reach k) | P(rank2=true\|reach k) | margin | rank-1 best? |
|---|---|---|---|---|
| 1 | 0.7335 | 0.1057 | **+0.628** | ✓ |
| 2 | 0.7655 | 0.1010 | **+0.665** | ✓ |
| 3 | 0.7966 | 0.0841 | **+0.713** | ✓ |
| 4 | 0.8260 | 0.0745 | **+0.752** | ✓ |
| 5 | 0.8408 | 0.0693 | **+0.772** | ✓ |
| 6 | 0.8379 | 0.0721 | **+0.766** | ✓ |
| 7 | 0.8496 | 0.0616 | **+0.788** | ✓ |

`P(rank2=true) = (1−q79[k])·ρ2_by_depth[k]`; rank-3/4 smaller still (declining ρ ladder). **Rank-1 is acceptance-DOMINANT at every depth** (7× more likely than rank-2 at k=1, 13× at k=7) → CE did **not** mis-order the head → re-ranking buys **0.0% E[T]**. **Consequence:** any LK gain must come from the *prediction-improvement* channel (changing the drafter's weights so more contexts land the true token at rank-1), not re-ranking. (The +1.55% #76-vs-#79 gap is probe variance on the same head, NOT headroom.)

**(b) The corrected greedy ceiling (the load-bearing correction).** Researcher-agent pulled the LK-Losses (arXiv:2602.23881) paper's own T=0-vs-T=1 tables (Llama-3.1-8B target):

| architecture | T=0 (greedy) gain | T=1 (sampling) gain | our analogue |
|---|---|---|---|
| EAGLE-3 (recurrent) | **+2.4%** | +2.7% | **upper** |
| Medusa (MLP head) | **+1.0%** | +7.6% | **lower** (single-layer, #80) |
| MLP-speculator | +1.2% | +3.3% | — |

The "+8-10%" in the PR body is the **sampling/T=1 family** (Medusa T=1 = +7.6%). Under sampling, acceptance = `min(1, p_t/p_d)` depends on full-distribution match (where CE-vs-acceptance gaps live); under our **greedy (T=0)** verify acceptance = pure top-1 argmax-match, and CE is Bayes-consistent there, so the gap shrinks 3-8×. Applied to our E[T]=3.844:
- **greedy ceiling (EAGLE-3 analogue):** +2.4% → E[T] 3.936 → **~493 TPS**
- **greedy floor (Medusa/single-layer analogue):** +1.0% → E[T] 3.883 → **~486 TPS**

This is the **same greedy-collapse direction my #88 proved** (Traversal Verification: +4.57% sampling → provably 0 greedy). LK doesn't go all the way to 0, but it collapses hard.

**(c) Reconciliation with #80 — partial extension, not full closure.** #80 retrained the drafter under CE / KL-distill / recipe sweeps → all MTP parity, concluding the ceiling is *architectural (single-layer head capacity)*. The sharp question was whether that **likelihood** closure extends to **acceptance**. Answer: **the re-ranking half extends (channel-1 closed), but the prediction half does NOT fully extend** — #80 only varied *likelihood* objectives; LK's acceptance gradient (focus mass on the acceptance gap, paper §4.1) is genuinely untested on our head. So I can't honestly claim channel-2 = 0. **However**, #80's *single-layer* finding says our head resembles the **Medusa/MLP T=0 class (+1.0%)** more than EAGLE-3 (+2.4%) → realistic central nearer the **floor**.

**Near-miss positions:** rank-2 catches **41.65%** of rejections (ρ2=0.4165) — but this is **TREE fodder** (land #71 width-2 branch, realized root-to-leaf per #88), **not** a mis-ranking a *linear* acceptance loss can flip (demoting rank-1 @0.73 to promote rank-2 @0.11 loses far more than it gains). wirbel #86's anti-correlation (drafter confidence → higher ρ2, r=−0.9688) says the residual misses are *fundamental uncertainty*. So the rank-2 mass does **not** add to the linear-drafter LK headroom.

### Step 3 — gate

**`lk_implied_ET_headroom_pct = +2.4%` (corrected greedy ceiling), realistic band +1.0…+2.4% → AMBER** (rule: GREEN ≥ +3% / AMBER +1–3% / RED < +1%).

**Verdict: AMBER — corrected ceiling reported, lane NOT closed but NOT full-launch.** The prize is real but marginal (+1–2.4% E[T], ~486–493 TPS), an order of magnitude smaller than the PR's +8% premise, and #80's capacity ceiling pushes the realistic value toward the floor. The right move is to **size it cheaply** before spending real quota.

### Recommendation
1. **Do NOT** transfer the +8-10% headline (it is sampling-regime) and **do NOT** full-launch an LK fine-tune unsized.
2. **Cheap discriminating probe** (advisor-gated, ~10-20% of a full run): retrain only the MTP head's final projection (or a LoRA) under an LK-style acceptance loss (`-log Σ min(p,q)` / TV-hybrid) with our exact **greedy** eval harness; gate on `heldout_native_accept_per_step` / E[T]. **Stop condition: if ΔE[T] < +0.5%, close the lane** (residual T=0 gap too small for our single-layer head). If +1–3%, escalate to a full retrain via a training-request issue.
3. The rank-2 mass (ρ2=0.4165) belongs to **land #71's tree**, which already harvests it root-to-leaf (#88) — it compounds with the tree and does not need a drafter retrain.

### Reproduction
```
cd target/ && python scripts/profiler/drafter_accept_objective_gate.py --wandb \
  --wandb-name "fern/drafter-accept-objective-gate" \
  --wandb-group "drafter-accept-objective-gate"
```
- CPU-only, ~1s, peak RSS < 60 MiB. Output: `research/drafter_accept_objective/gate_results.json`.
- Inputs (committed): `research/accept_calibration/accept_calibration_results.json` (#76), `research/rank_coverage/rank_coverage_results.json` (#79, run `z6wi4z4v`), `research/rank_coverage/entropy_branching_results.json` (#86).
- W&B run: `8kzjyzxb`. **No HF Job, no submission, no served-file change → official bar UNCHANGED 481.53.**

### Public evidence used
Leaderboard row `fa2sw-precache-splitkv-linear-mtp-k7` (481.53 TPS, the frontier this gate sizes against) and the deployed MTP head `kenyan-duma` (483.41 TPS / PPL 2.3769, 128/128).

### What happened
The hypothesis (CE ≠ acceptance leaves +8% E[T] on the table) is **mostly but not entirely** dissolved by greedy verification. The +8-10% is a sampling-regime number; the regime-correct T=0 figure is +1.0–2.4% for our nearest architecture analogues. Committed data closes the re-ranking channel exactly (argmax already acceptance-ordered), but cannot rule out the prediction-improvement channel that produces the literature's residual T=0 gain — #80 only tested likelihood objectives. So this gate corrects the prize from "+8% → 520 TPS, build it" to "+1-2.4% → 486-493 TPS, probe it first," with #80's single-layer ceiling arguing for the low end.

### Suggested follow-ups
- Run the cheap projection-layer/LoRA LK probe above (advisor-gated training-request) to resolve the channel-2 uncertainty for our specific head; stop if <0.5%.
- If head **capacity** is ever raised (the #80-banked HASS/EAGLE-3 multi-layer drop-in), re-run this gate — a bigger head moves q and the EAGLE-3 (+2.4%) analogue becomes more apt.
- Orthogonal lever flagged by the research: **GTO (arXiv:2509.22134)** targets *draft-policy* misalignment (single-path training vs tree verification) with larger T=0 gains — relevant to **land #71's tree**, a different axis than this loss-objective gate.
