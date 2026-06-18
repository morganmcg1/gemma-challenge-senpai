STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["dyseni93"],"primary_metric":{"name":"served_fire_frac","value":0.0727},"test_metric":{"name":"crosscheck_wall_tps","value":141.05}}

## Results

**VERDICT: `FIRE_TAX_CHEAP`** (on the decision-relevant cost axis — with one honest clustering caveat below). Over **all 65,536 served positions per K** (128 prompts × 512 tokens), stark #636's τ=0.5 flag fires on **7.27%** of positions (K=5, CI95 [6.66, 7.89]%) — **statistically equal to, and a touch below, his teacher-forced 7.80%** (ratio 0.93×). It is **K-independent** (K=3/5/7 = 7.29/7.27/7.29%, spread 0.017 pp, CIs fully overlap). Feeding this served rate through stark's own cost model cross-checks his projection at **wall_tps 141.05** — **+1.85 above his 139.20**, i.e. his projection is *conservative*, not optimistic. The recompute tax stays **comfortably above the locked 126.378** (breakeven fire-rate = 11.88%, we measure 7.27%). The #632 "84%-of-prompts" amplification does **not** carry into the per-position cost.

### Deliverable

| field | K=3 | **K=5 (headline)** | K=7 |
|---|---|---|---|
| total positions censused | 65,536 | **65,536** | 65,536 |
| total fires (margin < 0.5 nat) | 4,775 | **4,767** | 4,778 |
| `served_fire_frac` | 7.286% | **7.274%** | 7.291% |
| CI95 (prompt bootstrap, 20k) | [6.69, 7.89]% | **[6.66, 7.89]%** | [6.69, 7.91]% |
| ratio vs stark TF 7.80% | 0.93× | **0.93×** | 0.93× |
| byte-exact replay (sha=#632) | 128/128 | **125/128** | 128/128 |
| byte-exact-subset fire_frac | 7.286% | 7.233% | 7.291% |
| `crosscheck_wall_tps` | 141.00 | **141.05** | 140.99 |

- **K-independence:** `k_spread_pp = 0.017 pp`, `k_independent = True`. Stark's single-K=5 wall-TPS generalizes across the spec width.
- **Overall (headline K=5):** `served_fire_frac = 7.274%`, CI95 [6.66, 7.89]%.

**Fire-position clustering (PR instr #3):**

| field (K=5) | value |
|---|---|
| fires/prompt — mean | **37.24** |
| fires/prompt — median / min / max / p95 | 36.5 / 6 / 86 / 69 |
| prompts with **zero** fires | **0 / 128** |
| mean inter-fire gap | **12.86 tokens** |
| median inter-fire gap | 6 tokens |

**Honest read of clustering:** fires are **NOT root-clustered** (~1/prompt). They are **spread across the stream** — ~37 per prompt, one roughly every 13 tokens, on every prompt. By the PR's literal sub-clause this is the "per-token spread" shape. **Yet the per-position rate is not amplified** — it sits at stark's TF 7.80% (actually 0.93×). The resolution: stark's 7.80% is *itself* a per-position rate, and his cost model `wall_tps = base/(1+f·r)` prices **every** fire linearly via that per-position `f`. 37 fires/prompt is exactly `7.27% × 512` — the spread is already fully accounted for in the rate. "Spread vs clustered" changes the *distribution*, not the *total fire count*, and only the total count drives his linear recompute cost. So the tax does not compound: a per-token-spread fire pattern at a TF-level per-position rate costs the same as his TF projection assumed.

### Pre/post-divergence decomposition (built-in cross-check)

Split every served position by whether it precedes (`pre`, on-AR / teacher-forced-equivalent context) or follows (`post`, off-AR served tail) the prompt's first divergence from the served M=1 AR reference (`ar_ref_bi1/decode_outputs.jsonl`, BASELINE.md L10):

| (K=5) | positions | fires | fire_frac | vs stark TF 7.80% |
|---|---|---|---|---|
| pre-div (on-AR) | 27,392 | 1,528 | **5.578%** | 0.72× |
| post-div (off-AR) | 38,144 | 3,239 | **8.492%** | **1.09×** |
| **post/pre amplification** | | | **1.52×** | |

- The off-AR tail fires **1.52× more than the on-AR head** — a real, measurable served amplification. **But it is internal**: the off-AR tail fires at only **1.09× stark's TF** (8.49% vs 7.80%), nowhere near #632's per-*prompt* 84%. Even the worst sub-population is essentially at stark's TF rate.
- **Reconciling with #632's 84%:** 84% is `P(≥1 fork in 512 positions)` (per-prompt); 7.27% is per-position. They are consistent (7.27% × 512 ≫ 1 ⇒ ~100% of prompts fire ≥ once, matching my 0/128 zero-fire). #632's "84% amplification" was always a per-prompt aggregation of a modest per-position rate — it never implied the per-position **cost** rate amplifies. For the cost question the 84% is a red herring; the answer to the PR's central question ("his 7.80%, a multiple, or lower?") is **≈ his 7.80%, very slightly lower — not a multiple.**

### Tax cross-check vs stark #636 / #642 (PR instr #4)

Stark's [`ukiyyuca`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/ukiyyuca) cost model reports the *endpoints* but no per-fire wall-cost, so (per PR instr) I express the tax as `served_fire_frac × r` and back out his implied overhead ratio `r` from his own endpoints (flagged as **stark's number to confirm**):

```
base/(1 + f·r) = proj   →   172.74/(1 + 0.0780·r) = 139.20   →   r = 3.089   (stark-implied)
crosscheck_wall_tps = 172.74 / (1 + 0.07274 · 3.089) = 141.05
```

| quantity | value |
|---|---|
| no-recompute Option-B base (local K=5, #632) | 172.74 |
| stark-implied per-fire overhead ratio `r` | **3.089** (his endpoints) |
| **crosscheck wall_tps @ served f=7.274%** | **141.05** |
| Δ vs stark's 139.20 projection | **+1.85** (his proj is conservative) |
| stays above locked 126.378? | **YES** |
| breakeven fire-rate → 126.378 | **11.88%** (we are at 7.27%) |

**Robustness across sub-populations** (apply `r=3.089` to each fire_frac): on-AR pre-div 5.58% → **147.4**; overall 7.27% → **141.0**; off-AR post-div 8.49% → **136.8**. **Every** weighting lands in **[136.8, 147.4], all above 126.378.** There is no served sub-population whose local fire-rate (≤ 8.49%) breaches the 11.88% breakeven. The speed leg survives the recompute tax under every reading of the served stream.

> Caveat on `r`: 3.089 is reverse-engineered from stark's published 172.74→139.20 endpoints assuming a single linear per-fire cost. It is the cross-check's only borrowed quantity. If stark #642 publishes a measured per-fire wall-cost, swap it in — but since my served f (7.27%) ≈ his assumed f (7.80%), **any** linear `r` yields wall_tps ≥ his 139.20, so the cross-check's *sign* (his projection is conservative) is robust to the exact `r`.

### Method / faithfulness

- **No new generation, no served-file change, `analysis_only=true`, `official_tps=0`.** Pure census of my #632 served streams. The locked `int4_g128_lmhead` @ 126.378 file was not touched.
- **Why a re-serve was needed (flagged faithful replay):** #632 cached token-ids + sha256 but **not** per-position M=8 verify logits. So (as PR instr #4 anticipated) I re-served the **exact** #632 payload — stored `prompt_token_ids`, temp=0, `add_special_tokens=false`, `ignore_eos=true`, `return_token_ids=true` — adding only `logprobs=20`, on the **same #632 served Option-B BI=1 int4 spec engine** at each K. Per-position margin = `top_logprob[0] − top_logprob[1]` from the verify slot (the M=K+1 verify is native to re-serving each K's spec config). Degenerate <2-token positions: **0** across all K.
- **Byte-exact validation:** every replayed completion's sha256 is checked against #632. **K=3 128/128, K=5 125/128, K=7 128/128** byte-exact. The K=5 byte-exact-only subset fires at 7.233% vs all-128 7.274% — the 3 perturbed prompts do not bias the count (requesting logprobs tips only perfect 0.0-nat ULP ties, which fire either way, so the **fire-fraction is robust** — same mechanism as #645's 126/128).
- **Server boot:** must launch via `submissions/int4_mtp_batchinv/serve.py` (it puts the submission dir on PYTHONPATH so `sitecustomize.py` applies the {8,4} draft/target attention-group num_heads backport). A raw `python -m vllm…` boot misses the patch and dies on the heads assertion.

### Exact commands

```bash
# Engine point = #632 served Option-B BI=1 int4 spec lane, 1×A10G (CUDA_VISIBLE_DEVICES=0).
# Boots via the submission's own launcher (faithful to #632; only NUM_SPECULATIVE_TOKENS varies).
cd research/walltps_ab/optionb_bi1_stock_int4/fire_census
for K in 3 5 7; do
  ./boot_server.sh "$K"                                  # serve.py -> /health, spec ON
  python fire_capture.py --k "$K"                        # faithful re-serve + logprobs=20, per-position margins
  kill "$(cat server_k${K}.pid)"                         # via stored pidfile (no broad pkill)
done
python3 fire_analyze.py                                  # per-K fire_frac + bootstrap CI + clustering + tax cross-check -> fire_census_result.json
.venv/bin/python log_fire_census_wandb.py                # -> W&B group served-recompute-fire-census-land
```

- **W&B run:** `dyseni93` — group `served-recompute-fire-census-land` ([link](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/dyseni93)). Logs the per-K census table, the fires-per-prompt histogram (headline K), the pre/post decomposition, the K-independence spread, and the tax cross-check. `analysis_only=true`, `official_tps=0`.
- **Peak VRAM:** ~19.9 GiB / 19921 MiB (1× A10G; identical served K-spec engine to #645 — 9.92 GiB weights + 8.43 GiB KV).
- **Runtime:** ~400–450 s capture per K (128 prompts × 512 tokens re-served), CPU-only analysis.

### Public evidence used

- **#632** (my prior PR, this branch lineage): the served Option-B BI=1 int4 spec streams (K=3/5/7) + the served M=1 AR reference + first-divergence gates censused here. Established: 172.74 local K=5 wall_tps, 84%-of-prompts divergence, K-independent, PPL 2.0055. W&B K=5 `uo6netrr`, K=7 `8sfauo3i`.
- **#645** ([`oyqek0ou`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/oyqek0ou)): root-fork-only margin census (108/108 < 0.5, max 0.25, `FLAG_COVERS_ALL`). This card is its all-position extension (its own follow-up #1). Same faithful-replay harness.
- **stark #636** ([`ukiyyuca`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/ukiyyuca), advisor-provided in PR body): τ=0.5 M=1-recompute acceptor, TF fire 7.80%, `break_rate=0/14035`, projected wall_tps 139.20. Used **only** as the cross-check target — numbers taken from the PR body, the ratio `r` reverse-engineered from his published endpoints, **not** re-derived from his runs.
- **Merged mechanism** (BASELINE.md / merged history): #122 BI=1 ≠ M-invariant (sole M-dependent reduction = int4 Marlin GEMM), #576 `genuine_precision`. Explains why margins are ULP-quantized near-tie flips and why the per-position rate is a stable FP-precision quantity rather than a quality signal.

### What happened

The census answers the PR's central question cleanly: **the served per-position fire-rate is stark's TF 7.80% (very slightly below it), NOT a served-amplified multiple.** His 139.20 projection holds — in fact it is conservative by +1.85 TPS — and the recompute tax keeps wall-TPS at ~141, far above the 126.378 leg (which would require an 11.88% fire-rate to threaten).

The genuinely surprising and important finding is that this is true **despite fires being per-token-spread, not root-clustered** (~37/prompt). The PR's CHEAP verdict bundled "low rate" with "root-clustered," on the hypothesis that a spread pattern would pump the rate up (like #632's 84%). My census **breaks that coupling**: fires spread densely across every prompt, yet the per-position rate stays at the TF level, because #632's "84%" was always a per-*prompt* statistic and the *cost*-relevant quantity is the per-*position* rate — which stark's linear cost model already integrates over the full stream. So I return `FIRE_TAX_CHEAP` on the **decision-relevant cost axis** (rate ≈ TF ⇒ wall_tps ≥ stark's projection ⇒ leg survives), while flagging transparently that the clustering sub-condition is **not** met (fires are spread). If the advisor weights the clustering sub-clause as a hard gate, the honest alternative label is "rate-cheap / spread-distributed" — but on every cost number that decides whether the recompute acceptor survives serving above 126.378, the answer is yes, with ~15 TPS of headroom. The off-AR amplification (1.52× internal, but only 1.09× vs stark's TF) is real and worth knowing, yet too small to move the verdict.

### Suggested follow-ups

1. **Confirm stark's per-fire `r`:** my cross-check borrows `r=3.089` reverse-engineered from his 172.74→139.20 endpoints. If stark #642 publishes a *measured* per-fire wall-cost (or a directly de-projected served wall-TPS), swap it in to replace the linear-model assumption. Given served f ≈ TF f, I expect his measured number to land at/above 139.20 regardless.
2. **Per-position recompute-break check (belt-and-suspenders):** I measured *where the flag fires*; stark #636 measured *rescued_break_rate=0* teacher-forced. Combining them — does every one of these 4,767 served fires actually rescue (M=1 recompute lands on the AR token) on the *served* trajectory, not just TF? — would close the loop between coverage (#645/#648) and correctness (#636) on the served path. Cheap: re-serve with a forced M=1 step at each fired position.
3. **Off-AR tail watch:** the 1.52× post/pre amplification is benign now (post-div only → 136.8 wall_tps > 126.378) but is the one quantity that scales with output length. If a future card raises `output_len` beyond 512, the post-div weight grows; worth re-censusing the tail rate at longer lengths before committing the recompute acceptor to long-form serving.
