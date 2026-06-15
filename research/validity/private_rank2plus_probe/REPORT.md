# PR #263 — Private rank-2+ probe: does the width>1 tree recover the OOD E[T]-gap?

**Headline: NO. The tree's recovery mechanism DEGRADES out-of-distribution.**
Private rank-2+ draft coverage is **0.4768** (vs public **0.6532**, Δ **−0.1764**, −27.0% rel).
On the private proxy the true target-argmax token falls **beyond the width-4 tree more
than half the time** (beyond-top-4 = **0.5232** vs public 0.3468). `private_rank_recovery_robust = False`.

## Verdict table

| set | n_div | top-1 accept | rank-2+ coverage | beyond-top-4 | implied E[T] |
|---|---|---|---|---|---|
| **public** (anchor z6wi4z4v) | 12,869 | 0.7335 | **0.6532** | 0.3468 | 3.8445 (raw width-1) |
| **private** (measured he7glotf) | 17,677 | 0.5975 | **0.4768** | 0.5232 | 3.6406 (tree gap-recovered) |

`private_rank2plus_coverage + private_beyond_top4 = 0.4768 + 0.5232 = 1.0` (valid partition;
rank-1 contributes 0 at divergences by construction).

## The decisive linkage: the 505 TPS projection assumed public ρ

`tree_private_acceptance_gap` (ytxfi6zk) projected private tree **E[T]=4.92 → 505.46 TPS
("clears 500")** using the branch-salvage vector `rho_cond = [0.4165, 0.2655, 0.1908]` read
from `research/rank_coverage/rank_coverage_results.json` — i.e. the **public** z6wi4z4v ρ. It
degraded the *spine* (depth-1 width-1 acceptance) to private levels but held the *branch-salvage*
ρ at public values. **This probe measures that exact assumption** and it is false OOD:

| | ρ₂ | ρ₃ | ρ₄ |
|---|---|---|---|
| public (used by the 505 projection) | 0.4165 | 0.2655 | 0.1908 |
| **private (measured)** | **0.2782** | **0.1738** | **0.1226** |
| collapse | −33.2% | −34.5% | −35.7% (mean −34.5%) |

The private branch-salvage ρ-vector is ~34% weaker across all three lanes, so the 505.46
"clears 500" projection is **optimistic** — re-pricing the descent-walk topology with the
measured private ρ pushes it toward the `raw_proxy` regime (the same model gives 450 TPS at
the raw private drop). Path-A's private 500-clear is **structurally at-risk even with the tree**.

## Tree-recovery translation (bounded, self-test (d))

Using the width≤4 rank-2+ salvage as a fraction of the public salvage:
`private_tree_recovered_et = priv_raw + min(1, priv_cov/pub_cov)·(pub_raw − priv_raw)`
`= 3.0898 + min(1, 0.4768/0.6532)·(3.8445 − 3.0898) = 3.0898 + 0.730·0.7546 = 3.6406`.
The tree recovers only **73% of the intrinsic private E[T]-gap**; private E[T] stays depressed
at 3.64 (vs the 3.84 it would reach if coverage transferred). Bounded in [priv_raw 3.0898,
pub_raw 3.8445] ✓.

## Greedy/PPL-safety certificate

`rank_probe_analysis_only = True`. This is a rank-only forward (`--no-logits`, the proven #79
path) over a proxy dataset for MEASUREMENT — it does not change the served model, sampler, KV
cache, or emitted tokens; no served-file change, no HF Job, no submission. **BASELINE stays
481.53 TPS and the λ=1 ceiling 520.95 TPS — both unchanged (this leg adds 0 TPS).**

## Self-tests (PRIMARY = `private_rank2plus_probe_self_test_passes`)

- (a) public default reproduces banked 0.653 (resid ≤ 0.01): **PASS** (repro cov₄ = 0.6541, resid 0.0009, run pvt1rprm)
- (b) smoke_records > 0 (=74, fixed profiler) AND n_div_private > 0 (=17,677): **PASS**
- (c) partition cov₄ + beyond-4 = 1.0 (rank-1 = 0 at div): **PASS**
- (d) 3.0898 ≤ private_tree_recovered_et 3.6406 ≤ 3.8445: **PASS**
- (e) NaN-clean (align_bad=0 both runs): **PASS**
- (f) BASELINE 481.53 + 520.95 λ=1 ceiling unchanged (analysis-only): **PASS**

**PRIMARY `private_rank2plus_probe_self_test_passes` = True**
**TEST `private_rank2plus_coverage` = 0.4768**

## Hand-off (one sentence) to land #245 + fern #262 + human #124

*Private rank-2+ draft coverage is **0.4768** (vs public 0.653, Δ **−0.176**), so the width>1
tree recovers only **~73%** of the private E[T]-gap OOD — meaning the tree's recovery mechanism
**degrades** on the private distribution (branch-salvage ρ collapses ~34% to [0.278, 0.174,
0.123]) and Path-A's private validity is **structurally at-risk even with the tree**; land #245
must re-price the descent-walk topology with the measured private ρ (the 505.46 projection used
public ρ and is optimistic).*

## Public evidence imported (not re-derived)

- ubel #258 `2khp8gzs`: intrinsic discrimination loss, argmax-invariance ⇒ calibration share 0.
- `rank_coverage` `z6wi4z4v`: public rank2+ = 0.6532, beyond-4 = 0.3468, n_div 12,869, ρ [0.4165, 0.2655, 0.1908].
- `tree_private_acceptance_gap` `ytxfi6zk`: priv raw E[T] 3.0898, pub raw E[T] 3.8445, tree-priv proj 505.46 (USED public ρ).

## Profiler changes & fixes (scripts/profiler/rank_coverage.py — the only code change)

1. **`--dataset <path>` arg** (the assigned change): parameterizes the input prompt set only;
   drafter, target, rank-counting, greedy-exact verify unchanged; public default preserved.
2. **Fix — absolute `--out-dir`** (`args.out_dir = args.out_dir.resolve()`): the serve subprocess
   runs with `cwd=<scratch>`, so a RELATIVE out-dir doubled the records path (writer/reader
   disagreed → silent 0 records, indistinguishable from the dead #86 logits path). Resolving to
   absolute makes writer and reader agree. This was the first 0-records failure mode.
3. **Fix — prometheus `_IncludedRouter` boot guard in the scratch sitecustomize injection**
   (scratch-only, output-neutral): vLLM 0.22.1rc1 + prometheus_fastapi_instrumentator throws
   AttributeError in `_get_route_name` on pathless routes → `/v1/models` 500s → readiness never
   passes → 0 records. Ported the validated `_get_route_name` guard (kanna PR #177 / bjtwr9jn:
   token-ids 128/128 identical, PPL byte-identical, TPS +0.02%). Scratch-only — the committed
   served submission is byte-identical.

## ⚠ Latent bug flagged (NOT fixed here — out of scope for this analysis PR)

The **active 481.53 frontier `submissions/fa2sw_precache_kenyan/sitecustomize.py` lacks the
`_guard_included_router`** that the sibling `fa2sw_treeverify_kenyan` has (0 vs 6 matches). The
prometheus `_IncludedRouter` boot-500 **actually manifested on this local container** during
this profiling (it had to be guarded for records to flow) — corroborating darwin's 3× HF-runner
observation in #177. The 481.53 a10g-small job booted fine (intermittent/image-dependent), so
this is a **latent launch-boot risk**, not a guaranteed failure. Recommend landing the guard
(#71/#177) into the active frontier as cheap launch insurance — **separate served-file PR**.

## Suggested follow-ups

- **land #245**: re-price the descent-walk tree topology with the measured private ρ [0.278,
  0.174, 0.123] (not public [0.417, 0.266, 0.191]) — produces the honest private tree TPS; the
  505.46 figure is an upper bound.
- **fern #262**: the fidelity-safe shallow tree inherits the same OOD coverage collapse; its
  feasibility pricing should use private ρ.
- **human #124 / launch**: treat the private 500-clear as not-yet-established even with the tree.
- Bug-fix PR: port `_guard_included_router` into `fa2sw_precache_kenyan`.

## Reproduce

```
cd target/ && python scripts/profiler/rank_coverage.py --no-logits \
  --dataset data/private_proxy_sharegpt.json --num-prompts 128 --output-len 512 --seed 1 \
  --out-dir research/validity/private_rank2plus_probe \
  --wandb_group private-rank-probe --wandb_name ubel/private-rank2plus-probe
# 2-prompt smoke first: add --debug --no-wandb
# public reproduction (self-test a): drop --dataset, --wandb_name ubel/public-repro-rank2plus
```

- **Private run W&B**: `he7glotf` (wandb-applied-ai-team/gemma-challenge-senpai)
- **Public repro W&B**: pvt1rprm
- **Peak GPU**: ≈20.2 GiB reserved (GPU_MEMORY_UTILIZATION=0.90 of a 22.5 GiB A10G; KV cache 9.47 GiB available; the rank probe adds negligible memory)
