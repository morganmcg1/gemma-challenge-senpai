STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["wvy2k7w7"],"primary_metric":{"name":"fast_stack_reaches_identity_1p0_with_canonical_tiebreak","value":0},"test_metric":{"name":"canonical_tiebreak_self_test_passes","value":1}}

## Results

**Verdict: RED — a canonical tolerance tie-break applied *consistently to both the M=1 reference and the M=8 verify* does NOT make the fast stack byte-identical. It closes all 3 original flips but introduces 8 new ones, dropping served identity 0.9965986 → 0.9909297 (strictly worse, not 1.0). The rule is genuinely zero-cost and PPL-safe — but zero-cost × not-equivalent = 0 TPS reclaimable. The fastest *realizable strictly-equivalent* implementation stays blanket-strict at 467.14 TPS; `upside_over_blanket = 0.0`.** PRIMARY: **`fast_stack_reaches_identity_1p0_with_canonical_tiebreak = False`**. Self-test **33/33** (≥20 ✓), `canonical_tiebreak_self_test_passes = True`. W&B **`wvy2k7w7`** (group `canonical-tiebreak-fast-stack-identity`). Scope: `analysis_only=True` / `no_hf_job=True` / `no_served_file_change=True` / `official_tps=0`.

**The whole result in one sentence:** #405 showed a *one-sided* lowest-id override is net-negative (14 new flips) because id-ordering is uncorrelated with correctness; this card tests the natural fix — apply the canonical ≤ε→lowest-id rule **identically to both stacks** so the override can't be one-sided — and it still fails (8 new flips), because the rule's own gate (`gap ≤ ε*`) is **not robust to the ±1-ULP (0.125-nat) bf16-attention perturbation of the gap**: at **8/8** new-flip positions the M=1 and M=8 gaps land on *opposite sides of ε\***, so the "consistent" rule fires on one stack and not the other, manufacturing a fresh divergence exactly where it was supposed to prevent one.

### Primary — canonical rule applied to BOTH sides (heuristic = served/fast path, 126×8-row decode = 882 positions)
The rule (PR spec): among tokens within `ε*=0.125` of the top logprob, pick the lowest token-id; at a confident argmax (`gap > ε*`) it reduces to plain argmax. Applied to **both** the M=1 self-reference and the M=8 verify (realizable via the self-referential scorer gate, land #414 `bq7xkfcv`). Identity is `canon(M8) == canon(M1)` per position.

| quantity | value | reading |
|---|---|---|
| `fast_identity_baseline_argmax` | **0.9965986** (3 flips / 882) | reproduces #381/#397 exactly |
| `fast_identity_canonical_tiebreak` | **0.9909297** (8 flips / 882) | **worse**, not 1.0 |
| `fast_n_flips_closed` | **3** | all 3 original flips fixed |
| `fast_n_new_flips_introduced` | **8** | new divergences created by the rule |
| `fast_n_flips_unfixed` | 0 | no original flip left open |
| `reaches_identity_1p0` | **False** | PRIMARY falsified |

The 8 new flips dominate the 3 fixes → net identity drops. A zero-cost identity-1.0 fix requires `new = 0` AND `reaches_1p0`; it fails both.

### The mechanism — the ε\* gate straddles across stacks (this is the whole finding)
The 3 original flips are case-(a) **same-pattern**: both stacks see `gap ≤ ε*` *and* a reversed top-2 order, so the lowest-id rule fires on both and picks the same token → closed (`n_flips_same_pattern = 3`, `n_flips_diff_pattern = 0`, `all_flips_closed = True`). But the rule's gate is a hard threshold at ε\*, and bf16 attention perturbs each position's gap by up to one ULP (**0.125 nat at this magnitude**). Wherever a position's gap sits within one ULP of ε\*, the two stacks land on opposite sides of the threshold:

| prompt | pos | M=1 gap | M=8 gap | canon(M1) | canon(M8) | served argmax | straddles ε\* |
|---|---|---|---|---|---|---|---|
| 22 | 226 | **0.125** | 0.25 | 3861 | 45987 | 45987 | ✅ |
| 53 | 225 | 0.25 | **0.125** | 32481 | 21676 | 32481 | ✅ |
| 53 | 226 | **0.125** | 0.25 | 528 | 840 | 840 | ✅ |
| 57 | 226 | **0.125** | 0.25 | 529 | 10293 | 10293 | ✅ |
| 62 | 231 | **0.125** | 0.25 | 236770 | 236778 | 236778 | ✅ |
| 66 | 227 | 0.25 | **0.0** | 12868 | 577 | 12868 | ✅ |
| 116 | 230 | 0.25 | **0.125** | 506 | 496 | 506 | ✅ |
| 121 | 229 | **0.125** | 0.25 | 529 | 15633 | 15633 | ✅ |

**8/8 new flips straddle ε\*** — exactly one stack has `gap ≤ ε*` (rule fires, re-points to a lower id) while the other has `gap > ε*` (confident, keeps its argmax). Baseline these positions *agreed* (`canon_ver_tok == served_m1_tok` before the rule); the rule splits them. No fixed ε\* can avoid this: any threshold creates a straddle band of width ≈1 ULP around it that a ±1-ULP perturbation crosses. "Both sides consistent" fixes the *rule* but not its *input* — the gap the rule tests is itself stack-dependent.

### vs #405 — two-sided halves the damage but can't remove it
| rule | new flips @ ε\* | rule identity | why it fails |
|---|---|---|---|
| **#405** one-sided lowest-id (M8 only) | **14** | 0.9841270 | id-ordering uncorrelated with which FP candidate M1 picks (M1 = lower id only 65%) |
| **#421** two-sided canonical (this card) | **8** | 0.9909297 | gate `gap ≤ ε*` straddles ε\* under ±1-ULP bf16 perturbation (8/8 positions) |

Making the override symmetric removes the 65%-id-coin-flip failure mode (#405's 14 → 8) but exposes a second, irreducible one: the **threshold discontinuity**. Both are RED; the residual is a value-precision phenomenon at the bf16 floor, not anything an argmax post-process can resolve.

### PPL guard — the rule is safe, it just doesn't deliver identity
| field | value |
|---|---|
| `n_rule_fires_near_tie` | 39 |
| `n_changes_argmax` | 17 |
| **`n_changes_confident_argmax_FORBIDDEN`** | **0** |
| `max_logprob_delta_at_changed` | **0.125** (= ε\*, within band) |
| `rule_only_fires_at_near_ties` | True |
| `ppl_unchanged_under_canonical_tiebreak` | **True** |

The rule **never** re-points a confident argmax (0 forbidden changes); every change is within a 0.125-nat band, so PPL is preserved (PPL ≤ 2.42 gate untouched). The PPL half of the hypothesis holds — the identity half does not.

### Irreducible-tie-floor census (m1_self_gap = 0)
| field | heuristic | pinned |
|---|---|---|
| `tie_floor_count` (positions with M=1 self-gap = 0) | **14** (1.587%) | 13 (1.474%) |
| `tie_floor_flip_prompts` | [11, 18, 118] | [90] |
| m8 near-tie @ ε\*=0.125 | 40 (4.54%) | 39 (4.42%) |
| m8 near-tie @ 0.25 | 70 (7.94%) | 61 |
| m8 near-tie @ 0.5 | 106 (12.0%) | 97 |

All 3 fast flips (and the 1 strict flip) are exact bitwise ties on the M=1 side (`m1_self_gap = 0.0`, `m1_is_bitwise_tie = True`) sitting in a 0.125-nat M=8 band — i.e. the flips ARE in the tie-floor, but the tie-floor is a 14-position haystack and the rule's threshold catches a wider, asymmetric 39-position near-tie set.

### Both arms agree — even the batch-invariant pinned stack fails
| arm (bi_env) | baseline identity | flips | canonical: closed / new | canonical identity | reaches 1.0 |
|---|---|---|---|---|---|
| **heuristic** (served/fast, PRIMARY) | 0.9965986 (3) | 3 | 3 / **8** | 0.9909297 | ❌ |
| **pinned** (control, VLLM_BATCH_INVARIANT=1) | 0.9988662 (1) | 1 | 1 / **3** | 0.9965986 | ❌ |

Even with batch-invariant attention engaged (`attn_is_batch_invariant = True`), the pinned stack does not reach 1.0 under canonical tie-break: M=1 vs M=8 verify gaps still straddle ε\* at 3 positions. The failure is a property of the **threshold rule meeting two reduction orders**, independent of which attention kernel runs.

### Interpretation-robustness control (Issues #124/#192)
The win, were it real, would require the operational/self-referential reading (land #414: scorer runs the submission's own greedy as M=1 ref). I also measured the literal/strict reading — `canon(M8)` vs **vanilla M=1 argmax** (the #405 scenario):

| reading | identity | disagreements | reaches 1.0 |
|---|---|---|---|
| self-referential (`canon(M8)` vs `canon(M1)`) | 0.9909297 | 8 | ❌ |
| strict/literal (`canon(M8)` vs vanilla M1 argmax) | **0.9829932** | 15 | ❌ |

Both readings are RED (the strict reading reproduces #405's pinned 0.9829932 / 15 exactly). The negative result is **robust to the unresolved interpretation debate** — no reading of the canonical tie-break reaches identity 1.0.

### Cost model
| field | value |
|---|---|
| `tiebreak_residual_cost_tps` | **0.0** (reads existing in-register top-2, scalar compare + min) |
| `tiebreak_is_zero_cost` | True |
| `equiv_tps_fast_with_tiebreak` | **None** (only set if fast reaches 1.0 — it doesn't) |
| `fastest_realizable_strictly_equivalent_tps` | **467.14** (`blanket_strict`) |
| `upside_over_blanket_tps` | **0.0** |

The rule is correctly modeled as zero-cost (it touches no kernel; it post-processes logits already materialized). But zero-cost only converts to TPS if it yields byte-equivalence, which it does not. So the 14.39-TPS gap between the fast stack (481.53) and the realizable strict floor (467.14) is **not** reclaimable via canonical tie-break.

### Baseline comparison (per PR)
- Official frontier **481.53 TPS / PPL 2.3772 / 128-of-128 (PR #52, `2x9fm2zx`)** — **UNCHANGED**. Local identity micro-measurement only: **0 official TPS, no served-file change, no HF job, no submission.** ✅
- Fastest realizable *strictly-equivalent* = `blanket_strict` **467.14 TPS** (#393-class); the canonical tie-break does not move this. `equiv_tps_fast_with_tiebreak` would have been 481.53 (= +14.39 over blanket) had it reached 1.0 — it did not.
- #381 anchors reproduced byte-for-byte: heuristic 0.9965986 (3 flips @ 11/18/118), pinned 0.9988662 (1 flip @ 90); identities differ from #381's 0.996625/0.998875 only by the 882-vs-889 denominator.

### Determinism controls (signal, not noise)
Both arms: `determinism_M1_vs_M1 = determinism_M8_vs_M8 = within_batch_copy0_vs_copy1 = 1.000000`, `chunk_isolated_fraction = 1.0` (median width 7 = K_spec), `nan_clean = True`. Pinned `attn_is_batch_invariant = True`; heuristic `False` (control separation). ⇒ the 3 flips, the 39 near-ties, and the 8 new straddle-flips are all **deterministic** reduction-geometry effects, not jitter.

### Command
```
cd target/ && /workspace/senpai/target/.venv/bin/python \
    -m research.validity.canonical_tiebreak_fast_stack_identity.canonical_tiebreak_fast_stack_identity \
    --n-prompts 127 --wandb_group canonical-tiebreak-fast-stack-identity \
    --wandb_name stark/canonical-tiebreak-fast-stack-identity
```
Two GPU arms run as isolated subprocesses (CUDA_VISIBLE_DEVICES=0, VLLM_BATCH_INVARIANT∈{0,1}) on the on-target A10G; int4 substrate is the deployed `gemma-4-E4B-it-qat-w4a16-ct` snapshot. Per-position census logs **raw fp32 top-5 (id, logprob) for both M=1 and M=8** so the ε\*-band test is exact (Sterbenz — fp32 subtraction of nearby logprobs is exact; `band_truncation_positions_fast = 0`, no flip's band hit the top-5 logging cap). **127 prompts requested; 126 contributed clean isolated width-7 decode chunks** (one short/non-isolated chunk dropped deterministically in both arms, outside all flip indices → 3+1 flip structure reproduces #397). Peak GPU **12.25 GB** (heuristic) / 12.24 GB (pinned). `--reanalyze --no-wandb` recomputes the report + self-test from saved `arm_*.json` with 0 GPU.

### What happened
The card's optimistic hypothesis was **refuted for a clean, deeper reason than #405.** #405 killed the *one-sided* lowest-id override (it picks the wrong token 35% of the time because id-ordering carries no correctness signal). The natural rescue was to apply the tie-break **symmetrically** — same canonical ≤ε→lowest-id rule on both the M=1 reference and the M=8 verify — so the two stacks can't disagree by construction. That intuition is half-right: it does close all 3 original flips (they're genuine two-sided near-ties with reversed order) and it halves the collateral damage vs #405 (8 new flips vs 14). But it cannot reach 1.0, because the rule's **gate** — `gap ≤ ε*` — is a hard threshold, and the gap it tests is itself perturbed by ≈1 bf16 ULP (0.125 nat) between the M=1 and M=8 reduction orders. At every one of the 8 new-flip positions the two stacks' gaps land on *opposite sides* of ε\*: one fires the rule (re-points to a lower id), the other stays confident (keeps its argmax), and a position that previously agreed now diverges. This is irreducible for any fixed threshold: a discontinuous gate over a quantity with ±1-ULP cross-stack noise will always have a straddle band. The PPL guard confirms the rule is *safe* (0 forbidden confident-argmax changes, max Δ = 0.125 nat) — it just cannot buy identity. The interpretation control closes the last door: both the self-referential reading (0.9909) and the literal vanilla-M1 reading (0.9829) are RED, so the result does not hinge on the #124/#192 debate. **Net: the canonical two-sided tie-break is a real, zero-cost, PPL-safe rule that still does not make the 481.53-TPS fast stack byte-equivalent; no TPS is reclaimable this way, and the cheapest *correct* identity-1.0 path remains a value-level attention fix (#397's selective recompute) requiring a served-path change and human approval.**

### Suggested follow-ups
1. **Close the threshold-tie-break family.** #405 (one-sided) and #421 (two-sided) jointly show no fixed-ε argmax/id post-process reaches identity 1.0: the residual is value-precision at the bf16 floor, and any hard ε gate straddles under ±1-ULP cross-stack noise. Future identity-1.0 levers should act on the **attention reduction value**, not on logit post-processing.
2. **Quantify the straddle band directly (cheap, analysis-only).** The 8 new flips all sit at `min(gap_m1,gap_m8) = ε* = 0.125` with the other side at 0.25 (or 0.0). Sweep ε\* ∈ {0.0625, 0.125, 0.1875, 0.25} and plot `new_flips(ε)` — confirm the count is set by the population density in the [ε−ULP, ε+ULP] straddle band and is minimized but never zeroed. Bounds how badly any threshold rule can do across geometries.
3. **Realize and measure #397's selective recompute (served-path, needs human approval).** Still the only viable identity-1.0 path: fast attention everywhere → free near-tie gate (the margin is readable) → higher-precision reduction on the ≤ε steps only. Measure actual TPS vs the ~2.6-TPS model and the blanket-strict 467.14 floor on the official harness.
4. **Verify-width / prefix sensitivity.** All flips and straddles sit at the 0.125-nat bf16 floor; check whether M=4 or a different C changes the near-tie base rate and thus the 8:3 break:fix ratio. Qualitative conclusion (threshold straddle is irreducible) won't change, but it bounds the realizable selective-fix flagged fraction.

### Repro / bug notes
- Run with the serving venv interpreter (`/workspace/senpai/target/.venv/bin/python`, vLLM 0.22.0 + wandb 0.27.2); default `/usr/bin/python` lacks vllm. GPU phases self-isolate; `--reanalyze --no-wandb` re-derives the report + 33-check self-test from saved JSON with 0 GPU. Smoke (`--smoke --no-wandb`, 4 prompts) validated the path end-to-end before the full run.
- No bugs found in served files; this card touched only the new `research/validity/canonical_tiebreak_fast_stack_identity/` directory (analysis-only, no served-file change).

_Public-evidence note: all anchors cited (frontier 481.53 / PPL 2.3772 #52 `2x9fm2zx`; blanket-strict 467.14 #393-class; #381 pinned 0.998875 / heuristic 0.996625 @ flips 11/18/118/90; #405 one-sided 14 new / 0.9841 / frac_m1_lower 0.65 `j6h228xy`; #412 census machinery `dnjvqbtf`; self-referential scorer gate land #414 `bq7xkfcv`; #397 selective ~2.6 TPS) are reused, not re-derived. This card adds 0 official TPS and changes no served file._
