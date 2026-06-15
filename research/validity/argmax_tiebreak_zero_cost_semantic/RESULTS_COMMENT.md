STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["j6h228xy"],"primary_metric":{"name":"tiebreak_fix_tps_cost","value":2.5984251968503935},"test_metric":{"name":"argmax_tiebreak_zero_cost_self_test_passes","value":1}}

## Results

**Verdict: RED — a deterministic lowest-id tie-break is NOT free; it is net-negative. The #397 selective higher-precision attention fix (~2.6 TPS) stands as the cheapest *correct* decode-width identity-recovery path.** Required headline fields: `argmax_tiebreak_zero_cost_self_test_passes = True` (PRIMARY, **30/30 checks**, ≥20 ✓); `served_argmax_tiebreak_rule = "lowest_index_first"`; `m1_reference_uses_lowest_index = False`; `tie_identifiable_from_fast_path = True`; `free_tiebreak_recovers_identity_1p0 = False`; **`free_tiebreak_new_flips = 14`** (MUST be 0 to be free); `lowest_index_tiebreak_is_free = False`; `decode_width_identity_is_free = False`; `tiebreak_fix_tps_cost = 2.598`; `reconciles_397_selective_cost = True`. W&B run **`j6h228xy`** (group `argmax-tiebreak-zero-cost-semantic`). Scope: `analysis_only=True` / `no_hf_job=True` / `no_served_file_change=True` / `official_tps=0`.

**The whole result in one sentence:** the "M=1 reference = lower token-id of the tied pair" pattern that held for all 4/4 flips in #397 was a small-sample coincidence — across the full population of 40 near-tie verify positions the M=1 token is the lower id only **65%** of the time, so a *global* lowest-id override fixes the 3 real flips but **breaks 14 currently-correct near-ties** (where the served top-1 = M=1 reference is the *higher* id), dropping served identity from **0.9966 → 0.9841**; the margin is readable for free, but the *resolution* is not, so #397's 2.6-TPS selective recompute is the floor.

### Primary — global lowest-id rule simulation (heuristic = served/fast path, PRIMARY, 126×8-row decode = 882 positions)
The rule: resolve every ≤ε near-tie to the lower token-id (`pick = min(top1,top2) if gap≤ε else top1`), applied to the served argmax logits already in-register (zero recompute). A free fix requires `new_flips = 0` AND `recovers_identity_1p0`. It fails both, monotonically:

| ε (nat) | near-ties | correct near-ties | flips fixed | **new flips** | unfixed | rule identity | recovers 1.0 |
|---|---|---|---|---|---|---|---|
| **0.125** (eps*) | 40 | 37 | 3 | **14** | 0 | **0.9841270** | ❌ |
| 0.25 | 70 | 67 | 3 | **27** | 0 | 0.9693878 | ❌ |
| 0.5 | 106 | 103 | 3 | **45** | 0 | 0.9489796 | ❌ |

At every band the override breaks far more than it fixes (14:3 at the tightest bf16 floor), and the harm grows monotonically with ε. Baseline served identity 0.9965986 → 0.9841270 — **the "free" rule makes identity strictly worse.** `argmax_tiebreak_zero_cost_self_test_passes = True (30/30)`, including `new_flip_count_matches_higher_id_correct_near_ties` (an independent manual recount of the 14 broken positions matches the rule sim) and `reproduces_flip_structure_381_397`.

### The mechanism — M=1 is NOT always the lower id (this is the whole finding)
At ε=0.125, of the **40** near-tie positions where the M=1 reference token sits in the M=8 top-2:

| quantity | value | reading |
|---|---|---|
| `n_m1_is_lower` | **26 (65%)** | M=1 token is the lower id of the tied pair |
| `n_m1_is_higher` | **14 (35%)** | M=1 token is the **higher** id — a lowest-id override picks the WRONG token here |
| `m1_reference_uses_lowest_index` | **False** | the #397 4/4 "M1=lower-id" pattern does NOT generalize |
| `frac_m1_lower` | 0.65 | the id-ordering carries no reliable signal about which token is correct |

The 3 real flips happen to be cases where M=1 is the lower id (so a lowest-id rule fixes them); but they are drawn from a 40-position near-tie haystack in which M=1 is the higher id 14 times. Forcing lowest-id globally therefore trades 3 fixes for 14 breaks. **Token-id ordering is uncorrelated with FP-reduction-order correctness** — exactly the null prior, now measured.

### There is no bitwise tie to "break" (argmax probe)
| probe field | value | meaning |
|---|---|---|
| `served_argmax_tiebreak_rule` | `lowest_index_first` | torch.argmax already returns the lowest index on a **true** bitwise tie (CPU + CUDA, stable across repeats) |
| `cpu_returns_lowest_index` / `cuda_returns_lowest_index` | True / True | the "lowest-index tie-break" the card asked about is *already* the default for genuine ties |
| `bf16_strict_picks_larger` | **True** | when two logits differ by ≥1 bf16 ULP, argmax picks the strictly-larger one (no tie) |
| `bf16_one_ulp_gap_nats` | **0.0078125** | one bf16 ULP at this magnitude |

The flips sit at a **0.125-nat** gap = **16 bf16 ULPs** apart — these are genuine strict-greater decisions from M=8 reduction-order divergence, **not** bitwise ties. So the "zero-cost tie-break" is a category error: (1) for *true* ties argmax already picks lowest-index, but there are none at the flips; (2) catching the flips requires overriding genuine strict (non-tie) decisions inside a ≤ε *window*, and that override is wrong 35% of the time. You cannot decide the correct token from id-ordering — you need the higher-precision value (= #397's selective recompute).

### `tie_identifiable_from_fast_path = True` — the margin is free, the resolution is not
All **3/3** served-arm flips are identifiable from the fast path's own in-register top-2 (the disagreement *is* a readable ≤0.125-nat near-tie; `n_fast_flips=3`, `all_fast_flips_identifiable=True`). So a *selective gate* (flag near-ties) is free — this is precisely what makes #397's selective recompute cheap (2.6 TPS = flag 23.6% of steps, recompute only those at higher precision). But the **gate** being free does not make the **fix** free: resolving the flagged position correctly needs the precise attention reduction, because the lowest-id shortcut mis-resolves 14/37 correct near-ties.

### Both arms agree (control = pinned, VLLM_BATCH_INVARIANT=1)
| arm (bi_env) | identity | flips | rule@0.125 fixed/new/unfixed | rule identity | attn_bi |
|---|---|---|---|---|---|
| **heuristic** (served/fast, PRIMARY) | **0.9965986** (3/882) | 3 | 3 / **14** / 0 | 0.9841270 | False |
| **pinned** (control) | **0.9988662** (1/882) | 1 | 1 / **15** / 0 | 0.9829932 | True |

Both reproduce #381/#397 byte-for-byte at the flip level (heuristic 3 flips @ prompts 11/18/118; pinned 1 flip @ prompt 90; identities differ from #381's 0.998875/0.996625 only by the 882-vs-889 denominator). The pinned arm is even more lopsided (15 breaks for 1 fix), confirming the global rule is uniformly net-negative regardless of the attention reduction.

### Reconciliation with #397 and #364
- **`reconciles_397_selective_cost = True`** (`reproduces_flip_structure ∧ cost_logic_consistent`). #397 priced the *realizable, argmax-semantics-preserving* fix at 2.6 TPS (selective higher-precision attention on the 23.6% near-tie steps). #405 asked whether a free logit-level lowest-id rule could undercut that — **it cannot**: the free rule is not merely incomplete, it is *harmful* (−12.5% identity at eps*). So `tiebreak_fix_tps_cost` correctly stays at **2.598**.
- **#397 follow-up #2 resolved (the load-bearing caveat):** "does M1 = lower-id hold beyond n=4, or does a stable-lowest-id rule mis-resolve?" → it mis-resolves 14 times at eps*. The n=4 was coincidental.
- **#397 follow-up #3 resolved (this card's premise):** "logit-level tie-break as a near-free fix, separate scope" → **refuted.** A backend-free argmax override does not hold identity; the only correct cheap path remains the value-level selective recompute.
- **#364 precision-wall intuition vindicated at the resolution layer.** ubel #364 found you cannot cheaply *flag* the residual (low precision). #405 shows that even when you *can* flag it for free (the margin is readable), you still cannot cheaply *resolve* it — id-ordering is uncorrelated with correctness. The wall moved from detection to resolution, but it is still there.

### Determinism controls (signal, not noise)
Both arms: `determinism_M1_vs_M1 = determinism_M8_vs_M8 = within_batch_copy0_vs_copy1 = 1.000000`, `chunk_isolated_fraction = 1.0` (median width 7 = K_spec). Pinned: `attn_is_batch_invariant = True` (pin engaged); heuristic: `False` (control separation). ⇒ the 3 flips and the 40 near-ties are **deterministic** reduction-geometry effects, and the 14 new flips are a deterministic property of the rule, not jitter.

### Baseline comparison (per PR)
- Official frontier **481.53 TPS / PPL 2.3772 / 128-of-128 (PR #52, `2x9fm2zx`)** — **UNCHANGED**. This is a local identity micro-measurement: **0 official TPS, no served-file change, no HF job, no submission.** ✅
- Corrected realized strict base **471.42** (#390 `5y64zbjz`); gap_to_500 = 28.58; band ceiling 509.78. The free-tie-break route would have removed the 2.6-TPS (and 11-TPS blanket) identity tax at no cost; this card shows that route is closed, so **no TPS is reclaimable this way** — the 2.6-TPS selective recompute (still requiring a served-path change + human approval) remains the only identity-1.0 lever.
- #381 anchors reproduced: pinned 0.998875 (1 flip), heuristic 0.996625 (3 flips).

### Command
```
cd target/ && /workspace/senpai/target/.venv/bin/python \
    research/validity/argmax_tiebreak_zero_cost_semantic/argmax_tiebreak_zero_cost_semantic.py \
    --n-prompts 127 --wandb_group argmax-tiebreak-zero-cost-semantic \
    --wandb_name stark/argmax-tiebreak-zero-cost-semantic
```
Two GPU arms run as isolated subprocesses (CUDA_VISIBLE_DEVICES=0, VLLM_BATCH_INVARIANT∈{0,1}) on the on-target A10G; int4 substrate is the deployed `gemma-4-E4B-it-qat-w4a16-ct` snapshot. **127 prompts requested; 126 contributed clean isolated width-7 decode chunks** (prompt idx 105 dropped a short/non-isolated chunk deterministically in both arms — outside all flip-bearing indices, so the 3+1 flip structure reproduces #397 exactly). Peak GPU **12.25 GB** (heuristic) / 12.24 GB (pinned). `--reanalyze` recomposes the report + self-test from saved `arm_*.json` with 0 GPU.

### What happened
The card's optimistic hypothesis was **refuted, decisively and for a clean physical reason.** #397 observed that all 4 decode-width flips had the M=1 reference as the *lower* token-id of the 0.125-nat tied pair, raising the tantalizing possibility that a free, deterministic lowest-id argmax rule could recover identity 1.0 with no recompute — undercutting even the 2.6-TPS selective fix. The full per-position census kills it: that 4/4 pattern was small-sample luck. Across the 40 near-tie positions (37 of which the server already gets right), the M=1 token is the lower id only 65% of the time, so a global lowest-id override fixes 3 flips and breaks 14 — net identity 0.9966 → 0.9841 (pinned: 1 fixed, 15 broken). The argmax probe explains why at the bit level: there is no bitwise tie to break (argmax already picks lowest-index on true ties, of which there are none); the flips are genuine 16-ULP strict-greater decisions from reduction-order divergence, and token-id ordering carries no information about which of the two FP-legitimate candidates the M=1 reduction would have picked. The one genuinely positive sub-finding — `tie_identifiable_from_fast_path = True` — is exactly what keeps #397 alive: the *gate* is free (the disagreement is a readable ≤0.125-nat margin), so selective recompute can be cheap; but the *resolution* is not free, because only the higher-precision attention value disambiguates. Net: **the cheapest correct decode-width identity-1.0 fix remains #397's ~2.6-TPS position-selective higher-precision attention, and it requires a served-path change (human-approval-gated). No free lunch at the logit layer.**

### Suggested follow-ups
1. **Stop pursuing logit-/id-level free fixes for the decode-width residual.** This card closes that family: the residual is a value-precision phenomenon, not a tie-ordering one. Any identity-1.0 lever must act on the attention reduction value, not on the argmax post-process.
2. **Realize and measure #397's selective recompute (served-path, needs human approval).** This remains the only viable identity-1.0 path: fast attention everywhere → free margin gate (now confirmed readable, `tie_identifiable_from_fast_path=True`) → higher-precision reduction on the ≤ε steps only. Measure *actual* TPS vs the 2.6-TPS model and the 11-TPS blanket on the official harness. Coordinate with the attention-strict-pin-cost line that audits whole-backend η_attn.
3. **Quantify the per-token-id null directly (cheap, analysis-only).** Across the 40 near-ties, regress "M1 == lower-id" on candidate-id gap / logit magnitude to confirm 65% is consistent with a coin-flip (no exploitable structure). If some sub-class (e.g., adjacent BPE merges) *were* predictable, a *partial* id-rule on that sub-class might shave the selective fix's flagged fraction — low-value but bounded.
4. **Sensitivity to verify width / prefix length.** All flips sit at the 0.125-nat bf16 floor; check whether a different M (e.g., M=4) or C changes the near-tie base rate and thus the 14:3 break:fix ratio. Does not change the qualitative conclusion (id-ordering is uncorrelated), but bounds the selective fix's flagged fraction across geometries.

### Repro / bug notes
- Run with the serving venv interpreter (`/workspace/senpai/target/.venv/bin/python`, vLLM 0.22.0 + wandb 0.27.2); the default `/usr/bin/python` lacks vllm. GPU phases self-isolate; `--reanalyze` re-logs from saved JSON with 0 GPU. Smoke (`--smoke --no-wandb`, 4 prompts) validated the path end-to-end before the full run.
- No bugs found in served files; this card touched only the new `research/validity/argmax_tiebreak_zero_cost_semantic/` directory (analysis-only, no served-file change).

_Public-evidence note: all anchors cited (frontier 481.53 / PPL 2.3772 #52 `2x9fm2zx`; strict base 471.42 #390 `5y64zbjz`; band ceiling 509.78; #381 pinned 0.998875 / heuristic 0.996625; #397 selective 2.598 TPS / f_step 0.23622 / FA_SLIDING=0 η_attn ≈ 11 TPS / 4-of-4 M1-lower-id; #364 precision wall) are reused, not re-derived. This card adds 0 official TPS and changes no served file._
