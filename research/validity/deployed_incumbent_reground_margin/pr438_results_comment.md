STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["lbstimxo"],"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"beats_deployed_481":false,"honest_margin_tps":-14.28,"verdict_robust_to_cb3":false,"primary_metric":{"name":"honest_margin_tps","value":-14.28},"test_metric":{"name":"self_test_passes","value":1}}

## Results — the narrow deploy verdict (LOCAL pod A10G, analysis-only)

**One line the human can act on:** the fastest *strictly-equivalent* config that has actually been **MEASURED** (467.14 official-frame TPS, denken #423, consumed) is **SLOWER** than the freshly re-grounded deployed non-equivalent incumbent (**481.42** official-frame) by **−14.28 TPS (±0.27)**. Only the **MODELLED** +cb3 config (482.74) edges ahead, by **+1.32 TPS** — which needs cb3 to realise **≥91.6%** of its modelled +15.60 and is **NOT robust** to a cb3 haircut. So today's honest answer: **no equivalence-respecting config that has been measured beats the deployed 481.53; the only one that would is contingent on cb3 fully realising.**

`analysis_only=true · no_hf_job=true · no_served_file_change=true · official_tps=0`

### 1. Deployed incumbent re-grounded FRESH on this pod
Served the deployed surface (`submissions/fa2sw_precache_kenyan`, MTP M=8, ONEGRAPH, lm_head-prune-12k, fa_sliding, splitkv_verify, precache) once; 1 warm discard + 5 timed 128→512 decodes.

| metric | local | → official (×τ_lo 1.03524) | banked (PR #52 `2x9fm2zx`) |
|---|---|---|---|
| wall TPS mean | 465.04 | **481.42** | 481.53 |
| 95% t-CI half | ±0.26 | **±0.27** | — |
| CV over 5 reps | 0.045% | — | — |
| drift vs banked | — | **−0.11** | (anchor) |
| PPL | — | **2.3767** | 2.3772 |

Re-grounding **reproduces the banked 481.53 to −0.11 TPS** and PPL to −0.0005 → same config, same speed. PPL passes the ≤2.42 gate. CV 0.045% is an exceptionally tight 5-rep band (per-rep 464.70–465.22). τ_lo is my own validated local→official scalar (lawine #267 `nzqnd154`, stable to 0.135%).

### 2. Equivalence-respecting configs (CONSUMED, official frame — not re-derived here)
- **Best MEASURED strict-equiv:** 467.14 (denken #423 `5a6zq2yz`). `best_equivalent_is_measured=true`.
- **+cb3 (MODELLED):** 467.14 + 15.60 = 482.74 (kanna #403 `iv9i2wks`; stark owns realisation).
- **Naive strict-equiv floor measured locally for free:** the M=1 AR reference (speculation OFF, every other kernel identical) ran 156.20 local → **161.70 official** — i.e. with no speculation the strictly-equivalent path is **−319.7 TPS** below deployed. Speculation is doing all the work.

### 3. Honest verdict
| | best measured equiv (467.14) | +cb3 modelled (482.74) |
|---|---|---|
| `beats_deployed_481` | **false** | true |
| `honest_margin_tps` | **−14.28** | +1.32 |
| `margin_within_noise` | false (\|14.28\| ≫ 0.27) | — |
| cb3 must realise to TIE | — | **≥91.6% of +15.60** |

`verdict_robust_to_cb3 = false` — the win exists only on the modelled leg, and even then by +1.32 inside cb3's own realisation risk (stark already refuted the sibling pinned-K rung 496.74 → −5.82). **(a) cb3 fully realises:** equivalence wins by +1.32. **(b) cb3 haircuts:** equivalence loses (−14.28 at the measured floor). Not robust to (a)∧(b).

### 4. ⚠️ Greedy-identity census did NOT re-confirm 0.9966 / 3 flips — it read 0.4143 / 117 (honest finding)
The PR asked to re-confirm identity ~0.9966 / ~3 flips. The fresh **M=8-vs-own-M=1** census (official byte-exact verifier, no tolerance) instead read **identity 0.4143 (token) / 117 of 128 prompts divergent** at 512 forced tokens. I ran this down before reporting:

- **Not non-determinism:** all 5 M=8 reps are byte-identical (0/128 differ); M=1 confirmed spec-off (`speculative_config=None`); prompts byte-identical (128/128 sha256).
- **Genuine near-tie flips, not garbage:** sequences agree for a median of **114 tokens**, then flip at a near-tie position; 11 reconverge within 8 tokens, the rest cascade (ignore_eos forces 512 tokens). The 11 fully-identical prompts are all short high-confidence mmlu_pro answers.
- **Not a pure output-length artifact:** even truncated to the first 32 tokens, 22 prompts already diverge (identity 0.9336) — never near 0.9966 at any length.
- **Likely root cause:** the M=8 verify step batches 8 query positions/step; local vLLM `v0.22.1rc1.dev307` captures cudagraph sizes [1,2] (max 2), so the size-8 verify runs **eager** while M=1 (size 1) runs in cudagraph → divergent fp reductions. The flip **count** is environment-sensitive (HF a10g build → 3, local AWS A10G → 117); PPL is robust to these near-tie flips (still 2.3767).
- **Impact on the verdict: NONE.** The deployed config is **non-equivalent under both** measurements (3 or 117 flips → outside the #407 feasible set, *more* firmly so locally). The deploy margin is unchanged. **Caveat for the program:** a local greedy-identity census is **not an HF-faithful proxy** for speculative configs — don't trust a locally-measured "N flips" as the HF identity. Full reproducible breakdown in `research/validity/deployed_incumbent_reground_margin/census_discrepancy_analysis.json`.

### Required terminal fields
`deployed_tps_reground=481.42` · `deployed_tps_ci=±0.27` · `deployed_identity_reground=0.4143` (measured local; banked 0.9966 NOT re-confirmed — see §4) · `deployed_n_flips=117` (divergent prompts; banked 3) · `best_equivalent_tps=467.14` · `best_equivalent_is_measured=true` · `beats_deployed_481=false` · `honest_margin_tps=-14.28` · `margin_within_noise=false` · `verdict_robust_to_cb3=false` · `ppl=2.3767` (deployed, measured) · `self_test_passes=true` (7/7) · `analysis_only=true` · `no_hf_job=true` · `no_served_file_change=true` · `official_tps=0`.

### Reproduce / cost
```bash
cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  research/validity/deployed_incumbent_reground_margin/deployed_incumbent_reground_margin.py \
  --measure --reps 5 --wandb_group deployed-reground-margin \
  --wandb_name lawine/deployed-incumbent-reground-margin
# 0-GPU composition self-test gate: append --self-test --no-wandb  (no model load)
```
- **Peak VRAM:** 18.94 GB (single A10G). Process RSS 78.5 MiB.
- **W&B:** `lbstimxo` (wandb-applied-ai-team/gemma-challenge-senpai), group `deployed-reground-margin`.
- **PRIMARY** `self_test_passes=1` (a τ_lo round-trip, b measured-margin-is-deficit, c cb3-upside-not-robust, d cb3-required-threshold, e robustness-symmetry, f constants-exact, g nan-clean). **TEST** `honest_margin_tps=-14.28`.

### What happened
The fresh same-pod re-grounding cleanly confirms the incumbent's speed (481.42 vs 481.53, drift −0.11) and quality (PPL 2.3767). Against that anchor, the **best equivalence-respecting config that actually exists as a measurement (467.14) is 14.28 TPS slower** — the equivalence frontier only overtakes the deployed incumbent on the *modelled* +cb3 leg (482.74, +1.32), and only if cb3 realises ≥91.6% of its modelled delta. Given stark already refuted the sibling pinned-K rung (496.74 → −5.82 realised), banking on cb3 fully realising is exactly the contingency the human needs flagged. **Net: the fastest strictly-equivalent config is, on today's evidence, slower than the deployed non-equivalent 481.53; the equivalence "win" is modelled, +1.32, and not robust.** Surprise of the run: the deployed config's *local* greedy-identity is far lower than its banked HF identity (0.4143 vs 0.9966) — a real environment sensitivity in the M=8 verify path, not a bug, and one that doesn't move the verdict but does caution against trusting local identity censuses for spec configs.

### Suggested follow-ups
- If the verdict's +1.32 cb3 leg is to be load-bearing, gate it on **stark's realised-cb3 number** (budget-exact) rather than the modelled +15.60 — a sub-92% realisation flips it to a tie-or-loss.
- A true same-harness A/B would re-measure denken's 467.14 config **locally** on this pod and compare in local frame (removing the τ_lo bridge); worth doing if/when that config is inside this launch's isolation boundary.
- The 3→117 local-vs-HF identity gap deserves its own probe: raising local cudagraph capture to cover the size-8 verify (so M=8-verify and M=1 share the cudagraph path) would test whether the flips collapse back toward the banked 3 — relevant to how the whole program validates equivalence locally.

### Public evidence used
Consumed (as given in the PR body, official frame): deployed incumbent PR #52 `2x9fm2zx` (481.53 / 2.3772 / identity 0.9966); equivalence floor denken #423 `5a6zq2yz` (467.14, measured); +cb3 model kanna #403 `iv9i2wks` (+15.60); pinned-K refutation stark #433 `0pg4bz25` (496.74 → −5.82). This leg **re-grounds** the incumbent and **composes** the narrow deploy head-to-head; it does not re-derive cb3 or the demand axis (fern #357 / kanna #416 / land #436 own those).
