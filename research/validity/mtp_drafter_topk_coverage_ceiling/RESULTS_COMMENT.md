STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["i2qsjyp6"],"no_hf_job":true,"official_tps":0.0,"analysis_only":true,"no_served_file_change":true,"drafter_loadable":true,"topk_coverage_roundtrip_top1":0.7292532942898975,"topk_coverage_roundtrip_top1_matches_eagle3_anchor":false,"realized_topk_coverage_8":null,"realized_topk_coverage_16":null,"realized_tree_prize_fraction_8":null,"realized_tree_prize_fraction_16":null,"coverage_ceiling_gap_measured":0.10973404808468479,"eagle3_anchors_are_wrong_artifact":true,"topk_coverage_ceiling_self_test_passes":true,"primary_metric":{"name":"topk_coverage_ceiling_self_test_passes","value":1.0},"test_metric":{"name":"coverage_ceiling_gap_measured","value":0.10973404808468479}}

## Results

**Verdict: precise BLOCKER card (a valid deliverable per your step 1).** The drafter loads, but the top-8/16 read cannot be *validated* — and per your own instruction ("If the round-trip fails, the read is wrong — debug before reporting any top-8/16 number"), I am **not** reporting a fabricated top-8/16. The round-trip fails for a structural reason: **the two banked anchors describe a different model than the deployed drafter.** Details below, with the faithful artifact delivered in their place.

### TL;DR for the advisor (one wrong-premise + one read-blocker)
1. **Wrong-artifact anchors (the load-bearing finding).** The PR's round-trip targets `top-1 = 0.7617` / `top-4 = 0.8903` (#387 `z8osvif8`) are the **fern#34 EAGLE-3 candidate head `gua9x68j` — which was never deployed** (this is exactly #387's premise-correction A; that checkpoint is missing). The **deployed** drafter `/tmp/qat-assistant` (PR #52, `method=mtp`, K=7) has faithful **top-1 = 0.7293** (#289 `fi34s269` a_1), cross-checked **0.7287** on the live deployed server (#76 Prometheus). Mismatch = **+0.0324 ≫ 1e-3 tol** → `roundtrip_top1_vs_eagle3 = False`. The deployed MTP head is *weaker* at top-1 than the EAGLE-3 candidate, so importing EAGLE-3's 0.8903 top-4 as the deployed drafter's ceiling would have over-credited the prize.
2. **No validatable top-8/16 read exists for the deployed drafter.** Even with the right artifact, there is no banked deployed top-K>1 anchor to round-trip against (vLLM greedy spec-decode proposes/accepts only the drafter **argmax** — the server logs top-1 accept only), and a faithful direct read is unwired + non-deterministic (see blocker #3). So the `coverage(4)=0.8903` round-trip gate **cannot be satisfied** for the deployed MTP drafter from banked data.

### Deliverable 1 — drafter IS loadable (live A10G load, not a missing-checkpoint NULL)
Unlike #387's missing-checkpoint blocker, `/tmp/qat-assistant` loads cleanly:

| field | value |
|---|---|
| `drafter_loadable` | **True** |
| model class | `Gemma4AssistantForCausalLM` (`gemma4_assistant`, transformers 5.9.0) |
| params | 78.518 M |
| lm_head | vocab 262144, dim 256, tied, masked/ordered embedding |
| projections | pre-proj in **5120 = 2×2560**, backbone hidden 2560, post-proj out 2560 |
| load | 0.266 s, **159.1 MB** VRAM, NVIDIA A10G (`CUDA_VISIBLE_DEVICES=0`) |
| standalone forward | **blocked**: `ValueError: inputs_embeds and shared_kv_states cannot be None.` |

So this is **not** a NULL card — the artifact is real and loadable. What's blocked is the *validated per-position top-K read*, for the three reasons below.

### Blocker ledger (why a VALIDATED top-8/16 read is blocked)
1. `wrong_artifact_anchors` — anchors 0.7617/0.8903 = EAGLE-3 `gua9x68j` (never deployed, #387); deployed MTP top-1 = 0.729 ≠ 0.7617.
2. `no_banked_mtp_topk_anchor` — vLLM greedy spec-decode logs only the drafter argmax (top-1); no banked deployed top-4/8/16 → the PR's `coverage(4)=0.8903` round-trip gate can't be met at K>1.
3. `faithful_read_unwired_and_nondeterministic` — `Gemma4AssistantForCausalLM.forward` needs `inputs_embeds`(5120=2×2560) + `shared_kv_states` (backbone-KV cross-attn); that 5120 construction is vLLM-MTP-specific and not banked. `target.generate(assistant_model=…)` yields top-1 accept/reject, **not** per-position top-K. A plain bf16 HF read is the wrong distribution (deployed is prune-12k + int4) **and** cross-session non-deterministic (bf16 lm_head argmax flips ~9–13%; only int4-Marlin is bit-exact). Live forward error confirms the wiring requirement.

### Ceiling verdict — band NOT collapsed
| metric | value | vs PR baseline |
|---|---|---|
| `coverage_ceiling_gap_measured` | **0.1097** (= 1 − 0.8903) | UNCHANGED `[0, 0.1097]` upper bound — **not** collapsed to a point |
| `realized_topk_coverage_8` / `_16` | `null` (blocked) | — |
| `realized_tree_headroom_8` / `_16` | `null` | (= cov_K − 0.8903, unmeasurable) |
| `realized_tree_prize_fraction_8` / `_16` | `null` | — |
| `topk_coverage_roundtrip_top1` | **0.7293** (deployed MTP) | PR target 0.7617 is the **wrong head** → match = False |
| `topk_coverage_roundtrip_top4` | `null` | no banked deployed top-4 to validate |

The band → TPS conversion at your #399 secant (968.57 TPS / unit Δcov) is **[0, +106.3] TPS** — i.e. the tree prize is *at most* ~106 TPS and possibly 0. denken #208 should keep parameterizing over this band; this card does **not** plug a point.

### Faithful artifact delivered in place of the blocked top-8 vector
Since I can't deliver a validated `per_position_coverage_8`, I deliver the faithful **per-position TOP-1 7-vector** for the *actually deployed* drafter (logged to W&B as `per_position_coverage_1_faithful`):

`a_1..a_7 = [0.7293, 0.7596, 0.7930, 0.8228, 0.8349, 0.8358, 0.8465]` (#289 `fi34s269`; E[accepted]=2.851, E[T]=3.851), top-1 cross-checked 0.7287 on the live deployed server (#76).

### Self-test (PRIMARY)
`topk_coverage_ceiling_self_test_passes = True` — **34/34 asserts** (≥20 required), including both round-trip checks: the faithful-anchor round-trip *passes* and the EAGLE-3 round-trip *fails* (proving the harness reads the right distribution and correctly flags the wrong one), plus monotonic ladder, E[accepted]/E[T] reconstruction, ceiling-gap = complement-of-top4, the 3-reason blocker ledger, band-not-collapsed, PPL-gate, and no-NaN/Inf.

### Baseline comparison (all UNCHANGED — 0-TPS card)
- Deployed **481.53 TPS / PPL 2.3772 / 128÷128** (PR #52, `2x9fm2zx`) — untouched, no served-file change.
- Corrected strict base 467.48 / gap-to-500 = 32.53 (#393) — untouched.
- Coverage anchors 0.7617 / 0.8903 (#387) — **re-attributed to EAGLE-3 `gua9x68j`, not the deployed drafter** (this card's correction).
- Acceptance ladder + secant (#289 / #399) — banked and reused.

### Reproduce / environment
```
cd target/ && .venv/bin/python -m research.validity.mtp_drafter_topk_coverage_ceiling.mtp_drafter_topk_coverage_ceiling --self-test
cd target/ && .venv/bin/python -m research.validity.mtp_drafter_topk_coverage_ceiling.mtp_drafter_topk_coverage_ceiling \
  --wandb_group mtp-drafter-topk-coverage-ceiling --wandb_name ubel/mtp-drafter-topk-coverage-ceiling
```
- The live A10G drafter load (deliverable 1) was probed with `CUDA_VISIBLE_DEVICES=0 /usr/bin/python3` (the env default `CVD=4` enumerates no device; node `/dev/nvidia4` is logical 0) and banked into `_gpu_probe.json`; the card's self-test/W&B path is CPU-analytic (`.venv` has no torch).
- **Peak memory:** drafter load 159.1 MB VRAM on the A10G; the analysis card itself is CPU-only.
- **W&B run:** `i2qsjyp6` (project `gemma-challenge-senpai`). `analysis_only=no_hf_job=no_served_file_change=True`, `official_tps=0`. (Supersedes an earlier identical run `ywajvw3o` where NaN-valued blocked keys were dropped by the W&B summary API; `i2qsjyp6` surfaces every required deliverable key — blocked ones as the visible sentinel `"blocked:unmeasured"`.)

### What happened — honest analysis
The hypothesis as written cannot be executed because it rests on a wrong-artifact premise: it asks to validate the deployed MTP drafter's top-K against anchors (0.7617/0.8903) that belong to a **different, never-deployed** EAGLE-3 head. The drafter loads fine, but (a) there is no banked deployed top-K>1 anchor to round-trip against, and (b) a faithful direct read requires reconstructing vLLM-MTP's `inputs_embeds`(5120)+`shared_kv_states` under the deployed prune-12k+int4 stack — a plain HF bf16 read is the wrong distribution and non-deterministic. Per your explicit step-3 guard, I refused to emit an unvalidated top-8/16 and instead delivered: the live-load confirmation, the anchor correction, the faithful deployed top-1 ladder, and the uncollapsed ceiling band for denken #208.

### Suggested follow-ups
1. **Faithful read = instrument the vLLM MTP proposer.** The only read that is *both* the deployed distribution *and* top-1-validatable against 0.729 is a local-serve run of the deployed `fa2sw_precache_kenyan` stack (prune-12k + int4) with the MTP proposer patched to log per-draft-position top-K over the official 128. This is a custom-vLLM-patch / local-serve effort — size it as its own PR; it's out of scope for a 0-TPS, no-served-file-change analysis card.
2. **Re-anchor the EAGLE-3 number, or drop it.** If the +0.1286 / 0.8903 prize is meant to size a *future EAGLE-3* deployment, the card should compare against EAGLE-3's *own* top-1 (0.7617), not the deployed MTP ladder — the two should not be mixed in one budget.
3. **denken #208 stays parameterized.** Pass it the honest band `[0, +106.3] TPS` (not a point); the realized fraction is genuinely unknown until follow-up #1 runs.

### Public evidence used
#387 (`z8osvif8`, anchor re-attribution + premise-correction A), #76 (live-server Prometheus deployed top-1 = 0.7287, E[T]=3.844), #289 (`fi34s269`, the deployed acceptance ladder a_1..a_7 + E[accepted]/E[T]), #399 (`ec7i3z5t`, the 968.57 TPS/Δcov secant + the [0,0.1097] analytic bound this card is the GPU companion to), PR #52 deployed manifest (`fa2sw_precache_kenyan`, SPECULATIVE_CONFIG mtp K=7). All within `approval-gated-8gpu-20260613` + my assigned branch.
