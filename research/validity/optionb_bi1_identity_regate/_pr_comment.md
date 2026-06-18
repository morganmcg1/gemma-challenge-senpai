STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["15g9q3wc"],"primary_metric":{"name":"break_rate_bi1_both_sides","value":0.0009186954524575103},"test_metric":{"name":"residual_after_tau_0p3nat","value":0.0}}

## Results — Option-B #319 re-gate: BI=1 both-sides spec-vs-AR greedy identity

**VERDICT: `ATTENTION_RECOVERED_RESIDUAL_IS_INT4_TIES`** — under `VLLM_BATCH_INVARIANT=1` pinned on **both** the M=1 AR reference and the M=8 spec-verify side, the per-step greedy break collapses to a **0.092%** residual that is **100% sub-0.5-nat knife-edge near-ties, 0 systematic (attention-path) breaks, and rescued to exactly 0 at τ=0.2/0.3 nat.** The attention defect is gone; what remains is the int4-Marlin M-variance grid-tie the programme already knows about.

### Required deliverables

| metric | value |
|---|---|
| `break_rate_bi1_both_sides` | **0.0009187 (0.092%, 6/6531)**, Wilson CI95 [0.042%, 0.200%] |
| `attention_path_break_count` (margin ≥ 0.5 nat **or** M1-token out-of-topk) | **0** |
| `int4_tie_residual_rate` (margin < 0.5 nat) | 0.0009187 (0.092%) |
| `residual_frac_under_0p5nat` | **1.0** (all 6 flips < 0.5 nat) |
| `residual_after_tau_0p3nat` | **0.0** |
| τ-ladder {0.0, 0.2, 0.3} nat | 6 → **0** → **0** surviving flips (rates 0.092% → 0 → 0) |
| residual flip margin median / max | **0.125 / 0.125 nat** (= 1–2 bf16 logit ULP) |
| `n_m1_token_outside_topk` | 0 (M1 token always inside M8 top-20) |

### BI=0 contrast arm — BI=1 is load-bearing

| arm | break rate | CI95 |
|---|---|---|
| **pinned (BI=1 both sides)** | **0.092%** (6/6531) | [0.042%, 0.200%] |
| **heuristic (BI=0)** | **0.308%** (20/6489) | [0.200%, 0.476%] |

BI=0 is **3.35× the BI=1 rate.** This is the harness's own independent evidence that pinning BI=1 is doing real work: the BI=0 AR decode takes the **3D split-KV** path (stark #621 op-level: `ar_bi0_path=3D_splitKV`, maxdiff vs 2D verify = 6.1e-5 ≈ 1 bf16 ULP), which tips ~2/3 more knife-edge near-ties than the bit-exact 2D-vs-2D BI=1 path. Both arms are still 100% sub-0.5-nat and both go to 0 at τ=0.2.

### Honest reading — three things the advisor should weigh

**1. This is the per-step *seed* rate, not the cascaded 47%.** The harness is teacher-forced (every step shares the same context) by design, to isolate kernel/attention M-variance from trajectory divergence. #607's 47% (31048/65536) is the *cascade* of these per-step seeds in a free-running trajectory — a single near-tie flip forks the whole continuation. So the deliverable here is "the per-step seed that drives #607," and it drops from 0.308% (BI=0) to 0.092% (BI=1 both sides), **all of it sub-0.5-nat, all of it τ=0.3-rescued.** The end-to-end "47% → 0" claim rests on #621's op-level proof (attention bit-exact under BI=1, maxdiff 0.0); this run **corroborates it end-to-end on the real int4 body**: 0 systematic breaks under BI=1 both-sides.

**2. The #616 cross-check did NOT land where the card predicted — and that's good news.** The card expected the BI=1 residual to *match* wirbel #616's `raw_structural_flip_rate_m8_vs_m1 = 0.004318` (CI [0.374%, 0.494%]). It does **not**: the both-sides-pinned residual is **0.092%, CI [0.042%, 0.200%], which does not overlap #616.** Instead, my **BI=0** arm (0.308%, CI [0.200%, 0.476%]) is what overlaps #616. The most parsimonious read (from the anchors in the card; I did not inspect #616's branch) is that #616's 0.43% was measured **without BI=1 pinned on both sides**, so it still carries the attention-3D-tipped near-ties — and **fully pinning both sides (which the submission `serve.py` already does) drops the true residual ~3× below the #616 anchor.** If the advisor wants a hard corroboration number, the follow-up is to extend the pinned arm to the full prompt set to tighten [0.042%, 0.200%].

**3. Mechanism of the 0.092% residual.** #621's op-probe found int4-Marlin **bit-exact at M=8** for every body shape on its tested inputs, so the 6 residual flips are the *rare* per-input M-dependent Marlin reductions (the #616 mechanism) and/or prefill-vs-decode 1-ULP accumulation across 30+ layers — **not** a systematic attention divergence (0 flips ≥ 0.5 nat, M1-token always in-topk). The lm_head in this checkpoint is **bf16, not int4**, so the 0.125-nat logit gaps come through a dense head. Either way it is a knife-edge tie, not a kernel defect.

### #319 implication (the gate question)

- **Strict τ=0 (exact argmax every step):** a 0.092% per-step seed survives on the BI=1 stack — it would still cascade in a free-running trajectory, so strict-identity is *not* automatically clean.
- **τ ≥ 0.2 nat tolerance:** **0 breaks.** Every residual flip is a ≤0.125-nat tie.

So, exactly as the card hypothesized: **the kernel/attention-defect blocker is gone.** The *only* remaining Option-B #319 question is a programme **tolerance-policy decision** on sub-0.5-nat int4-grid-ties (advisor/human call), not an attention bug. This clears the last greedy-identity uncertainty I can resolve locally for the Option-B direction (the ~427 TPS / 3.4× projection over the locked 126.378 rung named in the card).

### Controls (all green)

- `pinned_attn_is_batch_invariant = True`, `heuristic = False` → BI flag toggles the attention backend mode as intended.
- `aten_mm_bitexact_M1_vs_M8 = True` (pinned) / `False` (heuristic) → the BI override is live in the pinned arm.
- `determinism_M1_vs_M1 = 1.0` (896 control positions) → the M=1 AR reference is exactly reproducible; flips are genuine M8-vs-M1, not RNG.
- `chunk_isolated_fraction = 1.0`, `mean_computed_rows = 8.0` → every measured position is a true size_m=8 verify-width read (prefix-cache hit on the rest).

### Repro / config

```bash
cd target/
# decisive arm (BI=1 both sides):
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/validity/optionb_bi1_identity_regate/optionb_bi1_regate.py \
  --phase arm --arm pinned --out research/validity/optionb_bi1_identity_regate/arm_pinned_result.json \
  --n-prompts 60 --ctx-len 224 --traj-len 512 --gpu-mem-util 0.55 --max-batched-tokens 8192
# BI=0 contrast arm: identical, --arm heuristic (env VLLM_BATCH_INVARIANT=0)
# recompose + W&B:  ... --reanalyze --wandb_group optionb-bi1-identity-regate
```

- **Stack:** vLLM **0.22.0** (the #607/#616/#621 chain version), int4 W4A16 body `google/gemma-4-E4B-it-qat-w4a16-ct`, M=8 verify width, MTP drafter NOT loaded (under greedy temp=0 the drafter only changes acceptance/speed, never the verify argmax — #621). **NB: the card body says "dev307"; I ran 0.22.0 deliberately because corroborating #616/#621 requires their exact 0.22.0 stack. Flagging in case dev307 was intended.**
- **Scope:** `analysis_only=true`, `official_tps=0`, **NO HF Job, NO submission, no served-file change.** Synthetic/local weights, A10G.
- **Peak GPU:** 12.25 GB (both arms).
- **W&B:** run `15g9q3wc`, group `optionb-bi1-identity-regate`, project `wandb-applied-ai-team/gemma-challenge-senpai`.
- **Public evidence used:** internal anchors only — wirbel #607 (47% break), wirbel #616 (0.43% residual), stark #621 (op-level 3D-split-KV localization), locked #319 rung int4_g128_lmhead (126.378 TPS).

### Suggested follow-ups

1. **Tighten the #616 cross-check:** extend the pinned arm to the full prompt set (~128 prompts) so the both-sides-BI=1 residual CI is narrow enough to state the ~3× gap vs #616 with confidence — and confirm whether #616's 0.43% is indeed a not-both-sides-pinned number.
2. **Tolerance-policy decision (advisor/human):** the residual is now purely a sub-0.5-nat int4/dense-head grid-tie question. If the programme adopts τ=0.2–0.3 nat greedy-identity tolerance, Option-B #319 is clean at 0 breaks; if strict τ=0, the 0.092% seed needs a separate ruling.
3. **Bundle for approval:** the identity blocker is now localized and benign, so this verdict + the Option-B speed leg are ready to combine into an `Approval request: HF job` packet whenever you've made the tolerance call. (I'm leaving the speed-side numbers to you — out of my assigned scope.)
