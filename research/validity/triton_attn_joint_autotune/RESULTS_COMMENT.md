STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"autotune_strict_endtoend_tps":475.88,"autotune_strict_endtoend_wall_tps":448.69,"autotune_realized_tps_delta":-5.65,"realized_wall_delta_pct":-1.173,"realized_wall_delta_ci95_pct":0.227,"autotune_beats_deployed_481":false,"crosses_481_ci_clean":false,"strict_null_confirmed":true,"p26_byte_exact_ceiling_tps":0.26,"realized_within_sigma_of_p26_ceiling":false,"byte_exact_identity":false,"census_frac_identical":0.531,"eager_floor_frac_identical":1.0,"bm4_breaks_greedy_identity":true,"e_accept_default":3.8416,"e_accept_bm4":3.8214,"e_accept_delta_pct":-0.526,"triton_verify_surface_includes_head256_FLAG":true,"larger_than_447_1p27pct_surface_FLAG":true,"ppl":2.3767,"ppl_gate":2.42,"self_test_passes":true,"classification":"evaporates","wandb_run_ids":["gyw2ksvs","yd5ywiwe","grrc3zms","cy0ijlit"],"primary_metric":{"name":"autotune_strict_endtoend_tps","value":475.88},"test_metric":{"name":"ppl","value":2.3767}}

## Results — 4th kernel-tiling lever confirmed strict-NULL + two LOUD flags

**One-line verdict:** the decisive served-stack wall A/B confirms the autotune as the **4th independent kernel-tiling strict-NULL** (joining verify-wall #447, int4-GEMM #448, drafter-kernel #449): bm4 (`{block_m:4, tile:32, warps:4, stages:2}`) realizes **−5.65 TPS** (wall **−1.173% ± 0.227%**, both seeds REAL-negative), `autotune_strict_endtoend_tps`=**475.88 < 481.53**, `crosses_481_ci_clean=False`. The +15.86 microbench Δ evaporated exactly as pre-adjudicated by #447 (+0.26 byte-exact ceiling). **But the result is NOT the clean +0.26 NULL #447 predicted — it carries two surprises I am flagging LOUD per your instruction:**

> **FLAG 1 — bm4 is NOT byte-exact on the served path (greedy-identity = 0.531, not 1.0).** The Triton **attention** tile retune is *itself* greedy-unsafe on the 3D split-KV path — #450's "greedy-unsafe FP-reassociating split-K" is not confined to the Marlin GEMM. Proven real (not eager-FP noise) by a clean 100% default-vs-default floor control.
>
> **FLAG 2 — the Triton verify-attention surface is materially larger than #447's 1.27%.** My injector forced bm4 on head-256 **sliding** (×5) AND head-512 global (×1) Triton 3D-split-KV calls in the M=8 verify step (`q_rows=8, IS_3D=True`) — head-256 sliding reaches **Triton**, not FA2. This reopens the strict-supply surface question for a *byte-exact* retune (which bm4 is not).

Analysis-only / 0-TPS: no served-file change, env-gated toggle reverted clean (`toggle_reverted_clean=True`, empty git diff), no HF job, `official_tps=0`. **Not flagging bm4 for deploy** — it failed on both TPS and identity. BASELINE stays 481.53.

---

### (1) Strict end-to-end served wall A/B — `autotune_strict_endtoend_tps = 475.88`, no CI-clean crossing

Default tiling vs bm4, served int4 stack (`google/gemma-4-E4B-it-qat-w4a16-ct`), 128 prompts × 512 decode, **2 seeds × 3 reps = n=6/arm**, closed-loop wall TPS, toggle→measure→revert. bm4 verified applied on every candidate run (30/30 forced-log hits, all 3D split-KV); baseline unpatched.

| arm | pooled p50 wall TPS (n=6) | ×1.06 → official anchor | E[T] (e_accept_exact) |
|---|---|---|---|
| **default** bm16/s3 | 454.00 | **481.57** (reproduces deployed 481.53, err 0.008%) | 3.8416 |
| **bm4 candidate** | 448.68 | **475.71** (lo-CI 466.38); on-incumbent **475.88** | 3.8214 |

- **realized wall Δ = −1.173% (CI95 ±0.227% ≈ ±1.1 TPS, lower bound −1.400%)** — both arms' CIs clear the σ_hw≈1%≈4.8 TPS bar by ~4×. Both seeds REAL-negative (seed1 −1.18%, seed2 −1.25%).
- **`autotune_strict_endtoend_tps` = 475.88** (N=6 median, incumbent-anchored) → **−5.65 TPS**. `crosses_481_central=False`, `crosses_481_ci_clean=False` — the entire CI sits below 481.53.
- **`realized_within_sigma_of_p26_ceiling = False`** — the realized Δ is −5.65, **not** within σ_hw of #447's +0.26 byte-exact ceiling. Section (2) explains why (bm4 is not byte-exact → the +0.26 ceiling does not apply to it).

**Answer to deliverable (1): the autotuned arm does NOT cross 481.53; the CI is clean and resolves a −5.65 TPS regression.**

### (2) Reconciliation with #447's +0.26 ceiling — collapse, and WHY it overshoots to negative

#447's +0.26 TPS is the ceiling for a **byte-exact** retune of the Triton attention *compute* (same token trajectory, faster kernel). bm4 lands at −5.65, far past that — because **bm4 is not byte-exact**, so the wall Δ is not a compute measurement at all. Per-arm telemetry decomposes the −1.12% (mean) wall:

| mechanism | measured | TPS contribution |
|---|---|---|
| **acceptance-trajectory drift** (bm4 emits different tokens → different greedy path) | e_accept 3.8416 → 3.8214 = **−0.526%**; accept_rate −0.29pp | **≈ −0.53%** |
| **grid-expansion step overhead** (BLOCK_Q 4→1 ⇒ grid (3,2,16)→(9,2,16) = **96→288 CTAs** on an already-occupancy-SATURATED 3D path, ~96>80 SMs) | residual of −1.12% wall | **≈ −0.59%** |
| **attention-compute speedup** | — | **≈ 0** (the +15.86 1.528× microbench gain does NOT exist in-graph) |

- The implied attention-compute Δ is **~0**, fully consistent with #447: a strict tile retune of the Triton attention is worth ≤+0.26 TPS (the slice is too small to move the needle). The +15.86 microbench (a 2D-occupancy fix measured on ~6 CTAs) has no purchase on the served 3D-saturated path.
- bm4's extra −5.65 is **not** a compute effect — it is the price of (a) breaking identity (acceptance drift) and (b) tripling CTAs on a saturated grid (launch overhead). **#447's +0.26 and bm4's −5.65 are not in contradiction — they measure different things.** Because bm4 perturbs the trajectory, the wall A/B *cannot* return a clean compute-only number; only a byte-exact config could, and bm4 isn't one.

**Answer to deliverable (2): the A/B shows COLLAPSE, not the modeled 45–58% speedup. Compute-Δ ≈ 0 (matches #447's +0.26 ceiling); the realized −5.65 is acceptance-drift + grid overhead, both consequences of bm4 not being byte-exact.**

### (3) Greedy-identity census + eager-noise floor — `byte_exact_identity = False` (FLAG 1)

Paired same-path census, 64 prompts × 512 (≥50 ✓), both arms served fresh and compared **against each other**, plus the default-vs-default floor control that makes the lawine-#438 eager artifact cancel:

| measurement | identical | byte_exact | reading |
|---|---|---|---|
| **bm4 vs default** | 34/64 = **53.1%** | False | 30 prompts diverge; first flip @ token 290 |
| **default vs default** (eager-noise FLOOR, fresh procs, identical config) | 64/64 = **100.0%** | **True** | eager path is deterministic across processes |

- **The floor is the lynchpin:** default-vs-default is **100% byte-identical across two independent processes** → the eager M=8 path is NOT a cross-process FP-noise source here, the #438 confound **does not apply**, and the 53% bm4 divergence is a **REAL identity break**, not eager noise.
- **So bm4 breaks greedy-identity on the served 3D split-KV path.** The injector's *a-priori* byte-exact argument (`reduce_segments` is BLOCK_Q-independent, num_stages is a pure pipeline knob — `served_bm4_injector.py` L19-32) is **empirically refuted**: changing BLOCK_M→4/BLOCK_Q→1 (and/or num_stages 3→2) **does** perturb the 3D split-K reduction order → FP-reassociation → flipped argmax. This is the *exact* greedy-unsafety class #450 attributed to the Marlin verify-GEMM — **and it is also present in the Triton 3D split-KV attention reduction.** That is why I ran the census instead of trusting the self-test assertion: the assertion was wrong.
- PPL still passes (**2.3767 ≤ 2.42**, anchor 2.3772) — expected and *not* reassuring: teacher-forced PPL is robust to sub-ULP per-logit perturbations while free-running greedy generation cascades from a single argmax flip. PPL-neutral ≠ greedy-identical. `verdict_byte_exact_and_ppl_pass=False`.

**Answer to deliverable (3): census is NOT byte-exact (53.1% identical, floor-proven real, not eager FP); `byte_exact_identity=False`, flips present. The Triton attention tile retune is greedy-UNSAFE on 3D split-KV. PPL gate passes but does not rescue identity.**

### (3b) FLAG 2 (LOUD) — Triton verify surface includes head-256 sliding → larger than #447's 1.27%

My injector gate fires only on launches **already at the deployed verify config** (`HEAD_SIZE∈{256,512}, num_queries_per_kv=4, BLOCK_M=16, BLOCK_Q=4, 2≤q_rows≤64`). It forced bm4 on, per server log:

```
CENSUS forced[1] head=256 IS_3D=True q_rows=8 grid (3,2,16)->(9,2,16) BLOCK_M 16->4 BLOCK_Q 4->1 num_stages->2
CENSUS forced[2] head=256 IS_3D=True q_rows=8 ...
CENSUS forced[3] head=512 IS_3D=True q_rows=8 ...
CENSUS forced[4..6] head=256 IS_3D=True q_rows=8 ...   (window 5×head256 : 1×head512 ≈ the 35:7 sliding:global layer ratio)
```

- `q_rows=8` uniquely isolates the **M=8 verify step** (drafter draft-steps are q_rows=1, excluded; prefill is q_rows≫64, excluded). So these are genuine verify-step attention calls.
- **head-256 (sliding-window) attention reaches the Triton `kernel_unified_attention` 3D split-KV path in verify — NOT FA2.** Most likely the deployed `splitkv_verify_patch.py` (`SPLITKV_VERIFY=1`, overrides max_seqlen_q) redirects the *entire* M=8 verify batch — sliding layers included — to Triton 3D split-KV, even though sliding uses FA2 in the prefill/decode path.
- **This contradicts #447's verify-wall map** (head-512 split-KV = the *only* Triton verify surface = 1.27%). If sliding head-256 (35 layers) also routes through Triton in verify, the Triton verify-attention surface is **materially larger** (up to all 42 layers), which **reopens the strict-supply question**: a *byte-exact* retune of that larger surface could exceed +0.26.
- **Important honesty caveat:** I measured end-to-end wall TPS + the routing (forced-log), **not** the per-kernel µs verify-fraction that #447 measured directly. So I am flagging a **routing discrepancy that implies a larger fraction**, not a competing fraction measurement. And it does **not** rescue bm4: bm4 is identity-breaking (so it can't capture byte-exact headroom) and its measured wall effect is −5.65. The byte-exact headroom of the larger head-256+512 surface is **unmeasured** — that is the reopener for you to reconcile with #447, not a beat.

### (4) Verdict — bank as 4th kernel-tiling strict-NULL; bm4 DEAD on two axes; lever closed

Per your branch ("realized Δ ≤ +2 TPS or no CI-clean crossing → bank the 4th realized-NULL and close the autotune lever"):

- **TPS:** realized **−5.65 TPS**, frontier **475.88 < 481.53**, no CI-clean crossing → `autotune_beats_deployed_481=False`. Compute-Δ ≈ 0 confirms #447's strict ceiling.
- **Identity:** `byte_exact_identity=False` (53.1%, floor-proven real) → not a byte-exact-safe deploy *even if it were faster*.

→ **4th independent kernel-tiling lever confirmed strict-NULL** (verify-wall #447 / int4-GEMM #448 / drafter-kernel #449 / **autotune bm4 #442**). The isolated-op Δ trap collapsed a 5th time: pinned-K +13.998→−5.82, cb3 +15.60→0.0, static-K +13.2%→−8.63%, autotune-isolated +15.86→**−5.65**. **bm4 is a DEAD deploy candidate; the autotune lever is CLOSED.** **No deploy, no HF job, no submission.** The only live thread is FLAG 2 (the larger Triton verify surface) — a *byte-exact* retune question I'm handing to you, not landing.

`served_bm4_wall_ab_self_test_passes=True` (7/7 required), `toggle_reverted_clean=True`.

---

### summary.json-equivalent fields
- **tps:** baseline 454.00 wall / 481.57 official-anchor (reproduces 481.53); candidate 448.68 wall / **475.88** incumbent-anchor. **`autotune_strict_endtoend_tps`=475.88, realized Δ=−5.65 (−1.173%).**
- **ppl:** 2.3767 (gate ≤2.42, anchor 2.3772) — passes; greedy-identity FAILS (53.1% identical, `byte_exact_identity=False`).
- **completed count:** A/B 128×512×2seeds×3reps (n=6/arm); census 64×512×2 arms; floor 64×512.
- **run_prefix:** `wirbel/served-bm4-wall-ab`, `wirbel/served-bm4-census`, `wirbel/served-bm4-eager-floor`.

### Peak memory
- Served int4 stack at the **deployed `gpu_memory_utilization`, unchanged** (bm4 is a launch-config knob, no extra allocation) → peak VRAM == deployed baseline, within the A10G 22 GiB serving envelope. The 0-GPU CPU self-test harness peak is 0.318 GiB.

### Reproduce / env
```bash
cd target/
# (1) decisive served wall A/B — 0 TPS, env-gated toggle, auto-reverted:
CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
  research/validity/triton_attn_joint_autotune/served_bm4_wall_ab.py \
  --num-prompts 128 --output-len 512 --seeds 1 2 --reps 3 \
  --block-m 4 --num-stages 2 --wandb_name wirbel/served-bm4-wall-ab
# (3) greedy-identity census + PPL:
CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
  research/validity/triton_attn_joint_autotune/served_bm4_census.py \
  --num-prompts 64 --output-len 512 --seed 1 --wandb_name wirbel/served-bm4-census
# (3-control) default-vs-default eager-noise FLOOR:
CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
  research/validity/triton_attn_joint_autotune/served_bm4_eager_floor.py \
  --num-prompts 64 --output-len 512 --seed 1 --wandb_name wirbel/served-bm4-eager-floor
```
- **Device:** NVIDIA A10G sm_86, torch 2.11.0+cu130, triton 3.6.0, vLLM v0.22.1rc1.
- **W&B:** A/B `gyw2ksvs` (seed1) / `yd5ywiwe` (seed2) / `lrepm2gj` (aggregate); census `grrc3zms`; eager-floor `cy0ijlit`. Group `triton-joint-autotune`.

### What happened — honest analysis
The joint sweep found a real **2D-isolation** lever (BLOCK_M 16→4, 1.528×), but it was a microbench-vs-served path artifact and it evaporated exactly as #447/#450 pre-adjudicated. The served M=8 verify is 3D split-KV, occupancy-saturated past the 80-SM A10G limit, so the BLOCK_M "occupancy fix" only adds ~288 CTAs of overhead. Two findings beyond the clean +0.26 NULL: (1) bm4 is **greedy-unsafe** on the 3D split-KV attention reduction (identity 53.1%, floor-proven) — #450's split-K FP-reassociation hazard lives in the Triton attention too, not just the Marlin GEMM; the injector's theoretical byte-exact claim was empirically wrong, vindicating the census-over-assertion approach. (2) head-256 **sliding** attention routes through Triton 3D split-KV in verify (not FA2), so the Triton verify surface looks larger than #447's 1.27% — a reconciliation item I'm flagging, not a beat. PPL survives because teacher-forcing hides the greedy cascade. Net: this served verify kernel is at its occupancy + FP-reduction-order floor; no tiling config recovers TPS without breaking identity.

### Suggested follow-ups
1. **Reconcile FLAG 2 with #447 (for you).** Does `splitkv_verify_patch.py` route head-256 sliding through Triton in the M=8 verify step? If so, #447's 1.27% (head-512 only) undercounts the Triton verify surface, and the *byte-exact* headroom of the full head-256+512 surface is unmeasured. I can run denken #447's direct per-kernel µs map on the head-256 sliding verify calls if you want the fraction nailed.
2. **Stop autotuning this kernel.** Four kernel-tiling levers (#447/#448/#449/#442) now confirmed strict-NULL; the supply-side attention lever is exhausted under byte-exact identity. Redirect to acceptance/coverage (the +1.40pp reopener from #244) — orthogonal to tiling and not bounded by this floor.
3. **If FLAG 2's larger surface is real and you want to chase it:** the only identity-safe path is a BLOCK_Q-invariant 3D split-K reduction (fixed partition) before any tile retune — a real kernel change, gated behind the #450 `relax-equivalence-prize` group if it can't be made byte-exact.
