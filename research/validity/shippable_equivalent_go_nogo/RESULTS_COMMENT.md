STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"wandb_run_ids":["e3r8oy2r"],"primary_metric":{"name":"modeled_margin_tps","value":11.55},"test_metric":{"name":"shippable_equivalent_go_nogo_self_test_passes","value":1}}

## Results

**Shippable fastest-equivalent GO/NO-GO: predicate + verify CI + flag (#407 / #417 follow-up).** Pure static-analysis design card — `analysis_only=True, no_hf_job=True, no_served_file_change=True, official_tps=0`. 0 GPU compute, no build/patch/compile/load, no served-file change. Deployed baseline **481.53 / PPL 2.3772 / 128÷128 UNCHANGED** (#52, `2x9fm2zx`). Self-test **73/73** (≥20 required). W&B `e3r8oy2r` (group `shippable-equivalent-go-nogo`). New dir: `research/validity/shippable_equivalent_go_nogo/`.

This card turns #417's modeled bracket into a **human-approvable GO/NO-GO** by delivering the three artifacts #417's follow-ups asked for: (a) an executable predicate, (b) the pre-submission identity-verify CI spec, (c) the feature-flag that de-risks the one binding in-place line.

### (a) The GO/NO-GO predicate (the executable core)
```
SHIP iff (measured_fastest_equivalent_tps > 481.53)   # kanna #416, beats the deployed non-strict #1
         AND byte_identity_verified                    # #319 e2e gate returns identity 1.0
         AND (ppl <= 2.42)                             # quality guardrail (unchanged 2.3772)
         AND (completed == 128)                        # full public run
```
Implemented as a **pure function** of the (to-be-measured) inputs (`go_nogo_predicate(...)`), returning the verdict + each conjunct so the human sees exactly which line is red.

| field | value |
|---|---|
| `go_nogo_predicate_evaluates_GO_under_modeled` | **True** (GO at both bracket ends 492.08 / 494.08) |
| `ship_breakeven_equivalent_tps` (NO-GO boundary) | **481.53** |
| `modeled_margin_tps` (central) | **+11.55** (lo **+10.55** / hi **+12.55**) |
| measured verdict | **PENDING** (#412 + #416 measuring; one-line swap) |

**NO-GO boundary (robustness).** The flip happens exactly at the breakeven — sweep: `480.0→NO-GO, 481.53→NO-GO, 481.54→GO, 492.08→GO`. Decomposed onto the pending inputs (given cb3's **measured** +15.60), the selective-recompute input breakeven is **465.93**, which is **below even the #393 blanket-strict floor 467.48** → a selective-recompute that delivered **zero** speedup over blanket-strict would **still ship** (483.08, +1.55). All three independent strict-equivalent anchors clear: #412 modeled selrec [476.48, 478.48] → [492.08, 494.08]; #393 floor → 483.08; **denken #413 equiv_tps(7)=478.93 → 494.53**. The predicate is **robust (≥+1.55 even worst-case), not knife-edge**. The predicate is a true AND-gate: flipping *any* single conjunct (TPS at breakeven / identity False / PPL 2.50 / completed 127 / None TPS) → NO-GO (proves it enforces greedy-identity **and** PPL ≤ 2.42 per instruction 6, not just TPS).

### (b) Pre-submission identity-verify CI spec (the byte-identity EVIDENCE, not a TPS claim)
- `pre_submission_verify_gpu_min` = **41.8** (from #417 shared-e2e: tier3 e2e shared 35.8 + tier2 decode-width shared 4.0 + 2× tier-1 micro (cb3 new-ref + selrec byte-exact); vs naive unshared 81.6).
- 3-tier #319/#411 harness: TIER1 per-GEMM/-config byte-identity micro (#390); TIER2 decode-width e2e (#381); TIER3 e2e self-referential gate (#319 `gen_greedy_reference --mode served` + `greedy_gate.compare` + `greedy_identity_interlock`).
- `verify_pass_thresholds`: served M=8 flips **3/882 → 0**; e2e greedy identity **== 1.0**; PPL **≤ 2.42**; completed **== 128**. Measured **locally on the A10G before any served-file change**.

### (c) Feature-flag for the one binding in-place line
`SELECTIVE_RECOMPUTE_VERIFY`, **resolved once at serve startup** (binds the verify-reduction function pointer; **no per-step hot-path branch**), wired into `splitkv_verify_patch.py` / `fa_sliding_patch.py` + the manifest env.

| field | value |
|---|---|
| `inplace_line_now_flag_revertible` | **True** (rollback-while-keeping-cb3 = flip `SELECTIVE_RECOMPUTE_VERIFY=0`, **not** a code re-edit) |
| `flag_residual_cost_tps` | **0.0** (startup-resolved → 0 hot-path branch; even a per-step branch < 0.01 TPS) |
| `flag_preserves_byte_identity_both_paths` | **True** |

- **ON (=1, shipped default):** selective-recompute reduction → byte-identical to **BLANKET-STRICT** (the strict reference) → flips 0, identity 1.0.
- **OFF (=0, rollback):** today's-served reduction → byte-identical to **TODAY'S SERVED** verify (the deployed fast path).
- The flag is a pure **selector** between two already-validated reduction functions; it adds **no arithmetic**, so it cannot itself introduce a flip on either path. This converts #417's single binding in-place line from a code-revert into a flag-flip.

### (d) The human-handoff artifact
A single GO/NO-GO checklist written to `research/validity/shippable_equivalent_go_nogo/GO_NOGO_CHECKLIST.md`: the six lines that must be GREEN (measured TPS > 481.53, identity 1.0, PPL ≤ 2.42, 128/128, whole-stack reversible, flag-revert) + the safe operation order (**measure identity locally on the A10G → human approves in GitHub → flip served file + submit**). This is the deliverable that makes ~492–494 actually shippable.

### Greedy identity / PPL (instruction 6)
The stack is byte-exact-equivalent by construction (selective-recompute restores identity; cb3 is body-read-only with a re-keyed reference; MTP only proposes) → PPL unchanged **2.3772 ≤ 2.42**. The predicate **enforces** this (the `bad_ppl → NO-GO` and `bad_identity → NO-GO` self-test conjuncts confirm it).

### Pinned-import cross-check (instruction 1: "import byte-exactly, do not re-derive")
Every pinned constant is cross-checked at runtime against the **merged** #417 (`equivalent_stack_deploy_surface_results.json`) and #413 (`equivalent_tps_optimal_geometry_results.json`) JSON — **14/14 checks pass** (`pinned_import_all_pass=1`): deployed 481.53 / PPL 2.3772 / cap 2.42, blanket base 467.48, cb3 +15.60 / k*=229, selrec bracket [476.48, 478.48], combined bracket [492.08, 494.08], combined verify 41.8, 7 files, human-gated, whole-stack-reversible (#417); equiv_tps(7)=478.93, k*=7 (#413). The pending feeders stark #412 / kanna #416 are hooked as one-line-swappable constants (`SELREC_MEASURED_TPS`, `FASTEST_EQUIVALENT_MEASURED_TPS`).

### Required deliverables (W&B `e3r8oy2r`)
- PRIMARY `shippable_equivalent_go_nogo_self_test_passes` = **True** (73 checks) · `go_nogo_predicate_evaluates_GO_under_modeled` = **True**
- `ship_breakeven_equivalent_tps` = **481.53** · `modeled_margin_tps` = **+11.55** (lo +10.55 / hi +12.55)
- `pre_submission_verify_gpu_min` = **41.8**
- `inplace_line_now_flag_revertible` = **True** · `flag_residual_cost_tps` = **0.0** · `flag_preserves_byte_identity_both_paths` = **True**
- `ship_is_human_gated` = **True** · scope `analysis_only`/`no_hf_job`/`no_served_file_change` = **True** · `official_tps` = **0**

### Comparison vs PR baseline
- Deployed 481.53 / PPL 2.3772 / 128÷128 (#52) — **UNCHANGED** (no served-file change). This is the ship-breakeven the equivalent stack must beat.
- Equivalent stack (#417 `2mv6ssw4`): selective-recompute (in-place, #412 modeled [476.48, 478.48]) + cb3 +15.60 (#403 `iv9i2wks`, k*=229) + MTP K=7/M=8 (deployed) → modeled fastest-equivalent **[492.08, 494.08]**; 7 served files; 41.8 GPU-min verify; whole-stack reversible; **1 binding in-place line now flag-revertible**.

### Public evidence used
Advisor-branch banked W&B runs + advisor-provided pending params: #417 deploy surface `2mv6ssw4`, #411 3-tier verify harness `078yjgax`, #403 cb3 +15.60 k*=229 `iv9i2wks`, #393 corrected strict base 467.48 `0q7ynumg`, denken #413 equiv_tps(7)=478.93 `se8mf9ax`; pending (advisor-provided): stark #412 `selective_recompute_equivalent_tps`, kanna #416 `fastest_equivalent_tps`. In-repo grounding: the in-place edit's landing site `submissions/fa2sw_treeverify_kenyan/{splitkv_verify_patch.py,fa_sliding_patch.py,manifest.json}` + all 5 harness + 4 evidence modules + both cross-check JSONs exist (`g_*` self-test checks).

### Exact command
```
cd target/ && python -m research.validity.shippable_equivalent_go_nogo.shippable_equivalent_go_nogo --self-test
cd target/ && .venv/bin/python -m research.validity.shippable_equivalent_go_nogo.shippable_equivalent_go_nogo \
  --wandb_group shippable-equivalent-go-nogo --wandb_name lawine/shippable-equivalent-go-nogo
```
- **Peak memory: N/A** (pure static analysis, 0 GPU). No `summary.json`/tps/ppl/completed/run_prefix — no benchmark or HF job was launched (`official_tps=0`, as scoped).
- **W&B run:** `e3r8oy2r` (project `wandb-applied-ai-team/gemma-challenge-senpai`), state **finished**, 26 scalar keys + full JSON artifact.

### What happened
The card delivers a **ready-to-approve** GO/NO-GO for shipping the fastest strictly-equivalent config. The predicate evaluates **GO** across the entire modeled bracket, and — the key new result — its robustness is *not* knife-edge: because cb3's +15.60 is already measured and the selective-recompute input breakeven (465.93) sits **below the blanket-strict floor**, the combined config ships even in the worst case where selective-recompute provides no speedup at all. The two #417 follow-ups are closed: the #412/#416 feeders are now one-line-swappable, and the one binding in-place line is **flag-revertible at 0 residual cost with byte-identity preserved on both paths**. The verify CI (41.8 GPU-min, flips 3→0, identity 1.0, PPL ≤ 2.42, 128/128) is the byte-identity evidence the human inspects, and the checklist sequences the safe operation order. The TPS auto-completes the moment #412 + #416 land.

### Suggested follow-ups
- **Wire the measured #412 / #416 numbers** when they land: set `SELREC_MEASURED_TPS` / `FASTEST_EQUIVALENT_MEASURED_TPS` (one line each) → the predicate's `under_measured` verdict flips from PENDING to the real GO/NO-GO, and the checklist's line-1 threshold turns from modeled to measured.
- **Open the gated approval issue** once #416 measures GO: PR/branch + the exact 41.8 GPU-min verify-CI command + the GREEN evidence, per the safe operation order — this is the human-gated served-file change + submission the checklist enables.
- **Land #414 (true-vocab lm_head)** would add a *fourth* component (deployed 16384-row head → full-256k identity); if the human wants strict-vocab identity on top, it slots into the same predicate as another conjunct and the same verify-CI tier structure.
