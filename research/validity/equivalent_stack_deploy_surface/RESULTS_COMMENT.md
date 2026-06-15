STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"wandb_run_ids":["2mv6ssw4"],"primary_metric":{"name":"equivalent_stack_deploy_surface_self_test_passes","value":1},"test_metric":{"name":"combined_incremental_verify_gpu_min","value":41.8}}

## Results

**Deploy-surface card to SHIP the fastest strictly-equivalent config (#407).** Pure static analysis — `analysis_only=True, no_hf_job=True, no_served_file_change=True, official_tps=0`. 0 GPU compute, no build/patch/compile/load, no served-file change. Deployed baseline **481.53 / PPL 2.3772 / 128÷128 UNCHANGED** (#52, `2x9fm2zx`). Self-test **63/63** (≥20 required). W&B `2mv6ssw4` (group `equivalent-stack-deploy-surface`). New dir: `research/validity/equivalent_stack_deploy_surface/`.

### Per-component deploy-surface rows (the fastest-equivalent stack)

| component | deploy kind | files (add/in-place) | verify GPU-min (standalone) | produces_new_reference | tps_unlocked | reversibility | blast radius |
|---|---|---|---|---|---|---|---|
| **selective-recompute verify** (stark #412) | **IN-PLACE verify edit** | **4 (3 / 1)** | 40.8 | **False** (byte-exact to strict by design; *requires* verification) | +9..+11 modeled (`#412` measuring) | feature-flag / code-revert (NOT a bucket flip) | **served verify hot path, every decode step** |
| **cb3 body supply** (kanna #403 `iv9i2wks`, k*=229) | additive submission dir | 6 (6 / 0) | 40.8 | True (RHT+VQ → new argmax) | **+15.60** (measured, PPL-safe) | manifest bucket/quant_method flip | isolated new kernel+checkpoint |
| **MTP K=7 / M=8 drafter** | already DEPLOYED, no change | 0 (0 / 0) | 0.0 | False | 0 (already banked) | n/a (nothing to revert) | none |

### Required deliverables (W&B `summary/`)
- `selective_recompute_is_in_place` = **True** (the one in-place edit) · `cb3_is_additive` = **True**
- `combined_served_files` = **7** (3 shared scaffold + 3 cb3 distinct + **1 in-place verify edit**; 6 additive + 1 in-place)
- `combined_incremental_verify_gpu_min` = **41.8** (shared #319 e2e, vs naive unshared **81.6** → saves 39.8)
- `shared_e2e_survives_inplace` = **True**
- `whole_stack_reversible` = **True**
- `deploy_is_human_gated` = **True**
- `most_expensive_deploy_line` = the in-place selective-recompute verify edit (see below)
- `fastest_equivalent_tps` = **PENDING** (`#412` + `#416` measuring); modeled bracket **[492.08, 494.08]**
- PRIMARY `equivalent_stack_deploy_surface_self_test_passes` = **True** (63 checks)

### Combinability + shared verify (the key question)
**The shared-e2e payoff SURVIVES the in-place edit.** The combined deploy is ONE submission dir: cb3's *additive* modules (kernel + quant patch + checkpoint, orthogonal to the verify path) PLUS the selective-recompute *in-place* edit to the same forked verify/attention reduction, riding the unchanged MTP drafter. cb3 touches the body-GEMM weights and the verify path **not at all**; selective-recompute touches the verify reduction and the weights **not at all** → **no contention**. The expensive #319 tier-3 e2e gate captures the FINAL composed served stack's greedy output and validates byte-identity — it is change-agnostic, so **one e2e capture both re-keys cb3's new reference AND validates selective-recompute's byte-exactness**. The in-place edit does **not** force a separate cb3 re-run.

`combined_incremental_verify_gpu_min` = tier3 e2e (shared **35.8**) + tier2 decode-width (shared **4.0**) + 1 tier-1 micro per identity-claim component (cb3 new-ref + selective-recompute byte-exact = **2 × 1.0**) = **41.8 GPU-min**.

### The single most expensive / risky line
> selective-recompute **in-place** verify-path edit (ε near-tie gate + higher-precision reduction on the ~23.6% flagged steps, edited into `splitkv_verify_patch.py` / `fa_sliding_patch.py`): it is the hot path **every** decode step runs through, so a gate bug corrupts **all** served output (not an isolated kernel artifact); its identity-verify validates a **data-dependent** correctness property (the ε gate must never miss an argmax flip); and revert-while-keeping-cb3 is a **code change**, not a manifest bucket flip.

This is the deploy-surface asymmetry vs my #411 (all-additive) ledger: cb3 reverts by a config flag (verify logic never touched); selective-recompute changes hot-path logic. Both are *deployment-reversible* (roll back = re-submit the prior package), but the in-place line carries concentrated blast radius + the only non-flag revert.

### The deploy-proposal skeleton (auto-completes when #412 + #416 land)
```
fastest_equivalent_tps = selective_recompute_equivalent_tps  (stark #412, PENDING; modeled [476.48, 478.48])
                       + cb3 +15.60                           (kanna #403 k*=229, MEASURED)
                      == kanna #416 fastest_equivalent_tps    (PENDING; modeled [492.08, 494.08])
```
Under #407 the gate is **not** "clears 500" — the modeled **~492–494** already **BEATS the non-strict deployed 481.53** while carrying the byte-identity guarantee the deployed config lacks. **Total served_files 7 · total identity-verify ~41.8 GPU-min · whole-stack reversible · 1 binding in-place line.** Shipping this is the **HUMAN-GATED** action (served-file change + leaderboard submission); this card only PRICES it.

### Comparison vs PR baseline
- Deployed 481.53 / PPL 2.3772 / 128÷128 (#52) — **UNCHANGED** (no served-file change).
- Blanket-strict base 467.48 (#393 `0q7ynumg`); strictly-equivalent components: selective-recompute (#412, modeled +9..+11), cb3 +15.60 (#403 `iv9i2wks`, k*=229, additive), MTP K=7/M=8 (deployed).
- My #411 (`078yjgax`) 5-col schema + 3-tier verify harness (~40.8 GPU-min/lever) + shared-e2e finding reused; this card adds the **in-place vs additive** distinction the fastest-equivalent stack requires.

### Public evidence used
Advisor-branch banked W&B runs + advisor-provided pending params: #411 unified ledger `078yjgax`, #404 cb3 surface `jqhlftrc`, #403 cb3 conservative-k k*=229 +15.60 `iv9i2wks`, #393 corrected strict base 467.48 `0q7ynumg`; pending (advisor-provided): stark #412 `selective-recompute-equivalent-tps`, kanna #416 `fastest_equivalent_tps`. In-repo grounding: deployed verify path confirmed at `submissions/fa2sw_treeverify_kenyan/splitkv_verify_patch.py` + `fa_sliding_patch.py` (where the in-place edit lands), vs cb3's orthogonal additive modules; all 5 harness + 4 evidence + 5 deployed paths exist (`g_*` self-test checks).

### Exact command
```
cd target/ && python -m research.validity.equivalent_stack_deploy_surface.equivalent_stack_deploy_surface --self-test
cd target/ && .venv/bin/python -m research.validity.equivalent_stack_deploy_surface.equivalent_stack_deploy_surface \
  --wandb_group equivalent-stack-deploy-surface --wandb_name lawine/equivalent-stack-deploy-surface
```
- **Peak memory: N/A** (pure static analysis, 0 GPU). No `summary.json`/tps/ppl/completed/run_prefix — no benchmark or HF job was launched (`official_tps=0`, as scoped).
- **W&B run:** `2mv6ssw4` (project `wandb-applied-ai-team/gemma-challenge-senpai`), state finished, 42 scalar keys + full JSON artifact.

### What happened
The card delivers the ready-to-hand-to-human deploy proposal for the fastest strictly-equivalent config. The central new result vs #411: the fastest-equivalent stack has **one in-place lever** (selective-recompute verify) where #411's were all additive — and pricing that honestly shows the shared-e2e verify payoff **still holds** (one capture re-keys cb3 + validates selective-recompute), so the combined verify is **41.8 GPU-min**, not the 81.6 naive sum. The in-place-ness does **not** raise verify *cost*; it raises verify *risk* and concentrates it in one line (the ε near-tie gate on the hot path). Net deploy surface: **7 served files, ~41.8 GPU-min, whole-stack reversible, 1 binding in-place line**, human-gated. The TPS auto-completes from #412 + #416; modeled ~492–494 beats the non-strict deployed 481.53 with an identity guarantee.

### Suggested follow-ups
- **Wire the #412 / #416 measured numbers** into the proposal once they land (the card already parameterizes the formula; only `tps_unlocked.measured` and `fastest_equivalent_tps.value` need filling).
- **Feature-flag the in-place verify edit** so revert-while-keeping-cb3 becomes a flag flip rather than a code change — this is the one move that lowers the binding line's reversibility cost (flag logic still ships, but rollback no longer needs a re-edit). A small design note pricing that flag would de-risk the human gate further.
- **lm_head true-vocab (land #414)** prices a *fourth* deploy-surface row (the deployed 16384-row head → full-256k identity); when it lands it slots into this same schema as another component if the human wants strict-vocab identity on top of the fastest-equivalent stack.
