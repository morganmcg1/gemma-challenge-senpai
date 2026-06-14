# Descent decode-path host-residency / graph-capture sweep + net step (PR #163)

**Lane:** host-residency / CUDA-graph-**capturability** of the descent decode path + the net step budget.
**Scope:** LOCAL CPU/A10G static-analysis + arithmetic + a tiny empirical capturability probe. NO vLLM
serve change, NO HF Job, NO submission, NO kernel deploy. BASELINE stays **481.53 TPS** (PPL 2.3777),
greedy/PPL untouched. **Produces a build-readiness inventory + a net-step bound; does NOT authorize a launch.**

Tool: `scripts/profiler/host_residency_sweep.py` · JSON: `research/spec_cost_model/host_residency_sweep/host_residency_sweep.json`
· W&B run `dmcskhwi` (group `descent-path-host-residency-sweep`).

---

## The decision this answers

My #157 (MERGED) found the `relocate_salvaged_kv` host loop is a silent step-collapsing landmine
(descent 522 → 77 TPS) — it is correctness/PPL-clean, so it passes every functional check, but as a
data-dependent Python loop it **cannot be CUDA-graph-captured** and pins the step host-bound. My #154
(MERGED) found a *second* host-residency op (the decode-path scatter+LP in `compute_logits`, eager
outside the graph). That is **TWO** independent host-resident ops on the descent decode path, each
individually capable of collapsing the step — and they were found **one at a time**.

Before the one irreversible shot, the decision-critical question is: **are #154 and #157 the ONLY
host-resident / graph-uncapturable ops on the descent decode path, or are there MORE landmines hiding?**
Finding them one-at-a-time is not a launch guarantee. This sweep enumerates the *whole field*.

**My lane is distinct from the adjacent sync-counting work.** A host Python loop need not register as a
CUDA *sync* yet still breaks graph capture — that gap is exactly what this sweep covers. lawine #147's
sync-point taxonomy is **consumed** here (not re-measured); lawine #161's spine op-count is **excluded**
(a slot is armed). My axis is graph-**capturability** + the net step.

---

## Result (headline)

| metric | value |
|---|---|
| **PRIMARY** `host_residency_sweep_self_test_passes` | **1** (10/10 self-tests PASS) |
| **TEST** `descent_path_residual_host_ops_count` | **0** (field swept clean) |
| ops enumerated on the timed decode window | 12 (6 host-resident) |
| both anchors (#154, #157) re-discovered | **True** |
| capturability probe ran / consistent | **True / True** (4/4 cases match) |
| `net_descent_step_pinned` (realizable build) | **1.2086 units** |
| `net_clear_500_bar` descent-only / both-bugs | **4.8241 / 4.8241** |
| fits inside the #136 anchor (1.2182) | **True** |
| peak GPU memory (probe) | 0.519 MiB |

**Verdict: the FULL descent decode path FITS inside the measured step ≈1.2182 once all greedy-safe
recoveries are applied.** No unclassified host op hides on the path
(`descent_path_residual_host_ops_count = 0`). The sole residual that blows the budget is a **host-loop**
relocate — and that is already classified (#157) with a greedy-safe vectorized design.

---

## (1) Static host-residency inventory — the whole field

Every op on the post-`PRECACHE_BENCH` timed decode window, classified by which of
{`a_host_roundtrip`, `b_datadep_pyloop`, `c_capture_break`} it triggers, with owner + status.

| # | op | site | host-residency class | owner | status |
|---|---|---|---|---|---|
| 1 | `drafter_propose_loop` | sitecustomize.py:158-203 `_run_graph_body` (LOOPGRAPH) | none | clean | CLEAN(captured) |
| 2 | `target_verify_forward` | vLLM cudagraph (42L, M=32 int4-Marlin); #136 `gemm_all_graphed` | none | clean | CLEAN(captured) — **the anchor step** |
| 3 | **`compute_logits_scatter_LP`** | serve_patch_pck04.py:335-342 → `_scatter_to_full_vocab`:113-168 | **c_capture_break** | **#154** | **RECOVERABLE (anchor #1)** |
| 4 | `fused_accept_prep` | sitecustomize.py:921-963 `_dixie_*` (TRITON) | none | clean | CLEAN(device-resident) |
| 5 | `descent_accept_walk` | land #71 build; modeled in salvage_walk_overhead.py | a_host_roundtrip, c_capture_break | #147 | design sync-free (GPU-hidden) |
| 6 | `salvage_branch_selection` | land #71; #147 taxonomy | a_host_roundtrip, c_capture_break | #147 | design sync-free |
| 7 | `accept_length_readout` | land #71; #147 taxonomy | a_host_roundtrip, c_capture_break | #147 | design sync-free |
| 8 | **`relocate_salvaged_kv`** | land #71; chiku-inu trace; priced in salvage_kv_relocation_audit.py | **a_host_roundtrip, b_datadep_pyloop, c_capture_break** | **#157** | **LANDMINE-if-host-loop / RECOVERABLE-vectorized (anchor #2)** |
| 9 | `kv_commit_blocktable_update` | sitecustomize.py:150-155 `_refresh_static_buffers` + land #71 commit | none | #157 | CLEAN(device-resident) — the slot-map relocate |
| 10 | `spine_conditional_depth1` | land #71; lawine #161 lane | **excluded** | #161 | EXCLUDED (slot armed) |
| 11 | `terminal_output_token_ids_cpu` | vLLM v1 `parse_output` (`accepted.cpu()`) | a_host_roundtrip | structural | UNAVOIDABLE-in-anchor |
| 12 | `input_ids_next_step_update` | sitecustomize.py:182,194 `input_ids[:1].copy_` | none | clean | CLEAN(captured) |

**Method-completeness check passes:** the sweep re-discovers **both** known anchors (#154 op 3, #157 op 8)
as host-resident with the right owner. Of the 6 host-resident ops, **2 are the anchors**, **3 are #147's
consumed sync surface** (ops 5/6/7), **1 is the structural terminal sync already in the anchor** (op 11).
After attributing every host-resident op to a lane, the **residual = 0**: no host op hides outside the
{#154, #157, #147-consumed, #161-excluded, structural} accounting.

### Why ops 9 and 11 are not new landmines
- **Op 9 (`kv_commit_blocktable_update`)** is host-bound *only* if `accept_len` is read to host — but that
  read *is* op 7 (`accept_length_readout`, #147's lane). If `accept_len` stays a device scalar (the #147
  sync-free rule), the commit/advance is a device `.copy_` = the relocate's zero-copy `paged_slotmap`
  variant (#157). Not a new op.
- **Op 11 (`terminal_output_token_ids_cpu`)** is the one structurally-unavoidable host sync per step
  (every decode step streams). It is already inside the 1.2182 anchor and GPU-hidden behind the GEMM tail.

---

## (2) Empirical capturability probe — the distinctive measurement of this lane

This is the leg that distinguishes capturability from sync-counting. Each case runs in its **own
subprocess** (a capture-break poisons the CUDA context). On the A10G all four cases match the taxonomy:

| case | captured | expected | match | mechanism |
|---|---|---|---|---|
| `device_vectorized_relocate` | **True** | True | ✓ | #157 design: device `[L,W,H,D]` `index_select`+`index_copy_` captures |
| `host_loop_relocate` | **False** | False | ✓ | #157 landmine: per-row `.to("cpu")` → `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture` |
| `sync_free_accept_walk` | **True** | True | ✓ | #147 sync-free: match-mask → cumprod → device argmax captures |
| `sync_bound_accept_walk` | **False** | False | ✓ | #147 sync-bound: per-node `bool(.item())` → `AcceleratorError: operation failed ... during capture` |

This **empirically grounds** the inventory's `c_capture_break` classification: the device-resident designs
capture; the host round-trips break capture with the exact CUDA errors the taxonomy predicts. (`consistent=True`,
`n_cases_ran=4`.) The probe gracefully skips when no GPU is exposed — the static + arithmetic legs stand alone.

---

## (3) Net the step budget

Composed from the consumed merged anchors: `anchor (1.2182) − #154 scatter+LP recovery + #157 relocate
(± vectorized) + NEW(=0)`. The #147 sync-free accept-walk is GPU-hidden (GREEN, +0 net at the bar). The
#161 both-bugs spine delta is armed (0 today). `K_CAL = 125.268`, `1 step-unit = 7982.89 µs`.

| scenario | net step (units) | bar | fits ≤1.2182 | descent E[T]=5.04 | both-bugs E[T]=5.207 |
|---|---|---|---|---|---|
| zero-recovery (reproduce #136 anchor) | 1.2182 | 4.8624 | — | 518 TPS ✓ | 535 TPS ✓ |
| descent + vectorized relocate (no #154) | 1.2226 | 4.8800 | False | 516 TPS ✓ | 534 TPS ✓ |
| **descent + vectorized relocate + #154 (realizable)** | **1.2086** | **4.8241** | **True** | **522 TPS ✓** | **540 TPS ✓** |
| descent + paged slot-map + #154 (zero-copy ideal) | 1.2067 | 4.8166 | True | 523 TPS ✓ | 541 TPS ✓ |
| descent + **HOST-LOOP** relocate (the landmine) | 8.1512 | 32.5351 | False | **77 TPS ✗** | 80 TPS ✗ |

`net_descent_step_pinned = 1.2086 units`; `net_clear_500_bar = 4.8241` (descent-only and both-bugs alike
today, since BUG-1 is a numerator fix — the #161 slot folds the both-bugs step delta when it lands).

**The realizable build fits inside the anchor with margin.** Vectorizing the relocate (#157) *adds* only
35.3 µs/step (+0.36%); recovering #154's scatter+LP *subtracts* ~111.9 µs/step. Net: the path lands
**below** the 1.2182 anchor (1.2086 units), so the clear-500 bar actually *falls* to 4.824 and both E[T]
regimes clear 500 with cushion (5.04 − 4.824 = +0.216; 5.207 − 4.824 = +0.383).

The **only** way the step collapses is a host-loop relocate (8.15 units, bar 32.5, → 77 TPS). That is
anchor #2, already classified with a greedy-safe vectorized/paged design.

---

## (4) Self-validation (PRIMARY)

10/10 self-tests pass → `host_residency_sweep_self_test_passes = 1`:

1. ✓ re-discovers both anchors (#154, #157) as host-resident
2. ✓ zero-recovery arithmetic reproduces the #136 anchor (1.2182) exactly
3. ✓ reproduces #157's published vectorized bar (4.880)
4. ✓ reproduces #157's published host-loop bar (32.59)
5. ✓ #154 recovery lands in the published 4.808–4.820 band (bar_154_only = 4.8064)
6. ✓ realizable build fits inside the anchor
7. ✓ feasibility binary: vectorized clears 500, host-loop does not
8. ✓ residual host-op count well-formed (finite non-negative int)
9. ✓ capturability probe consistent (ran, 4/4 match)
10. ✓ NaN-clean (every headline numeric finite)

`descent_path_residual_host_ops_count = 0` (TEST) — the field is swept clean.

---

## (5) Build-readiness hand-off (land #71)

**FIELD SWEPT CLEAN** — the descent decode-path host-residency surface is fully accounted: 2 anchors
(#154 scatter+LP, #157 relocate), #147's consumed accept-walk sync surface, #161's excluded spine
conditional, and the structural terminal sync. **0 new landmines.**

Ops that MUST be vectorized / device-resident in the land #71 build, with their greedy-safe designs:

| op | greedy-safe design |
|---|---|
| `compute_logits_scatter_LP` (#154) | on the token-selection path replace `scatter[M,262144]+LP+argmax_262144` with `argmax(pruned[M,12288])` → `kept_ids` remap (kept_ids ascending ⇒ first-occurrence tiebreak == full-vocab argmax; `equivalence_rate=1.0`). Keep full scatter+LP on the prompt_logprobs/PPL prefill path. |
| `relocate_salvaged_kv` (#157) | single FUSED device `index_select`+`index_copy_` over the `[L,W,H,D]` stack by a **device** commit-index in one launch, OR a paged slot-map update (zero-copy). The commit-index is produced on-device by the accept walk and consumed without a host readout → stays inside the captured graph. Bit-exact bf16 permutation (`equivalence_rate=1.0`). |
| `descent_accept_walk` (#147) | vLLM-v1 `RejectionSampler` (PR #14930, zero-sync): device match-mask/cumprod accept length; the next step's expand indexes by the device scalar (no `.item()`). |
| `salvage_branch_selection` (#147) | `best = branch_scores.argmax()`; gather the chosen branch by the **device** index. |
| `accept_length_readout` (#147) | keep `accept_len` on device; next-step KV/context indexing uses the device scalar. |

**Launch gate: INFORMS, does NOT authorize.** The descent build is launch-de-risked on the host-residency
axis **iff** the relocate is vectorized/paged **and** the accept-walk is sync-free. A host-loop relocate OR
a sync-bound walk is the only way the step collapses, and both are already classified with greedy-safe designs.

**Feeds:**
- **fern #155 consolidator:** `net_clear_500_bar_descent_only = 4.8241` — the realizable build's operative bar.
- **lawine #161 spine cost:** both-bugs step slot ARMED (`net_clear_500_bar_both_bugs = 4.8241`); fold
  #161's `both_bugs_step_delta_pct` when it lands.

---

## What happened

The sweep confirms — statically *and* empirically — that #154 and #157 are the **only two** host-resident
landmines on the descent decode path, and that with both greedy-safe recoveries applied the full path nets
**1.2086 units**, *below* the measured 1.2182 anchor. The empirical capturability probe is the load-bearing
new evidence: it shows on real hardware that the device-resident designs capture and the host round-trips
break capture with the exact CUDA errors the taxonomy predicts — closing the gap that pure sync-counting
(#147) leaves open (a host Python loop can break capture without registering as a sync).

This does **not** authorize a launch. It de-risks the *host-residency axis* of the land #71 build and hands
off the two ops that must be device-resident, each with a greedy-safe design and an equivalence guarantee.

## Suggested follow-ups

- **Validate the recoveries in the actual land #71 build** once it exists: re-run this sweep against the
  real descent kernel source (not the modeled inventory) to confirm `descent_path_residual_host_ops_count`
  stays 0 against shipped code.
- **Fold lawine #161** `both_bugs_step_delta_pct` into the armed slot (`PR161_BOTH_BUGS_STEP_DELTA_PCT`) to
  finalize the both-bugs bar; today it is identical to descent-only (BUG-1 is a numerator fix).
- **Micro-confirm the paged slot-map relocate** (zero-copy, 20.26 µs/step → bar 4.817) as the relocate
  implementation of choice over the fused-gather (35.3 µs/step) if block-table plumbing allows — it is the
  cheaper of the two greedy-safe designs.
