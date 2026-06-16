STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["3ohaod6u"],"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"bit_identical_supply_ceiling_tps":483.67844715271747,"bit_identical_supply_ceiling_floor_tps":482.7400155438763,"n_bit_identical_levers":1,"verify_sdpa_numstages_realized_tps":0.9384316088411424,"verify_sdpa_numstages_realized_tps_floor":0.0,"safe_frontier_beats_deployed_481":true,"self_test_passes":true,"primary_metric":{"name":"bit_identical_supply_ceiling_tps","value":483.67844715271747},"test_metric":{"name":"bit_identical_supply_ceiling_self_test_passes","value":1.0}}

## Results

**Analysis-only card** (`analysis_only=True, no_hf_job=True, no_served_file_change=True, official_tps=0`). No GPU, no build, no served-file change. Every number is re-priced from merged artifacts on `approval-gated-8gpu-20260613`; the module re-loads each JSON and cross-checks the pinned anchors in its self-test (**37/37**, incl. **12 artifact-provenance cross-checks**). The deployed senpai vLLM venv is not present in this launch env, so the verify-SDPA kernel numbers are **consumed from my own #279 local-A10G run** (`results.json`) exactly as instruction 1 sanctions ("consume/extend wirbel #279's measurement").

### Headline

The **SAFE bit-identical supply ceiling is a band `[482.74, 483.68]`** — the frozen floor (blanket-strict 467.14 #412 + cb3 +15.60 #403) **plus at most +0.94 TPS** from the one delivering bit-identical lever (verify-SDPA `num_stages 3→2`), and **possibly +0** once you price in-graph realism. **Either endpoint clears the deployed 481.53** (cb3 alone already does, +1.21) but **sits ~13.8–14.7 TPS below lawine #411's 497.44** — and that whole gap is the **reference-changing pinned-K** (+14.29, denken #427). **The bit-identical contribution to closing it is 0.** So the answer to the PR's question is the **decision-critical null**: the bit-identical-only path is **EXHAUSTED at cb3**; 497.44 / 500 is **only** reachable via denken's reference contract or stark's tie-break — the riskier levers are **necessary, not nice-to-have.**

### 1. Lever enumeration (instruction 1) — every maxdiff=0.0 supply lever above cb3

| Lever | PR | bit-identical | above cb3 | delivers | classification |
|---|---|:--:|:--:|:--:|---|
| **verify-SDPA `num_stages 3→2`** | #270/#279 | **True** (md=0.0 both shapes) | **True** | **YES** | the **only** bit-identical lever with nonzero supply above cb3 |
| int4-Marlin body GEMMs ×7 (q/k/v/o/gate/up/down) | #390 | True | False | no | **already byte-exact** at decode → 0 incremental (inside the floor) |
| int4-Marlin lm_head | #384 | True | False | no | **already byte-exact** (`deterministic_lmhead_recovers_deficit_tps=0.0`) → 0 incremental |
| pinned-K split-K reassociation | #400/#423/#427 | **False** | True | EXCLUDED | flips ~3/882 near-ties vs frozen ref (`multisplit_eq_serial_bytes=False`) → **reference-changing, denken #427's lane (+14.29)** |
| batch-invariant verify GEMM | #363 | **False** (md=9.77e-4) | True | EXCLUDED | new reference |
| drafter loopgraph fusion | #424 | **False** | True | EXCLUDED | **NO-GO (my #424)** |

**`n_bit_identical_levers = 1`.** The body GEMMs and lm_head are already strict, so they contribute zero *incremental* lift (they're already baked into the 482.74 floor). The reference-changing levers are explicitly out of this card's lane.

### 2. Re-pricing the verify-SDPA lever (instruction 2 — price realism, not roofline)

#279 measured `num_stages 3→2` on the **deployed 3D split-KV M=8 verify** shapes, **bit-identical on both**:

| shape | speedup (num_stages 3→2) | maxdiff | greedy 128-gate |
|---|---|:--:|:--:|
| global head-512 (×7 layers) | 1.018× | **0.0** | divergent=0 |
| sliding head-256 (×14 layers) | 1.093× | **0.0** | divergent=0 |

→ absolute bit-identical SDPA saving **15.55 µs/step** at the realistic decode ctx≈512 (sum over the 21 tunable TRITON_ATTN layers). *(Correction surfaced: the PR's "+1.097× at M=8" is the sliding head-256 shape; the global head-512 shape is only +1.8%. The step-level absolute saving blends both — bit-identity holds on each.)*

**The realism correction (the dominant move):** #279 priced this against a **STEP_US=1218.2 µs composition reference** → +1.293% / 487.76 TPS. But the **REAL wall-clock decode step is 8017 µs** (#284, directly CUDA-event measured: verify 6532 µs = 81.5%, drafter 1445 µs, host 40 µs; 99.5% GPU-bound). The verify body is on the critical path and is 81.5% of the step, so a verify-body saving comes off the step ~1:1. Re-basing the **absolute 15.55 µs** onto the real 8017 µs step:

```
roofline_gain = 15.55 / 8017          = +0.194%   (vs #279's step-inflated +1.293% → 6.6× HAIRCUT)
              = +0.938 TPS on the 482.74 floor  → roofline ceiling 483.68
```

**A second, unmeasurable-here discount remains.** #273 (static-K wall-clock A/B on this exact ONEGRAPH stack) measured a standalone/composition-predicted saving realizing at ratio **−2.02** — i.e. **negative**. The verify-SDPA lever is bit-identical and **E[T]-neutral**, so it lacks static-K's structural E[T] trade that drove that negative ratio; I therefore clamp the realization **floor at 0** (not −2.02 — applying it literally to a pure latency cut on the critical path would be nonsensical), not below. Confirming the in-graph (ONEGRAPH) realization needs a served-kernel-config A/B we do **not** build. Hence the realized contribution is a **BAND:**

```
verify_sdpa_numstages_realized_tps ∈ [0.0 (#273-cautionary floor), +0.938 (roofline UB)]
```

### 3. Verdict — the safe ceiling vs lawine #411 (instruction 3)

| config | TPS | reference | note |
|---|---|---|---|
| deployed (#52) | 481.53 | today's bytes | non-equivalent (3/882 M=8 flips) |
| blanket-strict (#412) | 467.14 | today's bytes | the −14.39 strict tax |
| **frozen floor = + cb3 (#403)** | **482.74** | today's bytes | **+1.21 identity-safe** (cb3 alone clears deployed) |
| **bit-id ceiling = + verify-SDPA roofline** | **483.68** | today's bytes (**bit-identical**) | **SAFE ceiling UB**; realized band → [482.74, 483.68] |
| lawine #411 (+ pinned-K) | 497.44 | **NEW** reference | **reference-changing** (human-gated) |

```
bit_identical_supply_ceiling_tps      = 483.68   (roofline UB)
bit_identical_supply_ceiling_floor_tps = 482.74   (realization floor; = cb3-only)
safe_frontier_beats_deployed_481       = True     (via cb3 alone, +1.21)
gap to lawine 497.44                    = 13.76–14.70 TPS  ≈ pinned-K +14.29 (REFERENCE-CHANGING)
```

The gap from the frozen floor to lawine's ceiling (14.70 TPS) is, to within 0.41 TPS, **exactly the reference-changing pinned-K lever** (+14.29). **No bit-identical lever closes any of it.** The safe frontier is exhausted at cb3 + a ≤+0.94 verify-SDPA crumb. **Crucially, the decision-critical conclusion is band-invariant:** whether the crumb realizes at 0 or +0.94, the safe ceiling clears 481.53 and falls ~14 TPS short of 497.44/500.

### 4. Deploy surface (instruction 4)

`verify-SDPA num_stages=2` is a **served-kernel-config change** at the deployed TRITON_ATTN `kernel_unified_attention` launch site (forcing `num_stages=2` on a bare `@triton.jit` launched at Triton defaults num_warps=4/num_stages=3) — a **FLAGGED served-file change**. The *question* (how much TPS, is it bit-identical → ≤+0.94 TPS, maxdiff=0.0) is fully answered **in-envelope**; only the **BUILD** is the flagged ask. **Flag it, do NOT build.**

### Reproduce

```bash
cd target/ && .venv/bin/python -m research.validity.bit_identical_supply_ceiling.bit_identical_supply_ceiling --self-test
cd target/ && .venv/bin/python -m research.validity.bit_identical_supply_ceiling.bit_identical_supply_ceiling \
    --wandb_group bit-id-supply-ceiling --wandb_name wirbel/bit-identical-supply-ceiling
```

- **Self-test:** 37/37 (`bit_identical_supply_ceiling_self_test_passes=True`), incl. 12 artifact-provenance cross-checks (#412/#403/#411/#284/#273/#423/#279 pinned constants re-loaded and verified against the merged JSONs).
- **Peak memory:** 12.2 MiB (pure analysis; no model load, no GPU).
- **W&B run ID:** `3ohaod6u` (group `bit-id-supply-ceiling`).
- **PPL:** unchanged 2.3772 ≤ 2.42 (bit-identical ⇒ no served-output change).

### Public evidence used (all on `approval-gated-8gpu-20260613`)

- **#270** (draft_attn_triton_autotune, `iwwcmvez`): VERIFY-side num_stages 3→2 is real and **bit-identical** (maxdiff=0.0) on head-512 SDPA.
- **#279** (verify_sdpa_linear_deploy, MY card): the consumed local-A10G measurement — `verify_sdpa_saving_us=15.55`, both shapes maxdiff=0.0, greedy-identical.
- **#284** (decode_host_overhead): the REAL 8017 µs decode step (verify 6532 = 81.5%, 99.5% GPU-bound) — the realism re-basing denominator.
- **#273** (static_k_wallclock_ab): `realization_ratio(K=4)=−2.02` — the negative-realization precedent that bands the contribution down to [0, roofline].
- **#412** (selective_recompute_equivalent_tps): blanket-strict measured **467.14** — the floor base.
- **#403** (cb3_conservative_k_deployable_lift, `iv9i2wks`): cb3 supply **+15.60** at k*=229 — the only banked equivalence-neutral lever.
- **#423** (byte_identical_reduction_tax_floor, MY card): frozen stack **482.74**, +1.21 knife-edge margin, `removable_tax_tps=0`.
- **#411** (flagged_supply_deploy_surface_ledger): supply-ledger ceiling **497.44** incl. reference-changing pinned-K (+14.29).

### What happened

The hypothesis was that the safe equivalence-respecting frontier might extend meaningfully above 482.74 using only bit-identical levers. **It doesn't.** A full scan of the merged record finds **exactly one** bit-identical lever with nonzero supply above cb3 — the verify-SDPA `num_stages 3→2` tune — and once its #279 saving is **re-priced from the composition step (1218.2 µs) onto the real wall-clock step (8017 µs, #284)** it collapses **6.6×** to **+0.194% / +0.94 TPS roofline**, and #273's negative-realization precedent bands the realized contribution down to **[0, +0.94]**. Everything else is either already byte-exact (body GEMMs, lm_head → zero incremental) or reference-changing (pinned-K, batch-invariant) or a NO-GO (drafter loopgraph). So `bit_identical_supply_ceiling_tps = 483.68` (UB) / 482.74 (floor): it clears the deployed 481.53 but lands **~14 TPS short of lawine's 497.44**, and that **entire** gap is the reference-changing pinned-K. **The bit-identical-only path is exhausted at cb3.** This is the decision-critical null the PR asked for: reaching 497.44/500 **requires** denken #427's reference contract or stark #421's tie-break — the riskier levers are *necessary*, not merely nice-to-have.

### Suggested follow-ups

1. **If +0.94 TPS is worth a flagged build:** a served-kernel-config A/B (force `num_stages=2` at the `kernel_unified_attention` launch) would collapse the [0, 0.94] band to a measured point and confirm in-graph (ONEGRAPH) realization — the only open uncertainty in this card. It's bit-identical (maxdiff=0.0 already proven by #279), so PPL/greedy are safe by construction; the only cost is the flagged served-file change. Low priority given the ≤+0.94 ceiling.
2. **The decision now lives entirely in the reference-contract lane.** With the bit-identical side exhausted, the next TPS must come from denken #427 (pinned-K → ~496.7, NEW reference, human-gated greedy+PPL re-capture) or stark #421 (tie-break). Recommend the advisor treat 482.74 as the *safe* frontier and route any push toward 497.44/500 through those two contract decisions.
3. **No further bit-identical supply work.** With one delivering lever priced at ≤+0.94 and everything else already-strict or reference-changing, there is no remaining bit-identical supply lever to probe. Future frontier gains on the frozen reference are closed on the bit-identical side.
