STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["5a6zq2yz"],"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"byte_identical_tax_floor_tps":14.38998445612367,"tax_decomp_fp32_accum_tps":0.0,"tax_decomp_serialization_tps":14.38998445612367,"tax_decomp_lost_batch_parallel_tps":0.0,"removable_tax_tps":0.0,"floor_tax_tps":14.38998445612367,"current_pin_gap_to_floor_tps":0.0,"knife_edge_margin_if_floor_reached_tps":1.2100155438763522,"byte_identical_floor_self_test_passes":true,"primary_metric":{"name":"byte_identical_tax_floor_tps","value":14.38998445612367},"test_metric":{"name":"byte_identical_floor_self_test_passes","value":1.0}}

## Results

**Analysis-only card** (`analysis_only=True, no_hf_job=True, no_served_file_change=True, official_tps=0`). No GPU, no build, no served-file change. All numbers are derived from merged artifacts on `approval-gated-8gpu-20260613`; the module re-loads each JSON and cross-checks the pinned constants in its self-test (59/59).

### Headline

The **14.39-TPS blanket-strict tax** (481.53 → 467.14, #412 measured) is a **single-component, byte-identity FLOOR**. It does not decompose into the three costs the PR hypothesized — it is **one** cost: single-segment serialization from M=1 draft-lane attention occupancy collapse. The cheapest byte-exact pin (#393 `fa2_unpack_ns1`) already **sits at the floor** (gap = 0). `removable_tax_tps = 0`. The only way under the floor is a **new-reference re-capture** (#400 pinned-K), which is greedy/PPL-valid but **NOT** byte-identical to today's served output — a contract decision for humans, not a byte-identical win.

### Tax decomposition (instruction 2)

| Component | TPS | Classification | Why |
|---|---|---|---|
| (a) fp32-accumulate overhead | **0.00** | n/a | FA fp32-accumulates on **both** the deployed split-K heuristic and the byte-exact un-pack. The byte break is reduction **ORDER**, not precision — flipping to fp32 changes nothing because it's already fp32. |
| (b) single-segment serialization | **14.39** | **FLOOR** | The whole tax. Forcing `num_splits=1` (un-pack) on the **M=1 draft-lane** attention collapses occupancy 88% → 10% (#400), far below the ~2% HBM-bandwidth floor → latency-bound, not bandwidth-bound. |
| (c) lost batch-parallelism | **0.00** | n/a | Blanket-strict has no per-step flagging, so nothing is serialized that could have been batched. The selective-peel alternative that *would* reclaim batching is net-**NEGATIVE −83 TPS** (#412: 384.11 < 467.14). |
| (d) launch/sync of flagged segment | **0.00** | n/a | Named as the 4th: there is no separable launch/sync component — un-pack is a single kernel. |
| **closure residual** | **0.00** | — | Decomposition closes exactly. |

**Correction surfaced to the advisor:** the PR frames the tax as living "over the M=8 verify body," but the merged budget files (#393/#400/#408) place it on the **M=1 draft lane**. #393 measured the **M=8 verify as `verify_penalty_free=True`** (num_splits-free), and #400/#408 locate the occupancy collapse at M=1 (`m1_attn_occupancy_frac=0.10`). The tax is real and exactly 14.39; it just lives one lane over from where the card's prose pointed. Decomposition and floor bound are unaffected.

### Floor bound (instruction 3/4) — PRIMARY

```
byte_identical_tax_floor_tps = 14.390   (the whole tax is irreducible under frozen byte-identity)
removable_tax_tps            =  0.000
current_pin_gap_to_floor_tps =  0.000   (#393 fa2_unpack_ns1 IS the floor)
```

**The gate is my #418 knife edge applied to split-K reassociation.** Recovering serialization means re-parallelizing the reduction (split-K / pinned-K), which reassociates a floating-point sum. The reassociation perturbation is **0.125 nat = 1 bf16-ULP = eps\*** exactly (the near-tie margin), so there is **zero proof margin** (`perturb ≥ eps*`, `has_proof_margin=False`), and the 40 near-ties **blanket all 7 chain positions** — no row is provably safe to re-parallelize. #400's own `new_reference_probe` independently **measured** `multisplit_eq_serial_bytes = False`. So a parallel reduction **flips** near-ties vs the frozen reference ⇒ **not byte-identical** ⇒ serialization is a genuine FLOOR, not removable.

### The only sub-floor path (instruction 5 — DO NOT build)

`served_change_to_approach_floor`: **NONE preserves byte-identity to the current served reference — this lever is CLOSED on the byte-identical side.** The single physical path under the floor is wirbel #400's deterministic **64-CTA `num_splits=8` pinned-K** split-reduce on the FA2 decode-attention kernel:

- **What it recovers:** ~13.998 TPS realistic (occupancy 10% → ~88%), residual floor only 0.184 TPS (98.7% of roofline).
- **Why it's legal-but-different:** it is **M-invariant byte-exact-feasible** (M=1 == M=8 under the *new* kernel; Marlin `atomic_add=False` / `fp32_reduce` / fixed-order grounds the M-invariance) and PPL-neutral — but it produces a **NEW reference**: #400 measured `multisplit_eq_serial_bytes=False`, and my #418 knife edge proves it flips ~O(3/882) near-ties vs **today's** served bytes.
- **Kernel surface / blast radius:** `num_splits>1` is `NotImplementedError` on shipped FA2 ⇒ a **kernel REBUILD** of the hottest decode kernel; a flagged served-file change. Identity-verify cost: a full greedy + PPL re-capture on the new reference (must re-clear ≤ 2.42).
- **Verdict:** greedy/PPL-**VALID** (a legal strictly-equivalent submission in the #407 M-invariance sense; stack → ~496.7 TPS, lawine #411 ceiling 497.44) but **NOT frozen-byte-identical**. This is a **CONTRACT decision** (frozen-byte vs M-invariant reference) for **human approval**. Recommend surfacing it as a re-capture proposal; **do not auto-build**.

### Stack arithmetic

| Config | TPS | Reference | Margin vs 481.53 |
|---|---|---|---|
| deployed (non-strict, #52) | 481.53 | today's bytes | — |
| blanket-strict measured (#412) | 467.14 ± 0.16 | today's bytes | −14.39 (the tax) |
| **frozen stack** = strict + cb3 supply (#403 +15.60) | **482.74** | today's bytes | **+1.21** (`knife_edge_margin_if_floor_reached_tps`) |
| re-capture stack = pinned-K + cb3 (#400/#411) | ~496.74 | **NEW** reference | +15.21 (NEW ref, human-gated) |

Even at the byte-identical floor, the **cb3 supply lever alone clears the deployed frontier by +1.21 TPS** with identity preserved. The 500 target is not reachable on the byte-identical side (margin +1.21); it needs the re-capture contract decision (→ ~496.7, still short of 500 by ~3.3) **plus** more supply.

### Reproduce

```bash
cd target/ && .venv/bin/python -m research.validity.byte_identical_reduction_tax_floor.byte_identical_reduction_tax_floor --self-test
cd target/ && .venv/bin/python -m research.validity.byte_identical_reduction_tax_floor.byte_identical_reduction_tax_floor \
    --wandb_group cb3-tax-floor --wandb_name denken/byte-identical-reduction-tax-floor
```

- **Self-test:** 59/59 checks pass (`byte_identical_floor_self_test_passes=True`), including 11 artifact-provenance cross-checks (#393/#400/#408/#412/#403 pinned constants re-loaded and verified against the merged JSONs).
- **Peak memory:** 13.7 MiB (pure analysis; no model load, no GPU).
- **W&B run ID:** `5a6zq2yz` (group `cb3-tax-floor`).
- **PPL:** unchanged 2.3772 ≤ 2.42 (no served-output change).

### Public evidence used (all on `approval-gated-8gpu-20260613`)

- **#412** (selective_recompute_equivalent_tps): blanket-strict measured **467.14 ± 0.16**, selective net-negative (384.11), `fastest_realizable_strict_config=blanket_strict` — anchors the 14.39 tax and kills component (c).
- **#393** (attention_strict_pin_cost, `0q7ynumg`): `cheapest_strict_attn_backend=fa2_unpack_ns1`, `n_byte_exact_attn_configs=1`, `verify_penalty_free=True` — the pin that sits at the floor; locates tax off the M=8 verify.
- **#400** (attn_pinnedk_headroom, `o7yhpkej`): M=1 occupancy 0.10 vs heuristic 0.88, `pinnedk_produces_new_reference=True`, measured `multisplit_eq_serial_bytes=False` — physical cause + the new-reference proof.
- **#408** (m1_decode_latency_budget, `qc9bz8sv`): `t_attn_frac=0.0951`, `attn_lever_gain_realistic=13.998` — the recoverable magnitude.
- **#405 / #418** (`uc7jg6vs`): eps\*=0.125 = reduction-order perturb 0.125, 40 near-ties blanketing all 7 positions — the knife-edge gate that makes serialization a floor.
- **#403** (cb3_conservative_k_deployable_lift, `iv9i2wks`): supply +15.60 at k\*=229 — the frozen-stack +1.21 margin.

### What happened

The hypothesis was that the 14.4-TPS tax splits into fp32-accum + serialization + lost-batch-parallelism. **It doesn't** — it's pure serialization (M=1 occupancy collapse), and that serialization is **irreducible under frozen byte-identity** because re-parallelizing the reduction reassociates a sum at exactly the eps\* knife edge (zero margin, near-ties blanket all positions). So `byte_identical_tax_floor_tps = 14.39`, the #393 pin is already AT the floor (gap 0), and `removable_tax_tps = 0`. The lever is **closed** on the byte-identical side. The +14 TPS is physically real but only accessible behind a **new-reference re-capture** (#400 pinned-K) — M-invariant and PPL-valid, but it flips ~3/882 near-ties vs today's bytes, making it a human contract decision rather than a byte-identical reduction. This closes the last tax-reduction lever named in my #418.

### Suggested follow-ups

1. **Re-capture contract proposal (human-gated):** if the team accepts an M-invariant new reference (frozen-byte → M-invariant), the #400 pinned-K rebuild → ~496.7 stacked TPS is the highest legal strictly-equivalent config. Needs a full greedy+PPL re-validation on the new bytes and a flagged served-file change — exactly the call the #407 re-scope was set up to make.
2. **Quantify the residual 3.3-TPS gap to 500** under the re-capture reference: pinned-K (~496.7) + cb3 (#403) still misses 500 by ~3.3; pair with the #411 supply ledger (ceiling 497.44) to see whether any *additional* equivalence-neutral supply closes it without a demand-side a1 break (#308 showed a1≈0.92 is out of reach).
3. **No further byte-identical attention work:** with the pin at the floor and the knife edge forbidding split-K, there is no remaining byte-identical attention lever to probe. Future TPS on the frozen reference must come from the supply side (cb3-style), not from reducing the strict-attention tax.
