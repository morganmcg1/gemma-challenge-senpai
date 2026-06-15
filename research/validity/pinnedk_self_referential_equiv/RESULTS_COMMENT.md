STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["132fgkbk"],"analysis_only":true,"no_hf_job":true,"no_served_file_change":true,"official_tps":0,"verdict":"legal_self_referential","pinnedk_self_referential_equivalent":true,"m_invariance_sufficient_for_self_referential":true,"pinnedk_recapture_stack_tps":496.7386162499593,"gap_to_500_tps":3.261383750040693,"gap_to_500_from_ceiling_411_tps":2.56,"deployed_passes_self_referential":false,"pinnedk_more_equivalent_than_deployed":true,"self_referential_frontier_tps":496.7386162499593,"frozen_byte_frontier_tps":482.7400155438763,"self_test_passes":true,"primary_metric":{"name":"pinnedk_recapture_stack_tps","value":496.7386162499593},"test_metric":{"name":"self_test_passes","value":1.0}}

## Results

**Analysis-only reconciliation card** (`analysis_only=True, no_hf_job=True, no_served_file_change=True, official_tps=0`). No GPU, no build, no served-file change, no submission. Every number is composed from MERGED artifacts on `approval-gated-8gpu-20260613`; the module re-loads each JSON and cross-checks the pinned constants in its self-test (52/52).

### Headline

**Pinned-K re-capture IS self-referentially equivalent — `verdict = legal_self_referential`.** My #423 conservatively filed pinned-K as a human "frozen-byte vs M-invariant" CONTRACT decision, assuming #407 equivalence meant byte-identity to the deployed 481.53 bytes. **That assumption is wrong.** The operative #407 gate is **SELF-REFERENTIAL** — it tests the submission's M=8 verify against **the submission's OWN M=1 AR greedy** (`SENPAI_REFERENCE_MODE` clears `SPECULATIVE_CONFIG` → spec-OFF M=1 on its own kernels/quant; #114 Step 1a mechanism, #414/#420 verdict), **NOT** byte-identity to today's served bytes. Pinned-K is M-invariant byte-exact (#400: M=1==M=8 under the rebuilt fixed-order `num_splits=8` kernel), so **its M=8 verify reproduces its own M=1 AR exactly ⇒ self-referential identity 1.0 ⇒ M-invariance is SUFFICIENT for equivalence.** The realizable fastest-equivalent frontier lifts **482.74 → 496.74**.

### The reconciliation twist (the part that flips #423)

The ~3/882 token flips that #423 used to gate pinned-K are flips **versus TODAY's bytes** — a *frozen-byte* property, **irrelevant** to a self-referential reference. And here is the decisive twist:

> **The deployed 481.53 config ITSELF fails the self-referential gate on the attention dimension** (self-ref identity **0.9966**, 3/882 M=8-vs-own-M=1 reduction-order flips, #114/#423). Pinned-K (identity **1.0**) is **MORE** self-referentially-equivalent than deployed.

So the self-referential gate **blesses pinned-K, not the reverse**. The reference that pinned-K must match is its own M=1 AR — which pinned-K's M-invariance makes byte-exact by construction. Re-using the 3/882 frozen-byte flips to *block* pinned-K would, under the self-referential gate, also have to block the **deployed** config — which is incoherent.

### Instruction 1 — the two equivalence notions, precisely

| Notion | Reference | pinned-K | deployed |
|---|---|---|---|
| **(a) self-referential** [OPERATIVE] | submission's OWN M=1 AR, re-run per submission (`SENPAI_REFERENCE_MODE` spec-OFF) | **compliant, identity 1.0** | **NOT compliant, identity 0.9966** |
| (b) frozen-byte [#423 conservative] | the currently-deployed 481.53 served bytes (fixed) | NOT compliant (`multisplit≠serial`, ~3/882) | trivially compliant |

Pinned-K is **(a)-compliant and (b)-non-compliant**. The whole question is *which notion is the operative #407 gate* — instruction 2.

### Instruction 2 — resolve the binding question from #414's scorer mechanics

```
scorer_is_self_referential               = True   (reference = own M=1 AR; #114 Step 1a + #414/#420)
official_harness_enforces_token_identity  = False  (#114 Step 1c: only TPS + 128/128 completion + PPL <= 2.42)
pinnedk_m_invariant                       = True   (#400: M=1==M=8, Marlin atomic_add=False/fp32_reduce/fixed-order)
=> m_invariance_SUFFICIENT_for_self_referential = True
=> pinnedk_self_referential_equivalent          = True
TWIST: deployed_passes_self_referential_attention = False (0.9966) ; pinnedk_more_equivalent_than_deployed = True
```

Three merged sources resolve this **decisively and consistently**, all responding to the same human #407 re-scope:
- **#114 Step 1a** (`self_referential_greedy_gate.md`): the gate is self-referential **by mechanism** — `SENPAI_REFERENCE_MODE=1` → spec-OFF M=1 AR on the submission's own engine. Step 1c: the **official harness runs NO token-identity check** ("128/128" is a completion count, not a greedy check) — only TPS + completion + PPL.
- **#414** (`bq7xkfcv`): `gate_for_respect_equivalence = "self_referential"` — "vs the submission's OWN truncated-head greedy AR, the official scorer's operative gate".
- **#420**: reaffirms self-referential; in-keepset drafter preserves it (`inkeepset_drafter_preserves_self_referential = True`).

The evidence is **not ambiguous**, so the verdict is the `legal_self_referential` branch, not `ambiguous_flag_to_human`.

**Honest caveat (kept explicit so the call isn't overstated):** "self-referential" is the operative reading of the **written** contract (program.md:27-28) + the local enforcement tooling + the advisor's own landed #414/#420 cards — it is **not** an automated leaderboard check (#114 Step 1c: the official harness computes only TPS + 128/128 completion + PPL, no greedy/token-identity comparison at all). So the equivalence QUESTION is resolved YES *under the written self-referential contract*, but nothing automated will *enforce* it on submission — which is exactly why realizing pinned-K stays a **human deployment approval** (instruction 4), not a contract re-litigation. This is a cleaner, narrower ask than #423's "frozen-byte vs M-invariant contract decision": the contract question is settled; only the rebuild+re-validation needs sign-off.

### Instruction 3 — price the re-capture stack + gap to 500

| Config | TPS | Reference | Frontier role |
|---|---|---|---|
| deployed (non-strict, #52) | 481.53 | own M=1 AR @ 0.9966 (**fails self-ref**) | today's bytes |
| frozen-byte frontier (#423) | 482.74 | today's bytes (strict + cb3) | +1.21 conservative |
| blanket-strict base (#412 measured) | 467.14 | num_splits=1 un-pack | stack base |
| **pinned-K re-capture stack** | **496.74** | **own M=1 AR @ 1.0 (self-ref LEGAL)** | **self-ref frontier** |
| lawine #411 ceiling | 497.44 | — | supply ceiling |

```
496.74 = blanket-strict 467.14 + pinned-K attn 13.998 (#408) + cb3 supply 15.60 (#403)
gap_to_500_tps               = 3.26   (stack)
gap_to_500_from_ceiling_411  = 2.56   (#411 ceiling)
self_ref uplift over frozen  = +13.998   over deployed = +15.21
```

The stack **misses 500 by 3.26** (2.56 from the #411 ceiling). Pinned-K alone caps at ~deployed speed (useful only *stacked*, #411); cb3+pinned-K still misses. Closing the last ~3 TPS needs additional equivalence-neutral **supply** — the demand-side a1≈0.92 break is out of reach (#308).

### Instruction 4 — the decision card (GO-to-human packet)

**The equivalence QUESTION is resolved YES.** What remains is a **DEPLOYMENT approval, NOT a contract decision** — the ONE allowed flagged ask:

1. **FLAGGED FA2 decode-kernel REBUILD** — `num_splits>1` is `NotImplementedError` on shipped FA2, so the fixed 64-CTA `num_splits=8` split-reduce needs a kernel rebuild (#400/#411: 5 files, **all additive, 0 in-place, NO checkpoint change** — a weightless reduction-order change to the hottest decode kernel).
2. **IDENTITY-VERIFY on the NEW served bytes** — a `SENPAI_REFERENCE_MODE` A/B (spec-ON M=8 vs own spec-OFF M=1 AR) must return GREEDY_IDENTICAL (self-ref identity 1.0), confirming the rebuilt kernel is genuinely M-invariant on-target. Cross-ref #411's deploy-surface ledger (one shared e2e self-referential capture re-keys all stacked levers at once) / #419's CI.
3. **PPL re-clear** — the new bytes must re-measure PPL ≤ 2.42 (expected neutral: a reduction-order change; PR #66 shows greedy near-tie flips don't enter the teacher-forced PPL).

Blast radius: a flagged served-file change; realistic vs floor recovery 98.7% vs 100% roofline. **PRIZE:** lifts the realizable fastest-EQUIVALENT frontier 482.74 → 496.74 (+13.99 with self-referential identity 1.0, which the deployed 481.53 LACKS at 0.9966). **Recommend:** surface as a self-referentially-legal re-capture GO request; do **NOT** auto-build (the rebuild + re-validation is the human-gated step).

### Reproduce

```bash
cd target/ && .venv/bin/python -m research.validity.pinnedk_self_referential_equiv.pinnedk_self_referential_equiv --self-test
cd target/ && .venv/bin/python -m research.validity.pinnedk_self_referential_equiv.pinnedk_self_referential_equiv \
    --wandb_group pinnedk-self-ref --wandb_name denken/pinnedk-self-referential-equiv
```

- **Self-test:** 52/52 checks pass (`self_test_passes=True`), including artifact-provenance cross-checks (#414/#420/#423 pinned constants re-loaded and verified against the merged JSONs) and all 3 verdict branches exercised (`legal_self_referential` / `human_contract_decision` / `ambiguous_flag_to_human`).
- **Peak memory:** 13.7 MiB (pure analysis; no model load, no GPU).
- **W&B run ID:** `132fgkbk` (group `pinnedk-self-ref`).
- **PPL:** unchanged 2.3772 ≤ 2.42 (no served-output change in this card; the GO packet's rebuild carries a re-validation step).

### Public evidence used (all on `approval-gated-8gpu-20260613`)

- **#114** (`self_referential_greedy_gate.md`): Step 1a — gate is self-referential by mechanism (`SENPAI_REFERENCE_MODE` → spec-OFF M=1 AR on own kernels); Step 1c — official harness runs NO greedy check. The decisive proof the gate is self-referential.
- **#414** (`bq7xkfcv`, truevocab_lmhead_equivalence_cost): `gate_for_respect_equivalence="self_referential"` — establishes the operative gate responding to the human #407 issue.
- **#420** (speculator_keepset_equivalence): reaffirms self-referential; in-keepset drafter preserves it.
- **#400** (`o7yhpkej`, attn_pinnedk_headroom): pinned-K M-invariant byte-exact-feasible (M=1==M=8); `multisplit_eq_serial_bytes=False` vs deployed serial; `attn_rebuild_is_flagged_served_change=True`.
- **#408** (`qc9bz8sv`, m1_decode_latency_budget): `attn_lever_gain_realistic=13.998` — the recoverable magnitude.
- **#403** (`iv9i2wks`, cb3_conservative_k_deployable_lift): supply +15.60 at k*=229.
- **#411** (`078yjgax`, flagged_supply_deploy_surface_ledger): ceiling 497.44; pinned-K caps at deployed alone, useful only stacked.
- **#412** (selective_recompute_equivalent_tps): blanket-strict measured base 467.14.
- **#423** (`5a6zq2yz`, byte_identical_reduction_tax_floor): the predecessor that filed pinned-K as a frozen-byte contract decision — now reconciled.

### What happened

#423's verdict ("human contract decision") rested on a **single unstated assumption**: that #407 equivalence means byte-identity to the deployed 481.53 bytes. The merged scorer-mechanics cards (#114 mechanism, #414 operative-gate, #420 reaffirmation) show that assumption is **false** — the gate is self-referential (own M=1 AR), and the official harness enforces no token-identity at all. Under the correct (self-referential) reading, pinned-K's M-invariance (M=1==M=8) is **exactly** what the gate requires, so pinned-K is a **legal** #407-equivalent submission. The 3/882 flips #423 leaned on are a frozen-byte artifact — and the same artifact makes the **deployed** config *fail* the self-referential gate (0.9966), so pinned-K is strictly *more* equivalent than what ships today. The equivalence question is resolved **YES**; what remains is a deployment approval (flagged kernel rebuild + A/B identity verify + PPL re-clear), lifting the realizable equivalent frontier 482.74 → 496.74. Still 3.26 short of 500 — the residual is a supply-side gap, not an equivalence one.

### Suggested follow-ups

1. **Re-capture GO request (human deployment approval):** package the #400 pinned-K rebuild as a self-referentially-legal re-capture — flagged FA2 decode-kernel rebuild + `SENPAI_REFERENCE_MODE` A/B identity verify (must return GREEDY_IDENTICAL on the new bytes) + PPL re-clear ≤ 2.42. This is the highest realizable strictly-equivalent config (~496.74). Pair with #411/#419's shared-capture deploy surface so all stacked levers re-key at once.
2. **Close the residual 3.26-TPS gap to 500** under the self-referential frontier: pinned-K (496.74) + cb3 still misses by 3.26 (2.56 from the #411 ceiling 497.44). Audit whether any *additional* equivalence-neutral supply lever (cb3-style, k>229) closes it without a demand-side a1≈0.92 break (#308 shows a1 is out of reach).
3. **No further byte-identical attention work:** with the equivalence question resolved on the self-referential side, the frozen-byte framing of #423 is superseded for the operative gate. Future TPS work should treat the self-referential frontier (496.74) as the realizable target, not the frozen-byte 482.74.
