# PR #189 — Executable fail-closed MUST-RETAIN submission gate: auto-catch the 85%-cost relocate-host-loop trap before the irreversible shot

**Verdict: SHIPS the enforcement.** ubel #186 proved that ONE dropped flag — the
`relocate_salvaged_kv` device-vectorization reverting to a 37-layer host Python
loop — silently costs **85% of projected throughput (516→77 TPS)** with no
submit-time warning. But #186 is a *document*. This PR converts it into an
**executable, fail-closed packaging gate**:

```python
verify_submission_gate(build_env, build_introspection)
  -> {packaging_verdict: GO|NO-GO, failing_rows[], per_row_assertions[],
      validity_class_failures[], binding_failure, ...}
```

It imports the #186 manifest JSON as the **source of rows + costs** (it does NOT
re-derive any cost), walks all **22** enumerated flags, asserts each of the **19**
MUST-RETAIN rows is present/correct and the **TRAP** (`LSK_SKIP_LAYERS`) is UNSET,
and is **fail-closed**: any MUST-RETAIN FAIL, a present TRAP, or
missing/unparseable introspection for a row → **NO-GO**, never silent-pass. On
any violation it names the exact failing row + its banked cost-of-omission.

- **PRIMARY `submission_gate_self_test_passes = 1` (34/34).**
- **TEST `gate_catches_row1_host_loop = 1`** — the gate returns NO-GO, flags row
  1, and attaches the imported **444.92 TPS / 85.17%** cost when the relocate
  reverts to a host loop.

**LOCAL CPU-only static analysis.** No GPU / vLLM / HF Job / submission /
served-file change / kernel deploy. Adds 0 TPS (primary = self-test). BASELINE
stays **481.53** (PPL 2.3772). Greedy/PPL untouched. It BLOCKS or CLEARS the
packaging precondition; a human still files `Approval request: HF job`.

Artifacts: `research/spec_cost_model/executable_submission_gate/verify_submission_gate.py`,
`.../executable_submission_gate.json`. W&B `pqpb8ugk` (group
`executable-submission-gate`). Imports #186 `u9kje7sn`.

## Public evidence used

- **ubel #186** `tree_submission_manifest.json` (W&B `u9kje7sn`) — the static
  MUST-RETAIN manifest this gate *enforces*: 22 flags, 19 must-retain, 5
  double-load-bearing, row-1 (`relocate_salvaged_kv`) = **85.17%** binding
  packaging cost. **Imported, not re-derived** (the manifest's own top follow-up,
  previously un-owned). *Extending* it from document → enforcement.
- **ubel #157** `report_salvage_kv_relocation_audit.md` — the 1571× host-loop vs
  vectorized microbench (145.2 ms vs 0.092 ms/call; descent 516→77 TPS) that
  defines the row-1 device-vectorized `[L,W,H,D]` `index_select`+`index_copy_`
  fingerprint the structural probe checks for.
- **Served `fa2sw_precache_kenyan/manifest.json` env** — the faithful 481.53 GO
  fixture (verified `LSK_SKIP_LAYERS` is correctly absent in the shipped env).

**Non-collision.** This is the executable packaging/flag-assertion gate — ubel's
cost/calibration/packaging lane. Distinct from #186 (the static manifest it
enforces), fern #185 (the numerical GO/NO-GO assembler, which consumes this
verdict as one ledger row), denken (output-validity preflight boot/PPL/128), and
land #71 (provides the build to introspect). **NOT open2. NOT a launch.**

## 1. Build-introspection schema (what the gate reads)

The gate reads two inputs from an assembled build:

**`build_env`** — the resolved served env dict. Read directly for served-flag
rows; JSON-embedded knobs are parsed out:
`num_speculative_tokens` from `SPECULATIVE_CONFIG`, `temperature` from
`OVERRIDE_GENERATION_CONFIG`.

**`build_introspection`** — a STRUCTURAL probe of the assembled
accept/relocate/decode path (served `sitecustomize.py` / `serve_patch_*.py` /
tree-build graph), one key per code-shaped row. **A missing key (or a None
field) is FAIL-CLOSED → MISSING → NO-GO**, never a silent pass.

| introspection key | row | PASS rule |
|---|---|---|
| `relocate_salvaged_kv` | row 1 (BUILD-BLOCKER) | `device_vectorized_op_present` AND NOT `host_layer_loop_present` AND NOT `host_sync_in_relocate` AND `n_host_layer_iterations==0` |
| `accept_walk` | row 6 (capturability) | `device_argmax_accept_len_present` AND NOT `host_item_call_present` |
| `decode_logits` | row 4/5 (denominator; double-load-bearing) | `decode_argmax_only` AND `prefill_full_scatter_lp_retained` |

The 22 rows decompose into **3 structural** (relocate / accept-walk / decode),
**2 env-json** (num_speculative_tokens / temperature), **16 env**, **1 trap**
(`LSK_SKIP_LAYERS`). Full schema in `executable_submission_gate.json:introspection_schema`.

## 2. The gate (worked GO + NO-GO)

For each of the 22 manifest rows the gate matches **exactly one** checker by a
unique flag-substring (a 0- or 2-match is itself a fail-closed construction
error), runs it, and attaches the row's banked `cost_sort_tps` **from the
manifest** to any FAIL. The aggregate verdict is fail-closed:

```
NO-GO  iff  any MUST-RETAIN row FAIL/MISSING  (a present TRAP is a FAIL of the LSK row)
GO     iff  all 19 MUST-RETAIN rows PASS and the TRAP is absent
```

**Worked GO** — the faithful shipped `fa2sw_precache_kenyan` env + land #71's
intended descending build spec:
`packaging_verdict = GO`, 0 failing rows, 21 PASS + 1 INFO (the free
logging row), 0 validity-class failures.

**Worked NO-GO** — relocate reverted to host-loop:
`packaging_verdict = NO-GO`, failing row =
`relocate_salvaged_kv == vectorized/device (NOT host-loop)`, **binding cost
444.92 TPS**, detail: *"37-layer host Python loop PRESENT (145ms/call landmine);
.item()/.cpu() host readout; n_host_layer_iterations=37 → host-bound, NOT
CUDA-graph-capturable (516→77 TPS)"*.

## 3. Row-1 binding check (the deliverable)

`row1_check_logic` distinguishes the device-vectorized relocate from a host loop
from a **static** introspection of the assembled path:

- **PASS** — `device_vectorized_op_present` (fused `[L,W,H,D]`
  `index_select`+`index_copy_` by device commit-index, one launch) AND no
  per-layer host loop AND no host `.item()`/`.cpu()` readout AND
  `n_host_layer_iterations==0`.
- **FAIL (host-loop)** — `host_layer_loop_present` OR `n_host_layer_iterations>0`
  OR `host_sync_in_relocate`: a data-dependent Python loop over 37 layers CANNOT
  be CUDA-graph-captured → pins the step host-bound (~122 ms vs the 9.7 ms
  captured target) → descent E[T]=5.04 (→522 TPS) collapses to ~77 TPS.
- **MISSING** — no `relocate_salvaged_kv` key / `n_host_layer_iterations is None`
  → NO-GO (fail-closed).

**`gate_catches_row1_host_loop = TRUE`**: on the host-loop fixture the gate
returns NO-GO, names row 1 as the binding failure, and attaches **444.92 TPS =
85.17% of projected official** (imported from #186, asserted equal to the loaded
JSON). The 85%-cost silent regression is now impossible to ship.

## 4. Validity seam (hand-off to denken)

The 5 **double-load-bearing** rows break PPL/greedy/scoring-basis (not just
speed) if dropped. The gate emits them in a `validity_class_failures` bucket so
its STATIC packaging verdict merges with denken's DYNAMIC output-validity
preflight (boot/PPL/128/greedy) into one launch-readiness surface:

| double-load-bearing row | breaks |
|---|---|
| decode-path argmax-only (prefill scatter+LP seam) | PPL |
| `OVERRIDE_GENERATION_CONFIG` temperature=0.0 | greedy token-identity (#124) |
| `MAX_NUM_SEQS`/`MAX_MODEL_LEN`/…/`DTYPE` | PPL/throughput scoring basis |
| `WEIGHTS_BUCKET`/`LOCAL_MODEL_DIR`/`PCK04_KEEPSET` | model identity/artifact |
| `LSK_SKIP_LAYERS` (TRAP) | output (decoder layer-skip) |

**Boundary:** this gate asserts flag **presence/shape** only; it does **NOT**
re-implement the PPL/greedy/128 output check — that is denken's lane.
`combined_rule = packaging_gate.verdict==GO AND output_gate{boot,ppl≤cap,
completed==128,greedy} all-pass`. Full shared schema in
`executable_submission_gate.json:validity_seam`.

## 5. Self-test (PRIMARY = 34/34)

Synthetic/mutated build-env + introspection fixtures:

| case | fixture | expected | result |
|---|---|---|---|
| (a) | faithful build (shipped #52 env + land #71 descending spec) | GO, 0 failing, every must-retain PASS, 22 rows, 0 construction errors | ✓ |
| (b) | relocate reverted to host-loop | NO-GO, row 1 binding, cost ~444.92, **speed-class not validity** (row 1 greedy-safe) | ✓ |
| (c) | `LSK_SKIP_LAYERS` set | NO-GO, TRAP in failing_rows AND validity_class | ✓ |
| (d) | `PRECACHE_BENCH=0` | NO-GO, row 2, cost ~18.33 (3.526%) | ✓ |
| (e) | each of the 5 double-load-bearing rows dropped | NO-GO with the right row in validity_class (×5) | ✓ |
| (f) | missing relocate introspection | NO-GO, status MISSING (fail-closed, not silent-pass) | ✓ |
| (f2) | empty env + empty introspection | NO-GO; 18 presence-rows fail, TRAP correctly PASSES (absence is correct) | ✓ |
| (g) | nan/exception-clean sweep over all fixtures | finite, no exceptions | ✓ |

One nuance surfaced and is encoded: on an empty build only **18** of 19
must-retain rows fail — the TRAP row PASSES because its assertion is *absence*,
and an empty build has `LSK_SKIP_LAYERS` unset. The gate is correct; the
assertion documents it.

## 6. Hand-off

> Run `verify_submission_gate(build_env, introspection)` against land #71's
> assembled build before any `Approval request: HF job`; a NO-GO names the exact
> failing flag + its banked cost — esp. row 1, the 85%-cost relocate-host-loop
> trap — and feeds fern #185's ledger as the `packaging-gate: GO` precondition row.

**Honest scope.** STATIC flag/shape assertion, no GPU, no new TPS measurement. It
consumes #186's costs and does NOT re-derive them, does NOT run the numerical
GO/NO-GO (fern #185), does NOT run the output-validity gate (denken), and
authorizes nothing. **NOT open2. NOT a launch.**

## Suggested follow-ups

- **Wire it into the live preflight**: have fern #185's assembler call
  `verify_submission_gate` on the resolved env + a real introspection dump of
  land #71's assembled `sitecustomize.py`/`serve_patch_*.py`, so the
  `packaging-gate: GO` ledger row is produced from the actual build rather than a
  fixture.
- **Real introspection extractor**: a small AST/symbol probe that emits the
  `build_introspection` dict from the served files (presence of the fused device
  op symbol, absence of a per-layer host iteration / `.item()` in the relocate
  and accept paths) — turning the schema's contract into an automatic dump.
- **Merge surface with denken**: co-locate `validity_class_failures` with
  denken's boot/PPL/128 output gate behind the `combined_rule` so one call
  answers "is this build launch-ready?".
