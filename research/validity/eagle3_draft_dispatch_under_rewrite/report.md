# PR #315 — Does the #312 loopgraph rewrite reintroduce draft-side dispatch risk?

**PRIMARY `draft_dispatch_under_rewrite_self_test_passes` = True** (all 7 conditions a–g)
**TEST `eagle_draft_capture_is_single_shape` = False** · **`draft_positions_fit_deployed_ceiling16` = True**
**W&B `jryber0b`** (group `eagle3-draft-dispatch`) · LOCAL read-only static analysis, 0 GPU, 0 TPS

> **Verdict:** the #312 T6/T7 loopgraph rewrite does **NOT reintroduce** the draft-side
> capture-dispatch (lawine #101) risk — **classification (a), CONDITIONAL on the emitted gate**.
> The deployed MTP draft is dispatch-safe by **MECHANISM** (`CUDAGraphMode.NONE` manual
> `torch.cuda.graph` capture, `sitecustomize.py:170`/`:243` → **OFF** vLLM's
> `cudagraph_capture_sizes` list), **NOT** by single-shape. EAGLE-3 **BREAKS** single-shape (its
> draft KV extent grows `1..K=7`), but a manual **UNROLLED** `NONE` capture records each per-step
> kernel at its **own static shape** — absorbing the variation into **ONE graph** rather than a
> capture-size **LIST**. So the #101 list-dispatch risk returns **only if** the rewrite abandons
> `NONE`; #312's plan keeps the manual scaffolding "structurally reusable". Belt-and-suspenders:
> the within-chain draft replay sizes `{1..7}` all fit the deployed ceiling-**16**, so even a
> degenerate list-route is draft-safe — **verify M=32** (needs `[24,32]` per #311) remains the
> **SOLE** #101 exposure. DERIVED from #312's spec (no EAGLE checkpoint); BASELINE 481.53
> untouched; 0 TPS.

## 1. Why the MTP draft is dispatch-safe — confirmed from the served source (instruction 1)

Parsed read-only from `submissions/fa2sw_precache_kenyan/{manifest.json, sitecustomize.py}`:

| marker | value | source |
|---|---|---|
| `num_speculative_tokens` (K) | **7** (mtp) | `manifest.json` `SPECULATIVE_CONFIG` |
| `cudagraph_runtime_mode=CUDAGraphMode.NONE` | **present** | `sitecustomize.py:170` (ubel #311 cite) |
| manual `with torch.cuda.graph(graph):` | **present** | `sitecustomize.py:243` |
| width-1 body `self.input_ids[:1].copy_(source)` | **present** | `:182` |
| single recycled `[1,2560]` hidden `self.hidden_states[:1].copy_(backbone_hidden[:1])` | **present** | `:189/:201` |
| "Q-only and KV-shared" invariant comment | **present** | `:25–32` |

**The load-bearing fact (mechanism, not shape):** the drafter sets
`cudagraph_runtime_mode=CUDAGraphMode.NONE`, so the engine does **not** route the draft forward
through its cudagraph dispatcher; the patch **itself** wraps the whole K-iteration loop in **ONE**
`torch.cuda.graph(graph)` replayed by `graph.replay()`. The draft chain therefore **bypasses**
vLLM's `cudagraph_capture_sizes` list entirely — it can **never** hit the lawine-#101 IndexError,
which is a *list-lookup* failure. The single width-1 shape (Q-only, KV-shared → width-1 is exact)
is merely what makes the MTP capture **trivial** — one body replayed K times — **not** what makes it
dispatch-safe.

## 2. #312's T6/T7 rewrite mapped onto the draft capture (instruction 2)

Banked from wirbel #312 (`9b1arani`, T7 `eagle3_delta`), the EAGLE-3 draft chain differs from MTP on
four axes (all **false** for the MTP single-shape invariant):

| delta | MTP draft | EAGLE-3 draft |
|---|---|---|
| own-KV layer | Q-only, KV-shared — never writes KV | full Llama decoder with own `k_proj/v_proj` — **writes** KV per draft position |
| growing draft KV | no cross-position deps — width-1 exact | step `i` reads KV it wrote at `0..i` — attention extent **GROWS 1..K** |
| `seq_len`/`slot_mapping` | fixed `[1]` buffer set once; metadata built once | must **advance per iteration INSIDE the captured graph** |
| aux seed dim | single recycled `[1,2560]` target hidden | fused `[1,7680]` aux cat of 3 backbone layers (2/21/39); `7680 = 3·2560` ✓ |

**Per-step EAGLE draft replay shape (DERIVED, not measured)** — the query stays **width-1** every
step, but the **attention KV extent grows**:

| step i | query | draft KV extent | slot-write | position | matches MTP single-shape |
|---|---|---|---|---|---|
| 0 | 1 | 1 | 0 | prefix+0 | **True** |
| 1 | 1 | 2 | 1 | prefix+1 | False |
| … | 1 | … | … | … | False |
| 6 | 1 | 7 | 6 | prefix+6 | False |

Each shape is **static per unrolled step** (`i` is a compile-time constant under unrolling), so the
whole K=7 chain is **capturable as ONE manual graph** — the per-step kernels differ but are each
fixed at capture and identical at every replay.

## 3. Classification — (a) vs (b) (instruction 3)

- **per-step shapes VARY** → `eagle_draft_capture_is_single_shape = **False**` (the MTP single-shape
  invariant is **broken**: draft KV extent `1..7`). So it is **not (a) "for free"**.
- **but that variation does NOT force a capture-size LIST** → it is absorbed by unrolling into **ONE**
  manual `CUDAGraphMode.NONE` graph. A manual graph is a flat recorded kernel stream with **no
  per-call list lookup**; the lawine-#101 IndexError is a property of vLLM's *list-dispatcher*, which
  `NONE` bypasses. → `reintroduces_list_dispatch_risk = **False**`.
- **belt-and-suspenders** → `draft_positions_fit_deployed_ceiling16 = **True**`: the within-chain
  replay sizes `{1..7}` and `max extent 7` are **all ≤ 16**, so even a degenerate list-route stays
  draft-safe. Only **verify M=32** breaches the ceiling.

→ **Classification = (a)**, `classification_is_conditional = True` (holds only if the gate is honored).
#312's banked plan keeps `_build_static_buffers`/`_capture_graph`/ping-pong "structurally reusable",
which is consistent with honoring the gate.

## 4. The dispatch-correctness gate the rewrite must honor (instruction 4)

Since classification is **(a)-conditional**, the invariant T6/T7 must **preserve** to keep draft-side
safety (the "fixed-shape slot-write KV advance" the card asks for):

1. **PRESERVE `CUDAGraphMode.NONE`** for the draft chain — keep the re-bodied `propose_onegraph`
   wrapping the K loop in a manual `torch.cuda.graph(graph)` replay so it stays **off** vLLM's
   `cudagraph_capture_sizes` list. *Abandoning `NONE` for FULL/PIECEWISE mode is the **only** way to
   reintroduce #101 on the draft side.* **(single load-bearing property)**
2. **PRE-ALLOCATE the draft KV to full K=7 extent** (static address) and advance by **fixed-address
   slot-writes** (slot `i` baked per unrolled iteration); **NO realloc/resize inside the captured
   region** — a realloc breaks the manual capture **outright** (a harder failure than a dispatch error).
3. **BAKE the per-iteration attention metadata** (`seq_len_i=prefix+i+1`, `slot_mapping` slot `i`) as
   **static compile-time-constant slices** so each unrolled step records a fixed-shape kernel identical
   at every replay (this is the "advance `seq_len/slot_mapping` INSIDE the captured graph" #312
   flagged — it must be a **static** schedule, not a data-dependent shape; requires a draft attention
   backend that admits a static per-step capture, as the deployed MTP path already does under `NONE`).
4. **CROSS-CHECK** the within-chain draft replay sizes `{1..7} ≤` deployed ceiling-16 (they are) — so
   even a degenerate list-route stays draft-safe; the **verify side** (M=32 needs `[24,32]` per #311)
   remains the **sole** #101 exposure.

## 5. Anchor cross-check + self-test (NaN-clean)

`#311 (os01ttw9)` + `#312 (9b1arani)` reloaded; all imported constants match to **`max_abs_err = 0.0`
(≤ 1e-6)**: K=7, M=8, ceiling-16, `max_safe_tree_width=16`, draft safe-by-construction /
not-list-dispatched, eager floor 402.1, regression 16.5%, T7 = correctness-rederivation, and the
`7680 = 3·2560` fused-aux consistency.

**Self-test (PRIMARY, a–g):** a MTP draft-safety mechanism confirmed from source ✓ · b #312 T6/T7
draft-chain deltas enumerated (4, all sourced to `9b1arani`) ✓ · c per-step EAGLE draft replay shape
derived (K=7, query width-1, KV extent 1..7) ✓ · d risk classified (a) with the 4-item gate emitted ✓
· e #311/#312 constants ≤1e-6 ✓ · f NaN-clean ✓ · g caveats carried (6) ✓ → **PRIMARY PASS**.

## 6. Honest caveats (carried in the artifact)

1. **DERIVED, NOT measured:** no EAGLE-3 checkpoint exists (training-gated). The per-step draft KV
   extents `0..i`, the fused `[1,7680]` seed, and the own-KV writes are derived from #312's banked
   rewrite spec + the deployed MTP source — not from a running `EagleProposer`. Prices the
   dispatch-correctness property; does not implement it.
2. **MECHANISM, not single-shape:** MTP draft-safety is `CUDAGraphMode.NONE`, not single-shape;
   EAGLE breaks single-shape but the variation is absorbed into one manual graph and does not by
   itself reintroduce #101.
3. **CONDITIONAL (a):** (a)-safety holds only if the gate is honored; #312's reusable scaffolding is
   consistent with that, but the gate is a **checklist** the implementation must satisfy.
4. **BACKEND assumption:** gate item 3 assumes the EAGLE draft attention backend admits a static
   per-step capture under `NONE` (the deployed MTP path already does). A non-capturable runtime-`seq_len`
   backend would fail item 3 — a flagged backend-compatibility risk.
5. **VERIFY side UNCHANGED:** this card scopes only the draft-side sub-axis; #311's verify gate
   (M≤16 safe, M=32 needs `[24,32]`) stands independently and orthogonally.
6. **0 TPS / config-correctness property:** depends only on integer token-counts / KV extents and the
   capture mode, not tensor values. NOT a launch / build / served-file change / HF Job.

## Greedy/PPL-safety certificate

`analysis_only = True`. No served-file change, no emitted-token change, no HF Job, no submission, NOT a
launch, NOT a build. BASELINE **481.53 TPS unchanged**; this leg adds **0 TPS**.

## Hand-off

EAGLE-3 draft-side dispatch-correctness priced under #312's T6/T7 rewrite: **classification (a)** — no
list-dispatch (#101) risk reintroduced on the draft side, **conditional on preserving
`CUDAGraphMode.NONE` + a static full-K slot-write KV-advance schedule**. The MTP single-shape invariant
**breaks** (draft KV grows `1..7`) but is absorbed into one manual unrolled graph, not a capture-size
list; and K=7 ≤ deployed ceiling-16 anyway. The human GO/NO-GO can treat draft-side dispatch as
**CLEARED provided the rewrite honors gate item (1)**; the verify-side #101 gate (#311) is unchanged.
