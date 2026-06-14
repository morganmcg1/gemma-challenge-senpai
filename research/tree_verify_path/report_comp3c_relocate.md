# PR #71 Component 3c — fused salvage-KV relocate: BUILT + validated (leg-2)

**Status:** the 3c KV-compaction blocker (Finding B) is now a built, locally
validated GPU kernel. This is the last *deferred* component of the continued-gen
tree-verify path and the answer to build-contract **leg-2** (ubel #157/#163).

Module: `scripts/profiler/tree_kv_relocate.py` (standalone-validatable, mirrors
`tree_accept_kernel.py`). Reference twin for the commit-index: new
`tree_spec.descend_accept_path` (the node path `descend_accept` commits).

## What it does (Finding B, restated)

After the descend walk commits a scattered root→leaf path through the M verify
rows, vLLM's next-step retention keeps the first `num_accepted` KV slots **in
layout order** (`num_accepted_tokens.gpu = (output_token_ids != -1).sum(dim=1)`,
`gpu_model_runner.py` — "Valid tokens are contiguous from position 0"). For a
tree the accepted path is e.g. nodes `[0, 2, 5]` → the first 3 slots `[0,1,2]`
are the WRONG rows → next-step prefix KV corrupt → PPL break. The relocate gathers
the accepted path's K/V into the first `len(path)` contiguous slots so the
count-based retention keeps the RIGHT rows. **Greedy-safe by construction:** pure
bf16 permute/copy, no cast/arithmetic → relocated K/V bit-identical to source.

## The build is shaped by the served layout (two verified facts)

1. **Per-layer separate allocations (NOT one arena).**
   `_allocate_kv_cache_tensors` (`gpu_model_runner.py:7012-7018`) does one
   `torch.zeros(size, int8)` per `KVCacheTensor`, so the 37 KV-cached attention
   layers are separate tensors — there is **no zero-copy `[L,…]` stack**. ubel
   #163's banked `index_select`+`index_copy_` assumes a layer-stack; on this
   served stack that stack does not exist, so the fused op must reach all 37
   layers another way.
2. **A slot's K/V is two non-contiguous spans.** Per-layer shape is
   `(num_blocks, 2, block_size, n_kv, head)` (`flash_attn.py:149`). The K/V dim
   (`2`) sits *between* block and offset, so one slot = two `EK=n_kv*head=512`
   contiguous spans separated by `block_size*EK`. No slot-major reshape is
   contiguous → a custom-index kernel is required, not a plain `view`.

## The served form: fused pointer-array Triton relocate (2 launches, sync-free)

`relocate_salvaged_kv_fused` vectorizes **all 37 layers in 2 launches** via a
layer `data_ptr()` array (Triton 3.6 int64→pointer, smoke-verified):
gather every accepted slot into a staging buffer (materialize → **aliasing-safe**),
then scatter into the contiguous dst slots. On-device `commit_map`, **no host
readout** → stays inside the captured decode graph (lawine #147 sync-free). Width
= `max_depth+1` = spine length (10 for both wirbel trees): the trailing verify
rows are discarded by vLLM anyway, so they are never moved.

`commit_map` is the fixed width-`W` map the descent kernel emits on-device: slot
`j` ← node `commit_map[j]` for the accepted prefix, identity (`j`) past it — a
fixed-shape op needing no host-side length (the sync-free contract). CPU twin:
`descend_accept_path`.

## Local validation (zero quota, 1 GPU, `/tmp/server-venv`, bf16, 37 layers)

`CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python scripts/profiler/tree_kv_relocate.py`

| check | result |
|---|---|
| **equivalence** (M16, M32, M32@base997) | FUSED == independent reference == naive per-layer == stacked `index_select`/`copy_`, **bit-exact (rate 1.0)** |
| **aliasing** (path `[0,3,1,2]`) | gather-then-scatter bit-exact; naive in-place sequential **corrupts** → materialize-first is required |
| **graph capture** | captured once, replayed with a **live-mutated** `commit_map` → bit-exact (proves the served step reads live on-device commit-index) |
| **SERVED cost (graph replay)** | **8.3 µs/step** — under ubel #163's banked 35.3 µs fused estimate → clear-500 bar ≥ 4.880, relocate ≈ free vs the 53%-decode verify GEMM |
| host-readout LANDMINE | 12.5 ms/step (~1500× the served cost) — empirically confirms the descent-TPS 522→77 collapse ubel #157 warned of |
| naive per-layer torch loop | 1.4 ms/step (74 launches) — too slow; quantifies *why* ubel mandates ONE fused launch (≠ "device-resident" alone) |

## Net

- **leg-2 closed on the build side:** the fused copy is built, bit-exact,
  graph-capturable, and **8.3 µs/step served** (cheaper than the banked estimate
  because the real cost is graph-replay, not eager launch). The paged re-point was
  already proven infeasible on this vLLM (PR comment, design-doc leg-2) → the fused
  copy is the variant, and it lands well inside budget.
- **Wires into continued-gen as:** descent kernel (`tree_accept_kernel.py`, leg-1)
  emits the on-device `commit_map` → `relocate_salvaged_kv_fused` compacts the KV
  → vLLM's unchanged count-based retention now keeps the correct prefix. Both
  kernels are standalone-validated; the live wiring is Step-2 (widened verify +
  descent kernel emit + this relocate in the captured decode step).
- **No PPL/greedy risk:** pure bf16 copy, equivalence 1.0; verifier numerics
  untouched. No HF launch — official confirmation stays human-approval-gated.
