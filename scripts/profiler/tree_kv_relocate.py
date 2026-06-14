"""Fused salvage-KV relocate -- PR #71 Component 3c / build-contract leg-2.

After the descending tree-accept walk (Component 4, ``tree_accept_kernel.py``)
commits a scattered root->leaf path through the M verify rows, vLLM's next-step
bookkeeping keeps the first ``num_accepted`` KV slots **in layout order**
(``num_accepted_tokens.gpu = (output_token_ids != -1).sum(dim=1)`` --
``gpu_model_runner.py``: "Valid tokens are contiguous from position 0"). For a
tree the accepted path is e.g. nodes ``[0, 2, 5]`` -> the first 3 slots
``[0,1,2]`` are the WRONG rows -> next-step prefix KV is corrupt -> PPL break.
This module relocates the accepted path's K/V into the first ``len(path)``
contiguous slots so the count-based retention keeps the RIGHT rows.

**Why this exact shape (ubel #157 / #163, MERGED):**
- ONE fused/vectorized GPU op, driven by an **on-device** ``commit_map`` (the
  accept walk emits the node path sync-free, lawine #147) and consumed **without a
  host readout**, so the relocate stays inside the CUDA-graph-captured decode step.
- A host-bound per-row loop (``.item()`` / ``.to('cpu')`` per slot) is
  correctness/PPL-clean but **breaks graph capture -> ~122 ms/step -> descent
  E[T]=5.04 collapses 522->77 TPS** (the silent step-collapsing landmine). The
  landmine reference here quantifies that gap.
- equivalence_rate = 1.0 **by construction**: pure bf16 permute/copy, no cast/
  arithmetic, so the relocated K/V is bit-identical to the source rows.
- leg-2 GATE (settled in the PR): the zero-copy paged block-table re-point is
  **infeasible** on this served vLLM (CPU-source block table + whole-block
  granularity), so the relocate is the banked **fused COPY**, not a re-point.

Forms:
- ``relocate_salvaged_kv``       -- REAL served layout: list of per-layer
  ``(num_blocks, 2, block_size, n_kv, head_size)`` tensors (separate allocations,
  ``gpu_model_runner.py:525``); per-layer advanced-index gather->scatter, on-device
  indices, no host sync. The integration-ready primary.
- ``relocate_salvaged_kv_stacked`` -- ubel #163's banked ``index_select`` +
  ``index_copy_`` on a contiguous ``[L, num_slots, 2, n_kv, head_size]`` stack
  (the 35.3 us fused number); also the cross-check oracle.
- ``relocate_salvaged_kv_hostloop`` -- the host-readout landmine (NOT for serving).

The accept walk produces ``commit_map`` (``tree_spec.descend_accept_path`` is the
CPU reference twin): a fixed width-M map where slot j receives node
``commit_map[j]``; entries past ``num_accepted`` are identity (j) so the op is a
fixed-shape no-host-length relocate -- exactly the sync-free contract.

Run standalone (validates every form bit-for-bit + times the landmine):
  CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python scripts/profiler/tree_kv_relocate.py
"""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_spec import (  # noqa: E402
    PARENT_M16,
    PARENT_M32,
    TreeSpec,
    descend_accept,
    descend_accept_path,
)

# Real served gemma-4-E4B-it verify-stack KV dims (config.json text_config):
# num_key_value_heads=2, head_dim=256, bf16. Design-doc KV-cached attention-layer
# stack = 37 (of 42 hidden layers). block_size is a vLLM kernel block (>=64).
N_KV = 2
HEAD_SIZE = 256
N_LAYERS = 37
KV_DTYPE = torch.bfloat16
EK = N_KV * HEAD_SIZE  # 512: contiguous (n_kv, head) span of one K (or V) half-slot
assert EK & (EK - 1) == 0, "EK must be a power of 2 for tl.arange"


def build_commit_map(path: list[int], width: int, device) -> torch.Tensor:
    """The on-device ``commit_map`` the descent kernel emits: slot j <- node
    ``path[j]`` for the accepted prefix, identity (j) past it so the relocate is a
    fixed width-M op needing no host-side length (sync-free)."""
    cm = list(range(width))
    for j, node in enumerate(path):
        cm[j] = node
    return torch.tensor(cm, dtype=torch.int64, device=device)


# ---------------------------------------------------------------------------
# Form 0 -- SERVED PRIMARY: fused pointer-array Triton relocate over the REAL
# per-layer separate-allocation layout (gpu_model_runner.py:7012-7018 allocates
# one int8 buffer per KVCacheTensor, so the 37 layers are NOT one arena and a
# zero-copy [L,...] stack does not exist). A naive per-layer torch loop is 37*2
# launches -> ~1.3 ms/step (measured) -> too slow; ubel #157 mandates ONE fused
# launch. This vectorizes ALL layers in 2 launches via a layer data-ptr array:
# gather every accepted slot into a staging buffer (materialize -> aliasing-safe),
# then scatter into the contiguous dst slots. On-device indices, no host readout.
# A slot's K/V is two EK-contiguous spans in (NB,2,BS,n_kv,head): the "2" (K/V)
# dim sits between block and offset, so element (blk,kv01,off,h,d) lives at
#   blk*(2*BS*EK) + kv01*(BS*EK) + off*EK + (h*head + d).
# ---------------------------------------------------------------------------
@triton.jit
def _kv_gather_kernel(layer_ptrs, src_idx, staging, BS, M, EK: tl.constexpr):
    layer = tl.program_id(0)
    j = tl.program_id(1)
    kv01 = tl.program_id(2)
    p = tl.load(layer_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    s = tl.load(src_idx + j)
    src_off = (s // BS) * (2 * BS * EK) + kv01 * (BS * EK) + (s % BS) * EK
    cols = tl.arange(0, EK)
    stg_off = (layer * M + j) * (2 * EK) + kv01 * EK
    tl.store(staging + stg_off + cols, tl.load(p + src_off + cols))


@triton.jit
def _kv_scatter_kernel(layer_ptrs, dst_idx, staging, BS, M, EK: tl.constexpr):
    layer = tl.program_id(0)
    j = tl.program_id(1)
    kv01 = tl.program_id(2)
    p = tl.load(layer_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    d = tl.load(dst_idx + j)
    dst_off = (d // BS) * (2 * BS * EK) + kv01 * (BS * EK) + (d % BS) * EK
    cols = tl.arange(0, EK)
    stg_off = (layer * M + j) * (2 * EK) + kv01 * EK
    tl.store(p + dst_off + cols, tl.load(staging + stg_off + cols))


def make_relocate_buffers(kv_list: list[torch.Tensor], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the (fixed-pointer) layer-ptr array + staging buffer ONCE; reused
    every decode step (so they're capturable graph state, not per-step allocs)."""
    dev = kv_list[0].device
    layer_ptrs = torch.tensor([t.data_ptr() for t in kv_list], dtype=torch.int64, device=dev)
    staging = torch.empty(N_LAYERS, width, 2, EK, dtype=KV_DTYPE, device=dev)
    return layer_ptrs, staging


def relocate_salvaged_kv_fused(
    kv_list: list[torch.Tensor],
    base: int,
    commit_map: torch.Tensor,
    block_size: int,
    layer_ptrs: torch.Tensor,
    staging: torch.Tensor,
) -> None:
    """SERVED form: two fused launches across all 37 layers, on-device commit_map,
    no host readout, aliasing-safe (staging materializes srcs first)."""
    m = commit_map.shape[0]
    dev = commit_map.device
    src_slots = base + commit_map
    dst_slots = base + torch.arange(m, device=dev, dtype=torch.int64)
    grid = (N_LAYERS, m, 2)
    _kv_gather_kernel[grid](layer_ptrs, src_slots, staging, block_size, m, EK)
    _kv_scatter_kernel[grid](layer_ptrs, dst_slots, staging, block_size, m, EK)


# ---------------------------------------------------------------------------
# Form 1 -- REAL served layout: list of per-layer (NB, 2, BS, n_kv, head) tensors.
# ---------------------------------------------------------------------------
def relocate_salvaged_kv(
    kv_list: list[torch.Tensor],
    base: int,
    commit_map: torch.Tensor,
    block_size: int,
) -> None:
    """In-place relocate on the real served KV layout. ``commit_map`` is on-device
    [M] (node id per dst slot). NO host readout -> graph-capturable. Gather
    materializes the source rows BEFORE the scatter, so it is aliasing-safe even
    when a dst slot is also a (later) src slot."""
    m = commit_map.shape[0]
    dst_slots = base + torch.arange(m, device=commit_map.device, dtype=torch.int64)
    src_slots = base + commit_map
    sb = src_slots // block_size
    so = src_slots % block_size
    db = dst_slots // block_size
    do = dst_slots % block_size
    for kv in kv_list:
        gathered = kv[sb, :, so, :, :]  # [M, 2, n_kv, head] -- materialized copy
        kv[db, :, do, :, :] = gathered


# ---------------------------------------------------------------------------
# Form 2 -- ubel #163 banked fused: index_select + index_copy_ on a contiguous
# [L, num_slots, 2, n_kv, head] stack (fully vectorized across layers).
# ---------------------------------------------------------------------------
def relocate_salvaged_kv_stacked(
    kv_stack: torch.Tensor,
    base: int,
    commit_map: torch.Tensor,
) -> None:
    """In-place relocate on a contiguous layer-stacked cache. Two launches total
    (index_select gathers -> index_copy_ scatters), no host sync. index_select
    materializes -> aliasing-safe."""
    m = commit_map.shape[0]
    dst_slots = base + torch.arange(m, device=commit_map.device, dtype=torch.int64)
    src_slots = base + commit_map
    gathered = kv_stack.index_select(1, src_slots)  # [L, M, 2, n_kv, head]
    kv_stack.index_copy_(1, dst_slots, gathered)


# ---------------------------------------------------------------------------
# Form 3 -- the host-readout LANDMINE (per-row .tolist()/python loop). NOT served.
# ---------------------------------------------------------------------------
def relocate_salvaged_kv_hostloop(
    kv_list: list[torch.Tensor],
    base: int,
    commit_map: torch.Tensor,
    block_size: int,
) -> None:
    """Correct but graph-capture-breaking: a host readout of commit_map plus a
    per-(layer,row) python loop. ubel #157's banned variant -- here only to
    measure the step-collapse it causes."""
    cm = commit_map.tolist()  # <-- HOST SYNC (kills capture)
    m = len(cm)
    for kv in kv_list:
        staging = []
        for j in range(m):
            s = base + cm[j]
            staging.append(kv[s // block_size, :, s % block_size, :, :].clone())
        for j in range(m):
            d = base + j
            kv[d // block_size, :, d % block_size, :, :] = staging[j]


# ---------------------------------------------------------------------------
# Independent ground-truth (obviously-correct gather-then-scatter, on CPU lists).
# ---------------------------------------------------------------------------
def _reference_relocated(
    kv_list: list[torch.Tensor],
    base: int,
    commit_map: torch.Tensor,
    block_size: int,
) -> list[torch.Tensor]:
    cm = commit_map.tolist()
    m = len(cm)
    out = [kv.clone() for kv in kv_list]
    for li, kv in enumerate(kv_list):
        gathered = [
            kv[(base + cm[j]) // block_size, :, (base + cm[j]) % block_size, :, :].clone()
            for j in range(m)
        ]
        for j in range(m):
            d = base + j
            out[li][d // block_size, :, d % block_size, :, :] = gathered[j]
    return out


def _make_kv_list(num_blocks: int, block_size: int, device, seed: int) -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.randn(
            num_blocks, 2, block_size, N_KV, HEAD_SIZE, generator=g, dtype=torch.float32
        ).to(device=device, dtype=KV_DTYPE)
        for _ in range(N_LAYERS)
    ]


def _stack_from_list(kv_list: list[torch.Tensor], block_size: int) -> torch.Tensor:
    """Contiguous [L, num_slots, 2, n_kv, head] holding the same logical slot K/V
    as the per-layer (NB,2,BS,n_kv,head) list (for the banked stacked form)."""
    nb = kv_list[0].shape[0]
    ns = nb * block_size
    stack = torch.empty(
        N_LAYERS, ns, 2, N_KV, HEAD_SIZE, dtype=KV_DTYPE, device=kv_list[0].device
    )
    for li, kv in enumerate(kv_list):
        # (NB,2,BS,n_kv,head) -> slot-major (NB,BS,2,n_kv,head) -> (NB*BS,2,n_kv,head)
        stack[li] = kv.permute(0, 2, 1, 3, 4).reshape(ns, 2, N_KV, HEAD_SIZE)
    return stack


def _read_slot_stack(stack: torch.Tensor, slot: int) -> torch.Tensor:
    return stack[:, slot, :, :, :]  # [L, 2, n_kv, head]


def _read_slot_list(kv_list: list[torch.Tensor], slot: int, block_size: int) -> torch.Tensor:
    b, o = slot // block_size, slot % block_size
    return torch.stack([kv[b, :, o, :, :] for kv in kv_list])  # [L, 2, n_kv, head]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _check_equiv(name: str, tree: TreeSpec, base: int, block_size: int, seed: int) -> int:
    """Build a random (node_argmax, draft_token), derive the real descent path,
    run the per-layer + stacked forms, assert both bit-identical to the reference
    AND to each other. Returns the committed path length."""
    dev = torch.device("cuda")
    m = tree.num_nodes
    g = torch.Generator().manual_seed(seed)
    node_argmax = torch.randint(0, 4, (m,), generator=g).tolist()
    draft_token = torch.randint(0, 4, (m,), generator=g).tolist()
    path = descend_accept_path(tree, node_argmax, draft_token)
    _, vc, _ = descend_accept(tree, node_argmax, draft_token)
    assert len(path) == vc, (name, len(path), vc)

    # relocate width = max committed = spine length (max_depth+1); the trailing
    # verify rows are discarded by vLLM's count-based retention, so never moved.
    width = tree.max_depth + 1
    nb = -(-(base + m) // block_size) + 1
    commit_map = build_commit_map(path, width, dev)

    kv_f = _make_kv_list(nb, block_size, dev, seed)
    ref = _reference_relocated(kv_f, base, commit_map, block_size)
    layer_ptrs, staging = make_relocate_buffers(kv_f, width)
    relocate_salvaged_kv_fused(kv_f, base, commit_map, block_size, layer_ptrs, staging)
    torch.cuda.synchronize()
    ok_fused = all(torch.equal(kv_f[li], ref[li]) for li in range(N_LAYERS))

    kv_a = _make_kv_list(nb, block_size, dev, seed)
    relocate_salvaged_kv(kv_a, base, commit_map, block_size)
    torch.cuda.synchronize()
    ok_perlayer = all(torch.equal(kv_a[li], ref[li]) for li in range(N_LAYERS))

    kv_b = _make_kv_list(nb, block_size, dev, seed)
    stack = _stack_from_list(kv_b, block_size)
    relocate_salvaged_kv_stacked(stack, base, commit_map)
    torch.cuda.synchronize()
    # the M dst slots of the stack must match the fused-relocated dst slots.
    ok_cross = all(
        torch.equal(_read_slot_stack(stack, base + j), _read_slot_list(kv_f, base + j, block_size))
        for j in range(len(path))
    )
    assert ok_fused, f"{name}: FUSED served form != reference"
    assert ok_perlayer, f"{name}: per-layer form != reference"
    assert ok_cross, f"{name}: stacked form != fused form"
    spans = (base + m - 1) // block_size - base // block_size
    print(
        f"  [ok] {name}: path={path} (len {len(path)}) | FUSED==ref AND per-layer==ref "
        f"AND stacked==fused bit-exact | dst span {spans + 1} block(s)"
    )
    return len(path)


def _check_aliasing() -> None:
    """A non-monotonic commit_map where a naive IN-PLACE sequential copy WOULD
    corrupt (a dst slot is read as a later src). Proves gather-then-scatter
    (materialize first) is robust regardless of node ordering."""
    dev = torch.device("cuda")
    bs = 64
    m = 8
    base = 500
    # map: slot0<-node0, slot1<-node3, slot2<-node1, slot3<-node2 (identity rest).
    # naive in-place: writing slot1<-node3 then slot2<-node1 reads node1 fine, but
    # slot3<-node2 reads node2 AFTER slot2 (node2's slot) was overwritten -> corrupt.
    path = [0, 3, 1, 2]
    commit_map = build_commit_map(path, m, dev)
    nb = -(-(base + m) // bs) + 1
    kv = _make_kv_list(nb, bs, dev, seed=99)
    ref = _reference_relocated(kv, base, commit_map, bs)

    # show the naive in-place sequential is WRONG (motivates materialize-first):
    naive = [t.clone() for t in kv]
    cm = commit_map.tolist()
    for t in naive:
        for j in range(m):
            s, d = base + cm[j], base + j
            t[d // bs, :, d % bs, :, :] = t[s // bs, :, s % bs, :, :]
    naive_wrong = any(not torch.equal(naive[li], ref[li]) for li in range(N_LAYERS))

    relocate_salvaged_kv(kv, base, commit_map, bs)
    torch.cuda.synchronize()
    ok = all(torch.equal(kv[li], ref[li]) for li in range(N_LAYERS))
    assert ok, "aliasing: gather-then-scatter form != reference"
    assert naive_wrong, "aliasing case did not actually exercise overlap"
    print(
        "  [ok] aliasing stress (path [0,3,1,2]): gather-then-scatter bit-exact; "
        "naive in-place sequential corrupts (confirms materialize-first is required)"
    )


def _time_ms(fn, iters: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _bench(tree: TreeSpec, base: int, block_size: int) -> None:
    dev = torch.device("cuda")
    m = tree.num_nodes
    node_argmax = torch.randint(0, 4, (m,)).tolist()
    draft_token = torch.randint(0, 4, (m,)).tolist()
    path = descend_accept_path(tree, node_argmax, draft_token)
    width = tree.max_depth + 1
    commit_map = build_commit_map(path, width, dev)
    nb = -(-(base + m) // block_size) + 1

    kv = _make_kv_list(nb, block_size, dev, seed=7)
    stack = _stack_from_list(kv, block_size)
    layer_ptrs, staging = make_relocate_buffers(kv, width)

    # warmup
    for _ in range(5):
        relocate_salvaged_kv_fused(kv, base, commit_map, block_size, layer_ptrs, staging)
        relocate_salvaged_kv(kv, base, commit_map, block_size)
        relocate_salvaged_kv_stacked(stack, base, commit_map)
    t_fused = _time_ms(
        lambda: relocate_salvaged_kv_fused(kv, base, commit_map, block_size, layer_ptrs, staging),
        200,
    )
    t_perlayer = _time_ms(lambda: relocate_salvaged_kv(kv, base, commit_map, block_size), 200)
    t_stacked = _time_ms(lambda: relocate_salvaged_kv_stacked(stack, base, commit_map), 200)
    t_host = _time_ms(
        lambda: relocate_salvaged_kv_hostloop(kv, base, commit_map, block_size), 20
    )
    print(
        f"  [time] tree M={m} (relocate width={width}, {N_LAYERS} layers, n_kv={N_KV}, "
        f"head={HEAD_SIZE}, bf16): FUSED-ptr {t_fused * 1e3:6.1f} us | naive per-layer "
        f"{t_perlayer * 1e3:7.1f} us | stacked(idx_sel+copy) {t_stacked * 1e3:6.1f} us | "
        f"HOST-LOOP landmine {t_host * 1e3:9.1f} us ({t_host / max(t_fused, 1e-9):.0f}x over fused)"
    )


def _check_graph_capture(tree: TreeSpec, base: int, block_size: int) -> None:
    """Capture the relocate in a CUDA graph, then change commit_map CONTENTS and
    replay -> proves the captured step reads the LIVE on-device commit_map (the
    sync-free integration contract). The host-loop form can't reach here -- it
    does a .tolist() host readout mid-step."""
    dev = torch.device("cuda")
    m = tree.num_nodes
    nb = -(-(base + m) // block_size) + 1
    width = tree.max_depth + 1
    kv_clean = _make_kv_list(nb, block_size, dev, seed=11)
    work = [t.clone() for t in kv_clean]  # fixed-pointer working buffer
    commit_map = build_commit_map(list(range(width)), width, dev)  # identity (no-op) to warm
    layer_ptrs, staging = make_relocate_buffers(work, width)  # fixed graph state

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            relocate_salvaged_kv_fused(work, base, commit_map, block_size, layer_ptrs, staging)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        relocate_salvaged_kv_fused(work, base, commit_map, block_size, layer_ptrs, staging)

    # now exercise a REAL path purely by mutating commit_map contents + replaying:
    node_argmax = torch.randint(0, 4, (m,)).tolist()
    draft_token = torch.randint(0, 4, (m,)).tolist()
    path = descend_accept_path(tree, node_argmax, draft_token)
    for li in range(N_LAYERS):
        work[li].copy_(kv_clean[li])  # reset buffer (same pointer)
    commit_map.copy_(build_commit_map(path, width, dev))  # live update, no recapture
    graph.replay()
    torch.cuda.synchronize()

    ref = _reference_relocated(kv_clean, base, build_commit_map(path, width, dev), block_size)
    ok = all(torch.equal(work[li], ref[li]) for li in range(N_LAYERS))
    assert ok, "graph replay with mutated commit_map != reference"
    t_replay = _time_ms(graph.replay, 500) * 1e3  # the actual SERVED relocate cost
    print(
        f"  [ok] CUDA-graph capture+replay (M={m}, width={width}): captured once, replayed "
        f"with a LIVE-mutated commit_map (path {path}) -> bit-exact. Sync-free contract holds. "
        f"SERVED relocate cost = {t_replay:.1f} us/step (graph replay)."
    )


def main() -> None:
    assert torch.cuda.is_available(), "needs a GPU (CUDA_VISIBLE_DEVICES=0)"
    print("=== tree_kv_relocate validation (PR #71 Component 3c / leg-2 fused copy) ===")
    print("-- equivalence (per-layer real layout == reference == stacked banked form) --")
    base = 500  # a realistic prefix slot -> M rows straddle a block boundary
    bs = 64
    _check_equiv("M16", TreeSpec(PARENT_M16), base, bs, seed=2)
    _check_equiv("M32", TreeSpec(PARENT_M32), base, bs, seed=3)
    _check_equiv("M32@base997", TreeSpec(PARENT_M32), 997, bs, seed=5)
    _check_aliasing()
    print("-- timing (fused vs the host-readout landmine) --")
    _bench(TreeSpec(PARENT_M16), base, bs)
    _bench(TreeSpec(PARENT_M32), base, bs)
    print("-- graph capture (sync-free contract) --")
    _check_graph_capture(TreeSpec(PARENT_M16), base, bs)
    _check_graph_capture(TreeSpec(PARENT_M32), base, bs)
    print("=== all tree_kv_relocate checks passed ===")


if __name__ == "__main__":
    main()
