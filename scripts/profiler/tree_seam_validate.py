"""End-to-end seam validation: descent-accept (leg-1) -> KV relocate (leg-2).

PR #71 continued-gen tree-verify path. The two GPU kernels are each standalone-
validated in their own modules:
  - ``tree_accept_kernel.py``  -- the descending accept walk; now ALSO emits the
    on-device ``commit_map`` (accepted node path, identity past valid_count).
  - ``tree_kv_relocate.py``    -- the fused KV relocate that compacts the accepted
    path's K/V into the first ``len(path)`` contiguous slots.

This module proves they COMPOSE: the descent kernel's emitted ``commit_map`` is fed
directly (on-device, no host readout) into the relocate, and the accepted path's
K/V lands bit-exactly in the contiguous dst slots vLLM's count-based retention
keeps. The only data crossing the seam is the on-device ``commit_map`` (int32) plus
a device-side int32->int64 cast -- so the whole descent->relocate step is sync-free
and CUDA-graph-capturable (proved here by capturing both kernels in ONE graph and
replaying with live-mutated drafter inputs).

Greedy-identity is preserved by construction across the seam: the descent commits
only verifier-argmax tokens (it changes HOW MANY commit, never WHICH), and the
relocate is a pure bf16 copy (bit-identical K/V). Nothing here can alter a token.

Run standalone:
  CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python scripts/profiler/tree_seam_validate.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_accept_kernel import _tree_accept_kernel, tree_accept  # noqa: E402
from tree_kv_relocate import (  # noqa: E402
    N_LAYERS,
    _make_kv_list,
    _read_slot_list,
    _reference_relocated,
    build_commit_map,
    make_relocate_buffers,
    relocate_salvaged_kv_fused,
)
from tree_spec import (  # noqa: E402
    PARENT_M16,
    PARENT_M32,
    TreeSpec,
    build_children_csr,
    descend_accept,
    descend_accept_path,
)


def _nb(base: int, m: int, block_size: int) -> int:
    """Blocks needed so slots [0, base+m) all exist (mirror tree_kv_relocate)."""
    return -(-(base + m) // block_size) + 1


def _check_seam_eager(name: str, tree: TreeSpec, base: int, block_size: int, seed: int) -> None:
    """Descent kernel -> commit_map -> fused relocate, eager. Proves the accepted
    path's K/V is compacted into contiguous dst slots, bit-exact, AND the trailing
    (identity) slots are untouched -- i.e. the two legs actually interoperate."""
    dev = torch.device("cuda")
    m = tree.num_nodes
    width = tree.max_depth + 1
    ptr, idx = build_children_csr(tree)
    children_ptr = torch.tensor(ptr, dtype=torch.int32, device=dev)
    children_idx = torch.tensor(idx, dtype=torch.int32, device=dev)

    g = torch.Generator().manual_seed(seed)
    node_argmax = torch.randint(0, 4, (m,), generator=g, dtype=torch.int64)
    draft_token = torch.randint(0, 4, (m,), generator=g, dtype=torch.int32)
    ref_path = descend_accept_path(tree, node_argmax.tolist(), draft_token.tolist())
    _, ref_vc, _ = descend_accept(tree, node_argmax.tolist(), draft_token.tolist())

    # leg-1: descent kernel emits the on-device commit_map
    out = torch.full((1, m + 1), -999, dtype=torch.int32, device=dev)
    _, vc, cmap = tree_accept(
        out, node_argmax.to(dev), draft_token.to(dev),
        children_ptr, children_idx, m, m, width,
    )
    torch.cuda.synchronize()
    vc_i = int(vc[0].item())
    exp_cmap = ref_path + list(range(len(ref_path), width))
    assert cmap[0].tolist() == exp_cmap, (name, cmap[0].tolist(), exp_cmap)
    assert vc_i == ref_vc, (name, vc_i, ref_vc)

    # leg-2: feed that SAME on-device commit_map (int32 -> int64 device cast) to the
    # fused relocate -- this is the seam (no host readout between the two kernels).
    kv = _make_kv_list(_nb(base, m, block_size), block_size, dev, seed=seed + 100)
    orig = [t.clone() for t in kv]
    commit_map = cmap[0].to(torch.int64)  # device-side cast, graph-capturable
    ref = _reference_relocated(orig, base, commit_map, block_size)
    layer_ptrs, staging = make_relocate_buffers(kv, width)
    relocate_salvaged_kv_fused(kv, base, commit_map, block_size, layer_ptrs, staging)
    torch.cuda.synchronize()

    ok_ref = all(torch.equal(kv[li], ref[li]) for li in range(N_LAYERS))
    assert ok_ref, f"{name}: fused relocate driven by kernel commit_map != reference"
    # direct seam proof: accepted node path[j]'s K/V is now in contiguous slot j;
    # the identity tail (slots >= vc) is a no-op (unchanged from orig).
    for j in range(vc_i):
        got = _read_slot_list(kv, base + j, block_size)
        want = _read_slot_list(orig, base + ref_path[j], block_size)
        assert torch.equal(got, want), f"{name}: slot {j} != node {ref_path[j]} K/V"
    for j in range(vc_i, width):
        got = _read_slot_list(kv, base + j, block_size)
        want = _read_slot_list(orig, base + j, block_size)
        assert torch.equal(got, want), f"{name}: identity slot {j} was mutated"
    print(
        f"  [ok] {name}: descent commit_map={cmap[0].tolist()} -> fused relocate "
        f"compacts path {ref_path} (len {vc_i}) into contiguous slots, bit-exact; "
        f"identity tail untouched"
    )


def _check_seam_branch_salvage(base: int, block_size: int) -> None:
    """The signature case: rank-1 spine MISSES, a rank-2 branch salvages. Prove the
    SALVAGED branch node's K/V (a scattered verify row) is what lands in the
    contiguous slot -- the exact thing the linear break-on-mismatch path can't do."""
    dev = torch.device("cuda")
    tree = TreeSpec(PARENT_M16)
    m = tree.num_nodes
    width = tree.max_depth + 1
    ptr, idx = build_children_csr(tree)
    children_ptr = torch.tensor(ptr, dtype=torch.int32, device=dev)
    children_idx = torch.tensor(idx, dtype=torch.int32, device=dev)

    # force g[0]==draft[2] (rank-2 branch) != draft[1] (spine), then descend node5.
    node_argmax = [900 + i for i in range(m)]
    draft_token = [500 + i for i in range(m)]
    node_argmax[0] = draft_token[2]
    node_argmax[2] = draft_token[5]
    ref_path = descend_accept_path(tree, node_argmax, draft_token)
    assert ref_path == [0, 2, 5], ref_path

    out = torch.full((1, m + 1), -999, dtype=torch.int32, device=dev)
    _, vc, cmap = tree_accept(
        out, torch.tensor(node_argmax, dtype=torch.int64, device=dev),
        torch.tensor(draft_token, dtype=torch.int32, device=dev),
        children_ptr, children_idx, m, m, width,
    )
    torch.cuda.synchronize()

    kv = _make_kv_list(_nb(base, m, block_size), block_size, dev, seed=314)
    orig = [t.clone() for t in kv]
    commit_map = cmap[0].to(torch.int64)
    layer_ptrs, staging = make_relocate_buffers(kv, width)
    relocate_salvaged_kv_fused(kv, base, commit_map, block_size, layer_ptrs, staging)
    torch.cuda.synchronize()

    # slot 1 must now hold node 2's (the salvaged rank-2 branch) K/V, NOT node 1's.
    got1 = _read_slot_list(kv, base + 1, block_size)
    node2 = _read_slot_list(orig, base + 2, block_size)
    node1 = _read_slot_list(orig, base + 1, block_size)
    assert torch.equal(got1, node2), "salvaged branch node 2 K/V did not reach slot 1"
    assert not torch.equal(node1, node2), "degenerate: node1==node2 K/V"
    # slot 2 holds node 5 (the branch's continuation).
    assert torch.equal(_read_slot_list(kv, base + 2, block_size), _read_slot_list(orig, base + 5, block_size))
    print(
        f"  [ok] branch-salvage seam: commit_map={cmap[0].tolist()} -> slot1<-node2 "
        f"(rank-2 branch), slot2<-node5; scattered salvage path compacted bit-exact"
    )


def _check_seam_graph(name: str, tree: TreeSpec, base: int, block_size: int, seed: int) -> None:
    """Capture BOTH kernels (descent + cast + fused relocate) in ONE CUDA graph, then
    replay with LIVE-mutated drafter inputs -- proving the whole descent->relocate
    step is sync-free (the on-device commit_map flows kernel-to-kernel with no host
    readout, the only thing that would break capture)."""
    dev = torch.device("cuda")
    m = tree.num_nodes
    width = tree.max_depth + 1
    ptr, idx = build_children_csr(tree)
    children_ptr = torch.tensor(ptr, dtype=torch.int32, device=dev)
    children_idx = torch.tensor(idx, dtype=torch.int32, device=dev)
    nb = _nb(base, m, block_size)

    # fixed-pointer graph state (mutated only via .copy_ between replays)
    node_argmax_buf = torch.zeros(m, dtype=torch.int64, device=dev)
    draft_token_buf = torch.zeros(m, dtype=torch.int32, device=dev)
    out_buf = torch.full((1, m + 1), -999, dtype=torch.int32, device=dev)
    nti_buf = torch.empty(1, dtype=torch.int32, device=dev)
    vc_buf = torch.empty(1, dtype=torch.int32, device=dev)
    cmap_buf = torch.empty((1, width), dtype=torch.int32, device=dev)
    cmap_i64 = torch.empty(width, dtype=torch.int64, device=dev)
    kv_clean = _make_kv_list(nb, block_size, dev, seed=seed + 7)
    work = [t.clone() for t in kv_clean]
    layer_ptrs, staging = make_relocate_buffers(work, width)

    def step() -> None:
        _tree_accept_kernel[(1,)](
            out_buf, nti_buf, vc_buf, cmap_buf, node_argmax_buf, draft_token_buf,
            children_ptr, children_idx, m, m, width,
        )
        cmap_i64.copy_(cmap_buf[0])  # device int32->int64 (the seam carry)
        relocate_salvaged_kv_fused(work, base, cmap_i64, block_size, layer_ptrs, staging)

    g = torch.Generator().manual_seed(seed)
    warm = torch.randint(0, 4, (m,), generator=g, dtype=torch.int64)
    node_argmax_buf.copy_(warm)
    draft_token_buf.copy_(warm.to(torch.int32))  # warmup inputs (some matches)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for li in range(N_LAYERS):
                work[li].copy_(kv_clean[li])
            step()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    # replay several DIFFERENT drafter inputs purely by mutating the input buffers;
    # each must reproduce the eager descent->relocate reference (no recapture).
    for trial in range(4):
        na = torch.randint(0, 4, (m,), generator=g, dtype=torch.int64)
        dt = torch.randint(0, 4, (m,), generator=g, dtype=torch.int32)
        path = descend_accept_path(tree, na.tolist(), dt.tolist())
        ref = _reference_relocated(kv_clean, base, build_commit_map(path, width, dev), block_size)
        node_argmax_buf.copy_(na)
        draft_token_buf.copy_(dt)
        for li in range(N_LAYERS):
            work[li].copy_(kv_clean[li])
        graph.replay()
        torch.cuda.synchronize()
        assert cmap_buf[0].tolist() == path + list(range(len(path), width)), (
            name, trial, cmap_buf[0].tolist(), path
        )
        ok = all(torch.equal(work[li], ref[li]) for li in range(N_LAYERS))
        assert ok, f"{name} graph replay trial {trial}: descent->relocate != reference"
    print(
        f"  [ok] {name}: descent+relocate captured in ONE graph; 4 live-mutated drafter "
        f"inputs replayed -> commit_map + compacted KV bit-exact (sync-free seam holds)"
    )


def main() -> None:
    assert torch.cuda.is_available(), "needs a GPU (CUDA_VISIBLE_DEVICES=0)"
    print("=== tree_seam_validate: descent-accept (leg-1) -> KV relocate (leg-2) ===")
    base, bs = 500, 64
    print("-- eager seam (commit_map -> fused relocate compaction, bit-exact) --")
    _check_seam_eager("M16", TreeSpec(PARENT_M16), base, bs, seed=2)
    _check_seam_eager("M32", TreeSpec(PARENT_M32), base, bs, seed=3)
    _check_seam_eager("M32@base997", TreeSpec(PARENT_M32), 997, bs, seed=5)
    _check_seam_branch_salvage(base, bs)
    print("-- one-graph capture of BOTH kernels (sync-free seam) --")
    _check_seam_graph("M16", TreeSpec(PARENT_M16), base, bs, seed=11)
    _check_seam_graph("M32", TreeSpec(PARENT_M32), base, bs, seed=13)
    print("=== all tree_seam_validate checks passed (legs 1 and 2 compose) ===")


if __name__ == "__main__":
    main()
