"""GPU Triton twin of ``tree_spec.descend_accept`` -- the descending tree-accept
walk that replaces the deployed linear break-on-mismatch
``_dixie_fused_accept_prep_kernel`` (PR #71, Component 4 / BUG-2).

The deployed accept kernel walks draft positions ``0,1,2,...`` linearly and stops
on the first mismatch. On a flat tree layout the rank-2/3 sibling branches (all
children of one parent) live at later positions, so the break never reaches them
-> chain-rejection -> ~3% salvage (the byteshark/cheesetaco failure signature).
This kernel instead descends the tree: at each accepted node it scans ALL
children (rank-1 spine + rank-2/3 branches) for the verifier argmax and follows
the first match, so a rank-2 branch salvages ~rho2 (0.4165) of first-divergences.

Greedy identity preserved by construction: every emitted token is a verifier
argmax (``node_argmax``); the tree only changes HOW MANY are committed per step,
never which token the verifier authoritatively chooses.

The kernel also emits the on-device ``commit_map`` (the accepted node path, width
W = max_depth + 1, identity past ``valid_count``) that the leg-2 KV relocate
(``tree_kv_relocate.relocate_salvaged_kv_fused``) consumes sync-free -- so the
descent walk and the KV compaction compose with no host readout between them.

Run standalone (validates the kernel bit-for-bit vs the CPU reference):
  CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python scripts/profiler/tree_accept_kernel.py
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
    build_children_csr,
    descend_accept,
    descend_accept_path,
    linear_parent,
)


@triton.jit(do_not_specialize=["num_nodes", "max_spec_len", "commit_width"])
def _tree_accept_kernel(
    output_token_ids_ptr,  # [batch, max_spec_len + 1] int32
    next_token_ids_ptr,  # [batch] int32
    valid_counts_ptr,  # [batch] int32
    commit_map_ptr,  # [batch * commit_width] int32 -- accepted node path; identity past count
    node_argmax_ptr,  # [batch * num_nodes] -- g[node] in node order, per request
    draft_token_ptr,  # [batch * num_nodes] -- d[node] (d[0] unused)
    children_ptr_ptr,  # [num_nodes + 1] CSR offsets (static, shared)
    children_idx_ptr,  # [num_edges] CSR child ids in rank order (static, shared)
    num_nodes,
    max_spec_len,
    commit_width,
) -> None:
    req_idx = tl.program_id(0)
    node_base = req_idx * num_nodes
    row_offset = req_idx * (max_spec_len + 1)
    cmap_base = req_idx * commit_width

    # commit_map = the node path the leg-2 relocate consumes (tree_kv_relocate):
    # slot j <- accepted node j, identity (j) past num_accepted so it is a fixed
    # width-W op needing no host-side length (the sync-free relocate contract).
    # Pre-fill identity; the descent below overwrites slots [0:count] with the
    # real path. (commit_width must be >= max committed = max_depth + 1.)
    for j in range(commit_width):
        tl.store(commit_map_ptr + cmap_base + j, j)

    current = 0
    count = 0
    rejected = False
    next_token_id = tl.load(node_argmax_ptr + node_base).to(tl.int32)

    # Bounded by num_nodes (descent depth + 1 <= num_nodes always). The
    # ``if not rejected`` guard turns post-stop iterations into no-ops -- the
    # same proven control-flow shape as the deployed linear accept kernel.
    for _ in range(num_nodes):
        if not rejected:
            g = tl.load(node_argmax_ptr + node_base + current).to(tl.int32)
            tl.store(output_token_ids_ptr + row_offset + count, g)
            tl.store(commit_map_ptr + cmap_base + count, current)
            count += 1
            next_token_id = g

            cstart = tl.load(children_ptr_ptr + current)
            cend = tl.load(children_ptr_ptr + current + 1)
            matched = -1
            # First child (rank order) whose drafted token == verifier argmax.
            for ci in range(cstart, cend):
                child = tl.load(children_idx_ptr + ci)
                d = tl.load(draft_token_ptr + node_base + child).to(tl.int32)
                if (d == g) and (matched < 0):
                    matched = child
            if matched < 0:
                rejected = True
            else:
                current = matched

    tl.store(next_token_ids_ptr + req_idx, next_token_id)
    tl.store(valid_counts_ptr + req_idx, count)


def tree_accept(
    output_token_ids: torch.Tensor,
    node_argmax: torch.Tensor,
    draft_token: torch.Tensor,
    children_ptr: torch.Tensor,
    children_idx: torch.Tensor,
    num_nodes: int,
    max_spec_len: int,
    commit_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Host wrapper. ``node_argmax``/``draft_token`` are [batch*num_nodes] in node
    order; the CSR arrays are the static tree topology. Returns
    ``(next_token_ids[batch], valid_counts[batch], commit_map[batch, commit_width])``
    and fills ``output_token_ids`` rows ``[:valid_count]`` with the committed greedy
    tokens. ``commit_map`` is the on-device node path the leg-2 relocate
    (``tree_kv_relocate.relocate_salvaged_kv_fused``) consumes: slot j <- accepted
    node j, identity (j) past ``valid_count``. ``commit_width`` must be >= the max
    committed length (= max_depth + 1 = the relocate width); the served path uses
    exactly that width."""
    batch = int(output_token_ids.shape[0])
    dev = output_token_ids.device
    next_token_ids = torch.empty((batch,), dtype=torch.int32, device=dev)
    valid_counts = torch.empty((batch,), dtype=torch.int32, device=dev)
    commit_map = torch.empty((batch, commit_width), dtype=torch.int32, device=dev)
    _tree_accept_kernel[(batch,)](
        output_token_ids,
        next_token_ids,
        valid_counts,
        commit_map,
        node_argmax,
        draft_token,
        children_ptr,
        children_idx,
        num_nodes,
        max_spec_len,
        commit_width,
    )
    return next_token_ids, valid_counts, commit_map


def _test_tree(tree: TreeSpec, name: str, trials: int, seed: int) -> None:
    """Validate the GPU kernel bit-for-bit against tree_spec.descend_accept on
    random (node_argmax, draft_token) -- a small token-id range forces frequent
    matches so deep descents + branch salvages are exercised."""
    dev = torch.device("cuda")
    m = tree.num_nodes
    ptr, idx = build_children_csr(tree)
    children_ptr = torch.tensor(ptr, dtype=torch.int32, device=dev)
    children_idx = torch.tensor(idx, dtype=torch.int32, device=dev)
    max_spec_len = m  # generous output row width (committed <= depth+1 <= m)
    g = torch.Generator().manual_seed(seed)

    cw = tree.max_depth + 1  # the leg-2 relocate width (spine length)
    n_full, n_mismatch = 0, 0
    max_committed = 0
    for t in range(trials):
        node_argmax = torch.randint(0, 5, (m,), generator=g, dtype=torch.int64)
        draft_token = torch.randint(0, 5, (m,), generator=g, dtype=torch.int32)
        ref_emit, ref_vc, _ = descend_accept(
            tree, node_argmax.tolist(), draft_token.tolist()
        )
        ref_path = descend_accept_path(tree, node_argmax.tolist(), draft_token.tolist())
        assert len(ref_path) == ref_vc, (name, t, len(ref_path), ref_vc)
        out = torch.full((1, max_spec_len + 1), -999, dtype=torch.int32, device=dev)
        nti, vc, cmap = tree_accept(
            out,
            node_argmax.to(dev),
            draft_token.to(dev),
            children_ptr,
            children_idx,
            m,
            max_spec_len,
            cw,
        )
        torch.cuda.synchronize()
        vc_i = int(vc[0].item())
        assert vc_i == ref_vc, f"{name}[{t}] valid_count {vc_i} != ref {ref_vc}"
        assert int(nti[0].item()) == ref_emit[-1], (
            f"{name}[{t}] next_token {int(nti[0])} != ref {ref_emit[-1]}"
        )
        got = out[0, :vc_i].tolist()
        assert got == ref_emit, f"{name}[{t}] emitted {got} != ref {ref_emit}"
        # commit_map = accepted node path then identity-fill (the relocate's input)
        exp_cmap = ref_path + list(range(len(ref_path), cw))
        got_cmap = cmap[0].tolist()
        assert got_cmap == exp_cmap, f"{name}[{t}] commit_map {got_cmap} != {exp_cmap}"
        max_committed = max(max_committed, vc_i)
        if ref_vc == tree.max_depth + 1:
            n_full += 1
        else:
            n_mismatch += 1
    print(
        f"  [ok] {name}: {trials} trials kernel==reference (emit+valid_count+commit_map; "
        f"full-accept={n_full}, stopped={n_mismatch}, max_committed={max_committed}, "
        f"depth+1={tree.max_depth + 1})"
    )


def _test_branch_salvage_explicit() -> None:
    """Deterministic proof the kernel REACHES a rank-2 branch when the rank-1
    spine misses -- the exact property the linear break-on-mismatch kernel lacks
    (its 3%-salvage bug). M16 root (node 0) has children [1 (spine), 2 (branch)].
    Force g[0] == draft[2] != draft[1]: a correct descend walk must salvage into
    node 2 (NOT stop like the linear kernel would)."""
    dev = torch.device("cuda")
    tree = TreeSpec(PARENT_M16)
    m = tree.num_nodes
    assert tree.children[0] == [1, 2], tree.children[0]
    ptr, idx = build_children_csr(tree)
    children_ptr = torch.tensor(ptr, dtype=torch.int32, device=dev)
    children_idx = torch.tensor(idx, dtype=torch.int32, device=dev)

    node_argmax = [900 + i for i in range(m)]  # default: no further matches
    draft_token = [500 + i for i in range(m)]
    node_argmax[0] = draft_token[2]  # root argmax == rank-2 branch token...
    assert draft_token[1] != node_argmax[0]  # ...and != rank-1 spine token
    # node 2's only child is node 5 (a depth-2 continuation of the branch):
    assert tree.children[2] == [5], tree.children[2]
    node_argmax[2] = draft_token[5]  # let the branch descend one more level

    ref_emit, ref_vc, ref_sal = descend_accept(tree, node_argmax, draft_token)
    # expect: emit g[0], descend node2, emit g[2], descend node5, emit g[5], stop.
    assert ref_emit == [node_argmax[0], node_argmax[2], node_argmax[5]], ref_emit
    assert ref_sal == [(0, 2)], ref_sal  # one rank-2 salvage at spine depth 0

    cw = tree.max_depth + 1
    out = torch.full((1, m + 1), -999, dtype=torch.int32, device=dev)
    nti, vc, cmap = tree_accept(
        out,
        torch.tensor(node_argmax, dtype=torch.int64, device=dev),
        torch.tensor(draft_token, dtype=torch.int32, device=dev),
        children_ptr,
        children_idx,
        m,
        m,
        cw,
    )
    torch.cuda.synchronize()
    got = out[0, : int(vc[0].item())].tolist()
    assert got == ref_emit, f"kernel salvage path {got} != ref {ref_emit}"
    assert int(vc[0].item()) == 3 and int(nti[0].item()) == node_argmax[5]
    ref_path = descend_accept_path(tree, node_argmax, draft_token)
    assert ref_path == [0, 2, 5], ref_path  # root -> rank-2 branch -> its child
    exp_cmap = ref_path + list(range(len(ref_path), cw))
    assert cmap[0].tolist() == exp_cmap, (cmap[0].tolist(), exp_cmap)
    print(
        f"  [ok] explicit rank-2 branch salvage: kernel descended spine-miss into "
        f"node 2 (branch) then node 5, committed {got}, commit_map={cmap[0].tolist()} "
        f"(linear kernel would stop at 1)"
    )


def main() -> None:
    assert torch.cuda.is_available(), "needs a GPU (CUDA_VISIBLE_DEVICES=0)"
    print("=== tree_accept_kernel GPU validation (vs tree_spec.descend_accept) ===")
    _test_tree(TreeSpec(linear_parent(8)), "lin8", trials=300, seed=1)
    _test_tree(TreeSpec(PARENT_M16), "M16", trials=400, seed=2)
    _test_tree(TreeSpec(PARENT_M32), "M32", trials=400, seed=3)
    _test_branch_salvage_explicit()
    print("=== all tree_accept_kernel GPU checks passed ===")


if __name__ == "__main__":
    main()
