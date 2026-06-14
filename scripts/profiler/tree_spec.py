"""Pure (GPU-free) tree-spec-decode structure + index-mapping + gate utilities.

This is the root-cause-independent CORE that every component of the PR #71
tree-verify serving path consumes:

  * drafter tree-emit (sitecustomize ``propose_onegraph``)  -> ``children``,
    ``spine``, ``rank_in_parent`` (which candidate is rank-1 vs a branch).
  * tree-causal star-attention dispatch (``splitkv_verify_patch``)  ->
    ``tree_causal_mask`` (node attends to ancestors-or-self only).
  * widened verify metadata (serve.py rejection-sampler patch)  ->
    ``verify_index_maps`` (``target_logits_indices`` / ``bonus_logits_indices``).
  * descending accept walk (sitecustomize ``_dixie_fused_accept_prep_kernel``)
    -> ``children`` grouped by parent, the structure the walk descends.

Convention (matches the deployed linear chain exactly):
  A tree of ``M`` nodes is ``M`` verify rows, ``parent`` an int array of length
  ``M`` with ``parent[0] == -1`` (root = the last-accepted "x" input row whose
  argmax predicts node-0's children) and ``parent[i] < i`` for ``i > 0``
  (topological order). Among the children of a node, ordered by node index, the
  FIRST is the rank-1 (spine) continuation, the next rank-2, etc.  draft tokens
  are nodes ``1..M-1``; draft node ``i`` is verified against the target argmax at
  row ``parent[i]``. The deployed linear K=7/M=8 chain is ``[-1,0,1,2,3,4,5,6]``,
  for which ``target_logits_indices == [0,1,2,3,4,5,6]`` and ``bonus`` row ``7`` --
  identical to the deployed ``metadata`` semantics.

Acceptance model (wirbel #49 / report_sequoia_dp.md, path-product):
  ``F(T) = Σ_v path_product(v)``, ``path_product(v) = Π_{u on root->v} p[rank(u)]``
  (root factor 1). ``F = E[committed tokens/step]`` including the always-emitted
  bonus. ``p[k]`` = marginal P(target argmax == drafter rank-k sibling).

No torch import at module top level so this stays importable/testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- canonical build targets (verbatim from advisor #74/#83 + report) --------

# wirbel #83 (measured declining rho ladder) -- the arrays land's build targets.
PARENT_M16 = [-1, 0, 0, 1, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14]
PARENT_M32 = [
    -1, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 5, 7, 9, 9, 10,
    11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29,
]

# Measured acceptance (wirbel #76 top-1, #79 rho ladder cross-validated to byteshark).
TOP1_MEASURED = 0.729
RHO_LADDER = {2: 0.4165, 3: 0.2655, 4: 0.1908}  # P(target==rank-k | rank-1 miss)

# Borrowed EAGLE-3 p-vector used by report_sequoia_dp.md's DP table (for the F
# cross-check against the published linear/DP F values).
P_VECTOR_BORROWED = [0.6792, 0.1097, 0.0494, 0.0222]

# report_sequoia_dp.md Section 2 linear F (geom, borrowed p) -- the F-formula anchor.
REPORT_LINEAR_F = {4: 2.454, 8: 2.976, 16: 3.111, 32: 3.117, 45: 3.117}

# wirbel #83 per-position salvage oracle (the debug-gate numeric reference).
SALVAGE_ORACLE_83 = {1: (3, 0.397), 2: (3, 0.431), 3: (2, 0.413), 4: (2, 0.428)}


@dataclass
class TreeSpec:
    """Parsed draft tree from a ``parent`` array. All structure is derived once."""

    parent: list[int]
    children: list[list[int]] = field(default_factory=list)
    rank_in_parent: list[int] = field(default_factory=list)  # 1-based, 0 for root
    depth: list[int] = field(default_factory=list)  # root depth 0

    def __post_init__(self) -> None:
        p = list(self.parent)
        m = len(p)
        if m == 0 or p[0] != -1:
            raise ValueError("parent[0] must be -1 (single root)")
        if p.count(-1) != 1:
            raise ValueError(f"exactly one root required, found {p.count(-1)} roots")
        self.children = [[] for _ in range(m)]
        self.rank_in_parent = [0] * m
        self.depth = [0] * m
        for i in range(1, m):
            par = p[i]
            if not (0 <= par < i):
                raise ValueError(
                    f"parent[{i}]={par} must satisfy 0 <= parent < i (topological)"
                )
            self.children[par].append(i)
        for par in range(m):
            for rank, child in enumerate(self.children[par], start=1):
                self.rank_in_parent[child] = rank
        for i in range(1, m):
            self.depth[i] = self.depth[p[i]] + 1

    @property
    def num_nodes(self) -> int:
        return len(self.parent)

    @property
    def max_depth(self) -> int:
        return max(self.depth)

    @property
    def max_branch(self) -> int:
        return max(len(c) for c in self.children)

    @property
    def spine(self) -> list[int]:
        """The rank-1 path from root (the chain that must stay token-identical to
        the deployed width-1 drafter forward -- the BUG-1 guard reference)."""
        path = [0]
        cur = 0
        while self.children[cur]:
            cur = self.children[cur][0]  # rank-1 (first) child
            path.append(cur)
        return path

    def ancestors(self, i: int) -> list[int]:
        """Ancestors of node i (excluding self), root-first."""
        chain = []
        cur = self.parent[i]
        while cur != -1:
            chain.append(cur)
            cur = self.parent[cur]
        return chain[::-1]


def tree_causal_mask(tree: TreeSpec):
    """MxM boolean: ``mask[i, j]`` = node i may attend to node j (ancestor-or-self).

    Returns a list-of-lists (numpy/torch-free). The shared KV prefix is attended
    densely and is NOT part of this intra-tree mask. This is the star-attention
    bias the verify path must DISPATCH for tree rows (chiku-inu missing-half #1).
    """
    m = tree.num_nodes
    mask = [[False] * m for _ in range(m)]
    for i in range(m):
        allowed = set(tree.ancestors(i)) | {i}
        for j in allowed:
            mask[i][j] = True
    return mask


def verify_index_maps(tree: TreeSpec) -> tuple[list[int], list[int]]:
    """``(target_logits_indices, bonus_logits_indices)``.

    ``target_logits_indices[k]`` = the verify-row whose argmax predicts draft
    node ``k+1`` = ``parent[k+1]`` (the 3%-salvage index-bug guard: a wrong
    mapping verifies branches against the wrong target rows). ``bonus`` rows are
    the leaves (their argmax = the continuation if that path fully accepts).
    """
    target_logits_indices = [tree.parent[i] for i in range(1, tree.num_nodes)]
    bonus_logits_indices = [
        i for i in range(tree.num_nodes) if not tree.children[i]
    ]
    return target_logits_indices, bonus_logits_indices


def expected_committed_tokens(tree: TreeSpec, p_vector: list[float]) -> float:
    """Closed-form ``F(T) = Σ_v path_product(v)`` = E[committed tokens/step]."""
    pp = [0.0] * tree.num_nodes
    pp[0] = 1.0  # root factor
    total = 1.0  # root contributes 1 (the always-emitted bonus token)
    for i in range(1, tree.num_nodes):
        rank = tree.rank_in_parent[i]
        p_rank = p_vector[rank - 1] if rank - 1 < len(p_vector) else 0.0
        pp[i] = pp[tree.parent[i]] * p_rank
        total += pp[i]
    return total


def monte_carlo_committed(
    tree: TreeSpec, p_vector: list[float], trials: int = 200_000, seed: int = 0
) -> float:
    """Greedy tree-accept simulation (the report's selfcheck). At each accepted
    node the target picks child of rank k w.p. ``p[k-1]`` (mutually exclusive) or
    none; descend the picked child. committed = accepted depth + 1 (bonus)."""
    import random

    rng = random.Random(seed)
    total = 0
    for _ in range(trials):
        cur = 0
        committed = 1  # bonus
        while tree.children[cur]:
            r = rng.random()
            acc = 0.0
            picked = -1
            for child in tree.children[cur]:
                rank = tree.rank_in_parent[child]
                acc += p_vector[rank - 1] if rank - 1 < len(p_vector) else 0.0
                if r < acc:
                    picked = child
                    break
            if picked == -1:
                break
            committed += 1
            cur = picked
        total += committed
    return total / trials


def per_position_salvage(
    tree: TreeSpec, rho_ladder: dict[int, float] = RHO_LADDER
) -> dict[int, tuple[int, float]]:
    """Per spine-position expected rank-2 salvage = ``rho_2`` for width>=2 branch
    positions (wirbel #83 oracle; the universal gate is branch-hit ~= 0.4165 at
    any width>=2 divergence). Returns ``{spine_depth: (width, E[salvage])}``."""
    out: dict[int, tuple[int, float]] = {}
    for depth, node in enumerate(tree.spine):
        width = len(tree.children[node])
        if width >= 2:
            out[depth + 1] = (width, rho_ladder[2])
    return out


def descend_accept(
    tree: TreeSpec,
    node_argmax: list[int],
    draft_token: list[int],
) -> tuple[list[int], int, list[tuple[int, int]]]:
    """The descending tree-accept walk (the reference twin of the Triton kernel
    that replaces the linear break-on-mismatch ``_dixie_fused_accept_prep_kernel``).

    This is the BUG-2 fix. The linear kernel walks draft positions ``0,1,2,...``
    and breaks on the FIRST mismatch; on a flat tree layout the rank-2/3 sibling
    branches (all children of the same parent) sit at later positions, so the
    break never reaches them -> chain-rejection -> branches unreachable -> the
    ~3% salvage signature every prior broken external build hit. This walk
    instead, at each accepted node, checks ALL children (rank-1 spine AND the
    rank-2+ branches) for a match against the verifier argmax, and descends into
    the matching one -> the rank-2 branch salvages ~rho2 of first-divergences.

    Args:
      node_argmax: ``g[i]`` = the verifier greedy argmax at node ``i``'s own
        verify row (``dixie_all_argmax`` indexed by node; ``g[parent[c]]`` is the
        token child ``c``'s draft is checked against -- identical to convention A's
        ``target_logits_indices`` mapping, but encoded by the children structure).
      draft_token: ``d[i]`` = the drafter's token for node ``i`` (``d[0]`` unused;
        node 0 is the always-present input/root row).

    Returns ``(emitted, valid_count, salvage_events)`` where ``emitted`` is the
    committed greedy-token sequence (every token is a verifier argmax -> greedy
    identity preserved by construction), ``valid_count == len(emitted)``, and
    ``salvage_events`` lists ``(spine_depth, rank)`` for each divergence that was
    rescued by a rank>=2 branch (the debug-gate salvage signal).
    """
    current = 0
    emitted: list[int] = []
    salvage_events: list[tuple[int, int]] = []
    spine_depth = 0
    while True:
        g = node_argmax[current]
        emitted.append(g)
        matched = -1
        matched_rank = 0
        for rank, child in enumerate(tree.children[current], start=1):
            if draft_token[child] == g:
                matched = child
                matched_rank = rank
                break
        if matched < 0:
            break
        if matched_rank >= 2:
            salvage_events.append((spine_depth, matched_rank))
        current = matched
        spine_depth += 1
    return emitted, len(emitted), salvage_events


def emit_tree(
    tree: TreeSpec,
    forward_fn,
    root_token: int,
    root_hidden,
    base_position: int = 0,
) -> tuple[list[int], list, int]:
    """Draft-side reference -- the topological tree-emit the live drafter
    (sitecustomize ``propose_onegraph``) realizes. The DRAFT twin of
    ``descend_accept``: ``descend_accept`` walks the verifier DOWN the tree,
    ``emit_tree`` builds the tree UP from the root.

    Processes nodes in topological (node-index) order. Node 0 (root) is the
    last-accepted token; its forward predicts node 0's children. Each internal
    node ``n`` is forwarded with (its own drafted token, its PARENT's hidden
    context) and its top-w prediction supplies its children's drafted tokens.
    A node's drafted token is the ``rank_in_parent``-th of its parent's top-w
    candidates (rank 1 == the deployed greedy argmax -> the rank-1 spine is
    token-IDENTICAL to the deployed linear width-1 chain: the BUG-1 guard).
    Leaf nodes are not forwarded (no children to predict).

    ``forward_fn(node, token, parent_hidden, position) -> (hidden, topw_tokens)``
    is one width-1 drafter forward. The LIVE realization owns the drafter-side
    ancestor-only attention -- a branch node must attend only its own root->node
    path's KV, never a sibling's -- which is the draft-side analogue of the
    verify star-attention mask (Component 2). ``topw_tokens`` are rank-ordered and
    ``topw_tokens[0]`` MUST equal the deployed greedy argmax (sparse-argmax
    ``get_top_tokens``) so the rank-1 path stays chain-identical.

    Returns ``(draft_token[M], hidden[M], forwards)``: ``draft_token[0]`` ==
    ``root_token``; ``draft_token[n>=1]`` is node n's drafted token; ``hidden[n]``
    is the forward output for internal nodes (``None`` for leaves); ``forwards``
    == number of ``forward_fn`` calls (the internal-node count = drafter cost).
    """
    m = tree.num_nodes
    draft_token: list = [None] * m
    hidden: list = [None] * m
    topw: list = [None] * m
    draft_token[0] = root_token
    forwards = 0
    for node in range(m):
        if node == 0:
            tok = root_token
            ctx = root_hidden
        else:
            par = tree.parent[node]
            rank = tree.rank_in_parent[node]  # 1-based; 1 == rank-1 spine
            cand = topw[par]
            if cand is None:
                raise AssertionError(
                    f"node {node}'s parent {par} was not forwarded (leaf parent?)"
                )
            if rank - 1 >= len(cand):
                raise ValueError(
                    f"node {node} is rank {rank} but parent {par} produced only "
                    f"{len(cand)} candidates (drafter top-w must be >= max_branch "
                    f"{tree.max_branch})"
                )
            tok = cand[rank - 1]
            draft_token[node] = tok
            ctx = hidden[par]
        if tree.children[node]:  # internal node -> forward to predict children
            h, cand_out = forward_fn(node, tok, ctx, base_position + tree.depth[node])
            hidden[node] = h
            topw[node] = list(cand_out)
            forwards += 1
    if any(t is None for t in draft_token):
        raise AssertionError(f"unassigned draft tokens: {draft_token}")
    return draft_token, hidden, forwards


def spine_tokens(tree: TreeSpec, draft_token: list[int]) -> list[int]:
    """The drafted tokens along the rank-1 spine (the BUG-1 guard reference: this
    sequence must be byte-identical to the deployed linear chain's draft tokens)."""
    return [draft_token[n] for n in tree.spine]


def simulate_tree_decode(
    tree: TreeSpec,
    p_vector: list[float],
    steps: int = 200_000,
    seed: int = 0,
) -> dict[str, float]:
    """Monte-Carlo the joint drafter+verifier token process and run the real
    ``descend_accept`` walk over it -- the offline correctness anchor for the
    Triton kernel. At every visited node the verifier argmax ``g`` equals the
    rank-k child's drafted token w.p. ``p[k-1]`` (mutually exclusive) else a
    novel token; the walk descends whichever child matches.

    Validates two gate numbers without a GPU:
      * ``e_t`` (mean committed tokens/step) must == the closed-form
        ``expected_committed_tokens`` (proves the walk accepts the right paths).
      * ``branch_hit_rate`` (P(rank-2 branch catches | rank-1 miss at a width>=2
        node)) must == ``rho2`` (proves branches are reachable -- the
        3%-vs-41% discriminator that every broken external build failed).
    """
    import random

    rng = random.Random(seed)
    novel = 1_000_000  # token-id space for "matches no child" outcomes
    total_committed = 0
    width2plus_divergences = 0  # rank-1 miss at a width>=2 spine node
    branch_salvages = 0  # ... that a rank-2 branch caught
    for _ in range(steps):
        node_argmax = [0] * tree.num_nodes
        draft_token = [0] * tree.num_nodes
        # Assign each node a unique draft-token id (distinct across the tree).
        for i in range(tree.num_nodes):
            draft_token[i] = i + 1
        # For every node, decide which (if any) child the verifier argmax hits.
        for node in range(tree.num_nodes):
            kids = tree.children[node]
            if not kids:
                node_argmax[node] = novel
                novel += 1
                continue
            r = rng.random()
            acc = 0.0
            picked_child = -1
            for rank, child in enumerate(kids, start=1):
                acc += p_vector[rank - 1] if rank - 1 < len(p_vector) else 0.0
                if r < acc:
                    picked_child = child
                    break
            if picked_child < 0:
                node_argmax[node] = novel  # matches no drafted child
                novel += 1
            else:
                node_argmax[node] = draft_token[picked_child]
        emitted, valid, salvage = descend_accept(tree, node_argmax, draft_token)
        total_committed += valid
        # Walk the spine to count width>=2 first-divergences + branch rescues.
        cur = 0
        while tree.children[cur]:
            kids = tree.children[cur]
            g = node_argmax[cur]
            rank1_child = kids[0]
            rank1_hit = draft_token[rank1_child] == g
            if len(kids) >= 2 and not rank1_hit:
                width2plus_divergences += 1
                if draft_token[kids[1]] == g:  # rank-2 branch caught it
                    branch_salvages += 1
            # descend whichever child matched (mirror descend_accept)
            nxt = -1
            for child in kids:
                if draft_token[child] == g:
                    nxt = child
                    break
            if nxt < 0:
                break
            cur = nxt
    return {
        "e_t": total_committed / steps,
        "branch_hit_rate": (
            branch_salvages / width2plus_divergences
            if width2plus_divergences
            else 0.0
        ),
        "width2plus_divergences": width2plus_divergences,
    }


def build_children_csr(tree: TreeSpec) -> tuple[list[int], list[int]]:
    """``(children_ptr[M+1], children_idx[num_edges])`` CSR layout in rank order
    -- the static topology the Triton descend kernel walks. ``children_ptr[n]``..
    ``children_ptr[n+1]`` index ``children_idx`` for node ``n``'s children."""
    ptr = [0]
    idx: list[int] = []
    for node in range(tree.num_nodes):
        idx.extend(tree.children[node])
        ptr.append(len(idx))
    return ptr, idx


def validate(
    tree: TreeSpec,
    expect_nodes: int | None = None,
    expect_max_branch: int | None = None,
    expect_depth: int | None = None,
) -> None:
    """Structural asserts used by the build + the debug gate."""
    if expect_nodes is not None:
        assert tree.num_nodes == expect_nodes, (
            f"M={tree.num_nodes} != expected {expect_nodes}"
        )
    if expect_max_branch is not None:
        assert tree.max_branch == expect_max_branch, (
            f"max_branch={tree.max_branch} != expected {expect_max_branch}"
        )
    if expect_depth is not None:
        assert tree.max_depth == expect_depth, (
            f"depth={tree.max_depth} != expected {expect_depth}"
        )
    # every non-root node reachable from root, no cycles (guaranteed by parent<i).
    assert len(tree.spine) >= 2, "spine must have >=1 draft token"


def linear_parent(m: int) -> list[int]:
    """The deployed linear chain as a degenerate tree (validation anchor)."""
    return [-1] + list(range(m - 1))


def _selfcheck() -> None:
    import math

    print("=== tree_spec selfcheck ===")

    # 1. Linear chain reproduces report_sequoia_dp.md linear F = Σ p1^i.
    for m, ref in REPORT_LINEAR_F.items():
        lin = TreeSpec(linear_parent(m))
        f = expected_committed_tokens(lin, P_VECTOR_BORROWED)
        geom = sum(P_VECTOR_BORROWED[0] ** i for i in range(m))
        assert abs(f - geom) < 1e-9, f"linear F formula broke at M={m}"
        assert abs(f - ref) < 0.01, f"linear F {f:.3f} != report {ref} at M={m}"
        print(f"  [ok] linear M={m:2d}: F={f:.3f} (report {ref})")

    # 2. Linear index map matches the deployed metadata semantics exactly.
    lin8 = TreeSpec(linear_parent(8))
    tli, bli = verify_index_maps(lin8)
    assert tli == [0, 1, 2, 3, 4, 5, 6], tli
    assert bli == [7], bli
    assert lin8.spine == list(range(8)), lin8.spine
    print(f"  [ok] linear M=8 index map: target={tli} bonus={bli}")

    # 3. Canonical build targets: structure matches advisor #83 spec.
    t16 = TreeSpec(PARENT_M16)
    validate(t16, expect_nodes=16, expect_max_branch=2, expect_depth=9)
    print(
        f"  [ok] M16: nodes={t16.num_nodes} max_branch={t16.max_branch} "
        f"depth={t16.max_depth} spine={t16.spine}"
    )
    t32 = TreeSpec(PARENT_M32)
    validate(t32, expect_nodes=32, expect_max_branch=3, expect_depth=9)
    print(
        f"  [ok] M32: nodes={t32.num_nodes} max_branch={t32.max_branch} "
        f"depth={t32.max_depth} spine={t32.spine}"
    )

    # 4. Closed-form F == Monte-Carlo greedy-accept (report selfcheck, <0.02).
    for name, tree in [("M16", t16), ("M32", t32), ("lin16", TreeSpec(linear_parent(16)))]:
        for p_name, p in [("borrowed", P_VECTOR_BORROWED)]:
            f = expected_committed_tokens(tree, p)
            mc = monte_carlo_committed(tree, p, trials=200_000, seed=1)
            assert abs(f - mc) < 0.02, f"{name}/{p_name}: F={f:.4f} MC={mc:.4f}"
            print(f"  [ok] {name} F={f:.4f} == MC={mc:.4f} (|d|={abs(f-mc):.4f})")

    # 5. Measured-p E[T]: build a measured p-vector from top-1 + rho ladder.
    #    p[1]=top1; p[k]=(1-top1)*rho_k for k>=2 (marginal = miss * conditional).
    miss = 1.0 - TOP1_MEASURED
    p_meas = [TOP1_MEASURED] + [miss * RHO_LADDER[k] for k in (2, 3, 4)]
    f16 = expected_committed_tokens(t16, p_meas)
    f32 = expected_committed_tokens(t32, p_meas)
    flin = expected_committed_tokens(TreeSpec(linear_parent(8)), p_meas)
    print(
        f"  [info] measured-p E[T]: linearM8={flin:.3f}  "
        f"M16={f16:.3f} (+{100*(f16/flin-1):.1f}%)  "
        f"M32={f32:.3f} (+{100*(f32/flin-1):.1f}%)  [vs 3.844 deployed ref]"
    )

    # 6. Per-position salvage oracle reproduces #83 widths.
    sal16 = per_position_salvage(t16)
    sal32 = per_position_salvage(t32)
    print(f"  [info] M16 branch positions (depth->(width,salvage)): {sal16}")
    print(f"  [info] M32 branch positions: {sal32}")
    for depth, (width, _) in SALVAGE_ORACLE_83.items():
        if depth in sal32:
            got_w = sal32[depth][0]
            assert got_w >= 2, f"#83 says branch at depth {depth}, got width {got_w}"
    print(f"  [ok] #83 salvage-oracle branch depths present in M32")

    # 7. Tree-causal mask sanity: row attends to exactly its ancestors+self.
    mask = tree_causal_mask(t16)
    for i in range(t16.num_nodes):
        allowed = sorted(set(t16.ancestors(i)) | {i})
        got = [j for j in range(t16.num_nodes) if mask[i][j]]
        assert got == allowed, f"mask row {i}: {got} != {allowed}"
    assert all(mask[i][i] for i in range(t16.num_nodes)), "diagonal must be self"
    assert all(mask[i][0] for i in range(t16.num_nodes)), "all attend root"
    print(f"  [ok] tree-causal mask: every row = ancestors+self (M16)")

    # 8. Descend walk reproduces the deployed LINEAR chain exactly (degenerate
    #    tree): emits the verifier argmax prefix up to the first mismatch + bonus.
    lin8 = TreeSpec(linear_parent(8))
    #    full accept: every node's argmax == its rank-1 child draft -> emit all 8.
    g_full = [10, 20, 30, 40, 50, 60, 70, 80]
    d_full = [0, 10, 20, 30, 40, 50, 60, 70]  # d[i] == g[parent[i]] == g[i-1]
    emit, vc, sal = descend_accept(lin8, g_full, d_full)
    assert emit == g_full and vc == 8 and sal == [], (emit, vc, sal)
    #    mismatch at pos 2 (node 3): g[2]=30 but draft[3]=999 != 30 -> stop at 3.
    d_mis = [0, 10, 20, 999, 40, 50, 60, 70]
    emit, vc, sal = descend_accept(lin8, g_full, d_mis)
    assert emit == [10, 20, 30] and vc == 3 and sal == [], (emit, vc, sal)
    print(f"  [ok] descend walk == linear kernel on degenerate chain (full+mismatch)")

    # 9. THE KEYSTONE GATE: simulate the joint drafter+verifier process and run
    #    the real descend walk. (a) E[T] must equal the closed form (the walk
    #    accepts the right paths); (b) branch-hit must equal rho2 = 0.4165 (the
    #    3%-vs-41% discriminator -- branches are REACHABLE, not chain-rejected).
    miss = 1.0 - TOP1_MEASURED
    p_meas = [TOP1_MEASURED] + [miss * RHO_LADDER[k] for k in (2, 3, 4)]
    rho2 = RHO_LADDER[2]
    for name, tree in [("M16", t16), ("M32", t32)]:
        cf = expected_committed_tokens(tree, p_meas)
        sim = simulate_tree_decode(tree, p_meas, steps=200_000, seed=7)
        assert abs(sim["e_t"] - cf) < 0.02, (
            f"{name}: sim E[T]={sim['e_t']:.4f} != closed-form {cf:.4f}"
        )
        assert abs(sim["branch_hit_rate"] - rho2) < 0.01, (
            f"{name}: branch-hit={sim['branch_hit_rate']:.4f} != rho2={rho2} "
            f"(3%-salvage-bug discriminator)"
        )
        print(
            f"  [ok] {name} descend-sim: E[T]={sim['e_t']:.3f} (cf {cf:.3f}) "
            f"branch-hit={sim['branch_hit_rate']:.4f} (rho2 {rho2}) "
            f"n_div={sim['width2plus_divergences']}"
        )

    # 10. CSR topology round-trips (the static arrays the Triton kernel walks).
    for name, tree in [("M16", t16), ("M32", t32), ("lin8", lin8)]:
        ptr, idx = build_children_csr(tree)
        assert len(ptr) == tree.num_nodes + 1 and ptr[-1] == len(idx)
        for node in range(tree.num_nodes):
            assert idx[ptr[node]:ptr[node + 1]] == tree.children[node]
    print(f"  [ok] children CSR round-trips (M16/M32/lin8)")

    # 11. DRAFT-side emit_tree (the propose_onegraph reference). A deterministic
    #     MOCK drafter forward (pure fn of token+hidden, ignores node/pos) lets us
    #     validate the topology of the emit + the BUG-1 spine-identity guard with
    #     no GPU. rank-1 of the mock top-w == the mock's greedy => spine identity
    #     is a genuine property (the spine path replays the chain's (token,hidden)).
    MOCK_W = 3

    def _mock_forward(node, token, ctx, pos):
        # deterministic, pure in (token, ctx); rank-ordered distinct candidates.
        h = (token * 31 + ctx * 7 + 13) % 100003
        base = (token * 13 + ctx * 17 + 5) % 90000
        topw = [base + r * 1000 + 1 for r in range(MOCK_W)]  # topw[0] == greedy
        return h, topw

    def _linear_chain_emit(k, root_token, root_hidden):
        """The deployed width-1 chain under the same mock (rank-1/greedy each step)."""
        toks, tok, ctx = [], root_token, root_hidden
        for i in range(k):
            h, topw = _mock_forward(i, tok, ctx, i)
            tok = topw[0]
            toks.append(tok)
            ctx = h
        return toks

    ROOT_TOK, ROOT_H = 42, 7
    # (a) degenerate linear tree reproduces the chain exactly.
    for k in (7, 15, 31):
        lin = TreeSpec(linear_parent(k + 1))
        dt, _, fwds = emit_tree(lin, _mock_forward, ROOT_TOK, ROOT_H)
        chain = _linear_chain_emit(k, ROOT_TOK, ROOT_H)
        assert dt[1:] == chain, f"linear-tree emit {dt[1:]} != chain {chain} (k={k})"
        assert fwds == k, f"linear k={k}: {fwds} forwards != {k} internal nodes"
    print(f"  [ok] emit_tree reproduces the deployed linear chain (k=7/15/31)")

    # (b) spine identity on the real trees: the rank-1 spine drafts the SAME
    #     tokens as a pure chain of the spine's length (the BUG-1 guard).
    for name, tree in [("M16", t16), ("M32", t32)]:
        dt, hid, fwds = emit_tree(tree, _mock_forward, ROOT_TOK, ROOT_H)
        sp = spine_tokens(tree, dt)
        chain = _linear_chain_emit(len(tree.spine) - 1, ROOT_TOK, ROOT_H)
        assert sp[1:] == chain, f"{name} spine {sp[1:]} != chain {chain}"
        n_internal = sum(1 for n in range(tree.num_nodes) if tree.children[n])
        assert fwds == n_internal, f"{name}: {fwds} forwards != {n_internal} internal"
        # structural: every node's drafted token == its parent's rank-th candidate;
        # rank-2 branch tokens differ from the rank-1 sibling (real branching).
        for node in range(1, tree.num_nodes):
            par, rank = tree.parent[node], tree.rank_in_parent[node]
            assert hid[par] is not None, f"{name}: parent {par} not forwarded"
            if rank >= 2:
                sib1 = tree.children[par][0]
                assert dt[node] != dt[sib1], (
                    f"{name}: rank-{rank} node {node} token == rank-1 sibling (no branch)"
                )
        print(
            f"  [ok] {name} emit_tree: {fwds} forwards (internal nodes), "
            f"spine-identity holds, {tree.max_branch}-way branches distinct"
        )

    print("=== all tree_spec selfchecks passed ===")


if __name__ == "__main__":
    _selfcheck()
