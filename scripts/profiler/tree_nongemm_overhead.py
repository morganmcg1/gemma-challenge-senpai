#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree-verify NON-GEMM systems-overhead audit at M=8/16/32 (PR #85).

WHAT THIS MEASURES
------------------
denken #68 priced the verify-GEMM SAVINGS side (weight matmuls over M rows:
bandwidth-bound, free-to-M<=32, +18.4% at M=32). wirbel #79 pinned rho ->
+21.8% GROSS tree re-price (acceptance x #68 GEMM). BUT the +21.8% cost model
(treeshape_real_cost #74) assumes S(M)=drafter+verify_GEMM(M) and treats the
tree's NON-GEMM machinery as FREE. Moving from the deployed LINEAR M=8 chain to
an M=32 TREE adds non-GEMM ops NOT in that model:
  1. tree-mask construction  (the [M,M] ancestor/causal mask among M tree nodes),
  2. candidate scatter/gather (place M tree-node tokens into the verify batch;
     gather the accepted spine hidden states for the next drafter pass),
  3. sampler-prep for M rows  (tree-build candidate sampling via the deployed
     centroid sparse-argmax kernel over M rows; the verify-side full-vocab argmax
     over [M,262144]; the per-row sampling-metadata expand),
  4. valid_counts / accepted-prefix scheduling (the deployed greedy rejection
     kernel that walks the M draft tokens for the longest accepted prefix; for a
     tree it walks parent pointers -- byteshark's broken tree-v2 `-1` crash).

This script TIMES each op in isolation at M=8 (linear baseline), M=16, M=32
(tile-top tree), the SAME launch-free CUDA-graph / eager methodology as my #77
(drafter non-GEMM profile) and #68 (verify-GEMM roofline). It also confirms the
#43 split-KV (FlashDecoding) verify-attention path AMORTIZES the shared-prefix KV
read across the M tree query rows (M=32 attention ~ M=8 attention, not 4x), the
load-bearing assumption behind "tree attention stays near the conc=1 floor".

DEPLOYED FAITHFULNESS
---------------------
- centroid sampler: the REAL Gemma4MTPMaskedEmbedder + a VERBATIM copy of the
  deployed fused sparse-argmax triton kernel (reused from #77 via import).
- accepted-prefix kernel: the REAL vLLM `rejection_greedy_sample_kernel` that the
  deployed serve.py (DIXIE SMP-02) launches at grid=(batch_size,)=(1,).
- verify argmax: full vocab 262144 (the deployed PCK-04 patch scatters pruned
  logits back into a full-vocab [M,262144] buffer, so the sampler argmaxes over
  262144 -- greedy identity requires the true argmax).
- attention: SDPA proxy at the served TARGET shapes (8 q-heads, 2 kv-heads,
  head_dim 256, 35 sliding + 7 full layers) + the deployed splitkv `would_redirect`
  predicate (proves M<=64 verify batches route to 3D split-KV, same path as M=8).
- timing: launch-free reps-in-one-CUDA-graph (deployed onegraph basis) + eager
  contrast, value-independent, no serve-path change, no HF Job.

ONEGRAPH NOTE (the #77 caveat, applied here)
--------------------------------------------
The deployed verify step runs under ONEGRAPH-style CUDA-graph capture, so these
prep ops are launch-free in deployment. We therefore report the launch-free
(graph-replay) per-op cost as the deployed-representative basis and the eager
cost as the without-graph upper bound. Unlike #77's fusible glue, most of these
ops are STANDALONE kernels (mask build, scatter, sampler, rejection walk), so the
launch-free sum is a faithful deployed cost, not an over-count.

Primary metric:  tree_overhead_nongemm_pct_decode  (total non-GEMM tree overhead
                 at M=32 as % of the 11.6 ms decode step; must be << the +21.8%
                 GEMM savings for the tree to net-win).
Test metric:     net_tree_gain_after_overhead_pct   (the +21.8% gross minus the
                 measured non-GEMM overhead erosion).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys

# Must be set before importing torch/vllm (see project_local_a10g_gpu_env memory):
# the container exposes one A10G as index 0 but inherits CUDA_VISIBLE_DEVICES=5
# (host id); flashinfer sampler JIT fails on incomplete CUDA dev headers.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Reuse #77's launch-free timing + the VERBATIM deployed centroid sampler kernel
# and real-module builder (no duplication; identical kernels => identical numbers).
from drafter_nongemm_profile import (  # noqa: E402
    build_modules,
    make_fused_top_tokens,
    time_op_eager,
    time_op_graph,
)
# Reuse the #49/#74 tree topology builders for a FAITHFUL tree shape.
from sequoia_dp_tree import build_linear, build_sequoia_tree, derive_per_rank  # noqa: E402

A10G_HBM_GBS = 600.0
BF16 = torch.bfloat16
DEFAULT_DRAFTER = "/tmp/qat-assistant"

# served TARGET (google/gemma-4-E4B-it-qat-w4a16-ct) attention config
TGT_HIDDEN = 2560
TGT_N_LAYERS = 42
TGT_N_HEADS = 8
TGT_N_KV = 2
TGT_HEAD_DIM = 256
TGT_SLIDING = 512
TGT_FULL_LAYERS = 7      # every 6th layer is full_attention (42/6)
TGT_SLIDING_LAYERS = 35
TGT_VOCAB = 262144


# --------------------------------------------------------------------------- #
# tree topology: the #74/#79 DP-optimal M=32 tree (faithful shape for the mask) #
# --------------------------------------------------------------------------- #
def tree_parent(M: int, top1: float, topW: float, W: int, max_branch: int,
                max_depth: int) -> list[int]:
    """Parent array for the DP-optimal tree of M nodes (#74 build target).
    M=8 falls back to the deployed LINEAR chain (the baseline-overhead reference)."""
    if M <= 8:
        return build_linear(M)
    p = derive_per_rank(top1, topW, W, "geom")
    par, _F, _d = build_sequoia_tree(p, M, max_depth, max_branch)
    return list(par)


def parent_depths(parent: list[int]) -> tuple[list[int], int]:
    depth = [0] * len(parent)
    for i in range(1, len(parent)):
        depth[i] = depth[parent[i]] + 1
    return depth, (max(depth) if depth else 0)


# --------------------------------------------------------------------------- #
# Op 1: tree-mask construction. Build the among-token [M,M] additive attention  #
# mask from the parent array: mask[i,j]=0 iff j is ancestor-or-self of i, else  #
# -inf. Faithful vectorized GPU build (walk parents `depth` times, the SpecInfer #
# /EAGLE-2 dynamic-tree approach). The shared-prefix [M, L] block is all-allowed #
# (same as the linear chain) -> only the [M,M] among-token block is tree-new.    #
# --------------------------------------------------------------------------- #
def make_tree_mask_builder(parent: list[int], device: str):
    M = len(parent)
    par_t = torch.tensor([p if p >= 0 else i for i, p in enumerate(parent)],
                         dtype=torch.long, device=device)  # root parent->self
    _, max_depth = parent_depths(parent)
    rows = torch.arange(M, device=device)
    # Pre-allocated DEVICE scalars. A python `True` / `float("-inf")` fed to an
    # index_put_/where during CUDA-graph capture is staged as a CPU scalar and
    # copied H2D -> "Cannot copy between CPU and CUDA tensors during capture".
    # Materialising them on-device once keeps the build() body capture-safe.
    true_dev = torch.ones((), dtype=torch.bool, device=device)
    zero_dev = torch.zeros((), dtype=torch.float32, device=device)
    neg_dev = torch.full((), float("-inf"), dtype=torch.float32, device=device)

    def build():
        # anc[i,j] = True iff j is ancestor-or-self of i. Walk parents up to depth.
        anc = torch.zeros(M, M, dtype=torch.bool, device=device)
        anc[rows, rows] = true_dev
        cur = rows.clone()
        for _ in range(max_depth):
            cur = par_t[cur]
            anc[rows, cur] = true_dev
        # additive mask via where over two device scalars (graph-safe).
        add_mask = torch.where(anc, zero_dev, neg_dev)
        return add_mask

    return build, max_depth


# --------------------------------------------------------------------------- #
# Op 4: the DEPLOYED greedy accepted-prefix kernel (vLLM, the one serve.py's     #
# DIXIE SMP-02 launches at grid=(1,)). Walks the M-1 draft tokens for the        #
# longest accepted prefix, appends the bonus token. O(M) sequential, one CTA.    #
# --------------------------------------------------------------------------- #
def make_rejection_greedy_runner(M: int, device: str):
    from vllm.v1.sample.rejection_sampler import (
        PLACEHOLDER_TOKEN_ID,
        rejection_greedy_sample_kernel,
    )
    K = M - 1  # draft tokens (deployed: M = K+1)
    batch = 1
    cu = torch.tensor([K], dtype=torch.int32, device=device)
    draft = torch.arange(K, dtype=torch.int32, device=device)
    targ = torch.arange(K, dtype=torch.int64, device=device)   # all-accept worst case (full walk)
    bonus = torch.zeros((batch, 1), dtype=torch.int64, device=device)
    out = torch.full((batch, M), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device)

    def run():
        rejection_greedy_sample_kernel[(batch,)](
            out, cu, draft, targ, bonus, None, K, None, None, SYNTHETIC_MODE=False)
        return out

    return run


# --------------------------------------------------------------------------- #
# Step 2: attention amortization. SDPA proxy at the served TARGET attention      #
# shapes, swept over M query rows at a FIXED KV length -> shows attention time   #
# is ~flat in M (the shared-prefix KV read dominates and is read once; the among #
# token [M,M] block is tiny). Mirrors #77's attention proxy.                     #
# --------------------------------------------------------------------------- #
def attn_kv_bytes(num_kv: int, head_dim: int, L: int) -> float:
    return 2.0 * L * num_kv * head_dim * 2.0  # K+V, bf16


def time_attn_proxy(num_heads, num_kv, head_dim, L, M, iters, warmup):
    """SDPA at served target shapes: q[1,H,M,Dh], k/v[1,Hkv,L+M,Dh] (tree rows
    attend the shared L-prefix + the M among-token block)."""
    import torch.nn.functional as F
    Lk = L + M
    q = torch.randn(1, num_heads, M, head_dim, device="cuda", dtype=BF16)
    k = torch.randn(1, num_kv, Lk, head_dim, device="cuda", dtype=BF16)
    v = torch.randn(1, num_kv, Lk, head_dim, device="cuda", dtype=BF16)

    def fn():
        try:
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=False)
        except TypeError:
            kk = k.repeat_interleave(num_heads // num_kv, dim=1)
            vv = v.repeat_interleave(num_heads // num_kv, dim=1)
            return F.scaled_dot_product_attention(q, kk, vv)

    us_g, captured = time_op_graph(fn, reps=32, iters=iters, warmup=warmup)
    return us_g, captured


def splitkv_routes_3d(M: int) -> bool:
    """The DEPLOYED splitkv-verify predicate: does an M-row verify batch route to
    the 3D split-KV (FlashDecoding) path (the amortizing path, same as M=1/8)?"""
    try:
        sys.path.insert(0, os.path.join(
            _HERE, "..", "..", "submissions", "fa2sw_precache_kenyan"))
        from splitkv_verify_patch import would_redirect
        return bool(would_redirect(
            q_rows=M, max_seqlen_q=M, segm_rows=max(M, 256),
            seq_threshold_3D=256, num_seqs=1))
    except Exception as exc:  # noqa: BLE001
        print(f"[tree-overhead] splitkv predicate unavailable: {exc!r}", flush=True)
        return M <= 64  # documented gate: 1 < M <= SPLITKV_VERIFY_MAX_Q(64)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter-dir", default=DEFAULT_DRAFTER)
    ap.add_argument("--top-k", type=int, default=64, help="centroid_intermediate_top_k (serve=64)")
    ap.add_argument("--fused-block", type=int, default=16, help="FUSED_SPARSE_ARGMAX_BLOCK (manifest=16)")
    ap.add_argument("--m-sweep", default="8,16,32", help="tree verify widths; 8=linear baseline")
    ap.add_argument("--attn-L", type=int, default=256, help="shared-prefix KV length for the attn proxy")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--decode-step-ms", type=float, default=11.6)
    # tree topology (the #74/#79 measured acceptance + max-branch-4 build target)
    ap.add_argument("--top1", type=float, default=0.6792)
    ap.add_argument("--topW", type=float, default=0.8605)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--max-branch", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=24)
    # net-gain accounting anchors
    ap.add_argument("--gross-gain-pct", type=float, default=21.8, help="wirbel #79 gross re-price")
    ap.add_argument("--wall-tps", type=float, default=454.0, help="lawine #72 local wall_tps anchor")
    ap.add_argument("--official-tps", type=float, default=481.53, help="deployed official frontier")
    ap.add_argument("--rel-baseline-pct", type=float, default=0.0)
    ap.add_argument("--output", default="research/spec_cost_model/tree_nongemm_overhead.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="tree-overhead-audit")
    ap.add_argument("--wandb_name", default="denken/tree-nongemm-overhead")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    m_sweep = [int(x) for x in args.m_sweep.split(",") if x.strip()]
    decode_us = args.decode_step_ms * 1000.0
    iters, warm = args.iters, args.warmup
    dev = "cuda"
    print(f"[tree-overhead] device: {torch.cuda.get_device_name(0)}", flush=True)

    # real drafter modules (for the deployed centroid sampler over M rows) ------
    mods, emb_w, me, lm_head, info = build_modules(args.drafter_dir, args.top_k)
    fused_sampler = make_fused_top_tokens(me, args.fused_block)
    print(f"[tree-overhead] drafter cfg: hidden={info['hidden']} vocab={info['vocab']} "
          f"num_selected={info['num_selected']}", flush=True)

    # per-M op timing ----------------------------------------------------------
    # ops dict: name -> {M -> (us_graph, us_eager)}
    ops: dict[str, dict[int, dict]] = {}
    trees: dict[int, dict] = {}

    def record(name, M, ug, ue, note=""):
        ops.setdefault(name, {})[M] = {"us_graph": ug, "us_eager": ue, "note": note}

    for M in m_sweep:
        parent = tree_parent(M, args.top1, args.topW, args.W, args.max_branch, args.max_depth)
        depth, max_depth = parent_depths(parent)
        n_branch = sum(1 for i in range(1, len(parent))
                       if sum(1 for j in range(len(parent)) if parent[j] == parent[i]) > 1)
        trees[M] = {"max_depth": max_depth, "is_linear": M <= 8, "parent": parent}
        print(f"\n[tree-overhead] ===== M={M} ({'linear' if M<=8 else 'DP-tree'}, "
              f"depth={max_depth}) =====", flush=True)

        # -- Op 1: tree-mask construction (among-token [M,M] additive mask) ------
        build_mask, md = make_tree_mask_builder(parent, dev)
        ug, _ = time_op_graph(build_mask, reps=64, iters=iters, warmup=warm)
        ue = time_op_eager(build_mask, max(40, iters // 4), warm)
        record("tree_mask_construct", M, ug, ue,
               f"[{M},{M}] ancestor mask, depth={md} walk; static-tree amortizes to ~0")
        print(f"[tree-overhead] {'tree_mask_construct':>24s} {ug:8.3f}us graph ({ue:8.3f} eager)", flush=True)

        # -- Op 2: candidate scatter/gather -------------------------------------
        # scatter: place M tree-node draft token ids into the flat verify buffer
        draft_ids = torch.arange(M, dtype=torch.int64, device=dev)
        flat_buf = torch.zeros(M, dtype=torch.int64, device=dev)
        scatter_idx = torch.arange(M, device=dev)
        def scatter_fn():
            return flat_buf.index_copy_(0, scatter_idx, draft_ids)
        ug_s, _ = time_op_graph(scatter_fn, reps=128, iters=iters, warmup=warm)
        ue_s = time_op_eager(scatter_fn, max(40, iters // 4), warm)
        record("scatter_tree_tokens", M, ug_s, ue_s, f"index_copy {M} draft tokens into verify buf")
        # gather: accepted spine hidden states [acc, hidden] from [M, hidden] (handoff
        # to next drafter pass / KV write). Worst case acc=M (all accepted).
        hid = torch.randn(M, TGT_HIDDEN, device=dev, dtype=BF16)
        gat_idx = torch.arange(M, device=dev)
        def gather_fn():
            return hid.index_select(0, gat_idx)
        ug_g, _ = time_op_graph(gather_fn, reps=128, iters=iters, warmup=warm)
        ue_g = time_op_eager(gather_fn, max(40, iters // 4), warm)
        record("gather_accepted_spine", M, ug_g, ue_g, f"index_select [{M},{TGT_HIDDEN}] accepted hidden")
        print(f"[tree-overhead] {'scatter_tree_tokens':>24s} {ug_s:8.3f}us  "
              f"{'gather_accepted_spine':>22s} {ug_g:8.3f}us", flush=True)

        # -- Op 3: sampler-prep for M rows --------------------------------------
        # (a) deployed centroid sparse-argmax sampler over M rows (tree-build
        #     candidate sampling); #77 measured 1 row.
        hsamp = torch.randn(M, info["hidden"], device=dev, dtype=BF16)
        def sampler_fn():
            return fused_sampler(hsamp, lm_head)
        ug_c, _ = time_op_graph(sampler_fn, reps=16, iters=iters, warmup=warm)
        ue_c = time_op_eager(sampler_fn, max(20, iters // 8), warm)
        record("centroid_sampler_Mrows", M, ug_c, ue_c,
               f"deployed fused sparse-argmax over {M} rows -> {info['num_selected']} cand each")
        # (b) verify-side full-vocab argmax over [M, 262144] (the dixie_all_argmax)
        logits = torch.randn(M, TGT_VOCAB, device=dev, dtype=BF16)
        def vargmax_fn():
            return logits.argmax(dim=-1)
        ug_a, _ = time_op_graph(vargmax_fn, reps=16, iters=iters, warmup=warm)
        ue_a = time_op_eager(vargmax_fn, max(20, iters // 8), warm)
        record("verify_argmax_Mrows", M, ug_a, ue_a, f"argmax [{M},{TGT_VOCAB}] full-vocab (greedy verify)")
        # (c) sampling-metadata expand [batch]->[M] (expand_batch_to_tokens-style)
        meta = torch.randn(1, device=dev)
        def meta_fn():
            return meta.expand(M).contiguous()
        ug_m, _ = time_op_graph(meta_fn, reps=128, iters=iters, warmup=warm)
        ue_m = time_op_eager(meta_fn, max(40, iters // 4), warm)
        record("sampling_meta_expand", M, ug_m, ue_m, f"[1]->[{M}] per-row sampling-metadata expand")
        print(f"[tree-overhead] {'centroid_sampler_Mrows':>24s} {ug_c:8.3f}us  "
              f"{'verify_argmax_Mrows':>22s} {ug_a:8.3f}us", flush=True)

        # -- Op 4: valid_counts / accepted-prefix (deployed greedy reject kernel) -
        try:
            rej = make_rejection_greedy_runner(M, dev)
            ug_r, _ = time_op_graph(rej, reps=64, iters=iters, warmup=warm)
            ue_r = time_op_eager(rej, max(40, iters // 4), warm)
        except Exception as exc:  # noqa: BLE001
            print(f"[tree-overhead]   rejection kernel failed: {exc!r}", flush=True)
            ug_r = ue_r = float("nan")
        record("accepted_prefix_kernel", M, ug_r, ue_r,
               f"deployed rejection_greedy_sample_kernel grid=(1,), walk {M-1} draft tokens")
        # seq_lens handoff: cad.seq_lens -= num_rejected ([1] scalar update)
        seq_lens = torch.zeros(1, dtype=torch.int32, device=dev)
        nrej = torch.ones(1, dtype=torch.int32, device=dev)
        def seqlen_fn():
            return seq_lens.sub_(nrej)
        ug_q, _ = time_op_graph(seqlen_fn, reps=256, iters=iters, warmup=warm)
        ue_q = time_op_eager(seqlen_fn, max(40, iters // 4), warm)
        record("seq_lens_handoff", M, ug_q, ue_q, "cad.seq_lens -= num_rejected ([1] scalar)")
        print(f"[tree-overhead] {'accepted_prefix_kernel':>24s} {ug_r:8.3f}us  "
              f"{'seq_lens_handoff':>22s} {ug_q:8.3f}us", flush=True)

    # ---- Step 2: attention amortization across the M tree query rows ----------
    print(f"\n[tree-overhead] ===== ATTENTION AMORTIZATION (split-KV, L={args.attn_L}) =====",
          flush=True)
    attn_rows = {}
    for M in [1] + m_sweep:
        us_sl, cap = time_attn_proxy(TGT_N_HEADS, TGT_N_KV, TGT_HEAD_DIM, args.attn_L, M, iters, warm)
        us_fl, _ = time_attn_proxy(TGT_N_HEADS, TGT_N_KV, TGT_HEAD_DIM, args.attn_L, M, iters, warm)
        # per decode-step attention = 35 sliding + 7 full layers (proxy uses same shape;
        # KV bytes are head_dim-driven and identical here -> one representative proxy/layer)
        per_step = us_sl * TGT_N_LAYERS
        kvb = attn_kv_bytes(TGT_N_KV, TGT_HEAD_DIM, args.attn_L + M) * TGT_N_LAYERS
        roof = kvb / A10G_HBM_GBS / 1e3
        routes3d = splitkv_routes_3d(M) if M > 1 else True
        attn_rows[M] = {"M": M, "proxy_us_per_layer": us_sl, "per_step_proxy_us": per_step,
                        "kv_bytes_per_step": kvb, "roofline_us_per_step": roof,
                        "routes_3d_splitkv": routes3d}
        print(f"[tree-overhead] attn M={M:3d}: {us_sl:7.3f}us/layer  per-step {per_step:8.1f}us  "
              f"roof {roof:7.1f}us  3D-splitkv={routes3d}", flush=True)

    a1 = attn_rows[1]["per_step_proxy_us"]
    a8 = attn_rows.get(8, attn_rows[m_sweep[0]])["per_step_proxy_us"]
    a32 = attn_rows[max(m_sweep)]["per_step_proxy_us"]
    attn_amortizes = (a32 / a8) < 2.0  # << 4x (the non-amortized ratio) => amortizes
    print(f"[tree-overhead] attention M=32/M=8 ratio = {a32/a8:.2f}x "
          f"({'AMORTIZES (<<4x)' if attn_amortizes else 'DOES NOT amortize'}); "
          f"M=32/M=1 = {a32/a1:.2f}x", flush=True)

    # ---- Aggregate per-op table + O(M) scaling -------------------------------
    M0, Mmax = m_sweep[0], max(m_sweep)
    op_names = list(ops.keys())
    table = []
    for name in op_names:
        row = {"op": name, "note": ops[name][Mmax]["note"]}
        for M in m_sweep:
            row[f"us_M{M}"] = ops[name][M]["us_graph"]
            row[f"pct_decode_M{M}"] = 100.0 * ops[name][M]["us_graph"] / decode_us
        # scaling exponent: us(Mmax)/us(M0) vs (Mmax/M0) -> ~1 flat, ~M linear, ~M^2 quad
        u0, umax = ops[name][M0]["us_graph"], ops[name][Mmax]["us_graph"]
        ratio = umax / u0 if u0 > 0 else float("nan")
        mr = Mmax / M0
        row["growth_ratio_M8toM32"] = ratio
        row["scaling_exponent"] = (math.log(ratio) / math.log(mr)) if ratio > 0 and mr > 1 else float("nan")
        table.append(row)

    # total non-GEMM tree overhead at each M (sum of all ops, launch-free basis).
    # We report TWO bases: dynamic-tree (mask built per step) and static-tree
    # (mask precomputed once -> amortized to ~0), since #74/#79 recommend a FIXED tree.
    # Op classification: the centroid sampler is DRAFTER-side (tree-build candidate
    # sampling, partly inside the #43 drafter budget); everything else is the
    # genuine VERIFY-side tree machinery that the GEMM-only cost model omitted.
    DRAFTER_SIDE = {"centroid_sampler_Mrows"}

    def total_overhead(M, include_mask=True, verify_only=False):
        s = 0.0
        for name in op_names:
            if name == "tree_mask_construct" and not include_mask:
                continue
            if verify_only and name in DRAFTER_SIDE:
                continue
            v = ops[name][M]["us_graph"]
            if v == v:  # not NaN
                s += v
        return s

    overhead = {}
    for M in m_sweep:
        dyn = total_overhead(M, include_mask=True)
        stat = total_overhead(M, include_mask=False)
        verify_stat = total_overhead(M, include_mask=False, verify_only=True)
        verify_dyn = total_overhead(M, include_mask=True, verify_only=True)
        overhead[M] = {
            "M": M,
            "total_us_dynamic_tree": dyn,
            "total_us_static_tree": stat,
            "pct_decode_dynamic": 100.0 * dyn / decode_us,
            "pct_decode_static": 100.0 * stat / decode_us,
            "verify_side_us_static": verify_stat,
            "verify_side_us_dynamic": verify_dyn,
            "verify_side_pct_decode_static": 100.0 * verify_stat / decode_us,
            "verify_side_pct_decode_dynamic": 100.0 * verify_dyn / decode_us,
        }
    # tree OVERHEAD vs the linear M=8 baseline (delta the tree actually adds)
    base = overhead[M0]
    for M in m_sweep:
        overhead[M]["delta_vs_M8_us_dynamic"] = overhead[M]["total_us_dynamic_tree"] - base["total_us_dynamic_tree"]
        overhead[M]["delta_vs_M8_us_static"] = overhead[M]["total_us_static_tree"] - base["total_us_static_tree"]
        overhead[M]["delta_pct_decode_dynamic"] = 100.0 * overhead[M]["delta_vs_M8_us_dynamic"] / decode_us
        overhead[M]["delta_pct_decode_static"] = 100.0 * overhead[M]["delta_vs_M8_us_static"] / decode_us

    # ---- Net-gain accounting on 3 bases --------------------------------------
    # The +21.8% gross is a TPS ratio under the GEMM-only cost model. A non-GEMM
    # overhead of `ov` us/step that the model omitted multiplies the real step by
    # (1 + ov/decode_us), eroding the gain: net = (1+gross)/(1+ov_frac) - 1.
    # We use the M=32 overhead DELTA vs the deployed M=8 (the genuinely-added cost),
    # static-tree basis (the recommended fixed-topology build), and also report the
    # conservative dynamic-tree basis.
    gross = args.gross_gain_pct / 100.0
    M32 = max(m_sweep)
    ov_static_us = max(0.0, overhead[M32]["delta_vs_M8_us_static"])
    ov_dynamic_us = max(0.0, overhead[M32]["delta_vs_M8_us_dynamic"])
    ov_static_frac = ov_static_us / decode_us
    ov_dynamic_frac = ov_dynamic_us / decode_us

    def net_gain(ov_frac):
        return (1.0 + gross) / (1.0 + ov_frac) - 1.0

    net_static = net_gain(ov_static_frac)
    net_dynamic = net_gain(ov_dynamic_frac)
    erosion_static_pp = (gross - net_static) * 100.0
    erosion_dynamic_pp = (gross - net_dynamic) * 100.0

    bases = {
        "relative_pct": {
            "baseline": args.rel_baseline_pct,
            "gross": args.gross_gain_pct,
            "net_static": net_static * 100.0,
            "net_dynamic": net_dynamic * 100.0,
        },
        "local_wall_tps_x454": {
            "baseline": args.wall_tps,
            "gross": args.wall_tps * (1 + gross),
            "net_static": args.wall_tps * (1 + net_static),
            "net_dynamic": args.wall_tps * (1 + net_dynamic),
        },
        "official_tps_x481_proj": {
            "baseline": args.official_tps,
            "gross": args.official_tps * (1 + gross),
            "net_static": args.official_tps * (1 + net_static),
            "net_dynamic": args.official_tps * (1 + net_dynamic),
        },
    }

    # ---- Cost-budget oracle for land #71 (per-op expected us/step at M=32) ----
    # Budget = measured M=32 launch-free cost x slack (1.5x). A build whose op
    # exceeds budget has the byteshark layout/cache bug -> fix before quota.
    SLACK = 1.5
    budget = []
    for name in op_names:
        m32 = ops[name][M32]["us_graph"]
        budget.append({
            "op": name,
            "expected_us_M32": m32,
            "budget_us_M32": (m32 * SLACK) if m32 == m32 else None,
            "pct_decode_M32": 100.0 * m32 / decode_us if m32 == m32 else None,
            "note": ops[name][M32]["note"],
        })

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    primary = overhead[M32]["pct_decode_static"]  # total non-GEMM tree overhead %decode at M=32 (static)
    verdict = {
        "primary_metric_name": "tree_overhead_nongemm_pct_decode",
        "tree_overhead_nongemm_pct_decode": primary,
        "tree_overhead_nongemm_pct_decode_dynamic": overhead[M32]["pct_decode_dynamic"],
        "tree_overhead_total_us_M32_static": overhead[M32]["total_us_static_tree"],
        "tree_overhead_total_us_M32_dynamic": overhead[M32]["total_us_dynamic_tree"],
        "tree_overhead_delta_vs_M8_us_static": overhead[M32]["delta_vs_M8_us_static"],
        "tree_overhead_delta_pct_decode_static": overhead[M32]["delta_pct_decode_static"],
        "verify_side_overhead_pct_decode_static_M32": overhead[M32]["verify_side_pct_decode_static"],
        "verify_side_overhead_us_static_M32": overhead[M32]["verify_side_us_static"],
        "test_metric_name": "net_tree_gain_after_overhead_pct",
        "net_tree_gain_after_overhead_pct": net_static * 100.0,
        "net_tree_gain_after_overhead_pct_dynamic": net_dynamic * 100.0,
        "gross_gain_pct": args.gross_gain_pct,
        "erosion_pp_static": erosion_static_pp,
        "erosion_pp_dynamic": erosion_dynamic_pp,
        "overhead_much_smaller_than_gemm_savings": primary < (args.gross_gain_pct * 0.25),
        "attention_amortizes_M32": attn_amortizes,
        "attn_M32_over_M8_ratio": a32 / a8,
        "attn_per_step_us_M8": a8,
        "attn_per_step_us_M32": a32,
        "decode_step_ms": args.decode_step_ms,
        "peak_gpu_mem_gib": peak_mem,
    }

    # ---- console summary -----------------------------------------------------
    print("\n[tree-overhead] ===== NON-GEMM TREE-OVERHEAD per-op TABLE (launch-free us/step) =====", flush=True)
    hdr = f"{'op':>24s}" + "".join(f"{'us@M'+str(M):>10s}" for M in m_sweep) + \
          "".join(f"{'%dec@'+str(M):>9s}" for M in m_sweep) + f"{'M8->M32x':>9s}{'exp':>6s}"
    print(hdr, flush=True)
    for r in table:
        line = f"{r['op']:>24s}"
        line += "".join(f"{r[f'us_M{M}']:>10.3f}" for M in m_sweep)
        line += "".join(f"{r[f'pct_decode_M{M}']:>8.3f}%" for M in m_sweep)
        line += f"{r['growth_ratio_M8toM32']:>9.2f}{r['scaling_exponent']:>6.2f}"
        print(line, flush=True)
    print(f"\n[tree-overhead] TOTAL non-GEMM tree overhead @ M=32:", flush=True)
    print(f"  static-tree (mask precomputed): {overhead[M32]['total_us_static_tree']:.1f}us/step "
          f"= {overhead[M32]['pct_decode_static']:.3f}% decode  "
          f"(delta vs M8 linear: {overhead[M32]['delta_vs_M8_us_static']:+.1f}us = "
          f"{overhead[M32]['delta_pct_decode_static']:+.3f}pp)", flush=True)
    print(f"  dynamic-tree (mask per step):   {overhead[M32]['total_us_dynamic_tree']:.1f}us/step "
          f"= {overhead[M32]['pct_decode_dynamic']:.3f}% decode", flush=True)
    print(f"  verify-side machinery only (excl. drafter centroid sampler): "
          f"{overhead[M32]['verify_side_us_static']:.1f}us/step "
          f"= {overhead[M32]['verify_side_pct_decode_static']:.3f}% decode (static)", flush=True)
    print(f"\n[tree-overhead] ATTENTION amortization: M=32/M=8 = {a32/a8:.2f}x "
          f"-> {'CONFIRMED (<<4x, KV read shared)' if attn_amortizes else 'EROSION'}", flush=True)
    print(f"\n[tree-overhead] NET-GAIN ACCOUNTING (gross +{args.gross_gain_pct}% from wirbel #79):", flush=True)
    print(f"  non-GEMM overhead erosion: static {erosion_static_pp:.2f}pp / dynamic {erosion_dynamic_pp:.2f}pp", flush=True)
    for bname, b in bases.items():
        print(f"  {bname:>26s}: gross {b['gross']:.2f} -> net(static) {b['net_static']:.2f} "
              f"/ net(dynamic) {b['net_dynamic']:.2f}", flush=True)
    print(f"\n[tree-overhead] PRIMARY tree_overhead_nongemm_pct_decode(M=32,static) = {primary:.3f}%  "
          f"(vs +{args.gross_gain_pct}% GEMM savings -> "
          f"{'<<savings, tree NET-WINS' if verdict['overhead_much_smaller_than_gemm_savings'] else 'material erosion'})",
          flush=True)
    print(f"[tree-overhead] TEST net_tree_gain_after_overhead_pct = {net_static:.4%} (static)", flush=True)
    print(f"[tree-overhead] peak GPU mem: {peak_mem:.2f} GiB", flush=True)

    payload = {
        "config": {
            "drafter_dir": args.drafter_dir, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0), "m_sweep": m_sweep,
            "attn_L": args.attn_L, "iters": iters, "warmup": warm,
            "decode_step_ms": args.decode_step_ms, "top_k": args.top_k,
            "fused_block": args.fused_block, "tree_params": {
                "top1": args.top1, "topW": args.topW, "W": args.W,
                "max_branch": args.max_branch, "max_depth": args.max_depth},
            "target_attn": {"n_heads": TGT_N_HEADS, "n_kv": TGT_N_KV,
                            "head_dim": TGT_HEAD_DIM, "n_layers": TGT_N_LAYERS,
                            "sliding": TGT_SLIDING, "vocab": TGT_VOCAB},
            "drafter_info": info, "A10G_HBM_GBS": A10G_HBM_GBS,
            "gross_gain_pct": args.gross_gain_pct, "wall_tps": args.wall_tps,
            "official_tps": args.official_tps, "peak_gpu_mem_gib": peak_mem,
            "note": "launch-free reps-in-CUDA-graph per-op timing (deployed onegraph "
                    "basis) + eager contrast; real drafter centroid sampler + real vLLM "
                    "rejection_greedy_sample_kernel + SDPA attention proxy at served "
                    "target shapes. Value-independent, no serve-path change, no HF Job.",
        },
        "trees": {str(M): trees[M] for M in m_sweep},
        "ops": {name: {str(M): ops[name][M] for M in m_sweep} for name in op_names},
        "per_op_table": table,
        "overhead_by_M": {str(M): overhead[M] for M in m_sweep},
        "attention_amortization": {str(M): attn_rows[M] for M in attn_rows},
        "net_gain_bases": bases,
        "cost_budget_oracle_land71": budget,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[tree-overhead] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[tree-overhead] W&B logging failed: {exc!r}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    m_sweep = payload["config"]["m_sweep"]
    # per-op table
    cols = ["op"] + [f"us_M{M}" for M in m_sweep] + [f"pct_decode_M{M}" for M in m_sweep] + \
           ["growth_ratio_M8toM32", "scaling_exponent", "note"]
    tbl = wandb.Table(columns=cols)
    for r in payload["per_op_table"]:
        tbl.add_data(*[r.get(c) for c in cols])
    run.log({"tree_overhead_op_table": tbl})
    # cost-budget oracle table
    bcols = ["op", "expected_us_M32", "budget_us_M32", "pct_decode_M32", "note"]
    btbl = wandb.Table(columns=bcols)
    for b in payload["cost_budget_oracle_land71"]:
        btbl.add_data(*[b.get(c) for c in bcols])
    run.log({"cost_budget_oracle_land71": btbl})
    # attention amortization line series
    acols = ["M", "per_step_proxy_us", "roofline_us_per_step", "routes_3d_splitkv"]
    atbl = wandb.Table(columns=acols)
    for M in sorted(payload["attention_amortization"], key=int):
        a = payload["attention_amortization"][M]
        atbl.add_data(a["M"], a["per_step_proxy_us"], a["roofline_us_per_step"], a["routes_3d_splitkv"])
    run.log({"attention_amortization": atbl})
    # overhead-by-M + net-gain summary
    run.summary.update({k: v for k, v in payload["verdict"].items() if not isinstance(v, (dict, list))})
    for M in sorted(payload["overhead_by_M"], key=int):
        o = payload["overhead_by_M"][M]
        run.log({"M": o["M"], "overhead_pct_decode_static": o["pct_decode_static"],
                 "overhead_pct_decode_dynamic": o["pct_decode_dynamic"],
                 "overhead_us_static": o["total_us_static_tree"]})
    run.finish()
    print(f"[tree-overhead] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
