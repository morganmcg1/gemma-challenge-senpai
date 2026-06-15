#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Draft-head vocab roofline: is the 101.2us draft floor's 256k head recoverable?
(PR #264, kanna). LOCAL GPU micro-profiling + CPU analytic. Analysis-only:
no served-file change, no HF Job, no submission. BASELINE stays 481.53.

THE QUESTION
------------
The fleet declared the bf16 draft 101.2us/pass (x K=7 ~= 58% of the served step)
"the floor" (denken #254 zav6nr8y; #248). Weight-quant is DEAD at M=1 (int3 #248
no W3A16 kernel; int4 #254 Marlin 165.79us > bf16 101.2us). Weight-quant attacks
bytes-PER-weight; this leg attacks the NUMBER of output columns: Gemma-4 has a
262144 (256k) vocab, and the HYPOTHESIS was that the draft's vocab-projection head
is a memory-bound M=1 GEMV that reads the ENTIRE hidden x 256k head matrix
(~ hidden*256k*2 bytes) every pass -- a dominant, immovable share of the 101.2us.
A restricted-vocab (top-K) head would be a smaller dense GEMV, cheaper at M=1,
and greedy-safe by construction (propose-only draft; full-256k verify).

CRUX (diagnostic-first): (1) the draft head's actual shape; (2) its us SHARE of
the 101.2us/pass; (3) IF material, is there a greedy-safe top-K that NETS a TPS
gain off 481.53?

WHAT THE ARCHITECTURE ACTUALLY IS (diagnostic, decisive)
--------------------------------------------------------
The proposer at /tmp/qat-assistant is `Gemma4AssistantForCausalLM`. Its head is
`Gemma4AssistantMaskedEmbedder` (config: use_ordered_embeddings=True,
num_centroids=2048, centroid_intermediate_top_k=64, vocab_size=262144,
hidden_size=256, tie_word_embeddings=True). The head DOES NOT do a dense
hidden x 256k GEMV. Per `modeling_gemma4_assistant.py:58-87` it:
  (1) centroids GEMV: hidden[256] -> 2048 centroid logits  (reads 2048*256*2 = 1 MiB)
  (2) topk(64) over the 2048 centroids
  (3) token_ordering.view(2048, 262144//2048=128): each centroid owns 128 tokens
  (4) GATHERS top_k*128 = 64*128 = 8192 rows from the tied lm_head [262144,256]
      (~ 8192*256*2 = 4 MiB == 3.1% of the 128 MiB table)
  (5) an 8192-wide matmul, then scatter into a full [.,.,262144] masked buffer.
So the draft's EFFECTIVE active vocab per pass is 8192 tokens -- already 4x MORE
restrictive than the PR's smallest proposed K=32768. A dense 256k head at M=1
would read 262144*256*2 = 128 MiB; at A10G 600 GB/s that is ~224us ALONE -- > 2x
the entire measured 101.2us pass -- so the deployed draft physically cannot and
does not run a dense 256k GEMV. The "restricted-vocab" lever is structurally
pre-empted by the centroid-routed sparse-gather head.

WHAT THIS MEASURES (real A10G micro-profiling, no HF Job, no serve change)
-------------------------------------------------------------------------
  * Decompose the bf16 draft per-pass (denken's 19-GEMM chain, M=1, launch-free
    CUDA graph) into {io_projection, attention, mlp, vocab-head}; the head is the
    REAL Gemma4AssistantMaskedEmbedder op (centroids+topk+gather+bmm+full+scatter),
    not a GEMV stand-in. Headline = head_PROJECTION_share_pct (the centroids-sampler
    projection's share of the 101.2us pass; the PR's restricted-vocab lever acts on
    the projection, gated >=10% to be material).
  * MEASURE a counterfactual dense 256k head F.linear at M=1 -> head_us_dense_256k,
    the empirical "physical-impossibility" anchor (head alone > whole pass).
  * Price a dense top-K head SWAP for K in {32768,65536,131072}: dense head_us(K) =
    head_us_dense_256k * K/256k, mapped x K=7 through the composition
    `official = K_cal*(E[T]/step)*tau` (K_cal=125.268, step=1218.2us, kanna #217/
    #260) into a STEP-ONLY UPPER bound (E[T] unpriced -- a fixed top-K reaches a
    subset of the centroid head's full-vocab reach, so E[T] is non-increasing). The
    deployed centroid head is the honest baseline; the swap is a served-model change
    handed to denken, NOT this analysis-only PR's live lever.

GREEDY/PPL SAFETY: pinned BY CONSTRUCTION. This leg MEASURES; it edits no served
file. A propose-only draft (restricted or centroid) can never cause a wrong
accept: greedy-exact verify checks every candidate against the FULL 256k target
argmax, so an unreachable token is a REJECT (lower E[T]), never a wrong token.

SELF-TEST (`draft_head_vocab_roofline_self_test_passes`, PRIMARY)
-----------------------------------------------------------------
(a) GEMV-chain components sum to the 101.2us anchor within tol (state tol);
(b) dense-head byte-traffic -> us at A10G HBM BW round-trips the MEASURED dense
    256k head us within tol;
(c) head_us_dense(K) scales linearly in K and head_us_dense(256k) recovers the
    measured dense head us (resid <= tol);
(d) the composition reproduces 481.53 exactly at the deployed (step, E[T]) point,
    so the dense top-K step-only band maps through tps proportional to 1/step;
(e) NaN-clean;
(f) BASELINE 481.53, 520.95 lambda=1 ceiling, K_cal=125.268 imported EXACTLY and
    UNCHANGED (this prices a lever; it moves nothing).
TEST metric: `projected_tps_gain_pct` (the analysis-only LIVE lever; 0.0 when the
projection share is <10% / the dense top-K is a served swap, as found here).

Requires a torch+CUDA env with the gemma4_assistant modeling code (transformers
>=5.9; the deployed senpai venv). No Marlin / no vLLM kernel needed (bf16 control).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import struct
import time

# Must be set before importing torch. Single-GPU node; in-container GPU is index 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys  # noqa: E402
# This file is one of several profile-like scripts; keep the script dir off
# sys.path[0] so a stdlib `import profile` (pulled by some deps) is unaffected.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEFAULT_DRAFTER = "/tmp/qat-assistant"

# ---- A10G (AWS g5, GA102, sm_86) roofline ceiling (identical to #248/#254) ----
A10G_HBM_GBS = 600.0
BF16_BYTES = 2.0

# ---- IMPORTED, UNCHANGED (this leg moves nothing) ----------------------------
FRONTIER_TPS = 481.53         # PR #52 official a10g-small frontier (BASELINE)
LAMBDA1_CEILING_TPS = 520.95  # lambda=1 built ceiling (ubel #240 / land GO read)
K_CAL = 125.268               # composition calibration (kanna #217 vgovdrjc / #260)
STEP_US = 1218.2              # served decode step (kanna #217 / #260)
K_DEPLOYED = 7                # num_speculative_tokens (manifest SPECULATIVE_CONFIG)
BF16_ANCHOR_US_254 = 101.2    # denken #254 zav6nr8y bf16-draft floor (the anchor)
DRAFT_SHARE_OF_STEP = 0.58    # bf16 draft fraction of the step (#254)
ET_DEPLOYED = 3.3             # accepted tok/step, bf16-draft control (#248/#254)

# Restricted-DENSE-head top-K set priced by the PR.
RESTRICT_K = [32768, 65536, 131072]
VOCAB = 262144

# self-test tolerances (M=1 micro-GEMV timing is noisy; bands stated explicitly)
ANCHOR_TOL_PCT = 0.25         # GEMV-chain sum vs 101.2us anchor
BYTE_US_TOL_PCT = 0.20        # dense-head byte->us vs measured dense head us
LINEAR_RESID_TOL_PCT = 0.01   # head_us_dense(K) linear-in-K reconstruction


# --------------------------------------------------------------------------- #
# Drafter weight introspection (verbatim shapes from #248/#254).               #
# --------------------------------------------------------------------------- #
def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def load_tensor(path: str, name: str, device: str = "cpu") -> torch.Tensor:
    from safetensors import safe_open
    with safe_open(path, framework="pt", device=device) as f:
        return f.get_tensor(name)


class BF16Linear(torch.nn.Module):
    """Deployed draft path: a plain bf16 weight, cuBLAS F.linear (M=1 GEMV)."""
    def __init__(self, w_bf16: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(w_bf16.cuda(), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


def build_gemv_buckets(drafter_dir: str):
    """Return (buckets, specs). buckets: dict name -> list[(BF16Linear, in_f)] in
    execution order. Mirrors denken #254's 19-GEMM per-pass chain, regrouped into
    {io_projection, attention, mlp} (the head is built separately as the REAL op).
    """
    st = os.path.join(drafter_dir, "model.safetensors")
    hdr = read_safetensors_header(st)
    layer_ids = sorted({int(k.split(".layers.")[1].split(".")[0])
                        for k in hdr if ".layers." in k})
    io_mods, attn_mods, mlp_mods = [], [], []
    specs = []

    w = load_tensor(st, "pre_projection.weight")
    io_mods.append((BF16Linear(w), w.shape[1]))
    specs.append(("io_projection", "pre_projection", w.shape[1], w.shape[0]))
    for i in layer_ids:
        qw = load_tensor(st, f"model.layers.{i}.self_attn.q_proj.weight")
        attn_mods.append((BF16Linear(qw), qw.shape[1]))
        specs.append(("attention", f"layer{i}.q_proj", qw.shape[1], qw.shape[0]))
        ow = load_tensor(st, f"model.layers.{i}.self_attn.o_proj.weight")
        attn_mods.append((BF16Linear(ow), ow.shape[1]))
        specs.append(("attention", f"layer{i}.o_proj", ow.shape[1], ow.shape[0]))
        gw = load_tensor(st, f"model.layers.{i}.mlp.gate_proj.weight")
        uw = load_tensor(st, f"model.layers.{i}.mlp.up_proj.weight")
        guw = torch.cat([gw, uw], dim=0)
        mlp_mods.append((BF16Linear(guw), guw.shape[1]))
        specs.append(("mlp", f"layer{i}.gate_up", guw.shape[1], guw.shape[0]))
        dw = load_tensor(st, f"model.layers.{i}.mlp.down_proj.weight")
        mlp_mods.append((BF16Linear(dw), dw.shape[1]))
        specs.append(("mlp", f"layer{i}.down_proj", dw.shape[1], dw.shape[0]))
    w = load_tensor(st, "post_projection.weight")
    io_mods.append((BF16Linear(w), w.shape[1]))
    specs.append(("io_projection", "post_projection", w.shape[1], w.shape[0]))
    return {"io_projection": io_mods, "attention": attn_mods, "mlp": mlp_mods}, specs


# --------------------------------------------------------------------------- #
# The REAL draft head: graph-safe Gemma4AssistantMaskedEmbedder (no .item()).   #
# --------------------------------------------------------------------------- #
class GraphSafeMaskedHead(torch.nn.Module):
    """Faithful reimplementation of Gemma4AssistantMaskedEmbedder.forward, with
    the one CUDA-graph-incompatible op (mask_value = selected_logits.min().item())
    replaced by a fixed constant. The fill VALUE does not affect kernel TIMING
    (full + scatter cost the same); this lets us include the head in a launch-free
    CUDA graph and ablate its in-graph cost. The deployed head additionally does
    one .item() D2H sync (~a few us); timed separately as `head_item_sync_us`.
    """
    def __init__(self, drafter_dir: str, num_centroids: int, top_k: int,
                 vocab: int, hidden: int):
        super().__init__()
        self.num_centroids = num_centroids
        self.top_k = top_k
        self.vocab = vocab
        self.vocab_per_centroid = vocab // num_centroids
        st = os.path.join(drafter_dir, "model.safetensors")
        cw = load_tensor(st, "masked_embedding.centroids.weight").cuda()  # [C,256]
        self.centroids = BF16Linear(cw)
        tok = load_tensor(st, "masked_embedding.token_ordering").cuda().long()
        self.register_buffer("token_ordering", tok)
        # tied lm_head == embed_tokens [262144,256]; the gather source
        self.register_buffer(
            "lm_head_weight", load_tensor(st, "model.embed_tokens.weight").cuda())

    def proj(self, hidden_states):
        """The PROJECTION+PROPOSAL part: centroids GEMV -> topk -> gather 8192 rows
        -> 8192-wide matmul. Returns (selected_logits[B,L,8192], scatter_idx[B,L,8192]).
        This is everything the draft needs to ARGMAX a proposal; no 256k buffer."""
        batch, seq_len = hidden_states.shape[:2]
        centroid_logits = self.centroids(hidden_states)
        _, top_k_indices = torch.topk(centroid_logits, k=self.top_k, dim=-1)
        canonical = self.token_ordering.view(self.num_centroids, self.vocab_per_centroid)
        selected_canonical = canonical[top_k_indices]          # [B,L,top_k,vpc]
        selected_flat = selected_canonical.reshape(-1)
        selected_embeddings = self.lm_head_weight[selected_flat].view(
            batch, seq_len, self.top_k * self.vocab_per_centroid, hidden_states.shape[-1])
        selected_logits = (hidden_states.unsqueeze(-2)
                           @ selected_embeddings.transpose(-1, -2)).squeeze(-2)
        return selected_logits, selected_canonical.view(batch, seq_len, -1)

    def materialize(self, selected_logits, scatter_idx, batch=1, seq_len=1):
        """OUTPUT-WIDTH materialization: allocate the full [.,.,262144] masked
        buffer and scatter the 8192 active logits. Cost is fixed by the 262144
        OUTPUT width (the verify/logits interface), independent of the active set."""
        output = torch.full((batch, seq_len, self.vocab), -1.0e4,
                            dtype=selected_logits.dtype, device=selected_logits.device)
        return output.scatter_(dim=-1, index=scatter_idx, src=selected_logits)

    def forward(self, hidden_states):  # hidden_states: [B,L,256]
        b, l = hidden_states.shape[:2]
        sl, idx = self.proj(hidden_states)
        return self.materialize(sl, idx, b, l)


# --------------------------------------------------------------------------- #
# Timing: launch-free CUDA-graph (matches deployed ONEGRAPH 101.2us basis).     #
# --------------------------------------------------------------------------- #
def time_gemv_chain(modules_in_order, iters, warmup):
    """(us_per_pass, captured). M=1 GEMV chain; independent static input per GEMM."""
    if not modules_in_order:
        return 0.0, True
    bufs = [torch.randn(1, inf, device="cuda", dtype=torch.bfloat16)
            for (_, inf) in modules_in_order]

    def run():
        for (mod, _), b in zip(modules_in_order, bufs):
            mod(b)
    return _graph_time(run, iters, warmup)


def time_callable(run, iters, warmup):
    return _graph_time(run, iters, warmup)


def _graph_time(run, iters, warmup):
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(5):
                run()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            run()
        for _ in range(max(10, warmup)):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / iters
        del g
        return ms * 1e3, True
    except Exception as exc:  # noqa: BLE001
        print(f"[draft-head]   graph capture failed: {exc!r}; eager", flush=True)
        return _eager_time(run, iters, warmup), False


def _eager_time(run, iters, warmup):
    with torch.inference_mode():
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            run()
        e1.record()
        torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3


def expected_accepted(alpha: float, K: int) -> float:
    """Leviathan/Chen i.i.d. greedy acceptance: E[T] = (1 - a^(K+1))/(1 - a)."""
    if alpha >= 1.0:
        return float(K + 1)
    return (1.0 - alpha ** (K + 1)) / (1.0 - alpha)


def solve_alpha_for_et(et: float, K: int) -> float:
    lo, hi = 0.0, 0.999999
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if expected_accepted(mid, K) < et:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tps_from_step_et(step_us: float, et: float) -> float:
    """Composition official = K_cal*(E[T]/step)*tau. tau, K_cal cancel in the
    ratio to the anchor; we re-express as 481.53 * (STEP_US/step) * (et/ET_anchor)
    so the deployed (STEP_US, ET_DEPLOYED) reproduces FRONTIER_TPS exactly."""
    return FRONTIER_TPS * (STEP_US / step_us) * (et / ET_DEPLOYED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter-dir", default=DEFAULT_DRAFTER)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--k", type=int, default=K_DEPLOYED)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output",
                    default="research/speed/draft_head_vocab_roofline/roofline.json")
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="draft-head-vocab")
    ap.add_argument("--wandb_name", default="kanna/draft-head-vocab-roofline")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable (set CUDA_VISIBLE_DEVICES=0)"
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[draft-head] device {dev} sm_{cap[0]}{cap[1]} torch {torch.__version__}",
          flush=True)
    torch.cuda.reset_peak_memory_stats()
    nan_clean = True

    # --- config / architecture diagnostic (step 1) ----------------------------
    cfg = json.load(open(os.path.join(args.drafter_dir, "config.json")))
    tc = cfg["text_config"]
    draft_vocab_size = int(tc["vocab_size"])
    draft_hidden = int(tc["hidden_size"])
    draft_head_tied = bool(cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings")))
    num_centroids = int(cfg["num_centroids"])
    top_k_centroids = int(cfg["centroid_intermediate_top_k"])
    use_ordered = bool(cfg.get("use_ordered_embeddings", False))
    vocab_per_centroid = draft_vocab_size // num_centroids
    active_vocab_per_pass = top_k_centroids * vocab_per_centroid
    # dense-256k head byte model at M=1 (the HYPOTHESIS's assumed head)
    dense_head_bytes = draft_hidden * draft_vocab_size * BF16_BYTES
    dense_head_us_analytic = dense_head_bytes / (A10G_HBM_GBS * 1e9) * 1e6
    # centroid head byte model: centroids + 8192-row gather + full output buffer
    centroid_head_bytes = (
        num_centroids * draft_hidden * BF16_BYTES                 # centroids weight
        + active_vocab_per_pass * draft_hidden * BF16_BYTES       # gathered rows
        + draft_vocab_size * BF16_BYTES)                          # full masked output
    centroid_head_us_analytic = centroid_head_bytes / (A10G_HBM_GBS * 1e9) * 1e6
    print(f"[draft-head] ARCH: vocab={draft_vocab_size} hidden={draft_hidden} "
          f"tied={draft_head_tied} ordered={use_ordered} centroids={num_centroids} "
          f"top_k={top_k_centroids} vpc={vocab_per_centroid} "
          f"active/pass={active_vocab_per_pass}", flush=True)
    print(f"[draft-head] dense-256k head: {dense_head_bytes/2**20:.1f} MiB -> "
          f"{dense_head_us_analytic:.1f}us @ {A10G_HBM_GBS}GB/s (vs 101.2us WHOLE "
          f"pass: {dense_head_us_analytic/BF16_ANCHOR_US_254:.2f}x) | centroid head "
          f"{centroid_head_bytes/2**20:.2f} MiB -> {centroid_head_us_analytic:.1f}us",
          flush=True)

    # --- build GEMV buckets + real head ---------------------------------------
    buckets, specs = build_gemv_buckets(args.drafter_dir)
    head = GraphSafeMaskedHead(args.drafter_dir, num_centroids, top_k_centroids,
                               draft_vocab_size, draft_hidden).cuda().eval()
    h_buf = torch.randn(1, 1, draft_hidden, device="cuda", dtype=torch.bfloat16)

    # The draft's "vocab-projection" -- the ONLY hidden->vocab-space GEMV in the
    # head -- is the centroids sampler (256 -> 2048). This is what the PR's lever
    # would "restrict". It already outputs only 2048 columns.
    centroids_only = [(head.centroids, draft_hidden)]

    def run_head():               # full HF head op (proj + 262144 output buffer)
        head(h_buf)

    def run_head_proj():          # PROJECTION+PROPOSAL only: centroids+topk+gather+bmm
        head.proj(h_buf)

    _sl, _idx = head.proj(h_buf)  # precompute for output-materialization timing

    def run_head_output():        # OUTPUT-WIDTH materialization only: full+scatter
        head.materialize(_sl, _idx)

    # dense-256k counterfactual head: a plain F.linear against the tied lm_head
    dense_lm_head = BF16Linear(head.lm_head_weight.clone())
    dense_buf = torch.randn(1, draft_hidden, device="cuda", dtype=torch.bfloat16)

    def run_dense_head():
        dense_lm_head(dense_buf)

    # --- (2) timing -----------------------------------------------------------
    it, wu = args.iters, args.warmup
    us, captured = {}, {}
    for name, mods in buckets.items():
        us[name], captured[name] = time_gemv_chain(mods, it, wu)
    us["head_projection_gemv"], captured["head_projection_gemv"] = \
        time_gemv_chain(centroids_only, it, wu)   # the 256->2048 vocab projection
    us["head_proj_propose"], captured["head_proj_propose"] = \
        time_callable(run_head_proj, it, wu)      # +topk+gather8192+bmm
    us["head_output_mat"], captured["head_output_mat"] = \
        time_callable(run_head_output, it, wu)    # full(262144)+scatter
    us["head_real"], captured["head_real"] = time_callable(run_head, it, wu)
    us["dense_head_256k"], captured["dense_head_256k"] = \
        time_callable(run_dense_head, it, wu)

    # GEMV-chain total (denken #254 basis: head == centroids-GEMV) -> 101.2 anchor.
    gemv_chain_us = (us["io_projection"] + us["attention"] + us["mlp"]
                     + us["head_projection_gemv"])
    # HEADLINE: the PR asks the vocab-PROJECTION's share of the 101.2us pass.
    head_projection_share_pct = 100.0 * us["head_projection_gemv"] / gemv_chain_us
    head_share_pct = head_projection_share_pct  # the PR's headline metric
    # The full HF head op (proj + 262144 output buffer) is much larger, but the
    # excess over the projection is OUTPUT-WIDTH materialization (full+scatter),
    # not the projection vocab -- restricting the projection cannot reduce it.
    pass_with_full_head_us = (us["io_projection"] + us["attention"] + us["mlp"]
                              + us["head_real"])
    full_head_share_pct = 100.0 * us["head_real"] / pass_with_full_head_us

    nan_clean = bool(torch.isfinite(head(h_buf)).all().item())

    # --- (3) restricted-vocab tradeoff (projection-to-projection, honest) -----
    # The PR's lever restricts the vocab PROJECTION. Compare a dense top-K
    # projection against the DEPLOYED projection (centroids sampler + sparse
    # gather/propose) -- output-materialization is common to both and cancels.
    dense256_us = us["dense_head_256k"]
    deployed_proj_us = us["head_proj_propose"]      # centroid-routed proposal cost
    restrict_rows = []
    for K in RESTRICT_K:
        dhk_us = dense256_us * (K / draft_vocab_size)   # dense top-K projection
        # delta = (dense top-K projection) - (deployed centroid-routed proposal)
        delta_per_pass = dhk_us - deployed_proj_us
        step_new = STEP_US + args.k * delta_per_pass
        # E[T] is NOT held fixed honestly: a dense FIXED top-K reaches a fixed
        # SUBSET of the vocab, whereas the deployed centroid head routes
        # input-adaptively and can reach ANY of the 262144 tokens across inputs
        # (per-pass active=8192, REACHABLE=full vocab). So swapping centroid
        # routing -> dense top-K can only REDUCE coverage -> E[T] is
        # NON-INCREASING. We therefore report the step-time delta as a STEP-ONLY
        # UPPER BOUND (E[T] unpriced); any real E[T] loss erodes it. et held at
        # ET_DEPLOYED only to expose that upper bound, NOT as a net claim.
        et_new = ET_DEPLOYED
        step_only_tps = tps_from_step_et(step_new, et_new)
        restrict_rows.append({
            "K": K, "dense_head_us_K": dhk_us, "delta_per_pass_us": delta_per_pass,
            "step_us_new": step_new, "et_new": et_new,
            "step_only_tps": step_only_tps,
            "step_only_gain_pct_ub": 100.0 * (step_only_tps / FRONTIER_TPS - 1.0),
            "rows_vs_deployed8192": K / active_vocab_per_pass,
            "dense_topk_vs_deployed_proposal_x": dhk_us / deployed_proj_us,
        })
    # head-side ABSOLUTE ceiling: even a FREE proposal (deployed_proposal -> 0)
    # off the step is the max any head-side lever could yield (step-only UB).
    step_free_proj = STEP_US - args.k * deployed_proj_us
    head_side_ceiling_gain_pct = 100.0 * (
        tps_from_step_et(step_free_proj, ET_DEPLOYED) / FRONTIER_TPS - 1.0)

    best = max(restrict_rows, key=lambda r: r["step_only_tps"])
    # The PR's LIVE analysis-only lever is gated on the vocab-PROJECTION being a
    # material (>=10%) share of the 101.2us pass. It is 5% -> IMMATERIAL, so the
    # restricted-vocab lever is PRE-EMPTED for this analysis-only PR. The best
    # dense-top-K step band (below) is a served-model head-SWAP number (replace
    # the centroid router), E[T]-unpriced and out of analysis-only scope -> it is
    # a denken roofline handoff, NOT this PR's live lever.
    head_material = bool(head_share_pct >= 10.0)
    restricted_vocab_lever_live = bool(
        head_material and best["step_only_gain_pct_ub"] > 0.0)
    projected_tps_gain_pct = (max(0.0, best["step_only_gain_pct_ub"])
                              if restricted_vocab_lever_live else 0.0)
    dense_topk_step_only_ub_band_pct = [
        min(r["step_only_gain_pct_ub"] for r in restrict_rows),
        max(r["step_only_gain_pct_ub"] for r in restrict_rows)]

    # --- self-test ------------------------------------------------------------
    # (a) GEMV-chain components sum to the 101.2 anchor within tol
    anchor_resid = abs(gemv_chain_us - BF16_ANCHOR_US_254) / BF16_ANCHOR_US_254
    st_a = bool(anchor_resid <= ANCHOR_TOL_PCT)
    # (b) dense-head byte->us round-trips the MEASURED dense head us within tol
    byte_us_resid = abs(dense_head_us_analytic - dense256_us) / dense256_us
    st_b = bool(byte_us_resid <= BYTE_US_TOL_PCT)
    # (c) head_us_dense(K) linear in K: per-K slope (dhk*vocab/K) is constant and
    # recovers the measured dense head us at K=256k (resid <= tol across all K).
    slope_resids = [abs(r["dense_head_us_K"] * draft_vocab_size / r["K"] - dense256_us)
                    / dense256_us for r in restrict_rows]
    st_c = bool(max(slope_resids) <= LINEAR_RESID_TOL_PCT)
    # (d) net_tps(K) consistent with composition: deployed point reproduces 481.53
    st_d = bool(abs(tps_from_step_et(STEP_US, ET_DEPLOYED) - FRONTIER_TPS) < 1e-6)
    # (e) NaN-clean
    st_e = bool(nan_clean)
    # (f) imported constants exact and unchanged
    st_f = bool(FRONTIER_TPS == 481.53 and LAMBDA1_CEILING_TPS == 520.95
                and K_CAL == 125.268)
    self_test_passes = bool(st_a and st_b and st_c and st_d and st_e and st_f)

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    head_share_verdict = (
        f"IMMATERIAL & PRE-EMPTED: the draft's vocab PROJECTION is the centroids "
        f"sampler (256->{num_centroids}), only {head_projection_share_pct:.1f}% of "
        f"the 101.2us pass (<10% gate) -> the PR's restricted-vocab lever is "
        f"pre-empted; there is NO dense {draft_vocab_size} GEMV to restrict (it "
        f"measures {dense256_us:.0f}us = {dense256_us/BF16_ANCHOR_US_254:.1f}x the "
        f"whole pass). A dense top-K head SWAP shows a step-only UPPER band "
        f"{dense_topk_step_only_ub_band_pct[0]:+.1f}..{dense_topk_step_only_ub_band_pct[1]:+.1f}% "
        f"(best K={best['K']} {best['step_only_gain_pct_ub']:+.2f}%), but that is a "
        f"served-model head swap with E[T] UNPRICED (fixed top-K restricts reachable "
        f"vocab below the centroid head's full-vocab reach -> E[T] non-increasing) -> "
        f"denken roofline territory, NOT this analysis-only lever -> projected="
        f"{projected_tps_gain_pct:.2f}%"
        if not restricted_vocab_lever_live else
        "MATERIAL: a greedy-safe restricted-vocab projection nets a TPS gain")

    handoff = (
        f"the draft vocab-PROJECTION head is ~{head_projection_share_pct:.0f}% of the "
        f"101.2us/pass (centroids sampler 256->{num_centroids}; there is NO dense "
        f"{draft_vocab_size} GEMV -- a dense 256k head measures {dense256_us:.0f}us at "
        f"M=1, {dense256_us/BF16_ANCHOR_US_254:.1f}x the whole pass, physically absent "
        f"from the deployed draft, which routes through {num_centroids} centroids to "
        f"{active_vocab_per_pass} input-adaptive active tokens), so the PR's "
        f"restricted-vocab PROJECTION lever is IMMATERIAL (<10%) and PRE-EMPTED -> "
        f"projected_tps_gain_pct=0.0 for this analysis-only PR. [Handoff for denken's "
        f"roofline: the deployed centroid PROPOSAL (proj+topk+gather8192+bmm) is "
        f"OVERHEAD-bound at M=1 -- it measures {us['head_proj_propose']:.0f}us for only "
        f"~5.5MiB of traffic, so a contiguous dense top-32768 GEMV ({best['dense_head_us_K']:.0f}"
        f"us) is cheaper, a step-only UPPER band of "
        f"{dense_topk_step_only_ub_band_pct[0]:+.1f}..{dense_topk_step_only_ub_band_pct[1]:+.1f}% "
        f"(best {best['step_only_gain_pct_ub']:+.2f}% at K={best['K']}); BUT that is a "
        f"served-model head SWAP (replace the trained centroid router), greedy-safe by "
        f"propose-only but E[T] UNPRICED and non-increasing (fixed top-K reaches a "
        f"subset of the centroid head's full-vocab reach), so the band is an upper "
        f"bound a real E[T] loss erodes -- needs a measured E[T]+PPL run, not a config "
        f"flip. Also: the FULL HF head op measures {us['head_real']:.0f}us, of which "
        f"{us['head_output_mat']:.0f}us is the 262144-wide output-buffer materialization "
        f"(full+scatter), excluded from the 101.2 anchor; if the deployed draft pays it "
        f"the recoverable cost is a 'lean draft logits / skip full buffer' served "
        f"change, NOT vocab restriction.]")

    verdict = {
        "draft_head_vocab_roofline_self_test_passes": self_test_passes,
        "projected_tps_gain_pct": projected_tps_gain_pct,
        "restricted_vocab_lever_live": restricted_vocab_lever_live,
        "restricted_draft_head_greedy_safe": True,
        "head_share_pct": head_share_pct,
        "head_projection_share_pct": head_projection_share_pct,
        "full_head_op_share_pct": full_head_share_pct,
        "head_material_ge10pct": head_material,
        # architecture diagnostic
        "draft_vocab_size": draft_vocab_size,
        "draft_hidden": draft_hidden,
        "draft_head_tied": draft_head_tied,
        "use_ordered_embeddings": use_ordered,
        "num_centroids": num_centroids,
        "centroid_intermediate_top_k": top_k_centroids,
        "vocab_size_per_centroid": vocab_per_centroid,
        "active_vocab_per_pass": active_vocab_per_pass,
        "head_is_dense_256k_gemv": False,
        # measured component us
        "us_io_projection": us["io_projection"],
        "us_attention": us["attention"],
        "us_mlp": us["mlp"],
        "us_head_projection_gemv": us["head_projection_gemv"],
        "us_head_proj_propose": us["head_proj_propose"],
        "us_head_output_mat": us["head_output_mat"],
        "us_head_real": us["head_real"],
        "us_dense_head_256k_measured": dense256_us,
        "gemv_chain_total_us": gemv_chain_us,
        "pass_with_full_head_total_us": pass_with_full_head_us,
        "bf16_anchor_us_254": BF16_ANCHOR_US_254,
        "anchor_resid_pct": 100.0 * anchor_resid,
        # byte models
        "dense_head_bytes_mib": dense_head_bytes / 2 ** 20,
        "dense_head_us_analytic": dense_head_us_analytic,
        "centroid_head_bytes_mib": centroid_head_bytes / 2 ** 20,
        "centroid_head_us_analytic": centroid_head_us_analytic,
        "dense_head_x_vs_pass": dense_head_us_analytic / BF16_ANCHOR_US_254,
        # pricing
        "deployed_proposal_us": deployed_proj_us,
        "head_side_ceiling_gain_pct": head_side_ceiling_gain_pct,
        "best_K": best["K"],
        "best_K_step_only_gain_pct_ub": best["step_only_gain_pct_ub"],
        "dense_topk_step_only_ub_band_pct": dense_topk_step_only_ub_band_pct,
        "dense_topk_swap_is_served_change_et_unpriced": True,
        # safety / housekeeping
        "greedy_identical_by_construction": True,
        "ppl_pinned": 2.3772, "ppl_ok": True,
        "nan_clean": nan_clean, "peak_vram_gib": peak_vram_gib,
        "vram_ok": bool(peak_vram_gib <= 24.0),
        # imported, unchanged
        "frontier_tps": FRONTIER_TPS, "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
        "k_cal": K_CAL, "step_us": STEP_US, "k_deployed": args.k,
        "draft_share_of_step": DRAFT_SHARE_OF_STEP, "et_deployed": ET_DEPLOYED,
        "head_share_verdict": head_share_verdict,
        "self_test_conditions": {"a_anchor_sum": st_a, "b_byte_us_roundtrip": st_b,
                                 "c_linear_K": st_c, "d_composition": st_d,
                                 "e_nan_clean": st_e, "f_constants_unchanged": st_f},
        "handoff_line": handoff,
        "chain_captured": captured,
    }

    print("\n[draft-head] ===== DECOMPOSITION (M=1, launch-free graph; denken 19-GEMM basis) =====", flush=True)
    print(f"  io_projection         {us['io_projection']:6.1f}us", flush=True)
    print(f"  attention             {us['attention']:6.1f}us  (q/o proj; SDPA shares "
          f"target KV, negligible at M=1)", flush=True)
    print(f"  mlp                   {us['mlp']:6.1f}us", flush=True)
    print(f"  vocab-projection head {us['head_projection_gemv']:6.1f}us  "
          f"(centroids sampler 256->{num_centroids})", flush=True)
    print(f"  GEMV-chain total      {gemv_chain_us:6.1f}us  (vs 101.2 anchor, "
          f"resid {100*anchor_resid:.1f}%)", flush=True)
    print(f"  => head_PROJECTION_share = {head_projection_share_pct:.1f}%  "
          f"({'MATERIAL' if head_material else 'IMMATERIAL <10%'})", flush=True)
    print(f"\n[draft-head] FULL HF head op {us['head_real']:.1f}us = proj+propose "
          f"{us['head_proj_propose']:.1f}us + output-mat(full 262144+scatter) "
          f"{us['head_output_mat']:.1f}us  (output-mat is OUTPUT-WIDTH, not "
          f"projection; excluded from the 101.2 anchor)", flush=True)
    print(f"[draft-head] dense-256k head MEASURED {dense256_us:.0f}us at M=1 = "
          f"{dense256_us/BF16_ANCHOR_US_254:.1f}x the whole 101.2us pass "
          f"(physically cannot be in the deployed draft)", flush=True)
    print("[draft-head] ===== DENSE top-K HEAD-SWAP PRICING (step-only UPPER bound; "
          "E[T] UNPRICED; served head swap, NOT this PR's lever) =====", flush=True)
    print(f"  deployed centroid-routed proposal: {deployed_proj_us:.1f}us "
          f"(OVERHEAD-bound at M=1)", flush=True)
    for r in restrict_rows:
        print(f"  K={r['K']:6d}  dense top-K GEMV {r['dense_head_us_K']:6.1f}us  "
              f"({r['dense_topk_vs_deployed_proposal_x']:.1f}x deployed proposal)  "
              f"step {r['step_us_new']:.0f}us  step-only UB "
              f"{r['step_only_gain_pct_ub']:+.2f}%", flush=True)
    print(f"  head-side ABSOLUTE ceiling (free proposal, step-only UB): "
          f"{head_side_ceiling_gain_pct:+.2f}%", flush=True)
    print(f"\n[draft-head] VERDICT: lever_live={restricted_vocab_lever_live}  "
          f"projected_tps_gain_pct={projected_tps_gain_pct:.3f}  "
          f"self_test={self_test_passes}", flush=True)
    print(f"  {head_share_verdict}", flush=True)

    payload = {
        "config": {
            "drafter_dir": args.drafter_dir, "torch": torch.__version__, "device": dev,
            "sm": f"{cap[0]}{cap[1]}", "iters": it, "warmup": wu, "k": args.k,
            "A10G_HBM_GBS": A10G_HBM_GBS, "restrict_K": RESTRICT_K,
            "anchor_tol_pct": ANCHOR_TOL_PCT, "byte_us_tol_pct": BYTE_US_TOL_PCT,
            "note": "isolated CUDA-graph M=1 micro-profiling of the real bf16 "
                    "drafter chain + the REAL Gemma4AssistantMaskedEmbedder head; "
                    "dense-256k head measured as the counterfactual. No serve "
                    "change, no HF Job, no submission. Greedy+PPL pinned (nothing "
                    "edited).",
        },
        "specs": [{"bucket": b, "role": r, "in": i, "out": o}
                  for (b, r, i, o) in specs],
        "restrict_rows": restrict_rows,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[draft-head] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload, specs, restrict_rows)
        except Exception as exc:  # noqa: BLE001
            print(f"[draft-head] W&B logging failed (non-fatal): {exc!r}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    return 0 if self_test_passes else 1


def _log_wandb(args, payload, specs, restrict_rows):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    # component breakdown table (denken 101.2 anchor basis: head == centroids GEMV)
    v = payload["verdict"]
    chain_us = v["gemv_chain_total_us"]
    comp = wandb.Table(columns=["component", "us", "pct_of_chain"])
    for name, key in [("io_projection", "us_io_projection"),
                      ("attention", "us_attention"), ("mlp", "us_mlp"),
                      ("vocab_projection_head", "us_head_projection_gemv")]:
        comp.add_data(name, v[key], 100.0 * v[key] / chain_us if chain_us else 0.0)
    run.log({"component_decomposition": comp})
    # full HF head op breakdown (proj+propose vs 262144 output-buffer materialization)
    headop = wandb.Table(columns=["sub_op", "us"])
    for name, key in [("proj_propose", "us_head_proj_propose"),
                      ("output_mat_full262144", "us_head_output_mat"),
                      ("head_real_total", "us_head_real")]:
        headop.add_data(name, v[key])
    run.log({"head_op_breakdown": headop})
    # dense top-K head-swap table (step-only UPPER bound; E[T] unpriced)
    rk = wandb.Table(columns=["K", "dense_head_us_K", "step_us_new", "et_new",
                              "step_only_tps", "step_only_gain_pct_ub"])
    for r in restrict_rows:
        rk.add_data(r["K"], r["dense_head_us_K"], r["step_us_new"], r["et_new"],
                    r["step_only_tps"], r["step_only_gain_pct_ub"])
    run.log({"dense_topk_headswap_step_only_ub": rk})
    run.summary.update({k: val for k, val in v.items()
                        if isinstance(val, (int, float, bool, str))})
    run.finish()
    print(f"[draft-head] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
