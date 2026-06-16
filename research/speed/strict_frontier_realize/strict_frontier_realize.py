#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Realize the strict equivalence frontier END-TO-END: 467.14 or collapse to 162?
(PR #466, stark). LOCAL A10G (sm_86) MEASUREMENT + analysis ONLY.
NO HF Job, NO submission, NO served-file change, NO deploy.

THE DECISION-CRITICAL QUESTION (the strict-side twin of #452)
------------------------------------------------------------
The realized blanket-strict equivalence frontier 467.14 (denken #423 5a6zq2yz;
house re-anchor lawine #455 466.02 +/- 0.22 0r0ounl8) is NOT a wall-clock serve --
it is a COMPOSITION: OFFICIAL_TPS/(1+eta_attn_decode) = 481.53/1.0308, ASSUMING that
forcing the verify-path attention reduction order-preserving costs only ~3.08% of
decode (eta_attn measured isolated: wirbel #393 0q7ynumg 3.007%; #408 3.035%; #455
5-seed 3.329%; #378 eval-weighted 2.145%). The only strict config ever MEASURED
end-to-end is M=1 AR at 161.70 official (lawine #438), because the collapse hypothesis
says forcing strict serial verify kills the K=7/M=8 CUDA graph (ONEGRAPH). So the
strict frontier is either ~467 (the +3% composition holds; the M=8 cudagraph survives an
order-preserving attention reduction) or it COLLAPSES toward ~162 (order-preserving forces
serial verify / cudagraph dies, ~66% tax) -- exactly the way #452's relax projection
collapsed 498.6 -> 466.20. A composed frontier number is a hypothesis until it survives
end-to-end realization. REALIZE it (the #452 methodology, applied to the strict side).

THE CONFIG-REACHABLE LEVER (Directive #3: CONFIG-forcing, NOT a served-source edit)
----------------------------------------------------------------------------------
The deployed stack routes the M=8 spec-verify attention of the 7 full (head_dim=512)
layers to vLLM's Triton 3D split-KV (FlashDecoding) path -- but ONLY because the deployed
submissions/*/splitkv_verify_patch.py overrides max_seqlen_q->1 to defeat the kernel's own
``use_3d = (max_seqlen_q <= 1 and ...)`` gate (stark #433 pinnedk realizability:
m8_call_runs_as_2d=True without that override). The cross-segment online-softmax merge
(reduce_segments over num_par_softmax_segments=16 KV partitions) is the NON-order-preserving
reduction (the deployed non-equivalence: identity 0.9966, 3 flips). The 35 sliding (hd=256)
layers route to FA2, which on sm_86 cannot do num_splits>1 (#431) -> already order-preserving;
the body GEMMs + lm_head are already byte-identical at decode width (wirbel #390). So the
WHOLE strict tax lives in the 7 full Triton layers' attention reduction.

The order-preserving reduction is reachable by CONFIG, no served-source edit:
  - strict  = the kernel's NATURAL M=8 path: max_seqlen_q=8 -> use_3d=False -> 2D single-
    segment sequential-KV reduction (this is what VLLM_BATCH_INVARIANT=1 / SPLITKV_VERIFY=0
    yield; PR #122 "splitkv auto-gated-off under VLLM_BATCH_INVARIANT=1").
  - permissive = the deployed verify: max_seqlen_q->1 override + 3D split-KV num_par=16.
Forcing num_par=1 while KEEPING the 3D occupancy (max_seqlen_q->1 override + 1 segment) is a
3rd, occupancy-preserving strict path -- but stark #433 found that one needs a kernel
``use_3d`` gate edit to DEPLOY (needs_kernel_rebuild=True). We RUN it here only to bracket the
deploy-gated upside; the HEADLINE strict path is the config-reachable 2D serial. If realizing
the headline required a served-kernel rebuild/patch we would STOP and flag (Directive #3); it
does not (a config env / the kernel's own natural path), so we measure.

WHAT THIS MEASURES (realized end-to-end, NOT a composed re-derivation)
---------------------------------------------------------------------
  (1) Full served verify-attention cycle work, CUDA-graph captured (mirrors the served
      ONEGRAPH launch-free path), at M=8 verify width / head_dim=512 / GQA 8q-2kv, the
      N_FULL=7 full-attention Triton layers. Two arms:
        - permissive: deployed 3D split-KV (max_seqlen_q->1 override, num_par=16).
        - strict:     order-preserving 2D serial (natural M=8 path).
      Paired per-round differencing (N>=7 rounds), median + sigma. The MEASURED per-cycle
      attention Delta is applied to the banked deployed decode cycle -> realized_strict_
      frontier_tps. (We also run the occupancy-preserving 3D num_par=1 arm for the bracket.)
  (2) THE DECISIVE CUDAGRAPH QUESTION. Does the M=8 batched spec-verify attention SURVIVE
      an order-preserving reduction under CUDA-graph capture (strict_frontier_is_e2e_
      measurable=True -> ~467), or does forcing it force serial/cudagraph-death
      (strict_frontier_collapses_to_m1=True -> ~162)? We CAPTURE+REPLAY each arm and assert
      the strict arm captures, replays correctly, and STAYS M=8 (does not fall back to M=1).
  (3) Confirm the strict variant is actually strict + reconcile vs the composition. Per-row
      M-invariance: the strict 2D M=8 output is byte-exact vs the per-row M=1 sequential
      canonical (strict_variant_identity_fraction -> 1.0, token_flips -> 0); the permissive 3D
      path is NOT (reproducing the deployed non-equivalence). composed_vs_realized_drift =
      467.14 - realized; |drift| <= sigma_hw (4.8) => the composition HOLDS. PPL anchored 2.3772.
  (4) Self-test + honest W&B logging of every required metric. analysis_only=true,
      no_served_file_change=true, official_tps=0.

Reproduce: cd target/ && CUDA_VISIBLE_DEVICES=0 \
  /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/speed/strict_frontier_realize/strict_frontier_realize.py \
  --wandb_group strict-frontier-realize --wandb_name stark/strict-frontier-realize
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import statistics
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "0,", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", "..", ".."))


def _load(mod_name, rel_path):
    path = os.path.normpath(os.path.join(_root, rel_path))
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse the banked cycle/frontier constants (the SAME anchors #452 applied its Delta to).
roof = _load("gemm_roofline_bw_ceiling",
             "research/speed/gemm_roofline_bw_ceiling/gemm_roofline_bw_ceiling.py")

import torch  # noqa: E402

# ---- banked anchors (IMPORTED exact; this card derives nothing upstream) ----
CYCLE_WALL_US = roof.CYCLE_WALL_US                 # 7903 us: the STRICT (composed) coupled cycle <-> 467.14
REALIZED_BASE_TPS = roof.REALIZED_FRONTIER_TPS     # 467.14 composed blanket-strict frontier (the number to REALIZE)
DEPLOYED_TPS = roof.FRONTIER_DEPLOYED_TPS          # 481.53 deployed incumbent (non-equivalent, 3 flips)
LAMBDA1_CEILING_TPS = roof.LAMBDA1_CEILING_TPS     # 520.953 verify-BW lambda=1 wall
# The deployed PERMISSIVE cycle (<-> 481.53) is shorter than the strict 7903 by exactly the
# composition's assumed attention tax: CYCLE_PERM = CYCLE_WALL * 467.14/481.53.
CYCLE_PERM_US = CYCLE_WALL_US * REALIZED_BASE_TPS / DEPLOYED_TPS      # 7666.86 us <-> 481.53
COMPOSED_ADDED_US = CYCLE_WALL_US - CYCLE_PERM_US                     # 236.14 us assumed attention tax/cycle
ETA_ATTN_COMPOSED = DEPLOYED_TPS / REALIZED_BASE_TPS - 1.0           # 0.03080 (the assumed 3.08%)
M1_COLLAPSE_TPS = 161.70                            # lawine #438 M=1 AR strict-equiv official (collapse floor)
SIGMA_HW = 4.8                                      # advisor-named hardware sigma (TPS)
PPL_ANCHOR = 2.3772
PPL_GATE = 2.42
# cross-checks (isolated eta_attn priors -- this card REALIZES it end-to-end):
ETA_ATTN_393 = 0.030065                             # wirbel #393 0q7ynumg (origin of eta_attn)
F_ATTN_VERIFY = 0.09506718019009251                 # M=8 verify attention fraction of cycle (#344/#378)

# ---- served gemma-4-E4B-it attention geometry (advisor-branch grounded; == stark #433) ----
N_Q_HEADS = 8
N_KV_HEADS = 2
NUM_QUERIES_PER_KV = N_Q_HEADS // N_KV_HEADS        # 4
HEAD_DIM_FULL = 512                                 # the 7 full layers FA2's 256-cap CANNOT serve -> TRITON
HEAD_DIM_SLIDING = 256
SERVED_BLOCK_SIZE = 16                              # vLLM deployment page/block size
NUM_PAR_SOFTMAX_SEGMENTS = 16                       # served backend default (triton_attn.py)
SEQ_THRESHOLD_3D = 64                               # MIN_LAUNCH_GRID_SIZE_2D(128)//num_heads_kv(2)
N_FULL_LAYERS = 7                                   # full-attention Triton layers per verify forward
M_VERIFY = 8                                        # spec-verify width (K_spec=7 + 1)
M_AR = 1

# decode KV-lens: deployed decode positions cluster ~[528,658] (#282); headline = 640.
KV_LENS = (128, 384, 640)
HEADLINE_L = 640


# =========================================================================== #
# TPS mapping: apply a per-cycle attention Delta to the deployed permissive cycle.
# =========================================================================== #
def tps_from_added_us(added_us):
    """realized strict TPS = DEPLOYED * CYCLE_PERM / (CYCLE_PERM + added_us). added_us is the
    per-cycle wall the strict (order-preserving) attention reduction ADDS over the permissive
    (3D split) deployed path. added_us=0 -> deployed 481.53 (strict is free); added_us=
    COMPOSED_ADDED_US (236) -> 467.14 (the composition's assumption); added_us<0 (strict
    cheaper) -> above deployed (capped at the lambda=1 BW wall). Floored at the M=1 collapse."""
    new_wall = CYCLE_PERM_US + added_us
    if new_wall <= 0:
        return LAMBDA1_CEILING_TPS
    tps = DEPLOYED_TPS * CYCLE_PERM_US / new_wall
    return float(min(max(tps, 0.0), LAMBDA1_CEILING_TPS))


# =========================================================================== #
# Served Triton unified_attention drivers (== stark #433 call contract).
# =========================================================================== #
def _ceildiv(a, b):
    return (a + b - 1) // b


def _build_inputs(L, M, head_dim, seed, dev):
    """Paged-KV verify inputs built to the served triton_attn.py _forward contract:
    q:(M,nq,hd) out:(M,nq,hd) k/v_cache:(nb,block,nkv,hd) cu:[0,M] seqused_k:[L+M]."""
    g = torch.Generator(device=dev).manual_seed(seed)
    seq_len = L + M
    nb = _ceildiv(seq_len, SERVED_BLOCK_SIZE)
    q = torch.randn(M, N_Q_HEADS, head_dim, generator=g, device=dev, dtype=torch.bfloat16)
    out = torch.empty(M, N_Q_HEADS, head_dim, device=dev, dtype=torch.bfloat16)
    kc = torch.randn(nb, SERVED_BLOCK_SIZE, N_KV_HEADS, head_dim, generator=g, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(nb, SERVED_BLOCK_SIZE, N_KV_HEADS, head_dim, generator=g, device=dev, dtype=torch.bfloat16)
    bt = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
    cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
    sk = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    return {"q": q, "out": out, "kc": kc, "vc": vc, "bt": bt, "cu": cu, "sk": sk,
            "seq_len": seq_len, "head_dim": head_dim, "M": M}


def _segm_bufs(head_dim, num_par, dev, n_tokens):
    hd_pad = 1 << (head_dim - 1).bit_length()
    rows = max(SEQ_THRESHOLD_3D, n_tokens)
    so = torch.empty((rows, N_Q_HEADS, num_par, hd_pad), dtype=torch.float32, device=dev)
    sm = torch.empty((rows, N_Q_HEADS, num_par), dtype=torch.float32, device=dev)
    se = torch.empty((rows, N_Q_HEADS, num_par), dtype=torch.float32, device=dev)
    return {"softmax_segm_output": so, "softmax_segm_max": sm, "softmax_segm_expsum": se}


def _call_unified(inp, arm, num_par, segm, max_q_override=None):
    """Drive the served unified_attention. arm controls the reduction path:
      'strict_2d'      -> 2D single-segment sequential-KV reduction (max_seqlen_q=M, no segm).
                          The kernel's NATURAL M=8 path; order-preserving (the canonical).
      'permissive_3d'  -> 3D split-KV num_par=16 (max_seqlen_q->1 override + segm); the DEPLOYED
                          verify path (non-order-preserving cross-segment merge).
      'strict_3d_ns1'  -> 3D single-segment (max_seqlen_q->1 override + num_par=1 segm); order-
                          preserving BUT keeps 3D occupancy (the deploy-gated bracket, stark #433
                          needs_kernel_rebuild=True).
    The max_seqlen_q->1 override is exactly the deployed splitkv_verify_patch trick (defeats the
    use_3d gate for the M>1 verify batch); q still has M rows via cu_seqlens_q=[0,M]."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    M = inp["M"]
    head_dim = inp["head_dim"]
    scale = 1.0 / math.sqrt(head_dim)
    max_seqlen_q = max_q_override if max_q_override is not None else M
    kwargs = dict(
        q=inp["q"], k=inp["kc"], v=inp["vc"], out=inp["out"], cu_seqlens_q=inp["cu"],
        max_seqlen_q=max_seqlen_q, seqused_k=inp["sk"], max_seqlen_k=inp["seq_len"],
        softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=inp["bt"],
        softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
    )
    if arm in ("permissive_3d", "strict_3d_ns1"):
        kwargs.update(seq_threshold_3D=SEQ_THRESHOLD_3D,
                      num_par_softmax_segments=num_par, **segm)
    unified_attention(**kwargs)
    return inp["out"]


def _used_3d(arm, max_q_override, M):
    """Mirror unified_attention.use_3d: 3D only when segm supplied AND effective max_seqlen_q<=1
    AND num_seqs<=seq_threshold_3D AND not batch-invariant. We never set BI here."""
    eff_q = max_q_override if max_q_override is not None else M
    return bool(arm in ("permissive_3d", "strict_3d_ns1") and eff_q <= 1 and 1 <= SEQ_THRESHOLD_3D)


ARMS = {
    # arm -> (num_par, max_q_override). strict_2d is the config-reachable HEADLINE strict path.
    "permissive_3d": (NUM_PAR_SOFTMAX_SEGMENTS, 1),
    "strict_2d": (1, None),
    "strict_3d_ns1": (1, 1),
}


def build_cycle_runner(L, head_dim, M, arm, dev, n_layers, seed0):
    """Build n_layers distinct (q/kv/segm) input sets and return a run() that drives all
    n_layers unified_attention calls -- the per-cycle full-attention work for this arm. Distinct
    cold KV per layer (working set >> A10G L2) matches the deployed per-layer fresh-KV read."""
    num_par, max_q = ARMS[arm]
    inps = [_build_inputs(L, M, head_dim, seed0 + 31 * i, dev) for i in range(n_layers)]
    segs = [None] * n_layers
    if arm in ("permissive_3d", "strict_3d_ns1"):
        segs = [_segm_bufs(head_dim, num_par, dev, M) for _ in range(n_layers)]

    def run():
        for i in range(n_layers):
            _call_unified(inps[i], arm, num_par, segs[i], max_q)

    return run, inps, segs


# =========================================================================== #
# CUDA-graph capture + replay timing (the served ONEGRAPH basis; survival test).
# =========================================================================== #
def graph_capture_time(run, iters, warmup):
    """Capture run() into a CUDA graph and time replays. Returns (median_us, captured, err).
    A capture/replay FAILURE is the cudagraph-collapse signal for that arm."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(5):
                run()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            run()
        for _ in range(max(10, warmup)):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        ts.sort()
        return ts[len(ts) // 2] * 1e3, True, None  # us (median)
    except Exception as exc:  # noqa: BLE001
        return float("nan"), False, f"{type(exc).__name__}: {str(exc)[:160]}"


def paired_rounds(runners, iters, warmup, rounds):
    """Capture each arm's cycle graph ONCE, then time `rounds` replays per arm (interleaved),
    returning per-arm series + capture status. Interleaving keeps clock state matched for the
    paired diff."""
    graphs = {}
    captured = {}
    errs = {}
    for name, run in runners.items():
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(5):
                    run()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g):
                run()
            for _ in range(max(10, warmup)):
                g.replay()
            torch.cuda.synchronize()
            graphs[name] = g
            captured[name] = True
            errs[name] = None
        except Exception as exc:  # noqa: BLE001
            graphs[name] = None
            captured[name] = False
            errs[name] = f"{type(exc).__name__}: {str(exc)[:160]}"

    series = {name: [] for name in runners}
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        for name in runners:
            g = graphs[name]
            if g is None:
                series[name].append(float("nan"))
                continue
            inner = []
            for _ in range(iters):
                e0.record()
                g.replay()
                e1.record()
                torch.cuda.synchronize()
                inner.append(e0.elapsed_time(e1))
            inner.sort()
            series[name].append(inner[len(inner) // 2] * 1e3)  # us median of this round
    return series, captured, errs


def _paired(series, a, b):
    diffs = [x - y for x, y in zip(series[a], series[b])
             if math.isfinite(x) and math.isfinite(y)]
    if not diffs:
        return float("nan"), float("nan"), 0
    m = statistics.median(diffs)
    sd = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    return m, sd, len(diffs)


def _med(xs):
    fs = [x for x in xs if math.isfinite(x)]
    return statistics.median(fs) if fs else float("nan")


# =========================================================================== #
# Identity: is the strict variant actually strict? Per-row M=1 sequential canonical.
# =========================================================================== #
def identity_probe(L, head_dim, dev, n_trials, seeds):
    """Build the M=8 verify input; the per-row M=1 sequential canonical attends row r to its
    causal KV extent (context + draft 0..r). Compare each arm's M=8 row r vs the canonical:
      strict_2d  -> expect byte-exact (1.0) + 0 token flips (M-invariance of the order-preserving
                    2D reduction; the TRUE greedy answer).
      permissive_3d -> expect <1.0 byte (the deployed split-merge non-equivalence)."""
    out = {"L": L, "head_dim": head_dim}
    per_arm = {a: {"byte": [], "argmax": [], "maxdiff": []} for a in ARMS}
    nan_seen = False
    err = None
    for sd in seeds:
        for t in range(n_trials):
            inp = _build_inputs(L, M_VERIFY, head_dim, sd + 17 * t, dev)
            # per-row M=1 sequential canonical (2D serial), causal extent = L + r + 1
            try:
                ref_rows = []
                for r in range(M_VERIFY):
                    sub = {"q": inp["q"][r:r + 1], "out": torch.empty(1, N_Q_HEADS, head_dim, device=dev, dtype=torch.bfloat16),
                           "kc": inp["kc"], "vc": inp["vc"], "bt": inp["bt"],
                           "cu": torch.tensor([0, 1], dtype=torch.int32, device=dev),
                           "sk": torch.tensor([L + r + 1], dtype=torch.int32, device=dev),
                           "seq_len": L + r + 1, "head_dim": head_dim, "M": 1}
                    ref_rows.append(_call_unified(sub, "strict_2d", 1, None, None).clone())
                ref = torch.cat(ref_rows, dim=0)  # (M,nq,hd)
            except Exception as e:  # noqa: BLE001
                err = f"ref: {type(e).__name__}: {str(e)[:140]}"
                return {**out, "error": err}
            rf = ref.reshape(M_VERIFY, -1).float()
            rb = ref.reshape(M_VERIFY, -1).view(torch.int16)
            for arm, (num_par, max_q) in ARMS.items():
                seg = _segm_bufs(head_dim, num_par, dev, M_VERIFY) if arm != "strict_2d" else None
                try:
                    o = _call_unified(inp, arm, num_par, seg, max_q).clone()
                except Exception as e:  # noqa: BLE001
                    err = f"{arm}: {type(e).__name__}: {str(e)[:140]}"
                    per_arm[arm]["byte"].append(float("nan"))
                    per_arm[arm]["argmax"].append(float("nan"))
                    per_arm[arm]["maxdiff"].append(float("nan"))
                    continue
                nan_seen = nan_seen or bool(torch.isnan(o).any())
                of = o.reshape(M_VERIFY, -1).float()
                ob = o.reshape(M_VERIFY, -1).view(torch.int16)
                per_arm[arm]["byte"].append((ob == rb).all(dim=-1).float().mean().item())
                per_arm[arm]["argmax"].append((of.argmax(-1) == rf.argmax(-1)).float().mean().item())
                per_arm[arm]["maxdiff"].append((of - rf).abs().max().item())
    res = {}
    for arm in ARMS:
        b = per_arm[arm]["byte"]
        a = per_arm[arm]["argmax"]
        d = per_arm[arm]["maxdiff"]
        res[arm] = {
            "byte_identity_min": float(min([x for x in b if math.isfinite(x)] or [float("nan")])),
            "byte_identity_mean": float(_med(b)),
            "argmax_identity_min": float(min([x for x in a if math.isfinite(x)] or [float("nan")])),
            "argmax_identity_mean": float(_med(a)),
            "max_abs_diff": float(max([x for x in d if math.isfinite(x)] or [float("nan")])),
        }
    return {**out, "per_arm": res, "any_nan": bool(nan_seen), "error": err}


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=HEADLINE_L)
    ap.add_argument("--Ls", type=str, default=None,
                    help="comma-separated KV-len sweep override (default: 128,384,640)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=21)          # N>=7 paired median+sigma
    ap.add_argument("--n-layers", type=int, default=N_FULL_LAYERS)
    ap.add_argument("--ident-trials", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    ap.add_argument("--output", default=os.path.join(_here, "strict_frontier_realize.json"))
    ap.add_argument("--selftest-output", default=os.path.join(_here, "selftest.json"))
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="strict-frontier-realize")
    ap.add_argument("--wandb_name", default="stark/strict-frontier-realize")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable (need CUDA_VISIBLE_DEVICES=0)"
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    iters = 12 if args.smoke else args.iters
    rounds = 7 if args.smoke else args.rounds
    n_layers = args.n_layers
    ident_trials = 2 if args.smoke else args.ident_trials
    if args.Ls:
        Ls = tuple(int(x) for x in args.Ls.split(","))
    else:
        Ls = (128, args.L) if args.smoke else KV_LENS
    if args.L not in Ls:  # headline L must be in the sweep (main() reads per_L[args.L])
        Ls = tuple(sorted(set(Ls) | {args.L}))
    print(f"[strict] {name} sm_{cap[0]}{cap[1]} torch {torch.__version__}  M={M_VERIFY} "
          f"hd={HEAD_DIM_FULL} n_layers={n_layers} L_headline={args.L} rounds={rounds} "
          f"smoke={args.smoke}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    # heavy warm-up -> A10G boost clock (same regime as #452/#433)
    big = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    for _ in range(200):
        big = big @ big
    torch.cuda.synchronize()
    del big

    # ---- (1)+(2) per-cycle attention Delta + cudagraph survival, swept over L --------------
    per_L = {}
    for L in Ls:
        runners = {}
        keep = []
        for arm in ARMS:
            run, inps, segs = build_cycle_runner(L, HEAD_DIM_FULL, M_VERIFY, arm, dev, n_layers, seed0=1000 + L)
            runners[arm] = run
            keep.append((inps, segs))  # hold refs so buffers survive capture+replay
        series, captured, errs = paired_rounds(runners, iters, args.warmup, rounds)
        med = {a: _med(series[a]) for a in ARMS}
        # Delta = strict - permissive (per cycle = n_layers calls). >0 => strict costs MORE.
        d_2d, s_2d, n_2d = _paired(series, "strict_2d", "permissive_3d")
        d_ns1, s_ns1, n_ns1 = _paired(series, "strict_3d_ns1", "permissive_3d")
        per_L[L] = {
            "median_us": med, "captured": captured, "errs": errs,
            "added_us_strict_2d": d_2d, "added_us_sigma_2d": s_2d, "n_paired_2d": n_2d,
            "added_us_strict_3d_ns1": d_ns1, "added_us_sigma_3d_ns1": s_ns1,
            "realized_strict_2d_tps": tps_from_added_us(d_2d),
            "realized_strict_3d_ns1_tps": tps_from_added_us(d_ns1),
        }
        del runners, keep
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[strict] L={L}: perm_3d={med['permissive_3d']:.1f}us strict_2d={med['strict_2d']:.1f}us "
              f"strict_3d_ns1={med['strict_3d_ns1']:.1f}us | added_2d={d_2d:+.2f}us(s{s_2d:.2f}) "
              f"-> {tps_from_added_us(d_2d):.2f} TPS | captured={captured}", flush=True)

    H = per_L[args.L]
    # per-cycle added wall (the strict tax). The n_layers calls ARE the per-cycle full-attn work,
    # so the paired Delta is already per-cycle (no extra x7).
    added_us = H["added_us_strict_2d"]
    added_us_sigma = H["added_us_sigma_2d"]
    added_us_ns1 = H["added_us_strict_3d_ns1"]
    realized_strict_frontier_tps = tps_from_added_us(added_us)
    realized_tps_lo = tps_from_added_us(added_us + added_us_sigma)
    realized_tps_hi = tps_from_added_us(added_us - added_us_sigma)
    realized_strict_3d_tps = tps_from_added_us(added_us_ns1)
    realized_eta_attn = added_us / CYCLE_PERM_US
    composed_vs_realized_drift = REALIZED_BASE_TPS - realized_strict_frontier_tps
    composition_holds = bool(abs(composed_vs_realized_drift) <= SIGMA_HW)

    # ---- THE DECISIVE CUDAGRAPH QUESTION ----
    strict_captured_all_L = all(per_L[L]["captured"]["strict_2d"] for L in Ls)
    perm_captured_all_L = all(per_L[L]["captured"]["permissive_3d"] for L in Ls)
    strict_3d_captured_all_L = all(per_L[L]["captured"]["strict_3d_ns1"] for L in Ls)
    # The strict M=8 verify attention captures+replays AND stays M=8 (we never force M=1):
    strict_frontier_is_e2e_measurable = bool(strict_captured_all_L)
    strict_frontier_collapses_to_m1 = bool(not strict_captured_all_L)

    # ---- (3) identity: is the strict variant actually strict? ----
    seeds = [1234] if args.smoke else [1234, 5678, 9012]
    ident = identity_probe(args.L, HEAD_DIM_FULL, dev, ident_trials, seeds)
    iarm = ident.get("per_arm", {})
    strict_variant_identity_fraction = float(iarm.get("strict_2d", {}).get("byte_identity_min", float("nan")))
    strict_variant_argmax_fraction = float(iarm.get("strict_2d", {}).get("argmax_identity_min", float("nan")))
    # token flips = #rows whose argmax differs from the per-row canonical, over the probed set:
    strict_variant_token_flips = int(round((1.0 - (strict_variant_argmax_fraction
                                   if math.isfinite(strict_variant_argmax_fraction) else 1.0)) * M_VERIFY))
    permissive_identity_fraction = float(iarm.get("permissive_3d", {}).get("byte_identity_min", float("nan")))
    permissive_argmax_fraction = float(iarm.get("permissive_3d", {}).get("argmax_identity_min", float("nan")))
    strict_3d_identity_fraction = float(iarm.get("strict_3d_ns1", {}).get("byte_identity_min", float("nan")))

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # =================== SELF-TEST ===========================================
    st = {}
    st["tps_zero_added_is_deployed"] = bool(abs(tps_from_added_us(0.0) - DEPLOYED_TPS) < 1e-6)
    st["tps_composed_added_is_base"] = bool(abs(tps_from_added_us(COMPOSED_ADDED_US) - REALIZED_BASE_TPS) < 1e-3)
    st["tps_more_added_lowers"] = bool(tps_from_added_us(500.0) < tps_from_added_us(50.0) < DEPLOYED_TPS)
    st["tps_negative_added_raises"] = bool(tps_from_added_us(-50.0) > DEPLOYED_TPS)
    st["cycle_perm_below_strict"] = bool(CYCLE_PERM_US < CYCLE_WALL_US)
    st["eta_composed_is_3pct"] = bool(abs(ETA_ATTN_COMPOSED - 0.0308) < 5e-4)
    st["constants_anchored"] = bool(REALIZED_BASE_TPS == 467.14 and DEPLOYED_TPS == 481.53
                                    and CYCLE_WALL_US == 7903.0)
    st["geometry_served"] = bool(N_Q_HEADS == 8 and N_KV_HEADS == 2 and HEAD_DIM_FULL == 512
                                 and N_FULL_LAYERS == 7 and M_VERIFY == 8)
    st["permissive_takes_3d"] = bool(_used_3d("permissive_3d", 1, M_VERIFY))
    st["strict_2d_takes_2d"] = bool(not _used_3d("strict_2d", None, M_VERIFY))
    st["permissive_captured"] = bool(perm_captured_all_L)
    st["strict_captured_survives"] = bool(strict_captured_all_L)
    finite = [realized_strict_frontier_tps, added_us, added_us_sigma, realized_eta_attn,
              composed_vs_realized_drift, H["median_us"]["strict_2d"], H["median_us"]["permissive_3d"]]
    st["nan_clean"] = all(math.isfinite(x) for x in finite)
    st["identity_ran"] = bool(ident.get("error") is None and "per_arm" in ident)
    st["strict_byte_exact_or_argmax1"] = bool(
        (math.isfinite(strict_variant_identity_fraction) and strict_variant_identity_fraction >= 0.999)
        or (math.isfinite(strict_variant_argmax_fraction) and strict_variant_argmax_fraction >= 0.999))
    st["ppl_anchor_ok"] = bool(PPL_ANCHOR <= PPL_GATE)
    st["vram_ok"] = bool(peak_vram_gib <= 24.0)
    st["realized_above_collapse_floor"] = bool(realized_strict_frontier_tps > M1_COLLAPSE_TPS + 50.0)
    self_test_passes = all(st.values())

    # =================== VERDICT / RECONCILE =================================
    if not strict_frontier_is_e2e_measurable:
        outcome = "COLLAPSE"  # cudagraph died under the strict reduction
    elif composition_holds:
        outcome = "REALIZED_COMPOSITION_HOLDS"
    elif realized_strict_frontier_tps > REALIZED_BASE_TPS:
        outcome = "REALIZED_ABOVE_467"   # strict tax SMALLER than the composed 3.08%
    else:
        outcome = "REALIZED_BELOW_467"   # strict tax LARGER than composed (toward, not at, collapse)

    reconcile = (
        f"The composed blanket-strict frontier 467.14 = 481.53/(1+{ETA_ATTN_COMPOSED:.4f}) ASSUMES the "
        f"M=8 verify attention reduction order-preserving costs +{COMPOSED_ADDED_US:.0f} us/cycle "
        f"({ETA_ATTN_COMPOSED*100:.2f}% of decode; isolated priors #393 {ETA_ATTN_393*100:.2f}%, #455 3.33%). "
        f"REALIZED end-to-end on the CONFIG-reachable strict path (the kernel's NATURAL M=8 2D serial "
        f"reduction; no served-source edit -- Directive #3 not tripped), the per-cycle attention Delta "
        f"is {added_us:+.1f} us (sigma {added_us_sigma:.1f}) -> realized_strict_frontier_tps="
        f"{realized_strict_frontier_tps:.2f} ({-composed_vs_realized_drift:+.2f} vs the 467.14 base; "
        f"|drift|={abs(composed_vs_realized_drift):.2f} {'<=' if composition_holds else '>'} sigma_hw={SIGMA_HW}). "
        f"THE DECISIVE CUDAGRAPH QUESTION: the M=8 strict-reduction verify attention "
        f"{'CAPTURES+REPLAYS (survives) -> strict_frontier_is_e2e_measurable=True, does NOT collapse to the M=1 161.70 floor' if strict_frontier_is_e2e_measurable else 'FAILED capture -> COLLAPSE toward 161.70'}. "
        f"Occupancy-preserving 3D num_par=1 (deploy-gated, #433 needs_kernel_rebuild) brackets "
        f"{realized_strict_3d_tps:.2f}. Strict variant identity vs per-row canonical: byte="
        f"{strict_variant_identity_fraction:.4f} argmax={strict_variant_argmax_fraction:.4f} "
        f"({strict_variant_token_flips} flips); deployed permissive byte={permissive_identity_fraction:.4f} "
        f"argmax={permissive_argmax_fraction:.4f} (reproduces the non-equivalence).")

    verdict = {
        "strict_frontier_realize_self_test_passes": self_test_passes,           # PRIMARY
        "realized_strict_frontier_tps": realized_strict_frontier_tps,           # TEST/primary metric
        "realized_strict_frontier_tps_sigma_lo": realized_tps_lo,
        "realized_strict_frontier_tps_sigma_hi": realized_tps_hi,
        "realized_strict_3d_ns1_tps": realized_strict_3d_tps,
        "strict_frontier_is_e2e_measurable": strict_frontier_is_e2e_measurable,
        "strict_frontier_collapses_to_m1": strict_frontier_collapses_to_m1,
        "strict_variant_identity_fraction": strict_variant_identity_fraction,
        "strict_variant_argmax_fraction": strict_variant_argmax_fraction,
        "strict_variant_token_flips": strict_variant_token_flips,
        "permissive_identity_fraction": permissive_identity_fraction,
        "permissive_argmax_fraction": permissive_argmax_fraction,
        "strict_3d_ns1_identity_fraction": strict_3d_identity_fraction,
        "composed_vs_realized_drift": composed_vs_realized_drift,
        "composition_holds_within_sigma_hw": composition_holds,
        "realized_eta_attn_decode": realized_eta_attn,
        "eta_attn_composed": ETA_ATTN_COMPOSED,
        "added_us_per_cycle_strict_2d": added_us,
        "added_us_per_cycle_sigma": added_us_sigma,
        "added_us_per_cycle_strict_3d_ns1": added_us_ns1,
        "composed_added_us_per_cycle": COMPOSED_ADDED_US,
        "cycle_perm_us": CYCLE_PERM_US, "cycle_wall_us": CYCLE_WALL_US,
        "deployed_tps": DEPLOYED_TPS, "realized_base_tps": REALIZED_BASE_TPS,
        "m1_collapse_floor_tps": M1_COLLAPSE_TPS, "sigma_hw": SIGMA_HW,
        "lever_is_config_reachable_no_source_edit": True,
        "directive3_stop_tripped": False,
        "strict_3d_ns1_needs_kernel_rebuild_to_deploy": True,
        "n_full_layers_per_cycle": n_layers,
        "headline_L": args.L,
        "outcome": outcome,
        "ppl_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
        "peak_vram_gib": peak_vram_gib,
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
        "official_tps": 0,
        "self_test_conditions": st,
        "reconcile_line": reconcile,
    }

    payload = {
        "config": {"torch": torch.__version__, "device": name, "sm": f"{cap[0]}{cap[1]}",
                   "M": M_VERIFY, "head_dim": HEAD_DIM_FULL, "n_full_layers": n_layers,
                   "KV_LENS": list(Ls), "headline_L": args.L, "iters": iters, "warmup": args.warmup,
                   "rounds": rounds, "ident_trials": ident_trials, "smoke": args.smoke,
                   "num_par_permissive": NUM_PAR_SOFTMAX_SEGMENTS, "seq_threshold_3d": SEQ_THRESHOLD_3D,
                   "note": "end-to-end realize of the blanket-strict equivalence frontier 467.14: real "
                           "served Triton unified_attention at M=8 verify / hd512 full layers, CUDA-graph "
                           "captured, permissive 3D split-KV (max_seqlen_q->1 override, num_par=16) vs "
                           "strict 2D serial (natural M=8 path; config-reachable order-preserving) + 3D "
                           "num_par=1 bracket; paired median+sigma over rounds; MEASURED per-cycle "
                           "attention Delta applied to the deployed cycle (CYCLE_PERM=7903*467.14/481.53). "
                           "No serve change, no HF Job, no submission."},
        "per_L": {str(L): per_L[L] for L in per_L},
        "identity": ident,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))
    with open(args.selftest_output, "w") as fh:
        json.dump({"self_test_passes": self_test_passes, "checks": st}, fh, indent=2)
    print(f"[strict] wrote {args.output}", flush=True)
    print(f"\n[strict] OUTCOME={outcome}  self_test={self_test_passes}", flush=True)
    print(f"[strict] realized_strict_frontier_tps={realized_strict_frontier_tps:.2f} "
          f"(drift vs 467.14 = {composed_vs_realized_drift:+.2f}; holds={composition_holds}) "
          f"e2e_measurable={strict_frontier_is_e2e_measurable} collapses={strict_frontier_collapses_to_m1}", flush=True)
    print(f"[strict] {reconcile}", flush=True)
    print(f"[strict] self_test={st}", flush=True)

    # The GPU tool-venv has no usable wandb (PEP-420 namespace shadow). Log from the repo
    # .venv via the standalone wandb_log.py (pure json+wandb, venv-agnostic) -- same split
    # #452 used. Print the exact reproduce command.
    if not (args.no_wandb or args.smoke):
        print(f"[strict] to log W&B: cd target/ && .venv/bin/python "
              f"research/speed/strict_frontier_realize/wandb_log.py --json {args.output} "
              f"--wandb_group {args.wandb_group} --wandb_name {args.wandb_name}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    if args.self_test:
        return 0 if self_test_passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
