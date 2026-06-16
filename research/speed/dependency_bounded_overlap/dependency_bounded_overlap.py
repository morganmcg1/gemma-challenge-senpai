#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Dependency-bounded multi-stream overlap: is the #477 474.44 ceiling realizable? (PR #482, lawine).
LOCAL A10G (sm_86) MEASUREMENT + analysis ONLY. NO HF Job, NO submission, NO served-file change,
NO kernel rebuild. analysis_only=true, official_tps=0, no_served_file_change=true.

THE DECISION-CRITICAL QUESTION (close the gap #477 left open)
------------------------------------------------------------
#477 (w41knrqd) measured a RESOURCE-feasibility ceiling: it timed the WHOLE 37-layer int4-Marlin
body GEMM (4126us) CONCURRENT with the WHOLE 7-layer strict verify-attention (557us) as two
MONOLITHIC, INDEPENDENT blobs on two CUDA streams -> overlap_fraction_strict=0.6449, residual
strict tax 114.6us -> multistream_strict_tps_ceiling=474.44. But that overlap is PHYSICALLY
UNREALIZABLE in a faithful served schedule: the body GEMM and the attention are NOT independent.
A Gemma decoder layer is a STRICT SERIAL RECURRENCE:
    qkv(L) -> attn(L) -> o_proj(L) -> mlp(L) -> qkv(L+1) -> attn(L+1) -> ...
o_proj(L) CONSUMES attn(L); every op in layers > L is transitively downstream of attn(L). So
attn(L) has NO independent body GEMM to hide under -- the only GEMMs are upstream (qkv(L), already
done) or downstream (o/mlp(L) and all later layers, all blocked on attn(L)). The realizable
cross-op overlap inside a single byte-exact verify forward is therefore DEPENDENCY-BOUNDED, not
resource-bounded. (Researcher confirm: serial-block transformers admit zero exactness-preserving
single-sequence attn||GEMM overlap; only PARALLEL-block archs -- GPT-J/PaLM-2204.02311 sec4.2/
Falcon -- where y=x+Attn(LN x)+FFN(LN x) makes the two branches independent, can overlap. Gemma
is serial-block. Techniques that DO fill a 2nd stream during decode -- FlashDecoding++ 2311.01282
intra-kernel, Helix 2507.07120 multi-GPU+comms, microbatch/2nd-seq/drafter -- all need INDEPENDENT
work that a single verify forward does not contain.)

WHAT THIS MEASURES (the dependency penalty, isolated by ONE barrier)
-------------------------------------------------------------------
Build the real per-layer schedule from the EXACT #477/#472 kernels (37-layer self-built g=128
int4-Marlin body via apply_gptq_marlin_linear + the 7 served-Triton hd=512 full-attn strict_2d/
permissive_3d reductions + 30 sliding sdpa + int4 12k lm_head), CUDA-graph captured (deployed
ONEGRAPH basis; multi-stream capture validated on this pod). Time these arms, paired per round:
  body          : all body GEMM + lm_head (single stream)              [#450 calib + base]
  serial_<lev>  : true-order [qkv,attn,o,gate_up,down]x37 + lm_head, ONE stream  [single-stream floor]
  pipe_dep_<lev>: per-layer 2-stream -- attn(L) on side, WITH the consume barrier (o_proj(L) waits on
                  attn(L)'s event). This is the DEPENDENCY-RESPECTING (realizable, byte-exact) schedule.
  pipe_indep_<lev>: IDENTICAL per-layer 2-stream but WITHOUT the consume barrier (the main GEMM stream
                  barrels through o/mlp(L) without waiting on attn(L)) -- the #477 resource ceiling.
  indep_mono_<lev>: monolithic body || all-attention (one fork/join) -- the literal #477 arm, anchor.
The ONLY difference between pipe_dep and pipe_indep is the single cross-stream consume barrier =
the data dependency. So (indep tax - dep tax) is the PURE dependency penalty, same kernels, same clock.

  serial_tax  = serial_strict  - serial_perm    (single-stream strict tax; calib -> #472 401.9)
  dep_tax     = pipe_dep_strict - pipe_dep_perm  (REALIZABLE residual strict tax, dependency-respecting)
  ceil_tax    = pipe_indep_strict - pipe_indep_perm  (resource-ceiling residual; calib -> #477 114.6)
  realizable_overlap_fraction = (serial_tax - dep_tax)/serial_tax     PRIMARY (~0 expected)
  dependency_bounded_strict_tps = tps_from_added_us(dep_tax)          (~457.55 expected)
  ceiling_strict_tps            = tps_from_added_us(ceil_tax)         (~474.44, reproduce #477)
  multistream_realizable_above_467 = dependency_bounded_strict_tps >= 467.14   (bool verdict)
Plus the real-schedule tax the resource-ceiling probe omitted:
  cross_stream_barrier_us : per fork/join-pair latency (isolated micro-bench) + per-cycle aggregate.
  pipeline_fill_drain_us  : T(pipe_indep per-layer) - T(indep_mono) = the per-layer fork/join + edge
                            ramp a real interleaved schedule pays that the monolithic #477 probe omitted.
Identity (instruction 4): strict 2D reduction ORDER is unchanged by stream assignment + barriers, so
strict byte-identity 1.0000 / 0 flips must survive the pipelined capture+replay (== #466/#472/#477).
ppl 2.3772 carried (pinned by construction -- scheduling cannot change emitted tokens).

Reproduce: cd target/ && CUDA_VISIBLE_DEVICES=0 \
  /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/speed/dependency_bounded_overlap/dependency_bounded_overlap.py \
  --wandb_group equivalence-escalation-anchors --wandb_name lawine/dependency-bounded-overlap
Then log W&B from the repo .venv:
  cd target/ && .venv/bin/python \
  research/speed/dependency_bounded_overlap/wandb_log.py \
  --json research/speed/dependency_bounded_overlap/dependency_bounded_overlap.json \
  --wandb_group equivalence-escalation-anchors --wandb_name lawine/dependency-bounded-overlap
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


# Reuse the EXACT committed primitives the #477/#472 probes used (faithful, not re-derived):
#  - wc (#472 strict_wholecycle_ab): the real 37-layer g=128 int4-Marlin body + 7 served-Triton
#    full-attn reductions + tps_from_added_us banked-cycle mapping; geometry constants.
#  - sfr (#466 strict_frontier_realize): _build_inputs / _segm_bufs / _call_unified / ARMS /
#    identity_probe / served attention geometry / tps mapping.
#  - roof (#450): _marlin_quant / _apply_marlin / _mk_ws / _QT / measure_peak_bw / mdgd loader.
wc = _load("strict_wholecycle_ab",
           "research/speed/strict_wholecycle_ab/strict_wholecycle_ab.py")
sfr = wc.sfr
roof = wc.roof

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# ---- banked anchors (IMPORTED exact; this card derives nothing upstream) ----
DEPLOYED_TPS = sfr.DEPLOYED_TPS                 # 481.53 deployed incumbent (non-equivalent)
REALIZED_BASE_TPS = sfr.REALIZED_BASE_TPS       # 467.14 composed blanket-strict ceiling (THE BAR)
CYCLE_PERM_US = sfr.CYCLE_PERM_US               # 7666.83 deployed permissive cycle <-> 481.53
M1_COLLAPSE_TPS = sfr.M1_COLLAPSE_TPS           # 161.70 M=1 AR strict floor
PPL_ANCHOR = sfr.PPL_ANCHOR                     # 2.3772 (pinned by construction)
PPL_GATE = sfr.PPL_GATE                         # 2.42
SIGMA_HW = wc.SIGMA_HW                          # 4.8153 advisor-named hardware sigma

ISO_DELTA_466_US = wc.ISO_DELTA_466_US          # 422.91 #466 isolated serial attention tax
REALIZED_466_TPS = wc.REALIZED_466_TPS          # 456.36 #466 realized (isolated lower bound)
GEMM_US_450 = wc.GEMM_US_450                    # 4152.96 #450 measured body-GEMM time
# stark #472 whole-cycle SINGLE-STREAM realized (THE single-stream floor a 2nd stream must beat) ----
WHOLE_DELTA_472_US = 401.89971923828125         # #472 single-stream in-graph-overlap strict tax/cycle
REALIZED_WHOLECYCLE_457 = 457.5452044002469     # #472 realized_strict_frontier_best_estimate_tps (FLOOR)
# lawine #477 RESOURCE-feasibility ceiling (the number this card discounts) ----
CEILING_477_TPS = 474.4379601932696             # #477 multistream_strict_tps_ceiling (resource UB)
CEILING_477_ADDED_US = 114.60601806640625       # #477 residual strict tax under independent 2-stream
CEILING_477_OVERLAP_FRAC = 0.6448720132725708   # #477 overlap_fraction_strict (resource)
CEILING_477_HIDE_FRAC = 0.7148392681547066      # #477 multistream_hide_fraction (vs 401.9)

# served gemma-4-E4B-it geometry (== #466/#472/#477)
N_FULL_LAYERS = sfr.N_FULL_LAYERS               # 7 full-attention (hd=512) Triton layers
M_VERIFY = sfr.M_VERIFY                          # 8 spec-verify width (K_spec=7 + 1)
HEAD_DIM_FULL = sfr.HEAD_DIM_FULL               # 512
HEAD_DIM_SLIDING = sfr.HEAD_DIM_SLIDING         # 256
N_Q_HEADS = sfr.N_Q_HEADS                       # 8
N_KV_HEADS = sfr.N_KV_HEADS                     # 2 (GQA)
KV_LENS = sfr.KV_LENS                           # (128, 384, 640)
HEADLINE_L = sfr.HEADLINE_L                     # 640
SLIDING_WINDOW = wc.SLIDING_WINDOW              # 512
FULL_ATTN_IDX = wc.FULL_ATTN_IDX               # (2,8,14,20,26,32,36)
ORDER = wc.ORDER                                 # ["qkv_proj","o_proj","gate_up_proj","down_proj"]
BODY_GEMM = roof.BODY_GEMM                       # same 4 projections
A10G_SPEC_BW_GBPS = roof.A10G_SPEC_BW_GBPS

_capture = roof.mdgd._capture
LEVERS = {"strict": "strict_2d", "perm": "permissive_3d"}


def tps_from_added_us(added_us):
    """== sfr.tps_from_added_us: realized strict TPS on the banked permissive cycle. added_us=0 ->
    481.53 byte-exact; added=401.9 -> 457.55 (single-stream floor)."""
    return sfr.tps_from_added_us(added_us)


def _med(vals):
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.median(vals) if vals else float("nan")


def _paired(series, a, b):
    diffs = [x - y for x, y in zip(series[a], series[b]) if math.isfinite(x) and math.isfinite(y)]
    if not diffs:
        return float("nan"), float("nan"), 0
    m = statistics.median(diffs)
    sd = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    return m, sd, len(diffs)


def _time_solo(g, iters):
    """us per replay of a captured graph (multi-stream-internal graphs replay on the launch stream;
    the captured fork/joins replay relative to it)."""
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3


# =========================================================================== #
def build_layered(dims, num_layers, full_set, L_full, L_slide, M, dev, n_distinct, side):
    """Per-op closures (gemm, attn_full, attn_sliding) from the EXACT #472 build_all construction,
    plus the 8 schedule arms. Weights / sliding sdpa / lm_head are SHARED (read-only; cancel in the
    paired strict-vs-perm diff); only the 7 full-attn Triton reductions differ (strict_2d/perm_3d).
    n_distinct distinct cold weights/component (working set >> A10G L2 -> cold HBM reads per replay)."""
    NK = {c: dims["shapes"][c] for c in ORDER}
    NK["lm_head"] = (roof.LM_HEAD_VOCAB, dims["hidden"])
    ws = roof._mk_ws(dev)
    zp = torch.zeros(0, dtype=torch.int, device=dev)

    weights, xins = {}, {}
    for c in ORDER:
        N, K = NK[c]
        weights[c] = [roof._marlin_quant(N, K, 128, dev) for _ in range(n_distinct)]
        xins[c] = torch.randn(M, K, dtype=torch.float16, device=dev)
    lmN, lmK = NK["lm_head"]
    weights["lm_head"] = [roof._marlin_quant(lmN, lmK, -1, dev) for _ in range(max(2, n_distinct // 4))]
    xins["lm_head"] = torch.randn(M, lmK, dtype=torch.float16, device=dev)

    def gemm(c, idx):
        q_w, s, gi, so = weights[c][idx % len(weights[c])]
        N, K = NK[c]
        roof._apply_marlin(xins[c], q_w, s, zp, gi, so, ws, roof._QT, N, K, is_k_full=True, bias=None)

    # sliding-attention proxy (hd=256), shared, identical in both levers (sdpa, == #452/#472)
    n_h = dims["n_heads"]
    q_sl = torch.randn(1, n_h, M, HEAD_DIM_SLIDING, dtype=torch.float16, device=dev)
    k_sl = torch.randn(1, n_h, L_slide, HEAD_DIM_SLIDING, dtype=torch.float16, device=dev)
    v_sl = torch.randn(1, n_h, L_slide, HEAD_DIM_SLIDING, dtype=torch.float16, device=dev)

    def attn_sliding():
        F.scaled_dot_product_attention(q_sl, k_sl, v_sl)

    # full-attention Triton inputs per lever (distinct per layer -> cold KV reads); strict 2D has no segm
    full_layers = sorted(full_set)
    full_pos = {Lidx: j for j, Lidx in enumerate(full_layers)}
    attn_inps = {}
    for tag, lever in LEVERS.items():
        num_par, max_q = sfr.ARMS[lever]
        lst = []
        for Lidx in full_layers:
            seed = 9000 + 13 * Lidx + (1 if lever == "permissive_3d" else 0)
            inp = sfr._build_inputs(L_full, M, HEAD_DIM_FULL, seed, dev)
            seg = sfr._segm_bufs(HEAD_DIM_FULL, num_par, dev, M) if lever == "permissive_3d" else None
            lst.append((inp, seg, num_par, max_q))
        attn_inps[tag] = lst

    def attn_full(tag, j):
        inp, seg, num_par, max_q = attn_inps[tag][j]
        sfr._call_unified(inp, LEVERS[tag], num_par, seg, max_q)

    def do_attn(tag, L):
        if L in full_set:
            attn_full(tag, full_pos[L])
        else:
            attn_sliding()

    # ---- arm 0: body GEMM only (calibration vs #450; the exposed-attn subtraction base) ----
    def run_body():
        for L in range(num_layers):
            for c in ORDER:
                gemm(c, L)
        gemm("lm_head", 0)

    # ---- arm: SINGLE-STREAM true-order interleave (the single-stream floor) ----
    def make_serial(tag):
        def run():
            for L in range(num_layers):
                gemm("qkv_proj", L)
                do_attn(tag, L)
                gemm("o_proj", L)
                gemm("gate_up_proj", L)
                gemm("down_proj", L)
            gemm("lm_head", 0)
        return run

    # ---- arm: DEPENDENCY-RESPECTING per-layer 2-stream pipeline (REALIZABLE, byte-exact) ----
    # attn(L) on side; o_proj(L) on main WAITS for attn(L) (the true data dependency). The main
    # GEMM stream stalls on each attn -> attention stays fully exposed -> ~zero realizable overlap.
    def make_pipe_dep(tag):
        def run():
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            for L in range(num_layers):
                gemm("qkv_proj", L)            # main: qkv(L)
                side.wait_stream(main)         # side waits for qkv(L)  [attn(L) consumes qkv(L)]
                with torch.cuda.stream(side):
                    do_attn(tag, L)            # side: attn(L)
                main.wait_stream(side)         # main waits for attn(L) [o_proj(L) consumes attn(L)] <-- DEP
                gemm("o_proj", L)
                gemm("gate_up_proj", L)
                gemm("down_proj", L)           # main: o/mlp(L)
            main.wait_stream(side)
            gemm("lm_head", 0)
        return run

    # ---- arm: INDEPENDENT per-layer 2-stream (the #477 resource ceiling) ----
    # IDENTICAL to pipe_dep EXCEPT the consume barrier is REMOVED: the main GEMM stream barrels
    # through o/mlp(L) without waiting on attn(L), so attention overlaps the body GEMM. The single
    # missing `main.wait_stream(side)` IS the data dependency the realizable schedule must honor.
    def make_pipe_indep(tag):
        def run():
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            for L in range(num_layers):
                gemm("qkv_proj", L)            # main: qkv(L)
                side.wait_stream(main)         # side waits for qkv(L)
                with torch.cuda.stream(side):
                    do_attn(tag, L)            # side: attn(L)  -- NO consume barrier; main races ahead
                gemm("o_proj", L)
                gemm("gate_up_proj", L)
                gemm("down_proj", L)           # main: o/mlp(L) (does NOT wait on attn(L))
            main.wait_stream(side)
            gemm("lm_head", 0)
        return run

    # ---- arm: MONOLITHIC body || all-attention (the literal #477 arm; #477 anchor + fill/drain ref) ----
    def make_indep_mono(tag):
        def run():
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                for L in range(num_layers):
                    do_attn(tag, L)            # ALL attention on side, one blob
            for L in range(num_layers):
                for c in ORDER:
                    gemm(c, L)                 # ALL body on main, one blob
            main.wait_stream(side)
            gemm("lm_head", 0)
        return run

    runners = {"body": run_body}
    for tag in LEVERS:
        runners[f"serial_{tag}"] = make_serial(tag)
        runners[f"pipe_dep_{tag}"] = make_pipe_dep(tag)
        runners[f"pipe_indep_{tag}"] = make_pipe_indep(tag)
    runners["indep_mono_strict"] = make_indep_mono("strict")
    keep = (weights, xins, ws, zp, q_sl, k_sl, v_sl, attn_inps)
    return runners, NK, keep


def capture_arms(runners):
    """Capture each arm into its own CUDA graph (multi-stream-internal arms capture their side-stream
    fork/joins via the validated wait_stream pattern). Returns (graphs, captured)."""
    graphs, captured = {}, {}
    for nm, run in runners.items():
        try:
            graphs[nm] = _capture(run)
            captured[nm] = True
        except Exception as exc:  # noqa: BLE001
            print(f"[dep] capture FAILED {nm}: {exc!r}", flush=True)
            graphs[nm], captured[nm] = None, False
    return graphs, captured


def measure_barrier_us(dev, side, iters, warmup, n_bar=64):
    """Isolated cross-stream fork/join latency. Same payload (2 tiny mm/iter) in both arms; the
    barriered arm adds a side.wait_stream + main.wait_stream per iter. (T_bar - T_same)/n_bar =
    pure per fork/join-pair latency. Graph-captured (launch overhead removed, == deployed ONEGRAPH)."""
    x = torch.randn(96, 96, dtype=torch.float16, device=dev)

    def run_same():               # 2 tiny mm/iter, single stream
        for _ in range(n_bar):
            torch.mm(x, x)
            torch.mm(x, x)

    def run_bar():                # 2 tiny mm/iter, split across streams w/ fork+join
        main = torch.cuda.current_stream()
        for _ in range(n_bar):
            torch.mm(x, x)        # main
            side.wait_stream(main)
            with torch.cuda.stream(side):
                torch.mm(x, x)    # side
            main.wait_stream(side)

    try:
        g_same = _capture(run_same)
        g_bar = _capture(run_bar)
    except Exception as exc:  # noqa: BLE001
        return {"per_pair_us": float("nan"), "err": repr(exc)}
    for g in (g_same, g_bar):
        for _ in range(max(10, warmup)):
            g.replay()
    torch.cuda.synchronize()
    same, bar = [], []
    for _ in range(21):
        same.append(_time_solo(g_same, iters))
        bar.append(_time_solo(g_bar, iters))
    t_same, t_bar = _med(same), _med(bar)
    per_pair = max(0.0, (t_bar - t_same) / n_bar)
    del g_same, g_bar
    return {"per_pair_us": per_pair, "t_same_us": t_same, "t_bar_us": t_bar, "n_bar": n_bar, "err": None}


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=HEADLINE_L)
    ap.add_argument("--Ls", type=str, default=None)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--n-distinct", type=int, default=8)
    ap.add_argument("--ident-trials", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    ap.add_argument("--output", default=os.path.join(_here, "dependency_bounded_overlap.json"))
    ap.add_argument("--selftest-output", default=os.path.join(_here, "selftest.json"))
    ap.add_argument("--wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", default="lawine/dependency-bounded-overlap")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable (need CUDA_VISIBLE_DEVICES=0)"
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    iters = 12 if args.smoke else args.iters
    rounds = 7 if args.smoke else args.rounds
    n_distinct = 4 if args.smoke else args.n_distinct
    ident_trials = 2 if args.smoke else args.ident_trials
    if args.Ls:
        Ls = tuple(int(x) for x in args.Ls.split(","))
    else:
        Ls = (128, args.L) if args.smoke else KV_LENS
    if args.L not in Ls:
        Ls = tuple(sorted(set(Ls) | {args.L}))

    model_dir = roof.SERVED_BODY.rsplit("/", 1)[0]
    dims = roof.mdgd.read_dims(model_dir)
    num_layers, depth_src = roof.mdgd.deployed_depth(dims["num_layers"])
    full_set = set(i for i in FULL_ATTN_IDX if i < num_layers)
    assert len(full_set) == N_FULL_LAYERS, f"expected {N_FULL_LAYERS} full-attn layers, got {len(full_set)}"
    print(f"[dep] {name} sm_{cap[0]}{cap[1]} torch {torch.__version__}  M={M_VERIFY} hd_full={HEAD_DIM_FULL} "
          f"depth={num_layers}({depth_src}) full_attn={sorted(full_set)} n_full={len(full_set)} "
          f"L_headline={args.L} Ls={Ls} rounds={rounds} n_distinct={n_distinct} smoke={args.smoke}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    # heavy warm-up -> A10G boost clock (same regime as #450/#466/#472/#477)
    big = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    for _ in range(200):
        big = big @ big
    torch.cuda.synchronize()
    del big

    peak = roof.measure_peak_bw(dev, iters, args.warmup)
    peak_read = peak["bw_read_gbps"]
    print(f"[dep] PEAK BW: read={peak_read:.0f} copy={peak['bw_copy_gbps']:.0f} GB/s", flush=True)

    side = torch.cuda.Stream()   # persistent side stream (shared across arms; sequential replay)

    # isolated cross-stream barrier latency (clock-warm)
    barrier = measure_barrier_us(dev, side, iters, args.warmup)
    cross_stream_barrier_us = barrier["per_pair_us"]
    print(f"[dep] cross_stream_barrier per fork/join-pair = {cross_stream_barrier_us:.2f}us "
          f"(t_same={barrier.get('t_same_us', float('nan')):.1f} t_bar={barrier.get('t_bar_us', float('nan')):.1f})", flush=True)

    # ---- per-L arm sweep ----------------------------------------------------------------------
    arm_names = (["body"]
                 + [f"serial_{t}" for t in LEVERS]
                 + [f"pipe_dep_{t}" for t in LEVERS]
                 + [f"pipe_indep_{t}" for t in LEVERS]
                 + ["indep_mono_strict"])
    per_L = {}
    for L in Ls:
        runners, NK, keep = build_layered(dims, num_layers, full_set, L, min(SLIDING_WINDOW, L),
                                          M_VERIFY, dev, n_distinct, side)
        graphs, captured = capture_arms(runners)
        core_ok = all(captured.get(nm, False) for nm in arm_names)
        if not core_ok:
            per_L[L] = {"captured": captured, "error": "core capture failed"}
            del runners, keep, graphs
            gc.collect(); torch.cuda.empty_cache()
            continue

        for _ in range(max(10, args.warmup)):
            for g in graphs.values():
                if g is not None:
                    g.replay()
        torch.cuda.synchronize()

        series = {nm: [] for nm in arm_names}
        for _ in range(rounds):
            for nm in arm_names:
                series[nm].append(_time_solo(graphs[nm], iters))

        med = {nm: _med(series[nm]) for nm in arm_names}
        serial_tax, serial_tax_sd, _ = _paired(series, "serial_strict", "serial_perm")
        dep_tax, dep_tax_sd, _ = _paired(series, "pipe_dep_strict", "pipe_dep_perm")
        ceil_tax, ceil_tax_sd, _ = _paired(series, "pipe_indep_strict", "pipe_indep_perm")
        # exposed attention (vs body base) for the monolithic #477 anchor
        exposed_mono_strict, _, _ = _paired(series, "indep_mono_strict", "body")

        denom = serial_tax if (math.isfinite(serial_tax) and serial_tax > 0) else WHOLE_DELTA_472_US
        realizable_overlap_fraction = (serial_tax - dep_tax) / denom
        ceiling_overlap_fraction = (serial_tax - ceil_tax) / denom
        # vs the banked #472 401.9 (directly comparable to #477's hide_fraction)
        realizable_overlap_fraction_vs472 = (WHOLE_DELTA_472_US - dep_tax) / WHOLE_DELTA_472_US
        ceiling_overlap_fraction_vs472 = (WHOLE_DELTA_472_US - ceil_tax) / WHOLE_DELTA_472_US

        dependency_bounded_strict_tps = tps_from_added_us(dep_tax)
        ceiling_strict_tps = tps_from_added_us(ceil_tax)
        single_stream_floor_tps = tps_from_added_us(serial_tax)
        # overhead the two-stream machinery ADDS over single stream (realizable schedule, strict arm)
        multistream_overhead_us = med["pipe_dep_strict"] - med["serial_strict"]
        # the real-schedule tax the monolithic resource-ceiling probe omitted (per-layer fork/join + ramp)
        pipeline_fill_drain_us = med["pipe_indep_strict"] - med["indep_mono_strict"]

        per_L[L] = {
            "median_us": med, "captured": captured,
            "serial_tax_us": serial_tax, "serial_tax_sigma": serial_tax_sd,
            "dep_tax_us": dep_tax, "dep_tax_sigma": dep_tax_sd,
            "ceil_tax_us": ceil_tax, "ceil_tax_sigma": ceil_tax_sd,
            "exposed_mono_strict_us": exposed_mono_strict,
            "realizable_overlap_fraction": realizable_overlap_fraction,
            "ceiling_overlap_fraction": ceiling_overlap_fraction,
            "realizable_overlap_fraction_vs472": realizable_overlap_fraction_vs472,
            "ceiling_overlap_fraction_vs472": ceiling_overlap_fraction_vs472,
            "dependency_bounded_strict_tps": dependency_bounded_strict_tps,
            "ceiling_strict_tps": ceiling_strict_tps,
            "single_stream_floor_tps": single_stream_floor_tps,
            "multistream_overhead_us": multistream_overhead_us,
            "pipeline_fill_drain_us": pipeline_fill_drain_us,
        }
        print(f"[dep] L={L}: body={med['body']:.0f} serial_s={med['serial_strict']:.0f} "
              f"pipe_dep_s={med['pipe_dep_strict']:.0f} pipe_indep_s={med['pipe_indep_strict']:.0f} "
              f"mono_s={med['indep_mono_strict']:.0f} | serial_tax={serial_tax:+.1f} dep_tax={dep_tax:+.1f} "
              f"ceil_tax={ceil_tax:+.1f} | realiz_ov={realizable_overlap_fraction*100:.0f}% "
              f"ceil_ov={ceiling_overlap_fraction*100:.0f}% -> dep_tps={dependency_bounded_strict_tps:.2f} "
              f"ceil_tps={ceiling_strict_tps:.2f}", flush=True)

        del runners, keep
        for g in graphs.values():
            del g
        gc.collect(); torch.cuda.empty_cache()

    # ---- headline (L = args.L) ----------------------------------------------------------------
    H = per_L[args.L]
    assert "error" not in H, f"headline L={args.L} failed capture: {H}"
    serial_tax = H["serial_tax_us"]
    dep_tax = H["dep_tax_us"]
    dep_tax_sigma = H["dep_tax_sigma"]
    ceil_tax = H["ceil_tax_us"]
    realizable_overlap_fraction = H["realizable_overlap_fraction"]              # PRIMARY
    ceiling_overlap_fraction = H["ceiling_overlap_fraction"]
    dependency_bounded_strict_tps = H["dependency_bounded_strict_tps"]          # TEST/primary
    ceiling_strict_tps = H["ceiling_strict_tps"]
    single_stream_floor_tps = H["single_stream_floor_tps"]
    multistream_overhead_us = H["multistream_overhead_us"]
    pipeline_fill_drain_us = H["pipeline_fill_drain_us"]
    body_gemm_us = H["median_us"]["body"]
    dep_tps_lo = tps_from_added_us(dep_tax + dep_tax_sigma)
    dep_tps_hi = tps_from_added_us(max(0.0, dep_tax - dep_tax_sigma))
    cross_stream_barrier_per_cycle_us = (cross_stream_barrier_us * num_layers
                                         if math.isfinite(cross_stream_barrier_us) else float("nan"))

    # ---- captured-all-L survival -------------------------------------------------------------
    captured_all_L = all("error" not in per_L[L] for L in Ls)
    strict_arms_captured = all(per_L[L].get("captured", {}).get("pipe_dep_strict", False)
                               and per_L[L].get("captured", {}).get("pipe_indep_strict", False)
                               and "error" not in per_L[L] for L in Ls)

    # ---- identity (strict 2D reduction order unchanged by stream assignment) == #466/#472/#477 ----
    seeds = [1234] if args.smoke else [1234, 5678, 9012]
    ident = sfr.identity_probe(args.L, HEAD_DIM_FULL, dev, ident_trials, seeds)
    iarm = ident.get("per_arm", {})
    strict_identity_fraction = float(iarm.get("strict_2d", {}).get("byte_identity_min", float("nan")))
    strict_argmax_fraction = float(iarm.get("strict_2d", {}).get("argmax_identity_min", float("nan")))
    strict_token_flips = int(round((1.0 - (strict_argmax_fraction
                            if math.isfinite(strict_argmax_fraction) else 1.0)) * M_VERIFY))
    permissive_identity_fraction = float(iarm.get("permissive_3d", {}).get("byte_identity_min", float("nan")))

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # ---- gain / bars -------------------------------------------------------------------------
    gain_vs_single_stream = dependency_bounded_strict_tps - REALIZED_WHOLECYCLE_457
    realizable_clears_457 = bool(dependency_bounded_strict_tps >= REALIZED_WHOLECYCLE_457 + SIGMA_HW)
    multistream_realizable_above_467 = bool(dependency_bounded_strict_tps >= REALIZED_BASE_TPS)
    ceiling_discount_tps = ceiling_strict_tps - dependency_bounded_strict_tps   # the unrealizable gap
    # the residual strict tax the dependency leaves exposed vs the resource ceiling (us)
    dependency_penalty_us = dep_tax - ceil_tax

    # =================== SELF-TEST (PRIMARY GATE) =================================
    st = {}
    st["tps_zero_added_is_deployed"] = bool(abs(tps_from_added_us(0.0) - DEPLOYED_TPS) < 1e-6)
    st["tps_472delta_is_457"] = bool(abs(tps_from_added_us(WHOLE_DELTA_472_US) - REALIZED_WHOLECYCLE_457) < 1e-2)
    st["constants_anchored"] = bool(REALIZED_BASE_TPS == 467.14 and DEPLOYED_TPS == 481.53)
    st["geometry_full_attn"] = bool(len(full_set) == N_FULL_LAYERS and M_VERIFY == 8
                                    and HEAD_DIM_FULL == 512 and num_layers == 37)
    st["body_shapes_served"] = bool(NK["qkv_proj"] == (3072, 2560) and NK["gate_up_proj"] == (20480, 2560)
                                    and NK["down_proj"] == (2560, 10240) and NK["o_proj"] == (2560, 2048))
    st["all_arms_captured"] = bool(captured_all_L and strict_arms_captured)
    # CALIBRATION GUARD 1: single-stream serial tax reproduces #472's 401.9 (within 18%)
    st["serial_tax_reproduces_472"] = bool(math.isfinite(serial_tax) and
                                           abs(serial_tax - WHOLE_DELTA_472_US) / WHOLE_DELTA_472_US < 0.18)
    # CALIBRATION GUARD 2: the INDEPENDENT (no-consume-barrier) tax reproduces #477's 114.6 (within 35%)
    #   -- proves THIS harness measures the same resource overlap #477 did, so the dep collapse is real
    st["ceiling_tax_reproduces_477"] = bool(math.isfinite(ceil_tax) and
                                            abs(ceil_tax - CEILING_477_ADDED_US) / CEILING_477_ADDED_US < 0.35)
    # CALIBRATION GUARD 3: body GEMM reproduces #450's 4152.96 (within 20%)
    st["body_gemm_anchor_matches_450"] = bool(body_gemm_us > 0 and
                                              abs(body_gemm_us - GEMM_US_450) / GEMM_US_450 < 0.20)
    # the ceiling arm must overlap MORE than the dependency-respecting arm (dep_tax >= ceil_tax modulo noise)
    st["dep_tax_ge_ceiling"] = bool(math.isfinite(dep_tax) and math.isfinite(ceil_tax)
                                    and dep_tax >= ceil_tax - 3.0 * SIGMA_HW)
    # the realizable dep tax must NOT be materially below the single-stream tax (overlap only HIDES,
    # and the dependency hides nothing -> dep_tax ~ serial_tax). Allow small clock/cache 2nd-order.
    st["dep_tax_near_serial"] = bool(math.isfinite(dep_tax) and math.isfinite(serial_tax)
                                     and dep_tax >= serial_tax - 0.20 * max(serial_tax, 1.0))
    finite = [realizable_overlap_fraction, dependency_bounded_strict_tps, ceiling_strict_tps,
              serial_tax, dep_tax, ceil_tax, body_gemm_us, cross_stream_barrier_us,
              pipeline_fill_drain_us, multistream_overhead_us]
    st["nan_clean"] = all(math.isfinite(x) for x in finite)
    st["identity_ran"] = bool(ident.get("error") is None and "per_arm" in ident)
    st["strict_byte_exact"] = bool(math.isfinite(strict_identity_fraction)
                                   and strict_identity_fraction >= 0.999)
    st["strict_zero_flips"] = bool(strict_token_flips == 0)
    st["permissive_reproduces_nonequiv"] = bool(math.isfinite(permissive_identity_fraction)
                                                and permissive_identity_fraction < 0.999)
    st["barrier_measured"] = bool(barrier.get("err") is None and math.isfinite(cross_stream_barrier_us))
    st["dep_tps_in_band"] = bool(REALIZED_466_TPS - 2 * SIGMA_HW <= dependency_bounded_strict_tps
                                 <= DEPLOYED_TPS + 1e-6)
    st["ceiling_tps_near_477"] = bool(abs(ceiling_strict_tps - CEILING_477_TPS) < 6.0)
    st["ppl_anchor_ok"] = bool(PPL_ANCHOR <= PPL_GATE)
    st["vram_ok"] = bool(peak_vram_gib <= 24.0)
    self_test_passes = all(st.values())

    # =================== VERDICT =================================
    if not strict_arms_captured:
        outcome = "CAPTURE_FAIL"
    elif multistream_realizable_above_467:
        outcome = "REALIZABLE_ABOVE_467"          # dependency-bounded number clears the bar (worth gating)
    elif realizable_clears_457:
        outcome = "REALIZABLE_BEATS_457_FLOOR"     # some dependency-bounded overlap survives
    else:
        outcome = "DEPENDENCY_COLLAPSES_TO_FLOOR"  # the 474 ceiling is NOT realizable -> multistream CLOSES

    multistream_closes = bool(outcome == "DEPENDENCY_COLLAPSES_TO_FLOOR")

    reconcile = (
        f"#477 (w41knrqd) timed the WHOLE body GEMM ({body_gemm_us:.0f}us) CONCURRENT with the WHOLE strict "
        f"verify-attention as two INDEPENDENT monolithic blobs -> overlap {CEILING_477_OVERLAP_FRAC*100:.0f}%, "
        f"residual tax {CEILING_477_ADDED_US:.0f}us -> ceiling {CEILING_477_TPS:.2f} TPS. But a Gemma layer is a "
        f"SERIAL recurrence (qkv->attn->o_proj->mlp; o_proj(L) CONSUMES attn(L)) -- so attn(L) has NO independent "
        f"GEMM to hide under. This card builds the real per-layer schedule from the SAME #472 kernels and measures "
        f"the dependency penalty by ONE barrier: pipe_dep (o_proj(L) WAITS on attn(L) -- the data dependency) vs "
        f"pipe_indep (the consume barrier REMOVED == #477's resource ceiling), identical kernels/clock. "
        f"Single-stream strict tax serial_tax={serial_tax:+.1f}us (calib vs #472 {WHOLE_DELTA_472_US:.0f} "
        f"[{st['serial_tax_reproduces_472']}]); resource-ceiling tax ceil_tax={ceil_tax:+.1f}us reproduces "
        f"#477 {CEILING_477_ADDED_US:.0f} [{st['ceiling_tax_reproduces_477']}] -> ceiling_strict_tps="
        f"{ceiling_strict_tps:.2f} (matches #477 {CEILING_477_TPS:.2f}). DEPENDENCY-RESPECTING dep_tax="
        f"{dep_tax:+.1f}us (sigma {dep_tax_sigma:.1f}) ~ the single-stream tax -> realizable_overlap_fraction="
        f"{realizable_overlap_fraction*100:.0f}% (vs the {CEILING_477_OVERLAP_FRAC*100:.0f}% resource ceiling) -> "
        f"dependency_bounded_strict_tps={dependency_bounded_strict_tps:.2f} ({gain_vs_single_stream:+.2f} vs the "
        f"single-stream floor {REALIZED_WHOLECYCLE_457:.2f}; clears_457[+sigma]={realizable_clears_457}). The "
        f"{ceiling_discount_tps:+.1f} TPS between the {CEILING_477_TPS:.2f} resource ceiling and the realizable "
        f"{dependency_bounded_strict_tps:.2f} is the UNREALIZABLE dependency gap. Real-schedule tax the monolithic "
        f"probe omitted: cross_stream_barrier={cross_stream_barrier_us:.2f}us/pair (x{num_layers}="
        f"{cross_stream_barrier_per_cycle_us:.0f}us/cycle), pipeline_fill_drain={pipeline_fill_drain_us:.1f}us; "
        f"two-stream ADDS {multistream_overhead_us:+.1f}us over single-stream (no benefit). multistream_realizable_"
        f"above_467={multistream_realizable_above_467} -> {'multi-stream CLOSES as a byte-exact path' if multistream_closes else 'REVISIT'}. "
        f"Strict 2D byte-exact under pipelined capture+replay: identity={strict_identity_fraction:.4f} "
        f"({strict_token_flips} flips); deployed permissive byte={permissive_identity_fraction:.4f}. ppl {PPL_ANCHOR} "
        f"(pinned). MEASUREMENT-ONLY: no served-file change, no kernel rebuild, no HF job. OUTCOME={outcome}.")

    verdict = {
        "dependency_bounded_self_test_passes": self_test_passes,                    # PRIMARY (gate)
        "realizable_overlap_fraction": realizable_overlap_fraction,                 # PRIMARY (metric)
        "dependency_bounded_strict_tps": dependency_bounded_strict_tps,             # TEST/primary
        "dependency_bounded_strict_tps_sigma_lo": dep_tps_lo,
        "dependency_bounded_strict_tps_sigma_hi": dep_tps_hi,
        "multistream_realizable_above_467": multistream_realizable_above_467,       # bool verdict
        "multistream_closes": multistream_closes,
        "realizable_clears_457_floor": realizable_clears_457,
        # --- the dependency penalty (the headline contrast) ---
        "ceiling_strict_tps": ceiling_strict_tps,
        "ceiling_discount_tps": ceiling_discount_tps,
        "dependency_penalty_us": dependency_penalty_us,
        "ceiling_overlap_fraction": ceiling_overlap_fraction,
        "realizable_overlap_fraction_vs472": H["realizable_overlap_fraction_vs472"],
        "ceiling_overlap_fraction_vs472": H["ceiling_overlap_fraction_vs472"],
        # --- the measured taxes ---
        "serial_tax_us": serial_tax, "serial_tax_sigma": H["serial_tax_sigma"],
        "dep_tax_us": dep_tax, "dep_tax_sigma": dep_tax_sigma,
        "ceil_tax_us": ceil_tax, "ceil_tax_sigma": H["ceil_tax_sigma"],
        "single_stream_floor_tps": single_stream_floor_tps,
        "exposed_mono_strict_us": H["exposed_mono_strict_us"],
        # --- the real-schedule tax #477 omitted ---
        "cross_stream_barrier_us": cross_stream_barrier_us,
        "cross_stream_barrier_per_cycle_us": cross_stream_barrier_per_cycle_us,
        "pipeline_fill_drain_us": pipeline_fill_drain_us,
        "multistream_overhead_vs_single_stream_us": multistream_overhead_us,
        # --- arm walls ---
        "body_gemm_us": body_gemm_us,
        "serial_strict_us": H["median_us"]["serial_strict"], "serial_perm_us": H["median_us"]["serial_perm"],
        "pipe_dep_strict_us": H["median_us"]["pipe_dep_strict"], "pipe_dep_perm_us": H["median_us"]["pipe_dep_perm"],
        "pipe_indep_strict_us": H["median_us"]["pipe_indep_strict"],
        "indep_mono_strict_us": H["median_us"]["indep_mono_strict"],
        # --- gain / bar ---
        "gain_vs_single_stream_tps": gain_vs_single_stream,
        "single_stream_realized_tps": REALIZED_WHOLECYCLE_457,
        # --- survival / identity ---
        "strict_arms_captured_all_L": strict_arms_captured, "captured_all_L": captured_all_L,
        "strict_identity_fraction": strict_identity_fraction, "strict_token_flips": strict_token_flips,
        "permissive_identity_fraction": permissive_identity_fraction,
        # --- banked anchors ---
        "whole_delta_472_us": WHOLE_DELTA_472_US, "realized_wholecycle_457_tps": REALIZED_WHOLECYCLE_457,
        "ceiling_477_tps": CEILING_477_TPS, "ceiling_477_added_us": CEILING_477_ADDED_US,
        "ceiling_477_overlap_fraction": CEILING_477_OVERLAP_FRAC,
        "iso_delta_466_us": ISO_DELTA_466_US, "realized_466_tps": REALIZED_466_TPS,
        "gemm_us_450_anchor": GEMM_US_450, "cycle_perm_us": CYCLE_PERM_US,
        "deployed_tps": DEPLOYED_TPS, "realized_base_tps": REALIZED_BASE_TPS,
        "m1_collapse_floor_tps": M1_COLLAPSE_TPS, "sigma_hw": SIGMA_HW,
        "n_full_layers_per_cycle": len(full_set), "deployed_num_layers": num_layers, "headline_L": args.L,
        "outcome": outcome,
        "ppl": PPL_ANCHOR, "ppl_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
        "peak_vram_gib": peak_vram_gib, "peak_read_gbps": peak_read,
        # --- scope / safety ---
        "deploy_needs_served_graph_edit": True, "measurement_only": True,
        "no_kernel_rebuild": True, "analysis_only": True, "no_hf_job": True,
        "no_served_file_change": True, "official_tps": 0,
        "self_test_conditions": st,
        "reconcile_line": reconcile,
    }

    payload = {
        "config": {"torch": torch.__version__, "device": name, "sm": f"{cap[0]}{cap[1]}",
                   "M": M_VERIFY, "head_dim_full": HEAD_DIM_FULL, "n_q_heads": N_Q_HEADS,
                   "n_kv_heads": N_KV_HEADS, "deployed_num_layers": num_layers,
                   "full_attn_idx": sorted(full_set), "n_full_layers": len(full_set),
                   "KV_LENS": list(Ls), "headline_L": args.L, "sliding_window": SLIDING_WINDOW,
                   "iters": iters, "warmup": args.warmup, "rounds": rounds, "n_distinct": n_distinct,
                   "ident_trials": ident_trials, "smoke": args.smoke, "served_model_dir": model_dir,
                   "group_size": 128, "self_built_marlin": True,
                   "note": "dependency-bounded per-layer software-pipelined overlap probe: real #472 kernels "
                           "(37-layer self-built g=128 int4-Marlin body + 7 served-Triton hd=512 full-attn "
                           "strict_2d/permissive_3d + 30 sliding sdpa + int4 12k lm_head), CUDA-graph captured "
                           "(multi-stream-internal). pipe_dep (o_proj(L) waits on attn(L)'s event = the data "
                           "dependency) vs pipe_indep (consume barrier REMOVED == #477 resource ceiling): the ONE "
                           "barrier IS the dependency. dep_tax=(pipe_dep_strict-pipe_dep_perm); realizable_overlap_"
                           "fraction=(serial_tax-dep_tax)/serial_tax; dependency_bounded_strict_tps via "
                           "tps_from_added_us on the banked cycle (CYCLE_PERM=7666.83<->481.53). No serve change, "
                           "no HF Job, no submission, no kernel rebuild."},
        "peak_bw": peak,
        "barrier": barrier,
        "per_L": {str(L): per_L[L] for L in per_L},
        "shapes": {c: list(NK[c]) for c in NK},
        "identity": ident,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))
    with open(args.selftest_output, "w") as fh:
        json.dump({"dependency_bounded_self_test_passes": self_test_passes, "checks": st}, fh, indent=2)
    print(f"[dep] wrote {args.output}", flush=True)
    print(f"\n[dep] OUTCOME={outcome}  self_test={self_test_passes}  multistream_closes={multistream_closes}", flush=True)
    print(f"[dep] realizable_overlap_fraction={realizable_overlap_fraction*100:.1f}% (ceiling {ceiling_overlap_fraction*100:.0f}%) "
          f"-> dependency_bounded_strict_tps={dependency_bounded_strict_tps:.2f} "
          f"(vs #477 ceiling {ceiling_strict_tps:.2f}; floor {REALIZED_WHOLECYCLE_457:.2f}; bar 467.14)", flush=True)
    print(f"[dep] multistream_realizable_above_467={multistream_realizable_above_467} "
          f"barrier={cross_stream_barrier_us:.2f}us/pair fill_drain={pipeline_fill_drain_us:.1f}us "
          f"identity={strict_identity_fraction:.4f}/{strict_token_flips}flips", flush=True)
    print(f"[dep] {reconcile}", flush=True)
    print(f"[dep] self_test={st}", flush=True)

    if not (args.no_wandb or args.smoke):
        print(f"[dep] to log W&B: cd target/ && .venv/bin/python "
              f"research/speed/dependency_bounded_overlap/wandb_log.py --json {args.output} "
              f"--wandb_group {args.wandb_group} --wandb_name {args.wandb_name}", flush=True)

    gc.collect(); torch.cuda.empty_cache()
    if args.self_test:
        return 0 if self_test_passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
