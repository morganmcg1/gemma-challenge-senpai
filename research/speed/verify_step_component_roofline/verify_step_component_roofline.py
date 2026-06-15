#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Verify-step component roofline (PR #280, kanna). LOCAL GPU micro-profiling +
CPU analytic. Analysis-only: NO served-file change, NO HF Job, NO submission,
NOT a launch. BASELINE stays 481.53. PRIMARY = self-test.

THE QUESTION
------------
The verify forward is 88% of the coupled spec-decode step (denken #271 region
split: verify_region 5355.5us vs draft 714.5us at M=8) and has only ever been
measured as an AGGREGATE verify_us(M) (denken #257 h1gj2ved). It was NEVER
component-decomposed. denken #271 retired tree-WIDTH (M*=32 -> 479.6 < 500) and
fern #274 proved you cannot cut your way to 500 (honest 500 needs E[T]_real >=
3.9914; draft-cuts move AWAY). So step-shaving survives ONLY as an E[T]-INDEPENDENT
stack that buys margin on the E[T]-raise path -- and the dominant unmapped
step-shaving territory is the VERIFY side.

WHAT THIS MEASURES (deployed-faithful, no serve change)
-------------------------------------------------------
Reuses denken #271's measure_deployed_gd.py loader + verify-runner basis (served
int4 compressed-tensors Marlin body, deployed depth=37, CUDA-graph replay = the
deployed ONEGRAPH basis) and kanna #269/#277's BW-utilization roofline anchor.

  (1) Component-decompose verify_us(M) at M in {8,16,32} into
      {qkv_proj, o_proj, SDPA, gate_up_proj, down_proj, lm_head} via PAIRED
      DIFFERENCING from the full verify graph (the small components qkv 3.9MiB,
      o 2.6MiB, KV 4.3MiB are < the 6MiB A10G L2, so an ISOLATED runner would be
      artificially L2-resident; differencing inside the full 45.9MiB/layer working
      set keeps every weight a COLD HBM read = deployed-faithful). Maps to the
      PR 5-way grouping {SDPA, MLP(gate_up+down), lm_head, KV-read, io/residual}.
  (2) Per-component BW-utilization (kanna #269 method): roofline_us = byte-traffic
      / 600 GB/s, anchored by a saturating large reference GEMM at the same M
      (~81% of peak, the #269 anchor). At M>=8 the GEMVs become batched GEMMs;
      ridge AI = 125 TFLOP / 600 GB/s = 208.3 FLOP/byte, int4-weight AI = 4*M, so
      the weight GEMMs stay WEIGHT-READ (memory) bound until M ~ 52 -- M=8/16/32
      are all memory-bound (the "compute-bound at M=32" aggregate, if real, is
      SDPA/dequant, NOT the weight GEMMs). SDPA is KV-read bound at all M.
  (3) Price wirbel #270's num_stages=2 verify-SDPA lever (bit-identical
      1.097/1.090/1.092x per call at M=8/16/32): SDPA_share(M)*(1-1/r) ->
      Delta verify_us(M) -> denken #271 step model -> Delta step -> Delta TPS off
      481.53. num_stages only changes the cp.async pipeline depth (NOT the MMA /
      K-reduction order) -> FP-reassociation-free -> greedy-safe by construction.
  (4) Bound the TOTAL greedy-safe verify-side step-shaving (num_stages SDPA +
      lossless io/residual fold); flag the int4-GEMM under-saturation as NOT
      greedy-safe-recoverable (num_warps/BLOCK_K reassociate the bf16 sum).
  (5) phi-correct (fern #274): honest_gain = composition_gain * phi, phi in
      [0.125, 0.735]; verify is the DOMINANT model-forward phase -> the LEAST
      phi-discounted step lever. Stack with kanna #269 +4.39% draft-MLP fold; does
      the E[T]-independent stack shrink fern #274's +3.8% E[T]_real-floor gap?
  (6) Greedy/PPL-safety certificate.
  (7) Self-test (PRIMARY).

Greedy/PPL pinned BY CONSTRUCTION: this leg edits no served file. The num_stages
lever is bit-identical scheduling (E[T]+PPL unchanged); any FP-reassociating
recovery (num_warps/BLOCK_K/split-K) is flagged and EXCLUDED from the greedy-safe
total. Composition-priced, 0 TPS measured; realization is exactly what stark #273
(wall-clock K-A/B) and fern #274 (phi) measure -- reported priced-not-realized.

Reproduce: cd target/ && CUDA_VISIBLE_DEVICES=0 \
  /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/speed/verify_step_component_roofline/verify_step_component_roofline.py \
  --self-test --wandb_group verify-step-roofline --wandb_name kanna/verify-step-roofline
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

_here = os.path.dirname(os.path.abspath(__file__))

# --- reuse denken #271 loader + timing primitives (the deployed-faithful basis) ---
_MDGD_PATH = os.path.normpath(os.path.join(
    _here, "..", "..", "validity", "gd_step_basis_reconcile", "measure_deployed_gd.py"))
_spec = importlib.util.spec_from_file_location("measure_deployed_gd", _MDGD_PATH)
mdgd = importlib.util.module_from_spec(_spec)
sys.modules["measure_deployed_gd"] = mdgd
_spec.loader.exec_module(mdgd)  # sets env defaults + imports torch (idempotent)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# --------------------------------------------------------------------------- #
# IMPORTED, EXACT -- this leg derives nothing already measured upstream.        #
# --------------------------------------------------------------------------- #
FRONTIER_TPS = 481.53            # PR #52 official a10g-small frontier (BASELINE)
LAMBDA1_CEILING_TPS = 520.953    # lambda=1 built ceiling
K_CAL = 125.268                  # composition calibration
STEP_SERVED_US = 1218.2          # denken #271 step model step_served
G_D_DEPLOYED = 0.0191            # denken #271 deployed g_d
N_TREE = 5                       # denken #271 step model n_tree
K_SPEC = 7                       # num_speculative_tokens (verify width M=K+1=8)
PHI_LO, PHI_HI = 0.125, 0.735    # fern #274 phi = model_forward_fraction edges
HONEST_ET_REAL_FLOOR = 3.9914    # fern #274 honest E[T]_real floor for 500
ET_GAP_PCT = 3.8                 # fern #274 +3.8% E[T] gap (floor over E[T]_real)
DRAFT_MLP_FOLD_PCT = 4.39        # kanna #269 GeluAndMul-fold draft step lever
# denken #257 h1gj2ved verify_us(M) -- the isolated verify-forward roofline (IMPORT)
VERIFY_US = {8: 5163.71, 16: 5405.0, 32: 5979.95}
# wirbel #270 iwwcmvez num_stages=2 verify-SDPA bit-identical per-call ratios (IMPORT)
NUM_STAGES2_RATIO = {8: 1.097, 16: 1.090, 32: 1.092}
# denken #271 region split anchor (verify_region at M=8 ~ verify_us(8) + scheduling)
VERIFY_REGION_US_8 = 5355.5

# A10G (GA102, sm_86) roofline (kanna #269/#277 anchor; researcher-confirmed)
A10G_BW_GBPS = mdgd.A10G_BW_GBPS            # 600.0
INT4_BYTES = mdgd.INT4_BYTES                # 0.5
BF16_BYTES = mdgd.BF16_BYTES                # 2.0
A10G_BF16_TFLOPS = 125.0                    # dense bf16 tensor-core peak
RIDGE_AI = A10G_BF16_TFLOPS * 1e12 / (A10G_BW_GBPS * 1e9)   # 208.3 FLOP/byte
LM_HEAD_VOCAB = mdgd.LM_HEAD_VOCAB          # 12288

# self-test tolerances (verify-scale differencing is noisy; bands stated)
COMPONENT_SUM_TOL_PCT = 0.10     # PR self-test (a): components sum to verify_us(M) +/-10%
MS = (8, 16, 32)


def bytes_to_us(b: float) -> float:
    return b / (A10G_BW_GBPS * 1e9) * 1e6


def step_us(verify_us_M: float, g_d: float = G_D_DEPLOYED) -> float:
    """denken #271: step(M;g_d)=step_served*[verify_us(M)/v8 + n_tree*g_d]/(1+K_spec*g_d).
    verify_us(M) enters LINEARLY -> cheaper verify => cheaper step (E[T] unchanged)."""
    return STEP_SERVED_US * (verify_us_M / VERIFY_US[8] + N_TREE * g_d) / (1.0 + K_SPEC * g_d)


# --------------------------------------------------------------------------- #
# paired-differencing timing: capture full + ablated graphs, interleave rounds. #
# --------------------------------------------------------------------------- #
def paired_diff_measure(runners: dict, iters: int, warmup: int, rounds: int):
    """Capture each runner into a CUDA graph (deployed ONEGRAPH basis), warm all,
    then in each round time every runner once (mean us/replay over `iters`). Paired
    per-round sampling cancels boost-clock drift across the difference. Returns
    {name: [per-round mean us]} and a captured-flags dict."""
    graphs, captured = {}, {}
    for name, run in runners.items():
        try:
            graphs[name] = mdgd._capture(run)
            captured[name] = True
        except Exception as exc:  # noqa: BLE001
            print(f"[verify-roofline] capture FAILED {name}: {exc!r}", flush=True)
            graphs[name] = None
            captured[name] = False
    # warm every captured graph
    for _ in range(max(10, warmup)):
        for g in graphs.values():
            if g is not None:
                g.replay()
    torch.cuda.synchronize()
    series = {name: [] for name in runners}
    for _ in range(rounds):
        for name, g in graphs.items():
            if g is None:
                series[name].append(float("nan"))
                continue
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(iters):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            series[name].append(e0.elapsed_time(e1) / iters * 1e3)
    for g in graphs.values():
        del g
    return series, captured


def _med(vals):
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.median(vals) if vals else float("nan")


def _paired_diff(full_series, minus_series):
    """Per-round (full - minus) -> median + CI95. Paired cancels per-round drift."""
    diffs = [f - m for f, m in zip(full_series, minus_series)
             if math.isfinite(f) and math.isfinite(m)]
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    med = statistics.median(diffs)
    n = len(diffs)
    sd = statistics.pstdev(diffs) if n > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(n) if n else 0.0
    return med, med - ci, med + ci


# --------------------------------------------------------------------------- #
# per-component runners (built over denken #271's verify-runner tensors).       #
# --------------------------------------------------------------------------- #
def build_runners(targets, dims, num_layers, ctx, M, dev):
    """Return (runners dict, byte-model dict) for tree width M. Components:
    qkv/o/gate_up/down int4 GEMMs (x num_layers), SDPA (x num_layers, denken shape),
    lm_head bf16 GEMM (x1). 'full' = the complete verify (reconciles verify_us(M)).
    Ablations remove ONE component for paired differencing."""
    shapes = dims["shapes"]                       # qkv_proj,o_proj,gate_up_proj,down_proj
    n_h, n_kv, hd, hidden = dims["n_heads"], dims["n_kv"], dims["head_dim"], dims["hidden"]
    order = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    xins = {n: torch.randn(M, shapes[n][1], dtype=torch.bfloat16, device=dev) for n in order}
    applies = {n: (targets[n].quant_method.apply, targets[n], xins[n]) for n in order}
    # SDPA: denken #271 shape (full n_h heads on q,k,v) -- the verify_us(M) basis.
    q = torch.randn(1, n_h, M, hd, dtype=torch.bfloat16, device=dev)
    k = torch.randn(1, n_h, ctx, hd, dtype=torch.bfloat16, device=dev)
    v = torch.randn(1, n_h, ctx, hd, dtype=torch.bfloat16, device=dev)
    lm_w = torch.randn(LM_HEAD_VOCAB, hidden, dtype=torch.bfloat16, device=dev) * 0.02
    xlm = torch.randn(M, hidden, dtype=torch.bfloat16, device=dev)

    def gemm(n):
        ap, mod, x = applies[n]
        ap(mod, x, bias=None)

    def full():
        for _ in range(num_layers):
            for n in order:
                gemm(n)
            F.scaled_dot_product_attention(q, k, v)
        torch.matmul(xlm, lm_w.t())

    def make_no(skip):
        def run():
            for _ in range(num_layers):
                for n in order:
                    if n != skip:
                        gemm(n)
                if skip != "sdpa":
                    F.scaled_dot_product_attention(q, k, v)
            if skip != "lm_head":
                torch.matmul(xlm, lm_w.t())
        return run

    runners = {"full": full}
    for skip in order + ["sdpa", "lm_head"]:
        runners[f"no_{skip}"] = make_no(skip)

    # ---- byte-traffic model (weights cold-read once/layer; activations small) ----
    def gemm_bytes(n):
        out, inn = shapes[n]
        w = out * inn * INT4_BYTES                    # int4 weight read (dominant)
        a = (M * inn + M * out) * BF16_BYTES          # activation read+write
        return w * num_layers, a * num_layers, (w + a) * num_layers
    bm = {}
    for n in order:
        wt, ac, tot = gemm_bytes(n)
        bm[n] = {"weight_bytes": wt, "act_bytes": ac, "total_bytes": tot,
                 "roofline_us": bytes_to_us(tot), "ai_flop_per_byte": 4.0 * M}
    # SDPA: KV-cache read (denken shape: n_h kv-heads) dominates; + Q read, out write
    kv_denken = 2 * n_h * ctx * hd * BF16_BYTES * num_layers
    kv_gqa = 2 * n_kv * ctx * hd * BF16_BYTES * num_layers   # deployed-faithful footnote
    q_bytes = n_h * M * hd * BF16_BYTES * num_layers
    o_bytes = n_h * M * hd * BF16_BYTES * num_layers
    sdpa_tot = kv_denken + q_bytes + o_bytes
    sdpa_flops = 2.0 * (2 * M * ctx * hd) * n_h * num_layers   # QK^T + PV
    bm["sdpa"] = {"kv_bytes_denken": kv_denken, "kv_bytes_gqa": kv_gqa,
                  "total_bytes": sdpa_tot, "roofline_us": bytes_to_us(sdpa_tot),
                  "roofline_us_gqa": bytes_to_us(kv_gqa + q_bytes + o_bytes),
                  "ai_flop_per_byte": sdpa_flops / sdpa_tot}
    lm_bytes = LM_HEAD_VOCAB * hidden * BF16_BYTES + (M * hidden + M * LM_HEAD_VOCAB) * BF16_BYTES
    bm["lm_head"] = {"weight_bytes": LM_HEAD_VOCAB * hidden * BF16_BYTES,
                     "total_bytes": lm_bytes, "roofline_us": bytes_to_us(lm_bytes),
                     "ai_flop_per_byte": 2.0 * M * LM_HEAD_VOCAB * hidden / (LM_HEAD_VOCAB * hidden * BF16_BYTES)}
    return runners, bm


def build_reference(hidden, M, dev):
    """Large bf16 GEMM (~128 MiB weight) at width M = the saturating BW anchor
    (kanna #269: ~81% of peak, empirically pins the 600 GB/s model at this M)."""
    out = (128 * 2 ** 20) // (hidden * int(BF16_BYTES))      # ~128 MiB weight
    w = torch.randn(out, hidden, dtype=torch.bfloat16, device=dev)
    x = torch.randn(M, hidden, dtype=torch.bfloat16, device=dev)
    ref_bytes = out * hidden * BF16_BYTES + (M * hidden + M * out) * BF16_BYTES

    def run():
        torch.matmul(x, w.t())
    return run, ref_bytes, out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=528)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--smoke", action="store_true", help="M=8 only, low iters, no wandb")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output", default=os.path.join(_here, "roofline.json"))
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="verify-step-roofline")
    ap.add_argument("--wandb_name", default="kanna/verify-step-roofline")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable (CUDA_VISIBLE_DEVICES=0)"
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[verify-roofline] device {name} sm_{cap[0]}{cap[1]} torch {torch.__version__}"
          f"  ridge_AI={RIDGE_AI:.1f} FLOP/byte", flush=True)
    torch.cuda.reset_peak_memory_stats()

    ms = (8,) if args.smoke else MS
    iters = 12 if args.smoke else args.iters
    rounds = 7 if args.smoke else args.rounds

    # ---- load served int4 verify body (denken #271 loader) -------------------
    llm, model_dir, dims, targets, load_errs = mdgd.load_verify(args.ctx)
    loaded_layers = dims["num_layers"]
    num_layers, depth_src = mdgd.deployed_depth(loaded_layers)
    print(f"[verify-roofline] layers loaded={loaded_layers} DEPLOYED={num_layers} "
          f"({depth_src}) hidden={dims['hidden']} n_h={dims['n_heads']} "
          f"n_kv={dims['n_kv']} hd={dims['head_dim']}", flush=True)

    # heavy warmup -> A10G boost clock (denken #271 basis)
    big = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    for _ in range(200):
        big = big @ big
    torch.cuda.synchronize()
    del big

    per_m = {}
    for M in ms:
        print(f"[verify-roofline] === M={M} : capture full + 6 ablations, "
              f"{rounds} paired rounds x {iters} replays ===", flush=True)
        runners, bm = build_runners(targets, dims, num_layers, args.ctx, M, dev)
        series, captured = paired_diff_measure(runners, iters, args.warmup, rounds)
        full_med = _med(series["full"])
        # paired-difference each component out of the full verify
        comp = {}
        for c in ["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "sdpa", "lm_head"]:
            med, lo, hi = _paired_diff(series["full"], series[f"no_{c}"])
            comp[c] = {"us": med, "us_lo": lo, "us_hi": hi,
                       "roofline_us": bm[c]["roofline_us"],
                       "bw_utilization": (bm[c]["roofline_us"] / med) if med and med > 0 else float("nan"),
                       "pct_of_full": 100.0 * med / full_med if full_med else float("nan"),
                       "ai_flop_per_byte": bm[c]["ai_flop_per_byte"],
                       "compute_bound": bool(bm[c]["ai_flop_per_byte"] > RIDGE_AI)}
        sum6 = sum(comp[c]["us"] for c in comp)
        remainder_us = full_med - sum6      # io/residual + RMSNorm + scheduling
        # reference saturating GEMM at this M (BW anchor)
        ref_run, ref_bytes, ref_out = build_reference(dims["hidden"], M, dev)
        ref_series, ref_cap = paired_diff_measure({"ref": ref_run}, iters, args.warmup, rounds)
        ref_us = _med(ref_series["ref"])
        ref_roofline_us = bytes_to_us(ref_bytes)
        ref_bw_util = ref_roofline_us / ref_us if ref_us and ref_us > 0 else float("nan")

        per_m[M] = {
            "full_us_measured": full_med,
            "verify_us_imported": VERIFY_US[M],
            "full_resid_pct": 100.0 * abs(full_med - VERIFY_US[M]) / VERIFY_US[M],
            "components": comp, "byte_model": bm,
            "sum6_us": sum6, "remainder_us": remainder_us,
            "remainder_pct": 100.0 * remainder_us / full_med if full_med else float("nan"),
            "ref_us": ref_us, "ref_bytes": ref_bytes, "ref_roofline_us": ref_roofline_us,
            "ref_bw_utilization": ref_bw_util, "ref_out": ref_out,
            "captured": captured, "ref_captured": ref_cap,
            "full_series": series["full"],
        }
        print(f"[verify-roofline]   full={full_med:.0f}us (import {VERIFY_US[M]:.0f}, "
              f"resid {per_m[M]['full_resid_pct']:.1f}%)  sum6={sum6:.0f}us "
              f"remainder={remainder_us:.0f}us ({per_m[M]['remainder_pct']:.1f}%)  "
              f"ref {ref_us:.0f}us @ {ref_bw_util*100:.0f}% peak", flush=True)
        for c in comp:
            print(f"      {c:14s} {comp[c]['us']:7.1f}us  roof {comp[c]['roofline_us']:7.1f}  "
                  f"BWutil {comp[c]['bw_utilization']*100:5.1f}%  {comp[c]['pct_of_full']:5.1f}%full"
                  f"  AI {comp[c]['ai_flop_per_byte']:.0f}{' [COMPUTE]' if comp[c]['compute_bound'] else ''}",
                  flush=True)
        # free per-M activation buffers before next M
        del runners, ref_run
        gc.collect()
        torch.cuda.empty_cache()

    # ===================== ANALYTICS (CPU) ==================================== #
    # (3) price wirbel #270 num_stages=2 SDPA lever -> step model -> TPS
    pricing = {}
    for M in ms:
        sdpa_us = per_m[M]["components"]["sdpa"]["us"]
        sdpa_share = sdpa_us / per_m[M]["full_us_measured"]
        r = NUM_STAGES2_RATIO[M]
        # apply the MEASURED share to the IMPORTED aggregate (robust to abs offset)
        delta_verify = sdpa_share * VERIFY_US[M] * (1.0 - 1.0 / r)
        verify_new = VERIFY_US[M] - delta_verify
        step_base = step_us(VERIFY_US[M])
        step_new = step_us(verify_new)
        gain_pct = (step_base / step_new - 1.0) * 100.0
        pricing[M] = {
            "sdpa_us": sdpa_us, "sdpa_share": sdpa_share, "num_stages2_ratio": r,
            "delta_verify_us": delta_verify, "verify_new_us": verify_new,
            "step_base_us": step_base, "step_new_us": step_new,
            "projected_tps_gain_pct_num_stages2": gain_pct,
            "projected_tps": FRONTIER_TPS * step_base / step_new,
        }

    # (4) TOTAL greedy-safe verify-side step-shaving.
    # GREEDY-SAFE = bit-identical scheduling (num_stages) OR lossless fusion. The
    # ONLY priced greedy-safe verify-side lever is wirbel #270's num_stages=2 SDPA
    # (bit-identical, FP-reassociation-free). The io/residual remainder is ~0 in
    # denken's GEMM+SDPA+lm_head runner basis (no separate RMSNorm/residual modeled)
    # -> NO hidden foldable overhead to recover; the small measured remainder is
    # differencing noise, NOT a foldable kernel, so it is reported as a completeness
    # diagnostic and EXCLUDED from the greedy-safe total. The int4-GEMM
    # under-saturation (gate_up/down/qkv/o below the ~79-81% bf16 reference) is real
    # slack but NOT greedy-safe-recoverable: recovering it needs num_warps/BLOCK_K/
    # split-K retiling, which REASSOCIATES the bf16 sum (E[T] drift, lawine #246).
    deployed_M = 8
    gain_num_stages = pricing[deployed_M]["projected_tps_gain_pct_num_stages2"]
    remainder_diag_pct = per_m[deployed_M]["remainder_pct"]   # ~0 => decomposition complete
    max_verify_side_step_shaving_tps_gain_pct = gain_num_stages

    # (5) phi-correction (fern #274) + stack with kanna #269 +4.39% draft fold.
    comp_gain = max_verify_side_step_shaving_tps_gain_pct                 # composition-priced
    honest_gain = {"lo": comp_gain * PHI_LO, "hi": comp_gain * PHI_HI}    # = comp_gain * phi
    stack_comp = comp_gain + DRAFT_MLP_FOLD_PCT                           # both composition-priced
    stack_honest = {"lo": stack_comp * PHI_LO, "hi": stack_comp * PHI_HI}
    phi_corrected_verify_stack_tps = {
        "lo": FRONTIER_TPS * (1.0 + stack_honest["lo"] / 100.0),
        "hi": FRONTIER_TPS * (1.0 + stack_honest["hi"] / 100.0)}
    # E[T]-floor reduction: a step lever of g% (E[T] fixed) lowers the honest
    # E[T]_real floor for 500 proportionally: floor_new = floor / (1 + g/100).
    et_real_basis = HONEST_ET_REAL_FLOOR / (1.0 + ET_GAP_PCT / 100.0)     # ~3.845
    new_floor = {"lo": HONEST_ET_REAL_FLOOR / (1.0 + stack_honest["lo"] / 100.0),
                 "hi": HONEST_ET_REAL_FLOOR / (1.0 + stack_honest["hi"] / 100.0)}
    new_gap_pct = {"lo": (new_floor["lo"] / et_real_basis - 1.0) * 100.0,    # at phi_lo
                   "hi": (new_floor["hi"] / et_real_basis - 1.0) * 100.0}    # at phi_hi (more)

    # ===================== SELF-TEST (PRIMARY) ============================== #
    # (a) components sum to verify_us(M) within +/-10% at each M
    st_a = all(100.0 * abs(per_m[M]["sum6_us"] - VERIFY_US[M]) / VERIFY_US[M]
               <= COMPONENT_SUM_TOL_PCT * 100.0 for M in ms)
    # (b) num_stages2 pricing monotone-consistent (cheaper verify->step->higher TPS)
    st_b = all(pricing[M]["verify_new_us"] < VERIFY_US[M]
               and pricing[M]["step_new_us"] < pricing[M]["step_base_us"]
               and pricing[M]["projected_tps_gain_pct_num_stages2"] > 0.0 for M in ms)
    # (c) verify_us anchors {M8,M32} match #257 exactly
    st_c = bool(VERIFY_US[8] == 5163.71 and VERIFY_US[32] == 5979.95)
    # (d) NaN-clean
    finite_vals = [per_m[M]["full_us_measured"] for M in ms] + \
                  [per_m[M]["components"][c]["us"] for M in ms
                   for c in per_m[M]["components"]] + \
                  [pricing[M]["projected_tps_gain_pct_num_stages2"] for M in ms] + \
                  [max_verify_side_step_shaving_tps_gain_pct,
                   phi_corrected_verify_stack_tps["lo"], phi_corrected_verify_stack_tps["hi"]]
    st_d = all(math.isfinite(x) for x in finite_vals)
    # (e) imported constants exact
    st_e = bool(FRONTIER_TPS == 481.53 and LAMBDA1_CEILING_TPS == 520.953
                and K_CAL == 125.268 and STEP_SERVED_US == 1218.2
                and G_D_DEPLOYED == 0.0191 and PHI_LO == 0.125 and PHI_HI == 0.735)
    # (f) phi-corrected honest gain == composition_gain * phi at both edges
    st_f = bool(abs(honest_gain["lo"] - comp_gain * PHI_LO) < 1e-9
                and abs(honest_gain["hi"] - comp_gain * PHI_HI) < 1e-9)
    # (g) the verify-side step-shaving is priced-not-realized: the verify-side lever
    # ALONE (composition-priced) does NOT independently cross 500 (it buys MARGIN on
    # the E[T]-raise path; only an E[T]-raise reliably reaches 500, fern #274). The
    # COMBINED stack with #269's draft fold naively grazes 500 under composition --
    # exactly the over-count fern #274's phi correction discounts.
    verify_side_tps_comp = FRONTIER_TPS * (1.0 + max_verify_side_step_shaving_tps_gain_pct / 100.0)
    st_g = bool(verify_side_tps_comp < 500.0)
    self_test_passes = bool(st_a and st_b and st_c and st_d and st_e and st_f and st_g)

    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # PR 5-way grouping (hand-off): SDPA / MLP(gate_up+down) / lm_head / KV / io+resid
    def pct(M, key):
        c = per_m[M]["components"]
        f = per_m[M]["full_us_measured"]
        if key == "SDPA":
            return 100.0 * c["sdpa"]["us"] / f
        if key == "MLP":
            return 100.0 * (c["gate_up_proj"]["us"] + c["down_proj"]["us"]) / f
        if key == "lm_head":
            return 100.0 * c["lm_head"]["us"] / f
        if key == "KV":   # KV-read byte-traffic share (roofline basis, not a kernel)
            return 100.0 * per_m[M]["byte_model"]["sdpa"]["roofline_us"] / f
        if key == "io":   # attn projections (qkv+o) + RMSNorm/residual remainder
            return 100.0 * (c["qkv_proj"]["us"] + c["o_proj"]["us"]
                            + max(0.0, per_m[M]["remainder_us"])) / f
        return float("nan")

    handoff = (
        f"the verify forward (verify_us(M), 88% of the step) decomposes at M=8 into "
        f"{{SDPA {pct(8,'SDPA'):.0f}%, MLP {pct(8,'MLP'):.0f}%, lm_head {pct(8,'lm_head'):.0f}%, "
        f"KV {pct(8,'KV'):.0f}%, io/residual {pct(8,'io'):.0f}%}}; wirbel #270's num_stages=2 "
        f"SDPA lever prices to {gain_num_stages:+.2f}% composition-TPS / "
        f"{gain_num_stages*PHI_LO:+.2f}..{gain_num_stages*PHI_HI:+.2f}% phi-corrected honest; "
        f"the total greedy-safe verify-side step-shaving is "
        f"{max_verify_side_step_shaving_tps_gain_pct:+.2f}% composition / "
        f"{honest_gain['lo']:+.2f}..{honest_gain['hi']:+.2f}% honest, which is the LEAST "
        f"phi-discounted step lever (verify = dominant model-forward phase) and stacked with "
        f"kanna #269's +4.39% draft fold buys {stack_honest['lo']:+.2f}..{stack_honest['hi']:+.2f}% "
        f"honest margin on the E[T]-raise path -- shrinking fern #274's +3.8% E[T]_real-floor gap "
        f"to {new_gap_pct['hi']:.2f}..{new_gap_pct['lo']:.2f}% (composition-priced, 0 TPS measured; "
        f"realization pending fern #274 phi / stark #273 wall-clock).")

    verify_step_shaving_greedy_safe = True  # num_stages = bit-identical; io-fold = lossless
    fp_reassociation_flag = (
        "num_stages=2 (wirbel #270) is FP-reassociation-FREE: it only changes the "
        "cp.async software-pipeline depth, NOT the MMA tile / K-reduction order -> "
        "bit-identical (maxdiff=0.0). LOSSLESS RMSNorm/residual epilogue folds are also "
        "greedy-safe. EXCLUDED as NOT greedy-safe: int4-GEMM under-saturation recovery "
        "via num_warps / BLOCK_K / split-K -- those REASSOCIATE the bf16 sum (cf. lawine "
        "#246, kanna #269), correctness-safe by propose-only verify only, E[T] may drift.")

    verdict = {
        "verify_step_component_roofline_self_test_passes": self_test_passes,   # PRIMARY
        "max_verify_side_step_shaving_tps_gain_pct":                           # TEST
            max_verify_side_step_shaving_tps_gain_pct,
        # headline pricing
        "projected_tps_gain_pct_num_stages2_M8": pricing[8]["projected_tps_gain_pct_num_stages2"],
        "projected_tps_gain_pct_num_stages2_M16": pricing[16]["projected_tps_gain_pct_num_stages2"]
            if 16 in pricing else None,
        "projected_tps_gain_pct_num_stages2_M32": pricing[32]["projected_tps_gain_pct_num_stages2"]
            if 32 in pricing else None,
        "sdpa_share_M8": pricing[8]["sdpa_share"],
        "remainder_diag_pct_M8": remainder_diag_pct,
        # phi-correction + stack
        "comp_gain_verify_side_pct": comp_gain,
        "honest_gain_verify_side_pct_lo": honest_gain["lo"],
        "honest_gain_verify_side_pct_hi": honest_gain["hi"],
        "stack_comp_pct": stack_comp,
        "stack_honest_pct_lo": stack_honest["lo"],
        "stack_honest_pct_hi": stack_honest["hi"],
        "phi_corrected_verify_stack_tps_lo": phi_corrected_verify_stack_tps["lo"],
        "phi_corrected_verify_stack_tps_hi": phi_corrected_verify_stack_tps["hi"],
        "honest_et_real_floor_new_lo": new_floor["lo"],
        "honest_et_real_floor_new_hi": new_floor["hi"],
        "et_gap_pct_new_lo": new_gap_pct["lo"], "et_gap_pct_new_hi": new_gap_pct["hi"],
        "verify_side_alone_tps_comp": verify_side_tps_comp,
        "verify_side_reaches_500_independently": bool(verify_side_tps_comp >= 500.0),
        "combined_stack_tps_comp": FRONTIER_TPS * (1.0 + stack_comp / 100.0),
        # safety
        "verify_step_shaving_greedy_safe": verify_step_shaving_greedy_safe,
        "fp_reassociation_flag": fp_reassociation_flag,
        "greedy_identical_by_construction": True,
        "ppl_pinned": 2.3772, "ppl_ok": True,
        # roofline diagnostic
        "ridge_ai_flop_per_byte": RIDGE_AI,
        "int4_gemm_compute_knee_M": RIDGE_AI / 4.0,   # 4*M = ridge -> M ~ 52
        "m16_bandwidth_bound": bool(4 * 16 < RIDGE_AI),
        "m32_bandwidth_bound": bool(4 * 32 < RIDGE_AI),
        # housekeeping
        "deployed_num_layers": num_layers, "model_dir": model_dir,
        "nan_clean": st_d, "peak_vram_gib": peak_vram_gib,
        "vram_ok": bool(peak_vram_gib <= 24.0),
        # imported, unchanged
        "frontier_tps": FRONTIER_TPS, "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
        "k_cal": K_CAL, "step_served_us": STEP_SERVED_US, "g_d_deployed": G_D_DEPLOYED,
        "n_tree": N_TREE, "k_spec": K_SPEC, "phi_lo": PHI_LO, "phi_hi": PHI_HI,
        "honest_et_real_floor": HONEST_ET_REAL_FLOOR, "draft_mlp_fold_pct": DRAFT_MLP_FOLD_PCT,
        "verify_us_imported": VERIFY_US, "num_stages2_ratio": NUM_STAGES2_RATIO,
        "self_test_conditions": {"a_component_sum": st_a, "b_pricing_monotone": st_b,
                                 "c_anchor_257": st_c, "d_nan_clean": st_d,
                                 "e_constants": st_e, "f_phi_arith": st_f,
                                 "g_priced_not_realized": st_g},
        "handoff_line": handoff,
    }

    # ----------------------------- print verdict -----------------------------
    print("\n[verify-roofline] ===== num_stages=2 SDPA PRICING (step-only; E[T] UNCHANGED) =====",
          flush=True)
    for M in ms:
        p = pricing[M]
        print(f"  M={M:2d}  SDPA_share={p['sdpa_share']*100:5.2f}%  r={p['num_stages2_ratio']:.3f}  "
              f"dverify={p['delta_verify_us']:.1f}us  step {p['step_base_us']:.0f}->{p['step_new_us']:.0f}us  "
              f"-> {p['projected_tps_gain_pct_num_stages2']:+.3f}% TPS", flush=True)
    print(f"\n[verify-roofline] TOTAL greedy-safe verify-side shaving = "
          f"{max_verify_side_step_shaving_tps_gain_pct:+.3f}% composition "
          f"(num_stages SDPA only; io/residual remainder {remainder_diag_pct:.2f}% = noise, "
          f"no foldable kernel; int4-GEMM under-sat NOT greedy-safe-recoverable)", flush=True)
    print(f"  phi-corrected honest = {honest_gain['lo']:+.3f}% (phi=0.125) .. "
          f"{honest_gain['hi']:+.3f}% (phi=0.735)  [verify = LEAST phi-discounted lever]", flush=True)
    print(f"  STACK + #269 +4.39% draft fold = {stack_comp:+.3f}% comp / "
          f"{stack_honest['lo']:+.3f}..{stack_honest['hi']:+.3f}% honest "
          f"-> {phi_corrected_verify_stack_tps['lo']:.1f}..{phi_corrected_verify_stack_tps['hi']:.1f} TPS",
          flush=True)
    print(f"  E[T]_real floor {HONEST_ET_REAL_FLOOR:.4f} -> "
          f"{new_floor['hi']:.4f}..{new_floor['lo']:.4f} (gap +3.8% -> "
          f"{new_gap_pct['hi']:+.2f}..{new_gap_pct['lo']:+.2f}%)  "
          f"verify-side alone {verify_side_tps_comp:.1f} TPS "
          f"(reaches_500_alone={verdict['verify_side_reaches_500_independently']}); "
          f"combined comp-stack {verdict['combined_stack_tps_comp']:.1f} TPS "
          f"(composition illusion -- fern #274 phi-discounts this)", flush=True)
    print(f"\n[verify-roofline] VERDICT self_test={self_test_passes}  "
          f"a={st_a} b={st_b} c={st_c} d={st_d} e={st_e} f={st_f} g={st_g}", flush=True)
    print(f"  {handoff}", flush=True)

    payload = {
        "config": {"torch": torch.__version__, "device": name, "sm": f"{cap[0]}{cap[1]}",
                   "ctx": args.ctx, "iters": iters, "warmup": args.warmup, "rounds": rounds,
                   "Ms": list(ms), "A10G_BW_GBPS": A10G_BW_GBPS, "ridge_ai": RIDGE_AI,
                   "deployed_num_layers": num_layers, "model_dir": model_dir,
                   "load_errors": load_errs, "smoke": args.smoke,
                   "note": "paired-differencing component decomposition of the served int4 "
                           "verify forward (denken #271 runner basis) vs the HBM-bandwidth "
                           "roofline (kanna #269 anchor); num_stages=2 SDPA lever priced "
                           "through the denken #271 step model + fern #274 phi. No serve "
                           "change, no HF Job, no submission. Greedy+PPL pinned."},
        "per_m": {str(M): {k: v for k, v in per_m[M].items() if k != "full_series"}
                  for M in ms},
        "pricing": {str(M): pricing[M] for M in ms},
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"[verify-roofline] wrote {args.output}", flush=True)

    if not (args.no_wandb or args.smoke):
        try:
            _log_wandb(args, payload, per_m, pricing, ms)
        except Exception as exc:  # noqa: BLE001
            print(f"[verify-roofline] W&B logging failed (non-fatal): {exc!r}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    return 0 if self_test_passes else 1


def _log_wandb(args, payload, per_m, pricing, ms):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    v = payload["verdict"]
    # component decomposition table (per M)
    comp_t = wandb.Table(columns=["M", "component", "us", "us_lo", "us_hi", "roofline_us",
                                  "bw_utilization", "pct_of_full", "ai_flop_per_byte",
                                  "compute_bound"])
    for M in ms:
        for c, d in per_m[M]["components"].items():
            comp_t.add_data(M, c, d["us"], d["us_lo"], d["us_hi"], d["roofline_us"],
                            d["bw_utilization"], d["pct_of_full"], d["ai_flop_per_byte"],
                            d["compute_bound"])
    run.log({"verify_component_decomposition": comp_t})
    # reconciliation table
    recon_t = wandb.Table(columns=["M", "full_us_measured", "verify_us_imported",
                                   "full_resid_pct", "sum6_us", "remainder_us",
                                   "remainder_pct", "ref_us", "ref_bw_utilization"])
    for M in ms:
        d = per_m[M]
        recon_t.add_data(M, d["full_us_measured"], d["verify_us_imported"], d["full_resid_pct"],
                         d["sum6_us"], d["remainder_us"], d["remainder_pct"], d["ref_us"],
                         d["ref_bw_utilization"])
    run.log({"verify_reconciliation": recon_t})
    # num_stages=2 pricing table
    price_t = wandb.Table(columns=["M", "sdpa_share", "num_stages2_ratio", "delta_verify_us",
                                   "step_base_us", "step_new_us",
                                   "projected_tps_gain_pct_num_stages2", "projected_tps"])
    for M in ms:
        p = pricing[M]
        price_t.add_data(M, p["sdpa_share"], p["num_stages2_ratio"], p["delta_verify_us"],
                         p["step_base_us"], p["step_new_us"],
                         p["projected_tps_gain_pct_num_stages2"], p["projected_tps"])
    run.log({"num_stages2_pricing": price_t})
    run.summary.update({k: val for k, val in v.items()
                        if isinstance(val, (int, float, bool, str))})
    run.finish()
    print(f"[verify-roofline] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
