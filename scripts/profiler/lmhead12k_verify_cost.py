#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""lmhead12k verify-forward cost model (PR #37).

Quantifies how ubel #14's row-pruned lm_head (262,144 -> 12,288 rows, x0.0469)
reduces the spec-VERIFY forward cost V_tree(M), and rebuilds the tree TPS-ceiling
input curve on the PR #33 tile-corrected basis.

STEP 1 (analytic, --analytic):
  The PR #28 profiler measures the verify step as
    t_step(M) = t_forward(M) + t_lmhead(M),
  with t_lmhead measured PER M (`compute_logits` GEMM [M,2560]x[2560,vocab] +
  full-vocab argmax). Re-derive the step with the lm_head term scaled x0.0469
  (kept_size/full_vocab = 12288/262144). Body GEMM + attention (t_forward) are
  untouched -- lmhead12k only row-prunes the head. Reports the per-M table and
  writes the reduced cost-model curve(s) the tree model (Step 3) consumes.

STEP 2 (measured, --measure):
  A faithful STANDALONE lm_head microbenchmark (no full vLLM load) replicating
  ubel's compute_logits + the profiler's verify argmax, for the FULL bf16 head
  vs the kept_size=12288 row-pruned head. THE HONEST CRUX: ubel's compute_logits
  (vllm_lmhead12k/model.py) does the cheap GEMM at kept_size but SCATTERS the
  result back into a full [M,262144] -inf tensor so greedy argmax stays
  full-vocab and identity-correct (kept rows keep their logit; pruned rows -inf
  never win). So the verify-path saving is GEMM-BANDWIDTH ONLY: the
  scatter-alloc + full-vocab argmax are RETAINED and form a residual floor that
  the x0.0469 analytic under-counts. We measure all three lenses:
    full          : full head GEMM->[M,262144] + softcap + argmax  (baseline)
    k12_scatter   : kept GEMM->[M,12288] + softcap + scatter->[M,262144] + argmax
                    (ubel's ACTUAL served path -- greedy-identity-correct)
    k12_gemm_only : kept GEMM->[M,12288] + softcap + argmax[M,12288]
                    (the pure x0.0469 ceiling if full-vocab identity were free)
  Calibrated against the PR #33 measured t_lmhead curve at the shared M.

Outputs: research/spec_cost_model/lmhead12k_verify_cost.json (+ reduced curves
for the tree model). LOCAL, single GPU, no HF Job, no submission, greedy/PPL
surface untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

KEPT_FRAC = 12288 / 262144  # 0.046875 -- ubel #14 head reduction (21.33x)
HIDDEN = 2560
FULL_VOCAB = 262144
KEPT_SIZE = 12288
TABLE_M = [1, 7, 17, 25, 33, 45, 49]  # PR #37 Step-1 reporting M (tree shapes)
MEASURE_M = [1, 7, 16, 17, 25, 33, 45, 49]  # +M=16 to calibrate vs profiler


# --------------------------------------------------------------------------- #
# Merged PR #28 (msweep, small M) + PR #33 (tile-boundary, M>=16) component curves
# --------------------------------------------------------------------------- #
def _node(path: str, key: str) -> dict:
    return json.load(open(path))["cost_model"][key]


def merged_components(msweep: str, tile: str, key: str):
    """Return (forward_by_M, lmhead_by_M) merged: tile-boundary overrides msweep
    at M it measured, msweep supplies the small-M (M<16) tail. Mirrors
    merge_tree_mask_curve.py's fold so the reduced curve is continuous with #33."""
    ms, tb = _node(msweep, key), _node(tile, key)

    def merge(field):
        m = {int(k): float(v) for k, v in ms[field].items()}
        for k, v in tb[field].items():
            m[int(k)] = float(v)
        return dict(sorted(m.items()))

    return merge("t_forward_ms_by_M"), merge("t_lmhead_ms_by_M")


def interp(tab: dict[int, float], M: float) -> float:
    xs = sorted(tab)
    if M <= xs[0]:
        return tab[xs[0]]
    if M >= xs[-1]:
        return tab[xs[-1]]
    lo = max(x for x in xs if x <= M)
    hi = min(x for x in xs if x >= M)
    if lo == hi:
        return tab[lo]
    t = (M - lo) / (hi - lo)
    return tab[lo] * (1 - t) + tab[hi] * t


# --------------------------------------------------------------------------- #
# Step 1: analytic re-derivation
# --------------------------------------------------------------------------- #
def analytic_table(fwd, lm, Ms, frac=KEPT_FRAC):
    rows = []
    for M in Ms:
        f = interp(fwd, M)
        l_full = interp(lm, M)
        l_12k = l_full * frac
        v_full = f + l_full
        v_12k = f + l_12k
        rows.append({
            "M": M,
            "t_forward_ms": f,
            "lm_head_ms_full": l_full,
            "lm_head_ms_12k": l_12k,
            "lm_head_saving_ms": l_full - l_12k,
            "V_tree_full": v_full,
            "V_tree_lmhead12k": v_12k,
            "step_reduction_pct": 100.0 * (v_full - v_12k) / v_full,
            "lmhead_frac_full": l_full / v_full,
            "lmhead_frac_12k": l_12k / v_12k,
        })
    return rows


def lmhead_m_scaling(lm):
    """Fit t_lmhead(M) ~= a + b*M over the measured tile-boundary M to expose how
    fixed (memory-bound weight read) vs M-linear (compute) the head term is."""
    Ms = sorted(m for m in lm if m >= 16)
    xs = Ms
    ys = [lm[m] for m in Ms]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    return {"intercept_ms": a, "slope_ms_per_M": b, "M_range": [Ms[0], Ms[-1]],
            "fixed_share_at_M45": a / (a + b * 45)}


# --------------------------------------------------------------------------- #
# Step 2: standalone lm_head microbenchmark (GPU)
# --------------------------------------------------------------------------- #
def measure(kept_ids_path, softcap, logits_fp32, steps, warmup, Ms):
    import torch

    dev = torch.device("cuda:0")
    torch.cuda.set_device(0)
    g = torch.Generator(device="cpu").manual_seed(0)
    # bf16 head weights [vocab, hidden] (timing depends on shape/dtype, not values)
    W_full = torch.randn(FULL_VOCAB, HIDDEN, generator=g, dtype=torch.float32).to(
        dev, dtype=torch.bfloat16)
    kept = json.load(open(kept_ids_path))["kept_ids"]
    idx_rows = torch.tensor(kept, dtype=torch.long, device=dev)
    W_12k = W_full.index_select(0, idx_rows).contiguous()  # [12288, hidden] bf16
    cap = float(softcap) if softcap else None

    def _softcap(x):
        if cap is None:
            return x
        return torch.tanh(x / cap) * cap

    def full_head(hs):
        logits = torch.nn.functional.linear(hs, W_full)  # [M, FULL_VOCAB]
        if logits_fp32:
            logits = logits.float()
        logits = _softcap(logits)
        _ = logits.argmax(dim=-1)
        return logits

    def k12_scatter(hs):
        partial = torch.nn.functional.linear(hs, W_12k)  # [M, KEPT_SIZE]
        if logits_fp32:
            partial = partial.float()
        partial = _softcap(partial)
        M = hs.shape[0]
        full = torch.full((M, FULL_VOCAB), float("-inf"),
                          dtype=partial.dtype, device=dev)
        idx = idx_rows.unsqueeze(0).expand(M, -1)
        full.scatter_(1, idx, partial)  # greedy-identity-correct full-vocab logits
        _ = full.argmax(dim=-1)
        return full

    def k12_gemm_only(hs):
        partial = torch.nn.functional.linear(hs, W_12k)  # [M, KEPT_SIZE]
        if logits_fp32:
            partial = partial.float()
        partial = _softcap(partial)
        _ = partial.argmax(dim=-1)
        return partial

    def k12_remap(hs):
        # PR #41 scatter-free FLOOR: kept GEMM -> [M,12288] + softcap + argmax
        # + kept_ids gather -> [M] ORIGINAL token ids (token-identical to the
        # full-vocab argmax because kept_ids is strictly ascending; proven by
        # scripts/profiler/lmhead12k_scatter_equiv.py). No [M,262144] tensor at
        # all. This is the analytic ceiling; deploying it needs a vLLM sampler
        # hook (the served path must return full-vocab logits for argmax->id and
        # for prompt_logprobs gather/log_softmax -- see model.py docstring).
        partial = torch.nn.functional.linear(hs, W_12k)  # [M, KEPT_SIZE]
        if logits_fp32:
            partial = partial.float()
        partial = _softcap(partial)
        idx = partial.argmax(dim=-1)  # [M] kept-row indices
        tok = idx_rows[idx]           # [M] original vocab ids (cheap gather)
        return tok

    # PR #41 DEPLOYABLE partial fix: a persistent [maxM,262144] -inf buffer whose
    # non-kept columns stay -inf forever; each step scatters only the kept columns
    # then argmaxes full-vocab. Eliminates the per-step alloc + -inf fill of the
    # 250k dead positions while returning a BIT-IDENTICAL full-vocab tensor (so
    # greedy/PPL identity is preserved by construction). Keeps the full-vocab
    # argmax read (unavoidable in-plugin). Buffer is keyed by (M, dtype) and
    # allocated during warmup so timed iters measure the reuse (fill-free) cost.
    _persist: dict = {}

    def k12_scatter_persistent(hs):
        partial = torch.nn.functional.linear(hs, W_12k)  # [M, KEPT_SIZE]
        if logits_fp32:
            partial = partial.float()
        partial = _softcap(partial)
        M = hs.shape[0]
        key = (M, partial.dtype)
        buf = _persist.get(key)
        if buf is None:
            buf = torch.full((M, FULL_VOCAB), float("-inf"),
                             dtype=partial.dtype, device=dev)
            _persist[key] = buf
        idx = idx_rows.unsqueeze(0).expand(M, -1)
        buf.scatter_(1, idx, partial)  # only kept cols written; dead cols stay -inf
        _ = buf.argmax(dim=-1)
        return buf

    fns = {"full": full_head, "k12_scatter": k12_scatter,
           "k12_scatter_persistent": k12_scatter_persistent,
           "k12_remap": k12_remap, "k12_gemm_only": k12_gemm_only}

    def time_fn(fn, M):
        hs = torch.randn(M, HIDDEN, generator=g, dtype=torch.float32).to(
            dev, dtype=torch.bfloat16)
        with torch.inference_mode():
            for _ in range(warmup):
                fn(hs)
            torch.cuda.synchronize()
            # PIPELINED (matches spec_cost_model._pipelined): back-to-back enqueue,
            # per-iter GPU deltas read after ONE final sync (no per-step bubble).
            e0 = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
            e1 = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
            torch.cuda.synchronize()
            for i in range(steps):
                e0[i].record()
                fn(hs)
                e1[i].record()
            torch.cuda.synchronize()
            ms = [e0[i].elapsed_time(e1[i]) for i in range(steps)]
        return statistics.median(ms), float(min(ms)), float(max(ms))

    out = {}
    for M in Ms:
        out[M] = {}
        for name, fn in fns.items():
            med, lo, hi = time_fn(fn, M)
            out[M][name] = {"median_ms": med, "min_ms": lo, "max_ms": hi}
    del W_full, W_12k
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# Build reduced cost-model curves for the tree model (Step 3 input)
# --------------------------------------------------------------------------- #
def build_reduced_curve(base_curve_json, key, lm_merged, reduce_fn, out_path,
                        label):
    """reduced step(M) = base_step(M) - (lm_full(M) - lm_reduced(M)).
    base_curve_json is the #33 headline curve (merged_treemask_flopideal.json or
    merged_dense_corrected.json); reduce_fn maps lm_full(M) -> lm_reduced(M)."""
    base = json.load(open(base_curve_json))
    node = base["cost_model"][key]
    lat = {int(k): float(v) for k, v in node["latency_ms_by_M"].items()}
    new_lat = {}
    detail = {}
    for M in sorted(lat):
        lf = interp(lm_merged, M)
        lr = reduce_fn(M, lf)
        new_lat[M] = lat[M] - (lf - lr)
        detail[M] = {"base_ms": lat[M], "lm_full_ms": lf, "lm_reduced_ms": lr,
                     "reduced_ms": new_lat[M]}
    out = {
        "config": {
            "source_base": base_curve_json, "key": key, "label": label,
            "kept_frac": KEPT_FRAC, "kept_size": KEPT_SIZE, "full_vocab": FULL_VOCAB,
            "detail_by_M": {str(m): detail[m] for m in sorted(detail)},
        },
        "cost_model": {key: {
            "latency_ms_by_M": {str(m): new_lat[m] for m in sorted(new_lat)},
            "attention_pct_step_by_M": node.get("attention_pct_step_by_M", {}),
        }},
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    return out_path, new_lat


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msweep", default="research/spec_cost_model/results_msweep.json")
    ap.add_argument("--tile", default="research/spec_cost_model/results_tile_boundary.json")
    ap.add_argument("--key", default="graph|ctx256")
    ap.add_argument("--flopideal-curve",
                    default="research/spec_cost_model/merged_treemask_flopideal.json")
    ap.add_argument("--dense-curve",
                    default="research/spec_cost_model/merged_dense_corrected.json")
    ap.add_argument("--kept-ids",
                    default="submissions/lmhead12k_empirical/kept_ids.json")
    ap.add_argument("--measure", action="store_true",
                    help="run the GPU lm_head microbenchmark (Step 2)")
    ap.add_argument("--softcap", type=float, default=30.0,
                    help="final_logit_softcapping (0 disables); calibrate vs #33")
    ap.add_argument("--logits-fp32", action="store_true",
                    help="upcast logits to fp32 before softcap/argmax (vLLM path)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--output",
                    default="research/spec_cost_model/lmhead12k_verify_cost.json")
    ap.add_argument("--wandb_project", "--wandb-project", dest="wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", "--wandb-entity", dest="wandb_entity",
                    default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="spec-verify-lmhead12k")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    fwd, lm = merged_components(args.msweep, args.tile, args.key)
    print(f"[lmhead12k] merged component M points: {sorted(fwd)}", flush=True)

    # ---- Step 1: analytic ----
    rows = analytic_table(fwd, lm, TABLE_M)
    scaling = lmhead_m_scaling(lm)
    print("\n[lmhead12k] ===== Step 1: analytic V_tree(M) with lm_head x0.0469 =====",
          flush=True)
    print(f"{'M':>3} {'fwd':>7} {'lm_full':>8} {'lm_12k':>7} {'save':>6} "
          f"{'V_full':>7} {'V_12k':>7} {'step_red%':>9} {'lmF%':>5}->{'lm12%':>5}",
          flush=True)
    for r in rows:
        print(f"{r['M']:>3} {r['t_forward_ms']:7.3f} {r['lm_head_ms_full']:8.3f} "
              f"{r['lm_head_ms_12k']:7.3f} {r['lm_head_saving_ms']:6.3f} "
              f"{r['V_tree_full']:7.3f} {r['V_tree_lmhead12k']:7.3f} "
              f"{r['step_reduction_pct']:8.2f}% {100*r['lmhead_frac_full']:4.1f}%"
              f"->{100*r['lmhead_frac_12k']:4.1f}%", flush=True)
    print(f"[lmhead12k] t_lmhead(M) fit: {scaling['intercept_ms']:.3f} + "
          f"{scaling['slope_ms_per_M']:.4f}*M ms over M="
          f"{scaling['M_range']}; fixed share @M45 = "
          f"{100*scaling['fixed_share_at_M45']:.1f}% -> the head term is "
          f"dominated by the M-INDEPENDENT weight-read (memory-bound).", flush=True)

    # ---- Step 2: measured ----
    measured = None
    calib = None
    if args.measure:
        print("\n[lmhead12k] ===== Step 2: GPU microbenchmark (full vs 12k-scatter) "
              "=====", flush=True)
        measured = measure(args.kept_ids, args.softcap, args.logits_fp32,
                           args.steps, args.warmup, MEASURE_M)
        print(f"{'M':>3} {'full':>7} {'scatter':>8} {'persist':>8} {'remap':>7} "
              f"{'gemm':>7} | {'floor':>7} {'fill':>6} {'argmax':>7}", flush=True)
        print("    (ms; floor=scatter-remap total scatter cost; "
              "fill=scatter-persist alloc+fill; argmax=persist-remap full-vocab read)",
              flush=True)
        for M in MEASURE_M:
            mf = measured[M]["full"]["median_ms"]
            ms = measured[M]["k12_scatter"]["median_ms"]
            mp = measured[M]["k12_scatter_persistent"]["median_ms"]
            mr = measured[M]["k12_remap"]["median_ms"]
            mg = measured[M]["k12_gemm_only"]["median_ms"]
            floor = ms - mr      # total scatter floor (the PR #41 saving target)
            fill = ms - mp       # alloc + -inf fill (the deployable persistent win)
            argmax = mp - mr     # residual full-vocab argmax read
            print(f"{M:>3} {mf:7.3f} {ms:8.3f} {mp:8.3f} {mr:7.3f} {mg:7.3f} | "
                  f"{floor:7.3f} {fill:6.3f} {argmax:7.3f}", flush=True)
        # calibration anchor M=45 highlight
        m45 = measured.get(45)
        if m45:
            s45, r45 = m45["k12_scatter"]["median_ms"], m45["k12_remap"]["median_ms"]
            print(f"[lmhead12k] M=45 scatter floor (scatter-remap) = "
                  f"{s45 - r45:.4f} ms  (scatter {s45:.4f} -> remap {r45:.4f})",
                  flush=True)
        # calibration: measured full vs profiler t_lmhead at shared M
        calib = {str(M): {"measured_full_ms": measured[M]["full"]["median_ms"],
                          "profiler_lmhead_ms": interp(lm, M),
                          "ratio": measured[M]["full"]["median_ms"] / interp(lm, M)}
                 for M in MEASURE_M}
        ratios = [c["ratio"] for c in calib.values()]
        print(f"[lmhead12k] calibration measured_full/profiler ratio: "
              f"mean={statistics.mean(ratios):.3f} "
              f"min={min(ratios):.3f} max={max(ratios):.3f}", flush=True)

    # ---- Build reduced curves for Step 3 ----
    # Analytic reduction: lm_reduced = x0.0469 * lm_full (optimistic ceiling).
    analytic_reduce = lambda M, lf: lf * KEPT_FRAC
    curves = {}
    for base_json, base_tag in [(args.flopideal_curve, "flopideal"),
                                (args.dense_curve, "dense")]:
        if not os.path.exists(base_json):
            continue
        out_path = base_json.replace(".json", "_lmhead12k_analytic.json")
        p, _ = build_reduced_curve(base_json, args.key, lm, analytic_reduce,
                                   out_path, f"{base_tag}+lmhead12k(analytic x0.0469)")
        curves[f"{base_tag}_analytic"] = p
        print(f"[lmhead12k] wrote reduced curve ({base_tag}, analytic): {p}", flush=True)

    # Measured reduction: lm_reduced(M) = measured k12_scatter, anchored to the
    # profiler scale via the per-M calibration ratio (so the residual scatter floor
    # is folded into the SAME basis as the #33 curve).
    if measured is not None:
        meas_tab = {M: measured[M]["k12_scatter"]["median_ms"] for M in MEASURE_M}
        full_tab = {M: measured[M]["full"]["median_ms"] for M in MEASURE_M}
        remap_tab = {M: measured[M]["k12_remap"]["median_ms"] for M in MEASURE_M}
        persist_tab = {M: measured[M]["k12_scatter_persistent"]["median_ms"]
                       for M in MEASURE_M}

        def _anchored(tab):
            # scale a measured microbench curve by (profiler_full / measured_full)
            # so it sits on the profiler's absolute t_lmhead scale (removes
            # microbench/vLLM-kernel offset); lf == profiler full lmhead at M.
            def reduce(M, lf):
                mfull = interp(full_tab, M)
                mv = interp(tab, M)
                return mv * (lf / mfull) if mfull > 0 else mv
            return reduce

        # (a) measured WITH scatter (current served path, PR #37 basis)
        # (b) scatter-free REMAP floor (PR #41 ceiling) -> *_scatter_free.json
        # (c) persistent-buffer DEPLOYABLE partial fix -> *_lmhead12k_persistent.json
        variants = [
            ("_lmhead12k_measured.json", _anchored(meas_tab), "measured(scatter)",
             "measured"),
            ("_scatter_free.json", _anchored(remap_tab), "scatter-free(remap floor)",
             "scatter_free"),
            ("_lmhead12k_persistent.json", _anchored(persist_tab),
             "persistent-buffer(deployable)", "persistent"),
        ]
        for base_json, base_tag in [(args.flopideal_curve, "flopideal"),
                                    (args.dense_curve, "dense")]:
            if not os.path.exists(base_json):
                continue
            for suffix, reduce_fn, label, ckey in variants:
                out_path = base_json.replace(".json", suffix)
                p, _ = build_reduced_curve(base_json, args.key, lm, reduce_fn,
                                           out_path, f"{base_tag}+lmhead12k({label})")
                curves[f"{base_tag}_{ckey}"] = p
                print(f"[lmhead12k] wrote reduced curve ({base_tag}, {label}): {p}",
                      flush=True)

    payload = {
        "config": {
            "kept_frac": KEPT_FRAC, "kept_size": KEPT_SIZE, "full_vocab": FULL_VOCAB,
            "hidden": HIDDEN, "table_M": TABLE_M, "measure_M": MEASURE_M,
            "msweep": args.msweep, "tile": args.tile, "key": args.key,
            "softcap": args.softcap, "logits_fp32": args.logits_fp32,
            "steps": args.steps, "warmup": args.warmup,
        },
        "step1_analytic": {"rows": rows, "lmhead_m_scaling": scaling},
        "step2_measured": measured,
        "step2_calibration": calib,
        "reduced_curves": curves,
        "merged_forward_by_M": {str(m): fwd[m] for m in sorted(fwd)},
        "merged_lmhead_by_M": {str(m): lm[m] for m in sorted(lm)},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(payload, open(args.output, "w"), indent=2)
    print(f"\n[lmhead12k] wrote {args.output}", flush=True)

    if not args.no_wandb and args.wandb_name:
        try:
            _log_wandb(args, payload)
        except Exception as e:  # noqa: BLE001
            print(f"[lmhead12k] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[lmhead12k] DONE", flush=True)


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    summary = {}
    for r in payload["step1_analytic"]["rows"]:
        t = f"M{r['M']}"
        summary[f"V_full_{t}"] = r["V_tree_full"]
        summary[f"V_lmhead12k_{t}"] = r["V_tree_lmhead12k"]
        summary[f"lm_saving_ms_{t}"] = r["lm_head_saving_ms"]
        summary[f"step_reduction_pct_{t}"] = r["step_reduction_pct"]
    sc = payload["step1_analytic"]["lmhead_m_scaling"]
    summary["lmhead_fit_intercept_ms"] = sc["intercept_ms"]
    summary["lmhead_fit_slope_ms_per_M"] = sc["slope_ms_per_M"]
    summary["lmhead_fixed_share_at_M45"] = sc["fixed_share_at_M45"]
    if payload["step2_measured"]:
        for M, d in payload["step2_measured"].items():
            summary[f"meas_full_M{M}"] = d["full"]["median_ms"]
            summary[f"meas_k12_scatter_M{M}"] = d["k12_scatter"]["median_ms"]
            summary[f"meas_k12_persistent_M{M}"] = \
                d["k12_scatter_persistent"]["median_ms"]
            summary[f"meas_k12_remap_M{M}"] = d["k12_remap"]["median_ms"]
            summary[f"meas_k12_gemm_only_M{M}"] = d["k12_gemm_only"]["median_ms"]
            # scatter floor decomposition (ms)
            summary[f"scatter_floor_M{M}"] = (
                d["k12_scatter"]["median_ms"] - d["k12_remap"]["median_ms"])
            summary[f"alloc_fill_M{M}"] = (
                d["k12_scatter"]["median_ms"]
                - d["k12_scatter_persistent"]["median_ms"])
            summary[f"fullvocab_argmax_M{M}"] = (
                d["k12_scatter_persistent"]["median_ms"]
                - d["k12_remap"]["median_ms"])
        cr = payload["step2_calibration"]
        summary["calib_ratio_mean"] = statistics.mean(c["ratio"] for c in cr.values())
    cols = ["M", "t_forward_ms", "lm_head_ms_full", "lm_head_ms_12k",
            "lm_head_saving_ms", "V_tree_full", "V_tree_lmhead12k",
            "step_reduction_pct"]
    tbl = wandb.Table(columns=cols)
    for r in payload["step1_analytic"]["rows"]:
        tbl.add_data(*[r[c] for c in cols])
    run.log({"analytic_table": tbl})
    run.summary.update({k: v for k, v in summary.items() if v is not None})
    run.finish()
    print(f"[lmhead12k] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
