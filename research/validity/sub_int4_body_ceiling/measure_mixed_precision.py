#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Mixed-precision sub-int4 body-shrink MEASUREMENT (PR #372) — the last
unexplored facet of lever (a): sensitivity-weighted bit ALLOCATION.

WHY THIS LEG (PR #372)
----------------------
#355 (RTN 3-bit, +13.94% rel PPL) and #367 (codebook+incoherence 3-bit, +5.87%)
both quantized ALL 258 body Linear modules UNIFORMLY to 3-bit and both blew the
razor-thin +1.81% relative PPL headroom (gate 2.42 vs deployed 2.3772). #367's
conclusion — "the ceiling is set by the headroom, not the method" — is about the
quantizer FAMILY, not the bit ALLOCATION. This file measures whether a
sensitivity-weighted MIXED-precision allocation (3-bit codebook on the
least-sensitive modules, int4 on the sensitive ones) can buy a positive
body-read shrink while keeping the MEASURED gate-transfer PPL <= 2.42.

GATE-TRANSFER PPL (the binding wall, same as #355/#367)
------------------------------------------------------
  gate_PPL = SERVED_INT4_PPL_SPEC * (PPL_config / PPL_int4_localproxy)
where PPL_int4_localproxy is the in-run all-int4 anchor (#355/#367: 1.9512) and
SERVED_INT4_PPL_SPEC=2.3772. Pass iff gate_PPL <= 2.42  <=>  PPL_config <=
1.9512 * (2.42/2.3772) = 1.9863  (a +1.80% relative local budget).

THE 3-bit OPTION = #367's best calibration-free codebook: per-group (g128) RHT
incoherence + fixed Gaussian VQ lattice (vq_dim=2, K=64), eff 3.125 bpw. The
4-bit option = the deployed int4 (RTN per-group g128), eff 4.125 bpw. Same 258
body linears as #355/#367 (lm_head/embed/norms held at source precision).

DELIVERABLES (PR #372)
----------------------
 1. PER-MODULE sensitivity: each (layer x proj) = exactly one Linear (258 total).
    Quantize ONE module to cb3 (rest int4), measure gate-transfer PPL delta vs the
    all-int4 baseline, on the SAME official methodology (128 records, 61,797 tok,
    token-weighted). -> sensitivity distribution + `sensitivity_is_concentrated`.
 2. GREEDY allocation under the headroom: rank modules ascending by sensitivity;
    cumulatively assign cb3 to the least-sensitive; find the largest 3-bit fraction
    whose MEASURED gate-transfer PPL <= 2.42 (NOT the additive sum — quant errors
    interact). Report frac_modules_3bit_within_headroom, achievable_avg_bpw,
    body_read_shrink_frac, mixed_config_measured_gate_ppl, additivity_gap.
 3. Ceiling-lift translation: body_read_shrink_frac -> batch=1 TPS lift via
    1/(0.943*body_bytes_frac + 0.057) on BOTH bases (165.44 non-spec, 520.953
    lambda=1 spec). Clears private-500 residual? (pred_spec*0.957 >= 500). Beats
    the +33.62-over-520.953 #360 mark? Anchor: uniform cb3 -> 214.5 (== #367).
 4. Greedy-identity sanity (report-only): 128-tok greedy decode on >=8 prompts vs
    the int4 reference -> mixed_greedy_frac_mismatch (the #319 gate is
    self-referential; any deterministic quant passes by construction).

GO: mixed_precision_ceiling_lift_go = (achievable_avg_bpw < 4.125 within the PPL
gate) AND (pred_ceiling_lift_tps > 0).

RESUMABILITY: 258 PPL evals (~29s each @128 rec) > the 90-min run cap, so the
sweep checkpoints after EVERY module to `mixed_precision_checkpoint.json` and a
single invocation stops cleanly at `--max-seconds`. Re-run to resume; the run
that completes the sweep also runs finalize (allocation + ceiling-lift + GO +
W&B). 0 official TPS (feasibility ceiling-lift card; no submission, no
served-file change).

Run (chunked; re-run until it prints FINALIZE):
    cd target/ && CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online .venv/bin/python \
      research/validity/sub_int4_body_ceiling/measure_mixed_precision.py \
      --max-seconds 4500 --wandb_group mixed-precision-bit-allocation \
      --wandb_name lawine/mixed-precision-subint4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import measure_subint4 as m355  # noqa: E402
import measure_codebook as mcb  # noqa: E402
import sub_int4_body_ceiling as card  # noqa: E402

# ---- anchors / constants (single source of truth) -------------------------- #
PPL_GATE = card.PPL_GATE                                   # 2.42
STRICT_NONSPEC_FLOOR_TPS = card.STRICT_NONSPEC_FLOOR_TPS   # 165.44 (lawine #196)
SERVED_INT4_PPL_SPEC = m355.SERVED_INT4_PPL_SPEC           # 2.3772 (gate's named anchor)
DEFAULT_PPL_DATASET = m355.DEFAULT_PPL_DATASET
DEFAULT_BASE_MODEL = m355.DEFAULT_BASE_MODEL

INT4_BPW = 4.125          # 4 index + 16/128 scale
CB3_BPW = 3.125           # 3 index + 16/128 scale (codebook amortization ~0; #367)
BODY_HBM_FRAC = 0.943     # body weight read = 94.3% of step HBM (denken #344)
NONBODY_HBM_FRAC = 0.057
SPEC_CEILING_LAMBDA1 = 520.953   # lambda=1 spec ceiling (PR #372 baseline)
PRIVATE_GAP_FACTOR = 0.957       # public->private (measured 4.3% gap)
LEVER_A_MARK_360 = 33.62         # #360 lever-(a) requirement: lift ceiling by >= this

CKPT_NAME = "mixed_precision_checkpoint.json"
RESULTS_NAME = "measure_mixed_precision_results.json"
FINALIZE_RESERVE_S = 1000.0  # if the sweep completes with less than this budget left,
# defer finalize to a fresh invocation so it never gets killed mid-run by the 90-min cap.


# --------------------------------------------------------------------------- #
# Byte / ceiling-lift model (PR #372 deliverable 3).
# --------------------------------------------------------------------------- #
def tps_lift_factor(body_bytes_frac: float) -> float:
    """Batch=1 BW-bound lift factor when body bytes shrink to `body_bytes_frac` of
    the all-int4 body read. 1/(0.943*frac + 0.057). frac=1 -> 1.0;
    frac=3.125/4.125 (uniform cb3) -> 1.2963 (=> 165.44*1.2963=214.5, == #367)."""
    return 1.0 / (BODY_HBM_FRAC * body_bytes_frac + NONBODY_HBM_FRAC)


def ceiling_lift(avg_bpw: float) -> dict[str, Any]:
    body_bytes_frac = avg_bpw / INT4_BPW
    body_read_shrink_frac = 1.0 - body_bytes_frac
    f = tps_lift_factor(body_bytes_frac)
    spec = SPEC_CEILING_LAMBDA1 * f
    nonspec = STRICT_NONSPEC_FLOOR_TPS * f
    return {
        "avg_bpw": avg_bpw,
        "body_bytes_frac": body_bytes_frac,
        "body_read_shrink_frac": body_read_shrink_frac,
        "tps_lift_factor": f,
        "pred_ceiling_lift_tps_spec": spec,
        "pred_ceiling_lift_tps_nonspec": nonspec,
        "pred_spec_ceiling_delta_tps": spec - SPEC_CEILING_LAMBDA1,
        "pred_nonspec_ceiling_delta_tps": nonspec - STRICT_NONSPEC_FLOOR_TPS,
        "spec_private_after_gap": spec * PRIVATE_GAP_FACTOR,
        "spec_clears_private_500": bool(spec * PRIVATE_GAP_FACTOR >= 500.0),
        "spec_beats_360_mark": bool((spec - SPEC_CEILING_LAMBDA1) >= LEVER_A_MARK_360),
    }


def gate_transfer_ppl(ppl_config: float, ppl_int4: float) -> float:
    return SERVED_INT4_PPL_SPEC * (ppl_config / ppl_int4)


# --------------------------------------------------------------------------- #
# Module grouping: each (layer, proj) is exactly one body Linear.
# --------------------------------------------------------------------------- #
def parse_module(name: str) -> tuple[int, str]:
    parts = name.split(".")
    layer = -1
    for i, p in enumerate(parts):
        if p == "layers":
            layer = int(parts[i + 1])
            break
    return layer, parts[-1]


# --------------------------------------------------------------------------- #
# Per-module quant state machine (avoids re-applying all 258 each config).
# --------------------------------------------------------------------------- #
class QuantModel:
    """Holds the model in a per-module {bf16,int4,cb3} state; flips lazily."""

    def __init__(self, model, snap, group_size, scheme, vq_dim, codebook, R, device):
        self.model = model
        self.snap = snap
        self.mods = dict(model.named_modules())
        self.group_size = group_size
        self.scheme = scheme
        self.vq_dim = vq_dim
        self.codebook = codebook
        self.R = R
        self.device = device
        self.state = {name: "bf16" for name in snap}   # model loads original bf16
        self._cb3_cache: dict[str, torch.Tensor] = {}

    def _cb3(self, name: str) -> torch.Tensor:
        cached = self._cb3_cache.get(name)
        if cached is not None:
            return cached.to(self.device)
        w0 = self.snap[name].to(self.device)
        wq = mcb.codebook_quant_per_group(w0, self.group_size, self.vq_dim, self.codebook, self.R)
        self._cb3_cache[name] = wq.detach().to("cpu")
        del w0
        return wq

    def _weight_for(self, name: str, kind: str) -> torch.Tensor:
        if kind == "bf16":
            return self.snap[name].to(self.device)
        if kind == "int4":
            return m355.fake_quant_per_group(self.snap[name].to(self.device), 4, self.group_size, self.scheme)
        if kind == "cb3":
            return self._cb3(name)
        raise ValueError(kind)

    def set_module(self, name: str, kind: str) -> None:
        if self.state[name] == kind:
            return
        with torch.no_grad():
            wq = self._weight_for(name, kind)
            self.mods[name].weight.data.copy_(wq.to(self.mods[name].weight.dtype))
        self.state[name] = kind

    def set_config(self, cb3_names: set[str], base: str = "int4") -> None:
        for name in self.snap:
            self.set_module(name, "cb3" if name in cb3_names else base)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def set_all(self, kind: str) -> None:
        for name in self.snap:
            self.set_module(name, kind)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# Checkpoint I/O (atomic).
# --------------------------------------------------------------------------- #
def load_ckpt(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {"anchors": {}, "modules": {}, "meta": {}}


def save_ckpt(path: Path, ckpt: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ckpt, default=float))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Sweep phase.
# --------------------------------------------------------------------------- #
def run_anchors(qm: QuantModel, records, max_records, greedy_prompts, greedy_tokens,
                ckpt: dict[str, Any], ckpt_path: Path) -> None:
    anchors = ckpt["anchors"]

    def need(k):
        return k not in anchors

    if need("bf16"):
        qm.set_all("bf16")
        ppl = m355.measure_ppl(qm.model, records, qm.device, max_records)
        g = [m355.greedy_decode(qm.model, p, greedy_tokens, qm.device) for p in greedy_prompts]
        anchors["bf16"] = {"ppl": ppl["ppl"], "nll": ppl["neg_log_likelihood"],
                           "num_tokens": ppl["num_tokens"], "greedy": g}
        save_ckpt(ckpt_path, ckpt)
        print(f"[mix] anchor bf16: PPL={ppl['ppl']:.4f}", flush=True)

    if need("int4"):
        qm.set_all("int4")
        ppl = m355.measure_ppl(qm.model, records, qm.device, max_records)
        g = [m355.greedy_decode(qm.model, p, greedy_tokens, qm.device) for p in greedy_prompts]
        anchors["int4"] = {"ppl": ppl["ppl"], "nll": ppl["neg_log_likelihood"],
                           "num_tokens": ppl["num_tokens"], "greedy": g}
        save_ckpt(ckpt_path, ckpt)
        print(f"[mix] anchor int4 (gate denom): PPL={ppl['ppl']:.6f}", flush=True)

    if need("uniform_cb3"):
        qm.set_all("cb3")
        ppl = m355.measure_ppl(qm.model, records, qm.device, max_records)
        g = [m355.greedy_decode(qm.model, p, greedy_tokens, qm.device) for p in greedy_prompts]
        anchors["uniform_cb3"] = {"ppl": ppl["ppl"], "nll": ppl["neg_log_likelihood"],
                                  "num_tokens": ppl["num_tokens"], "greedy": g}
        save_ckpt(ckpt_path, ckpt)
        gt = gate_transfer_ppl(ppl["ppl"], anchors["int4"]["ppl"])
        print(f"[mix] anchor uniform_cb3: PPL={ppl['ppl']:.4f} gate={gt:.4f} "
              f"(rel {100*(ppl['ppl']/anchors['int4']['ppl']-1):.2f}%)", flush=True)


def run_sweep(qm: QuantModel, records, max_records, ckpt: dict[str, Any],
              ckpt_path: Path, order: list[str], param_counts: dict[str, int],
              t_start: float, max_seconds: float) -> bool:
    """Per-module cb3 sensitivity. Returns True if the sweep is complete."""
    int4 = ckpt["anchors"]["int4"]
    nll_int4 = int4["nll"]
    ppl_int4 = int4["ppl"]
    tok = int4["num_tokens"]
    qm.set_all("int4")  # baseline state for the whole sweep
    done = 0
    for name in order:
        if name in ckpt["modules"]:
            continue
        if time.time() - t_start > max_seconds:
            print(f"[mix] budget {max_seconds}s hit; {len(ckpt['modules'])}/{len(order)} modules done; "
                  f"stopping cleanly (resume by re-running)", flush=True)
            return False
        layer, proj = parse_module(name)
        qm.set_module(name, "cb3")
        ppl = m355.measure_ppl(qm.model, records, qm.device, max_records)
        qm.set_module(name, "int4")
        d_nll = ppl["neg_log_likelihood"] - nll_int4
        d_ppl = ppl["ppl"] - ppl_int4
        gt = gate_transfer_ppl(ppl["ppl"], ppl_int4)
        ckpt["modules"][name] = {
            "layer": layer, "proj": proj, "params": param_counts[name],
            "cb3_ppl": ppl["ppl"], "cb3_nll": ppl["neg_log_likelihood"],
            "delta_nll": d_nll, "delta_ppl": d_ppl, "gate_transfer_ppl": gt,
        }
        save_ckpt(ckpt_path, ckpt)
        done += 1
        if done <= 3 or done % 20 == 0:
            print(f"[mix] sweep {len(ckpt['modules'])}/{len(order)} {name.split('language_model.')[-1]}: "
                  f"PPL={ppl['ppl']:.4f} dNLL={d_nll:+.2f} gate={gt:.4f}", flush=True)
    print(f"[mix] sweep COMPLETE: {len(ckpt['modules'])}/{len(order)} modules", flush=True)
    return True


# --------------------------------------------------------------------------- #
# Finalize: per-proj aggregates, allocation, ceiling-lift, greedy, GO.
# --------------------------------------------------------------------------- #
def avg_bpw_for(cb3_names: set[str], param_counts: dict[str, int]) -> float:
    total = sum(param_counts.values())
    s = sum(param_counts[n] * (CB3_BPW if n in cb3_names else INT4_BPW) for n in param_counts)
    return s / total


def additive_pred_ppl(cb3_names: set[str], modules: dict[str, Any],
                      nll_int4: float, tok: int) -> float:
    pred_nll = nll_int4 + sum(modules[n]["delta_nll"] for n in cb3_names)
    return math.exp(pred_nll / tok)


def finalize(qm: QuantModel, records, max_records, ckpt: dict[str, Any],
             param_counts: dict[str, int], greedy_prompts, greedy_tokens,
             args) -> dict[str, Any]:
    anchors = ckpt["anchors"]
    modules = ckpt["modules"]
    nll_int4 = anchors["int4"]["nll"]
    ppl_int4 = anchors["int4"]["ppl"]
    tok = anchors["int4"]["num_tokens"]
    names = list(modules)                       # universe = swept modules (all 258 in a full run)
    param_counts = {n: param_counts[n] for n in names}
    total_params = sum(param_counts.values())

    # ---- 1. sensitivity distribution + concentration ---------------------- #
    deltas = sorted(((modules[n]["delta_nll"], n) for n in names))  # ascending
    order_asc = [n for _, n in deltas]
    dvals = [d for d, _ in deltas]
    pos_total = sum(max(d, 0.0) for d in dvals)
    n_dec = max(1, len(names) // 10)
    top_decile = sorted(dvals, reverse=True)[:n_dec]
    top_decile_share = (sum(max(d, 0.0) for d in top_decile) / pos_total) if pos_total > 0 else float("nan")
    # Gini of the (clamped-positive) sensitivity mass.
    pos = sorted(max(d, 0.0) for d in dvals)
    n = len(pos)
    cum = sum((i + 1) * v for i, v in enumerate(pos))
    gini = (2 * cum / (n * sum(pos)) - (n + 1) / n) if sum(pos) > 0 else float("nan")
    sensitivity_is_concentrated = bool(not math.isnan(top_decile_share) and top_decile_share >= 0.40)

    # per-proj and per-layer marginal aggregates (from the per-module deltas)
    by_proj: dict[str, dict[str, float]] = {}
    by_layer: dict[int, dict[str, float]] = {}
    for n_, m in modules.items():
        by_proj.setdefault(m["proj"], {"delta_nll": 0.0, "params": 0, "count": 0})
        by_proj[m["proj"]]["delta_nll"] += m["delta_nll"]
        by_proj[m["proj"]]["params"] += m["params"]
        by_proj[m["proj"]]["count"] += 1
        by_layer.setdefault(m["layer"], {"delta_nll": 0.0, "count": 0})
        by_layer[m["layer"]]["delta_nll"] += m["delta_nll"]
        by_layer[m["layer"]]["count"] += 1

    # ---- per-proj-type aggregate MEASUREMENT (additivity cross-check) ------ #
    proj_aggr_measured: dict[str, Any] = {}
    for proj in sorted(by_proj):
        cb3 = {n_ for n_ in names if modules[n_]["proj"] == proj}
        qm.set_config(cb3)
        ppl = m355.measure_ppl(qm.model, records, qm.device, max_records)
        measured_gt = gate_transfer_ppl(ppl["ppl"], ppl_int4)
        pred_ppl = additive_pred_ppl(cb3, modules, nll_int4, tok)
        proj_aggr_measured[proj] = {
            "n_modules": len(cb3), "params": sum(param_counts[n_] for n_ in cb3),
            "measured_ppl": ppl["ppl"], "measured_gate_ppl": measured_gt,
            "additive_pred_ppl": pred_ppl,
            "additive_pred_gate_ppl": gate_transfer_ppl(pred_ppl, ppl_int4),
            "additivity_gap_gate_ppl": measured_gt - gate_transfer_ppl(pred_ppl, ppl_int4),
        }
        print(f"[mix] proj-aggr {proj:>10}: measured gate={measured_gt:.4f} "
              f"additive={gate_transfer_ppl(pred_ppl, ppl_int4):.4f} "
              f"gap={proj_aggr_measured[proj]['additivity_gap_gate_ppl']:+.4f}", flush=True)

    # ---- 2. greedy allocation under the headroom (ascending sensitivity) --- #
    # additive-predicted gate-transfer PPL at every cumulative cutoff k.
    cum_nll = nll_int4
    pred_gate_curve = []
    for k, n_ in enumerate(order_asc, start=1):
        cum_nll += modules[n_]["delta_nll"]
        pred_gate_curve.append(gate_transfer_ppl(math.exp(cum_nll / tok), ppl_int4))
    k_pred = 0
    for k in range(len(order_asc), 0, -1):
        if pred_gate_curve[k - 1] <= PPL_GATE:
            k_pred = k
            break

    # MEASURE the allocation curve at decile cutoffs + refine around the crossing.
    measured_cache: dict[int, dict[str, Any]] = {0: {
        "k": 0, "measured_ppl": ppl_int4, "measured_gate_ppl": SERVED_INT4_PPL_SPEC,
        "additive_pred_gate_ppl": SERVED_INT4_PPL_SPEC,
    }}

    def measure_k(k: int) -> dict[str, Any]:
        if k in measured_cache:
            return measured_cache[k]
        cb3 = set(order_asc[:k])
        qm.set_config(cb3)
        ppl = m355.measure_ppl(qm.model, records, qm.device, max_records)
        gt = gate_transfer_ppl(ppl["ppl"], ppl_int4)
        pred = additive_pred_ppl(cb3, modules, nll_int4, tok)
        rec = {
            "k": k, "frac_modules": k / len(names),
            "frac_params": sum(param_counts[n_] for n_ in cb3) / total_params,
            "avg_bpw": avg_bpw_for(cb3, param_counts),
            "measured_ppl": ppl["ppl"], "measured_gate_ppl": gt,
            "additive_pred_ppl": pred, "additive_pred_gate_ppl": gate_transfer_ppl(pred, ppl_int4),
            "additivity_gap_gate_ppl": gt - gate_transfer_ppl(pred, ppl_int4),
            "passes_gate": bool(gt <= PPL_GATE),
        }
        measured_cache[k] = rec
        print(f"[mix] alloc k={k:>3} ({100*k/len(names):4.1f}% mod, {100*rec['frac_params']:4.1f}% par, "
              f"bpw {rec['avg_bpw']:.3f}): measured gate={gt:.4f} additive={rec['additive_pred_gate_ppl']:.4f} "
              f"gap={rec['additivity_gap_gate_ppl']:+.4f} pass={rec['passes_gate']}", flush=True)
        return rec

    deciles = sorted(set(round(len(names) * f / 10) for f in range(1, 11)))
    for k in deciles:
        if k > 0:
            measure_k(k)
    # bracket-refine the measured crossing near k_pred and near the decile crossing
    candidates = set()
    last_pass = max((k for k, r in measured_cache.items() if r.get("passes_gate", k == 0)), default=0)
    first_fail = min((k for k, r in measured_cache.items() if not r.get("passes_gate", False)),
                     default=len(names))
    # binary-search the measured crossing between last_pass and first_fail
    lo, hi = last_pass, first_fail
    for _ in range(6):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        r = measure_k(mid)
        if r["passes_gate"]:
            lo = mid
        else:
            hi = mid
    # also pin the additive-predicted threshold so measured-vs-additive is comparable there
    if k_pred not in measured_cache and 0 < k_pred <= len(names):
        measure_k(k_pred)
        if measured_cache[k_pred]["passes_gate"]:
            lo = max(lo, k_pred)

    k_star = max((k for k, r in measured_cache.items() if r.get("passes_gate", k == 0)), default=0)
    achievable = measured_cache[k_star]
    achievable_cb3 = set(order_asc[:k_star])
    achievable_avg_bpw = avg_bpw_for(achievable_cb3, param_counts) if k_star > 0 else INT4_BPW

    # ---- byte-aware (knapsack-greedy) allocation: maximize bytes shrunk per --- #
    # ---- unit NLL cost. Secondary; the real objective is body_read_shrink.   --- #
    bytes_saved = {n_: param_counts[n_] * (INT4_BPW - CB3_BPW) for n_ in names}
    eps = 1e-9
    ratio_order = sorted(
        names,
        key=lambda n_: (modules[n_]["delta_nll"] > 0,
                        modules[n_]["delta_nll"] / max(bytes_saved[n_], eps)),
    )  # free (delta<=0) first, then smallest NLL-cost-per-byte
    cum_nll_b = nll_int4
    ba_pred_curve = []
    for n_ in ratio_order:
        cum_nll_b += modules[n_]["delta_nll"]
        ba_pred_curve.append(gate_transfer_ppl(math.exp(cum_nll_b / tok), ppl_int4))
    k_ba_pred = 0
    for k in range(len(ratio_order), 0, -1):
        if ba_pred_curve[k - 1] <= PPL_GATE:
            k_ba_pred = k
            break
    byte_aware = None
    if k_ba_pred > 0:
        cb3 = set(ratio_order[:k_ba_pred])
        qm.set_config(cb3)
        ppl = m355.measure_ppl(qm.model, records, qm.device, max_records)
        gt = gate_transfer_ppl(ppl["ppl"], ppl_int4)
        # shrink toward passing if additive over-optimistic
        while gt > PPL_GATE and k_ba_pred > 1:
            k_ba_pred -= max(1, len(ratio_order) // 40)
            cb3 = set(ratio_order[:k_ba_pred])
            qm.set_config(cb3)
            ppl = m355.measure_ppl(qm.model, records, qm.device, max_records)
            gt = gate_transfer_ppl(ppl["ppl"], ppl_int4)
        pred = additive_pred_ppl(cb3, modules, nll_int4, tok)
        byte_aware = {
            "k": k_ba_pred, "frac_modules": k_ba_pred / len(names),
            "frac_params": sum(param_counts[n_] for n_ in cb3) / total_params,
            "avg_bpw": avg_bpw_for(cb3, param_counts),
            "measured_ppl": ppl["ppl"], "measured_gate_ppl": gt,
            "additive_pred_gate_ppl": gate_transfer_ppl(pred, ppl_int4),
            "passes_gate": bool(gt <= PPL_GATE),
        }
        print(f"[mix] byte-aware k={k_ba_pred} ({100*byte_aware['frac_params']:.1f}% par, "
              f"bpw {byte_aware['avg_bpw']:.3f}): gate={gt:.4f} pass={byte_aware['passes_gate']}", flush=True)

    # ---- 3. ceiling-lift translation (ascending-sensitivity achievable) ---- #
    cl = ceiling_lift(achievable_avg_bpw)
    cl_byteaware = ceiling_lift(byte_aware["avg_bpw"]) if (byte_aware and byte_aware["passes_gate"]) else None

    # ---- 4. greedy-identity sanity on the achievable mixed config ---------- #
    mixed_greedy_frac_mismatch = float("nan")
    if k_star > 0:
        qm.set_config(achievable_cb3)
        g_mixed = [m355.greedy_decode(qm.model, p, greedy_tokens, qm.device) for p in greedy_prompts]
        ref = anchors["int4"]["greedy"]
        per = [m355.divergence(r, c) for r, c in zip(ref, g_mixed)]
        tot_mis = sum(d["num_mismatched"] for d in per)
        tot_cmp = sum(d["n_compared"] for d in per)
        mixed_greedy_frac_mismatch = tot_mis / tot_cmp if tot_cmp else 0.0
    # context floors
    bf16_g = anchors["bf16"]["greedy"]
    int4_g = anchors["int4"]["greedy"]
    cb3_g = anchors["uniform_cb3"]["greedy"]
    def frac_mis(a, b):
        per = [m355.divergence(r, c) for r, c in zip(a, b)]
        tm = sum(d["num_mismatched"] for d in per); tc = sum(d["n_compared"] for d in per)
        return tm / tc if tc else 0.0
    bf16_drift = frac_mis(int4_g, bf16_g)
    cb3_drift = frac_mis(int4_g, cb3_g)

    # ---- single GO/NO-GO --------------------------------------------------- #
    pred_lift_tps = cl["pred_nonspec_ceiling_delta_tps"]  # >0 iff any cb3 fraction
    achievable_under_bpw = bool(achievable_avg_bpw < INT4_BPW)
    mixed_precision_ceiling_lift_go = bool(achievable_under_bpw and pred_lift_tps > 0)

    res = {
        "config": {
            "base_model": args.base_model, "scheme": args.scheme, "group_size": args.group_size,
            "vq_dim": args.vq_dim, "seed": args.seed, "n_modules": len(names),
            "max_records": max_records or len(records), "num_tokens": tok,
            "greedy_prompts": len(greedy_prompts), "greedy_tokens": greedy_tokens,
            "ppl_gate": PPL_GATE, "int4_localproxy_ppl": ppl_int4,
            "served_int4_ppl_spec_anchor": SERVED_INT4_PPL_SPEC,
            "strict_nonspec_floor_tps": STRICT_NONSPEC_FLOOR_TPS,
            "spec_ceiling_lambda1": SPEC_CEILING_LAMBDA1, "private_gap_factor": PRIVATE_GAP_FACTOR,
            "int4_bpw": INT4_BPW, "cb3_bpw": CB3_BPW, "device": qm.device,
        },
        "anchors": {
            "bf16_ppl": anchors["bf16"]["ppl"], "int4_ppl": ppl_int4,
            "uniform_cb3_ppl": anchors["uniform_cb3"]["ppl"],
            "uniform_cb3_gate_ppl": gate_transfer_ppl(anchors["uniform_cb3"]["ppl"], ppl_int4),
            "uniform_cb3_rel_increase": anchors["uniform_cb3"]["ppl"] / ppl_int4 - 1.0,
            "uniform_cb3_pred_tps_nonspec": STRICT_NONSPEC_FLOOR_TPS * tps_lift_factor(CB3_BPW / INT4_BPW),
        },
        "sensitivity": {
            "by_module": modules,
            "order_ascending": order_asc,
            "delta_nll_min": dvals[0], "delta_nll_max": dvals[-1],
            "delta_nll_median": dvals[len(dvals) // 2],
            "n_modules_improving": int(sum(1 for d in dvals if d <= 0)),
            "top_decile_share": top_decile_share, "gini": gini,
            "sensitivity_is_concentrated": sensitivity_is_concentrated,
            "by_proj_marginal": by_proj, "by_layer_marginal": by_layer,
            "by_proj_measured": proj_aggr_measured,
        },
        "allocation": {
            "k_star": k_star, "k_additive_pred": k_pred,
            "frac_modules_3bit_within_headroom": k_star / len(names),
            "frac_params_3bit_within_headroom": (
                sum(param_counts[n_] for n_ in achievable_cb3) / total_params) if k_star else 0.0,
            "achievable_avg_bpw": achievable_avg_bpw,
            "mixed_config_measured_gate_ppl": achievable["measured_gate_ppl"],
            "mixed_config_additive_pred_gate_ppl": achievable.get("additive_pred_gate_ppl"),
            "additivity_gap": achievable.get("additivity_gap_gate_ppl"),
            "measured_curve": [measured_cache[k] for k in sorted(measured_cache)],
            "byte_aware": byte_aware,
        },
        "ceiling_lift": cl,
        "ceiling_lift_byteaware": cl_byteaware,
        "greedy_identity": {
            "ref": "int4", "mixed_greedy_frac_mismatch": mixed_greedy_frac_mismatch,
            "bf16_drift_vs_int4": bf16_drift, "uniform_cb3_drift_vs_int4": cb3_drift,
            "mixed_worse_than_uniform_cb3": bool(
                not math.isnan(mixed_greedy_frac_mismatch) and mixed_greedy_frac_mismatch > cb3_drift + 0.02),
        },
        "go": {
            "mixed_precision_ceiling_lift_go": mixed_precision_ceiling_lift_go,
            "achievable_avg_bpw_under_int4": achievable_under_bpw,
            "pred_ceiling_lift_tps_positive": bool(pred_lift_tps > 0),
            "spec_clears_private_500": cl["spec_clears_private_500"],
            "spec_beats_360_mark": cl["spec_beats_360_mark"],
        },
    }
    return res


def print_report(res: dict[str, Any]) -> None:
    cfg = res["config"]; a = res["anchors"]; s = res["sensitivity"]
    al = res["allocation"]; cl = res["ceiling_lift"]; go = res["go"]
    print("\n" + "=" * 104, flush=True)
    print("MIXED-PRECISION SUB-INT4 BODY-SHRINK (PR #372) — sensitivity-weighted bit allocation", flush=True)
    print("=" * 104, flush=True)
    print(f"  base={cfg['base_model']} g{cfg['group_size']} vq{cfg['vq_dim']} seed{cfg['seed']} "
          f"records={cfg['max_records']} tok={cfg['num_tokens']} modules={cfg['n_modules']}", flush=True)
    print(f"  int4 local-proxy PPL={a['int4_ppl']:.6f}  gate denom; local budget PPL<="
          f"{a['int4_ppl']*PPL_GATE/SERVED_INT4_PPL_SPEC:.4f}  (gate_PPL=2.3772*PPL/int4 <= 2.42)", flush=True)
    print(f"  uniform cb3 PPL={a['uniform_cb3_ppl']:.4f} gate={a['uniform_cb3_gate_ppl']:.4f} "
          f"(rel {100*a['uniform_cb3_rel_increase']:+.2f}%) pred_tps_nonspec={a['uniform_cb3_pred_tps_nonspec']:.2f}",
          flush=True)
    print("-" * 104, flush=True)
    print(f"  SENSITIVITY: dNLL min={s['delta_nll_min']:+.3f} median={s['delta_nll_median']:+.3f} "
          f"max={s['delta_nll_max']:+.3f}  improving={s['n_modules_improving']}", flush=True)
    print(f"    top-decile share={s['top_decile_share']:.3f} gini={s['gini']:.3f} "
          f"-> sensitivity_is_concentrated={s['sensitivity_is_concentrated']}", flush=True)
    print("    per-proj measured gate_ppl (all-of-proj at cb3, rest int4):", flush=True)
    for proj in sorted(s["by_proj_measured"]):
        pm = s["by_proj_measured"][proj]
        print(f"      {proj:>10}: gate={pm['measured_gate_ppl']:.4f} (additive {pm['additive_pred_gate_ppl']:.4f} "
              f"gap {pm['additivity_gap_gate_ppl']:+.4f}) n={pm['n_modules']} par={pm['params']/1e6:.0f}M", flush=True)
    print("-" * 104, flush=True)
    print(f"  ALLOCATION (ascending sensitivity, MEASURED gate <= 2.42):", flush=True)
    print(f"    k*={al['k_star']}/{cfg['n_modules']}  frac_modules_3bit={al['frac_modules_3bit_within_headroom']:.4f} "
          f"frac_params_3bit={al['frac_params_3bit_within_headroom']:.4f}", flush=True)
    print(f"    achievable_avg_bpw={al['achievable_avg_bpw']:.4f}  measured_gate_ppl="
          f"{al['mixed_config_measured_gate_ppl']:.4f}  additive_pred={al['mixed_config_additive_pred_gate_ppl']} "
          f"additivity_gap={al['additivity_gap']}", flush=True)
    if al["byte_aware"]:
        ba = al["byte_aware"]
        print(f"    [byte-aware] k={ba['k']} frac_params_3bit={ba['frac_params']:.4f} bpw={ba['avg_bpw']:.4f} "
              f"gate={ba['measured_gate_ppl']:.4f} pass={ba['passes_gate']}", flush=True)
    print("-" * 104, flush=True)
    print(f"  CEILING-LIFT (ascending achievable, bpw {cl['avg_bpw']:.4f}, body_read_shrink="
          f"{cl['body_read_shrink_frac']:.4f}):", flush=True)
    print(f"    spec: {SPEC_CEILING_LAMBDA1:.3f} -> {cl['pred_ceiling_lift_tps_spec']:.2f} "
          f"(+{cl['pred_spec_ceiling_delta_tps']:.2f})  private*0.957={cl['spec_private_after_gap']:.2f} "
          f">=500? {cl['spec_clears_private_500']}  beats +33.62? {cl['spec_beats_360_mark']}", flush=True)
    print(f"    nonspec: {STRICT_NONSPEC_FLOOR_TPS:.2f} -> {cl['pred_ceiling_lift_tps_nonspec']:.2f} "
          f"(+{cl['pred_nonspec_ceiling_delta_tps']:.2f})", flush=True)
    gi = res["greedy_identity"]
    print(f"  GREEDY (report-only): mixed drift vs int4={gi['mixed_greedy_frac_mismatch']} "
          f"(bf16 {gi['bf16_drift_vs_int4']:.3f}, uniform_cb3 {gi['uniform_cb3_drift_vs_int4']:.3f}; "
          f"worse_than_uniform? {gi['mixed_worse_than_uniform_cb3']})", flush=True)
    print("-" * 104, flush=True)
    print(f"  >>> mixed_precision_ceiling_lift_go = {go['mixed_precision_ceiling_lift_go']}  "
          f"(avg_bpw<4.125: {go['achievable_avg_bpw_under_int4']}, lift>0: {go['pred_ceiling_lift_tps_positive']})",
          flush=True)
    print("=" * 104, flush=True)


# --------------------------------------------------------------------------- #
# W&B.
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict[str, Any]) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        import wandb  # noqa: F401
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[mix] wandb unavailable: {exc}", flush=True)
        return
    res = payload["result"]
    cfg = res["config"]
    run = init_wandb_run(
        job_type="validity-gate", agent="lawine", name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "mixed-precision-bit-allocation", "sub-int4", "codebook-quant",
              "sensitivity-weighted", "non-spec-frontier", "ceiling-lift", "pr-372"],
        config={k: v for k, v in cfg.items()},
    )
    if run is None:
        print("[mix] wandb: no run — skipping", flush=True)
        return
    a = res["anchors"]; s = res["sensitivity"]; al = res["allocation"]
    cl = res["ceiling_lift"]; go = res["go"]; gi = res["greedy_identity"]
    summary: dict[str, Any] = {
        "mixed_precision_ceiling_lift_go": int(bool(go["mixed_precision_ceiling_lift_go"])),
        "achievable_avg_bpw": al["achievable_avg_bpw"],
        "frac_modules_3bit_within_headroom": al["frac_modules_3bit_within_headroom"],
        "frac_params_3bit_within_headroom": al["frac_params_3bit_within_headroom"],
        "body_read_shrink_frac": cl["body_read_shrink_frac"],
        "mixed_config_measured_gate_ppl": al["mixed_config_measured_gate_ppl"],
        "additivity_gap": al["additivity_gap"],
        "sensitivity_is_concentrated": int(bool(s["sensitivity_is_concentrated"])),
        "top_decile_share": s["top_decile_share"], "gini": s["gini"],
        "pred_ceiling_lift_tps_spec": cl["pred_ceiling_lift_tps_spec"],
        "pred_ceiling_lift_tps_nonspec": cl["pred_ceiling_lift_tps_nonspec"],
        "pred_spec_ceiling_delta_tps": cl["pred_spec_ceiling_delta_tps"],
        "spec_private_after_gap": cl["spec_private_after_gap"],
        "spec_clears_private_500": int(bool(cl["spec_clears_private_500"])),
        "spec_beats_360_mark": int(bool(cl["spec_beats_360_mark"])),
        "int4_localproxy_ppl": a["int4_ppl"], "uniform_cb3_ppl": a["uniform_cb3_ppl"],
        "uniform_cb3_gate_ppl": a["uniform_cb3_gate_ppl"],
        "k_star": al["k_star"], "k_additive_pred": al["k_additive_pred"],
        "mixed_greedy_frac_mismatch": gi["mixed_greedy_frac_mismatch"],
        "n_modules_improving": s["n_modules_improving"],
        "peak_mem_mib": payload["peak_mem_mib"],
    }
    if al["byte_aware"]:
        summary["byteaware_frac_params_3bit"] = al["byte_aware"]["frac_params"]
        summary["byteaware_avg_bpw"] = al["byte_aware"]["avg_bpw"]
        if res["ceiling_lift_byteaware"]:
            summary["byteaware_pred_ceiling_lift_tps_spec"] = \
                res["ceiling_lift_byteaware"]["pred_ceiling_lift_tps_spec"]
            summary["byteaware_body_read_shrink_frac"] = \
                res["ceiling_lift_byteaware"]["body_read_shrink_frac"]
    for proj, pm in s["by_proj_measured"].items():
        summary[f"proj_gate_ppl_{proj}"] = pm["measured_gate_ppl"]
        summary[f"proj_additivity_gap_{proj}"] = pm["additivity_gap_gate_ppl"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v))}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="mixed_precision_subint4_measured", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[mix] wandb logged: {len(summary)} metrics", flush=True)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", "--base_model", dest="base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--scheme", choices=["asym", "sym"], default="asym")
    ap.add_argument("--group-size", "--group_size", dest="group_size", type=int, default=128)
    ap.add_argument("--vq-dim", "--vq_dim", dest="vq_dim", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ppl-dataset", "--ppl_dataset", dest="ppl_dataset", default=str(DEFAULT_PPL_DATASET))
    ap.add_argument("--max-records", "--max_records", dest="max_records", type=int, default=0,
                    help="0 = all (128) records — the official methodology")
    ap.add_argument("--greedy-prompts", "--greedy_prompts", dest="greedy_prompts", type=int, default=8)
    ap.add_argument("--greedy-tokens", "--greedy_tokens", dest="greedy_tokens", type=int, default=128)
    ap.add_argument("--max-seconds", "--max_seconds", dest="max_seconds", type=float, default=4500.0,
                    help="wall-clock budget for THIS invocation's sweep (stops cleanly under the 90-min cap)")
    ap.add_argument("--limit-modules", dest="limit_modules", type=int, default=0,
                    help="DEBUG: only sweep the first N modules (smoke test)")
    ap.add_argument("--out-dir", dest="out_dir", default=str(HERE))
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="mixed-precision-bit-allocation")
    args = ap.parse_args(argv)

    t_start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[mix] WARNING CUDA not available; set CUDA_VISIBLE_DEVICES=0", flush=True)
    records = m355.read_ppl_records(Path(args.ppl_dataset))
    max_records = args.max_records if args.max_records > 0 else None

    model, tok = m355.load_model(args.base_model, device)
    print(f"[mix] model loaded; GPU {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
    snap = m355.snapshot_body(model)
    param_counts = {name: int(w.numel()) for name, w in snap.items()}
    print(f"[mix] {len(snap)} body linears, {sum(param_counts.values())/1e9:.3f}B params", flush=True)

    R = mcb.rht_matrix(args.group_size, device, seed=args.seed)
    codebook = mcb.build_gaussian_codebook(3, args.vq_dim, device, seed=args.seed)
    qm = QuantModel(model, snap, args.group_size, args.scheme, args.vq_dim, codebook, R, device)

    greedy_prompts = [records[i]["ids"][: m355._prompt_len(records[i])]
                      for i in range(min(args.greedy_prompts, len(records)))]

    ckpt_path = Path(args.out_dir) / CKPT_NAME
    ckpt = load_ckpt(ckpt_path)
    ckpt.setdefault("anchors", {})
    ckpt.setdefault("modules", {})
    ckpt["meta"] = {"base_model": args.base_model, "group_size": args.group_size,
                    "vq_dim": args.vq_dim, "seed": args.seed, "scheme": args.scheme,
                    "max_records": max_records or len(records)}

    run_anchors(qm, records, max_records, greedy_prompts, args.greedy_tokens, ckpt, ckpt_path)

    order = list(snap)
    if args.limit_modules > 0:
        order = order[: args.limit_modules]
    complete = run_sweep(qm, records, max_records, ckpt, ckpt_path, order, param_counts,
                         t_start, args.max_seconds)

    if not complete:
        done = len(ckpt["modules"])
        print(f"[mix] PARTIAL: {done}/{len(order)} modules swept; re-run to continue.", flush=True)
        return 0

    # The sweep is complete. If it only just finished near the wall-clock deadline,
    # defer finalize (~15 min of joint-config evals) to a fresh invocation so it is
    # not killed mid-run by the 90-min cap. On a cached-complete resume the elapsed
    # time is tiny (model load only), so this guard does not trip and finalize runs.
    if time.time() - t_start > args.max_seconds - FINALIZE_RESERVE_S:
        print(f"[mix] sweep COMPLETE but only "
              f"{args.max_seconds - (time.time() - t_start):.0f}s budget left "
              f"(< {FINALIZE_RESERVE_S:.0f}s reserve); deferring FINALIZE to next invocation.",
              flush=True)
        return 0

    print("[mix] FINALIZE", flush=True)
    res = finalize(qm, records, max_records, ckpt, param_counts, greedy_prompts,
                   args.greedy_tokens, args)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_mib = (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0
    payload = {
        "created_at": created_at, "pr": 372, "agent": "lawine",
        "kind": "mixed-precision-subint4-measured",
        "elapsed_s": round(time.time() - t_start, 1), "peak_mem_mib": round(peak_mib, 1),
        "result": res,
    }
    print_report(res)
    out_path = Path(args.out_dir) / RESULTS_NAME
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    print(f"[mix] wrote {out_path} (elapsed {payload['elapsed_s']}s, peak {payload['peak_mem_mib']} MiB)",
          flush=True)
    maybe_log_wandb(args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
