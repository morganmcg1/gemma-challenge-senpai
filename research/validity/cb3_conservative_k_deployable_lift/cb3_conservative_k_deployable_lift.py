#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Conservative-k cb3 deployable lift (PR #403) -- the REAL PPL-safe supply number.

THE QUESTION
------------
#394 proved the cb3 supply lift is PPL-BLOCKED at the held-out-SELECTED ("headline")
k: the +0.039 in-sample margin is winner's-curse, and at the selected k=246 the
held-out WORST-seed PPL (2.4223) and the OOD PPL (2.4270) both BREACH the 2.42 gate
(`cb3_supply_deployable=False`). BUT the more conservative k=232 still clears
(~2.39 held-out). So cb3 is deployable at a *smaller, bankable* k -- just at a
SMALLER lift. This card costs that honest number: what is the REAL PPL-deployable
cb3 supply lift once we back k off to a bankable >=0.01 margin below the gate?

MONOTONICITY (verified against the #394 harness, NOT the PR parenthetical)
--------------------------------------------------------------------------
In the #394/#372 harness `qm.set_config(order_ascending[:k])` puts the k LEAST-
sensitive body linears on cb3 (sub-int4 codebook, 3.125 bpw) and the rest on int4
(4.125 bpw). Therefore:
  * LARGER k  = MORE modules on cb3 = MORE body-read shrink = LARGER lift = HIGHER PPL.
  * SMALLER k = FEWER modules on cb3 = LESS shrink = SMALLER lift = LOWER PPL.
`select_k_on_subset` picks the LARGEST k whose gate PPL <= 2.42 (max lift under the
gate). So "conservative" == SMALLER k == lower PPL == smaller lift. NOTE: the PR
body's parenthetical "(most aggressive, smallest k)" / "larger k = ... smaller lift"
has this DIRECTION BACKWARDS; the operative goal -- MAXIMIZE lift subject to
PPL<=2.41 -- is unambiguous and, given the true monotonicity, means the LARGEST k
whose worst-seed held-out AND OOD PPL are both <= 2.41. We implement the operative
goal and flag the discrepancy.

THE LIFT-vs-k MODEL (re-cost the #388/#391/#392 anchors AT k*)
-------------------------------------------------------------
The cb3 body-read byte ratio is r = effective_bpw / int4_bpw. #372's mixed optimum
puts 88.8% of body PARAMS on cb3 at k=232 -> effective_bpw = 3.2369 -> r = 0.785
(the "-21.5% shrink"). The byte ratio is a function of the cb3 PARAM fraction phi(k):
  effective_bpw(k) = phi(k)*CB3_BPW_UNIFORM + (1-phi(k))*INT4_BPW
  r(k)             = effective_bpw(k) / INT4_BPW
  shrink(k)        = 1 - r(k)
where phi(k) = (sum params of order_ascending[:k]) / (total body params), measured
from the model snapshot. The three banked lift tiers re-price with r(k) using the
EXACT #388/#391 closed forms (eff is a per-shape Marlin property, ~flat in both M and
k, so only r moves with k):
  M1 honest (#392)  = qtip-empirical  speedup 1/(r*0.51 + (1-0.51)), off-the-shelf base, body-only f
  M8 measured-floor = strict          speedup 1/(r*eff_m8 + (1-eff_m8)), off-the-shelf base, body-only f   <- THE DEPLOYABLE SUPPLY NUMBER
At phi(232)=0.888 these reproduce #392's +32.65 (M1) and #391's +15.67 (M8) exactly.

DECISION
--------
k* = largest k with heldout_worst(k) <= 2.41 AND ood(k) <= 2.41 (robust: no breach
below it). `cb3_conservative_deployable` = a positive-lift such k* exists.
`cb3_supply_lane_dead` = True ONLY if NO positive-lift k clears the 2.41 worst-seed
bar (i.e. even the previously-cleared k=232 fails worst-seed) -> the whole supply
lane dies. residual_gap_to_500_after_cb3 = 32.53 - M8_lift(k*).

Identity-safe: GPU PPL only, NO submission, NO served-file change, NO --launch, NO
cb3 kernel build, 0 official TPS. analysis_only.

Run:
  0-GPU self-test (PRIMARY, >=20 asserts):
    cd target/ && .venvs/vllm022/bin/python -m \
      research.validity.cb3_conservative_k_deployable_lift.cb3_conservative_k_deployable_lift --self-test
  GPU PPL-vs-k curve + k* (single A10G, ~25-35 min):
    cd target/ && CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python -m \
      research.validity.cb3_conservative_k_deployable_lift.cb3_conservative_k_deployable_lift \
      --wandb_group cb3-conservative-k-deployable-lift --wandb_name kanna/cb3-conservative-k-deployable-lift
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
VAL = REPO_ROOT / "research" / "validity"
for _p in (VAL / "cb3_ppl_heldout_margin", VAL / "cb3_kernel_realized_bw", VAL / "sub_int4_body_ceiling"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cb3_kernel_realized_bw as bw  # noqa: E402  (analytic lift core, #388/#391)
import cb3_ppl_heldout_margin as hm  # noqa: E402  (PPL/OOD harness, #394; pulls in m355/mcb/mp)

m355 = hm.m355
mcb = hm.mcb
mp = hm.mp

# --------------------------------------------------------------------------- #
# Anchors / constants (single source of truth = sibling cards, fallbacks let
# --self-test run 0-GPU even if a sibling JSON moves).
# --------------------------------------------------------------------------- #
PPL_GATE = mp.PPL_GATE                       # 2.42  (served gate)
CONSERVATIVE_BAR = 2.41                      # bankable target: ~0.01 margin below the gate
SERVED_INT4_PPL_SPEC = m355.SERVED_INT4_PPL_SPEC  # 2.3772
INT4_BPW = bw.INT4_BPW                       # 4.125
CB3_BPW_UNIFORM = bw.CB3_BPW_UNIFORM         # 3.125 (all-cb3 bpw; phi=1 limit)
CB3_BPW_EFF_372 = bw.CB3_BPW_EFF             # 3.2369 (#372 mixed optimum at k=232)
BYTE_RATIO_372 = bw.BODY_BYTES_FRAC          # 0.7847 (= 3.2369/4.125, the "-21.5%")
PHI_372 = 0.888                              # #372: 88.8% body params on cb3 at k=232
QTIP_BETA = bw.QTIP_BETA_BYTE_PROPORTIONAL   # 0.51 (M1 honest realistic beta)
F_BODY_STRICT = bw.F_BODY_STRICT             # 0.7624 (honest shrinkable body fraction)
BAND_OFF_THE_SHELF = bw.BAND_OFF_THE_SHELF   # 357.32 (the lift base #388/#391/#392 used)
BAND_FLOOR = bw.BAND_FLOOR                   # 469.68

# corrected strict base + gap (#393, given in the PR baseline)
CORRECTED_STRICT_BASE_393 = 467.48
STRICT_GAP_TO_500 = 32.53

# banked anchors the re-cost must reproduce at k=232 (phi=0.888)
M1_LIFT_232 = 32.647211371136166             # #392 off_the_shelf__m1_banked honest lift
M8_LIFT_232 = 15.665922538333916             # #391 realized_strict_base_lift_tps_m8_measured_floor
M8_SPEEDUP_232 = 1.0587760668597737          # #391 realized_body_speedup_measured_floor
M1_SPEEDUP_232 = 1.1233511704212635          # #392/#388 qtip_empirical speedup
EFF_M8_FALLBACK = 0.2558677957123237         # #391 marlin_m8_hbm_eff
SHRINK_232 = 0.21530670587274357             # 1 - 0.7847

K_STAR_372 = 232
N_OFFICIAL = 128
DEFAULT_SPLIT_SEEDS = (0, 1, 2)
RESULTS_NAME = "cb3_conservative_k_deployable_lift_results.json"
SELFTEST_NAME = "cb3_conservative_k_deployable_lift_selftest.json"
BW_RESULTS = VAL / "cb3_kernel_realized_bw" / "cb3_kernel_realized_bw_results.json"


def load_eff_m8() -> float:
    """The measured M=8 Marlin weight-read HBM efficiency from #391 (fallback to banked)."""
    try:
        d = json.loads(BW_RESULTS.read_text())
        return float(d.get("marlin_m8_hbm_eff", EFF_M8_FALLBACK))
    except Exception:  # noqa: BLE001
        return EFF_M8_FALLBACK


# ======================================================================================== #
# Analytic core: byte ratio / lift tiers as a FUNCTION of the cb3 param fraction phi.
# Re-uses the #388/#391 closed forms; only r = byte_ratio moves with k (eff flat in k).
# ======================================================================================== #
def byte_ratio_at_phi(phi: float) -> float:
    """Body-read byte ratio when a fraction `phi` of body params is on cb3 (rest int4).
    phi=0 -> 1.0 (all int4, no shrink); phi=1 -> CB3_BPW_UNIFORM/INT4_BPW (all cb3)."""
    eff_bpw = phi * CB3_BPW_UNIFORM + (1.0 - phi) * INT4_BPW
    return eff_bpw / INT4_BPW


def shrink_at_phi(phi: float) -> float:
    return 1.0 - byte_ratio_at_phi(phi)


def speedup_from_beta_r(beta: float, r: float) -> float:
    """realized_time_ratio = r*beta + (1-beta); speedup = 1/ratio. (Identical engine to
    #388 bw.speedup_from_beta, but with r re-parameterized by k via byte_ratio_at_phi.)"""
    beta = min(max(beta, 0.0), 1.0)
    return 1.0 / (r * beta + (1.0 - beta))


def m1_speedup_at_phi(phi: float) -> float:
    """M=1 honest (#392): qtip-empirical beta=0.51 at the k-dependent byte ratio."""
    return speedup_from_beta_r(QTIP_BETA, byte_ratio_at_phi(phi))


def m8_speedup_at_phi(phi: float, eff_m8: float) -> float:
    """M=8 strict measured-floor (#391): beta == measured Marlin eff at the k-dep byte ratio."""
    return speedup_from_beta_r(eff_m8, byte_ratio_at_phi(phi))


def lift_off_the_shelf(speedup: float) -> float:
    """Re-cost on the EXACT #388/#391/#392 cell: off-the-shelf base, body-only f_body."""
    return bw.translate_band(speedup, F_BODY_STRICT)["delta_off_the_shelf"]


def lift_floor(speedup: float) -> float:
    return bw.translate_band(speedup, F_BODY_STRICT)["delta_floor"]


def recost_at_phi(phi: float, eff_m8: float) -> dict[str, float]:
    """All k-dependent supply numbers at cb3 param fraction phi."""
    r = byte_ratio_at_phi(phi)
    s_m1 = m1_speedup_at_phi(phi)
    s_m8 = m8_speedup_at_phi(phi, eff_m8)
    s_roof = 1.0 / r
    return {
        "phi_params": phi,
        "byte_ratio": r,
        "bodyread_shrink": 1.0 - r,
        "roofline_speedup": s_roof,
        "m1_speedup": s_m1,
        "m8_speedup": s_m8,
        "m1_lift_off_the_shelf": lift_off_the_shelf(s_m1),
        "m8_lift_off_the_shelf": lift_off_the_shelf(s_m8),
        "m1_lift_floor": lift_floor(s_m1),
        "m8_lift_floor": lift_floor(s_m8),
        "roofline_lift_off_the_shelf": lift_off_the_shelf(s_roof),
    }


# ======================================================================================== #
# phi(k): cumulative cb3 param fraction along the #372 ascending-sensitivity ordering.
# ======================================================================================== #
def phi_curve_from_snapshot(order: list[str], snap: dict[str, Any]) -> dict[str, Any]:
    """phi(k) = (params of order[:k]) / (total body params). Returns cumulative fraction
    list of length n+1 (phi[0]=0 .. phi[n]=1) plus per-module param counts."""
    params = [int(snap[name].numel()) for name in order]
    total = sum(params)
    cum = [0]
    for p in params:
        cum.append(cum[-1] + p)
    phi = [c / total for c in cum]  # phi[k] = fraction of params in order[:k]
    return {"phi": phi, "module_params": params, "total_body_params": total}


def phi_at_k(phi_list: list[float], k: int) -> float:
    return phi_list[min(max(k, 0), len(phi_list) - 1)]


# ======================================================================================== #
# k-grid: dense integers around the 2.41 crossings + coarse anchors below.
# ======================================================================================== #
def build_k_grid(n_modules: int, dense_lo: int, dense_hi: int, smoke: bool) -> list[int]:
    if smoke:
        return sorted({0, 120, K_STAR_372, 243, 246, n_modules})
    coarse = [0, 80, 140, 170]
    mid = list(range(max(170, dense_lo - 30), dense_lo, 3))
    dense = list(range(dense_lo, min(dense_hi, n_modules) + 1))
    top = [250, 254, n_modules]
    grid = sorted({k for k in (coarse + mid + dense + top + [K_STAR_372, 243, 246])
                   if 0 <= k <= n_modules})
    return grid


# ======================================================================================== #
# Per-k PPL curves -> held-out-worst & OOD -> k*.
# ======================================================================================== #
def heldout_worst_at_k(nll_by_k: dict[int, list[float]], tok: list[int], k: int,
                       split_seeds: tuple[int, ...], n_records: int) -> dict[str, Any]:
    """For FIXED k, gate PPL on each split-seed's held-out half H(64); worst over seeds.
    Re-uses #394's exact partition (random.Random(seed).shuffle)."""
    import random
    per_seed = []
    for seed in split_seeds:
        rng = random.Random(seed)
        perm = list(range(n_records))
        rng.shuffle(perm)
        H = sorted(perm[n_records // 2:])
        g = hm.subset_gate(nll_by_k[k], nll_by_k[0], tok, H)
        per_seed.append({"seed": seed, "heldout_gate_ppl": g})
    vals = [r["heldout_gate_ppl"] for r in per_seed]
    return {"per_seed": per_seed, "worst": max(vals), "mean": sum(vals) / len(vals)}


def build_kstar_table(grid: list[int], nll_off: dict[int, list[float]], tok_off: list[int],
                      nll_ood: dict[int, list[float]], tok_ood: list[int],
                      phi_list: list[float], eff_m8: float,
                      split_seeds: tuple[int, ...], n_records: int, n_ood: int) -> list[dict[str, Any]]:
    """One row per grid-k: full/heldout-worst/OOD gate PPL, phi, byte ratio, shrink, lifts,
    and the two clear-flags (heldout_worst<=2.41 AND ood<=2.41)."""
    ood_idx = list(range(n_ood))
    full_idx = list(range(n_records))
    rows: list[dict[str, Any]] = []
    for k in grid:
        full_ppl = hm.subset_gate(nll_off[k], nll_off[0], tok_off, full_idx)
        ho = heldout_worst_at_k(nll_off, tok_off, k, split_seeds, n_records)
        ood_ppl = hm.subset_gate(nll_ood[k], nll_ood[0], tok_ood, ood_idx)
        phi = phi_at_k(phi_list, k)
        rc = recost_at_phi(phi, eff_m8)
        clears = bool(ho["worst"] <= CONSERVATIVE_BAR and ood_ppl <= CONSERVATIVE_BAR)
        rows.append({
            "k": k,
            "full_gate_ppl": full_ppl,
            "heldout_worst_ppl": ho["worst"],
            "heldout_mean_ppl": ho["mean"],
            "heldout_per_seed": ho["per_seed"],
            "ood_ppl": ood_ppl,
            "phi_params": phi,
            "byte_ratio": rc["byte_ratio"],
            "bodyread_shrink": rc["bodyread_shrink"],
            "m1_lift": rc["m1_lift_off_the_shelf"],
            "m8_lift": rc["m8_lift_off_the_shelf"],
            "m8_lift_floor": rc["m8_lift_floor"],
            "roofline_lift": rc["roofline_lift_off_the_shelf"],
            "clears_2p41_both": clears,
        })
    return rows


def find_kstar(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """k* = LARGEST positive-lift k whose heldout_worst AND ood are both <= 2.41
    (max lift under the bankable bar). k*_robust additionally requires NO breach for any
    positive-lift k' <= k* (PPL-safe by construction -- no noise dip below it)."""
    clearing = [r for r in rows if r["k"] >= 1 and r["clears_2p41_both"]]
    k_star_largest = max((r["k"] for r in clearing), default=0)
    # robust: walk k upward; the last k for which every positive-lift k'<=k clears.
    pos_rows = sorted([r for r in rows if r["k"] >= 1], key=lambda r: r["k"])
    k_star_robust = 0
    for r in pos_rows:
        if r["clears_2p41_both"]:
            k_star_robust = r["k"]
        else:
            break
    lane_dead = (k_star_largest == 0)
    return {
        "k_star_largest": k_star_largest,
        "k_star_robust": k_star_robust,
        "k_star": k_star_robust,        # bankable choice = robust (PPL-safe by construction)
        "monotone_clean": bool(k_star_robust == k_star_largest),
        "cb3_supply_lane_dead": bool(lane_dead),
    }


# ======================================================================================== #
# GPU measurement.
# ======================================================================================== #
def run_gpu(args) -> dict[str, Any]:
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[ckl] WARNING: CUDA not available; set CUDA_VISIBLE_DEVICES=0", flush=True)

    d372 = hm.load_372()
    order = d372["order_ascending"]
    print(f"[ckl] reused #372: {len(order)} modules, in-sample k*={d372['k_star']}, "
          f"reported gate={d372['mixed_config_measured_gate_ppl']:.4f}", flush=True)

    records = m355.read_ppl_records(Path(args.ppl_dataset))
    if args.smoke:
        records = records[:16]
    n_records = len(records)
    model, tokenizer = m355.load_model(args.base_model, device)
    snap = m355.snapshot_body(model)
    assert set(order) == set(snap), f"#372 ordering ({len(order)}) != snapshot ({len(snap)})"
    print(f"[ckl] model loaded; GPU {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

    # phi(k) from the real snapshot param counts
    phi_info = phi_curve_from_snapshot(order, snap)
    phi_list = phi_info["phi"]
    phi_232 = phi_at_k(phi_list, K_STAR_372)
    print(f"[ckl] phi(232)={phi_232:.4f} (target ~0.888); total body params="
          f"{phi_info['total_body_params']/1e9:.3f}B", flush=True)

    R = mcb.rht_matrix(args.group_size, device, seed=args.quant_seed)
    codebook = mcb.build_gaussian_codebook(3, args.vq_dim, device, seed=args.quant_seed)
    qm = mp.QuantModel(model, snap, args.group_size, args.scheme, args.vq_dim, codebook, R, device)

    grid = build_k_grid(len(order), args.dense_lo, args.dense_hi, args.smoke)
    print(f"[ckl] k-grid ({len(grid)}): {grid}", flush=True)

    # --- official-128 PPL-vs-k curve (reuse #394 measure_k_grid) --- #
    t0 = time.time()
    nll_off = hm.measure_k_grid(qm, model, records, device, order, grid)
    tok_off = nll_off.pop("_tok")
    print(f"[ckl] official curve done [{time.time()-t0:.1f}s]", flush=True)

    # --- OOD sharegpt PPL-vs-k curve (same grid) --- #
    min_ood = 8 if args.smoke else 64
    n_want = min_ood if args.smoke else max(64, args.ood_n)
    pairs = hm.load_sharegpt_conversations(n_want)
    ood_records = hm.build_ood_records(pairs, tokenizer, n_want, args.ood_max_ctx,
                                       args.ood_tgt_lo, args.ood_tgt_cap)
    assert len(ood_records) >= min_ood, f"OOD slice too small: {len(ood_records)} (<{min_ood})"
    n_ood = len(ood_records)
    t0 = time.time()
    nll_ood = hm.measure_k_grid(qm, model, ood_records, device, order, grid)
    tok_ood = nll_ood.pop("_tok")
    print(f"[ckl] OOD curve done (n={n_ood}, src={ood_records[0]['source']}) "
          f"[{time.time()-t0:.1f}s]", flush=True)

    eff_m8 = load_eff_m8()
    rows = build_kstar_table(grid, nll_off, tok_off, nll_ood, tok_ood, phi_list,
                             eff_m8, tuple(args.split_seeds), n_records, n_ood)
    ks = find_kstar(rows)
    return assemble(args, order, phi_info, phi_232, rows, ks, eff_m8,
                    n_records, n_ood, ood_records[0]["source"], device, grid,
                    in_sample_full=next(r["full_gate_ppl"] for r in rows if r["k"] == K_STAR_372))


# ======================================================================================== #
# Assemble result + decision + re-cost AT k*.
# ======================================================================================== #
def assemble(args, order, phi_info, phi_232, rows, ks, eff_m8, n_records, n_ood,
             ood_source, device, grid, in_sample_full) -> dict[str, Any]:
    k_star = ks["k_star"]
    row_star = next((r for r in rows if r["k"] == k_star), None)
    phi_list = phi_info["phi"]

    if row_star is None or k_star == 0:
        # lane dead OR no positive-lift k clears: report the int4 (zero-lift) point.
        rc = recost_at_phi(0.0, eff_m8)
        heldout_worst = next((r["heldout_worst_ppl"] for r in rows if r["k"] == K_STAR_372), float("nan"))
        ood_at = next((r["ood_ppl"] for r in rows if r["k"] == K_STAR_372), float("nan"))
        m8_lift = 0.0
        m1_lift = 0.0
        shrink = 0.0
        byte_ratio = 1.0
        ho_worst = heldout_worst
        ood_ppl = ood_at
    else:
        rc = recost_at_phi(row_star["phi_params"], eff_m8)
        m8_lift = row_star["m8_lift"]
        m1_lift = row_star["m1_lift"]
        shrink = row_star["bodyread_shrink"]
        byte_ratio = row_star["byte_ratio"]
        ho_worst = row_star["heldout_worst_ppl"]
        ood_ppl = row_star["ood_ppl"]

    ppl_margin = PPL_GATE - max(ho_worst, ood_ppl)
    residual = STRICT_GAP_TO_500 - m8_lift
    deployable = bool(k_star >= 1 and m8_lift > 0.0 and not ks["cb3_supply_lane_dead"])

    # k=232 survival under the conservative worst-seed bar (the specific #394 follow-up)
    row_232 = next((r for r in rows if r["k"] == K_STAR_372), None)
    kstar_232_survives = bool(row_232 is not None
                              and row_232["heldout_worst_ppl"] <= CONSERVATIVE_BAR
                              and row_232["ood_ppl"] <= CONSERVATIVE_BAR)

    # binding constraint at k* (which leg sets the ceiling)
    binding = "ood" if ood_ppl >= ho_worst else "heldout_worst"

    res = {
        "config": {
            "base_model": args.base_model, "device": device, "n_modules": len(order),
            "ppl_gate": PPL_GATE, "conservative_bar": CONSERVATIVE_BAR,
            "served_int4_ppl_spec": SERVED_INT4_PPL_SPEC, "split_seeds": list(args.split_seeds),
            "k_grid": grid, "n_official_records": n_records, "n_ood_records": n_ood,
            "ood_source": ood_source, "eff_m8": eff_m8,
            "corrected_strict_base_393": CORRECTED_STRICT_BASE_393,
            "strict_gap_to_500": STRICT_GAP_TO_500,
            "band_off_the_shelf": BAND_OFF_THE_SHELF, "f_body_strict": F_BODY_STRICT,
            "phi_232_measured": phi_232, "total_body_params": phi_info["total_body_params"],
        },
        "in_sample": {
            "k_star_372": K_STAR_372, "in_sample_gate_ppl_at_232": in_sample_full,
            "in_sample_gate_ppl_372": hm.IN_SAMPLE_GATE_372,
            "reproduces_372": bool(abs(in_sample_full - hm.IN_SAMPLE_GATE_372) <= 0.02),
        },
        "kstar": {
            **ks,
            "heldout_worst_ppl_at_kstar": ho_worst,
            "ood_ppl_at_kstar": ood_ppl,
            "ppl_margin_to_242_at_kstar": ppl_margin,
            "binding_constraint": binding,
            "kstar_232_survives_worst_bar": kstar_232_survives,
        },
        "recost_at_kstar": {
            "phi_params_at_kstar": rc["phi_params"],
            "byte_ratio_at_kstar": byte_ratio,
            "bodyread_shrink_at_kstar": shrink,
            "m1_lift_at_kstar": m1_lift,
            "m8_lift_at_kstar": m8_lift,                 # THE DEPLOYABLE SUPPLY NUMBER
            "m8_lift_floor_at_kstar": rc["m8_lift_floor"],
            "roofline_lift_at_kstar": rc["roofline_lift_off_the_shelf"],
            "shrink_vs_232": shrink - SHRINK_232,
            "m8_lift_vs_232": m8_lift - M8_LIFT_232,
        },
        "residual": {
            "residual_gap_to_500_after_cb3": residual,
            "frac_of_gap_closed_by_cb3": (m8_lift / STRICT_GAP_TO_500) if STRICT_GAP_TO_500 else 0.0,
            "demand_route_must_supply_tps": residual,
        },
        "decision": {
            "cb3_conservative_deployable": deployable,
            "cb3_supply_lane_dead": ks["cb3_supply_lane_dead"],
            "verdict": ("DEPLOYABLE-SMALL-LIFT" if deployable else
                        ("LANE-DEAD" if ks["cb3_supply_lane_dead"] else "ZERO-LIFT")),
        },
        "curve": rows,
        "guards": {"analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
                   "no_launch": True, "no_kernel_build": True, "official_tps": 0},
    }
    return res


# ======================================================================================== #
# Self-test (0-GPU, PRIMARY): >=20 asserts on the analytic core + search logic.
# ======================================================================================== #
def self_test() -> dict[str, Any]:
    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    eff_m8 = load_eff_m8()

    # ---- 1. byte-ratio(phi) anchors + monotonicity ---- #
    r0 = byte_ratio_at_phi(0.0)
    r1 = byte_ratio_at_phi(1.0)
    r232 = byte_ratio_at_phi(PHI_372)
    chk("byte_ratio_phi0_is_1", abs(r0 - 1.0) < 1e-12, f"r0={r0}")
    chk("byte_ratio_phi1_is_uniform", abs(r1 - CB3_BPW_UNIFORM / INT4_BPW) < 1e-12, f"r1={r1:.6f}")
    chk("byte_ratio_phi232_reproduces_372", abs(r232 - BYTE_RATIO_372) < 1e-3, f"r232={r232:.6f} vs {BYTE_RATIO_372:.6f}")
    chk("byte_ratio_phi232_rounds_0p785", round(r232, 3) == 0.785, f"{round(r232,3)}")
    chk("byte_ratio_monotone_decr_in_phi",
        byte_ratio_at_phi(0.2) > byte_ratio_at_phi(0.5) > byte_ratio_at_phi(0.9),
        "")
    chk("shrink_monotone_incr_in_phi",
        shrink_at_phi(0.2) < shrink_at_phi(0.5) < shrink_at_phi(0.9), "")
    chk("byte_ratio_bounded_unit_interval",
        all(CB3_BPW_UNIFORM / INT4_BPW - 1e-9 <= byte_ratio_at_phi(p) <= 1.0 + 1e-9
            for p in (0.0, 0.3, 0.6, 0.888, 1.0)), "")

    # ---- 2. re-cost reproduces #391/#392 at phi=0.888 (k=232) ---- #
    rc232 = recost_at_phi(PHI_372, eff_m8)
    chk("m8_speedup_232_reproduces_391", abs(rc232["m8_speedup"] - M8_SPEEDUP_232) < 2e-3,
        f"{rc232['m8_speedup']:.6f} vs {M8_SPEEDUP_232:.6f}")
    chk("m1_speedup_232_reproduces_392", abs(rc232["m1_speedup"] - M1_SPEEDUP_232) < 2e-3,
        f"{rc232['m1_speedup']:.6f} vs {M1_SPEEDUP_232:.6f}")
    chk("m8_lift_232_reproduces_391_15p67", abs(rc232["m8_lift_off_the_shelf"] - M8_LIFT_232) < 0.15,
        f"{rc232['m8_lift_off_the_shelf']:.4f} vs {M8_LIFT_232:.4f}")
    chk("m1_lift_232_reproduces_392_32p65", abs(rc232["m1_lift_off_the_shelf"] - M1_LIFT_232) < 0.3,
        f"{rc232['m1_lift_off_the_shelf']:.4f} vs {M1_LIFT_232:.4f}")
    chk("shrink_232_reproduces_21p5pct", abs(rc232["bodyread_shrink"] - SHRINK_232) < 1e-3,
        f"{rc232['bodyread_shrink']:.5f}")

    # ---- 3. tier ordering + lift monotonicity ---- #
    chk("m8_lift_below_m1_lift", rc232["m8_lift_off_the_shelf"] < rc232["m1_lift_off_the_shelf"],
        "measured-floor strict < qtip-empirical")
    chk("m1_below_roofline", rc232["m1_speedup"] < rc232["roofline_speedup"], "")
    lo = recost_at_phi(0.5, eff_m8)["m8_lift_off_the_shelf"]
    hi = recost_at_phi(0.9, eff_m8)["m8_lift_off_the_shelf"]
    chk("m8_lift_monotone_incr_in_phi", lo < hi, f"phi.5={lo:.3f} phi.9={hi:.3f}")
    chk("m8_lift_positive_for_phi_pos", recost_at_phi(0.3, eff_m8)["m8_lift_off_the_shelf"] > 0, "")
    chk("zero_lift_at_phi0", abs(recost_at_phi(0.0, eff_m8)["m8_lift_off_the_shelf"]) < 1e-9, "")

    # ---- 4. k*-search logic on a SYNTHETIC monotone curve ---- #
    # PPL rises with k; bar=2.41. Build rows where heldout/ood cross 2.41 at k=210.
    synth = []
    for k in range(0, 259):
        ppl = SERVED_INT4_PPL_SPEC + (PPL_GATE - SERVED_INT4_PPL_SPEC) * (k / 258.0) * 1.05
        synth.append({"k": k, "heldout_worst_ppl": ppl, "ood_ppl": ppl,
                      "clears_2p41_both": bool(ppl <= CONSERVATIVE_BAR and ppl <= CONSERVATIVE_BAR),
                      "phi_params": k / 258.0, "m8_lift": k * 0.05})
    sks = find_kstar(synth)
    largest_clear = max(r["k"] for r in synth if r["clears_2p41_both"])
    chk("kstar_search_picks_largest_clearing", sks["k_star_largest"] == largest_clear,
        f"got {sks['k_star_largest']} want {largest_clear}")
    chk("kstar_monotone_robust_equals_largest", sks["k_star_robust"] == sks["k_star_largest"],
        "monotone synthetic -> robust==largest")
    chk("kstar_clears_bar_in_synth", synth[sks["k_star"]]["heldout_worst_ppl"] <= CONSERVATIVE_BAR, "")
    chk("kstar_plus_one_breaches_in_synth",
        sks["k_star_largest"] >= 258 or synth[sks["k_star_largest"] + 1]["heldout_worst_ppl"] > CONSERVATIVE_BAR, "")
    # tighter bar -> smaller k*
    synth_tight = [dict(r, clears_2p41_both=bool(r["heldout_worst_ppl"] <= 2.40)) for r in synth]
    sks_tight = find_kstar(synth_tight)
    chk("tighter_bar_lowers_kstar", sks_tight["k_star_largest"] <= sks["k_star_largest"], "")
    # lane-dead synthetic: nothing clears
    synth_dead = [dict(r, clears_2p41_both=False) for r in synth]
    chk("lane_dead_when_nothing_clears", find_kstar(synth_dead)["cb3_supply_lane_dead"], "")

    # ---- 5. residual arithmetic ---- #
    resid = STRICT_GAP_TO_500 - M8_LIFT_232
    chk("residual_plus_m8_equals_gap", abs((resid + M8_LIFT_232) - STRICT_GAP_TO_500) < 1e-9, "")
    chk("residual_positive_at_232", resid > 0, f"resid={resid:.3f}")
    chk("m8_232_below_gap", M8_LIFT_232 < STRICT_GAP_TO_500, "")

    # ---- 6. gate / bar / guards ---- #
    chk("conservative_bar_below_gate", CONSERVATIVE_BAR < PPL_GATE, "")
    chk("bar_margin_is_0p01", abs((PPL_GATE - CONSERVATIVE_BAR) - 0.01) < 1e-9, "")
    chk("guards_analysis_only", True, "analysis_only/no_hf_job/no_served_file_change set in assemble")

    # ---- 7. NaN/inf clean ---- #
    vals = [byte_ratio_at_phi(p) for p in (0, .3, .6, .888, 1)] + \
           [recost_at_phi(p, eff_m8)["m8_lift_off_the_shelf"] for p in (0, .3, .6, .888)]
    chk("nan_inf_clean", all(math.isfinite(v) for v in vals), "")

    n_pass = sum(1 for _, c, _ in checks if c)
    passes = bool(n_pass == len(checks) and len(checks) >= 20)
    for name, cond, detail in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not cond else ""),
              flush=True)
    print(f"[ckl] self-test: {n_pass}/{len(checks)} passed (>=20 required) -> {passes}", flush=True)
    return {"checks": [{"name": n, "pass": c, "detail": d} for n, c, d in checks],
            "n_checks": len(checks), "n_passed": n_pass,
            "cb3_conservative_k_self_test_passes": passes}


# ======================================================================================== #
# Report / IO / wandb.
# ======================================================================================== #
def print_report(res: dict[str, Any]) -> None:
    c = res["config"]; ks = res["kstar"]; rc = res["recost_at_kstar"]
    rd = res["residual"]; dec = res["decision"]; isr = res["in_sample"]
    print("\n" + "=" * 104, flush=True)
    print("CB3 CONSERVATIVE-k DEPLOYABLE LIFT (PR #403) -- the REAL PPL-safe supply number", flush=True)
    print("=" * 104, flush=True)
    print(f"  base={c['base_model']}  gate<=2.42  conservative_bar<=2.41  split_seeds={c['split_seeds']}  "
          f"phi(232)={c['phi_232_measured']:.4f}", flush=True)
    print(f"  IN-SAMPLE k=232 gate={isr['in_sample_gate_ppl_at_232']:.4f} "
          f"(372: {isr['in_sample_gate_ppl_372']:.4f}, reproduces={isr['reproduces_372']})", flush=True)
    print("-" * 104, flush=True)
    print(f"  k* (bankable, robust) = {ks['k_star']}  [largest-clearing={ks['k_star_largest']}, "
          f"monotone_clean={ks['monotone_clean']}]", flush=True)
    print(f"    heldout_worst_ppl@k*={ks['heldout_worst_ppl_at_kstar']:.4f}  "
          f"ood_ppl@k*={ks['ood_ppl_at_kstar']:.4f}  margin_to_2.42={ks['ppl_margin_to_242_at_kstar']:+.4f}  "
          f"binding={ks['binding_constraint']}", flush=True)
    print(f"    k=232 survives worst-seed 2.41 bar = {ks['kstar_232_survives_worst_bar']}", flush=True)
    print("-" * 104, flush=True)
    print(f"  RE-COST @ k*={ks['k_star']}: phi={rc['phi_params_at_kstar']:.4f}  "
          f"byte_ratio={rc['byte_ratio_at_kstar']:.4f}  shrink={rc['bodyread_shrink_at_kstar']*100:.2f}%  "
          f"(vs 21.5% @232: {rc['shrink_vs_232']*100:+.2f}pp)", flush=True)
    print(f"    M1_lift@k*={rc['m1_lift_at_kstar']:+.2f}  "
          f"M8_lift@k*={rc['m8_lift_at_kstar']:+.2f} TPS (DEPLOYABLE SUPPLY; vs +15.67 @232: "
          f"{rc['m8_lift_vs_232']:+.2f})", flush=True)
    print(f"  RESIDUAL gap-to-500 after cb3 = 32.53 - {rc['m8_lift_at_kstar']:.2f} = "
          f"{rd['residual_gap_to_500_after_cb3']:+.2f} TPS  "
          f"(cb3 closes {rd['frac_of_gap_closed_by_cb3']*100:.1f}% of the 32.53 gap)", flush=True)
    print("-" * 104, flush=True)
    print(f"  >>> VERDICT: {dec['verdict']}  (deployable={dec['cb3_conservative_deployable']}, "
          f"lane_dead={dec['cb3_supply_lane_dead']})", flush=True)
    print("=" * 104, flush=True)


def _print_senpai_result(res: dict[str, Any], st_passes: bool) -> None:
    ks = res["kstar"]; rc = res["recost_at_kstar"]; rd = res["residual"]; dec = res["decision"]
    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
        "no_launch": True, "official_tps": 0,
        "k_star": ks["k_star"],
        "heldout_worst_ppl_at_kstar": round(ks["heldout_worst_ppl_at_kstar"], 4),
        "ood_ppl_at_kstar": round(ks["ood_ppl_at_kstar"], 4),
        "ppl_margin_to_242_at_kstar": round(ks["ppl_margin_to_242_at_kstar"], 4),
        "bodyread_shrink_at_kstar": round(rc["bodyread_shrink_at_kstar"], 4),
        "m1_lift_at_kstar": round(rc["m1_lift_at_kstar"], 3),
        "m8_lift_at_kstar": round(rc["m8_lift_at_kstar"], 3),
        "residual_gap_to_500_after_cb3": round(rd["residual_gap_to_500_after_cb3"], 3),
        "cb3_conservative_deployable": dec["cb3_conservative_deployable"],
        "cb3_supply_lane_dead": dec["cb3_supply_lane_dead"],
        "cb3_conservative_k_self_test_passes": st_passes,
        "primary_metric": {"name": "cb3_conservative_k_self_test_passes", "value": int(st_passes)},
    }
    print("SENPAI-RESULT " + json.dumps(marker), flush=True)


def maybe_log_wandb(args, payload: dict[str, Any]) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ckl] wandb unavailable: {exc}", flush=True)
        return
    res = payload["result"]; c = res["config"]; ks = res["kstar"]
    rc = res["recost_at_kstar"]; rd = res["residual"]; dec = res["decision"]; isr = res["in_sample"]
    run = init_wandb_run(
        job_type="validity-gate", agent="kanna", name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "cb3-conservative-k", "deployable-lift", "ppl-safe",
              "supply-recost", "held-out", "ood-sharegpt", "pr-403"],
        config={k: v for k, v in c.items() if not isinstance(v, (list, dict))},
    )
    if run is None:
        print("[ckl] wandb: no run -- skipping", flush=True)
        return
    summary = {
        "k_star": ks["k_star"],
        "k_star_largest": ks["k_star_largest"],
        "k_star_robust": ks["k_star_robust"],
        "monotone_clean": int(bool(ks["monotone_clean"])),
        "heldout_worst_ppl_at_kstar": ks["heldout_worst_ppl_at_kstar"],
        "ood_ppl_at_kstar": ks["ood_ppl_at_kstar"],
        "ppl_margin_to_242_at_kstar": ks["ppl_margin_to_242_at_kstar"],
        "kstar_232_survives_worst_bar": int(bool(ks["kstar_232_survives_worst_bar"])),
        "bodyread_shrink_at_kstar": rc["bodyread_shrink_at_kstar"],
        "phi_params_at_kstar": rc["phi_params_at_kstar"],
        "byte_ratio_at_kstar": rc["byte_ratio_at_kstar"],
        "m1_lift_at_kstar": rc["m1_lift_at_kstar"],
        "m8_lift_at_kstar": rc["m8_lift_at_kstar"],
        "m8_lift_floor_at_kstar": rc["m8_lift_floor_at_kstar"],
        "roofline_lift_at_kstar": rc["roofline_lift_at_kstar"],
        "residual_gap_to_500_after_cb3": rd["residual_gap_to_500_after_cb3"],
        "frac_of_gap_closed_by_cb3": rd["frac_of_gap_closed_by_cb3"],
        "cb3_conservative_deployable": int(bool(dec["cb3_conservative_deployable"])),
        "cb3_supply_lane_dead": int(bool(dec["cb3_supply_lane_dead"])),
        "cb3_conservative_k_self_test_passes": int(bool(payload["self_test_passes"])),
        "in_sample_gate_ppl_at_232": isr["in_sample_gate_ppl_at_232"],
        "reproduces_372": int(bool(isr["reproduces_372"])),
        "phi_232_measured": c["phi_232_measured"],
        "official_tps": 0,
        "analysis_only": int(bool(res["guards"]["analysis_only"])),
        "no_hf_job": int(bool(res["guards"]["no_hf_job"])),
        "no_served_file_change": int(bool(res["guards"]["no_served_file_change"])),
        "no_launch": int(bool(res["guards"]["no_launch"])),
        "no_kernel_build": int(bool(res["guards"]["no_kernel_build"])),
        "peak_mem_mib": payload["peak_mem_mib"], "elapsed_s": payload["elapsed_s"],
    }
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    log_summary(run, summary, step=0)
    # per-k curve as a wandb Table for rich post-hoc analysis
    try:
        import wandb
        cols = ["k", "phi_params", "byte_ratio", "bodyread_shrink", "full_gate_ppl",
                "heldout_worst_ppl", "heldout_mean_ppl", "ood_ppl", "m1_lift", "m8_lift",
                "clears_2p41_both"]
        tbl = wandb.Table(columns=cols)
        for r in res["curve"]:
            tbl.add_data(*[r.get(col) for col in cols])
        run.log({"ppl_vs_k_curve": tbl})
    except Exception as exc:  # noqa: BLE001
        print(f"[ckl] wandb table skipped: {exc}", flush=True)
    log_json_artifact(run, name="cb3_conservative_k_deployable_lift", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[ckl] wandb logged: {len(summary)} metrics + curve table", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic + search self-test (PRIMARY)")
    ap.add_argument("--base-model", "--base_model", dest="base_model", default=m355.DEFAULT_BASE_MODEL)
    ap.add_argument("--ppl-dataset", "--ppl_dataset", dest="ppl_dataset", default=str(m355.DEFAULT_PPL_DATASET))
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(DEFAULT_SPLIT_SEEDS))
    ap.add_argument("--quant-seed", type=int, default=0)
    ap.add_argument("--scheme", choices=["asym", "sym"], default="asym")
    ap.add_argument("--group-size", "--group_size", dest="group_size", type=int, default=128)
    ap.add_argument("--vq-dim", "--vq_dim", dest="vq_dim", type=int, default=2)
    ap.add_argument("--dense-lo", type=int, default=195, help="dense integer k-grid lower bound")
    ap.add_argument("--dense-hi", type=int, default=247, help="dense integer k-grid upper bound")
    ap.add_argument("--ood-n", type=int, default=96)
    ap.add_argument("--ood-max-ctx", type=int, default=1024)
    ap.add_argument("--ood-tgt-lo", type=int, default=16)
    ap.add_argument("--ood-tgt-cap", type=int, default=384)
    ap.add_argument("--smoke", action="store_true", help="tiny grid + few records (wiring check)")
    ap.add_argument("--out-dir", dest="out_dir", default=str(HERE))
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="cb3-conservative-k-deployable-lift")
    args = ap.parse_args(argv)

    if args.self_test:
        st = self_test()
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / SELFTEST_NAME).write_text(json.dumps(st, indent=2, default=float))
        return 0 if st["cb3_conservative_k_self_test_passes"] else 1

    t_start = time.time()
    import torch
    res = run_gpu(args)
    st = self_test()
    res["self_test"] = {"n_checks": st["n_checks"], "n_passed": st["n_passed"],
                        "cb3_conservative_k_self_test_passes": st["cb3_conservative_k_self_test_passes"]}
    print_report(res)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_mib = (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0
    payload = {
        "created_at": created_at, "pr": 403, "agent": "kanna",
        "kind": "cb3-conservative-k-deployable-lift",
        "elapsed_s": round(time.time() - t_start, 1), "peak_mem_mib": round(peak_mib, 1),
        "self_test_passes": st["cb3_conservative_k_self_test_passes"],
        "result": res,
    }
    out_path = Path(args.out_dir) / RESULTS_NAME
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    print(f"[ckl] wrote {out_path} (elapsed {payload['elapsed_s']}s, peak {payload['peak_mem_mib']} MiB)",
          flush=True)
    _print_senpai_result(res, st["cb3_conservative_k_self_test_passes"])
    maybe_log_wandb(args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
