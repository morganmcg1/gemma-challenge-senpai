#!/usr/bin/env python3
"""PR #185 -- Launch-trigger calculator: one-call verified GO/NO-GO + filled approval
block from land #71's measured tuple.

This is the EXECUTABLE measured-tuple trigger. `launch_decision(measured_tuple)` applies
the full composition law `official = K_cal*(E[T]/step)*tau` + the lambda-acceptance gate
to land #71's measured ladder and emits a human-ready GO/NO-GO decision plus the filled
`Approval request: HF job` block. It PRODUCES the approval block; it does NOT file it -- a
human must still approve the filed issue before any spend.

It REUSES merged-PR engines verbatim (does NOT re-derive geometry):
  * denken #183 margin-aware lambda-acceptance card -> the BINDING build-gate bar
    lambda*_LCB (finite-sample-LCB-clears-500), the forward map, and the inverse
    measured-ladder -> lambda_hat_built, via
    research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card.py (imports #178/#175/#159).
  * fern #179 packet geometry -> `cell()` (proj_private, LCB(P>=0.9), P(clear-500)) via
    scripts/profiler/launch_packet_refresh.py (which imports fern #174 -> #167).
  * denken #178 self-KV forward/inverse map (q_d(lambda), E[T](lambda)) via
    research/oracle_readout/realistic_selfkv_floor/realistic_selfkv_floor.py (imports #172).
  * wirbel #170 over-accept / greedy-exactness gate via
    research/oracle_readout/overaccept_signature/overaccept_signature.py.

LAMBDA-GATE (advisor-pinned, denken #183 MERGED): the binding per-topology build-gate is
`lambda_hat_built >= lambda*_LCB(topology, tau)` read from #183's REAL card -- both-bugs
0.9052 (tau=1) / 0.9234 (tau=0.9924), descent-only 0.9750 / 0.9926 -- STRICTER than #178's
central point bar (0.838 both / 0.909 descent). The #178 central bar is reported as the
flagged-INACTIVE looser reference only.

TWO-LCB DESIGN (transparency for the advisor): the OVERALL per-topology launch verdict ANDs
TWO finite-sample lower bounds that cross 500 at DIFFERENT lambda:
  * #183 BUILD-gate LCB = central(lam) - z95*sqrt(SE_tps^2 + sigma_hw^2)   [sigma_hw FOLDED,
    NO private drop] -> crosses 500 at lambda*_LCB (0.9052 both / 0.9750 descent at tau=1).
  * #179 LAUNCH-projection cell-LCB = central -> proj_private(#164 drop) -> finite-sample CI
    [sigma_hw on a SEPARATE best-of-2 axis] -> 514.88 both / 499.97 descent at lambda=1.
For descent-only these DIVERGE: at lambda=1 it CLEARS the #183 build-bar (LCB 505.53, lam
1.0 >= 0.9750) but MISSES the #179 launch projection (499.97 < 500) -- a knife-edge launch
miss on a build-acceptable kernel. both-bugs clears BOTH at lambda=1 (the robust GO).

DEGRADATION (PR step 5e): ubel #181's tau-pin and wirbel #184's lambda-robust contingency
topology are IN-FLIGHT (not landed) -> named as the gap-fallback restoration levers and flagged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import resource
import sys
import time
from typing import Any

# ----------------------------------------------------------------------------------------- #
# Engine loading (reuse merged-PR machinery verbatim via dynamic import).
# ----------------------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _load(name: str, relpath: str):
    path = os.path.join(REPO_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


V179 = _load("launch_packet_refresh", "scripts/profiler/launch_packet_refresh.py")
V183 = _load("lambda_acceptance_card",
             "research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card.py")
V178 = _load("realistic_selfkv_floor",
             "research/oracle_readout/realistic_selfkv_floor/realistic_selfkv_floor.py")
V170 = _load("overaccept_signature",
             "research/oracle_readout/overaccept_signature/overaccept_signature.py")
D172 = V178.D172

# ----------------------------------------------------------------------------------------- #
# Banked launch constants (imported, not re-derived).
# ----------------------------------------------------------------------------------------- #
K_CAL = V179.K_CAL                       # 125.26795005202914
TARGET_OFFICIAL = 500.0
Z_P90 = V179.Z_P90                       # 1.281552 (one-sided P>=0.9)
P_GO = V179.P_GO                         # 0.90
PPL_BAR = 2.42                           # Issue #124 / denken #166 validity bar
PASS_REQUIRED = 128                      # 128/128 completions
TAU_HEADLINE = 1.0                       # headline lambda*_LCB gate corner (advisor: 0.9052 both)
TAU_CONSERVATIVE = V183.TAU_CONSERVATIVE  # 0.9924 tree-class floor (#183 conservative bar)

# Numerator E[T] full-recovery anchors (lawine #161 / denken #172).
E_T_STAR_DESCENT = V178.IMPORTED_CENTRAL_5p0564   # 5.056404568844709
E_T_STAR_BOTH = V178.IMPORTED_BOTH_BUGS_5p2070    # 5.206954309441963

# v over-accept noise floor (wirbel #170 trustworthy region: v<=v_tol is greedy-exact).
V_TOL = 1.52587890625e-05

# Dependency provenance / degrade flags (PR step 5e).
DEP_FLAGS = {
    "denken_183_margin_aware_lambda_card": "LANDED -- CONSUMED. The binding build-gate bar "
        "lambda*_LCB is read from #183's REAL card (lambda_star_lcb): both-bugs 0.9052 (tau=1) "
        "/ 0.9234 (tau=0.9924), descent-only 0.9750 / 0.9926. #178's central point bar "
        "(0.838/0.909) is the flagged-INACTIVE looser reference only.",
    "ubel_181_tau_pin": "IN-FLIGHT (not landed) -- tau band [tau_low, 1.0] from stark #164 used; "
        "the conservative tau=0.9924 corner is reported as the #181 stand-in floor.",
    "wirbel_184_lambda_robust_topology": "IN-FLIGHT (not landed) -- named as the gap-fallback "
        "restoration lever; the contingency topology itself is not yet importable.",
}


def _finite(x: float) -> float:
    return float(x) if (x is not None and math.isfinite(float(x))) else float("nan")


# ----------------------------------------------------------------------------------------- #
# cell() (#179) + lambda-card (#183) machinery assembled ONCE on shared anchors.
# ----------------------------------------------------------------------------------------- #
class _Machinery:
    def __init__(self) -> None:
        cons = V179.cons
        base_step = V179.STEP_BASE                     # 1.2182
        self.base_step = base_step
        self.knots = V179.load_stark151_retention(step=base_step)["knots"]
        self.kcal_band = V179.load_kcal_band()
        self.sampling = V179.SamplingModel(
            n_steps=cons.env.ORACLE_STEPS, n_boot=2000, seed=174,
            step=base_step, step_rel_hw=V179.STEP_REL_DEFAULT)
        self.b_both = V179.build_joint_b_dict(cons.RHO_OPT_JSON, cons.ORACLE_LIVE_JSON)
        self.b_desc = V179.b_dict_for_depth1(self.b_both, V179.DEPTH1_DESCENT_ONLY)

        self.framings = V179.load_step168_framings()
        self.shipped_step = self.framings["shipped"]   # 1.2181727676912677
        pinned = V179.load_stark156_pinned()
        self.drop_desc = pinned["drop_descent_only"]
        self.drop_both = pinned["drop_both_bugs"]
        self.sigma_hw = V179.load_kanna159_sigma_hw()
        self.stark164 = V179.load_stark164_private()
        self.tau_low = self.stark164["tau_low"]        # 0.9924318649123313

        # Shared #178/#183 anchors -> the REAL #183 lambda-card context + endpoint spines.
        anchors = D172.load_anchors(
            D172.DEFAULT_BUG2_ANCHOR, D172.DEFAULT_TOPO_JSON, D172.DEFAULT_ACCEPT_JSON,
            D172.DEFAULT_RANKCOV_JSON, D172.DEFAULT_DECOMP_JSON)
        self.ctx183 = V183.build_topologies(anchors)   # denken #183 card context
        self.ep = self.ctx183["ep"]
        self.H = len(self.ctx183["topo"]["both_bugs"]["q_full"])
        # liveprobe lambda_hat (#178 formula, recomputed from the same anchors) for self-test.
        self.lam_hat_liveprobe = (
            (V178.LIVEPROBE_WALK_TOPW0_HIT - self.ctx183["topo"]["descent_only"]["q_floor"][0])
            / (V178.LIVEPROBE_LINEAR_TOP1 - self.ctx183["topo"]["descent_only"]["q_floor"][0]))

    # ---- #183 card accessors (q_floor/q_full per topology) ---- #
    def qfqF_183(self, topo: str):
        t = self.ctx183["topo"][topo]
        return t["q_floor"], t["q_full"]

    def lambda_star_lcb_183(self, topo: str, tau: float = TAU_HEADLINE) -> float:
        qf, qF = self.qfqF_183(topo)
        return V183.lambda_star_lcb(self.ctx183, qf, qF, tau)

    def lambda_star_central_178(self, topo: str, tau: float = TAU_HEADLINE) -> float:
        qf, qF = self.qfqF_183(topo)
        return V183.lambda_star_central(self.ctx183, qf, qF, tau)

    def lambda_built_from_ladder(self, q_ladder: list[float], topo: str = "both_bugs") -> float:
        """#183 inverse: pooled per-depth lambda_hat_built from a measured q[2..9] ladder.
        Topology-invariant for depths>=2 (descent/both differ only at depth-1)."""
        qf, qF = self.qfqF_183(topo)
        return V183.lambda_from_measured_ladder(self.ctx183, list(q_ladder), qf, qF)["lambda_hat_built"]

    def metrics_183(self, lam: float, topo: str, tau: float = TAU_HEADLINE) -> dict:
        qf, qF = self.qfqF_183(topo)
        return V183.metrics_at(self.ctx183, lam, qf, qF, tau)

    def et_of_lambda(self, lam: float, topo: str) -> float:
        return self.metrics_183(lam, topo)["E_T"]

    def canonical_ladder(self, lam: float, topo: str = "both_bugs") -> list[float]:
        """The depth-2..9 q ladder of a constant-lambda spine (8 entries, #183 d_lo..d_hi)."""
        qf, qF = self.qfqF_183(topo)
        spine = V178.spine_from_profile(self.ep, V178.constant_lambda(len(qF), lam), qf, qF)
        return [D172.qd_at(spine, d) for d in range(2, 10)]

    # ---- #179 cell() (proj_private / LCB(P>=0.9) / P(clear-500)) ---- #
    def topo_profiles(self, topo: str):
        if topo == "both_bugs":
            return self.b_both, self.drop_both, E_T_STAR_BOTH
        return self.b_desc, self.drop_desc, E_T_STAR_DESCENT

    def cell_for(self, E_T: float, topo: str, step: float, validity: dict) -> dict:
        b_dict, drop, _ = self.topo_profiles(topo)
        return V179.cell(E_T, topo, drop, step, b_dict, self.knots, self.sampling,
                         self.kcal_band, validity)


_M = _Machinery()


# ----------------------------------------------------------------------------------------- #
# LAUNCH-CI LEDGER (advisor #185 update 2026-06-14T16:53Z): the launch CI fractured into FOUR
# orthogonal numerical noise axes + one hard packaging precondition. Build the ledger to ingest
# each as a TYPED row so the final GO/NO-GO recomposes when they land. Do NOT hardcode the iid
# +-10.9 / public-0.9052 bar: carry `binding_bar = max(public #183, ICC-refined #190, private
# #191)` and a composed half-width that folds the independent axes in quadrature. Every in-flight
# axis (#190/#187/#188/#191) and the packaging gate (#189) is NOT yet landed -> each row runs in
# flagged `iid-fallback` / `pending`, and the ledger reproduces the #183/#179 numbers exactly.
# ----------------------------------------------------------------------------------------- #
def _try_load_axis(name: str, relpath: str):
    """Best-effort import of an in-flight axis module; None if not landed (-> fallback). The
    conventional paths below do not exist yet, so every axis degrades to its flagged fallback;
    when an axis lands at its path, the loader consumes it and the ledger recomposes."""
    path = os.path.join(REPO_ROOT, relpath)
    if not os.path.exists(path):
        return None
    try:
        return _load(name, relpath)
    except Exception:                       # noqa: BLE001
        return None


def numerical_ci_ledger(topo: str, tau: float = TAU_HEADLINE) -> list[dict]:
    """The numerical launch-CI axes as typed rows. Axis 0 (wirbel #175 iid sampling) is LANDED
    and is the fallback baseline; axes #190/#187/#188/#191 are IN-FLIGHT -> flagged fallback."""
    public_bar = _M.lambda_star_lcb_183(topo, tau)          # #183 iid public bar (#175 (+) #159)
    rows = []

    # Axis 0 -- wirbel #175 iid sampling leg (LANDED): the +-10.906 numerator, a LOWER bound.
    rows.append({
        "axis": "sampling_iid", "pr": 175, "slug": "et-second-moment", "kind": "numerical",
        "status": "LANDED", "flag": "consumed",
        "halfwidth_tps_iid": (V183.WIRBEL_HALFWIDTH_BOTH_BUGS if topo == "both_bugs" else None),
        "note": "iid +-10.906 numerator leg (B=16384); a LOWER bound on the true half-width.",
    })
    # Axis 1 -- wirbel #190 icc-neff-launch-ci (IN-FLIGHT): realistic within-prompt ICC shrinks
    #           N_eff and RAISES the bar. iid +-10.9 is a LOWER bound. Fallback = iid public bar.
    m190 = _try_load_axis("icc_neff_launch_ci",
                          "research/oracle_readout/icc_neff_launch_ci/icc_neff_launch_ci.py")
    rows.append({
        "axis": "sampling_icc", "pr": 190, "slug": "icc-neff-launch-ci", "kind": "numerical",
        "status": "LANDED" if m190 else "IN-FLIGHT",
        "flag": "consumed" if m190 else "iid-fallback",
        "lambda_bar": public_bar,    # fallback; when landed -> lambda_star_lcb_realistic_icc
        "expected_when_landed": "halfwidth_realistic + lambda_star_lcb_realistic_icc (>= iid bar)",
        "note": "iid +-10.9 is a LOWER bound (ICC=1 -> N_eff 128->54.9 -> LCB 480.5 FLIPS); "
                "consume the correlation-refined bar (stricter-or-equal) when #190 lands.",
    })
    # Axis 2 -- denken #187 lambda-built-ci (IN-FLIGHT): INPUT-side -- how tightly land #71's
    #           measured q[2..9] resolves the bar. Fallback = point lambda_hat (0 input CI).
    m187 = _try_load_axis("lambda_built_ci",
                          "research/oracle_readout/lambda_built_ci/lambda_built_ci.py")
    rows.append({
        "axis": "input_lambda_built", "pr": 187, "slug": "lambda-built-ci", "kind": "numerical",
        "status": "LANDED" if m187 else "IN-FLIGHT",
        "flag": "consumed" if m187 else "pending-input-ci",
        "lambda_built_halfwidth": 0.0,    # fallback: point estimate, no input-side CI
        "expected_when_landed": "lambda_built_halfwidth (dual of #175's output-side TPS CI)",
        "note": "input-side CI on lambda_hat_built; widens the effective bar margin when landed.",
    })
    # Axis 3 -- kanna #188 oneshot-hw-bound (IN-FLIGHT): is sigma_hw=4.86 right for a SINGLE
    #           launch draw? Fallback = sigma_hw 4.86 (#159).
    m188 = _try_load_axis("oneshot_hw_bound",
                          "research/oracle_readout/oneshot_hw_bound/oneshot_hw_bound.py")
    rows.append({
        "axis": "hardware_oneshot", "pr": 188, "slug": "oneshot-hw-bound", "kind": "numerical",
        "status": "LANDED" if m188 else "IN-FLIGHT",
        "flag": "consumed" if m188 else "oneshot-fallback",
        "sigma_tps": V183.SIGMA_HW,       # fallback: kanna #159 sigma_hw 4.86
        "expected_when_landed": "sigma_oneshot (within-run vs between-device/thermal decomposition)",
        "note": "composed in quadrature with the sampling leg; fallback uses #159 sigma_hw=4.86.",
    })
    # Axis 4 -- stark #191 private-build-bar (IN-FLIGHT): PRIVATE-side bar via #176 adverse-skew
    #           drop (2.300% descent) through #183's forward map. Fallback = public bar.
    m191 = _try_load_axis("private_build_bar",
                          "research/oracle_readout/private_build_bar/private_build_bar.py")
    rows.append({
        "axis": "private_bar", "pr": 191, "slug": "private-build-bar", "kind": "numerical",
        "status": "LANDED" if m191 else "IN-FLIGHT",
        "flag": "consumed" if m191 else "public-fallback",
        "lambda_bar": public_bar,    # fallback; when landed -> lambda_star_lcb_private
        "expected_when_landed": "lambda_star_lcb_private + valid_at_bar (may exceed public 0.9052)",
        "note": "private adverse-skew drop may demand a STRICTER bar than public; max()'d into binding_bar.",
    })
    return rows


def binding_bar(topo: str, tau: float = TAU_HEADLINE) -> dict:
    """binding_bar = max(public #183, ICC-refined #190, private #191). Every in-flight axis falls
    back to the public iid bar, so in fallback binding_bar == the #183 public bar (0.9052 both /
    0.9750 descent at tau=1). When #190/#191 land, this can only get stricter."""
    ledger = numerical_ci_ledger(topo, tau)
    public = _M.lambda_star_lcb_183(topo, tau)
    icc_row = next(r for r in ledger if r["axis"] == "sampling_icc")
    priv_row = next(r for r in ledger if r["axis"] == "private_bar")
    icc, private = icc_row["lambda_bar"], priv_row["lambda_bar"]
    bar = max(public, icc, private)
    if abs(bar - icc) < 1e-12 and icc_row["flag"] == "consumed" and icc > public + 1e-12:
        source = "icc_190"
    elif abs(bar - private) < 1e-12 and priv_row["flag"] == "consumed" and private > public + 1e-12:
        source = "private_191"
    else:
        source = "public_183" + ("_iid_fallback" if icc_row["flag"] != "consumed" else "")
    return {
        "binding_bar": _finite(bar), "public_183": _finite(public),
        "icc_190": _finite(icc), "private_191": _finite(private),
        "binding_source": source,
        "icc_flag": icc_row["flag"], "private_flag": priv_row["flag"],
        "any_iid_fallback_active": bool(icc_row["flag"] != "consumed"
                                        or priv_row["flag"] != "consumed"),
    }


def composed_lcb(lam: float, topo: str, tau: float = TAU_HEADLINE) -> float:
    """Composed finite-sample LCB at lambda = central - z*sqrt(SUM independent axis halfwidths^2).
    In all-fallback this is exactly #183's lcb_full (iid sampling #175 (+) sigma_hw #159); when
    #190 (ICC), #188 (sigma_oneshot), #187 (input-side) land, the quadrature inflates here."""
    return _finite(_M.metrics_183(lam, topo, tau)["lcb_full_tps"])


def precondition_ledger(build_lambda_ok: bool, build_topo: str) -> list[dict]:
    """Hard precondition rows (kind=precondition). Any NO-GO blocks an ACTUAL launch even if the
    composed-LCB clears. Operational rows (PRECACHE serve-config, #189 packaging gate, human
    approval) are PENDING for a pre-launch calculator run -> launch_authorized stays False."""
    m189 = _try_load_axis(
        "executable_submission_gate",
        "research/oracle_readout/executable_submission_gate/executable_submission_gate.py")
    return [
        {"row": "boot_fix_kanna_177", "pr": 177, "kind": "precondition", "status": "GO",
         "flag": "landed", "note": "darwin _IncludedRouter boot-validation startup-500 fix; "
                                   "output-neutral (land #71 banks the diff)."},
        {"row": "precache_bench", "kind": "precondition", "status": "PENDING",
         "flag": "serve-config",
         "note": "PRECACHE_BENCH=1 must be set on the served path at launch."},
        {"row": "executable_submission_gate_ubel_189", "pr": 189, "kind": "precondition",
         "status": "GO" if m189 else "PENDING",
         "flag": "consumed" if m189 else "pending-packaging",
         "expected_when_landed": "verify_submission_gate(build_env, introspection) -> GO/NO-GO "
                                 "+ failing flag + banked cost (row-1 relocate-host-loop = 85% TPS)",
         "note": "packaging precondition; UNVERIFIED until #189 lands -> blocks an actual launch."},
        {"row": "build_lambda_geq_binding_bar", "pr": 71, "kind": "precondition",
         "status": "GO" if build_lambda_ok else "NO-GO", "flag": "measured",
         "note": "land #71's measured lambda_hat_built >= binding_bar for topology %s." % build_topo},
        {"row": "human_approval", "kind": "precondition", "status": "PENDING", "flag": "human-gate",
         "note": "a human must approve the filed `Approval request: HF job` issue before any spend."},
    ]


# ----------------------------------------------------------------------------------------- #
# STEP 1 -- measured-tuple schema.
# ----------------------------------------------------------------------------------------- #
def tuple_schema() -> dict:
    """The measured tuple land #71 must emit (PR step 1). `drop`/`tau`/sigma_hw are BANKED
    legs (stark #164 / kanna #159), not measured-kernel fields, so they are NOT in the tuple."""
    return {
        "name": "str -- submission name (used in the Approval request title).",
        "E_T_descent": "float -- E[T] under #154 argmax-only (descent-only) decode.",
        "E_T_both": "float -- E[T] under the both-bugs descending accept-prep decode.",
        "v": "float -- over-accept rate vs wirbel #170 trustworthy region "
             "(v<=%.3e is greedy-exact; v>v_tol => illusory speedup)." % V_TOL,
        "step_us": "float -- measured denominator step (overlap, microsec-convention "
                   "matching K_cal=%.5f)." % K_CAL,
        "q_ladder": "list[float] -- per-depth accepted-rate ladder q[2..9] (8 entries, the "
                    "self-KV depths). Inverted through #183's q_d(lambda) -> lambda_hat_built.",
        "PPL": "float -- validation perplexity (bar <= %.2f)." % PPL_BAR,
        "pass_128": "int -- completions passing (need %d/%d)." % (PASS_REQUIRED, PASS_REQUIRED),
        "greedy_exact": "bool -- greedy-decode identity preserved (Issue #124).",
        "boots_ok": "bool -- bootstrap CI available for the finite-sample leg.",
    }


def make_measured_tuple(name: str, E_T_descent: float, E_T_both: float, v: float,
                        step_us: float, q_ladder: list[float], PPL: float, pass_128: int,
                        greedy_exact: bool = True, boots_ok: bool = True) -> dict:
    return {
        "name": str(name),
        "E_T_descent": float(E_T_descent),
        "E_T_both": float(E_T_both),
        "v": float(v),
        "step_us": float(step_us),
        "q_ladder": [float(x) for x in q_ladder],
        "PPL": float(PPL),
        "pass_128": int(pass_128),
        "greedy_exact": bool(greedy_exact),
        "boots_ok": bool(boots_ok),
    }


def synth_land71_tuple(name: str, lam: float) -> dict:
    """Synthesize a measured tuple from a constant-lambda deep-spine recovery (for self-test).
    The q_ladder is the #183 canonical q[2..9] spine at `lam`; E_T_{topo} = #183 E[T] at `lam`."""
    return make_measured_tuple(
        name=name,
        E_T_descent=_M.et_of_lambda(lam, "descent_only"),
        E_T_both=_M.et_of_lambda(lam, "both_bugs"),
        v=0.0,
        step_us=_M.shipped_step,
        q_ladder=_M.canonical_ladder(lam, "both_bugs"),   # depths 2..9 (8 entries)
        PPL=2.39, pass_128=128, greedy_exact=True, boots_ok=True)


# ----------------------------------------------------------------------------------------- #
# lambda_hat inference (#183 inverse) + lambda* bars (#183 real card).
# ----------------------------------------------------------------------------------------- #
def lambda_hat_per_depth(q_ladder: list[float], topo: str = "both_bugs") -> dict:
    """#183 inverse: per-depth lambda_hat_d + pooled lambda_hat_built over the measured ladder."""
    qf, qF = _M.qfqF_183(topo)
    res = V183.lambda_from_measured_ladder(_M.ctx183, list(q_ladder), qf, qF)
    return {
        "per_depth": [{"depth": p["depth"], "lambda_hat_d": _finite(p["lambda_hat_d"])}
                      for p in res["per_depth"]],
        "lambda_hat_built": _finite(res["lambda_hat_built"]),
    }


def lambda_hat_et_matched(E_T: float, topo: str) -> float:
    """Diagnostic: the constant-lambda whose #183 forward-map E[T] == measured E_T."""
    et0, et1 = _M.et_of_lambda(0.0, topo), _M.et_of_lambda(1.0, topo)
    if E_T <= et0:
        return 0.0 if abs(E_T - et0) < 1e-9 else (E_T - et0) / max(et1 - et0, 1e-12)
    if E_T >= et1:
        return 1.0 if abs(E_T - et1) < 1e-9 else 1.0 + (E_T - et1) / max(et1 - et0, 1e-12)
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _M.et_of_lambda(mid, topo) >= E_T:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------------------- #
# STEP 2 -- per-topology projection (composition law + cell() CI).
# ----------------------------------------------------------------------------------------- #
def official_direct(E_T: float, step: float, tau: float) -> float:
    """Headline composition law: official = K_cal*(E[T]/step)*tau (matches #178 official_tps;
    scales with the MEASURED E_T, unlike cell()'s b_dict-anchored geom_tps_public)."""
    return K_CAL * (E_T / step) * tau


def project_topology(E_T: float, topo: str, step: float, tau_high: float, tau_low: float,
                     validity: dict) -> dict:
    """official (direct, tau-banded) + proj_private / LCB(P>=0.9) / P(clear-500) from cell()."""
    c = _M.cell_for(E_T, topo, step, validity)
    sigma_fold = V179.fold_sigma_hw(c, _M.sigma_hw["sigma_hw_pct"])
    return {
        "E_T": _finite(E_T),
        "official_tps": _finite(official_direct(E_T, step, tau_high)),       # tau=1.0 headline
        "official_tps_tau_low": _finite(official_direct(E_T, step, tau_low)),
        "proj_private_tps": _finite(c["proj_private_tps"]),
        "lcb_p90": _finite(c["lcb_p90"]),
        "p_clear_500": _finite(c["p_clear_500"]),
        "validity_gate": c["validity_gate"],
        # sigma_hw on a SEPARATE hardware axis (best-of-2 retires it); naive-fold for sensitivity.
        "sigma_hw_naive_fold": {"lcb_p90_4term": _finite(sigma_fold["lcb_p90_4term"]),
                                "p_clear_500_4term": _finite(sigma_fold["p_clear_500_4term"])},
    }


# ----------------------------------------------------------------------------------------- #
# STEP 3 -- the one-call launch decision.
# ----------------------------------------------------------------------------------------- #
def _classify_overaccept(E_T: float, v: float, e_star: float) -> dict:
    """wirbel #170 three-region classifier: TRUSTWORTHY / OVER_ACCEPT_BUG2 / ANOMALOUS."""
    gate = V170.make_land_gate(e_star)
    trustworthy = bool(gate(E_T, v))
    ceiling = V170.et_of_v(V_TOL, e_star)
    if trustworthy:
        region = "TRUSTWORTHY"
    elif v > V_TOL and E_T > ceiling:
        region = "OVER_ACCEPT_BUG2"      # inflated E[T] sitting on the over-accept locus
    else:
        region = "ANOMALOUS"             # E[T] above ceiling but v ~ 0 (does not add up)
    return {"region": region, "trustworthy": trustworthy,
            "v_implied": _finite(V170.v_of_et(E_T, e_star)), "et_ceiling": _finite(ceiling)}


def _topology_verdict(topo: str, E_T: float, lam_built: float, t: dict, step: float,
                      tau_low: float, validity: dict, validity_ok: bool) -> dict:
    e_star = E_T_STAR_BOTH if topo == "both_bugs" else E_T_STAR_DESCENT
    oa = _classify_overaccept(E_T, t["v"], e_star)
    proj = project_topology(E_T, topo, step, 1.0, tau_low, validity)

    # ---- BUILD-gate: lambda_hat_built >= binding_bar = max(public#183, ICC#190, private#191) ----
    bbar = binding_bar(topo, TAU_HEADLINE)                             # ledger-composed bar
    lam_star_lcb_hi = bbar["binding_bar"]                             # 0.9052 both / 0.9750 desc (fallback)
    lam_star_lcb_lo = _M.lambda_star_lcb_183(topo, TAU_CONSERVATIVE)  # 0.9234 / 0.9926 (tau floor)
    lam_central_178 = _M.lambda_star_central_178(topo, TAU_HEADLINE)  # 0.8384 / 0.9091 (INACTIVE)
    build_gate_pass = bool(math.isfinite(lam_star_lcb_hi)
                           and lam_built >= lam_star_lcb_hi - 1e-9)
    build_gate_pass_cons = bool(math.isfinite(lam_star_lcb_lo)
                                and lam_built >= lam_star_lcb_lo - 1e-9)
    # composed build-LCB at the measured lambda (the bar's own LCB, for the divergence readout).
    build_lcb_at_lam = composed_lcb(lam_built, topo, TAU_HEADLINE)

    # ---- #179 LAUNCH-projection cell-LCB gate (unchanged integrator) ----
    clear500_lcb_pass = bool(proj["lcb_p90"] >= TARGET_OFFICIAL - 1e-9)

    # ---- overall per-topology GO: validity AND trustworthy AND both LCB gates ----
    go = bool(validity_ok and oa["trustworthy"] and build_gate_pass and clear500_lcb_pass)

    # Failing-gate diagnosis + restoration lever (PR step 4).
    failing_gate, restoration = None, None
    if not validity_ok:
        failing_gate = "validity (PPL/128/greedy-exact)"
        restoration = "fix build quality to satisfy PPL<=2.42, 128/128, greedy-exact."
    elif not oa["trustworthy"]:
        failing_gate = "over-accept (#170): E[T] is NOT greedy-trustworthy"
        restoration = ("measured speedup is illusory (region=%s); re-bench under greedy-exact "
                       "decode before reading E[T] as headroom." % oa["region"])
    elif not build_gate_pass:
        in_gap = bool(math.isfinite(lam_central_178) and lam_built >= lam_central_178 - 1e-9)
        failing_gate = ("lambda-acceptance build-gate (#183): lambda_hat_built=%.4f < "
                        "lambda*_LCB=%.4f" % (lam_built, lam_star_lcb_hi))
        if in_gap:
            restoration = ("lambda_hat_built clears the INACTIVE #178 central bar %.4f but sits "
                           "in the GAP below #183's margin-aware bar %.4f -> fall back to wirbel "
                           "#184's lambda-robust contingency topology, or ship #154 argmax-only "
                           "decode (+3.96 TPS LCB)." % (lam_central_178, lam_star_lcb_hi))
        else:
            restoration = ("recover deeper self-KV: lambda_hat_built=%.4f is below even the #178 "
                           "central bar %.4f; the spine has not risen enough (cause-#2 starvation)."
                           % (lam_built, lam_central_178))
    elif not clear500_lcb_pass:
        failing_gate = ("clear-500 launch-projection LCB (#179): cell-LCB(P>=0.9)=%.2f < 500 "
                        "(build-gate PASSED at lambda*_LCB=%.4f -- a build-acceptable kernel that "
                        "still misses the private-drop launch projection)" % (proj["lcb_p90"],
                                                                              lam_star_lcb_hi))
        restoration = ("ship #154 argmax-only decode (+3.96 TPS LCB -> realizable) or land the "
                       "both-bugs kernel; descent-only is a knife-edge launch miss at the shipped "
                       "step even though it clears the #183 build bar.")

    return {
        "topo": topo,
        "GO": go,
        "verdict": "GO" if go else "NO-GO",
        **proj,
        "overaccept": oa,
        "lambda_gate": {
            "lambda_hat_built": _finite(lam_built),
            "binding_bar": bbar,                                            # max(public,ICC,private)
            "lambda_star_lcb_183_headline": _finite(lam_star_lcb_hi),       # BINDING (tau=1, = binding_bar)
            "lambda_star_lcb_183_conservative": _finite(lam_star_lcb_lo),   # tau=0.9924
            "lambda_star_central_178_INACTIVE": _finite(lam_central_178),   # looser reference only
            "build_gate_pass": build_gate_pass,
            "build_gate_pass_conservative": build_gate_pass_cons,
            "build_lcb_at_lambda_built": build_lcb_at_lam,
            "clear500_launch_lcb_pass": clear500_lcb_pass,
            "numerical_ci_ledger": numerical_ci_ledger(topo, TAU_HEADLINE),
            "binding_rule": "lambda_hat_built >= binding_bar = max(public#183 0.9052, ICC#190, "
                            "private#191); in-flight axes -> iid-fallback. STRICTER than #178 central.",
            "two_lcb_divergence": (
                "composed build-LCB (#175(+)#159, axes 1-4) and #179 launch cell-LCB cross 500 at "
                "DIFFERENT lambda; build_gate=%s, launch_lcb_gate=%s" % (build_gate_pass,
                                                                         clear500_lcb_pass)),
        },
        "failing_gate": failing_gate,
        "restoration_lever": restoration,
    }


def launch_decision(measured_tuple: dict, step_override: float | None = None) -> dict:
    """ONE-CALL verified GO/NO-GO from land #71's measured tuple.

    Returns {verdict, headline_topology, per-topology {official, proj_private, LCB,
    P(clear-500), GO/NO-GO}, lambda_gate_logic, both_bugs_go_at_lambda_star,
    filled_approval_block | nogo_restoration, dependency flags}.
    PRODUCES the approval block; does NOT file it (human approval still required)."""
    t = measured_tuple
    step = float(step_override) if step_override is not None else float(t["step_us"])
    tau_low = _M.tau_low

    # ---- validity gate (hard) ----
    validity_ok = bool(t["PPL"] <= PPL_BAR and t["pass_128"] >= PASS_REQUIRED
                       and t.get("greedy_exact", False))
    validity_cell = {"ppl": t["PPL"], "boots": bool(t.get("boots_ok", True)),
                     "completed": int(t["pass_128"])}
    validity = {
        "ppl": _finite(t["PPL"]), "ppl_bar": PPL_BAR, "ppl_ok": bool(t["PPL"] <= PPL_BAR),
        "pass_128": int(t["pass_128"]), "pass_ok": bool(t["pass_128"] >= PASS_REQUIRED),
        "greedy_exact": bool(t.get("greedy_exact", False)),
        "validity_ok": validity_ok,
    }

    # ---- lambda_hat_built from the measured q[2..9] ladder (#183 inverse, topology-invariant) ----
    inv = lambda_hat_per_depth(t["q_ladder"], "both_bugs")
    lam_built = inv["lambda_hat_built"]

    # ---- per-topology verdicts ----
    per_topo = {
        "both_bugs": _topology_verdict("both_bugs", t["E_T_both"], lam_built, t, step, tau_low,
                                       validity_cell, validity_ok),
        "descent_only": _topology_verdict("descent_only", t["E_T_descent"], lam_built, t, step,
                                          tau_low, validity_cell, validity_ok),
    }

    # ---- overall verdict: both-bugs is the primary GO path; descent-only is the fallback ----
    bb, do = per_topo["both_bugs"], per_topo["descent_only"]
    overall_go = bool(bb["GO"] or do["GO"])
    if bb["GO"]:
        headline = "both_bugs"
    elif do["GO"]:
        headline = "descent_only"
    else:
        headline = "both_bugs"           # primary path named even on NO-GO
    verdict = "GO" if overall_go else "NO-GO"

    # ---- lambda_gate_logic (PR step 3 report) ----
    lambda_gate_logic = {
        "rule": "per-topology GO iff lambda_hat_built >= lambda*_LCB(topo, tau=1) [#183 REAL "
                "card BUILD-gate: both-bugs 0.9052 / descent 0.9750, STRICTER than #178's "
                "central 0.838/0.909] AND #179 launch-projection cell-LCB(P>=0.9) >= 500. In "
                "the gap [#178-central, #183-LCB) -> wirbel #184 contingency / #154 restore.",
        "lambda_hat_built_from_ladder": _finite(lam_built),
        "ladder_inverse_per_depth": inv["per_depth"],
        "both_bugs": bb["lambda_gate"],
        "descent_only": do["lambda_gate"],
        "card_is_monotone": bool(_M.metrics_183(0.5, "both_bugs")["E_T"] > 0),  # placeholder; set below
        "degraded_dependencies": DEP_FLAGS,
    }
    # real monotone flag from #183's forward map (both topologies, tau=1).
    lambda_gate_logic["card_is_monotone"] = _card_monotone_183()

    # ---- TEST metric: #183 build-gate verdict at exactly both-bugs lambda*_LCB ----
    both_bugs_go_at_lambda_star = _build_gate_at_lambda_star("both_bugs")

    # ---- launch-CI ledger (advisor 16:53Z): binding bars + precondition rows + authorization ----
    bb_binding = binding_bar("both_bugs", TAU_HEADLINE)
    do_binding = binding_bar("descent_only", TAU_HEADLINE)
    headline_binding = bb_binding if headline == "both_bugs" else do_binding
    build_lambda_ok = bool(math.isfinite(headline_binding["binding_bar"])
                           and lam_built >= headline_binding["binding_bar"] - 1e-9)
    preconds = precondition_ledger(build_lambda_ok, headline)
    preconds_all_go = all(r["status"] == "GO" for r in preconds)
    # launch_authorized = analytic-GO AND every precondition GO. Operational rows (PRECACHE
    # serve-config, #189 packaging, human approval) are PENDING pre-launch -> authorizes NOTHING.
    launch_authorized = bool(overall_go and preconds_all_go)
    any_iid_fallback = bool(bb_binding["any_iid_fallback_active"]
                            or any(r["flag"] != "consumed"
                                   for r in numerical_ci_ledger("both_bugs", TAU_HEADLINE)
                                   if r["axis"] != "sampling_iid"))

    out = {
        "verdict": verdict,
        "overall_go": overall_go,
        "headline_topology": headline,
        "name": t["name"],
        "step_used": _finite(step),
        "tau_band": [_finite(tau_low), 1.0],
        "validity": validity,
        "per_topology": {
            k: {"official_tps": v["official_tps"],
                "official_tps_tau_low": v["official_tps_tau_low"],
                "proj_private_tps": v["proj_private_tps"],
                "lcb_p90": v["lcb_p90"],
                "p_clear_500": v["p_clear_500"],
                "verdict": v["verdict"],
                "failing_gate": v["failing_gate"],
                "restoration_lever": v["restoration_lever"],
                "overaccept_region": v["overaccept"]["region"]}
            for k, v in per_topo.items()},
        "lambda_gate_logic": lambda_gate_logic,
        "both_bugs_go_at_lambda_star": both_bugs_go_at_lambda_star,
        "binding_bar": {"both_bugs": bb_binding, "descent_only": do_binding},
        "launch_ci_ledger": {
            "net_rule": "launch_authorized = (analytic GO: composed-LCB >= 500 at binding_bar "
                        "AND validity AND over-accept) AND (all precondition rows GO). Numerical "
                        "axes #190/#187/#188/#191 fold in quadrature; binding_bar = max(public#183, "
                        "ICC#190, private#191). In-flight axes -> iid-fallback (flagged); the "
                        "ledger recomposes when they land.",
            "numerical_axes": {"both_bugs": numerical_ci_ledger("both_bugs", TAU_HEADLINE),
                               "descent_only": numerical_ci_ledger("descent_only", TAU_HEADLINE)},
            "preconditions": preconds,
            "preconditions_all_go": preconds_all_go,
            "any_iid_fallback_active": any_iid_fallback,
        },
        "launch_authorized": {
            "authorized": launch_authorized,
            "analytic_verdict": verdict,
            "preconditions_all_go": preconds_all_go,
            "note": "PRODUCED, not FILED. analytic_verdict can be GO while authorized=False because "
                    "operational preconditions (PRECACHE serve-config, #189 packaging gate, human "
                    "approval) are PENDING. The calculator authorizes NOTHING.",
        },
        "hardware_axis_sigma_hw": {
            "sigma_hw_tps": _finite(_M.sigma_hw["sigma_hw_tps"]),
            "best_of_2_p": _finite(_M.sigma_hw["best_of_2"]),
            "note": "RETIRED on a SEPARATE hardware axis by best-of-2 official draws (P=%.4f>=0.90);"
                    " does NOT subtract from the #179 projection-axis LCB. (#183's build-LCB folds"
                    " sigma_hw in quadrature instead -- the documented two-LCB divergence.)"
                    % _M.sigma_hw["best_of_2"],
        },
        "_full_per_topology": per_topo,
    }

    if overall_go:
        out["filled_approval_block"] = render_approval_block(out, per_topo)
        out["nogo_restoration"] = None
    else:
        out["filled_approval_block"] = None
        out["nogo_restoration"] = render_nogo_restoration(out, per_topo)
    out["human_deferral"] = ("This block is PRODUCED, not FILED. A human must approve the filed "
                             "`Approval request: HF job` issue before any spend.")
    return out


def _card_monotone_183() -> bool:
    """#183 forward map monotonicity (both topologies, tau=1) -- read from the REAL card."""
    ok = True
    for topo in ("both_bugs", "descent_only"):
        qf, qF = _M.qfqF_183(topo)
        ls_lcb = _M.lambda_star_lcb_183(topo, TAU_HEADLINE)
        ls_cen = _M.lambda_star_central_178(topo, TAU_HEADLINE)
        fm = V183.forward_map(_M.ctx183, qf, qF, TAU_HEADLINE,
                              _M.lam_hat_liveprobe, ls_lcb, ls_cen)
        ok = ok and bool(fm["card_is_monotone"])
    return bool(ok)


def _build_gate_at_lambda_star(topo: str) -> bool:
    """TEST: the BUILD-gate verdict evaluated at EXACTLY this topology's binding_bar. In fallback
    binding_bar == #183's lambda*_LCB; the composed-LCB == 500 there by construction and
    lambda_hat_built == binding_bar, so the build-gate passes inclusively -> True (the advisor's
    headline gate). Uses binding_bar so the metric stays correct when #190/#191 land (stricter)."""
    lam_star = binding_bar(topo, TAU_HEADLINE)["binding_bar"]
    if not math.isfinite(lam_star):
        return False
    ladder = _M.canonical_ladder(lam_star, topo)
    lam_built = _M.lambda_built_from_ladder(ladder, topo)
    return bool(lam_built >= lam_star - 1e-9)


# ----------------------------------------------------------------------------------------- #
# STEP 4 -- filled approval block (GO) / restoration (NO-GO).
# ----------------------------------------------------------------------------------------- #
def render_approval_block(out: dict, per_topo: dict) -> str:
    h = out["headline_topology"]
    cell = per_topo[h]
    name = out["name"]
    lg = per_topo[h]["lambda_gate"]
    tbl = ["| topology | official (tau=1.0) | proj_private | LCB(P>=0.9) | P(clear-500) | verdict |",
           "|---|---|---|---|---|---|"]
    for k in ("both_bugs", "descent_only"):
        c = per_topo[k]
        tbl.append("| %s | %.2f | %.2f | %.2f | %.4f | %s |" % (
            k, c["official_tps"], c["proj_private_tps"], c["lcb_p90"], c["p_clear_500"], c["verdict"]))
    table_md = "\n".join(tbl)
    binding = out["binding_bar"][h]
    led = out["launch_ci_ledger"]
    num_md = ["| axis | PR | status | flag | bar/value |", "|---|---|---|---|---|"]
    for r in led["numerical_axes"][h]:
        val = r.get("lambda_bar", r.get("sigma_tps", r.get("lambda_built_halfwidth",
                    r.get("halfwidth_tps_iid"))))
        val_s = ("%.4f" % val) if isinstance(val, (int, float)) else str(val)
        num_md.append("| %s | #%s | %s | %s | %s |" % (
            r["axis"], r["pr"], r["status"], r["flag"], val_s))
    num_table = "\n".join(num_md)
    pre_md = ["| precondition | status | flag |", "|---|---|---|"]
    for r in led["preconditions"]:
        pre_md.append("| %s | %s | %s |" % (r["row"], r["status"], r["flag"]))
    pre_table = "\n".join(pre_md)
    la = out["launch_authorized"]
    return f"""### Approval request: HF job for {name}

**PRE-FILLED DRAFT (NOT FILED).** A human must approve this filed issue before any spend.
**launch_authorized = {la['authorized']}** (analytic_verdict {la['analytic_verdict']}; preconditions_all_go {la['preconditions_all_go']}).

**ANALYTIC VERDICT: {out['verdict']}** on the **{h}** path at the shipped step {out['step_used']:.4f}
(LCB(P>=0.9) {cell['lcb_p90']:.2f} TPS, P(clear-500) {cell['p_clear_500']*100:.2f}%, official
{cell['official_tps']:.2f} TPS at tau=1.0; tau-low {cell['official_tps_tau_low']:.2f}).

**Composition:** official = K_cal*(E[T]/step)*tau, K_cal={K_CAL:.5f}, tau in [{out['tau_band'][0]:.4f}, 1.0].
clear-500 verdict ANDs TWO finite-sample lower bounds: (1) the BUILD-gate lambda_hat_built
{lg['lambda_hat_built']:.4f} >= binding_bar {binding['binding_bar']:.4f}
(= max(public#183 {binding['public_183']:.4f}, ICC#190, private#191); source {binding['binding_source']};
STRICTER than #178's central 0.838 both / 0.909 descent), and (2) the #179 launch-projection
cell-LCB(P>=0.9) >= 500. sigma_hw RETIRED on a separate hardware axis by best-of-2 official
draws (P={out['hardware_axis_sigma_hw']['best_of_2_p']:.4f}>=0.90).

**Measured-tuple GO table:**
{table_md}

**Launch-CI ledger -- numerical axes (binding_bar = max; in-flight -> iid-fallback):**
{num_table}

**Launch-CI ledger -- hard precondition rows (any non-GO blocks the launch):**
{pre_table}

**Submission command (named hard deps -- the human runs this AFTER approval):**
```
PRECACHE_BENCH=1 <serve-harness with land #71 {h} kernel + kanna darwin _IncludedRouter boot-fix>
```
  - PRECACHE_BENCH=1 set on the served path.
  - kanna's darwin _IncludedRouter boot-validation startup-500 fix folded into the serve harness.

**Dependency ledger (all required-GREEN before approval):**
  - [BANKED] Numerator E[T] (#160/#165/#172): descent {E_T_STAR_DESCENT:.4f} / both {E_T_STAR_BOTH:.4f}.
  - [BANKED] Denominator step (#168): shipped {out['step_used']:.4f}.
  - [BANKED] Hardware sigma_hw (#159): {_M.sigma_hw['sigma_hw_tps']:.2f} TPS; best-of-2 P={_M.sigma_hw['best_of_2']:.4f}.
  - [BANKED] Validity (#166/#124): PPL {out['validity']['ppl']:.3f}<=2.42; {out['validity']['pass_128']}/128; greedy-exact={out['validity']['greedy_exact']}.
  - [BANKED] Private drop (#164): tau-low {out['tau_band'][0]:.4f}.
  - [BANKED] denken #183 margin-aware lambda-card: binding_bar {binding['binding_bar']:.4f} (tau=1) / {lg['lambda_star_lcb_183_conservative']:.4f} (tau=0.9924).
  - [IN-FLIGHT] launch-CI axes #190 (ICC) / #187 (lambda-built-CI) / #188 (sigma-oneshot) / #191 (private-bar) -> iid-fallback until landed (bar can only get stricter).
  - [IN-FLIGHT] ubel #189 executable-submission-gate (packaging precondition -- UNVERIFIED).
  - [PENDING-BUILD] land #71 measured tuple (THIS tuple).

**Launch gates (ALL required):** (1) land #71 builds the {h} kernel; (2) darwin _IncludedRouter
boot-fix folded; (3) PRECACHE_BENCH=1; (4) a human-approved `Approval request: HF job` issue.
"""


def render_nogo_restoration(out: dict, per_topo: dict) -> str:
    lines = [f"### NO-GO for {out['name']} at step {out['step_used']:.4f}", ""]
    for k in ("both_bugs", "descent_only"):
        c = per_topo[k]
        lines.append("- **%s -> %s** (official %.2f, LCB %.2f, P %.4f). Failing gate: %s. "
                     "Restoration: %s" % (k, c["verdict"], c["official_tps"], c["lcb_p90"],
                                          c["p_clear_500"], c["failing_gate"], c["restoration_lever"]))
    lines.append("")
    lines.append("No approval block emitted: at least the primary (both-bugs) path must clear "
                 "BOTH the #183 build-gate (lambda_hat_built >= lambda*_LCB) AND the #179 "
                 "launch-projection cell-LCB(P>=0.9) >= 500 for a robust GO.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------- #
# STEP 5 -- self-test (PRIMARY metric).
# ----------------------------------------------------------------------------------------- #
def self_test() -> dict:
    results: dict[str, Any] = {}
    tol = 0.05

    # (a) lambda=1 ladder reproduces #179 both LCB 514.88 / descent 499.97; both GO, descent NO-GO.
    t_full = synth_land71_tuple("self-test-lambda1", 1.0)
    d_full = launch_decision(t_full, step_override=_M.shipped_step)
    bb_lcb = d_full["per_topology"]["both_bugs"]["lcb_p90"]
    do_lcb = d_full["per_topology"]["descent_only"]["lcb_p90"]
    a_ok = (abs(bb_lcb - 514.877540689496) <= tol and abs(do_lcb - 499.96519706601964) <= tol
            and d_full["per_topology"]["both_bugs"]["verdict"] == "GO"
            and d_full["per_topology"]["descent_only"]["verdict"] == "NO-GO"
            and d_full["verdict"] == "GO" and d_full["headline_topology"] == "both_bugs")
    results["a_lambda1_reproduces_179"] = {
        "pass": bool(a_ok), "both_lcb": bb_lcb, "both_anchor": 514.877540689496,
        "descent_lcb": do_lcb, "descent_anchor": 499.96519706601964,
        "both_verdict": d_full["per_topology"]["both_bugs"]["verdict"],
        "descent_verdict": d_full["per_topology"]["descent_only"]["verdict"]}

    # (b) lambda_hat=0.342 ladder reproduces #178 misses descent 404.06 / both 416.31; NO-GO both.
    t_floor = synth_land71_tuple("self-test-lambdahat", _M.lam_hat_liveprobe)
    d_floor = launch_decision(t_floor, step_override=_M.shipped_step)
    bb_off = d_floor["per_topology"]["both_bugs"]["official_tps"]
    do_off = d_floor["per_topology"]["descent_only"]["official_tps"]
    b_ok = (abs(do_off - 404.06468476135797) <= tol and abs(bb_off - 416.307156176311) <= tol
            and d_floor["per_topology"]["both_bugs"]["verdict"] == "NO-GO"
            and d_floor["per_topology"]["descent_only"]["verdict"] == "NO-GO"
            and d_floor["verdict"] == "NO-GO")
    results["b_lambdahat_reproduces_178_misses"] = {
        "pass": bool(b_ok), "descent_official": do_off, "descent_anchor": 404.06468476135797,
        "both_official": bb_off, "both_anchor": 416.307156176311,
        "verdict": d_floor["verdict"]}

    # (c) over-accept v>0 flags OVER_ACCEPT/ANOMALOUS; inflated E[T] NOT read as headroom.
    e_inflated = 5.6   # above the both ceiling
    v_onlocus = V170.v_of_et(e_inflated, E_T_STAR_BOTH)
    t_oa_locus = make_measured_tuple("self-test-overaccept-locus", E_T_STAR_DESCENT * 1.1,
                                     e_inflated, v_onlocus, _M.shipped_step,
                                     _M.canonical_ladder(1.0, "both_bugs"), 2.39, 128, True, True)
    d_oa = launch_decision(t_oa_locus, step_override=_M.shipped_step)
    reg_both = d_oa["_full_per_topology"]["both_bugs"]["overaccept"]["region"]
    t_oa_anom = make_measured_tuple("self-test-overaccept-anom", E_T_STAR_DESCENT,
                                    e_inflated, 0.0, _M.shipped_step,
                                    _M.canonical_ladder(1.0, "both_bugs"), 2.39, 128, True, True)
    d_anom = launch_decision(t_oa_anom, step_override=_M.shipped_step)
    reg_anom = d_anom["_full_per_topology"]["both_bugs"]["overaccept"]["region"]
    c_ok = (reg_both == "OVER_ACCEPT_BUG2" and reg_anom == "ANOMALOUS"
            and d_oa["_full_per_topology"]["both_bugs"]["verdict"] == "NO-GO"
            and d_anom["_full_per_topology"]["both_bugs"]["verdict"] == "NO-GO")
    results["c_overaccept_flagged"] = {
        "pass": bool(c_ok), "region_on_locus": reg_both, "region_anomalous": reg_anom,
        "both_verdict_locus": d_oa["_full_per_topology"]["both_bugs"]["verdict"]}

    # (d) approval block well-formed on GO + names PRECACHE/boot-fix deps.
    blk = d_full["filled_approval_block"]
    d_ok = (blk is not None and "Approval request: HF job for" in blk
            and "PRECACHE_BENCH=1" in blk and "_IncludedRouter boot-fix" in blk
            and "human must approve" in blk and "NOT FILED" in blk)
    results["d_approval_block_wellformed"] = {
        "pass": bool(d_ok), "approval_block_wellformed": bool(d_ok),
        "names_precache": ("PRECACHE_BENCH=1" in (blk or "")),
        "names_bootfix": ("_IncludedRouter boot-fix" in (blk or ""))}

    # (e) #183 REAL card consumed: lambda*_LCB reproduces 0.9052/0.9750 (tau=1) & 0.9234/0.9926
    #     (tau_cons); card monotone; forward map 0.9052->500.0 & 0.342->404.14; #178 central
    #     bar (0.8384/0.9091) reported but flagged INACTIVE.
    lg = d_full["lambda_gate_logic"]
    bb_lg, do_lg = lg["both_bugs"], lg["descent_only"]
    fm_bb = V183.forward_map(_M.ctx183, *_M.qfqF_183("both_bugs"), TAU_HEADLINE,
                             _M.lam_hat_liveprobe, _M.lambda_star_lcb_183("both_bugs"),
                             _M.lambda_star_central_178("both_bugs"))
    row_lcb = next(r for r in fm_bb["rows"]
                   if abs(r["lambda"] - round(_M.lambda_star_lcb_183("both_bugs"), 5)) < 1e-9)
    row_live = next(r for r in fm_bb["rows"]
                    if abs(r["lambda"] - round(_M.lam_hat_liveprobe, 5)) < 1e-9)
    e_ok = (abs(bb_lg["lambda_star_lcb_183_headline"] - 0.905229319301184) <= 1e-3
            and abs(do_lg["lambda_star_lcb_183_headline"] - 0.9750199960244741) <= 1e-3
            and abs(bb_lg["lambda_star_lcb_183_conservative"] - 0.923358337599465) <= 1e-3
            and abs(do_lg["lambda_star_lcb_183_conservative"] - 0.9925986560663185) <= 1e-3
            and abs(bb_lg["lambda_star_central_178_INACTIVE"] - 0.8383898298915815) <= 1e-3
            and abs(do_lg["lambda_star_central_178_INACTIVE"] - 0.9091326079857753) <= 1e-3
            and bool(lg["card_is_monotone"])
            and abs(row_lcb["predicted_lcb_tps"] - 500.0) <= 0.1
            and bool(row_lcb["predicted_lcb_clears_500"])
            and abs(row_live["predicted_lcb_tps"] - 404.14) <= 0.5
            and not bool(row_live["predicted_lcb_clears_500"])
            and "LANDED -- CONSUMED" in lg["degraded_dependencies"]["denken_183_margin_aware_lambda_card"]
            and "INACTIVE" in str(list(bb_lg.keys())))
    results["e_183_real_card_consumed"] = {
        "pass": bool(e_ok),
        "both_lambda_star_lcb_tau1": bb_lg["lambda_star_lcb_183_headline"],
        "descent_lambda_star_lcb_tau1": do_lg["lambda_star_lcb_183_headline"],
        "both_lambda_star_lcb_tau_cons": bb_lg["lambda_star_lcb_183_conservative"],
        "both_central_178_inactive": bb_lg["lambda_star_central_178_INACTIVE"],
        "card_is_monotone": bool(lg["card_is_monotone"]),
        "fwd_lcb_at_lambda_star": row_lcb["predicted_lcb_tps"],
        "fwd_lcb_at_liveprobe": row_live["predicted_lcb_tps"],
        "183_flag": "LANDED -- CONSUMED"}

    # (f) NaN-clean across all reported numerics.
    def _all_finite(obj) -> bool:
        if isinstance(obj, float):
            return math.isfinite(obj)
        if isinstance(obj, dict):
            return all(_all_finite(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_all_finite(v) for v in obj)
        return True
    nan_targets = {k: v for k, v in d_full.items() if k != "filled_approval_block"}
    f_ok = _all_finite(nan_targets) and _all_finite(
        {k: v for k, v in d_floor.items() if k != "nogo_restoration"})
    results["f_nan_clean"] = {"pass": bool(f_ok)}

    # (g) two-LCB divergence documented: at lambda=1 descent PASSES the #183 build-gate
    #     (1.0 >= 0.9750) but FAILS the #179 launch cell-LCB (499.97 < 500) -> the two LCBs
    #     diverge for descent. both-bugs clears BOTH (the robust GO).
    do_full_lg = d_full["_full_per_topology"]["descent_only"]["lambda_gate"]
    bb_full_lg = d_full["_full_per_topology"]["both_bugs"]["lambda_gate"]
    g_ok = (bool(do_full_lg["build_gate_pass"]) and not bool(do_full_lg["clear500_launch_lcb_pass"])
            and bool(bb_full_lg["build_gate_pass"]) and bool(bb_full_lg["clear500_launch_lcb_pass"]))
    results["g_two_lcb_divergence_documented"] = {
        "pass": bool(g_ok),
        "descent_build_gate_pass": bool(do_full_lg["build_gate_pass"]),
        "descent_launch_lcb_pass": bool(do_full_lg["clear500_launch_lcb_pass"]),
        "descent_build_lcb_at_lambda1": do_full_lg["build_lcb_at_lambda_built"],
        "both_build_gate_pass": bool(bb_full_lg["build_gate_pass"]),
        "both_launch_lcb_pass": bool(bb_full_lg["clear500_launch_lcb_pass"])}

    # (h) launch-CI ledger well-formed (advisor 16:53Z): the FOUR numerical axes (#190/#187/
    #     #188/#191) present + flagged iid-fallback (none landed); binding_bar = max() reduces to
    #     public 0.9052 both / 0.9750 descent; composed-LCB at binding_bar == 500; #189 packaging
    #     precondition present & PENDING; launch_authorized==False while analytic verdict==GO.
    led = d_full["launch_ci_ledger"]
    axes_bb = {r["axis"]: r for r in led["numerical_axes"]["both_bugs"]}
    inflight_axes = {"sampling_icc", "input_lambda_built", "hardware_oneshot", "private_bar"}
    all_inflight_flagged = all(axes_bb[a]["status"] == "IN-FLIGHT"
                               and axes_bb[a]["flag"] != "consumed" for a in inflight_axes)
    bb_bar = d_full["binding_bar"]["both_bugs"]
    do_bar = d_full["binding_bar"]["descent_only"]
    composed_at_bar = composed_lcb(bb_bar["binding_bar"], "both_bugs", TAU_HEADLINE)
    pack_row = next(r for r in led["preconditions"]
                    if r["row"] == "executable_submission_gate_ubel_189")
    h_ok = (all_inflight_flagged
            and abs(bb_bar["binding_bar"] - 0.905229319301184) <= 1e-3
            and bb_bar["binding_source"].startswith("public_183")
            and bb_bar["any_iid_fallback_active"] is True
            and abs(do_bar["binding_bar"] - 0.9750199960244741) <= 1e-3
            and abs(composed_at_bar - 500.0) <= 0.1
            and pack_row["status"] == "PENDING" and pack_row["flag"] == "pending-packaging"
            and led["any_iid_fallback_active"] is True
            and d_full["launch_authorized"]["authorized"] is False
            and d_full["launch_authorized"]["analytic_verdict"] == "GO")
    results["h_launch_ci_ledger_wellformed"] = {
        "pass": bool(h_ok),
        "binding_bar_both": bb_bar["binding_bar"], "binding_source": bb_bar["binding_source"],
        "binding_bar_descent": do_bar["binding_bar"],
        "composed_lcb_at_binding_bar": composed_at_bar,
        "inflight_axes_all_iid_fallback": bool(all_inflight_flagged),
        "packaging_189_status": pack_row["status"],
        "launch_authorized": d_full["launch_authorized"]["authorized"],
        "analytic_verdict": d_full["launch_authorized"]["analytic_verdict"]}

    passes = bool(all(v["pass"] for v in results.values()))
    test_metric = bool(d_full["both_bugs_go_at_lambda_star"])
    return {
        "launch_trigger_calculator_self_test_passes": passes,
        "both_bugs_go_at_lambda_star": test_metric,
        "conditions": results,
    }


# ----------------------------------------------------------------------------------------- #
# wandb + main.
# ----------------------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if args.no_wandb:
        return
    try:
        import wandb
    except Exception as exc:               # noqa: BLE001
        print(f"[wandb] unavailable ({exc}); skipping.", file=sys.stderr)
        return
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group,
                     config={"k_cal": K_CAL, "ppl_bar": PPL_BAR, "target_official": TARGET_OFFICIAL,
                             "step_shipped": _M.shipped_step,
                             "lambda_star_lcb_both_tau1": _M.lambda_star_lcb_183("both_bugs"),
                             "lambda_star_lcb_descent_tau1": _M.lambda_star_lcb_183("descent_only")})
    st = payload["self_test"]
    flat = {
        "launch_trigger_calculator_self_test_passes": st["launch_trigger_calculator_self_test_passes"],
        "both_bugs_go_at_lambda_star": st["both_bugs_go_at_lambda_star"],
    }
    for cond, v in st["conditions"].items():
        flat[f"selftest/{cond}"] = bool(v["pass"])
    wp = payload["worked_example"]
    flat["worked_example/verdict_go"] = bool(wp["overall_go"])
    la = wp["launch_authorized"]
    flat["worked_example/launch_authorized"] = bool(la["authorized"])
    flat["worked_example/analytic_verdict_go"] = la["analytic_verdict"] == "GO"
    flat["worked_example/preconditions_all_go"] = bool(la["preconditions_all_go"])
    flat["worked_example/any_iid_fallback_active"] = bool(
        wp["launch_ci_ledger"]["any_iid_fallback_active"])
    for topo in ("both_bugs", "descent_only"):
        c = wp["per_topology"][topo]
        lgc = wp["_full_per_topology"][topo]["lambda_gate"]
        bb = wp["binding_bar"][topo]
        flat[f"worked_example/{topo}/official_tps"] = c["official_tps"]
        flat[f"worked_example/{topo}/lcb_p90"] = c["lcb_p90"]
        flat[f"worked_example/{topo}/p_clear_500"] = c["p_clear_500"]
        flat[f"worked_example/{topo}/lambda_hat_built"] = lgc["lambda_hat_built"]
        flat[f"worked_example/{topo}/lambda_star_lcb_183"] = lgc["lambda_star_lcb_183_headline"]
        flat[f"worked_example/{topo}/binding_bar"] = bb["binding_bar"]
        flat[f"worked_example/{topo}/binding_source"] = bb["binding_source"]
        flat[f"worked_example/{topo}/build_gate_pass"] = bool(lgc["build_gate_pass"])
        flat[f"worked_example/{topo}/clear500_launch_lcb_pass"] = bool(
            lgc["clear500_launch_lcb_pass"])
    wandb.log(flat)
    wandb.summary.update(flat)
    run.finish()


def run(args) -> dict:
    t0 = time.time()
    st = self_test()
    # worked example = land #71 at full self-KV recovery (lambda=1) -- the GO-path demonstration.
    worked = launch_decision(synth_land71_tuple("land-71-bothbugs-kernel", 1.0),
                             step_override=_M.shipped_step)
    handoff = (
        "launch_trigger_calculator: one-call launch_decision(measured_tuple) -> verified "
        "GO/NO-GO + filled (un-filed) Approval request. Self-test %s. The BINDING lambda-gate "
        "is a TYPED launch-CI ledger (advisor 16:53Z): binding_bar = max(public#183 %.4f, "
        "ICC#190, private#191) -- all four in-flight axes (#190 ICC, #187 lambda-built-CI, "
        "#188 sigma-oneshot, #191 private-bar) degrade to the iid-fallback public bar until "
        "they land, then the verdict RECOMPOSES (bar can only get stricter). land #71 must show "
        "lambda_hat_built >= binding_bar (%.4f both-bugs tau=1; %.4f tau=0.9924), STRICTER than "
        "#178's 0.838 central bar; AND the #179 launch-projection cell-LCB(P>=0.9) >= 500. "
        "launch_authorized = (analytic-GO AND all precondition rows GO); preconditions "
        "(PRECACHE_BENCH, ubel #189 packaging-gate, human approval) are PENDING -> "
        "launch_authorized=False (authorizes nothing). both-bugs clears both LCBs at lambda=1 "
        "(514.88, robust GO); descent clears the #183 build bar but misses the #179 launch "
        "projection (499.97). both_bugs_go_at_lambda_star=%s. Human approval still required "
        "before any HF spend." % (
            "PASSES" if st["launch_trigger_calculator_self_test_passes"] else "FAILS",
            _M.lambda_star_lcb_183("both_bugs", TAU_HEADLINE),
            _M.lambda_star_lcb_183("both_bugs", TAU_HEADLINE),
            _M.lambda_star_lcb_183("both_bugs", TAU_CONSERVATIVE),
            st["both_bugs_go_at_lambda_star"]))
    payload = {
        "pr": 185,
        "agent": "fern",
        "kind": "launch_trigger_calculator",
        "primary_metric_name": "launch_trigger_calculator_self_test_passes",
        "launch_trigger_calculator_self_test_passes": st["launch_trigger_calculator_self_test_passes"],
        "test_metric_name": "both_bugs_go_at_lambda_star",
        "both_bugs_go_at_lambda_star": st["both_bugs_go_at_lambda_star"],
        "tuple_schema": tuple_schema(),
        "self_test": st,
        "worked_example": worked,
        "handoff_line": handoff,
        "elapsed_sec": time.time() - t0,
        "peak_mem_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR #185 launch-trigger calculator")
    ap.add_argument("--out", default="research/validity/launch_trigger/launch_trigger_calculator_results.json")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", "--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", "--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/launch-trigger-calculator")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-trigger-calculator")
    args = ap.parse_args(argv)

    payload = run(args)
    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)

    st = payload["self_test"]
    print("launch_trigger_calculator_self_test_passes =",
          st["launch_trigger_calculator_self_test_passes"])
    print("both_bugs_go_at_lambda_star =", st["both_bugs_go_at_lambda_star"])
    for cond, v in st["conditions"].items():
        print(f"  {cond}: {'PASS' if v['pass'] else 'FAIL'}")
    print(payload["handoff_line"])
    _maybe_log_wandb(args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
