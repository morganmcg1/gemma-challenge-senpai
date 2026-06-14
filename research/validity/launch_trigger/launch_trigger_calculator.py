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

# Dependency provenance / degrade flags (PR step 5e). RE-RUN (advisor 17:27Z + 17:52Z): the launch-CI
# numerical axes #190/#191/#188/#187, the #189 packaging gate, #194 re-draw budget AND ubel #195
# cross-axis covariance have ALL MERGED; the ledger now reads their REAL banked scalars, the binding
# bar RECOMPOSES, and the combined single-shot sigma DE-DUPS (the 4-axis quadrature was invalid). The
# iid placeholder is superseded and the ledger is now CLOSED (no pending numerical axes).
DEP_FLAGS = {
    "denken_183_margin_aware_lambda_card": "LANDED -- CONSUMED. #183's REAL card (lambda_star_lcb) "
        "supplies the PUBLIC iid leg: both-bugs 0.9052 (tau=1) / 0.9234 (tau=0.9924), descent 0.9750 "
        "/ 0.9926. Post-recompose this is the visible iid-fallback, NOT the binding bar; #178's "
        "central point bar (0.838/0.909) stays flagged-INACTIVE (looser reference only).",
    "launch_ci_axes_190_191_188_187": "LANDED -- CONSUMED. wirbel #190 (realistic within-prompt "
        "ICC=0.145 -> half-width iid +-10.9 -> realistic +-22.9, build bar 0.9513), stark #191 "
        "(private adverse-skew bar 0.9780, DOMINATES; descent private LCB 490.16<500 -> UNREACHABLE), "
        "kanna #188 (sigma_oneshot=4.86 confirms #159), denken #187 (input-side lambda_built CI "
        "+-0.017, 89%% overlap). binding_bar = max(0.9052, 0.9513, 0.9780) = 0.9780 (private, "
        "both-bugs); descent UNREACHABLE.",
    "executable_submission_gate_ubel_189": "LANDED -- CONSUMED as a precondition. Faithful "
        "device-vectorized relocate build -> GO; auto-catches the row-1 relocate-as-host-Python-loop "
        "trap (banked 444.92 TPS, ~85%% cost). MUST be re-verified against land #71's real build.",
    "redraw_budget_kanna_194": "LANDED -- CONSUMED (informational). best-of-N official re-draw "
        "budget N* for P(clear-500)>=0.95; sizes the launch, NOT a binding_bar axis. (Landed at "
        "advisor branch 96e9b25, after the 17:27Z comment named it WIP.)",
    "cost_budget_kanna_200": "LANDED -- CONSUMED (annotation only; advisor 18:03Z). Cost-aware layer "
        "on #194's cost-invariant N*(mu): the REALISTIC spend at the bar is SEQUENTIAL early-stop "
        "E[shots]=1.94 (NOT fixed-5; ~half clear on shot 1). build-higher (mu>=512.2/N=1) beats "
        "stay-at-bar iff reaching mu costs < 4 official shots' GPU-$ (crossover c*=3.04*b fixed-N; "
        "early-stop raises it to c*=12.97*b). Prices the budget ROW the human reads; does NOT touch "
        "the binding bar or the single-shot sigma -- single-shot GO/NO-GO is UNCHANGED.",
    "cross_axis_covariance_ubel_195": "LANDED -- CONSUMED (advisor 17:52Z); de-dup MECHANISM retained "
        "as provenance, the 7.26/17.04 NUMBERS SUPERSEDED by ubel #201. The 4-axis quadrature is "
        "INVALID: rho(sampling#175, input-lambda#187) = +0.945 -> a DOUBLE-COUNT (overlap=rho^2=0.893; "
        "OUTPUT accept-length scatter and INPUT lambda_hat CI are two views of the SAME accept draw). "
        "FIX: de-dup A1+A2 into ONE acceptance axis (overlap-corrected 5.32 TPS) -- this sets the "
        "acceptance-axis IDENTITY that #201 then rescales by realistic ICC. The #195 stand-alone "
        "single-shot sigma 7.26/17.04 is now the ICC=0 corner of #201's combined LAUNCH sigma.",
    "launch_sigma_closure_ubel_201": "LANDED -- CONSUMED (advisor 18:23Z, W&B spau6tch). SUPERSEDES "
        "#195's 7.26/17.04 in the combined-sigma row: de-dup (#195, IDENTITY 5.32 iid) x realistic "
        "ICC (#190, MAGNITUDE sqrt(D)=2.100 -> 11.17) are ORTHOGONAL corrections to the SAME "
        "acceptance axis -> combined LAUNCH sigma 12.215 central / 13.796 worst-case. P95 GO trigger "
        "mu >= 500 + z_p95*sigma = 520.09 central / 522.69 worst-case vs the lambda=1 ceiling 520.95 "
        "-> central P95-reachable (+0.86), worst-case UNREACHABLE (-1.74). ICC erodes ~8 TPS of "
        "launch headroom (lifts #194's iid break-even 512.16 by +7.94). WIRE the mechanism, HOLD the "
        "verdict: this row is PROVISIONAL + NON-GATING (NOT hard-wired vs the ceiling). Two open "
        "levers finalize it: ubel #204 (clean-1-sigma unit rebase, IN FLIGHT, ~3 TPS, direction OPEN "
        "-> can FLIP the central verdict) + land #71 co-log (n=385 cross-device allocations retires "
        "the rho(*,hw) [-0.3,+0.3] band). NO change to the binding BUILD bar (private 0.9780, #191) "
        "-- purely the launch sigma->LCB row.",
    "ubel_181_tau_pin": "LANDED (advisor branch) -- the tau band [tau_low, 1.0] floor (stark #164) "
        "with the conservative tau=0.9924 corner; referenced as the floor, not a banked ledger axis.",
    "wirbel_184_lambda_robust_topology": "LANDED (advisor branch) -- named as the gap-fallback "
        "restoration lever; not consumed as a banked ledger axis by this calculator.",
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
# LAUNCH-CI LEDGER (advisor #185 updates 16:53Z + 17:27Z): the launch CI fractured into orthogonal
# numerical noise axes + one hard packaging precondition. Each is a TYPED row so the verdict
# recomposes as axes land. Do NOT hardcode the iid +-10.9 / public-0.9052 bar: carry
# `binding_bar = max(public #183, ICC-refined #190, private #191)` and a composed half-width.
#
# RE-RUN STATE (advisor 17:27Z + 17:52Z, 2026-06-14): #190 (ICC), #191 (private), #189 (packaging),
# #188 (sigma-oneshot), #187 (input-side), #194 (re-draw budget) AND #195 (cross-axis covariance)
# have ALL MERGED -> each row reads its REAL banked scalar from the merged module's results.json
# (consumed, NOT the iid placeholder). The binding both-bugs bar is now the PRIVATE bar 0.9780
# (dominates 0.9052 public / 0.9513 ICC), the sampling half-width is the realistic +-22.9 (ICC=0.145),
# not the iid +-10.9. #195 proved the 4-axis quadrature INVALID (rho(sampling,input)=0.945 double-
# count) -> de-dup into ONE acceptance axis -> combined single-shot sigma 7.26 central / 17.04 worst-
# case (the conservative GO/NO-GO corner). The ledger is now CLOSED: no pending numerical axes.
# ----------------------------------------------------------------------------------------- #
def _load_axis_json(relpath: str):
    """Read a merged axis's banked results.json from the working tree. None if absent (-> fallback);
    when an axis lands at its path, the ledger consumes its banked scalars and recomposes."""
    path = os.path.join(REPO_ROOT, relpath)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:                       # noqa: BLE001
        return None


def _dig(d: Any, *path: str, default: Any = None) -> Any:
    """Safe nested-dict getter; returns `default` if any key is missing or a leaf is hit early."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# REAL merged-axis result paths (all landed on the advisor branch this cycle, #195 included).
_AXIS_PATHS = {
    "icc_190": "research/validity/icc_neff/icc_neff_results.json",
    "private_191": "research/validity/private_build_bar/results.json",
    "oneshot_188": "research/validity/oneshot_hw_bound/oneshot_hw_bound_results.json",
    "lambda_built_187": "research/validity/lambda_built_ci/lambda_built_ci_results.json",
    "redraw_194": "research/validity/redraw_budget/redraw_budget_results.json",
    "packaging_189": "research/spec_cost_model/executable_submission_gate/executable_submission_gate.json",
    # ubel #195 cross-axis CI covariance -- MERGED this turn (advisor 17:52Z): the 4-axis quadrature
    # is INVALID (rho(sampling,input)=0.945 double-count); consume the de-dup'd combined sigma.
    "covariance_195": "research/validity/ci_axis_covariance/ci_axis_covariance_results.json",
    # kanna #200 cost-aware re-draw budget -- MERGED (advisor 18:03Z): sequential early-stop spend
    # (E[shots]=1.94 at the bar, NOT fixed-5) + the build-higher-vs-stay cost toggle. Annotates the
    # budget row ONLY; the binding bar and single-shot sigma are UNCHANGED.
    "cost_budget_200": "research/validity/cost_budget/cost_budget_results.json",
    # ubel #201 launch-sigma closure -- MERGED (advisor 18:23Z): the de-dup (#195) acceptance axis
    # evaluated under REALISTIC ICC (#190) -> combined launch sigma 12.215 central / 13.796 worst-case,
    # REPLACING #195's 7.26 / 17.04 in the combined-sigma row. Wire the MECHANISM; HOLD the verdict
    # PROVISIONAL (do NOT hard-wire GO/NO-GO vs the lambda=1 ceiling) pending #204 + land #71.
    "sigma_closure_201": "research/validity/launch_sigma_closure/launch_sigma_closure_results.json",
}


class _BankedAxes:
    """Reads the REAL banked scalars from each merged axis's results.json (advisor 17:27Z + 17:52Z:
    the iid placeholder is superseded). #190 ICC, #191 private, #188 sigma-oneshot, #187 input-side,
    #189 packaging, #194 re-draw budget AND #195 cross-axis covariance are ALL consumed; the ledger
    is CLOSED (the #195 de-dup proved the 4-axis quadrature invalid -> combined sigma 7.26/17.04)."""
    def __init__(self) -> None:
        self.icc = _load_axis_json(_AXIS_PATHS["icc_190"])
        self.private = _load_axis_json(_AXIS_PATHS["private_191"])
        self.oneshot = _load_axis_json(_AXIS_PATHS["oneshot_188"])
        self.lambda_built = _load_axis_json(_AXIS_PATHS["lambda_built_187"])
        self.redraw = _load_axis_json(_AXIS_PATHS["redraw_194"])
        self.packaging = _load_axis_json(_AXIS_PATHS["packaging_189"])
        self.covariance = _load_axis_json(_AXIS_PATHS["covariance_195"])
        self.cost_budget = _load_axis_json(_AXIS_PATHS["cost_budget_200"])
        self.sigma_closure = _load_axis_json(_AXIS_PATHS["sigma_closure_201"])

    # ---- wirbel #190 realistic within-prompt ICC / N_eff ---- #
    def icc_landed(self) -> bool:
        return self.icc is not None

    def halfwidth_iid(self) -> float:
        return float(_dig(self.icc, "realistic_ci", "halfwidth_iid_tps",
                          default=V183.WIRBEL_HALFWIDTH_BOTH_BUGS))

    def halfwidth_realistic(self) -> float | None:
        v = _dig(self.icc, "realistic_ci", "halfwidth_realistic_tps")
        return float(v) if v is not None else None

    def icc_bar(self) -> float | None:
        v = _dig(self.icc, "build_bar", "lambda_star_lcb_realistic_icc")
        return float(v) if v is not None else None

    def icc_hat(self) -> float | None:
        v = _dig(self.icc, "icc_estimate", "icc_hat")
        return float(v) if v is not None else None

    def design_effect(self) -> float | None:
        v = _dig(self.icc, "realistic_ci", "design_effect_hat")
        return float(v) if v is not None else None

    def realistic_launch_lcb(self, topo: str) -> float | None:
        key = "bothbugs_hat" if topo == "both_bugs" else "descent_hat"
        v = _dig(self.icc, "go_robustness", "section4", key, "lcb_tps")
        return float(v) if v is not None else None

    def realistic_launch_clears(self, topo: str) -> bool:
        key = "bothbugs_hat" if topo == "both_bugs" else "descent_hat"
        return bool(_dig(self.icc, "go_robustness", "section4", key, "lcb_clears_500", default=False))

    # ---- stark #191 private-side build bar ---- #
    def private_landed(self) -> bool:
        return self.private is not None

    def private_bar(self, topo: str):
        h = _dig(self.private, "synthesis", "headline", default={}) or {}
        if topo == "both_bugs":
            v = h.get("lambda_star_lcb_private_both")
        else:
            v = h.get("lambda_star_lcb_private_descent")  # null => UNREACHABLE at full recovery
        return float(v) if v is not None else None

    def private_valid_at_bar(self) -> bool:
        return bool(_dig(self.private, "synthesis", "headline", "valid_at_bar", default=False))

    def descent_private_lcb_lambda1(self) -> float | None:
        m = _dig(self.private, "synthesis", "headline", "descent_private_lcb_margin_at_lambda1")
        return (TARGET_OFFICIAL + float(m)) if m is not None else None  # margin is vs 500

    def both_required_at_private_bar(self) -> bool:
        return bool(_dig(self.private, "synthesis", "headline",
                         "both_bugs_required_at_private_bar", default=False))

    def private_drop_pct(self) -> float | None:
        v = _dig(self.private, "synthesis", "headline", "private_drop_at_bar_pct")
        return float(v) if v is not None else None

    # ---- kanna #188 one-shot hardware bound ---- #
    def oneshot_landed(self) -> bool:
        return self.oneshot is not None

    def sigma_oneshot(self) -> float:
        return float(_dig(self.oneshot, "sigma_oneshot", default=V183.SIGMA_HW))

    # ---- denken #187 lambda_hat_built measurement-CI (input-side) ---- #
    def lambda_built_landed(self) -> bool:
        return self.lambda_built is not None

    def lambda_built_halfwidth(self) -> float:
        return float(_dig(self.lambda_built, "synthesis", "lambda_built_ci",
                          "lambda_built_halfwidth", default=0.0))

    def lambda_built_overlap(self) -> float | None:
        v = _dig(self.lambda_built, "synthesis", "input_output_compose", "overlap_fraction")
        return float(v) if v is not None else None

    # ---- kanna #194 official re-draw budget N* ---- #
    def redraw_landed(self) -> bool:
        return self.redraw is not None

    def n_shots_at_bar(self):
        return _dig(self.redraw, "n_shots_for_p95_at_bar")

    def n_shots_at_lambda1(self):
        return _dig(self.redraw, "n_shots_for_p95_at_lambda1")

    # ---- ubel #195 cross-axis CI covariance (MERGED 17:52Z: quadrature INVALID, de-dup) ---- #
    def covariance_landed(self) -> bool:
        return self.covariance is not None

    def quadrature_valid(self) -> bool:
        # False: rho(sampling,input-lambda)=0.945 is a double-count, not an independent axis.
        return bool(_dig(self.covariance, "combined", "quadrature_valid", default=True))

    def combined_sigma(self, kind: str) -> float | None:
        """The combined single-shot sigma (TPS). kind: 'quadrature' (naive, INVALID),
        'corrected' (full additive law incl the +rho double-count term -- the inflate artifact),
        'worstcase' (PSD-clamped, hardware<->acceptance coupling bounded), 'dedup' (PHYSICALLY-
        CORRECT: collapse sampling+input into ONE acceptance axis -> 3 independent axes)."""
        if kind == "dedup":
            v = _dig(self.covariance, "combined", "dedup_reading", "combined_sigma_dedup_tps")
        else:
            v = _dig(self.covariance, "combined", "combined_sigma_%s" % kind)
        return float(v) if v is not None else None

    def dedup_acceptance_block(self) -> float | None:
        v = _dig(self.covariance, "combined", "dedup_reading", "overlap_corrected_sampling_block_tps")
        return float(v) if v is not None else None

    def rho_sampling_input(self) -> float | None:
        v = _dig(self.covariance, "rho", "pairs", "sampling__input_lambda", "rho_central")
        return float(v) if v is not None else None

    def overlap_fraction_input_output(self) -> float | None:
        v = _dig(self.covariance, "axis_sigmas", "input_lambda", "overlap_fraction")
        return float(v) if v is not None else None

    def combined_z_p90(self) -> float:
        return float(_dig(self.covariance, "combined", "z_p90_one_sided", default=Z_P90))

    def combined_proj_private_both(self) -> float | None:
        v = _dig(self.covariance, "combined", "proj_private_tps_both")
        return float(v) if v is not None else None

    def lcb_shift_worstcase(self) -> float | None:
        v = _dig(self.covariance, "combined", "lcb_shift_worstcase_tps")
        return float(v) if v is not None else None

    def icc1_corner_sigma(self, kind: str) -> float | None:
        v = _dig(self.covariance, "combined", "icc1_corner", "combined_sigma_%s_icc1" % kind)
        return float(v) if v is not None else None

    def combined_launch_lcb(self, kind: str) -> float | None:
        """proj_private_central - z_p90 * combined_sigma(kind). The de-dup central is the BEST
        estimate; the worst-case is the conservative GO/NO-GO corner (advisor 17:52Z)."""
        central = self.combined_proj_private_both()
        sig = self.combined_sigma(kind)
        if central is None or sig is None:
            return None
        return float(central - self.combined_z_p90() * sig)

    # ---- kanna #200 cost-aware re-draw budget (MERGED 18:03Z: sequential spend + cost toggle) ---- #
    # ANNOTATION ONLY: does NOT touch the binding bar or the single-shot sigma; it prices the budget
    # row the human reads (build-higher-vs-stay-at-bar), reusing #194's N*(mu) cost-invariant frontier.
    def cost_budget_landed(self) -> bool:
        return self.cost_budget is not None

    def expected_shots_sequential_at_bar(self) -> float | None:
        v = _dig(self.cost_budget, "expected_shots_sequential_at_bar")
        return float(v) if v is not None else None

    def cost_optimal_n_at_bar(self):
        return _dig(self.cost_budget, "cost_optimal_n_at_bar")

    def p_single_clear_at_bar(self) -> float | None:
        v = _dig(self.cost_budget, "sequential_savings", "p_single_at_bar")
        return float(v) if v is not None else None

    def saved_shots_at_bar(self) -> float | None:
        v = _dig(self.cost_budget, "sequential_savings", "saved_shots_at_bar")
        return float(v) if v is not None else None

    def cost_crossover_fixedn_per_b(self) -> float | None:
        v = _dig(self.cost_budget, "cost_optimal", "c_star_slope_per_b_fixedN")
        return float(v) if v is not None else None

    def cost_crossover_sequential_per_b(self) -> float | None:
        v = _dig(self.cost_budget, "cost_optimal", "c_star_slope_per_b_sequential")
        return float(v) if v is not None else None

    def cost_crossover_total_shots(self) -> float | None:
        v = _dig(self.cost_budget, "cost_optimal", "crossover_total_shots")
        return float(v) if v is not None else None

    def cost_delta_mu_n1(self) -> float | None:
        v = _dig(self.cost_budget, "cost_optimal", "delta_mu_tps")
        return float(v) if v is not None else None

    def cost_per_mu_sequential_frontier(self):
        return _dig(self.cost_budget, "sequential_savings", "per_mu_sequential", default=[])

    # ---- ubel #201 launch-sigma closure (MERGED 18:23Z: de-dup x realistic-ICC -> launch sigma) ---- #
    # SUPERSEDES #195's 7.26/17.04 in the combined-sigma row. de-dup (#195) sets the acceptance-axis
    # IDENTITY (5.32 iid); realistic ICC (#190) sets its MAGNITUDE (5.32 * sqrt(D)=2.100 -> 11.17).
    # The trigger is held PROVISIONAL (advisor: do NOT hard-wire vs the lambda=1 ceiling) -- two open
    # levers (#204 unit-rebase, land #71 co-log n=385) finalize it.
    def sigma_closure_landed(self) -> bool:
        return self.sigma_closure is not None

    def combined_sigma_launch(self, kind: str) -> float | None:
        """The combined LAUNCH sigma (1-sigma TPS). kind: 'central' (rho(*,hw)=0) / 'worstcase'
        (rho(*,hw)=+0.3 PSD-clamped). These REPLACE #195's de-dup 7.26 / worst-case 17.04."""
        v = _dig(self.sigma_closure, "combined", "combined_sigma_launch_%s" % kind)
        return float(v) if v is not None else None

    def mu_clears_500(self, kind: str) -> float | None:
        """The P95 GO-trigger mu: the mu at which LCB(mu)=mu-z_p95*sigma_launch = 500."""
        v = _dig(self.sigma_closure, "lcb", "mu_clears_500_%s" % kind)
        return float(v) if v is not None else None

    def lambda1_ceiling_mu(self) -> float | None:
        v = _dig(self.sigma_closure, "lcb", "lambda1_ceiling_mu")
        return float(v) if v is not None else None

    def lambda1_clears_500(self, kind: str) -> bool:
        return bool(_dig(self.sigma_closure, "lcb", "lambda1_clears_500_%s" % kind, default=False))

    def margin_at_lambda1(self, kind: str) -> float | None:
        v = _dig(self.sigma_closure, "lcb", "%s_margin_at_lambda1_tps" % kind)
        return float(v) if v is not None else None

    def sigma_closure_z_p95(self) -> float:
        return float(_dig(self.sigma_closure, "lcb", "z_one_sided_p95", default=1.6448536269514722))

    def acceptance_sigma_dedup_iid(self) -> float | None:
        v = _dig(self.sigma_closure, "dedup_x_icc", "acceptance_sigma_dedup_iid")
        return float(v) if v is not None else None

    def acceptance_sigma_dedup_realistic_icc(self) -> float | None:
        v = _dig(self.sigma_closure, "dedup_x_icc", "acceptance_sigma_dedup_realistic_icc")
        return float(v) if v is not None else None

    def sqrt_design_effect(self) -> float | None:
        v = _dig(self.sigma_closure, "dedup_x_icc", "sqrt_design_effect_inflation")
        return float(v) if v is not None else None

    def design_effect_201(self) -> float | None:
        v = _dig(self.sigma_closure, "dedup_x_icc", "design_effect_algebra_1_plus_mbarm1_icc")
        return float(v) if v is not None else None

    def sigma_vector_leg(self, leg: str) -> float | None:
        v = _dig(self.sigma_closure, "combined", "sigma_vector_tps", leg)
        return float(v) if v is not None else None

    def headroom_shift_from_iid(self) -> float | None:
        v = _dig(self.sigma_closure, "lcb", "revises_194_break_even", "shift_tps")
        return float(v) if v is not None else None

    def iid_break_even_194(self) -> float | None:
        v = _dig(self.sigma_closure, "lcb", "revises_194_break_even", "iid_break_even_194")
        return float(v) if v is not None else None

    def icc0_combined_sigma_central(self) -> float | None:
        # ICC=0 corner reproduces #195's de-dup central 7.2617 -> proves #201 superset (applies ICC).
        v = _dig(self.sigma_closure, "icc_band_envelope", "icc0_iid", "combined_sigma_central")
        return float(v) if v is not None else None

    def colog_n_allocations(self):
        return _dig(self.sigma_closure, "colog_spec", "colog_n_allocations_for_rho_ci")

    def sigma_closure_rho_in_out(self) -> float | None:
        v = _dig(self.sigma_closure, "legs", "rho_in_out_195")
        return float(v) if v is not None else None

    def sigma_closure_overlap_fraction(self) -> float | None:
        v = _dig(self.sigma_closure, "legs", "overlap_fraction")
        return float(v) if v is not None else None

    def sigma_closure_unit_convention_note(self):
        return _dig(self.sigma_closure, "convention_note")

    # ---- ubel #189 executable submission gate (packaging precondition) ---- #
    def packaging_landed(self) -> bool:
        return self.packaging is not None

    def packaging_go_verdict(self) -> str | None:
        return _dig(self.packaging, "worked_go_example", "packaging_verdict")

    def packaging_trap(self) -> dict:
        bf = _dig(self.packaging, "worked_nogo_example", "binding_failure", default={}) or {}
        return {"flag": bf.get("flag"), "banked_cost_tps": bf.get("banked_cost_tps")}


_BANKED = _BankedAxes()


def numerical_ci_ledger(topo: str, tau: float = TAU_HEADLINE) -> list[dict]:
    """The numerical launch-CI axes as typed rows. RE-RUN (advisor 17:27Z + 17:52Z): #190/#187/#188/
    #191/#194 AND #195 have all MERGED -> each reads its REAL banked scalar (consumed); the ledger is
    now CLOSED (no pending axes). The iid #175 leg is the visible `iid-fallback` baseline. ubel #195
    proved the 4-axis quadrature INVALID: rho(sampling#175,input#187)=0.945 is a DOUBLE-COUNT, so the
    two are collapsed into ONE acceptance axis -> de-dup combined sigma 7.26 / worst-case 17.04."""
    public_bar = _M.lambda_star_lcb_183(topo, tau)          # #183 iid public bar (#175 (+) #159)
    rows = []

    # Axis 0 -- wirbel #175 iid sampling leg (LANDED): the +-10.906 numerator -- now the visible
    #           iid-fallback baseline (a LOWER bound; superseded as binding by #190's realistic ICC).
    rows.append({
        "axis": "sampling_iid", "pr": 175, "slug": "et-second-moment", "kind": "numerical",
        "status": "LANDED", "flag": "iid-fallback-visible",
        "halfwidth_tps_iid": _BANKED.halfwidth_iid(),
        "double_count_with": "input_lambda_built",
        "rho_with_input_lambda": (_finite(_BANKED.rho_sampling_input())
                                  if _BANKED.covariance_landed() else None),
        "note": "iid +-10.906 OUTPUT-side numerator leg (B=16384); a LOWER bound. ubel #195: rho=0.945 "
                "correlated with #187's INPUT-side lambda_hat CI (SAME accept draw) -> the two are "
                "COLLAPSED into ONE acceptance axis (5.32 TPS), NOT quadrature-summed. The BINDING "
                "sampling half-width is #190's realistic +-22.9.",
    })
    # Axis 1 -- wirbel #190 icc-neff (MERGED, CONSUMED): realistic within-prompt ICC shrinks N_eff,
    #           inflates the half-width iid->realistic and RAISES the bar 0.9052->0.9513.
    icc_ok = _BANKED.icc_landed()
    icc_bar_val = _BANKED.icc_bar() if (icc_ok and topo == "both_bugs") else public_bar
    rows.append({
        "axis": "sampling_icc", "pr": 190, "slug": "icc-neff-launch-ci", "kind": "numerical",
        "status": "LANDED" if icc_ok else "IN-FLIGHT",
        "flag": "consumed" if icc_ok else "iid-fallback",
        "lambda_bar": _finite(icc_bar_val) if icc_bar_val is not None else None,
        "halfwidth_realistic_tps": _BANKED.halfwidth_realistic() if icc_ok else None,
        "halfwidth_iid_tps": _BANKED.halfwidth_iid(),
        "icc_hat": _BANKED.icc_hat() if icc_ok else None,
        "design_effect": _BANKED.design_effect() if icc_ok else None,
        "realistic_launch_lcb_tps": _BANKED.realistic_launch_lcb(topo) if icc_ok else None,
        "realistic_launch_clears_500": _BANKED.realistic_launch_clears(topo) if icc_ok else None,
        "note": "iid +-10.9 was a LOWER bound; realistic ICC=%.4f -> half-width +-%.2f (Deff=%.2f). "
                "both-bugs ICC bar 0.9513; descent's realistic launch-LCB MISSES 500." % (
                    _BANKED.icc_hat() or float("nan"), _BANKED.halfwidth_realistic() or float("nan"),
                    _BANKED.design_effect() or float("nan")) if icc_ok else "iid-fallback (not landed).",
    })
    # Axis 2 -- denken #187 lambda-built-ci (MERGED, CONSUMED): INPUT-side resolvability of the bar.
    l187 = _BANKED.lambda_built_landed()
    rows.append({
        "axis": "input_lambda_built", "pr": 187, "slug": "lambda-built-ci", "kind": "numerical",
        "status": "LANDED" if l187 else "IN-FLIGHT",
        "flag": "consumed" if l187 else "pending-input-ci",
        "lambda_built_halfwidth": _BANKED.lambda_built_halfwidth() if l187 else 0.0,
        "overlap_fraction_with_175": _BANKED.lambda_built_overlap() if l187 else None,
        "double_count_with": "sampling_iid",
        "rho_with_sampling": (_finite(_BANKED.rho_sampling_input())
                              if (l187 and _BANKED.covariance_landed()) else None),
        "note": "INPUT-side CI on lambda_hat_built +-%.4f (89%% overlap with #175's OUTPUT-side). "
                "ubel #195: rho=0.945 (overlap=rho^2) -> a DOUBLE-COUNT, NOT an independent axis; "
                "collapsed with #175 into the acceptance axis (5.32 TPS). on-bar builds are "
                "measurement-unresolvable, so land #71 should build with margin." % (
                    _BANKED.lambda_built_halfwidth() if l187 else 0.0),
    })
    # Axis 3 -- kanna #188 oneshot-hw-bound (MERGED, CONSUMED): sigma_hw=4.86 confirmed correct for a
    #           SINGLE launch draw (within-run (+) between-device/thermal reproduces 4.86).
    o188 = _BANKED.oneshot_landed()
    rows.append({
        "axis": "hardware_oneshot", "pr": 188, "slug": "oneshot-hw-bound", "kind": "numerical",
        "status": "LANDED" if o188 else "IN-FLIGHT",
        "flag": "consumed" if o188 else "oneshot-fallback",
        "sigma_tps": _BANKED.sigma_oneshot() if o188 else V183.SIGMA_HW,
        "note": "sigma_oneshot=%.3f confirms #159's sigma_hw=4.86 IS the right single-draw sigma. "
                "ubel #195: the 3rd INDEPENDENT axis (after the de-dup'd acceptance axis) -> folds "
                "with acceptance to the combined sigma 7.26 (central) / 17.04 (worst-case, hw-coupling "
                "bounded). The hw<->acceptance coupling is UNMEASURED (within-device only)." % (
                    _BANKED.sigma_oneshot() if o188 else V183.SIGMA_HW),
    })
    # Axis 4 -- stark #191 private-build-bar (MERGED, CONSUMED): PRIVATE-side bar via #176 adverse-skew
    #           drop through #183's forward map. both-bugs 0.9780 (DOMINATES); descent UNREACHABLE.
    p191 = _BANKED.private_landed()
    priv_bar_val = _BANKED.private_bar(topo) if p191 else public_bar
    rows.append({
        "axis": "private_bar", "pr": 191, "slug": "private-build-bar", "kind": "numerical",
        "status": "LANDED" if p191 else "IN-FLIGHT",
        "flag": "consumed" if p191 else "public-fallback",
        "lambda_bar": _finite(priv_bar_val) if priv_bar_val is not None else None,
        "private_unreachable": bool(p191 and topo != "both_bugs" and priv_bar_val is None),
        "valid_at_bar": _BANKED.private_valid_at_bar() if p191 else None,
        "private_drop_pct": _BANKED.private_drop_pct() if p191 else None,
        "descent_private_lcb_at_lambda1": (_BANKED.descent_private_lcb_lambda1()
                                           if (p191 and topo != "both_bugs") else None),
        "note": ("private adverse-skew bar 0.9780 (both-bugs) DOMINATES public 0.9052 + ICC 0.9513 "
                 "-> binding. descent's private LCB tops out at 490.16<500 even at lambda=1 -> "
                 "UNREACHABLE (both_bugs_required_at_private_bar=True).") if p191 else "public-fallback.",
    })
    # Axis 5 -- kanna #194 official re-draw budget N* (MERGED, CONSUMED): informational best-of-N
    #           shot budget; does NOT enter binding_bar (it sizes the launch, not the bar). kanna
    #           #200 (MERGED 18:03Z) ANNOTATES it with the SEQUENTIAL early-stop spend + the
    #           build-higher-vs-stay cost toggle -- still NOT a binding axis (cost-invariant N*).
    r194 = _BANKED.redraw_landed()
    cba = cost_budget_annotation()
    c200 = bool(cba.get("landed"))
    rows.append({
        "axis": "redraw_budget", "pr": 194, "slug": "redraw-budget", "kind": "numerical-budget",
        "status": "LANDED" if r194 else "IN-FLIGHT",
        "flag": "consumed" if r194 else "single-shot-fallback",
        "n_shots_for_p95_at_bar": _BANKED.n_shots_at_bar() if r194 else None,
        "n_shots_for_p95_at_lambda1": _BANKED.n_shots_at_lambda1() if r194 else None,
        # kanna #200 cost annotation (sequential spend + cost toggle; cost-invariant N*).
        "cost_budget_200_landed": c200,
        "expected_shots_sequential_at_bar": (cba["stay_at_bar"]["expected_shots_sequential"]
                                             if c200 else None),
        "fixed_n_naive_at_bar": (cba["stay_at_bar"]["fixed_n_naive"] if c200 else None),
        "build_higher_mu_safe_n1_tps": (cba["build_higher"]["mu_safe_n1_tps"] if c200 else None),
        "build_vs_stay_crossover_total_shots": (cba["crossover_total_shots"] if c200 else None),
        "c_star_fixedN_per_b": (cba["c_star_fixedN_per_b"] if c200 else None),
        "c_star_sequential_per_b": (cba["c_star_sequential_per_b"] if c200 else None),
        "note": ("best-of-N official re-draw budget for P(clear-500)>=0.95: N*=%s at the build bar, "
                 "%s at full recovery lambda=1. kanna #200: the REALISTIC spend at the bar is the "
                 "SEQUENTIAL early-stop E[shots]=%.2f (NOT fixed-%s); build-higher (mu>=%.1f/N=1) beats "
                 "stay-at-bar iff reaching mu costs < %.0f shots' GPU-$ (c*=%.2f*b fixed / %.2f*b "
                 "sequential). Sizes/prices the launch budget; NOT a binding_bar axis (N* is "
                 "cost-invariant)." % (
                     _BANKED.n_shots_at_bar(), _BANKED.n_shots_at_lambda1(),
                     cba["stay_at_bar"]["expected_shots_sequential"],
                     cba["stay_at_bar"]["fixed_n_naive"], cba["build_higher"]["mu_safe_n1_tps"],
                     cba["crossover_total_shots"], cba["c_star_fixedN_per_b"],
                     cba["c_star_sequential_per_b"])) if (r194 and c200)
                else ("best-of-N official re-draw budget for P(clear-500)>=0.95: N*=%s at the build "
                      "bar, %s at full recovery lambda=1. Sizes the launch budget; not a binding_bar "
                      "axis. (kanna #200 cost annotation not landed.)" % (
                          _BANKED.n_shots_at_bar(), _BANKED.n_shots_at_lambda1())) if r194
                else "single-shot fallback (N=1).",
    })
    # Axis 6 -- ubel #195 cross-axis CI covariance (MERGED 17:52Z, CONSUMED): the quadrature is
    #           INVALID -- rho(sampling,input-lambda)=0.945 is a DOUBLE-COUNT. De-dup -> ONE acceptance
    #           axis (5.32 iid IDENTITY). This row reads #195's banked scalars DIRECTLY (decoupled
    #           from the #201 combined-sigma row, which supersedes the 7.26/17.04 single-shot numbers).
    #           CLOSES the ledger (no pending axes). The 7.26/17.04 here are #195's stand-alone reading;
    #           #201 rescales the acceptance axis by realistic ICC -> the launch-sigma row below.
    c195 = _BANKED.covariance_landed()
    cov_lcb_dedup = _BANKED.combined_launch_lcb("dedup") if c195 else None
    cov_lcb_worst = _BANKED.combined_launch_lcb("worstcase") if c195 else None
    rows.append({
        "axis": "cross_axis_covariance", "pr": 195, "slug": "ci-axis-covariance",
        "kind": "numerical-covariance",
        "status": "LANDED" if c195 else "IN-FLIGHT",
        "flag": "consumed" if c195 else "pending-covariance",
        "quadrature_valid": (_BANKED.quadrature_valid() if c195 else None),
        "rho_sampling_input": (_finite(_BANKED.rho_sampling_input()) if c195 else None),
        "sigma_quadrature_invalid_tps": (_finite(_BANKED.combined_sigma("quadrature")) if c195 else None),
        "sigma_dedup_central_tps": (_finite(_BANKED.combined_sigma("dedup")) if c195 else None),
        "sigma_worstcase_tps": (_finite(_BANKED.combined_sigma("worstcase")) if c195 else None),
        "acceptance_dedup_block_tps": (_finite(_BANKED.dedup_acceptance_block()) if c195 else None),
        "launch_lcb_dedup_central_tps": (_finite(cov_lcb_dedup) if cov_lcb_dedup is not None else None),
        "launch_lcb_worstcase_tps": (_finite(cov_lcb_worst) if cov_lcb_worst is not None else None),
        "worstcase_corner_clears_500": (bool(cov_lcb_worst is not None
                                             and cov_lcb_worst >= TARGET_OFFICIAL - 1e-9) if c195 else None),
        "superseded_by": "launch_sigma_closure_201" if c195 else None,
        "note": ("quadrature INVALID: rho(sampling#175,input#187)=0.945 double-count -> collapse to "
                 "ONE acceptance axis (5.32 TPS iid, the IDENTITY). #195's stand-alone de-dup combined "
                 "sigma 7.26 / worst-case 17.04 (NOT quadrature 12.54). ubel #201 SUPERSEDES these "
                 "single-shot numbers: it rescales the 5.32 acceptance axis by realistic ICC "
                 "(sqrt(D)=2.100 -> 11.17) -> the combined LAUNCH sigma row (12.22/13.80). CLOSES the "
                 "ledger (no pending numerical axes).") if c195
                else "not landed -> assume independence (quadrature); conservative until it lands.",
    })
    return rows


def binding_bar(topo: str, tau: float = TAU_HEADLINE) -> dict:
    """binding_bar = max(public #183, ICC-refined #190, private #191). RE-RUN (advisor 17:27Z):
    #190/#191 LANDED, so for both-bugs binding_bar = private 0.9780 (DOMINATES 0.9052 public /
    0.9513 ICC). For descent the private bar is UNREACHABLE (no lambda<=1 clears 500) -> binding
    is UNREACHABLE -> descent NO-GO on the build side. The iid 0.9052 is the visible fallback only."""
    ledger = numerical_ci_ledger(topo, tau)
    public = _M.lambda_star_lcb_183(topo, tau)
    icc_row = next(r for r in ledger if r["axis"] == "sampling_icc")
    priv_row = next(r for r in ledger if r["axis"] == "private_bar")
    icc = icc_row["lambda_bar"]
    private = priv_row["lambda_bar"]
    private_unreachable = bool(priv_row.get("private_unreachable"))

    candidates = [("public_183", public), ("icc_190", icc), ("private_191", private)]
    finite = [(s, v) for s, v in candidates if v is not None and math.isfinite(v)]
    if private_unreachable:
        # descent: private LCB never reaches 500 -> the bar does not exist -> UNREACHABLE.
        bar = float("inf")
        source = "private_191_UNREACHABLE"
    else:
        source, bar = max(finite, key=lambda kv: kv[1])
    icc_consumed = icc_row["flag"] == "consumed"
    priv_consumed = priv_row["flag"] == "consumed"
    binding_is_iid_fallback = bool(source == "public_183" and not icc_consumed and not priv_consumed)
    return {
        # UNREACHABLE -> None (NaN/inf-clean); the boolean flag + source string carry the meaning.
        "binding_bar": (None if private_unreachable else _finite(bar)),
        "binding_bar_unreachable": private_unreachable,
        "public_183": _finite(public),
        "icc_190": (_finite(icc) if icc is not None else None),
        "private_191": (_finite(private) if private is not None else None),
        "binding_source": source,
        "icc_flag": icc_row["flag"], "private_flag": priv_row["flag"],
        "binding_is_iid_fallback": binding_is_iid_fallback,
        # post-recompose: the binding bar is the REAL private bar (or UNREACHABLE), not the iid placeholder.
        "any_iid_fallback_active": binding_is_iid_fallback,
    }


def composed_lcb(lam: float, topo: str, tau: float = TAU_HEADLINE) -> float:
    """The #183 BUILD-gate composed LCB at lambda = central - z*sqrt(SE_tps^2 + sigma_hw^2) (iid
    sampling #175 (+) sigma_hw #159, confirmed one-shot by #188). This is the BUILD-side LCB that
    crosses 500 at lambda*_LCB; distinct from the #179/#190 LAUNCH-projection LCB (realistic ICC)."""
    return _finite(_M.metrics_183(lam, topo, tau)["lcb_full_tps"])


def realistic_launch_lcb(topo: str) -> dict:
    """The #179 LAUNCH-projection cell-LCB recomposed under #190's realistic +-22.9 ICC half-width
    (advisor 17:27Z: report the launch LCB under +-22.9, not iid +-10.9). Consumes #190's banked
    go_robustness section-4 LCB at full recovery (the launch operating point). Returns the binding
    realistic LCB + the iid-fallback cell-LCB for the visible optimistic leg."""
    rl = _BANKED.realistic_launch_lcb(topo)
    return {
        "realistic_lcb_tps": _finite(rl) if rl is not None else None,
        "realistic_clears_500": _BANKED.realistic_launch_clears(topo),
        "halfwidth_realistic_tps": _BANKED.halfwidth_realistic(),
        "halfwidth_iid_tps": _BANKED.halfwidth_iid(),
        "landed": _BANKED.icc_landed(),
    }


def combined_sigma_corner() -> dict:
    """The combined-sigma row. ubel #201 (MERGED 18:23Z, W&B spau6tch) SUPERSEDES ubel #195's
    7.26 / 17.04: the de-dup (#195) and realistic-ICC (#190) corrections are ORTHOGONAL on the SAME
    acceptance axis -- de-dup sets its IDENTITY (overlap-corrected 5.32 TPS iid, removing the
    rho(sampling#175,input#187)=0.945 double-count), realistic ICC sets its MAGNITUDE
    (design-effect D=1+(m_bar-1)*ICC=4.4106 -> sqrt(D)=2.100 -> 5.32 -> 11.17 TPS). Folding that with
    sigma_hw (#188 4.86) and sigma_private (#176/#191 0.88) gives the combined LAUNCH sigma
    **12.215 central / 13.796 worst-case** (rho(*,hw) bounded [-0.3,+0.3]), REPLACING the #195 numbers.

    P95 framing: the GO trigger is mu >= 500 + z_p95(1.6449)*sigma = **520.09 central / 522.69
    worst-case**, against the lambda=1 ceiling **520.95** TPS. Central is P95-reachable (+0.86 TPS),
    worst-case is P95-UNREACHABLE (-1.74 TPS), even at lambda=1. ICC erodes ~8 TPS of launch headroom,
    lifting the trigger off #194's iid break-even (512.16) onto/over the ceiling.

    HOLD PROVISIONAL (advisor 18:23Z: do NOT hard-wire GO/NO-GO vs the ceiling). Two open levers
    finalize it: (a) ubel #204 clean-1-sigma unit rebase (IN FLIGHT) may shift the central trigger
    ~3 TPS, direction OPEN -> can FLIP the central verdict (central sits only +0.86 under the ceiling);
    (b) land #71 co-log of per-allocation acceptance x wall-TPS (n=385 cross-device) retires the
    rho(*,hw) [-0.3,+0.3] band -> collapses [central, worst-case] onto a single trigger. ICC=0 here
    reproduces #195's de-dup central 7.2617 -> #201 is the strict superset (it applies ICC).
    (both-bugs only; descent is already NO-GO via #191.) NON-GATING: this row does NOT gate the
    analytic go -- it is the PROVISIONAL launch sigma->LCB readout the human reads."""
    if not _BANKED.sigma_closure_landed():
        return {"landed": False, "note": "launch_sigma_closure #201 not landed -> fall back to the "
                                         "#195 de-dup row (7.26/17.04) if present; else quadrature."}
    sc, wc = _BANKED.combined_sigma_launch("central"), _BANKED.combined_sigma_launch("worstcase")
    mu_c, mu_w = _BANKED.mu_clears_500("central"), _BANKED.mu_clears_500("worstcase")
    ceiling = _BANKED.lambda1_ceiling_mu()
    central_reach = _BANKED.lambda1_clears_500("central")     # True: 520.09 <= 520.95
    worst_reach = _BANKED.lambda1_clears_500("worstcase")     # False: 522.69 > 520.95
    return {
        "landed": True,
        "source_pr": 201,
        "supersedes_195_726_1704": True,
        "provisional": True,                                  # advisor 18:23Z: HOLD the verdict
        "gates_analytic_go": False,                           # NON-gating: not wired into `go`
        # ---- #201 headline combined LAUNCH sigma (1-sigma), REPLACES #195's 7.26 / 17.04 ----
        "combined_sigma_launch_central_tps": _finite(sc),     # 12.215
        "combined_sigma_launch_worstcase_tps": _finite(wc),   # 13.796
        # ---- P95 GO-trigger vs the lambda=1 ceiling ----
        "z_p95": _finite(_BANKED.sigma_closure_z_p95()),      # 1.6449
        "go_trigger_mu_central_tps": _finite(mu_c),           # 520.09
        "go_trigger_mu_worstcase_tps": _finite(mu_w),         # 522.69
        "lambda1_ceiling_mu_tps": _finite(ceiling),           # 520.95
        "central_p95_reachable": bool(central_reach),         # True  (+0.86)
        "worstcase_p95_reachable": bool(worst_reach),         # False (-1.74)
        "central_margin_at_lambda1_tps": _finite(_BANKED.margin_at_lambda1("central")),     # +0.86
        "worstcase_margin_at_lambda1_tps": _finite(_BANKED.margin_at_lambda1("worstcase")),  # -1.74
        # ---- the de-dup x ICC mechanism (SOLID -- banked) ----
        "acceptance_axis_dedup_iid_tps": _finite(_BANKED.acceptance_sigma_dedup_iid()),       # 5.32 (IDENTITY)
        "design_effect": _finite(_BANKED.design_effect_201()),                                # 4.4106
        "sqrt_design_effect": _finite(_BANKED.sqrt_design_effect()),                          # 2.100
        "acceptance_axis_realistic_icc_tps": _finite(_BANKED.acceptance_sigma_dedup_realistic_icc()),  # 11.17 (MAGNITUDE)
        "sigma_vector_tps": {
            "acceptance": _finite(_BANKED.sigma_vector_leg("acceptance")),                    # 11.17
            "hardware": _finite(_BANKED.sigma_vector_leg("hardware")),                        # 4.86
            "private": _finite(_BANKED.sigma_vector_leg("private")),                          # 0.88
        },
        # ---- headroom erosion vs #194's iid break-even ----
        "iid_break_even_194_tps": _finite(_BANKED.iid_break_even_194()),                      # 512.16
        "headroom_shift_tps": _finite(_BANKED.headroom_shift_from_iid()),                     # 7.94
        # ---- ICC=0 corner reproduces #195's de-dup central 7.2617 (proves the superset) ----
        "icc0_combined_sigma_central_tps": _finite(_BANKED.icc0_combined_sigma_central()),    # 7.2617 (= #195)
        # ---- #195 de-dup PROVENANCE (the mechanism that sets the acceptance-axis identity) ----
        "dedup_provenance_195": {
            "quadrature_valid": _BANKED.quadrature_valid() if _BANKED.covariance_landed() else False,
            "rho_sampling_input": _finite(_BANKED.sigma_closure_rho_in_out()),   # 0.945 double-count
            "overlap_fraction": _finite(_BANKED.sigma_closure_overlap_fraction()),  # 0.893 = rho^2
            "sigma_dedup_iid_195_tps": _finite(_BANKED.combined_sigma("dedup"))
                                       if _BANKED.covariance_landed() else _finite(
                                           _BANKED.icc0_combined_sigma_central()),  # 7.26
            "sigma_worstcase_195_tps": _finite(_BANKED.combined_sigma("worstcase"))
                                       if _BANKED.covariance_landed() else None,  # 17.04 (superseded)
        },
        # ---- two open levers that FINALIZE the PROVISIONAL trigger (advisor 18:23Z) ----
        "open_levers": {
            "ubel_204_unit_rebase": "clean-1-sigma unit rebase IN FLIGHT; may shift the central "
                "trigger ~3 TPS (direction OPEN) -> can FLIP the central verdict (central 520.09 "
                "sits only +0.86 under the 520.95 ceiling).",
            "land_71_colog": "co-log per-allocation acceptance x wall-TPS across n=%s cross-device "
                "allocations -> measures rho(*,hw) directly and RETIRES the [-0.3,+0.3] band, "
                "collapsing the [central, worst-case] interval onto a single trigger." % (
                    _BANKED.colog_n_allocations()),
        },
        "colog_n_allocations": _BANKED.colog_n_allocations(),  # 385
        "unit_convention_note": _BANKED.sigma_closure_unit_convention_note(),
        "verdict_line": "launch is on a knife-edge vs the lambda=1 ceiling -- P95-reachable at "
                        "central ICC (+%.2f TPS margin) but P95-UNREACHABLE at worst-case rho(*,hw) "
                        "(%.2f TPS), even at lambda=1; central verdict PENDING ubel #204's "
                        "unit-direction." % (
                            _BANKED.margin_at_lambda1("central") or float("nan"),
                            _BANKED.margin_at_lambda1("worstcase") or float("nan")),
        "note": "#201 REPLACES #195's 7.26/17.04 -> combined launch sigma %.2f central / %.2f "
                "worst-case (de-dup IDENTITY 5.32 * sqrt(D)=2.100 ICC MAGNITUDE -> acceptance 11.17, "
                "(+) sigma_hw 4.86 (+) sigma_priv 0.88). P95 GO trigger mu>=%.2f central / %.2f "
                "worst-case vs the lambda=1 ceiling %.2f -> central reachable (+%.2f), worst-case "
                "UNREACHABLE (%.2f). PROVISIONAL: HOLD the verdict (not hard-wired vs the ceiling) "
                "pending #204 unit-rebase (~3 TPS, direction open) + land #71 co-log (n=%s) retiring "
                "the rho(*,hw) band." % (
                    _finite(sc), _finite(wc), _finite(mu_c), _finite(mu_w), _finite(ceiling),
                    _BANKED.margin_at_lambda1("central") or float("nan"),
                    _BANKED.margin_at_lambda1("worstcase") or float("nan"),
                    _BANKED.colog_n_allocations()),
    }


def cost_budget_annotation() -> dict:
    """kanna #200 (MERGED 18:03Z): the cost-aware re-draw budget. ANNOTATION ONLY -- it does NOT
    touch the binding bar or the single-shot sigma; the single-shot GO/NO-GO is UNCHANGED. It
    prices the multi-shot budget ROW the human reads, on top of #194's cost-invariant N*(mu)
    frontier (N=5@500, N=1@mu>=512.2):
      (1) the REALISTIC spend at the bar is SEQUENTIAL early-stop = E[shots]=1.94 (NOT fixed-5),
          because ~half the draws clear on shot 1 (p_single=0.5);
      (2) a build-higher-vs-stay-at-bar TOGGLE: build to mu=512.2 / N=1 iff reaching mu=512.2 costs
          < 4 official shots' GPU-$ (fixed-N crossover c* = 3.04*b; sequential early-stop RAISES it
          to c* = 12.97*b -- early-stop substantially weakens the case for building higher);
      (3) N at a fixed mu is cost-INVARIANT (= #194), so single-shot logic is untouched."""
    if not _BANKED.cost_budget_landed():
        return {"landed": False, "note": "cost_budget #200 not landed -> budget row reports the "
                                         "naive fixed-N=5 spend (conservative; over-states the bill)."}
    e_seq = _BANKED.expected_shots_sequential_at_bar()
    n_bar = _BANKED.cost_optimal_n_at_bar()
    dmu = _BANKED.cost_delta_mu_n1()
    return {
        "landed": True,
        "single_shot_go_unchanged": True,            # advisor 18:03Z: annotation only.
        "stay_at_bar": {                             # build mu=500, best-of-N
            "mu_tps": float(TARGET_OFFICIAL),
            "n_max": n_bar,                          # = #194 N*(500) = 5
            "expected_shots_sequential": _finite(e_seq) if e_seq is not None else None,  # 1.94
            "fixed_n_naive": n_bar,                  # the naive (over-stated) fixed-5
            "p_single_clear": _finite(_BANKED.p_single_clear_at_bar())
                              if _BANKED.p_single_clear_at_bar() is not None else None,   # 0.5
            "saved_shots_vs_fixed": _finite(_BANKED.saved_shots_at_bar())
                                    if _BANKED.saved_shots_at_bar() is not None else None,  # 3.06
        },
        "build_higher": {                            # build mu>=512.2, N=1
            "mu_safe_n1_tps": float(TARGET_OFFICIAL + (dmu or 0.0)),  # 512.16
            "delta_mu_tps": _finite(dmu) if dmu is not None else None,  # 12.16
            "n_max": 1,
            "expected_shots_sequential": 1.0,
        },
        # the toggle: build-higher beats stay-at-bar once the per-shot $ exceeds these *b multiples.
        "crossover_total_shots": _finite(_BANKED.cost_crossover_total_shots())
                                 if _BANKED.cost_crossover_total_shots() is not None else None,  # 4.0
        "c_star_fixedN_per_b": _finite(_BANKED.cost_crossover_fixedn_per_b())
                               if _BANKED.cost_crossover_fixedn_per_b() is not None else None,  # 3.04
        "c_star_sequential_per_b": _finite(_BANKED.cost_crossover_sequential_per_b())
                                   if _BANKED.cost_crossover_sequential_per_b() is not None else None,  # 12.97
        "n_star_cost_invariant_194": True,           # N at fixed mu = #194 (cost-invariant).
        "per_mu_sequential_frontier": _BANKED.cost_per_mu_sequential_frontier(),
        "note": "budget row = {stay-at-bar mu=500 -> N_max=%s, pay E[shots]=%.2f sequential} vs "
                "{build-higher mu>=%.1f -> N=1}; pick the cheaper once land #71's build->mu cost (b) "
                "is banked. Build-higher wins iff reaching mu=%.1f costs < %.0f official shots' GPU-$ "
                "(fixed-N c*=%.2f*b; early-stop raises it to c*=%.2f*b). PRICES a plan; takes NO "
                "draws; authorizes nothing." % (
                    n_bar, e_seq if e_seq is not None else float("nan"),
                    TARGET_OFFICIAL + (dmu or 0.0), TARGET_OFFICIAL + (dmu or 0.0),
                    _BANKED.cost_crossover_total_shots() or float("nan"),
                    _BANKED.cost_crossover_fixedn_per_b() or float("nan"),
                    _BANKED.cost_crossover_sequential_per_b() or float("nan")),
    }


def precondition_ledger(build_lambda_ok: bool, build_topo: str) -> list[dict]:
    """Hard precondition rows (kind=precondition). Any NO-GO blocks an ACTUAL launch even if the
    composed-LCB clears. RE-RUN (advisor 17:27Z): ubel #189's executable packaging gate has MERGED
    -> consume its real GO verdict + failing-flag + banked cost-of-omission (the row-1 host-loop trap
    = 444.92 TPS), replacing the PENDING stub. Operational rows (PRECACHE serve-config, the gate's
    re-verification against land #71's REAL build introspection, human approval) keep the launch
    UN-authorized for a pre-launch run -> launch_authorized stays False."""
    p189 = _BANKED.packaging_landed()
    pkg_go = _BANKED.packaging_go_verdict()                  # "GO" for a faithful build (worked example)
    trap = _BANKED.packaging_trap()                          # host-loop binding failure + banked cost
    return [
        {"row": "boot_fix_kanna_177", "pr": 177, "kind": "precondition", "status": "GO",
         "flag": "landed", "note": "darwin _IncludedRouter boot-validation startup-500 fix; "
                                   "output-neutral (land #71 banks the diff)."},
        {"row": "precache_bench", "kind": "precondition", "status": "PENDING",
         "flag": "serve-config",
         "note": "PRECACHE_BENCH=1 must be set on the served path at launch."},
        {"row": "executable_submission_gate_ubel_189", "pr": 189, "kind": "precondition",
         # CONSUMED: the verifier exists. For a faithful both-bugs build the gate returns GO; it
         # must be RE-RUN against land #71's real introspection at launch (auto-catches the trap).
         "status": (pkg_go or "PENDING") if p189 else "PENDING",
         "flag": "consumed" if p189 else "pending-packaging",
         "verifier_landed": bool(p189),
         "faithful_build_verdict": pkg_go,
         "banked_trap_flag": trap.get("flag"),
         "banked_trap_cost_tps": trap.get("banked_cost_tps"),
         "note": "EXECUTABLE gate consumed (pqpb8ugk): a faithful device-vectorized relocate build -> "
                 "GO; auto-catches the row-1 relocate-as-host-Python-loop trap (banked %s TPS, ~85%% "
                 "cost). MUST be re-verified against land #71's REAL build introspection at launch." % (
                     ("%.2f" % trap["banked_cost_tps"]) if trap.get("banked_cost_tps") else "n/a")},
        {"row": "build_lambda_geq_binding_bar", "pr": 71, "kind": "precondition",
         "status": "GO" if build_lambda_ok else "NO-GO", "flag": "measured",
         "note": "land #71's measured lambda_hat_built >= binding_bar (%s) for topology %s." % (
             "0.9780 private" if build_topo == "both_bugs" else "UNREACHABLE", build_topo)},
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
    # RE-RUN (advisor 17:27Z): both-bugs binding = PRIVATE 0.9780 (dominates 0.9052 public / 0.9513
    # ICC); descent binding = UNREACHABLE (private LCB never reaches 500). The #183 public bar is the
    # visible iid-fallback only.
    bbar = binding_bar(topo, TAU_HEADLINE)
    binding_bar_val = bbar["binding_bar"]                            # 0.9780 both / inf descent
    binding_unreachable = bool(bbar["binding_bar_unreachable"])      # False both / True descent
    public_bar = _M.lambda_star_lcb_183(topo, TAU_HEADLINE)         # 0.9052 both / 0.9750 desc (iid-fallback)
    icc_bar = bbar["icc_190"]                                       # 0.9513 both
    lam_star_lcb_lo = _M.lambda_star_lcb_183(topo, TAU_CONSERVATIVE)  # tau floor
    lam_central_178 = _M.lambda_star_central_178(topo, TAU_HEADLINE)  # 0.8384 / 0.9091 (INACTIVE)
    # The visible iid-fallback "build-acceptable" check (descent clears 0.9750) AND the BINDING
    # private build gate (descent UNREACHABLE).
    public_build_gate_pass = bool(math.isfinite(public_bar) and lam_built >= public_bar - 1e-9)
    binding_build_gate_pass = bool((not binding_unreachable) and binding_bar_val is not None
                                   and math.isfinite(binding_bar_val)
                                   and lam_built >= binding_bar_val - 1e-9)
    build_gate_pass_cons = bool(math.isfinite(lam_star_lcb_lo) and lam_built >= lam_star_lcb_lo - 1e-9)
    build_lcb_at_lam = composed_lcb(lam_built, topo, TAU_HEADLINE)   # #183 build-LCB (divergence readout)

    # ---- LAUNCH-projection LCB under the REALISTIC +-22.9 ICC half-width (#190), NOT iid +-10.9 ----
    rl = realistic_launch_lcb(topo)                                  # 510.63 both / 495.04 descent
    realistic_lcb = rl["realistic_lcb_tps"]
    realistic_launch_pass = bool(rl["realistic_clears_500"])         # both True, descent False
    iid_cell_lcb = proj["lcb_p90"]                                   # 514.88 both / 499.97 desc (iid-fallback)

    # ---- PRIVATE launch LCB (#191): both-bugs valid_at_bar; descent UNREACHABLE (490.16<500) ----
    if topo == "both_bugs":
        private_launch_lcb = None                                    # valid (>=500) at the private bar
        private_launch_pass = bool(_BANKED.private_valid_at_bar())   # True
    else:
        private_launch_lcb = _BANKED.descent_private_lcb_lambda1()   # 490.16
        private_launch_pass = bool(private_launch_lcb is not None
                                   and private_launch_lcb >= TARGET_OFFICIAL - 1e-9)  # False

    # ---- combined-sigma row (#201 ubel, advisor 18:23Z): SUPERSEDES #195's 7.26/17.04. The de-dup
    #      (#195) acceptance axis (5.32 iid IDENTITY) evaluated under realistic ICC (#190 sqrt(D)=2.100
    #      MAGNITUDE -> 11.17) (+) sigma_hw (+) sigma_private -> combined LAUNCH sigma 12.215 central /
    #      13.796 worst-case. P95 GO trigger mu>=520.09 central / 522.69 worst-case vs the lambda=1
    #      ceiling 520.95 -> central P95-reachable (+0.86), worst-case UNREACHABLE (-1.74).
    #      "Wire the MECHANISM, HOLD the verdict": this row is PROVISIONAL and NON-GATING -- it does
    #      NOT enter `go` (do NOT hard-wire GO/NO-GO vs the ceiling). Two open levers finalize it
    #      (#204 unit-rebase ~3 TPS direction-open; land #71 co-log n=385 retires the rho(*,hw) band).
    #      both-bugs only; descent is already NO-GO via #191 so the row is non-binding there. ----
    if topo == "both_bugs":
        csc = combined_sigma_corner()
        lsc_landed = bool(csc.get("landed"))
        lsc_central_p95_reachable = bool(csc.get("central_p95_reachable")) if lsc_landed else None
        lsc_worstcase_p95_reachable = bool(csc.get("worstcase_p95_reachable")) if lsc_landed else None
        lsc_provisional = bool(csc.get("provisional", True)) if lsc_landed else None
    else:
        csc = None
        lsc_central_p95_reachable = None
        lsc_worstcase_p95_reachable = None
        lsc_provisional = None

    # ---- overall per-topology GO: validity AND trustworthy AND binding-build AND BOTH launch LCBs
    #      (realistic ICC #190 + private #191). The #201 combined-sigma row is PROVISIONAL/NON-GATING
    #      (advisor 18:23Z: HOLD the verdict -- do NOT hard-wire the GO/NO-GO vs the lambda=1 ceiling
    #      while #204 + land #71 are open) -> it is surfaced in lambda_gate but NOT folded into `go`. ----
    go = bool(validity_ok and oa["trustworthy"] and binding_build_gate_pass
              and realistic_launch_pass and private_launch_pass)

    # Failing-gate diagnosis + restoration lever (PR step 4).
    failing_gate, restoration = None, None
    if not validity_ok:
        failing_gate = "validity (PPL/128/greedy-exact)"
        restoration = "fix build quality to satisfy PPL<=2.42, 128/128, greedy-exact."
    elif not oa["trustworthy"]:
        failing_gate = "over-accept (#170): E[T] is NOT greedy-trustworthy"
        restoration = ("measured speedup is illusory (region=%s); re-bench under greedy-exact "
                       "decode before reading E[T] as headroom." % oa["region"])
    elif binding_unreachable or not binding_build_gate_pass:
        if binding_unreachable:
            failing_gate = ("binding build-gate (#191 private): PRIVATE-UNREACHABLE -- the adverse-"
                            "skew private LCB tops out at %.2f<500 even at lambda=1, so NO lambda<=1 "
                            "clears the private bar (both_bugs_required_at_private_bar=True). The "
                            "build clears the #183 public bar %.4f (iid-fallback) but that is "
                            "superseded by the private bar." % (
                                private_launch_lcb if private_launch_lcb is not None else float("nan"),
                                public_bar))
            restoration = ("ship #154 argmax-only decode (realizes step 1.2047, lifts the projection) "
                           "OR land the both-bugs kernel (the robust GO path). descent-only is "
                           "private-UNREACHABLE under the adverse-skew bar -- not a thin miss.")
        else:
            in_gap = bool(math.isfinite(lam_central_178) and lam_built >= lam_central_178 - 1e-9)
            failing_gate = ("binding build-gate (#191 private): lambda_hat_built=%.4f < binding_bar="
                            "%.4f (private; dominates ICC 0.9513 / public 0.9052)" % (
                                lam_built, binding_bar_val))
            restoration = (("lambda_hat_built clears the iid-fallback public bar %.4f but sits below "
                            "the BINDING private bar %.4f -> recover deeper self-KV or ship #154." % (
                                public_bar, binding_bar_val)) if in_gap else
                           ("recover deeper self-KV: lambda_hat_built=%.4f is below the binding bar "
                            "%.4f." % (lam_built, binding_bar_val)))
    elif not realistic_launch_pass:
        failing_gate = ("clear-500 launch-projection LCB under realistic ICC (#190): LCB=%.2f<500 at "
                        "the realistic +-%.2f half-width (the iid +-%.2f LCB %.2f was optimistic). "
                        "Binding build-gate PASSED -- a build-acceptable kernel that misses the "
                        "correlation-refined launch projection." % (
                            realistic_lcb, rl["halfwidth_realistic_tps"], rl["halfwidth_iid_tps"],
                            iid_cell_lcb))
        restoration = ("ship #154 argmax-only decode (realizes step 1.2047) or land the both-bugs "
                       "kernel; this path misses once within-prompt correlation widens the CI.")
    elif not private_launch_pass:
        failing_gate = ("clear-500 launch-projection LCB on the PRIVATE axis (#191): private LCB=%.2f"
                        "<500 even at lambda=1." % (
                            private_launch_lcb if private_launch_lcb is not None else float("nan")))
        restoration = "land the both-bugs kernel (the robust GO path); this path is private-unreachable."
    # NOTE: the #201 combined-sigma row is PROVISIONAL/NON-GATING (advisor 18:23Z) -> it is NOT a
    # failing-gate candidate. A worst-case-P95-unreachable corner does NOT flip the analytic verdict.

    return {
        "topo": topo,
        "GO": go,
        "verdict": "GO" if go else "NO-GO",
        **proj,
        "overaccept": oa,
        "lambda_gate": {
            "lambda_hat_built": _finite(lam_built),
            "binding_bar": bbar,                                            # max(public,ICC,private)
            "binding_bar_value": (None if binding_unreachable else _finite(binding_bar_val)),
            "binding_bar_unreachable": binding_unreachable,
            "binding_source": bbar["binding_source"],                       # private_191 (both)
            "public_bar_183_iid_fallback": _finite(public_bar),             # 0.9052 / 0.9750 (visible only)
            "icc_bar_190": (_finite(icc_bar) if icc_bar is not None else None),  # 0.9513 (both)
            "lambda_star_lcb_183_conservative": _finite(lam_star_lcb_lo),   # tau=0.9924
            "lambda_star_central_178_INACTIVE": _finite(lam_central_178),   # looser reference only
            "public_build_gate_pass": public_build_gate_pass,              # descent clears the iid-fallback
            "build_gate_pass": binding_build_gate_pass,                    # BINDING (private) build gate
            "build_gate_pass_conservative": build_gate_pass_cons,
            "build_lcb_at_lambda_built": build_lcb_at_lam,
            # launch-projection LCBs: realistic (binding) + iid cell (fallback) + private.
            "realistic_launch_lcb_tps": _finite(realistic_lcb) if realistic_lcb is not None else None,
            "realistic_launch_clears_500": realistic_launch_pass,
            "halfwidth_realistic_tps": _finite(rl["halfwidth_realistic_tps"]),
            "halfwidth_iid_tps": _finite(rl["halfwidth_iid_tps"]),
            "iid_cell_lcb_p90_fallback": _finite(iid_cell_lcb),            # 514.88 / 499.97 (optimistic)
            "private_launch_lcb_tps": (_finite(private_launch_lcb)
                                       if private_launch_lcb is not None else None),
            "private_launch_clears_500": private_launch_pass,
            "clear500_launch_lcb_pass": bool(realistic_launch_pass and private_launch_pass),
            # combined-sigma row (#201, SUPERSEDES #195): the de-dup x realistic-ICC launch sigma
            # 12.215/13.796 -> P95 GO trigger 520.09/522.69 vs the lambda=1 ceiling 520.95.
            # PROVISIONAL + NON-GATING (advisor 18:23Z: HOLD the verdict). both-bugs only (None on descent).
            "combined_sigma_corner": csc,
            "launch_sigma_central_p95_reachable": lsc_central_p95_reachable,     # True  (+0.86)
            "launch_sigma_worstcase_p95_reachable": lsc_worstcase_p95_reachable,  # False (-1.74)
            "launch_sigma_provisional": lsc_provisional,                         # HOLD pending #204 + land #71
            "launch_sigma_gates_go": False,                                      # advisor 18:23Z: NOT hard-wired
            "numerical_ci_ledger": numerical_ci_ledger(topo, TAU_HEADLINE),
            "binding_rule": "lambda_hat_built >= binding_bar = max(public#183 0.9052, ICC#190 0.9513, "
                            "private#191 0.9780) = 0.9780 (private, both-bugs); descent private bar "
                            "UNREACHABLE. AND launch-projection LCB >= 500 under realistic +-22.9 (#190) "
                            "AND on the private axis (#191).",
            "two_lcb_divergence": (
                "BUILD gate (binding private bar) and LAUNCH-projection LCB (realistic ICC + private) "
                "are DIFFERENT bars. both-bugs clears all (robust GO). descent clears the iid-fallback "
                "public build bar (%.4f) but is private-UNREACHABLE on the build side AND misses the "
                "realistic (%.2f) and private (%.2f) launch LCBs -> NO-GO, doubly hardened. "
                "public_build=%s binding_build=%s realistic_launch=%s private_launch=%s" % (
                    public_bar,
                    realistic_lcb if realistic_lcb is not None else float("nan"),
                    private_launch_lcb if private_launch_lcb is not None else float("nan"),
                    public_build_gate_pass, binding_build_gate_pass, realistic_launch_pass,
                    private_launch_pass)),
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
        "rule": "per-topology GO iff lambda_hat_built >= binding_bar(topo) = max(public#183, "
                "ICC#190, private#191) [RECOMPOSED: both-bugs 0.9780 private (dominates 0.9052 "
                "public / 0.9513 ICC); descent private-UNREACHABLE] AND the #179 launch-projection "
                "cell-LCB(P>=0.9) >= 500 under the REALISTIC #190 +-22.9 half-width (not iid +-10.9) "
                "AND on the #191 private axis. descent is doubly hardened NO-GO (UNREACHABLE build "
                "bar + misses realistic 495.04 and private 490.16 launch LCBs) -> #154 / both-bugs.",
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
    _hb = headline_binding["binding_bar"]
    build_lambda_ok = bool(_hb is not None and math.isfinite(_hb) and lam_built >= _hb - 1e-9)
    preconds = precondition_ledger(build_lambda_ok, headline)
    preconds_all_go = all(r["status"] == "GO" for r in preconds)
    # launch_authorized = analytic-GO AND every precondition GO. Operational rows (PRECACHE
    # serve-config, #189 packaging re-verify, human approval) are PENDING pre-launch -> authorizes NOTHING.
    launch_authorized = bool(overall_go and preconds_all_go)
    # RE-RUN: the binding bar is the REAL private bar (not the iid placeholder). ALL numerical axes
    # (including ubel #195 cross-axis covariance) have landed -> the ledger is CLOSED, no pending axes.
    # The iid #175 leg is a visible fallback row only, not a binding fallback.
    bb_ledger = numerical_ci_ledger("both_bugs", TAU_HEADLINE)
    pending_axes = [r["axis"] for r in bb_ledger
                    if r["axis"] not in ("sampling_iid",) and r["flag"] != "consumed"]
    ledger_closed = bool(len(pending_axes) == 0)
    binding_on_iid_fallback = bool(bb_binding["binding_is_iid_fallback"])
    any_iid_fallback = binding_on_iid_fallback                       # binding bar no longer iid -> False
    # combined-sigma row (#201, SUPERSEDES #195): the de-dup x realistic-ICC launch sigma -> P95
    # GO-trigger-vs-ceiling readout. PROVISIONAL + NON-GATING (advisor 18:23Z: HOLD the verdict).
    csc_ledger = combined_sigma_corner()
    # cost-aware re-draw budget (#200): sequential early-stop spend + build-higher-vs-stay toggle.
    # ANNOTATION ONLY -- single-shot GO/NO-GO + binding bar + sigma are unchanged.
    cba_ledger = cost_budget_annotation()

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
            "net_rule": "launch_authorized = (analytic GO: lambda_hat_built >= binding_bar AND the "
                        "realistic-ICC launch-LCB >= 500 AND the private-axis launch-LCB >= 500 AND "
                        "validity AND over-accept) AND (all precondition rows GO). The #201 "
                        "combined-sigma row is PROVISIONAL + NON-GATING (advisor 18:23Z: HOLD the "
                        "verdict -- it is surfaced, NOT folded into the analytic GO). RECOMPOSED "
                        "post-merge: binding_bar = max(public#183 0.9052, ICC#190 0.9513, private#191 "
                        "0.9780) = 0.9780 (private#191 dominates, both-bugs); descent private is "
                        "UNREACHABLE. Sampling half-width is the REALISTIC ICC#190 +/-22.9 TPS "
                        "(design-effect %.4f over iid +/-10.9), NOT the iid placeholder. CLOSED "
                        "post-#195 (the 4-axis quadrature was INVALID: rho(sampling,input)=0.945 "
                        "double-count). #201 SUPERSEDES #195's 7.26/17.04: the de-dup acceptance axis "
                        "(5.32 iid IDENTITY) under realistic ICC (sqrt(D)=2.100 MAGNITUDE -> 11.17), "
                        "(+) sigma_hw (+) sigma_private -> combined LAUNCH sigma 12.215 central / "
                        "13.796 worst-case; P95 GO trigger mu>=520.09 central / 522.69 worst-case vs "
                        "the lambda=1 ceiling 520.95 -> central P95-reachable (+0.86), worst-case "
                        "UNREACHABLE (-1.74). PROVISIONAL pending ubel #204 (unit-rebase ~3 TPS, "
                        "direction open) + land #71 co-log (n=385) retiring the rho(*,hw) band. iid "
                        "#175 leg kept as a visible fallback row only. NO pending numerical axes "
                        "(ledger CLOSED). #200 cost annotation (budget row only, single-shot logic "
                        "UNCHANGED): realistic spend at the bar is SEQUENTIAL E[shots]=1.94 (not "
                        "fixed-5); build-higher (mu>=512.2/N=1) beats stay-at-bar iff reaching mu "
                        "costs < 4 shots' GPU-$ (c*=3.04*b fixed / 12.97*b sequential)."
                        % _BANKED.design_effect(),
            "numerical_axes": {"both_bugs": bb_ledger,
                               "descent_only": numerical_ci_ledger("descent_only", TAU_HEADLINE)},
            "combined_sigma_corner": csc_ledger,
            "cost_budget_annotation": cba_ledger,
            "preconditions": preconds,
            "preconditions_all_go": preconds_all_go,
            "any_iid_fallback_active": any_iid_fallback,
            "binding_on_iid_fallback": binding_on_iid_fallback,
            "pending_numerical_axes": pending_axes,
            "sole_pending_axis": (pending_axes[0] if len(pending_axes) == 1 else None),
            "all_binding_axes_landed": (not any_iid_fallback),
            "ledger_closed": ledger_closed,
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
    """TEST: the BUILD-gate verdict evaluated at EXACTLY this topology's binding_bar. RECOMPOSED:
    binding_bar = max(public#183, ICC#190, private#191) = 0.9780 (private, both-bugs). A build at
    the bar inverts back to lambda_hat_built == binding_bar, so the build-gate passes inclusively
    -> True. For descent binding_bar is None (UNREACHABLE) -> False (no lambda<=1 clears the private
    bar). Uses binding_bar so the metric tracks the recomposition (private dominant)."""
    lam_star = binding_bar(topo, TAU_HEADLINE)["binding_bar"]
    if lam_star is None or not math.isfinite(lam_star):
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
                    r.get("halfwidth_tps_iid", r.get("n_shots_for_p95_at_bar")))))
        val_s = ("%.4f" % val) if isinstance(val, (int, float)) else (
            str(val) if val is not None else "pending")
        num_md.append("| %s | #%s | %s | %s | %s |" % (
            r["axis"], r["pr"], r["status"], r["flag"], val_s))
    num_table = "\n".join(num_md)
    pre_md = ["| precondition | status | flag |", "|---|---|---|"]
    for r in led["preconditions"]:
        pre_md.append("| %s | %s | %s |" % (r["row"], r["status"], r["flag"]))
    pre_table = "\n".join(pre_md)
    # combined-sigma worst-case corner (#195) + cost-aware budget (#200) annotation lines.
    csc = led.get("combined_sigma_corner") or {}
    cba = led.get("cost_budget_annotation") or {}
    if csc.get("landed"):
        sigma_line = (
            "**Combined-sigma row (#201, SUPERSEDES #195's 7.26/17.04 -- PROVISIONAL, does NOT gate "
            "the verdict):** de-dup (#195) sets the acceptance-axis IDENTITY (5.32 TPS iid, removing "
            "the rho(sampling,input)=%.3f double-count); realistic ICC (#190) sets its MAGNITUDE "
            "(sqrt(D)=%.3f -> %.2f TPS). (+) sigma_hw (+) sigma_private -> combined LAUNCH sigma "
            "**%.2f central / %.2f worst-case** (1-sigma). P95 GO trigger mu >= 500 + z_p95*sigma = "
            "**%.2f central / %.2f worst-case** vs the lambda=1 ceiling **%.2f** -> central is "
            "P95-reachable (**+%.2f TPS**) but worst-case is P95-UNREACHABLE (**%.2f TPS**), even at "
            "lambda=1. ICC erodes ~%.1f TPS of launch headroom (lifts #194's iid break-even %.2f). "
            "**Knife-edge** -- HELD PROVISIONAL pending two open levers: ubel #204 (clean-1-sigma "
            "unit rebase, IN FLIGHT, ~3 TPS, direction OPEN -> can FLIP the central verdict) + land "
            "#71 co-log (n=%s cross-device allocations retire the rho(*,hw) [-0.3,+0.3] band). NO "
            "change to the binding BUILD bar (private 0.9780, #191) -- purely the launch sigma->LCB row." % (
                csc["dedup_provenance_195"]["rho_sampling_input"], csc["sqrt_design_effect"],
                csc["acceptance_axis_realistic_icc_tps"],
                csc["combined_sigma_launch_central_tps"], csc["combined_sigma_launch_worstcase_tps"],
                csc["go_trigger_mu_central_tps"], csc["go_trigger_mu_worstcase_tps"],
                csc["lambda1_ceiling_mu_tps"], csc["central_margin_at_lambda1_tps"],
                csc["worstcase_margin_at_lambda1_tps"], csc["headroom_shift_tps"],
                csc["iid_break_even_194_tps"], csc["colog_n_allocations"]))
    else:
        sigma_line = "**Combined-sigma row (#201):** not landed -> #195 de-dup fallback (7.26/17.04)."
    if cba.get("landed"):
        budget_line = (
            "**Multi-shot budget (#200, cost-aware -- prices the spend, single-shot GO/NO-GO "
            "UNCHANGED):** {stay-at-bar mu=%.0f -> N_max=%s, pay **E[shots]=%.2f sequential** "
            "(early-stop, ~half clear on shot 1) -- NOT fixed-%s} **vs** {build-higher mu>=%.1f -> "
            "N=1}. Pick the cheaper once land #71's build->mu cost (b) is banked: build-higher wins "
            "iff reaching mu>=%.1f costs < %.0f official shots' GPU-$ (crossover c*=%.2f*b fixed-N; "
            "early-stop raises it to c*=%.2f*b). N at a fixed mu is cost-invariant (= #194)." % (
                cba["stay_at_bar"]["mu_tps"], cba["stay_at_bar"]["n_max"],
                cba["stay_at_bar"]["expected_shots_sequential"], cba["stay_at_bar"]["fixed_n_naive"],
                cba["build_higher"]["mu_safe_n1_tps"], cba["build_higher"]["mu_safe_n1_tps"],
                cba["crossover_total_shots"], cba["c_star_fixedN_per_b"],
                cba["c_star_sequential_per_b"]))
    else:
        budget_line = "**Multi-shot budget (#200):** not landed -> naive fixed-N=5 spend (over-stated)."
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
(= max(public#183 {binding['public_183']:.4f}, ICC#190 {binding['icc_190']:.4f}, private#191 {binding['private_191']:.4f});
source {binding['binding_source']}; private#191 DOMINATES, STRICTER than #178's central 0.838), and
(2) the #179 launch-projection cell-LCB(P>=0.9) >= 500 under the REALISTIC #190 +-{lg['halfwidth_realistic_tps']:.2f}
ICC half-width (realistic launch-LCB {lg['realistic_launch_lcb_tps']:.2f} TPS; the optimistic iid
+-{lg['halfwidth_iid_tps']:.2f} cell-LCB {lg['iid_cell_lcb_p90_fallback']:.2f} is the visible fallback) AND
clears the #191 private axis. sigma_hw RETIRED on a separate hardware axis by best-of-2 official
draws (P={out['hardware_axis_sigma_hw']['best_of_2_p']:.4f}>=0.90).

**Measured-tuple GO table:**
{table_md}

**Launch-CI ledger -- numerical axes (binding_bar = max; #190/#191/#188/#187/#194/#195 ALL CONSUMED; iid #175 leg visible; ledger CLOSED; #201 combined-sigma row below):**
{num_table}

**Launch-CI ledger -- hard precondition rows (any non-GO blocks the launch):**
{pre_table}

{sigma_line}

{budget_line}

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
  - [BANKED] denken #183 margin-aware lambda-card (PUBLIC iid leg): {binding['public_183']:.4f} (tau=1) / {lg['lambda_star_lcb_183_conservative']:.4f} (tau=0.9924).
  - [LANDED-CONSUMED] launch-CI axes #190 (ICC bar {binding['icc_190']:.4f}, realistic +-{lg['halfwidth_realistic_tps']:.2f}) / #187 (lambda-built-CI) / #188 (sigma-oneshot) / #191 (private bar {binding['private_191']:.4f}, BINDING). binding_bar {binding['binding_bar']:.4f}.
  - [LANDED-CONSUMED] kanna #194 re-draw budget N* (informational; sizes the launch, not the bar).
  - [LANDED-CONSUMED] kanna #200 cost-aware budget: sequential E[shots]=1.94 at the bar (not fixed-5) + build-higher-vs-stay toggle (c*=3.04*b fixed / 12.97*b sequential). Annotation only; single-shot GO/NO-GO unchanged.
  - [LANDED-CONSUMED] ubel #189 executable-submission-gate: faithful build -> GO; re-verify vs land #71 at launch.
  - [LANDED-CONSUMED] ubel #195 cross-axis CI covariance -- quadrature INVALID (rho(sampling,input)=0.945 double-count) -> de-dup acceptance axis 5.32 iid (sets the IDENTITY; the 7.26/17.04 single-shot numbers are now #201's ICC=0 corner).
  - [LANDED-CONSUMED] ubel #201 launch-sigma closure (PROVISIONAL, NON-GATING) -- de-dup x realistic-ICC combined LAUNCH sigma 12.215 central / 13.796 worst-case; P95 GO trigger 520.09/522.69 vs the lambda=1 ceiling 520.95 (central reachable +0.86, worst-case UNREACHABLE -1.74). HOLD the verdict pending ubel #204 (unit-rebase) + land #71 co-log (n=385).
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
                 "the BINDING build-gate (lambda_hat_built >= binding_bar = max(public#183, ICC#190, "
                 "private#191 0.9780)) AND the launch-projection cell-LCB(P>=0.9) >= 500 under the "
                 "realistic #190 +-22.9 ICC half-width AND on the #191 private axis for a robust GO.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------- #
# STEP 5 -- self-test (PRIMARY metric).
# ----------------------------------------------------------------------------------------- #
def self_test() -> dict:
    results: dict[str, Any] = {}
    tol = 0.05

    # (a) lambda=1: both-bugs GO SURVIVES the RECOMPOSED binding_bar 0.9780 (private#191, dominates
    #     0.9052 public / 0.9513 ICC) under the REALISTIC +-22.9 ICC half-width (#190 launch LCB
    #     510.63 >= 500 AND private valid). descent is a DOUBLY-HARDENED NO-GO: private build bar
    #     UNREACHABLE (no lambda<=1 clears) AND the realistic 495.04 + private 490.16 launch LCBs
    #     both miss 500. The iid cell-LCBs (514.88 both / 499.97 descent) remain the visible fallback.
    t_full = synth_land71_tuple("self-test-lambda1", 1.0)
    d_full = launch_decision(t_full, step_override=_M.shipped_step)
    bb_lcb = d_full["per_topology"]["both_bugs"]["lcb_p90"]
    do_lcb = d_full["per_topology"]["descent_only"]["lcb_p90"]
    bb_g = d_full["_full_per_topology"]["both_bugs"]["lambda_gate"]
    do_g = d_full["_full_per_topology"]["descent_only"]["lambda_gate"]
    a_ok = (abs(bb_lcb - 514.877540689496) <= tol and abs(do_lcb - 499.96519706601964) <= tol
            and abs(bb_g["binding_bar_value"] - 0.9780112973731208) <= 1e-3
            and not bool(bb_g["binding_bar_unreachable"])
            and bb_g["binding_source"] == "private_191"
            and bool(bb_g["build_gate_pass"]) and bool(bb_g["realistic_launch_clears_500"])
            and abs(bb_g["realistic_launch_lcb_tps"] - 510.626905623982) <= tol
            and abs(bb_g["halfwidth_realistic_tps"] - 22.90457745058264) <= 1e-2
            and bool(do_g["binding_bar_unreachable"]) and not bool(do_g["build_gate_pass"])
            and not bool(do_g["realistic_launch_clears_500"])
            and abs(do_g["realistic_launch_lcb_tps"] - 495.0395325187374) <= tol
            and not bool(do_g["private_launch_clears_500"])
            and abs(do_g["private_launch_lcb_tps"] - 490.16268553751826) <= tol
            and d_full["per_topology"]["both_bugs"]["verdict"] == "GO"
            and d_full["per_topology"]["descent_only"]["verdict"] == "NO-GO"
            and d_full["verdict"] == "GO" and d_full["headline_topology"] == "both_bugs")
    results["a_lambda1_both_go_survives_recomposed_binding"] = {
        "pass": bool(a_ok), "both_lcb_iid_fallback": bb_lcb, "descent_lcb_iid_fallback": do_lcb,
        "both_binding_bar": bb_g["binding_bar_value"], "both_binding_anchor": 0.9780112973731208,
        "both_binding_source": bb_g["binding_source"],
        "both_realistic_launch_lcb": bb_g["realistic_launch_lcb_tps"], "both_realistic_anchor": 510.626905623982,
        "halfwidth_realistic_tps": bb_g["halfwidth_realistic_tps"],
        "descent_binding_unreachable": bool(do_g["binding_bar_unreachable"]),
        "descent_realistic_launch_lcb": do_g["realistic_launch_lcb_tps"], "descent_realistic_anchor": 495.0395325187374,
        "descent_private_launch_lcb": do_g["private_launch_lcb_tps"], "descent_private_anchor": 490.16268553751826,
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

    # (e) #183 REAL card consumed: its lambda*_LCB is now the PUBLIC iid-fallback leg (both 0.9052 /
    #     descent 0.9750 at tau=1; 0.9234/0.9926 at tau_cons) -- SUPERSEDED as binding by private#191.
    #     card monotone; forward map 0.9052->500.0 & 0.342->404.14; #178 central bar (0.8384/0.9091)
    #     reported but flagged INACTIVE.
    lg = d_full["lambda_gate_logic"]
    bb_lg, do_lg = lg["both_bugs"], lg["descent_only"]
    fm_bb = V183.forward_map(_M.ctx183, *_M.qfqF_183("both_bugs"), TAU_HEADLINE,
                             _M.lam_hat_liveprobe, _M.lambda_star_lcb_183("both_bugs"),
                             _M.lambda_star_central_178("both_bugs"))
    row_lcb = next(r for r in fm_bb["rows"]
                   if abs(r["lambda"] - round(_M.lambda_star_lcb_183("both_bugs"), 5)) < 1e-9)
    row_live = next(r for r in fm_bb["rows"]
                    if abs(r["lambda"] - round(_M.lam_hat_liveprobe, 5)) < 1e-9)
    e_ok = (abs(bb_lg["public_bar_183_iid_fallback"] - 0.905229319301184) <= 1e-3
            and abs(do_lg["public_bar_183_iid_fallback"] - 0.9750199960244741) <= 1e-3
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
    results["e_183_real_card_consumed_as_public_leg"] = {
        "pass": bool(e_ok),
        "both_public_bar_iid_fallback_tau1": bb_lg["public_bar_183_iid_fallback"],
        "descent_public_bar_iid_fallback_tau1": do_lg["public_bar_183_iid_fallback"],
        "both_lambda_star_lcb_tau_cons": bb_lg["lambda_star_lcb_183_conservative"],
        "both_central_178_inactive": bb_lg["lambda_star_central_178_INACTIVE"],
        "card_is_monotone": bool(lg["card_is_monotone"]),
        "fwd_lcb_at_lambda_star": row_lcb["predicted_lcb_tps"],
        "fwd_lcb_at_liveprobe": row_live["predicted_lcb_tps"],
        "183_flag": "LANDED -- CONSUMED (public iid-fallback leg)"}

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

    # (g) two-LCB divergence, RECOMPOSED + DOUBLY-HARDENED: at lambda=1 descent PASSES the visible
    #     PUBLIC iid-fallback build bar (1.0 >= 0.9750) but FAILS the BINDING private build gate
    #     (UNREACHABLE) AND BOTH launch LCBs -- realistic (495.04 < 500) and private (490.16 < 500).
    #     both-bugs clears all four (the robust GO). public/binding-build divergence is the new axis.
    do_full_lg = d_full["_full_per_topology"]["descent_only"]["lambda_gate"]
    bb_full_lg = d_full["_full_per_topology"]["both_bugs"]["lambda_gate"]
    g_ok = (bool(do_full_lg["public_build_gate_pass"])
            and bool(do_full_lg["binding_bar_unreachable"])
            and not bool(do_full_lg["build_gate_pass"])
            and not bool(do_full_lg["realistic_launch_clears_500"])
            and not bool(do_full_lg["private_launch_clears_500"])
            and not bool(do_full_lg["clear500_launch_lcb_pass"])
            and bool(bb_full_lg["build_gate_pass"])
            and bool(bb_full_lg["realistic_launch_clears_500"])
            and bool(bb_full_lg["clear500_launch_lcb_pass"]))
    results["g_two_lcb_divergence_doubly_hardened"] = {
        "pass": bool(g_ok),
        "descent_public_build_gate_pass": bool(do_full_lg["public_build_gate_pass"]),
        "descent_binding_build_unreachable": bool(do_full_lg["binding_bar_unreachable"]),
        "descent_binding_build_gate_pass": bool(do_full_lg["build_gate_pass"]),
        "descent_realistic_launch_pass": bool(do_full_lg["realistic_launch_clears_500"]),
        "descent_private_launch_pass": bool(do_full_lg["private_launch_clears_500"]),
        "descent_launch_lcb_pass": bool(do_full_lg["clear500_launch_lcb_pass"]),
        "both_binding_build_gate_pass": bool(bb_full_lg["build_gate_pass"]),
        "both_realistic_launch_pass": bool(bb_full_lg["realistic_launch_clears_500"]),
        "both_launch_lcb_pass": bool(bb_full_lg["clear500_launch_lcb_pass"])}

    # (h) launch-CI ledger RECOMPOSED + CLOSED (advisor 17:27Z + 17:52Z RE-RUN): ALL six numerical
    #     axes (#190/#187/#188/#191/#194 + #195 cross-axis covariance) have LANDED -> flag "consumed";
    #     binding_bar = max() = private 0.9780 (both) and UNREACHABLE (descent); the iid #175 leg is
    #     the SOLE visible fallback row; NO pending axis (ledger_closed); the #189 packaging
    #     precondition is GO/consumed; launch_authorized==False (precache + human-approval PENDING)
    #     while analytic verdict==GO. The binding bar is NO LONGER on the iid fallback.
    led = d_full["launch_ci_ledger"]
    axes_bb = {r["axis"]: r for r in led["numerical_axes"]["both_bugs"]}
    consumed_axes = {"sampling_icc", "input_lambda_built", "hardware_oneshot", "private_bar",
                     "redraw_budget", "cross_axis_covariance"}
    all_consumed = all(axes_bb[a]["status"] == "LANDED" and axes_bb[a]["flag"] == "consumed"
                       for a in consumed_axes)
    iid_row = axes_bb["sampling_iid"]
    cov_row = axes_bb["cross_axis_covariance"]
    bb_bar = d_full["binding_bar"]["both_bugs"]
    do_bar = d_full["binding_bar"]["descent_only"]
    pack_row = next(r for r in led["preconditions"]
                    if r["row"] == "executable_submission_gate_ubel_189")
    h_ok = (all_consumed
            and iid_row["flag"] == "iid-fallback-visible" and iid_row["status"] == "LANDED"
            and cov_row["flag"] == "consumed" and cov_row["status"] == "LANDED"
            and cov_row["quadrature_valid"] is False
            and abs(bb_bar["binding_bar"] - 0.9780112973731208) <= 1e-3
            and bb_bar["binding_source"] == "private_191"
            and bb_bar["binding_is_iid_fallback"] is False
            and bool(do_bar["binding_bar_unreachable"])
            and do_bar["binding_source"] == "private_191_UNREACHABLE"
            and abs(axes_bb["sampling_icc"]["halfwidth_realistic_tps"] - 22.90457745058264) <= 1e-2
            and pack_row["status"] == "GO" and pack_row["flag"] == "consumed"
            and bool(pack_row["verifier_landed"])
            and led["any_iid_fallback_active"] is False
            and led["all_binding_axes_landed"] is True
            and led["pending_numerical_axes"] == []
            and led["sole_pending_axis"] is None
            and led["ledger_closed"] is True
            and d_full["launch_authorized"]["authorized"] is False
            and d_full["launch_authorized"]["analytic_verdict"] == "GO")
    results["h_launch_ci_ledger_recomposed_closed"] = {
        "pass": bool(h_ok),
        "binding_bar_both": bb_bar["binding_bar"], "binding_source_both": bb_bar["binding_source"],
        "binding_bar_descent_unreachable": bool(do_bar["binding_bar_unreachable"]),
        "binding_source_descent": do_bar["binding_source"],
        "all_numerical_axes_consumed": bool(all_consumed),
        "cov_row_flag": cov_row["flag"], "cov_row_status": cov_row["status"],
        "iid_fallback_row_visible": iid_row["flag"],
        "n_pending_numerical_axes": len(led["pending_numerical_axes"]),
        "sole_pending_axis": led["sole_pending_axis"],
        "ledger_closed": led["ledger_closed"],
        "any_iid_fallback_active": led["any_iid_fallback_active"],
        "packaging_189_status": pack_row["status"], "packaging_189_flag": pack_row["flag"],
        "launch_authorized": d_full["launch_authorized"]["authorized"],
        "analytic_verdict": d_full["launch_authorized"]["analytic_verdict"]}

    # (i) combined-sigma row CLOSURE (advisor 18:23Z, ubel #201): SUPERSEDES #195's 7.26/17.04. The
    #     de-dup acceptance axis (#195, IDENTITY 5.32 iid) x realistic ICC (#190, MAGNITUDE
    #     sqrt(D)=2.100 -> 11.17) (+) sigma_hw (+) sigma_private -> combined LAUNCH sigma 12.215
    #     central / 13.796 worst-case. P95 GO trigger mu>=520.09 central / 522.69 worst-case vs the
    #     lambda=1 ceiling 520.95 -> central P95-reachable (+0.86), worst-case UNREACHABLE (-1.74).
    #     ICC=0 here reproduces #195's de-dup central 7.2617 (proves the superset). The row is
    #     PROVISIONAL + NON-GATING: "wire the mechanism, HOLD the verdict" -- the worst-case
    #     P95-unreachable corner does NOT flip both-bugs GO (the verdict is held, not hard-wired vs the
    #     ceiling), pending ubel #204 (unit-rebase) + land #71 co-log (n=385) retiring the rho(*,hw) band.
    csc = led["combined_sigma_corner"]
    bb_full_g = d_full["_full_per_topology"]["both_bugs"]["lambda_gate"]
    do_full_g = d_full["_full_per_topology"]["descent_only"]["lambda_gate"]
    prov = csc["dedup_provenance_195"]
    i_ok = (bool(csc["landed"]) and csc["source_pr"] == 201
            and csc["supersedes_195_726_1704"] is True
            and csc["provisional"] is True and csc["gates_analytic_go"] is False
            # #201 headline launch sigma REPLACES #195's 7.26/17.04:
            and abs(csc["combined_sigma_launch_central_tps"] - 12.215) <= 5e-2
            and abs(csc["combined_sigma_launch_worstcase_tps"] - 13.796) <= 5e-2
            # P95 GO trigger vs the lambda=1 ceiling:
            and abs(csc["z_p95"] - 1.6449) <= 5e-3
            and abs(csc["go_trigger_mu_central_tps"] - 520.092) <= 5e-2
            and abs(csc["go_trigger_mu_worstcase_tps"] - 522.692) <= 5e-2
            and abs(csc["lambda1_ceiling_mu_tps"] - 520.953) <= 5e-2
            and csc["central_p95_reachable"] is True
            and csc["worstcase_p95_reachable"] is False
            and abs(csc["central_margin_at_lambda1_tps"] - 0.860) <= 5e-2
            and abs(csc["worstcase_margin_at_lambda1_tps"] - (-1.739)) <= 5e-2
            # de-dup x ICC mechanism (de-dup IDENTITY * sqrt(D) MAGNITUDE):
            and abs(csc["acceptance_axis_dedup_iid_tps"] - 5.319) <= 5e-2
            and abs(csc["design_effect"] - 4.4106) <= 5e-3
            and abs(csc["sqrt_design_effect"] - 2.1001) <= 5e-3
            and abs(csc["acceptance_axis_realistic_icc_tps"] - 11.170) <= 5e-2
            and abs(csc["headroom_shift_tps"] - 7.935) <= 5e-2
            # ICC=0 corner reproduces #195's de-dup central 7.2617 (the superset proof):
            and abs(csc["icc0_combined_sigma_central_tps"] - 7.2617) <= 5e-3
            # #195 de-dup provenance retained (rho double-count that sets the acceptance IDENTITY):
            and abs(prov["rho_sampling_input"] - 0.9449) <= 5e-3
            and abs(prov["overlap_fraction"] - 0.893) <= 5e-3
            and prov["quadrature_valid"] is False
            and csc["colog_n_allocations"] == 385
            # lambda_gate surfaces the PROVISIONAL/NON-GATING readout (both-bugs only):
            and bb_full_g["combined_sigma_corner"] is not None
            and bb_full_g["launch_sigma_central_p95_reachable"] is True
            and bb_full_g["launch_sigma_worstcase_p95_reachable"] is False
            and bb_full_g["launch_sigma_provisional"] is True
            and bb_full_g["launch_sigma_gates_go"] is False
            and do_full_g["combined_sigma_corner"] is None
            and do_full_g["launch_sigma_central_p95_reachable"] is None
            # THE HELD-VERDICT INVARIANT: worst-case P95-UNREACHABLE, yet both-bugs is STILL GO
            # (the row is non-gating -> the analytic verdict is not hard-wired vs the ceiling):
            and d_full["per_topology"]["both_bugs"]["verdict"] == "GO"
            and d_full["verdict"] == "GO")
    results["i_combined_sigma_closure_201_provisional"] = {
        "pass": bool(i_ok),
        "source_pr": csc["source_pr"], "supersedes_195": csc["supersedes_195_726_1704"],
        "provisional": csc["provisional"], "gates_analytic_go": csc["gates_analytic_go"],
        "combined_sigma_launch_central_tps": csc["combined_sigma_launch_central_tps"],
        "combined_sigma_launch_worstcase_tps": csc["combined_sigma_launch_worstcase_tps"],
        "go_trigger_mu_central_tps": csc["go_trigger_mu_central_tps"],
        "go_trigger_mu_worstcase_tps": csc["go_trigger_mu_worstcase_tps"],
        "lambda1_ceiling_mu_tps": csc["lambda1_ceiling_mu_tps"],
        "central_p95_reachable": csc["central_p95_reachable"],
        "worstcase_p95_reachable": csc["worstcase_p95_reachable"],
        "central_margin_at_lambda1_tps": csc["central_margin_at_lambda1_tps"],
        "worstcase_margin_at_lambda1_tps": csc["worstcase_margin_at_lambda1_tps"],
        "acceptance_axis_realistic_icc_tps": csc["acceptance_axis_realistic_icc_tps"],
        "icc0_reproduces_195_dedup_726": abs(csc["icc0_combined_sigma_central_tps"] - 7.2617) <= 5e-3,
        "headroom_shift_tps": csc["headroom_shift_tps"],
        "colog_n_allocations": csc["colog_n_allocations"],
        "verdict_held_go_despite_worstcase_unreachable": (
            d_full["per_topology"]["both_bugs"]["verdict"] == "GO"
            and csc["worstcase_p95_reachable"] is False)}

    # (j) cost-aware re-draw budget ANNOTATION (advisor 18:03Z, kanna #200): the budget ROW the human
    #     reads is priced -- the REALISTIC spend at the bar is SEQUENTIAL early-stop E[shots]=1.9375
    #     (NOT fixed-5; ~half clear on shot 1, p_single=0.5), and a build-higher-vs-stay TOGGLE
    #     (build mu>=512.16/N=1 iff reaching it costs < 4 official shots' GPU-$; crossover c*=3.04*b
    #     fixed-N, 12.97*b sequential early-stop). N at a fixed mu is cost-INVARIANT (= #194), so the
    #     SINGLE-SHOT GO/NO-GO is UNCHANGED: both-bugs GO, descent NO-GO, overall GO -- exactly as
    #     before #200 (annotation only, no bar/sigma change).
    cba = d_full["launch_ci_ledger"]["cost_budget_annotation"]
    redraw_row = next(r for r in d_full["launch_ci_ledger"]["numerical_axes"]["both_bugs"]
                      if r["axis"] == "redraw_budget")
    j_ok = (bool(cba["landed"]) and cba["single_shot_go_unchanged"] is True
            and abs(cba["stay_at_bar"]["expected_shots_sequential"] - 1.9375) <= 5e-3
            and cba["stay_at_bar"]["n_max"] == 5
            and cba["stay_at_bar"]["fixed_n_naive"] == 5
            and abs(cba["stay_at_bar"]["p_single_clear"] - 0.5) <= 5e-3
            and abs(cba["build_higher"]["mu_safe_n1_tps"] - 512.157071171610028) <= 5e-2
            and cba["build_higher"]["n_max"] == 1
            and abs(cba["crossover_total_shots"] - 4.0) <= 1e-6
            and abs(cba["c_star_fixedN_per_b"] - 3.039267792902507) <= 5e-3
            and abs(cba["c_star_sequential_per_b"] - 12.967542583050696) <= 5e-3
            and cba["n_star_cost_invariant_194"] is True
            and bool(redraw_row["cost_budget_200_landed"])
            and abs(redraw_row["expected_shots_sequential_at_bar"] - 1.9375) <= 5e-3
            # single-shot verdict UNCHANGED by the annotation:
            and d_full["per_topology"]["both_bugs"]["verdict"] == "GO"
            and d_full["per_topology"]["descent_only"]["verdict"] == "NO-GO"
            and d_full["verdict"] == "GO")
    results["j_cost_budget_annotation_200"] = {
        "pass": bool(j_ok),
        "expected_shots_sequential_at_bar": cba["stay_at_bar"]["expected_shots_sequential"],
        "fixed_n_naive_at_bar": cba["stay_at_bar"]["fixed_n_naive"],
        "p_single_clear_at_bar": cba["stay_at_bar"]["p_single_clear"],
        "build_higher_mu_safe_n1_tps": cba["build_higher"]["mu_safe_n1_tps"],
        "crossover_total_shots": cba["crossover_total_shots"],
        "c_star_fixedN_per_b": cba["c_star_fixedN_per_b"],
        "c_star_sequential_per_b": cba["c_star_sequential_per_b"],
        "single_shot_go_unchanged": cba["single_shot_go_unchanged"],
        "verdict_unchanged_go": d_full["verdict"] == "GO"}

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
    bb_bind = binding_bar("both_bugs", TAU_HEADLINE)
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group,
                     config={"k_cal": K_CAL, "ppl_bar": PPL_BAR, "target_official": TARGET_OFFICIAL,
                             "step_shipped": _M.shipped_step,
                             "public_bar_183_iid_fallback_both_tau1": _M.lambda_star_lcb_183("both_bugs"),
                             "public_bar_183_iid_fallback_descent_tau1": _M.lambda_star_lcb_183("descent_only"),
                             "binding_bar_both": bb_bind["binding_bar"],
                             "binding_bar_icc_190": bb_bind["icc_190"],
                             "binding_bar_private_191": bb_bind["private_191"],
                             "binding_source_both": bb_bind["binding_source"],
                             "halfwidth_iid_tps": _BANKED.halfwidth_iid(),
                             "halfwidth_realistic_tps": _BANKED.halfwidth_realistic(),
                             "icc_hat": _BANKED.icc_hat(), "design_effect": _BANKED.design_effect(),
                             # combined-sigma row (#201, SUPERSEDES #195's 7.26/17.04): de-dup x
                             # realistic-ICC launch sigma -> P95 GO-trigger-vs-ceiling. PROVISIONAL.
                             "combined_sigma_launch_central_tps_201": _BANKED.combined_sigma_launch("central"),
                             "combined_sigma_launch_worstcase_tps_201": _BANKED.combined_sigma_launch("worstcase"),
                             "go_trigger_mu_central_tps_201": _BANKED.mu_clears_500("central"),
                             "go_trigger_mu_worstcase_tps_201": _BANKED.mu_clears_500("worstcase"),
                             "lambda1_ceiling_mu_tps_201": _BANKED.lambda1_ceiling_mu(),
                             "central_p95_reachable_201": _BANKED.lambda1_clears_500("central"),
                             "worstcase_p95_reachable_201": _BANKED.lambda1_clears_500("worstcase"),
                             "central_margin_at_lambda1_tps_201": _BANKED.margin_at_lambda1("central"),
                             "worstcase_margin_at_lambda1_tps_201": _BANKED.margin_at_lambda1("worstcase"),
                             "acceptance_axis_dedup_iid_tps_201": _BANKED.acceptance_sigma_dedup_iid(),
                             "acceptance_axis_realistic_icc_tps_201": _BANKED.acceptance_sigma_dedup_realistic_icc(),
                             "sqrt_design_effect_201": _BANKED.sqrt_design_effect(),
                             "headroom_shift_tps_201": _BANKED.headroom_shift_from_iid(),
                             "icc0_combined_sigma_central_tps_201": _BANKED.icc0_combined_sigma_central(),
                             "colog_n_allocations_201": _BANKED.colog_n_allocations(),
                             # #195 de-dup provenance (sets the acceptance-axis IDENTITY; sigma SUPERSEDED by #201).
                             "combined_sigma_quadrature_invalid_tps_195": _BANKED.combined_sigma("quadrature"),
                             "combined_sigma_dedup_central_tps_195": _BANKED.combined_sigma("dedup"),
                             "combined_sigma_worstcase_tps_195_superseded": _BANKED.combined_sigma("worstcase"),
                             "combined_sigma_quadrature_valid": _BANKED.quadrature_valid(),
                             "rho_sampling_input_195": _BANKED.rho_sampling_input(),
                             "acceptance_dedup_block_tps_195": _BANKED.dedup_acceptance_block(),
                             # kanna #200 cost-aware budget annotation (single-shot logic unchanged).
                             "expected_shots_sequential_at_bar_200": _BANKED.expected_shots_sequential_at_bar(),
                             "cost_optimal_n_at_bar_200": _BANKED.cost_optimal_n_at_bar(),
                             "build_vs_stay_crossover_total_shots_200": _BANKED.cost_crossover_total_shots(),
                             "c_star_fixedN_per_b_200": _BANKED.cost_crossover_fixedn_per_b(),
                             "c_star_sequential_per_b_200": _BANKED.cost_crossover_sequential_per_b()})
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
    led = wp["launch_ci_ledger"]
    flat["worked_example/any_iid_fallback_active"] = bool(led["any_iid_fallback_active"])
    flat["worked_example/all_binding_axes_landed"] = bool(led["all_binding_axes_landed"])
    flat["worked_example/sole_pending_axis"] = str(led["sole_pending_axis"])
    flat["worked_example/n_pending_numerical_axes"] = len(led["pending_numerical_axes"])
    flat["worked_example/ledger_closed"] = bool(led["ledger_closed"])
    # combined-sigma row (#201, SUPERSEDES #195): launch sigma -> P95 GO-trigger-vs-ceiling. PROVISIONAL.
    csc = led["combined_sigma_corner"]
    if csc.get("landed"):
        flat["worked_example/combined_sigma_launch_central_tps"] = csc["combined_sigma_launch_central_tps"]
        flat["worked_example/combined_sigma_launch_worstcase_tps"] = csc["combined_sigma_launch_worstcase_tps"]
        flat["worked_example/go_trigger_mu_central_tps"] = csc["go_trigger_mu_central_tps"]
        flat["worked_example/go_trigger_mu_worstcase_tps"] = csc["go_trigger_mu_worstcase_tps"]
        flat["worked_example/lambda1_ceiling_mu_tps"] = csc["lambda1_ceiling_mu_tps"]
        flat["worked_example/central_p95_reachable"] = bool(csc["central_p95_reachable"])
        flat["worked_example/worstcase_p95_reachable"] = bool(csc["worstcase_p95_reachable"])
        flat["worked_example/central_margin_at_lambda1_tps"] = csc["central_margin_at_lambda1_tps"]
        flat["worked_example/worstcase_margin_at_lambda1_tps"] = csc["worstcase_margin_at_lambda1_tps"]
        flat["worked_example/acceptance_axis_realistic_icc_tps"] = csc["acceptance_axis_realistic_icc_tps"]
        flat["worked_example/sqrt_design_effect"] = csc["sqrt_design_effect"]
        flat["worked_example/headroom_shift_tps"] = csc["headroom_shift_tps"]
        flat["worked_example/icc0_combined_sigma_central_tps"] = csc["icc0_combined_sigma_central_tps"]
        flat["worked_example/combined_sigma_provisional"] = bool(csc["provisional"])
        flat["worked_example/combined_sigma_gates_analytic_go"] = bool(csc["gates_analytic_go"])
        flat["worked_example/rho_sampling_input_195_provenance"] = csc["dedup_provenance_195"]["rho_sampling_input"]
    # cost-aware re-draw budget annotation (#200): sequential spend + build-vs-stay cost toggle.
    cba = led["cost_budget_annotation"]
    if cba.get("landed"):
        flat["worked_example/cost_expected_shots_sequential_at_bar"] = cba["stay_at_bar"]["expected_shots_sequential"]
        flat["worked_example/cost_fixed_n_naive_at_bar"] = cba["stay_at_bar"]["fixed_n_naive"]
        flat["worked_example/cost_build_higher_mu_safe_n1_tps"] = cba["build_higher"]["mu_safe_n1_tps"]
        flat["worked_example/cost_crossover_total_shots"] = cba["crossover_total_shots"]
        flat["worked_example/cost_c_star_fixedN_per_b"] = cba["c_star_fixedN_per_b"]
        flat["worked_example/cost_c_star_sequential_per_b"] = cba["c_star_sequential_per_b"]
        flat["worked_example/cost_single_shot_go_unchanged"] = bool(cba["single_shot_go_unchanged"])
    for topo in ("both_bugs", "descent_only"):
        c = wp["per_topology"][topo]
        lgc = wp["_full_per_topology"][topo]["lambda_gate"]
        bb = wp["binding_bar"][topo]
        flat[f"worked_example/{topo}/official_tps"] = c["official_tps"]
        flat[f"worked_example/{topo}/lcb_p90_iid_fallback"] = c["lcb_p90"]
        flat[f"worked_example/{topo}/p_clear_500"] = c["p_clear_500"]
        flat[f"worked_example/{topo}/lambda_hat_built"] = lgc["lambda_hat_built"]
        flat[f"worked_example/{topo}/public_bar_183_iid_fallback"] = lgc["public_bar_183_iid_fallback"]
        # binding_bar is None for the UNREACHABLE (descent) topology -> log the flag, not a float.
        flat[f"worked_example/{topo}/binding_bar_unreachable"] = bool(bb["binding_bar_unreachable"])
        if bb["binding_bar"] is not None and math.isfinite(bb["binding_bar"]):
            flat[f"worked_example/{topo}/binding_bar"] = bb["binding_bar"]
        flat[f"worked_example/{topo}/binding_source"] = bb["binding_source"]
        flat[f"worked_example/{topo}/build_gate_pass"] = bool(lgc["build_gate_pass"])
        flat[f"worked_example/{topo}/realistic_launch_clears_500"] = bool(
            lgc["realistic_launch_clears_500"])
        if lgc["realistic_launch_lcb_tps"] is not None:
            flat[f"worked_example/{topo}/realistic_launch_lcb_tps"] = lgc["realistic_launch_lcb_tps"]
        flat[f"worked_example/{topo}/halfwidth_realistic_tps"] = lgc["halfwidth_realistic_tps"]
        flat[f"worked_example/{topo}/halfwidth_iid_tps"] = lgc["halfwidth_iid_tps"]
        flat[f"worked_example/{topo}/private_launch_clears_500"] = bool(
            lgc["private_launch_clears_500"])
        if lgc["private_launch_lcb_tps"] is not None:
            flat[f"worked_example/{topo}/private_launch_lcb_tps"] = lgc["private_launch_lcb_tps"]
        flat[f"worked_example/{topo}/clear500_launch_lcb_pass"] = bool(
            lgc["clear500_launch_lcb_pass"])
        # #201 combined-sigma row: PROVISIONAL + NON-GATING readout (None on descent -> skip).
        if lgc.get("launch_sigma_central_p95_reachable") is not None:
            flat[f"worked_example/{topo}/launch_sigma_central_p95_reachable"] = bool(
                lgc["launch_sigma_central_p95_reachable"])
            flat[f"worked_example/{topo}/launch_sigma_worstcase_p95_reachable"] = bool(
                lgc["launch_sigma_worstcase_p95_reachable"])
            flat[f"worked_example/{topo}/launch_sigma_provisional"] = bool(
                lgc["launch_sigma_provisional"])
            flat[f"worked_example/{topo}/launch_sigma_gates_go"] = bool(lgc["launch_sigma_gates_go"])
    wandb.log(flat)
    wandb.summary.update(flat)
    run.finish()


def run(args) -> dict:
    t0 = time.time()
    st = self_test()
    # worked example = land #71 at full self-KV recovery (lambda=1) -- the GO-path demonstration.
    worked = launch_decision(synth_land71_tuple("land-71-bothbugs-kernel", 1.0),
                             step_override=_M.shipped_step)
    bb_bind = binding_bar("both_bugs", TAU_HEADLINE)
    csc = combined_sigma_corner()
    cba = cost_budget_annotation()
    handoff = (
        "launch_trigger_calculator: one-call launch_decision(measured_tuple) -> verified "
        "GO/NO-GO + filled (un-filed) Approval request. Self-test %s. RECOMPOSED post-merge "
        "(advisor 17:27Z + 17:52Z + 18:03Z + 18:23Z): the TYPED launch-CI ledger reads REAL banked "
        "scalars -- #190 ICC, #191 private, #188 sigma-oneshot, #187 input-side, #194 re-draw budget, "
        "#195 cross-axis covariance, #200 cost AND #201 launch-sigma closure have ALL LANDED "
        "(flag=consumed); the ledger is CLOSED. binding_bar = max(public#183 %.4f, ICC#190 %.4f, "
        "private#191 %.4f) = %.4f (private DOMINATES, both-bugs); descent private bar is UNREACHABLE. "
        "land #71 must show lambda_hat_built >= %.4f AND the #179 launch-projection cell-LCB(P>=0.9) "
        ">= 500 under the REALISTIC #190 +-%.2f ICC half-width (NOT iid +-%.2f) AND on the #191 "
        "private axis. launch_authorized = (analytic-GO AND all precondition rows GO); preconditions "
        "(PRECACHE_BENCH, ubel #189 packaging-gate=GO but re-verify-pending, human approval) PENDING "
        "-> launch_authorized=False (authorizes nothing). both-bugs SURVIVES at lambda=1: realistic "
        "launch-LCB 510.63>=500 + private valid -> robust GO (HELD). descent is DOUBLY-HARDENED NO-GO: "
        "private build bar UNREACHABLE AND misses realistic 495.04 + private 490.16 launch LCBs. "
        "COMBINED-SIGMA ROW (#201, advisor 18:23Z -- SUPERSEDES #195's 7.26/17.04; PROVISIONAL + "
        "NON-GATING): de-dup (#195, rho=%.3f double-count -> IDENTITY 5.32 iid) x realistic ICC (#190, "
        "MAGNITUDE -> %.2f) (+) sigma_hw (+) sigma_private -> combined LAUNCH sigma %.2f central / "
        "%.2f worst-case; P95 GO trigger mu>=%.2f central / %.2f worst-case vs the lambda=1 ceiling "
        "%.2f -> central P95-reachable (+%.2f) but worst-case UNREACHABLE (%.2f), even at lambda=1 "
        "(ICC erodes ~%.1f TPS of launch headroom). KNIFE-EDGE: WIRE the mechanism, HOLD the verdict "
        "-- this row does NOT gate `go` (not hard-wired vs the ceiling); two open levers finalize it "
        "(ubel #204 unit-rebase ~3 TPS direction-OPEN -> can FLIP central; land #71 co-log n=%s "
        "retires the rho(*,hw) band). #200 cost annotation (budget row ONLY, single-shot logic "
        "UNCHANGED): realistic spend at the bar is SEQUENTIAL E[shots]=%.2f (not fixed-5); "
        "build-higher (mu>=%.1f/N=1) beats stay-at-bar iff reaching mu costs < %.0f shots' GPU-$ "
        "(c*=%.2f*b fixed / %.2f*b sequential). both_bugs_go_at_lambda_star=%s. Human approval still "
        "required before any HF spend." % (
            "PASSES" if st["launch_trigger_calculator_self_test_passes"] else "FAILS",
            bb_bind["public_183"], bb_bind["icc_190"], bb_bind["private_191"], bb_bind["binding_bar"],
            bb_bind["binding_bar"], _BANKED.halfwidth_realistic(), _BANKED.halfwidth_iid(),
            csc["dedup_provenance_195"]["rho_sampling_input"], csc["acceptance_axis_realistic_icc_tps"],
            csc["combined_sigma_launch_central_tps"], csc["combined_sigma_launch_worstcase_tps"],
            csc["go_trigger_mu_central_tps"], csc["go_trigger_mu_worstcase_tps"],
            csc["lambda1_ceiling_mu_tps"], csc["central_margin_at_lambda1_tps"],
            csc["worstcase_margin_at_lambda1_tps"], csc["headroom_shift_tps"], csc["colog_n_allocations"],
            cba["stay_at_bar"]["expected_shots_sequential"], cba["build_higher"]["mu_safe_n1_tps"],
            cba["crossover_total_shots"], cba["c_star_fixedN_per_b"], cba["c_star_sequential_per_b"],
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
