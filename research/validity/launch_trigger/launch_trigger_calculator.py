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
    "launch_sigma_closure_ubel_201": "LANDED -- CONSUMED (advisor 18:23Z, W&B spau6tch). The de-dup x "
        "realistic-ICC MECHANISM is retained as provenance: de-dup (#195, IDENTITY 5.32 iid) x realistic "
        "ICC (#190, MAGNITUDE sqrt(D)=2.100 -> 11.17) are ORTHOGONAL corrections to the SAME acceptance "
        "axis. BUT #201's sigma->LCB TRIGGER (12.215/13.796 -> 520.09/522.69, worst-case UNREACHABLE) is "
        "RETIRED by ubel #204: the 11.17 acceptance leg was a z=1.96 two-sided HALF-WIDTH double-counted "
        "with the z=1.645 P95 LCB (a units bug). The PROVISIONAL knife-edge verdict (central +0.86, "
        "worst-case -1.74) is SUPERSEDED -- see launch_sigma_unit_rebase_ubel_204. The de-dup IDENTITY "
        "(5.32), ICC=0 corner (7.2617 = #195), and headroom-erosion provenance (lifts #194's 512.16) "
        "stay banked. NO change to the binding BUILD bar (private 0.9780, #191) -- purely the launch "
        "sigma->LCB row.",
    "launch_sigma_unit_rebase_ubel_204": "LANDED -- CONSUMED (advisor 18:46Z, W&B m7vwuus2). RETIRES "
        "#201's sigma->LCB trigger on a UNITS bug: the acceptance leg 11.17 was a z=1.96 two-sided "
        "HALF-WIDTH; clean 1-sigma = 11.17/1.95996 = 5.699. Re-footed onto #194's clean convention "
        "(every leg 1-sigma; LCB(mu)=mu-z1*sigma) -> combined LAUNCH sigma 7.545 central / 8.897 "
        "worst-case (was 12.215/13.796); P95 GO trigger mu >= 500 + z_p95*sigma = 512.41 central / "
        "514.63 worst-case (was 520.09/522.69) vs the lambda=1 ceiling 520.95 -> BOTH BELOW -> lambda=1 "
        "clears 500 at P95 CENTRALLY (+8.54) AND worst-case (+6.32). Verdict FLIPS PROVISIONAL "
        "knife-edge -> RESOLVED-YES. Direction came out SIGN-BACKWARDS (the dominant acceptance leg is "
        "DIVIDED down by z2, so the trigger DROPS 7.68 TPS, not up ~3). Anchors reproduce #195's de-dup "
        "central 7.2617 and #194's break-even 512.16 EXACTLY (err 0.0). Still NON-GATING ('it does not "
        "authorize a launch'). ubel #207 (advisor 19:40Z) RESOLVED the #175 two-readings caveat in "
        "FAVOR of the YES (the larger 10.906 reading is the B=16384 128-tok SUB-bench, RETIRED -> the "
        "launch-correct 512.41/514.63 trigger STANDS); only land #71 co-log (n=385) remains open, and "
        "it now TIGHTENS a YES (retires rho(*,hw) [-0.3,+0.3]) rather than rescuing a NO. private-footing "
        "sensitivity -0.063 TPS (negligible). NO change to the binding BUILD bar (private 0.9780).",
    "liveprobe_depth_budget_denken_197": "LANDED -- CONSUMED (advisor 18:39Z, W&B wqr94io4). The GO "
        "leg GATES on land #71's MEASURED full-ladder q[2..9] >= 0.9780 -- NEVER a depth-1-only or "
        "spine-inferred read. liveprobe is a beta~1 CONFIRMATION, not a discovery: at the grounded "
        "beta=0.765 the mechanism CANNOT clear the private bar (even PERFECT depth-1 -> private LCB "
        "419.6 << 500), so a real GO needs beta~1 across the MEASURED ladder, not a point lambda_hat. "
        "A depth-1-only read FALSELY declares GO (overstatement 85.2 TPS) -> false_go_risk_depth1_only="
        "True. min_depths_for_decisive=full-ladder (depth-1+2 does NOT suffice). The decisive "
        "certification cost ROW is now REPLACED by denken #205's realistic SPRT (see "
        "liveprobe_sprt_budget_denken_205): the ~30,455 fixed-N Neyman trials @lambda=1 (shallow-heavy "
        "N_d[1..9]; 1.43x efficient over equal-allocation) survives only as the worst-case CAP. GATES "
        "the GO leg (the worked tuple's 8-entry ladder SATISFIES the guard); REINFORCES the HOLD, does "
        "NOT flip it.",
    "liveprobe_sprt_budget_denken_205": "LANDED -- CONSUMED (advisor 19:09Z, W&B oracle_readout/"
        "sprt_liveprobe_budget). REPLACES #197's fixed-N liveprobe-cost row with the realistic "
        "SPRT (expected-N, operating-characteristic): the measurement-cost row is no longer '30k "
        "draws' -- it is E[N]~405 on a clear-NO-GO build (1788 under realistic ICC), ~14,915 "
        "near-bar, and <=24,398 worst-case (ASN peak; 107,610 realistic-ICC). A 75.12x collapse vs "
        "#197's 30,455 fixed-N on the NO-GO build. (alpha,power)=(0.05,0.95) realized EXACTLY; "
        "Wald boundaries A/B=+-2.9444; Deff=4.41 (#190). The full-ladder GO guard is UNCHANGED -- "
        "this is a cost-row swap, NOT a verdict change: a real GO STILL needs beta~1 across the "
        "MEASURED q[2..9] ladder (at beta=0.765 the NO-GO private LCB 419.6 << 500). Carry #197's "
        "30,455 as the fixed-N worst-case cap. REINFORCES the HOLD.",
    "frozen_budget_kanna_202": "LANDED -- CONSUMED (advisor 18:39Z, W&B 533jd6l1; annotation only). "
        "Multi-shot budget under the conservative FROZEN regime (the DEFAULT until the human pins the "
        "harness): under fixed prompts + deterministic greedy the official harness re-benchmarks "
        "IDENTICAL tokens, so per-checkpoint sampling deviation is a COMMON bias and best-of-N beats "
        "down ONLY sigma_hw (66%% of one-sigma). So #194's N=5-at-the-bar does NOT reach P>=0.95 under "
        "freeze (frozen P=0.810, not fresh 0.969); the default build-bar input is mu_bar_frozen_p95="
        "504.87 (not fresh 499.08). THE HEDGE: build-to-mu=512.2 / N=1 is fully freeze-robust "
        "(n_shots_frozen=1; a single draw has the same sigma_draw in both regimes) -> the SAFE "
        "recommendation is UNTOUCHED. Only build-at-bar+best-of-N is frozen-fragile (E[shots]=2.34 vs "
        "fresh 1.94; exhausts WITHOUT clearing 19%%; breakeven f*=0.846). kanna -> #206 frozen-cost "
        "crossover. Does NOT touch the binding bar or single-shot sigma; REINFORCES the HOLD.",
    "launch_sigma_175_reconcile_ubel_207": "LANDED -- CONSUMED (advisor 19:40Z). RESOLVES the #204 "
        "open caveat in FAVOR of the YES: the two #175 readings (h_out 5.178 @ B=65536 full-gen vs "
        "#175-sampling 10.906 @ B=16384 128-tok window) are the SAME finite-sample TPS CI at DIFFERENT "
        "bench sizes, NOT sqrt(D)-apart (the 2.106 ratio = bench-sqrtN 2.04 x op-point 1.03, only "
        "COINCIDENTALLY ~= #190's sqrt(D) 2.100; reading 10.906 as h_out*sqrt(D) double-counts the ICC). "
        "The launch-correct full-generation h_out -> trigger 512.41/514.63 STANDS, both < the 520.95 "
        "ceiling -> #204's robust-YES SURVIVES; the 10.906 sub-bench reading is RETIRED. Only land #71 "
        "co-log (n=385) remains open and now merely TIGHTENS the YES. NO change to the binding bar.",
    "winners_curse_budget_kanna_210": "LANDED -- CONSUMED (advisor 19:40Z, W&B hwvv7nn1; build-target "
        "row). CORRECTS the #202 hedge against the binding PRIVATE bar: best-of-N does NOT relax it -- "
        "the conditional private clear is EXACTLY FLAT in N (n_star_private=1) because selection is on "
        "non-replicating public noise and the private grade is one fresh draw (Capen 1971 winner's "
        "curse / Smith-Winkler 2006 optimizer's curse). The freeze-robust mu=512.2/N=1 does NOT survive "
        "privately (p=0.3120<0.95); to clear 500-PRIVATE at P>=0.95 under best-of-5 the PUBLIC build "
        "must reach mu_bar_private_corrected=528.48 (+23.61 winner's-curse tax over #202's 504.87 = "
        "7.28 evaporating public discount + 16.33 private-drop gross-up). REGIME-INVARIANT. N=1 STANDS "
        "(build higher, do NOT re-draw). Does NOT touch the sigma->LCB PUBLIC trigger 512.41; "
        "REINFORCES the HOLD.",
    "kernel_budget_lambda_wirbel_213": "LANDED -- CONSUMED (advisor 19:40Z; lane-a #192 capstone). "
        "Grades the compliant-spec batch-invariant verify kernel's overhead budget vs lambda: "
        "max_kernel_overhead_pct(lambda) opens from <=0 at lambda_hat=0.342 (the realistic floor "
        "already misses 500 -- even a FREE kernel fails) to 7.33% both / 4.12% descent at lambda=1; the "
        "zero-overhead path first clears 500 at lambda_crit=0.8345 both / 0.9067 descent. So the ONLY "
        "compliant 500-lane (lane-a) is a DOUBLE gate: self-KV-recovery lambda >= lambda_crit AND kernel "
        "under max_overhead(lambda). Off-shelf #122 (+51.78%) clears at NO physical lambda<=1 (~7.1x "
        "over the lambda=1 budget). EXTENDS the #192 compliance bracket ABOVE the sigma math; does NOT "
        "change the #204/#207 trigger; REINFORCES the HOLD.",
    "sprt_ar_asn_denken_212": "LANDED -- CONSUMED (advisor 19:54Z; cost-band sharpening). CONFIRMS "
        "#205's flat xDeff=4.41 is CONSERVATIVE and SHARPENS it: folding #190's DECAYING within-prompt "
        "ACF (rho(1)=0.2583) into the SPRT partial-sum variance tightens the realism band 1.59-2.66x. "
        "The E[N]_nogo band becomes [405 IID -> 672 AR(1)-optimistic -> 1,125 measured-ACF-realistic -> "
        "1,788 flat-loose]; the data-grounded point is 1,125 (the measured ACF decays slower than pure "
        "AR(1): rho(2)=0.168 >> rho(1)^2=0.067). The 75.12x collapse vs #197's fixed-N is Deff-INVARIANT, "
        "and realized (alpha,power)=(0.05,0.95) + the bar 0.9780 are UNCHANGED -- the AR correction only "
        "sharpens the absolute band, never flips the verdict. Orthogonal to #192. REINFORCES the HOLD.",
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
    # ubel #201 launch-sigma closure -- MERGED (advisor 18:23Z): the de-dup (#195) x realistic-ICC
    # (#190) MECHANISM (banked as provenance + the leg vector). Its sigma->LCB TRIGGER (12.215/13.796
    # -> 520.09/522.69) is RETIRED by ubel #204 (a z=1.96 half-width units bug); see unit_rebase_204.
    "sigma_closure_201": "research/validity/launch_sigma_closure/launch_sigma_closure_results.json",
    # denken #197 liveprobe depth-budget -- MERGED (advisor 18:39Z): the certification COST row + the
    # FULL-LADDER GO requirement + the depth-1-only FALSE-GO guard. At the grounded beta=0.765 the
    # mechanism CANNOT clear the private bar (even PERFECT depth-1 -> private_LCB 419.6 << 500), so a
    # GO requires beta~1 across the MEASURED full ladder, not a point lambda_hat. REINFORCES the HOLD.
    "liveprobe_197": "research/oracle_readout/liveprobe_depth_budget/liveprobe_depth_budget_results.json",
    # kanna #202 frozen-sampling re-draw budget -- MERGED (advisor 18:39Z): the multi-shot budget
    # under the conservative FROZEN regime. Conservative default build-bar input mu_bar_frozen_p95=
    # 504.87 (not fresh 499.08); the build-to-512.2/N=1 hedge is freeze-robust (n_shots_frozen_at_512
    # =1). Only build-at-bar+best-of-N is frozen-fragile. REINFORCES the HOLD; budget-row annotation.
    "frozen_budget_202": "research/validity/frozen_budget/frozen_budget_results.json",
    # ubel #204 clean-1-sigma unit rebase -- MERGED (advisor 18:46Z): RETIRES #201's σ→LCB trigger.
    # #201's 520.09/522.69 was a UNITS BUG (the acceptance leg 11.17 was a z=1.96 two-sided HALF-WIDTH
    # double-counted with the z=1.645 P95 LCB). Clean 1-sigma = 11.17/1.95996 = 5.699 -> combined LAUNCH
    # sigma 7.545/8.897; P95 trigger mu 512.41/514.63 vs the lambda=1 ceiling 520.95 -> BOTH BELOW ->
    # lambda=1 clears 500 at P95 CENTRALLY (+8.54) AND worst-case (+6.32). Verdict FLIPS to RESOLVED-YES.
    "unit_rebase_204": "research/validity/launch_sigma_unit_rebase/launch_sigma_unit_rebase_results.json",
    # denken #205 SPRT liveprobe budget -- MERGED (advisor 19:09Z): REPLACES #197's fixed-N 30k cost
    # row with the realistic sequential expected-N. E[N]≈405 on a clear-NO-GO build (75.12x cheaper),
    # 14,915 near-bar, 24,398 worst-case (ASN peak); (α,power)=(0.05,0.95), boundaries ±2.9444, Deff=4.41.
    "sprt_liveprobe_205": "research/oracle_readout/sprt_liveprobe_budget/sprt_liveprobe_budget_results.json",
    # lawine #196 compliant non-spec floor -- MERGED (advisor 18:58Z): lane-b. Under strict #192 the
    # compliant non-spec int4 path floors at ≈165.44 official TPS (66.9% below 500); NO compliant
    # non-spec 500-lane. token-identity 1.0, PPL 2.3766<2.42, 128/128. Adds the #192 compliance precond.
    "compliant_nonspec_196": "research/validity/compliant_nonspec_floor/floor_report.json",
    # wirbel #199 compliant-spec E[T] ceiling -- MERGED (advisor 19:09Z): lane-a, the ONLY compliant
    # 500 route under strict #192. Ceiling 536.66 (lower-CI 525.73>500 -> clears), floor 416.31; clears
    # ONLY if kernel overhead < 7.33% both / 4.12% descent (UNMEASURED; off-shelf #122 +51.78% ~7x over).
    "compliant_spec_199": "research/validity/compliant_spec_et/compliant_spec_et_results.json",
    # ubel #207 launch-sigma #175-reading reconcile -- MERGED (advisor 19:40Z): RESOLVES the #204
    # open caveat. The larger 10.906 #175-sampling HW is the B=16384 128-tok sub-bench, NOT launch-
    # correct; the ratio 2.106 vs h_out 5.178 = bench-sqrtN(2.04) x op-point(1.03), only COINCIDENTALLY
    # ~= #190's sqrt(D)=2.100 (reading it as h_out*sqrt(D) double-counts the ICC). Launch-correct = the
    # B=65536 full-generation h_out -> trigger 512.41/514.63, both < 520.95 ceiling -> robust-YES SURVIVES.
    "launch_sigma_175_reconcile_207": "research/validity/launch_sigma_175_reconcile/launch_sigma_175_reconcile_results.json",
    # kanna #210 winner's-curse re-draw -- MERGED (advisor 19:40Z): best-of-N does NOT relax the binding
    # PRIVATE bar. The conditional private clear is FLAT in N (n_star_private=1): selection is on non-
    # replicating public noise, the private grade is one fresh draw. To clear 500-private at P>=0.95 under
    # best-of-5 the PUBLIC build must reach mu_bar_private_corrected=528.48 (+23.61 winner's-curse tax over
    # #202's 504.87). The #202 freeze-robust mu=512.2/N=1 does NOT survive privately (p=0.3120). Build-higher/N=1.
    "winners_curse_210": "research/validity/winners_curse_budget/winners_curse_budget_results.json",
    # wirbel #213 compliant-kernel overhead budget vs lambda -- MERGED (advisor 19:40Z): the lane-a capstone.
    # max_kernel_overhead_pct(lambda): 7.33% both / 4.12% descent at lambda=1 (<->#199); lambda_crit=0.8345
    # both / 0.9067 descent (below it even a FREE batch-invariant kernel misses 500); lambda_hat=0.342 budget
    # -16.74%. Off-shelf #122 (+51.78%) clears at NO physical lambda<=1. Strict-#192 500-path is a DOUBLE gate.
    "kernel_budget_lambda_213": "research/validity/kernel_budget_lambda/kernel_budget_lambda_results.json",
    # denken #212 AR(1)-corrected SPRT ASN -- MERGED (advisor 19:54Z): sharpens #205's flat-Deff cost row.
    # #205's flat xDeff=4.41 is CONFIRMED CONSERVATIVE; folding #190's DECAYING within-prompt ACF tightens
    # the realism band 1.59-2.66x. E[N]_nogo band = [405 IID-floor -> 672 AR(1)-opt -> 1,125 measured-ACF-
    # realistic -> 1,788 flat-loose]; data-grounded point 1,125 (rho(2)=0.168 >> rho(1)^2=0.067). UNCHANGED:
    # (alpha,power)=(0.05,0.95), bar 0.9780, 75.12x collapse is Deff-INVARIANT. Orthogonal to #192.
    "sprt_ar_asn_212": "research/validity/sprt_ar_asn/sprt_ar_asn_results.json",
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
        self.liveprobe = _load_axis_json(_AXIS_PATHS["liveprobe_197"])
        self.frozen_budget = _load_axis_json(_AXIS_PATHS["frozen_budget_202"])
        self.unit_rebase = _load_axis_json(_AXIS_PATHS["unit_rebase_204"])
        self.sprt = _load_axis_json(_AXIS_PATHS["sprt_liveprobe_205"])
        self.nonspec_floor = _load_axis_json(_AXIS_PATHS["compliant_nonspec_196"])
        self.compliant_spec = _load_axis_json(_AXIS_PATHS["compliant_spec_199"])
        self.sigma_reconcile = _load_axis_json(_AXIS_PATHS["launch_sigma_175_reconcile_207"])
        self.winners_curse = _load_axis_json(_AXIS_PATHS["winners_curse_210"])
        self.kernel_budget = _load_axis_json(_AXIS_PATHS["kernel_budget_lambda_213"])
        self.sprt_ar = _load_axis_json(_AXIS_PATHS["sprt_ar_asn_212"])

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

    # ---- denken #197 liveprobe depth-budget (MERGED 18:39Z: full-ladder GO gate + cost row) ---- #
    # The GO must gate on land #71's MEASURED full-ladder q[2..9] >= 0.9780, never a depth-1-only /
    # spine-inferred read (a FALSE GO worth 85.2 TPS). At beta=0.765 the mechanism CANNOT clear the
    # private bar (perfect depth-1 -> private_LCB 419.6 << 500). The cost ROW is REPLACED by denken #205's
    # realistic SPRT (E[N]~405 NO-GO / 14,915 near-bar / <=24,398 worst-case); ~30,455 fixed-N -> worst-case cap.
    def liveprobe_landed(self) -> bool:
        return self.liveprobe is not None

    def liveprobe_mech_can_clear_private(self) -> bool:
        return bool(_dig(self.liveprobe, "synthesis", "mechanism_feasibility",
                         "mechanism_can_clear_private_bar", default=False))

    def liveprobe_private_lcb_perfect_depth1(self) -> float | None:
        v = _dig(self.liveprobe, "synthesis", "mechanism_feasibility",
                 "private_lcb_at_perfect_depth1_mech")
        return float(v) if v is not None else None

    def liveprobe_false_go_risk_depth1(self) -> bool:
        return bool(_dig(self.liveprobe, "synthesis", "depth1_false_go",
                         "false_go_risk_depth1_only", default=False))

    def liveprobe_depth1_overstatement(self) -> float | None:
        v = _dig(self.liveprobe, "synthesis", "depth1_false_go", "overstatement_tps")
        return float(v) if v is not None else None

    def liveprobe_true_private_lcb_at_lambda1(self) -> float | None:
        v = _dig(self.liveprobe, "synthesis", "depth1_false_go", "true_private_lcb_at_lambda1_eq_1")
        return float(v) if v is not None else None

    def liveprobe_min_depths_for_decisive(self):
        return _dig(self.liveprobe, "synthesis", "beta_depth_count", "min_depths_for_decisive")

    def liveprobe_min_depths_int(self):
        return _dig(self.liveprobe, "synthesis", "beta_depth_count", "min_depths_for_decisive_int")

    def liveprobe_depth1_plus_2_suffices(self) -> bool:
        return bool(_dig(self.liveprobe, "synthesis", "beta_depth_count",
                         "depth1_plus_2_suffices", default=False))

    def liveprobe_beta_primary(self) -> float | None:
        v = _dig(self.liveprobe, "synthesis", "imports", "beta_primary")
        return float(v) if v is not None else None

    def liveprobe_private_bar(self) -> float | None:
        v = _dig(self.liveprobe, "synthesis", "imports", "private_bar_both_0p9780")
        return float(v) if v is not None else None

    def liveprobe_decisive_total_trials(self) -> float | None:
        v = _dig(self.liveprobe, "synthesis", "decisive_budget", "total_trials_for_decisive_private")
        return float(v) if v is not None else None

    def liveprobe_N_d_budget(self):
        return _dig(self.liveprobe, "synthesis", "decisive_budget", "N_d_budget_1to9", default=[])

    def liveprobe_neyman_efficiency_gain(self) -> float | None:
        v = _dig(self.liveprobe, "synthesis", "decisive_budget", "efficiency_gain_neyman_vs_equal")
        return float(v) if v is not None else None

    def liveprobe_decisive_margin_at_lambda1(self) -> float | None:
        # the per_lam_true_table row at lam_true=1.0 -> structural margin 0.022 (private bar so tight).
        rows = _dig(self.liveprobe, "synthesis", "decisive_budget", "per_lam_true_table", default=[])
        for r in (rows or []):
            if isinstance(r, dict) and abs(float(r.get("lam_true", 0.0)) - 1.0) < 1e-9:
                v = r.get("margin")
                return float(v) if v is not None else None
        return None

    # ---- kanna #202 frozen-sampling re-draw budget (MERGED 18:39Z: conservative FROZEN regime) ---- #
    # Multi-shot budget under the FROZEN regime (fixed prompts + deterministic greedy => the harness
    # re-benchmarks IDENTICAL tokens => best-of-N beats ONLY sigma_hw, not the per-checkpoint
    # sigma_sample bias). Conservative DEFAULT build-bar input mu_bar_frozen_p95=504.87 (not fresh
    # 499.08). THE HEDGE: build-to-512.2/N=1 is freeze-robust (n_shots_frozen_at_512=1). REINFORCES HOLD.
    def frozen_budget_landed(self) -> bool:
        return self.frozen_budget is not None

    def mu_bar_frozen_p95(self) -> float | None:
        v = _dig(self.frozen_budget, "mu_bar_frozen_p95")
        return float(v) if v is not None else None

    def mu_bar_fresh_p95_n5(self) -> float | None:
        v = _dig(self.frozen_budget, "build_bar", "mu_bar_fresh_p95_n5")
        return float(v) if v is not None else None

    def p_bar_n5_frozen(self) -> float | None:
        v = _dig(self.frozen_budget, "p_bar_n5_frozen")
        return float(v) if v is not None else None

    def frozen_n_shots_at_512(self):
        return _dig(self.frozen_budget, "build_bar", "n_shots_frozen_at_512")

    def frozen_mu_safe_tps(self) -> float | None:
        v = _dig(self.frozen_budget, "build_bar", "mu_safe_fresh_tps")
        return float(v) if v is not None else None

    def frozen_delta_mu(self) -> float | None:
        v = _dig(self.frozen_budget, "delta_mu_frozen")
        return float(v) if v is not None else None

    def frozen_fraction_breakeven(self) -> float | None:
        v = _dig(self.frozen_budget, "frozen_fraction_breakeven")
        return float(v) if v is not None else None

    def frozen_e_shots_at_bar(self) -> float | None:
        v = _dig(self.frozen_budget, "harness_sensitivity", "p95_shots_at_bar",
                 "e_shots_frozen_at_bar")
        return float(v) if v is not None else None

    def fresh_e_shots_at_bar(self) -> float | None:
        v = _dig(self.frozen_budget, "harness_sensitivity", "p95_shots_at_bar",
                 "e_shots_fresh_at_bar")
        return float(v) if v is not None else None

    def frozen_p_exhaust_without_clear(self) -> float | None:
        v = _dig(self.frozen_budget, "harness_sensitivity", "p95_shots_at_bar",
                 "p_exhaust_without_clear_frozen")
        return float(v) if v is not None else None

    def frozen_sigma_fraction_beatable(self) -> float | None:
        v = _dig(self.frozen_budget, "import_194", "sigma_fraction_beatable_frozen")
        return float(v) if v is not None else None

    # ---- ubel #204 clean-1-sigma unit rebase (MERGED 18:46Z: RETIRES #201's sigma->LCB trigger) ---- #
    # #201's 520.09/522.69 was a UNITS BUG: the acceptance leg (11.17) was a z=1.96 two-sided HALF-
    # WIDTH double-counted with the z=1.645 P95 LCB. Clean 1-sigma = 11.17/1.95996 = 5.699 -> combined
    # LAUNCH sigma 7.545 central / 8.897 worst-case; P95 GO trigger mu 512.41 / 514.63 vs the lambda=1
    # ceiling 520.95 -> BOTH BELOW -> lambda=1 clears 500 at P95 CENTRALLY (+8.54) AND worst-case
    # (+6.32). Verdict FLIPS from PROVISIONAL knife-edge to RESOLVED-YES. Still NON-GATING.
    def unit_rebase_landed(self) -> bool:
        return self.unit_rebase is not None

    def rebase_mu_clears_500(self, kind: str) -> float | None:
        v = _dig(self.unit_rebase, "clean_trigger", "mu_clears_500_clean_%s" % kind)
        return float(v) if v is not None else None

    def rebase_combined_sigma(self, kind: str) -> float | None:
        v = _dig(self.unit_rebase, "clean_trigger", "combined_sigma_launch_clean_%s" % kind)
        return float(v) if v is not None else None

    def rebase_lambda1_ceiling(self) -> float | None:
        v = _dig(self.unit_rebase, "clean_trigger", "lambda1_ceiling_mu")
        return float(v) if v is not None else None

    def rebase_headroom_below_ceiling(self, kind: str) -> float | None:
        v = _dig(self.unit_rebase, "clean_trigger", "%s_headroom_below_ceiling_tps" % kind)
        return float(v) if v is not None else None

    def rebase_lambda1_clears_500(self, kind: str) -> bool:
        return bool(_dig(self.unit_rebase, "clean_trigger", "lambda1_clears_500_clean_%s" % kind,
                         default=False))

    def rebase_does_lambda1_clear_p95_centrally(self):
        return _dig(self.unit_rebase, "clean_trigger", "does_lambda1_clear_500_at_p95_centrally")

    def rebase_acceptance_1sigma_clean(self) -> float | None:
        v = _dig(self.unit_rebase, "clean_trigger", "acceptance_1sigma_clean")
        return float(v) if v is not None else None

    def rebase_delta_mu(self, kind: str) -> float | None:
        v = _dig(self.unit_rebase, "clean_trigger", "delta_mu_rebase_%s" % kind)
        return float(v) if v is not None else None

    def rebase_clean_vs_ceiling(self, kind: str):
        return _dig(self.unit_rebase, "clean_trigger", "clean_%s_vs_ceiling" % kind)

    def rebase_private_footing_shift(self) -> float | None:
        v = _dig(self.unit_rebase, "clean_trigger", "private_footing_sensitivity",
                 "shift_vs_primary_tps")
        return float(v) if v is not None else None

    def rebase_direction_matches_prediction(self) -> bool:
        return bool(_dig(self.unit_rebase, "direction_reconciliation",
                         "rebase_direction_matches_prediction", default=False))

    def rebase_mu_201_imported(self, kind: str) -> float | None:
        v = _dig(self.unit_rebase, "imported_legs_201", "mu_201_%s" % kind)
        return float(v) if v is not None else None

    def rebase_combined_201_central(self) -> float | None:
        v = _dig(self.unit_rebase, "imported_legs_201", "combined_201_central")
        return float(v) if v is not None else None

    def rebase_anchor_err_195(self) -> float | None:
        v = _dig(self.unit_rebase, "anchors", "anchor_err_195_dedup")
        return float(v) if v is not None else None

    def rebase_anchor_err_194(self) -> float | None:
        v = _dig(self.unit_rebase, "anchors", "anchor_err_194_breakeven")
        return float(v) if v is not None else None

    def rebase_self_test_passes(self) -> bool:
        return bool(_dig(self.unit_rebase, "unit_rebase_self_test_passes", default=False))

    # ---- denken #205 SPRT liveprobe budget (MERGED 19:09Z: REPLACES #197's fixed-N 30k cost row) ---- #
    # Sequential probability ratio test: the EXPECTED measurement cost is E[N]≈405 on a clear-NO-GO
    # build (75.12x cheaper than #197's 30,455 fixed-N), 14,915 near-bar, 24,398 worst-case (ASN peak);
    # (alpha,power)=(0.05,0.95), boundaries A/B=+-2.9444, Deff=4.41. Keep 30k as the fixed-N cap.
    def sprt_landed(self) -> bool:
        return self.sprt is not None

    def sprt_expected_n_nogo(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "headline", "expected_n_sprt_nogo")
        return float(v) if v is not None else None

    def sprt_expected_n_nogo_realistic_icc(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "headline", "expected_n_sprt_nogo_realistic_icc")
        return float(v) if v is not None else None

    def sprt_expected_n_nearbar(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "headline", "expected_n_sprt_nearbar")
        return float(v) if v is not None else None

    def sprt_worst_case_expected_n(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "operating_characteristic", "worst_case_expected_n")
        return float(v) if v is not None else None

    def sprt_worst_case_expected_n_realistic_icc(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "operating_characteristic",
                 "worst_case_expected_n_realistic_icc")
        return float(v) if v is not None else None

    def sprt_savings_vs_fixed_n(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "headline", "savings_vs_fixed_n_nogo")
        return float(v) if v is not None else None

    def sprt_n_fixed_197(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "headline", "n_fixed_z95_197")
        return float(v) if v is not None else None

    def sprt_boundary(self, which: str) -> float | None:
        v = _dig(self.sprt, "synthesis", "sprt_setup", "sprt_boundaries", which)
        return float(v) if v is not None else None

    def sprt_realized_alpha(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "operating_characteristic", "realized_alpha_false_nogo_at_mu1")
        return float(v) if v is not None else None

    def sprt_realized_power(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "operating_characteristic", "realized_power_decide_nogo_at_bar")
        return float(v) if v is not None else None

    def sprt_target_alpha(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "operating_characteristic", "target_alpha")
        return float(v) if v is not None else None

    def sprt_target_power(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "operating_characteristic", "target_power")
        return float(v) if v is not None else None

    def sprt_deff_190(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "imports", "deff_190")
        return float(v) if v is not None else None

    def sprt_beta_nogo(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "headline", "beta_nogo")
        return float(v) if v is not None else None

    def sprt_private_lcb_nogo(self) -> float | None:
        v = _dig(self.sprt, "synthesis", "headline", "private_lcb_nogo_tps")
        return float(v) if v is not None else None

    def sprt_self_test_passes(self) -> bool:
        return bool(_dig(self.sprt, "synthesis", "self_test", "sprt_budget_self_test_passes",
                         default=False))

    # ---- lawine #196 compliant non-spec floor (MERGED 18:58Z: lane-b; the #192 compliance precond) ---- #
    # Under strict #192 the compliant non-spec int4 path floors at ~165.44 official TPS (66.9% below
    # 500) -- token-identity 1.0, PPL 2.3766<2.42, 128/128. There is NO compliant non-spec 500-lane;
    # the speculation premium (316.1 TPS, 191%) is existential. Gates the launch on the #192 ruling.
    def nonspec_floor_landed(self) -> bool:
        return self.nonspec_floor is not None

    def nonspec_official_tps(self) -> float | None:
        v = _dig(self.nonspec_floor, "nonspec_official_tps_est")
        return float(v) if v is not None else None

    def nonspec_floor_band(self, which: str):
        return _dig(self.nonspec_floor, "nonspec_official_tps_est_%s_band" % which, default=[])

    def nonspec_token_identity(self) -> float | None:
        v = _dig(self.nonspec_floor, "nonspec_token_identity_rate")
        return float(v) if v is not None else None

    def nonspec_ppl(self) -> float | None:
        v = _dig(self.nonspec_floor, "ppl_nonspec")
        return float(v) if v is not None else None

    def nonspec_completes_128(self) -> bool:
        return bool(_dig(self.nonspec_floor, "nonspec_completes_128", default=False))

    def nonspec_clears_500(self) -> bool:
        return bool(_dig(self.nonspec_floor, "nonspec_clears_500", default=False))

    def nonspec_margin_pct(self) -> float | None:
        v = _dig(self.nonspec_floor, "margin_to_500_pct")
        return float(v) if v is not None else None

    def nonspec_spec_premium_tps(self) -> float | None:
        v = _dig(self.nonspec_floor, "spec_premium_tps")
        return float(v) if v is not None else None

    def nonspec_spec_premium_pct(self) -> float | None:
        v = _dig(self.nonspec_floor, "spec_premium_pct")
        return float(v) if v is not None else None

    def nonspec_verdict_label(self):
        return _dig(self.nonspec_floor, "verdict_label")

    def nonspec_floor_self_test_passes(self) -> bool:
        return bool(_dig(self.nonspec_floor, "nonspec_floor_self_test_passes", default=False))

    # ---- wirbel #199 compliant-spec E[T] ceiling (MERGED 19:09Z: lane-a, the ONLY compliant 500 route) ---- #
    # Under strict #192 the batch-invariant int4 VERIFY kernel (lane a) ceiling = 536.66 official TPS
    # (lower-CI 525.73 > 500 -> clears), floor = 416.31 (misses). It clears 500 ONLY if kernel overhead
    # < 7.33% (both-bugs) / 4.12% (descent) -- UNMEASURED; kanna #122 off-the-shelf int4 is +51.78%
    # (~7x over). So lane-a is the single compliant route, conditioned on a <7.3% overhead feasibility.
    def compliant_spec_landed(self) -> bool:
        return self.compliant_spec is not None

    def compliant_spec_ceiling(self) -> float | None:
        v = _dig(self.compliant_spec, "synthesis", "headline", "compliant_spec_tps_ceiling")
        return float(v) if v is not None else None

    def compliant_spec_floor(self) -> float | None:
        v = _dig(self.compliant_spec, "synthesis", "headline", "compliant_spec_tps_floor")
        return float(v) if v is not None else None

    def compliant_spec_clears_500(self) -> bool:
        return bool(_dig(self.compliant_spec, "synthesis", "headline", "compliant_spec_clears_500",
                         default=False))

    def compliant_spec_floor_clears_500(self) -> bool:
        return bool(_dig(self.compliant_spec, "synthesis", "headline",
                         "compliant_spec_floor_clears_500", default=False))

    def compliant_spec_max_overhead(self, topo: str) -> float | None:
        if topo == "both_bugs":
            v = _dig(self.compliant_spec, "synthesis", "headline",
                     "max_kernel_overhead_pct_to_clear_500")
        else:
            v = _dig(self.compliant_spec, "synthesis", "brackets", "descent_only", "ceiling_propagate",
                     "tau_central_1p0", "max_kernel_overhead_pct_to_clear_500")
        return float(v) if v is not None else None

    def compliant_spec_ceiling_ci_lower(self, topo: str) -> float | None:
        v = _dig(self.compliant_spec, "synthesis", "brackets", topo, "ceiling_finite_sample_ci_tau1",
                 "ci_lower_tps")
        return float(v) if v is not None else None

    def compliant_spec_lower_clears_500(self, topo: str) -> bool:
        return bool(_dig(self.compliant_spec, "synthesis", "brackets", topo,
                         "ceiling_finite_sample_ci_tau1", "lower_clears_500", default=False))

    def compliant_spec_descent_ceiling(self) -> float | None:
        v = _dig(self.compliant_spec, "synthesis", "brackets", "descent_only",
                 "ceiling_finite_sample_ci_tau1", "central_tps")
        return float(v) if v is not None else None

    def compliant_spec_offshelf_overhead_ref(self) -> float | None:
        v = _dig(self.compliant_spec, "synthesis", "composition",
                 "kanna122_offshelf_overhead_nonworking_ref")
        return float(v) if v is not None else None

    def compliant_spec_self_test_passes(self) -> bool:
        return bool(_dig(self.compliant_spec, "synthesis", "self_test",
                         "compliant_spec_et_self_test_passes", default=False))

    # ---- ubel #207 launch-sigma #175-reading reconcile (MERGED 19:40Z: RESOLVES the #204 caveat) ---- #
    # The two #175 readings (h_out 5.178 @ B=65536 full-gen vs #175-sampling 10.906 @ B=16384 128-tok
    # window) are the SAME finite-sample TPS CI at DIFFERENT bench sizes, NOT sqrt(D)-apart (ratio 2.106
    # = bench-sqrtN 2.04 x op-point 1.03, COINCIDENTALLY ~= #190's sqrt(D) 2.100). Launch-correct = the
    # full-generation h_out -> trigger 512.41/514.63, both < 520.95 ceiling -> the robust-YES SURVIVES.
    def sigma_reconcile_landed(self) -> bool:
        return self.sigma_reconcile is not None

    def sigma_reconcile_self_test_passes(self) -> bool:
        return bool(_dig(self.sigma_reconcile, "reconcile_175_self_test_passes", default=False))

    def sigma_reconcile_robust_yes_survives(self) -> bool:
        return bool(_dig(self.sigma_reconcile, "robust_yes_survives", default=False))

    def sigma_reconcile_lambda1_clears_conservative(self) -> bool:
        return bool(_dig(self.sigma_reconcile, "lambda1_clears_under_conservative_reading",
                         default=False))

    def sigma_reconcile_conservative_is_launch_correct(self) -> bool:
        return bool(_dig(self.sigma_reconcile, "verdict", "conservative_reading_is_launch_correct",
                         default=False))

    def sigma_reconcile_launch_correct_reading(self):
        return _dig(self.sigma_reconcile, "launch_correct_reading")

    def sigma_reconcile_ratio_175(self) -> float | None:
        v = _dig(self.sigma_reconcile, "ratio_175_readings")
        return float(v) if v is not None else None

    def sigma_reconcile_ratio_equals_sqrtd(self) -> bool:
        return bool(_dig(self.sigma_reconcile, "ratio_equals_sqrtD", default=False))

    def sigma_reconcile_trigger_hout(self, kind: str) -> float | None:
        v = _dig(self.sigma_reconcile, "trigger_%s_hout" % kind)
        return float(v) if v is not None else None

    def sigma_reconcile_trigger_175sampling(self, kind: str) -> float | None:
        v = _dig(self.sigma_reconcile, "trigger_%s_175sampling" % kind)
        return float(v) if v is not None else None

    def sigma_reconcile_lambda1_ceiling(self) -> float | None:
        v = _dig(self.sigma_reconcile, "lambda1_ceiling")
        return float(v) if v is not None else None

    def sigma_reconcile_delta_trigger(self) -> float | None:
        v = _dig(self.sigma_reconcile, "delta_trigger_reading")
        return float(v) if v is not None else None

    def sigma_reconcile_acceptance_1sigma_hout(self) -> float | None:
        v = _dig(self.sigma_reconcile, "acceptance_1sigma_hout")
        return float(v) if v is not None else None

    def sigma_reconcile_headroom(self, kind: str) -> float | None:
        v = _dig(self.sigma_reconcile, "readings", "A_hout_launch_correct", "trigger",
                 "%s_headroom_below_ceiling_tps" % kind)
        return float(v) if v is not None else None

    # ---- kanna #210 winner's-curse re-draw (MERGED 19:40Z: best-of-N does NOT relax the PRIVATE bar) ---- #
    # The conditional private clear is FLAT in N (n_star_private=1): selection is on non-replicating
    # public noise, the private grade is one fresh draw. To clear 500-private at P>=0.95 under best-of-5
    # the PUBLIC build must reach mu_bar_private_corrected=528.48 (+23.61 winner's-curse tax over #202's
    # 504.87 = 7.28 evaporating public discount + 16.33 private-drop grossup). REGIME-INVARIANT. Build-higher/N=1.
    def winners_curse_landed(self) -> bool:
        return self.winners_curse is not None

    def winners_curse_self_test_passes(self) -> bool:
        return bool(_dig(self.winners_curse, "winners_curse_self_test_passes", default=False))

    def winners_curse_mu_bar_private_corrected(self) -> float | None:
        v = _dig(self.winners_curse, "mu_bar_private_corrected")
        return float(v) if v is not None else None

    def winners_curse_delta_mu(self) -> float | None:
        v = _dig(self.winners_curse, "delta_mu_winners_curse")
        return float(v) if v is not None else None

    def winners_curse_n_star_private(self) -> int | None:
        v = _dig(self.winners_curse, "n_star_private")
        return int(v) if v is not None else None

    def winners_curse_private_clear_flat_in_n(self) -> bool:
        return bool(_dig(self.winners_curse, "private_clear_flat_in_n", default=False))

    def winners_curse_mu_bar_frozen_202(self) -> float | None:
        v = _dig(self.winners_curse, "private_corrected_bar", "mu_bar_frozen_public_202")
        return float(v) if v is not None else None

    def winners_curse_mu_safe_fresh_194(self) -> float | None:
        v = _dig(self.winners_curse, "private_corrected_bar", "mu_safe_fresh_194")
        return float(v) if v is not None else None

    def winners_curse_freeze_robust_512_survives(self) -> bool:
        return bool(_dig(self.winners_curse, "private_corrected_bar",
                         "freeze_robust_512_survives_private", default=False))

    def winners_curse_regime_invariant(self) -> bool:
        return bool(_dig(self.winners_curse, "private_corrected_bar", "regime_invariant",
                         default=False))

    def winners_curse_p_private_at_512(self) -> float | None:
        v = _dig(self.winners_curse, "private_corrected_bar", "p_private_clear_at_mu512p2_n1")
        return float(v) if v is not None else None

    def winners_curse_tax_decomposition(self) -> dict:
        return _dig(self.winners_curse, "private_corrected_bar", "tax_decomposition", default={}) or {}

    def winners_curse_tps_n5(self, regime: str) -> float | None:
        v = _dig(self.winners_curse, "winners_curse_tps_n5_%s" % regime)
        return float(v) if v is not None else None

    def winners_curse_lambda_star_191(self) -> float | None:
        v = _dig(self.winners_curse, "import_banked", "lambda_star_191")
        return float(v) if v is not None else None

    # ---- wirbel #213 compliant-kernel overhead budget vs lambda (MERGED 19:40Z: lane-a capstone) ---- #
    # max_kernel_overhead_pct(lambda): the batch-invariant verify kernel's budget opens from <=0 at
    # lambda_hat=0.342 (the realistic floor already misses 500 -- even a FREE kernel fails) to 7.33%
    # (both) / 4.12% (descent) at lambda=1; the zero-overhead path first clears 500 at lambda_crit=0.8345
    # both / 0.9067 descent. Off-shelf #122 (+51.78%) clears at NO physical lambda<=1. DOUBLE gate.
    def kernel_budget_landed(self) -> bool:
        return self.kernel_budget is not None

    def kernel_budget_self_test_passes(self) -> bool:
        return bool(_dig(self.kernel_budget, "synthesis", "self_test",
                         "kernel_budget_lambda_self_test_passes", default=False))

    def kernel_budget_lambda_crit(self, topo: str) -> float | None:
        v = _dig(self.kernel_budget, "synthesis", "headline",
                 "lambda_crit_clears_500_zero_overhead_%s_tau1" % topo)
        return float(v) if v is not None else None

    def kernel_budget_overhead_at_lambda1(self, topo: str) -> float | None:
        v = _dig(self.kernel_budget, "synthesis", "headline",
                 "overhead_budget_at_lambda_1_%s_tau1" % topo)
        return float(v) if v is not None else None

    def kernel_budget_overhead_at_lambda_hat(self) -> float | None:
        v = _dig(self.kernel_budget, "synthesis", "headline",
                 "overhead_budget_at_lambda_hat_both_bugs_tau1")
        return float(v) if v is not None else None

    def kernel_budget_lambda_hat(self) -> float | None:
        v = _dig(self.kernel_budget, "synthesis", "lambda_hat")
        return float(v) if v is not None else None

    def kernel_budget_offshelf_clears_at_physical_lambda(self) -> bool:
        return bool(_dig(self.kernel_budget, "synthesis", "headline",
                         "off_the_shelf_122_clears_at_physical_lambda_both_bugs_tau1", default=False))

    def kernel_budget_offshelf_overhead_ref(self) -> float | None:
        v = _dig(self.kernel_budget, "synthesis", "composition",
                 "kanna122_offshelf_overhead_nonworking_ref")
        return float(v) if v is not None else None

    def kernel_budget_max_at_saturation(self) -> float | None:
        v = _dig(self.kernel_budget, "synthesis", "headline",
                 "max_budget_pct_at_prob_saturation_both_bugs_tau1")
        return float(v) if v is not None else None

    def kernel_budget_verdict(self):
        return _dig(self.kernel_budget, "synthesis", "verdict")

    def kernel_budget_curve(self, topo: str) -> list[dict]:
        rows = _dig(self.kernel_budget, "synthesis", "regimes", topo, "overhead_budget_vs_lambda",
                    default=[]) or []
        out = []
        for r in rows:
            tc = r.get("tau_central_1p0", {}) or {}
            out.append({"lambda": _finite(r.get("lambda")),
                        "max_kernel_overhead_pct": _finite(tc.get("max_kernel_overhead_pct")),
                        "clears_500_zero_overhead": bool(tc.get("clears_500_zero_overhead", False))})
        return out

    # ---- denken #212 AR(1)-corrected SPRT ASN (MERGED 19:54Z: sharpens #205's flat-Deff cost row) ---- #
    # Folding #190's DECAYING within-prompt ACF into the SPRT partial-sum variance tightens #205's flat
    # xDeff=4.41 by 1.59-2.66x. E[N]_nogo band = [405 IID -> 672 AR(1)-opt -> 1,125 measured-ACF-realistic
    # -> 1,788 flat-loose]; data-grounded point 1,125 (rho(2)=0.168 >> rho(1)^2=0.067). UNCHANGED:
    # (alpha,power)=(0.05,0.95), bar 0.9780, the 75.12x collapse vs #197's fixed-N is Deff-INVARIANT.
    def sprt_ar_landed(self) -> bool:
        return self.sprt_ar is not None

    def sprt_ar_self_test_passes(self) -> bool:
        return bool(_dig(self.sprt_ar, "synthesis", "self_test", "sprt_ar_self_test_passes",
                         default=False))

    def sprt_ar_expected_n_nogo(self, model: str) -> float | None:
        v = _dig(self.sprt_ar, "synthesis", "expected_n_corrected", "rows", "nogo", model)
        return float(v) if v is not None else None

    def sprt_ar_realistic_nogo(self) -> float | None:
        v = _dig(self.sprt_ar, "synthesis", "expected_n_corrected", "expected_n_nogo_empirical_acf")
        return float(v) if v is not None else None

    def sprt_ar_band(self) -> list:
        # [IID-floor, AR(1)-optimistic, measured-ACF-realistic, flat-loose]
        return [self.sprt_ar_expected_n_nogo("iid"), self.sprt_ar_expected_n_nogo("ar1"),
                self.sprt_ar_expected_n_nogo("empirical_acf"), self.sprt_ar_expected_n_nogo("flat_441")]

    def sprt_ar_deff(self, which: str) -> float | None:
        v = _dig(self.sprt_ar, "synthesis", "deff_comparison", which)
        return float(v) if v is not None else None

    def sprt_ar_savings_invariant(self) -> bool:
        return bool(_dig(self.sprt_ar, "synthesis", "expected_n_corrected", "savings_invariance",
                         "deff_invariant", default=False))

    def sprt_ar_savings_ratio(self) -> float | None:
        v = _dig(self.sprt_ar, "synthesis", "expected_n_corrected", "savings_invariance",
                 "savings_ratio_iid")
        return float(v) if v is not None else None

    def sprt_ar_n_fss_197(self) -> float | None:
        v = _dig(self.sprt_ar, "synthesis", "expected_n_corrected", "savings_invariance", "n_fss_197")
        return float(v) if v is not None else None

    def sprt_ar_realized_alpha(self) -> float | None:
        v = _dig(self.sprt_ar, "synthesis", "operating_characteristic_invariance", "realized_alpha")
        return float(v) if v is not None else None

    def sprt_ar_realized_power(self) -> float | None:
        v = _dig(self.sprt_ar, "synthesis", "operating_characteristic_invariance", "realized_power")
        return float(v) if v is not None else None

    def sprt_ar_private_bar(self) -> float | None:
        v = _dig(self.sprt_ar, "synthesis", "imports", "private_bar_both")
        return float(v) if v is not None else None

    def sprt_ar_rho_lag1(self) -> float | None:
        v = _dig(self.sprt_ar, "synthesis", "imports", "rho_lag1_190")
        return float(v) if v is not None else None

    def sprt_ar_tightening(self, which: str) -> float | None:
        v = _dig(self.sprt_ar, "synthesis", "deff_comparison", "tightening_%s_vs_flat" % which)
        return float(v) if v is not None else None

    def sprt_ar_flat_is_conservative(self) -> bool:
        return bool(_dig(self.sprt_ar, "synthesis", "realism_band", "flat_441_is_conservative",
                         default=False))

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
    # kanna #202 frozen-regime annotation (advisor 18:39Z): the conservative default build-bar input
    # is mu_bar_frozen_p95=504.87 (not fresh 499.08); the build-to-512.2/N=1 hedge is freeze-robust.
    # ANNOTATION ONLY -- single-shot GO/NO-GO + binding bar + sigma UNCHANGED. REINFORCES the HOLD.
    fba = frozen_budget_annotation()
    c202 = bool(fba.get("landed"))
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
        # kanna #202 frozen-regime annotation (conservative default + freeze-robust hedge).
        "frozen_budget_202_landed": c202,
        "frozen_regime_default": (fba.get("regime_default") if c202 else None),
        "mu_bar_frozen_p95_tps": (fba["mu_bar_frozen_p95_tps"] if c202 else None),
        "mu_bar_fresh_p95_n5_tps": (fba["mu_bar_fresh_p95_n5_tps"] if c202 else None),
        "p_bar_n5_frozen": (fba["p_bar_n5_frozen"] if c202 else None),
        "freeze_robust_build_to_mu_tps": (fba["freeze_robust_hedge"]["build_to_mu_tps"]
                                          if c202 else None),
        "n_shots_frozen_at_512": (fba["freeze_robust_hedge"]["n_shots_frozen"] if c202 else None),
        "frozen_fragile_e_shots_at_bar": (fba["build_at_bar_best_of_n_fragile"]["e_shots_frozen"]
                                          if c202 else None),
        "frozen_fraction_breakeven": (fba["build_at_bar_best_of_n_fragile"]["frozen_fraction_breakeven"]
                                      if c202 else None),
        "frozen_cost_crossover_206_pending": (fba.get("frozen_cost_crossover_206_pending")
                                              if c202 else None),
        "frozen_note": (fba["note"] if c202 else None),
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
    # Axis 7 -- denken #197 liveprobe measurement-cost (MERGED 18:39Z, CONSUMED): the INPUT-side
    #           CERTIFICATION budget + the FULL-LADDER GO requirement + the depth-1-only FALSE-GO
    #           guard. Distinct from #187 (input-side CI precision): #197 sizes how many liveprobe
    #           trials DECISIVELY certify the measured ladder clears the PRIVATE bar 0.9780. A
    #           depth-1-only / spine-inferred read is a FALSE GO worth 85.2 TPS; at beta=0.765 the
    #           mechanism CANNOT clear the private bar (perfect depth-1 -> 419.6 << 500). The COST row
    #           is REPLACED by denken #205 (SPRT): realistic E[N]~405 NO-GO, ~24,398 worst-case; #197's
    #           ~30,455 fixed-N is the worst-case CAP. REINFORCES the HOLD (does NOT flip it).
    lp = liveprobe_measurement_cost(None)
    lp_ok = bool(lp.get("landed"))
    rows.append({
        "axis": "liveprobe_measurement_cost", "pr": 197, "slug": "liveprobe-depth-budget",
        "kind": "measurement-cost",
        "cost_row_source_pr": (lp.get("cost_row_source_pr") if lp_ok else None),   # 205 (SPRT replaces #197)
        "status": "LANDED" if lp_ok else "IN-FLIGHT",
        "flag": "consumed" if lp_ok else "pending-liveprobe",
        "reinforces_hold": True,
        "full_ladder_required": (lp.get("full_ladder_required") if lp_ok else None),
        "min_depths_for_decisive": (lp.get("min_depths_for_decisive") if lp_ok else None),
        "depth1_plus_2_suffices": (lp.get("depth1_plus_2_suffices") if lp_ok else None),
        "false_go_risk_depth1_only": (lp.get("false_go_risk_depth1_only") if lp_ok else None),
        "depth1_overstatement_tps": (lp.get("depth1_overstatement_tps") if lp_ok else None),
        "mechanism_can_clear_private_bar": (lp.get("mechanism_can_clear_private_bar") if lp_ok else None),
        "private_lcb_perfect_depth1_tps": (lp.get("private_lcb_perfect_depth1_tps") if lp_ok else None),
        "beta_primary": (lp.get("beta_primary") if lp_ok else None),
        # cost row: denken #205 SPRT expected-N (the realistic measurement cost); 30k fixed-N = cap.
        "expected_n_sprt_nogo": (lp.get("expected_n_sprt_nogo") if lp_ok else None),
        "expected_n_sprt_nearbar": (lp.get("expected_n_sprt_nearbar") if lp_ok else None),
        "worst_case_expected_n_sprt": (lp.get("worst_case_expected_n_sprt") if lp_ok else None),
        "sprt_savings_vs_fixed_n": (lp.get("sprt_savings_vs_fixed_n") if lp_ok else None),
        "sprt_boundary_A_upper_decide_go": (lp.get("sprt_boundary_A_upper_decide_go") if lp_ok else None),
        "sprt_boundary_B_lower_decide_nogo": (lp.get("sprt_boundary_B_lower_decide_nogo") if lp_ok else None),
        "sprt_realized_alpha": (lp.get("sprt_realized_alpha") if lp_ok else None),
        "sprt_realized_power": (lp.get("sprt_realized_power") if lp_ok else None),
        "sprt_deff_190": (lp.get("sprt_deff_190") if lp_ok else None),
        "decisive_total_trials_lambda1": (lp.get("decisive_total_trials_lambda1") if lp_ok else None),
        "neyman_efficiency_gain_vs_equal": (lp.get("neyman_efficiency_gain_vs_equal") if lp_ok else None),
        "decisive_margin_at_lambda1": (lp.get("decisive_margin_at_lambda1") if lp_ok else None),
        "sequential_via_205_pending": (lp.get("sequential_via_205_pending") if lp_ok else None),
        "sequential_sprt_205_landed": (lp.get("sequential_sprt_205_landed") if lp_ok else None),
        "fixed_n_worst_case_cap_trials": (lp.get("fixed_n_worst_case_cap_trials") if lp_ok else None),
        "note": (lp.get("note") if lp_ok else "liveprobe #197 not landed -> no explicit full-ladder "
                 "/ false-GO guard; the GO leg uses the measured ladder as-is."),
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
    """The combined-sigma row. ubel #204 (MERGED 18:46Z, W&B m7vwuus2) RETIRES ubel #201's sigma->LCB
    trigger: #201's 520.09/522.69 was a UNITS BUG. The acceptance leg carried into #201 was wirbel
    #175's **z=1.96 two-sided HALF-WIDTH** (11.17 TPS), then double-counted against the z=1.645 P95
    LCB -- a 1-sigma quantity was inflated by z=1.96. #204 rebases every leg onto a clean 1-sigma
    footing: acceptance 11.17/1.95996 = **5.699 TPS**, folded with sigma_hw (#188 4.86) and
    sigma_private (#176/#191 0.88) -> combined LAUNCH sigma **7.545 central / 8.897 worst-case**
    (rho(*,hw) bounded [-0.3,+0.3]), REPLACING #201's 12.215/13.796.

    P95 framing (clean): the GO trigger is mu >= 500 + z_p95(1.6449)*sigma = **512.41 central /
    514.63 worst-case**, against the lambda=1 ceiling **520.95** TPS. BOTH are BELOW the ceiling ->
    lambda=1 clears 500 at P95 **CENTRALLY (+8.54 TPS) AND worst-case (+6.32 TPS)**. The verdict FLIPS
    from #201's PROVISIONAL knife-edge (worst-case NOT clearing) to a robust **RESOLVED-YES**. The
    de-dup x ICC mechanism is unchanged; only the unit footing is corrected (and the rebase direction
    came out SIGN-BACKWARDS vs the naive prediction -- the trigger went DOWN, not up).

    Still NON-GATING (advisor 18:46Z: "it does not authorize a launch"). ubel #207 (MERGED 19:40Z)
    RESOLVES the first caveat in FAVOR of the YES: the #175 two-readings tension is settled -- the
    larger 10.906 reading is the B=16384 128-tok-window SUB-bench, NOT launch-correct (the 2.106 ratio
    vs h_out 5.178 is bench-sqrtN x op-point, only COINCIDENTALLY ~= #190's sqrt(D) 2.100; reading it
    as h_out*sqrt(D) double-counts the ICC). The launch-correct full-generation h_out -> trigger
    512.41/514.63 STANDS, both < the 520.95 ceiling -> the robust-YES SURVIVES (under the conservative
    10.906 reading the worst-case 523.60 would sit ABOVE the ceiling, but that reading is mis-selected).
    Now ONE caveat remains OPEN, and it only TIGHTENS the YES: land #71 co-log (n=385 cross-device)
    retires the rho(*,hw) band. private-footing sensitivity is -0.063 TPS (negligible). (both-bugs only;
    descent is already NO-GO via #191.) NON-GATING: this row does NOT gate the analytic go -- it is the
    launch sigma->LCB readout the human reads. Falls back to #201's PROVISIONAL row if #204 is absent."""
    rebased = _BANKED.unit_rebase_landed()
    if not _BANKED.sigma_closure_landed() and not rebased:
        return {"landed": False, "note": "neither ubel #204 (clean rebase) nor #201 (launch-sigma "
                                         "closure) landed -> fall back to the #195 de-dup row "
                                         "(7.26/17.04) if present; else quadrature."}
    # #201 legs (the de-dup x ICC mechanism is SOLID; #201 supplies the leg vector + RETIRED trigger).
    sc201, wc201 = _BANKED.combined_sigma_launch("central"), _BANKED.combined_sigma_launch("worstcase")
    mu201_c, mu201_w = _BANKED.mu_clears_500("central"), _BANKED.mu_clears_500("worstcase")
    ceiling = _BANKED.rebase_lambda1_ceiling() if rebased else _BANKED.lambda1_ceiling_mu()
    if rebased:
        # ---- ubel #204 CLEAN-1-sigma trigger (RETIRES #201's 520.09/522.69) ----
        sc, wc = _BANKED.rebase_combined_sigma("central"), _BANKED.rebase_combined_sigma("worstcase")
        mu_c, mu_w = _BANKED.rebase_mu_clears_500("central"), _BANKED.rebase_mu_clears_500("worstcase")
        central_reach = _BANKED.rebase_lambda1_clears_500("central")    # True
        worst_reach = _BANKED.rebase_lambda1_clears_500("worstcase")    # True (NOW reachable)
        central_margin = _BANKED.rebase_headroom_below_ceiling("central")   # +8.54
        worst_margin = _BANKED.rebase_headroom_below_ceiling("worstcase")   # +6.32
    else:
        sc, wc = sc201, wc201
        mu_c, mu_w = mu201_c, mu201_w
        central_reach = _BANKED.lambda1_clears_500("central")
        worst_reach = _BANKED.lambda1_clears_500("worstcase")
        central_margin = _BANKED.margin_at_lambda1("central")
        worst_margin = _BANKED.margin_at_lambda1("worstcase")
    resolved_yes = bool(rebased and central_reach and worst_reach)
    return {
        "landed": True,
        "source_pr": 201,                                     # the de-dup x ICC mechanism/provenance
        "source_pr_trigger": 204 if rebased else 201,         # the trigger NUMBERS source
        "supersedes_195_726_1704": True,
        "supersedes_201_trigger": bool(rebased),              # #204 RETIRES #201's 520.09/522.69
        "provisional": (not rebased),                         # #204 RESOLVES it -> False
        "resolved": bool(rebased),                            # #204 -> RESOLVED-YES
        "resolved_yes": resolved_yes,                         # lambda=1 clears 500 at P95 both ends
        "gates_analytic_go": False,                           # NON-gating: not wired into `go`
        # ---- headline combined LAUNCH sigma (1-sigma): clean #204 (or #201 fallback) ----
        "combined_sigma_launch_central_tps": _finite(sc),     # 7.545 clean (was 12.215)
        "combined_sigma_launch_worstcase_tps": _finite(wc),   # 8.897 clean (was 13.796)
        # ---- P95 GO-trigger vs the lambda=1 ceiling ----
        "z_p95": _finite(_BANKED.sigma_closure_z_p95()),      # 1.6449
        "go_trigger_mu_central_tps": _finite(mu_c),           # 512.41 clean (was 520.09)
        "go_trigger_mu_worstcase_tps": _finite(mu_w),         # 514.63 clean (was 522.69)
        "lambda1_ceiling_mu_tps": _finite(ceiling),           # 520.95
        "central_p95_reachable": bool(central_reach),         # True  (+8.54)
        "worstcase_p95_reachable": bool(worst_reach),         # True  (+6.32) -- was False under #201
        "central_margin_at_lambda1_tps": _finite(central_margin),    # +8.54 (was +0.86)
        "worstcase_margin_at_lambda1_tps": _finite(worst_margin),    # +6.32 (was -1.74)
        # ---- ubel #204 clean-rebase specifics ----
        "unit_rebase_204_landed": bool(rebased),
        "clean_acceptance_1sigma_tps": (_finite(_BANKED.rebase_acceptance_1sigma_clean())
                                        if rebased else None),                # 5.699 (= 11.17/1.95996)
        "delta_mu_rebase_central_tps": (_finite(_BANKED.rebase_delta_mu("central"))
                                        if rebased else None),                # -7.68 (trigger went DOWN)
        "delta_mu_rebase_worstcase_tps": (_finite(_BANKED.rebase_delta_mu("worstcase"))
                                          if rebased else None),              # -8.06
        "clean_central_vs_ceiling": (_BANKED.rebase_clean_vs_ceiling("central") if rebased else None),    # BELOW
        "clean_worstcase_vs_ceiling": (_BANKED.rebase_clean_vs_ceiling("worstcase") if rebased else None),  # BELOW
        "does_lambda1_clear_500_at_p95_centrally": (
            _BANKED.rebase_does_lambda1_clear_p95_centrally() if rebased else None),   # "YES"
        "private_footing_shift_tps": (_finite(_BANKED.rebase_private_footing_shift())
                                      if rebased else None),                  # -0.063 (negligible)
        "rebase_direction_matches_prediction": (_BANKED.rebase_direction_matches_prediction()
                                                if rebased else None),        # False (sign-backwards)
        "anchor_err_195_dedup": (_finite(_BANKED.rebase_anchor_err_195()) if rebased else None),  # 0.0
        "anchor_err_194_breakeven": (_finite(_BANKED.rebase_anchor_err_194()) if rebased else None),  # 0.0
        # ---- #201 RETIRED trigger provenance (the units-bug numbers #204 supersedes) ----
        "superseded_201_trigger": ({
            "mu_201_central_tps": _finite(mu201_c),           # 520.09 RETIRED
            "mu_201_worstcase_tps": _finite(mu201_w),         # 522.69 RETIRED
            "combined_201_central_tps": _finite(sc201),       # 12.215 RETIRED (z=1.96 halfwidth)
            "note": "#201's trigger 520.09/522.69 RETIRED -- the acceptance leg 11.17 was a z=1.96 "
                    "two-sided HALF-WIDTH double-counted with the z=1.645 P95 LCB. #204 rebases to a "
                    "clean 1-sigma (5.699) -> trigger drops to 512.41/514.63, both clearing.",
        } if rebased else None),
        # ---- the de-dup x ICC mechanism (SOLID -- banked; sets the acceptance-axis identity) ----
        "acceptance_axis_dedup_iid_tps": _finite(_BANKED.acceptance_sigma_dedup_iid()),       # 5.32 (#195 IDENTITY)
        "design_effect": _finite(_BANKED.design_effect_201()),                                # 4.4106
        "sqrt_design_effect": _finite(_BANKED.sqrt_design_effect()),                          # 2.100
        "acceptance_axis_realistic_icc_halfwidth_tps": _finite(
            _BANKED.acceptance_sigma_dedup_realistic_icc()),   # 11.17 z=1.96 HALF-WIDTH (#204 rebases -> 5.699)
        "acceptance_axis_realistic_icc_tps": _finite(_BANKED.acceptance_sigma_dedup_realistic_icc()),  # 11.17 (legacy key)
        "sigma_vector_tps": {
            "acceptance": _finite(_BANKED.sigma_vector_leg("acceptance")),                    # 11.17 (#201 footing)
            "hardware": _finite(_BANKED.sigma_vector_leg("hardware")),                        # 4.86
            "private": _finite(_BANKED.sigma_vector_leg("private")),                          # 0.88
        },
        # ---- headroom vs #194's iid break-even (#201's erosion; #204 removes the double-count) ----
        "iid_break_even_194_tps": _finite(_BANKED.iid_break_even_194()),                      # 512.16
        "headroom_shift_tps": _finite(_BANKED.headroom_shift_from_iid()),                     # 7.94 (#201 erosion)
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
        # ---- ubel #207 (MERGED 19:40Z): the #175 two-readings tension RESOLVED in favor of the YES ----
        "sigma_reconcile_207": ({
            "landed": True,
            "robust_yes_survives": _BANKED.sigma_reconcile_robust_yes_survives(),          # True
            "launch_correct_reading": _BANKED.sigma_reconcile_launch_correct_reading(),    # h_out 5.178 (B=65536)
            "conservative_reading_is_launch_correct": _BANKED.sigma_reconcile_conservative_is_launch_correct(),  # False
            "lambda1_clears_under_conservative_reading": _BANKED.sigma_reconcile_lambda1_clears_conservative(),  # False
            "ratio_175_readings": _finite(_BANKED.sigma_reconcile_ratio_175()),            # 2.106
            "ratio_equals_sqrtD": _BANKED.sigma_reconcile_ratio_equals_sqrtd(),            # False (coincidence)
            "trigger_central_launch_correct_tps": _finite(_BANKED.sigma_reconcile_trigger_hout("central")),    # 512.41
            "trigger_worstcase_launch_correct_tps": _finite(_BANKED.sigma_reconcile_trigger_hout("worstcase")),  # 514.63
            "trigger_central_175sampling_subbench_tps": _finite(_BANKED.sigma_reconcile_trigger_175sampling("central")),    # 520.98 (RETIRED reading)
            "trigger_worstcase_175sampling_subbench_tps": _finite(_BANKED.sigma_reconcile_trigger_175sampling("worstcase")),  # 523.60 (RETIRED reading)
            "delta_trigger_reading_tps": _finite(_BANKED.sigma_reconcile_delta_trigger()),  # 8.57
            "note": "the 10.906 #175-sampling HW is the B=16384 128-tok-window SUB-bench, NOT the "
                    "official full-generation (B=65536) CI; the 2.106 ratio vs h_out 5.178 = bench-"
                    "sqrtN(2.04) x op-point(1.03), only COINCIDENTALLY ~= #190's sqrt(D) 2.100 (reading "
                    "10.906 as h_out*sqrt(D) double-counts the ICC). Launch-correct trigger 512.41/514.63 "
                    "STANDS; #204's robust-YES SURVIVES. The 10.906 reading is RETIRED.",
        } if _BANKED.sigma_reconcile_landed() else {"landed": False}),
        # ---- caveats: #207 RESOLVES the #175 tension; only land #71 remains open (and it TIGHTENS) ----
        "resolved_caveats": {
            "issue_207_175_two_readings": "RESOLVED by ubel #207 (19:40Z): the larger 10.906 reading is "
                "the B=16384 sub-bench, NOT launch-correct; the launch-correct h_out 5.178 -> 512.41/"
                "514.63 STANDS and the robust-YES SURVIVES. The 10.906 reading is retired.",
        },
        "open_caveats": {
            "land_71_colog": "co-log per-allocation acceptance x wall-TPS across n=%s cross-device "
                "allocations now TIGHTENS a YES (retires the rho(*,hw) [-0.3,+0.3] band), rather than "
                "rescuing a NO as it would have under #201." % (_BANKED.colog_n_allocations()),
        },
        "colog_n_allocations": _BANKED.colog_n_allocations(),  # 385
        "unit_convention_note": _BANKED.sigma_closure_unit_convention_note(),
        "verdict_line": (
            "RESOLVED-YES: after #204's clean-1-sigma rebase the launch sigma is %.2f central / %.2f "
            "worst-case; lambda=1 clears 500 at P95 CENTRALLY (+%.2f TPS) AND worst-case (+%.2f TPS) "
            "-- trigger %.2f/%.2f both BELOW the lambda=1 ceiling %.2f. NON-GATING: it does not "
            "authorize a launch." % (
                _finite(sc), _finite(wc), _finite(central_margin), _finite(worst_margin),
                _finite(mu_c), _finite(mu_w), _finite(ceiling)) if resolved_yes
            else "launch sigma->LCB row PROVISIONAL (ubel #204 clean rebase absent): central reachable "
                 "(+%.2f), worst-case %s." % (
                     _finite(central_margin) if central_margin is not None else float("nan"),
                     "reachable" if worst_reach else "UNREACHABLE")),
        "note": (
            "ubel #204 RETIRES #201's 520.09/522.69 (a z=1.96 two-sided HALF-WIDTH double-counted with "
            "the z=1.645 LCB). Clean 1-sigma acceptance 5.699 -> combined launch sigma %.2f central / "
            "%.2f worst-case ((+) sigma_hw 4.86 (+) sigma_priv 0.88). P95 GO trigger mu>=%.2f central / "
            "%.2f worst-case vs the lambda=1 ceiling %.2f -> BOTH BELOW -> RESOLVED-YES (central +%.2f, "
            "worst-case +%.2f). Direction came out sign-backwards (trigger DOWN by %.2f, not up). "
            "NON-GATING; ubel #207 RESOLVED the #175 two-readings tension (10.906 is the B=16384 "
            "sub-bench, retired) so the robust-YES SURVIVES; only land #71 co-log (n=%s) remains open "
            "and it merely TIGHTENS the YES." % (
                _finite(sc), _finite(wc), _finite(mu_c), _finite(mu_w), _finite(ceiling),
                _finite(central_margin), _finite(worst_margin),
                _finite(_BANKED.rebase_delta_mu("central")) if rebased else float("nan"),
                _BANKED.colog_n_allocations()) if rebased
            else "#201 PROVISIONAL fallback (ubel #204 absent): launch sigma %.2f/%.2f, trigger "
                 "%.2f/%.2f vs ceiling %.2f." % (
                     _finite(sc), _finite(wc), _finite(mu_c), _finite(mu_w), _finite(ceiling))),
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


def liveprobe_measurement_cost(ladder: list | None = None) -> dict:
    """denken #197 (MERGED 18:39Z, W&B wqr94io4): the liveprobe certification COST row + the
    FULL-LADDER GO requirement + the depth-1-only FALSE-GO guard. The GO must gate on land #71's
    MEASURED full-ladder q[2..9] clearing 0.9780 -- never a depth-1-only / spine-inferred read (which
    is a FALSE GO worth 85.2 TPS of overstatement). At the grounded beta=0.765 the mechanism CANNOT
    clear the private bar (even PERFECT depth-1 -> private_LCB 419.6 << 500), so a GO requires beta~1
    across the MEASURED ladder, not a point lambda_hat.

    COST ROW REPLACED by denken #205 (MERGED 19:09Z, SPRT): the measurement-cost row is no longer
    #197's truth-INDEPENDENT fixed-N ~30,455 draws; it is the REALISTIC sequential expected-N --
    E[N]~405 trials on a clear-NO-GO build (grounded beta=0.765, private_LCB 419.6<<500), a ~75.12x
    collapse, rising to ~14,915 only if the build is genuinely near the bar and PEAKING at the
    indifference point (worst-case ASN ~24,398). Boundaries A/B=+-2.9444 deliver realized
    (alpha,power)=(0.05,0.95); the ICC band (Deff=4.41) scales absolute counts but NOT the saving
    ratio. The SPRT KEEPS #197's shallow-heavy order (depths {2,3,4} carry 65% of the decisive info).
    #197's 30,455 fixed-N is retained as the worst-case-CAP reference. The FULL-LADDER GO guard is
    UNCHANGED. REINFORCES the HOLD (sharpens the measurement spec); does NOT flip it.

    COST-BAND SHARPENED by denken #212 (MERGED 19:54Z, AR(1)-corrected ASN): #205's flat xDeff=4.41
    is CONFIRMED CONSERVATIVE; folding #190's DECAYING within-prompt ACF (rho(1)=0.2583) into the SPRT
    partial-sum variance tightens the realism band 1.59-2.66x. The E[N]_nogo band becomes [405 IID ->
    672 AR(1)-optimistic -> 1,125 measured-ACF-realistic -> 1,788 flat-loose]; the data-grounded point
    is 1,125 (the measured ACF decays slower than pure AR(1): rho(2)=0.168 >> rho(1)^2=0.067). The
    75.12x collapse vs #197's fixed-N is Deff-INVARIANT, and realized (alpha,power)=(0.05,0.95) + the
    bar 0.9780 are UNCHANGED -- the AR correction only sharpens the absolute band. Orthogonal to #192."""
    if not _BANKED.liveprobe_landed():
        return {"landed": False, "note": "liveprobe #197 not landed -> GO leg uses the measured "
                                         "ladder as-is; no explicit full-ladder / false-GO guard."}
    n_meas = (len([q for q in ladder if q is not None]) if ladder is not None else None)
    full_ladder_measured = bool(n_meas is not None and n_meas >= 8)
    sprt = _BANKED.sprt_landed()
    return {
        "landed": True,
        "source_pr": 197,
        "cost_row_source_pr": 205 if sprt else 197,            # #205 SPRT replaces the #197 cost row
        "reinforces_hold": True,
        "flips_verdict": False,
        # ---- the FULL-LADDER GO requirement (measurement spec; UNCHANGED by #205) ----
        "full_ladder_required": True,
        "min_depths_for_decisive": _BANKED.liveprobe_min_depths_for_decisive(),     # "full-ladder"
        "min_depths_int": _BANKED.liveprobe_min_depths_int(),                       # 9
        "depth1_plus_2_suffices": _BANKED.liveprobe_depth1_plus_2_suffices(),       # False
        "measured_depths_in_tuple": n_meas,                                        # 8 (worked example)
        "full_ladder_measured": full_ladder_measured,                              # True (worked example)
        # ---- the depth-1-only FALSE-GO guard ----
        "false_go_risk_depth1_only": _BANKED.liveprobe_false_go_risk_depth1(),      # True
        "depth1_overstatement_tps": _finite(_BANKED.liveprobe_depth1_overstatement()),  # 85.21
        "true_private_lcb_at_lambda1_tps": _finite(_BANKED.liveprobe_true_private_lcb_at_lambda1()),  # 419.6
        # ---- the beta~1 confirmation (mechanism feasibility) ----
        "mechanism_can_clear_private_bar": _BANKED.liveprobe_mech_can_clear_private(),  # False
        "private_lcb_perfect_depth1_tps": _finite(_BANKED.liveprobe_private_lcb_perfect_depth1()),  # 419.6
        "beta_primary": _finite(_BANKED.liveprobe_beta_primary()),                  # 0.765
        "private_bar_both": _finite(_BANKED.liveprobe_private_bar()),               # 0.9780
        # ---- the certification COST row: REALISTIC SPRT expected-N (#205), 30k fixed-N as the CAP ----
        "sequential_via_205_pending": False,                                       # #205 has LANDED
        "sequential_sprt_205_landed": sprt,                                         # denken #205 SPRT
        # the REALISTIC measurement-cost the human reads now (NO-GO build is the realistic case):
        "expected_n_sprt_nogo": (_finite(_BANKED.sprt_expected_n_nogo()) if sprt else None),       # ~405
        "expected_n_sprt_nogo_realistic_icc": (_finite(_BANKED.sprt_expected_n_nogo_realistic_icc())
                                               if sprt else None),                  # ~1788 (Deff band)
        "expected_n_sprt_nearbar": (_finite(_BANKED.sprt_expected_n_nearbar()) if sprt else None),  # ~14,915
        "worst_case_expected_n_sprt": (_finite(_BANKED.sprt_worst_case_expected_n()) if sprt else None),  # ~24,398
        "worst_case_expected_n_sprt_realistic_icc": (
            _finite(_BANKED.sprt_worst_case_expected_n_realistic_icc()) if sprt else None),  # ~107,610
        "sprt_savings_vs_fixed_n": (_finite(_BANKED.sprt_savings_vs_fixed_n()) if sprt else None),  # ~75.12x
        "sprt_boundary_A_upper_decide_go": (_finite(_BANKED.sprt_boundary("A_upper_decide_go"))
                                            if sprt else None),                     # +2.9444
        "sprt_boundary_B_lower_decide_nogo": (_finite(_BANKED.sprt_boundary("B_lower_decide_nogo"))
                                              if sprt else None),                   # -2.9444
        "sprt_realized_alpha": (_finite(_BANKED.sprt_realized_alpha()) if sprt else None),  # 0.05
        "sprt_realized_power": (_finite(_BANKED.sprt_realized_power()) if sprt else None),  # 0.95
        "sprt_target_alpha": (_finite(_BANKED.sprt_target_alpha()) if sprt else None),      # 0.05
        "sprt_target_power": (_finite(_BANKED.sprt_target_power()) if sprt else None),      # 0.95
        "sprt_deff_190": (_finite(_BANKED.sprt_deff_190()) if sprt else None),              # 4.4106
        "sprt_beta_nogo": (_finite(_BANKED.sprt_beta_nogo()) if sprt else None),            # 0.765
        "sprt_private_lcb_nogo_tps": (_finite(_BANKED.sprt_private_lcb_nogo()) if sprt else None),  # 419.6
        # #197's fixed-N Neyman trials retained as the worst-case CAP reference:
        "decisive_total_trials_lambda1": _finite(_BANKED.liveprobe_decisive_total_trials()),  # 30455.40
        "decisive_N_d_budget_1to9": _BANKED.liveprobe_N_d_budget(),
        "neyman_efficiency_gain_vs_equal": _finite(_BANKED.liveprobe_neyman_efficiency_gain()),  # 1.43
        "decisive_margin_at_lambda1": _finite(_BANKED.liveprobe_decisive_margin_at_lambda1()),  # 0.022
        "fixed_n_worst_case_cap_trials": _finite(_BANKED.liveprobe_decisive_total_trials()),  # 30k fixed-N cap
        # ---- denken #212 (MERGED 19:54Z): AR(1)-corrected realism band for the SPRT cost ----
        # #205's flat xDeff=4.41 is CONFIRMED CONSERVATIVE; folding #190's DECAYING within-prompt ACF
        # tightens the realism band 1.59-2.66x. The data-grounded realistic E[N]_nogo is 1,125 (measured-
        # ACF), NOT the flat-loose 1,788; 405 (IID) and 1,788 (flat) are the envelope ends. The 75.12x
        # collapse vs #197's fixed-N is Deff-INVARIANT; realized (alpha,power)=(0.05,0.95) + bar 0.9780 UNCHANGED.
        "ar_corrected_cost_band_212": ({
            "landed": True,
            "source_pr": 212,
            "expected_n_nogo_band": _BANKED.sprt_ar_band(),    # [405 IID, 672 AR(1), 1125 measured-ACF, 1788 flat]
            "expected_n_nogo_iid_floor": _finite(_BANKED.sprt_ar_expected_n_nogo("iid")),          # 405
            "expected_n_nogo_ar_optimistic": _finite(_BANKED.sprt_ar_expected_n_nogo("ar1")),      # 672
            "expected_n_nogo_realistic_measured_acf": _finite(_BANKED.sprt_ar_realistic_nogo()),   # 1125 (data-grounded)
            "expected_n_nogo_flat_loose": _finite(_BANKED.sprt_ar_expected_n_nogo("flat_441")),    # 1788 (= #205 conservative end)
            "deff_ar": _finite(_BANKED.sprt_ar_deff("deff_ar_at_mbar")),                # 1.658 (optimistic)
            "deff_empirical_acf": _finite(_BANKED.sprt_ar_deff("deff_empirical_acf_measured")),  # 2.774 (realistic)
            "deff_flat_441": _finite(_BANKED.sprt_ar_deff("deff_flat_441")),            # 4.411 (conservative)
            "rho_lag1": _finite(_BANKED.sprt_ar_rho_lag1()),                            # 0.2583
            "tightening_empirical_vs_flat": _finite(_BANKED.sprt_ar_tightening("empirical")),  # 1.59x
            "tightening_ar_vs_flat": _finite(_BANKED.sprt_ar_tightening("ar")),         # 2.66x
            "flat_441_is_conservative": _BANKED.sprt_ar_flat_is_conservative(),         # True
            "savings_ratio_deff_invariant": _BANKED.sprt_ar_savings_invariant(),        # True
            "savings_ratio_unchanged": _finite(_BANKED.sprt_ar_savings_ratio()),        # 75.12 (UNCHANGED)
            "n_fss_197": _finite(_BANKED.sprt_ar_n_fss_197()),                          # 30455
            "realized_alpha_unchanged": _finite(_BANKED.sprt_ar_realized_alpha()),      # 0.05 (UNCHANGED)
            "realized_power_unchanged": _finite(_BANKED.sprt_ar_realized_power()),      # 0.95 (UNCHANGED)
            "private_bar_both_unchanged": _finite(_BANKED.sprt_ar_private_bar()),       # 0.9780 (UNCHANGED)
            "orthogonal_to_192": True,
            "self_test_passes": _BANKED.sprt_ar_self_test_passes(),
            "note": "folding #190's DECAYING ACF (rho(1)=0.2583) into the SPRT partial-sum variance "
                    "tightens #205's flat xDeff=4.41 by 1.59-2.66x. E[N]_nogo band = [%.0f IID -> %.0f "
                    "AR(1)-opt -> %.0f measured-ACF-realistic -> %.0f flat-loose]; the data-grounded "
                    "realistic point is %.0f (rho(2)=0.168 >> rho(1)^2=0.067, slower than pure AR(1)). "
                    "flat x4.41 is the CONSERVATIVE (loose) end. The 75.12x collapse is Deff-INVARIANT; "
                    "(alpha,power)=(0.05,0.95) + bar 0.9780 UNCHANGED. Orthogonal to #192." % (
                        _BANKED.sprt_ar_expected_n_nogo("iid") or float("nan"),
                        _BANKED.sprt_ar_expected_n_nogo("ar1") or float("nan"),
                        _BANKED.sprt_ar_realistic_nogo() or float("nan"),
                        _BANKED.sprt_ar_expected_n_nogo("flat_441") or float("nan"),
                        _BANKED.sprt_ar_realistic_nogo() or float("nan")),
        } if _BANKED.sprt_ar_landed() else {"landed": False}),
        # the data-grounded realistic measurement-cost the human reads now (denken #212, supersedes
        # #205's flat-loose 1,788 as the realistic point; 405/1,788 are the tight/loose envelope ends):
        "expected_n_nogo_realistic_measured_acf": (_finite(_BANKED.sprt_ar_realistic_nogo())
                                                   if _BANKED.sprt_ar_landed() else None),   # 1125
        "note": ("the GO must gate on land #71's MEASURED full-ladder q[2..9] >= 0.9780 (min_depths="
                 "full-ladder; depth-1+2 does NOT suffice) -- a depth-1-only read is a FALSE GO worth "
                 "%.1f TPS (true private LCB 419.6 << 500). At beta=0.765 NO build clears the private "
                 "bar (mech_can_clear=False) -> a GO needs beta~1 across the ladder. COST (denken #205 "
                 "SPRT, REPLACES #197's fixed-N): realistic E[N]~%.0f trials on a clear-NO-GO build "
                 "(%.1fx collapse vs #197's %.0f fixed-N), ~%.0f near-bar, worst-case ASN ~%.0f at the "
                 "indifference point; realized (alpha,power)=(%.2f,%.2f), boundaries +-%.4f; Deff=%.2f "
                 "scales counts not the ratio. Carry #197's %.0f fixed-N as the worst-case CAP. "
                 "REINFORCES the HOLD." % (
                     _BANKED.liveprobe_depth1_overstatement() or float("nan"),
                     _BANKED.sprt_expected_n_nogo() or float("nan"),
                     _BANKED.sprt_savings_vs_fixed_n() or float("nan"),
                     _BANKED.sprt_n_fixed_197() or float("nan"),
                     _BANKED.sprt_expected_n_nearbar() or float("nan"),
                     _BANKED.sprt_worst_case_expected_n() or float("nan"),
                     _BANKED.sprt_realized_alpha() or float("nan"),
                     _BANKED.sprt_realized_power() or float("nan"),
                     _BANKED.sprt_boundary("A_upper_decide_go") or float("nan"),
                     _BANKED.sprt_deff_190() or float("nan"),
                     _BANKED.liveprobe_decisive_total_trials() or float("nan"))
                 if sprt else
                 "the GO must gate on land #71's MEASURED full-ladder q[2..9] >= 0.9780; cost ~%.0f "
                 "fixed-N Neyman trials @lambda=1 (denken #205 SPRT not yet landed). REINFORCES the "
                 "HOLD." % (_BANKED.liveprobe_decisive_total_trials() or float("nan"))),
    }


def frozen_budget_annotation() -> dict:
    """kanna #202 (MERGED 18:39Z, W&B 533jd6l1): the multi-shot budget under the conservative FROZEN
    regime. Under fixed prompts + deterministic greedy the official harness re-benchmarks IDENTICAL
    tokens, so the per-checkpoint sampling deviation is a COMMON bias and best-of-N beats down ONLY
    sigma_hw (66% of one-sigma), NOT sigma_sample. So #194's N=5-at-the-bar does NOT reach P>=0.95
    under freeze (frozen P=0.810, not fresh 0.969). Carry the conservative default build-bar input
    mu_bar_frozen_p95=504.87 (not fresh 499.08). THE HEDGE: build-to-mu=512.2 / N=1 is fully
    freeze-robust (a single draw has the same sigma_draw in both regimes; n_shots_frozen_at_512=1),
    so the SAFE recommendation survives untouched. Only build-at-bar+best-of-N is frozen-fragile
    (E[shots]=2.34, exhausts-without-clearing 19%, breakeven f*=0.846). REINFORCES the HOLD."""
    if not _BANKED.frozen_budget_landed():
        return {"landed": False, "note": "frozen_budget #202 not landed -> multi-shot budget assumes "
                                         "the FRESH regime (#194 N=5@bar P=0.969); optimistic if the "
                                         "harness re-benchmarks frozen tokens."}
    return {
        "landed": True,
        "source_pr": 202,
        "reinforces_hold": True,
        "regime_default": "FROZEN",          # conservative default until the human pins the harness
        "regime_is_open": True,              # WHICH regime applies is the harness-owner's open question
        # ---- conservative default build-bar input under freeze ----
        "mu_bar_frozen_p95_tps": _finite(_BANKED.mu_bar_frozen_p95()),           # 504.87
        "mu_bar_fresh_p95_n5_tps": _finite(_BANKED.mu_bar_fresh_p95_n5()),       # 499.08
        "p_bar_n5_frozen": _finite(_BANKED.p_bar_n5_frozen()),                   # 0.810 (NOT 0.969)
        "delta_mu_frozen_tps": _finite(_BANKED.frozen_delta_mu()),               # -7.28
        "sigma_fraction_beatable_frozen": _finite(_BANKED.frozen_sigma_fraction_beatable()),  # 0.658
        # ---- THE HEDGE: build-to-512.2 / N=1 is freeze-robust (the SAFE recommendation, untouched) ----
        "freeze_robust_hedge": {
            "build_to_mu_tps": _finite(_BANKED.frozen_mu_safe_tps()),            # 512.16
            "n_shots_frozen": _BANKED.frozen_n_shots_at_512(),                   # 1
            "regime_invariant": True,
            "note": "build clear of the bar (mu>=512.2) and take ONE shot -- a single draw has the "
                    "same sigma_draw in both regimes, so this path is fully freeze-robust.",
        },
        # ---- the frozen-FRAGILE shortcut (build-at-bar + best-of-N) ----
        "build_at_bar_best_of_n_fragile": {
            "e_shots_frozen": _finite(_BANKED.frozen_e_shots_at_bar()),          # 2.34
            "e_shots_fresh": _finite(_BANKED.fresh_e_shots_at_bar()),            # 1.94
            "exhaust_without_clear_frac_frozen": _finite(_BANKED.frozen_p_exhaust_without_clear()),  # 0.19
            "frozen_fraction_breakeven": _finite(_BANKED.frozen_fraction_breakeven()),  # 0.846
            "note": "build-at-bar mu=500 + best-of-N is frozen-fragile: pays E[shots]=2.34 (vs fresh "
                    "1.94) and exhausts WITHOUT clearing 19% of the time; f*=0.846 of sigma_sample "
                    "must re-randomize for N=5 to hold P>=0.95.",
        },
        "frozen_cost_crossover_206_pending": True,   # kanna -> #206 (build_higher_dominates_below_b)
        "note": "multi-shot budget under the conservative FROZEN default: best-of-N beats ONLY "
                "sigma_hw -> N=5@bar clears P=%.3f (NOT fresh 0.969); the default build-bar input is "
                "mu_bar_frozen_p95=%.2f (not fresh %.2f). THE HEDGE (carry prominently): build-to-"
                "mu=512.2 / N=1 is fully freeze-robust (n_shots_frozen=1) -> the SAFE recommendation "
                "is untouched. Only build-at-bar+best-of-N is frozen-fragile (E[shots]=%.2f, exhausts "
                "19%%, breakeven f*=%.3f). kanna -> #206 frozen-cost crossover (carry "
                "build_higher_dominates_below_b when it lands). REINFORCES the HOLD." % (
                    _BANKED.p_bar_n5_frozen() or float("nan"),
                    _BANKED.mu_bar_frozen_p95() or float("nan"),
                    _BANKED.mu_bar_fresh_p95_n5() or float("nan"),
                    _BANKED.frozen_e_shots_at_bar() or float("nan"),
                    _BANKED.frozen_fraction_breakeven() or float("nan")),
    }


def winners_curse_annotation() -> dict:
    """kanna #210 (MERGED 19:40Z, W&B hwvv7nn1): best-of-N does NOT relax the binding PRIVATE bar.
    The conditional private clear is EXACTLY FLAT in N (n_star_private=1): best-of-N selects the MAX
    public shot, inflating it by sigma_sel*E[Z_(N:N)] of NON-replicating noise that never appears in
    the private re-benchmark (Capen 1971 winner's curse / Smith-Winkler 2006 optimizer's curse). The
    private grade is one fresh draw whose distribution is unchanged by the public selection. So to
    clear 500-PRIVATE at P>=0.95 under a best-of-5 launch trigger the PUBLIC build must reach
    mu_bar_private_corrected=**528.48** (+23.61 winner's-curse tax over #202's public-only frozen bar
    504.87 = 7.28 evaporating public best-of-N discount + 16.33 private-drop gross-up). REGIME-INVARIANT.

    This CORRECTS the #202 hedge: build-to-512.2 / N=1 is freeze-robust for the PUBLIC number but does
    NOT survive the PRIVATE winner's curse (p_private=0.3120, NOT >=0.95) -- the 2.35% adverse drop x
    tau_low sinks its private mean to 496.38 < 500. The N=1 prescription STANDS (against the private
    bar, BUILDING HIGHER strictly dominates RE-DRAWING MORE; best-of-N is self-defeating), but the
    BUILD TARGET rises to 528.48. This is the multi-shot BUILD-target row; it does NOT touch the
    sigma->LCB PUBLIC trigger (512.41, ubel #204/#207 -- a different quantity). REINFORCES the HOLD."""
    if not _BANKED.winners_curse_landed():
        return {"landed": False, "note": "winners_curse #210 not landed -> the multi-shot budget row "
                                         "may UNDER-state the private build target (best-of-N looks "
                                         "like it relaxes the bar; it does not -- the private clear is "
                                         "flat in N). Use #202's 504.87 public bar as a LOWER bound."}
    tax = _BANKED.winners_curse_tax_decomposition()
    return {
        "landed": True,
        "source_pr": 210,
        "reinforces_hold": True,
        "regime_invariant": _BANKED.winners_curse_regime_invariant(),       # True (FRESH/FROZEN same answer)
        # ---- the headline: best-of-N does NOT relax the PRIVATE bar (flat in N) ----
        "private_clear_flat_in_n": _BANKED.winners_curse_private_clear_flat_in_n(),   # True
        "n_star_private": _BANKED.winners_curse_n_star_private(),            # 1 (build higher, not re-draw)
        "recommendation": "build higher (mu_pub >= 528.48), N=1",
        # ---- the winner's-curse-corrected PRIVATE build target ----
        "mu_bar_private_corrected_tps": _finite(_BANKED.winners_curse_mu_bar_private_corrected()),  # 528.48
        "mu_bar_frozen_public_202_tps": _finite(_BANKED.winners_curse_mu_bar_frozen_202()),         # 504.87
        "mu_safe_fresh_194_tps": _finite(_BANKED.winners_curse_mu_safe_fresh_194()),                # 512.16
        "delta_mu_winners_curse_tps": _finite(_BANKED.winners_curse_delta_mu()),                    # +23.61 tax
        "tax_decomposition_tps": {
            "public_bestofN_discount_evaporates": _finite(tax.get("public_bestofN_discount_evaporates_tps")),  # 7.28
            "private_drop_grossup": _finite(tax.get("private_drop_grossup_tps")),                   # 16.33
            "sum": _finite(tax.get("sum_tps")),                                                     # 23.61
        },
        # ---- the #202 hedge (build-to-512.2/N=1) does NOT survive the PRIVATE winner's curse ----
        "freeze_robust_512_survives_private": _BANKED.winners_curse_freeze_robust_512_survives(),   # False
        "p_private_clear_at_mu512p2_n1": _finite(_BANKED.winners_curse_p_private_at_512()),         # 0.3120 (< 0.95)
        # ---- the winner's-curse inflation that EVAPORATES privately (best-of-5) ----
        "winners_curse_tps_n5_frozen": _finite(_BANKED.winners_curse_tps_n5("frozen")),             # 5.66
        "winners_curse_tps_n5_fresh": _finite(_BANKED.winners_curse_tps_n5("fresh")),               # 8.60
        "private_bar_lambda_star_191": _finite(_BANKED.winners_curse_lambda_star_191()),            # 0.9780
        # ---- explicit: this is the BUILD-target row, NOT the sigma->LCB PUBLIC trigger ----
        "does_not_change_sigma_lcb_trigger": True,   # 512.41 (public-bar P95 trigger) is a different quantity
        "note": "best-of-N does NOT relax the binding PRIVATE bar (stark #191 lambda*_LCB 0.9780): the "
                "conditional private clear is FLAT in N (n_star_private=1) because selection is on non-"
                "replicating public noise and the private grade is one fresh draw. To clear 500-private "
                "at P>=0.95 under best-of-5 the PUBLIC build must reach mu_bar_private_corrected=%.2f "
                "(+%.2f winner's-curse tax over #202's 504.87 = %.2f evaporating public discount + %.2f "
                "private-drop gross-up). The #202 freeze-robust mu=512.2/N=1 does NOT survive privately "
                "(p=%.4f < 0.95). N=1 STANDS (building higher dominates re-drawing more); the BUILD "
                "TARGET rises to %.2f. Does NOT touch the sigma->LCB public trigger 512.41. REINFORCES "
                "the HOLD." % (
                    _BANKED.winners_curse_mu_bar_private_corrected() or float("nan"),
                    _BANKED.winners_curse_delta_mu() or float("nan"),
                    (tax.get("public_bestofN_discount_evaporates_tps") or float("nan")),
                    (tax.get("private_drop_grossup_tps") or float("nan")),
                    _BANKED.winners_curse_p_private_at_512() or float("nan"),
                    _BANKED.winners_curse_mu_bar_private_corrected() or float("nan")),
    }


def compliant_lane_bracket() -> dict:
    """lawine #196 (lane-b non-spec floor, MERGED 18:58Z) + wirbel #199 (lane-a compliant-spec E[T]
    ceiling, MERGED 19:09Z): the ISSUE #192 COMPLIANCE bracket that sits ABOVE the sigma->LCB trigger.

    Issue #192 -- the human ruling on whether the speculative int4 verify token must be IDENTICAL to
    the AR greedy argmax -- decides WHICH lane is admissible at 500 TPS:

      * lane-b (non-spec, lawine #196, EMPIRICAL): a compliant non-spec int4 path IS token-identical
        (rate 1.0), PPL 2.3766 < 2.42, 128/128 complete -- but it FLOORS at ~165.44 official TPS,
        66.9% BELOW 500. There is NO compliant non-spec 500-lane; the speculation premium (316.1 TPS,
        191%) is EXISTENTIAL.

      * lane-a (compliant-spec, wirbel #199, BRACKET): the batch-invariant int4 VERIFY kernel ceiling
        = 536.66 official TPS (finite-sample lower-CI 525.73 > 500 -> CLEARS), floor = 416.31 (MISSES).
        Clears 500 ONLY IF the batch-invariant kernel costs < 7.33% (both-bugs) / 4.12% (descent)
        verify-step overhead -- UNMEASURED; kanna #122's off-the-shelf int4 is +51.78% (~7x over budget).

    wirbel #213 (lane-a capstone, MERGED 19:40Z) grades lane-a's overhead budget vs lambda:
    max_kernel_overhead_pct(lambda) opens from <=0 at lambda_hat=0.342 (the realistic floor already
    misses 500 -- even a FREE kernel fails) to 7.33% (both) / 4.12% (descent) at lambda=1; the zero-
    overhead path first clears 500 at lambda_crit=**0.8345** both / **0.9067** descent. So lane-a is a
    DOUBLE gate: build self-KV-recovery lambda >= 0.8345 AND hold the batch-invariant verify kernel
    under the lambda-graded budget. Off-shelf #122 (+51.78%) clears at NO physical lambda<=1.

    NET: under strict #192, lane-a is the SINGLE compliant route to 500, a DOUBLE gate (lambda>=0.8345
    AND kernel-overhead < max_overhead(lambda), the latter UNMEASURED). This does NOT change the
    sigma->LCB GO trigger (ubel #204/#207's 512.41/514.63 stands); it adds a COMPLIANCE PRECONDITION
    that gates the launch ABOVE the sigma math. The launch stays HELD on the three hard gates (land #71
    build, measured lambda_hat >= 0.9780 q[2..9] direct, issue #192 human ruling)."""
    nl = _BANKED.nonspec_floor_landed()
    cl = _BANKED.compliant_spec_landed()
    if not (nl and cl):
        return {"landed": False, "source_prs": [196, 199],
                "nonspec_floor_landed": bool(nl), "compliant_spec_landed": bool(cl),
                "note": "compliant-lane modules not both present -> #192 bracket unresolved."}
    ceiling = _BANKED.compliant_spec_ceiling()
    floor = _BANKED.compliant_spec_floor()
    ci_lower_bb = _BANKED.compliant_spec_ceiling_ci_lower("both_bugs")
    lower_clears = _BANKED.compliant_spec_lower_clears_500("both_bugs")
    oh_bb = _BANKED.compliant_spec_max_overhead("both_bugs")
    oh_desc = _BANKED.compliant_spec_max_overhead("descent_only")
    offshelf = _BANKED.compliant_spec_offshelf_overhead_ref()
    nonspec = _BANKED.nonspec_official_tps()
    margin = _BANKED.nonspec_margin_pct()
    kb = _BANKED   # wirbel #213 kernel-budget accessors (lane-a capstone)
    return {
        "landed": True,
        "source_prs": [196, 199, 213] if kb.kernel_budget_landed() else [196, 199],
        "reinforces_hold": True,
        # CRUX: this row sits ABOVE the sigma math -- it does NOT gate or change #204's sigma->LCB trigger.
        "gates_sigma_lcb_trigger": False,
        "sits_above_sigma_math": True,
        "does_not_change_204_trigger": True,
        "issue_192_human_ruling_pending": True,
        "binding_compliance_gate": "issue_192_human_ruling",
        # ---- lane-b: non-spec compliant floor (lawine #196, EMPIRICAL) ----
        "lane_b_nonspec": {
            "official_tps_floor": _finite(nonspec),                          # 165.44
            "hw_band_tps": _BANKED.nonspec_floor_band("hw"),                 # [160.58, 170.30]
            "token_identity_rate": _finite(_BANKED.nonspec_token_identity()),  # 1.0
            "ppl": _finite(_BANKED.nonspec_ppl()),                           # 2.3766
            "completes_128": _BANKED.nonspec_completes_128(),               # True
            "clears_500": _BANKED.nonspec_clears_500(),                      # False
            "margin_to_500_pct": _finite(margin),                           # -66.9
            "spec_premium_tps": _finite(_BANKED.nonspec_spec_premium_tps()),   # 316.1
            "spec_premium_pct": _finite(_BANKED.nonspec_spec_premium_pct()),   # 191.1
            "verdict_label": _BANKED.nonspec_verdict_label(),               # STRUCTURAL_GAP_SPEC_EXISTENTIAL
            "self_test_passes": _BANKED.nonspec_floor_self_test_passes(),
        },
        # ---- lane-a: compliant-spec E[T] ceiling (wirbel #199, BRACKET) ----
        "lane_a_compliant_spec": {
            "tps_ceiling": _finite(ceiling),                  # 536.66
            "tps_floor": _finite(floor),                      # 416.31
            "ceiling_ci_lower_tps": _finite(ci_lower_bb),     # 525.73
            "ceiling_lower_clears_500": bool(lower_clears),   # True
            "ceiling_clears_500": _BANKED.compliant_spec_clears_500(),      # True
            "floor_clears_500": _BANKED.compliant_spec_floor_clears_500(),  # False
            "max_kernel_overhead_pct_both_bugs": _finite(oh_bb),    # 7.33
            "max_kernel_overhead_pct_descent": _finite(oh_desc),    # 4.12
            "offshelf_overhead_ref_122": _finite(offshelf),         # 0.5178 (+51.78%)
            "overhead_is_measured": False,                          # UNMEASURED feasibility
            "self_test_passes": _BANKED.compliant_spec_self_test_passes(),
        },
        # ---- lane-a capstone: the kernel-overhead budget vs lambda (wirbel #213, MERGED 19:40Z) ----
        # max_kernel_overhead_pct(lambda) -- the lane-a 500-route is a DOUBLE gate: self-KV-recovery
        # lambda >= lambda_crit AND batch-invariant verify kernel under the lambda-graded overhead budget.
        "lane_a_kernel_budget_213": ({
            "landed": True,
            "source_pr": 213,
            "lambda_crit_clears_500_both_bugs": _finite(kb.kernel_budget_lambda_crit("both_bugs")),   # 0.8345
            "lambda_crit_clears_500_descent": _finite(kb.kernel_budget_lambda_crit("descent")),       # 0.9067
            "overhead_budget_pct_at_lambda1_both_bugs": _finite(kb.kernel_budget_overhead_at_lambda1("both_bugs")),  # 7.33
            "overhead_budget_pct_at_lambda1_descent": _finite(kb.kernel_budget_overhead_at_lambda1("descent")),      # 4.12
            "overhead_budget_pct_at_lambda_hat_both_bugs": _finite(kb.kernel_budget_overhead_at_lambda_hat()),       # -16.74
            "lambda_hat": _finite(kb.kernel_budget_lambda_hat()),                  # 0.342 (realistic floor)
            "offshelf_122_clears_at_physical_lambda": kb.kernel_budget_offshelf_clears_at_physical_lambda(),  # False
            "max_budget_pct_at_prob_saturation_both_bugs": _finite(kb.kernel_budget_max_at_saturation()),     # 50.10
            "budget_curve_both_bugs": kb.kernel_budget_curve("both_bugs"),         # [(lambda, pct, clears)...]
            "verdict": kb.kernel_budget_verdict(),                                  # BUDGET-OPENS-ONLY-ABOVE-LAMBDA-CRIT
            "self_test_passes": kb.kernel_budget_self_test_passes(),
            "note": "the budget opens from <=0 at lambda_hat=0.342 (the realistic floor already misses "
                    "500 -- even a FREE batch-invariant kernel fails) to 7.33%% (both) / 4.12%% (descent) "
                    "at lambda=1; the zero-overhead path first clears 500 at lambda_crit=0.8345 both / "
                    "0.9067 descent. Off-shelf #122 (+51.78%%) clears at NO physical lambda<=1 (the "
                    "lambda=1 budget is only 7.33%%, ~7.1x over). So lane-a is a DOUBLE gate: build "
                    "self-KV-recovery lambda >= 0.8345 AND hold the kernel under max_overhead(lambda).",
        } if kb.kernel_budget_landed() else {"landed": False}),
        # ---- NET: the #192 compliance precondition ----
        "compliant_500_lane_exists": bool(lower_clears),     # lane-a lower-CI clears -> a compliant 500-path CAN exist...
        "compliant_500_lane_is_spec_only": True,             # ...but ONLY via lane-a (the batch-invariant verify kernel)
        "compliant_500_conditioned_on_unmeasured_overhead": True,
        "compliant_500_lane_is_double_gate": bool(kb.kernel_budget_landed()),   # #213: lambda>=lambda_crit AND kernel-under-budget
        "lane_a_lambda_crit_both_bugs": _finite(kb.kernel_budget_lambda_crit("both_bugs")) if kb.kernel_budget_landed() else None,  # 0.8345
        "lane_a_lambda_crit_descent": _finite(kb.kernel_budget_lambda_crit("descent")) if kb.kernel_budget_landed() else None,      # 0.9067
        "note": "ISSUE #192 compliance bracket (ABOVE the sigma->LCB trigger; does NOT change ubel "
                "#204's 512.41/514.63). lane-b (non-spec, EMPIRICAL): token-identical (1.0), PPL "
                "%.4f<2.42, 128/128 -- but FLOORS at %.2f official TPS (%.1f%% below 500); NO compliant "
                "non-spec 500-lane, the spec premium (%.1f TPS, %.1f%%) is EXISTENTIAL. lane-a "
                "(compliant-spec, BRACKET): ceiling %.2f (lower-CI %.2f > 500 -> CLEARS), floor %.2f "
                "(misses); clears 500 ONLY if the batch-invariant verify-kernel overhead < %.2f%% both "
                "/ %.2f%% descent -- UNMEASURED (off-shelf #122 is +%.1f%%, ~7x over). NET: lane-a is "
                "the SINGLE compliant route to 500, conditioned on an unmeasured feasibility; the launch "
                "stays HELD on the three hard gates (land #71 build, measured lambda_hat>=0.9780 "
                "q[2..9] direct, issue #192 human ruling). REINFORCES the HOLD." % (
                    _BANKED.nonspec_ppl() or float("nan"),
                    nonspec or float("nan"),
                    abs(margin) if margin is not None else float("nan"),
                    _BANKED.nonspec_spec_premium_tps() or float("nan"),
                    _BANKED.nonspec_spec_premium_pct() or float("nan"),
                    ceiling or float("nan"),
                    ci_lower_bb or float("nan"),
                    floor or float("nan"),
                    oh_bb or float("nan"),
                    oh_desc or float("nan"),
                    (offshelf * 100.0) if offshelf is not None else float("nan")),
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
        # ISSUE #192 human ruling (lawine #196 lane-b + wirbel #199 lane-a compliance bracket): a HARD
        # gate ABOVE the sigma->LCB trigger. Under strict enforcement lane-a (batch-invariant verify
        # kernel ceiling 536.66, lower-CI 525.73>500) is the ONLY compliant 500-route; lane-b non-spec
        # floors at 165.44 (66.9% below 500). PENDING -> keeps launch_authorized=False.
        {"row": "issue_192_human_ruling", "pr": 192, "kind": "precondition", "status": "PENDING",
         "flag": "human-ruling",
         "note": "issue #192 (the human ruling on whether the speculative int4 verify token must be "
                 "IDENTICAL to the AR greedy argmax) must resolve in favor of an admissible compliant "
                 "lane. Under strict enforcement lane-a (batch-invariant verify kernel, wirbel #199 "
                 "ceiling 536.66, lower-CI 525.73>500) is the ONLY compliant 500-route -- conditioned "
                 "on an UNMEASURED <7.33%% kernel overhead; lane-b non-spec floors at 165.44 (lawine "
                 "#196, 66.9%% below 500). Sits ABOVE the sigma->LCB trigger (does NOT change #204's "
                 "512.41/514.63)."},
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

    # ---- combined-sigma row (ubel #204, advisor 18:46Z): RESOLVES the #201 row. #201's TRIGGER was a
    #      UNITS bug (acceptance leg 11.17 = z=1.96 two-sided HALF-WIDTH double-counted with the z=1.645
    #      P95 LCB). #204 re-foots every leg onto a clean 1-sigma: acceptance 11.17/1.95996 = 5.699 ->
    #      combined LAUNCH sigma 7.545 central / 8.897 worst-case (was 12.215/13.796). P95 GO trigger
    #      mu>=512.41 central / 514.63 worst-case (was 520.09/522.69) vs the lambda=1 ceiling 520.95 ->
    #      BOTH BELOW -> lambda=1 clears 500 at P95 CENTRALLY (+8.54) AND worst-case (+6.32). The verdict
    #      FLIPS from #201's PROVISIONAL knife-edge (worst-case UNREACHABLE) to a robust RESOLVED-YES.
    #      Still NON-GATING -- it does NOT enter `go` (advisor 18:46Z: "it does not authorize a launch").
    #      The de-dup x ICC MECHANISM (#195/#190) is unchanged; only the unit footing is corrected.
    #      both-bugs only; descent is already NO-GO via #191 so the row is non-binding there. ----
    if topo == "both_bugs":
        csc = combined_sigma_corner()
        lsc_landed = bool(csc.get("landed"))
        lsc_central_p95_reachable = bool(csc.get("central_p95_reachable")) if lsc_landed else None
        lsc_worstcase_p95_reachable = bool(csc.get("worstcase_p95_reachable")) if lsc_landed else None
        lsc_provisional = bool(csc.get("provisional", True)) if lsc_landed else None
        lsc_resolved = bool(csc.get("resolved", False)) if lsc_landed else None
    else:
        csc = None
        lsc_central_p95_reachable = None
        lsc_worstcase_p95_reachable = None
        lsc_provisional = None
        lsc_resolved = None

    # ---- FULL-LADDER GO requirement + depth-1-only FALSE-GO guard (#197 denken, advisor 18:39Z) ----
    # The GO leg must gate on land #71's MEASURED full-ladder q[2..9] (>=8 depths) clearing the private
    # bar 0.9780 -- NEVER a depth-1-only or spine-inferred read. A depth-1-only GO is a FALSE GO worth
    # 85.2 TPS (true private LCB 419.6 << 500; at the grounded beta=0.765 the mechanism CANNOT clear
    # the private bar, so a real GO needs beta~1 across the MEASURED ladder, not a point lambda_hat).
    # full_ladder_ok GATES `go` (it BLOCKS a depth-1-only false GO) but is NON-FLIPPING for a properly
    # measured tuple: it passes when #197 is not landed OR the measured ladder carries >=8 depths. The
    # worked example carries the 8-entry q[2..9] spine -> full_ladder_ok=True -> both-bugs GO HELD.
    lpc = liveprobe_measurement_cost(t.get("q_ladder"))
    lp_landed = bool(lpc.get("landed"))
    full_ladder_measured = bool(lpc.get("full_ladder_measured")) if lp_landed else None
    full_ladder_ok = bool((not lp_landed) or lpc.get("full_ladder_measured"))

    # ---- overall per-topology GO: validity AND trustworthy AND binding-build AND BOTH launch LCBs
    #      (realistic ICC #190 + private #191) AND the #197 full-ladder measurement guard. The #201
    #      combined-sigma row is PROVISIONAL/NON-GATING (advisor 18:23Z: HOLD the verdict -- do NOT
    #      hard-wire the GO/NO-GO vs the lambda=1 ceiling while #204 + land #71 are open) -> it is
    #      surfaced in lambda_gate but NOT folded into `go`. The #197 full-ladder guard, by contrast,
    #      DOES gate `go` (advisor 18:39Z: "require the full-ladder measurement before emitting GO") --
    #      but is SATISFIED by the worked example, so the analytic verdict is unchanged. ----
    go = bool(validity_ok and oa["trustworthy"] and binding_build_gate_pass
              and realistic_launch_pass and private_launch_pass and full_ladder_ok)

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
    elif not full_ladder_ok:
        failing_gate = ("full-ladder measurement guard (#197): GO requires land #71's MEASURED "
                        "full-ladder q[2..9] (>=8 depths) clearing %.4f -- the tuple carries only %s "
                        "depth(s). A depth-1-only / spine-inferred read is a FALSE GO worth %.1f TPS "
                        "(true private LCB %.1f<<500; at beta=%.3f the mechanism CANNOT clear the "
                        "private bar, so a real GO needs beta~1 across the MEASURED ladder)." % (
                            lpc.get("private_bar_both") or float("nan"),
                            lpc.get("measured_depths_in_tuple"),
                            lpc.get("depth1_overstatement_tps") or float("nan"),
                            lpc.get("true_private_lcb_at_lambda1_tps") or float("nan"),
                            lpc.get("beta_primary") or float("nan")))
        restoration = ("measure land #71's full q[2..9] ladder before emitting GO (the REALISTIC cost "
                       "is denken #205's sequential E[N]~%.0f trials on a clear-NO-GO build, ~%.0f "
                       "worst-case at the bar; #197's ~%.0f fixed-N is the worst-case CAP); a "
                       "depth-1-only read is not certifiable." % (
                           lpc.get("expected_n_sprt_nogo") or float("nan"),
                           lpc.get("worst_case_expected_n_sprt") or float("nan"),
                           lpc.get("decisive_total_trials_lambda1") or float("nan")))
    # NOTE: the #204 combined-sigma row is RESOLVED-YES/NON-GATING (advisor 18:46Z) -> it is NOT a
    # gating candidate either way. lambda=1 now clears 500 at P95 both ends, but the row still does NOT
    # authorize a launch; the analytic verdict is held on the three hard gates, not this sigma row.

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
            # combined-sigma row (ubel #204, RESOLVES #201): the clean-1-sigma launch sigma 7.545/8.897
            # -> P95 GO trigger 512.41/514.63 vs the lambda=1 ceiling 520.95 -> BOTH BELOW -> lambda=1
            # clears 500 at P95 centrally (+8.54) AND worst-case (+6.32). RESOLVED-YES + NON-GATING
            # (advisor 18:46Z: "it does not authorize a launch"). both-bugs only (None on descent).
            "combined_sigma_corner": csc,
            "launch_sigma_central_p95_reachable": lsc_central_p95_reachable,     # True  (+8.54)
            "launch_sigma_worstcase_p95_reachable": lsc_worstcase_p95_reachable,  # True  (+6.32) -- was False under #201
            "launch_sigma_provisional": lsc_provisional,                         # False -- #204 RESOLVES it
            "launch_sigma_resolved": lsc_resolved,                               # True  -- RESOLVED-YES
            "launch_sigma_gates_go": False,                                      # advisor 18:46Z: NOT hard-wired
            # full-ladder GO requirement + depth-1-only FALSE-GO guard (#197, advisor 18:39Z). UNLIKE
            # the #201 row, this measurement-spec guard DOES gate `go` (it BLOCKS a depth-1-only false
            # GO) -- but is SATISFIED by the worked example's 8-entry measured ladder -> both-bugs GO HELD.
            "liveprobe_measurement_cost": lpc,
            "full_ladder_required": (lpc.get("full_ladder_required") if lp_landed else None),
            "full_ladder_measured": full_ladder_measured,
            "full_ladder_ok": full_ladder_ok,
            "full_ladder_gates_go": True,                                        # advisor 18:39Z: REQUIRED before GO
            "false_go_risk_depth1_only": (lpc.get("false_go_risk_depth1_only") if lp_landed else None),
            "depth1_overstatement_tps": (lpc.get("depth1_overstatement_tps") if lp_landed else None),
            "mechanism_can_clear_private_bar": (lpc.get("mechanism_can_clear_private_bar")
                                                if lp_landed else None),
            # cost row: denken #205 SPRT expected-N (REPLACES #197's fixed-N); 30k retained as the cap.
            "decisive_total_trials_lambda1": (lpc.get("decisive_total_trials_lambda1")
                                              if lp_landed else None),
            "fixed_n_worst_case_cap_trials": (lpc.get("fixed_n_worst_case_cap_trials")
                                              if lp_landed else None),
            "liveprobe_sequential_via_205_pending": (lpc.get("sequential_via_205_pending")
                                                     if lp_landed else None),
            "liveprobe_sequential_sprt_205_landed": (lpc.get("sequential_sprt_205_landed")
                                                     if lp_landed else None),
            "liveprobe_expected_n_sprt_nogo": (lpc.get("expected_n_sprt_nogo") if lp_landed else None),
            "liveprobe_expected_n_sprt_nearbar": (lpc.get("expected_n_sprt_nearbar")
                                                  if lp_landed else None),
            "liveprobe_worst_case_expected_n_sprt": (lpc.get("worst_case_expected_n_sprt")
                                                     if lp_landed else None),
            "liveprobe_sprt_savings_vs_fixed_n": (lpc.get("sprt_savings_vs_fixed_n")
                                                  if lp_landed else None),
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
    # combined-sigma row (ubel #204, RESOLVES #201): the clean-1-sigma launch sigma -> P95
    # GO-trigger-vs-ceiling readout. RESOLVED-YES + NON-GATING (advisor 18:46Z: does not authorize a launch).
    csc_ledger = combined_sigma_corner()
    # cost-aware re-draw budget (#200): sequential early-stop spend + build-higher-vs-stay toggle.
    # ANNOTATION ONLY -- single-shot GO/NO-GO + binding bar + sigma are unchanged.
    cba_ledger = cost_budget_annotation()
    # liveprobe certification cost + FULL-LADDER GO requirement (#197 denken; cost row REPLACED by #205
    # SPRT, advisor 19:09Z): the GO leg GATES on land #71's MEASURED full-ladder q[2..9] (a depth-1-only
    # read is a FALSE GO worth 85.2 TPS); REALISTIC cost = E[N]~405 SPRT (30k fixed-N = cap). REINFORCES HOLD.
    lpc_ledger = liveprobe_measurement_cost(t.get("q_ladder"))
    # multi-shot budget under the conservative FROZEN regime (#202 kanna, advisor 18:39Z): default
    # build-bar input mu_bar_frozen_p95=504.87; build-to-512.2/N=1 hedge is freeze-robust. REINFORCES HOLD.
    fba_ledger = frozen_budget_annotation()
    # ISSUE #192 compliance bracket (lawine #196 lane-b + wirbel #199 lane-a, advisor 18:58Z/19:09Z):
    # sits ABOVE the sigma->LCB trigger. lane-b non-spec floors at 165.44 (66.9% below 500); lane-a
    # compliant-spec ceiling 536.66 (lower-CI 525.73>500) is the ONLY compliant 500-route, conditioned
    # on an UNMEASURED <7.33% kernel overhead. Does NOT change #204's trigger; REINFORCES the HOLD.
    # wirbel #213 (advisor 19:40Z) extends clb with the kernel-overhead-vs-lambda budget (DOUBLE gate).
    clb_ledger = compliant_lane_bracket()
    # winner's-curse correction to the multi-shot budget (kanna #210, advisor 19:40Z): best-of-N does
    # NOT relax the PRIVATE bar (flat in N, n_star_private=1); the private build target is 528.48 (+23.61
    # tax over #202's 504.87). Build-higher/N=1. Does NOT touch the sigma->LCB public trigger (512.41).
    wca_ledger = winners_curse_annotation()

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
                        "validity AND over-accept) AND (all precondition rows GO). The combined-sigma "
                        "row is RESOLVED-YES + NON-GATING (ubel #204, advisor 18:46Z: surfaced, NOT "
                        "folded into the analytic GO -- 'it does not authorize a launch'). RECOMPOSED "
                        "post-merge: binding_bar = max(public#183 0.9052, ICC#190 0.9513, private#191 "
                        "0.9780) = 0.9780 (private#191 dominates, both-bugs); descent private is "
                        "UNREACHABLE. Sampling half-width is the REALISTIC ICC#190 +/-22.9 TPS "
                        "(design-effect %.4f over iid +/-10.9), NOT the iid placeholder. CLOSED "
                        "post-#195 (the 4-axis quadrature was INVALID: rho(sampling,input)=0.945 "
                        "double-count). The de-dup x ICC mechanism (#195/#190): the de-dup acceptance "
                        "axis (5.32 iid IDENTITY) under realistic ICC (sqrt(D)=2.100 MAGNITUDE -> 11.17 "
                        "z=1.96 HALF-WIDTH). ubel #204 RETIRES #201's trigger (a UNITS bug -- that 11.17 "
                        "half-width was double-counted with the z=1.645 P95 LCB): clean 1-sigma = "
                        "11.17/1.95996 = 5.699 -> combined LAUNCH sigma 7.545 central / 8.897 worst-case "
                        "(was 12.215/13.796); P95 GO trigger mu>=512.41 central / 514.63 worst-case (was "
                        "520.09/522.69) vs the lambda=1 ceiling 520.95 -> BOTH BELOW -> lambda=1 clears "
                        "500 at P95 CENTRALLY (+8.54) AND worst-case (+6.32) -> RESOLVED-YES (was a "
                        "PROVISIONAL knife-edge; direction came out sign-backwards, trigger DROPPED 7.68 "
                        "not up). ubel #207 (advisor 19:40Z) RESOLVED the #175 two-readings tension in "
                        "FAVOR of the YES: the larger 10.906 reading is the B=16384 128-tok SUB-bench "
                        "(the 2.106 ratio vs h_out 5.178 = bench-sqrtN x op-point, only COINCIDENTALLY "
                        "~= sqrt(D) 2.100), NOT launch-correct -> the 512.41/514.63 trigger STANDS and "
                        "the robust-YES SURVIVES (10.906 retired); only land #71 co-log (n=385, now "
                        "TIGHTENS the YES) remains open. iid "
                        "#175 leg kept as a visible fallback row only. NO pending numerical axes "
                        "(ledger CLOSED). #200 cost annotation (budget row only, single-shot logic "
                        "UNCHANGED): realistic spend at the bar is SEQUENTIAL E[shots]=1.94 (not "
                        "fixed-5); build-higher (mu>=512.2/N=1) beats stay-at-bar iff reaching mu "
                        "costs < 4 shots' GPU-$ (c*=3.04*b fixed / 12.97*b sequential). #197 (denken, "
                        "advisor 18:39Z): the GO leg GATES on land #71's MEASURED full-ladder q[2..9] "
                        ">= 0.9780 -- NEVER depth-1-only/spine-inferred (a depth-1-only read is a FALSE "
                        "GO worth 85.2 TPS; at beta=0.765 the mechanism CANNOT clear the private bar, "
                        "so a real GO needs beta~1 across the MEASURED ladder). #205 (denken, advisor "
                        "19:09Z) REPLACES #197's fixed-N cost row with the realistic SPRT: the "
                        "measurement-cost row is E[N]~405 on a clear-NO-GO build (75.1x collapse vs "
                        "#197's 30,455 fixed-N), ~14,915 near-bar, <=24,398 worst-case ASN; realized "
                        "(alpha,power)=(0.05,0.95), boundaries +-2.9444. The full-ladder GO guard is "
                        "UNCHANGED (cost-row swap, not a verdict change); carry 30,455 as the fixed-N "
                        "worst-case cap. #212 (denken, advisor 19:54Z) SHARPENS the SPRT cost band: "
                        "#205's flat xDeff=4.41 is CONFIRMED CONSERVATIVE; folding #190's DECAYING ACF "
                        "tightens it 1.59-2.66x -> E[N]_nogo band [405 IID -> 672 AR(1) -> 1,125 "
                        "measured-ACF-realistic -> 1,788 flat-loose], data-grounded point 1,125. The "
                        "75.1x collapse is Deff-INVARIANT; (alpha,power)=(0.05,0.95) + bar 0.9780 "
                        "UNCHANGED. #202 (kanna, advisor 18:39Z): the multi-shot budget "
                        "DEFAULTS to the conservative FROZEN regime (best-of-N beats ONLY sigma_hw -> "
                        "N=5@bar P=0.810 not fresh 0.969; default build-bar input mu_bar_frozen_p95="
                        "504.87 not fresh 499.08); THE HEDGE -- build-to-mu=512.2/N=1 is fully freeze-"
                        "robust (n_shots_frozen=1), so the SAFE recommendation is untouched; only "
                        "build-at-bar+best-of-N is frozen-fragile (E[shots]=2.34, exhausts 19%%, "
                        "breakeven f*=0.846). #210 (kanna, advisor 19:40Z) CORRECTS the #202 hedge "
                        "against the PRIVATE bar: best-of-N does NOT relax it (the conditional private "
                        "clear is FLAT in N, n_star_private=1 -- selection is on non-replicating public "
                        "noise, the private grade is one fresh draw), and the freeze-robust mu=512.2/N=1 "
                        "does NOT survive privately (p=0.3120<0.95). To clear 500-PRIVATE at P>=0.95 the "
                        "PUBLIC build must reach mu_bar_private_corrected=528.48 (+23.61 winner's-curse "
                        "tax over #202's 504.87 = 7.28 evaporating public discount + 16.33 private-drop "
                        "gross-up); N=1 STANDS (build higher, do NOT re-draw). This is the BUILD-target "
                        "row; it does NOT touch the sigma->LCB PUBLIC trigger 512.41. BOTH #197 + #202 "
                        "REINFORCE the HOLD -- they sharpen the measurement spec + budget robustness, "
                        "they do NOT flip the verdict. ISSUE #192 "
                        "COMPLIANCE BRACKET (lawine #196 lane-b + wirbel #199 lane-a, advisor 18:58Z/"
                        "19:09Z) sits ABOVE this sigma math and does NOT change the #204 trigger: lane-b "
                        "(non-spec, EMPIRICAL) is token-identical but FLOORS at 165.44 official TPS "
                        "(66.9%% below 500) -> NO compliant non-spec 500-lane (spec premium 316.1 TPS / "
                        "191%% existential); lane-a (compliant-spec int4 VERIFY kernel) ceiling 536.66 "
                        "(lower-CI 525.73>500 -> CLEARS), floor 416.31, clears 500 ONLY at <7.33%% both "
                        "/ 4.12%% descent kernel overhead (UNMEASURED; off-shelf #122 +51.78%%, ~7x "
                        "over). wirbel #213 (advisor 19:40Z) grades lane-a's overhead budget vs lambda: "
                        "max_overhead(lambda) opens from <=0 at lambda_hat=0.342 (the realistic floor "
                        "misses 500 even with a FREE kernel) to 7.33%% both / 4.12%% descent at lambda=1; "
                        "the zero-overhead path first clears 500 at lambda_crit=0.8345 both / 0.9067 "
                        "descent. So lane-a is a DOUBLE gate (lambda>=0.8345 AND kernel-under-budget), "
                        "the SINGLE compliant 500-route, and a NEW hard precondition "
                        "row (issue_192_human_ruling, PENDING) gates the launch on the #192 ruling -- "
                        "keeping launch_authorized=False independent of the analytic/sigma logic."
                        % _BANKED.design_effect(),
            "numerical_axes": {"both_bugs": bb_ledger,
                               "descent_only": numerical_ci_ledger("descent_only", TAU_HEADLINE)},
            "combined_sigma_corner": csc_ledger,
            "cost_budget_annotation": cba_ledger,
            "liveprobe_measurement_cost": lpc_ledger,
            "frozen_budget_annotation": fba_ledger,
            "winners_curse_annotation": wca_ledger,
            "compliant_lane_bracket": clb_ledger,
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
                    r.get("halfwidth_tps_iid", r.get("n_shots_for_p95_at_bar",
                    r.get("decisive_total_trials_lambda1"))))))
        val_s = ("%.4f" % val) if isinstance(val, (int, float)) else (
            str(val) if val is not None else "pending")
        num_md.append("| %s | #%s | %s | %s | %s |" % (
            r["axis"], r["pr"], r["status"], r["flag"], val_s))
    num_table = "\n".join(num_md)
    pre_md = ["| precondition | status | flag |", "|---|---|---|"]
    for r in led["preconditions"]:
        pre_md.append("| %s | %s | %s |" % (r["row"], r["status"], r["flag"]))
    pre_table = "\n".join(pre_md)
    # combined-sigma row: ubel #204 clean-1-sigma trigger (RESOLVED-YES) + cost-aware budget (#200).
    csc = led.get("combined_sigma_corner") or {}
    cba = led.get("cost_budget_annotation") or {}
    if csc.get("landed"):
        sigma_line = (
            "**Combined-sigma row (ubel #204 -- RESOLVED-YES, NON-GATING, does NOT authorize a "
            "launch):** de-dup (#195) sets the acceptance-axis IDENTITY (5.32 TPS iid, removing the "
            "rho(sampling,input)=%.3f double-count); realistic ICC (#190) sets its MAGNITUDE "
            "(sqrt(D)=%.3f -> %.2f TPS). ubel #204 caught a UNITS bug: that %.2f TPS was a z=1.96 "
            "two-sided HALF-WIDTH double-counted with the z=1.645 P95 LCB -> clean 1-sigma = **%.3f "
            "TPS**. (+) sigma_hw (+) sigma_private -> combined LAUNCH sigma **%.2f central / %.2f "
            "worst-case** (1-sigma). P95 GO trigger mu >= 500 + z_p95*sigma = **%.2f central / %.2f "
            "worst-case** vs the lambda=1 ceiling **%.2f** -> **BOTH BELOW** -> lambda=1 clears 500 at "
            "P95 **CENTRALLY (+%.2f TPS) AND worst-case (+%.2f TPS)**. The verdict FLIPS from #201's "
            "PROVISIONAL knife-edge to a robust **RESOLVED-YES** (direction came out sign-backwards: "
            "the dominant acceptance leg is DIVIDED down by z2, so the trigger SHIFTED %.2f TPS -- DOWN, "
            "not up). ubel #207 RESOLVED the #175 two-readings caveat in FAVOR of the YES (the larger "
            "10.906 reading is the B=16384 SUB-bench, RETIRED -> 512.41/514.63 STANDS); only land #71 "
            "co-log (n=%s) remains open, and it now TIGHTENS the YES (retiring the rho(*,hw) [-0.3,+0.3] "
            "band). NO change to the binding BUILD bar (private 0.9780, #191) -- purely the launch "
            "sigma->LCB row." % (
                csc["dedup_provenance_195"]["rho_sampling_input"], csc["sqrt_design_effect"],
                csc["acceptance_axis_realistic_icc_halfwidth_tps"],
                csc["acceptance_axis_realistic_icc_halfwidth_tps"], csc["clean_acceptance_1sigma_tps"],
                csc["combined_sigma_launch_central_tps"], csc["combined_sigma_launch_worstcase_tps"],
                csc["go_trigger_mu_central_tps"], csc["go_trigger_mu_worstcase_tps"],
                csc["lambda1_ceiling_mu_tps"], csc["central_margin_at_lambda1_tps"],
                csc["worstcase_margin_at_lambda1_tps"], csc["delta_mu_rebase_central_tps"],
                csc["colog_n_allocations"]))
    else:
        sigma_line = ("**Combined-sigma row:** ubel #204 clean rebase not landed -> #201 PROVISIONAL "
                      "fallback (12.215/13.796, trigger 520.09/522.69).")
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
    # liveprobe certification cost + FULL-LADDER GO requirement (#197) + frozen-budget regime (#202).
    lpc = led.get("liveprobe_measurement_cost") or {}
    fba = led.get("frozen_budget_annotation") or {}
    if lpc.get("landed"):
        liveprobe_line = (
            "**Full-ladder GO requirement + depth-1-only FALSE-GO guard (#197 denken -- GATES the GO "
            "leg; the worked tuple SATISFIES it):** GO requires land #71's **MEASURED full-ladder "
            "q[2..9]** (>=8 depths; min_depths=%s, depth1+2 does NOT suffice) clearing the private bar "
            "%.4f -- NEVER a depth-1-only/spine-inferred read. A depth-1-only GO is a **FALSE GO worth "
            "%.1f TPS** (true private LCB %.1f << 500). At the grounded **beta=%.3f the mechanism "
            "CANNOT clear the private bar** (mechanism_can_clear=%s; perfect depth-1 -> %.1f << 500), "
            "so a real GO needs beta~1 ACROSS the measured ladder, not a point lambda_hat. The worked "
            "tuple carries the full 8-entry q[2..9] spine (full_ladder_measured=%s) -> guard SATISFIED, "
            "verdict HELD. Decisive certification cost (denken #205 SPRT REPLACES #197's fixed-N row): "
            "**E[N]~%.0f trials on a clear-NO-GO build** (%.1fx collapse vs #197's %.0f fixed-N "
            "shallow-heavy @lambda=1), ~%.0f near-bar, **<=%.0f worst-case ASN**; realized "
            "(alpha,power)=(%.2f,%.2f), Wald boundaries +-%.4f. Full-ladder GO guard UNCHANGED (cost-row "
            "swap, not a verdict change); carry %.0f fixed-N as the worst-case cap. REINFORCES the HOLD." % (
                lpc.get("min_depths_for_decisive"), lpc.get("private_bar_both") or float("nan"),
                lpc.get("depth1_overstatement_tps") or float("nan"),
                lpc.get("true_private_lcb_at_lambda1_tps") or float("nan"),
                lpc.get("beta_primary") or float("nan"),
                lpc.get("mechanism_can_clear_private_bar"),
                lpc.get("private_lcb_perfect_depth1_tps") or float("nan"),
                lpc.get("full_ladder_measured"),
                lpc.get("expected_n_sprt_nogo") or float("nan"),
                lpc.get("sprt_savings_vs_fixed_n") or float("nan"),
                lpc.get("decisive_total_trials_lambda1") or float("nan"),
                lpc.get("expected_n_sprt_nearbar") or float("nan"),
                lpc.get("worst_case_expected_n_sprt") or float("nan"),
                lpc.get("sprt_realized_alpha") or float("nan"),
                lpc.get("sprt_realized_power") or float("nan"),
                lpc.get("sprt_boundary_A_upper_decide_go") or float("nan"),
                lpc.get("fixed_n_worst_case_cap_trials") or float("nan")))
    else:
        liveprobe_line = ("**Full-ladder GO requirement (#197):** not landed -> the GO leg uses the "
                          "measured ladder as-is; no explicit full-ladder / depth-1 false-GO guard.")
    if fba.get("landed"):
        hedge = fba.get("freeze_robust_hedge", {})
        fragile = fba.get("build_at_bar_best_of_n_fragile", {})
        frozen_line = (
            "**Multi-shot budget under the conservative FROZEN regime (#202 kanna -- the DEFAULT; "
            "REINFORCES the HOLD):** under fixed prompts + deterministic greedy the official harness "
            "re-benchmarks IDENTICAL tokens, so per-checkpoint sampling deviation is a COMMON bias and "
            "best-of-N beats down ONLY sigma_hw (%.0f%% of one-sigma). So #194's N=5-at-the-bar does "
            "**NOT** reach P>=0.95 under freeze (**frozen P=%.3f, not fresh 0.969**); the default "
            "build-bar input is **mu_bar_frozen_p95=%.2f** (not fresh %.2f; delta %.2f TPS). **THE "
            "HEDGE (carry prominently):** build-to-**mu=%.1f / N=1** is fully freeze-robust "
            "(n_shots_frozen=%s; a single draw has the same sigma_draw in both regimes), so the SAFE "
            "recommendation is **untouched**. Only build-at-bar+best-of-N is frozen-fragile "
            "(E[shots]=%.2f vs fresh %.2f; exhausts WITHOUT clearing %.0f%% of the time; breakeven "
            "f*=%.3f of sigma_sample must re-randomize). kanna -> #206 frozen-cost crossover (carry "
            "build_higher_dominates_below_b when it lands). REINFORCES the HOLD." % (
                (fba.get("sigma_fraction_beatable_frozen") or 0.0) * 100.0,
                fba.get("p_bar_n5_frozen") or float("nan"),
                fba.get("mu_bar_frozen_p95_tps") or float("nan"),
                fba.get("mu_bar_fresh_p95_n5_tps") or float("nan"),
                fba.get("delta_mu_frozen_tps") or float("nan"),
                hedge.get("build_to_mu_tps") or float("nan"), hedge.get("n_shots_frozen"),
                fragile.get("e_shots_frozen") or float("nan"),
                fragile.get("e_shots_fresh") or float("nan"),
                (fragile.get("exhaust_without_clear_frac_frozen") or 0.0) * 100.0,
                fragile.get("frozen_fraction_breakeven") or float("nan")))
    else:
        frozen_line = ("**Frozen-budget regime (#202):** not landed -> the multi-shot budget assumes "
                       "the FRESH regime (#194 N=5@bar P=0.969); optimistic if the harness re-benches "
                       "frozen tokens.")
    # ISSUE #192 compliance bracket (lawine #196 lane-b + wirbel #199 lane-a) -- ABOVE the sigma math.
    clb = led.get("compliant_lane_bracket") or {}
    if clb.get("landed"):
        lb = clb["lane_b_nonspec"]
        lna = clb["lane_a_compliant_spec"]
        compliant_line = (
            "**Issue #192 compliance bracket (lawine #196 lane-b + wirbel #199 lane-a -- sits ABOVE "
            "the sigma->LCB trigger; does NOT change ubel #204's 512.41/514.63):** issue #192 (whether "
            "the speculative int4 verify token must be IDENTICAL to the AR greedy argmax) decides which "
            "lane is admissible at 500. **lane-b (non-spec, EMPIRICAL):** token-identical (%.3f), PPL "
            "%.4f<2.42, 128/128 -- but **FLOORS at %.2f official TPS (%.1f%% below 500)**; there is NO "
            "compliant non-spec 500-lane (spec premium %.1f TPS / %.1f%% is EXISTENTIAL; %s). **lane-a "
            "(compliant-spec batch-invariant VERIFY kernel, BRACKET):** ceiling **%.2f** official TPS "
            "(finite-sample lower-CI **%.2f > 500 -> CLEARS**), floor %.2f (misses); clears 500 **ONLY "
            "IF** kernel overhead < **%.2f%% (both-bugs) / %.2f%% (descent)** -- **UNMEASURED** "
            "(off-shelf #122 is +%.1f%%, ~7x over budget). **NET:** lane-a is the SINGLE compliant "
            "500-route, conditioned on an unmeasured feasibility -> a NEW hard precondition "
            "(issue_192_human_ruling, PENDING) gates the launch ABOVE the sigma math. REINFORCES the "
            "HOLD." % (
                lb.get("token_identity_rate") or float("nan"), lb.get("ppl") or float("nan"),
                lb.get("official_tps_floor") or float("nan"),
                abs(lb.get("margin_to_500_pct")) if lb.get("margin_to_500_pct") is not None else float("nan"),
                lb.get("spec_premium_tps") or float("nan"), lb.get("spec_premium_pct") or float("nan"),
                lb.get("verdict_label"),
                lna.get("tps_ceiling") or float("nan"), lna.get("ceiling_ci_lower_tps") or float("nan"),
                lna.get("tps_floor") or float("nan"),
                lna.get("max_kernel_overhead_pct_both_bugs") or float("nan"),
                lna.get("max_kernel_overhead_pct_descent") or float("nan"),
                (lna.get("offshelf_overhead_ref_122") or 0.0) * 100.0))
    else:
        compliant_line = ("**Issue #192 compliance bracket (#196/#199):** not landed -> no compliant-"
                          "lane row; the #192 ruling precondition is carried abstractly.")
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

{liveprobe_line}

{frozen_line}

{compliant_line}

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
  - [LANDED-CONSUMED] ubel #201 launch-sigma closure -- de-dup x realistic-ICC MECHANISM retained as provenance (acceptance axis 5.32 iid IDENTITY x sqrt(D)=2.100 -> 11.17 z=1.96 half-width). Its sigma->LCB TRIGGER (12.215/13.796 -> 520.09/522.69, worst-case UNREACHABLE) is RETIRED by ubel #204 (a units bug); see below.
  - [LANDED-CONSUMED] ubel #204 launch-sigma UNIT-REBASE (RESOLVED-YES, NON-GATING) -- RETIRES #201's trigger: the 11.17 acceptance leg was a z=1.96 two-sided HALF-WIDTH double-counted with the z=1.645 P95 LCB; clean 1-sigma = 11.17/1.95996 = 5.699 -> combined LAUNCH sigma 7.545 central / 8.897 worst-case; P95 GO trigger 512.41/514.63 vs the lambda=1 ceiling 520.95 -> BOTH BELOW -> lambda=1 clears 500 at P95 CENTRALLY (+8.54) AND worst-case (+6.32). Verdict FLIPS PROVISIONAL knife-edge -> RESOLVED-YES (direction sign-backwards: trigger DROPPED 7.68, not up). Caveats #207 (#175 two-readings) + land #71 co-log (n=385, now TIGHTENS the YES) remain open but neither flips it. "It does not authorize a launch."
  - [LANDED-CONSUMED] denken #197 liveprobe depth-budget -- the GO leg GATES on land #71's MEASURED full-ladder q[2..9] >= 0.9780 (depth-1-only/spine-inferred is a FALSE GO worth 85.2 TPS; at beta=0.765 mechanism CANNOT clear the private bar -> needs beta~1 across the ladder). Cost row REPLACED by #205 SPRT (below); ~30,455 fixed-N -> worst-case cap. REINFORCES the HOLD; the worked tuple's 8-entry ladder SATISFIES the guard.
  - [LANDED-CONSUMED] denken #205 SPRT liveprobe-budget -- REPLACES #197's fixed-N cost row with the realistic expected-N: the measurement-cost row is E[N]~405 on a clear-NO-GO build (75.1x collapse vs #197's 30,455 fixed-N), ~14,915 near-bar, <=24,398 worst-case ASN (107,610 realistic-ICC); realized (alpha,power)=(0.05,0.95), Wald boundaries +-2.9444, Deff=4.41. Full-ladder GO guard UNCHANGED (cost-row swap, not a verdict change); carry 30,455 as the fixed-N worst-case cap. REINFORCES the HOLD.
  - [LANDED-CONSUMED] denken #212 AR(1)-corrected SPRT cost band -- SHARPENS #205's flat-Deff cost row: folding #190's DECAYING within-prompt ACF (rho(1)=0.2583) into the SPRT partial-sum variance tightens the realism band 1.59-2.66x. Certification-cost band E[N]_nogo = [405 IID-floor -> 672 AR(1)-optimistic -> 1,125 measured-ACF-REALISTIC -> 1,788 flat-loose]; the data-grounded point is 1,125 (rho(2)=0.168 >> rho(1)^2=0.067, decays slower than pure AR(1)), confirming #205's flat x4.41 as the CONSERVATIVE loose end. The 75.12x collapse vs #197's 30,455 fixed-N is Deff-INVARIANT (both scale by the same cluster Deff); realized (alpha,power)=(0.05,0.95) and the binding bar 0.9780 are UNTOUCHED. Orthogonal to #192 (liveprobe cost, not the greedy-identity reading). REINFORCES the HOLD.
  - [LANDED-CONSUMED] kanna #202 frozen-budget regime (conservative DEFAULT) -- best-of-N beats only sigma_hw, so N=5@bar clears P=0.810 (not fresh 0.969); default build-bar input mu_bar_frozen_p95=504.87 (not fresh 499.08). HEDGE: build-to-512.2/N=1 is fully freeze-robust (n_shots_frozen=1) -> SAFE recommendation untouched; only build-at-bar+best-of-N is frozen-fragile (E[shots]=2.34, exhausts 19%, f*=0.846). kanna -> #206 crossover. REINFORCES the HOLD.
  - [LANDED-CONSUMED] kanna #210 winner's-curse private-bar correction -- best-of-N does NOT relax the PRIVATE bar (selection on non-replicating public noise; the private grade is one fresh draw -> conditional clear is FLAT in N, n_star_private=1). To clear 500-private at P>=0.95 under a best-of-5 trigger, public must build to mu_pub=528.48 (+23.61 winner's-curse tax over #202's frozen 504.87 = 7.28 evaporating public best-of-N discount + 16.33 private-drop gross-up); #202's freeze-robust 512.2/N=1 hedge does NOT survive privately (p=0.312). Budget DEFAULT: BUILD HIGHER, N=1 (the #206 hedge, now winner's-curse-justified). Does NOT change #204's 512.41/514.63 sigma->LCB trigger (a different quantity). REINFORCES the HOLD.
  - [LANDED-CONSUMED] lawine #196 compliant non-spec floor (lane-b, EMPIRICAL) -- under strict #192 the compliant non-spec int4 path is token-identical (1.0), PPL 2.3766<2.42, 128/128, but FLOORS at 165.44 official TPS (66.9% below 500). NO compliant non-spec 500-lane; spec premium 316.1 TPS / 191% is existential. Adds the #192 compliance precondition ABOVE the sigma->LCB trigger (does NOT change #204's 512.41/514.63).
  - [LANDED-CONSUMED] wirbel #199 compliant-spec E[T] ceiling (lane-a, BRACKET) -- the batch-invariant int4 VERIFY kernel ceiling 536.66 official TPS (lower-CI 525.73>500 -> CLEARS), floor 416.31 (misses); clears 500 ONLY if kernel overhead < 7.33% both / 4.12% descent (UNMEASURED; off-shelf #122 +51.78%, ~7x over). lane-a is the SINGLE compliant 500-route -> gated on the issue_192_human_ruling precondition (PENDING). REINFORCES the HOLD.
  - [LANDED-CONSUMED] wirbel #213 compliant-kernel lambda-budget curve max_kernel_overhead_pct(lambda) -- prices #199's lane-a feasibility as a function of the build: lambda=1 budget 7.33% both / 4.12% descent (<->#199); lambda_crit=0.834 both / 0.907 descent (below it even a FREE batch-invariant kernel misses 500); lambda_hat=0.342 budget -16.74% (negative -> infeasible); off-the-shelf #122 (+51.78%) clears at NO physical lambda<=1. So the strict-#192 compliant 500-path is a DOUBLE gate (lambda >= 0.834 AND kernel-under-budget). Does NOT change #204's 512.41/514.63 trigger (sits ABOVE the sigma math). REINFORCES the HOLD.
  - [PENDING-RULING] issue #192 human ruling (compliance gate, ABOVE the sigma math) -- decides whether the speculative int4 verify lane is admissible; keeps launch_authorized=False until resolved.
  - [PENDING-BUILD] land #71 measured tuple (THIS tuple).

**Launch gates (ALL required):** (1) land #71 builds the {h} kernel; (2) darwin _IncludedRouter
boot-fix folded; (3) PRECACHE_BENCH=1; (4) a human-approved `Approval request: HF job` issue;
(5) the issue #192 human ruling resolving in favor of an admissible compliant lane (lane-a
batch-invariant verify kernel, the only compliant 500-route per wirbel #199).
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

    # (i) combined-sigma row RESOLUTION (advisor 18:46Z, ubel #204): RETIRES #201's sigma->LCB trigger
    #     on a UNITS bug. #201's acceptance leg 11.17 was a z=1.96 two-sided HALF-WIDTH double-counted
    #     with the z=1.645 P95 LCB; the clean 1-sigma = 11.17/1.95996 = 5.699. Re-footed onto #194's
    #     clean convention -> combined LAUNCH sigma 7.545 central / 8.897 worst-case (was 12.215/13.796);
    #     P95 GO trigger mu>=512.41 central / 514.63 worst-case (was 520.09/522.69) vs the lambda=1
    #     ceiling 520.95 -> BOTH BELOW -> lambda=1 clears 500 at P95 CENTRALLY (+8.54) AND worst-case
    #     (+6.32). The verdict FLIPS PROVISIONAL knife-edge -> RESOLVED-YES. The de-dup x ICC MECHANISM
    #     is unchanged (anchors reproduce #195's de-dup central 7.2617 and #194's break-even 512.16 with
    #     err 0.0; ICC=0 corner still 7.2617). Direction came out SIGN-BACKWARDS (the dominant acceptance
    #     leg is DIVIDED down by z2 -> trigger DROPS 7.68 TPS, not up). Still NON-GATING: "it does not
    #     authorize a launch" -- the RESOLVED-YES does NOT enter `go`; the verdict is held on the three
    #     hard gates. #201's 520.09/522.69/12.215 are carried as superseded_201_trigger provenance.
    csc = led["combined_sigma_corner"]
    bb_full_g = d_full["_full_per_topology"]["both_bugs"]["lambda_gate"]
    do_full_g = d_full["_full_per_topology"]["descent_only"]["lambda_gate"]
    prov = csc["dedup_provenance_195"]
    sup201 = csc["superseded_201_trigger"]
    i_ok = (bool(csc["landed"]) and csc["source_pr"] == 201
            and csc["source_pr_trigger"] == 204
            and csc["supersedes_195_726_1704"] is True
            and csc["supersedes_201_trigger"] is True
            and csc["provisional"] is False and csc["resolved"] is True
            and csc["resolved_yes"] is True and csc["gates_analytic_go"] is False
            # #204 clean-1-sigma launch sigma REPLACES #201's 12.215/13.796:
            and abs(csc["combined_sigma_launch_central_tps"] - 7.545) <= 5e-2
            and abs(csc["combined_sigma_launch_worstcase_tps"] - 8.897) <= 5e-2
            # clean P95 GO trigger vs the lambda=1 ceiling (RETIRES 520.09/522.69):
            and abs(csc["z_p95"] - 1.6449) <= 5e-3
            and abs(csc["go_trigger_mu_central_tps"] - 512.410) <= 5e-2
            and abs(csc["go_trigger_mu_worstcase_tps"] - 514.635) <= 5e-2
            and abs(csc["lambda1_ceiling_mu_tps"] - 520.953) <= 5e-2
            and csc["central_p95_reachable"] is True
            and csc["worstcase_p95_reachable"] is True
            and abs(csc["central_margin_at_lambda1_tps"] - 8.543) <= 5e-2
            and abs(csc["worstcase_margin_at_lambda1_tps"] - 6.318) <= 5e-2
            # clean-rebase specifics: acceptance 1-sigma + signed direction (sign-backwards):
            and abs(csc["clean_acceptance_1sigma_tps"] - 5.699) <= 5e-2
            and abs(csc["delta_mu_rebase_central_tps"] - (-7.682)) <= 5e-2
            and csc["rebase_direction_matches_prediction"] is False
            and csc["does_lambda1_clear_500_at_p95_centrally"] == "YES"
            # de-dup x ICC mechanism (UNCHANGED -- the half-width footing #204 rebases):
            and abs(csc["acceptance_axis_dedup_iid_tps"] - 5.319) <= 5e-2
            and abs(csc["design_effect"] - 4.4106) <= 5e-3
            and abs(csc["sqrt_design_effect"] - 2.1001) <= 5e-3
            and abs(csc["acceptance_axis_realistic_icc_halfwidth_tps"] - 11.170) <= 5e-2
            and abs(csc["headroom_shift_tps"] - 7.935) <= 5e-2
            # ICC=0 corner reproduces #195's de-dup central 7.2617 (the superset proof, unchanged):
            and abs(csc["icc0_combined_sigma_central_tps"] - 7.2617) <= 5e-3
            # clean anchors reproduce #195 de-dup + #194 break-even EXACTLY (the rebase preserves them):
            and abs(csc["anchor_err_195_dedup"]) <= 1e-6
            and abs(csc["anchor_err_194_breakeven"]) <= 1e-6
            # #195 de-dup provenance retained (rho double-count that sets the acceptance IDENTITY):
            and abs(prov["rho_sampling_input"] - 0.9449) <= 5e-3
            and abs(prov["overlap_fraction"] - 0.893) <= 5e-3
            and prov["quadrature_valid"] is False
            and csc["colog_n_allocations"] == 385
            # #201's RETIRED trigger carried as provenance (the units-bug numbers #204 supersedes):
            and abs(sup201["mu_201_central_tps"] - 520.092) <= 5e-2
            and abs(sup201["mu_201_worstcase_tps"] - 522.692) <= 5e-2
            and abs(sup201["combined_201_central_tps"] - 12.215) <= 5e-2
            # lambda_gate surfaces the RESOLVED-YES/NON-GATING readout (both-bugs only):
            and bb_full_g["combined_sigma_corner"] is not None
            and bb_full_g["launch_sigma_central_p95_reachable"] is True
            and bb_full_g["launch_sigma_worstcase_p95_reachable"] is True
            and bb_full_g["launch_sigma_provisional"] is False
            and bb_full_g["launch_sigma_resolved"] is True
            and bb_full_g["launch_sigma_gates_go"] is False
            and do_full_g["combined_sigma_corner"] is None
            and do_full_g["launch_sigma_central_p95_reachable"] is None
            # THE NON-GATING INVARIANT: lambda=1 now clears 500 at P95 BOTH ends, yet the row STILL does
            # NOT authorize a launch -- both-bugs GO is held on the three hard gates, not hard-wired here:
            and d_full["per_topology"]["both_bugs"]["verdict"] == "GO"
            and d_full["verdict"] == "GO")
    results["i_combined_sigma_closure_204_resolved_yes"] = {
        "pass": bool(i_ok),
        "source_pr_mechanism": csc["source_pr"], "source_pr_trigger": csc["source_pr_trigger"],
        "supersedes_195": csc["supersedes_195_726_1704"],
        "supersedes_201_trigger": csc["supersedes_201_trigger"],
        "provisional": csc["provisional"], "resolved": csc["resolved"],
        "resolved_yes": csc["resolved_yes"], "gates_analytic_go": csc["gates_analytic_go"],
        "combined_sigma_launch_central_tps": csc["combined_sigma_launch_central_tps"],
        "combined_sigma_launch_worstcase_tps": csc["combined_sigma_launch_worstcase_tps"],
        "go_trigger_mu_central_tps": csc["go_trigger_mu_central_tps"],
        "go_trigger_mu_worstcase_tps": csc["go_trigger_mu_worstcase_tps"],
        "lambda1_ceiling_mu_tps": csc["lambda1_ceiling_mu_tps"],
        "central_p95_reachable": csc["central_p95_reachable"],
        "worstcase_p95_reachable": csc["worstcase_p95_reachable"],
        "central_margin_at_lambda1_tps": csc["central_margin_at_lambda1_tps"],
        "worstcase_margin_at_lambda1_tps": csc["worstcase_margin_at_lambda1_tps"],
        "clean_acceptance_1sigma_tps": csc["clean_acceptance_1sigma_tps"],
        "delta_mu_rebase_central_tps": csc["delta_mu_rebase_central_tps"],
        "rebase_direction_matches_prediction": csc["rebase_direction_matches_prediction"],
        "icc0_reproduces_195_dedup_726": abs(csc["icc0_combined_sigma_central_tps"] - 7.2617) <= 5e-3,
        "anchor_err_195_dedup": csc["anchor_err_195_dedup"],
        "anchor_err_194_breakeven": csc["anchor_err_194_breakeven"],
        "headroom_shift_tps": csc["headroom_shift_tps"],
        "colog_n_allocations": csc["colog_n_allocations"],
        "retired_201_trigger": sup201,
        "resolved_yes_but_nongating_verdict_held_go": (
            d_full["per_topology"]["both_bugs"]["verdict"] == "GO"
            and csc["resolved_yes"] is True
            and csc["gates_analytic_go"] is False)}

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

    # (k) FULL-LADDER GO requirement + depth-1-only FALSE-GO guard (denken #197) + the COST row now
    #     REPLACED by denken #205 (SPRT, advisor 19:09Z). THREE assertions: (1) the HELD-VERDICT
    #     INVARIANT -- the worked tuple carries the full 8-entry ladder, so the full-ladder guard is
    #     SATISFIED and both-bugs stays GO (#197 REINFORCES, does NOT flip); (2) the guard BITES -- a
    #     depth-1-only tuple (1-entry ladder) is BLOCKED to NO-GO with a full-ladder failing_gate, even
    #     though every OTHER gate (validity, trustworthy, build, both launch LCBs) WOULD pass; (3) the
    #     COST row is the REALISTIC SPRT expected-N (E[N]~405 NO-GO, ~14,915 near-bar, ~24,398 worst-case
    #     ASN; realized (alpha,power)=(0.05,0.95), boundaries +-2.9444, Deff=4.41, ~75.12x collapse) --
    #     NOT #197's truth-independent fixed-N, which is retained only as the 30,455 worst-case CAP. The
    #     full-ladder GUARD is UNCHANGED; #205 only re-prices the cost row. At beta=0.765 the mechanism
    #     CANNOT clear the private bar -> a real GO needs beta~1 across the MEASURED ladder.
    lpc = d_full["launch_ci_ledger"]["liveprobe_measurement_cost"]
    lp_row = next(r for r in d_full["launch_ci_ledger"]["numerical_axes"]["both_bugs"]
                  if r["axis"] == "liveprobe_measurement_cost")
    t_d1 = make_measured_tuple("self-test-depth1-falsego", _M.et_of_lambda(1.0, "descent_only"),
                               E_T_STAR_BOTH, 0.0, _M.shipped_step, [0.99], 2.39, 128)
    v_d1 = _topology_verdict("both_bugs", E_T_STAR_BOTH, 1.0, t_d1, _M.shipped_step, _M.tau_low,
                             {"ppl": 2.39, "boots": True, "completed": 128}, True)
    k_ok = (bool(lpc["landed"]) and lpc["reinforces_hold"] is True and lpc["flips_verdict"] is False
            and lpc["full_ladder_required"] is True
            and lpc["min_depths_for_decisive"] == "full-ladder" and lpc["min_depths_int"] == 9
            and lpc["depth1_plus_2_suffices"] is False
            and lpc["full_ladder_measured"] is True          # worked tuple carries q[2..9]
            and lpc["false_go_risk_depth1_only"] is True
            and lpc["mechanism_can_clear_private_bar"] is False
            and abs(lpc["depth1_overstatement_tps"] - 85.21434075500031) <= 5e-2
            and abs(lpc["true_private_lcb_at_lambda1_tps"] - 419.6445574528826) <= 5e-2
            and abs(lpc["private_lcb_perfect_depth1_tps"] - 419.6445574528826) <= 5e-2
            and abs(lpc["beta_primary"] - 0.765124365433998) <= 5e-3
            and abs(lpc["private_bar_both"] - 0.9780112973731208) <= 1e-3
            # COST row REPLACED by denken #205 SPRT (the realistic expected-N):
            and lpc["cost_row_source_pr"] == 205
            and lpc["sequential_via_205_pending"] is False     # #205 LANDED (no longer pending)
            and lpc["sequential_sprt_205_landed"] is True
            and abs(lpc["expected_n_sprt_nogo"] - 405.42403511311863) <= 1.0
            and abs(lpc["expected_n_sprt_nearbar"] - 14915.057585591705) <= 1.0
            and abs(lpc["worst_case_expected_n_sprt"] - 24398.04273973794) <= 1.0
            and abs(lpc["sprt_savings_vs_fixed_n"] - 75.11987975965602) <= 5e-2
            and abs(lpc["sprt_boundary_A_upper_decide_go"] - 2.9444389791664403) <= 5e-3
            and abs(lpc["sprt_boundary_B_lower_decide_nogo"] - (-2.9444389791664394)) <= 5e-3
            and abs(lpc["sprt_realized_alpha"] - 0.05) <= 5e-3
            and abs(lpc["sprt_realized_power"] - 0.95) <= 5e-3
            and abs(lpc["sprt_deff_190"] - 4.410614351127293) <= 5e-3
            # #197's fixed-N retained as the worst-case CAP (and still emitted for the cap row):
            and abs(lpc["decisive_total_trials_lambda1"] - 30455.404769372028) <= 1.0
            and abs(lpc["fixed_n_worst_case_cap_trials"] - 30455.404769372028) <= 1.0
            and abs(lpc["neyman_efficiency_gain_vs_equal"] - 1.4336929857369356) <= 5e-3
            and lp_row["status"] == "LANDED" and lp_row["flag"] == "consumed"
            and lp_row["cost_row_source_pr"] == 205
            # lambda_gate surfacing on the worked (full-ladder) both-bugs cell:
            and bb_g["full_ladder_ok"] is True and bb_g["full_ladder_measured"] is True
            and bb_g["full_ladder_gates_go"] is True
            and bb_g["liveprobe_sequential_sprt_205_landed"] is True
            and abs(bb_g["liveprobe_expected_n_sprt_nogo"] - 405.42403511311863) <= 1.0
            # HELD-VERDICT INVARIANT: guard present + satisfied -> verdict stays GO.
            and d_full["per_topology"]["both_bugs"]["verdict"] == "GO"
            and d_full["verdict"] == "GO"
            # the guard BITES on a depth-1-only read (otherwise-GO cell -> NO-GO):
            and v_d1["GO"] is False and v_d1["verdict"] == "NO-GO"
            and v_d1["lambda_gate"]["full_ladder_ok"] is False
            and v_d1["lambda_gate"]["full_ladder_measured"] is False
            and v_d1["lambda_gate"]["build_gate_pass"] is True          # build gate WOULD pass
            and v_d1["lambda_gate"]["realistic_launch_clears_500"] is True
            and v_d1["lambda_gate"]["private_launch_clears_500"] is True
            and v_d1["failing_gate"] is not None and "full-ladder" in v_d1["failing_gate"])
    results["k_full_ladder_guard_197_sprt_cost_205"] = {
        "pass": bool(k_ok),
        "cost_row_source_pr": lpc["cost_row_source_pr"],
        "full_ladder_required": lpc["full_ladder_required"],
        "full_ladder_measured_worked": lpc["full_ladder_measured"],
        "false_go_risk_depth1_only": lpc["false_go_risk_depth1_only"],
        "mechanism_can_clear_private_bar": lpc["mechanism_can_clear_private_bar"],
        "depth1_overstatement_tps": lpc["depth1_overstatement_tps"],
        "expected_n_sprt_nogo": lpc["expected_n_sprt_nogo"],
        "expected_n_sprt_nearbar": lpc["expected_n_sprt_nearbar"],
        "worst_case_expected_n_sprt": lpc["worst_case_expected_n_sprt"],
        "sprt_savings_vs_fixed_n": lpc["sprt_savings_vs_fixed_n"],
        "sprt_boundary_A_upper_decide_go": lpc["sprt_boundary_A_upper_decide_go"],
        "sprt_realized_alpha": lpc["sprt_realized_alpha"],
        "sprt_realized_power": lpc["sprt_realized_power"],
        "sprt_deff_190": lpc["sprt_deff_190"],
        "fixed_n_worst_case_cap_trials": lpc["fixed_n_worst_case_cap_trials"],
        "decisive_total_trials_lambda1": lpc["decisive_total_trials_lambda1"],
        "beta_primary": lpc["beta_primary"],
        "worked_verdict_held_go": d_full["per_topology"]["both_bugs"]["verdict"] == "GO",
        "depth1_guard_bites_nogo": v_d1["verdict"] == "NO-GO",
        "depth1_failing_gate": v_d1["failing_gate"]}

    # (l) multi-shot budget under the conservative FROZEN regime (advisor 18:39Z, kanna #202): the
    #     budget DEFAULTS to FROZEN (best-of-N beats only sigma_hw -> N=5@bar P=0.810, NOT fresh 0.969);
    #     the default build-bar input is mu_bar_frozen_p95=504.87 (not fresh 499.08). THE HEDGE: build-
    #     to-512.2 / N=1 is fully freeze-robust (n_shots_frozen=1) -> the SAFE recommendation is
    #     untouched; only build-at-bar+best-of-N is frozen-fragile (E[shots]=2.34, exhausts 19%,
    #     f*=0.846). ANNOTATION only -- single-shot GO/NO-GO UNCHANGED (both GO, descent NO-GO, GO).
    fba = d_full["launch_ci_ledger"]["frozen_budget_annotation"]
    hedge = fba["freeze_robust_hedge"]
    fragile = fba["build_at_bar_best_of_n_fragile"]
    l_ok = (bool(fba["landed"]) and fba["reinforces_hold"] is True
            and fba["regime_default"] == "FROZEN" and fba["regime_is_open"] is True
            and abs(fba["mu_bar_frozen_p95_tps"] - 504.87342465668917) <= 5e-2
            and abs(fba["mu_bar_fresh_p95_n5_tps"] - 499.08467835746706) <= 5e-2
            and abs(fba["p_bar_n5_frozen"] - 0.8097690471233381) <= 5e-3
            and abs(fba["delta_mu_frozen_tps"] - (-7.283646514920861)) <= 5e-2
            and abs(fba["sigma_fraction_beatable_frozen"] - 0.6581633898900652) <= 5e-3
            and hedge["n_shots_frozen"] == 1 and hedge["regime_invariant"] is True
            and abs(hedge["build_to_mu_tps"] - 512.15707117161) <= 5e-2
            and abs(fragile["e_shots_frozen"] - 2.3367894101342586) <= 5e-2
            and abs(fragile["e_shots_fresh"] - 1.9375) <= 5e-3
            and abs(fragile["exhaust_without_clear_frac_frozen"] - 0.19023095287666192) <= 5e-3
            and abs(fragile["frozen_fraction_breakeven"] - 0.8455321793444455) <= 5e-3
            and fba["frozen_cost_crossover_206_pending"] is True
            # redraw_budget row carries the #202 annotation:
            and bool(redraw_row["frozen_budget_202_landed"])
            and abs(redraw_row["mu_bar_frozen_p95_tps"] - 504.87342465668917) <= 5e-2
            and redraw_row["n_shots_frozen_at_512"] == 1
            # single-shot verdict UNCHANGED by the annotation:
            and d_full["per_topology"]["both_bugs"]["verdict"] == "GO"
            and d_full["per_topology"]["descent_only"]["verdict"] == "NO-GO"
            and d_full["verdict"] == "GO")
    results["l_frozen_budget_annotation_202"] = {
        "pass": bool(l_ok),
        "regime_default": fba["regime_default"],
        "mu_bar_frozen_p95_tps": fba["mu_bar_frozen_p95_tps"],
        "mu_bar_fresh_p95_n5_tps": fba["mu_bar_fresh_p95_n5_tps"],
        "p_bar_n5_frozen": fba["p_bar_n5_frozen"],
        "freeze_robust_build_to_mu_tps": hedge["build_to_mu_tps"],
        "n_shots_frozen": hedge["n_shots_frozen"],
        "frozen_fragile_e_shots": fragile["e_shots_frozen"],
        "frozen_fraction_breakeven": fragile["frozen_fraction_breakeven"],
        "verdict_unchanged_go": d_full["verdict"] == "GO"}

    # (m) ISSUE #192 compliance bracket (lawine #196 lane-b + wirbel #199 lane-a, advisor 18:58Z/
    #     19:09Z): lane-b non-spec FLOORS at 165.44 (66.9% below 500, token-identical) -> NO compliant
    #     non-spec 500-lane; lane-a compliant-spec ceiling 536.66 (lower-CI 525.73>500) is the ONLY
    #     compliant 500-route, conditioned on an UNMEASURED <7.33%/4.12% kernel overhead. This sits
    #     ABOVE the sigma->LCB trigger (does NOT change #204's 512.41/514.63) and adds a NEW hard
    #     precondition (issue_192_human_ruling, PENDING) -> launch_authorized stays False.
    clb = d_full["launch_ci_ledger"]["compliant_lane_bracket"]
    lb = clb["lane_b_nonspec"]
    lna = clb["lane_a_compliant_spec"]
    p192 = next((r for r in d_full["launch_ci_ledger"]["preconditions"]
                 if r["row"] == "issue_192_human_ruling"), None)
    m_ok = (bool(clb["landed"]) and clb["reinforces_hold"] is True
            # CRUX: sits ABOVE the sigma math; does NOT gate or change the #204 trigger.
            and clb["gates_sigma_lcb_trigger"] is False
            and clb["sits_above_sigma_math"] is True
            and clb["does_not_change_204_trigger"] is True
            # lane-b non-spec floor (lawine #196):
            and abs(lb["official_tps_floor"] - 165.43791973106974) <= 5e-2
            and abs(lb["token_identity_rate"] - 1.0) <= 1e-9
            and abs(lb["ppl"] - 2.376564864651504) <= 5e-3
            and lb["completes_128"] is True and lb["clears_500"] is False
            and abs(lb["margin_to_500_pct"] - (-66.91241605378605)) <= 5e-2
            and abs(lb["spec_premium_tps"] - 316.0920802689302) <= 5e-2
            and lb["verdict_label"] == "STRUCTURAL_GAP_SPEC_EXISTENTIAL"
            and bool(lb["self_test_passes"])
            # lane-a compliant-spec ceiling (wirbel #199):
            and abs(lna["tps_ceiling"] - 536.6590426143789) <= 5e-2
            and abs(lna["tps_floor"] - 416.307156176311) <= 5e-2
            and abs(lna["ceiling_ci_lower_tps"] - 525.7290377676009) <= 5e-2
            and lna["ceiling_lower_clears_500"] is True
            and lna["ceiling_clears_500"] is True and lna["floor_clears_500"] is False
            and abs(lna["max_kernel_overhead_pct_both_bugs"] - 7.331808522875782) <= 5e-3
            and abs(lna["max_kernel_overhead_pct_descent"] - 4.123450699935671) <= 5e-3
            and abs(lna["offshelf_overhead_ref_122"] - 0.5178) <= 5e-4
            and lna["overhead_is_measured"] is False
            and bool(lna["self_test_passes"])
            # the NET compliance-precondition framing:
            and clb["compliant_500_lane_exists"] is True
            and clb["compliant_500_lane_is_spec_only"] is True
            and clb["compliant_500_conditioned_on_unmeasured_overhead"] is True
            # NEW hard precondition row, PENDING -> launch stays un-authorized:
            and p192 is not None and p192["status"] == "PENDING"
            and p192["pr"] == 192 and p192["flag"] == "human-ruling"
            and d_full["launch_authorized"]["authorized"] is False
            # the sigma->LCB trigger is UNCHANGED by this row (#204 still 512.41/514.63):
            and abs(d_full["launch_ci_ledger"]["combined_sigma_corner"]["go_trigger_mu_central_tps"]
                    - 512.4101095400661) <= 5e-2)
    results["m_compliant_lane_bracket_192_196_199"] = {
        "pass": bool(m_ok),
        "sits_above_sigma_math": clb["sits_above_sigma_math"],
        "does_not_change_204_trigger": clb["does_not_change_204_trigger"],
        "lane_b_nonspec_floor_tps": lb["official_tps_floor"],
        "lane_b_clears_500": lb["clears_500"],
        "lane_a_ceiling_tps": lna["tps_ceiling"],
        "lane_a_ceiling_ci_lower_tps": lna["ceiling_ci_lower_tps"],
        "lane_a_lower_clears_500": lna["ceiling_lower_clears_500"],
        "lane_a_max_overhead_both_pct": lna["max_kernel_overhead_pct_both_bugs"],
        "lane_a_overhead_unmeasured": (not lna["overhead_is_measured"]),
        "issue_192_precondition_pending": (p192 is not None and p192["status"] == "PENDING"),
        "launch_authorized": d_full["launch_authorized"]["authorized"]}

    # (n) ubel #207 (MERGED 19:40Z): the #175 two-readings tension RESOLVED in FAVOR of the YES. The
    #     larger 10.906 reading is the B=16384 128-tok-window SUB-bench, NOT launch-correct (the 2.106
    #     ratio vs h_out 5.178 = bench-sqrtN x op-point, only COINCIDENTALLY ~= sqrt(D) 2.100). The
    #     launch-correct trigger 512.41/514.63 STANDS -> #204's robust-YES SURVIVES; issue_207 moves
    #     from open_caveats to resolved_caveats; the go_trigger_mu_central is UNCHANGED (512.41).
    csc = d_full["launch_ci_ledger"]["combined_sigma_corner"]
    sr = csc["sigma_reconcile_207"]
    n_ok = (bool(sr["landed"]) and sr["robust_yes_survives"] is True
            and sr["lambda1_clears_under_conservative_reading"] is False
            and sr["ratio_equals_sqrtD"] is False
            and sr["conservative_reading_is_launch_correct"] is False
            and abs(sr["trigger_central_launch_correct_tps"] - 512.4101095400661) <= tol
            and abs(sr["trigger_worstcase_launch_correct_tps"] - 514.6346173476741) <= tol
            and abs(sr["ratio_175_readings"] - 2.106153062206173) <= 5e-3
            and abs(sr["delta_trigger_reading_tps"] - 8.570958365039814) <= 5e-2
            # issue_207 RESOLVED (now in resolved_caveats), and NO LONGER open; only land #71 remains open:
            and "issue_207_175_two_readings" in csc["resolved_caveats"]
            and "issue_207_175_two_readings" not in csc["open_caveats"]
            and "land_71_colog" in csc["open_caveats"]
            and bool(_BANKED.sigma_reconcile_self_test_passes())
            # CRUX: the sigma->LCB trigger is UNCHANGED by #207 (still 512.41):
            and abs(csc["go_trigger_mu_central_tps"] - 512.4101095400661) <= tol)
    results["n_sigma_reconcile_207"] = {
        "pass": bool(n_ok),
        "robust_yes_survives": sr["robust_yes_survives"],
        "lambda1_clears_under_conservative_reading": sr["lambda1_clears_under_conservative_reading"],
        "ratio_equals_sqrtD": sr["ratio_equals_sqrtD"],
        "ratio_175_readings": sr["ratio_175_readings"],
        "trigger_central_launch_correct_tps": sr["trigger_central_launch_correct_tps"],
        "trigger_worstcase_launch_correct_tps": sr["trigger_worstcase_launch_correct_tps"],
        "issue_207_resolved": ("issue_207_175_two_readings" in csc["resolved_caveats"]),
        "issue_207_no_longer_open": ("issue_207_175_two_readings" not in csc["open_caveats"]),
        "go_trigger_mu_central_unchanged": abs(csc["go_trigger_mu_central_tps"] - 512.4101095400661) <= tol}

    # (o) kanna #210 (MERGED 19:40Z): best-of-N does NOT relax the binding PRIVATE bar -- the conditional
    #     private clear is FLAT in N (n_star_private=1; selection is on non-replicating public noise, the
    #     private grade is one fresh draw). To clear 500-PRIVATE at P>=0.95 under best-of-5 the PUBLIC build
    #     must reach mu_bar_private_corrected=528.48 (+23.61 winner's-curse tax over #202's 504.87). The
    #     #202 freeze-robust mu=512.2/N=1 does NOT survive privately (p=0.3120). BUILD-target row only --
    #     it does NOT touch the sigma->LCB PUBLIC trigger (512.41 STANDS).
    wca = d_full["launch_ci_ledger"]["winners_curse_annotation"]
    o_ok = (bool(wca["landed"]) and wca["reinforces_hold"] is True
            and abs(wca["mu_bar_private_corrected_tps"] - 528.4835555959944) <= tol
            and abs(wca["delta_mu_winners_curse_tps"] - 23.610130939305236) <= 5e-2
            and wca["n_star_private"] == 1
            and wca["private_clear_flat_in_n"] is True
            and wca["regime_invariant"] is True
            and wca["freeze_robust_512_survives_private"] is False
            and abs(wca["p_private_clear_at_mu512p2_n1"] - 0.31197802246730244) <= 1e-3
            and abs(wca["mu_bar_frozen_public_202_tps"] - 504.87342465668917) <= tol
            and abs(wca["tax_decomposition_tps"]["sum"] - 23.610130939305236) <= 5e-2
            and bool(_BANKED.winners_curse_self_test_passes())
            # CRUX: the BUILD target does NOT change the sigma->LCB PUBLIC trigger (still 512.41):
            and wca["does_not_change_sigma_lcb_trigger"] is True
            and abs(csc["go_trigger_mu_central_tps"] - 512.4101095400661) <= tol)
    results["o_winners_curse_210"] = {
        "pass": bool(o_ok),
        "mu_bar_private_corrected_tps": wca["mu_bar_private_corrected_tps"],
        "delta_mu_winners_curse_tps": wca["delta_mu_winners_curse_tps"],
        "n_star_private": wca["n_star_private"],
        "private_clear_flat_in_n": wca["private_clear_flat_in_n"],
        "freeze_robust_512_survives_private": wca["freeze_robust_512_survives_private"],
        "p_private_clear_at_mu512p2_n1": wca["p_private_clear_at_mu512p2_n1"],
        "does_not_change_sigma_lcb_trigger": wca["does_not_change_sigma_lcb_trigger"]}

    # (p) wirbel #213 (MERGED 19:40Z): the compliant-spec verify-kernel overhead budget vs lambda. lane-a
    #     is a DOUBLE gate -- self-KV-recovery lambda >= lambda_crit AND the batch-invariant kernel under
    #     max_kernel_overhead_pct(lambda). lambda_crit=0.8345 both / 0.9067 descent; budget@lambda=1 is
    #     7.33% both / 4.12% descent; off-shelf #122 (+51.78%) clears at NO physical lambda<=1. This
    #     EXTENDS the #192 compliance bracket; it does NOT change the #204/#207 sigma->LCB trigger.
    clb = d_full["launch_ci_ledger"]["compliant_lane_bracket"]
    kb213 = clb["lane_a_kernel_budget_213"]
    p_ok = (bool(kb213["landed"])
            and abs(kb213["lambda_crit_clears_500_both_bugs"] - 0.8344533978886615) <= 5e-3
            and abs(kb213["lambda_crit_clears_500_descent"] - 0.9066754940814947) <= 5e-3
            and abs(kb213["overhead_budget_pct_at_lambda1_both_bugs"] - 7.331808522875782) <= 5e-3
            and abs(kb213["overhead_budget_pct_at_lambda1_descent"] - 4.123450699935671) <= 5e-3
            and abs(kb213["overhead_budget_pct_at_lambda_hat_both_bugs"] - (-16.738568764737806)) <= 5e-2
            and kb213["offshelf_122_clears_at_physical_lambda"] is False
            and bool(kb213["self_test_passes"])
            # the NET double-gate framing on the bracket:
            and clb["compliant_500_lane_is_double_gate"] is True
            and abs(clb["lane_a_lambda_crit_both_bugs"] - 0.8344533978886615) <= 5e-3
            and abs(clb["lane_a_lambda_crit_descent"] - 0.9066754940814947) <= 5e-3
            # CRUX: still sits ABOVE the sigma math -> the #204/#207 trigger is UNCHANGED:
            and clb["does_not_change_204_trigger"] is True
            and abs(csc["go_trigger_mu_central_tps"] - 512.4101095400661) <= tol)
    results["p_kernel_budget_213"] = {
        "pass": bool(p_ok),
        "lambda_crit_both_bugs": kb213["lambda_crit_clears_500_both_bugs"],
        "lambda_crit_descent": kb213["lambda_crit_clears_500_descent"],
        "overhead_budget_pct_at_lambda1_both_bugs": kb213["overhead_budget_pct_at_lambda1_both_bugs"],
        "overhead_budget_pct_at_lambda1_descent": kb213["overhead_budget_pct_at_lambda1_descent"],
        "offshelf_122_clears_at_physical_lambda": kb213["offshelf_122_clears_at_physical_lambda"],
        "compliant_500_lane_is_double_gate": clb["compliant_500_lane_is_double_gate"]}

    # (q) denken #212 (MERGED 19:54Z): the AR(1)-corrected SPRT cost band. Folding #190's DECAYING ACF
    #     (rho(1)=0.2583) into the SPRT partial-sum variance tightens #205's flat xDeff=4.41 by 1.59-2.66x
    #     -> E[N]_nogo band [405 IID -> 672 AR(1) -> 1,125 measured-ACF-realistic -> 1,788 flat-loose]; the
    #     data-grounded realistic point is 1,125. The 75.12x collapse is Deff-INVARIANT; realized
    #     (alpha,power)=(0.05,0.95) + bar 0.9780 are UNCHANGED -- the band only SHARPENS, never flips.
    lpc = d_full["launch_ci_ledger"]["liveprobe_measurement_cost"]
    ar = lpc["ar_corrected_cost_band_212"]
    band = ar["expected_n_nogo_band"]
    q_ok = (bool(ar["landed"]) and isinstance(band, list) and len(band) == 4
            and abs(ar["expected_n_nogo_iid_floor"] - 405.42403511311863) <= 1.0
            and abs(ar["expected_n_nogo_ar_optimistic"] - 672.3420962564048) <= 1.0
            and abs(ar["expected_n_nogo_realistic_measured_acf"] - 1124.7628546877863) <= 1.0
            and abs(ar["expected_n_nogo_flat_loose"] - 1788.1690675618568) <= 1.0
            and ar["flat_441_is_conservative"] is True
            and ar["savings_ratio_deff_invariant"] is True
            and abs(ar["savings_ratio_unchanged"] - 75.11987975965602) <= 5e-2
            and abs(ar["realized_alpha_unchanged"] - 0.04999999999999993) <= 1e-3
            and abs(ar["realized_power_unchanged"] - 0.95) <= 1e-3
            and abs(ar["private_bar_both_unchanged"] - 0.9780112973731208) <= 1e-3
            and abs(ar["rho_lag1"] - 0.2583178258286258) <= 5e-3
            and bool(ar["self_test_passes"])
            # the headline realistic measurement-cost the human reads (1,125, data-grounded):
            and abs(lpc["expected_n_nogo_realistic_measured_acf"] - 1124.7628546877863) <= 1.0)
    results["q_sprt_ar_212"] = {
        "pass": bool(q_ok),
        "expected_n_nogo_band": band,
        "expected_n_nogo_realistic_measured_acf": ar["expected_n_nogo_realistic_measured_acf"],
        "flat_441_is_conservative": ar["flat_441_is_conservative"],
        "savings_ratio_deff_invariant": ar["savings_ratio_deff_invariant"],
        "savings_ratio_unchanged": ar["savings_ratio_unchanged"],
        "realized_alpha_unchanged": ar["realized_alpha_unchanged"],
        "realized_power_unchanged": ar["realized_power_unchanged"],
        "private_bar_both_unchanged": ar["private_bar_both_unchanged"]}

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
                             # combined-sigma row -- ubel #204 CLEAN-1-sigma trigger (RESOLVED-YES,
                             # NON-GATING; RETIRES #201's trigger). The de-dup x ICC mechanism is solid;
                             # #201's 12.215/520.09 was a z=1.96 half-width units bug -> rebased to 5.699.
                             "combined_sigma_launch_central_tps_204": _BANKED.rebase_combined_sigma("central"),
                             "combined_sigma_launch_worstcase_tps_204": _BANKED.rebase_combined_sigma("worstcase"),
                             "go_trigger_mu_central_tps_204": _BANKED.rebase_mu_clears_500("central"),
                             "go_trigger_mu_worstcase_tps_204": _BANKED.rebase_mu_clears_500("worstcase"),
                             "lambda1_ceiling_mu_tps_204": _BANKED.rebase_lambda1_ceiling(),
                             "central_p95_reachable_204": _BANKED.rebase_lambda1_clears_500("central"),
                             "worstcase_p95_reachable_204": _BANKED.rebase_lambda1_clears_500("worstcase"),
                             "central_margin_at_lambda1_tps_204": _BANKED.rebase_headroom_below_ceiling("central"),
                             "worstcase_margin_at_lambda1_tps_204": _BANKED.rebase_headroom_below_ceiling("worstcase"),
                             "clean_acceptance_1sigma_tps_204": _BANKED.rebase_acceptance_1sigma_clean(),
                             "delta_mu_rebase_central_tps_204": _BANKED.rebase_delta_mu("central"),
                             "rebase_direction_matches_prediction_204": _BANKED.rebase_direction_matches_prediction(),
                             "anchor_err_195_dedup_204": _BANKED.rebase_anchor_err_195(),
                             "anchor_err_194_breakeven_204": _BANKED.rebase_anchor_err_194(),
                             # #201 RETIRED trigger provenance (the units-bug numbers #204 supersedes):
                             "retired_combined_sigma_launch_central_tps_201": _BANKED.combined_sigma_launch("central"),
                             "retired_combined_sigma_launch_worstcase_tps_201": _BANKED.combined_sigma_launch("worstcase"),
                             "retired_go_trigger_mu_central_tps_201": _BANKED.mu_clears_500("central"),
                             "retired_go_trigger_mu_worstcase_tps_201": _BANKED.mu_clears_500("worstcase"),
                             "lambda1_ceiling_mu_tps_201": _BANKED.lambda1_ceiling_mu(),
                             "acceptance_axis_dedup_iid_tps_201": _BANKED.acceptance_sigma_dedup_iid(),
                             "acceptance_axis_realistic_icc_halfwidth_tps_201": _BANKED.acceptance_sigma_dedup_realistic_icc(),
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
                             "c_star_sequential_per_b_200": _BANKED.cost_crossover_sequential_per_b(),
                             # denken #197 liveprobe depth-budget: full-ladder GO requirement + depth-1
                             # FALSE-GO guard + decisive certification cost (GATES the GO leg).
                             "liveprobe_landed_197": _BANKED.liveprobe_landed(),
                             "liveprobe_false_go_risk_depth1_197": _BANKED.liveprobe_false_go_risk_depth1(),
                             "liveprobe_depth1_overstatement_tps_197": _BANKED.liveprobe_depth1_overstatement(),
                             "liveprobe_mech_can_clear_private_197": _BANKED.liveprobe_mech_can_clear_private(),
                             "liveprobe_private_lcb_perfect_depth1_tps_197": _BANKED.liveprobe_private_lcb_perfect_depth1(),
                             "liveprobe_beta_primary_197": _BANKED.liveprobe_beta_primary(),
                             "liveprobe_min_depths_int_197": _BANKED.liveprobe_min_depths_int(),
                             "liveprobe_decisive_total_trials_lambda1_197": _BANKED.liveprobe_decisive_total_trials(),
                             "liveprobe_neyman_efficiency_gain_197": _BANKED.liveprobe_neyman_efficiency_gain(),
                             "liveprobe_private_bar_both_197": _BANKED.liveprobe_private_bar(),
                             # denken #205 SPRT liveprobe-budget: REPLACES #197's fixed-N cost row with
                             # the realistic expected-N / operating-characteristic (cost-row swap only).
                             "sprt_landed_205": _BANKED.sprt_landed(),
                             "sprt_expected_n_nogo_205": _BANKED.sprt_expected_n_nogo(),
                             "sprt_expected_n_nogo_realistic_icc_205": _BANKED.sprt_expected_n_nogo_realistic_icc(),
                             "sprt_expected_n_nearbar_205": _BANKED.sprt_expected_n_nearbar(),
                             "sprt_worst_case_expected_n_205": _BANKED.sprt_worst_case_expected_n(),
                             "sprt_worst_case_expected_n_realistic_icc_205": _BANKED.sprt_worst_case_expected_n_realistic_icc(),
                             "sprt_savings_vs_fixed_n_205": _BANKED.sprt_savings_vs_fixed_n(),
                             "sprt_n_fixed_197_cap_205": _BANKED.sprt_n_fixed_197(),
                             "sprt_boundary_A_upper_decide_go_205": _BANKED.sprt_boundary("A_upper_decide_go"),
                             "sprt_boundary_B_lower_decide_nogo_205": _BANKED.sprt_boundary("B_lower_decide_nogo"),
                             "sprt_realized_alpha_205": _BANKED.sprt_realized_alpha(),
                             "sprt_realized_power_205": _BANKED.sprt_realized_power(),
                             "sprt_target_alpha_205": _BANKED.sprt_target_alpha(),
                             "sprt_target_power_205": _BANKED.sprt_target_power(),
                             "sprt_deff_190_205": _BANKED.sprt_deff_190(),
                             "sprt_beta_nogo_205": _BANKED.sprt_beta_nogo(),
                             "sprt_private_lcb_nogo_tps_205": _BANKED.sprt_private_lcb_nogo(),
                             # kanna #202 frozen-budget regime (conservative DEFAULT; annotation only).
                             "frozen_budget_landed_202": _BANKED.frozen_budget_landed(),
                             "mu_bar_frozen_p95_tps_202": _BANKED.mu_bar_frozen_p95(),
                             "mu_bar_fresh_p95_n5_tps_202": _BANKED.mu_bar_fresh_p95_n5(),
                             "p_bar_n5_frozen_202": _BANKED.p_bar_n5_frozen(),
                             "delta_mu_frozen_tps_202": _BANKED.frozen_delta_mu(),
                             "freeze_robust_build_to_mu_tps_202": _BANKED.frozen_mu_safe_tps(),
                             "n_shots_frozen_at_512_202": _BANKED.frozen_n_shots_at_512(),
                             "frozen_e_shots_at_bar_202": _BANKED.frozen_e_shots_at_bar(),
                             "frozen_fraction_breakeven_202": _BANKED.frozen_fraction_breakeven(),
                             "sigma_fraction_beatable_frozen_202": _BANKED.frozen_sigma_fraction_beatable(),
                             # ISSUE #192 compliance bracket -- lawine #196 lane-b (non-spec floor) +
                             # wirbel #199 lane-a (compliant-spec ceiling). Sits ABOVE the sigma->LCB
                             # trigger (does NOT change #204's 512.41/514.63); adds the #192 precondition.
                             "compliant_nonspec_floor_landed_196": _BANKED.nonspec_floor_landed(),
                             "compliant_nonspec_floor_tps_196": _BANKED.nonspec_official_tps(),
                             "compliant_nonspec_token_identity_196": _BANKED.nonspec_token_identity(),
                             "compliant_nonspec_ppl_196": _BANKED.nonspec_ppl(),
                             "compliant_nonspec_clears_500_196": _BANKED.nonspec_clears_500(),
                             "compliant_nonspec_margin_to_500_pct_196": _BANKED.nonspec_margin_pct(),
                             "compliant_nonspec_spec_premium_tps_196": _BANKED.nonspec_spec_premium_tps(),
                             "compliant_nonspec_spec_premium_pct_196": _BANKED.nonspec_spec_premium_pct(),
                             "compliant_spec_landed_199": _BANKED.compliant_spec_landed(),
                             "compliant_spec_tps_ceiling_199": _BANKED.compliant_spec_ceiling(),
                             "compliant_spec_tps_floor_199": _BANKED.compliant_spec_floor(),
                             "compliant_spec_ceiling_ci_lower_both_199": _BANKED.compliant_spec_ceiling_ci_lower("both_bugs"),
                             "compliant_spec_lower_clears_500_both_199": _BANKED.compliant_spec_lower_clears_500("both_bugs"),
                             "compliant_spec_clears_500_199": _BANKED.compliant_spec_clears_500(),
                             "compliant_spec_floor_clears_500_199": _BANKED.compliant_spec_floor_clears_500(),
                             "compliant_spec_max_overhead_both_pct_199": _BANKED.compliant_spec_max_overhead("both_bugs"),
                             "compliant_spec_max_overhead_descent_pct_199": _BANKED.compliant_spec_max_overhead("descent_only"),
                             "compliant_spec_offshelf_overhead_ref_122_199": _BANKED.compliant_spec_offshelf_overhead_ref(),
                             # ubel #207 launch-sigma #175-reading reconcile -- RESOLVES the #204 caveat in
                             # FAVOR of the YES (the 10.906 reading is the B=16384 sub-bench, RETIRED). The
                             # launch-correct trigger 512.41/514.63 STANDS; robust-YES SURVIVES.
                             "sigma_reconcile_landed_207": _BANKED.sigma_reconcile_landed(),
                             "sigma_reconcile_self_test_passes_207": _BANKED.sigma_reconcile_self_test_passes(),
                             "sigma_reconcile_robust_yes_survives_207": _BANKED.sigma_reconcile_robust_yes_survives(),
                             "sigma_reconcile_lambda1_clears_conservative_207": _BANKED.sigma_reconcile_lambda1_clears_conservative(),
                             "sigma_reconcile_conservative_is_launch_correct_207": _BANKED.sigma_reconcile_conservative_is_launch_correct(),
                             "sigma_reconcile_ratio_175_207": _BANKED.sigma_reconcile_ratio_175(),
                             "sigma_reconcile_ratio_equals_sqrtd_207": _BANKED.sigma_reconcile_ratio_equals_sqrtd(),
                             "sigma_reconcile_trigger_central_hout_tps_207": _BANKED.sigma_reconcile_trigger_hout("central"),
                             "sigma_reconcile_trigger_worstcase_hout_tps_207": _BANKED.sigma_reconcile_trigger_hout("worstcase"),
                             "sigma_reconcile_trigger_central_175sampling_tps_207": _BANKED.sigma_reconcile_trigger_175sampling("central"),
                             "sigma_reconcile_delta_trigger_tps_207": _BANKED.sigma_reconcile_delta_trigger(),
                             "sigma_reconcile_lambda1_ceiling_tps_207": _BANKED.sigma_reconcile_lambda1_ceiling(),
                             # kanna #210 winner's-curse: best-of-N does NOT relax the PRIVATE bar (flat in N,
                             # n_star_private=1); the build target rises to 528.48 (+23.61 tax). N=1 STANDS;
                             # does NOT touch the sigma->LCB PUBLIC trigger (512.41).
                             "winners_curse_landed_210": _BANKED.winners_curse_landed(),
                             "winners_curse_self_test_passes_210": _BANKED.winners_curse_self_test_passes(),
                             "winners_curse_mu_bar_private_corrected_tps_210": _BANKED.winners_curse_mu_bar_private_corrected(),
                             "winners_curse_delta_mu_tps_210": _BANKED.winners_curse_delta_mu(),
                             "winners_curse_n_star_private_210": _BANKED.winners_curse_n_star_private(),
                             "winners_curse_private_clear_flat_in_n_210": _BANKED.winners_curse_private_clear_flat_in_n(),
                             "winners_curse_regime_invariant_210": _BANKED.winners_curse_regime_invariant(),
                             "winners_curse_freeze_robust_512_survives_private_210": _BANKED.winners_curse_freeze_robust_512_survives(),
                             "winners_curse_p_private_at_512_210": _BANKED.winners_curse_p_private_at_512(),
                             "winners_curse_mu_bar_frozen_public_202_tps_210": _BANKED.winners_curse_mu_bar_frozen_202(),
                             "winners_curse_mu_safe_fresh_194_tps_210": _BANKED.winners_curse_mu_safe_fresh_194(),
                             "winners_curse_tps_n5_frozen_210": _BANKED.winners_curse_tps_n5("frozen"),
                             "winners_curse_tps_n5_fresh_210": _BANKED.winners_curse_tps_n5("fresh"),
                             # wirbel #213 compliant-kernel overhead budget vs lambda -- lane-a DOUBLE gate
                             # (lambda >= lambda_crit AND kernel under max_overhead(lambda)). Extends the #192
                             # bracket; off-shelf #122 (+51.78%) clears at NO physical lambda<=1.
                             "kernel_budget_landed_213": _BANKED.kernel_budget_landed(),
                             "kernel_budget_self_test_passes_213": _BANKED.kernel_budget_self_test_passes(),
                             "kernel_budget_lambda_crit_both_bugs_213": _BANKED.kernel_budget_lambda_crit("both_bugs"),
                             "kernel_budget_lambda_crit_descent_213": _BANKED.kernel_budget_lambda_crit("descent"),
                             "kernel_budget_overhead_at_lambda1_both_pct_213": _BANKED.kernel_budget_overhead_at_lambda1("both_bugs"),
                             "kernel_budget_overhead_at_lambda1_descent_pct_213": _BANKED.kernel_budget_overhead_at_lambda1("descent"),
                             "kernel_budget_overhead_at_lambda_hat_both_pct_213": _BANKED.kernel_budget_overhead_at_lambda_hat(),
                             "kernel_budget_lambda_hat_213": _BANKED.kernel_budget_lambda_hat(),
                             "kernel_budget_offshelf_clears_at_physical_lambda_213": _BANKED.kernel_budget_offshelf_clears_at_physical_lambda(),
                             "kernel_budget_offshelf_overhead_ref_122_213": _BANKED.kernel_budget_offshelf_overhead_ref(),
                             "kernel_budget_max_at_saturation_pct_213": _BANKED.kernel_budget_max_at_saturation(),
                             # denken #212 AR(1)-corrected SPRT cost band -- sharpens #205's flat xDeff=4.41
                             # by 1.59-2.66x. E[N]_nogo band [405 IID -> 672 AR(1) -> 1,125 measured-ACF-
                             # realistic -> 1,788 flat-loose]; the 75.12x collapse is Deff-INVARIANT.
                             "sprt_ar_landed_212": _BANKED.sprt_ar_landed(),
                             "sprt_ar_self_test_passes_212": _BANKED.sprt_ar_self_test_passes(),
                             "sprt_ar_expected_n_nogo_iid_212": _BANKED.sprt_ar_expected_n_nogo("iid"),
                             "sprt_ar_expected_n_nogo_ar1_212": _BANKED.sprt_ar_expected_n_nogo("ar1"),
                             "sprt_ar_expected_n_nogo_realistic_measured_acf_212": _BANKED.sprt_ar_realistic_nogo(),
                             "sprt_ar_expected_n_nogo_flat_441_212": _BANKED.sprt_ar_expected_n_nogo("flat_441"),
                             "sprt_ar_deff_ar_212": _BANKED.sprt_ar_deff("deff_ar_at_mbar"),
                             "sprt_ar_deff_empirical_acf_212": _BANKED.sprt_ar_deff("deff_empirical_acf_measured"),
                             "sprt_ar_deff_flat_441_212": _BANKED.sprt_ar_deff("deff_flat_441"),
                             "sprt_ar_rho_lag1_212": _BANKED.sprt_ar_rho_lag1(),
                             "sprt_ar_tightening_empirical_vs_flat_212": _BANKED.sprt_ar_tightening("empirical"),
                             "sprt_ar_tightening_ar_vs_flat_212": _BANKED.sprt_ar_tightening("ar"),
                             "sprt_ar_flat_441_is_conservative_212": _BANKED.sprt_ar_flat_is_conservative(),
                             "sprt_ar_savings_ratio_deff_invariant_212": _BANKED.sprt_ar_savings_invariant(),
                             "sprt_ar_savings_ratio_unchanged_212": _BANKED.sprt_ar_savings_ratio(),
                             "sprt_ar_realized_alpha_unchanged_212": _BANKED.sprt_ar_realized_alpha(),
                             "sprt_ar_realized_power_unchanged_212": _BANKED.sprt_ar_realized_power(),
                             "sprt_ar_private_bar_both_unchanged_212": _BANKED.sprt_ar_private_bar()})
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
    # combined-sigma row (ubel #204, RESOLVES #201): clean-1-sigma launch sigma -> P95 GO-trigger-vs-
    # ceiling. RESOLVED-YES (lambda=1 clears 500 at P95 both ends), NON-GATING.
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
        flat["worked_example/clean_acceptance_1sigma_tps"] = csc["clean_acceptance_1sigma_tps"]
        flat["worked_example/delta_mu_rebase_central_tps"] = csc["delta_mu_rebase_central_tps"]
        flat["worked_example/rebase_direction_matches_prediction"] = bool(csc["rebase_direction_matches_prediction"])
        flat["worked_example/acceptance_axis_realistic_icc_halfwidth_tps"] = csc["acceptance_axis_realistic_icc_halfwidth_tps"]
        flat["worked_example/sqrt_design_effect"] = csc["sqrt_design_effect"]
        flat["worked_example/headroom_shift_tps"] = csc["headroom_shift_tps"]
        flat["worked_example/icc0_combined_sigma_central_tps"] = csc["icc0_combined_sigma_central_tps"]
        flat["worked_example/anchor_err_195_dedup"] = csc["anchor_err_195_dedup"]
        flat["worked_example/anchor_err_194_breakeven"] = csc["anchor_err_194_breakeven"]
        flat["worked_example/combined_sigma_provisional"] = bool(csc["provisional"])
        flat["worked_example/combined_sigma_resolved"] = bool(csc["resolved"])
        flat["worked_example/combined_sigma_resolved_yes"] = bool(csc["resolved_yes"])
        flat["worked_example/combined_sigma_gates_analytic_go"] = bool(csc["gates_analytic_go"])
        flat["worked_example/combined_sigma_source_pr_trigger"] = int(csc["source_pr_trigger"])
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
    # denken #197 liveprobe: full-ladder GO requirement + depth-1 FALSE-GO guard (GATES the GO leg).
    lpc = led.get("liveprobe_measurement_cost") or {}
    if lpc.get("landed"):
        flat["worked_example/liveprobe_full_ladder_required"] = bool(lpc["full_ladder_required"])
        flat["worked_example/liveprobe_full_ladder_measured_worked"] = bool(lpc["full_ladder_measured"])
        flat["worked_example/liveprobe_false_go_risk_depth1_only"] = bool(lpc["false_go_risk_depth1_only"])
        flat["worked_example/liveprobe_mechanism_can_clear_private_bar"] = bool(
            lpc["mechanism_can_clear_private_bar"])
        flat["worked_example/liveprobe_depth1_overstatement_tps"] = lpc["depth1_overstatement_tps"]
        flat["worked_example/liveprobe_true_private_lcb_at_lambda1_tps"] = lpc["true_private_lcb_at_lambda1_tps"]
        flat["worked_example/liveprobe_beta_primary"] = lpc["beta_primary"]
        flat["worked_example/liveprobe_decisive_total_trials_lambda1"] = lpc["decisive_total_trials_lambda1"]
        flat["worked_example/liveprobe_neyman_efficiency_gain"] = lpc["neyman_efficiency_gain_vs_equal"]
        flat["worked_example/liveprobe_reinforces_hold"] = bool(lpc["reinforces_hold"])
        flat["worked_example/liveprobe_flips_verdict"] = bool(lpc["flips_verdict"])
        # denken #205 SPRT cost row (REPLACES #197's fixed-N; cost-row swap only, guard unchanged).
        flat["worked_example/liveprobe_cost_row_source_pr"] = int(lpc["cost_row_source_pr"])
        flat["worked_example/liveprobe_sequential_sprt_205_landed"] = bool(lpc["sequential_sprt_205_landed"])
        flat["worked_example/liveprobe_expected_n_sprt_nogo"] = lpc["expected_n_sprt_nogo"]
        flat["worked_example/liveprobe_expected_n_sprt_nogo_realistic_icc"] = lpc["expected_n_sprt_nogo_realistic_icc"]
        flat["worked_example/liveprobe_expected_n_sprt_nearbar"] = lpc["expected_n_sprt_nearbar"]
        flat["worked_example/liveprobe_worst_case_expected_n_sprt"] = lpc["worst_case_expected_n_sprt"]
        flat["worked_example/liveprobe_sprt_savings_vs_fixed_n"] = lpc["sprt_savings_vs_fixed_n"]
        flat["worked_example/liveprobe_fixed_n_worst_case_cap_trials"] = lpc["fixed_n_worst_case_cap_trials"]
        flat["worked_example/liveprobe_sprt_realized_alpha"] = lpc["sprt_realized_alpha"]
        flat["worked_example/liveprobe_sprt_realized_power"] = lpc["sprt_realized_power"]
        flat["worked_example/liveprobe_sprt_boundary_A_upper_decide_go"] = lpc["sprt_boundary_A_upper_decide_go"]
        flat["worked_example/liveprobe_sprt_deff_190"] = lpc["sprt_deff_190"]
        flat["worked_example/liveprobe_sprt_beta_nogo"] = lpc["sprt_beta_nogo"]
    # kanna #202 frozen-budget regime (conservative DEFAULT; annotation only).
    fba = led.get("frozen_budget_annotation") or {}
    if fba.get("landed"):
        flat["worked_example/frozen_regime_default"] = str(fba["regime_default"])
        flat["worked_example/frozen_mu_bar_frozen_p95_tps"] = fba["mu_bar_frozen_p95_tps"]
        flat["worked_example/frozen_mu_bar_fresh_p95_n5_tps"] = fba["mu_bar_fresh_p95_n5_tps"]
        flat["worked_example/frozen_p_bar_n5_frozen"] = fba["p_bar_n5_frozen"]
        flat["worked_example/frozen_delta_mu_tps"] = fba["delta_mu_frozen_tps"]
        flat["worked_example/frozen_freeze_robust_build_to_mu_tps"] = fba["freeze_robust_hedge"]["build_to_mu_tps"]
        flat["worked_example/frozen_n_shots_at_512"] = fba["freeze_robust_hedge"]["n_shots_frozen"]
        flat["worked_example/frozen_fragile_e_shots_at_bar"] = fba["build_at_bar_best_of_n_fragile"]["e_shots_frozen"]
        flat["worked_example/frozen_fraction_breakeven"] = fba["build_at_bar_best_of_n_fragile"]["frozen_fraction_breakeven"]
        flat["worked_example/frozen_reinforces_hold"] = bool(fba["reinforces_hold"])
    # ISSUE #192 compliance bracket (lawine #196 lane-b + wirbel #199 lane-a) -- ABOVE the sigma math.
    clb = led.get("compliant_lane_bracket") or {}
    if clb.get("landed"):
        lb = clb["lane_b_nonspec"]
        lna = clb["lane_a_compliant_spec"]
        flat["worked_example/compliant_sits_above_sigma_math"] = bool(clb["sits_above_sigma_math"])
        flat["worked_example/compliant_does_not_change_204_trigger"] = bool(clb["does_not_change_204_trigger"])
        flat["worked_example/compliant_500_lane_exists"] = bool(clb["compliant_500_lane_exists"])
        flat["worked_example/compliant_500_lane_is_spec_only"] = bool(clb["compliant_500_lane_is_spec_only"])
        flat["worked_example/compliant_500_conditioned_on_unmeasured_overhead"] = bool(
            clb["compliant_500_conditioned_on_unmeasured_overhead"])
        flat["worked_example/lane_b_nonspec_floor_tps"] = lb["official_tps_floor"]
        flat["worked_example/lane_b_nonspec_token_identity_rate"] = lb["token_identity_rate"]
        flat["worked_example/lane_b_nonspec_ppl"] = lb["ppl"]
        flat["worked_example/lane_b_nonspec_clears_500"] = bool(lb["clears_500"])
        flat["worked_example/lane_b_nonspec_margin_to_500_pct"] = lb["margin_to_500_pct"]
        flat["worked_example/lane_b_nonspec_spec_premium_tps"] = lb["spec_premium_tps"]
        flat["worked_example/lane_b_nonspec_spec_premium_pct"] = lb["spec_premium_pct"]
        flat["worked_example/lane_a_compliant_spec_ceiling_tps"] = lna["tps_ceiling"]
        flat["worked_example/lane_a_compliant_spec_floor_tps"] = lna["tps_floor"]
        flat["worked_example/lane_a_compliant_spec_ceiling_ci_lower_tps"] = lna["ceiling_ci_lower_tps"]
        flat["worked_example/lane_a_compliant_spec_lower_clears_500"] = bool(lna["ceiling_lower_clears_500"])
        flat["worked_example/lane_a_compliant_spec_floor_clears_500"] = bool(lna["floor_clears_500"])
        flat["worked_example/lane_a_max_kernel_overhead_pct_both"] = lna["max_kernel_overhead_pct_both_bugs"]
        flat["worked_example/lane_a_max_kernel_overhead_pct_descent"] = lna["max_kernel_overhead_pct_descent"]
        flat["worked_example/lane_a_offshelf_overhead_ref_122"] = lna["offshelf_overhead_ref_122"]
        flat["worked_example/lane_a_overhead_is_measured"] = bool(lna["overhead_is_measured"])
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
        # #197 full-ladder GO guard: GATES `go` (None on a non-landed read -> skip).
        if lgc.get("full_ladder_ok") is not None:
            flat[f"worked_example/{topo}/full_ladder_ok"] = bool(lgc["full_ladder_ok"])
            flat[f"worked_example/{topo}/full_ladder_measured"] = bool(lgc["full_ladder_measured"])
            flat[f"worked_example/{topo}/full_ladder_gates_go"] = bool(lgc["full_ladder_gates_go"])
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
    lpc = liveprobe_measurement_cost(_M.canonical_ladder(1.0, "both_bugs"))   # #197/#205/#212, worked full ladder
    fba = frozen_budget_annotation()                                         # #202
    clb = compliant_lane_bracket()                                           # #196/#199/#213 (#192 bracket)
    wca = winners_curse_annotation()                                         # #210 winner's-curse private bar
    ar212 = lpc.get("ar_corrected_cost_band_212") or {}                      # #212 AR(1) cost band
    kb213 = clb.get("lane_a_kernel_budget_213") or {}                        # #213 kernel-overhead-vs-lambda
    handoff = (
        "launch_trigger_calculator: one-call launch_decision(measured_tuple) -> verified "
        "GO/NO-GO + filled (un-filed) Approval request. Self-test %s. RECOMPOSED post-merge "
        "(advisor 17:27Z + 17:52Z + 18:03Z + 18:23Z + 18:39Z + 18:46Z + 19:09Z): the TYPED launch-CI "
        "ledger reads REAL banked scalars -- #190 ICC, #191 private, #188 sigma-oneshot, #187 "
        "input-side, #194 re-draw budget, #195 cross-axis covariance, #200 cost, #201->#204 "
        "launch-sigma closure+UNIT-REBASE, #197->#205 liveprobe depth-budget+SPRT, #202 frozen-budget "
        "AND #196/#199 #192-compliant lanes have ALL LANDED (flag=consumed); the ledger is CLOSED. "
        "binding_bar = max(public#183 %.4f, ICC#190 %.4f, "
        "private#191 %.4f) = %.4f (private DOMINATES, both-bugs); descent private bar is UNREACHABLE. "
        "land #71 must show lambda_hat_built >= %.4f AND the #179 launch-projection cell-LCB(P>=0.9) "
        ">= 500 under the REALISTIC #190 +-%.2f ICC half-width (NOT iid +-%.2f) AND on the #191 "
        "private axis. launch_authorized = (analytic-GO AND all precondition rows GO); preconditions "
        "(PRECACHE_BENCH, ubel #189 packaging-gate=GO but re-verify-pending, human approval) PENDING "
        "-> launch_authorized=False (authorizes nothing). both-bugs SURVIVES at lambda=1: realistic "
        "launch-LCB 510.63>=500 + private valid -> robust GO (HELD). descent is DOUBLY-HARDENED NO-GO: "
        "private build bar UNREACHABLE AND misses realistic 495.04 + private 490.16 launch LCBs. "
        "COMBINED-SIGMA ROW (ubel #204, advisor 18:46Z -- RETIRES #201's trigger on a UNITS bug; "
        "RESOLVED-YES + NON-GATING): de-dup (#195, rho=%.3f double-count -> IDENTITY 5.32 iid) x "
        "realistic ICC (#190, MAGNITUDE -> %.2f z=1.96 HALF-WIDTH); #204 rebases to a clean 1-sigma "
        "%.3f (+) sigma_hw (+) sigma_private -> combined LAUNCH sigma %.2f central / %.2f worst-case; "
        "P95 GO trigger mu>=%.2f central / %.2f worst-case vs the lambda=1 ceiling %.2f -> BOTH BELOW "
        "-> lambda=1 clears 500 at P95 CENTRALLY (+%.2f) AND worst-case (+%.2f). RESOLVED-YES (was a "
        "PROVISIONAL knife-edge; direction sign-backwards -> trigger SHIFTED %.2f, DOWN not up). "
        "NON-GATING -- this row does NOT gate `go` (it does not authorize a launch). ubel #207 "
        "(advisor 19:40Z) RESOLVED the #175 two-readings caveat in FAVOR of the YES: the larger "
        "10.906 reading is the B=16384 128-tok SUB-bench (RETIRED), so the launch-correct trigger "
        "512.41/514.63 STANDS and the robust-YES SURVIVES; only land #71 co-log n=%s remains open "
        "(it now merely TIGHTENS the YES, never flips it). #200 cost annotation (budget row ONLY, "
        "single-shot logic "
        "UNCHANGED): realistic spend at the bar is SEQUENTIAL E[shots]=%.2f (not fixed-5); "
        "build-higher (mu>=%.1f/N=1) beats stay-at-bar iff reaching mu costs < %.0f shots' GPU-$ "
        "(c*=%.2f*b fixed / %.2f*b sequential). #197 (denken, advisor 18:39Z): the GO leg now GATES on "
        "land #71's MEASURED full-ladder q[2..9] >= 0.9780 -- a depth-1-only/spine-inferred read is a "
        "FALSE GO worth %.1f TPS (true private LCB %.1f<<500; at beta=%.3f the mechanism CANNOT clear "
        "the private bar, mech_can_clear=%s -> a real GO needs beta~1 across the MEASURED ladder). "
        "#205 (denken, advisor 19:09Z) REPLACES #197's fixed-N cost row with the realistic SPRT: "
        "E[N]~%.0f trials on a clear-NO-GO build (%.1fx collapse vs #197's %.0f fixed-N), ~%.0f "
        "near-bar, <=%.0f worst-case ASN; realized (alpha,power)=(%.2f,%.2f). Full-ladder GO guard "
        "UNCHANGED (cost-row swap, not a verdict change). The worked tuple's 8-entry ladder SATISFIES "
        "the guard -> verdict HELD. #202 (kanna, advisor 18:39Z): the "
        "multi-shot budget DEFAULTS to the conservative FROZEN regime (best-of-N beats only sigma_hw -> "
        "N=5@bar P=%.3f not fresh 0.969; default build-bar input mu_bar_frozen_p95=%.2f not fresh "
        "%.2f); THE HEDGE -- build-to-mu=%.1f/N=1 is fully freeze-robust (n_shots_frozen=%s) so the "
        "SAFE recommendation is untouched; only build-at-bar+best-of-N is frozen-fragile (E[shots]="
        "%.2f, exhausts 19%%, breakeven f*=%.3f). BOTH #197 + #202 REINFORCE the HOLD -- they sharpen "
        "the measurement spec + budget robustness, not flip it. ISSUE #192 COMPLIANCE BRACKET (lawine "
        "#196 lane-b + wirbel #199 lane-a, advisor 18:58Z/19:09Z) sits ABOVE the sigma math and does "
        "NOT change the #204 trigger: lane-b (non-spec, EMPIRICAL) is token-identical but FLOORS at "
        "%.2f official TPS (%.1f%% below 500) -> NO compliant non-spec 500-lane (spec premium %.1f TPS "
        "existential); lane-a (compliant-spec int4 VERIFY kernel) ceiling %.2f (lower-CI %.2f>500 -> "
        "CLEARS), floor %.2f, clears 500 ONLY at <%.2f%% both / %.2f%% descent kernel overhead "
        "(UNMEASURED; off-shelf #122 +%.1f%%). So lane-a is the SINGLE compliant 500-route, and a NEW "
        "hard precondition (issue_192_human_ruling, PENDING) gates the launch on the #192 ruling -- "
        "keeping launch_authorized=False. wirbel #213 (advisor 19:40Z) grades lane-a's overhead "
        "budget vs lambda: max_overhead(lambda) opens to %.2f%% both / %.2f%% descent at lambda=1, "
        "the zero-overhead path first clears 500 at lambda_crit=%.4f both / %.4f descent -> lane-a is "
        "a DOUBLE gate (lambda>=lambda_crit AND kernel-under-budget); off-shelf #122 (+51.78%%) clears "
        "at NO physical lambda<=1. kanna #210 (advisor 19:40Z) CORRECTS the #202 hedge against the "
        "PRIVATE bar: best-of-N does NOT relax it (private clear FLAT in N, n_star_private=%s), the "
        "freeze-robust mu=512.2/N=1 does NOT survive privately (p=%.4f<0.95), so the PUBLIC build must "
        "reach mu_bar_private_corrected=%.2f (+%.2f winner's-curse tax over #202's 504.87) -- build "
        "higher, N=1; this BUILD target does NOT touch the %.2f sigma->LCB PUBLIC trigger. denken #212 "
        "(advisor 19:54Z) SHARPENS #205's flat xDeff=4.41 cost: folding #190's DECAYING ACF gives the "
        "E[N]_nogo band [%.0f IID -> %.0f AR(1) -> %.0f measured-ACF-realistic -> %.0f flat-loose] "
        "(data-grounded point %.0f); the 75.12x collapse is Deff-INVARIANT, (alpha,power)=(0.05,0.95) "
        "+ bar 0.9780 UNCHANGED. both_bugs_go_at_lambda_star=%s. Human "
        "approval still required before any HF spend." % (
            "PASSES" if st["launch_trigger_calculator_self_test_passes"] else "FAILS",
            bb_bind["public_183"], bb_bind["icc_190"], bb_bind["private_191"], bb_bind["binding_bar"],
            bb_bind["binding_bar"], _BANKED.halfwidth_realistic(), _BANKED.halfwidth_iid(),
            csc["dedup_provenance_195"]["rho_sampling_input"],
            csc["acceptance_axis_realistic_icc_halfwidth_tps"], csc["clean_acceptance_1sigma_tps"],
            csc["combined_sigma_launch_central_tps"], csc["combined_sigma_launch_worstcase_tps"],
            csc["go_trigger_mu_central_tps"], csc["go_trigger_mu_worstcase_tps"],
            csc["lambda1_ceiling_mu_tps"], csc["central_margin_at_lambda1_tps"],
            csc["worstcase_margin_at_lambda1_tps"], csc["delta_mu_rebase_central_tps"],
            csc["colog_n_allocations"],
            cba["stay_at_bar"]["expected_shots_sequential"], cba["build_higher"]["mu_safe_n1_tps"],
            cba["crossover_total_shots"], cba["c_star_fixedN_per_b"], cba["c_star_sequential_per_b"],
            lpc["depth1_overstatement_tps"], lpc["true_private_lcb_at_lambda1_tps"],
            lpc["beta_primary"], lpc["mechanism_can_clear_private_bar"],
            lpc["expected_n_sprt_nogo"], lpc["sprt_savings_vs_fixed_n"],
            lpc["decisive_total_trials_lambda1"], lpc["expected_n_sprt_nearbar"],
            lpc["worst_case_expected_n_sprt"], lpc["sprt_realized_alpha"],
            lpc["sprt_realized_power"],
            fba["p_bar_n5_frozen"], fba["mu_bar_frozen_p95_tps"], fba["mu_bar_fresh_p95_n5_tps"],
            fba["freeze_robust_hedge"]["build_to_mu_tps"], fba["freeze_robust_hedge"]["n_shots_frozen"],
            fba["build_at_bar_best_of_n_fragile"]["e_shots_frozen"],
            fba["build_at_bar_best_of_n_fragile"]["frozen_fraction_breakeven"],
            clb["lane_b_nonspec"]["official_tps_floor"],
            abs(clb["lane_b_nonspec"]["margin_to_500_pct"]),
            clb["lane_b_nonspec"]["spec_premium_tps"],
            clb["lane_a_compliant_spec"]["tps_ceiling"],
            clb["lane_a_compliant_spec"]["ceiling_ci_lower_tps"],
            clb["lane_a_compliant_spec"]["tps_floor"],
            clb["lane_a_compliant_spec"]["max_kernel_overhead_pct_both_bugs"],
            clb["lane_a_compliant_spec"]["max_kernel_overhead_pct_descent"],
            clb["lane_a_compliant_spec"]["offshelf_overhead_ref_122"] * 100.0,
            # wirbel #213 lane-a kernel-overhead-vs-lambda DOUBLE gate:
            kb213.get("overhead_budget_pct_at_lambda1_both_bugs", float("nan")),
            kb213.get("overhead_budget_pct_at_lambda1_descent", float("nan")),
            kb213.get("lambda_crit_clears_500_both_bugs", float("nan")),
            kb213.get("lambda_crit_clears_500_descent", float("nan")),
            # kanna #210 winner's-curse private build target:
            wca["n_star_private"], wca["p_private_clear_at_mu512p2_n1"],
            wca["mu_bar_private_corrected_tps"], wca["delta_mu_winners_curse_tps"],
            csc["go_trigger_mu_central_tps"],
            # denken #212 AR(1)-corrected E[N]_nogo cost band:
            ar212.get("expected_n_nogo_iid_floor", float("nan")),
            ar212.get("expected_n_nogo_ar_optimistic", float("nan")),
            ar212.get("expected_n_nogo_realistic_measured_acf", float("nan")),
            ar212.get("expected_n_nogo_flat_loose", float("nan")),
            ar212.get("expected_n_nogo_realistic_measured_acf", float("nan")),
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
