#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Compliant-spec E[T] CEILING (PR #199, wirbel) — CPU-only analytic synthesis.

THE QUESTION (Issue #192 lane-a)
--------------------------------
Issue #192 (the live Greedy Decode Correctness gate) may invalidate the int4-*speculative*
500-path: kanna #114 (`9q5yy9l1`) measured 56.08% token-divergence from plain greedy AR,
structural in the int4 Marlin spec-verify GEMM batch-VARIANCE; kanna #122 (`n5bypf5h`)
proved off-the-shelf ``VLLM_BATCH_INVARIANT=1`` fails (56→58.57%, +51.78% TPS). The
advisor's #192 reply named two compliant lanes: (a) a custom batch-INVARIANT int4 verify
kernel (spec ON, token-identical) and (b) non-speculative int4 AR (spec OFF). lawine #196
measures lane-b's PUBLIC TPS floor empirically. THIS leg is the analytic twin: lane-a's
E[T] **CEILING** — what acceptance length, and therefore what TPS, a token-identical
(batch-invariant) verify would achieve, and whether that compliant-spec config can clear
500 at all.

THE BRACKET (why this is a bracket, not a point)
------------------------------------------------
Under a batch-invariant greedy verify the accept rule becomes EXACTLY "draft top-1 == the
true greedy token", so the rank-1-coverage rate IS the compliant-lane accept probability,
and the current batch-VARIANT stack's spurious argmax-flip rejects (the 56% divergence)
are precisely the accepts a batch-invariant verify would RECOVER. Therefore the compliant
E[T] is bounded:
  * BELOW by the current measured spec E[T] at the realistic λ̂=0.342 (denken #178's graded
    self-KV curve) — batch-invariance only REMOVES spurious rejects, so E[T] cannot fall
    below the batch-variant value at the same λ.  ``et_compliant_floor``.
  * ABOVE by the rank-1-coverage ceiling — propagate the measured per-depth rank-1 ladder
    ``q_compliant`` through the #175/#184 reach-DP. This is the JOINT best case: full
    rank-1 coverage (no spurious rejects) AND the full-self-KV conditional ladder (≈λ=1),
    so it sits at the top of #178's E[T](λ) envelope.  ``et_compliant_ceiling``.

HONEST DATA CAVEAT (stated, drives the fall-back-to-bracket)
-----------------------------------------------------------
The PR#86 rankprobe (the in-scope ``rankprobe_records.jsonl.118860`` decomposed in #190)
records, per draft position, the rank of the TARGET greedy token in the drafter top-W.
But the rankprobe's "true token" is ``target_argmax`` — the DEPLOYED (batch-VARIANT) int4
Marlin verify GEMM's argmax (rankprobe_patch.py ``_log_verify``), NOT a clean batch-
invariant / AR greedy argmax. So the data does NOT cleanly expose the batch-invariant
token; we therefore DO NOT claim a point estimate of the compliant accept and FALL BACK TO
THE BRACKET (PR leg-1 instruction). The rank-1-coverage ladder is used as the OPTIMISTIC
ceiling (an upper bound on the achievable compliant acceptance), not as the realized
compliant accept.

COMPOSITION (fully pinned; imported, NOT re-derived)
----------------------------------------------------
official = K_cal·(E[T]/step)·τ ; K_cal=125.268, step=1.2182, τ∈[0.9924,1.0]. Clear-500
bar E[T] ≥ 4.862. A real batch-invariant kernel carries an UNKNOWN per-step cost (kanna
#122's off-the-shelf +51.78% is a NON-working reference, not a bound); we report TPS at
the CURRENT step (zero-kernel-overhead ceiling) AND the max verify-step inflation that
still clears 500 at the bracket — the kernel-dev budget.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / kernel build / served-file
change. BASELINE stays 481.53; adds **0 TPS**; greedy/PPL untouched. Imports #190
(`fva6o4ug`) rankprobe reconstruction, #175 (`zh1accmi`) reach-DP+pmf+σ_L, #184
(`7uek36mx`) E[T] machinery + max-E[T]@λ envelope, denken #178 (`zjdc7hhh`) graded E[T](λ)
+ λ̂, #172/#160 anchors (descent 5.0564 / both-bugs 5.2070). Does NOT re-derive any of them.
**NOT open2. NOT a launch.** Pairs with lawine #196 (lane-b empirical floor twin).

PRIMARY metric  compliant_spec_et_self_test_passes
TEST    metric  compliant_spec_tps_ceiling  (both-bugs, τ=1, zero-kernel-overhead)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import resource
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Imports — path-based (mirrors #178 / #184). Do NOT re-derive imported machinery.
# --------------------------------------------------------------------------- #
def _load(name: str, relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D172 = _load("descent_et_dp_audit", "research/validity/descent_et_audit/descent_et_dp_audit.py")
SELF = _load("realistic_selfkv_floor",
             "research/oracle_readout/realistic_selfkv_floor/realistic_selfkv_floor.py")
ETSM = _load("et_second_moment", "research/oracle_readout/et_second_moment/et_second_moment.py")

sys.path.insert(0, str(REPO_ROOT / "scripts/profiler"))
import treeshape_measured_accept as TS  # noqa: E402

# --------------------------------------------------------------------------- #
# Committed launch-composition constants (imported).
# --------------------------------------------------------------------------- #
K_CAL = D172.K_CAL                              # 125.26795005202914 (ubel #148/#169)
STEP = D172.STEP_OVERLAP                        # 1.2182 (lawine #168 realized step)
TAU_CENTRAL = 1.0
TAU_CONS = 0.9924                               # served-fraction conservative corner (#181)
TAU_CORNERS = (("tau_central_1p0", TAU_CENTRAL), ("tau_conservative_0p9924", TAU_CONS))
TARGET = D172.TARGET_OFFICIAL                   # 500.0
Z95 = ETSM.Z95                                  # 1.959963984540054
BENCH_TOKENS = ETSM.BENCH_TOKENS                # 16384 (128×128 PR contract)

IMPORTED_DESCENT_5p0564 = D172.IMPORTED_DESCENT_ONLY_0679   # 5.056404568844709
IMPORTED_BOTH_BUGS_5p2070 = D172.IMPORTED_BOTH_BUGS         # 5.206954309441963
DESCENT_ONLY_D1 = D172.ORACLE_DEPTH1_ALT                    # 0.679 (BUG-1-unfixed depth-1 spine)

# in-scope rankprobe shard (#190 source; read-only) and its published cross-check ladder.
RANKPROBE_SHARD = REPO_ROOT / "research/rank_coverage/pr86/rankprobe_records.jsonl.118860"
# Published rank-1 conditional ladder from a DIFFERENT probe shard (z6wi4z4v, root
# rank_coverage_results.json) — sanity cross-check only (different draw → ~1% noise).
PUBLISHED_RANK1_LADDER_Z6 = [0.7335390946502057, 0.7655308967906939, 0.7966375687035241,
                             0.8260281385281385, 0.840812315754995, 0.8379431242695754,
                             0.8496048349604834]
# accept_calibration #76 deployed conditional ladder (5m17r52s) — the q_deployed spine.
ACCEPT76_CONDITIONAL = [0.728739760479042, 0.7589764102641635, 0.7924989076194682,
                        0.821702519412012, 0.8342716929825772, 0.8352594665096346,
                        0.8472621220149911]

# divergence / cost evidence (board context for the hand-off; not used in the DP).
KANNA114_DIVERGENCE = 0.5608                    # 56.08% token-divergence (9q5yy9l1)
KANNA122_OFFSHELF_OVERHEAD = 0.5178             # +51.78% off-the-shelf (n5bypf5h) — NON-working ref

TOL_PROV = 1e-9                                 # DP-entry-point agreement (provenance)
TOL_REPRO = 1e-6                                # endpoint reproduction vs #172/#184 anchors


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Shared context: both-bugs + descent λ-spine endpoints (imported #172/#178/#184).
# --------------------------------------------------------------------------- #
def _build_context() -> dict[str, Any]:
    anchors = D172.load_anchors(D172.DEFAULT_BUG2_ANCHOR, D172.DEFAULT_TOPO_JSON,
                                D172.DEFAULT_ACCEPT_JSON, D172.DEFAULT_RANKCOV_JSON,
                                D172.DEFAULT_DECOMP_JSON)
    ep = SELF.build_endpoints(anchors)          # q_full (descent d1=0.679), q_floor (d1=0.674)
    q_full_bb = list(ep["q_deployed"])          # both-bugs full: d1=0.7287, rising
    q_floor_bb = list(ep["q_floor"]); q_floor_bb[0] = ep["q_deployed"][0]   # d1 λ-insensitive
    lam_hat = ((SELF.LIVEPROBE_WALK_TOPW0_HIT - ep["q_floor"][0]) /
               (SELF.LIVEPROBE_LINEAR_TOP1 - ep["q_floor"][0]))             # ≈ 0.342
    return {
        "anchors": anchors, "ep": ep,
        "parent": list(ep["parent"]), "rho_cond": ep["rho_cond"], "W": ep["W"],
        "H": ep["horizon"], "n_nodes": len(ep["parent"]),
        "q_full_descent": list(ep["q_full"]), "q_floor_descent": list(ep["q_floor"]),
        "q_full_bb": q_full_bb, "q_floor_bb": q_floor_bb,
        "lam_hat": lam_hat,
    }


CTX = _build_context()


# --------------------------------------------------------------------------- #
# DP entry points: floor (= #178 et_backward) and ceiling (= #175 pmf-mean), the
# SAME object — score_tree_depthrank(build_depth_pvecs_measured(spine)) == et_backward.
# --------------------------------------------------------------------------- #
def et_floor_of_spine(spine: list[float]) -> float:
    """#178 / #172 first-moment DP (et_backward) on a λ-spine."""
    return SELF.et_of_spine(CTX["ep"], spine)


def et_via_reachdp(spine: list[float]) -> dict[str, Any]:
    """Propagate a per-depth rank-1 spine through the #175/#184 reach-DP.

    Returns the pmf-mean E[T] (≡ score_tree_depthrank ≡ et_backward), σ_L (#175 second
    moment), the score-tree E[T] and et_backward as independent provenance reads, and the
    accepted-length pmf total mass. flat-extrapolation past the measured horizon (the
    build_depth_pvecs_measured default), matching #178/#184.
    """
    parent = CTX["parent"]
    _, depth = D172.build_children(parent)
    maxd = max(depth)
    pvecs = TS.build_depth_pvecs_measured(list(spine), CTX["rho_cond"], CTX["W"], maxd, "flat")
    pmf, _, _, _ = ETSM.dp_accepted_length_pmf(parent, pvecs)
    mom = ETSM.pmf_moments(pmf)
    score_et, _ = TS.score_tree_depthrank(parent, pvecs)
    children, dep = D172.build_children(parent)
    back_et = D172.et_backward(parent, children, dep, list(spine), CTX["rho_cond"], CTX["W"])
    return {
        "et_pmf_mean": mom["mean"], "et_score_tree": score_et, "et_backward": back_et,
        "sigma_L": mom["std"], "var_L": mom["var"], "total_mass": mom["total_mass"],
    }


# --------------------------------------------------------------------------- #
# (1) Compliant-lane accept ladder from rank-1 coverage (in-scope rankprobe shard).
# --------------------------------------------------------------------------- #
def load_q_compliant(path: Path, max_records: int | None = None) -> dict[str, Any]:
    """Per-depth rank-1 conditional accept ladder from the PR#86 rankprobe.

    Each record: n drafted positions, fd = first divergence (draft != target argmax;
    positions 0..fd-1 accepted at rank-1, hr[d]==1), all_acc. A record REACHES position d
    iff it accepted 0..d-1 (d ≤ fd) and the position exists (d < n); it ACCEPTS at d iff
    d < fd. The conditional ladder q_compliant[d] = accept[d] / reached[d] is exactly
    "draft top-1 == target greedy token | reached depth d" — the batch-invariant accept
    event under the stated (caveated) assumption.
    """
    reached: dict[int, int] = defaultdict(int)
    accept: dict[int, int] = defaultdict(int)
    n_records = 0
    n_align_bad = 0
    n_all_acc = 0
    maxpos = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("align", True):
                n_align_bad += 1
                continue                         # the probe drops mis-aligned proposals
            n = int(rec["n"])
            fd = int(rec["fd"])
            if rec.get("all_acc"):
                n_all_acc += 1
            n_records += 1
            maxpos = max(maxpos, n - 1)
            for d in range(n):
                if d <= fd:
                    reached[d] += 1
                if d < fd:
                    accept[d] += 1
            if max_records is not None and n_records >= max_records:
                break
    K = maxpos + 1
    per_reached = [reached[d] for d in range(K)]
    per_accept = [accept[d] for d in range(K)]
    q = [(accept[d] / reached[d]) if reached[d] else None for d in range(K)]
    tot_reached = sum(per_reached)
    tot_accept = sum(per_accept)
    return {
        "shard": str(path.relative_to(REPO_ROOT)),
        "n_records": n_records, "n_align_bad": n_align_bad, "n_all_acc": n_all_acc,
        "measured_depths": K,
        "per_depth_reached": per_reached, "per_depth_accept": per_accept,
        "q_compliant": q,                              # rank-1 conditional ladder (depth 1..K)
        "rank1_coverage_top1": q[0] if q else None,    # depth-1 top-1 match rate
        "rank1_coverage_rate_pooled": (tot_accept / tot_reached) if tot_reached else None,
        "mean_accepted_run_length_fd": (
            sum(per_accept) / n_records if n_records else None),  # E[fd] = Σ survival
    }


def _ladder_diff(a: list[float | None], b: list[float]) -> float:
    n = min(len([x for x in a if x is not None]), len(b))
    return max((abs(a[i] - b[i]) for i in range(n) if a[i] is not None), default=float("nan"))


# --------------------------------------------------------------------------- #
# (3) TPS composition + clear-500 + kernel-overhead budget.
# --------------------------------------------------------------------------- #
def official_tps(et: float, tau: float, step: float = STEP) -> float:
    return D172.official_tps(et, step, K_CAL, tau)


def clear_bar(tau: float) -> float:
    return D172.clear500_bar(STEP, K_CAL, tau, TARGET)


def max_kernel_overhead_pct(et: float, tau: float) -> float:
    """Max verify-step inflation (1+o) that still clears 500 at this E[T].

    official = K_cal·(E[T]/(step·(1+o)))·τ ≥ 500  ⇔  (1+o) ≤ E[T]/bar(τ)  ⇔  o ≤ E[T]/bar − 1.
    Negative ⇒ misses 500 even at zero kernel overhead.
    """
    return (et / clear_bar(tau) - 1.0) * 100.0


def propagate(et: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag, tau in TAU_CORNERS:
        tps = official_tps(et, tau)
        out[tag] = {
            "tau": tau, "official_tps": tps, "clear500_bar_et": clear_bar(tau),
            "clears_500": bool(tps >= TARGET),
            "tps_margin_over_500": tps - TARGET,
            "max_kernel_overhead_pct_to_clear_500": max_kernel_overhead_pct(et, tau),
        }
    return out


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(shard: Path, max_records: int | None = None) -> dict[str, Any]:
    ep = CTX["ep"]
    H = CTX["H"]
    lam_hat = CTX["lam_hat"]
    bar1 = clear_bar(TAU_CENTRAL)

    # ---------- (1) rank-1-coverage ladder q_compliant ---------- #
    rc = load_q_compliant(shard, max_records)
    q_compliant = [x for x in rc["q_compliant"] if x is not None]
    data_exposes_batch_invariant_cleanly = False   # target_argmax is batch-VARIANT (caveat)
    sanity = {
        "max_abs_diff_vs_published_z6wi4z4v": _ladder_diff(rc["q_compliant"],
                                                           PUBLISHED_RANK1_LADDER_Z6),
        "max_abs_diff_vs_accept76_deployed": _ladder_diff(rc["q_compliant"],
                                                          ACCEPT76_CONDITIONAL),
    }

    # ceiling spines: descent keeps BUG-1-unfixed depth-1 (0.679); both-bugs uses the
    # measured rank-1 depth-1 (BUG-1 fixed). Deeper depths = measured rank-1 coverage.
    ceil_spine = {
        "descent_only": [DESCENT_ONLY_D1] + q_compliant[1:],
        "both_bugs": list(q_compliant),
    }
    # #178-anchored full-coverage endpoints (provenance: reproduce 5.0564 / 5.2070 exactly).
    anchor_spine = {
        "descent_only": list(CTX["q_full_descent"]),
        "both_bugs": list(CTX["q_full_bb"]),
    }
    anchor_et_target = {
        "descent_only": IMPORTED_DESCENT_5p0564,
        "both_bugs": IMPORTED_BOTH_BUGS_5p2070,
    }
    floor_endpoints = {
        "descent_only": (CTX["q_floor_descent"], CTX["q_full_descent"]),
        "both_bugs": (CTX["q_floor_bb"], CTX["q_full_bb"]),
    }

    # ---------- (2) bracket per topology ---------- #
    brackets: dict[str, Any] = {}
    for topo in ("descent_only", "both_bugs"):
        q_floor, q_full = floor_endpoints[topo]
        spine_floor = SELF.spine_from_profile(ep, SELF.constant_lambda(H, lam_hat),
                                              q_floor, q_full)
        et_floor = et_floor_of_spine(spine_floor)             # #178 graded E[T] at λ̂
        ceil = et_via_reachdp(ceil_spine[topo])
        et_ceiling = ceil["et_pmf_mean"]
        anchor = et_via_reachdp(anchor_spine[topo])
        brackets[topo] = {
            "et_compliant_floor": et_floor,
            "et_compliant_ceiling": et_ceiling,
            "ceiling_sigma_L": ceil["sigma_L"],
            "bracket_ordered_floor_le_ceiling": bool(et_floor <= et_ceiling + 1e-12),
            "bracket_width_E_T": et_ceiling - et_floor,
            "ceiling_spine": ceil_spine[topo],
            "floor_spine_at_lambda_hat": spine_floor,
            "floor_propagate": propagate(et_floor),
            "ceiling_propagate": propagate(et_ceiling),
            "ceiling_finite_sample_ci_tau1": ETSM.finite_sample_tps_ci(
                et_ceiling, ceil["sigma_L"], BENCH_TOKENS, STEP, TAU_CENTRAL, Z95),
            # provenance reads (self-test d): three DP entry points must agree
            "ceiling_dp_provenance": {
                "et_pmf_mean": ceil["et_pmf_mean"], "et_score_tree": ceil["et_score_tree"],
                "et_backward": ceil["et_backward"],
                "max_abs_resid": max(abs(ceil["et_pmf_mean"] - ceil["et_score_tree"]),
                                     abs(ceil["et_pmf_mean"] - ceil["et_backward"])),
            },
            # anchored full-coverage endpoint reproduces #172/#184 (self-test b)
            "anchor_full_coverage": {
                "et": anchor["et_pmf_mean"], "target": anchor_et_target[topo],
                "resid": abs(anchor["et_pmf_mean"] - anchor_et_target[topo]),
                "reproduces": bool(abs(anchor["et_pmf_mean"] - anchor_et_target[topo])
                                   <= TOL_REPRO),
            },
            # self-test (a): ceiling-DP fed the floor's λ̂ spine reproduces the floor
            "selftest_a_floor_reproduce": {
                "et_reachdp_on_floor_spine": et_via_reachdp(spine_floor)["et_pmf_mean"],
                "et_floor_backward": et_floor,
                "resid": abs(et_via_reachdp(spine_floor)["et_pmf_mean"] - et_floor),
            },
        }

    # headline = both-bugs (BUG-1 fixed + batch-invariant = the compliant best case).
    head = brackets["both_bugs"]
    compliant_spec_tps_ceiling = head["ceiling_propagate"]["tau_central_1p0"]["official_tps"]
    compliant_spec_tps_floor = head["floor_propagate"]["tau_central_1p0"]["official_tps"]
    compliant_spec_clears_500 = bool(
        head["ceiling_propagate"]["tau_central_1p0"]["clears_500"])     # zero-kernel-overhead
    max_kernel_overhead = head["ceiling_propagate"]["tau_central_1p0"][
        "max_kernel_overhead_pct_to_clear_500"]

    # ---------- (4) self-test (PRIMARY) ---------- #
    cond_a = all(brackets[t]["selftest_a_floor_reproduce"]["resid"] <= TOL_PROV
                 for t in brackets)
    cond_b = all(brackets[t]["anchor_full_coverage"]["reproduces"] for t in brackets)
    cond_c = all(brackets[t]["bracket_ordered_floor_le_ceiling"] for t in brackets)
    cond_d = all(brackets[t]["ceiling_dp_provenance"]["max_abs_resid"] <= TOL_PROV
                 for t in brackets)
    # (e) NaN-clean is set by the caller after the full payload walk.
    conditions = {
        "a_zero_spurious_reject_reproduces_current_spec_ET": bool(cond_a),
        "b_rank_coverage_to_1_reproduces_184_maxET_envelope": bool(cond_b),
        "c_bracket_ordered_floor_le_ceiling": bool(cond_c),
        "d_reproduce_175_pmf_mean_to_ET_provenance": bool(cond_d),
        "e_nan_clean": True,
    }

    verdict = _verdict(compliant_spec_clears_500,
                       head["floor_propagate"]["tau_central_1p0"]["clears_500"],
                       max_kernel_overhead)
    handoff = _handoff(brackets, lam_hat, compliant_spec_tps_ceiling, compliant_spec_tps_floor,
                       compliant_spec_clears_500, max_kernel_overhead)

    return {
        "self_test": {
            "compliant_spec_et_self_test_passes": bool(all(conditions.values())),
            "conditions": conditions,
        },
        "test_metric": {"compliant_spec_tps_ceiling": compliant_spec_tps_ceiling},
        "headline": {
            "compliant_spec_tps_ceiling": compliant_spec_tps_ceiling,        # TEST (both-bugs, τ=1, 0-overhead)
            "compliant_spec_tps_floor": compliant_spec_tps_floor,
            "compliant_spec_clears_500": compliant_spec_clears_500,          # ceiling, zero kernel overhead
            "compliant_spec_floor_clears_500":
                bool(head["floor_propagate"]["tau_central_1p0"]["clears_500"]),
            "max_kernel_overhead_pct_to_clear_500": max_kernel_overhead,     # kernel-dev budget at the ceiling
            "et_compliant_bracket_both_bugs":
                [head["et_compliant_floor"], head["et_compliant_ceiling"]],
            "et_compliant_bracket_descent_only":
                [brackets["descent_only"]["et_compliant_floor"],
                 brackets["descent_only"]["et_compliant_ceiling"]],
        },
        "rank1_coverage": {
            **{k: rc[k] for k in ("shard", "n_records", "n_align_bad", "n_all_acc",
                                  "measured_depths", "per_depth_reached", "per_depth_accept",
                                  "q_compliant", "rank1_coverage_top1",
                                  "rank1_coverage_rate_pooled", "mean_accepted_run_length_fd")},
            "data_exposes_batch_invariant_cleanly": data_exposes_batch_invariant_cleanly,
            "assumption_note": (
                "rankprobe true token == target_argmax == the DEPLOYED batch-VARIANT int4 "
                "Marlin verify GEMM argmax (rankprobe_patch.py _log_verify), NOT a clean "
                "batch-invariant/AR greedy argmax. Data does NOT cleanly expose the batch-"
                "invariant token → FALL BACK TO BRACKET: rank-1 coverage is used as the "
                "OPTIMISTIC ceiling (upper bound), not a point estimate of compliant accept."),
            "sanity_crosschecks": sanity,
        },
        "brackets": brackets,
        "lambda_hat": lam_hat,
        "composition": {
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONS, "target_official": TARGET,
            "clear500_bar_et_tau1": bar1, "clear500_bar_et_tau_cons": clear_bar(TAU_CONS),
            "bench_tokens": BENCH_TOKENS,
            "kanna114_divergence": KANNA114_DIVERGENCE,
            "kanna122_offshelf_overhead_nonworking_ref": KANNA122_OFFSHELF_OVERHEAD,
        },
        "verdict": verdict,
        "handoff_line": handoff,
    }


def _verdict(ceiling_clears: bool, floor_clears: bool, max_ovh_pct: float) -> str:
    if ceiling_clears and not floor_clears:
        return "COMPLIANT-CEILING-CLEARS-FLOOR-MISSES"   # a token-identical 500-path EXISTS iff kernel-overhead < budget
    if ceiling_clears and floor_clears:
        return "COMPLIANT-BRACKET-CLEARS-BOTH"
    return "COMPLIANT-CEILING-MISSES-500"                # spec 500-path existentially blocked under #192


def _handoff(brackets: dict, lam_hat: float, tps_ceil: float, tps_floor: float,
             clears: bool, max_ovh: float) -> str:
    bb = brackets["both_bugs"]
    verb = "CLEARS" if clears else "MISSES"
    return (
        f"COMPLIANT-SPEC E[T] CEILING (Issue #192 lane-a, batch-invariant verify): the "
        f"token-identical-verify E[T] is BRACKETED [{bb['et_compliant_floor']:.4f}, "
        f"{bb['et_compliant_ceiling']:.4f}] (both-bugs) → TPS [{tps_floor:.1f}, {tps_ceil:.1f}]. "
        f"The optimistic ceiling (full self-KV recovery + zero spurious rejects) {verb} 500 "
        f"at the current step and AFFORDS up to {max_ovh:.1f}% verify-step inflation as the "
        f"kernel-dev budget — vs kanna #122's off-the-shelf +51.78% (a NON-working ref that "
        f"blows the budget ~{KANNA122_OFFSHELF_OVERHEAD * 100 / max(max_ovh, 1e-9):.1f}×). The "
        f"floor (realistic λ̂={lam_hat:.3f}, batch-variant) MISSES at {tps_floor:.1f}. HONEST "
        f"SCOPE: the rankprobe's true token is the batch-VARIANT argmax, so this is an E[T] "
        f"BRACKET, not a measured compliant accept; the ceiling conflates full self-KV "
        f"recovery (λ=1) with batch-invariance, so it is an UPPER bound. Pairs with lawine "
        f"#196 (lane-b non-spec PUBLIC floor, empirical): if the lane-a ceiling clears 500 at "
        f"a feasible kernel overhead, a compliant 500-path EXISTS (motivates the batch-"
        f"invariant kernel-dev); the realistic-λ̂ floor says it is NOT free. NOT a launch. "
        f"NOT open2."
    )


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #178/#184; never fatal).
# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "result") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


def _print_report(syn: dict) -> None:
    rc, br, st = syn["rank1_coverage"], syn["brackets"], syn["self_test"]
    hd, comp = syn["headline"], syn["composition"]
    print("\n" + "=" * 92, flush=True)
    print("COMPLIANT-SPEC E[T] CEILING (PR #199, wirbel) — Issue #192 lane-a, CPU-only", flush=True)
    print("=" * 92, flush=True)
    print(f"  (1) RANK-1 COVERAGE (in-scope shard {rc['shard']}, n={rc['n_records']}, "
          f"align_bad={rc['n_align_bad']})", flush=True)
    print(f"      q_compliant[1..{rc['measured_depths']}] = "
          f"{[round(x, 4) if x is not None else None for x in rc['q_compliant']]}", flush=True)
    print(f"      top1={rc['rank1_coverage_top1']:.4f}  pooled-rate="
          f"{rc['rank1_coverage_rate_pooled']:.4f}  E[fd]={rc['mean_accepted_run_length_fd']:.4f}",
          flush=True)
    print(f"      data exposes batch-invariant cleanly = "
          f"{rc['data_exposes_batch_invariant_cleanly']}  → FALL BACK TO BRACKET", flush=True)
    print(f"      sanity: |Δ| vs published z6 ladder={rc['sanity_crosschecks']['max_abs_diff_vs_published_z6wi4z4v']:.4f}  "
          f"vs accept76 deployed={rc['sanity_crosschecks']['max_abs_diff_vs_accept76_deployed']:.4f}", flush=True)
    print("-" * 92, flush=True)
    print(f"  (2) E[T] BRACKET  (clear-500 bar E[T]={comp['clear500_bar_et_tau1']:.4f} @ τ=1)", flush=True)
    for topo in ("descent_only", "both_bugs"):
        b = br[topo]
        fp = b["floor_propagate"]["tau_central_1p0"]
        cp = b["ceiling_propagate"]["tau_central_1p0"]
        print(f"      {topo:<13} E[T] [{b['et_compliant_floor']:.4f}, {b['et_compliant_ceiling']:.4f}]"
              f"  ordered={b['bracket_ordered_floor_le_ceiling']}", flush=True)
        print(f"                    TPS  [{fp['official_tps']:.1f} (floor, clears={fp['clears_500']}), "
              f"{cp['official_tps']:.1f} (ceiling, clears={cp['clears_500']})]  "
              f"kernel-budget={cp['max_kernel_overhead_pct_to_clear_500']:.1f}%", flush=True)
        print(f"                    anchor full-cov E[T]={b['anchor_full_coverage']['et']:.6f} "
              f"(target {b['anchor_full_coverage']['target']:.4f}, "
              f"reproduces={b['anchor_full_coverage']['reproduces']})", flush=True)
    print("-" * 92, flush=True)
    print(f"  (3) HEADLINE (both-bugs, τ=1, zero-kernel-overhead)", flush=True)
    print(f"      compliant_spec_tps_ceiling = {hd['compliant_spec_tps_ceiling']:.2f}  "
          f"(floor {hd['compliant_spec_tps_floor']:.2f})", flush=True)
    print(f"      compliant_spec_clears_500  = {hd['compliant_spec_clears_500']}  "
          f"(floor clears={hd['compliant_spec_floor_clears_500']})", flush=True)
    print(f"      max_kernel_overhead_pct_to_clear_500 = "
          f"{hd['max_kernel_overhead_pct_to_clear_500']:.2f}%  "
          f"(off-the-shelf kanna#122 ref +{comp['kanna122_offshelf_overhead_nonworking_ref']*100:.1f}%)",
          flush=True)
    print("-" * 92, flush=True)
    print(f"  (4) PRIMARY compliant_spec_et_self_test_passes = "
          f"{st['compliant_spec_et_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 92, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[compliant-spec-et] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    run = init_wandb_run(
        job_type="compliant-spec-et-ceiling",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["compliant-spec-et-ceiling", "issue-192", "batch-invariant", "validity-gate",
              "et-bracket"],
        config={
            "K_cal": K_CAL, "step": STEP, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONS, "n_nodes": CTX["n_nodes"],
            "imported_descent_5p0564": IMPORTED_DESCENT_5p0564,
            "imported_both_bugs_5p2070": IMPORTED_BOTH_BUGS_5p2070,
            "lambda_hat": CTX["lam_hat"], "bench_tokens": BENCH_TOKENS,
            "rankprobe_shard": syn["rank1_coverage"]["shard"],
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[compliant-spec-et] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st, hd, rc = syn["self_test"], syn["headline"], syn["rank1_coverage"]
    bb, des = syn["brackets"]["both_bugs"], syn["brackets"]["descent_only"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "compliant_spec_et_self_test_passes":
            int(bool(st["compliant_spec_et_self_test_passes"])),
        "compliant_spec_tps_ceiling": hd["compliant_spec_tps_ceiling"],
        # headline bracket
        "compliant_spec_tps_floor": hd["compliant_spec_tps_floor"],
        "compliant_spec_clears_500": int(bool(hd["compliant_spec_clears_500"])),
        "compliant_spec_floor_clears_500": int(bool(hd["compliant_spec_floor_clears_500"])),
        "max_kernel_overhead_pct_to_clear_500": hd["max_kernel_overhead_pct_to_clear_500"],
        # both-bugs bracket E[T]
        "et_compliant_floor_both_bugs": bb["et_compliant_floor"],
        "et_compliant_ceiling_both_bugs": bb["et_compliant_ceiling"],
        "bracket_width_both_bugs": bb["bracket_width_E_T"],
        "ceiling_sigma_L_both_bugs": bb["ceiling_sigma_L"],
        "ceiling_lcb_tps_both_bugs": bb["ceiling_finite_sample_ci_tau1"]["ci_lower_tps"],
        # descent-only bracket E[T]
        "et_compliant_floor_descent_only": des["et_compliant_floor"],
        "et_compliant_ceiling_descent_only": des["et_compliant_ceiling"],
        # rank-1 coverage
        "rank1_coverage_top1": rc["rank1_coverage_top1"],
        "rank1_coverage_rate_pooled": rc["rank1_coverage_rate_pooled"],
        "rankprobe_n_records": rc["n_records"],
        "data_exposes_batch_invariant_cleanly": int(bool(rc["data_exposes_batch_invariant_cleanly"])),
        # composition bars
        "clear500_bar_et_tau1": syn["composition"]["clear500_bar_et_tau1"],
        "lambda_hat": syn["lambda_hat"],
        "verdict_ceiling_clears_floor_misses":
            int(syn["verdict"] == "COMPLIANT-CEILING-CLEARS-FLOOR-MISSES"),
        "verdict_ceiling_misses": int(syn["verdict"] == "COMPLIANT-CEILING-MISSES-500"),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="compliant_spec_et_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[compliant-spec-et] wandb logged: {summary}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--shard", type=Path, default=RANKPROBE_SHARD,
                    help="in-scope PR#86 rankprobe shard (read-only)")
    ap.add_argument("--max-records", type=int, default=None, help="debug: cap records parsed")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="compliant-spec-et-ceiling")
    args = ap.parse_args(argv)

    syn = synthesize(args.shard, args.max_records)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 199, "agent": "wirbel",
        "kind": "compliant-spec-et-ceiling", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["e_nan_clean"] = not nan_paths
    syn["self_test"]["compliant_spec_et_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    if nan_paths:
        print(f"[compliant-spec-et] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compliant_spec_et_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[compliant-spec-et] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["compliant_spec_et_self_test_passes"] and payload["nan_clean"]
        print(f"[compliant-spec-et] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
