#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #313 (student lawine) -- Pre-register the read-only probe that flips #309's YELLOW.

WHAT THIS CARD DOES (0-GPU, 0-TPS, no served-file change, no HF Job)
--------------------------------------------------------------------
lawine #309 (7tkn4d9x, MERGED) showed the M=8 verify-tree relaxes the EAGLE-3 drafter demand from raw
a1 -> 0.9213 (denken #304's hard line) down to raw a1 -> 0.7731 -- but YELLOW. The single missing input
is whether the {2,21,39}-fusion draft inherits the deployed LINEAR spine's rank-2+ coverage
(cov4=0.6532, measured by wirbel #79 z6wi4z4v). If the fusion draft's rank-1 misses fall FURTHER down
the rank list (frac_true_beyond_top4 > the spine's 0.347), cov_W drops and the relaxed demand rises
back toward 0.92.

This card BUILDS and PRE-REGISTERS the read-only probe that resolves the YELLOW. It is pure analysis +
a CPU dry-run of the BANKED #79 harness logic. It does NOT re-run #79's GPU probe and it does NOT
require a fusion checkpoint (none exists). Concretely it:

  1. Re-implements #309's salvage operator c1_eff(a1)=a1+(1-a1)*cov_W AND its inverse, and reproduces
     #309's banked a1-required curve (W2 0.8651 / W3 0.8164 / W4 0.7731; zero-salvage 0.9213==#304) to
     <= 1e-6 against the on-disk #309 artifact.
  2. Derives the DECISION THRESHOLD: solves the fusion cov_W (and implied frac_true_beyond_topW) at
     which the salvaged raw-a1 demand exactly equals #304's 0.9213 (RED edge) and a chosen trainable
     band edge 0.80 (GREEN edge), per width W in {2,3,4}. Output: the falsifiable rule.
  3. Specifies the read-only probe harness (scripts/profiler/rank_coverage.py RANKPROBE_W interface)
     and runs a CPU DRY-RUN on #79's BANKED pooled rank histogram through the REAL analyze() to
     reproduce cov4=0.6532 -- proving the harness logic is checkpoint-ready.
  4. Carries the honest caveat: the fusion cov_W is UNMEASURED; the linear-spine cov_W is the best-case
     transfer. This PRE-REGISTERS the test, it does not run it.

THE SALVAGE OPERATOR (the recovered estimand, #309)
---------------------------------------------------
At a verify tree of branch width W, the true token is accepted at position-1 if it is the draft rank-1
token (prob a1) OR it is in the rank-2..W branch when rank-1 missed (prob (1-a1)*cov_W), where
cov_W = P(true token at draft rank 2..W | rank-1 miss) is wirbel #79's measured cumulative coverage.
So c1_eff(a1) = a1 + (1-a1)*cov_W, and the raw draft a1 needed to reach an EFFECTIVE target T is
a1 = (T - cov_W)/(1 - cov_W). The decision threshold inverts THAT once more: the cov_W at which the
required raw a1 equals a band edge d is cov_W* = (T - d)/(1 - d), with frac_true_beyond_topW* = 1 -
cov_W* (the #79 partition cov_W + frac_beyond_topW == 1 holds exactly on the banked record).

LOCAL CPU-only analytic + harness-logic card. No GPU / vLLM / model forward / training / HF Job /
submission / served-file change. NOT a launch. BASELINE stays 481.53 (0 TPS). Greedy/PPL untouched.

PRIMARY metric  fusion_rankcov_probe_self_test_passes
TEST    metrics frac_beyond_top4_threshold_for_green (float)  +  probe_reproduces_linear_cov4 (bool)

Reproduce:
    cd target/ && .venv/bin/python \\
        research/validity/eagle3_fusion_rankcov_probe/eagle3_fusion_rankcov_probe.py \\
        --self-test --wandb_group eagle3-rankcov-probe --wandb_name lawine/eagle3-rankcov-probe
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# Imported fleet anchors (cited EXACTLY, UNCHANGED; this card re-derives none).
# All three are committed on the advisor branch; the self-test re-loads the
# on-disk artifacts and cross-checks these literals to <= 1e-6 (no silent drift).
# --------------------------------------------------------------------------- #
# denken #304 (dtf1ouml): the no-salvage hard line == the EFFECTIVE per-position target T.
A1_REQUIRED_611_NOSALVAGE = 0.9213011665456927   # T: effective a1 for E[T]=6.11 (#304, no tree salvage)
A1_DEPLOYED = 0.72925                             # deployed raw a1 cliff (#304)

# lawine #309 (7tkn4d9x): salvage operator + inverted raw-a1 demand by tree width.
BANKED_309_A1_REQ_BY_W = {2: 0.8651251447964469, 3: 0.8163598752994596, 4: 0.7730729805683441}

# wirbel #79 (z6wi4z4v): MEASURED rank-coverage on the deployed LINEAR spine (16,524 records,
# align_bad=0). cov_W = cumulative P(true token caught at draft rank <= W | rank-1 miss).
COV_W = {2: 0.4165047789261015, 3: 0.5714507731758489, 4: 0.6531976066516435}
FRAC_TRUE_BEYOND_TOP4 = 0.3468023933483565       # 1 - cov4 (irreducible width-4 miss mass, linear spine)
# pooled rank-of-true histogram at first divergence (rank 0 == "beyond top-W"). Sum == n_divergences.
RANK_FD_HIST_POOLED = {0: 4463, 2: 5360, 3: 1994, 4: 1052}
N_DIVERGENCES = 12869

# Banked artifacts (committed; read-only validation targets for the self-test).
PR79_RESULTS = REPO_ROOT / "research" / "rank_coverage" / "rank_coverage_results.json"
PR309_RESULTS = (REPO_ROOT / "research" / "validity" / "eagle3_tree_salvage_a1"
                 / "eagle3_tree_salvage_a1_results.json")
RANKPROBE = REPO_ROOT / "scripts" / "profiler" / "rank_coverage.py"

# Band edges on the RAW-a1 demand axis.
TRAINABLE_BAND_EDGE = 0.80                        # PR-chosen GREEN edge (raw a1 demand "trainable" cap)
DEPLOYED_PLUS_10PCT = A1_DEPLOYED * 1.10          # 0.802175 secondary GREEN edge (#309 verdict anchor)
HARD_LINE_304 = A1_REQUIRED_611_NOSALVAGE         # RED edge

PRIMARY_W = 4
WIDTHS = (2, 3, 4)
# the a1 grid #309 banked its effective-a1 curve over (reproduced here to <= 1e-6).
A1_SWEEP = [0.65, 0.70, 0.73, 0.75, 0.80, 0.85, 0.90, 0.92]

TOL = 1e-9
TOL_REPRO = 1e-6


# --------------------------------------------------------------------------- #
# The salvage operator + its two inversions (independently re-implemented).
# --------------------------------------------------------------------------- #
def tree_recovered(base: float, cov: float) -> float:
    """Salvage operator: effective position-1 acceptance after a width-W verify tree.

    true token accepted if rank-1 (prob base) OR in the rank-2..W branch on a rank-1 miss
    (prob (1-base)*cov). Clipped to a valid acceptance [0,1].
    """
    return min(1.0, max(0.0, base + (1.0 - base) * cov))


def a1_draft_for_effective(a_eff_target: float, cov: float) -> float:
    """Invert the salvage in a1: raw draft a1 so that tree_recovered(a1, cov) == a_eff_target.

    a1 = (a_eff_target - cov)/(1 - cov). cov->0 returns a_eff_target (no salvage, reproduces #304);
    cov->1 returns 0 (perfect coverage needs no draft acceptance). Clipped to [0,1].
    """
    if cov >= 1.0 - TOL:
        return 0.0
    return min(1.0, max(0.0, (a_eff_target - cov) / (1.0 - cov)))


def cov_for_demand(a1_demand: float, a_eff_target: float = HARD_LINE_304) -> float:
    """Decision threshold: the cov_W at which the salvaged raw-a1 demand equals ``a1_demand``.

    Inverts a1_draft_for_effective in cov: solving (a_eff_target - cov)/(1 - cov) == a1_demand gives
    cov* = (a_eff_target - a1_demand)/(1 - a1_demand). a1_demand == a_eff_target -> cov* = 0 (RED: full
    demand only at zero salvage). a1_demand < a_eff_target -> cov* > 0 (the coverage bar to clear).
    Independent of W: the salvage inverse depends only on cov and the fixed effective target T.
    """
    if a1_demand >= 1.0 - TOL:
        return 0.0
    return (a_eff_target - a1_demand) / (1.0 - a1_demand)


def frac_beyond_from_hist(hist: dict[int, int], w: int) -> float:
    """frac_true_beyond_topW from the pooled rank histogram (rank 0 == beyond, ranks > W also beyond)."""
    n_div = sum(hist.values())
    beyond = sum(c for r, c in hist.items() if r == 0 or r > w)
    return beyond / n_div


def cov_from_hist(hist: dict[int, int], w: int) -> float:
    """cov_W = #(2 <= rank <= W) / n_div from the pooled rank histogram (mirrors #79 analyze())."""
    n_div = sum(hist.values())
    caught = sum(c for r, c in hist.items() if 2 <= r <= w)
    return caught / n_div


# --------------------------------------------------------------------------- #
# CPU dry-run of #79's harness logic on the BANKED pooled histogram.
# --------------------------------------------------------------------------- #
def dryrun_probe_on_banked(hist: dict[int, int]) -> dict[str, Any]:
    """Reconstruct minimal divergence records from the banked pooled histogram and push them through
    the REAL #79 analyze() (scripts/profiler/rank_coverage.py) to reproduce cov2/cov3/cov4 + frac.

    Proves the committed harness logic reproduces #79's banked coverage with ZERO GPU and no fusion
    checkpoint, so the probe is checkpoint-ready. The raw #79 record shard is gitignored / not on disk;
    the pooled histogram (committed in rank_coverage_results.json) is a sufficient statistic for cov_W.
    """
    out: dict[str, Any] = {"used_real_analyze": False, "import_error": None}
    try:
        from scripts.profiler.rank_coverage import analyze  # GPU-free import (stdlib-only harness/paths)
        out["used_real_analyze"] = True
    except Exception as exc:  # noqa: BLE001 -- fall back to the local mirror, flag it loudly
        out["import_error"] = repr(exc)
        analyze = None  # type: ignore[assignment]

    with tempfile.TemporaryDirectory() as td:
        records_path = Path(td) / "rankprobe_records.jsonl"
        shard = Path(td) / "rankprobe_records.jsonl.0"
        # fd=0 < n=7 => each row is a first-divergence (rank-1 miss); rank_fd carries the true-token rank.
        with shard.open("w") as fh:
            for rank, count in sorted(hist.items()):
                row = json.dumps({"n": 7, "fd": 0, "rank_fd": int(rank), "align": True})
                for _ in range(int(count)):
                    fh.write(row + "\n")
        if analyze is not None:
            a = analyze(records_path, W=4, max_depth=7)
            cov = {int(k): v for k, v in a["cumulative_coverage"].items()}
            out["analyze_cumulative_coverage"] = a["cumulative_coverage"]
            out["analyze_frac_true_beyond_topW"] = a["frac_true_beyond_topW"]
            out["analyze_n_divergences"] = a["n_divergences"]
        else:
            cov = {w: cov_from_hist(hist, w) for w in WIDTHS}
            out["analyze_cumulative_coverage"] = {str(w): cov[w] for w in WIDTHS}
            out["analyze_frac_true_beyond_topW"] = frac_beyond_from_hist(hist, 4)
            out["analyze_n_divergences"] = sum(hist.values())

    out["reproduced_cov_W"] = {str(w): cov[w] for w in WIDTHS}
    out["cov4_abs_diff_vs_banked"] = abs(cov[4] - COV_W[4])
    out["cov3_abs_diff_vs_banked"] = abs(cov[3] - COV_W[3])
    out["cov2_abs_diff_vs_banked"] = abs(cov[2] - COV_W[2])
    out["frac_abs_diff_vs_banked"] = abs(out["analyze_frac_true_beyond_topW"] - FRAC_TRUE_BEYOND_TOP4)
    out["probe_reproduces_linear_cov4"] = bool(out["cov4_abs_diff_vs_banked"] <= TOL_REPRO
                                               and out["used_real_analyze"])
    return out


# --------------------------------------------------------------------------- #
# Load banked artifacts (read-only) for the <=1e-6 cross-checks.
# --------------------------------------------------------------------------- #
def load_banked() -> dict[str, Any]:
    d79 = json.loads(PR79_RESULTS.read_text()) if PR79_RESULTS.exists() else {}
    d309 = json.loads(PR309_RESULTS.read_text()) if PR309_RESULTS.exists() else {}
    return {"pr79": d79, "pr309": d309}


# --------------------------------------------------------------------------- #
# Synthesis (steps 1-4).
# --------------------------------------------------------------------------- #
def synthesize(banked: dict[str, Any]) -> dict[str, Any]:
    d79 = banked.get("pr79", {})
    d309 = banked.get("pr309", {})

    # ---- STEP 1: reproduce #309's salvage operator + inverse to <= 1e-6. ---- #
    # (a) inverse: raw a1 demanded after salvage, per width, vs the banked #309 curve.
    a1_req_by_w = {w: a1_draft_for_effective(HARD_LINE_304, COV_W[w]) for w in WIDTHS}
    banked_309_req = ((d309.get("synthesis", {}) or {}).get("step3_invert", {})
                      or {}).get("a1_draft_required_by_W", {}) or {}
    # JSON keys are strings; coerce to compare.
    banked_309_req = {int(k): float(v) for k, v in banked_309_req.items()} or BANKED_309_A1_REQ_BY_W
    a1_req_max_abs_diff = max(abs(a1_req_by_w[w] - banked_309_req[w]) for w in WIDTHS)

    # (b) operator: c1_eff over the banked a1 grid, per width, vs the banked #309 curve.
    curves = {f"W{w}": [{"a1_draft": a1, "c1_eff": tree_recovered(a1, COV_W[w])} for a1 in A1_SWEEP]
              for w in WIDTHS}
    banked_309_curve = ((d309.get("synthesis", {}) or {}).get("step2_fusion_curve", {})
                        or {}).get("effective_a1_curves", {}) or {}
    curve_max_abs_diff = 0.0
    if banked_309_curve:
        for w in WIDTHS:
            mine = curves[f"W{w}"]
            theirs = banked_309_curve.get(f"W{w}", [])
            for pm, pt in zip(mine, theirs):
                curve_max_abs_diff = max(curve_max_abs_diff, abs(pm["c1_eff"] - float(pt["c1_eff"])))

    # (c) zero-salvage reconciles #304 exactly.
    a1_req_zero_salvage = a1_draft_for_effective(HARD_LINE_304, 0.0)
    zero_salvage_abs_diff = abs(a1_req_zero_salvage - HARD_LINE_304)

    step1 = {
        "a1_draft_required_by_W": a1_req_by_w,
        "banked_309_a1_draft_required_by_W": banked_309_req,
        "a1_req_max_abs_diff_vs_309": a1_req_max_abs_diff,
        "effective_a1_curves": curves,
        "curve_max_abs_diff_vs_309": curve_max_abs_diff,
        "a1_req_zero_salvage": a1_req_zero_salvage,
        "zero_salvage_abs_diff_vs_304": zero_salvage_abs_diff,
        "reproduces_309_curve": bool(a1_req_max_abs_diff <= TOL_REPRO
                                     and curve_max_abs_diff <= TOL_REPRO
                                     and zero_salvage_abs_diff <= TOL_REPRO),
        "note": ("salvage operator c1_eff=a1+(1-a1)*cov_W and its a1-inverse reproduce #309's banked "
                 "a1-required curve (W2 {:.4f}/W3 {:.4f}/W4 {:.4f}) and effective-a1 grid to <= 1e-6; "
                 "zero salvage (cov=0) reproduces #304's 0.9213 exactly."
                 .format(a1_req_by_w[2], a1_req_by_w[3], a1_req_by_w[4])),
    }

    # ---- STEP 2: derive the DECISION THRESHOLD (cov_W* / frac*). ------------- #
    # The threshold is W-INVARIANT on the cov axis: it depends only on the band edge and the fixed T.
    cov_star_green = cov_for_demand(TRAINABLE_BAND_EDGE)          # GREEN edge (raw a1 demand 0.80)
    frac_star_green = 1.0 - cov_star_green
    cov_star_red = cov_for_demand(HARD_LINE_304)                 # RED edge (raw a1 demand 0.9213)
    frac_star_red = 1.0 - cov_star_red
    cov_star_green_p10 = cov_for_demand(DEPLOYED_PLUS_10PCT)     # secondary GREEN edge (deployed*1.10)
    frac_star_green_p10 = 1.0 - cov_star_green_p10

    # per width: banked coverage, implied frac, demand at banked cov, GREEN margin, verdict.
    per_w = {}
    for w in WIDTHS:
        cov_banked = COV_W[w]
        frac_banked = frac_beyond_from_hist(RANK_FD_HIST_POOLED, w)   # == 1 - cov_banked (partition)
        demand = a1_draft_for_effective(HARD_LINE_304, cov_banked)
        green_margin_cov = cov_banked - cov_star_green               # >0 => clears the GREEN bar
        if demand <= TRAINABLE_BAND_EDGE + TOL:
            verdict = "GREEN"
        elif demand >= HARD_LINE_304 - TOL:
            verdict = "RED"
        else:
            verdict = "YELLOW"
        per_w[w] = {
            "cov_W_banked_linear": cov_banked,
            "frac_true_beyond_topW_banked_linear": frac_banked,
            "cov_W_star_green": cov_star_green,
            "frac_true_beyond_topW_star_green": frac_star_green,
            "cov_W_star_red": cov_star_red,
            "frac_true_beyond_topW_star_red": frac_star_red,
            "salvaged_raw_a1_demand_at_banked_cov": demand,
            "green_margin_cov": green_margin_cov,
            "green_margin_frac": frac_star_green - frac_banked,
            "linear_spine_verdict": verdict,
        }

    # headroom at the primary W (how far the fusion cov4 may degrade before leaving GREEN).
    frac_headroom_abs = frac_star_green - FRAC_TRUE_BEYOND_TOP4
    frac_headroom_rel = frac_headroom_abs / FRAC_TRUE_BEYOND_TOP4
    cov_headroom_abs = COV_W[PRIMARY_W] - cov_star_green
    cov_headroom_rel = cov_headroom_abs / COV_W[PRIMARY_W]

    # monotonicity (self-test 5b): banked cov increasing, demand decreasing, GREEN margin increasing.
    cov_monotone = COV_W[2] < COV_W[3] < COV_W[4]
    demand_monotone = per_w[2]["salvaged_raw_a1_demand_at_banked_cov"] > \
        per_w[3]["salvaged_raw_a1_demand_at_banked_cov"] > \
        per_w[4]["salvaged_raw_a1_demand_at_banked_cov"]
    green_margin_monotone = per_w[2]["green_margin_cov"] < per_w[3]["green_margin_cov"] \
        < per_w[4]["green_margin_cov"]

    falsifiable_rule = (
        "Measure the FUSION draft's frac_true_beyond_top4 (share of rank-1 misses whose true token is "
        "beyond draft rank 4) via the #79 RANKPROBE_W=4 probe. GREEN (salvaged raw-a1 demand < {edge}, "
        "trainable) iff frac_true_beyond_top4 < {fg:.4f}  (equiv. cov4 > {cg:.4f}). RED (#304's full "
        "{T:.4f} demand reinstated) only at frac_true_beyond_top4 = {fr:.1f} i.e. cov4 = {cr:.1f} -- "
        "UNREACHABLE for any positive rank-2+ transfer. The deployed LINEAR spine sits at "
        "frac_true_beyond_top4 = {fl:.4f} (cov4 = {cl:.4f}) -> GREEN with frac-headroom {hf:.4f} "
        "(cov4 may degrade {cr_rel:.1f}% rel before YELLOW). 0.80<=demand<0.9213 (i.e. {fg:.4f}<frac<"
        "1.0) is YELLOW: trainable but above the chosen band edge."
        .format(edge=TRAINABLE_BAND_EDGE, fg=frac_star_green, cg=cov_star_green, T=HARD_LINE_304,
                fr=frac_star_red, cr=cov_star_red, fl=FRAC_TRUE_BEYOND_TOP4, cl=COV_W[PRIMARY_W],
                hf=frac_headroom_abs, cr_rel=100.0 * cov_headroom_rel))

    step2 = {
        "effective_target_T": HARD_LINE_304,
        "trainable_band_edge_green": TRAINABLE_BAND_EDGE,
        "hard_line_edge_red": HARD_LINE_304,
        "cov_W_star_green": cov_star_green,
        "frac_true_beyond_topW_star_green": frac_star_green,
        "cov_W_star_red": cov_star_red,
        "frac_true_beyond_topW_star_red": frac_star_red,
        "secondary_green_edge_deployed_plus_10pct": DEPLOYED_PLUS_10PCT,
        "cov_W_star_green_deployed_plus_10pct": cov_star_green_p10,
        "frac_true_beyond_topW_star_green_deployed_plus_10pct": frac_star_green_p10,
        "threshold_is_W_invariant_on_cov_axis": True,
        "per_width": per_w,
        "primary_W": PRIMARY_W,
        "frac_headroom_abs_primary": frac_headroom_abs,
        "frac_headroom_rel_primary": frac_headroom_rel,
        "cov_headroom_abs_primary": cov_headroom_abs,
        "cov_headroom_rel_primary": cov_headroom_rel,
        "banked_cov_monotone_in_W": bool(cov_monotone),
        "demand_monotone_decreasing_in_W": bool(demand_monotone),
        "green_margin_monotone_increasing_in_W": bool(green_margin_monotone),
        "falsifiable_rule": falsifiable_rule,
        "note": ("decision threshold cov_W*=(T-d)/(1-d), frac*=1-cov_W*, W-invariant on the cov axis. "
                 "GREEN edge d=0.80 -> cov4*={:.4f}/frac4*={:.4f}; RED edge d=0.9213 -> cov*=0/frac*=1 "
                 "(unreachable for positive transfer). Only W=4's banked linear cov ({:.4f}) clears the "
                 "GREEN bar; W=2/W=3 sit YELLOW. The GREEN/YELLOW decision reduces to ONE coverage bar."
                 .format(cov_star_green, frac_star_green, COV_W[PRIMARY_W])),
    }

    # ---- STEP 3: probe-harness spec + CPU dry-run on banked records. -------- #
    dry = dryrun_probe_on_banked(RANK_FD_HIST_POOLED)
    probe_spec = {
        "script": "scripts/profiler/rank_coverage.py",
        "estimand": ("first-divergence on-path: at the first depth where the draft rank-1 token misses "
                     "the target greedy argmax, the prefix is the true continuation; read the rank of "
                     "the true token in the drafter's top-W there. cov_W = #(2<=rank_fd<=W)/n_div; "
                     "frac_true_beyond_topW = #(rank_fd==0)/n_div; cov_W + frac_beyond_topW == 1."),
        "env_interface": {
            "RANKPROBE_ENABLE": "1 -> load rankprobe_patch in the scratch sitecustomize",
            "RANKPROBE_OUTPUT": "absolute path for per-process record shards {OUTPUT}.{pid}",
            "RANKPROBE_W": "tree branch width to score coverage up to (banked run W=4)",
            "RANKPROBE_LOGITS": "1 -> also capture drafter top-W probs+entropy (PR #86; optional)",
            "LOOPGRAPH_WARMUP_CALLS": "huge -> never capture the CUDA graph, so eager base_propose "
                                      "(the per-depth greedy override) fires every draft depth",
        },
        "inputs_needed_from_fusion_draft": ("per-token first-divergence records {n, fd, rank_fd, align}: "
                                            "n draft tokens proposed, fd first-divergence depth, rank_fd "
                                            "rank of the true token in the fusion draft's top-W at fd "
                                            "(0 == beyond top-W), align == byte-identity self-check."),
        "outputs": ["cumulative_coverage cov_W", "frac_true_beyond_topW", "rho_marginal",
                    "conditional_rank1_acceptance_q", "n_divergences"],
        "zero_served_file_change_invariant": ("runs on a SCRATCH copy of the submission; served files "
                                              "stay byte-identical (only ADD logging + force eager "
                                              "base_propose). Each record's align flag self-checks that "
                                              "the emitted draft chain is byte-identical to production. "
                                              "No served-file edit, no HF Job, no submission launch."),
        "checkpoint_ready": ("point RANKPROBE at a fusion EAGLE-3 submission once a draft head exists; "
                             "analyze() returns the fusion frac_true_beyond_top4 to drop into the "
                             "step-2 rule. Zero served-file change; no extra GPU beyond the single "
                             "assigned student GPU; greedy/PPL untouched."),
        "cpu_dryrun": dry,
    }

    # ---- STEP 4: honest caveat (the YELLOW is pre-registered, not run). ----- #
    step4 = {
        "fusion_cov_W_measured": False,
        "fusion_checkpoint_exists": False,
        "linear_spine_is_best_case_transfer": True,
        "caveat": ("the GREEN/RED thresholds are EXACT arithmetic on the fixed salvage operator and the "
                   "fixed effective target T=0.9213. The FUSION draft's cov_W is UNMEASURED -- no "
                   "fusion EAGLE-3 checkpoint exists -- so this card PRE-REGISTERS the test, it does not "
                   "run it. The imported cov_W is wirbel #79's deployed LINEAR-spine coverage, the "
                   "best-case transfer: a {2,21,39}-fusion draft has a distinct rank-2+ candidate "
                   "distribution; if its rank-1 misses fall further down the rank list its "
                   "frac_true_beyond_top4 rises and the verdict can move GREEN->YELLOW. RED stays "
                   "unreachable for any positive transfer."),
        "what_flips_the_verdict": ("run the #79 RANKPROBE_W=4 probe on the fusion draft head; compare "
                                   "its frac_true_beyond_top4 to the {:.4f} GREEN threshold."
                                   .format(frac_star_green)),
    }

    return {
        "step1_reproduce_309": step1,
        "step2_decision_threshold": step2,
        "step3_probe_spec_and_dryrun": probe_spec,
        "step4_caveat": step4,
        "test_metrics": {
            "frac_beyond_top4_threshold_for_green": frac_star_green,
            "cov4_threshold_for_green": cov_star_green,
            "probe_reproduces_linear_cov4": dry["probe_reproduces_linear_cov4"],
        },
        "imported": {
            "T_a1_required_611_nosalvage_304": HARD_LINE_304,
            "a1_deployed_304": A1_DEPLOYED,
            "cov_W_measured_79": COV_W,
            "frac_true_beyond_top4_79": FRAC_TRUE_BEYOND_TOP4,
            "rank_fd_hist_pooled_79": RANK_FD_HIST_POOLED,
            "n_divergences_79": N_DIVERGENCES,
            "banked_309_a1_req_by_W": BANKED_309_A1_REQ_BY_W,
            "provenance": ("salvage operator + a1-inverse lawine #309 (7tkn4d9x); measured cov_W + "
                           "frac_true_beyond_top4 + pooled rank histogram wirbel #79 (z6wi4z4v); "
                           "effective target T=0.9213 denken #304 (dtf1ouml). Decision threshold and "
                           "probe pre-registration are this card (#313)."),
        },
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any], banked: dict[str, Any]) -> dict[str, Any]:
    s1 = syn["step1_reproduce_309"]
    s2 = syn["step2_decision_threshold"]
    s3 = syn["step3_probe_spec_and_dryrun"]
    s4 = syn["step4_caveat"]
    c: dict[str, bool] = {}

    # (a) salvage operator + inverse reproduce #309's curve to <= 1e-6.
    c["01_inverse_reproduces_309_a1req_by_W"] = bool(s1["a1_req_max_abs_diff_vs_309"] <= TOL_REPRO)
    c["02_operator_reproduces_309_curve"] = bool(s1["curve_max_abs_diff_vs_309"] <= TOL_REPRO)
    c["03_zero_salvage_reconciles_304"] = bool(s1["zero_salvage_abs_diff_vs_304"] <= TOL_REPRO)

    # (b) thresholds solved + monotone in W.
    # GREEN threshold round-trips: plugging cov*_green back through the salvage inverse yields 0.80.
    rt_green = a1_draft_for_effective(HARD_LINE_304, s2["cov_W_star_green"])
    c["04_green_threshold_roundtrips_to_edge"] = bool(abs(rt_green - TRAINABLE_BAND_EDGE) <= TOL_REPRO)
    # RED threshold is exactly zero coverage / unit frac.
    c["05_red_threshold_is_zero_cov_unit_frac"] = bool(
        abs(s2["cov_W_star_red"] - 0.0) <= TOL and abs(s2["frac_true_beyond_topW_star_red"] - 1.0) <= TOL)
    # the #79 partition cov_W + frac_beyond_topW == 1 holds per width on the banked histogram.
    c["06_frac_partition_holds_per_W"] = bool(all(
        abs(COV_W[w] + frac_beyond_from_hist(RANK_FD_HIST_POOLED, w) - 1.0) <= TOL_REPRO for w in WIDTHS))
    c["07_banked_cov_monotone_in_W"] = bool(s2["banked_cov_monotone_in_W"])
    c["08_demand_monotone_decreasing_in_W"] = bool(s2["demand_monotone_decreasing_in_W"])
    c["09_green_margin_monotone_increasing_in_W"] = bool(s2["green_margin_monotone_increasing_in_W"])
    # exactly the primary W=4 clears GREEN on the linear spine (W2/W3 YELLOW).
    c["10_only_primary_W4_clears_green_on_linear"] = bool(
        s2["per_width"][4]["linear_spine_verdict"] == "GREEN"
        and s2["per_width"][3]["linear_spine_verdict"] == "YELLOW"
        and s2["per_width"][2]["linear_spine_verdict"] == "YELLOW")

    # (c) CPU dry-run reproduces #79's cov4 from banked records via the REAL analyze().
    dry = s3["cpu_dryrun"]
    c["11_dryrun_reproduces_linear_cov4"] = bool(
        dry["probe_reproduces_linear_cov4"]
        and dry["cov2_abs_diff_vs_banked"] <= TOL_REPRO
        and dry["cov3_abs_diff_vs_banked"] <= TOL_REPRO
        and dry["frac_abs_diff_vs_banked"] <= TOL_REPRO
        and dry["used_real_analyze"])

    # (d) #79/#309/#304 constants match the on-disk banked artifacts to <= 1e-6 (no silent drift).
    d79 = banked.get("pr79", {})
    d309 = banked.get("pr309", {})
    consts_ok = True
    if d79:
        a79 = d79.get("analysis", {})
        cc = a79.get("cumulative_coverage", {})
        consts_ok = consts_ok and all(abs(COV_W[w] - float(cc[str(w)])) <= TOL_REPRO for w in WIDTHS)
        consts_ok = consts_ok and abs(
            FRAC_TRUE_BEYOND_TOP4 - float(a79.get("frac_true_beyond_topW"))) <= TOL_REPRO
        consts_ok = consts_ok and int(a79.get("n_divergences")) == N_DIVERGENCES
        hist79 = {int(k): int(v) for k, v in a79.get("rank_fd_hist_pooled", {}).items()}
        consts_ok = consts_ok and hist79 == RANK_FD_HIST_POOLED
    if d309:
        imp = (d309.get("synthesis", {}) or {}).get("imported", {})
        consts_ok = consts_ok and abs(
            float(imp.get("a1_required_611_nosalvage")) - HARD_LINE_304) <= TOL_REPRO
        consts_ok = consts_ok and abs(float(imp.get("a1_deployed")) - A1_DEPLOYED) <= TOL_REPRO
    c["12_constants_match_banked_artifacts"] = bool(consts_ok and bool(d79) and bool(d309))

    # (f) caveats carried (pre-registration honesty).
    c["13_caveats_carried"] = bool(
        s4["fusion_cov_W_measured"] is False
        and s4["fusion_checkpoint_exists"] is False
        and s4["linear_spine_is_best_case_transfer"] is True
        and isinstance(s4["caveat"], str) and len(s4["caveat"]) > 80)

    gate = all(bool(v) for v in c.values())
    return {"fusion_rankcov_probe_self_test_passes": gate, "checks": c}


# --------------------------------------------------------------------------- #
# NaN-clean walk.
# --------------------------------------------------------------------------- #
def assert_nan_clean(payload: dict) -> list[str]:
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

    walk(payload, "result")
    return bad


# --------------------------------------------------------------------------- #
# W&B logging (summary/ namespace; robust; never fatal).
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-rankcov-probe] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    s1, s2, s4 = (syn["step1_reproduce_309"], syn["step2_decision_threshold"], syn["step4_caveat"])
    dry = syn["step3_probe_spec_and_dryrun"]["cpu_dryrun"]
    st = payload["self_test"]
    tm = syn["test_metrics"]

    run = init_wandb_run(
        job_type="validity-analytic", agent="lawine", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3-rankcov-probe", "validity-analytic", "rank-coverage", "tree-verify", "eagle3",
              "decision-threshold", "pre-registration", "bank-the-analysis"],
        config={
            "pr": 313, "effective_target_T": HARD_LINE_304, "trainable_band_edge_green": TRAINABLE_BAND_EDGE,
            "cov_W_measured_79": COV_W, "frac_true_beyond_top4_79": FRAC_TRUE_BEYOND_TOP4,
            "primary_W": PRIMARY_W, "a1_deployed": A1_DEPLOYED, "wandb_group": args.wandb_group,
            "imports": syn["imported"]["provenance"],
        },
    )
    if run is None:
        print("[eagle3-rankcov-probe] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "fusion_rankcov_probe_self_test_passes": int(bool(st["fusion_rankcov_probe_self_test_passes"])),
        "frac_beyond_top4_threshold_for_green": tm["frac_beyond_top4_threshold_for_green"],
        "cov4_threshold_for_green": tm["cov4_threshold_for_green"],
        "probe_reproduces_linear_cov4": int(bool(tm["probe_reproduces_linear_cov4"])),
        # decision threshold
        "cov_W_star_green": s2["cov_W_star_green"],
        "frac_star_green": s2["frac_true_beyond_topW_star_green"],
        "cov_W_star_red": s2["cov_W_star_red"],
        "frac_star_red": s2["frac_true_beyond_topW_star_red"],
        "frac_headroom_abs_primary": s2["frac_headroom_abs_primary"],
        "frac_headroom_rel_primary": s2["frac_headroom_rel_primary"],
        "cov_headroom_rel_primary": s2["cov_headroom_rel_primary"],
        # per-width demand at banked linear cov (== #309 curve)
        "demand_at_linear_cov_W2": s2["per_width"][2]["salvaged_raw_a1_demand_at_banked_cov"],
        "demand_at_linear_cov_W3": s2["per_width"][3]["salvaged_raw_a1_demand_at_banked_cov"],
        "demand_at_linear_cov_W4": s2["per_width"][4]["salvaged_raw_a1_demand_at_banked_cov"],
        "verdict_linear_W2": s2["per_width"][2]["linear_spine_verdict"],
        "verdict_linear_W3": s2["per_width"][3]["linear_spine_verdict"],
        "verdict_linear_W4": s2["per_width"][4]["linear_spine_verdict"],
        # step-1 reproduction residuals
        "a1_req_max_abs_diff_vs_309": s1["a1_req_max_abs_diff_vs_309"],
        "curve_max_abs_diff_vs_309": s1["curve_max_abs_diff_vs_309"],
        "zero_salvage_abs_diff_vs_304": s1["zero_salvage_abs_diff_vs_304"],
        # dry-run residuals
        "dryrun_cov4_abs_diff_vs_banked": dry["cov4_abs_diff_vs_banked"],
        "dryrun_used_real_analyze": int(bool(dry["used_real_analyze"])),
        "dryrun_n_divergences": dry["analyze_n_divergences"],
        # imported anchors
        "cov4_linear_79": COV_W[4],
        "frac_true_beyond_top4_linear_79": FRAC_TRUE_BEYOND_TOP4,
        "effective_target_T_304": HARD_LINE_304,
        # caveat flags
        "fusion_cov_W_measured": int(bool(s4["fusion_cov_W_measured"])),
        # hygiene
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["checks"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_fusion_rankcov_probe_result", artifact_type="validity",
                      data=payload)
    rid = getattr(run, "id", None)
    print(f"[eagle3-rankcov-probe] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    syn = payload["synthesis"]
    s1, s2, s4 = (syn["step1_reproduce_309"], syn["step2_decision_threshold"], syn["step4_caveat"])
    dry = syn["step3_probe_spec_and_dryrun"]["cpu_dryrun"]
    st = payload["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("EAGLE-3 FUSION RANK-COVERAGE PROBE (PR #313) -- pre-register the test that flips #309's YELLOW",
          flush=True)
    print("=" * 100, flush=True)
    print("STEP 1 -- reproduce #309 salvage operator + inverse:", flush=True)
    print(f"  a1-required by W: W2 {s1['a1_draft_required_by_W'][2]:.4f} | "
          f"W3 {s1['a1_draft_required_by_W'][3]:.4f} | W4 {s1['a1_draft_required_by_W'][4]:.4f}  "
          f"(max |diff| vs #309 = {s1['a1_req_max_abs_diff_vs_309']:.2e})", flush=True)
    print(f"  effective-a1 curve max |diff| vs #309 = {s1['curve_max_abs_diff_vs_309']:.2e}; "
          f"zero-salvage |diff| vs #304 = {s1['zero_salvage_abs_diff_vs_304']:.2e}", flush=True)
    print("-" * 100, flush=True)
    print("STEP 2 -- decision threshold (W-invariant on cov axis):", flush=True)
    print(f"  GREEN edge d={s2['trainable_band_edge_green']}: cov4* = {s2['cov_W_star_green']:.4f}, "
          f"frac4* = {s2['frac_true_beyond_topW_star_green']:.4f}", flush=True)
    print(f"  RED edge d={s2['hard_line_edge_red']:.4f}: cov* = {s2['cov_W_star_red']:.4f}, "
          f"frac* = {s2['frac_true_beyond_topW_star_red']:.1f} (unreachable for positive transfer)",
          flush=True)
    print("  per width (banked LINEAR spine):", flush=True)
    for w in (2, 3, 4):
        pw = s2["per_width"][w]
        print(f"    W={w}: cov={pw['cov_W_banked_linear']:.4f} frac={pw['frac_true_beyond_topW_banked_linear']:.4f} "
              f"-> demand={pw['salvaged_raw_a1_demand_at_banked_cov']:.4f} "
              f"[margin_cov={pw['green_margin_cov']:+.4f}] {pw['linear_spine_verdict']}", flush=True)
    print(f"  primary W=4 headroom: frac {s2['frac_headroom_abs_primary']:+.4f} "
          f"(cov4 may degrade {100.0 * s2['cov_headroom_rel_primary']:.1f}% rel before YELLOW)", flush=True)
    print(f"  RULE: {s2['falsifiable_rule']}", flush=True)
    print("-" * 100, flush=True)
    print("STEP 3 -- probe spec + CPU dry-run on #79 banked records:", flush=True)
    print(f"  reproduced cov_W = {dry['reproduced_cov_W']} (used_real_analyze={dry['used_real_analyze']}, "
          f"n_div={dry['analyze_n_divergences']})", flush=True)
    print(f"  cov4 |diff| vs banked = {dry['cov4_abs_diff_vs_banked']:.2e} -> "
          f"probe_reproduces_linear_cov4 = {dry['probe_reproduces_linear_cov4']}", flush=True)
    print("-" * 100, flush=True)
    print("STEP 4 -- caveat:", flush=True)
    print(f"  fusion cov_W measured? {s4['fusion_cov_W_measured']} | fusion checkpoint exists? "
          f"{s4['fusion_checkpoint_exists']} | linear spine = best-case transfer? "
          f"{s4['linear_spine_is_best_case_transfer']}", flush=True)
    print(f"  {s4['caveat']}", flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) fusion_rankcov_probe_self_test_passes = "
          f"{st['fusion_rankcov_probe_self_test_passes']}", flush=True)
    for k, val in st["checks"].items():
        print(f"   - {k}: {val}", flush=True)
    print(f"nan_clean = {payload['nan_clean']}   peak_mem_mib = {payload['peak_mem_mib']}", flush=True)
    print("=" * 100 + "\n", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="eagle3-rankcov-probe")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    banked = load_banked()
    syn = synthesize(banked)
    st = self_test(syn, banked)

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 313, "agent": "lawine", "kind": "eagle3-fusion-rankcov-probe",
        "eagle3_fusion_rankcov_probe_analysis_only": True,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[eagle3-rankcov-probe] WARNING non-finite at: {nan_paths}", flush=True)
    gate = bool(st["fusion_rankcov_probe_self_test_passes"] and payload["nan_clean"])
    st["fusion_rankcov_probe_self_test_passes"] = gate

    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_fusion_rankcov_probe_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eagle3-rankcov-probe] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    tm = syn["test_metrics"]
    print(f"  PRIMARY fusion_rankcov_probe_self_test_passes = {gate}", flush=True)
    print(f"  TEST frac_beyond_top4_threshold_for_green = "
          f"{tm['frac_beyond_top4_threshold_for_green']:.6f}", flush=True)
    print(f"  TEST probe_reproduces_linear_cov4 = {tm['probe_reproduces_linear_cov4']}", flush=True)
    print(f"  wandb run = {rid}", flush=True)

    if args.self_test:
        print(f"[eagle3-rankcov-probe] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
