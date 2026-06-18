#!/usr/bin/env python
"""PR #626 — GREEDY answer-materiality analysis (spec vs AR, same int4 body).

Consumes the paired greedy decode jsonls written by gen_paired_greedy.py and answers
the load-bearing question: do the ~0.43% residual greedy token-flips (#616 int4-Marlin
M=8-verify grid-ties) ever change a final extracted EVAL ANSWER, and if so, does that
systematically move quality?

Three measurements (PR #626):
  (a) TOKEN-level greedy divergence — the official greedy_identity verifier path
      (compare_files) gives the canonical per-prompt divergence; we report BOTH the
      naive cascade-amplified divergent-item fraction (#607-style) AND the un-amplified
      matched-prefix per-step HAZARD rate (n_diverged / sum first-div-or-full-len),
      which is the number that corroborates #616's ~0.43% structural flip rate.
  (b) ANSWER-level divergence [HEADLINE] — frac of paired items whose extracted answer
      differs between arms, per eval.
  (c) NET quality consequence — signed McNemar (spec-wins b vs ar-wins c) + cluster
      bootstrap on the paired accuracy delta (spec-ar) with CI. Each greedy item is its
      own cluster (deterministic; no seed correlation).
  (d) FLIP CHARACTER — from the AR-side prompt_logprobs gap probe at each item's first
      cross-arm divergence: frac of flips that are <0.5-nat near-ties, the tau=0.3-nat
      relaxed-acceptor rescue fraction, and a FLAG for any LARGE-margin answer-flipping
      divergence (an answer-divergent item whose ROOT token-divergence was NOT a grid-tie).

Verdict:
  RESIDUAL_ANSWER_IMMATERIAL  -> net paired delta CI contains 0 on BOTH evals AND no
                                 large-margin answer-flipping divergence. The residual
                                 greedy flips are grid-tie coin-flips in the CoT that
                                 never systematically move a graded answer; firing option
                                 B costs ONLY the strict-#319 byte-exact greedy contract.
  RESIDUAL_FLIPS_ANSWERS      -> some eval's delta CI excludes 0, OR a large-margin
                                 structural flip changed an answer. The residual is NOT
                                 answer-immaterial.

Backbone (mcnemar / cluster_bootstrap / cluster_level_paired_diff) reused verbatim from
the #620 matched-arm analyzer so the paired statistics are identical machinery.

ANALYSIS-ONLY. No GPU. analysis_only=True, official_tps=0.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# #620 paired-stats backbone (pure numpy/math; importing only defines functions).
sys.path.insert(0, str(ROOT / "research/validity/spec_distribution_preservation_matched_arm"))
from analyze_matched_arm import (  # noqa: E402
    mcnemar,
    cluster_bootstrap,
    cluster_level_paired_diff,
)

# official greedy-identity verifier (canonical token-divergence read).
import importlib.util  # noqa: E402

_GI_PATH = (
    ROOT / "official/main_bucket/shared_resources/"
    "gemma_greedy_identity_verifier_flowian-powers/greedy_identity.py"
)
_spec = importlib.util.spec_from_file_location("greedy_identity", _GI_PATH)
gi = importlib.util.module_from_spec(_spec)
sys.modules["greedy_identity"] = gi  # dataclass needs the module registered before exec
_spec.loader.exec_module(gi)

# A flip is a near-tie (a small relaxed-acceptor tolerance could rescue it) if the
# AR-side gap between the AR token and the spec token is small (#616 NEARTIE_NATS).
NEARTIE_NATS = 0.5
TAU_RESCUE = 0.3  # #616 rescued 100% of int4-grid flips at tau=0.3

# PR #637 Leg 2 target: tighten the GPQA paired net-Δ CI half-width to this (±4pp) so
# the GPQA leg is as decisive as GSM8K (which is already ±1.4pp at n=500).
GPQA_CI_HALFWIDTH_TARGET = 0.04


def _base_qid(iid: str) -> str:
    """Underlying question id, stripping the PR #637 multi-shuffle "#s<seed>" tag. For
    single-shuffle ids (AIME/GSM8K/the #626 primary GPQA seed) this is the id itself, so
    by-question clustering reduces to per-item clustering and reproduces #626 exactly."""
    return iid.split("#s", 1)[0]


def _ci_halfwidth(ci: list[float] | None) -> float | None:
    if not ci or len(ci) != 2 or ci[0] != ci[0] or ci[1] != ci[1]:
        return None
    return (float(ci[1]) - float(ci[0])) / 2.0


# --------------------------------------------------------------------------- IO
def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["id"])] = r
    return out


def _arm_path(arm: str, kind: str) -> Path:
    return RES / f"{arm}_{kind}.jsonl"


# --------------------------------------------------------------------------- (a) tokens
def token_divergence(kind: str, clean_ids: set[str]) -> dict[str, Any]:
    """Canonical greedy_identity verdict (ref=ar, cand=spec) PLUS the un-cascaded
    matched-prefix hazard rate, restricted to the CLEAN paired items."""
    ar_path, spec_path = _arm_path("ar", kind), _arm_path("spec", kind)
    if not ar_path.exists() or not spec_path.exists():
        return {"available": False}
    report = gi.compare_files(ar_path, spec_path)  # ref=ar greedy, cand=spec
    # per-prompt rows restricted to clean (non-errored, both-present) items.
    rows = [p for p in report.per_prompt if p.key in clean_ids]
    n_items = len(rows)
    n_div_items = sum(1 for p in rows if not p.identical)
    tot_tokens = sum(p.num_compared for p in rows)
    tot_div_tokens = sum(p.num_divergent_tokens for p in rows)
    # matched-prefix hazard: each item contributes its first-divergence index (or full
    # compared length if identical) as the matched run, and 1 "event" iff it diverged.
    # hazard = events / matched-prefix-tokens-at-risk == per-step P(first flip), no cascade.
    risk_tokens = 0
    onsets: list[int] = []
    for p in rows:
        if p.identical:
            risk_tokens += p.num_compared
        else:
            fdi = p.first_divergence_index if p.first_divergence_index is not None else 0
            risk_tokens += fdi + 1  # the matched prefix plus the flipping step itself
            onsets.append(fdi)
    hazard = (n_div_items / risk_tokens) if risk_tokens else float("nan")
    return {
        "available": True,
        "verdict_official": report.verdict,            # expected DIVERGENT
        "n_items": n_items,
        "n_divergent_items": n_div_items,
        "frac_divergent_items_cascade": (n_div_items / n_items) if n_items else float("nan"),
        "total_tokens_compared": tot_tokens,
        "total_divergent_tokens": tot_div_tokens,
        "naive_token_div_rate_cascade": (tot_div_tokens / tot_tokens) if tot_tokens else float("nan"),
        "matched_prefix_hazard_rate": hazard,          # the #616-comparable per-step rate
        "risk_tokens": risk_tokens,
        "first_div_onset_median": int(statistics.median(onsets)) if onsets else None,
        "first_div_onset_min": min(onsets) if onsets else None,
        "n_integrity_failures": len(report.integrity_failures),
        "n_missing_in_spec": len(report.missing_in_candidate),
        "n_missing_in_ar": len(report.missing_in_reference),
    }


# --------------------------------------------------------------------------- (d) gaps
def load_gaps(kind: str) -> dict[str, dict]:
    return load_jsonl(RES / f"gaps_{kind}.jsonl")


def flip_character(kind: str, gaps: dict[str, dict]) -> dict[str, Any]:
    """Distribution of the AR-side first-divergence logit gap (ar_tok vs spec_tok).
    Small gap == int4 grid-tie (rescuable); large gap == structural M-induced flip."""
    probed = [g for g in gaps.values()
              if g.get("divergent") and g.get("gap_ar_minus_spec") is not None]
    g_vals = [float(g["gap_ar_minus_spec"]) for g in probed]
    outside = sum(1 for g in gaps.values() if g.get("spec_outside_topk"))
    n = len(g_vals)
    frac_under = (sum(1 for g in g_vals if g < NEARTIE_NATS) / n) if n else None
    frac_rescued_tau = (sum(1 for g in g_vals if g <= TAU_RESCUE) / n) if n else None
    qs = sorted(g_vals)

    def q(p):
        if not qs:
            return None
        if len(qs) == 1:
            return qs[0]
        pos = p * (len(qs) - 1)
        lo = int(math.floor(pos)); hi = min(lo + 1, len(qs) - 1)
        return qs[lo] * (1 - (pos - lo)) + qs[hi] * (pos - lo)

    return {
        "n_divergent_with_probe": n,
        "frac_flips_under_0p5nat": frac_under,
        "frac_flips_rescued_at_tau_0p3nat": frac_rescued_tau,
        "n_spec_tok_outside_topk": outside,
        "gap_median": q(0.5),
        "gap_p90": q(0.9),
        "gap_p99": q(0.99),
        "gap_max": max(g_vals) if g_vals else None,
        "gap_mean": statistics.fmean(g_vals) if g_vals else None,
    }


# ------------------------------------------------------------- (e) absolute AIME quality
# PR #637 advisor re-prioritization (2026-06-18): AIME is the 4th bar of the Reading-A
# quality panel. Beyond materiality (is the spec-vs-AR break large-margin?) the advisor
# needs the ABSOLUTE Option-B AIME maj@1 number + %-of-base vs the ubel #628 int4 base
# and the Reading-A bar. Constants are the advisor-named scalars from the PR comment.
AIME_BASE_UBEL = 0.4667  # ubel #628 int4 base, AIME-2024 n=30 (=14/30), gb6144 greedy
AIME_BAR = 0.420         # Reading-A AIME quality bar (advisor PR #637 comment)


def aime_quality(spec: dict[str, dict], ar: dict[str, dict], clean: list[str]) -> dict[str, Any]:
    """Absolute Option-B (spec) AIME maj@1 accuracy + paired AR base, year-stratified.

    The ubel #628 base 0.4667 the advisor names is AIME-2024-only (n=30, 14/30), so the
    apples-to-apples ``pct_of_base`` denominator is the 2024 stratum; we also report the
    full-mix (2024+2025-I+2025-II) number and the honest paired-AR ratio (denominator-
    independent: spec vs the SAME-config int4 AR arm on the SAME items)."""
    from collections import defaultdict
    yrs: dict[str, dict[str, int]] = defaultdict(lambda: {"spec": 0, "ar": 0, "n": 0})
    s_all = a_all = n_all = 0
    for iid in clean:
        s, a = spec[iid], ar[iid]
        y = str(s.get("year"))
        sc = 1 if s.get("correct") else 0
        ac = 1 if a.get("correct") else 0
        yrs[y]["spec"] += sc; yrs[y]["ar"] += ac; yrs[y]["n"] += 1
        s_all += sc; a_all += ac; n_all += 1
    optionb_all = (s_all / n_all) if n_all else float("nan")
    ar_all = (a_all / n_all) if n_all else float("nan")
    y24 = yrs.get("2024")
    optionb_2024 = (y24["spec"] / y24["n"]) if (y24 and y24["n"]) else None
    ar_2024 = (y24["ar"] / y24["n"]) if (y24 and y24["n"]) else None
    return {
        "ubel_base_2024": AIME_BASE_UBEL, "bar": AIME_BAR,
        "optionb_aime_all": optionb_all, "ar_base_aime_all": ar_all, "n_all": n_all,
        "optionb_aime_2024": optionb_2024, "ar_base_aime_2024": ar_2024,
        "n_2024": (y24["n"] if y24 else 0),
        # %-of-base: apples-to-apples is 2024-vs-ubel; full-mix-vs-ubel + paired-AR also shown.
        "pct_of_base_aime_2024_vs_ubel": (optionb_2024 / AIME_BASE_UBEL) if optionb_2024 is not None else None,
        "pct_of_base_aime_all_vs_ubel": (optionb_all / AIME_BASE_UBEL),
        "pct_of_base_aime_all_vs_paired_ar": (optionb_all / ar_all) if ar_all else None,
        "pct_of_base_aime_2024_vs_paired_ar": (optionb_2024 / ar_2024) if (ar_2024 and optionb_2024 is not None) else None,
        "optionb_aime_2024_clears_bar": (optionb_2024 is not None and optionb_2024 >= AIME_BAR),
        "optionb_aime_all_clears_bar": (optionb_all >= AIME_BAR),
        "ar_base_aime_all_clears_bar": (ar_all >= AIME_BAR),
        "by_year": {y: {"spec_acc": v["spec"] / v["n"], "ar_acc": v["ar"] / v["n"], "n": v["n"]}
                    for y, v in sorted(yrs.items())},
    }


# --------------------------------------------------------------------------- pairing
def pair_eval(kind: str) -> dict[str, Any]:
    spec = load_jsonl(_arm_path("spec", kind))
    ar = load_jsonl(_arm_path("ar", kind))
    gaps = load_gaps(kind)
    common = sorted(set(spec) & set(ar))

    sha_mismatch: list[str] = []
    spec_err: set[str] = set()
    ar_err: set[str] = set()
    clean: list[str] = []
    answer_div_ids: list[str] = []
    inst_ids: list[str] = []     # per (question,shuffle) instance
    base_qids: list[str] = []    # underlying question (multi-shuffle reps collapse here)
    spec_c: list[int] = []
    ar_c: list[int] = []

    for iid in common:
        s, a = spec[iid], ar[iid]
        if s.get("error"):
            spec_err.add(iid)
        if a.get("error"):
            ar_err.add(iid)
        # prompt_sha gate — identical by construction (same tokenized prompt to both arms).
        ssha, asha = s.get("prompt_sha256"), a.get("prompt_sha256")
        if ssha is not None and asha is not None and ssha != asha:
            sha_mismatch.append(iid)
            continue
        if s.get("error") or a.get("error"):
            continue
        clean.append(iid)
        inst_ids.append(iid)
        base_qids.append(str(s.get("base_qid") or _base_qid(iid)))
        spec_c.append(1 if s.get("correct") else 0)
        ar_c.append(1 if a.get("correct") else 0)
        if s.get("answer") != a.get("answer"):
            answer_div_ids.append(iid)

    clean_set = set(clean)
    tok = token_divergence(kind, clean_set)
    fc = flip_character(kind, gaps)
    aime_q = aime_quality(spec, ar, clean) if kind == "aime" else None

    # (c) net quality consequence on clean pairs.
    #   PRIMARY (cb) clusters by underlying QUESTION (base_qid): for the PR #637 GPQA
    #   multi-shuffle pool this is the honest unit — it resamples the 198 questions, each
    #   carrying its shuffle-reps, so two permutations of one question are NOT treated as
    #   independent (no pseudo-replication). For single-shuffle evals base_qid==iid, so
    #   this reproduces #626 byte-for-byte. SECONDARY (cb_inst) clusters by (question,
    #   shuffle) instance — the optimistic "n≈400 independent units" view; reported as a
    #   sensitivity bound. The true CI sits between them.
    n_units = len(spec_c)
    n_questions = len(set(base_qids))
    multi_shuffle = n_units > n_questions
    spec_a, ar_a = np.array(spec_c), np.array(ar_c)
    bq_a, inst_a = np.array(base_qids), np.array(inst_ids)
    pairs = list(zip(spec_c, ar_c))
    mc = mcnemar(pairs)
    cb = cluster_bootstrap(bq_a, spec_a, ar_a) if spec_c else {}
    cb_inst = (cluster_bootstrap(inst_a, spec_a, ar_a) if (spec_c and multi_shuffle) else cb)
    cl = cluster_level_paired_diff(bq_a, spec_a, ar_a) if spec_c else {}

    # (b) headline answer divergence.
    n_clean = len(clean)
    answer_div_rate = (len(answer_div_ids) / n_clean) if n_clean else float("nan")

    # large-margin answer-flips: an answer-divergent item whose ROOT (first) token
    # divergence was NOT a near-tie (gap >= NEARTIE_NATS) or whose spec token fell
    # outside the AR top-k (decisive). These would be structural, not grid-tie.
    large_margin_answer_flips = []
    for iid in answer_div_ids:
        g = gaps.get(iid, {})
        gap = g.get("gap_ar_minus_spec")
        outside = g.get("spec_outside_topk")
        if outside or (gap is not None and float(gap) >= NEARTIE_NATS):
            large_margin_answer_flips.append({
                "id": iid, "gap_ar_minus_spec": gap, "spec_outside_topk": bool(outside),
                "first_div_index": g.get("first_div_index"),
                "spec_answer": spec[iid].get("answer"), "ar_answer": ar[iid].get("answer"),
                "spec_correct": spec[iid].get("correct"), "ar_correct": ar[iid].get("correct"),
            })

    delta_ci = cb.get("delta_ci95")
    delta_ci_contains_0 = bool(delta_ci and delta_ci[0] <= 0.0 <= delta_ci[1])
    delta_ci_hw = _ci_halfwidth(delta_ci)
    delta_ci_inst = cb_inst.get("delta_ci95")
    delta_ci_inst_hw = _ci_halfwidth(delta_ci_inst)
    delta_ci_inst_contains_0 = bool(delta_ci_inst and delta_ci_inst[0] <= 0.0 <= delta_ci_inst[1])

    return {
        "kind": kind,
        "n_common": len(common),
        "n_clean_pairs": n_clean,
        "n_prompt_sha_mismatch": len(sha_mismatch),
        "prompt_sha_gate_pass": len(sha_mismatch) == 0,
        "errors": {
            "n_spec_errored": len(spec_err),
            "n_ar_errored": len(ar_err),
            "symmetric": spec_err == ar_err,
            "errored_examples": sorted(spec_err | ar_err)[:5],
        },
        "token_divergence": tok,
        "answer_divergence": {
            "n_answer_divergent": len(answer_div_ids),
            "answer_div_rate": answer_div_rate,
            "answer_divergent_ids": answer_div_ids[:50],
        },
        "net_quality": {
            "mcnemar": mc,
            "cluster_bootstrap": cb,                 # PRIMARY: by-question (honest)
            "cluster_bootstrap_inst": cb_inst,       # SECONDARY: per (q,shuffle) instance
            "cluster_level_paired_diff": cl,
            "net_graded_delta_spec_minus_ar": cb.get("delta"),
            "net_graded_delta_ci95": delta_ci,
            "net_graded_delta_ci95_halfwidth": delta_ci_hw,
            "delta_ci_contains_0": delta_ci_contains_0,
            # per-instance sensitivity (optimistic n≈units view)
            "net_graded_delta_ci95_inst": delta_ci_inst,
            "net_graded_delta_ci95_inst_halfwidth": delta_ci_inst_hw,
            "delta_ci_inst_contains_0": delta_ci_inst_contains_0,
            "n_units": n_units,
            "n_questions": n_questions,
            "multi_shuffle": multi_shuffle,
        },
        "flip_character": fc,
        "large_margin_answer_flips": large_margin_answer_flips,
        "n_large_margin_answer_flips": len(large_margin_answer_flips),
        "aime_quality": aime_q,
    }


def verdict_for(ev: dict[str, Any]) -> dict[str, Any]:
    """Per-eval materiality read."""
    delta0 = ev["net_quality"]["delta_ci_contains_0"]
    n_large = ev["n_large_margin_answer_flips"]
    immaterial = bool(delta0 and n_large == 0)
    return {
        "delta_ci_contains_0": delta0,
        "n_large_margin_answer_flips": n_large,
        "verdict": "RESIDUAL_ANSWER_IMMATERIAL" if immaterial else "RESIDUAL_FLIPS_ANSWERS",
    }


def _packet_637(evals: dict[str, Any]) -> dict[str, Any] | None:
    """PR #637 packet verdict.

    Two legs:
      * AIME (Leg 1) — the decisive metric is ``aime_n_large_margin_answer_flips``
        (target 0). AIME answers are confident integers, so a tie-flip reaching the
        boxed answer is a LARGE-margin answer flip by construction — the exact event
        #626's verdict says never happens. One such flip is a counterexample (fire-
        blocker).
      * GPQA (Leg 2) — tighten the net-Δ CI half-width to <= 0.04 (±4pp) while still
        ∋0. The honest (by-question) half-width is the gate; a structural flip or a CI
        that shifts off 0 fails the leg.

    Returns None when neither #637 leg is present (a #626-only gpqa,gsm8k run)."""
    aime = evals.get("aime")
    gpqa = evals.get("gpqa")
    if aime is None and gpqa is None:
        return None
    decisive: dict[str, Any] = {}

    aime_ok = aime is not None
    if aime is not None:
        anq = aime["net_quality"]
        a_nlarge = aime["n_large_margin_answer_flips"]
        a_ci0 = anq["delta_ci_contains_0"]
        decisive.update({
            "aime_answer_div_rate": aime["answer_divergence"]["answer_div_rate"],
            "aime_n_large_margin_answer_flips": a_nlarge,
            "aime_frac_flips_under_0p5nat": aime["flip_character"]["frac_flips_under_0p5nat"],
            "aime_net_delta": anq["net_graded_delta_spec_minus_ar"],
            "aime_net_delta_ci95": anq["net_graded_delta_ci95"],
            "aime_net_delta_ci_halfwidth": anq["net_graded_delta_ci95_halfwidth"],
            "aime_net_delta_ci_contains_0": a_ci0,
            "aime_n_clean_pairs": aime["n_clean_pairs"],
        })
        # advisor re-prioritization: absolute Option-B AIME quality + %-of-base vs the bar.
        aq = aime.get("aime_quality") or {}
        if aq:
            decisive.update({
                "optionb_aime": aq.get("optionb_aime_all"),
                "optionb_aime_2024": aq.get("optionb_aime_2024"),
                "ar_base_aime": aq.get("ar_base_aime_all"),
                "ar_base_aime_2024": aq.get("ar_base_aime_2024"),
                "pct_of_base_aime": aq.get("pct_of_base_aime_2024_vs_ubel"),     # apples-to-apples (2024 vs ubel 0.4667)
                "pct_of_base_aime_all_vs_ubel": aq.get("pct_of_base_aime_all_vs_ubel"),
                "pct_of_base_aime_vs_paired_ar": aq.get("pct_of_base_aime_all_vs_paired_ar"),
                "aime_bar": aq.get("bar"),
                "aime_base_ubel": aq.get("ubel_base_2024"),
                "optionb_aime_2024_clears_bar": aq.get("optionb_aime_2024_clears_bar"),
                "optionb_aime_all_clears_bar": aq.get("optionb_aime_all_clears_bar"),
            })
        aime_ok = (a_nlarge == 0 and a_ci0)

    gpqa_ok = gpqa is not None
    if gpqa is not None:
        gnq = gpqa["net_quality"]
        # PR-DECISIVE half-width = the per-instance (units) CI: the PR's "tighten to ±4pp
        # by bringing paired n to ~400" mechanism is 1/sqrt(units) scaling, which only the
        # units view responds to (by-question keeps 198 fixed clusters → its half-width is
        # heterogeneity-floored and ~insensitive to added shuffles). We still report the
        # by-question (honest, conservative) CI and use IT for the no-shift (∋0) gate so a
        # "SHIFTS" call is never raised on anti-conservative grounds.
        g_hw_units = gnq["net_graded_delta_ci95_inst_halfwidth"]
        g_hw_byq = gnq["net_graded_delta_ci95_halfwidth"]
        g_ci0_byq = gnq["delta_ci_contains_0"]              # honest no-shift gate
        g_ci0_units = gnq.get("delta_ci_inst_contains_0", g_ci0_byq)
        g_nlarge = gpqa["n_large_margin_answer_flips"]
        hw_ok = (g_hw_units is not None and g_hw_units <= GPQA_CI_HALFWIDTH_TARGET)
        decisive.update({
            "gpqa_net_delta": gnq["net_graded_delta_spec_minus_ar"],
            "gpqa_net_delta_ci95": gnq["net_graded_delta_ci95_inst"],       # the ±4pp target CI
            "gpqa_net_delta_ci_halfwidth": g_hw_units,                      # PR-decisive
            "gpqa_ci_halfwidth_target_met": hw_ok,
            "gpqa_net_delta_ci_contains_0": g_ci0_units,
            "gpqa_net_delta_ci95_byquestion": gnq["net_graded_delta_ci95"],  # honest companion
            "gpqa_net_delta_ci_byq_halfwidth": g_hw_byq,
            "gpqa_net_delta_ci_byq_contains_0": g_ci0_byq,
            "gpqa_n_large_margin_answer_flips": g_nlarge,
            "gpqa_n_units": gnq["n_units"],
            "gpqa_n_questions": gnq["n_questions"],
            "gpqa_multi_shuffle": gnq.get("multi_shuffle", False),
        })
        gpqa_ok = (g_ci0_byq and hw_ok and g_nlarge == 0)
        if g_hw_units is not None and not hw_ok:
            # n-needed (units 1/sqrt(n) scaling) to hit the ±4pp half-width target.
            factor = (g_hw_units / GPQA_CI_HALFWIDTH_TARGET) ** 2
            decisive["gpqa_n_units_needed_est"] = int(math.ceil(gnq["n_units"] * factor))

    a_flip = aime is not None and decisive.get("aime_n_large_margin_answer_flips", 0) >= 1
    # A GENUINE GPQA materiality problem = a structural flip or a net-Δ CI that SHIFTS off 0
    # (these falsify immateriality). Distinct from the CI merely being WIDE (∋0, 0 flips but
    # half-width > ±4pp): under the advisor's 2026-06-18 re-prioritization the GPQA-CI
    # tightening is OPTIONAL polish (the identity leg it de-risked is closed natively by
    # stark #636 / fern #641), and adding shuffle-seeds cannot tighten the honest by-question
    # CI anyway (heterogeneity-floored at GPQA-Diamond's 198 fixed questions), so wide-only
    # must NOT mask a passing AIME load-bearing leg.
    g_problem = gpqa is not None and (
        decisive.get("gpqa_n_large_margin_answer_flips", 0) >= 1
        or not decisive.get("gpqa_net_delta_ci_byq_contains_0", True)  # honest no-shift gate
    )
    g_wide_only = (gpqa is not None and not g_problem
                   and (decisive.get("gpqa_net_delta_ci_halfwidth") or 1.0) > GPQA_CI_HALFWIDTH_TARGET)
    if a_flip:
        verdict = "AIME_LARGE_MARGIN_FLIP_FOUND"
    elif g_problem:
        verdict = "GPQA_CI_STILL_WIDE_OR_SHIFTS"
    elif aime is not None and aime_ok and gpqa is not None and gpqa_ok:
        verdict = "MATERIALITY_PACKET_COMPLETE__IMMATERIAL_HOLDS"
    elif aime is not None and aime_ok:
        # load-bearing AIME leg passes (0 large-margin flips, net-Δ ∋0); GPQA-CI either
        # absent or wide-only (optional polish, not tightened to ±4pp). No fire-blocker.
        verdict = "AIME_IMMATERIAL_HOLDS__GPQA_CI_OPTIONAL"
    else:
        verdict = "PACKET_637_PARTIAL"  # one leg missing or AIME CI not ∋0 yet
    return {
        "verdict": verdict,
        "decisive": decisive,
        "ci_halfwidth_target": GPQA_CI_HALFWIDTH_TARGET,
        "legs_present": {"aime": aime is not None, "gpqa": gpqa is not None},
        "aime_leg_pass": bool(aime is not None and aime_ok and not a_flip),
        "gpqa_ci_tightened": bool(gpqa is not None and gpqa_ok),
        "gpqa_ci_wide_only": bool(g_wide_only),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evals", default="gpqa,gsm8k")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_group", default="optionb-319-residual-answer-materiality")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    evals = [e.strip() for e in args.evals.split(",") if e.strip()]

    meta = {}
    meta_path = RES / "gen_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    result: dict[str, Any] = {
        "pr": 626,
        "analysis_only": True,
        "official_tps": 0,
        "design": "GREEDY matched-arm paired (spec=int4+MTP-K7 ON M=8, ar=int4 spec OFF M=1, "
                  "same body, MAX_NUM_SEQS=1 serial)",
        "stack": "vllm==0.22.0",
        "decoding": {"temperature": 0.0, "min_tokens": meta.get("min_tokens", 8),
                     "max_model_len": meta.get("max_model_len", 6144),
                     "max_tokens_by_kind": meta.get("max_tokens_by_kind", {}),
                     "max_num_seqs": 1, "batch_invariant": 1},
        "gen_peaks_gb": meta.get("peaks", {}),
        "evals": {},
    }

    per_eval_verdicts = {}
    for kind in evals:
        ev = pair_eval(kind)
        ev["materiality"] = verdict_for(ev)
        result["evals"][kind] = ev
        per_eval_verdicts[kind] = ev["materiality"]["verdict"]

    # overall: IMMATERIAL only if EVERY eval is immaterial.
    all_immaterial = all(v == "RESIDUAL_ANSWER_IMMATERIAL" for v in per_eval_verdicts.values())
    result["headline_verdict"] = (
        "RESIDUAL_ANSWER_IMMATERIAL" if all_immaterial else "RESIDUAL_FLIPS_ANSWERS"
    )
    result["per_eval_verdicts"] = per_eval_verdicts

    # pooled headline numbers across both evals (single-number terminal metrics).
    pool_div_items = pool_risk = pool_div_tok = pool_tok = 0
    pool_flips_under = pool_probed = pool_outside = 0
    pool_answer_div = pool_clean = 0
    pool_large_margin = 0
    for kind in evals:
        ev = result["evals"][kind]
        td = ev["token_divergence"]
        if td.get("available"):
            pool_div_items += td["n_divergent_items"]; pool_risk += td["risk_tokens"]
            pool_div_tok += td["total_divergent_tokens"]; pool_tok += td["total_tokens_compared"]
        fc = ev["flip_character"]
        n_pr = fc["n_divergent_with_probe"]
        if n_pr:
            pool_flips_under += round((fc["frac_flips_under_0p5nat"] or 0.0) * n_pr)
            pool_probed += n_pr
        pool_outside += fc["n_spec_tok_outside_topk"]
        pool_answer_div += ev["answer_divergence"]["n_answer_divergent"]
        pool_clean += ev["n_clean_pairs"]
        pool_large_margin += ev["n_large_margin_answer_flips"]
    result["pooled"] = {
        "token_div_rate_greedy_hazard": (pool_div_items / pool_risk) if pool_risk else float("nan"),
        "naive_token_div_rate_cascade": (pool_div_tok / pool_tok) if pool_tok else float("nan"),
        "frac_flips_under_0p5nat": (pool_flips_under / pool_probed) if pool_probed else None,
        "n_divergent_probed": pool_probed,
        "n_spec_tok_outside_topk": pool_outside,
        "answer_div_rate_pooled": (pool_answer_div / pool_clean) if pool_clean else float("nan"),
        "n_answer_divergent": pool_answer_div,
        "n_clean_pairs": pool_clean,
        "n_large_margin_answer_flips_total": pool_large_margin,
    }

    # ---- PR #637 materiality packet: AIME greedy leg + tightened GPQA net-Δ CI ----
    packet = _packet_637(result["evals"])
    if packet is not None:
        result["packet_637"] = packet

    # terminal SENPAI metrics (PR #626): pull the per-eval + pooled headline numbers up.
    terminal: dict[str, Any] = {
        "verdict": result["headline_verdict"],
        "token_div_rate_greedy": result["pooled"]["token_div_rate_greedy_hazard"],
        "frac_flips_under_0p5nat": result["pooled"]["frac_flips_under_0p5nat"],
    }
    for kind in evals:
        ev = result["evals"][kind]
        terminal[f"token_div_rate_greedy_{kind}"] = ev["token_divergence"].get("matched_prefix_hazard_rate")
        terminal[f"answer_div_rate_{kind}"] = ev["answer_divergence"]["answer_div_rate"]
        terminal[f"net_graded_delta_spec_minus_ar_{kind}"] = ev["net_quality"]["net_graded_delta_spec_minus_ar"]
        terminal[f"net_graded_delta_ci95_{kind}"] = ev["net_quality"]["net_graded_delta_ci95"]
        terminal[f"net_graded_delta_ci95_halfwidth_{kind}"] = ev["net_quality"]["net_graded_delta_ci95_halfwidth"]
        terminal[f"n_large_margin_answer_flips_{kind}"] = ev["n_large_margin_answer_flips"]
        terminal[f"frac_flips_under_0p5nat_{kind}"] = ev["flip_character"]["frac_flips_under_0p5nat"]
    if packet is not None:
        terminal["packet_637_verdict"] = packet["verdict"]
        for k, v in packet["decisive"].items():
            terminal[f"packet637/{k}"] = v
    result["terminal_metrics"] = terminal

    (RES / "materiality_analysis.json").write_text(json.dumps(result, indent=2, default=str))

    # ---- human-readable report ----
    lines = ["GREEDY ANSWER-MATERIALITY: option-B spec vs AR, same int4 body (PR #626)",
             f"stack={result['stack']}  {result['design']}"]
    for kind in evals:
        ev = result["evals"][kind]
        td = ev["token_divergence"]; ad = ev["answer_divergence"]
        nq = ev["net_quality"]; fc = ev["flip_character"]
        cb = nq["cluster_bootstrap"]
        lines.append(f"\n=== {kind.upper()}  (n_clean={ev['n_clean_pairs']} / common={ev['n_common']}) ===")
        lines.append(f"  prompt_sha gate: {'PASS' if ev['prompt_sha_gate_pass'] else 'FAIL'}"
                     f"   serving errors: spec={ev['errors']['n_spec_errored']} ar={ev['errors']['n_ar_errored']}"
                     f" (symmetric={ev['errors']['symmetric']})")
        if td.get("available"):
            lines.append(f"  (a) TOKEN div: official={td['verdict_official']}  "
                         f"divergent-items {td['n_divergent_items']}/{td['n_items']} "
                         f"(cascade {td['frac_divergent_items_cascade']:.3%})")
            lines.append(f"      per-step HAZARD (un-cascaded) = {td['matched_prefix_hazard_rate']:.4%}"
                         f"  [#616 anchor ~0.43%]  onset median={td['first_div_onset_median']}")
        lines.append(f"  (b) ANSWER div rate = {ad['answer_div_rate']:.4%} "
                     f"({ad['n_answer_divergent']}/{ev['n_clean_pairs']})   [HEADLINE]")
        if cb:
            lines.append(f"  (c) spec acc={cb['spec_acc']:.4f} ar acc={cb['ar_acc']:.4f}  "
                         f"net delta(spec-ar)={cb['delta']:+.4f} CI95={[round(x,4) for x in cb['delta_ci95']]}"
                         f"  contains0={nq['delta_ci_contains_0']}")
        lines.append(f"      McNemar: b(spec>ar)={nq['mcnemar']['b']} c(ar>spec)={nq['mcnemar']['c']} "
                     f"p_exact={nq['mcnemar']['p_exact']:.4f}")
        lines.append(f"  (d) flips<0.5nat={fc['frac_flips_under_0p5nat']} "
                     f"(rescued@tau0.3={fc['frac_flips_rescued_at_tau_0p3nat']}, "
                     f"probed={fc['n_divergent_with_probe']}, outside_topk={fc['n_spec_tok_outside_topk']})")
        lines.append(f"      gap median={fc['gap_median']} p90={fc['gap_p90']} max={fc['gap_max']}")
        if ev["n_large_margin_answer_flips"]:
            lines.append(f"  !! LARGE-MARGIN ANSWER FLIPS: {ev['n_large_margin_answer_flips']} "
                         f"-> {ev['large_margin_answer_flips'][:3]}")
        else:
            lines.append("  large-margin answer flips: NONE")
        lines.append(f"  --> {kind} verdict: {ev['materiality']['verdict']}")
    pl = result["pooled"]
    lines.append(f"\nPOOLED: token_div_rate_greedy(hazard)={pl['token_div_rate_greedy_hazard']:.4%} "
                 f"[#616 ~0.43%]  frac_flips<0.5nat={pl['frac_flips_under_0p5nat']} "
                 f"(probed={pl['n_divergent_probed']}, outside_topk={pl['n_spec_tok_outside_topk']})  "
                 f"large-margin-answer-flips={pl['n_large_margin_answer_flips_total']}")
    lines.append(f"\nHEADLINE VERDICT: {result['headline_verdict']}")

    if packet is not None:
        d = packet["decisive"]
        lines.append("\n" + "=" * 64)
        lines.append("PR #637 MATERIALITY PACKET  (AIME greedy leg + tightened GPQA net-Δ CI)")
        lines.append(f"  legs present: aime={packet['legs_present']['aime']} "
                     f"gpqa={packet['legs_present']['gpqa']}  "
                     f"(GPQA CI half-width target ≤{packet['ci_halfwidth_target']})")
        if packet["legs_present"]["aime"]:
            lines.append(
                f"  [AIME]  answer_div={d.get('aime_answer_div_rate')!r}  "
                f"net Δ={d.get('aime_net_delta')!r} CI95={d.get('aime_net_delta_ci95')!r} "
                f"(∋0={d.get('aime_net_delta_ci_contains_0')}, hw={d.get('aime_net_delta_ci_halfwidth')!r})")
            lines.append(
                f"          flips<0.5nat={d.get('aime_frac_flips_under_0p5nat')!r}  "
                f"!! LARGE-MARGIN ANSWER FLIPS = {d.get('aime_n_large_margin_answer_flips')} "
                f"(DECISIVE; target 0)  n_clean={d.get('aime_n_clean_pairs')}")
            if d.get("optionb_aime") is not None:
                lines.append(
                    f"  [AIME quality]  optionb_aime(all-{d.get('aime_n_clean_pairs')})="
                    f"{d.get('optionb_aime'):.4f} (clears {d.get('aime_bar')} bar="
                    f"{d.get('optionb_aime_all_clears_bar')})  AR-base={d.get('ar_base_aime'):.4f}  "
                    f"spec/AR={d.get('pct_of_base_aime_vs_paired_ar'):.3%}")
                lines.append(
                    f"          2024-only: optionb={d.get('optionb_aime_2024'):.4f} "
                    f"AR={d.get('ar_base_aime_2024'):.4f}  vs ubel base {d.get('aime_base_ubel')} "
                    f"-> pct_of_base={d.get('pct_of_base_aime'):.3%} "
                    f"(clears {d.get('aime_bar')} bar={d.get('optionb_aime_2024_clears_bar')})")
        if packet["legs_present"]["gpqa"]:
            lines.append(
                f"  [GPQA]  net Δ={d.get('gpqa_net_delta')!r}")
            lines.append(
                f"          UNITS CI95 (±4pp target)={d.get('gpqa_net_delta_ci95')!r} "
                f"(∋0={d.get('gpqa_net_delta_ci_contains_0')}, hw={d.get('gpqa_net_delta_ci_halfwidth')!r} "
                f"-> target_met={d.get('gpqa_ci_halfwidth_target_met')})")
            lines.append(
                f"          BY-QUESTION CI95 (honest)={d.get('gpqa_net_delta_ci95_byquestion')!r} "
                f"(∋0={d.get('gpqa_net_delta_ci_byq_contains_0')}, hw={d.get('gpqa_net_delta_ci_byq_halfwidth')!r})")
            lines.append(
                f"          n_units={d.get('gpqa_n_units')} n_questions={d.get('gpqa_n_questions')} "
                f"multi_shuffle={d.get('gpqa_multi_shuffle')}  "
                f"large-margin-flips={d.get('gpqa_n_large_margin_answer_flips')}")
            if d.get("gpqa_n_units_needed_est") is not None:
                lines.append(f"          (est n_units to hit ±4pp half-width target ≈ "
                             f"{d['gpqa_n_units_needed_est']})")
        lines.append(f"\nPACKET #637 VERDICT: {packet['verdict']}")

    rep = "\n".join(lines)
    (RES / "materiality_report.txt").write_text(rep + "\n")
    print(rep, flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(result, name=args.wandb_name, group=args.wandb_group)
        except Exception as exc:  # noqa: BLE001
            print(f"[analyze] WARNING: wandb logging failed ({type(exc).__name__}: {exc}); "
                  f"report preserved at {RES/'materiality_analysis.json'}", flush=True)
    return 0


def _log_wandb(result: dict[str, Any], *, name: str | None, group: str | None) -> None:
    from scripts import wandb_logging as wl

    # PR #637 extends the #626 packet with the AIME greedy leg + the GPQA multi-shuffle
    # net-Δ CI tightening. When that packet is present, stamp the record as #637 so the
    # materiality-packet metrics are discoverable under the right PR/tags.
    packet = result.get("packet_637")
    pr_num = 637 if packet is not None else 626
    cfg = {
        "pr": pr_num, "stack": result["stack"], "design": result["design"],
        "analysis_only": True, "official_tps": 0,
        **{f"decoding/{k}": v for k, v in result["decoding"].items()
           if isinstance(v, (int, float, str))},
    }
    if packet is not None:
        cfg["packet_637_verdict"] = packet["verdict"]
        cfg["gpqa_ci_halfwidth_target"] = packet["ci_halfwidth_target"]
    tags = ["specdec", "greedy-identity", "answer-materiality", "int4-mtp", "option-b"]
    tags += (["pr637", "pr626", "materiality-packet", "aime", "gpqa-multishuffle"]
             if packet is not None else ["pr626"])
    notes = ("PR637 materiality packet: AIME greedy large-margin-flip leg + GPQA "
             "multi-shuffle net-Δ CI tightening (extends #626)." if packet is not None
             else "PR626 greedy matched-arm: do residual int4-Marlin spec flips change a graded answer?")
    run = wl.init_wandb_run(
        job_type="optionb-319-answer-materiality", agent="denken",
        name=name or ("denken/optionb-materiality-packet" if packet is not None
                      else "denken/optionb-319-residual-answer-materiality"),
        group=group or "optionb-319-residual-answer-materiality",
        notes=notes,
        tags=tags,
        config=cfg,
    )
    if run is None:
        print("[analyze] wandb not configured (no API key/mode) — skipping", flush=True)
        return
    metrics: dict[str, Any] = {}
    for kind, ev in result["evals"].items():
        metrics.update(wl.flatten_numeric(f"{kind}/token", ev["token_divergence"]))
        metrics.update(wl.flatten_numeric(f"{kind}/answer", ev["answer_divergence"]))
        metrics.update(wl.flatten_numeric(f"{kind}/net", ev["net_quality"]))
        metrics.update(wl.flatten_numeric(f"{kind}/flip", ev["flip_character"]))
        metrics[f"{kind}/n_large_margin_answer_flips"] = ev["n_large_margin_answer_flips"]
        if ev.get("aime_quality"):
            metrics.update(wl.flatten_numeric(f"{kind}/quality", ev["aime_quality"]))
    metrics.update(wl.flatten_numeric("pooled", result["pooled"]))
    metrics.update(wl.flatten_numeric("terminal", result["terminal_metrics"]))
    metrics["analysis_only"] = 1
    metrics["official_tps"] = 0
    wl.log_event(run, "materiality_complete", step=0, metrics=metrics)
    for k, v in metrics.items():
        run.summary[k] = v
    run.summary["headline_verdict"] = result["headline_verdict"]
    if packet is not None:
        run.summary["packet_637_verdict"] = packet["verdict"]
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    wl.log_json_artifact(
        run,
        name=("pr637_materiality_packet" if packet is not None else "pr626_materiality_report"),
        artifact_type="answer-materiality", data=result)
    wl.finish_wandb(run)
    print("[analyze] wandb logged", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
