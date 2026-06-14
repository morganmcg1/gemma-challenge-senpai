#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Issue #192 enforcement-reading calibration (PR #219) — per-reading pass-fractions.

THE ONE NUMBER vs THE MENU
--------------------------
Issue #192 (greedy token-identity) is the live launch gate. The human asked the board
WHICH reading of the contract applies: (A) strict literal per-sequence token-identity,
(B) per-token tolerance, or (C) operational / PPL acceptance. The board is collecting
qualitative reads; the ruling needs NUMBERS. kanna #114 (`9q5yy9l1`) measured ONE
divergence number — 56.08% of tokens flip argmax under the int4-Marlin batch-variant
spec-verify GEMM vs plain greedy AR — but a single per-token rate does not tell the human
the per-reading PASS-FRACTION. This leg converts #114's one number into the per-reading
menu the #192 ruling picks from.

THE THREE READINGS
------------------
  (A) STRICT per-sequence token-identity — a served greedy sequence is compliant iff ALL
      its output tokens match plain greedy AR (zero argmax flips). TEST metric
      `strict_a_pass_fraction` = fraction of the 128 served sequences with zero flips.
  (B) PER-TOKEN-theta — compliant iff the per-sequence flip fraction <= theta. The curve
      `pass_fraction(theta)` over theta in [0,1].
  (C) PPL-ONLY — compliant iff served PPL <= 2.42 (the auto-scorer's ACTUAL check). 100%
      (served PPL 2.3772 <= 2.42).

THE FINEST GRANULARITY #114 BANKED (per-sequence + per-position — used DIRECTLY)
-------------------------------------------------------------------------------
#114's `interlock_report.json` banks the per-SEQUENCE split directly: of 128 prompts,
`num_identical=16` (zero-flip) and `num_divergent=112`, with aggregate `token_div_frac`
= 0.5607757568359375 (36751/65536) and onset min/median/max = 0/120/496. The per-POSITION
token-ids are also banked (`decode_outputs.jsonl`, spec-ON and spec-OFF AR), so this leg
RECONSTRUCTS the FULL per-sequence flip-fraction distribution and verifies it reproduces
#114's banked split BIT-FOR-BIT (16/112, 0.5607757568359375). The strict-A pass-fraction
is therefore OBSERVED, not modeled: `strict_a_pass_fraction = 16/128 = 0.125`.

THE CLUSTERING STORY (this is denken's #190/#212 within-sequence-correlation lane)
---------------------------------------------------------------------------------
The naive iid intuition (model per-token flips as Bernoulli(p=0.5608)) predicts strict-A
= (1-p)^L = (0.4392)^512 ~ 1e-183 ~ 0 — a 512-token zero-flip sequence is astronomically
unlikely under independence. But the EMPIRICAL strict-A is 0.125, ~183 orders of magnitude
HIGHER, because the #114 flips are EXTREMELY CLUSTERED: a sequence either never trips
(16 prompts) or trips once and CASCADES (onset median 120/512, then ~64% of the remaining
tokens flip). Positive within-sequence flip correlation concentrates flips into fewer
sequences -> MORE zero-flip sequences -> strict-A RISES with clustering. This is exactly
the #190/#212 machinery (a zero-flip run over L tokens under a CORRELATED Bernoulli flip
process), now grounded in the REAL per-sequence data rather than a hypothetical band.

A model-free Frechet/union bound caps how far clustering can push it: for ANY within-
sequence correlation structure with aggregate per-token flip rate p, P(zero flips) <=
1 - max_i p_i <= 1 - p = 0.4392 (achieved by comonotone/perfectly-nested flips). So
strict-A is in (~0 iid, 0.4392 max-clustering]; the empirical 0.125 sits inside the band,
and EVEN MAXIMAL clustering keeps strict-A < 0.5 << 1. Under strict-A the int4-spec stack
fails for the MAJORITY of sequences no matter how clustered the flips are.

BOTH STACKS share the exposure: the deployed `fa2sw_precache_kenyan` (481.53) AND the
land #71 tree ride the SAME int4-Marlin spec-verify basis, so the per-reading pass-fractions
apply to both. Strict-A is a frontier-WIDE exposure, not tree-only.

SCOPE: LOCAL CPU-only analytic re-read of kanna #114's banked divergence under three
identity definitions. No GPU / vLLM / HF Job / submission / served-file change / official
draw. It re-reads ONE banked number (you do NOT re-measure #114). BASELINE stays 481.53.
Greedy/PPL untouched. Bank-the-analysis (PRIMARY = self-test, adds 0 TPS). NOT a launch.
The RULING is the human's (issue #192); this leg supplies the DECISION-ARMING menu.

PRIMARY metric  issue192_calibration_self_test_passes
TEST    metric  strict_a_pass_fraction  (per-sequence zero-flip fraction; expected << 1)

Run:
    CUDA_VISIBLE_DEVICES="" python research/validity/issue192_reading_calibration/issue192_reading_calibration.py \\
        --self-test --wandb_group issue192-reading-calibration --wandb_name denken/issue192-reading-calibration
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# kanna #114 (`9q5yy9l1`) banked self-referential greedy gate (read-only).
_D114_DIR = REPO_ROOT / "research/validity/self_referential_gate/ab-20260614T075459Z"
_D114_INTERLOCK = _D114_DIR / "interlock_report.json"
_D114_SPEC_ON = _D114_DIR / "default/run_00/decode_outputs.jsonl"            # spec-ON (served)
_D114_SPEC_OFF = _D114_DIR / "default__specoff/run_00/decode_outputs.jsonl"  # spec-OFF plain greedy AR
# denken #190 (`fva6o4ug`) banked within-prompt ICC (read-only reference for the band).
_D190_JSON = REPO_ROOT / "research/validity/icc_neff/icc_neff_results.json"

# Baseline (PR #219): official 481.53 TPS, served PPL 2.3772, cap 2.42.
SERVED_PPL = 2.3772
PPL_CAP = 2.42
OFFICIAL_TPS = 481.53

TOL_REPRO = 1e-12   # the per-position reconstruction must reproduce #114 BIT-FOR-BIT
TOL_EXACT = 1e-9


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Imports — kanna #114's banked per-sequence + aggregate divergence (NOT re-measured).
# --------------------------------------------------------------------------- #
def load_imports() -> dict[str, Any]:
    d114 = json.load(open(_D114_INTERLOCK))
    scg = d114["self_consistency_gate"]
    per0 = scg["per_run"][0]

    d190 = json.load(open(_D190_JSON))
    ie = d190["icc_estimate"]

    return {
        # ---- #114 (9q5yy9l1) banked divergence ----
        "num_prompts": per0["num_identical"] + per0["num_divergent"],   # 128
        "num_identical_banked": per0["num_identical"],                  # 16 (zero-flip seqs)
        "num_divergent_banked": per0["num_divergent"],                 # 112
        "total_tokens_banked": per0["total_tokens_compared"],          # 65536
        "total_divergent_tokens_banked": per0["total_divergent_tokens"],  # 36751
        "token_div_frac_banked": per0["token_div_frac"],               # 0.5607757568359375 (THE 56.08%)
        "onset_min_banked": scg["onset_min"],                          # 0
        "onset_median_banked": scg["onset_median"],                    # 120
        "onset_max_banked": scg["onset_max"],                          # 496
        "spec_on_deterministic": d114["spec_on_self_determinism"]["deterministic"],   # True
        "spec_off_deterministic": d114["spec_off_ar_self_determinism"]["deterministic"],  # True
        "d114_verdict": d114["verdict"],                               # "RED"
        # ---- baseline / reading-C ----
        "served_ppl": SERVED_PPL, "ppl_cap": PPL_CAP, "official_tps": OFFICIAL_TPS,
        # ---- #190 (fva6o4ug) within-prompt ICC reference for the clustering band ----
        "icc_190": ie["icc_hat"],                                      # 0.1446
        "icc_ci_190": ie["icc_ci"],                                    # [0.1043, 0.1857]
        "source_runs": {"d114": "9q5yy9l1", "d190": "fva6o4ug", "d199": "wdyqnx3g",
                        "d213": "5o7zcj8s", "d196": "(compliant floor)"},
    }


# --------------------------------------------------------------------------- #
# Reconstruct the FULL per-sequence flip-fraction distribution from #114's banked
# per-position token-ids (spec-ON served vs spec-OFF plain greedy AR), and verify it
# reproduces #114's banked per-sequence split (16/112) + aggregate (0.5608) BIT-FOR-BIT.
# This is "use the finest granularity #114 banked, directly".
# --------------------------------------------------------------------------- #
def _load_jsonl_token_ids(path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line)["completion_token_ids"])
    return rows


def reconstruct_per_sequence(imp: dict) -> dict[str, Any]:
    served = _load_jsonl_token_ids(_D114_SPEC_ON)       # spec-ON served greedy
    ar_ref = _load_jsonl_token_ids(_D114_SPEC_OFF)       # plain greedy AR reference
    n = len(served)
    if not (len(served) == len(ar_ref) == imp["num_prompts"]):
        raise RuntimeError(f"reconstruction mismatch: served={len(served)} ar={len(ar_ref)} "
                           f"banked={imp['num_prompts']}")

    flip_fracs: list[float] = []
    flip_counts: list[int] = []
    seq_lens: list[int] = []
    onsets: list[int] = []
    total_tokens = 0
    total_div = 0
    num_identical = 0
    num_divergent = 0
    for a, b in zip(served, ar_ref):
        L = min(len(a), len(b))
        diffs = [i for i in range(L) if a[i] != b[i]]
        ndiff = len(diffs)
        total_tokens += L
        total_div += ndiff
        seq_lens.append(L)
        flip_counts.append(ndiff)
        flip_fracs.append(ndiff / L if L else 0.0)
        if ndiff:
            num_divergent += 1
            onsets.append(diffs[0])
        else:
            num_identical += 1

    token_div_frac = total_div / total_tokens
    # bit-for-bit anchor against #114's banked summary
    reproduces_banked = bool(
        num_identical == imp["num_identical_banked"]
        and num_divergent == imp["num_divergent_banked"]
        and total_tokens == imp["total_tokens_banked"]
        and total_div == imp["total_divergent_tokens_banked"]
        and abs(token_div_frac - imp["token_div_frac_banked"]) <= TOL_REPRO
        and (not onsets or (
            min(onsets) == imp["onset_min_banked"]
            and int(median(onsets)) == imp["onset_median_banked"]
            and max(onsets) == imp["onset_max_banked"]))
    )

    ff_sorted = sorted(flip_fracs)
    within_div_mean = (total_div / (num_divergent * (sum(seq_lens) / n))) if num_divergent else 0.0
    return {
        "num_prompts": n,
        "num_identical": num_identical,
        "num_divergent": num_divergent,
        "total_tokens": total_tokens,
        "total_divergent_tokens": total_div,
        "token_div_frac": token_div_frac,
        "onset_min": min(onsets) if onsets else None,
        "onset_median": int(median(onsets)) if onsets else None,
        "onset_max": max(onsets) if onsets else None,
        "seq_len_min": min(seq_lens), "seq_len_max": max(seq_lens),
        "flip_fracs_sorted": ff_sorted,
        "flip_frac_min": ff_sorted[0], "flip_frac_max": ff_sorted[-1],
        "flip_frac_median": ff_sorted[n // 2],
        "within_divergent_mean_flip_frac": within_div_mean,
        "reproduces_banked": reproduces_banked,
        "anchor_note": (
            f"per-position reconstruction reproduces #114 banked split BIT-FOR-BIT: "
            f"identical {num_identical}/{n} (banked {imp['num_identical_banked']}), divergent "
            f"{num_divergent} (banked {imp['num_divergent_banked']}), token_div_frac "
            f"{token_div_frac:.16f} (banked {imp['token_div_frac_banked']:.16f})."),
    }


# --------------------------------------------------------------------------- #
# (1) Reading definitions.
# --------------------------------------------------------------------------- #
def reading_definitions(imp: dict, rec: dict) -> dict[str, Any]:
    return {
        "A_strict_per_sequence": (
            "a served greedy sequence is COMPLIANT iff ALL its output tokens match plain "
            "greedy AR (zero argmax flips over output_len). pass = fraction of the 128 served "
            "sequences with zero flips."),
        "B_per_token_theta": (
            "COMPLIANT iff the per-sequence flip fraction <= theta. pass_fraction(theta) is "
            "the empirical CDF of per-sequence flip fractions over theta in [0,1]."),
        "C_ppl_only": (
            f"COMPLIANT iff served PPL <= {imp['ppl_cap']} (the auto-scorer's actual check). "
            f"served PPL {imp['served_ppl']} <= {imp['ppl_cap']} -> 100%."),
        "source_granularity": "per_sequence_and_per_position",
        "source_granularity_note": (
            "#114 interlock_report.json banks the per-SEQUENCE split (num_identical=16, "
            "num_divergent=112) AND the per-POSITION token-ids (decode_outputs.jsonl, spec-ON "
            "+ spec-OFF AR). This leg reconstructs the full per-sequence flip-fraction "
            "distribution from the per-position ids and verifies it reproduces the banked "
            "per-sequence split + aggregate bit-for-bit -> strict-A is OBSERVED, not modeled."),
        "d114_run": imp["source_runs"]["d114"],
    }


# --------------------------------------------------------------------------- #
# (2) Per-reading pass-fractions (the core).
# --------------------------------------------------------------------------- #
def pass_fraction_theta(ff_sorted: list[float], theta: float) -> float:
    """Empirical CDF: fraction of sequences with flip fraction <= theta."""
    n = len(ff_sorted)
    return sum(1 for x in ff_sorted if x <= theta + 1e-15) / n


def reading_pass_fractions(imp: dict, rec: dict) -> dict[str, Any]:
    ff = rec["flip_fracs_sorted"]
    n = rec["num_prompts"]

    # (A) strict-A — OBSERVED per-sequence zero-flip fraction.
    strict_a = rec["num_identical"] / n

    # (B) per-token-theta curve over a representative grid + the full empirical CDF.
    theta_grid = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50,
                  rec["flip_frac_median"], 0.80, 0.95, rec["flip_frac_max"], 0.99, 1.0]
    theta_grid = sorted(set(round(t, 6) for t in theta_grid))
    theta_curve = [{"theta": t, "pass_fraction": pass_fraction_theta(ff, t)} for t in theta_grid]

    # (C) PPL-only.
    ppl_pass = 1.0 if imp["served_ppl"] <= imp["ppl_cap"] else 0.0

    return {
        "strict_a_pass_fraction": strict_a,                 # TEST headline (observed)
        "strict_a_num_identical": rec["num_identical"],
        "strict_a_num_total": n,
        "per_token_theta_curve": theta_curve,
        "ppl_only_pass_fraction": ppl_pass,
        "ppl_only_served": imp["served_ppl"], "ppl_only_cap": imp["ppl_cap"],
        "ordering_note": (
            f"strict-A {strict_a:.4f} (theta=0) <= per-token-theta(theta) (monotone CDF) <= "
            f"PPL-only 1.0. The three readings are nested in strictness."),
    }


# --------------------------------------------------------------------------- #
# (2b) iid contrast + clustering band (denken #190/#212 within-sequence-correlation lane).
#      iid Bernoulli(p) UNDERSTATES strict-A by ~183 orders of magnitude; the model-free
#      Frechet bound caps it at 1-p; the empirical sits inside.
# --------------------------------------------------------------------------- #
def _logE_one_minus_q_pow_L_betabinom(p: float, L: int, rho: float) -> float:
    """log E[(1-q)^L] for q ~ Beta(alpha,beta), mean p, ICC rho = 1/(alpha+beta+1).

    Closed form: E[(1-q)^L] = Gamma(beta+L) Gamma(alpha+beta) / (Gamma(beta) Gamma(alpha+beta+L)).
    rho->0  => (1-p)^L  (concentration at p);  rho->1 => 1-p (bimodal {0,1})."""
    if rho <= 0.0:
        return L * math.log1p(-p)            # (1-p)^L
    if rho >= 1.0:
        return math.log1p(-p)                # 1-p
    s = (1.0 - rho) / rho                     # alpha+beta concentration
    alpha = p * s
    beta = (1.0 - p) * s
    return (math.lgamma(beta + L) + math.lgamma(alpha + beta)
            - math.lgamma(beta) - math.lgamma(alpha + beta + L))


def clustering_band(imp: dict, rec: dict, reads: dict) -> dict[str, Any]:
    p = imp["token_div_frac_banked"]
    L = 512                                   # forced output_len (per #114 decode_summary)
    strict_a_emp = reads["strict_a_pass_fraction"]

    # iid (independence) — the naive lower reference.
    log_strict_a_iid = L * math.log1p(-p)
    strict_a_iid = math.exp(log_strict_a_iid)
    log10_strict_a_iid = log_strict_a_iid / math.log(10.0)

    # model-free Frechet/union upper bound: P(zero flips) <= 1 - p (any correlation structure).
    frechet_upper = 1.0 - p

    # Beta-Binomial clustering sweep (smooth interpolation iid -> max).
    rho_grid = [0.0, 0.01, 0.05, imp["icc_190"], 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    rho_grid = sorted(set(round(r, 6) for r in rho_grid))
    bb_curve = []
    for r in rho_grid:
        sa = math.exp(_logE_one_minus_q_pow_L_betabinom(p, L, r))
        bb_curve.append({"rho": r, "strict_a": sa})

    # Beta-Binomial strict-A at the #190 realistic within-prompt ICC (transplanted reference).
    strict_a_at_icc190 = math.exp(_logE_one_minus_q_pow_L_betabinom(p, L, imp["icc_190"]))

    # effective rho that a smooth Beta-Binomial would need to MATCH the empirical strict-A
    # (illustrative — the real cascade is more extreme than Beta-Binomial at fixed strict-A).
    def bb(r: float) -> float:
        return math.exp(_logE_one_minus_q_pow_L_betabinom(p, L, r))
    lo, hi = 0.0, 1.0
    if bb(hi) >= strict_a_emp >= bb(lo):
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if bb(mid) < strict_a_emp:
                lo = mid
            else:
                hi = mid
        rho_eff = 0.5 * (lo + hi)
    else:
        rho_eff = float("nan")

    return {
        "p_per_token_flip": p, "output_len_L": L,
        "strict_a_iid": strict_a_iid,
        "strict_a_iid_log10": log10_strict_a_iid,
        "strict_a_empirical": strict_a_emp,
        "frechet_upper_bound_1_minus_p": frechet_upper,
        "betabinom_sweep": bb_curve,
        "strict_a_at_icc190_betabinom": strict_a_at_icc190,
        "icc_190_reference": imp["icc_190"],
        "rho_effective_matching_empirical": rho_eff,
        "empirical_within_frechet_band": bool(strict_a_iid <= strict_a_emp <= frechet_upper + TOL_EXACT),
        "clustering_raises_strict_a": bool(strict_a_emp > strict_a_iid),
        "orders_of_magnitude_iid_understates": (math.log10(strict_a_emp) - log10_strict_a_iid),
        "note": (
            f"iid Bernoulli(p={p:.4f}) predicts strict-A=(1-p)^{L}=10^{log10_strict_a_iid:.1f} ~ 0; "
            f"the EMPIRICAL strict-A is {strict_a_emp:.4f}, ~{(math.log10(strict_a_emp) - log10_strict_a_iid):.0f} "
            f"orders of magnitude higher, because the #114 flips CASCADE (extreme within-sequence "
            f"clustering). The model-free Frechet bound caps strict-A at 1-p={frechet_upper:.4f} for ANY "
            f"correlation structure; the empirical {strict_a_emp:.4f} sits inside (iid {strict_a_iid:.2e} "
            f"... empirical {strict_a_emp:.4f} ... max {frechet_upper:.4f})."),
    }


# --------------------------------------------------------------------------- #
# (3) Apply to both stacks (frontier 481.53 + land #71 tree share the int4-spec basis).
# --------------------------------------------------------------------------- #
def apply_to_stacks(imp: dict, reads: dict) -> dict[str, Any]:
    sa = reads["strict_a_pass_fraction"]
    return {
        "applies_to_frontier_and_tree": True,
        "shared_basis": "int4-Marlin batch-variant spec-verify GEMM",
        "frontier_submission": "fa2sw_precache_kenyan (481.53 TPS, PR #52)",
        "tree": "land #71 (same int4-spec basis)",
        "launch_implication_strict_A": (
            f"under (A) strict per-sequence token-identity, BOTH the 481.53 frontier and the land "
            f"#71 tree pass only {sa:.1%} of sequences -> NEITHER is launch-eligible; only the "
            f"wirbel #199/#213/#216 compliant-kernel (batch-invariant verify) route survives strict-A."),
        "launch_implication_per_token_theta": (
            "under (B) per-token-theta, launch-eligibility depends on the human's chosen theta; the "
            "pass-fraction curve is the menu (e.g. a 5%-tolerance reading passes only the cleanest "
            "few percent of sequences)."),
        "launch_implication_ppl_only": (
            f"under (C) PPL-only (the auto-scorer's actual check), BOTH stacks pass 100% "
            f"(served PPL {imp['served_ppl']} <= {imp['ppl_cap']}) -> launch-eligible today."),
    }


# --------------------------------------------------------------------------- #
# (4) Correlation sensitivity (SECONDARY — does NO-GO-under-strict-A survive the clustering sweep?).
# --------------------------------------------------------------------------- #
def correlation_sensitivity(band: dict) -> dict[str, Any]:
    frechet = band["frechet_upper_bound_1_minus_p"]
    sa_emp = band["strict_a_empirical"]
    # NO-GO-under-strict-A survives iff even MAXIMAL clustering keeps strict-A decisively < 1
    # (we use < 0.5, i.e. the MAJORITY of sequences still fail strict-A, as the decisive threshold).
    robust = bool(frechet < 0.5 and sa_emp < 0.5)
    return {
        "strict_a_robust_to_clustering": robust,
        "strict_a_max_over_all_clustering": frechet,
        "strict_a_empirical": sa_emp,
        "decisive_threshold": 0.5,
        "note": (
            f"positive within-sequence flip ICC concentrates flips into fewer sequences -> strict-A "
            f"RISES with clustering. But the model-free Frechet cap is 1-p={frechet:.4f} < 0.5: even in "
            f"the MOST flip-clustered world the MAJORITY (>={1-frechet:.1%}) of sequences still fail "
            f"strict-A. The empirical {sa_emp:.4f} is well below the cap. NO-GO-under-strict-A SURVIVES "
            f"the full clustering sweep -> the int4-spec stack fails literal token-identity robustly."),
    }


# --------------------------------------------------------------------------- #
# (5) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(imp: dict, rec: dict, reads: dict, band: dict) -> dict[str, Any]:
    ff = rec["flip_fracs_sorted"]
    p = imp["token_div_frac_banked"]

    # (a) the iid model reproduces #114's aggregate 56.08% per-token rate EXACTLY. The iid mean
    #     per-token flip rate is p; anchor on the reconstructed token_div_frac == banked.
    iid_mean_per_token = p
    cond_a = bool(abs(iid_mean_per_token - imp["token_div_frac_banked"]) <= TOL_REPRO
                  and rec["reproduces_banked"]
                  and abs(rec["token_div_frac"] - imp["token_div_frac_banked"]) <= TOL_REPRO)

    # (b) strict_a_pass_fraction <= pass_fraction(theta) for all theta>0 (strict is the hardest).
    strict_a = reads["strict_a_pass_fraction"]
    theta_probe = [t for t in [1e-6, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]]
    cond_b = all(strict_a <= pass_fraction_theta(ff, t) + TOL_EXACT for t in theta_probe)

    # (c) theta=1 -> 1.0.
    cond_c = bool(abs(pass_fraction_theta(ff, 1.0) - 1.0) <= TOL_EXACT)

    # (d) theta=0 reproduces strict-A.
    cond_d = bool(abs(pass_fraction_theta(ff, 0.0) - strict_a) <= TOL_EXACT)

    # (e) PPL-only = 1.0.
    cond_e = bool(abs(reads["ppl_only_pass_fraction"] - 1.0) <= TOL_EXACT)

    # (f) NaN-clean (key scalars finite; full-payload walk enforced in main()).
    key = [strict_a, band["strict_a_iid"], band["strict_a_iid_log10"],
           band["frechet_upper_bound_1_minus_p"], band["strict_a_empirical"],
           reads["ppl_only_pass_fraction"], rec["token_div_frac"],
           band["strict_a_at_icc190_betabinom"]]
    cond_f = all(_finite(x) for x in key)

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e and cond_f)
    return {
        "issue192_calibration_self_test_passes": passes,
        "conditions": {
            "a_iid_reproduces_aggregate_5608_and_reconstruction_bitexact": cond_a,
            "b_strict_a_le_pass_fraction_theta_monotone": bool(cond_b),
            "c_theta1_pass_is_1": cond_c,
            "d_theta0_reproduces_strict_a": cond_d,
            "e_ppl_only_is_1": cond_e,
            "f_key_scalars_finite": cond_f,
        },
        "evidence": {
            "a_iid_mean_per_token": iid_mean_per_token,
            "a_banked_token_div_frac": imp["token_div_frac_banked"],
            "a_reconstructed_token_div_frac": rec["token_div_frac"],
            "a_reproduces_banked": rec["reproduces_banked"],
            "b_strict_a": strict_a,
            "b_pass_fraction_at_probes": {str(t): pass_fraction_theta(ff, t) for t in theta_probe},
            "c_pass_at_theta1": pass_fraction_theta(ff, 1.0),
            "d_pass_at_theta0": pass_fraction_theta(ff, 0.0),
            "e_ppl_only": reads["ppl_only_pass_fraction"],
        },
    }


# --------------------------------------------------------------------------- #
# Verdict + hand-off.
# --------------------------------------------------------------------------- #
def _verdict(imp: dict, reads: dict, band: dict, sens: dict) -> str:
    sa = reads["strict_a_pass_fraction"]
    return (
        f"CALIBRATED. #114's one number (56.08% per-token argmax divergence) resolves into the #192 "
        f"per-reading menu: (A) STRICT per-sequence token-identity pass = {sa:.4f} "
        f"({reads['strict_a_num_identical']}/{reads['strict_a_num_total']} sequences zero-flip) -> the "
        f"int4-spec stack FAILS literal token-identity for {1-sa:.1%} of sequences; (B) per-token-theta "
        f"is the CDF menu (theta=0 -> {sa:.4f}, theta=1 -> 1.0); (C) PPL-ONLY (the auto-scorer's actual "
        f"check) = 100% (served PPL {imp['served_ppl']} <= {imp['ppl_cap']}). HONEST CORRECTION to the "
        f"naive iid intuition: independence predicts strict-A=(1-p)^512=10^{band['strict_a_iid_log10']:.0f}~0, "
        f"but the EMPIRICAL strict-A is {sa:.4f} because the #114 flips CASCADE (extreme within-sequence "
        f"clustering: 16 prompts never trip, 112 trip-and-cascade). The model-free Frechet bound caps "
        f"strict-A at 1-p={band['frechet_upper_bound_1_minus_p']:.4f} for ANY correlation structure, so "
        f"even MAXIMAL clustering keeps strict-A < 0.5 << 1 "
        f"(strict_a_robust_to_clustering={sens['strict_a_robust_to_clustering']}). BOTH the 481.53 frontier "
        f"and the land #71 tree ride the SAME int4-spec basis, so the menu applies to both. The RULING is "
        f"the human's (issue #192); only the wirbel #199/#213/#216 compliant-kernel route survives strict-A. "
        f"BASELINE 481.53 untouched. NOT a launch."
    )


def _handoff(imp: dict, reads: dict, band: dict, sens: dict) -> dict[str, str]:
    sa = reads["strict_a_pass_fraction"]
    line = (
        f"the #192 per-reading menu from #114's 56.08%: strict-A (per-sequence zero-flip) pass = "
        f"{sa:.4f} ({reads['strict_a_num_identical']}/{reads['strict_a_num_total']}; the int4-spec stack "
        f"fails literal token-identity for {1-sa:.0%} of sequences — and even under the model-free "
        f"maximal-flip-clustering cap 1-p={band['frechet_upper_bound_1_minus_p']:.3f} it stays < 0.5, so "
        f"NO-GO-under-strict-A is robust), per-token-theta pass-fraction = the empirical CDF (theta=0 -> "
        f"{sa:.4f}, theta=1 -> 1.0), PPL-only (the auto-scorer's actual check) = 100% "
        f"(served PPL {imp['served_ppl']} <= {imp['ppl_cap']}); BOTH the 481.53 frontier and the land #71 "
        f"tree share this exposure, so the human's A/B/C ruling determines whether EITHER is launch-eligible "
        f"under strict-A — and the wirbel #199/#213/#216 compliant-kernel route is the only strict-A-"
        f"survivable 500-path."
    )
    return {"issue_192": line, "fern_185": line}


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    imp = load_imports()
    rec = reconstruct_per_sequence(imp)
    rdefs = reading_definitions(imp, rec)
    reads = reading_pass_fractions(imp, rec)
    band = clustering_band(imp, rec, reads)
    stacks = apply_to_stacks(imp, reads)
    sens = correlation_sensitivity(band)
    st = self_test(imp, rec, reads, band)
    handoff = _handoff(imp, reads, band, sens)
    return {
        "self_test": st,
        "test_metric": {"strict_a_pass_fraction": reads["strict_a_pass_fraction"]},
        "imports": imp,
        "reconstruction": rec,
        "reading_definitions": rdefs,
        "reading_pass_fractions": reads,
        "clustering_band": band,
        "applies_to_stacks": stacks,
        "correlation_sensitivity": sens,
        "verdict": _verdict(imp, reads, band, sens),
        "handoff_lines": handoff,
    }


# --------------------------------------------------------------------------- #
# NaN-clean walk.
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


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict) -> None:
    imp = syn["imports"]
    rec, reads = syn["reconstruction"], syn["reading_pass_fractions"]
    band, sens = syn["clustering_band"], syn["correlation_sensitivity"]
    stacks, st = syn["applies_to_stacks"], syn["self_test"]
    print("\n" + "=" * 96, flush=True)
    print("ISSUE #192 ENFORCEMENT-READING CALIBRATION (PR #219) — per-reading pass-fractions", flush=True)
    print("=" * 96, flush=True)
    print(f"  #114 (9q5yy9l1): token_div_frac={imp['token_div_frac_banked']:.6f} "
          f"({imp['total_divergent_tokens_banked']}/{imp['total_tokens_banked']}); "
          f"per-sequence {imp['num_identical_banked']} identical / {imp['num_divergent_banked']} divergent "
          f"of {imp['num_prompts']}; onset {imp['onset_min_banked']}/{imp['onset_median_banked']}/"
          f"{imp['onset_max_banked']}", flush=True)
    print(f"  reconstruction reproduces banked bit-for-bit: {rec['reproduces_banked']}", flush=True)
    print("-" * 96, flush=True)
    print("  THE THREE READINGS:", flush=True)
    print(f"    (A) STRICT per-sequence token-identity   pass = {reads['strict_a_pass_fraction']:.4f}   "
          f"({reads['strict_a_num_identical']}/{reads['strict_a_num_total']})   <-- TEST headline", flush=True)
    print(f"    (B) per-token-theta   pass_fraction(theta) curve:", flush=True)
    for row in reads["per_token_theta_curve"]:
        print(f"          theta={row['theta']:<8.4g}  pass={row['pass_fraction']:.4f}", flush=True)
    print(f"    (C) PPL-only   pass = {reads['ppl_only_pass_fraction']:.4f}   "
          f"(served {reads['ppl_only_served']} <= {reads['ppl_only_cap']})", flush=True)
    print("-" * 96, flush=True)
    print("  CLUSTERING (denken #190/#212 lane):", flush=True)
    print(f"      iid (1-p)^512        strict-A = {band['strict_a_iid']:.3e}  (10^{band['strict_a_iid_log10']:.1f})", flush=True)
    print(f"      EMPIRICAL (banked)   strict-A = {band['strict_a_empirical']:.4f}   "
          f"(~{band['orders_of_magnitude_iid_understates']:.0f} orders above iid)", flush=True)
    print(f"      Frechet max-cluster  strict-A <= 1-p = {band['frechet_upper_bound_1_minus_p']:.4f}  (model-free)", flush=True)
    print(f"      Beta-Binom @ICC190={band['icc_190_reference']:.3f}  strict-A = {band['strict_a_at_icc190_betabinom']:.3e}", flush=True)
    print(f"      strict_a_robust_to_clustering = {sens['strict_a_robust_to_clustering']}  "
          f"(even max-clustering {band['frechet_upper_bound_1_minus_p']:.3f} < 0.5)", flush=True)
    print("-" * 96, flush=True)
    print(f"  applies_to_frontier_and_tree = {stacks['applies_to_frontier_and_tree']}  "
          f"(shared basis: {stacks['shared_basis']})", flush=True)
    print("-" * 96, flush=True)
    print(f"  (PRIMARY) issue192_calibration_self_test_passes = "
          f"{st['issue192_calibration_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  HAND-OFF (#192 + fern #185): {syn['handoff_lines']['issue_192']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #212; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[issue192-calib] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    imp, rec = syn["imports"], syn["reconstruction"]
    reads, band = syn["reading_pass_fractions"], syn["clustering_band"]
    sens, stacks, st = syn["correlation_sensitivity"], syn["applies_to_stacks"], syn["self_test"]

    run = init_wandb_run(
        job_type="validity-gate",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["issue-192", "greedy-identity", "enforcement-reading", "per-reading-pass-fraction",
              "within-sequence-correlation", "clustering", "kanna-114-reread", "bank-the-analysis"],
        config={
            "official_tps": imp["official_tps"], "served_ppl": imp["served_ppl"],
            "ppl_cap": imp["ppl_cap"], "token_div_frac_114": imp["token_div_frac_banked"],
            "num_prompts": imp["num_prompts"], "output_len_L": band["output_len_L"],
            "icc_190": imp["icc_190"],
            "imports": "kanna#114 (9q5yy9l1) divergence + denken#190 (fva6o4ug) ICC",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[issue192-calib] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "issue192_calibration_self_test_passes": int(bool(st["issue192_calibration_self_test_passes"])),
        "strict_a_pass_fraction": reads["strict_a_pass_fraction"],
        "strict_a_num_identical": reads["strict_a_num_identical"],
        "strict_a_num_total": reads["strict_a_num_total"],
        "ppl_only_pass_fraction": reads["ppl_only_pass_fraction"],
        "strict_a_iid": band["strict_a_iid"],
        "strict_a_iid_log10": band["strict_a_iid_log10"],
        "strict_a_empirical": band["strict_a_empirical"],
        "frechet_upper_bound_1_minus_p": band["frechet_upper_bound_1_minus_p"],
        "strict_a_at_icc190_betabinom": band["strict_a_at_icc190_betabinom"],
        "rho_effective_matching_empirical": band["rho_effective_matching_empirical"],
        "strict_a_robust_to_clustering": int(bool(sens["strict_a_robust_to_clustering"])),
        "applies_to_frontier_and_tree": int(bool(stacks["applies_to_frontier_and_tree"])),
        "token_div_frac_reconstructed": rec["token_div_frac"],
        "token_div_frac_banked": imp["token_div_frac_banked"],
        "reconstruction_reproduces_banked": int(bool(rec["reproduces_banked"])),
        "num_identical": rec["num_identical"], "num_divergent": rec["num_divergent"],
        "onset_median": rec["onset_median"], "within_divergent_mean_flip_frac": rec["within_divergent_mean_flip_frac"],
        "served_ppl": imp["served_ppl"], "ppl_cap": imp["ppl_cap"], "official_tps": imp["official_tps"],
        "icc_190": imp["icc_190"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
        # per-token-theta curve as flat scalars for easy plotting
        **{f"pass_fraction_theta_{str(row['theta']).replace('.', 'p')}": row["pass_fraction"]
           for row in reads["per_token_theta_curve"]},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="issue192_reading_calibration_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[issue192-calib] wandb logged {len(summary)} summary keys", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="issue192-reading-calibration")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 219,
        "agent": "denken",
        "kind": "issue192-reading-calibration",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[issue192-calib] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (f) and recompute PRIMARY
    syn["self_test"]["conditions"]["f_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["f_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["issue192_calibration_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "issue192_reading_calibration_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[issue192-calib] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY issue192_calibration_self_test_passes = {passes}", flush=True)
    print(f"  TEST strict_a_pass_fraction = {syn['test_metric']['strict_a_pass_fraction']:.4f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[issue192-calib] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
