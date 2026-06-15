#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #321 (student stark) -- Does the 586 EAGLE-3 private-500 projection survive a 47.7% haircut?

THE STRESS TEST
---------------
stark #298 (xp974x58) MEASURED that a *composed* step-side kernel gain realizes at only
realization_ratio_487 = 0.4769 on the host-to-host wall (the banked 487.7 free step ceiling
does NOT realize: free_ceiling_realizes_on_wall = False). The EAGLE-3 GREEN private-500 verdict
(fern #310, 2u3kcnv5: private 586.08 @ E[T]=6.11, +17.2%) is a *composed* E[T] raise. If the
EAGLE-3 raise suffered the same ~47.7% realization haircut, 586 could collapse below the 500
private bar. This card determines the CORRECT realization model for an E[T]-side raise and
stress-tests the 586 headline against stark's own measured 47.7% step-side ratio.

THE MECHANICAL DISTINCTION (the crux the PR asks us to nail)
-----------------------------------------------------------
The composition law is  official_TPS = K_cal * (E[T] / step) * tau .

  * stark #298's lever is a STEP-SIDE / DENOMINATOR lever (verify SDPA num_stages 3->2 shaves
    ~15.48 us off the kernel). On the wall, TPS = tokens / (kernel + FIXED_overhead). A kernel-us
    saving Delta shrinks the wall step by Delta, but the FRACTIONAL gain is Delta/(kernel+fixed)
    << Delta/kernel: the large FIXED serving overhead (host scheduling, Python round-trips,
    sampling, detok) does NOT shrink -> the gain is DILUTED to 47.7% realization.

  * EAGLE-3's lever is an E[T]-SIDE / NUMERATOR lever (more accepted tokens per verify). On the
    wall, TPS = E[T]_accepted / T_macrostep. Raising E[T] by factor f with T_macrostep fixed
    raises TPS by *exactly* f. The fixed overhead lives in the DENOMINATOR (amortised over MORE
    tokens) -- it does NOT dilute a numerator multiply. A pure E[T] numerator raise realises at
    ~100%, NOT 47.7%.

  * EAGLE-3's ONE denominator cost -- its heavier draft inflates T_macrostep from the deployed
    1218.2 us to step_central 1499.13 us -- is ALREADY priced INTO the 586 projection, and it is
    priced with the denken #278 bridge: step_central = new_step + bridge(0.2147) * draft_wall_delta
    (eagle3_draft_norm_us = wall_delta * bridge). The bridge IS the draft-side wall-realisation
    accounting (the 4.82x batch=1 over-credit). It is draft-side-specific: verify-side bridge = 1.0
    (kanna #286).

VERDICT (computed below): the bridge ALREADY captures the E[T]-side (draft-denominator)
realisation loss; there is NO additional 47.7%-style wall haircut on the E[T] NUMERATOR, because a
numerator raise is not a time-slice shrink and is not subject to the fixed-overhead dilution that
produced #298's 0.4769. stark #298's 47.7% is therefore the WRONG reference for the E[T]-side
lever; the E[T] numerator realises at a categorically HIGHER rate. Even when the (wrong, harsher)
47.7% step-side ratio is applied to the residual E[T] raise, 586 -> ~517-521 TPS, STILL clears 500.

DELIVERABLES
  1. Decompose 586 into the part already discounted by the bridge (draft denominator) vs the
     residual un-bridged E[T] numerator. Robustness across the banked bridge/step band.
  2. Residual realisation-ratio break-even: rho_real on the residual E[T] raise where 586 -> 500.
  3. State whether 47.7% (step-side) is the right reference for an E[T]-side lever (it is not).
  4. Headroom margin (586 - 500 = 86 TPS, +17.2%): how much realisation loss it can absorb.

SCOPE. LOCAL CPU-only analytic robustness closer over banked MERGED constants. 0 TPS added;
BASELINE 481.53 untouched; greedy/PPL untouched. NO GPU / vLLM / HF Job / submission /
served-file change. Authorises NOTHING. NOT a launch. Reachability of E[T]=6.11 and the OOD
private factor are upstream (fern #310 / go-card #305) and OUT OF SCOPE here.

PRIMARY metric  realization_haircut_survives_self_test_passes
TEST    metric  private_tps_at_worstcase_realization (float)

Run:
    cd target/ && .venv/bin/python \\
        research/validity/eagle3_realization_haircut/eagle3_realization_haircut.py \\
        --wandb_group eagle3-realization --wandb_name stark/eagle3-realization-haircut
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
V = REPO_ROOT / "research" / "validity"

# Source banked runs (import-not-rederive). Each constant cites file + JSON path.
RECONCILE = V / "eagle3_private_perposition_reconcile/eagle3_private_perposition_reconcile_results.json"  # fern #310
STEP_PROFILE = V / "eagle3_step_profile/eagle3_step_profile_results.json"        # wirbel #295
GO_CARD = V / "eagle3_go_card/eagle3_go_card_results.json"                       # fern #305
AB298 = V / "free_ceiling_wallclock_realize/ab_out/results.json"                 # stark #298 (xp974x58)

TARGET = 500.0
IMPORT_TOL = 1e-6      # self-test: every imported float constant matches its source to <=1e-6
REPRO_TOL = 1e-6       # self-test: reproduce the banked 586.08 / 622.08 to <=1e-6


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dig(d: Any, path: str) -> Any:
    cur = d
    for seg in path.strip("/").split("/"):
        cur = cur[int(seg)] if isinstance(cur, list) else cur[seg]
    return cur


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


# --------------------------------------------------------------------------- #
# Step 1 -- import every banked constant + self-verify to <=1e-6.
# --------------------------------------------------------------------------- #
def load_constants() -> dict[str, Any]:
    # (cited value, source file, source JSON path). The loader reads the source and the self-test
    # asserts |cited - source| <= 1e-6; the SOURCE value is used downstream.
    spec: dict[str, tuple[float, Path, str]] = {
        # --- the projection under stress (fern #310, 2u3kcnv5) ---
        "private_586": (586.0766391463308, RECONCILE, "/private_tps_at_611_perposition"),
        "honest_public_622": (622.080888, RECONCILE, "/reconcile_honest_single_tax/honest_public_at_611"),
        "margin_tps_310": (86.07663914633076, RECONCILE, "/reconcile_honest_single_tax/perposition_margin_tps"),
        "margin_pct_310": (17.21532782926615, RECONCILE, "/reconcile_honest_single_tax/perposition_margin_pct"),
        "breakeven_rho_ood": (0.8037539966988988, RECONCILE, "/reconcile_honest_single_tax/breakeven_rho"),
        "rho_priv_e3": (0.9421228821714434, RECONCILE, "/constants/rho_priv_e3"),
        "K_cal": (125.268, RECONCILE, "/constants/K_cal"),
        "K_cal_eagle3": (101.79332412387028, RECONCILE, "/constants/K_cal_eagle3"),
        "step_us": (1218.2, RECONCILE, "/constants/step_us"),
        "step_central_us": (1499.1305069702046, RECONCILE, "/constants/step_central_us"),
        "E_T_central": (6.1112149873699195, RECONCILE, "/constants/E_T_central"),
        "E_T_deployed": (3.844, RECONCILE, "/constants/E_T_deployed"),
        "official_baseline": (481.53, RECONCILE, "/constants/official_baseline"),
        "private_verified": (460.85, RECONCILE, "/constants/private_verified"),
        "private_factor_scalar": (0.804, RECONCILE, "/constants/private_factor"),
        # --- the bridge + banked step band (wirbel #295 / denken #278) ---
        "bridge_draft": (0.2147122962556323, STEP_PROFILE, "/synthesis/constants/bridge_draft"),
        "new_step_us": (1202.7171244939168, STEP_PROFILE, "/synthesis/constants/new_step_us"),
        "step_faithful_us": (1315.7371300068355, STEP_PROFILE, "/synthesis/collapse_faithful/eagle3_step_measured_us"),
        "step_additive_us": (1682.5238839335736, STEP_PROFILE, "/synthesis/collapse_additive/eagle3_step_measured_us"),
        "eagle3_draft_wall_us": (1233.2343231724515, STEP_PROFILE, "/synthesis/collapse_faithful/eagle3_draft_wall_us"),
        "eagle3_draft_norm_us": (113.0200055129188, STEP_PROFILE, "/synthesis/collapse_faithful/eagle3_draft_norm_us"),
        # --- stark #298 step-side realization (xp974x58) -- the reference being stress-tested ---
        "rr487_stepside": (0.47691696793341565, AB298, "/aggregate/realization_ratio_487"),
        "composed_dpct_298": (1.287324774110809, AB298, "/aggregate/composed_delta_pct"),
        "realized_dpct_298": (0.6139470280144962, AB298, "/aggregate/realized_delta_pct_wall"),
        "s3_wall_298": (454.05771252324206, AB298, "/aggregate/s3_pooled_p50_wall_tps"),
        # --- go-card #305 scalar double-tax anchor (context) ---
        "go305_private_402": (402.0, GO_CARD, "/deterministic_card/private_at_central_tps"),
    }
    cache: dict[Path, dict] = {}
    out: dict[str, Any] = {}
    verify: list[dict[str, Any]] = []
    for name, (cited, path, jpath) in spec.items():
        if path not in cache:
            cache[path] = _load(path)
        src_val = float(_dig(cache[path], jpath))
        err = abs(src_val - cited)
        out[name] = src_val
        verify.append({
            "name": name, "cited": cited, "source_value": src_val, "abs_err": err,
            "matches_source": bool(err <= IMPORT_TOL),
            "source": str(path.relative_to(REPO_ROOT)) + "#" + jpath,
        })
    out["_verify"] = verify
    out["_all_match"] = bool(all(v["matches_source"] for v in verify))

    # non-float context (imported, not verified by the float path).
    ab = cache[AB298]["aggregate"]
    out["free_ceiling_realizes_on_wall_298"] = bool(ab["free_ceiling_realizes_on_wall"])
    out["classification_298"] = str(ab["classification"])

    out["bridge_overcredit"] = 1.0 / out["bridge_draft"]   # 4.658x (PR/denken #278 cite ~4.82x)
    out["provenance"] = (
        "fern#310 (2u3kcnv5) private 586.08 / honest_public 622.08 / margin 86.08 (+17.2%) / "
        "rho_priv_e3 0.9421 / breakeven_rho_ood 0.8038 x wirbel#295 (c334qaqu) K_cal 125.268 / "
        "step_us 1218.2 / step_central 1499.13 / bridge_draft 0.21471 / step band [faithful "
        "1315.74, additive 1682.52] x fern#305 (m4nmtdl9) scalar double-tax private 402.0 x "
        "stark#298 (xp974x58) step-side realization_ratio_487 0.4769 / free_ceiling_realizes False.")
    return out


# --------------------------------------------------------------------------- #
# Step 2 -- reproduce the banked 586.08 / 622.08 (round-trip <=1e-6).
# --------------------------------------------------------------------------- #
def reproduce(C: dict) -> dict[str, Any]:
    # honest_public(6.11) = K_cal * E[T] * (step_us / step_central)  -- bridge-discounted step baked in.
    hp = C["K_cal"] * C["E_T_central"] * (C["step_us"] / C["step_central_us"])
    private_586 = C["rho_priv_e3"] * hp
    # The E[T] numerator at the DEPLOYED step (no draft inflation): the bridge's "free draft" corner.
    public_numerator_only = C["K_cal"] * C["E_T_central"]                    # 765.63
    out = {
        "honest_public_611": hp,
        "private_586_reproduced": private_586,
        "public_numerator_only_deployed_step": public_numerator_only,
        "resid_honest_public": abs(hp - C["honest_public_622"]),
        "resid_private_586": abs(private_586 - C["private_586"]),
    }
    out["reproduces_586"] = bool(out["resid_honest_public"] <= REPRO_TOL * max(1.0, hp)
                                 and out["resid_private_586"] <= REPRO_TOL * max(1.0, private_586))
    return out


# --------------------------------------------------------------------------- #
# Step 3 (DELIVERABLE 1) -- decompose 586 into bridge-discounted vs residual; bridge-band robustness.
# --------------------------------------------------------------------------- #
def decompose(C: dict) -> dict[str, Any]:
    et = C["E_T_central"]
    hp = C["honest_public_622"]
    rho = C["rho_priv_e3"]

    # The 586 path from the deployed frontier, factor by factor:
    #   481.53 --[E[T] numerator x1.5897]--> 765.63 --[bridge-disc. draft x0.8127]--> 622.08
    #          --[OOD private tax x0.9421]--> 586.08
    public_numerator_only = C["K_cal"] * et                                  # 765.63 (deployed step)
    step_haircut_factor = C["step_us"] / C["step_central_us"]                # 0.8127  (bridge-priced)
    bridge_discounted_draft_haircut_public = public_numerator_only - hp     # 143.55 public TPS removed
    bridge_discounted_draft_haircut_private = (public_numerator_only - hp) * rho  # 135.27 private TPS

    # Bridge-band robustness: vary the step convention across the banked band and re-price private.
    # All three already APPLY the bridge (eagle3_draft_norm = wall_delta * bridge); they differ only in
    # WHICH draft wall delta is charged (per-step faithful -> full-chain additive). step_central is the
    # go-card midpoint. The additive corner is the most pessimistic banked draft denominator.
    band = {
        "faithful_1316": C["step_faithful_us"],
        "central_1499": C["step_central_us"],
        "additive_1683": C["step_additive_us"],
    }
    bridge_band = {}
    for label, step in band.items():
        hp_b = C["K_cal"] * et * (C["step_us"] / step)
        priv_b = rho * hp_b
        bridge_band[label] = {
            "step_us": step,
            "honest_public": hp_b,
            "private_tps": priv_b,
            "clears_500": bool(priv_b >= TARGET),
            "margin_tps": priv_b - TARGET,
        }
    bridge_band_all_clear = bool(all(v["clears_500"] for v in bridge_band.values()))

    return {
        "narrative": "586 = 481.53 x [E[T] numerator 1.5897] x [bridge-disc. draft 0.8127] x [OOD 0.9421]",
        "public_numerator_only_deployed_step": public_numerator_only,
        "step_haircut_factor_bridge_priced": step_haircut_factor,
        "bridge_discounted_draft_haircut_public_tps": bridge_discounted_draft_haircut_public,
        "bridge_discounted_draft_haircut_private_tps": bridge_discounted_draft_haircut_private,
        "draft_wall_delta_us": C["eagle3_draft_wall_us"],
        "draft_norm_us_bridged": C["eagle3_draft_norm_us"],
        "bridge_draft": C["bridge_draft"],
        "bridge_overcredit_x": C["bridge_overcredit"],
        "residual_unbridged_axis": "E[T] NUMERATOR (token-multiply); NOT a time-slice shrink",
        "bridge_band_private": bridge_band,
        "bridge_band_all_clear_500": bridge_band_all_clear,
        "statement": (
            "The bridge (denken #278, 0.21471, draft-side-specific) ALREADY discounts EAGLE-3's draft "
            "denominator INTO the 586: step 1218.2->%.0f us removes %.1f public TPS (%.1f private) before "
            "the OOD tax. The residual un-bridged part is the E[T] NUMERATOR raise. Across the banked "
            "bridge/step band [faithful %.0f, central %.0f, additive %.0f us] private = [%.1f, %.1f, %.1f] "
            "-- ALL clear 500 (the most pessimistic banked draft-denominator corner still holds the bar)."
            % (C["step_central_us"], bridge_discounted_draft_haircut_public,
               bridge_discounted_draft_haircut_private, C["step_faithful_us"], C["step_central_us"],
               C["step_additive_us"], bridge_band["faithful_1316"]["private_tps"],
               bridge_band["central_1499"]["private_tps"], bridge_band["additive_1683"]["private_tps"])),
    }


# --------------------------------------------------------------------------- #
# Step 4 (DELIVERABLE 2) -- residual realisation-ratio break-even where 586 -> 500.
# --------------------------------------------------------------------------- #
def residual_breakeven(C: dict) -> dict[str, Any]:
    """Apply a realisation ratio rho_real to ONLY the residual (un-bridged) E[T] raise -- the increment
    over the wall-honest anchor -- and solve for where the 586 projection falls to the 500 bar. Two
    conventions reported; the public-space one is the more conservative (higher break-even)."""
    private_586 = C["private_586"]
    hp = C["honest_public_622"]
    rho_ood = C["rho_priv_e3"]

    # Convention P (private-space): anchor to the organizer-VERIFIED deployed private 460.85 (a real
    # measured wall number). Residual raise = 586.08 - 460.85; haircut it directly.
    raise_priv = private_586 - C["private_verified"]                         # 125.23
    be_priv = (TARGET - C["private_verified"]) / raise_priv                  # 0.3126

    # Convention Q (public-space): anchor to the deployed public 481.53; haircut the public E[T] raise
    # (481.53 -> 622.08), THEN apply the OOD tax. Avoids mixing the deployed OOD factor (0.957) with the
    # EAGLE-3 one (0.9421); slightly more conservative.
    raise_pub = hp - C["official_baseline"]                                  # 140.55
    be_pub = ((TARGET / rho_ood) - C["official_baseline"]) / raise_pub       # 0.3500

    return {
        "convention_P_private_space": {
            "anchor": "verified deployed private 460.85",
            "residual_raise_tps": raise_priv,
            "breakeven_rho_real": be_priv,
            "formula": "460.85 + rho_real*(586.08-460.85) = 500",
        },
        "convention_Q_public_space": {
            "anchor": "deployed public 481.53, OOD tax after",
            "residual_raise_tps": raise_pub,
            "breakeven_rho_real": be_pub,
            "formula": "(481.53 + rho_real*(622.08-481.53)) * 0.9421 = 500",
        },
        "breakeven_rho_real_band": [be_priv, be_pub],
        "breakeven_rho_real_conservative": max(be_priv, be_pub),
        "below_stepside_4769": bool(max(be_priv, be_pub) < C["rr487_stepside"]),
        "statement": (
            "The residual (un-bridged) E[T] raise needs to realise at only rho_real in [%.3f (private-"
            "space), %.3f (public-space)] for 586 to hold the 500 bar -- BELOW stark #298's step-side "
            "0.4769 and FAR below the ~1.0 a numerator raise actually realises. 586 breaches 500 only if "
            "the E[T] raise realises worse than ~%.0f%%." % (be_priv, be_pub, 100.0 * max(be_priv, be_pub))),
    }


# --------------------------------------------------------------------------- #
# Step 5 (DELIVERABLE 3) -- is 47.7% the right reference? + literal-4769 stress survival.
# --------------------------------------------------------------------------- #
def stepside_vs_etside(C: dict) -> dict[str, Any]:
    """The 47.7% is a DENOMINATOR (fixed-overhead-dilution) phenomenon; an E[T] NUMERATOR raise is not
    subject to it. So 47.7% is the wrong reference and the E[T]-side realises higher. As a hard stress,
    we nonetheless apply the literal 0.4769 to the residual E[T] raise in both conventions: 586 still
    clears 500 -> the headline survives even under the wrong (harsher) reference."""
    rr = C["rr487_stepside"]
    private_586 = C["private_586"]
    hp = C["honest_public_622"]
    rho_ood = C["rho_priv_e3"]

    # apply literal 47.7% to the residual raise.
    raise_priv = private_586 - C["private_verified"]
    priv_at_4769_P = C["private_verified"] + rr * raise_priv                 # 520.57
    raise_pub = hp - C["official_baseline"]
    priv_at_4769_Q = (C["official_baseline"] + rr * raise_pub) * rho_ood     # 516.81

    survives_P = bool(priv_at_4769_P >= TARGET)
    survives_Q = bool(priv_at_4769_Q >= TARGET)
    worst_case = min(priv_at_4769_P, priv_at_4769_Q)

    # The EXTREME double-pessimistic corner (additive step AND 47.7% numerator haircut). This
    # DOUBLE-CHARGES the draft (the additive step already maxes the draft denominator) and mis-applies a
    # denominator-dilution ratio to a numerator -- reported for transparency, NOT the physical model.
    hp_additive = C["K_cal"] * C["E_T_central"] * (C["step_us"] / C["step_additive_us"])
    priv_additive = rho_ood * hp_additive
    raise_priv_add = priv_additive - C["private_verified"]
    priv_double_pessimistic = C["private_verified"] + rr * raise_priv_add

    return {
        "why_4769_is_wrong_reference": (
            "47.7% is a STEP-SIDE / DENOMINATOR realisation ratio: a kernel-us saving is diluted by the "
            "large FIXED serving overhead that does NOT shrink (TPS=tokens/(kernel+fixed)). An E[T]-SIDE "
            "raise is a NUMERATOR multiply (TPS=E[T]/T_macrostep): the fixed overhead is AMORTISED over "
            "more tokens, not dragged through a shrinking slice, so it does NOT dilute the gain. The "
            "E[T] numerator realises at ~100%; EAGLE-3's only denominator cost (heavier draft) is "
            "already bridge-priced into step_central. Hence 47.7% over-states the E[T]-side haircut."),
        "etside_expected_realization": "~1.0 (numerator), denominator already bridged",
        "literal_4769_applied_private_space": priv_at_4769_P,
        "literal_4769_applied_public_space": priv_at_4769_Q,
        "survives_4769_private_space": survives_P,
        "survives_4769_public_space": survives_Q,
        "survives_4769_both": bool(survives_P and survives_Q),
        "private_tps_at_worstcase_realization": worst_case,
        "double_pessimistic_additive_and_4769": priv_double_pessimistic,
        "double_pessimistic_note": (
            "additive-step (max draft denominator) AND a 47.7%% numerator haircut TOGETHER give %.1f "
            "(<500). This double-charges the draft (bridge already prices it) and mis-applies a "
            "denominator ratio to a numerator -- it is the unphysical worst-of-both corner, not the "
            "model." % priv_double_pessimistic),
        "statement": (
            "Applying stark #298's literal step-side 0.4769 to the residual E[T] raise: 586 -> %.1f "
            "(private-space) / %.1f (public-space) -- BOTH clear 500. The 586 headline SURVIVES even the "
            "wrong, harsher 47.7%% reference. The correct E[T]-side realisation is ~100%% (numerator) with "
            "the draft denominator already bridged, so the real margin is far larger."
            % (priv_at_4769_P, priv_at_4769_Q)),
    }


# --------------------------------------------------------------------------- #
# Step 6 (DELIVERABLE 4) -- headroom margin: how much realisation loss 586 can absorb.
# --------------------------------------------------------------------------- #
def headroom(C: dict) -> dict[str, Any]:
    private_586 = C["private_586"]
    headroom_tps = private_586 - TARGET                                      # 86.08
    headroom_pct_over_bar = 100.0 * headroom_tps / TARGET                    # 17.2%
    # The whole projection can lose this fraction of its VALUE before breaching 500.
    absorbable_haircut_frac = headroom_tps / private_586                     # 0.1469
    overall_realization_floor = TARGET / private_586                         # 0.8531 (realized/composed)
    return {
        "headroom_tps": headroom_tps,
        "headroom_pct_over_500_bar": headroom_pct_over_bar,
        "absorbable_total_haircut_frac": absorbable_haircut_frac,
        "overall_realization_floor_on_full_586": overall_realization_floor,
        "ood_breakeven_rho_cross_ref": C["breakeven_rho_ood"],
        "statement": (
            "Headroom 586.08 - 500 = %.1f TPS (+%.1f%% over the bar). The full projection can absorb a "
            "%.1f%% haircut on its value (overall realised/composed floor %.4f) before breaching 500. "
            "Separately, on the OOD axis the projection breaks even if the private factor falls from "
            "0.9421 to %.4f (banked fern #310). The realisation axis is far safer: the residual E[T] "
            "raise need only realise at ~31-35%%."
            % (headroom_tps, headroom_pct_over_bar, 100.0 * absorbable_haircut_frac,
               overall_realization_floor, C["breakeven_rho_ood"])),
    }


# --------------------------------------------------------------------------- #
# Step 7 -- honest caveats (carried explicitly).
# --------------------------------------------------------------------------- #
def caveats() -> dict[str, str]:
    return {
        "realization_axis_only": (
            "This card prices the WALL-REALISATION axis of the 586 projection (does a composed E[T] "
            "raise realise on the host-to-host wall). It takes the upstream numbers (E[T]=6.11 "
            "reachability, OOD private factor rho_priv_e3=0.9421) as GIVEN -- those are fern #310 / "
            "go-card #305 axes and OUT OF SCOPE here."),
        "rho_modeled_not_measured": (
            "rho_priv_e3=0.9421 is MODELED from the deployed linear spine's deep-position fidelity "
            "(lawine #300); a trained {2,21,39}-fusion EAGLE-3 draft may carry a different private tax. "
            "The realisation argument here is orthogonal to that modelling risk."),
        "bridge_is_draft_side": (
            "The denken #278 bridge (0.21471) is DRAFT-SIDE-SPECIFIC (kanna #286: verify-side bridge=1.0). "
            "It prices the batch=1 draft wall cost into the normalised step (the 4.82x/4.66x over-credit). "
            "It does NOT cover the fixed-overhead dilution that hit #298's verify-side step lever -- but "
            "that dilution is a DENOMINATOR effect and does not apply to an E[T] NUMERATOR raise."),
        "verify_gemm_m_cliff": (
            "The one un-modelled residual denominator risk is the verify-GEMM M-width tile cliff (flat "
            "M<=32, jump at M=33): a wider tree could push M past the cliff. This is bounded, separate "
            "from the bridge, and not a numerator dilution. The deployed M=8 convention stays sub-cliff."),
        "scope": (
            "LOCAL CPU-only analytic robustness closer over banked MERGED constants. 0 TPS added; "
            "BASELINE 481.53 untouched; greedy/PPL untouched. NO GPU / vLLM / HF Job / submission / "
            "served-file change. Authorises NOTHING. NOT a launch."),
    }


# --------------------------------------------------------------------------- #
# Step 8 -- self-test (PRIMARY metric).
# --------------------------------------------------------------------------- #
def self_test(C: dict, rep: dict, dec: dict, res: dict, ss: dict, hd: dict, cav: dict) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # (1) every imported constant matches its source <=1e-6.
    results["01_imports_match_source"] = {
        "pass": bool(C["_all_match"]),
        "max_abs_err": max((v["abs_err"] for v in C["_verify"]), default=0.0),
        "mismatches": [v["name"] for v in C["_verify"] if not v["matches_source"]],
    }

    # (2) reproduce the banked 586.08 / 622.08 <=1e-6 (the projection under stress is faithfully rebuilt).
    results["02_reproduces_586"] = {
        "pass": bool(rep["reproduces_586"]),
        "resid_private_586": rep["resid_private_586"],
        "resid_honest_public": rep["resid_honest_public"],
    }

    # (3) the bridge-discounted draft haircut is the right SIGN and magnitude (the bridge removes TPS,
    #     i.e. the EAGLE-3 step is heavier than deployed; haircut public > 0 and < the numerator gross).
    h = dec["bridge_discounted_draft_haircut_public_tps"]
    results["03_bridge_haircut_signed"] = {
        "pass": bool(0.0 < h < dec["public_numerator_only_deployed_step"]),
        "haircut_public_tps": h,
    }

    # (4) the residual break-even is in (0,1) and BELOW the step-side 0.4769 (the central claim:
    #     an E[T]-side raise needs far less realisation than a step-side one to hold the bar).
    be = res["breakeven_rho_real_conservative"]
    results["04_residual_breakeven_below_stepside"] = {
        "pass": bool(0.0 < be < C["rr487_stepside"] < 1.0),
        "breakeven_rho_real_conservative": be,
        "stepside_4769": C["rr487_stepside"],
    }

    # (5) the headline SURVIVES the literal 47.7% haircut in BOTH conventions (the robustness verdict).
    results["05_survives_literal_4769_both"] = {
        "pass": bool(ss["survives_4769_both"]),
        "private_space": ss["literal_4769_applied_private_space"],
        "public_space": ss["literal_4769_applied_public_space"],
    }

    # (6) bridge-band robustness: across [faithful, central, additive] every private projection clears.
    results["06_bridge_band_all_clear"] = {
        "pass": bool(dec["bridge_band_all_clear_500"]),
        "additive_private": dec["bridge_band_private"]["additive_1683"]["private_tps"],
    }

    # (7) headroom is the banked +17.2% / +86 TPS (round-trips fern #310 to <=1e-6).
    results["07_headroom_matches_310"] = {
        "pass": bool(abs(hd["headroom_tps"] - C["margin_tps_310"]) <= 1e-6
                     and abs(hd["headroom_pct_over_500_bar"] - C["margin_pct_310"]) <= 1e-6),
        "headroom_tps": hd["headroom_tps"],
    }

    # (8) NaN-clean across every reported numeric.
    payload_numeric = {"rep": rep, "dec": {k: v for k, v in dec.items() if k != "bridge_band_private"},
                       "res": res, "ss": ss, "hd": hd,
                       "constants": {k: v for k, v in C.items() if not k.startswith("_") and _finite(v)}}
    nan_paths = _nan_paths(payload_numeric, "selftest")
    results["08_nan_clean"] = {"pass": bool(len(nan_paths) == 0), "nan_paths": nan_paths}

    # (9) caveats carried (realization-axis-only + bridge-draft-side + M-cliff + BASELINE untouched).
    g_ok = ("OUT OF SCOPE" in cav["realization_axis_only"]
            and "DRAFT-SIDE-SPECIFIC" in cav["bridge_is_draft_side"]
            and "M-width" in cav["verify_gemm_m_cliff"]
            and "481.53" in cav["scope"] and "NOT a launch" in cav["scope"])
    results["09_caveats_carried"] = {"pass": bool(g_ok)}

    card_valid = bool(all(v["pass"] for v in results.values()))
    # PRIMARY: the card is internally valid AND the 586 headline survives the 47.7% haircut.
    survives = bool(ss["survives_4769_both"] and dec["bridge_band_all_clear_500"])
    return {
        "realization_haircut_survives_self_test_passes": bool(card_valid and survives),
        "card_valid": card_valid,
        "headline_survives_4769": survives,
        "conditions": results,
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    C = load_constants()
    rep = reproduce(C)
    dec = decompose(C)
    res = residual_breakeven(C)
    ss = stepside_vs_etside(C)
    hd = headroom(C)
    cav = caveats()
    st = self_test(C, rep, dec, res, ss, hd, cav)

    constants_public = {k: v for k, v in C.items() if not k.startswith("_")}
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 321, "agent": "stark", "kind": "eagle3_realization_haircut",
        "analysis_only": True,
        "primary_metric_name": "realization_haircut_survives_self_test_passes",
        "realization_haircut_survives_self_test_passes":
            st["realization_haircut_survives_self_test_passes"],
        "test_metric_names": ["private_tps_at_worstcase_realization"],
        "private_tps_at_worstcase_realization": ss["private_tps_at_worstcase_realization"],
        "reproduce": rep,
        "deliverable1_decompose": dec,
        "deliverable2_residual_breakeven": res,
        "deliverable3_stepside_vs_etside": ss,
        "deliverable4_headroom": hd,
        "caveats": cav,
        "self_test": st,
        "import_verification": C["_verify"],
        "all_imports_match_source": C["_all_match"],
        "constants": constants_public,
        "provenance": C["provenance"],
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    payload["nan_clean"] = len(_nan_paths({"a": rep, "b": dec, "c": res, "d": ss, "e": hd})) == 0
    return payload


# --------------------------------------------------------------------------- #
# console report
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    dec = payload["deliverable1_decompose"]
    res = payload["deliverable2_residual_breakeven"]
    ss = payload["deliverable3_stepside_vs_etside"]
    hd = payload["deliverable4_headroom"]
    st = payload["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("PR #321 -- Does the 586 EAGLE-3 private-500 projection survive a 47.7% haircut?", flush=True)
    print("=" * 100, flush=True)
    print("REPRODUCE banked projection (<=1e-6):", flush=True)
    r = payload["reproduce"]
    print(f"  honest_public(6.11)={r['honest_public_611']:.4f}  private={r['private_586_reproduced']:.4f}"
          f"  [reproduces_586={r['reproduces_586']}]", flush=True)
    print("-" * 100, flush=True)
    print("D1 DECOMPOSE:", dec["statement"], flush=True)
    for label, mm in dec["bridge_band_private"].items():
        print(f"    {label:<14} step={mm['step_us']:7.1f}us  public={mm['honest_public']:7.1f}"
              f"  private={mm['private_tps']:7.1f}  clears={mm['clears_500']}", flush=True)
    print("-" * 100, flush=True)
    print("D2 RESIDUAL BREAK-EVEN:", res["statement"], flush=True)
    print(f"    private-space rho_real={res['convention_P_private_space']['breakeven_rho_real']:.4f}"
          f"   public-space rho_real={res['convention_Q_public_space']['breakeven_rho_real']:.4f}"
          f"   (step-side ref 0.4769)", flush=True)
    print("-" * 100, flush=True)
    print("D3 47.7% IS THE WRONG REFERENCE:", ss["statement"], flush=True)
    print(f"    literal-0.4769 stress: private-space={ss['literal_4769_applied_private_space']:.2f}"
          f"  public-space={ss['literal_4769_applied_public_space']:.2f}"
          f"  survives_both={ss['survives_4769_both']}", flush=True)
    print(f"    [double-pessimistic additive+0.4769 corner = {ss['double_pessimistic_additive_and_4769']:.1f} "
          f"(<500, unphysical double-charge)]", flush=True)
    print("-" * 100, flush=True)
    print("D4 HEADROOM:", hd["statement"], flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) realization_haircut_survives_self_test_passes = "
          f"{st['realization_haircut_survives_self_test_passes']}  "
          f"(card_valid={st['card_valid']} survives_4769={st['headline_survives_4769']})", flush=True)
    for k, v in st["conditions"].items():
        print(f"   - {k}: {'PASS' if v['pass'] else 'FAIL'}", flush=True)
    print(f"(TEST) private_tps_at_worstcase_realization = "
          f"{payload['private_tps_at_worstcase_realization']:.2f}", flush=True)
    print(f"nan_clean={payload['nan_clean']}  peak_mem_mib={payload['peak_mem_mib']}", flush=True)
    print("=" * 100 + "\n", flush=True)


# --------------------------------------------------------------------------- #
# wandb logging (robust; never fatal)
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[haircut] wandb logging unavailable: {exc}", flush=True)
        return None

    C = payload["constants"]
    dec = payload["deliverable1_decompose"]
    res = payload["deliverable2_residual_breakeven"]
    ss = payload["deliverable3_stepside_vs_etside"]
    hd = payload["deliverable4_headroom"]
    st = payload["self_test"]
    run = init_wandb_run(
        job_type="validity-analytic", agent="stark", name=args.wandb_name, group=args.wandb_group,
        tags=["eagle3-realization", "validity-analytic", "realization-haircut", "eagle3", "private-500",
              "bridge", "go-no-go", "bank-the-analysis"],
        config={
            "pr": 321, "K_cal": C["K_cal"], "step_us": C["step_us"], "step_central_us": C["step_central_us"],
            "E_T_central": C["E_T_central"], "rho_priv_e3": C["rho_priv_e3"],
            "bridge_draft": C["bridge_draft"], "private_586": C["private_586"],
            "honest_public_622": C["honest_public_622"], "rr487_stepside": C["rr487_stepside"],
            "provenance": payload["provenance"], "scope": payload["caveats"]["scope"],
        },
    )
    if run is None:
        print("[haircut] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        "realization_haircut_survives_self_test_passes":
            int(bool(st["realization_haircut_survives_self_test_passes"])),
        "private_tps_at_worstcase_realization": payload["private_tps_at_worstcase_realization"],
        "card_valid": int(bool(st["card_valid"])),
        "headline_survives_4769": int(bool(st["headline_survives_4769"])),
        "bridge_discounted_draft_haircut_public_tps": dec["bridge_discounted_draft_haircut_public_tps"],
        "bridge_discounted_draft_haircut_private_tps": dec["bridge_discounted_draft_haircut_private_tps"],
        "bridge_band_all_clear_500": int(bool(dec["bridge_band_all_clear_500"])),
        "additive_corner_private_tps": dec["bridge_band_private"]["additive_1683"]["private_tps"],
        "breakeven_rho_real_private_space": res["convention_P_private_space"]["breakeven_rho_real"],
        "breakeven_rho_real_public_space": res["convention_Q_public_space"]["breakeven_rho_real"],
        "breakeven_rho_real_conservative": res["breakeven_rho_real_conservative"],
        "residual_below_stepside_4769": int(bool(res["below_stepside_4769"])),
        "literal_4769_private_space": ss["literal_4769_applied_private_space"],
        "literal_4769_public_space": ss["literal_4769_applied_public_space"],
        "survives_4769_both": int(bool(ss["survives_4769_both"])),
        "double_pessimistic_additive_and_4769": ss["double_pessimistic_additive_and_4769"],
        "headroom_tps": hd["headroom_tps"],
        "headroom_pct_over_500_bar": hd["headroom_pct_over_500_bar"],
        "absorbable_total_haircut_frac": hd["absorbable_total_haircut_frac"],
        "overall_realization_floor_on_full_586": hd["overall_realization_floor_on_full_586"],
        "stepside_realization_ratio_298": C["rr487_stepside"],
        "max_import_abs_err": st["conditions"]["01_imports_match_source"]["max_abs_err"],
        "nan_clean": int(bool(payload["nan_clean"])), "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v["pass"])) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_realization_haircut_result",
                      artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    print(f"[haircut] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PR #321 EAGLE-3 realization-haircut robustness closer")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="eagle3-realization")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    payload = run()
    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_realization_haircut_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[haircut] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    gate = bool(payload["realization_haircut_survives_self_test_passes"])
    print(f"  PRIMARY realization_haircut_survives_self_test_passes = {gate}", flush=True)
    print(f"  TEST private_tps_at_worstcase_realization = "
          f"{payload['private_tps_at_worstcase_realization']:.2f}", flush=True)
    print(f"  wandb run = {rid}", flush=True)
    if args.self_test:
        print(f"[haircut] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
