"""PR #664 spec-regime isolation -- SYNTHESIS (FULL: cross-student authorized).

Human issue #666 AUTHORIZED cross-student reads (morganmcg1, OWNER, 2026-06-18T14:54Z:
"Authorized!"). So this completes ALL of #664: the espec discriminator (Step 1), the
config-convergence attribution (Step 2), ratio-stability (Step 3), the #319 identity
re-gate + explicit break_rate/spec_fire_rate scalars (Step 4), and the SURFACE framing
(Step 5) -- with stark's #642 spec-path config + K6 espec READ directly from his runs.

THE FINDING -- REGIME_IS_ACCEPTANCE. land's 170.21 and stark's 155.57 un-rescued K6
ceilings are BOTH NUM_SPECULATIVE_TOKENS=6 on the SAME int4_mtp_batchinv stack (same
w4a16-ct body, BI=1, vllm 0.22.0, 128x512 seed1), measured by the SAME shared
scripts/profiler/paired_tps_ab.py runner logging the SAME `e_accept_exact` key. The
ONLY differing spec-path field is DRAFTER_MODEL:
  land  : /tmp/qat-assistant (local QAT-matched gemma4_assistant)  espec 3.6574
  stark : google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant (stock manifest)  espec 3.3332
The espec gap (3.6574 vs 3.3332 = +9.73%) ~exactly tracks the wall_tps gap (+9.41%);
the implied per-spec-step rate is IDENTICAL (46.54 vs 46.67 steps/s, -0.28%). So the
locus is ACCEPTANCE (a better drafter), NOT a per-step LATENCY config knob -- the own-
config knob sweep (sampler BOOT_FAIL, eager -73%+id-break, flashattn) corroborates that
no identity-safe latency knob moves the 170 regime.

SHIPPABILITY: the OFFICIAL submission ships the STOCK Hub drafter, so the official
harness realizes stark's ~155 / rescued ~135 regime. land's 170 depends on a LOCAL-only
drafter (/tmp/qat-assistant, not on the Hub, not in the submission). The swap is identity-
safe BY CONSTRUCTION (the target verify enforces greedy-identity for ANY drafter; the
re-gate's break_rate=0 does not even load the drafter) and PPL-neutral (output identical)
-> publishing /tmp/qat-assistant + repointing the manifest is a clean SURFACE candidate
for #481 (lifts official-equiv ~135 -> ~147, clears the +10 bar). It is NOT a fire.

analysis_only=True, official_tps=0. NO HF Job / NO submission. Locked 126.378 untouched.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KSWEEP = ROOT / "research/walltps_ab/optionb_bi1_stock_int4/ksweep"

# ---- stark #642 constants -- READ from his runs (issue #666 human-authorized 2026-06-18) ----
# Source: stark branch stark/optionb-rescue-deproject, research/validity/optionb_rescue_deproject/
# bc/paired_ab.json (run 6uepftr6) + arm_d (2rc1sku2) + rate_sweep_captured (henp1fb8).
STARK_K6_UNRESCUED = 155.5693177204223      # 6uepftr6 candidate arm wall_tps median (stock drafter)
STARK_K6_ESPEC = 3.33324213626396           # 6uepftr6 candidate arm e_accept_exact mean (SAME key as land)
STARK_AR_RUNG_LOCAL = 126.75181166586952    # 6uepftr6 baseline arm: int4_g128_lmhead local AR rung
STARK_AR_REF_RECOMPUTE = 77.89              # 2rc1sku2 arm-d: w4a16-ct M=1 recompute cost (K-indep)
STARK_RESCUED_K6_CAPTURED = 135.27          # henp1fb8: live-acceptor captured rescued K6 @ tau=0.3 (MEASURED)
STARK_RESC_OVER_UNRESC_CAPTURED = 135.27 / 155.58   # stark's measured captured tax ratio (0.8695)
STARK_DRAFTER = "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"  # stock manifest (Hub)
LAND_DRAFTER = "/tmp/qat-assistant"         # local QAT-matched gemma4_assistant (NOT on Hub / not in submission)

LAND_AR_REF_A = 77.962            # land #658 own AR rung (w4a16-ct M=1; ~= stark's 77.89)
SPEC_FIRE_RATE = 0.0727386474609375   # land #648 served_fire_frac_overall (tau=0.5, K-indep)
OPTIMAL_TAU = 0.3                 # stark #642 / land re-gate: min tau holding break_rate=0
PPL_SPEC = 2.005501029415618      # land ksweep ppl_k5_spec (target-only, drafter/K-independent)
PPL_GATE = 2.42
LOCKED_OFFICIAL = 126.378
# A SURFACE-to-#481 speedup must be MATERIAL, not merely statistically resolvable. The paired
# operative MDE at N=3 is ~0.10%; a knob can clear that yet still be physically trivial (FLASH_ATTN
# lands +0.142%, ~0.24 TPS, within run-to-run/hardware noise -- CV 0.009%, CI95 +/-0.055%). The fire
# framework's incremental bar is +10 TPS (+7.9%); a knob worth spending official quota to re-measure
# should move >=1% (~7x the noise floor). Below that, an identity-safe "speedup" is reported for
# transparency but is NOT a shippable lever and does NOT warrant an official re-measure.
MATERIAL_SPEEDUP_PCT = 1.0


def reprice(U: float, f: float, A: float) -> float:
    """rescued = 1/(1/U + f/A)  (additive de-projection: pay the fire tax f at the AR ref A)."""
    return 1.0 / (1.0 / U + f / A)


def official_equiv(rescued_local: float) -> float:
    """Project a LOCAL rescued wall_tps to the OFFICIAL scale via the g128 AR rung, which is
    drafter-INDEPENDENT and measured both locally (126.75) and officially (126.378). This is
    the PR Step-3 formula `rescued_K6 / AR_rung_local * 126.378_official`."""
    return rescued_local / STARK_AR_RUNG_LOCAL * LOCKED_OFFICIAL


def sha_map(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["dataset_index"]] = d["completion_token_sha256"]
    return out


def byte_identity(cand_decode: Path, base_decode: Path) -> dict[str, Any]:
    """Strict per-prompt sha256 match of two FULL spec trajectories. Valid identity gate
    here because same-config is 128/128 deterministic across restarts (verified): any
    divergence is the knob perturbing an argmax. A byte-identical trajectory inherits the
    baseline's tau=0.3 break_rate=0 exactly."""
    c, b = sha_map(cand_decode), sha_map(base_decode)
    keys = sorted(set(c) & set(b))
    match = sum(1 for k in keys if c[k] == b[k])
    return {"n_compared": len(keys), "n_match": match,
            "byte_identical": (len(keys) > 0 and match == len(keys)),
            "n_divergent_prompts": len(keys) - match}


def arm_wall(paired_json: Path, which: str) -> dict[str, Any]:
    d = json.loads(paired_json.read_text())
    a = d["arms"][which]
    w = a.get("wall_tps", {})
    ea = a.get("e_accept_exact", {})
    return {"median": w.get("median"), "mean": w.get("mean"), "cv_pct": w.get("cv_pct"),
            "n": w.get("n"), "e_accept_mean": ea.get("mean"),
            "describe": d[which].get("describe")}


def load_knob_records(out_dir: Path, label: str, base_decode: Path) -> dict[str, Any] | None:
    """Fallback when paired_ab.json is absent (e.g. the eager arm was stopped after its decisive
    N=1 run to free the GPU for the #319 re-gate -- a -73% effect is unambiguous at N=1). Builds the
    same knob entry from the per-run records.jsonl: real measured wall_tps, not a re-run."""
    rj = out_dir / "records.jsonl"
    if not rj.exists():
        return None
    base_tps, cand_tps, cand_eacc = [], [], []
    for line in rj.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        arm = d.get("arm")
        if arm == "k6_base":
            base_tps.append(d["wall_tps"])
        elif arm == label:
            cand_tps.append(d["wall_tps"])
            if d.get("e_accept_exact") is not None:
                cand_eacc.append(d["e_accept_exact"])
    if not cand_tps:
        return None
    base_med = statistics.median(base_tps) if base_tps else None
    cand_med = statistics.median(cand_tps)
    delta_pct = (100.0 * (cand_med - base_med) / base_med) if base_med else None
    cand_decode = out_dir / label / "decode" / "run00.jsonl"
    ident = byte_identity(cand_decode, base_decode)
    return {
        "label": label,
        "baseline_median": base_med,
        "candidate_median": cand_med,
        "delta_pct": delta_pct,
        # a -73% (eager) move dwarfs the #72 operative MDE (0.1%); REAL by inspection.
        "verdict": "REAL" if (delta_pct is not None and abs(delta_pct) > 0.1) else "NULL",
        "op_threshold_pct": 0.1,
        "candidate_cv_pct": (100.0 * statistics.pstdev(cand_tps) / cand_med
                             if len(cand_tps) > 1 and cand_med else None),
        "candidate_e_accept": (statistics.median(cand_eacc) if cand_eacc else None),
        "identity_vs_baseline": ident,
        "identity_safe": ident["byte_identical"],
        "n_runs": len(cand_tps),
        "source": "records.jsonl(fallback,N=%d)" % len(cand_tps),
    }


def load_knob(out_dir: Path, label: str, base_decode: Path) -> dict[str, Any] | None:
    pj = out_dir / "paired_ab.json"
    if not pj.exists():
        return load_knob_records(out_dir, label, base_decode)
    d = json.loads(pj.read_text())
    v = d["verdict"]
    cand = arm_wall(pj, "candidate")
    base = arm_wall(pj, "baseline")
    cand_decode = out_dir / label / "decode" / "run00.jsonl"
    ident = byte_identity(cand_decode, base_decode)
    return {
        "label": label,
        "baseline_median": base["median"],
        "candidate_median": cand["median"],
        "delta_pct": v.get("delta_median_pct"),
        "verdict": v.get("verdict"),
        "op_threshold_pct": v.get("operative_threshold_pct"),
        "candidate_cv_pct": cand["cv_pct"],
        "candidate_e_accept": cand["e_accept_mean"],
        "identity_vs_baseline": ident,
        "identity_safe": ident["byte_identical"],
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    base_decode = HERE / "sampler" / "k6_base" / "decode" / "run00.jsonl"

    # ---- knob attribution (bootable knobs) ----
    knobs = []
    for out_sub, label in (("eager", "eager1"), ("attn", "flashattn")):
        k = load_knob(HERE / out_sub, label, base_decode)
        if k:
            knobs.append(k)

    # ---- boot-fail knobs (server never reached readiness -> not a viable shippable lever) ----
    boot_fail = []
    for out_sub in ("sampler",):
        bf = HERE / out_sub / "BOOT_FAIL.json"
        if bf.exists():
            boot_fail.append(json.loads(bf.read_text()))

    baseline_median = next((k["baseline_median"] for k in knobs if k["baseline_median"]), None)

    # ---- K-definition sensitivity (OWN stack, already measured in ksweep) ----
    kdef = None
    kj = KSWEEP / "k6" / "paired_ab.json"
    if kj.exists():
        d = json.loads(kj.read_text())
        k6 = arm_wall(kj, "candidate")   # NUM_SPECULATIVE_TOKENS=6
        k7 = arm_wall(kj, "baseline")    # NUM_SPECULATIVE_TOKENS=7
        kdef = {
            "k6_median": k6["median"], "k6_e_accept": k6["e_accept_mean"],
            "k7_median": k7["median"], "k7_e_accept": k7["e_accept_mean"],
            "k7_minus_k6_pct": (100.0 * (k7["median"] - k6["median"]) / k6["median"]
                                if (k6["median"] and k7["median"]) else None),
            # how close is my OWN K7 to stark's READ un-rescued K6 155.57? (both stock-K NUM_SPEC)
            "k7_vs_stark155_pct": (100.0 * (k7["median"] - STARK_K6_UNRESCUED)
                                   / STARK_K6_UNRESCUED if k7["median"] else None),
        }

    # ---- identity re-gate (#319 teacher-forced, tau-ladder) ----
    regate = None
    rg = HERE / "regate_report.json"
    if rg.exists():
        r = json.loads(rg.read_text())
        regate = {
            "break_rate_bi1_both_sides": r.get("break_rate_bi1_both_sides"),
            "residual_after_tau_0p3nat": r.get("residual_after_tau_0p3nat"),
            "attention_path_break_count": r.get("attention_path_break_count"),
            "tau_ladder": r.get("tau_ladder"),
            "verdict": r.get("verdict"),
            "n_prompts": r.get("n_prompts"),
            "total_positions": r.get("pinned_total_positions"),
        }
    break_rate = regate["residual_after_tau_0p3nat"] if regate else None

    # ---- STEP 1: espec discriminator (latency vs acceptance) -- cross-student, #666-authorized ----
    # land K6 (170) and stark K6 (155) are BOTH NUM_SPECULATIVE_TOKENS=6 on int4_mtp_batchinv,
    # measured by the SAME paired_tps_ab.py runner logging the SAME `e_accept_exact` key.
    land_k6_wall = kdef["k6_median"] if kdef else None         # 170.21 (ksweep candidate)
    land_k6_espec = kdef["k6_e_accept"] if kdef else None      # 3.6574 (ksweep candidate e_accept_exact)
    espec_ratio = (land_k6_espec / STARK_K6_ESPEC) if land_k6_espec else None       # 1.097
    walltps_ratio = (land_k6_wall / STARK_K6_UNRESCUED) if land_k6_wall else None    # 1.094
    step_rate_land = (land_k6_wall / land_k6_espec) if (land_k6_wall and land_k6_espec) else None
    step_rate_stark = STARK_K6_UNRESCUED / STARK_K6_ESPEC
    step_rate_ratio = (step_rate_land / step_rate_stark) if step_rate_land else None
    # Discriminator: stark espec < land espec (and step-rate identical) => the gap is ACCEPTANCE.
    espec_is_acceptance = (land_k6_espec is not None and STARK_K6_ESPEC < land_k6_espec - 0.02)
    discriminator = {
        "land_k6_espec": land_k6_espec, "stark_k6_espec": STARK_K6_ESPEC,
        "espec_delta": (land_k6_espec - STARK_K6_ESPEC) if land_k6_espec else None,
        "espec_ratio": espec_ratio, "walltps_ratio": walltps_ratio,
        "step_rate_land_per_s": step_rate_land, "step_rate_stark_per_s": step_rate_stark,
        "step_rate_ratio": step_rate_ratio,
        "espec_key_parity": "both `e_accept_exact` via scripts/profiler/paired_tps_ab.py (identical defn)",
        "reading": ("ACCEPTANCE: stark espec 3.333 < land espec 3.657; step-rate identical "
                    "(46.5 vs 46.7/s) => per-step latency equal, gap is drafter acceptance"),
    }

    # ---- STEP 2: config attribution -- the SINGLE differing spec-path field is the drafter ----
    drafter_attribution = {
        "differing_field": "DRAFTER_MODEL",
        "land_value": LAND_DRAFTER, "stark_value": STARK_DRAFTER,
        "land_espec": land_k6_espec, "stark_espec": STARK_K6_ESPEC,
        "identical_fields": ["MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct", "NUM_SPECULATIVE_TOKENS=6",
                             "VLLM_BATCH_INVARIANT=1", "vllm==0.22.0", "MAX_NUM_SEQS=1",
                             "workload=128x512 seed=1", "runner=paired_tps_ab.py"],
        "land_drafter_kind": "local-only gemma4_assistant (QAT-matched); NOT on Hub, NOT in submission",
        "stark_drafter_kind": "stock manifest default (Hub model id)",
        "official_harness_realizes": "stark ~155 regime (shipped submission uses the stock Hub drafter)",
        "note": ("NOT a per-step latency knob -> Step-2 'pin to stark's values' latency A/B is moot; "
                 "the own-config knob sweep (below) corroborates no identity-safe latency mover exists."),
    }

    # ---- STEP 3: ratio-stability -- does the 9.4% un-rescued delta PROPAGATE to the rescued
    #      official-equiv, or CANCEL?  The rescued ceiling scales with acceptance (drafter), but
    #      the AR-rung denominator (126.75) is drafter-INDEPENDENT -> it PROPAGATES.
    # stark regime (stock drafter): rescued is MEASURED (live acceptor, henp1fb8 = 135.27).
    rescued_stark = STARK_RESCUED_K6_CAPTURED
    off_equiv_stark = official_equiv(rescued_stark)                         # 134.87 (matches stark #642)
    # land regime (better drafter): rescued is PROJECTED. Transfer stark's MEASURED captured tax
    # ratio (0.8695, K- and drafter-independent: target-body recompute on target-logit-gap fires)
    # onto land's higher ceiling. Additive reprice gives a cross-check upper bound.
    rescued_land = (land_k6_wall * STARK_RESC_OVER_UNRESC_CAPTURED) if land_k6_wall else None
    rescued_land_additive = reprice(land_k6_wall, SPEC_FIRE_RATE, LAND_AR_REF_A) if land_k6_wall else None
    off_equiv_land = official_equiv(rescued_land) if rescued_land else None  # ~147.5 (projected)
    off_equiv_land_additive = official_equiv(rescued_land_additive) if rescued_land_additive else None
    propagated_delta_pct = (100.0 * (off_equiv_land - off_equiv_stark) / off_equiv_stark
                            if off_equiv_land else None)
    ratio_stability = {
        "ar_rung_local": STARK_AR_RUNG_LOCAL, "scaling_local_to_official": LOCKED_OFFICIAL / STARK_AR_RUNG_LOCAL,
        "stark_regime_stock_drafter": {
            "unrescued_k6": STARK_K6_UNRESCUED, "rescued_k6_MEASURED": rescued_stark,
            "official_equiv": off_equiv_stark,
            "pct_over_locked": 100.0 * (off_equiv_stark - LOCKED_OFFICIAL) / LOCKED_OFFICIAL,
            "clears_plus10_bar": off_equiv_stark >= LOCKED_OFFICIAL + 10.0},
        "land_regime_better_drafter": {
            "unrescued_k6": land_k6_wall, "rescued_k6_PROJECTED_taxratio": rescued_land,
            "rescued_k6_additive_xcheck": rescued_land_additive,
            "official_equiv": off_equiv_land, "official_equiv_additive": off_equiv_land_additive,
            "pct_over_locked": (100.0 * (off_equiv_land - LOCKED_OFFICIAL) / LOCKED_OFFICIAL
                                if off_equiv_land else None),
            "clears_plus10_bar": (off_equiv_land >= LOCKED_OFFICIAL + 10.0) if off_equiv_land else None,
            "shippability": "requires /tmp/qat-assistant published to Hub + manifest repoint"},
        "verdict": "PROPAGATES",
        "propagated_delta_pct": propagated_delta_pct,
        "reading": ("the 9.4% un-rescued delta PROPAGATES to the rescued official-equiv (~135 stock vs "
                    "~147 better-drafter); it does NOT cancel because the AR-rung denominator is "
                    "drafter-independent. The drafter is the lever that flips +6.7% -> clears +10 bar."),
    }

    # ---- knob-existence deliverable ----
    # A knob "moves the regime" if its paired delta is REAL (beyond the #72 operative MDE 0.10%).
    # It is a SURFACE-to-#481 *speedup* only if it is REAL, identity-safe, faster than the 170
    # baseline, AND MATERIAL (>= MATERIAL_SPEEDUP_PCT). A statistically-real but sub-1% move is
    # within run-to-run/hardware noise and cannot "make the slow regime run fast" (Step 5), so it
    # does NOT warrant spending official quota. A REAL identity-safe knob that SLOWS (toward ~155)
    # is attribution-relevant (it bounds how much of the 170<->155 gap one own-stack knob could
    # explain) but is NOT a speedup to surface. Boot-fail knobs (e.g. sampler1) are non-viable.
    knob_moves = [k for k in knobs if k["verdict"] == "REAL"]
    identity_safe_movers = [k for k in knob_moves if k["identity_safe"]]
    speedup_movers = [k for k in identity_safe_movers
                      if baseline_median and k["candidate_median"] > baseline_median]
    slowdown_movers = [k for k in identity_safe_movers
                       if baseline_median and k["candidate_median"] < baseline_median]
    material_speedup_movers = [k for k in speedup_movers
                               if k["delta_pct"] is not None and abs(k["delta_pct"]) >= MATERIAL_SPEEDUP_PCT]
    # headline deliverable: a MATERIAL identity-safe speedup worth surfacing for an official re-measure.
    # `speedup_movers` (any stat-real identity-safe speedup, incl. sub-material) is kept for transparency.
    shippable_knob_exists = len(material_speedup_movers) > 0

    out = {
        "pr": 664, "analysis_only": True, "official_tps": 0, "fires": False,
        "stack": "int4_mtp_batchinv",
        "scope": "FULL: cross-student stark #642/#663 reads authorized (human issue #666, 2026-06-18)",
        "baseline_k6_median_wall_tps": baseline_median,
        # STEP 1 -- espec discriminator (latency vs acceptance)
        "discriminator": discriminator,
        # STEP 2 -- config attribution (single differing spec-path field)
        "drafter_attribution": drafter_attribution,
        # STEP 3 -- ratio-stability (does the un-rescued delta propagate to official-equiv)
        "ratio_stability": ratio_stability,
        "knob_attribution": knobs,
        "k_definition_sensitivity": kdef,
        # STEP 4 -- #319 identity re-gate + explicit audit scalars
        "identity_regate": regate,
        "spec_fire_rate": SPEC_FIRE_RATE,
        "break_rate_tau0p3": break_rate,
        "ppl_spec": PPL_SPEC, "ppl_gate": PPL_GATE, "ppl_passes": PPL_SPEC <= PPL_GATE,
        "knob_boot_fail": boot_fail,
        "shippable_identity_safe_speedup_exists": shippable_knob_exists,
        "material_speedup_threshold_pct": MATERIAL_SPEEDUP_PCT,
        "identity_safe_speedup_movers": [{"label": k["label"], "delta_pct": k["delta_pct"],
                                          "is_material": abs(k["delta_pct"]) >= MATERIAL_SPEEDUP_PCT}
                                         for k in speedup_movers],
        "identity_safe_slowdown_movers": [{"label": k["label"], "delta_pct": k["delta_pct"]}
                                          for k in slowdown_movers],
        "verdict": "REGIME_IS_ACCEPTANCE",
        "verdict_reason": (
            "The land-170 vs stark-155 un-rescued K6 gap is ACCEPTANCE, not a per-step latency "
            "config knob. Both are NUM_SPECULATIVE_TOKENS=6 on the same int4_mtp_batchinv stack "
            "measured by the same paired_tps_ab.py runner logging the same e_accept_exact key; "
            "the only differing spec-path field is DRAFTER_MODEL (land /tmp/qat-assistant espec "
            "3.657 vs stark stock-Hub q4_0-unquantized espec 3.333). The +9.7%% espec gap tracks "
            "the +9.4%% wall_tps gap and the per-step rate is identical (46.5 vs 46.7 steps/s), so "
            "there is no latency component. The OFFICIAL submission ships the stock Hub drafter -> "
            "the harness realizes stark's ~155/rescued-~135 regime; land's 170 needs the LOCAL-only "
            "drafter. Own-config knob sweep (sampler BOOT_FAIL+greedy-moot; eager -73%% wall_tps "
            "+ id-break 18/128, disqualified; flashattn FLASH_ATTN byte-identical 128/128 but only "
            "+0.142%% = ~0.24 TPS within run-to-run/hardware noise, immaterial <1%%) finds NO MATERIAL "
            "identity-safe latency mover, corroborating acceptance."),
    }
    (HERE / "synthesis_664.json").write_text(json.dumps(out, indent=2, default=str))

    # ---- console ----
    print("\n=== PR#664 spec-regime isolation -- SYNTHESIS (FULL: #666-authorized) ===")
    print(f"baseline K6 wall_tps median = {baseline_median}")
    print("\nSTEP 1 -- ESPEC DISCRIMINATOR (latency vs acceptance):")
    print(f"  land  K6: wall={land_k6_wall:.2f}  espec={land_k6_espec:.4f}  step_rate={step_rate_land:.2f}/s  ({LAND_DRAFTER})")
    print(f"  stark K6: wall={STARK_K6_UNRESCUED:.2f}  espec={STARK_K6_ESPEC:.4f}  step_rate={step_rate_stark:.2f}/s  ({STARK_DRAFTER})")
    print(f"  espec_ratio={espec_ratio:.4f}  walltps_ratio={walltps_ratio:.4f}  step_rate_ratio={step_rate_ratio:.4f}")
    print(f"  => {discriminator['reading']}")
    print("\nSTEP 2 -- CONFIG ATTRIBUTION: single differing spec-path field =",
          drafter_attribution["differing_field"])
    print(f"  land={drafter_attribution['land_value']}  stark={drafter_attribution['stark_value']}")
    print(f"  official harness realizes: {drafter_attribution['official_harness_realizes']}")
    print("\nSTEP 3 -- RATIO-STABILITY:", ratio_stability["verdict"],
          f"(propagated_delta={propagated_delta_pct:+.2f}%)")
    print(f"  stark regime: rescued={rescued_stark:.2f}(MEASURED) -> off_equiv={off_equiv_stark:.2f} "
          f"(+{ratio_stability['stark_regime_stock_drafter']['pct_over_locked']:.1f}% over locked, clears+10={ratio_stability['stark_regime_stock_drafter']['clears_plus10_bar']})")
    print(f"  land  regime: rescued={rescued_land:.2f}(PROJECTED) -> off_equiv={off_equiv_land:.2f} "
          f"(+{ratio_stability['land_regime_better_drafter']['pct_over_locked']:.1f}% over locked, clears+10={ratio_stability['land_regime_better_drafter']['clears_plus10_bar']})")
    print(f"\nKNOB ATTRIBUTION (vs baseline {baseline_median}):")
    print(f"  {'knob':12s} {'cand_tps':>9s} {'delta%':>8s} {'verdict':>6s} {'ident_safe':>10s}")
    for k in knobs:
        print(f"  {k['label']:12s} {k['candidate_median']:>9.3f} {k['delta_pct']:>+8.3f} "
              f"{k['verdict']:>6s} {str(k['identity_safe']):>10s}  "
              f"(byte {k['identity_vs_baseline']['n_match']}/{k['identity_vs_baseline']['n_compared']})")
    for b in boot_fail:
        print(f"  {b['knob']:12s} {'--':>9s} {'--':>8s} {'BOOT_FAIL':>6s} "
              f"  ({b['land_value']}; greedy-moot)")
    if kdef:
        print(f"\nK-DEFINITION (own stack): K6={kdef['k6_median']:.2f} (e_acc {kdef['k6_e_accept']:.3f})  "
              f"K7={kdef['k7_median']:.2f} (e_acc {kdef['k7_e_accept']:.3f})  "
              f"K7-K6={kdef['k7_minus_k6_pct']:+.2f}%  K7-vs-stark155={kdef['k7_vs_stark155_pct']:+.2f}%")
    if regate:
        print(f"\nIDENTITY re-gate (#319 teacher-forced): break_rate(tau0.3)={break_rate} "
              f"attn_path_breaks={regate['attention_path_break_count']} verdict={regate['verdict']}")
    print(f"spec_fire_rate={SPEC_FIRE_RATE:.6f}  PPL={PPL_SPEC:.4f} (<= {PPL_GATE})")
    print(f"\nMATERIAL identity-safe SPEEDUP exists = {shippable_knob_exists} (>= {MATERIAL_SPEEDUP_PCT}%)  "
          f"stat-real-speedups={[(m['label'], round(m['delta_pct'], 3), 'material' if m['is_material'] else 'IMMATERIAL') for m in out['identity_safe_speedup_movers']]}  "
          f"slowdowns={[m['label'] for m in out['identity_safe_slowdown_movers']]}")
    print(f"VERDICT = {out['verdict']}")

    if args.no_wandb:
        return 0

    # ---- W&B (land-attributed; logs the explicit audit-gap scalars) ----
    from scripts import wandb_logging
    run = wandb_logging.init_wandb_run(
        job_type="spec_regime_isolation", agent="land",
        name="land/spec-regime-isolation",
        group="spec-regime-isolation-land",
        notes=("PR#664 FULL (#666-authorized cross-student read): VERDICT REGIME_IS_ACCEPTANCE. "
               "land-170 vs stark-155 un-rescued K6 gap is the DRAFTER (espec 3.657 vs 3.333), "
               "not a per-step latency knob -- step-rate identical (46.5 vs 46.7/s). Own-config "
               "knob sweep (sampler BOOT_FAIL, eager -73%+id-break, flashattn) + #319 teacher-forced "
               "identity re-gate (explicit break_rate/spec_fire_rate scalars, closing the #660 audit "
               "gap). Official harness ships stock Hub drafter -> realizes ~155/rescued-~135; land's "
               "170 needs LOCAL /tmp/qat-assistant. Ratio-stability PROPAGATES (off-equiv ~135 vs ~147)."),
        config={
            "pr": 664, "analysis_only": True, "official_tps": 0,
            "stack": "int4_mtp_batchinv", "drafter": LAND_DRAFTER,
            "batch_invariant": 1, "max_num_seqs": 1, "greedy": True, "K": 6,
            "num_prompts": 128, "output_len": 512, "seed": 1, "vllm": "0.22.0",
            "stark_drafter": STARK_DRAFTER, "stark_unrescued_k6_READ": STARK_K6_UNRESCUED,
            "stark_k6_espec_READ": STARK_K6_ESPEC, "stark_ar_rung_local_READ": STARK_AR_RUNG_LOCAL,
            "stark_rescued_k6_measured_READ": STARK_RESCUED_K6_CAPTURED,
            "locked_official": LOCKED_OFFICIAL, "scope": "full-666-authorized",
        },
        tags=["optionb", "batch_invariant", "pr664", "spec_regime", "knob_sweep",
              "served", "REGIME_IS_ACCEPTANCE", "identity_regate", "drafter_attribution"],
    )
    if run is not None:
        import wandb
        # explicit audit-gap scalars (machine-checkable in-run)
        summary = {
            "gate/break_rate_tau0p3": break_rate,
            "gate/spec_fire_rate": SPEC_FIRE_RATE,
            "gate/ppl_spec": PPL_SPEC, "gate/ppl_passes": int(PPL_SPEC <= PPL_GATE),
            "baseline/k6_median_wall_tps": baseline_median,
            "deliverable/shippable_identity_safe_speedup_exists": int(shippable_knob_exists),
            "deliverable/material_speedup_threshold_pct": MATERIAL_SPEEDUP_PCT,
            "deliverable/n_statreal_identity_safe_speedups": len(speedup_movers),
            "deliverable/n_material_identity_safe_speedups": len(material_speedup_movers),
            "deliverable/n_boot_fail_knobs": len(boot_fail),
            "decision/verdict": out["verdict"],
            "decision/official_tps": 0, "decision/fires": 0,
            # STEP 1 -- espec discriminator scalars (the verdict driver)
            "disc/land_k6_espec": land_k6_espec, "disc/stark_k6_espec": STARK_K6_ESPEC,
            "disc/espec_ratio": espec_ratio, "disc/walltps_ratio": walltps_ratio,
            "disc/step_rate_land_per_s": step_rate_land, "disc/step_rate_stark_per_s": step_rate_stark,
            "disc/step_rate_ratio": step_rate_ratio,
            "disc/espec_is_acceptance": int(espec_is_acceptance),
            # STEP 3 -- ratio-stability official-equiv for BOTH regimes
            "ratio/verdict_propagates": int(ratio_stability["verdict"] == "PROPAGATES"),
            "ratio/propagated_delta_pct": propagated_delta_pct,
            "ratio/stark_rescued_k6_MEASURED": rescued_stark,
            "ratio/stark_official_equiv": off_equiv_stark,
            "ratio/stark_clears_plus10": int(ratio_stability["stark_regime_stock_drafter"]["clears_plus10_bar"]),
            "ratio/land_rescued_k6_PROJECTED": rescued_land,
            "ratio/land_official_equiv": off_equiv_land,
            "ratio/land_clears_plus10": int(bool(ratio_stability["land_regime_better_drafter"]["clears_plus10_bar"])),
        }
        if regate:
            summary["gate/attention_path_break_count"] = regate["attention_path_break_count"]
            summary["gate/break_rate_bi1_both_sides"] = regate["break_rate_bi1_both_sides"]
            for t in (regate["tau_ladder"] or []):
                summary[f"gate/tau_{str(t['tau']).replace('.', 'p')}_rate"] = t["rate"]
        if kdef:
            summary["kdef/k6_median"] = kdef["k6_median"]
            summary["kdef/k7_median"] = kdef["k7_median"]
            summary["kdef/k7_minus_k6_pct"] = kdef["k7_minus_k6_pct"]
            summary["kdef/k7_vs_stark155_pct"] = kdef["k7_vs_stark155_pct"]
        cols = ["knob", "land_value", "candidate_wall_tps", "delta_pct", "verdict",
                "identity_safe", "byte_match", "candidate_e_accept"]
        tbl = wandb.Table(columns=cols)
        knob_vals = {"sampler1": "VLLM_USE_FLASHINFER_SAMPLER=1",
                     "eager1": "ENFORCE_EAGER=1", "flashattn": "VLLM_ATTENTION_BACKEND=FLASH_ATTN"}
        for k in knobs:
            iv = k["identity_vs_baseline"]
            tbl.add_data(k["label"], knob_vals.get(k["label"], "?"), k["candidate_median"],
                         k["delta_pct"], k["verdict"], k["identity_safe"],
                         f"{iv['n_match']}/{iv['n_compared']}", k["candidate_e_accept"])
            run.log({f"knob/{k['label']}_delta_pct": k["delta_pct"],
                     f"knob/{k['label']}_wall_tps": k["candidate_median"],
                     f"knob/{k['label']}_identity_safe": int(k["identity_safe"])})
        for b in boot_fail:
            tbl.add_data(b["knob"], b["land_value"], None, None, "BOOT_FAIL",
                         b.get("identity_safe", False), "n/a", None)
        run.log({"knob_attribution": tbl})
        wandb_logging.log_summary(run, summary, step=6)
        wandb_logging.log_json_artifact(run, name="spec_regime_isolation_664",
                                        artifact_type="analysis", data=out)
        url = getattr(run, "url", ""); rid = getattr(run, "id", "")
        wandb_logging.finish_wandb(run)
        print(f"[wandb] spec-regime-isolation id={rid} url={url}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
