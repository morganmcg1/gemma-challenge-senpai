"""Realize-or-invert wall A/B for the +1.1982 byte-exact ``num_stages 3->2`` ceiling
(PR #468, wirbel) — the THIRD and final closure of the Triton verify-attention surface.

My #459 ([`6pwhesdy`]) closed the surface two ways (greedy-unsafe tile retune #442 FLAG-1
AND byte-exact-immaterial-modeled) and produced the program's last positive byte-exact
supply number: a full-surface ``num_stages 3->2`` retune of the deployed Triton
``kernel_unified_attention`` 3D split-KV launch, ``maxdiff == 0.0`` on BOTH heads
(identity-preserving), modeled at a +1.1982 TPS Amdahl ceiling

    ceiling = (n512*saving512 + n256*saving256) * tps_per_us
            = (7*0.6656 + 30*0.549547) * 0.056663
            = 21.14561 us * 0.056663 = 1.19817 TPS         (#459 reconcile)

This orchestrator answers the ONE open question: does that modeled +1.20 REALIZE on the
served wall, or does it INVERT like the five prior modeled microbench leads
(pinned-K +13.998->-5.82 #433 / cb3 +15.60->0.0 #437 / static-K +13.2%->-8.63% #273 /
autotune-isolated +15.86->-5.65 #442 / relax-prize +17->-0.94 #452)?

WHY ``num_stages`` is the RARE legitimate byte-exact lever (unlike bm4): ``num_stages``
sets the cp.async software-pipeline depth; 3->2 frees shared memory (occupancy) WITHOUT
reordering the online-softmax FMA sequence -> bit-identical output (maxdiff 0.0, the banked
#270 result). Crucially it does NOT change the grid/CTA count, so it should NOT carry
bm4's CTA-tripling penalty (96->288 CTAs) that drove #442's -5.65. The deployed launch
(triton_unified_attention.py L967) passes NO num_stages -> Triton default 3; forcing 2 is
a pure config override.

MECHANISM — same env-gated, auto-reverted injector A/B harness as #442/#459, NO new served
file, NO kernel rebuild, NO source patch: the candidate arm sets
``WIRBEL_BM4_AB=1 WIRBEL_BM4_BLOCK_M=16 WIRBEL_BM4_NUM_STAGES=2``. With BLOCK_M held at the
deployed 16 the injector's grid recompute is a NO-OP (BLOCK_Q stays 4, grid dim0 =
q_rows//4 + num_seqs unchanged) -> the ONLY delta vs baseline is ``num_stages`` 3->2. The
served_bm4_injector docstring names this exact path: "WIRBEL_BM4_BLOCK_M=16 ->
num_stages-only isolation (BLOCK_M unchanged, BLOCK_Q=4)". Baseline arm = deployed default
(WIRBEL_BM4_AB unset -> injector not even imported). So this is genuinely a one-variable
``num_stages``-only A/B through the proven, reverted harness.

ONEGRAPH-SURVIVAL (the stark #466 cross-check): the meta-path finder patches the kernel
module BEFORE vLLM imports it -> the num_stages=2 cubin is compiled during warmup and baked
into the captured ONEGRAPH whole-step graph. #459 proved the M=8 verify is captured (the
Python wrapper fires ONLY during the brief capture warmup, then the graph replays). So a
healthy realization shows: PATCHED present, forced>0, the forced count BOUNDED (warmup-only,
~1e2, NOT the ~1e5+ a per-step eager fallback would log), and the candidate wall_tps NOT
collapsed toward the BI=1/serial 161.70. That bounded-forced + non-collapse is the direct
datapoint #466 needs about whether a config-level attention change survives capture.

HONEST EXPECTATION (stated up front, matching the PR): +1.1982 < +2 materiality bar
(< sigma_hw 4.8) -> even a clean realization does NOT reopen the frontier. The num_stages-
only sub-lever precedent realized <=+0.94 vs +13.23 modeled (#452, a ~14x kernel-vs-wall
haircut); the most likely outcome here is a small realize or within-noise. This is the
THIRD closure (realized, not just modeled) + a harness cross-check, not a reopener.

Adds 0 TPS, changes NO served file (the sitecustomize hook is env-gated, reverted, NEVER
submitted; the PR diff carries only research/**). NOT an HF Job, NOT a submission, NOT a
launch. BASELINE stays 481.53.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
# Reuse the proven #442 toggle + injector + per-seed paired-A/B machinery verbatim.
from research.validity.triton_attn_joint_autotune import served_bm4_wall_ab as ab  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants imported EXACT from the #459 reconcile/microbench (6pwhesdy);
# NOT re-derived. (research/equivalence_escalation/triton_verify_surface/*.json)
# ---------------------------------------------------------------------------
DEPLOYED_INCUMBENT_TPS = 481.53          # PR #52 deployed (identity 0.9966, 3 flips {11,18,118})
STRICT_BASE_TPS = 467.14                 # denken #423 realized blanket-strict frontier
MODELED_CEILING_TPS = 1.1981736994299998  # #459 full-surface byte-exact num_stages 3->2
HEAD512_ONLY_CEILING_TPS = 0.2640042496  # #447 head-512-only (= best realized byte-exact to date)
TPS_PER_US = 0.056663
SAVING_US_256 = 0.549547
SAVING_US_512 = 0.6656
TOTAL_BYTE_EXACT_SAVING_US = 21.145609999999998
GEOMETRY = {"n256_sliding": 30, "n512_global": 7, "n_total": 37}
MATERIALITY_TPS = 2.0
SIGMA_HW_TPS = 4.8
BEST_REALIZED_BYTE_EXACT_TPS = 0.26      # #447 head-512-only realized byte-exact lever
PPL_ANCHOR = 2.3772
PPL_GATE = 2.42
# num_stages-only sub-lever precedent (the kernel-vs-wall haircut): modeled vs realized UB.
NUMSTAGES_ONLY_MODELED_TPS = 13.226254718079531   # #452/#428 num_stages-only modeled
NUMSTAGES_ONLY_REALIZED_UB_TPS = 0.94             # its realized upper band
# Five modeled microbench leads that INVERTED end-to-end (the realize-or-collapse ledger).
INVERSION_LEDGER = {
    "pinned_K_pr433": [13.998, -5.82], "cb3_pr437": [15.60, 0.0],
    "static_K_pr273_pct": [13.2, -8.63], "autotune_isolated_pr442": [15.86, -5.65],
    "relax_prize_pr452": [17.0, -0.94],
}

REALIZES_RATIO = 0.8                      # >=80% of modeled realized & significant -> "realizes"
Z95 = 1.959963984540054
# Student-t 0.975 two-sided multipliers (small df = n_seeds-1); falls back to Z for df>=30.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060}

# ONEGRAPH-survival: a per-step EAGER fallback would force ~(prompts*steps*layers) ~ 1e5+
# Python launches; a captured-and-baked override fires only during warmup -> ~1e2. This
# threshold sits two orders of magnitude below per-step eager and an order above warmup.
ONEGRAPH_BOUNDED_FORCED_MAX = 5000
# A capture-broken serial/BI=1 fallback collapses the candidate toward 161.70; a survived
# override keeps it within a hair of the baseline arm. Flag collapse below this fraction.
ONEGRAPH_NONCOLLAPSE_FRAC = 0.90


def _log(msg: str) -> None:
    print(f"[numstages-wall] {msg}", file=sys.stderr, flush=True)


def _finite_pos(xs: list) -> list[float]:
    return [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(x) and x > 0]


def _t975(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df >= 30:
        return Z95
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    for k in keys:
        if k >= df:
            return _T975[k]
    return Z95


# ---------------------------------------------------------------------------
# Per-seed runner args: reuse ab.run_seed (it wires paired_tps_ab with the
# candidate-env num_stages override and reuses an on-disk completed seed).
# ---------------------------------------------------------------------------
def _seed_args(args) -> argparse.Namespace:
    return argparse.Namespace(
        block_m=args.block_m, num_stages=args.num_stages,
        baseline_label=args.baseline_label, candidate_label=args.candidate_label,
        num_prompts=args.num_prompts, output_len=args.output_len,
        wandb_group=args.wandb_group, wandb_name=args.wandb_name,
        no_wandb=args.no_wandb, fresh=args.fresh,
    )


# ---------------------------------------------------------------------------
# ONEGRAPH-survival: parse the injector's bounded forced-launch milestones.
# ---------------------------------------------------------------------------
_FORCED_RE = re.compile(r"forced bm4 \(count=(\d+)\)")


def _max_forced_milestone(log_paths: list[Path]) -> int:
    mx = 0
    for p in log_paths:
        try:
            for m in _FORCED_RE.finditer(p.read_text(errors="replace")):
                mx = max(mx, int(m.group(1)))
        except Exception:  # noqa: BLE001
            pass
    return mx


def onegraph_survival(out_root: Path, seeds: list[int], verifies: dict[int, dict],
                      agg: dict[str, Any], cand_label: str) -> dict[str, Any]:
    """Did the num_stages override apply INSIDE the captured ONEGRAPH whole-step graph?

    Survives  <=>  PATCHED (wrapper installed before kernel import) AND forced>0 (override
    fired) AND the forced count is BOUNDED (warmup-only, ~1e2 -> baked into the replayed
    graph; NOT the ~1e5+ of a per-step eager fallback) AND the candidate wall_tps did NOT
    collapse toward the serial/BI=1 161.70 (which a capture-break would cause)."""
    cand_logs: list[Path] = []
    for s in seeds:
        cand_logs += ab._arm_run_logs(out_root / f"seed{s}" / cand_label)
    max_forced = _max_forced_milestone(cand_logs)

    all_patched = bool(verifies) and all(v.get("candidate_all_runs_patched") for v in verifies.values())
    served_3d = bool(verifies) and all(v.get("served_verify_is_3d") for v in verifies.values())
    heads = sorted({h for v in verifies.values() for h in (v.get("census_heads") or [])})
    both_heads = (256 in heads and 512 in heads)
    forced_total = sum(v.get("candidate_forced_log_hits", 0) for v in verifies.values())

    bounded = (0 < max_forced <= ONEGRAPH_BOUNDED_FORCED_MAX) if max_forced else (forced_total > 0)
    base_p50 = agg.get("base_pooled_p50_wall_tps")
    cand_p50 = agg.get("cand_pooled_p50_wall_tps")
    not_collapsed = (isinstance(base_p50, (int, float)) and isinstance(cand_p50, (int, float))
                     and base_p50 > 0 and cand_p50 >= ONEGRAPH_NONCOLLAPSE_FRAC * base_p50)

    survives = bool(all_patched and served_3d and bounded and not_collapsed)
    mechanism = (
        "num_stages=2 cubin compiled during warmup and BAKED into the captured ONEGRAPH "
        "whole-step graph; the Python wrapper fires only during capture warmup "
        f"(forced bounded, max milestone {max_forced}), then the graph replays the baked "
        "kernel. Candidate wall_tps did not collapse toward the serial 161.70 -> the "
        "config-level attention change SURVIVED capture (no recapture, no BI=1 fallback)."
        if survives else
        "config-level num_stages override did NOT cleanly survive capture: "
        f"patched={all_patched} bounded_forced={bounded} (max milestone {max_forced}) "
        f"served_3d={served_3d} not_collapsed={not_collapsed}."
    )
    return {
        "numstages_survives_onegraph": survives,
        "candidate_all_runs_patched": all_patched,
        "served_verify_is_3d": served_3d,
        "census_heads": heads, "both_heads_overridden": both_heads,
        "max_forced_milestone": max_forced,
        "forced_log_hits_total": forced_total,
        "forced_bounded_warmup_only": bounded,
        "candidate_not_collapsed": not_collapsed,
        "base_pooled_p50_wall_tps": base_p50, "cand_pooled_p50_wall_tps": cand_p50,
        "mechanism": mechanism,
    }


# ---------------------------------------------------------------------------
# Aggregate: pooled + per-seed-paired realized delta vs the modeled +1.1982.
# ---------------------------------------------------------------------------
def aggregate(seed_jsons: list[Path]) -> dict[str, Any]:
    base_vals: list[float] = []
    cand_vals: list[float] = []
    per_seed = []
    seed_delta_pcts: list[float] = []
    for pj in seed_jsons:
        d = json.loads(pj.read_text())
        seed = d["workload"]["seed"]
        b = d["arms"]["baseline"]["wall_tps"]
        c = d["arms"]["candidate"]["wall_tps"]
        bv = _finite_pos(b.get("values") or [])
        cv = _finite_pos(c.get("values") or [])
        base_vals += bv
        cand_vals += cv
        bm = statistics.median(bv) if bv else None
        cm = statistics.median(cv) if cv else None
        sd = (100.0 * (cm - bm) / bm) if (bm and cm) else None
        if sd is not None:
            seed_delta_pcts.append(sd)
        proj = (d.get("projection") or {}).get("arms") or {}
        per_seed.append({
            "seed": seed, "base_median": bm, "cand_median": cm,
            "base_values": bv, "cand_values": cv, "delta_pct": sd,
            "base_projected_official": (proj.get("baseline") or {}).get("projected_official"),
            "cand_projected_official": (proj.get("candidate") or {}).get("projected_official"),
        })

    out: dict[str, Any] = {"per_seed": per_seed, "n_base": len(base_vals), "n_cand": len(cand_vals),
                           "n_seeds": len(seed_delta_pcts)}
    if not base_vals or not cand_vals:
        out["error"] = "missing pooled wall_tps values"
        return out

    base_p50 = statistics.median(base_vals)
    cand_p50 = statistics.median(cand_vals)
    pooled_delta_pct = 100.0 * (cand_p50 - base_p50) / base_p50

    # Pooled two-sample CI (cross-check; includes between-seed spread).
    sd_b = statistics.stdev(base_vals) if len(base_vals) > 1 else 0.0
    sd_c = statistics.stdev(cand_vals) if len(cand_vals) > 1 else 0.0
    se_diff = math.hypot(sd_b / math.sqrt(len(base_vals)), sd_c / math.sqrt(len(cand_vals)))
    pooled_ci_pct = 100.0 * Z95 * se_diff / base_p50

    # PRIMARY: per-seed-paired delta (proper paired design across N seeds).
    if seed_delta_pcts:
        mean_delta_pct = statistics.mean(seed_delta_pcts)
        if len(seed_delta_pcts) > 1:
            se_seed = statistics.stdev(seed_delta_pcts) / math.sqrt(len(seed_delta_pcts))
            ci_seed_pct = _t975(len(seed_delta_pcts) - 1) * se_seed
        else:
            se_seed = None
            ci_seed_pct = pooled_ci_pct  # single seed -> fall back to pooled CI
    else:
        mean_delta_pct = pooled_delta_pct
        se_seed = None
        ci_seed_pct = pooled_ci_pct

    # Realized TPS delta on the deployed incumbent (the headline numstages_realized_tps_delta).
    realized_tps = DEPLOYED_INCUMBENT_TPS * mean_delta_pct / 100.0
    ci_tps = DEPLOYED_INCUMBENT_TPS * ci_seed_pct / 100.0
    delta_ci_lo = realized_tps - ci_tps
    delta_ci_hi = realized_tps + ci_tps
    ratio = realized_tps / MODELED_CEILING_TPS

    # realize / partial / invert / within_noise vs the modeled +1.1982.
    if delta_ci_hi < 0:
        cls = "inverts"
    elif delta_ci_lo > 0:
        cls = "realizes" if ratio >= REALIZES_RATIO else "partial"
    else:
        cls = "within_noise"
    realizes = (cls == "realizes")

    # Three-way surface closure (deliverable #5): greedy-unsafe tile (#442) + byte-exact-
    # immaterial-modeled (#459) + byte-exact-REALIZED (this run): the realized CI upper bound
    # stays below the +2 materiality bar -> the surface is closed a third way (realized).
    realized_immaterial = bool(delta_ci_hi < MATERIALITY_TPS)
    closed_three_ways = bool(
        True                                  # (1) greedy-unsafe tile retune closed (#442 bm4 -5.65)
        and (MODELED_CEILING_TPS < MATERIALITY_TPS)  # (2) byte-exact modeled immaterial (#459)
        and realized_immaterial               # (3) byte-exact realized immaterial (this run)
    )

    base_proj = [ps["base_projected_official"] for ps in per_seed
                 if isinstance(ps["base_projected_official"], (int, float))]
    cand_proj = [ps["cand_projected_official"] for ps in per_seed
                 if isinstance(ps["cand_projected_official"], (int, float))]

    out.update({
        "base_pooled_p50_wall_tps": base_p50,
        "cand_pooled_p50_wall_tps": cand_p50,
        "base_pooled_values": base_vals, "cand_pooled_values": cand_vals,
        "seed_delta_pcts": seed_delta_pcts,
        # primary (per-seed-paired)
        "mean_delta_pct": mean_delta_pct,
        "se_seed_delta_pct": se_seed,
        "ci95_delta_pct": ci_seed_pct,
        "numstages_realized_tps_delta": realized_tps,
        "realized_tps_delta_ci95": ci_tps,
        "realized_tps_delta_ci95_lo": delta_ci_lo,
        "realized_tps_delta_ci95_hi": delta_ci_hi,
        # cross-check (pooled)
        "pooled_delta_pct": pooled_delta_pct,
        "pooled_ci95_pct": pooled_ci_pct,
        # verdict
        "modeled_ceiling_tps": MODELED_CEILING_TPS,
        "realization_ratio": ratio,
        "classification": cls,
        "numstages_realizes": realizes,
        "realized_immaterial_below_2tps": realized_immaterial,
        "attention_surface_closed_three_ways": closed_three_ways,
        "base_projected_official_median": statistics.median(base_proj) if base_proj else None,
        "cand_projected_official_median": statistics.median(cand_proj) if cand_proj else None,
    })
    return out


# ---------------------------------------------------------------------------
# Census ingest (deliverable #3): served greedy-identity + PPL, run separately by
# served_bm4_census.py --block-m 16 --num-stages 2 (num_stages-only token A/B).
# ---------------------------------------------------------------------------
def ingest_census(census_json: Path | None) -> dict[str, Any]:
    if census_json is None or not Path(census_json).exists():
        return {"census_run": False, "numstages_identity_fraction": None,
                "byte_exact": None, "ppl": PPL_ANCHOR, "note": "census not provided"}
    d = json.loads(Path(census_json).read_text())
    cmp = d.get("comparison") or {}
    return {
        "census_run": True,
        "numstages_identity_fraction": cmp.get("frac_identical"),
        "frac_token_prefix_match": cmp.get("frac_token_prefix_match"),
        "byte_exact": cmp.get("byte_exact"),
        "n_prompts": cmp.get("n_prompts"),
        "ppl": d.get("ppl_bm4") if d.get("ppl_bm4") is not None else PPL_ANCHOR,
        "ppl_ok": d.get("ppl_ok"),
        "census_verdict_pass": d.get("verdict_byte_exact_and_ppl_pass"),
        "census_attestation": d.get("attestation"),
        "census_config": d.get("config"),
    }


# ---------------------------------------------------------------------------
# Self-test (PRIMARY headline).
# ---------------------------------------------------------------------------
def constants_exact() -> dict[str, Any]:
    surf = GEOMETRY["n512_global"] * SAVING_US_512 + GEOMETRY["n256_sliding"] * SAVING_US_256
    ceil_recompute = surf * TPS_PER_US
    checks = {
        "ceiling_recomputes": abs(ceil_recompute - MODELED_CEILING_TPS) < 1e-6,
        "surface_us_exact": abs(surf - TOTAL_BYTE_EXACT_SAVING_US) < 1e-6,
        "geometry_37": GEOMETRY["n_total"] == GEOMETRY["n256_sliding"] + GEOMETRY["n512_global"] == 37,
        "ceiling_below_materiality": MODELED_CEILING_TPS < MATERIALITY_TPS,
        "ceiling_below_sigma_hw": MODELED_CEILING_TPS < SIGMA_HW_TPS,
        "head512_only_is_447_026": abs(HEAD512_ONLY_CEILING_TPS - 0.2640042496) < 1e-9,
    }
    return {"all_exact": all(checks.values()), "checks": checks,
            "ceiling_recompute": ceil_recompute, "surface_us": surf}


def self_test(agg: dict[str, Any], verifies: dict[int, dict], survival: dict[str, Any],
              census: dict[str, Any], smoke: bool) -> dict[str, Any]:
    cexact = constants_exact()
    base_vals = agg.get("base_pooled_values") or []
    cand_vals = agg.get("cand_pooled_values") or []
    all_finite = (bool(base_vals) and bool(cand_vals)
                  and len(_finite_pos(base_vals)) == len(base_vals)
                  and len(_finite_pos(cand_vals)) == len(cand_vals))

    candidate_applied = bool(verifies) and all(v.get("applied_ok") for v in verifies.values())
    both_heads = bool(survival.get("both_heads_overridden"))
    classified = agg.get("classification") in ("realizes", "partial", "inverts", "within_noise")

    required = {
        "constants_exact": cexact["all_exact"],
        "candidate_actually_applied_numstages": candidate_applied,
        "both_heads_overridden_256_and_512": both_heads,
        "served_verify_is_3d_census": bool(survival.get("served_verify_is_3d")),
        "realized_delta_classified": classified,
        "all_tps_finite_positive": all_finite,
        "onegraph_survival_resolved": isinstance(survival.get("numstages_survives_onegraph"), bool),
    }
    if not smoke:
        # N>=5 seeds required by the PR; identity census must have been run + byte-exact.
        required["at_least_5_seeds"] = (agg.get("n_seeds") or 0) >= 5
        required["identity_census_byte_exact"] = bool(census.get("census_run") and census.get("byte_exact"))

    passes = all(required.values())
    return {"numstages_self_test_passes": passes, "required": required, "smoke": smoke,
            "constants": cexact}


# ---------------------------------------------------------------------------
# W&B (the PR-required metric surface).
# ---------------------------------------------------------------------------
def log_wandb(args, result: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        _log(f"wandb_logging import failed ({exc}); skipping")
        return None
    agg = result["aggregate"]
    st = result["self_test"]
    surv = result["onegraph"]
    cen = result["census"]
    run = wandb_logging.init_wandb_run(
        job_type="numstages-realize-wall-ab", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["pr468", "equivalence-escalation", "numstages-realize", "triton-attn",
              "byte-exact", ab.SUBMISSION],
        config={
            "deployed_incumbent_tps": DEPLOYED_INCUMBENT_TPS, "strict_base_tps": STRICT_BASE_TPS,
            "modeled_ceiling_tps": MODELED_CEILING_TPS, "tps_per_us": TPS_PER_US,
            "saving_us_256": SAVING_US_256, "saving_us_512": SAVING_US_512,
            "geometry": GEOMETRY, "materiality_tps": MATERIALITY_TPS, "sigma_hw_tps": SIGMA_HW_TPS,
            "block_m": args.block_m, "num_stages": args.num_stages, "baseline_num_stages": 3,
            "seeds": args.seeds, "n": args.n, "num_prompts": args.num_prompts,
            "output_len": args.output_len, "smoke": args.smoke,
            "lever": "verify_triton_num_stages_3to2_block_m_16_byte_exact",
            "analysis_only": True, "no_served_file_change": True, "official_tps": 0,
        },
    )
    if run is None:
        _log("wandb disabled (no API key / WANDB_DISABLED); skipping")
        return None
    run_id = getattr(run, "id", None)
    try:
        flat: dict[str, Any] = {
            # ---- PR-required metric surface ----
            "numstages_realized_tps_delta": agg.get("numstages_realized_tps_delta"),
            "numstages_modeled_ceiling": MODELED_CEILING_TPS,
            "numstages_realizes": float(bool(agg.get("numstages_realizes"))),
            "numstages_survives_onegraph": float(bool(surv.get("numstages_survives_onegraph"))),
            "attention_surface_closed_three_ways": float(bool(agg.get("attention_surface_closed_three_ways"))),
            "numstages_self_test_passes": float(bool(st["numstages_self_test_passes"])),
            "analysis_only": 1.0, "no_served_file_change": 1.0, "official_tps": 0.0,
            "ppl": cen.get("ppl") if cen.get("ppl") is not None else PPL_ANCHOR,
            # ---- realization detail ----
            "realized_tps_delta_ci95": agg.get("realized_tps_delta_ci95"),
            "realized_tps_delta_ci95_lo": agg.get("realized_tps_delta_ci95_lo"),
            "realized_tps_delta_ci95_hi": agg.get("realized_tps_delta_ci95_hi"),
            "realization_ratio": agg.get("realization_ratio"),
            "mean_delta_pct": agg.get("mean_delta_pct"),
            "ci95_delta_pct": agg.get("ci95_delta_pct"),
            "pooled_delta_pct": agg.get("pooled_delta_pct"),
            "pooled_ci95_pct": agg.get("pooled_ci95_pct"),
            "base_pooled_p50_wall_tps": agg.get("base_pooled_p50_wall_tps"),
            "cand_pooled_p50_wall_tps": agg.get("cand_pooled_p50_wall_tps"),
            "base_projected_official_median": agg.get("base_projected_official_median"),
            "cand_projected_official_median": agg.get("cand_projected_official_median"),
            "n_seeds": agg.get("n_seeds"), "n_base_runs": agg.get("n_base"),
            "n_cand_runs": agg.get("n_cand"),
            # ---- onegraph-survival detail ----
            "onegraph_max_forced_milestone": surv.get("max_forced_milestone"),
            "onegraph_forced_bounded": float(bool(surv.get("forced_bounded_warmup_only"))),
            "onegraph_candidate_not_collapsed": float(bool(surv.get("candidate_not_collapsed"))),
            # ---- identity census detail ----
            "numstages_identity_fraction": cen.get("numstages_identity_fraction"),
            "numstages_identity_byte_exact": (float(bool(cen.get("byte_exact")))
                                              if cen.get("byte_exact") is not None else None),
            "census_run": float(bool(cen.get("census_run"))),
        }
        flat = {k: v for k, v in flat.items() if isinstance(v, (int, float))}
        for k, v in st["required"].items():
            flat[f"selftest/{k}"] = float(bool(v))
        run.summary["classification"] = agg.get("classification")
        run.summary["onegraph_mechanism"] = surv.get("mechanism")
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="numstages_realize_wall_ab", artifact_type="wall-realization", data=result)
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN wandb logging error: {exc}")
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass
    return run_id


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="1,2,3,4,5", help="comma-separated seeds (>=5 for headline)")
    ap.add_argument("--n", type=int, default=2, help="fresh runs per arm per seed (median-of-N)")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    # num_stages-ONLY: BLOCK_M held at the deployed 16 (grid recompute is a no-op) so the
    # only delta vs baseline is num_stages 3->2. DO NOT change --block-m (that is the bm4
    # grid-changing, greedy-unsafe lever; 16 is the byte-exact num_stages isolation).
    ap.add_argument("--block-m", dest="block_m", type=int, default=16)
    ap.add_argument("--num-stages", dest="num_stages", type=int, default=2)
    ap.add_argument("--baseline-label", default="bm16_s3")
    ap.add_argument("--candidate-label", default="bm16_s2")
    ap.add_argument("--census-json", type=Path, default=None,
                    help="served_bm4_census results.json (block_m=16) for identity+PPL ingest")
    ap.add_argument("--smoke", action="store_true",
                    help="cheap boot+patch-fires check: n=1, seeds=1, 5-seed/census not required")
    ap.add_argument("--self-test", dest="self_test_exit", action="store_true",
                    help="exit non-zero if the self-test fails")
    ap.add_argument("--no-toggle", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore on-disk paired_ab.json, re-run every seed")
    ap.add_argument("--out-root", type=Path, default=HERE / "numstages_ab_out")
    ap.add_argument("--wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", default="wirbel/numstages-realize")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.block_m != 16:
        raise SystemExit(f"--block-m must be 16 for the num_stages-only byte-exact isolation "
                         f"(got {args.block_m}); any other value is the bm4 grid-changing lever.")
    if args.smoke:
        args.n = 1
        seeds = [1]
    else:
        seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    _log(f"seeds={seeds} n={args.n} smoke={args.smoke} cfg=bm{args.block_m}_s{args.num_stages} "
         f"(num_stages-only; baseline s3) workload={args.num_prompts}x{args.output_len} -> {out_root}")
    _log(f"modeled_ceiling={MODELED_CEILING_TPS:+.4f} TPS ({100*MODELED_CEILING_TPS/DEPLOYED_INCUMBENT_TPS:+.4f}% "
         f"of {DEPLOYED_INCUMBENT_TPS}); materiality +{MATERIALITY_TPS} sigma_hw {SIGMA_HW_TPS}")

    seed_args = _seed_args(args)
    t0 = time.time()
    original_bytes = None
    toggled = False
    try:
        if not args.no_toggle:
            ab.ensure_clean_toggle()
            original_bytes = ab.apply_toggle()
            toggled = True
        seed_jsons: list[Path] = []
        verifies: dict[int, dict] = {}
        for seed in seeds:
            pj = ab.run_seed(seed, args.n, out_root, seed_args)
            seed_jsons.append(pj)
            verifies[seed] = ab.verify_arms_applied(out_root / f"seed{seed}",
                                                    args.candidate_label, args.baseline_label)
            _log(f"seed {seed}: applied_ok={verifies[seed].get('applied_ok')} "
                 f"forced_hits={verifies[seed].get('candidate_forced_log_hits')} "
                 f"served_3d={verifies[seed].get('served_verify_is_3d')} "
                 f"heads={verifies[seed].get('census_heads')}")
    finally:
        toggle_clean = ab.revert_toggle(original_bytes) if toggled and original_bytes is not None else True

    agg = aggregate(seed_jsons)
    survival = onegraph_survival(out_root, seeds, verifies, agg, args.candidate_label)
    census = ingest_census(args.census_json)
    st = self_test(agg, verifies, survival, census, smoke=args.smoke)
    st["toggle_reverted_clean"] = toggle_clean
    if not toggle_clean:
        st["required"]["toggle_reverted_clean"] = False
        st["numstages_self_test_passes"] = False

    result = {
        "experiment": "numstages_realize_wall_ab", "pr": 468, "student": "wirbel",
        "question": "Does the modeled +1.1982 TPS byte-exact num_stages 3->2 ceiling realize on "
                    "the served wall, or invert (the 6th isolation-trap)?",
        "lever": "verify Triton num_stages 3->2 (BLOCK_M=16 held -> grid unchanged -> byte-exact)",
        "analysis_only": True, "no_served_file_change": True, "official_tps": 0,
        "not_a_launch": True, "not_a_build": True, "not_a_submission": True,
        "config": {"block_m": args.block_m, "num_stages": args.num_stages, "baseline_num_stages": 3,
                   "tile": 32, "num_warps": 4},
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len, "seeds": seeds},
        "inversion_ledger": INVERSION_LEDGER,
        "numstages_only_precedent": {"modeled_tps": NUMSTAGES_ONLY_MODELED_TPS,
                                     "realized_ub_tps": NUMSTAGES_ONLY_REALIZED_UB_TPS},
        "elapsed_s": time.time() - t0,
        "aggregate": agg, "onegraph": survival, "census": census, "self_test": st,
    }
    (out_root / "results.json").write_text(json.dumps(result, indent=2, default=str))

    # ---- console verdict ----
    print("\n" + "=" * 78, flush=True)
    print("num_stages 3->2 BYTE-EXACT REALIZE-OR-INVERT WALL A/B (PR #468, wirbel)", flush=True)
    print("=" * 78, flush=True)
    if "error" not in agg:
        print(f"  A bm16_s3 (deployed) pooled p50 wall_tps = {agg['base_pooled_p50_wall_tps']:.4f}", flush=True)
        print(f"  B bm16_s2 (num_stages=2) pooled p50 wall_tps = {agg['cand_pooled_p50_wall_tps']:.4f}", flush=True)
        print(f"  per-seed-paired Δ = {agg['mean_delta_pct']:+.4f}% (CI95 ±{agg['ci95_delta_pct']:.4f}%, "
              f"{agg['n_seeds']} seeds)", flush=True)
        print(f"  >>> numstages_realized_tps_delta = {agg['numstages_realized_tps_delta']:+.4f} TPS "
              f"[CI95 {agg['realized_tps_delta_ci95_lo']:+.4f} .. {agg['realized_tps_delta_ci95_hi']:+.4f}]  "
              f"(modeled +{MODELED_CEILING_TPS:.4f})", flush=True)
        print(f"  >>> realization_ratio = {agg['realization_ratio']:+.4f}  [{agg['classification']}]  "
              f"realizes={agg['numstages_realizes']}", flush=True)
        print(f"  >>> attention_surface_closed_three_ways = {agg['attention_surface_closed_three_ways']}", flush=True)
        print(f"  >>> numstages_survives_onegraph = {survival['numstages_survives_onegraph']} "
              f"(max_forced={survival['max_forced_milestone']}, heads={survival['census_heads']})", flush=True)
        print(f"      {survival['mechanism']}", flush=True)
        if census.get("census_run"):
            print(f"  >>> numstages_identity_fraction = {census['numstages_identity_fraction']} "
                  f"(byte_exact={census['byte_exact']}); ppl={census['ppl']} (gate {PPL_GATE})", flush=True)
        else:
            print("  >>> identity census not yet ingested (run served_bm4_census --block-m 16 "
                  "--num-stages 2, pass --census-json)", flush=True)
    else:
        print(f"  AGGREGATE ERROR: {agg['error']}", flush=True)
    for ps in agg.get("per_seed", []):
        print(f"   seed{ps['seed']}: A={ps.get('base_median')} B={ps.get('cand_median')} "
              f"Δ={ps.get('delta_pct')}%", flush=True)
    print(f"  toggle_reverted_clean = {toggle_clean}", flush=True)
    print(f"  >>> SELF-TEST PASSES = {st['numstages_self_test_passes']} required={st['required']}", flush=True)
    print("=" * 78 + "\n", flush=True)

    run_id = log_wandb(args, result)
    if run_id:
        result["wandb_run_id"] = run_id
        (out_root / "results.json").write_text(json.dumps(result, indent=2, default=str))
        print(f"[numstages-wall] wandb run id = {run_id}", flush=True)
    print(f"[numstages-wall] artifacts -> {out_root / 'results.json'}", flush=True)

    if args.self_test_exit and not st["numstages_self_test_passes"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
