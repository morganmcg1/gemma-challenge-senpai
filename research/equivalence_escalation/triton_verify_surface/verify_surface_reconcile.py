#!/usr/bin/env python
"""PR #459 -- Reconcile the Triton verify-attention surface (#447 1.27% vs #442 head-256)
and compute the BYTE-EXACT full-surface retune ceiling.

This is the orchestrator. It consumes two MEASURED artifacts and does pure arithmetic
(no GPU):
  * ``microbench_results.json`` (verify_surface_microbench.py) -- per-kernel deployed us
    and the num_stages-only BYTE-EXACT saving for BOTH served head dims (head-512 global,
    head-256 sliding), plus the realized-frontier Amdahl slope tps_per_us.
  * ``census_results.json`` (verify_surface_census.py) -- the served routing count: how
    many head-256 sliding layers (n256) actually reach the Triton 3D split-KV kernel at
    the M=8 verify, derived per-forward off the 7 head-512 global layers as a clock.

THREE DELIVERABLES (PR #459):
  (1) the TRUE Triton verify-attn fraction of T_verify (head-256 sliding + head-512 global,
      Triton per-call), vs #447's head-512-only 1.274%;
  (2) whether #447 UNDERCOUNTED and by how much (it timed the head-256 sliding layers as
      FA2 @21.81us x35 and excluded them from the Triton surface -- but #442's served
      census shows they route the Triton kernel at verify);
  (3) the byte-exact full-surface retune CEILING
        byte_exact_full_surface_ceiling_tps_delta = (n256*s256 + 7*s512) * tps_per_us
      with the verdict: < +2 TPS  -> STRICT-NULL (reconciliation closes clean);
                         >= +2 TPS (CI-clean) -> REOPENS strict-supply (LOUD; bring the
                         exact byte-exact kernel change, do NOT deploy).

byte-exact == num_stages 3->2 ONLY (maxdiff == 0.0 exactly); bm4/TILE/num_warps are
greedy-unsafe (#442 FLAG-1) and excluded. The ceiling is T_verify-INDEPENDENT (it is a
sum of per-layer us savings times the Amdahl slope), so it does not inherit the verify
GEMM-body-count uncertainty.

#447 anchors (advisor branch approval-gated-8gpu-20260613, run crrq2e1y):
  t_verify_us = 5911.459832 ; tri3d_h512_each = 10.758827 ; tri3d_h512_x7 = 75.311787
  -> #447 Triton verify surface = 75.311787 / 5911.459832 = 1.2740% (head-512 ONLY)
  fa2_h256_each = 21.814613 (the head-256 sliding layers #447 attributed to FA2)
materiality bar = +2 TPS ; sigma_hw ~ 4.8 TPS ; PPL gate <= 2.42 (byte-exact -> 2.3772).

Analysis ONLY. NOT an HF Job, NOT a submission, NOT a launch. official_tps=0, no
served-file change. group=equivalence-escalation-anchors.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
MICROBENCH_JSON = HERE / "microbench_results.json"
CENSUS_JSON = HERE / "census_out" / "census_results.json"

# ---------------------------------------------------------------------------
# #447 frozen anchors (verify_wall_tile_scan_results.json, run crrq2e1y).
# ---------------------------------------------------------------------------
T_VERIFY_US_447 = 5911.459832
TRI3D_H512_EACH_447 = 10.758827
TRI3D_H512_X7_447 = 75.311787
FA2_H256_EACH_447 = 21.814613
FRAC_447_HEAD512_ONLY = TRI3D_H512_X7_447 / T_VERIFY_US_447   # 0.012740 = 1.2740%
TPS_PER_US_447 = 0.056663

N_GLOBAL_LAYERS = 7      # head-512 global, ALWAYS Triton at verify
N_SLIDING_LAYERS = 30    # head-256 sliding (split Triton/FA2 by FA_SLIDING)
N_TOTAL_LAYERS = 37

MATERIALITY_TPS = 2.0    # advisor bar: < +2 -> STRICT-NULL ; >= +2 (CI-clean) -> REOPEN
PPL_ANCHOR = 2.3772      # byte-exact -> served anchor PPL (gate <= 2.42)


def _log(msg: str) -> None:
    print(f"[reconcile] {msg}", file=sys.stderr, flush=True)


def _load(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(p).read_text())
    except Exception as exc:  # noqa: BLE001
        _log(f"could not load {p}: {exc!r}")
        return None


def _mb_savings(mb: dict) -> dict[str, Any]:
    """Pull the per-head byte-exact savings + deployed us + Amdahl slope from the
    microbench artifact."""
    heads = mb.get("heads", {}) or {}
    h512 = heads.get("head512_global", {}) or {}
    h256 = heads.get("head256_sliding", {}) or {}
    amd = mb.get("amdahl", {}) or {}
    return {
        "s512": float(h512.get("byte_exact_saving_us_per_layer")),
        "s256": float(h256.get("byte_exact_saving_us_per_layer")),
        "deployed_us_512": float(h512.get("deployed_us")),
        "deployed_us_256": float(h256.get("deployed_us")),
        "best_num_stages_512": h512.get("best_num_stages"),
        "best_num_stages_256": h256.get("best_num_stages"),
        "speedup_pct_512": h512.get("byte_exact_speedup_pct"),
        "speedup_pct_256": h256.get("byte_exact_speedup_pct"),
        "tps_per_us": float(amd.get("tps_per_us")),
        "cycle_us": float(amd.get("cycle_us")),
        "realized_tps_k7": float(amd.get("realized_tps_k7")),
        "control_reproduces_447": bool(
            (mb.get("control_vs_447", {}) or {}).get("h512_deployed_within_10pct_of_447")
            and (mb.get("control_vs_447", {}) or {}).get("h512_saving_within_0p3us_of_447")),
    }


def compose_ceiling(s256: float, s512: float, n256: int, tps_per_us: float,
                    n512: int = N_GLOBAL_LAYERS) -> dict[str, Any]:
    total_saving_us = n256 * s256 + n512 * s512
    tps_delta = total_saving_us * tps_per_us
    h512_only_us = n512 * s512
    h512_only_tps = h512_only_us * tps_per_us
    return {
        "n256_triton_layers": n256, "n512_triton_layers": n512,
        "saving_us_256": s256, "saving_us_512": s512,
        "total_byte_exact_saving_us": total_saving_us,
        "byte_exact_full_surface_ceiling_tps_delta": tps_delta,
        "head512_only_ceiling_us": h512_only_us,
        "head512_only_ceiling_tps_delta": h512_only_tps,
        "full_surface_minus_head512_only_tps": tps_delta - h512_only_tps,
        "tps_per_us": tps_per_us,
    }


def reconcile_fraction(n256: int, deployed_us_512: float, deployed_us_256: float) -> dict[str, Any]:
    """The TRUE Triton verify-attn fraction (head-512 global Triton + n256 head-256 sliding
    Triton) vs #447's head-512-only 1.2740%. Per-call us are the measured Triton deployed
    costs (the microbench reproduces #447's tri3d_h512_each), anchored to #447's T_verify so
    the comparison is apples-to-apples on the SAME denominator."""
    triton_h512_total = N_GLOBAL_LAYERS * deployed_us_512
    triton_h256_total = n256 * deployed_us_256
    triton_total = triton_h512_total + triton_h256_total
    corrected_frac = triton_total / T_VERIFY_US_447
    # what #447 had instead for those n256 layers (counted as FA2, excluded from Triton):
    fa2_misattributed_us = n256 * FA2_H256_EACH_447
    return {
        "T_verify_us_anchor_447": T_VERIFY_US_447,
        "triton_verify_attn_total_us": triton_total,
        "triton_h512_global_total_us": triton_h512_total,
        "triton_h256_sliding_total_us": triton_h256_total,
        "triton_h512_each_us": deployed_us_512,
        "triton_h256_each_us": deployed_us_256,
        "triton_verify_attn_frac_of_verify": corrected_frac,
        "frac_447_head512_only": FRAC_447_HEAD512_ONLY,
        "undercount_pp": (corrected_frac - FRAC_447_HEAD512_ONLY) * 100.0,
        "surface_ratio_vs_447": (corrected_frac / FRAC_447_HEAD512_ONLY
                                 if FRAC_447_HEAD512_ONLY > 0 else None),
        "triton_surface_larger_than_447": bool(corrected_frac > FRAC_447_HEAD512_ONLY),
        "n447_undercounted_head256_layers": n256,
        "h256_was_misattributed_to_fa2_us": fa2_misattributed_us,
        "fa2_h256_each_447": FA2_H256_EACH_447,
    }


def self_test(mb_s: dict, n256: int) -> dict[str, Any]:
    res: dict[str, bool] = {}

    def ck(name, cond):
        res[name] = bool(cond)
        print(f"        {'ok ' if cond else 'XX '} {name}", flush=True)

    # #447 anchor arithmetic: 75.311787 / 5911.459832 == 1.2740%.
    ck("a_447_anchor_127pct", abs(FRAC_447_HEAD512_ONLY - 0.012740) < 1e-5)
    # head-512-only ceiling reproduces #447's banked +0.2613 TPS (saving 0.659, slope 0.0567).
    c447 = compose_ceiling(0.0, 0.658774, n256=0, tps_per_us=TPS_PER_US_447)
    ck("b_head512_only_447_0p26",
       abs(c447["byte_exact_full_surface_ceiling_tps_delta"] - 0.2613) < 0.01)
    # ceiling is monotone increasing in n256.
    cps = mb_s["tps_per_us"]
    c0 = compose_ceiling(mb_s["s256"], mb_s["s512"], 0, cps)
    cN = compose_ceiling(mb_s["s256"], mb_s["s512"], n256, cps)
    cmax = compose_ceiling(mb_s["s256"], mb_s["s512"], N_SLIDING_LAYERS, cps)
    ck("c_monotone_in_n256",
       cmax["byte_exact_full_surface_ceiling_tps_delta"]
       >= cN["byte_exact_full_surface_ceiling_tps_delta"]
       >= c0["byte_exact_full_surface_ceiling_tps_delta"])
    # the PHYSICAL maximum (all 30 sliding layers route Triton) still caps under +2 TPS:
    # this makes the STRICT-NULL verdict robust to the exact census n256.
    ck("d_physical_max_under_2tps",
       cmax["byte_exact_full_surface_ceiling_tps_delta"] < MATERIALITY_TPS)
    # reopening would need ~35us total byte-exact saving (2.0 / slope).
    us_needed = MATERIALITY_TPS / cps
    ck("e_reopen_needs_35us", 34.0 < us_needed < 36.5)
    # microbench savings are byte-exact via num_stages==2 on BOTH heads.
    ck("f_both_heads_s2",
       mb_s["best_num_stages_512"] == 2 and mb_s["best_num_stages_256"] == 2)
    # head-512 control reproduced #447 (harness validated).
    ck("g_control_reproduces_447", mb_s["control_reproduces_447"])
    # savings positive + finite.
    ck("h_savings_pos_finite",
       mb_s["s256"] > 0 and mb_s["s512"] > 0
       and all(math.isfinite(x) for x in (mb_s["s256"], mb_s["s512"], cps, us_needed)))
    npass = sum(1 for v in res.values() if v)
    return {"passes": npass == len(res), "n_pass": npass, "n_total": len(res), "checks": res}


def log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        _log(f"wandb_logging import failed ({exc}); skipping")
        return
    rec = result["reconciliation"]
    cei = result["ceiling"]
    st = result["self_test"]
    run = wandb_logging.init_wandb_run(
        job_type="verify-surface-reconcile", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["pr459", "equivalence-escalation", "triton-verify-surface",
              "byte-exact", "reconcile-447"],
        config={
            "n256_triton": result["n256"], "n512_triton": N_GLOBAL_LAYERS,
            "n_total_layers": N_TOTAL_LAYERS,
            "s256_byte_exact_us": result["microbench"]["s256"],
            "s512_byte_exact_us": result["microbench"]["s512"],
            "tps_per_us": result["microbench"]["tps_per_us"],
            "frac_447_head512_only": FRAC_447_HEAD512_ONLY,
            "materiality_tps": MATERIALITY_TPS,
            "byte_exact_axis": "num_stages_3to2_only",
            "analysis_only": True, "no_served_file_change": True, "official_tps": 0,
        },
    )
    if run is None:
        _log("wandb disabled (no API key); skipping")
        return
    try:
        flat = {
            # required #459 fields:
            "triton_verify_attn_frac_of_verify": rec["triton_verify_attn_frac_of_verify"],
            "triton_surface_larger_than_447": float(bool(rec["triton_surface_larger_than_447"])),
            "head256_sliding_routes_triton": float(bool(result["head256_sliding_routes_triton"])),
            "byte_exact_full_surface_ceiling_tps_delta":
                cei["byte_exact_full_surface_ceiling_tps_delta"],
            "reopens_strict_supply": float(bool(result["reopens_strict_supply"])),
            "analysis_only": 1.0, "no_served_file_change": 1.0, "official_tps": 0.0,
            "ppl": PPL_ANCHOR,
            # supporting:
            "n256_triton": float(result["n256"]),
            "frac_447_head512_only": FRAC_447_HEAD512_ONLY,
            "undercount_pp": rec["undercount_pp"],
            "surface_ratio_vs_447": rec["surface_ratio_vs_447"] or 0.0,
            "triton_verify_attn_total_us": rec["triton_verify_attn_total_us"],
            "head512_only_ceiling_tps_delta": cei["head512_only_ceiling_tps_delta"],
            "total_byte_exact_saving_us": cei["total_byte_exact_saving_us"],
            "s256_byte_exact_us": result["microbench"]["s256"],
            "s512_byte_exact_us": result["microbench"]["s512"],
            "tps_per_us": result["microbench"]["tps_per_us"],
            "reconcile_self_test_passes": float(bool(st["passes"])),
        }
        for k, v in st["checks"].items():
            flat[f"selftest/{k}"] = float(bool(v))
        run.summary["verdict"] = result["verdict"]
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="verify_surface_reconcile",
            artifact_type="equivalence-anchor", data=result)
        result["wandb_run_id"] = getattr(run, "id", None)
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN wandb logging error: {exc}")
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--microbench-json", type=Path, default=MICROBENCH_JSON)
    ap.add_argument("--census-json", type=Path, default=CENSUS_JSON)
    ap.add_argument("--n256", type=int, default=None,
                    help="override head-256 Triton layer count (else read from census)")
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", default="wirbel/verify-surface-reconcile")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    mb = _load(args.microbench_json)
    if mb is None:
        raise SystemExit(f"microbench artifact required: {args.microbench_json}")
    mb_s = _mb_savings(mb)

    census = _load(args.census_json)
    cder = (census or {}).get("derived", {}) or {}
    if args.n256 is not None:
        n256 = args.n256
        n256_source = "cli_override"
        head256_routes = n256 > 0
    elif cder.get("n256_triton_round") is not None:
        n256 = int(cder["n256_triton_round"])
        n256_source = "census"
        head256_routes = bool(cder.get("head256_sliding_routes_triton"))
    else:
        n256 = None
        n256_source = "missing"
        head256_routes = None

    if n256 is None:
        _log("no n256 (census missing and no --n256). Running self-test only with n256=14 stub.")
        st = self_test(mb_s, 14)
        return 0 if st["passes"] else 1

    rec = reconcile_fraction(n256, mb_s["deployed_us_512"], mb_s["deployed_us_256"])
    cei = compose_ceiling(mb_s["s256"], mb_s["s512"], n256, mb_s["tps_per_us"])
    ceiling_tps = cei["byte_exact_full_surface_ceiling_tps_delta"]
    reopens = bool(ceiling_tps >= MATERIALITY_TPS)
    verdict = ("REOPENS-STRICT-SUPPLY" if reopens else "STRICT-NULL")
    st = self_test(mb_s, n256)

    result: dict[str, Any] = {
        "experiment": "verify_surface_reconcile", "pr": 459, "student": "wirbel",
        "analysis_only": True, "no_served_file_change": True, "official_tps": 0,
        "ppl": PPL_ANCHOR,
        "n256": n256, "n256_source": n256_source,
        "head256_sliding_routes_triton": head256_routes,
        "geometry": {"n_global_head512": N_GLOBAL_LAYERS,
                     "n_sliding_head256": N_SLIDING_LAYERS, "n_total": N_TOTAL_LAYERS},
        "microbench": mb_s,
        "census_derived": {k: cder.get(k) for k in (
            "verify_M", "n_forwards", "n512_triton_per_forward",
            "n256_triton_per_forward", "n256_fa2_per_forward", "n256_triton_round",
            "n256_fa2_round", "sliding_completeness_sum", "head256_sliding_routes_triton",
            "served_h512_per_call_us", "served_h256_per_call_us")},
        "reconciliation": rec,
        "ceiling": cei,
        "byte_exact_full_surface_ceiling_tps_delta": ceiling_tps,
        "materiality_tps": MATERIALITY_TPS,
        "reopens_strict_supply": reopens,
        "verdict": verdict,
        "byte_exact_kernel_change": "num_stages 3 -> 2 (cp.async pipeline depth) on the "
            "Triton kernel_unified_attention 3D split-KV launch; maxdiff == 0.0 exactly. "
            "NOT bm4/TILE/num_warps (greedy-unsafe). DO NOT DEPLOY (analysis-only).",
        "self_test": st,
    }

    # ----- console verdict -----
    print("\n" + "=" * 78, flush=True)
    print("VERIFY-SURFACE RECONCILE (#447 1.27% vs #442 head-256) -- PR #459 wirbel", flush=True)
    print("=" * 78, flush=True)
    print(f"  n256 (head-256 sliding routing Triton) = {n256}  [{n256_source}]   "
          f"head256_routes_triton={head256_routes}", flush=True)
    print(f"  byte-exact savings: s512={mb_s['s512']:.4f}us/layer  s256={mb_s['s256']:.4f}us/layer "
          f"(num_stages 3->2, maxdiff 0.0)   slope={mb_s['tps_per_us']:.6f} TPS/us", flush=True)
    print("  -- fraction reconciliation (anchored to #447 T_verify=5911.46us) --", flush=True)
    print(f"     #447 head-512-only Triton surface = {FRAC_447_HEAD512_ONLY*100:.4f}% "
          f"({TRI3D_H512_X7_447:.2f}us)", flush=True)
    print(f"     TRUE Triton verify surface        = {rec['triton_verify_attn_frac_of_verify']*100:.4f}% "
          f"({rec['triton_verify_attn_total_us']:.2f}us)  "
          f"[+{rec['undercount_pp']:.4f}pp, {rec['surface_ratio_vs_447']:.2f}x]", flush=True)
    print(f"     #447 UNDERCOUNTED: {n256} head-256 layers it timed as FA2 "
          f"(@{FA2_H256_EACH_447:.2f}us) actually route Triton @{mb_s['deployed_us_256']:.2f}us",
          flush=True)
    print("  -- byte-exact full-surface retune CEILING --", flush=True)
    print(f"     total byte-exact saving = {cei['total_byte_exact_saving_us']:.3f}us "
          f"(n256*{mb_s['s256']:.3f} + 7*{mb_s['s512']:.3f})", flush=True)
    print(f"     >>> byte_exact_full_surface_ceiling_tps_delta = {ceiling_tps:+.4f} TPS "
          f"(head-512-only was {cei['head512_only_ceiling_tps_delta']:+.4f})", flush=True)
    print(f"     materiality bar = +{MATERIALITY_TPS} TPS   physical-max(n256=30) ceiling = "
          f"{compose_ceiling(mb_s['s256'], mb_s['s512'], N_SLIDING_LAYERS, mb_s['tps_per_us'])['byte_exact_full_surface_ceiling_tps_delta']:+.4f}",
          flush=True)
    print(f"  >>> VERDICT = {verdict}  (reopens_strict_supply={reopens})", flush=True)
    print(f"  >>> SELF-TEST = {st['passes']} ({st['n_pass']}/{st['n_total']})", flush=True)
    print("=" * 78 + "\n", flush=True)

    log_wandb(args, result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reconcile_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[reconcile] artifacts -> {args.out_dir / 'reconcile_results.json'}", flush=True)

    # SENPAI-RESULT primary/test metric line (printed for the PR comment).
    senpai = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [result.get("wandb_run_id")] if result.get("wandb_run_id") else [],
        "primary_metric": {"name": "byte_exact_full_surface_ceiling_tps_delta",
                           "value": round(ceiling_tps, 4)},
        "test_metric": {"name": "ppl", "value": PPL_ANCHOR},
    }
    print("SENPAI-RESULT: " + json.dumps(senpai), flush=True)

    if args.self_test and not st["passes"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
