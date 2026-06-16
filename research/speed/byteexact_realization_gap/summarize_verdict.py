"""PR #523 -- consolidate the three measurement batches into ONE component-attributed
realization-gap verdict + the six KEY OUTPUTS. CPU-only, no serve, no GPU.

Reads the per-batch ``realization_gap_result.json`` (ledger / geometry / levers) and the
``microbench_sweep_summary.json`` that the GPU runs already produced, recomputes the
cross-batch deltas the per-batch ``build_ledger`` could not (each batch only saw its own
arms), and emits ``realization_gap_verdict.json`` + a consolidated W&B summary run.

Run under the repo .venv (has wandb)::

    .venv/bin/python -m research.speed.byteexact_realization_gap.summarize_verdict \
        --wandb-name lawine/realization-gap-verdict --wandb-group byteexact-realization-gap
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "research" / "speed" / "byteexact_realization_gap"

# PR #523 banked anchors (body). Local scale; official = tau_lo~1.0352 * local (lawine #267).
SURGICAL_ANCHOR = 357.6          # PR #488 byte-exact 2D rung (local headline)
BYTEEXACT_ANCHOR_32x256 = 399.97  # PR #496 byte-exact fixed-3D rung @ 32x256 headline
STRICT_FRONTIER_PRED = 457.5     # #466/#474 microbench projection (never served)
PPL_GATE = 2.42
SIGMA_HW = 4.864


def _load(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except OSError:
        return {}


def _tps(batch: dict[str, Any], arm: str) -> float | None:
    rec = (batch.get("arms") or {}).get(arm)
    if not rec:
        return None
    v = rec.get("median_wall_tps")
    return v if isinstance(v, (int, float)) and v == v else None  # drop NaN


def _ppl(batch: dict[str, Any], arm: str) -> float | None:
    rec = (batch.get("arms") or {}).get(arm)
    return rec.get("ppl") if rec else None


def _exec_gpu(batch: dict[str, Any], arm: str) -> float | None:
    rec = (batch.get("arms") or {}).get(arm)
    if not rec:
        return None
    return ((rec.get("mechanism") or {}).get("steptime") or {}).get("exec", {}).get("gpu_mean")


def build_verdict() -> dict[str, Any]:
    ledger_b = _load(OUT_ROOT / "ledger" / "realization_gap_result.json")
    geom_b = _load(OUT_ROOT / "geometry" / "realization_gap_result.json")
    levers_b = _load(OUT_ROOT / "levers" / "realization_gap_result.json")
    micro = _load(OUT_ROOT / "microbench" / "microbench_sweep_summary.json")

    # --- same-session ledger arms (batch A, n=3 back-to-back, shared sigma_hw) ----------
    surgical = _tps(ledger_b, "surgical")          # byte-exact 2D in-order (357 floor)
    byteexact = _tps(ledger_b, "bx_T4_S64")        # byte-exact fixed-3D split-KV (packaged)
    deployed = _tps(ledger_b, "deployed")          # NON-byte-exact adaptive-3D split-KV

    # --- geometry segment sweep (hold coverage=4096 keys; vary parallel segments @L=512) -
    geom = {
        "seg2_T16_S16": _tps(geom_b, "bx_T16_S16"),
        "seg4_T8_S32": _tps(geom_b, "bx_T8_S32"),
        "seg8_T4_S64": byteexact,                  # the packaged config (measured in batch A)
        "seg16_T2_S128": _tps(geom_b, "bx_T2_S128"),
    }
    geom_named = {k: v for k, v in geom.items() if v is not None}
    geom_best_arm = max(geom_named, key=geom_named.get) if geom_named else None

    # --- levers --------------------------------------------------------------------------
    eager_drafter = _tps(levers_b, "bx_eager_drafter")    # ONEGRAPH=0 (drafter eager)
    fisampler = _tps(levers_b, "bx_fisampler")            # NaN -> None (local cuRAND JIT crash)
    cudagraph_drafter_benefit = (
        byteexact - eager_drafter if (byteexact and eager_drafter) else None
    )

    # --- the gap decomposition (same-session basis) --------------------------------------
    # surgical 351.97 --[+geometry, byte-exact]--> byteexact 439.71
    #                  --[+fixed->adaptive, NOT byte-exact]--> deployed 453.93
    #                  --[projection overshoot, never served]--> 457.5
    lever_geometry_byteexact = (byteexact - surgical) if (byteexact and surgical) else None
    tax_fixed_vs_adaptive = (deployed - byteexact) if (deployed and byteexact) else None
    projection_overshoot = (STRICT_FRONTIER_PRED - deployed) if deployed else None

    # PR-anchor-basis gap (the task's literal definition)
    realization_gap_tps = STRICT_FRONTIER_PRED - SURGICAL_ANCHOR  # 457.5 - 357.6

    # fraction of the same-session surgical->457.5 span that is byte-exact-recoverable
    span = (STRICT_FRONTIER_PRED - surgical) if surgical else None
    frac_kernel_overhead = (lever_geometry_byteexact / span) if (lever_geometry_byteexact and span) else None

    all_geom_byteexact = bool(micro.get("all_fixed_geometry_byteexact_0of8"))
    adaptive_contrast_flips = micro.get("adaptive_contrast_straddle_flips")

    # largest realizable lever = the byte-exact split-KV geometry (vs surgical 2D floor)
    largest_lever_name = "attention_splitkv_geometry_2D_inorder_to_fixed3D"
    largest_lever_tps = lever_geometry_byteexact
    largest_lever_is_byteexact = all_geom_byteexact and (byteexact is not None)

    gap_is_kernel_overhead = bool(
        frac_kernel_overhead is not None and frac_kernel_overhead >= 0.5
    )

    verdict = {
        "pr": 523,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "official_tps": 0,
        "workload": {"num_prompts": 128, "output_len": 512, "seed": 1, "metric": "wall_tps (median, same-session)"},
        "sigma_hw": SIGMA_HW,
        "ppl_gate": PPL_GATE,

        # ---- same-session served local wall_tps (the clean controlled comparison) -------
        "served_local_wall_tps": {
            "surgical_2D_byteexact_floor": surgical,
            "byteexact_fixed3D_splitkv": byteexact,
            "deployed_adaptive3D_nonexact": deployed,
            "strict_frontier_457p5_projection": STRICT_FRONTIER_PRED,
        },
        "ppl": {
            "surgical": _ppl(ledger_b, "surgical"),
            "byteexact": _ppl(ledger_b, "bx_T4_S64"),
            "deployed": _ppl(ledger_b, "deployed"),
        },

        # ---- component-attributed ledger of the 357->457.5 gap --------------------------
        "gap_ledger_same_session": {
            "floor_surgical_2D_byteexact": surgical,
            "lever1_splitkv_geometry_BYTEEXACT_tps": lever_geometry_byteexact,
            "after_lever1_byteexact_fixed3D": byteexact,
            "lever2_fixed_to_adaptive_NONEXACT_tps": tax_fixed_vs_adaptive,
            "after_lever2_deployed_adaptive": deployed,
            "residual_projection_overshoot_tps": projection_overshoot,
            "top_457p5_projection_never_served": STRICT_FRONTIER_PRED,
            "byteexact_recoverable_fraction_of_span": frac_kernel_overhead,
        },

        # ---- steptime corroboration (verify-step GPU ms; attention lives here) ----------
        "steptime_exec_gpu_ms": {
            "surgical": _exec_gpu(ledger_b, "surgical"),
            "byteexact": _exec_gpu(ledger_b, "bx_T4_S64"),
            "deployed": _exec_gpu(ledger_b, "deployed"),
        },

        # ---- geometry segment-count sweep (all byte-exact) ------------------------------
        "geometry_segment_sweep_tps": geom_named,
        "geometry_sweep_best_arm": geom_best_arm,
        "geometry_all_configs_byteexact_0of8": all_geom_byteexact,
        "geometry_adaptive_contrast_straddle_flips": adaptive_contrast_flips,

        # ---- other levers ---------------------------------------------------------------
        "cudagraph_drafter_benefit_tps": cudagraph_drafter_benefit,
        "flashinfer_sampler_local": "UNMEASURABLE_local_cuRAND_JIT_crash (works on HF a10g; not dominant)",
        "flashinfer_sampler_tps": fisampler,

        # ---- 457.5 provenance (task step 1) ---------------------------------------------
        "strict_frontier_457p5_provenance": {
            "457p5_is_128x512_measured": False,
            "established_at_workload": "single-shape microbench, KV-len 640 (headline_L), M=8 verify, hd512, "
                                       "n_full_layers=7; per-cycle attention added-us applied to the deployed "
                                       "481.53 decode cycle. NOT a served 128x512 run.",
            "source_artifacts": [
                "research/speed/strict_frontier_realize/strict_frontier_realize.json "
                "(realized_strict_frontier_tps=456.36, strict_2d projection @L640)",
                "research/speed/surgical_attn_realize/PR488_verdict_integrated.json "
                "(modeled_surgical_estimate=457.0 PROJECTION vs surgical_attn_only_tps=357.64 REALIZED; "
                "is_457_a_mirage='partially -- ~22% overshoot ... Realized surgical = 357.6, not 457')",
            ],
            "what_actually_serves": "byte-exact serve at full 128x512 realizes surgical 351.97 / byteexact 439.71 "
                                    "local; the fast NON-byte-exact deployed stack tops 453.93 same-session "
                                    "(465 local / 481.53 official other-session). 457.5 exceeds even the real "
                                    "fast stack -> a projection, not a frontier.",
        },

        # ---- KEY OUTPUTS (PR #523) ------------------------------------------------------
        "KEY_OUTPUTS": {
            "realization_gap_tps": realization_gap_tps,
            "gap_largest_lever_name": largest_lever_name,
            "gap_largest_lever_tps": largest_lever_tps,
            "gap_largest_lever_is_byteexact": bool(largest_lever_is_byteexact),
            "gap_is_kernel_overhead": gap_is_kernel_overhead,
            "457p5_is_128x512_measured": False,
        },

        # ---- honest census caveat (lawine #500/#496) ------------------------------------
        "census_caveat": "The +87.74 lever is byte-exact at the ATTENTION KERNEL (0/8 microbench, all geometry "
                         "configs). End-to-end token census (lawine #500, #461 locus): byteexact 5 ULP-tie flips "
                         "(0 semantic) vs surgical 1 -- driven by the SHARED un-taxed Marlin matmul, NOT the "
                         "attention geometry. So the rung is draw-ready on speed/PPL/completion/self-determinism "
                         "(r1-r2=1.0) but NOT census-tight to surgical's 1-flip bar; matching it needs the matmul "
                         "tax (~135 TPS, #488), defeating the purpose.",
        "wandb_batch_runs": {
            "ledger": ledger_b.get("wandb_run_id"),
            "geometry": geom_b.get("wandb_run_id"),
            "levers": levers_b.get("wandb_run_id"),
        },
    }
    return verdict


def log_wandb(args, verdict: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[verdict] wandb_logging import failed ({exc}); skipping", flush=True)
        return None
    try:
        run = wandb_logging.init_wandb_run(
            job_type="byteexact-realization-gap", agent="lawine",
            name=args.wandb_name or "lawine/realization-gap-verdict",
            group=args.wandb_group,
            tags=["byteexact-realization-gap", "pr523", "analysis-only", "verdict"],
            config={"analysis_only": True, "official_tps": 0, "sigma_hw": SIGMA_HW,
                    "strict_frontier_pred_457p5": STRICT_FRONTIER_PRED,
                    "surgical_anchor": SURGICAL_ANCHOR},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[verdict] wandb init failed ({exc}); skipping", flush=True)
        return None
    if run is None:
        print("[verdict] wandb disabled (no API key); skipping", flush=True)
        return None
    run_id = getattr(run, "id", None)
    try:
        flat: dict[str, Any] = {f"key/{k}": v for k, v in verdict["KEY_OUTPUTS"].items()
                                if isinstance(v, (int, float, bool))}
        flat["key/gap_largest_lever_name"] = verdict["KEY_OUTPUTS"]["gap_largest_lever_name"]
        for k, v in verdict["served_local_wall_tps"].items():
            if isinstance(v, (int, float)):
                flat[f"served/{k}"] = v
        for k, v in verdict["gap_ledger_same_session"].items():
            if isinstance(v, (int, float)):
                flat[f"ledger/{k}"] = v
        for k, v in verdict["geometry_segment_sweep_tps"].items():
            flat[f"geom/{k}"] = v
        if isinstance(verdict.get("cudagraph_drafter_benefit_tps"), (int, float)):
            flat["lever/cudagraph_drafter_benefit_tps"] = verdict["cudagraph_drafter_benefit_tps"]
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="realization_gap_verdict",
            artifact_type="byteexact-realization-gap", data=verdict,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[verdict] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass
    return run_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wandb-name", default="lawine/realization-gap-verdict")
    ap.add_argument("--wandb-group", default="byteexact-realization-gap")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    verdict = build_verdict()
    run_id = log_wandb(args, verdict)
    verdict["wandb_verdict_run_id"] = run_id
    out = OUT_ROOT / "realization_gap_verdict.json"
    out.write_text(json.dumps(verdict, indent=2, default=float))

    k = verdict["KEY_OUTPUTS"]
    print("\n================= PR #523 REALIZATION-GAP VERDICT =================", flush=True)
    print(f"  realization_gap_tps (457.5-357.6)      = {k['realization_gap_tps']:.2f}", flush=True)
    print(f"  gap_largest_lever_name                 = {k['gap_largest_lever_name']}", flush=True)
    lv = k['gap_largest_lever_tps']
    print(f"  gap_largest_lever_tps                  = {lv:.2f}" if isinstance(lv, (int, float)) else f"  gap_largest_lever_tps = {lv}", flush=True)
    print(f"  gap_largest_lever_is_byteexact         = {k['gap_largest_lever_is_byteexact']}", flush=True)
    print(f"  gap_is_kernel_overhead                 = {k['gap_is_kernel_overhead']}", flush=True)
    print(f"  457p5_is_128x512_measured              = {k['457p5_is_128x512_measured']}", flush=True)
    led = verdict["gap_ledger_same_session"]
    print("  --- same-session ledger ---", flush=True)
    print(f"  surgical 2D byte-exact floor           = {led['floor_surgical_2D_byteexact']}", flush=True)
    print(f"  + split-KV geometry (BYTE-EXACT)       = +{led['lever1_splitkv_geometry_BYTEEXACT_tps']:.2f} -> {led['after_lever1_byteexact_fixed3D']}", flush=True)
    print(f"  + fixed->adaptive (NON-byte-exact)     = +{led['lever2_fixed_to_adaptive_NONEXACT_tps']:.2f} -> {led['after_lever2_deployed_adaptive']}", flush=True)
    print(f"  + projection overshoot (never served)  = +{led['residual_projection_overshoot_tps']:.2f} -> {led['top_457p5_projection_never_served']}", flush=True)
    print(f"  byteexact-recoverable fraction of span = {led['byteexact_recoverable_fraction_of_span']:.3f}", flush=True)
    print(f"\n  artifacts -> {out}  wandb_verdict_run_id={run_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
