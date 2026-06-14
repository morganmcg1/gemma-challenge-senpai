#!/usr/bin/env python3
"""Log the PR #138 block64 K-sweep as a single clean W&B curve (group ``k-sweep-block64-reopt``).

The ``paired_tps_ab.py`` arms already log rich per-run records to W&B, but the PR #138
tracking contract asks for one queryable curve carrying exactly:
``num_speculative_tokens``, ``fused_sparse_argmax_block``, ``wall_tps_median``,
``accept_length_mean``, ``step_time_ms``. This reads the per-arm ``paired_ab.json`` and
logs one step per K (``global_step=K``) so ``wall_tps_median`` vs ``num_speculative_tokens``
renders as the K* curve, plus the anchor (K7-block16) row and the final gate as summary.

Usage:
    .venv/bin/python research/walltps_ab/log_block64_sweep_wandb.py \
        --wandb-name kanna/k-sweep-block64-summary --wandb-group k-sweep-block64-reopt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "research" / "walltps_ab"

K_ARMS = {6: "k6_block64", 7: "k7_block64", 8: "k8_block64", 9: "k9_block64"}
ANCHOR_ARM = "k7_block64"
LAWINE_K7_BLOCK16_REF = 454.338
OP_THRESHOLD_PCT = 0.10


def _load(arm_dir: str) -> dict | None:
    p = SWEEP / arm_dir / "paired_ab.json"
    return json.loads(p.read_text()) if p.exists() else None


def _step_ms(wall_tps: float | None, e_accept: float | None) -> float | None:
    return 1000.0 * e_accept / wall_tps if (wall_tps and e_accept) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", default="kanna/k-sweep-block64-summary")
    ap.add_argument("--wandb-group", default="k-sweep-block64-reopt")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts import wandb_logging

    anchor = _load(ANCHOR_ARM)
    if not anchor:
        raise SystemExit(f"anchor arm not found: {SWEEP / ANCHOR_ARM}")
    base = anchor["arms"]["baseline"]      # K7-block16
    k7b64 = anchor["arms"]["candidate"]    # K7-block64
    anchor_med = base["wall_tps"]["median"]
    anchor_eacc = base["e_accept_exact"]["mean"]
    k7b64_med = k7b64["wall_tps"]["median"]

    # Collect per-K block64 points.
    points = {}
    for K, arm_dir in K_ARMS.items():
        if K == 7:
            med, eacc = k7b64_med, k7b64["e_accept_exact"]["mean"]
        else:
            d = _load(arm_dir)
            if not d:
                continue
            cand = d["arms"]["candidate"]
            med, eacc = cand["wall_tps"]["median"], cand["e_accept_exact"]["mean"]
        points[K] = {"wall_tps_median": med, "accept_length_mean": eacc,
                     "step_time_ms": _step_ms(med, eacc)}

    if args.no_wandb:
        print(json.dumps({"anchor_block16": anchor_med, "points": points}, indent=2))
        return 0

    run = wandb_logging.init_wandb_run(
        job_type="walltps-ksweep-summary", agent="kanna",
        name=args.wandb_name, group=args.wandb_group,
        tags=["k-sweep-block64-reopt", "fa2sw_precache_kenyan", "block64"],
        notes="PR #138 block64 K-sweep curve; one step per K (global_step=K).",
        config={
            "submission": "fa2sw_precache_kenyan",
            "fused_sparse_argmax_block": 64,
            "anchor_k7_block16_wall_tps": anchor_med,
            "anchor_k7_block16_accept_length": anchor_eacc,
            "lawine_k7_block16_ref": LAWINE_K7_BLOCK16_REF,
            "k7_block64_ref_wall_tps": k7b64_med,
            "n_per_arm": anchor["n"],
            "workload": anchor["workload"],
        },
    )
    if run is None:
        print("[k64-wandb] disabled (no API key / WANDB_DISABLED); printed summary only")
        print(json.dumps({"anchor_block16": anchor_med, "points": points}, indent=2))
        return 0

    # Anchor row at K=7, block16 (global_step uses 70+K-ish offset to keep separate? no —
    # keep the curve clean: log block64 points at global_step=K; log the block16 anchor as
    # a distinct series so a single chart shows both).
    for K in sorted(points):
        p = points[K]
        wandb_logging.log_event(run, "k_point", step=K, metrics={
            "num_speculative_tokens": K,
            "fused_sparse_argmax_block": 64,
            "wall_tps_median": p["wall_tps_median"],
            "accept_length_mean": p["accept_length_mean"],
            "step_time_ms": p["step_time_ms"],
            "delta_vs_k7_block16_pct": 100.0 * (p["wall_tps_median"] - anchor_med) / anchor_med,
            "delta_vs_k7_block64_pct": 100.0 * (p["wall_tps_median"] - k7b64_med) / k7b64_med,
            # block16 anchor as a flat reference line on the same chart.
            "k7_block16_anchor_wall_tps": anchor_med,
        })

    done = list(points.items())
    best_k, best_p = max(done, key=lambda kv: kv[1]["wall_tps_median"])
    better = [K for K, p in done if K != 7
              and (p["wall_tps_median"] - k7b64_med) / k7b64_med * 100.0 >= OP_THRESHOLD_PCT]
    k_star = max(better, key=lambda K: points[K]["wall_tps_median"]) if better else 7
    wandb_logging.log_summary(run, {
        "k_star_block64": k_star,
        "k_optimal_wall_tps_block64": best_p["wall_tps_median"],
        "k7_block64_wall_tps": k7b64_med,
        "k7_block16_anchor_wall_tps": anchor_med,
        "block64_step_time_gain_pct_at_k7": 100.0 * (k7b64_med - anchor_med) / anchor_med,
        "k_star_shifted": int(k_star != 7),
    }, step=max(points) + 1)
    wandb_logging.finish_wandb(run)
    print(f"[k64-wandb] logged curve: K*={k_star} best_wall_tps={best_p['wall_tps_median']:.3f} "
          f"run={args.wandb_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
