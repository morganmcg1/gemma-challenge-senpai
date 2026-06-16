#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Clean same-env A/B: the VLLM_BATCH_INVARIANT=1 global-flag tax on the deployed precache stack.

Advisor ask (PR #473, 2026-06-16 10:14Z interim ruling): after the clause-3d strict pre-check came back
OUT of band (222.32 TPS vs the [452.7, 462.4] band the 457.55 estimate implies), run a clean same-env A/B
to pin the EXACT local delta the global flag costs:

  arm (a) deployed config, NO flag   -> submissions/fa2sw_precache_kenyan, single-stream
  arm (b) strict config, flag ON     -> + VLLM_BATCH_INVARIANT=1, single-stream

Both arms are identical except the single env var, run back-to-back in the same warm serve env (same venv,
same baked weights, same A10G, VLLM_USE_FLASHINFER_SAMPLER=0, CUDA_VISIBLE_DEVICES=0). This isolates the
cost of the GLOBAL batch-invariant flag, which routes ALL matmuls (lm_head, MLP, QKV) through the
deterministic-but-slow matmul_persistent Triton kernel -- not just the attention reduction.

This is a LOCAL exploratory measurement (A10G != official a10g-small). It does NOT launch an HF Job, does
NOT submit, and does NOT edit any served file. It only reads the two local_summary.json artifacts the
local_prevalidate.py runs already wrote, computes the ratio/tax, and logs the anchors to W&B.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEPLOYED_NOFLAG_DIR = REPO_ROOT / "research/local_validation/ab_deployed_noflag_20260616"
STRICT_FLAG_DIR = REPO_ROOT / "research/local_validation/ab_strict_flag_20260616"

# Reconciliation anchors (advisor PR #473 body + 10:14Z ruling).
OFFICIAL_STRICT_TPS_STARK472 = 457.55      # stark #472 (wfggu51k) "honest" in-harness strict re-anchor
OFFICIAL_DEPLOYED_TPS = 481.53             # deployed non-strict (PR #52), identity 0.9966
SIGMA_HW = 4.8153
OFFICIAL_BAND = (round(OFFICIAL_STRICT_TPS_STARK472 - SIGMA_HW, 2),
                 round(OFFICIAL_STRICT_TPS_STARK472 + SIGMA_HW, 2))   # [452.74, 462.37]
UBEL470_OFFICIAL_TPS = 234.47              # ubel #470 (ugqnytji) official global-flag full serve
UBEL470_LOCAL_TPS = 221.16                 # ubel #470 local global-flag full serve
PR122_BLANKET_PIN_TAX_PCT = 51.78          # PR #122 measured blanket-pin cost


def load_arm(d: Path) -> dict[str, Any]:
    s = json.loads((d / "local_summary.json").read_text(encoding="utf-8"))
    return {
        "dir": str(d.relative_to(REPO_ROOT)),
        "tps": float(s["tps"]),
        "ppl": float(s["ppl"]),
        "completed": int(s["completed"]),
        "decode_num_records": int(s.get("decode_num_records", 0)),
        "decode_num_completion_tokens": int(s.get("decode_num_completion_tokens", 0)),
        "decode_duration_s": float(s.get("decode_duration_s", 0.0)),
    }


def build_payload() -> dict[str, Any]:
    a = load_arm(DEPLOYED_NOFLAG_DIR)   # deployed, NO flag
    b = load_arm(STRICT_FLAG_DIR)       # strict, flag ON
    ratio = b["tps"] / a["tps"] if a["tps"] else 0.0
    tax_pct = round((1.0 - ratio) * 100.0, 2)
    # Does the in-harness "strict" 457.55 look like the GLOBAL flag locally, or like deployed-no-flag?
    in_harness_strict_matches_local_deployed = bool(abs(OFFICIAL_STRICT_TPS_STARK472 - a["tps"]) <= 2 * SIGMA_HW)
    in_harness_strict_matches_local_strict = bool(abs(OFFICIAL_STRICT_TPS_STARK472 - b["tps"]) <= 2 * SIGMA_HW)
    # Naive ratio extrapolation of the official deployed draw down by the measured global-flag tax.
    naive_official_strict_extrapolation = round(OFFICIAL_DEPLOYED_TPS * ratio, 2)
    return {
        "arm_a_deployed_noflag": a,
        "arm_b_strict_flag": b,
        "ratio_strict_over_deployed": round(ratio, 4),
        "global_flag_tax_pct": tax_pct,
        "ppl_neutral": bool(abs(a["ppl"] - b["ppl"]) < 0.01),
        "both_complete_128": bool(a["completed"] >= 128 and b["completed"] >= 128),
        "reconciliation": {
            "official_strict_tps_stark472": OFFICIAL_STRICT_TPS_STARK472,
            "official_band": list(OFFICIAL_BAND),
            "local_strict_in_official_band": bool(OFFICIAL_BAND[0] <= b["tps"] <= OFFICIAL_BAND[1]),
            "in_harness_strict_matches_local_deployed_noflag": in_harness_strict_matches_local_deployed,
            "in_harness_strict_matches_local_strict_flag": in_harness_strict_matches_local_strict,
            "naive_official_strict_extrapolation_tps": naive_official_strict_extrapolation,
            "ubel470_official_global_flag_tps": UBEL470_OFFICIAL_TPS,
            "ubel470_local_global_flag_tps": UBEL470_LOCAL_TPS,
            "pr122_blanket_pin_tax_pct": PR122_BLANKET_PIN_TAX_PCT,
            "interpretation": (
                "The global VLLM_BATCH_INVARIANT=1 flag costs ~{tax}% locally (deployed {a:.2f} -> strict "
                "{b:.2f}, ratio {r:.3f}), reproducing PR #122's 51.78% blanket-pin cost and matching ubel "
                "#470 (234.47 official / 221.16 local). The in-harness 'strict' 457.55 (stark #472) is within "
                "2*sigma_hw of the local DEPLOYED-no-flag {a:.2f}, NOT the local global-flag strict {b:.2f} -- "
                "i.e. the 457.55 measurement does NOT carry the global-flag tax, so it was almost certainly a "
                "surgical attention-only strict change, NOT the global env flag. Firing the global-flag GO "
                "config would land the official draw near ~{ext:.0f} TPS (naive extrapolation), far below the "
                "~457.5 the human approved on #474."
            ).format(tax=tax_pct, a=a["tps"], b=b["tps"], r=ratio,
                     ext=naive_official_strict_extrapolation),
        },
        "analysis_only": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }


def maybe_log_wandb(args, payload: dict[str, Any]) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[ab-tax] wandb logging unavailable: {exc}", flush=True)
        return
    a, b, rec = payload["arm_a_deployed_noflag"], payload["arm_b_strict_flag"], payload["reconciliation"]
    run = init_wandb_run(
        job_type="strict-global-flag-ab",
        agent="land",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["strict-global-flag-ab", "equivalence-escalation-anchors", "clause3d-reconcile",
              "global-flag-tax", "blocker2", "analysis-only", "local-exploratory"],
        config={
            "submission": "submissions/fa2sw_precache_kenyan",
            "flag": "VLLM_BATCH_INVARIANT=1",
            "single_stream": True,
            "official_strict_tps_stark472": OFFICIAL_STRICT_TPS_STARK472,
            "official_deployed_tps": OFFICIAL_DEPLOYED_TPS,
            "sigma_hw": SIGMA_HW,
            "ubel470_official_global_flag_tps": UBEL470_OFFICIAL_TPS,
            "pr122_blanket_pin_tax_pct": PR122_BLANKET_PIN_TAX_PCT,
        },
    )
    if run is None:
        print("[ab-tax] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return
    summary = {
        "ab_deployed_noflag_tps": a["tps"],
        "ab_strict_flag_tps": b["tps"],
        "ab_ratio_strict_over_deployed": payload["ratio_strict_over_deployed"],
        "ab_global_flag_tax_pct": payload["global_flag_tax_pct"],
        "ab_deployed_noflag_ppl": a["ppl"],
        "ab_strict_flag_ppl": b["ppl"],
        "ab_deployed_noflag_completed": a["completed"],
        "ab_strict_flag_completed": b["completed"],
        "ab_ppl_neutral": int(bool(payload["ppl_neutral"])),
        "ab_both_complete_128": int(bool(payload["both_complete_128"])),
        "local_strict_in_official_band": int(bool(rec["local_strict_in_official_band"])),
        "in_harness_strict_matches_local_deployed_noflag": int(bool(rec["in_harness_strict_matches_local_deployed_noflag"])),
        "in_harness_strict_matches_local_strict_flag": int(bool(rec["in_harness_strict_matches_local_strict_flag"])),
        "naive_official_strict_extrapolation_tps": rec["naive_official_strict_extrapolation_tps"],
        "analysis_only": 1,
        "no_served_file_change": 1,
        "official_tps": 0,
        "ppl": b["ppl"],
    }
    log_summary(run, summary, step=0)
    try:
        run.summary["interpretation"] = rec["interpretation"]
    except Exception:  # noqa: BLE001
        pass
    log_json_artifact(run, name="strict_global_flag_ab", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[ab-tax] wandb logged {len(summary)} keys; run id {getattr(run, 'id', '?')}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "research/local_validation")
    args = ap.parse_args(argv)
    payload = build_payload()
    out = Path(args.out_dir) / "ab_global_flag_tax_20260616.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    a, b = payload["arm_a_deployed_noflag"], payload["arm_b_strict_flag"]
    print("\n=== Clean same-env A/B: VLLM_BATCH_INVARIANT=1 global-flag tax ===")
    print(f"arm (a) deployed NO flag : tps={a['tps']:.2f}  ppl={a['ppl']:.4f}  completed={a['completed']}")
    print(f"arm (b) strict flag ON   : tps={b['tps']:.2f}  ppl={b['ppl']:.4f}  completed={b['completed']}")
    print(f"ratio strict/deployed    : {payload['ratio_strict_over_deployed']:.4f}  "
          f"(global-flag tax {payload['global_flag_tax_pct']:.2f}%)")
    print(f"reconciliation           : {payload['reconciliation']['interpretation']}")
    print(f"[ab-tax] wrote {out.relative_to(REPO_ROOT)}")
    maybe_log_wandb(args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
