#!/usr/bin/env python3
"""Log the PR #755 served num_splits=1 / strict-identity result to wandb (0-GPU).

Reads the artifacts produced on the vLLM-0.22.0 GPU venv:
  * numsplits_probe/report.json          -- instr 1: served num_splits PROBE
  * strict_census_force_ns1/report.json  -- instr 2/3: force num_splits=1 census + TPS
  * strict_census_eager/report.json       -- localization arm (enforce_eager)
and emits ONE run in group ``pubk4_numsplits1_byteexact`` carrying the deliverable:

  served_numsplits1_strict_identity (primary), tps_pubk4_numsplits1_anchored (test),
  the byte-exactness tax vs #752 (236.02 / 198.00), and the reconciling mechanism
  (num_splits is ALREADY 1 under BI=1 for BOTH decode and verify -> the #747/#752
  hinge is NOT num_splits activation; the force is a measured no-op).

Run under the repo .venv (has wandb): ``.venv/bin/python``. analysis_only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import wandb

HERE = Path(__file__).resolve().parent


def _load(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _probe_buckets(probe: dict[str, Any]) -> dict[str, Any]:
    """Pull the decode (msq=1) and verify (msq=K+1) nseg from the probe spec side."""
    out = {"is_batch_invariant_live": None, "decode_nseg": None, "verify_nseg": None,
           "decode_use3d": None, "verify_use3d": None}
    spec = probe.get("spec", {}) if probe else {}
    out["is_batch_invariant_live"] = spec.get("is_batch_invariant_live")
    k = probe.get("k", 4)
    for key in (spec.get("buckets", {}) or {}):
        # key = msq=<>|local=<>|use3d=<>|nseg=<>
        parts = dict(kv.split("=") for kv in key.split("|"))
        msq = int(parts.get("msq", -1)); nseg = int(parts.get("nseg", -1)); u3 = int(parts.get("use3d", -1))
        if msq == 1:
            out["decode_nseg"], out["decode_use3d"] = nseg, u3
        elif msq == k + 1:
            out["verify_nseg"], out["verify_use3d"] = nseg, u3
    return out


def _arm_flat(tag: str, rep: dict[str, Any], flat: dict[str, Any]) -> None:
    r = rep.get("result", {})
    cfg = rep.get("config", {})
    tax = rep.get("tax_vs_752", {})
    flat[f"{tag}/strict_seq_exact"] = r.get("strict_seq_exact")
    flat[f"{tag}/strict_num_identical"] = r.get("strict_num_identical")
    flat[f"{tag}/strict_num_divergent"] = r.get("strict_num_divergent")
    flat[f"{tag}/strict_num_prompts_compared"] = r.get("strict_num_prompts_compared")
    flat[f"{tag}/strict_token_identity"] = r.get("strict_token_identity")
    flat[f"{tag}/strict_verdict"] = r.get("strict_verdict")
    flat[f"{tag}/wall_tps_local"] = r.get("wall_tps_local")
    flat[f"{tag}/official_equiv_floor"] = r.get("official_equiv_floor")
    flat[f"{tag}/official_equiv_anchored"] = r.get("official_equiv_anchored")
    flat[f"{tag}/ppl"] = r.get("ppl")
    flat[f"{tag}/ppl_ok"] = int(bool(r.get("ppl_ok")))
    flat[f"{tag}/self_consistent_tau03"] = int(bool(r.get("self_consistent_tau03")))
    flat[f"{tag}/tau03_headroom_nat"] = (r.get("tau03") or {}).get("headroom_nat")
    flat[f"{tag}/ET"] = (r.get("acceptance") or {}).get("mean_acceptance_length_ET")
    flat[f"{tag}/accept_rate"] = (r.get("acceptance") or {}).get("avg_draft_acceptance_rate")
    flat[f"{tag}/complete_128_128"] = int(bool(r.get("complete_128_128")))
    flat[f"{tag}/onset_median"] = (r.get("onset") or {}).get("onset_median")
    flat[f"{tag}/peak_vram_gb"] = r.get("peak_vram_gb")
    flat[f"{tag}/enforce_eager"] = int(bool(cfg.get("enforce_eager")))
    flat[f"{tag}/force_numsplits1"] = int(bool(cfg.get("force_numsplits1")))
    flat[f"{tag}/d_anchored_vs_752"] = tax.get("d_anchored")
    flat[f"{tag}/pct_anchored_vs_752"] = tax.get("pct_anchored")
    flat[f"{tag}/d_strict_seq_exact_vs_752"] = tax.get("d_strict_seq_exact")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    ap.add_argument("--probe", type=Path, default=HERE / "runs" / "numsplits_probe" / "report.json")
    ap.add_argument("--project", default="gemma-challenge-senpai")
    ap.add_argument("--entity", default="wandb-applied-ai-team")
    ap.add_argument("--group", default="pubk4_numsplits1_byteexact")
    ap.add_argument("--name", default="lawine/pubk4-numsplits1-byteexact")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    probe = _load(args.probe) or {}
    force = _load(args.runs_dir / "strict_census_force_ns1" / "report.json")
    eager = _load(args.runs_dir / "strict_census_eager" / "report.json")
    if force is None:
        raise SystemExit(f"missing force_ns1 report under {args.runs_dir}")

    pb = _probe_buckets(probe)
    fr = force.get("result", {})
    base752 = force.get("baseline_752", {})

    # ---- reconciling verdict --------------------------------------------------
    force_noop = (pb.get("decode_nseg") == 1 and pb.get("verify_nseg") == 1
                  and bool(pb.get("is_batch_invariant_live")))
    eager_seq = (eager or {}).get("result", {}).get("strict_seq_exact")
    eager_literal_strict = (eager_seq is not None and eager_seq >= 0.999)
    if eager is None:
        localization = "eager_arm_pending"
    elif eager_literal_strict:
        localization = "cudagraph_batch_keyed_capture_byte_exact_under_enforce_eager"
    else:
        # residual survives ALL of: num_splits force, BI=1 (aten-matmul + single-split
        # attention), AND enforce_eager (no cudagraphs). By elimination the only
        # remaining M-dependent kernel in the served int4 stack is the un-BI-patched
        # int4 Marlin quantized matmul (M=1 GEMV vs M=K+1 verify-GEMM reduction order).
        localization = "residual_int4_marlin_quant_matmul_M_dep_survives_split_BI_and_enforce_eager"

    verdict = {
        "pr": 755,
        # mechanism (instr 1)
        "num_splits_hypothesis_refuted": int(force_noop),
        "served_decode_nseg": pb.get("decode_nseg"),
        "served_verify_nseg": pb.get("verify_nseg"),
        "is_batch_invariant_live": int(bool(pb.get("is_batch_invariant_live"))),
        "force_is_measured_noop": int(force_noop),
        # deliverable (instr 2/3)
        "served_numsplits1_strict_identity": fr.get("strict_seq_exact"),
        "served_numsplits1_num_identical": fr.get("strict_num_identical"),
        "tps_pubk4_numsplits1_anchored": fr.get("official_equiv_anchored"),
        "tps_pubk4_numsplits1_wall": fr.get("wall_tps_local"),
        "byteexact_tax_anchored_vs_752": (force.get("tax_vs_752") or {}).get("d_anchored"),
        "byteexact_tax_pct_vs_752": (force.get("tax_vs_752") or {}).get("pct_anchored"),
        "ppl": fr.get("ppl"),
        "self_consistent_tau03": int(bool(fr.get("self_consistent_tau03"))),
        # localization
        "eager_strict_seq_exact": eager_seq,
        "literal_strict_achievable": int(bool(eager_literal_strict)),
        "localization": localization,
        "can_make_pubk4_literally_strict_via_numsplits1": 0,  # refuted: already 1
    }

    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group,
        name=args.name, id=args.run_id, resume=("allow" if args.run_id else None),
        config={
            "pr": 755, "phase": "served_numsplits1_strict_identity",
            "analysis_only": True, "official_tps": 0,
            "vllm_version": force.get("vllm_version"),
            "model_dir": (force.get("config") or {}).get("model_dir"),
            "drafter": (force.get("config") or {}).get("drafter"),
            "k": force.get("k"),
            "batch_invariant": 1,
            "baseline_752_anchored": base752.get("anchored"),
            "baseline_752_wall": base752.get("wall_tps"),
            "baseline_752_strict_seq_exact": base752.get("strict_seq_exact"),
            "anchor_bar": 126.378,
        },
        tags=["pr755", "specdec", "publishable-drafter", "num_splits", "byte-exact",
              "self-consistency", "served", "G1-immune"],
    )

    flat: dict[str, Any] = {}
    for k, v in pb.items():
        flat[f"mechanism/{k}"] = v if not isinstance(v, bool) else int(v)
    _arm_flat("force_ns1", force, flat)
    if eager is not None:
        _arm_flat("eager", eager, flat)
    for k, v in verdict.items():
        flat[f"verdict/{k}"] = int(v) if isinstance(v, bool) else v
    run.summary.update({k: v for k, v in flat.items() if v is not None})

    # full reports as artifacts
    art = wandb.Artifact("pr755_numsplits1_reports", type="strict-identity")
    for nm, obj in (("probe", probe), ("force_ns1", force), ("eager", eager)):
        if obj is not None:
            with art.new_file(f"{nm}.json", mode="w") as fh:
                json.dump(obj, fh, indent=2, default=str)
    run.log_artifact(art)

    print(f"[wandb] run {run.id} group={args.group}")
    print(f"[wandb] MECHANISM: is_bi_live={pb.get('is_batch_invariant_live')} "
          f"decode_nseg={pb.get('decode_nseg')} verify_nseg={pb.get('verify_nseg')} "
          f"-> num_splits hypothesis refuted={force_noop}")
    print(f"[wandb] DELIVERABLE: served_numsplits1_strict_identity={fr.get('strict_seq_exact')} "
          f"({fr.get('strict_num_identical')}/{fr.get('strict_num_prompts_compared')}) "
          f"anchored={fr.get('official_equiv_anchored')} ppl={fr.get('ppl')} "
          f"self_consistent={fr.get('self_consistent_tau03')}")
    print(f"[wandb] LOCALIZATION: eager_seq_exact={eager_seq} -> {localization}")
    run.finish()
    print(f"RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
