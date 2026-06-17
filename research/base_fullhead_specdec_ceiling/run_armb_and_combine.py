#!/usr/bin/env python
"""PR #572 follow-up — run ARM B (no-spec matched control) and combine with the
already-measured ARM A (base_fullhead + spec MTP K=7) into the final #572 report.

ARM A was measured cleanly by ``probe_specdec_ceiling.py`` (decode_specon_full
r1/r2 + server_specon_full.log: wall_tps 252.35/255.62, e_accept_exact 3.844,
steady_gen 243.04 over 71 intervals) but that run crashed on a *disk-full* before
ARM B (the no-spec M=1 AR control) and the final report/W&B. Disk is freed; this
re-runs ONLY ARM B (no wasteful ARM A re-run) and assembles the combined report,
greedy-identity (ARM A r1 vs ARM B r1), gates, and W&B log.

Run under the repo .venv python from a cwd WITHOUT a ./wandb dir (the target dir
has one, which shadows ``import wandb``)::

    cd research/base_fullhead_specdec_ceiling
    /workspace/senpai/target/.venv/bin/python run_armb_and_combine.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path("/workspace/senpai/target")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# the probe module lives next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_specdec_ceiling as P  # noqa: E402
from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

OUT = P.OUT


def _wall_tps(summary_file: Path) -> float:
    return P.decode_wall_tps(json.loads(summary_file.read_text()))


def reconstruct_arm_a() -> dict:
    """Rebuild ARM A (spec ON) result dict from the artifacts the crashed run left."""
    r1 = OUT / "decode_specon_full_r1.jsonl"
    r2 = OUT / "decode_specon_full_r2.jsonl"
    s1 = OUT / "decode_specon_full_r1.summary.json"
    s2 = OUT / "decode_specon_full_r2.summary.json"
    log = OUT / "server_specon_full.log"
    tps_runs = [_wall_tps(s1), _wall_tps(s2)]
    arm: dict = {
        "tag": "specon_full",
        "spec_on": True,
        "serve_ok": True,
        "reconstructed_from_crashed_run": True,
        "tps_runs": tps_runs,
        "warm_median_tps": median(tps_runs),
        "decode_files": [str(r1), str(r2)],
        "server_log": str(log),
        "spec_log": serve_profile.parse_spec_log(log.read_text()),
        "spec_metrics": {"note": "live /metrics not captured (run crashed before teardown); "
                                 "server-log SpecDecoding counters are the exact source"},
        "peak_gpu_mib": None,  # GPU sampler thread died with the crashed run
    }
    sd = P.self_det(r1, r2)
    arm.update({f"selfdet_{k}": v for k, v in sd.items()})
    return arm


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for note in paths.prepare_local_gpu_env():
        print(f"[combine] {note}", flush=True)

    manifest = harness.load_manifest(P.SUB)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    spec_cfg = (manifest.get("env") or {}).get("SPECULATIVE_CONFIG", "")
    print(f"[combine] SPECULATIVE_CONFIG (ship surgical-357) = {spec_cfg}", flush=True)

    report: dict = {
        "pr": 572,
        "submission": str(P.SUB.relative_to(ROOT)),
        "substrate": "base_fullhead (stock base-int4 + native 262k head, NO bake, NO prune)",
        "model_snapshot": P.BASE_INT4,
        "speculative_config": spec_cfg,
        "spec_drafter": "mtp_k7",
        "num_prompts": paths.NUM_PROMPTS,
        "output_len": paths.OUTPUT_LEN,
        "analysis_only": True,
        "official_tps": 0,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "ARM A reused from crashed-run artifacts; ARM B (no-spec control) re-run here.",
    }

    # ARM A (spec ON) — reconstruct from existing clean artifacts.
    print("\n===== ARM A: base_fullhead + spec (MTP K=7) [reconstructed] =====", flush=True)
    arm_on = reconstruct_arm_a()
    report["arm_spec_on"] = arm_on
    print(f"[combine] ARM A warm_median_tps={arm_on['warm_median_tps']:.3f} "
          f"runs={arm_on['tps_runs']} e_accept={arm_on['spec_log'].get('e_accept_exact')}", flush=True)

    # ARM B (no-spec, M=1 AR matched control) — re-run now.
    print("\n===== ARM B: base_fullhead no-spec (M=1 AR reference) =====", flush=True)
    arm_off = P.serve_arm(server_python, spec_on=False, num_prompts=paths.NUM_PROMPTS,
                          output_len=paths.OUTPUT_LEN, n_decodes=2, port=8000, tag="specoff_full")
    report["arm_spec_off"] = arm_off

    # ---- Acceptance length (primary new measurement) ----
    sl = arm_on.get("spec_log") or {}
    acc_log = sl.get("e_accept_exact")
    acc_log_interval = sl.get("e_accept_interval_mean")
    acceptance_length = acc_log or acc_log_interval
    acc_source = ("server_log_exact" if acc_log else
                  "server_log_interval_mean" if acc_log_interval else "none")
    report["acceptance_length"] = acceptance_length
    report["acceptance_length_source"] = acc_source
    report["acceptance_detail"] = {
        "server_log_e_accept_exact": acc_log,
        "server_log_e_accept_interval_mean": acc_log_interval,
        "server_log_draft_acceptance_rate": sl.get("draft_acceptance_rate"),
        "num_speculative_tokens": sl.get("num_speculative_tokens"),
        "total_accepted_tokens": sl.get("total_accepted_tokens"),
        "total_drafted_tokens": sl.get("total_drafted_tokens"),
        "steady_gen_tps_mean": sl.get("steady_gen_tps_mean"),
        "intervals": sl.get("intervals"),
    }

    # ---- Greedy identity (measured, light; denken #576 owns the rigorous census) ----
    gid: dict = {"error": "missing decode files"}
    if arm_on.get("decode_files") and arm_off.get("decode_files"):
        gid = P.greedy_identity(Path(arm_on["decode_files"][0]), Path(arm_off["decode_files"][0]))
    report["greedy_identity"] = gid
    report["greedy_identity_vs_base_fullhead"] = gid.get("greedy_identity_vs_base_fullhead", False)

    # ---- TPS + gates ----
    tps = arm_on.get("warm_median_tps", float("nan"))
    nospec_tps = arm_off.get("warm_median_tps", float("nan"))
    nospec_steady = ((arm_off.get("spec_log") or {}).get("steady_gen_tps_mean")
                     if arm_off.get("spec_log") else None)
    # ARM B is no-spec, so serve_arm did NOT parse spec_log; parse the server log here
    # for the steady gen-tps cross-check (vLLM logs "Avg generation throughput" w/o spec).
    if nospec_steady is None:
        off_log = Path(arm_off.get("server_log", "")) if arm_off.get("server_log") else None
        if off_log and off_log.exists():
            nospec_steady = serve_profile.parse_spec_log(off_log.read_text()).get("steady_gen_tps_mean")
    spec_steady = sl.get("steady_gen_tps_mean")

    report["base_fullhead_spec_tps"] = tps
    report["base_fullhead_nospec_tps_local"] = nospec_tps
    report["spec_lift_over_nospec_local"] = (tps - nospec_tps
                                             if tps == tps and nospec_tps == nospec_tps else None)
    report["spec_lift_pct_over_nospec_local"] = (
        100.0 * (tps - nospec_tps) / nospec_tps
        if tps == tps and nospec_tps == nospec_tps and nospec_tps else None)
    report["steady_gen_tps"] = {
        "spec_on": spec_steady, "nospec": nospec_steady,
        "spec_lift": (spec_steady - nospec_steady
                      if spec_steady is not None and nospec_steady is not None else None),
    }
    report["official_projected_tps"] = tps * P.TAU_LO if tps == tps else float("nan")

    report["gates"] = {
        "exceeds_ship": bool(tps == tps and tps >= P.SHIP_FLIP_TPS),
        "gap_to_ship": P.SHIP_FLIP_TPS - tps if tps == tps else float("nan"),
        "beats_capstone_floor": bool(tps == tps and tps > P.CAPSTONE_FLOOR_TPS),
        "ship_flip_tps": P.SHIP_FLIP_TPS,
        "capstone_floor_tps": P.CAPSTONE_FLOOR_TPS,
        "exceeds_ship_official_proj": bool(tps == tps and tps * P.TAU_LO >= P.SHIP_FLIP_TPS),
        "gap_to_ship_official_proj": P.SHIP_FLIP_TPS - tps * P.TAU_LO if tps == tps else float("nan"),
        "ship_flip_local_equiv": P.SHIP_FLIP_TPS / P.TAU_LO,
        "unit_note": ("ship 375.857 is OFFICIAL; floor 311.25 and anchors 252.69/291.36 are LOCAL; "
                      "measured TPS is LOCAL. Verdict robust on either basis (clean miss)."),
    }
    report["quality_gate_passes_by_construction"] = True
    report["self_det"] = arm_on.get("selfdet_self_det")
    report["self_det_nospec"] = arm_off.get("selfdet_self_det")
    report["nan_clean"] = all(
        (v == v) for v in [tps, nospec_tps, report["official_projected_tps"]]
        if isinstance(v, float))

    out_json = OUT / "specdec_ceiling_full.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[combine] wrote {out_json}", flush=True)

    g = report["gates"]
    print("\n========== BASE_FULLHEAD + SPEC-DEC CEILING (combined) ==========", flush=True)
    print(f"base_fullhead_spec_tps (local wall)   = {tps:.2f}", flush=True)
    print(f"base_fullhead_nospec_tps (local wall) = {nospec_tps:.2f}  (anchor 252.69)", flush=True)
    print(f"spec_lift_over_nospec (wall)          = {report['spec_lift_over_nospec_local']} "
          f"({report['spec_lift_pct_over_nospec_local']}%)", flush=True)
    print(f"steady_gen_tps spec_on/nospec         = {spec_steady} / {nospec_steady}", flush=True)
    print(f"acceptance_length                     = {acceptance_length} (src {acc_source})", flush=True)
    print(f"official_projected_tps (x{P.TAU_LO})  = {report['official_projected_tps']:.2f}", flush=True)
    print(f"greedy_identity_vs_base_fullhead      = {report['greedy_identity_vs_base_fullhead']} "
          f"(seq {gid.get('greedy_identity_seq_frac')}, per-step {gid.get('per_step_argmax_identity')})", flush=True)
    print(f"self_det spec/nospec                  = {report['self_det']} / {report['self_det_nospec']}", flush=True)
    print(f"exceeds_ship (>= {P.SHIP_FLIP_TPS})       = {g['exceeds_ship']}  gap {g['gap_to_ship']:.2f}", flush=True)
    print(f"beats_capstone_floor (> {P.CAPSTONE_FLOOR_TPS})  = {g['beats_capstone_floor']}", flush=True)

    P._log_wandb(report, "lawine/base-fullhead-specdec-ceiling", "base-fullhead-specdec-ceiling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
