#!/usr/bin/env python
"""PR #572 — assemble the final base_fullhead + spec-dec ceiling report from the
clean artifacts the crashed probe run already produced. NO re-run (the disk-full
crash that killed the original run already left every load-bearing measurement on
disk; re-serving only risks another disk crash on a 95%-full shared node).

Artifacts consumed (all complete unless noted):
  ARM A  base_fullhead + MTP K=7 spec   decode_specon_full_r1/r2.jsonl (128 each)
                                         server_specon_full.log  (acceptance, steady-gen)
  ARM B  base_fullhead no-spec (M=1 AR)  decode_specoff_full_r1.jsonl (128)
         SAME engine/kernels/quant,      decode_specoff_full_r2.jsonl (97, partial -> self-det only)
         only the drafter removed        server_specoff_full.log (steady-gen)

KEY CORRECTION vs the crashed combine script's framing
-------------------------------------------------------
The #572 card's stated anchor "base_fullhead no-spec = 252.69 TPS (wirbel #553)"
is NOT reproduced on this pod. Verified facts:
  * reference mode (serve.py:disable_speculation_for_reference_mode) clears ONLY
    SPECULATIVE_CONFIG -> SAME engine/kernels/cudagraph/quant as the spec arm
    (both server logs show identical PIECEWISE cudagraph capture, enforce_eager
    off; the ONLY config delta is speculative_config mtp vs None).
  * the stock gemma-4-E4B-it-qat-w4a16-ct keeps `lm_head` in the quant *ignore*
    list -> the full native 262,144-row head is bf16 (~1.34 GB), not int4.
  * measured same-pod no-spec wall_tps == steady "Avg generation throughput"
    == 83.4 TPS (12 ms/token), the memory-bound reality of loading the int4
    body + bf16 262k head once per decode step at M=1.
A bf16 262k head loaded per token makes 252.69 TPS (3.96 ms/token) physically
unreachable as *plain* no-spec, so 252.69 must come from a lighter/quantized head
or a different metric. => 83.44 is the honest same-pod no-spec comparator; 252.69
is reported as the card anchor and flagged as inconsistent. The PRIMARY verdict
(spec TPS vs ship 375.857 / floor 311.25) is robust on either basis: clean miss.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path("/workspace/senpai/target")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_specdec_ceiling as P  # noqa: E402
from scripts.local_validation import serve_profile  # noqa: E402

OUT = P.OUT

# Card anchors / gates.
SHIP_FLIP_TPS = P.SHIP_FLIP_TPS                 # 375.857 official ship (verdict_flip_condition)
CAPSTONE_FLOOR_TPS = P.CAPSTONE_FLOOR_TPS       # 311.25 magically-free-head floor
CARD_NOSPEC_ANCHOR = P.BASE_FULLHEAD_NOSPEC_ANCHOR   # 252.69 wirbel #553 (card)
CANDIDATE_VERIFY_ANCHOR = P.CANDIDATE_VERIFY_ANCHOR   # 291.36 fern #560
TAU_LO = P.TAU_LO                               # 1.03524 local->official (#267)


def _wall(summary: str) -> float:
    return P.decode_wall_tps(json.loads(Path(summary).read_text()))


def main() -> int:
    a_r1 = OUT / "decode_specon_full_r1.jsonl"
    a_r2 = OUT / "decode_specon_full_r2.jsonl"
    b_r1 = OUT / "decode_specoff_full_r1.jsonl"
    b_r2 = OUT / "decode_specoff_full_r2.jsonl"          # partial (97) -> self-det only
    a_log = OUT / "server_specon_full.log"
    b_log = OUT / "server_specoff_full.log"

    # --- TPS (warm-median wall_tps == steady-gen within each arm) ---
    a_runs = [_wall(str(a_r1).replace(".jsonl", ".summary.json")),
              _wall(str(a_r2).replace(".jsonl", ".summary.json"))]
    spec_tps = median(a_runs)
    nospec_tps = _wall(str(b_r1).replace(".jsonl", ".summary.json"))

    # --- acceptance + steady-gen from server logs (the exact SpecDecoding source) ---
    a_spec = serve_profile.parse_spec_log(a_log.read_text())
    b_spec = serve_profile.parse_spec_log(b_log.read_text())   # no-spec: steady-gen only
    acc = a_spec.get("e_accept_exact") or a_spec.get("e_accept_interval_mean")
    spec_steady = a_spec.get("steady_gen_tps_mean")
    nospec_steady = b_spec.get("steady_gen_tps_mean")

    # --- greedy identity (LIGHT; denken #576 owns the rigorous census) ---
    gid = P.greedy_identity(a_r1, b_r1)
    # --- self-determinism (run-to-run reproducibility of each path) ---
    sd_spec = P.self_det(a_r1, a_r2)
    sd_nospec = P.self_det(b_r1, b_r2)   # 97 common records (r2 partial)

    # --- gates ---
    gates = {
        "exceeds_ship": bool(spec_tps >= SHIP_FLIP_TPS),
        "gap_to_ship": SHIP_FLIP_TPS - spec_tps,
        "beats_capstone_floor": bool(spec_tps > CAPSTONE_FLOOR_TPS),
        "ship_flip_tps": SHIP_FLIP_TPS,
        "capstone_floor_tps": CAPSTONE_FLOOR_TPS,
        "exceeds_ship_official_proj": bool(spec_tps * TAU_LO >= SHIP_FLIP_TPS),
        "gap_to_ship_official_proj": SHIP_FLIP_TPS - spec_tps * TAU_LO,
        "ship_flip_local_equiv": SHIP_FLIP_TPS / TAU_LO,
    }

    report = {
        "pr": 572,
        "submission": str(P.SUB.relative_to(ROOT)),
        "substrate": "base_fullhead (stock base-int4 + native bf16 262k head, NO bake, NO prune)",
        "model_snapshot": P.BASE_INT4,
        "speculative_config": (json.loads((P.SUB / "manifest.json").read_text())
                               .get("env", {}).get("SPECULATIVE_CONFIG", "")),
        "spec_drafter": "mtp_k7",
        "num_prompts": 128,
        "output_len": 512,
        "n_warm_decodes_spec": len(a_runs),
        "analysis_only": True,
        "official_tps": 0,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assembled_from_crashed_run": True,
        "no_rerun_reason": "disk-full crash left all load-bearing artifacts complete; "
                           "re-serving risks another crash on a 95%-full shared node.",

        # ---- PRIMARY: base_fullhead + spec TPS ----
        "base_fullhead_spec_tps": spec_tps,
        "base_fullhead_spec_tps_runs": a_runs,
        "base_fullhead_spec_steady_gen_tps": spec_steady,
        "official_projected_tps": spec_tps * TAU_LO,

        # ---- no-spec comparator: measured same-pod (honest) vs card anchor (flagged) ----
        "base_fullhead_nospec_tps_local_measured": nospec_tps,
        "base_fullhead_nospec_steady_gen_tps": nospec_steady,
        "card_nospec_anchor_wirbel553": CARD_NOSPEC_ANCHOR,
        "nospec_anchor_discrepancy": {
            "measured_same_pod": nospec_tps,
            "card_anchor": CARD_NOSPEC_ANCHOR,
            "ratio_anchor_over_measured": CARD_NOSPEC_ANCHOR / nospec_tps,
            "reference_mode_changes_only_speculative_config": True,
            "lm_head_in_quant_ignore_list_bf16": True,
            "explanation": ("reference mode keeps identical engine/kernels/cudagraph/quant "
                            "(only SPECULATIVE_CONFIG cleared); stock lm_head is bf16 (quant "
                            "ignore list); a bf16 262k head (~1.34GB) loaded per M=1 decode "
                            "step makes 252.69 TPS (3.96 ms/tok) unreachable as plain no-spec. "
                            "83.44 (wall==steady) is the memory-bound reality. 252.69 must use "
                            "a lighter/quantized head or a different metric -> the 311.25 floor "
                            "(derived from 252.69) inherits this caveat."),
        },
        "spec_lift_over_measured_nospec": spec_tps - nospec_tps,
        "spec_lift_pct_over_measured_nospec": 100.0 * (spec_tps - nospec_tps) / nospec_tps,
        "candidate_verify_anchor_fern560": CANDIDATE_VERIFY_ANCHOR,

        # ---- acceptance ----
        "acceptance_length": acc,
        "acceptance_length_source": "server_log_exact",
        "acceptance_detail": {
            "e_accept_exact": a_spec.get("e_accept_exact"),
            "e_accept_interval_mean": a_spec.get("e_accept_interval_mean"),
            "draft_acceptance_rate": a_spec.get("draft_acceptance_rate"),
            "num_speculative_tokens": a_spec.get("num_speculative_tokens"),
            "total_accepted_tokens": a_spec.get("total_accepted_tokens"),
            "total_drafted_tokens": a_spec.get("total_drafted_tokens"),
            "intervals": a_spec.get("intervals"),
        },

        # ---- greedy identity (light, deferred) ----
        "greedy_identity": gid,
        "greedy_identity_vs_base_fullhead": gid.get("greedy_identity_vs_base_fullhead", False),
        "greedy_identity_note": ("LIGHT sanity gate; denken #576 owns the rigorous served "
                                 "byte-exact #319 census. NOISE-LIMITED: no-spec self-det only "
                                 f"{sd_nospec['self_det']:.3f} (engine-wide int4 FP/ULP "
                                 "nondeterminism; sampler is torch-native argmax, "
                                 "VLLM_USE_FLASHINFER_SAMPLER=0), so seq-level identity cannot "
                                 "exceed that ceiling and cannot certify/refute #319 here."),

        # ---- self-determinism (run-to-run) ----
        "self_det": sd_spec["self_det"],
        "self_det_spec_detail": sd_spec,
        "self_det_nospec": sd_nospec["self_det"],
        "self_det_nospec_detail": sd_nospec,

        # ---- gates / quality ----
        "gates": gates,
        "exceeds_ship": gates["exceeds_ship"],
        "gap_to_ship": gates["gap_to_ship"],
        "beats_capstone_floor": gates["beats_capstone_floor"],
        "quality_gate_passes_by_construction": True,

        "peak_gpu_mib": 19409,   # from smoke (identical substrate); full-run sampler died on crash
        "nan_clean": all(v == v for v in [spec_tps, nospec_tps, acc, spec_tps * TAU_LO]),
    }

    out_json = OUT / "specdec_ceiling_full.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"[assemble] wrote {out_json}\n")

    g = report["gates"]
    print("========== BASE_FULLHEAD + SPEC-DEC CEILING (final) ==========")
    print(f"base_fullhead_spec_tps (local, warm-median) = {spec_tps:.2f}  runs={[round(x,2) for x in a_runs]}")
    print(f"  steady-gen 'Avg generation throughput'    = {spec_steady}")
    print(f"base_fullhead_nospec_tps (SAME pod/kernels) = {nospec_tps:.2f}  (steady {nospec_steady})")
    print(f"  card anchor (wirbel553)                   = {CARD_NOSPEC_ANCHOR}  <-- NOT reproduced (bf16 head)")
    print(f"spec_lift_over_measured_nospec              = +{report['spec_lift_over_measured_nospec']:.2f} "
          f"(+{report['spec_lift_pct_over_measured_nospec']:.0f}%)")
    print(f"acceptance_length (MTP K=7)                 = {acc}")
    print(f"official_projected_tps (x{TAU_LO})          = {report['official_projected_tps']:.2f}")
    print(f"greedy_identity_vs_base_fullhead (LIGHT)    = {report['greedy_identity_vs_base_fullhead']} "
          f"(seq {gid.get('greedy_identity_seq_frac'):.3f}, onset_frac_med {gid.get('onset_frac_median')})")
    print(f"self_det spec / nospec                      = {sd_spec['self_det']:.3f} / {sd_nospec['self_det']:.3f}")
    print(f"exceeds_ship (>= {SHIP_FLIP_TPS})              = {g['exceeds_ship']}  gap {g['gap_to_ship']:.2f}")
    print(f"beats_capstone_floor (> {CAPSTONE_FLOOR_TPS})     = {g['beats_capstone_floor']}")
    print(f"nan_clean                                   = {report['nan_clean']}")

    _log_wandb(report)
    return 0


def _log_wandb(report: dict) -> None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] skipped ({exc})")
        return
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name="lawine/base-fullhead-specdec-ceiling",
            group="base-fullhead-specdec-ceiling",
            job_type="probe",
            config={k: report[k] for k in (
                "pr", "submission", "substrate", "model_snapshot", "speculative_config",
                "spec_drafter", "num_prompts", "output_len", "analysis_only", "official_tps")},
        )
        gid = report["greedy_identity"]
        disc = report["nospec_anchor_discrepancy"]
        g = report["gates"]
        flat = {
            "base_fullhead_spec_tps": report["base_fullhead_spec_tps"],
            "base_fullhead_spec_steady_gen_tps": report["base_fullhead_spec_steady_gen_tps"],
            "base_fullhead_nospec_tps_local_measured": report["base_fullhead_nospec_tps_local_measured"],
            "base_fullhead_nospec_steady_gen_tps": report["base_fullhead_nospec_steady_gen_tps"],
            "card_nospec_anchor_wirbel553": report["card_nospec_anchor_wirbel553"],
            "nospec_anchor_ratio_over_measured": disc["ratio_anchor_over_measured"],
            "candidate_verify_anchor_fern560": report["candidate_verify_anchor_fern560"],
            "spec_lift_over_measured_nospec": report["spec_lift_over_measured_nospec"],
            "spec_lift_pct_over_measured_nospec": report["spec_lift_pct_over_measured_nospec"],
            "official_projected_tps": report["official_projected_tps"],
            "acceptance_length": report["acceptance_length"],
            "greedy_identity_vs_base_fullhead": report["greedy_identity_vs_base_fullhead"],
            "greedy_identity_seq_frac": gid.get("greedy_identity_seq_frac"),
            "per_step_argmax_identity": gid.get("per_step_argmax_identity"),
            "num_divergent_seqs": gid.get("num_divergent_seqs"),
            "onset_frac_median": gid.get("onset_frac_median"),
            "onset_signature_late_spread": gid.get("onset_signature_late_spread"),
            "self_det": report["self_det"],
            "self_det_nospec": report["self_det_nospec"],
            "exceeds_ship": g["exceeds_ship"],
            "gap_to_ship": g["gap_to_ship"],
            "beats_capstone_floor": g["beats_capstone_floor"],
            "exceeds_ship_official_proj": g["exceeds_ship_official_proj"],
            "quality_gate_passes_by_construction": True,
            "analysis_only": True,
            "official_tps": 0,
            "peak_gpu_mib": report["peak_gpu_mib"],
            "nan_clean": report["nan_clean"],
        }
        for k, v in report["acceptance_detail"].items():
            flat[f"acceptance/{k}"] = v
        run.summary.update(flat)
        rid = run.id
        run.finish()
        print(f"[wandb] logged run {rid}")
        (OUT / "wandb_run_id.txt").write_text(rid)
        report["wandb_run_id"] = rid
        (OUT / "specdec_ceiling_full.json").write_text(json.dumps(report, indent=2, default=str))
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed ({exc})")


if __name__ == "__main__":
    raise SystemExit(main())
