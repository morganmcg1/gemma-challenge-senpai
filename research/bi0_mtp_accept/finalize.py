"""Finalize PR #792 sweep after the driver died mid-post-processing.

The sweep (research.bi0_mtp_accept.sweep 32 64 128) completed all THREE
measurement points on disk — decode_ctk{32,64,128}.summary.json + .jsonl,
ppl_ctk{32,64,128}.summary.json, server_ctk{32,64,128}.log — but the driver
process exited before it appended ctk128 to sweep_partial.json, scored greedy
identity, logged wandb, or wrote sweep_report.json. This re-derives the final
artifacts from the on-disk measurements WITHOUT re-running any serve/decode.

ctk32 and ctk64 records are taken verbatim from sweep_partial.json (clean: the
driver snapshotted their decode-phase server log BEFORE PPL ran). ctk128 is
rebuilt from its summaries + server log:
  * wall_tps, duration_s, completed_128  <- decode_ctk128.summary.json (PURE
    decode; PPL never touches it).
  * E_accept / accept_rate / accepted / drafted <- parse_spec_log on the full
    server_ctk128.log. VERIFIED pollution-immune: PPL is teacher-forced
    (prompt_logprobs), runs ZERO drafts, and emits NO SpecDecoding
    Accepted:/Drafted: lines (ctk128 has 31 such intervals, same 30-32 range as
    the clean ctk32/ctk64 logs).
  * steady_gen_tps_mean (SECONDARY) <- mean of the server log's "Avg generation
    throughput" lines with the PPL tail dropped (values < 1 tok/s; a steady
    spec-decode decode never logs sub-1 tok/s, so this filter is a no-op on a
    clean decode log and removes only the idle/PPL scoring lines).
  * ppl <- ppl_ctk128.summary.json.

Run under the repo .venv python (has wandb; serve venv does not, local ./wandb
shadows the import):
    cd target && .venv/bin/python -m research.bi0_mtp_accept.finalize
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import serve_profile  # noqa: E402
from research.bi0_mtp_accept import sweep  # noqa: E402

OUT_DIR = sweep.OUT_DIR


def steady_gen_tps_decode_only(log_text: str) -> float | None:
    """Mean decode-phase 'Avg generation throughput'; drop the PPL/idle tail."""
    import re

    vals = [float(x) for x in re.findall(r"Avg generation throughput:\s*([\d.]+)", log_text)]
    decode_vals = [v for v in vals if v >= 1.0]  # drop idle/PPL sub-1 tok/s lines
    return statistics.fmean(decode_vals) if decode_vals else None


def rebuild_ctk128() -> dict:
    tag = "ctk128"
    top_k = 128
    decode_sum = json.loads((OUT_DIR / f"decode_{tag}.summary.json").read_text())
    ppl_sum = json.loads((OUT_DIR / f"ppl_{tag}.summary.json").read_text())
    log_text = (OUT_DIR / f"server_{tag}.log").read_text()

    spec = serve_profile.parse_spec_log(log_text)
    num_completion = int(decode_sum["num_completion_tokens"])
    duration_s = float(decode_sum["duration_s"])
    num_records = int(decode_sum["num_records"])
    wall_tps = num_completion / duration_s if duration_s else float("nan")
    e_accept = spec.get("e_accept_exact")
    accept_rate = spec.get("draft_acceptance_rate")
    cycle_wall_ms = (
        1000.0 * e_accept / wall_tps if (e_accept and wall_tps == wall_tps) else None
    )
    ppl = ppl_sum.get("ppl") or ppl_sum.get("perplexity")
    rec = {
        "centroid_top_k": top_k,
        "drafter_dir": "/tmp/drafter_ctk128",
        "wall_tps": wall_tps,
        "num_completion_tokens": num_completion,
        "num_records": num_records,
        "completed_128": num_records == 128 and num_completion == 128 * 512,
        "duration_s": duration_s,
        "e_accept": e_accept,
        "accept_rate": accept_rate,
        "num_speculative_tokens": spec.get("num_speculative_tokens"),
        "total_accepted_tokens": spec.get("total_accepted_tokens"),
        "total_drafted_tokens": spec.get("total_drafted_tokens"),
        "e_accept_interval_mean": spec.get("e_accept_interval_mean"),
        "cycle_wall_ms": cycle_wall_ms,
        # SECONDARY metric — recomputed decode-only (driver lost the pre-PPL snapshot)
        "steady_gen_tps_mean": steady_gen_tps_decode_only(log_text),
        "steady_gen_tps_note": "recomputed post-hoc from combined log; PPL tail (<1 tok/s) dropped",
        "ppl": ppl,
        "ppl_summary": ppl_sum,
        "decode_out": str(OUT_DIR / f"decode_{tag}.jsonl"),
    }
    return rec


def main() -> int:
    partial = json.loads((OUT_DIR / "sweep_partial.json").read_text())
    native_top_k = partial["native_top_k"]
    by_k = {r["centroid_top_k"]: r for r in partial["records"]}
    if 32 not in by_k or 64 not in by_k:
        raise SystemExit("sweep_partial.json missing ctk32/ctk64 — cannot finalize")

    records = [by_k[32], by_k[64], rebuild_ctk128()]

    identity = sweep.score_identity(records)
    wandb_ids = {}
    for rec in records:
        wandb_ids[str(rec["centroid_top_k"])] = sweep.log_wandb(rec)

    report = {
        "submission": str(sweep.SUBMISSION),
        "native_top_k": native_top_k,
        "control_top_k": sweep.CONTROL_TOP_K,
        "points": [32, 64, 128],
        "wandb_group": sweep.WANDB_GROUP,
        "wandb_run_ids": wandb_ids,
        "identity_vs_control": identity,
        "records": records,
        "finalized_post_hoc": True,
        "note": "driver died before finalize; ctk128 rebuilt from on-disk artifacts",
    }
    (OUT_DIR / "sweep_report.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n========== bi0 centroid_top_k acceptance/TPS sweep (FINALIZED) ==========", flush=True)
    print(f"native centroid_intermediate_top_k = {native_top_k} (control = {sweep.CONTROL_TOP_K})", flush=True)
    print(f"{'top_k':>6} {'wall_tps':>9} {'E_accept':>9} {'accept':>8} "
          f"{'cycle_ms':>9} {'steadyTPS':>10} {'PPL':>7} {'128/128':>8} {'==ctrl':>7}", flush=True)
    for rec in records:
        print(
            f"{rec['centroid_top_k']:>6} "
            f"{rec.get('wall_tps', float('nan')):>9.2f} "
            f"{(rec.get('e_accept') or float('nan')):>9.4f} "
            f"{(rec.get('accept_rate') or float('nan')):>8.4f} "
            f"{(rec.get('cycle_wall_ms') or float('nan')):>9.3f} "
            f"{(rec.get('steady_gen_tps_mean') or float('nan')):>10.2f} "
            f"{(rec.get('ppl') or float('nan')):>7.4f} "
            f"{str(rec.get('completed_128')):>8} "
            f"{str(rec.get('identical_to_control')):>7}",
            flush=True,
        )
    print(f"\nidentity vs control (ctk{sweep.CONTROL_TOP_K}):", flush=True)
    for k, v in identity.items():
        print(f"  ctk{k}: matched {v['matched']}/{v['compared']} "
              f"identical={v['identical_to_control']} n_mismatched={v['n_mismatched']}", flush=True)
    print(f"\nwandb group {sweep.WANDB_GROUP} ids {wandb_ids}", flush=True)
    print(f"report -> {OUT_DIR / 'sweep_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
