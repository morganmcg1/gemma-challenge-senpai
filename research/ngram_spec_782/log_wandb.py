#!/usr/bin/env python3
"""PR #782 — log each A/B cell to W&B group ``bi0-ngram-spec`` (one run/cell).

Consumes ``analysis.json`` (written by ``analyze.py``), which already carries,
per cell: wall_tps, spec-decode acceptance (rate / E[T] tok-per-step / raw
counters), PPL, and the greedy-identity verdict both vs the committed bi0
plain-AR reference R and vs the MTP control C captured in the same harness.

Each cell becomes one ``ngram-spec-ab`` run so the dashboard shows the drafter
sweep side-by-side: config = drafter (method / num_speculative_tokens /
prompt_lookup_{max,min}); summary = throughput, acceptance, PPL, and the two
greedy-gate verdicts + flip rates. The control's wall_tps is the local A/B
denominator (``tps_vs_control_ratio``); the official bi0 anchor (218.02 TPS /
PPL 2.0058, W&B s63tb03x) is logged as config context, not a local measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary  # noqa: E402

OFFICIAL_BI0_TPS = 218.02
OFFICIAL_BI0_PPL = 2.0058


def _drafter_cfg(extra_env: dict) -> dict:
    env = extra_env or {}
    method = env.get("SPECULATIVE_METHOD") or "mtp"
    cfg = {"drafter_method": method}
    if method == "ngram":
        cfg["num_speculative_tokens"] = int(env.get("NUM_SPECULATIVE_TOKENS", 0) or 0)
        cfg["prompt_lookup_max"] = int(env.get("PROMPT_LOOKUP_MAX", 0) or 0)
        cfg["prompt_lookup_min"] = int(env.get("PROMPT_LOOKUP_MIN", 0) or 0)
    else:
        # MTP control: K comes from the submission manifest (6), not extra_env.
        cfg["num_speculative_tokens"] = int(env.get("NUM_SPECULATIVE_TOKENS", 6) or 6)
        cfg["drafter_model"] = "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
    return cfg


def _gate_metrics(prefix: str, gate: dict | None) -> dict:
    if not isinstance(gate, dict):
        return {}
    out = {f"{prefix}_verdict": gate.get("verdict")}
    nd = gate.get("num_divergent")
    ni = gate.get("num_identical")
    tot_tok = gate.get("total_tokens_compared")
    div_tok = gate.get("total_divergent_tokens")
    if nd is not None:
        out[f"{prefix}_num_divergent"] = nd
        out[f"{prefix}_num_identical"] = ni
    if tot_tok:
        out[f"{prefix}_flip_rate_per_token"] = (div_tok or 0) / tot_tok
        out[f"{prefix}_total_divergent_tokens"] = div_tok
    npc = gate.get("num_prompts_compared")
    if npc:
        out[f"{prefix}_flip_rate_per_prompt"] = (nd or 0) / npc
    onset = gate.get("onset") or {}
    for k in ("onset_min", "onset_median", "onset_max"):
        if k in onset:
            out[f"{prefix}_{k}"] = onset[k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", default=str(REPO / "research/ngram_spec_782/analysis.json"))
    ap.add_argument("--group", default="bi0-ngram-spec")
    ap.add_argument("--agent", default="land")
    ap.add_argument("--dry-run", action="store_true", help="print payloads, do not touch W&B")
    args = ap.parse_args()

    analysis = json.loads(Path(args.analysis).read_text())
    cells = analysis.get("cells", [])
    control_label = analysis.get("control_label")
    control_tps = next(
        (c.get("wall_tps") for c in cells if c.get("label") == control_label), None
    )

    run_ids = []
    for c in cells:
        label = c.get("label")
        # Smoke cells are 4-prompt startup checks (INCOMPARABLE vs the 128-prompt
        # gate, misleading >1 flip rate); they are not A/B data points, so keep
        # them on disk as audit artifacts but out of the bi0-ngram-spec dashboard.
        if str(label).startswith("smoke"):
            print(f"[log_wandb] skip {label} (smoke; not an A/B cell)")
            continue
        dcfg = _drafter_cfg(c.get("extra_env") or {})
        wall_tps = c.get("wall_tps")
        summary = {
            "wall_tps": wall_tps,
            "acceptance_rate": c.get("acceptance_rate"),
            "mean_tokens_per_step_ET": c.get("mean_tokens_per_step_ET"),
            "spec_accepted": c.get("spec_accepted"),
            "spec_draft": c.get("spec_draft"),
            "spec_drafts": c.get("spec_drafts"),
            "ppl": c.get("ppl"),
            "ppl_num_records": c.get("ppl_num_records"),
            "decode_num_records": c.get("decode_num_records"),
            "is_control": label == control_label,
        }
        if control_tps and wall_tps is not None:
            summary["tps_vs_control_ratio"] = wall_tps / control_tps
            summary["tps_minus_control"] = wall_tps - control_tps
        summary.update(_gate_metrics("vs_R", c.get("vs_reference_R")))
        summary.update(_gate_metrics("vs_C", c.get("vs_control_C")))

        submission = (
            "int4_ngram_bi0_surgattn"
            if dcfg["drafter_method"] == "ngram"
            else "int4_mtp_bi0_surgattn"
        )
        config = {
            "submission": submission,
            "cell_label": label,
            "wandb_group": args.group,
            "baseline_official_tps": OFFICIAL_BI0_TPS,
            "baseline_official_ppl": OFFICIAL_BI0_PPL,
            "baseline_wandb_run": "s63tb03x",
            "control_label": control_label,
            **dcfg,
        }

        if args.dry_run:
            print(f"\n=== {label} ===")
            print("config:", json.dumps(config, sort_keys=True))
            print("summary:", json.dumps(summary, sort_keys=True, default=str))
            continue

        run = init_wandb_run(
            job_type="ngram-spec-ab",
            agent=args.agent,
            name=f"{args.agent}/{label}",
            group=args.group,
            tags=[args.group, dcfg["drafter_method"]],
            notes="PR #782 bi0 ngram-vs-MTP drafter A/B (local, temp=0 greedy-verified).",
            config=config,
        )
        if run is None:
            print(f"[log_wandb] {label}: W&B run not created (no creds/disabled)")
            continue
        log_summary(run, summary, step=0)
        run_ids.append(run.id)
        print(f"[log_wandb] {label}: logged run {run.id}  wall_tps={wall_tps}  "
              f"vsR={summary.get('vs_R_verdict')} vsC={summary.get('vs_C_verdict')}")
        finish_wandb(run)

    if run_ids:
        print("\n[log_wandb] run ids:", " ".join(run_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
