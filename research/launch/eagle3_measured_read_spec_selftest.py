#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 measured-read launch spec — completeness self-test (PR #322, Issue #319 A).

0 GPU, 0 TPS. This does NOT launch anything. It machine-verifies that the launch
spec at ``research/launch/eagle3_measured_read_spec.md`` is complete and internally
consistent — every required section is present and every load-bearing number/flag the
human needs to fire the single ``a10g-small`` measured read is actually in the file.

PRIMARY metric : ``measured_read_spec_complete_self_test_passes`` (bool → 1 if all
                 completeness conditions hold against the real spec file).
TEST   metric  : ``go_threshold_single_stream_tps`` = 500.0 (the >500 GO target; the
                 481.53 frontier is the no-regression floor).

Run:
    python research/launch/eagle3_measured_read_spec_selftest.py --self-test \
      --wandb_group eagle3-measured-read-spec
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "research" / "launch" / "eagle3_measured_read_spec.md"

GO_THRESHOLD_SINGLE_STREAM_TPS = 500.0
FRONTIER_FLOOR_TPS = 481.53
A1_SALVAGE_BAR = 0.7731
ET_HONEST_500_FLOOR = 3.9914
ET_SIZING_TARGET = 6.11
PPL_CAP = 2.42


def _has(text: str, *needles: str) -> bool:
    """All needles present (case-insensitive substring)."""
    low = text.lower()
    return all(n.lower() in low for n in needles)


def evaluate(spec_text: str) -> dict[str, Any]:
    """Machine-check the spec's completeness conditions against the real file."""
    conditions: dict[str, bool] = {}

    # 1. All six required sections present.
    conditions["sec_config"] = _has(spec_text, "## §1", "config")
    conditions["sec_eval_protocol"] = _has(spec_text, "## §2", "eval protocol")
    conditions["sec_cost"] = _has(spec_text, "## §3", "cost")
    conditions["sec_go_nogo"] = _has(spec_text, "## §4", "go/no-go")
    conditions["sec_failure_branches"] = _has(spec_text, "## §5", "failure branches")
    conditions["sec_prelaunch_checklist"] = _has(spec_text, "## §6", "pre-launch checklist")

    # 2. Config: served basis + #317 guard + method/drafter delta + vLLM/hw.
    conditions["config_basis_precache_317guard"] = _has(
        spec_text, "fa2sw_precache_kenyan", "#317", "_guard_included_router"
    )
    conditions["config_method_swap"] = _has(spec_text, '"mtp"', '"eagle3"', "num_speculative_tokens")
    conditions["config_drafter_artifact"] = _has(spec_text, "DRAFTER_BUCKET", "DRAFTER_SHA256", "LOCAL_DRAFTER_DIR")
    conditions["config_aux_layers"] = _has(spec_text, "[2,21,39]")
    conditions["config_vllm_hw"] = _has(spec_text, "0.22.1rc1", "a10g-small", "sm_86")

    # 3. Eval protocol: single-stream, 512, 128, gates.
    conditions["eval_single_stream"] = _has(spec_text, "MAX_NUM_SEQS=1") and _has(spec_text, "MAX_CONCURRENCY")
    conditions["eval_output_len_512"] = _has(spec_text, "512")
    conditions["eval_128_prompts"] = _has(spec_text, "128")
    conditions["eval_ppl_gate"] = _has(spec_text, "2.42")
    conditions["eval_completion_gate"] = _has(spec_text, "128/128") or _has(spec_text, "== 128")

    # 4. Cost: one run, <40min wall, VRAM fit.
    conditions["cost_one_run"] = _has(spec_text, "a10g-small") and _has(spec_text, "one")
    conditions["cost_wall_40min"] = _has(spec_text, "40 min") or _has(spec_text, "2400")
    conditions["cost_vram_fit"] = _has(spec_text, "20.10") and _has(spec_text, "headroom")

    # 5. GO/NO-GO: all numeric bars present.
    conditions["go_a1_bar"] = "0.7731" in spec_text
    conditions["go_et_floor"] = "3.9914" in spec_text
    conditions["go_et_target"] = "6.11" in spec_text
    conditions["go_tps_floor_and_target"] = ("481.53" in spec_text) and ("500" in spec_text)
    conditions["go_ppl_cap"] = "2.42" in spec_text

    # 6. Failure branches: NO-GO (kill) vs Option B (retrain) vs eager-shortfall-not-nogo.
    conditions["fail_nogo_kill"] = _has(spec_text, "NO-GO") and _has(spec_text, "kill")
    conditions["fail_option_b_retrain"] = _has(spec_text, "Option B", "retrain")
    conditions["fail_eager_not_nogo"] = _has(spec_text, "eager") and _has(spec_text, "not a NO-GO") or _has(
        spec_text, "NOT a NO-GO"
    )

    # 7. Pre-launch checklist maps every merged closure + named YELLOW-closer.
    merged = ["#308", "#310", "#315", "#295", "#299", "#317"]
    conditions["checklist_merged_closures"] = all(c in spec_text for c in merged)
    yellow = ["#314", "#316", "#318", "#320", "#321"]
    conditions["checklist_yellow_closers"] = all(c in spec_text for c in yellow)

    # 8. Critical blocker + exact launch command behind DO-NOT-LAUNCH gate.
    conditions["blocker_checkpoint_publish"] = _has(
        spec_text, "gua9x68j", "Eagle3LlamaForCausalLM"
    ) and (_has(spec_text, "publish") or _has(spec_text, "Hub") or _has(spec_text, "bucket"))
    conditions["blocker_vllm_load_smoke"] = _has(spec_text, "smoke") and _has(spec_text, "greedy")
    conditions["launch_cmd_present"] = _has(spec_text, "train.py", "--submission", "--launch")
    conditions["do_not_launch_gate"] = _has(spec_text, "DO NOT") or _has(spec_text, "DO-NOT-LAUNCH") or _has(
        spec_text, "do not run"
    )

    # Numeric anchors are exactly the constants this card commits to.
    conditions["anchor_go_tps_500"] = f"{GO_THRESHOLD_SINGLE_STREAM_TPS:g}" in spec_text or "500.0" in spec_text
    conditions["anchor_frontier_481"] = f"{FRONTIER_FLOOR_TPS}" in spec_text

    self_test_passes = all(bool(v) for v in conditions.values())

    return {
        "measured_read_spec_complete_self_test_passes": bool(self_test_passes),
        "go_threshold_single_stream_tps": GO_THRESHOLD_SINGLE_STREAM_TPS,
        "frontier_floor_tps": FRONTIER_FLOOR_TPS,
        "a1_salvage_bar": A1_SALVAGE_BAR,
        "et_honest_500_floor": ET_HONEST_500_FLOOR,
        "et_sizing_target": ET_SIZING_TARGET,
        "ppl_cap": PPL_CAP,
        "spec_path": str(SPEC_PATH.relative_to(ROOT)),
        "spec_bytes": len(spec_text.encode("utf-8")),
        "n_conditions": len(conditions),
        "n_conditions_pass": sum(1 for v in conditions.values() if v),
        "conditions": conditions,
    }


def log_wandb(syn: dict[str, Any], args: argparse.Namespace) -> None:
    try:
        import wandb as _wb  # noqa: F401

        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub wandb with no .init")
        sys.path.insert(0, str(ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb,
            init_wandb_run,
            log_json_artifact,
            log_summary,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[eagle3-measured-read-spec] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return

    try:
        run = init_wandb_run(
            job_type="analysis",
            agent="ubel",
            name=args.wandb_name or "ubel/eagle3-measured-read-spec",
            notes="0-GPU launch-spec completeness self-test for the EAGLE-3 measured read (Issue #319 A / PR #322).",
            group=args.wandb_group,
            tags=["eagle3", "measured-read", "launch-spec", "0-tps", "issue-319"],
            config={
                "pr": 322,
                "issue": 319,
                "wandb_group": args.wandb_group,
                "spec_path": syn["spec_path"],
            },
        )
    except Exception as exc:  # pragma: no cover
        print(f"[eagle3-measured-read-spec] wandb init failed (analysis unaffected): {exc}", flush=True)
        return

    if run is None:
        print("[eagle3-measured-read-spec] wandb: no run (no API key / disabled) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "measured_read_spec_complete_self_test_passes": int(syn["measured_read_spec_complete_self_test_passes"]),
        "go_threshold_single_stream_tps": syn["go_threshold_single_stream_tps"],
        "frontier_floor_tps": syn["frontier_floor_tps"],
        "a1_salvage_bar": syn["a1_salvage_bar"],
        "et_honest_500_floor": syn["et_honest_500_floor"],
        "et_sizing_target": syn["et_sizing_target"],
        "ppl_cap": syn["ppl_cap"],
        "spec_bytes": syn["spec_bytes"],
        "n_conditions": syn["n_conditions"],
        "n_conditions_pass": syn["n_conditions_pass"],
        "tps_added_by_this_card": 0,
    }
    summary.update({f"selftest_{k}": int(bool(v)) for k, v in syn["conditions"].items()})

    try:
        log_summary(run, summary, step=0)
        log_json_artifact(
            run,
            name="eagle3_measured_read_spec_selftest",
            artifact_type="analysis",
            data=syn,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[eagle3-measured-read-spec] wandb summary/artifact skipped: {exc}", flush=True)
    finish_wandb(run)
    print(f"[eagle3-measured-read-spec] wandb run logged: {getattr(run, 'id', '?')}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--no-wandb", action="store_true", help="skip W&B logging")
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument(
        "--wandb-group", "--wandb_group", dest="wandb_group", default="eagle3-measured-read-spec"
    )
    args = ap.parse_args(argv)

    if not SPEC_PATH.exists():
        print(f"[eagle3-measured-read-spec] FATAL: spec file missing at {SPEC_PATH}", flush=True)
        return 2

    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    syn = evaluate(spec_text)

    failed = [k for k, v in syn["conditions"].items() if not v]
    print(
        f"[eagle3-measured-read-spec] self_test_passes="
        f"{syn['measured_read_spec_complete_self_test_passes']} "
        f"({syn['n_conditions_pass']}/{syn['n_conditions']} conditions) "
        f"go_threshold_single_stream_tps={syn['go_threshold_single_stream_tps']}",
        flush=True,
    )
    if failed:
        print(f"[eagle3-measured-read-spec] FAILED conditions: {failed}", flush=True)

    if not args.no_wandb:
        log_wandb(syn, args)

    if args.self_test and not syn["measured_read_spec_complete_self_test_passes"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
