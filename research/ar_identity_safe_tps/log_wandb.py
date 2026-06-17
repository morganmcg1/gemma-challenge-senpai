"""Log the PR #601 AR-frame identity-safe TPS deliverable to W&B as ONE rich run.

Decoupled from run_arm.py on purpose: each arm is an expensive ~9-min serve+decode,
so a W&B hiccup must never cost a re-run. This is a pure replay over saved artifacts.

For each research/ar_identity_safe_tps/<arm>/arm_result.json it reads the per-arm
wall_tps / official_proj / serve config, and for each requested (reference, candidate)
pair it recomputes the OFFICIAL byte-identity verdict via the submission's own
check_greedy_identity.compare (zero-tolerance, the wirbel #588 / #319 predicate).

Emits a single grouped run with:
  * per-arm scalars + an "arms" wandb.Table (arm x {knob, wall_tps, official_proj, identity})
  * an "identity" wandb.Table (pair x {verdict, num_identical, divergent_tokens, first_div})
  * headline summary scalars + the qualitative findings as config.

official_tps is kept 0 (projected only — ANALYSIS-ONLY per the PR; the official-proxy
TPS is the local decode wall_tps, proven == sglang output_throughput in PR #72).

Run under the repo .venv (has wandb); import wandb FIRST so the real package wins
over the repo-root ./wandb run dir that would otherwise shadow the import.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import wandb  # noqa: F401  (import first to win over the ./wandb shadow dir)

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts import wandb_logging  # noqa: E402

ARM_DIR = ROOT / "research/ar_identity_safe_tps"
CHECK = ROOT / "submissions/int4_g128_lmhead/check_greedy_identity.py"
BASELINE_OFFICIAL_TPS = 126.378  # operative int4_g128_lmhead rung (PR #601)
TAU = 1.03524  # local wall_tps -> official scalar (#267)


def _load_check_module():
    spec = importlib.util.spec_from_file_location("cgi_mod", CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arm_record(arm: str) -> dict[str, Any] | None:
    f = ARM_DIR / arm / "arm_result.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    flags = r.get("extra_flag") or []
    knob = " ".join(flags) if flags else "(shipped serve)"
    return {
        "arm": arm,
        "knob": knob,
        "wall_tps": r.get("wall_tps"),
        "official_proj_tps": r.get("tau_official_proj"),
        "ready_s": r.get("ready_s"),
        "duration_s": r.get("duration_s"),
        "num_completion_tokens": r.get("num_completion_tokens"),
    }


def _compare(cgi, ref_arm: str, cand_arm: str) -> dict[str, Any] | None:
    ref = ARM_DIR / ref_arm / "decode_outputs.jsonl"
    cand = ARM_DIR / cand_arm / "decode_outputs.jsonl"
    if not ref.exists() or not cand.exists():
        return None
    rep = cgi.compare(cgi.load_decode_outputs(ref), cgi.load_decode_outputs(cand))
    fd = rep.get("first_divergence") or {}
    return {
        "verdict": rep["verdict"],
        "identity_pass": rep["verdict"] == "GREEDY_IDENTICAL",
        "num_prompts_compared": rep["num_prompts_compared"],
        "num_identical": rep["num_identical"],
        "num_divergent": rep["num_divergent"],
        "total_tokens_compared": rep["total_tokens_compared"],
        "total_divergent_tokens": rep["total_divergent_tokens"],
        "first_divergence_key": fd.get("key"),
        "first_divergence_index": fd.get("first_divergence_index"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="lawine/ar-identity-safe-tps")
    ap.add_argument("--group", default="int4g128-ar-identity-safe-tps")
    ap.add_argument("--arms", nargs="*", default=["ref", "ref2", "fulldecode",
                                                  "refS1", "refS1b", "fulldecodeS1"],
                    help="arm dir names to log if present")
    ap.add_argument("--compare", action="append", default=[],
                    help="ref_arm:cand_arm:label triples; repeatable. "
                         "If omitted, a sensible default set is auto-built from present arms.")
    args = ap.parse_args(argv)

    cgi = _load_check_module()

    arms = [a for a in (_arm_record(x) for x in args.arms) if a is not None]
    present = {a["arm"] for a in arms}

    # Default compare set (only pairs whose arms exist get logged).
    pairs: list[tuple[str, str, str]] = []
    if args.compare:
        for spec in args.compare:
            r, c, lbl = spec.split(":", 2)
            pairs.append((r, c, lbl))
    else:
        cand_pairs = [
            ("ref", "ref2", "run_to_run_floor@max_num_seqs=256(shipped)"),
            ("ref", "fulldecode", "cudagraph_FULL_DECODE_ONLY@seqs256"),
            ("refS1", "refS1b", "run_to_run_floor@max_num_seqs=1(predicate)"),
            ("refS1", "fulldecodeS1", "cudagraph_FULL_DECODE_ONLY@seqs1"),
        ]
        pairs = [(r, c, l) for (r, c, l) in cand_pairs if r in present and c in present]

    cmp_rows = []
    for r, c, lbl in pairs:
        rep = _compare(cgi, r, c)
        if rep is None:
            continue
        rep.update({"label": lbl, "reference": r, "candidate": c})
        cmp_rows.append(rep)

    run = wandb_logging.init_wandb_run(
        job_type="analysis",
        agent="lawine",
        name=args.name,
        group=args.group,
        notes="PR #601: cheapest identity-safe AR-frame speedup over int4_g128_lmhead 126.378 "
              "(body/attention/graph axis). ANALYSIS-ONLY, local. official_tps=0 (projected).",
        tags=["pr601", "int4_g128_lmhead", "ar-identity-safe", "analysis-only"],
        config={
            "submission": "int4_g128_lmhead",
            "baseline_official_tps": BASELINE_OFFICIAL_TPS,
            "plus10_target_official_tps": 136.4,
            "tau_local_to_official": TAU,
            "official_tps": 0,  # projected only (PR rule)
            "predicate": "warm steady greedy, MAX_NUM_SEQS=1, spec-OFF, temp=0, "
                         "byte-identical (zero-tol) 128x512 sharegpt (wirbel #588 / #319)",
            "vllm_version": "0.22.1rc1.dev307+g3e8afdf78 (local serve venv)",
            # qualitative findings:
            "flashinfer_reachable": False,  # config.py force-pins TRITON_ATTN (het head dims 256/512)
            "cudagraph_default_full_decode": True,  # shipped default = FULL_AND_PIECEWISE
            "model_weights_gib_on_gpu": 9.85,
            "decode_gpu_bound": True,  # 94-100% util; matmul ~90.4% of kernel time (Phase 1)
        },
    )
    if run is None:
        print("[wandb] disabled / no API key — printing summary only", flush=True)

    # per-arm scalars + table
    arm_cols = ["arm", "knob", "wall_tps", "official_proj_tps", "ready_s",
                "duration_s", "num_completion_tokens"]
    arm_tbl = wandb.Table(columns=arm_cols) if run is not None else None
    for i, a in enumerate(arms):
        beats = (a["wall_tps"] or 0) > BASELINE_OFFICIAL_TPS
        print(f"[arm] {a['arm']:14s} wall_tps={a['wall_tps']:.3f} "
              f"official_proj={a['official_proj_tps']:.3f} knob={a['knob']}", flush=True)
        if run is not None:
            wandb.log({f"arm/{a['arm']}/wall_tps": a["wall_tps"],
                       f"arm/{a['arm']}/official_proj_tps": a["official_proj_tps"],
                       f"arm/{a['arm']}/beats_baseline_tps": int(beats),
                       "global_step": i})
            arm_tbl.add_data(*[a[c] for c in arm_cols])

    # identity compares + table
    cmp_cols = ["label", "reference", "candidate", "verdict", "identity_pass",
                "num_identical", "num_prompts_compared", "total_divergent_tokens",
                "total_tokens_compared", "first_divergence_index"]
    cmp_tbl = wandb.Table(columns=cmp_cols) if run is not None else None
    for j, rep in enumerate(cmp_rows):
        print(f"[identity] {rep['label']:48s} {rep['verdict']:16s} "
              f"{rep['num_identical']}/{rep['num_prompts_compared']} prompts, "
              f"{rep['total_divergent_tokens']}/{rep['total_tokens_compared']} div tok", flush=True)
        if run is not None:
            wandb.log({f"identity/{rep['label']}/num_identical": rep["num_identical"],
                       f"identity/{rep['label']}/divergent_tokens": rep["total_divergent_tokens"],
                       f"identity/{rep['label']}/pass": int(rep["identity_pass"]),
                       "global_step": j})
            cmp_tbl.add_data(*[rep.get(c) for c in cmp_cols])

    if run is not None:
        wandb.log({"arms_table": arm_tbl, "identity_table": cmp_tbl})
        best = max((a["wall_tps"] for a in arms), default=0.0)
        run.summary["best_arm_wall_tps"] = best
        run.summary["baseline_official_tps"] = BASELINE_OFFICIAL_TPS
        run.summary["any_fireable_variant"] = int(any(
            (a["wall_tps"] or 0) > BASELINE_OFFICIAL_TPS for a in arms))
        run.summary["official_tps"] = 0
        wandb_logging.finish_wandb(run)
        print(f"[wandb] logged run: {run.id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
