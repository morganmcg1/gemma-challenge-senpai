"""Controlled A/B for the splitkv-verify patch: serve the SAME submission with
SPLITKV_VERIFY=1 (patched, 3D split-KV verify) vs SPLITKV_VERIFY=0 (baseline, stock
2D verify) under an identical workload on the same GPU, then for each arm:

  * greedy-compare the served decode against the submission's canonical M=1 AR
    reference (the leaderboard validity anchor) -> verdict + per-prompt identical
    count (read on the OVERLAPPING prompts, so a 16-vs-128 count mismatch that the
    official gate labels INCOMPARABLE still yields the real identical/divergent
    split);
  * record vLLM's own whole-run steady decode TPS, E_accept, and verify GPU p50.

This isolates whether the 3D split-KV verify path (a) changes greedy tokens vs the
stock 2D path and (b) moves end-to-end TPS. extra_env wins over the manifest's
SPLITKV_VERIFY=1 (harness LocalServer applies extra_env last), so the baseline arm
genuinely runs the unpatched dispatch.

Local A10G exploratory probe — NOT the official a10g-small TPS.

    /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
        research/profiling/splitkv_verify/ab_verify.py --num-prompts 16 --arms baseline,patched
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, harness, paths, serve_profile  # noqa: E402

SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
SERVER_PYTHON = Path("/tmp/senpai-venvs/5f4c623f772358a2/bin/python")
ARM_ENV = {
    "patched": {"SPLITKV_VERIFY": "1"},
    "baseline": {"SPLITKV_VERIFY": "0"},
}


def _reference_path() -> Path:
    manifest = harness.load_manifest(SUBMISSION)
    ref_id = harness.reference_identity(harness.serve_model_id(manifest, SUBMISSION), SUBMISSION)
    return greedy_gate.reference_for(ref_id)


def _greedy_for(out_dir: Path, label: str, reference: Path) -> dict:
    candidate = out_dir / f"decode_{label}.jsonl"
    report = greedy_gate.compare(reference, candidate)
    onset = greedy_gate.onset_summary(report)
    return {
        "verdict": report.verdict,
        "num_prompts_compared": report.num_prompts_compared,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "total_divergent_tokens": report.total_divergent_tokens,
        "onset_min": onset.get("onset_min"),
        "onset_median": onset.get("onset_median"),
        "onset_max": onset.get("onset_max"),
        # identical fraction on the OVERLAPPING (compared) prompts — the real signal
        # even when the overall verdict is INCOMPARABLE due to a count mismatch.
        "identical_frac_overlap": (report.num_identical / report.num_prompts_compared
                                   if report.num_prompts_compared else None),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--arms", default="baseline,patched",
                    help="comma list from {baseline,patched}; order = serve order")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "research" / "profiling" / "splitkv_verify" / "ab_verify")
    args = ap.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARM_ENV]
    if bad:
        raise SystemExit(f"unknown arms {bad}; choose from {list(ARM_ENV)}")

    for note in paths.prepare_local_gpu_env():
        print(f"[ab] {note}", flush=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = _reference_path()
    print(f"[ab] reference: {reference} (exists={reference.exists()})", flush=True)
    print(f"[ab] workload: {args.num_prompts} prompts x {args.output_len} tok; arms={arms}", flush=True)

    result: dict = {
        "submission": str(SUBMISSION),
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "reference": str(reference),
        "arms": {},
    }
    for arm in arms:
        extra = ARM_ENV[arm]
        print(f"\n===== ARM {arm} (extra_env={extra}) =====", flush=True)
        timing = serve_profile.run_timing_pass(
            SUBMISSION, SERVER_PYTHON, out_dir, arm,
            num_prompts=args.num_prompts, output_len=args.output_len, extra_env=extra,
        )
        spec_log = timing.get("spec_log") or {}
        steptime = timing.get("steptime") or {}
        greedy = _greedy_for(out_dir, arm, reference)
        result["arms"][arm] = {
            "extra_env": extra,
            "greedy": greedy,
            "steady_gen_tps_mean": spec_log.get("steady_gen_tps_mean"),
            "e_accept_exact": spec_log.get("e_accept_exact"),
            "e_accept_interval_mean": spec_log.get("e_accept_interval_mean"),
            "verify_gpu_ms_p50": steptime.get("verify_gpu_ms"),
            "drafter_gpu_ms_p50": steptime.get("drafter_gpu_ms"),
            "decode_records": (timing.get("decode_summary") or {}).get("num_records"),
        }
        (out_dir / "ab_result.json").write_text(json.dumps(result, indent=2))
        g = greedy
        print(f"[ab] {arm}: greedy verdict={g['verdict']} "
              f"identical={g['num_identical']}/{g['num_prompts_compared']} "
              f"(overlap frac={g['identical_frac_overlap']}) "
              f"onset min/med/max={g['onset_min']}/{g['onset_median']}/{g['onset_max']} | "
              f"steady_tps={result['arms'][arm]['steady_gen_tps_mean']} "
              f"E_accept={result['arms'][arm]['e_accept_exact']} "
              f"verify_gpu_ms={result['arms'][arm]['verify_gpu_ms_p50']}", flush=True)

    # A/B deltas when both arms ran.
    a = result["arms"]
    if "patched" in a and "baseline" in a:
        p, b = a["patched"], a["baseline"]
        if p["steady_gen_tps_mean"] and b["steady_gen_tps_mean"]:
            result["tps_delta"] = p["steady_gen_tps_mean"] - b["steady_gen_tps_mean"]
            result["frac_improvement"] = result["tps_delta"] / b["steady_gen_tps_mean"]
        result["greedy_identical_match"] = (
            p["greedy"]["num_identical"] == b["greedy"]["num_identical"]
            and p["greedy"]["num_divergent"] == b["greedy"]["num_divergent"]
        )
    (out_dir / "ab_result.json").write_text(json.dumps(result, indent=2))
    print(f"\n[ab] ===== SUMMARY =====\n{json.dumps(result, indent=2)}", flush=True)
    print(f"[ab] artifacts -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
