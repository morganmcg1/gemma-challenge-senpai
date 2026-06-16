"""PR #525 — byte-exact split-KV NUM_SEGMENTS occupancy sweep toward the ~457 frontier.

Single-variable sweep of ``BYTEEXACT_NUM_SEGMENTS`` (the parallel-softmax segment
COUNT / occupancy knob) holding ``BYTEEXACT_FIXED_TPS=4`` FIXED (the byte-exact
reduction-order invariant) and everything else identical to #519's variant
(same submission ``fa2sw_strict_byteexact_splitkv399``, same 128x512 workload,
seed 1, 3 decodes, full PPL, same served stack). LOCAL ONLY: ``analysis_only``,
``official_tps=0``, NO HF job, NO ``--launch``, NO submission. Challenge PAUSED.

Mechanism (read from the stock wheel + ``byteexact_splitkv_patch.py``)
---------------------------------------------------------------------
The fixed-order patch rewrites ``tiles_per_segment = cdiv(seq_len, S*TILE_SIZE)``
to the literal ``tiles_per_segment = T`` in both @triton.jit kernels, so each
parallel-softmax segment covers a FIXED span of ``T*TILE_SIZE = 4*16 = 64`` keys
at a fixed absolute key position (-> M-invariant byte-exact reduction). The 3D
grid is ``(num_q_blocks, num_kv_heads, S)`` where ``S=BYTEEXACT_NUM_SEGMENTS``;
segment ``segm_idx`` early-exits when ``segm_idx*64 >= seq_len``. So the number
of ACTIVE (real-work) segments is ``ceil(seq_len/64)`` -- pinned by seq_len and
T, INDEPENDENT of S. ``reduce_segments`` masks to that same active set, so the
byte-exact result is identical for every ``S >= ceil(seq_len/64)``.

Two HARD kernel bounds on the runnable S (BOTH must hold):
  * Power-of-2: ``reduce_segments`` does ``tl.arange(0, NUM_PAR_SOFTMAX_SEGMENTS)``
    and Triton requires an arange range to be a power of 2 -> S MUST be a power of
    2 (S=48 / S=96 raise ``ValueError: arange's range must be a power of 2`` ~80s
    into engine init -- confirmed by smoke).
  * Coverage: ``coverage = T*S*TILE_SIZE = 64*S`` keys must be >= the longest
    decode KV length, or the kernel silently drops the context tail (PPL/identity
    break). #519's full workload max KV = 2939 (prompt 2427 + 512), so the safe
    floor is ``S >= ceil(2939/64) = 46``; S=32 (coverage 2048) under-covers.

Together these leave only powers of 2 >= 46, i.e. {64, 128, 256, ...}. The
SMALLEST valid S for this workload is therefore 64 -- already the #519 value --
so the sweep can only explore UPWARD. And upward is predicted flat-to-slightly
-slower: active segments = ceil(seq_len/64) (mean ~13, max ~46) is pinned by
FIXED_TPS, so S>64 only adds idle early-exit grid blocks + a wider scratch buffer
/ masked reduction (FlashDecoding SM-fill optimum is ~SMs/(batch*heads) << 64).
That is NOT the +15 TPS the realization-gap target would need.

HARD GATES per config (a config failing any is reported but EXCLUDED from "best"):
self-determinism == 1.0, PPL <= 2.42, 128/128 completion, and byte-exact
mechanism_valid. The occupancy-optimal byte-exact S is the gate-passing config
with the highest warm-median TPS.

Usage::

    .venv/bin/python -m research.validity.splitkv_numseg_sweep.sweep_numseg \
        --segments 48 64 96 128
    .venv/bin/python -m research.validity.splitkv_numseg_sweep.sweep_numseg --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.validity.splitkv399_full_recert import recert_ab  # noqa: E402
from scripts.local_validation import paths  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent

# Provenance anchors (this PR's baselines, from the #519 merge).
S519_TPS = 442.35              # #519 variant @ S=64, full 128x512 (run kwhylaeg)
SURGICAL357_LOCAL_TPS = 357.06   # same-session surgical-357 control, local
SURGICAL357_OFFICIAL_TPS = 375.857  # surgical-357 official ship (j7qao5e9)
FRONTIER_TARGET_TPS = 457.5    # strict-frontier prediction (the ceiling)
PPL_GATE = recert_ab.PPL_GATE  # 2.42
SIGMA_HW = recert_ab.SIGMA_HW  # 4.864
TILE_SIZE = recert_ab.BYTEEXACT_TILE_SIZE  # 16
WORKLOAD_MAX_KV_519 = 2939     # measured in #519 decode artifacts (id gpqa_diamond-1d37a7a51d)


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def validate_segments(segs: list[int], max_kv: int = WORKLOAD_MAX_KV_519) -> list[str]:
    """Reject S values the byte-exact split-KV kernel cannot run (fail fast, not 80s
    into engine init). Returns a list of human-readable rejection reasons (empty = ok).

    Two hard kernel bounds: S must be a power of 2 (``tl.arange(0, S)`` in
    ``reduce_segments``) and ``coverage = 64*S`` must cover the longest decode KV.
    """
    bad: list[str] = []
    for s in segs:
        if not _is_pow2(s):
            bad.append(f"S={s}: not a power of 2 (reduce_segments tl.arange(0,S) "
                       f"requires a power-of-2 range)")
        elif 64 * s < max_kv:
            bad.append(f"S={s}: coverage {64 * s} < workload max KV {max_kv} "
                       f"(would drop the context tail)")
    return bad


def _decode_max_kv(out_dir: Path) -> int | None:
    """Authoritative per-run longest decode KV = max(num_prompt_tokens + num_completion_tokens)."""
    f = out_dir / "decode_round00.jsonl"
    if not f.exists():
        return None
    mx = 0
    with f.open() as fh:
        for line in fh:
            r = json.loads(line)
            mx = max(mx, int(r.get("num_prompt_tokens", 0)) + int(r.get("num_completion_tokens", 0)))
    return mx or None


def _make_args(s: int, smoke: bool, n_decodes: int) -> SimpleNamespace:
    return SimpleNamespace(
        arm="variant",
        smoke=smoke,
        num_prompts=(8 if smoke else paths.NUM_PROMPTS),
        output_len=(16 if smoke else paths.OUTPUT_LEN),
        seed=paths.SEED,
        n_decodes=(2 if smoke else n_decodes),
        do_ppl=(not smoke),
        num_segments=s,
        out_tag=(f"smoke_s{s}" if smoke else f"s{s}"),
        out_root=str(OUT_ROOT),
    )


def _augment(result: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Attach per-run coverage/active-segment fields computed from the decode artifacts."""
    run_max_kv = _decode_max_kv(out_dir)
    coverage = result.get("coverage_keys")
    result["run_max_kv"] = run_max_kv
    result["coverage_ge_run_max_kv"] = (
        bool(coverage is not None and run_max_kv is not None and coverage >= run_max_kv)
    )
    if run_max_kv:
        result["active_segments_max_kv"] = -(-run_max_kv // (recert_ab.BYTEEXACT_TILE_SIZE * 4))
    return result


def run_one_s(s: int, smoke: bool, n_decodes: int) -> dict[str, Any]:
    args = _make_args(s, smoke, n_decodes)
    print(f"\n[sweep] ===== S={s} (smoke={smoke}) =====", flush=True)
    result = recert_ab.run_one_arm("variant", args)
    return _augment(result, OUT_ROOT / args.out_tag)


def _load_s(s: int, smoke: bool) -> dict[str, Any] | None:
    """Reload a previously-run S from disk (for --summarize-only / crash recovery)."""
    tag = f"smoke_s{s}" if smoke else f"s{s}"
    out_dir = OUT_ROOT / tag
    p = out_dir / "arm_result.json"
    if not p.exists():
        return None
    return _augment(json.loads(p.read_text()), out_dir)


def _gate_pass(r: dict[str, Any]) -> bool:
    return bool(
        r.get("self_determinism_rate") == 1.0
        and r.get("ppl_passes_gate")
        and r.get("completion_128_128")
        and r.get("mechanism_valid")
        and r.get("coverage_ge_run_max_kv")
    )


def summarize(results: dict[int, dict[str, Any]], smoke: bool) -> dict[str, Any]:
    table = []
    for s in sorted(results):
        r = results[s]
        table.append({
            "num_segments": s,
            "coverage_keys": r.get("coverage_keys"),
            "run_max_kv": r.get("run_max_kv"),
            "active_segments_max_kv": r.get("active_segments_max_kv"),
            "warm_median_tps": r.get("median_warm_wall_tps"),
            "ppl": r.get("ppl"),
            "ppl_passes_gate": r.get("ppl_passes_gate"),
            "self_determinism": r.get("self_determinism_rate"),
            "completion_128_128": r.get("completion_128_128"),
            "byteexact_mechanism_valid": r.get("mechanism_valid"),
            "peak_gpu_mem_mib": r.get("peak_gpu_mem_mib"),
            "gate_pass": _gate_pass(r),
        })

    passing = [row for row in table if row["gate_pass"]
               and isinstance(row["warm_median_tps"], (int, float))]
    if passing:
        best = max(passing, key=lambda row: row["warm_median_tps"])
        best_s = best["num_segments"]
        best_tps = best["warm_median_tps"]
        s64 = next((row for row in table if row["num_segments"] == 64), None)
        s64_tps = s64["warm_median_tps"] if s64 else None
        vs_519 = (best_tps - S519_TPS) if isinstance(best_tps, (int, float)) else None
        vs_s64_insession = (best_tps - s64_tps) if isinstance(s64_tps, (int, float)) else None
        vs_surg_local = (best_tps - SURGICAL357_LOCAL_TPS) if isinstance(best_tps, (int, float)) else None
        reaches_457 = bool(best_tps >= FRONTIER_TARGET_TPS - SIGMA_HW)
        beats_519 = bool(vs_519 is not None and vs_519 > SIGMA_HW)
        if reaches_457:
            verdict = (f"FRONTIER REACHED: byte-exact S={best_s} -> {best_tps:.2f} TPS "
                       f">= ~457 within sigma_hw; the NUM_SEGMENTS occupancy knob closes the gap.")
        elif beats_519:
            verdict = (f"PARTIAL: byte-exact S={best_s} -> {best_tps:.2f} TPS beats #519 S=64 "
                       f"by {vs_519:+.2f} (> sigma_hw) but short of ~457.")
        else:
            verdict = (f"FLAT: occupancy-optimal byte-exact S={best_s} -> {best_tps:.2f} TPS; "
                       f"NUM_SEGMENTS does NOT move TPS beyond #519's 442 within sigma_hw "
                       f"(vs #519 {vs_519:+.2f}, vs in-session S=64 "
                       f"{vs_s64_insession if vs_s64_insession is None else round(vs_s64_insession, 2)}). "
                       f"Realization gap 442->457 is NOT a segment-count effect: active segments "
                       f"= ceil(seq_len/64) is pinned by FIXED_TPS, so S only adds idle blocks.")
    else:
        best_s = best_tps = vs_519 = vs_s64_insession = vs_surg_local = None
        reaches_457 = beats_519 = False
        verdict = "FAIL: no config passed all hard gates (self-det/PPL/128-128/byte-exact/coverage)."

    summary = {
        "pr": 525,
        "exp": "byte-exact split-KV NUM_SEGMENTS occupancy sweep toward ~457",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "smoke": smoke,
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "no_served_file_change": True,
        "fixed_tps_held": 4,
        "workload_max_kv_519": WORKLOAD_MAX_KV_519,
        "safe_floor_segments": -(-WORKLOAD_MAX_KV_519 // (TILE_SIZE * 4)),
        # ---- KEY OUTPUTS (PR-required) ----
        "best_byteexact_s": best_s,
        "best_s_warm_median_tps": best_tps,
        "best_s_vs_519_delta": vs_519,
        "best_s_vs_insession_s64_delta": vs_s64_insession,
        "best_s_vs_surgical357_local_delta": vs_surg_local,
        "best_s_vs_surgical357_official_delta": (
            (best_tps - SURGICAL357_OFFICIAL_TPS) if isinstance(best_tps, (int, float)) else None),
        "all_swept_configs_byteexact": bool(all(row["byteexact_mechanism_valid"] for row in table)),
        "all_swept_configs_ppl_pass": bool(all(row["ppl_passes_gate"] for row in table)),
        "reaches_457": reaches_457,
        "beats_519_by_sigma": beats_519,
        "sigma_hw": SIGMA_HW,
        "frontier_target_tps": FRONTIER_TARGET_TPS,
        "baseline_519_tps": S519_TPS,
        "per_s_table": table,
        "verdict": verdict,
    }
    summary_path = OUT_ROOT / ("sweep_summary_smoke.json" if smoke else "sweep_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n[sweep] ================= NUM_SEGMENTS SWEEP SUMMARY =================", flush=True)
    print(f"  {'S':>4} {'cov':>6} {'maxKV':>6} {'TPS':>8} {'PPL':>9} {'selfdet':>8} "
          f"{'128/128':>7} {'byteexact':>9} {'mem_MiB':>8} {'gate':>5}", flush=True)
    for row in table:
        print(f"  {row['num_segments']:>4} {row['coverage_keys']:>6} "
              f"{str(row['run_max_kv']):>6} "
              f"{(row['warm_median_tps'] if row['warm_median_tps'] is None else round(row['warm_median_tps'],2)):>8} "
              f"{str(row['ppl']):>9} {str(row['self_determinism']):>8} "
              f"{str(row['completion_128_128']):>7} {str(row['byteexact_mechanism_valid']):>9} "
              f"{str(row['peak_gpu_mem_mib']):>8} {str(row['gate_pass']):>5}", flush=True)
    print(f"  best_byteexact_s = {best_s}  best_tps = {best_tps}  "
          f"vs_519(442.35) = {vs_519}  reaches_457 = {reaches_457}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"[sweep] artifacts -> {summary_path}", flush=True)
    return summary


def _log_wandb(summary: dict[str, Any], results: dict[int, dict[str, Any]],
               wandb_group: str) -> None:
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[sweep] wandb import failed ({exc}); skip", flush=True)
        return
    # One run per S (grouped) + a summary run, so the W&B group shows the sweep.
    for s in sorted(results):
        r = results[s]
        run = wandb_logging.init_wandb_run(
            job_type="splitkv-numseg-sweep",
            agent="stark",
            name=f"stark/splitkv-numseg-s{s}",
            group=wandb_group,
            tags=["pr525", "splitkv-numseg-sweep", "analysis-only", "byteexact-splitkv",
                  f"S{s}"],
            config={
                "pr": 525, "num_segments": s, "fixed_tps": r.get("byteexact_fixed_tps"),
                "coverage_keys": r.get("coverage_keys"), "run_max_kv": r.get("run_max_kv"),
                "workload_prompts": paths.NUM_PROMPTS, "workload_output_len": paths.OUTPUT_LEN,
                "seed": paths.SEED, "analysis_only": True, "official_tps": 0,
                "submission": r.get("submission"),
            },
        )
        if run is None:
            print("[sweep] wandb disabled (no API key); skip", flush=True)
            return
        flat = {
            "sweep/num_segments": s,
            "sweep/warm_median_tps": r.get("median_warm_wall_tps"),
            "sweep/ppl": r.get("ppl"),
            "sweep/ppl_passes_gate": int(bool(r.get("ppl_passes_gate"))),
            "sweep/self_determinism": r.get("self_determinism_rate"),
            "sweep/completion_128_128": int(bool(r.get("completion_128_128"))),
            "sweep/byteexact_mechanism_valid": int(bool(r.get("mechanism_valid"))),
            "sweep/coverage_keys": r.get("coverage_keys"),
            "sweep/run_max_kv": r.get("run_max_kv"),
            "sweep/active_segments_max_kv": r.get("active_segments_max_kv"),
            "sweep/peak_gpu_mem_mib": r.get("peak_gpu_mem_mib"),
            "sweep/gate_pass": int(_gate_pass(r)),
            "sweep/vs_519_delta": (
                (r.get("median_warm_wall_tps") - S519_TPS)
                if isinstance(r.get("median_warm_wall_tps"), (int, float)) else None),
        }
        flat = {k: v for k, v in flat.items() if isinstance(v, (int, float))}
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.finish_wandb(run)
        print(f"[sweep] wandb logged S={s} run_id={getattr(run, 'id', None)}", flush=True)

    run = wandb_logging.init_wandb_run(
        job_type="splitkv-numseg-sweep",
        agent="stark",
        name="stark/splitkv-numseg-sweep-summary",
        group=wandb_group,
        tags=["pr525", "splitkv-numseg-sweep", "analysis-only", "byteexact-splitkv", "summary"],
        config={"pr": 525, "fixed_tps": 4, "baseline_519_tps": S519_TPS,
                "frontier_target_tps": FRONTIER_TARGET_TPS, "analysis_only": True,
                "official_tps": 0, "safe_floor_segments": summary["safe_floor_segments"]},
    )
    if run is None:
        return
    keys = ["best_byteexact_s", "best_s_warm_median_tps", "best_s_vs_519_delta",
            "best_s_vs_insession_s64_delta", "best_s_vs_surgical357_local_delta",
            "best_s_vs_surgical357_official_delta", "reaches_457", "beats_519_by_sigma",
            "all_swept_configs_byteexact", "all_swept_configs_ppl_pass", "safe_floor_segments"]
    flat = {}
    for k in keys:
        v = summary.get(k)
        if isinstance(v, bool):
            flat[f"summary/{k}"] = int(v)
        elif isinstance(v, (int, float)):
            flat[f"summary/{k}"] = v
    wandb_logging.log_summary(run, flat, step=0)
    run.summary["verdict"] = summary["verdict"]
    try:
        import wandb
        tbl = wandb.Table(columns=list(summary["per_s_table"][0].keys()))
        for row in summary["per_s_table"]:
            tbl.add_data(*[row[c] for c in tbl.columns])
        run.log({"per_s_table": tbl})
    except Exception as exc:  # noqa: BLE001
        print(f"[sweep] wandb table log skipped: {exc}", flush=True)
    wandb_logging.log_json_artifact(run, name="splitkv_numseg_sweep",
                                    artifact_type="operative-cert", data=summary)
    wandb_logging.finish_wandb(run)
    print(f"[sweep] wandb summary run_id={getattr(run, 'id', None)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segments", type=int, nargs="+", default=[64, 128, 256],
                    help="BYTEEXACT_NUM_SEGMENTS values to sweep. Must be powers of 2 "
                         "(Triton arange) with coverage 64*S >= workload max KV 2939 -> "
                         "valid set is {64, 128, 256, ...}; 48/96 fail Triton, 32 under-covers")
    ap.add_argument("--allow-invalid", action="store_true",
                    help="skip the power-of-2 / coverage pre-check (for deliberate failure probes)")
    ap.add_argument("--n-decodes", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny serve+decode plumbing check (one S, 8x16, 2 decodes, no ppl)")
    ap.add_argument("--summarize-only", action="store_true",
                    help="skip serving; rebuild the combined summary from on-disk s<S>/arm_result.json")
    ap.add_argument("--wandb-group", default="splitkv-numseg-sweep")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    segs = args.segments[:1] if args.smoke else args.segments
    if not args.allow_invalid:
        bad = validate_segments(segs)
        if bad:
            for reason in bad:
                print(f"[sweep] REJECT {reason}", flush=True)
            raise SystemExit(
                "invalid S value(s); the byte-exact split-KV kernel needs powers of 2 "
                "with 64*S >= 2939. Use {64,128,256} or pass --allow-invalid to probe a failure.")
    results: dict[int, dict[str, Any]] = {}
    for s in segs:
        if args.summarize_only:
            r = _load_s(s, args.smoke)
            if r is None:
                print(f"[sweep] WARN no on-disk result for S={s}; skipping", flush=True)
                continue
            results[s] = r
        else:
            results[s] = run_one_s(s, args.smoke, args.n_decodes)
    if not results:
        raise SystemExit("no results to summarize")

    summary = summarize(results, args.smoke)
    if not args.no_wandb and not args.smoke:
        _log_wandb(summary, results, args.wandb_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
