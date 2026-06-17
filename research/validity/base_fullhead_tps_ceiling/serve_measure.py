"""PR #544 -- base_fullhead TPS-ceiling: serve+measure the two ends of the gap.

LOCAL MEASUREMENT ONLY. ``analysis_only=true``, ``official_tps=0``. No HF job, no
submission, no served-file change. Serves the PACKAGED ``fa2sw_strict_surgical357``
submission twice back-to-back on the same pod and measures the robust official-spec
``wall_tps`` (= num_completion_tokens / decode duration_s) at the full 128x512
workload, plus realized E[T] and the env-gated ``steptime_patch`` (STEPTIME=1)
per-step split:

  exec.gpu   verify-step GPU time (M=8 spec-verify: attention + body GEMM + lm_head)
  exec.gap   host/python gap between steps (the drafter propose lands here)
  exec.cpu   verify-step call wall (host)
  draft.gpu  drafter (MTP propose) GPU time

Two arms (same submission, only env differs -> a clean A/B):

  base_fullhead  full 42L int4 body (google/gemma-4-E4B-it-qat-w4a16-ct snapshot) +
                 FULL 262k tied bf16 lm_head (LM_HEAD_PRUNE=0, PCK04 inactive).
                 fern #535 whh42dgd recipe: serve_ok, PPL 2.006 byte-exact, ~253.78 TPS.
  osoi5_ship     DEFAULT manifest = 37L baked body (/tmp/osoi5-v0-baked) + 16k int4
                 pruned head (LM_HEAD_PRUNE=1) -- the unsafe ship, ~353.73 local TPS.

E[T] (realized accept length) is captured two independent ways and cross-checked:
  (a) /metrics: vLLM spec_decode accepted/draft counter delta around the decodes,
      E[T] = 1 + K*accepted/drafted (K=num_speculative_tokens=7);
  (b) steptime-derived: E[T] = wall_tps * t_cycle, t_cycle=(exec.gap+exec.cpu)/1e3.

Hard-reject guard: refuse to serve base_fullhead if the resolved LOCAL_MODEL_DIR's
lm_head row count < 262144 (a silent 16k fallback must not masquerade as full-head).

Run under the repo .venv; serve/decode subprocs use the submission serve venv::

    .venv/bin/python -m research.validity.base_fullhead_tps_ceiling.serve_measure \
        --arms base_fullhead,osoi5_ship --n-decodes 2
"""
from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from research.speed.byteexact_realization_gap.run_realization_gap import (  # noqa: E402
    grep_log,
    parse_steptime,
)
from research.tps_noise_floor.run_noise_floor import (  # noqa: E402
    preflight_gpu,
    _gpu_mem_used_mib,
)

OUT_ROOT = ROOT / "research" / "validity" / "base_fullhead_tps_ceiling"
K_SPEC = 7
PPL_GATE = 2.42

# Anchors from the PR body (LOCAL scale; never mix with official).
BFH_TPS_ANCHOR = 253.78       # fern #535 whh42dgd base_fullhead local wall_tps
BFH_PPL_ANCHOR = 2.006        # fern #535 base_fullhead PPL byte-exact
OSOI5_TPS_ANCHOR = 353.73     # unsafe osoi5 ship local wall_tps
UNSAFE_FRONTIER_LOCAL = 442.0 # byte-exact equivalence-legal frontier (#523 lineage)


def _resolve_qat_snapshot() -> str:
    base = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    snaps = sorted(p for p in base.glob("*") if (p / "config.json").exists())
    if not snaps:
        raise RuntimeError(f"no qat-w4a16-ct snapshot under {base}")
    return str(snaps[0])


def _lm_head_rows(model_dir: str) -> int:
    """Read the safetensors header (no tensor load) and return the lm_head output rows.

    Handles both the stock dense head (lm_head.weight [V,H]) and the PCK04 int4
    packed head (lm_head.weight_packed [V,H/8])."""
    sft = Path(model_dir) / "model.safetensors"
    with open(sft, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for key in ("lm_head.weight", "lm_head.weight_packed"):
        if key in header:
            return int(header[key]["shape"][0])
    raise RuntimeError(f"no lm_head tensor in {sft}; keys={list(header)[:8]}")


def guard_full_head(model_dir: str) -> int:
    rows = _lm_head_rows(model_dir)
    if rows < 262144:
        raise RuntimeError(
            f"FULL-HEAD GUARD TRIPPED: resolved lm_head rows={rows} < 262144 in "
            f"{model_dir} -- a silent 16k fallback cannot masquerade as full-head."
        )
    print(f"[ceiling] full-head guard OK: lm_head rows={rows} in {model_dir}", flush=True)
    return rows


def _arms(qat_snapshot: str) -> dict[str, dict[str, Any]]:
    return {
        "base_fullhead": {
            "submission": "fa2sw_strict_surgical357",
            "extra_env": {
                "LOCAL_MODEL_DIR": qat_snapshot,
                "PLE_FOLD_TARGET_MODEL": qat_snapshot,
                "LM_HEAD_PRUNE": "0",
                "LM_HEAD_PRUNE_REQUIRE": "0",
                "PCK04_KEEPSET": "",
            },
            "full_head_guard": True,
            "label": "full 42L int4 body + FULL 262k bf16 lm_head (fern #535 recipe)",
            "tps_anchor": BFH_TPS_ANCHOR,
        },
        "osoi5_ship": {
            "submission": "fa2sw_strict_surgical357",
            "extra_env": {},  # default manifest = 37L baked + 16k int4 pruned head
            "full_head_guard": False,
            "label": "37L baked body + 16k int4 pruned head (unsafe osoi5 ship)",
            "tps_anchor": OSOI5_TPS_ANCHOR,
        },
    }


# ---------------------------------------------------------------------------
# E[T] via Prometheus /metrics (spec_decode counters)
# ---------------------------------------------------------------------------
def _fetch_metrics(base_url: str) -> str:
    url = base_url.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        print(f"[ceiling] /metrics fetch failed: {exc}", flush=True)
        return ""


def _spec_counters(metrics_text: str) -> dict[str, float]:
    """Pull cumulative spec-decode accepted/draft token counters from /metrics.

    vLLM v1 exposes (names vary by version):
      vllm:spec_decode_num_accepted_tokens_total
      vllm:spec_decode_num_draft_tokens_total
    We grep robustly for accepted/draft families and return the first match each."""
    import re

    out: dict[str, float] = {}
    pats = {
        "accepted": r"vllm:spec_decode_num_accepted_tokens(?:_total)?\b(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
        "draft": r"vllm:spec_decode_num_draft_tokens(?:_total)?\b(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
        "drafts": r"vllm:spec_decode_num_drafts(?:_total)?\b(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
        "emitted": r"vllm:spec_decode_num_emitted_tokens(?:_total)?\b(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
    }
    for name, pat in pats.items():
        vals = [float(x) for x in re.findall(pat, metrics_text)]
        if vals:
            out[name] = sum(vals)
    return out


def _etp_from_counters(pre: dict[str, float], post: dict[str, float]) -> dict[str, Any]:
    acc = post.get("accepted", 0.0) - pre.get("accepted", 0.0)
    drf = post.get("draft", 0.0) - pre.get("draft", 0.0)
    out: dict[str, Any] = {
        "metrics_accepted_delta": acc,
        "metrics_draft_delta": drf,
        "metrics_available": bool(post),
    }
    if drf > 0:
        rate = acc / drf
        out["accept_rate"] = rate
        out["e_t_metrics"] = 1.0 + K_SPEC * rate
    else:
        out["accept_rate"] = None
        out["e_t_metrics"] = None
    return out


# ---------------------------------------------------------------------------
# One arm: fresh server, N back-to-back decodes (median wall_tps), E[T], PPL
# ---------------------------------------------------------------------------
def run_arm(
    arm_name: str, arm: dict[str, Any], server_python: Path, out_dir: Path,
    *, n_decodes: int, num_prompts: int, output_len: int, seed: int,
    do_ppl: bool,
) -> dict[str, Any]:
    submission_dir = (ROOT / "submissions" / arm["submission"]).resolve()
    arm_dir = out_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    server_log = arm_dir / "server.log"
    extra_env = dict(arm["extra_env"])
    extra_env["STEPTIME"] = "1"  # per-step split; events at python boundary, no compute change
    extra_env["STEPTIME_REPORT_EVERY"] = "512"  # emit agg often; we read the decode-phase one
    print(f"\n[ceiling] ===== ARM {arm_name} :: {arm['label']} =====", flush=True)
    print(f"[ceiling] submission={arm['submission']} extra_env={extra_env}", flush=True)

    if arm.get("full_head_guard"):
        guard_full_head(extra_env["LOCAL_MODEL_DIR"])

    preflight_gpu()
    decodes: list[dict[str, Any]] = []
    peak_mem_mib = 0
    server_ready_s = None
    ppl_summary: dict[str, Any] | None = None
    server_error: str | None = None
    etp_metrics: dict[str, Any] = {"metrics_available": False}
    spec_pre: dict[str, float] = {}
    spec_post: dict[str, float] = {}
    steptime_decode: dict[str, Any] = {}

    t_load0 = time.time()
    try:
        with harness.LocalServer(
            submission_dir, server_python=server_python,
            log_path=server_log, extra_env=extra_env,
        ) as server:
            server_ready_s = time.time() - t_load0
            print(f"[ceiling] {arm_name}: server ready in {server_ready_s:.0f}s", flush=True)
            m = _gpu_mem_used_mib()
            if m:
                peak_mem_mib = max(peak_mem_mib, m)
            spec_pre = _spec_counters(_fetch_metrics(server.base_url))
            for i in range(n_decodes):
                decode_out = arm_dir / f"decode_round{i:02d}.jsonl"
                decode_summary = arm_dir / f"decode_round{i:02d}.summary.json"
                summary = harness.capture_decode(
                    server_python, base_url=server.base_url,
                    model=server.served_model_name, out_file=decode_out,
                    summary_file=decode_summary, num_prompts=num_prompts,
                    output_len=output_len, seed=seed,
                )
                n_tok = int(summary.get("num_completion_tokens", 0))
                dur = float(summary.get("duration_s", 0.0))
                wall_tps = n_tok / dur if dur > 0 else float("nan")
                n_completed = int(summary.get("num_records", 0))
                decodes.append({
                    "round": i, "wall_tps": wall_tps, "num_completion_tokens": n_tok,
                    "decode_duration_s": dur, "num_completed_prompts": n_completed,
                    "expected_tokens": num_prompts * output_len,
                })
                print(f"[ceiling] {arm_name} round {i}: wall_tps={wall_tps:.2f} "
                      f"tok={n_tok}/{num_prompts * output_len} dur={dur:.1f}s "
                      f"completed={n_completed}", flush=True)
                mm = _gpu_mem_used_mib()
                if mm:
                    peak_mem_mib = max(peak_mem_mib, mm)
            spec_post = _spec_counters(_fetch_metrics(server.base_url))
            etp_metrics = _etp_from_counters(spec_pre, spec_post)
            print(f"[ceiling] {arm_name}: E[T]_metrics={etp_metrics.get('e_t_metrics')} "
                  f"(acc_delta={etp_metrics.get('metrics_accepted_delta')} "
                  f"draft_delta={etp_metrics.get('metrics_draft_delta')})", flush=True)
            # Capture steptime from the DECODE phase BEFORE PPL runs -- PPL's prefill-heavy
            # exec steps would otherwise contaminate the cumulative steady-state agg.
            try:
                steptime_decode = parse_steptime(server_log.read_text(errors="replace"))
            except OSError:
                steptime_decode = {}
            if do_ppl:
                try:
                    ppl_summary = harness.run_ppl(
                        server_python, base_url=server.base_url,
                        model=server.served_model_name,
                        out_file=arm_dir / "ppl.jsonl",
                        summary_file=arm_dir / "ppl.summary.json",
                    )
                    print(f"[ceiling] {arm_name}: PPL={ppl_summary.get('ppl')} "
                          f"records={ppl_summary.get('num_records')}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[ceiling] {arm_name}: WARN PPL failed: {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001
        server_error = repr(exc)
        print(f"[ceiling] {arm_name}: SERVER ERROR: {server_error}", flush=True)

    mech = grep_log(server_log)
    wall_tps_vals = [d["wall_tps"] for d in decodes if d["wall_tps"] == d["wall_tps"]]
    median_tps = statistics.median(wall_tps_vals) if wall_tps_vals else float("nan")

    # steptime-derived E[T] = wall_tps * t_cycle, t_cycle = (exec.gap + exec.cpu)/1e3.
    # Use the DECODE-PHASE steptime (captured before PPL) not the post-PPL grep_log one.
    st = (steptime_decode or {}).get("exec", {})
    dr = (steptime_decode or {}).get("draft", {})
    t_cycle_ms = None
    e_t_steptime = None
    if st.get("gap_mean") is not None and st.get("cpu_mean") is not None and median_tps == median_tps:
        t_cycle_ms = st["gap_mean"] + st["cpu_mean"]
        e_t_steptime = median_tps * t_cycle_ms / 1e3

    arm_rec = {
        "arm": arm_name, "submission": arm["submission"], "label": arm["label"],
        "extra_env": arm["extra_env"], "tps_anchor": arm.get("tps_anchor"),
        "median_wall_tps": median_tps, "wall_tps_values": wall_tps_vals,
        "wall_tps_n": len(wall_tps_vals),
        "wall_tps_std": statistics.stdev(wall_tps_vals) if len(wall_tps_vals) > 1 else 0.0,
        "server_ready_s": server_ready_s, "peak_gpu_mem_mib": peak_mem_mib,
        "ppl": (ppl_summary or {}).get("ppl"), "ppl_num_records": (ppl_summary or {}).get("num_records"),
        "num_completed_prompts": decodes[0]["num_completed_prompts"] if decodes else None,
        "completion_full": bool(decodes and decodes[0]["num_completion_tokens"] == num_prompts * output_len),
        "server_error": server_error,
        "steptime_exec": st, "steptime_draft": dr,
        "t_cycle_ms": t_cycle_ms,
        "e_t_steptime": e_t_steptime,
        "etp_metrics": etp_metrics,
        "spec_counters_pre": spec_pre, "spec_counters_post": spec_post,
        "mechanism": {k: v for k, v in mech.items() if k != "steptime"},
        "decodes": decodes,
    }
    print(f"[ceiling] ARM {arm_name} SUMMARY: median_wall_tps={median_tps:.2f} "
          f"(n={len(wall_tps_vals)}) PPL={arm_rec['ppl']} "
          f"E[T]_metrics={etp_metrics.get('e_t_metrics')} E[T]_steptime={e_t_steptime} "
          f"| exec gpu={st.get('gpu_mean')} gap={st.get('gap_mean')} cpu={st.get('cpu_mean')} "
          f"draft gpu={dr.get('gpu_mean')} | onegraph={mech.get('onegraph_captured')} "
          f"surgical={mech.get('surgical_armed')}", flush=True)
    return arm_rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="base_fullhead,osoi5_ship")
    ap.add_argument("--n-decodes", type=int, default=2)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--no-ppl", dest="do_ppl", action="store_false", default=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny serve+decode sanity (8 prompts x 16 tok, 1 decode, no ppl)")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args(argv)

    if args.smoke:
        args.num_prompts = min(args.num_prompts, 8)
        args.output_len = min(args.output_len, 16)
        args.n_decodes = 1
        args.do_ppl = False

    for note in paths.prepare_local_gpu_env():
        print(f"[ceiling] {note}", flush=True)

    qat_snapshot = _resolve_qat_snapshot()
    print(f"[ceiling] qat snapshot = {qat_snapshot}", flush=True)
    arms = _arms(qat_snapshot)
    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in want if a not in arms]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(arms)}")

    base_manifest = harness.load_manifest(
        (ROOT / "submissions" / "fa2sw_strict_surgical357").resolve())
    server_python = harness.ensure_server_venv(base_manifest["dependencies"])
    print(f"[ceiling] server_python={server_python}", flush=True)

    out_dir = (OUT_ROOT / ("smoke" if args.smoke else args.tag)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ceiling] arms={want} n_decodes={args.n_decodes} "
          f"workload={args.num_prompts}x{args.output_len} seed={args.seed} -> {out_dir}", flush=True)

    t0 = time.time()
    arm_recs: dict[str, dict[str, Any]] = {}
    for name in want:
        arm_recs[name] = run_arm(
            name, arms[name], server_python, out_dir,
            n_decodes=args.n_decodes, num_prompts=args.num_prompts,
            output_len=args.output_len, seed=args.seed, do_ppl=args.do_ppl,
        )
    elapsed = time.time() - t0

    result = {
        "schema": "base_fullhead_serve_measure_v1",
        "analysis_only": True, "official_tps": 0,
        "qat_snapshot": qat_snapshot,
        "k_spec": K_SPEC, "ppl_gate": PPL_GATE,
        "num_prompts": args.num_prompts, "output_len": args.output_len, "seed": args.seed,
        "n_decodes": args.n_decodes, "smoke": args.smoke,
        "elapsed_s": elapsed,
        "anchors": {
            "bfh_tps": BFH_TPS_ANCHOR, "bfh_ppl": BFH_PPL_ANCHOR,
            "osoi5_tps": OSOI5_TPS_ANCHOR, "unsafe_frontier_local": UNSAFE_FRONTIER_LOCAL,
        },
        "arms": arm_recs,
    }
    out_json = out_dir / "serve_results.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\n[ceiling] wrote {out_json} (elapsed {elapsed:.0f}s)", flush=True)
    for name, rec in arm_recs.items():
        print(f"[ceiling]   {name}: wall_tps={rec['median_wall_tps']:.2f} "
              f"PPL={rec['ppl']} E[T]_m={rec['etp_metrics'].get('e_t_metrics')} "
              f"E[T]_st={rec.get('e_t_steptime')} exec_gpu={rec['steptime_exec'].get('gpu_mean')}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
