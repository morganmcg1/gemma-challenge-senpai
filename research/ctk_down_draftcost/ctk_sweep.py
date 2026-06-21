#!/usr/bin/env python
"""PR #824 — CENTROID_TOP_K (ctk) DOWN-sweep on int4_mtp_bi0_int4head.

ctk = the MTP drafter's ``centroid_intermediate_top_k``: per draft step the
masked-embedding head scores 2048 centroids, keeps the top ctk, and gathers
ctk*128 token rows from the bf16 draft lm_head. #792 swept ctk UP (32->64->128,
null). This sweeps it DOWN {32(control=shipped), 24, 16, 8} to test whether
cutting the drafter's per-step gather cost (the drafter is ~20% of the int4head
decode cycle) nets a TPS gain despite a possibly weaker draft (lower E_accept).

Config-only: ``CENTROID_TOP_K`` env -> serve.py overlays a drafter dir with a
patched ``config.json`` (no kernel/weight change). control = ctk unset = stock
shipped drafter (ctk=32). Greedy temp=0 decode is LOSSLESS regardless of ctk
(the rejection sampler short-circuits to target-argmax), so the emitted tokens,
128/128 and PPL are ctk-INDEPENDENT; only SPEED (drafter_gpu_ms, E_accept, TPS)
moves. PPL+128/128 are therefore measured once on the control arm.

The int4head verifier is served from this user's own HF cache (hardlinked,
zero-disk) with HF_HUB_OFFLINE=1 so there is no auth/download. All numbers are
LOCAL A10G exploratory probes, NOT the official a10g-small TPS.

Per arm (one server, N decode reps): wall TPS (per-rep, mean+/-std), steady TPS
(whole-arm engine-meter intervals, mean+/-std), drafter_gpu_ms / verify_gpu_ms
(STEPTIME p50), E_accept (exact, 1+K*acc/draft), and per-position accept rate
(positions 1..K of K=6 from /metrics). Writes incremental JSON; a soft wall-clock
cap leaves margin under the hard run timeout.

Usage:
  python ctk_sweep.py [REPS]          # full sweep (default REPS=3)
  python ctk_sweep.py --smoke         # control-vs-ctk16 losslessness + log check
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

SUB = ROOT / "submissions" / "int4_mtp_bi0_int4head"
OUT = ROOT / "research" / "ctk_down_draftcost"
GROUP = "bi0-ctk-down-draftcost"

# Arms: (label, ctk). ctk=None -> CENTROID_TOP_K unset -> stock shipped drafter
# (ctk=32), byte-identical control. Capped to {32,24,16,8} per the PR.
ARMS: list[tuple[str, int | None]] = [
    ("ctk32_control", None),
    ("ctk24", 24),
    ("ctk16", 16),
    ("ctk8", 8),
]
K = 6  # num_speculative_tokens (knee; not varied per PR cap)

# Serve the int4head verifier from the local (hardlinked) cache, fully offline:
# no auth, no download. DISABLE_LOG_STATS=0 re-enables the spec_decode_* counters
# the manifest's --disable-log-stats would hide (canonical E_accept + per-pos).
BASE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "DISABLE_LOG_STATS": "0",
}

# Soft wall-clock cap (s): stop launching new reps/arms past this, so partial
# results are always persisted before the hard run timeout kills the process.
SOFT_CAP_S = float(os.environ.get("CTK_SOFT_CAP_S", "4680"))  # 78 min

_CENTROID_LINE = re.compile(r"top_k=(\d+),\s*active_tokens=(\d+)/(\d+)")
_PERPOS = re.compile(
    r"^vllm:spec_decode_num_accepted_tokens_per_pos(?:_total)?\{([^}]*)\}\s+([\d.eE+-]+)",
    re.M,
)


def _ensure_clean_slate(port: int = 8000, timeout_s: float = 90.0) -> None:
    """Kill any lingering vLLM server and wait for ``port`` + GPU to drain BEFORE
    spawning a new arm. Guards the teardown race observed in #824 smoke: when a
    prior arm's EngineCore outlives ``LocalServer.__exit__`` (parent dies fast,
    SIGTERM'd child shuts down slowly), the next arm's ``_wait_ready`` polls the
    stale :port server, returns "ready in 0s", and the real freshly-spawned server
    fails to bind :port and LEAKS. A clean slate makes each arm hit its OWN server
    (validated downstream via centroid_top_k_logged + a real ~255s decode)."""
    subprocess.run(["pkill", "-KILL", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    subprocess.run(["pkill", "-KILL", "-f", "EngineCore"], check=False)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        port_busy = subprocess.run(
            ["bash", "-c", f"ss -ltn 2>/dev/null | grep -q ':{port} '"], check=False
        ).returncode == 0
        try:
            used = int(
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True,
                ).stdout.strip().splitlines()[0]
            )
        except Exception:  # noqa: BLE001
            used = 0
        if not port_busy and used < 1500:
            return
        time.sleep(2)
    print(f"[ctk] WARN: clean-slate wait timed out (port_busy={port_busy} gpu_used={used}MiB)", flush=True)


def _per_position(metrics_text: str) -> dict[int, float]:
    """position-label -> accepted-token count (summed across engine label sets)."""
    by: dict[int, float] = {}
    for m in _PERPOS.finditer(metrics_text):
        pm = re.search(r'position="(\d+)"', m.group(1))
        if pm:
            p = int(pm.group(1))
            by[p] = by.get(p, 0.0) + float(m.group(2))
    return by


def _finite(x: object) -> bool:
    """True iff x is a real, non-nan, non-inf number (guards resume validity)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and x not in (
        float("inf"), float("-inf"),
    )


def _stats(vals: list[float]) -> dict[str, float]:
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    if not vals:
        return {"n": 0}
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": mean,
        "std": std,
        "min": min(vals),
        "max": max(vals),
        "cv_pct": (100.0 * std / mean) if mean else 0.0,
    }


def _arm_env(ctk: int | None, expected_steps: int) -> dict[str, str]:
    env = dict(BASE_ENV)
    env.update(serve_profile._steptime_env(expected_steps))
    if ctk is not None:
        env["CENTROID_TOP_K"] = str(ctk)
        env["CENTROID_TOP_K_OUT"] = f"/tmp/drafter_ctk{ctk}"
    return env


def run_arm(
    label: str,
    ctk: int | None,
    server_python: Path,
    *,
    reps: int,
    num_prompts: int,
    output_len: int,
    do_ppl: bool,
    t_start: float,
) -> dict:
    out_dir = OUT / "sweep" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"
    expected_steps = max(64, num_prompts * output_len // 2)
    env = _arm_env(ctk, expected_steps)
    _ensure_clean_slate()
    arm: dict = {
        "label": label,
        "ctk_requested": ctk if ctk is not None else 32,
        "num_prompts": num_prompts,
        "output_len": output_len,
        "K": K,
        "reps": [],
    }
    with harness.LocalServer(
        SUB, server_python=server_python, port=8000, log_path=log_path,
        extra_env=env, startup_timeout_s=1800,
    ) as srv:
        # Confirm the ctk override actually took effect at MTP construction.
        cl = _CENTROID_LINE.search(log_path.read_text())
        if cl:
            arm["centroid_top_k_logged"] = int(cl.group(1))
            arm["active_tokens_logged"] = int(cl.group(2))
            print(f"[ctk] {label}: serve log centroids top_k={cl.group(1)} "
                  f"active_tokens={cl.group(2)}/{cl.group(3)}", flush=True)
        for rep in range(1, reps + 1):
            if time.time() - t_start > SOFT_CAP_S:
                print(f"[ctk] soft cap hit before {label} rep{rep}; stopping reps", flush=True)
                arm["soft_capped"] = True
                break
            dout = out_dir / f"decode_rep{rep}.jsonl"
            dsum = out_dir / f"decode_rep{rep}.summary.json"
            t0 = time.time()
            summary = harness.capture_decode(
                server_python, base_url=srv.base_url, model=srv.served_model_name,
                out_file=dout, summary_file=dsum,
                num_prompts=num_prompts, output_len=output_len, timeout_s=3600,
            )
            wall = time.time() - t0
            dur = float(summary.get("duration_s", wall))
            ntok = int(summary.get("num_completion_tokens", 0))
            nrec = int(summary.get("num_records", 0))
            wall_tps = ntok / dur if dur else float("nan")
            rep_rec = {
                "rep": rep, "num_records": nrec, "num_completion_tokens": ntok,
                "duration_s": dur, "wall_s_outer": wall, "wall_tps": wall_tps,
            }
            arm["reps"].append(rep_rec)
            print(f"[ctk] {label} rep{rep}/{reps}: nrec={nrec} ntok={ntok} "
                  f"dur={dur:.1f}s wall_tps={wall_tps:.2f} "
                  f"(elapsed {time.time()-t_start:.0f}s)", flush=True)
            (out_dir / "arm_partial.json").write_text(json.dumps(arm, indent=2))
        # Whole-run /metrics: spec counters + per-position (cumulative across reps).
        try:
            mtext = serve_profile._get_text(f"{srv.base_url}/metrics")
            arm["spec_metrics"] = serve_profile.parse_spec_metrics(mtext)
            arm["per_position_accepted"] = {str(k): v for k, v in _per_position(mtext).items()}
        except Exception as exc:  # noqa: BLE001
            arm["spec_metrics_error"] = str(exc)
        # PPL + 128/128 are ctk-independent -> measure once (control arm only).
        if do_ppl:
            try:
                arm["ppl"] = harness.run_ppl(
                    server_python, base_url=srv.base_url, model=srv.served_model_name,
                    out_file=out_dir / "ppl.jsonl", summary_file=out_dir / "ppl.summary.json",
                    timeout_s=1800,
                )
                print(f"[ctk] {label} PPL={arm['ppl'].get('ppl')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                arm["ppl_error"] = str(exc)

    # Whole-arm log parse: STEPTIME GPU split + spec-log E_accept + steady TPS.
    # The served EngineCore flushes its piped stdout buffer only as it winds down,
    # a beat AFTER LocalServer.__exit__'s proc.wait() returns. Reading the log the
    # instant the `with` block closed raced that flush and parsed drafter_gpu_ms to
    # nan for some arms (#824 ctk24: 0 raw records at read time, 4000 a few s later).
    # Retry-read until the [steptime] raw records land so drafter_gpu_ms is captured
    # for EVERY arm, not just the ones whose read happened to win the race.
    log_text = log_path.read_text()
    steptime = serve_profile.parse_steptime(log_text)
    for _attempt in range(20):
        if steptime.get("raw_exec_steps"):
            break
        time.sleep(1)
        log_text = log_path.read_text()
        steptime = serve_profile.parse_steptime(log_text)
    if not steptime.get("raw_exec_steps"):
        print(f"[ctk] WARN: {label} no [steptime] raw records after 20s "
              f"(drafter_gpu_ms will be nan)", flush=True)
    arm["steptime"] = steptime
    arm["spec_log"] = serve_profile.parse_spec_log(log_text)
    gen_tps = [float(x) for x in serve_profile._GEN_TPS_RE.findall(log_text)]
    arm["steady_gen_tps_stats"] = _stats(gen_tps)

    # Headline scalars.
    arm["drafter_gpu_ms"] = arm["steptime"].get("drafter_gpu_ms")
    arm["verify_gpu_ms"] = arm["steptime"].get("verify_gpu_ms")
    arm["e_accept"] = arm["spec_log"].get("e_accept_exact")
    arm["steady_gen_tps_mean"] = arm["spec_log"].get("steady_gen_tps_mean")
    arm["wall_tps_stats"] = _stats([r["wall_tps"] for r in arm["reps"]])

    # Per-position accept RATE = accepted-at-pos / num_drafts (positions 0..K-1
    # -> reported as draft positions 1..K).
    nd = (arm.get("spec_metrics") or {}).get("num_drafts")
    if nd:
        ppos = arm.get("per_position_accepted") or {}
        arm["per_position_accept_rate"] = {
            f"pos{int(p)+1}": ppos[str(p)] / nd for p in range(K) if str(p) in ppos
        }
    arm["num_drafts"] = nd

    (out_dir / "arm.json").write_text(json.dumps(arm, indent=2))
    return arm


def smoke_test(server_python: Path) -> int:
    """control vs ctk16 on 2 prompts: assert the override logs AND that greedy
    output is byte-identical (lossless) per prompt, before the full sweep."""
    print("[ctk] SMOKE: control vs ctk16, 2 prompts x 32 tokens", flush=True)
    sha: dict[str, list] = {}
    for label, ctk in [("control", None), ("ctk16", 16)]:
        out_dir = OUT / "smoke" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "server.log"
        env = _arm_env(ctk, expected_steps=64)
        _ensure_clean_slate()
        with harness.LocalServer(
            SUB, server_python=server_python, port=8000, log_path=log_path,
            extra_env=env, startup_timeout_s=1800,
        ) as srv:
            harness.capture_decode(
                server_python, base_url=srv.base_url, model=srv.served_model_name,
                out_file=out_dir / "decode.jsonl", summary_file=out_dir / "decode.summary.json",
                num_prompts=2, output_len=32, timeout_s=600,
            )
        cl = _CENTROID_LINE.search(log_path.read_text())
        groups = cl.groups() if cl else None
        rows = [json.loads(x) for x in (out_dir / "decode.jsonl").read_text().splitlines() if x.strip()]
        rows.sort(key=lambda r: r.get("index", 0))
        sha[label] = [r.get("completion_token_sha256") for r in rows]
        print(f"[ctk] SMOKE {label}: centroids_line={groups} "
              f"n_rows={len(rows)} sha={sha[label]}", flush=True)
    expect = {"control": ("32",), "ctk16": ("16",)}
    lossless = sha.get("control") == sha.get("ctk16") and all(sha.get("control") or [None])
    print(f"\n[ctk] SMOKE lossless(control==ctk16 token sha256)={lossless}", flush=True)
    (OUT / "smoke" / "smoke_result.json").write_text(
        json.dumps({"sha256": sha, "lossless": lossless}, indent=2)
    )
    return 0 if lossless else 1


def aggregate(arms: list[dict]) -> dict:
    rows = []
    for a in arms:
        rows.append({
            "label": a["label"],
            "ctk": a.get("centroid_top_k_logged", a.get("ctk_requested")),
            "active_tokens": a.get("active_tokens_logged"),
            "drafter_gpu_ms": a.get("drafter_gpu_ms"),
            "verify_gpu_ms": a.get("verify_gpu_ms"),
            "e_accept": a.get("e_accept"),
            "wall_tps_mean": (a.get("wall_tps_stats") or {}).get("mean"),
            "wall_tps_std": (a.get("wall_tps_stats") or {}).get("std"),
            "wall_tps_cv_pct": (a.get("wall_tps_stats") or {}).get("cv_pct"),
            "steady_tps_mean": a.get("steady_gen_tps_mean"),
            "steady_tps_std": (a.get("steady_gen_tps_stats") or {}).get("std"),
            "ppl": (a.get("ppl") or {}).get("ppl"),
            "decode_records": [r["num_records"] for r in a.get("reps", [])],
            "per_position_accept_rate": a.get("per_position_accept_rate"),
        })
    # Deltas vs control (first arm).
    ctrl = rows[0] if rows else {}
    c_wall = ctrl.get("wall_tps_mean") or 0.0
    c_steady = ctrl.get("steady_tps_mean") or 0.0
    for r in rows:
        if c_wall and r.get("wall_tps_mean") is not None:
            r["wall_tps_delta_pct"] = 100.0 * (r["wall_tps_mean"] - c_wall) / c_wall
        if c_steady and r.get("steady_tps_mean") is not None:
            r["steady_tps_delta_pct"] = 100.0 * (r["steady_tps_mean"] - c_steady) / c_steady
    return {"group": GROUP, "K": K, "arms": rows}


def _print_curve(agg: dict) -> None:
    print("\n" + "=" * 96, flush=True)
    print("CTK DOWN-SWEEP CURVE (local A10G; not official a10g-small TPS)", flush=True)
    print("=" * 96, flush=True)
    hdr = (f"{'arm':14s} {'ctk':>4s} {'active':>6s} {'draft_ms':>8s} "
           f"{'E_acc':>6s} {'wallTPS':>10s} {'d%':>7s} {'steadyTPS':>11s} {'d%':>7s} {'PPL':>6s}")
    print(hdr, flush=True)
    print("-" * 96, flush=True)
    for r in agg["arms"]:
        wt = r.get("wall_tps_mean")
        st = r.get("steady_tps_mean")
        print(
            f"{r['label']:14s} {str(r.get('ctk')):>4s} {str(r.get('active_tokens')):>6s} "
            f"{(r.get('drafter_gpu_ms') or float('nan')):8.3f} "
            f"{(r.get('e_accept') or float('nan')):6.3f} "
            f"{(wt if wt is not None else float('nan')):7.2f}"
            f"+/-{(r.get('wall_tps_std') or 0):.1f} "
            f"{(r.get('wall_tps_delta_pct') if r.get('wall_tps_delta_pct') is not None else float('nan')):+6.2f} "
            f"{(st if st is not None else float('nan')):8.2f}"
            f"+/-{(r.get('steady_tps_std') or 0):.1f} "
            f"{(r.get('steady_tps_delta_pct') if r.get('steady_tps_delta_pct') is not None else float('nan')):+6.2f} "
            f"{(r.get('ppl') or float('nan')):6.3f}",
            flush=True,
        )
    print("-" * 96, flush=True)


def _load_arm(label: str, *, min_reps: int, expect_ppl: bool) -> dict | None:
    """Resume helper: return a VALID finished arm dict from disk, or None if the
    arm still needs a (re)run. An arm is valid iff it has >= min_reps decode reps
    AND a finite drafter_gpu_ms (the headline quantity #824 measures); the control
    arm additionally needs PPL. If the stored drafter_gpu_ms is nan but the now
    fully-flushed server.log carries the [steptime] raw records, repair it in place
    (this recovers arms hit by the flush-race without re-serving them)."""
    out_dir = OUT / "sweep" / label
    ap = out_dir / "arm.json"
    if not ap.exists():
        return None
    try:
        arm = json.loads(ap.read_text())
    except Exception:  # noqa: BLE001
        return None
    log_path = out_dir / "server.log"
    if not _finite(arm.get("drafter_gpu_ms")) and log_path.exists():
        st = serve_profile.parse_steptime(log_path.read_text())
        if st.get("raw_exec_steps"):
            arm["steptime"] = st
            arm["drafter_gpu_ms"] = st.get("drafter_gpu_ms")
            arm["verify_gpu_ms"] = st.get("verify_gpu_ms")
            ap.write_text(json.dumps(arm, indent=2))
            print(f"[ctk] {label}: repaired drafter_gpu_ms from log -> "
                  f"{arm['drafter_gpu_ms']}", flush=True)
    reps_ok = len([r for r in arm.get("reps", []) if r.get("num_records")]) >= min_reps
    ppl_ok = (not expect_ppl) or _finite((arm.get("ppl") or {}).get("ppl"))
    if reps_ok and _finite(arm.get("drafter_gpu_ms")) and ppl_ok:
        return arm
    return None


def main() -> int:
    if "--smoke" in sys.argv:
        for note in paths.prepare_local_gpu_env():
            print(f"[ctk] {note}", flush=True)
        server_python = harness.ensure_server_venv(harness.load_manifest(SUB)["dependencies"])
        print(f"[ctk] server_python={server_python}", flush=True)
        return smoke_test(server_python)

    reps = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
    num_prompts = int(os.environ.get("CTK_NUM_PROMPTS", str(paths.NUM_PROMPTS)))
    output_len = int(os.environ.get("CTK_OUTPUT_LEN", str(paths.OUTPUT_LEN)))
    (OUT / "sweep").mkdir(parents=True, exist_ok=True)

    for note in paths.prepare_local_gpu_env():
        print(f"[ctk] {note}", flush=True)
    server_python = harness.ensure_server_venv(harness.load_manifest(SUB)["dependencies"])
    print(f"[ctk] server_python={server_python} reps={reps} "
          f"workload={num_prompts}x{output_len} soft_cap={SOFT_CAP_S:.0f}s", flush=True)

    fresh = "--fresh" in sys.argv  # force re-run all arms, ignoring any on-disk arm.json
    t_start = time.time()
    arms: list[dict] = []
    for i, (label, ctk) in enumerate(ARMS):
        cached = None if fresh else _load_arm(label, min_reps=reps, expect_ppl=(i == 0))
        if cached is not None:
            print(f"\n##### ARM {label}: RESUME from disk "
                  f"(drafter_gpu_ms={cached.get('drafter_gpu_ms')}, "
                  f"reps={len(cached.get('reps', []))}, "
                  f"wall_tps_mean={(cached.get('wall_tps_stats') or {}).get('mean')}) #####",
                  flush=True)
            arms.append(cached)
            (OUT / "sweep" / "sweep_partial.json").write_text(json.dumps(arms, indent=2))
            continue
        if time.time() - t_start > SOFT_CAP_S:
            print(f"[ctk] soft cap hit before arm {label}; stopping", flush=True)
            break
        print(f"\n##### ARM {label} (ctk={ctk if ctk is not None else 32}) "
              f"elapsed={time.time()-t_start:.0f}s #####", flush=True)
        # PPL/128-128 are ctk-independent (greedy verify short-circuits to target
        # argmax), so measure once on the control -- PLUS the most aggressive arm
        # (lowest ctk) as a quality spot-check that the accumulated ULP near-tie
        # flips #792 saw stay PPL-neutral even at the extreme.
        arm = run_arm(
            label, ctk, server_python,
            reps=reps, num_prompts=num_prompts, output_len=output_len,
            do_ppl=(i == 0 or i == len(ARMS) - 1), t_start=t_start,
        )
        arms.append(arm)
        (OUT / "sweep" / "sweep_partial.json").write_text(json.dumps(arms, indent=2))

    agg = aggregate(arms)
    (OUT / "sweep" / "ctk_sweep_results.json").write_text(
        json.dumps({"aggregate": agg, "arms_full": arms}, indent=2)
    )
    _print_curve(agg)
    print(f"\n[ctk] results -> {OUT / 'sweep' / 'ctk_sweep_results.json'} "
          f"(total {time.time()-t_start:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
