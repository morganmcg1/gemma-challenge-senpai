#!/usr/bin/env python
"""PR #443 — Static CUDA-graph capture of the K=7/M=8 spec-decode loop (LOCAL, analysis-only).

Hypothesis (advisor): static CUDA-graph capture of the inter-step spec-decode loop
recovers per-step CPU/Python *launch* overhead as realized TPS, byte-exact.

CODE FINDING (submissions/fa2sw_treeverify_kenyan/sitecustomize.py): the served
fa2sw K=7 stack ALREADY captures the whole K=7 draft loop in ONE CUDA graph —
``ONEGRAPH=1`` -> ``_capture_graph`` records ``_run_graph_body`` (the 7 width-1
drafter forwards + fused-sparse-argmax + hidden copies) with ``torch.cuda.CUDAGraph``
and the hot path is a single ``graph.replay()`` (propose_onegraph). The paged-KV
block-table refresh stays OUTSIDE the graph (``_refresh_static_buffers``), exactly as
the PR requires. The DIXIE accept-prep is a fused Triton kernel, not Python. So the
"inter-step orchestration is NOT inside a graph capture" premise is FALSE for the
served stack: the lever is already deployed.

This probe therefore MEASURES, on the pod A10G (sm_86, on-target):

  1. self-abort gate  -> per-step host/CPU overhead as a fraction of cycle wall on the
                         DEPLOYED graph-on stack (PR #443 instr. 2: <0.5% => measured
                         abort, bank the negative, do NOT prototype).
  2. launch overhead  -> the per-step wall the deployed loop-graph already recovers,
                         measured directly as graph-ON vs graph-OFF (ONEGRAPH 1 vs 0,
                         a one-variable serve-time delta). This sizes the cost land
                         #444 (async pipelined drafting) would RE-INTRODUCE if a second
                         async stream forces graphs OFF — the number the advisor asked
                         to flag.
  3. equivalence      -> strict byte-exact greedy-token identity graph-ON vs graph-OFF
                         over the official eval set (official check_greedy_identity),
                         and PPL on the deployed arm (gate <= 2.42).

Instrument: the shipped STEPTIME probe (perf_counter + CUDA-event at the python call
boundary, OUTSIDE any graph). Per-step decode wall = exec_cpu + host_gap (p50 steady
state, the #275 host-to-host method). host overhead = wall - GPU-busy(verify+drafter).
This is the same harness ubel #284 used to measure host overhead 0.50% at out=512;
this card re-measures at the PR's 128->128 point and adds the graph-OFF arm.

Changes NO served file, NO emitted token, NO submission. NOT a launch. official_tps=0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import (  # noqa: E402
    greedy_gate,
    harness,
    paths,
    ppl_runner,
    serve_profile,
)
from scripts.profiler import prefill_denominator_probe as pdp  # noqa: E402

OUT_DIR = ROOT / "research" / "validity" / "cudagraph_capture_specloop"
DEFAULT_SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"

# --- anchors (PR #443 baseline block + ubel #284), imported exact ----------- #
FRONTIER_TPS = 467.14          # denken #423 strict realized-equivalence frontier (anchor base)
DEPLOYED_TPS = 481.53          # PR #52 deployed incumbent (NON-equiv, identity 0.9966)
DEPLOYED_PPL = 2.3772          # PR #52 deployed PPL
PPL_GATE = 2.42
SELF_ABORT_GATE_FRAC = 0.005   # PR #443 instr. 2: CPU overhead < 0.5% of cycle -> abort
UBEL284_HOST_FRAC = 0.0050     # ubel #284 (PR #284) measured host-overhead frac at out=512

# Arms: a one-variable serve-time delta on the loop-graph capture ONLY. Everything
# else (precache, MTP drafter, FA2 sliding, fused argmax, fused accept-prep, lmhead
# prune, detok-endonly) is held identical so the delta isolates the capture.
ARMS: dict[str, dict[str, str]] = {
    "graph_on": {},  # deployed: ONEGRAPH=1 (manifest default) -> K=7 loop in one CUDA graph
    "graph_off": {  # eager K=7 width-1 python loop (sitecustomize propose, no inter-step graph)
        "ONEGRAPH": "0",
        "LOOPGRAPH_REQUIRE_CAPTURE": "0",
    },
}


def _f(x: Any) -> float:
    try:
        v = float(x)
        return v if v == v else 0.0
    except (TypeError, ValueError):
        return 0.0


def _dump(out_dir: Path, name: str, obj: Any) -> None:
    (out_dir / name).write_text(json.dumps(obj, indent=2, default=str))
    print(f"[write] {out_dir / name}", flush=True)


# --------------------------------------------------------------------------- #
# one arm: serve once, measure timing + capture greedy tokens + (graph_on) PPL
# --------------------------------------------------------------------------- #
def serve_and_measure(
    arm: str,
    arm_env: dict[str, str],
    *,
    submission: Path,
    server_python: Path,
    out_dir: Path,
    num_prompts: int,
    output_len: int,
    seed: int,
    port: int,
    do_ppl: bool,
    do_capture: bool,
) -> dict[str, Any]:
    log_path = out_dir / f"server_{arm}.log"
    expected_steps = max(64, num_prompts * output_len // 4)
    guard_files = pdp._install_serve_guard(server_python)
    extra_env = {
        **pdp._steptime_env(expected_steps),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "DISABLE_LOG_STATS": "0",      # expose /metrics spec counters
        "PREFILL_PROBE_GUARD": "1",    # the validated pathless-route prometheus guard
        "PRECACHE_DATASET": str(paths.EVAL_PROMPTS),  # local precache warmup (manifest path absent locally)
        **arm_env,
    }
    res: dict[str, Any] = {
        "arm": arm, "arm_env": arm_env, "submission": str(submission),
        "num_prompts": num_prompts, "output_len": output_len, "seed": seed,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "analysis_only": True,
    }
    prompts = pdp.load_bench_prompts(num_prompts, seed)
    print(f"\n===== arm={arm} env={arm_env} prompts={num_prompts} out={output_len} =====", flush=True)
    try:
        with harness.LocalServer(
            submission, server_python=server_python, port=port,
            log_path=log_path, extra_env=extra_env, startup_timeout_s=1800,
        ) as srv:
            time.sleep(1.0)
            spec_base = serve_profile.parse_spec_metrics(pdp._get_text(f"{srv.base_url}/metrics", 10.0))
            # --- STEPTIME timing window (chat, faithful to the deployed decode) ---
            t0 = time.time()
            ok = 0
            for i, p in enumerate(prompts):
                try:
                    pdp.chat_completion(srv.base_url, srv.served_model_name, p["prompt_text"], output_len)
                    ok += 1
                except Exception as exc:  # noqa: BLE001 - a single bad request must not abort the window
                    print(f"[{arm}] timing request {i} failed: {exc}", flush=True)
            res["timing_wall_s"] = time.time() - t0
            res["timing_requests_ok"] = ok
            spec_final = serve_profile.parse_spec_metrics(pdp._get_text(f"{srv.base_url}/metrics", 10.0))
            # Snapshot the log NOW so the STEPTIME p50 isolates the timing window;
            # the capture/PPL drives below would otherwise append raw records.
            time.sleep(1.0)
            steptime_text = log_path.read_text(errors="ignore")
            res["steptime"] = serve_profile.parse_steptime(steptime_text)
            res["spec"] = {
                k: (_f(spec_final.get(k)) - _f(spec_base.get(k)))
                for k in ("num_drafts", "num_accepted_tokens", "num_draft_tokens")
            }
            nd, na = res["spec"]["num_drafts"], res["spec"]["num_accepted_tokens"]
            res["e_accept"] = (1.0 + na / nd) if nd else float("nan")
            # --- official greedy token-id capture (byte-exact identity input) ---
            if do_capture:
                cap = out_dir / f"decode_{arm}.jsonl"
                try:
                    csum = harness.capture_decode(
                        server_python, base_url=srv.base_url, model=srv.served_model_name,
                        out_file=cap, summary_file=out_dir / f"decode_{arm}_summary.json",
                        num_prompts=num_prompts, output_len=output_len, seed=seed,
                    )
                    res["capture"] = str(cap)
                    res["capture_records"] = csum.get("num_records")
                    res["capture_completion_tokens"] = csum.get("num_completion_tokens")
                except Exception as exc:  # noqa: BLE001
                    res["capture_error"] = repr(exc)
                    print(f"[{arm}] capture_decode failed: {exc}", flush=True)
            # --- PPL on the deployed (graph_on) arm only ---
            if do_ppl:
                try:
                    psum = ppl_runner.score_endpoint(
                        srv.base_url, srv.served_model_name,
                        out_dir=out_dir / f"ppl_{arm}", runner_python=server_python,
                    )
                    res["ppl"] = _f(psum.get("ppl"))
                    res["ppl_num_tokens"] = psum.get("num_tokens")
                except Exception as exc:  # noqa: BLE001
                    res["ppl_error"] = repr(exc)
                    print(f"[{arm}] ppl failed: {exc}", flush=True)
    finally:
        for fpath in guard_files:
            fpath.unlink(missing_ok=True)
        print("[guard] removed scratch prometheus guard from venv", flush=True)

    # --- derived per-step decode wall (the #275 host-to-host method) ---
    st = res.get("steptime") or {}
    exec_cpu_us = _f(st.get("exec_cpu_ms")) * 1000.0
    host_gap_us = _f(st.get("host_gap_ms")) * 1000.0
    verify_gpu_us = _f(st.get("verify_gpu_ms")) * 1000.0
    drafter_gpu_us = _f(st.get("drafter_gpu_ms")) * 1000.0
    wall_us = exec_cpu_us + host_gap_us
    gpu_busy_us = verify_gpu_us + drafter_gpu_us
    host_overhead_us = max(0.0, wall_us - gpu_busy_us)
    ea = _f(res.get("e_accept"))
    res["per_step"] = {
        "decode_wall_us": wall_us,
        "exec_cpu_us": exec_cpu_us,
        "host_gap_us": host_gap_us,
        "verify_gpu_us": verify_gpu_us,
        "drafter_gpu_us": drafter_gpu_us,
        "gpu_busy_us": gpu_busy_us,
        "host_overhead_us": host_overhead_us,
        "host_overhead_frac": (host_overhead_us / wall_us) if wall_us else float("nan"),
        "gpu_busy_share": (gpu_busy_us / wall_us) if wall_us else float("nan"),
        # local output throughput at conc=1: E_accept tokens emitted per decode cycle.
        "output_tps_local": (ea / (wall_us / 1e6)) if (wall_us and ea) else float("nan"),
    }
    print(f"[{arm}] wall={wall_us:.0f}us gpu_busy={gpu_busy_us:.0f}us "
          f"host_overhead={host_overhead_us:.0f}us ({100 * res['per_step']['host_overhead_frac']:.2f}%) "
          f"E_accept={ea:.3f} tps_local={res['per_step']['output_tps_local']:.1f}", flush=True)
    return res


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def analyze(arms: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    on = arms["graph_on"]
    off = arms.get("graph_off") or {}
    on_ps = on.get("per_step", {})
    off_ps = off.get("per_step", {})

    host_frac_on = _f(on_ps.get("host_overhead_frac"))
    wall_on = _f(on_ps.get("decode_wall_us"))
    wall_off = _f(off_ps.get("decode_wall_us"))
    tps_on = _f(on_ps.get("output_tps_local"))
    tps_off = _f(off_ps.get("output_tps_local"))

    have_off = bool(wall_off) and "serve_error" not in off
    launch_overhead_us = (wall_off - wall_on) if have_off else float("nan")
    launch_overhead_frac_of_off = (launch_overhead_us / wall_off) if (have_off and wall_off) else float("nan")
    # Realized TPS the deployed loop-graph already recovers (graph_on - graph_off).
    realized_delta_tps = (tps_on - tps_off) if have_off else float("nan")
    # Anchor: the graph-on point IS the deployed/frontier operating point; price the
    # measured fractional saving onto the 467.14 strict-equivalence frontier anchor.
    frac_saving = (launch_overhead_us / wall_off) if (have_off and wall_off) else float("nan")
    frontier_if_graphs_off = (FRONTIER_TPS * (1.0 - frac_saving)) if frac_saving == frac_saving else float("nan")
    land444_cost_tps_on_frontier = (FRONTIER_TPS - frontier_if_graphs_off) if frontier_if_graphs_off == frontier_if_graphs_off else float("nan")

    # self-abort gate (PR #443 instr. 2)
    self_abort = bool(host_frac_on < SELF_ABORT_GATE_FRAC) if host_frac_on == host_frac_on else None
    # This card adds NO new graph capture (it is already deployed): incremental
    # realized TPS contributed by *this PR* is 0 by construction.
    incremental_realized_tps = 0.0
    crosses_deployed_481 = False  # graph capture is already inside the deployed numbers

    # --- equivalence: byte-exact greedy identity graph_on vs graph_off ---
    identity: dict[str, Any] = {"checked": False}
    cap_on, cap_off = on.get("capture"), off.get("capture")
    if cap_on and cap_off and Path(cap_on).exists() and Path(cap_off).exists():
        try:
            rep = greedy_gate.compare(Path(cap_on), Path(cap_off))
            identity = {
                "checked": True,
                "verdict": rep.verdict,
                "num_prompts_compared": rep.num_prompts_compared,
                "num_identical": rep.num_identical,
                "num_divergent": rep.num_divergent,
                "total_tokens_compared": rep.total_tokens_compared,
                "total_divergent_tokens": rep.total_divergent_tokens,
                "byte_exact": bool(rep.verdict == "GREEDY_IDENTICAL"),
            }
        except Exception as exc:  # noqa: BLE001
            identity = {"checked": False, "error": repr(exc)}

    ppl_on = _f(on.get("ppl")) if on.get("ppl") is not None else float("nan")
    ppl_pass = bool(ppl_on <= PPL_GATE) if ppl_on == ppl_on and ppl_on > 0 else None

    # --- self-test ---
    nan_clean = all(v == v for v in [host_frac_on, wall_on, tps_on])
    gate_decided = self_abort is not None
    identity_ok = bool(identity.get("byte_exact")) if identity.get("checked") else None
    consistent_with_284 = bool(abs(host_frac_on - UBEL284_HOST_FRAC) < 0.01)  # within 1pp of #284's out=512 read
    self_test_passes = bool(nan_clean and gate_decided and (identity_ok is not False)
                            and (ppl_pass is not False))

    return {
        "pr": 443,
        "analysis_only": True,
        "is_launch": False,
        "official_tps": 0.0,
        "operating_point": "fa2sw_precache_kenyan precache-on single-stream greedy K=7(M=8) "
                           f"{args.num_prompts}->{args.output_len}",
        "anchors": {
            "frontier_tps": FRONTIER_TPS, "deployed_tps": DEPLOYED_TPS,
            "deployed_ppl": DEPLOYED_PPL, "ppl_gate": PPL_GATE,
            "self_abort_gate_frac": SELF_ABORT_GATE_FRAC, "ubel284_host_frac": UBEL284_HOST_FRAC,
        },
        "graph_on": on.get("per_step"),
        "graph_off": off.get("per_step") if have_off else {"unavailable": off.get("serve_error", "not run")},
        "e_accept": {"graph_on": _f(on.get("e_accept")),
                     "graph_off": _f(off.get("e_accept")) if have_off else None},
        "self_abort_gate": {
            "host_overhead_frac_graph_on": host_frac_on,
            "gate_frac": SELF_ABORT_GATE_FRAC,
            "self_abort": self_abort,
            "verdict": ("MEASURED-ABORT: host/CPU overhead below 0.5% gate; loop already "
                        "graph-captured (ONEGRAPH); no prototype warranted"
                        if self_abort else
                        ("host/CPU overhead >= 0.5% gate; flag to land #444"
                         if self_abort is False else "undecided (measurement missing)")),
        },
        "launch_overhead": {
            "graph_on_wall_us": wall_on,
            "graph_off_wall_us": wall_off if have_off else None,
            "launch_overhead_us": launch_overhead_us,
            "launch_overhead_frac_of_graphoff_wall": launch_overhead_frac_of_off,
            "realized_delta_tps_graph_recovers": realized_delta_tps,
            "frontier_if_graphs_off_tps": frontier_if_graphs_off,
            "land444_cost_tps_if_graphs_disabled": land444_cost_tps_on_frontier,
            "note": ("the deployed ONEGRAPH already captures this; land #444 re-introduces "
                     "this per-step cost iff a second async stream forces graphs OFF"),
        },
        "realized_tps_contribution_of_this_pr": incremental_realized_tps,
        "crosses_deployed_481": crosses_deployed_481,
        "equivalence": {"identity_graph_on_vs_off": identity, "ppl_graph_on": ppl_on,
                        "ppl_gate": PPL_GATE, "ppl_pass": ppl_pass},
        "self_test": {
            "nan_clean": nan_clean, "gate_decided": gate_decided,
            "identity_byte_exact": identity_ok, "ppl_pass": ppl_pass,
            "consistent_with_ubel284_host_frac": consistent_with_284,
            "self_test_passes": self_test_passes,
        },
        "primary_metric": {"name": "host_overhead_frac", "value": host_frac_on},
        "test_metric": {"name": "ppl", "value": ppl_on},
        "greedy_ppl_safety_certificate": {
            "analysis_only": True, "served_file_changed": False, "emitted_token_changed": False,
            "hf_job_or_submission": False, "is_launch": False,
            "baseline_tps_unchanged": FRONTIER_TPS, "tps_added_by_this_pr": 0.0,
        },
    }


def render_md(r: dict[str, Any]) -> str:
    g = r["self_abort_gate"]
    lo = r["launch_overhead"]
    on = r["graph_on"] or {}
    off = r["graph_off"] or {}
    eq = r["equivalence"]
    st = r["self_test"]
    ident = eq["identity_graph_on_vs_off"]
    L = ["# PR #443 — Static CUDA-graph capture of the K=7/M=8 spec-decode loop\n"]
    L.append(f"**PRIMARY `host_overhead_frac` (graph-on) = {100 * _f(g['host_overhead_frac_graph_on']):.2f}%** "
             f"· gate {100 * SELF_ABORT_GATE_FRAC:.1f}% · **self_abort = {g['self_abort']}**  ")
    L.append(f"**TEST `ppl` (graph-on) = {_f(eq['ppl_graph_on']):.4f}** (gate <= {eq['ppl_gate']}, "
             f"pass = {eq['ppl_pass']})  ")
    L.append(f"**realized TPS contributed by THIS PR = {r['realized_tps_contribution_of_this_pr']:.2f}** "
             f"(the loop-graph is ALREADY deployed; `crosses_deployed_481` = {r['crosses_deployed_481']})\n")
    L.append(f"> **Verdict:** {g['verdict']}. The served K=7 stack ALREADY replays the whole "
             f"K=7 draft loop as one CUDA graph (ONEGRAPH=1); the premise that the inter-step "
             f"orchestration is un-captured is false. The per-step host/CPU overhead that remains "
             f"OUTSIDE the graph is **{_f(on.get('host_overhead_us')):.0f} us "
             f"({100 * _f(on.get('host_overhead_frac')):.2f}%)** of the "
             f"**{_f(on.get('decode_wall_us')):.0f} us** cycle (GPU-busy share "
             f"**{100 * _f(on.get('gpu_busy_share')):.1f}%**). Adding *more* static capture cannot "
             f"recover it without fusing verify+draft+accept into one mega-graph, which is forbidden "
             f"(paged-KV block-table updates must stay outside the graph).\n")

    L.append("## 1. Per-step decode wall (STEPTIME p50 steady-state, host-to-host)\n")
    L.append("| quantity (us) | graph_on | graph_off |")
    L.append("|---|---|---|")
    for k, lab in [("verify_gpu_us", "verify (execute_model) GPU"),
                   ("drafter_gpu_us", "drafter (propose) GPU"),
                   ("exec_cpu_us", "exec host-call wall"),
                   ("host_gap_us", "inter-step gap"),
                   ("decode_wall_us", "**decode wall / step**"),
                   ("gpu_busy_us", "GPU-busy (verify+drafter)"),
                   ("host_overhead_us", "host overhead (wall - GPU-busy)")]:
        L.append(f"| {lab} | {_f(on.get(k)):.0f} | "
                 f"{(f'{_f(off.get(k)):.0f}' if off and 'unavailable' not in off else 'n/a')} |")
    L.append("")

    L.append("## 2. Launch overhead the deployed loop-graph already recovers (graph_on vs graph_off)\n")
    if lo["graph_off_wall_us"]:
        L.append(f"- graph_off wall - graph_on wall = **{_f(lo['launch_overhead_us']):.0f} us/step** "
                 f"({100 * _f(lo['launch_overhead_frac_of_graphoff_wall']):.2f}% of the graph-off cycle)")
        L.append(f"- realized output-TPS the graph recovers (local, conc=1) = "
                 f"**+{_f(lo['realized_delta_tps_graph_recovers']):.2f} TPS**")
        L.append(f"- priced onto the 467.14 frontier: disabling graphs would drop it to "
                 f"**{_f(lo['frontier_if_graphs_off_tps']):.2f} TPS** "
                 f"(**-{_f(lo['land444_cost_tps_if_graphs_disabled']):.2f} TPS** — the land #444 cost)")
    else:
        L.append("- graph_off arm unavailable; launch overhead not directly measured this run.")
    L.append(f"- {lo['note']}\n")

    L.append("## 3. Equivalence (byte-exact greedy identity graph_on vs graph_off)\n")
    if ident.get("checked"):
        L.append(f"- official verdict: **{ident.get('verdict')}** "
                 f"({ident.get('num_identical')}/{ident.get('num_prompts_compared')} prompts identical, "
                 f"{ident.get('total_divergent_tokens')} divergent tokens of "
                 f"{ident.get('total_tokens_compared')})")
    else:
        L.append(f"- identity check unavailable: {ident.get('error', 'captures missing')}")
    L.append(f"- PPL (graph_on, official ground truth) = **{_f(eq['ppl_graph_on']):.4f}** "
             f"(gate <= {eq['ppl_gate']}; deployed ref {DEPLOYED_PPL})\n")

    L.append("## 4. Self-test\n")
    for k, v in st.items():
        L.append(f"- {k}: **{v}**")
    L.append("")
    L.append("## Greedy/PPL-safety certificate\n")
    L.append("`analysis_only = True`. STEPTIME timing-only + official decode/PPL capture; no served-file "
             "change, no emitted-token change, no HF Job, no submission, NOT a launch. The loop-graph "
             "capture is ALREADY deployed, so this PR adds 0 TPS; the graph_on/off delta sizes what is "
             "already realized (and the land #444 cost of disabling it).")
    return "\n".join(L)


def _log_wandb(report: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.wandb_name:
        return
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb.init(
            project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
            name=args.wandb_name, group=args.wandb_group, job_type="analysis",
            config={"pr": 443, "num_prompts": args.num_prompts, "output_len": args.output_len,
                    "operating_point": report["operating_point"]},
        )
        flat: dict[str, Any] = {
            "host_overhead_frac_graph_on": _f(report["self_abort_gate"]["host_overhead_frac_graph_on"]),
            "self_abort": int(bool(report["self_abort_gate"]["self_abort"])),
            "launch_overhead_us": _f(report["launch_overhead"]["launch_overhead_us"]),
            "realized_delta_tps_graph_recovers": _f(report["launch_overhead"]["realized_delta_tps_graph_recovers"]),
            "land444_cost_tps_if_graphs_disabled": _f(report["launch_overhead"]["land444_cost_tps_if_graphs_disabled"]),
            "realized_tps_contribution_of_this_pr": _f(report["realized_tps_contribution_of_this_pr"]),
            "ppl_graph_on": _f(report["equivalence"]["ppl_graph_on"]),
            "identity_byte_exact": int(bool(report["self_test"]["identity_byte_exact"])),
            "self_test_passes": int(bool(report["self_test"]["self_test_passes"])),
            "graph_on_wall_us": _f((report["graph_on"] or {}).get("decode_wall_us")),
            "graph_off_wall_us": _f((report["graph_off"] or {}).get("decode_wall_us")),
            "gpu_busy_share_graph_on": _f((report["graph_on"] or {}).get("gpu_busy_share")),
        }
        wandb.log(flat)
        wandb.summary.update(flat)
        run.finish()
        print(f"[wandb] logged run {args.wandb_name}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed ({exc}); continuing", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="cudagraph-capture-kopt")
    ap.add_argument("--skip-graph-off", action="store_true", help="measure only the deployed graph_on arm")
    ap.add_argument("--smoke", action="store_true", help="tiny self-check (2 prompts, out 16, on-arm only, no PPL)")
    args = ap.parse_args()

    if args.smoke:
        args.num_prompts, args.output_len, args.skip_graph_off = 2, 16, True

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu-env] {note}", flush=True)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    submission = args.submission.resolve()
    manifest = harness.load_manifest(submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    arms: dict[str, dict[str, Any]] = {}
    arms["graph_on"] = serve_and_measure(
        "graph_on", ARMS["graph_on"], submission=submission, server_python=server_python,
        out_dir=out_dir, num_prompts=args.num_prompts, output_len=args.output_len,
        seed=args.seed, port=args.port, do_ppl=not args.smoke, do_capture=not args.smoke,
    )
    _dump(out_dir, "measure_graph_on.json", arms["graph_on"])

    if not args.skip_graph_off:
        try:
            arms["graph_off"] = serve_and_measure(
                "graph_off", ARMS["graph_off"], submission=submission, server_python=server_python,
                out_dir=out_dir, num_prompts=args.num_prompts, output_len=args.output_len,
                seed=args.seed, port=args.port, do_ppl=False, do_capture=True,
            )
            _dump(out_dir, "measure_graph_off.json", arms["graph_off"])
        except Exception as exc:  # noqa: BLE001
            arms["graph_off"] = {"arm": "graph_off", "serve_error": repr(exc)}
            print(f"[graph_off] serve failed: {exc}", flush=True)
            _dump(out_dir, "measure_graph_off.json", arms["graph_off"])

    report = analyze(arms, args)
    _dump(out_dir, "report.json", report)
    (out_dir / "report.md").write_text(render_md(report))
    print(f"[write] {out_dir / 'report.md'}", flush=True)
    _log_wandb(report, args)

    g = report["self_abort_gate"]
    lo = report["launch_overhead"]
    print("\n========== CUDA-GRAPH CAPTURE OF SPEC LOOP (PR #443) ==========", flush=True)
    print(f"PRIMARY host_overhead_frac (graph_on) = {100 * _f(g['host_overhead_frac_graph_on']):.2f}%  "
          f"self_abort = {g['self_abort']}", flush=True)
    print(f"launch overhead (graph_off - graph_on) = {_f(lo['launch_overhead_us']):.0f} us/step  "
          f"-> realized +{_f(lo['realized_delta_tps_graph_recovers']):.2f} TPS already deployed", flush=True)
    print(f"land #444 cost if graphs disabled = -{_f(lo['land444_cost_tps_if_graphs_disabled']):.2f} TPS "
          f"on the {FRONTIER_TPS} frontier", flush=True)
    print(f"TEST ppl (graph_on) = {_f(report['equivalence']['ppl_graph_on']):.4f}  "
          f"identity_byte_exact = {report['self_test']['identity_byte_exact']}", flush=True)
    print(f"realized TPS contributed by THIS PR = {report['realized_tps_contribution_of_this_pr']:.2f} "
          f"(lever already deployed)  self_test_passes = {report['self_test']['self_test_passes']}", flush=True)
    print(f"artifacts -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
