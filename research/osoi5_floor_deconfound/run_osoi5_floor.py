#!/usr/bin/env python3
"""PR #548: de-confound the shipped osoi5 quality collapse — is it body-bake
damage, or a recoverable first-token-EOS serving artifact?

Serves the AS-SHIPPED osoi5 (submissions/fa2sw_strict_surgical357: 12k-pruned
lm_head + dropped/baked int4 body, drafter + onegraph ON — the live 375.857
ship) and evaluates the short-chain quality axes BOTH as-served
(MIN_TOKENS_FLOOR off) AND with the #545 chat-scoped min_tokens=8 floor on,
reporting per-arm empty/EOS rate so a recoverable first-token-EOS empty is not
read as a reasoning loss.

One LocalServer per arm; MMLU-Pro (PRIMARY) + GSM8K (sampled) + GPQA-Diamond run
against that single server, each result saved immediately. The min_tokens floor
is the EXACT already-merged #545 mechanism (serve.py
ChatCompletionRequest.to_sampling_params); the only knob is the server-side
MIN_TOKENS_FLOOR env ("" = floor off / as-served, "8" = floored).

HARD-REJECT GUARD: at serve time the loaded lm_head must be the 12k pruned width
(12288 rows, NOT 262k full / 16k unpruned) and the body must be the dropped/baked
int4 body (37 layers, NOT the 42L intact body) — a silent fallback to a different
config raises so the cell cannot masquerade.

LOCAL ONLY. analysis_only=true, official_tps=0. No HF job, no submission, no
served-file change beyond locally reusing the merged floor mechanism.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # target/
sys.path.insert(0, str(ROOT))

SUBMISSION = ROOT / "submissions" / "fa2sw_strict_surgical357"
GSM8K_EVAL = ROOT / "research" / "downstream_quality_gsm8k" / "gsm8k_eval.py"
MMLU_GPQA_EVAL = ROOT / "research" / "validity" / "downstream_quality_eval" / "run_eval.py"

# ---- guard expectations (the as-shipped osoi5 substrate) -------------------------
EXPECT_LM_HEAD_ROWS = 12288   # 12k PCK04 keepset (the pruned ship)
EXPECT_LAYERS = 37            # dropped/baked body (surgical layer removal)
INTACT_LAYERS = 42           # the 42L intact body we must NOT be serving

# ---- public-evidence anchors (referenced for pct_of_base / gate) -----------------
# ubel #538 intact-body full-head control (un-collapsed ceiling) == PR #511 base arm.
BASE_MMLU_PRO = 0.668
BASE_GPQA_D = 0.444
BASE_GSM8K_SAMPLED = 0.878    # vanilla base GSM8K sampled (gate denominator, #541)
# Morgan #524 gate floors
GATE_MMLU_PRO = 0.601
GATE_GPQA_D = 0.400
GATE_GSM8K_FRAC = 0.90        # >= 90% of vanilla base


def _lm_head_rows(model_dir: str) -> int | None:
    from safetensors import safe_open

    for st in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(st, framework="pt", device="cpu") as h:
            keys = set(h.keys())
            for k in ("lm_head.weight", "lm_head.weight_packed", "lm_head.weight_shape"):
                if k in keys:
                    if k == "lm_head.weight_shape":
                        return int(h.get_tensor(k)[0].item())
                    return int(h.get_slice(k).get_shape()[0])
    return None


def _config_layers_quant(model_dir: str) -> tuple[int | None, str | None]:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None, None
    cfg = json.loads(pathlib.Path(cfg_path).read_text())
    layers = cfg.get("num_hidden_layers")
    if layers is None:
        layers = (cfg.get("text_config") or {}).get("num_hidden_layers")
    quant = (cfg.get("quantization_config") or {}).get("quant_method")
    return layers, quant


def _gpu_mem_used_gb() -> float:
    """Snapshot of max memory.used across visible GPUs (peak proxy, model resident)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        vals = [float(x.strip()) for x in out.splitlines() if x.strip()]
        return round(max(vals) / 1024.0, 3) if vals else 0.0
    except Exception:
        return 0.0


def _served_dir_from_log(log_path: pathlib.Path) -> str | None:
    """Parse the active pruned dst the server actually loaded."""
    if not log_path.exists():
        return None
    served = None
    for line in log_path.read_text(errors="ignore").splitlines():
        if "[lmhead-prune] active dst=" in line:
            # format: [lmhead-prune] active dst=/tmp/osoi5-12k-baked keepset=...
            frag = line.split("active dst=", 1)[1].split()[0].strip()
            served = frag
    return served


def _floor_active_in_log(log_path: pathlib.Path) -> bool:
    if not log_path.exists():
        return False
    return "[min-tokens] chat-endpoint min_tokens floor active" in log_path.read_text(errors="ignore")


def run_guard(arm: str, floor_on: bool, log_path: pathlib.Path) -> dict:
    """Hard-reject guard: confirm the served substrate is the 12k-pruned/baked ship
    AND that the floor state matches the requested arm. Raises on any mismatch so a
    silent fallback to a different config cannot masquerade as the osoi5 cell."""
    served_dir = _served_dir_from_log(log_path)
    if served_dir is None:
        raise RuntimeError(
            "GUARD FAIL: no '[lmhead-prune] active dst=' in server log — the "
            "as-shipped osoi5 prune phase did not run (LM_HEAD_PRUNE must be 1). "
            "Refusing to attribute results to the pruned ship."
        )
    rows = _lm_head_rows(served_dir)
    layers, quant = _config_layers_quant(served_dir)
    floor_logged = _floor_active_in_log(log_path)

    problems = []
    if rows != EXPECT_LM_HEAD_ROWS:
        problems.append(
            f"lm_head rows={rows} != expected pruned 12k width {EXPECT_LM_HEAD_ROWS} "
            f"(served_dir={served_dir})"
        )
    if layers == INTACT_LAYERS:
        problems.append(
            f"num_hidden_layers={layers} == intact 42L body — expected the "
            f"dropped/baked body ({EXPECT_LAYERS}L)"
        )
    if layers != EXPECT_LAYERS:
        problems.append(
            f"num_hidden_layers={layers} != expected baked-body {EXPECT_LAYERS}L"
        )
    if quant != "compressed-tensors":
        problems.append(f"quant_method={quant!r} != 'compressed-tensors' (int4 baked body)")
    if floor_on and not floor_logged:
        problems.append("floored arm requested but server log shows NO min_tokens floor active")
    if (not floor_on) and floor_logged:
        problems.append("as-served arm requested but server log shows a min_tokens floor active")

    guard = {
        "arm": arm,
        "floor_on": floor_on,
        "served_dir": served_dir,
        "lm_head_rows": rows,
        "num_hidden_layers": layers,
        "quant_method": quant,
        "floor_active_in_log": floor_logged,
        "is_12k_pruned_head": rows == EXPECT_LM_HEAD_ROWS,
        "is_baked_body": layers == EXPECT_LAYERS and quant == "compressed-tensors",
        "passed": not problems,
    }
    if problems:
        raise RuntimeError("GUARD FAIL: " + " | ".join(problems) + f" :: {json.dumps(guard)}")
    print(f"[guard] PASS arm={arm} floor_on={floor_on} served_dir={served_dir} "
          f"lm_head_rows={rows} layers={layers} quant={quant} floor_logged={floor_logged}",
          flush=True)
    return guard


def _stream(cmd: list[str], env: dict | None = None, cwd: str | None = None) -> int:
    print(f"[run] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=cwd)
    print(f"[run] rc={proc.returncode} wall={time.time() - t0:.0f}s :: {cmd[0]} ...", flush=True)
    return proc.returncode


def run_mmlu(base_url: str, model: str, arm: str, n: int, seed: int,
             max_tokens: int, max_conn: int, limit: int, eval_python: str) -> dict:
    out = HERE / f"osoi5_{arm}_mmlu_pro.json"
    cmd = [eval_python, str(MMLU_GPQA_EVAL),
           "--task", "mmlu_pro", "--arm", f"osoi5_{arm}",
           "--out", str(out), "--n", str(n), "--seed", str(seed),
           "--max-tokens", str(max_tokens), "--max-connections", str(max_conn),
           "--base-url", f"{base_url.rstrip('/')}/v1", "--model", model]
    if limit:
        cmd += ["--limit", str(limit)]
    rc = _stream(cmd)
    return {"axis": "mmlu_pro", "rc": rc, "out": str(out)}


def run_gpqa(base_url: str, model: str, arm: str, seed: int,
             max_tokens: int, max_conn: int, limit: int, eval_python: str) -> dict:
    out = HERE / f"osoi5_{arm}_gpqa_diamond.json"
    cmd = [eval_python, str(MMLU_GPQA_EVAL),
           "--task", "gpqa_diamond", "--arm", f"osoi5_{arm}",
           "--out", str(out), "--seed", str(seed),
           "--max-tokens", str(max_tokens), "--max-connections", str(max_conn),
           "--base-url", f"{base_url.rstrip('/')}/v1", "--model", model]
    if limit:
        cmd += ["--limit", str(limit)]
    rc = _stream(cmd)
    return {"axis": "gpqa_diamond", "rc": rc, "out": str(out)}


def run_gsm8k(base_url: str, model: str, arm: str, n: int, seed: int,
              concurrency: int, limit: int) -> dict:
    label = f"osoi5_{arm}"
    cmd = [sys.executable, str(GSM8K_EVAL),
           "--base-url", base_url, "--model", model,
           "--label", label, "--regimes", "sampled",
           "--n", str(n), "--seed", str(seed),
           "--concurrency", str(concurrency), "--save-text",
           "--out-dir", str(HERE)]
    if limit:
        cmd += ["--limit", str(limit)]
    rc = _stream(cmd, cwd=str(ROOT))
    return {"axis": "gsm8k", "rc": rc, "out": str(HERE / f"{label}_sampled.json")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["as_served", "floored"])
    ap.add_argument("--axes", default="mmlu_pro,gsm8k,gpqa_d",
                    help="comma list, run in this order; mmlu_pro first = PRIMARY")
    ap.add_argument("--limit", type=int, default=0, help="smoke cap per axis (0 = full)")
    ap.add_argument("--mmlu-n", type=int, default=500)
    ap.add_argument("--gsm8k-n", type=int, default=500)
    ap.add_argument("--seed-mmlu", type=int, default=12345)
    ap.add_argument("--seed-gsm8k", type=int, default=1234)
    ap.add_argument("--mmlu-max-tokens", type=int, default=2048)
    ap.add_argument("--gpqa-max-tokens", type=int, default=3072)
    ap.add_argument("--max-conn", type=int, default=16)
    ap.add_argument("--gsm8k-conc", type=int, default=32)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--startup-timeout-s", type=int, default=1800)
    ap.add_argument("--eval-python", default="/tmp/osoi5-eval-venv/bin/python")
    args = ap.parse_args()

    floor_on = args.arm == "floored"
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]

    from scripts.local_validation import harness, paths  # noqa: E402

    for note in paths.prepare_local_gpu_env():
        print(f"[env] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    # As-shipped osoi5 default env (LM_HEAD_PRUNE=1, baked body, drafter, onegraph)
    # + local-serve overrides (precache off; eval concurrency) + floor toggle.
    overrides = {
        "PRECACHE_BENCH": "0",
        "PRECACHE_REQUIRE": "0",
        "PRECACHE_DATASET": "/tmp/senpai_gsm8k_no_precache.json",
        "MAX_NUM_SEQS": str(args.max_num_seqs),
        # The ONLY knob: the #545 server-side floor. "" disables the patch entirely
        # (as-served); "8" arms the chat-endpoint floor (floored).
        "MIN_TOKENS_FLOOR": "8" if floor_on else "",
    }
    # ensure the no-precache stub exists
    pathlib.Path("/tmp/senpai_gsm8k_no_precache.json").write_text("[]")

    log_path = HERE / f"server_osoi5_{args.arm}.log"
    print(f"[serve] osoi5 AS-SHIPPED arm={args.arm} floor_on={floor_on} "
          f"overrides={overrides} log={log_path}", flush=True)

    arm_summary: dict = {
        "pr": 548,
        "arm": args.arm,
        "floor_on": floor_on,
        "analysis_only": True,
        "official_tps": 0,
        "limit": args.limit,
        "axes": axes,
        "overrides": overrides,
        "results": {},
    }

    t_start = time.time()
    with harness.LocalServer(
        SUBMISSION,
        server_python=server_python,
        port=args.port,
        startup_timeout_s=args.startup_timeout_s,
        log_path=log_path,
        extra_env=overrides,
    ) as srv:
        guard = run_guard(args.arm, floor_on, log_path)
        arm_summary["guard"] = guard
        arm_summary["peak_vram_gb"] = _gpu_mem_used_gb()
        model = srv.served_model_name

        for axis in axes:
            print(f"\n========== axis={axis} arm={args.arm} ==========", flush=True)
            if axis == "mmlu_pro":
                r = run_mmlu(srv.base_url, model, args.arm, args.mmlu_n, args.seed_mmlu,
                             args.mmlu_max_tokens, args.max_conn, args.limit, args.eval_python)
            elif axis == "gsm8k":
                r = run_gsm8k(srv.base_url, model, args.arm, args.gsm8k_n, args.seed_gsm8k,
                              args.gsm8k_conc, args.limit)
            elif axis == "gpqa_d":
                r = run_gpqa(srv.base_url, model, args.arm, args.seed_mmlu,
                             args.gpqa_max_tokens, args.max_conn, args.limit, args.eval_python)
            else:
                print(f"[warn] unknown axis {axis!r}, skipping", flush=True)
                continue
            arm_summary["results"][axis] = r
            arm_summary["peak_vram_gb"] = max(arm_summary["peak_vram_gb"], _gpu_mem_used_gb())
            # incremental save after each axis
            (HERE / f"arm_summary_{args.arm}.json").write_text(json.dumps(arm_summary, indent=2))

    arm_summary["wall_s"] = round(time.time() - t_start, 1)
    (HERE / f"arm_summary_{args.arm}.json").write_text(json.dumps(arm_summary, indent=2))
    print(f"\n[done] arm={args.arm} wall={arm_summary['wall_s']:.0f}s "
          f"peak_vram={arm_summary['peak_vram_gb']:.1f}GiB -> arm_summary_{args.arm}.json",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
