#!/usr/bin/env python
"""Local tree-submission preflight: the scorer's THREE hard validity gates.

We get ONE human-approved official shot at the tree submission. A brand-new
served stack (tree-emit drafter + descending accept-walk + M=32 verify +
tree-mask attention) can FAIL the scorer in ways unrelated to TPS:

  * (A) engine-init crash      — boots/serves at all? (#141-class catch)
  * (B) PPL > 2.42             — greedy-exactness regression in the accept-walk
  * (C) < 128/128 completion   — a hang/OOM on some prompt

This harness validates the fully-assembled submission against those three gates
*locally* (A10G + CPU, no HF Job, no leaderboard spend) and emits a single
**READY / NOT-READY** verdict naming the failing gate(s) — so a wasted shot is
caught BEFORE a human is asked to authorize it.

Fidelity: it mirrors ``hf_bucket_single_job.py`` by serving the submission with
the same participant venv + manifest env + serve command
(``scripts.local_validation.harness.LocalServer``) and then driving the
*official* ``ppl_endpoint.py`` / ``decode_outputs.py`` against the live endpoint.
The PPL definition and the 128-prompt / output_len-512 protocol are NOT
reimplemented — the scorer's own scripts and datasets compute them.

Scope: VALIDITY (will it score), NOT TPS. A READY verdict is the validity leg of
an ``Approval request: HF job`` evidence-line; it does NOT authorize the spend.

Usage (drop-in for any submission dir, e.g. the tree stack the instant it lands):

    .venv/bin/python research/tree_submission_preflight/preflight.py \\
        --submission submissions/<tree-dir> \\
        --server-python /tmp/server-venv/bin/python \\
        --wandb-group tree-submission-preflight

Self-validation (PRIMARY deliverable until the tree stack lands):

    .venv/bin/python research/tree_submission_preflight/preflight.py --self-test \\
        --submission submissions/fa2sw_precache_kenyan \\
        --server-python /tmp/server-venv/bin/python \\
        --wandb-group tree-submission-preflight
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Repo root (research/tree_submission_preflight/preflight.py -> three parents up)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
from scripts.local_validation.ppl_runner import _headroom_overrides  # noqa: E402

DEFAULT_SERVER_PYTHON = Path("/tmp/server-venv/bin/python")
DEFAULT_PPL_CAP = 2.42
A10G_CEILING_MIB = 23028  # the official a10g-small / local A10G VRAM ceiling
GPU_INDEX = 0  # single in-container GPU (see project_local_a10g_gpu_env memory)


# --------------------------------------------------------------------------- #
# Verdict logic (pure function — unit-checkable in the self-test)
# --------------------------------------------------------------------------- #
def evaluate_verdict(gate_a: dict, gate_b: dict, gate_c: dict) -> dict:
    """READY iff A AND B AND C all passed; else NOT-READY naming the failing gates.

    Each gate dict carries ``{"name", "passed", "detail", ...}``. A gate whose
    ``passed`` is not exactly ``True`` (False, or ``None`` for "did not run") is
    treated as failing — fail-closed, so an unmeasured gate never rubber-stamps a
    launch.
    """
    gates = {"A": gate_a, "B": gate_b, "C": gate_c}
    failing = [k for k, g in gates.items() if g.get("passed") is not True]
    ready = not failing
    return {
        "verdict": "READY" if ready else "NOT-READY",
        "ready": ready,
        "failing_gates": failing,
        "gate_a_passed": gate_a.get("passed"),
        "gate_b_passed": gate_b.get("passed"),
        "gate_c_passed": gate_c.get("passed"),
    }


# --------------------------------------------------------------------------- #
# GPU memory peak sampler
# --------------------------------------------------------------------------- #
class GpuMemSampler:
    """Background sampler of ``nvidia-smi`` ``memory.used`` (MiB) for a peak."""

    def __init__(self, index: int = GPU_INDEX, interval_s: float = 1.0) -> None:
        self.index = index
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> int | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.index),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        try:
            return int(out.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            mib = self._sample_once()
            if mib is not None and mib > self.peak_mib:
                self.peak_mib = mib
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuMemSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# Gate A — boot/serve smoke
# --------------------------------------------------------------------------- #
def _post_completion(base_url: str, payload: dict, timeout_s: int = 120) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def smoke_decode(
    base_url: str,
    model: str,
    *,
    num_prompts: int = 3,
    max_tokens: int = 8,
) -> dict:
    """Tiny decode + finite-logprob probe over the live endpoint.

    Two endpoint-observable failure modes are checked:

      * tokens emitted — a few short greedy decodes must each return >=1
        completion token (the engine actually produces output);
      * finite logits — a ``prompt_logprobs=1`` request on a short integer prompt
        must return *finite* prompt-token logprobs. NaN/Inf logits (the silent
        corruption an accept-walk bug or a bad fused kernel can produce) survive
        argmax but show up as non-finite log-softmax here. This is the cheap
        local analogue of "no NaN/Inf logits".
    """
    detail: dict[str, Any] = {"tokens_emitted": [], "errors": []}

    # 1) Greedy decodes emit tokens.
    prompts = [
        "Explain how a transformer decodes one token at a time.",
        "List three prime numbers.",
        "Translate 'good morning' into French.",
    ][:num_prompts]
    emitted_ok = True
    for text in prompts:
        try:
            resp = _post_completion(
                base_url,
                {
                    "model": model,
                    "prompt": text,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": False,
                    "ignore_eos": True,
                    "return_token_ids": True,
                },
            )
            choice = (resp.get("choices") or [{}])[0]
            tok = choice.get("token_ids")
            usage = resp.get("usage") or {}
            n = (
                len(tok)
                if isinstance(tok, list)
                else (usage.get("completion_tokens") or 0)
            )
            detail["tokens_emitted"].append(n)
            if not n:
                emitted_ok = False
        except (urllib.error.URLError, OSError, ValueError) as exc:
            emitted_ok = False
            detail["errors"].append(f"decode: {exc}")

    # 2) Finite prompt-token logprobs (no NaN/Inf logits).
    finite_ok = True
    try:
        # A short, valid integer-token prompt (BOS + a few common ids).
        probe_tokens = [2, 105, 2364, 107, 7925, 506]
        resp = _post_completion(
            base_url,
            {
                "model": model,
                "prompt": probe_tokens,
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
                "prompt_logprobs": 1,
                "add_special_tokens": False,
                "return_token_ids": True,
            },
        )
        choice = (resp.get("choices") or [{}])[0]
        plp = choice.get("prompt_logprobs") or resp.get("prompt_logprobs")
        checked = 0
        if not isinstance(plp, list):
            finite_ok = False
            detail["errors"].append("prompt_logprobs missing from smoke response")
        else:
            for entry in plp:
                if entry is None:
                    continue
                if isinstance(entry, dict):
                    for value in entry.values():
                        lp = value.get("logprob") if isinstance(value, dict) else value
                        if lp is None:
                            continue
                        checked += 1
                        if not _finite(lp):
                            finite_ok = False
                            detail["errors"].append(f"non-finite logprob: {lp}")
        detail["logprobs_checked"] = checked
        if checked == 0:
            finite_ok = False
            detail["errors"].append("no prompt logprobs available to check finiteness")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        finite_ok = False
        detail["errors"].append(f"logprob probe: {exc}")

    detail["tokens_emitted_ok"] = emitted_ok
    detail["finite_logprobs_ok"] = finite_ok
    detail["passed"] = bool(emitted_ok and finite_ok)
    return detail


# --------------------------------------------------------------------------- #
# Gate B — PPL on the scorer's exact convention
# --------------------------------------------------------------------------- #
def run_gate_b(
    server_python: Path,
    base_url: str,
    model: str,
    *,
    out_dir: Path,
    cap: float,
    dataset: Path | None = None,
    tag: str = "ppl",
) -> dict:
    """Run the official ppl_endpoint.py and assert PPL <= cap (report the margin)."""
    summary = harness.run_ppl(
        server_python,
        base_url=base_url,
        model=model,
        out_file=out_dir / f"{tag}_results.jsonl",
        summary_file=out_dir / f"{tag}_summary.json",
        dataset=dataset,
    )
    ppl = summary["ppl"]
    passed = _finite(ppl) and ppl <= cap
    return {
        "name": "B_ppl",
        "passed": bool(passed),
        "ppl": ppl,
        "cap": cap,
        "margin": (cap - ppl) if _finite(ppl) else None,
        "num_tokens": summary.get("num_tokens"),
        "num_records": summary.get("num_records"),
        "summary_file": str(out_dir / f"{tag}_summary.json"),
        "detail": f"PPL={ppl:.4f} <= {cap} margin={cap - ppl:+.4f}"
        if _finite(ppl)
        else f"PPL non-finite ({ppl})",
    }


# --------------------------------------------------------------------------- #
# Gate C — 128/128 completion at output_len 512
# --------------------------------------------------------------------------- #
def run_gate_c(
    server_python: Path,
    base_url: str,
    model: str,
    *,
    out_dir: Path,
    num_prompts: int,
    output_len: int,
) -> dict:
    """Run the official decode_outputs.py over all prompts; assert full completion.

    Passes iff every prompt produced a record AND every record decoded the full
    ``output_len`` tokens (``ignore_eos: true`` forces exactly output_len per
    prompt, so num_completion_tokens == num_prompts * output_len when nothing
    hangs / OOMs / short-circuits). Peak GPU memory is sampled during the run and
    reported against the A10G ceiling.
    """
    subset = num_prompts < paths.NUM_PROMPTS
    with GpuMemSampler() as mem:
        summary = harness.capture_decode(
            server_python,
            base_url=base_url,
            model=model,
            out_file=out_dir / "decode_outputs.jsonl",
            summary_file=out_dir / "decode_summary.json",
            num_prompts=num_prompts,
            output_len=output_len,
        )
    records = summary.get("num_records", 0)
    completion_tokens = summary.get("num_completion_tokens", 0)
    expected_tokens = num_prompts * output_len
    records_ok = records == num_prompts
    tokens_ok = completion_tokens == expected_tokens
    passed = records_ok and tokens_ok
    peak_mib = mem.peak_mib
    return {
        "name": "C_completion",
        "passed": bool(passed),
        "completed": records,
        "num_prompts": num_prompts,
        "num_completion_tokens": completion_tokens,
        "expected_completion_tokens": expected_tokens,
        "output_len": output_len,
        "subset": subset,
        "peak_gpu_mem_mib": peak_mib,
        "a10g_ceiling_mib": A10G_CEILING_MIB,
        "peak_gpu_mem_frac": round(peak_mib / A10G_CEILING_MIB, 4) if peak_mib else None,
        "duration_s": summary.get("duration_s"),
        "summary_file": str(out_dir / "decode_summary.json"),
        "detail": (
            f"{records}/{num_prompts} prompts, {completion_tokens}/{expected_tokens} tokens"
            + (
                f"; SUBSET (full protocol = {paths.NUM_PROMPTS}); extrapolation: "
                "a clean subset is necessary-not-sufficient — re-run full 128 before launch"
                if subset
                else ""
            )
            + (f"; peak GPU {peak_mib} MiB / {A10G_CEILING_MIB} MiB" if peak_mib else "")
        ),
    }


# --------------------------------------------------------------------------- #
# Single preflight run (one submission, optional injected boot fault)
# --------------------------------------------------------------------------- #
def run_preflight(
    submission: Path,
    server_python: Path,
    *,
    out_dir: Path,
    num_prompts: int,
    output_len: int,
    cap: float,
    port: int,
    inject_boot_fault: bool = False,
    run_gates_bc: bool = True,
) -> dict:
    """Serve the submission and run gates A/B/C; return the full result dict.

    If ``inject_boot_fault`` is set, a wrong ``DRAFTER_SHA256`` is injected so
    serve.py's ``ensure_drafter()`` raises before the engine starts — exercising
    the #141-class "server exited before readiness" catch end-to-end.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = harness.load_manifest(submission)
    overrides = _headroom_overrides(manifest.get("env", {}))
    if inject_boot_fault:
        overrides = {**overrides, "DRAFTER_SHA256": "0" * 64}

    result: dict[str, Any] = {
        "submission": str(submission),
        "submission_name": submission.name,
        "num_prompts": num_prompts,
        "output_len": output_len,
        "ppl_cap": cap,
        "inject_boot_fault": inject_boot_fault,
        "out_dir": str(out_dir),
    }

    gate_a = {"name": "A_boot", "passed": None, "detail": "not run"}
    gate_b = {"name": "B_ppl", "passed": None, "detail": "not run"}
    gate_c = {"name": "C_completion", "passed": None, "detail": "not run"}
    log_path = out_dir / "server.log"

    try:
        with harness.LocalServer(
            submission,
            server_python=server_python,
            port=port,
            log_path=log_path,
            extra_env=overrides,
        ) as srv:
            result["model_id"] = srv.model_id
            result["served_model_name"] = srv.served_model_name
            # Gate A: boot succeeded (we are inside the context) + smoke decode.
            smoke = smoke_decode(srv.base_url, srv.served_model_name)
            gate_a = {
                "name": "A_boot",
                "passed": bool(smoke["passed"]),
                "boot_ok": True,
                "smoke": smoke,
                "detail": "boots+serves; smoke "
                + ("OK" if smoke["passed"] else f"FAILED: {smoke['errors']}"),
            }
            if run_gates_bc:
                gate_b = run_gate_b(
                    server_python, srv.base_url, srv.served_model_name,
                    out_dir=out_dir, cap=cap,
                )
                gate_c = run_gate_c(
                    server_python, srv.base_url, srv.served_model_name,
                    out_dir=out_dir, num_prompts=num_prompts, output_len=output_len,
                )
    except Exception as exc:  # boot failure / engine-init crash -> Gate A fails
        tail = ""
        if log_path.exists():
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-25:])
        gate_a = {
            "name": "A_boot",
            "passed": False,
            "boot_ok": False,
            "failure_site": str(exc),
            "server_log_tail": tail,
            "detail": f"engine-init crash (#141-class): {exc}",
        }

    result["gate_a"] = gate_a
    result["gate_b"] = gate_b
    result["gate_c"] = gate_c
    result.update(evaluate_verdict(gate_a, gate_b, gate_c))
    return result


# --------------------------------------------------------------------------- #
# Self-test: known-good READY + injected faults NOT-READY (naming the gate)
# --------------------------------------------------------------------------- #
def run_self_test(
    submission: Path,
    server_python: Path,
    *,
    out_root: Path,
    num_prompts: int,
    output_len: int,
    cap: float,
    port: int,
) -> dict:
    """Certify the harness PASSES a known-good stack AND CATCHES injected faults.

    Conditions for ``harness_self_test_passes`` (all must hold):
      1. known-good linear stack  -> READY (A & B & C pass) [end-to-end, full 128/128]
      2. injected boot fault       -> NOT-READY, Gate A catches it [end-to-end]
      3. injected over-cap PPL     -> NOT-READY naming gate B [verdict-logic check]
      4. under-count completion    -> NOT-READY naming gate C [verdict-logic check]

    Gates B and C are certified with verdict-logic checks (a synthetic over-cap
    PPL / under-count completion fed through ``evaluate_verdict``), NOT by feeding
    corrupted inputs to the official scorer. The end-to-end MEASUREMENT path for
    both is already exercised by the known-good arm (a real PPL + a real 128/128).
    A corrupted-ground-truth token swap was dropped on purpose: the official,
    read-only ``ppl_endpoint.py`` calls an unguarded ``math.exp`` that OVERFLOWS
    on adversarial tokens (per-record mean NLL/token > 709) — that is a scorer
    crash, not a clean over-cap PPL, and the failing subprocess masqueraded as a
    boot (Gate A) failure. The verdict-logic checks isolate exactly the gate
    behavior we must trust (threshold + fail-closed) and can never flake.
    """
    checks: dict[str, Any] = {}

    # 1) Known-good linear stack: full A/B/C on a live server -> must be READY.
    good_dir = out_root / "known_good"
    good = run_preflight(
        submission, server_python, out_dir=good_dir,
        num_prompts=num_prompts, output_len=output_len, cap=cap, port=port,
    )
    checks["known_good"] = good

    # 2) Injected boot fault — separate (fast-failing) boot with a wrong drafter
    #    sha so serve.py's ensure_drafter() raises before the engine starts.
    #    Success = Gate A catches it AND the cause is the INJECTED sha mismatch
    #    (not an unrelated flake). B/C are intentionally skipped, so they stay
    #    fail-closed in failing_gates — correct behavior — hence we assert "A
    #    failed and is the named cause", not strict failing_gates == ["A"].
    boot_fault_dir = out_root / "injected_boot_fault"
    boot_fault = run_preflight(
        submission, server_python, out_dir=boot_fault_dir,
        num_prompts=num_prompts, output_len=output_len, cap=cap, port=port,
        inject_boot_fault=True, run_gates_bc=False,
    )
    ga = boot_fault.get("gate_a", {})
    boot_fault["names_gate_a"] = bool(
        ga.get("passed") is False
        and boot_fault.get("verdict") == "NOT-READY"
        and "A" in boot_fault.get("failing_gates", [])
        and "DRAFTER_SHA256 mismatch" in (ga.get("server_log_tail") or "")
    )
    checks["injected_boot_fault"] = boot_fault

    # 3) Injected over-cap PPL (verdict-logic): a measured PPL above the cap must
    #    trip Gate B and name B. Mirrors the Gate-C check below.
    over_cap_ppl = round(cap + 0.5, 4)
    ppl_fault_b = {
        "name": "B_ppl", "passed": False, "ppl": over_cap_ppl, "cap": cap,
        "margin": cap - over_cap_ppl,
        "detail": f"synthetic over-cap PPL={over_cap_ppl} > cap {cap} -> Gate B must fail",
    }
    ppl_fault = {
        "gate_b": ppl_fault_b,
        **evaluate_verdict({"passed": True}, ppl_fault_b, {"passed": True}),
    }
    ppl_fault["names_gate_b"] = ppl_fault["failing_gates"] == ["B"]
    checks["injected_ppl_fault"] = ppl_fault

    # 4) Under-count completion (verdict-logic): < 128/128 must trip Gate C / name C.
    logic_c = evaluate_verdict(
        {"passed": True},
        {"passed": True},
        {"name": "C_completion", "passed": False, "completed": 120, "num_prompts": 128},
    )
    checks["completion_logic_check"] = {
        **logic_c,
        "names_gate_c": logic_c["failing_gates"] == ["C"],
    }

    cond_good = good.get("verdict") == "READY"
    cond_boot = bool(boot_fault.get("names_gate_a"))
    cond_ppl = ppl_fault.get("verdict") == "NOT-READY" and ppl_fault.get("names_gate_b")
    cond_comp = logic_c["verdict"] == "NOT-READY" and checks["completion_logic_check"]["names_gate_c"]
    passes = bool(cond_good and cond_boot and cond_ppl and cond_comp)

    return {
        "harness_self_test_passes": passes,
        "conditions": {
            "known_good_READY": cond_good,
            "boot_fault_NOT_READY_names_A": cond_boot,
            "ppl_fault_NOT_READY_names_B": cond_ppl,
            "completion_logic_NOT_READY_names_C": cond_comp,
        },
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# W&B logging
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # logging must never break the gate
        print(f"[preflight] wandb logging unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="tree-submission-preflight",
        agent="senpai",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["tree-submission-preflight", "validity-gate"],
        config={
            "submission": str(args.submission),
            "num_prompts": args.num_prompts,
            "output_len": args.output_len,
            "ppl_cap": args.ppl_cap,
            "self_test": args.self_test,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[preflight] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return
    summary: dict[str, Any] = {}
    if args.self_test:
        st = payload["self_test"]
        summary["harness_self_test_passes"] = int(bool(st["harness_self_test_passes"]))
        for key, value in st["conditions"].items():
            summary[f"selftest_{key}"] = int(bool(value))
        good = st["checks"]["known_good"]
        gb = good.get("gate_b", {})
        gc = good.get("gate_c", {})
        if _finite(gb.get("ppl")):
            summary["known_good_ppl"] = gb["ppl"]
            summary["known_good_ppl_margin"] = gb.get("margin")
        if isinstance(gc.get("completed"), int):
            summary["known_good_completed"] = gc["completed"]
        if isinstance(gc.get("peak_gpu_mem_mib"), int) and gc["peak_gpu_mem_mib"]:
            summary["known_good_peak_gpu_mem_mib"] = gc["peak_gpu_mem_mib"]
        summary["known_good_ready"] = int(good.get("verdict") == "READY")
    if "preflight" in payload:
        pf = payload["preflight"]
        summary["live_preflight_ready"] = int(pf.get("verdict") == "READY")
        gb, gc = pf.get("gate_b", {}), pf.get("gate_c", {})
        if _finite(gb.get("ppl")):
            summary["live_ppl"] = gb["ppl"]
            summary["live_ppl_margin"] = gb.get("margin")
        if isinstance(gc.get("completed"), int):
            summary["live_completed"] = gc["completed"]
        if isinstance(gc.get("peak_gpu_mem_mib"), int) and gc["peak_gpu_mem_mib"]:
            summary["live_peak_gpu_mem_mib"] = gc["peak_gpu_mem_mib"]
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="tree_preflight_result", artifact_type="preflight", data=payload)
    finish_wandb(run)
    print(f"[preflight] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_verdict(label: str, result: dict) -> None:
    line = "=" * 14 + f" {label} " + "=" * 14
    print("\n" + line, flush=True)
    print(f"submission: {result.get('submission')}", flush=True)
    for key in ("gate_a", "gate_b", "gate_c"):
        g = result.get(key, {})
        mark = {True: "PASS", False: "FAIL", None: "----"}.get(g.get("passed"), "?")
        print(f"  [{mark}] {g.get('name', key)}: {g.get('detail', '')}", flush=True)
    print(f"VERDICT: {result.get('verdict')}"
          + (f"  (failing: {result.get('failing_gates')})" if result.get("failing_gates") else ""),
          flush=True)
    print("=" * len(line), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", type=Path, required=True, help="submission dir (drop-in for the tree stack)")
    ap.add_argument("--server-python", type=Path, default=DEFAULT_SERVER_PYTHON,
                    help="python with the pinned vLLM wheel (default: /tmp/server-venv/bin/python)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS,
                    help="Gate C prompt count (default 128; < 128 is a documented subset)")
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--ppl-cap", type=float, default=DEFAULT_PPL_CAP)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="run the PRIMARY self-validation (known-good READY + injected faults NOT-READY)")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="tree-submission-preflight")
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[preflight] {note}", flush=True)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "runs" / f"{args.submission.name}-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "created_at": stamp,
        "submission": str(args.submission),
        "server_python": str(args.server_python),
        "ppl_cap": args.ppl_cap,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
    }

    exit_code = 0
    if args.self_test:
        st = run_self_test(
            args.submission, args.server_python, out_root=out_dir,
            num_prompts=args.num_prompts, output_len=args.output_len,
            cap=args.ppl_cap, port=args.port,
        )
        payload["self_test"] = st
        _print_verdict("KNOWN-GOOD (must be READY)", st["checks"]["known_good"])
        _print_verdict("INJECTED BOOT FAULT (must be NOT-READY / gate A)", st["checks"]["injected_boot_fault"])
        print("\n-- injected PPL token-swap fault (must be NOT-READY / gate B) --", flush=True)
        pf = st["checks"]["injected_ppl_fault"]
        print(f"  verdict={pf.get('verdict')} failing={pf.get('failing_gates')} "
              f"ppl={pf.get('gate_b', {}).get('ppl')}", flush=True)
        print("\n-- completion verdict-logic check (must be NOT-READY / gate C) --", flush=True)
        print(f"  {st['checks']['completion_logic_check']}", flush=True)
        print(f"\nharness_self_test_passes = {st['harness_self_test_passes']}", flush=True)
        print(f"conditions: {json.dumps(st['conditions'])}", flush=True)
        if not st["harness_self_test_passes"]:
            exit_code = 1
    else:
        pf = run_preflight(
            args.submission, args.server_python, out_dir=out_dir,
            num_prompts=args.num_prompts, output_len=args.output_len,
            cap=args.ppl_cap, port=args.port,
        )
        payload["preflight"] = pf
        _print_verdict("TREE-SUBMISSION PREFLIGHT", pf)
        if pf.get("verdict") != "READY":
            exit_code = 1

    result_file = out_dir / "preflight_result.json"
    result_file.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n[preflight] result -> {result_file}", flush=True)
    _maybe_log_wandb(args, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
