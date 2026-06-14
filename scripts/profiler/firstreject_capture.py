#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Serve-faithful per-step MTP first-reject capture (PR #89 capture step).

Records, on the DEPLOYED spec-decode path (onegraph drafter unchanged), the MTP
chain accept length ``fd`` (== m; ``fd == 0`` is the first-reject / m=0 miss) at every
decode step, plus the exact emitted tokens so the offline aligner can pin every step
to an absolute generation position. This is the GPU half of PR #89; the CPU
intersection with prompt-lookup hits lives in
``scripts/analyze_prompt_lookup.py --overlap-*``.

Why a FRESH aligned pass (not reuse of #81/#79): #81's prompt-lookup q and #79's
first-reject fd came from SEPARATE runs and carry no shared position key (#79's
records have only a global step counter, no prompt/pos), so they cannot be
intersected by position. This single pass emits BOTH the greedy completion
(``decode_outputs.jsonl``, used to recompute q) AND the position-aligned fd-stream
from the SAME decode, making the intersection exact.

Greedy-identity self-check: the concatenated emit-stream MUST equal the greedy
completion tokens (the validity-contract trace), which the aligner asserts.

LOCAL ONLY. Single assigned GPU. No HF Job, no submission launch, no served-file
change (scratch copy only).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

DEFAULT_SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
OUT_DIR = ROOT / "research" / "local_validation" / "prompt_lookup" / "firstreject_capture"
PATCH_SRC = Path(__file__).resolve().parent / "firstreject_patch.py"

_HOOK_MARKER = "# --- first-reject capture probe (PR #89, scratch only) ---"


def build_scratch(submission: Path, scratch: Path) -> Path:
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(
        submission, scratch,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(PATCH_SRC, scratch / "firstreject_patch.py")
    sc = scratch / "sitecustomize.py"
    text = sc.read_text()
    if _HOOK_MARKER not in text:
        text += (
            f"\n\n{_HOOK_MARKER}\n"
            "import os as _fr_os  # noqa: E402\n"
            "if _fr_os.environ.get('FRPROBE_ENABLE') == '1':\n"
            "    try:\n"
            "        import firstreject_patch  # noqa: E402,F401\n"
            "    except Exception as _fr_exc:  # noqa: BLE001\n"
            "        import sys as _fr_sys\n"
            "        print(f'[frprobe] import failed: {_fr_exc!r}', file=_fr_sys.stderr, flush=True)\n"
        )
        sc.write_text(text)
    return scratch


def run_capture(scratch: Path, *, num_prompts: int, output_len: int, seed: int,
                records_path: Path, decode_out: Path, decode_summary: Path,
                log_path: Path) -> dict[str, Any]:
    manifest = harness.load_manifest(scratch)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    extra_env = {
        "FRPROBE_ENABLE": "1",
        "FRPROBE_OUTPUT": str(records_path),
        # native sampler (cuRAND JIT dodge) + re-enable stat loggers for an
        # independent E[T]/acceptance read in the same log. None change tokens.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "DISABLE_LOG_STATS": "0",
    }
    report: dict[str, Any] = {
        "submission": str(scratch),
        "num_prompts": num_prompts, "output_len": output_len, "seed": seed, "conc": 1,
        "spec": "ON (deployed)",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    t0 = time.time()
    with harness.LocalServer(
        scratch, server_python=server_python, port=8000, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        summary = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_out, summary_file=decode_summary,
            num_prompts=num_prompts, output_len=output_len, seed=seed, timeout_s=7200,
        )
        report["decode_summary"] = summary
    report["decode_wall_s"] = time.time() - t0
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--debug", action="store_true",
                    help="tiny 2-prompt/64-token smoke run to validate the harness")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_debug" if args.debug else ""
    records_path = args.out_dir / f"firstreject_records{tag}.jsonl"
    decode_out = args.out_dir / f"decode_outputs{tag}.jsonl"
    decode_summary = args.out_dir / f"decode_outputs{tag}.summary.json"
    log_path = args.out_dir / f"server_firstreject{tag}.log"

    for note in paths.prepare_local_gpu_env():
        print(f"[frcap] {note}", flush=True)
    num_prompts = 2 if args.debug else args.num_prompts
    output_len = 64 if args.debug else args.output_len
    scratch = args.out_dir / "_scratch_submission"
    build_scratch(args.submission.resolve(), scratch)
    print(f"[frcap] scratch submission at {scratch}", flush=True)

    report = run_capture(
        scratch, num_prompts=num_prompts, output_len=output_len, seed=args.seed,
        records_path=records_path, decode_out=decode_out, decode_summary=decode_summary,
        log_path=log_path,
    )
    out_json = args.out_dir / f"firstreject_capture_report{tag}.json"
    out_json.write_text(json.dumps(report, indent=2))

    print("\n========== FIRST-REJECT CAPTURE ==========", flush=True)
    print(f"records glob   : {records_path}.*", flush=True)
    print(f"decode out     : {decode_out}", flush=True)
    ds = report.get("decode_summary", {})
    print(f"decode summary : completed={ds.get('completed')} tps={ds.get('tps')}", flush=True)
    print(f"decode wall_s  : {report['decode_wall_s']:.1f}", flush=True)
    print(f"report         : {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
