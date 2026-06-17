#!/usr/bin/env python3
"""PR #542 diagnostic — is the fresh-base degeneration a batch-width serve artifact?

base_fullhead is served at max_num_seqs=1 (MTP single-stream, batch-INVARIANT).
Plain base was served at max_num_seqs=16, where it CRATERS: item 10064 reasons
correctly for ~700 chars then collapses into a pure greedy repetition loop
("threetimesuparrowthreetimesuparrow..." to max_tokens) — classic batch-variant
int4 greedy degeneration. base_fullhead converges the SAME item ("ANSWER: I").

So fullhead@seqs1 vs base@seqs16 confounds the fast kernels WITH batch width.
The clean control the PR demands ("only the fast kernels move") is base served at
the SAME batch width as fullhead: max_num_seqs=1.

This re-serves plain base at seqs=1 and re-runs a FIXED paired subset of the
seqs=16 item set (exact ids via --ids-file, byte-identical prompts). We compare,
on identical ids:
  * accuracy  seqs1 vs seqs16
  * parsed / truncation rate (does the repetition-loop degeneration vanish at seqs1?)

If seqs1 recovers toward the documented anchor (MMLU 0.668 / GPQA 0.470) and the
loops disappear, the seqs=16 degeneration is a SERVE ARTIFACT and the non-confounded
fresh denominator is base@seqs1 (== anchor regime). If seqs1 stays ~0.43 and still
loops, vanilla base genuinely degenerates and base_fullhead legitimately clears it.

LOCAL, analysis_only, NO FIRE. conc=32 client pin held; only server batch width -> 1.

Usage:
  base_seqs1_diag.py [--mmlu-n 80] [--gpqa-n 60]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path("/workspace/senpai/target/research/validity/base_fullhead_shortchain_quality")
ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))

# Reuse run_2x2's EXACT serve helpers so the only change vs the seqs=16 base serve
# is max_num_seqs (1 instead of 16) — nothing else about the serve diverges.
_spec = importlib.util.spec_from_file_location("run_2x2", HERE / "run_2x2.py")
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)


def _first_ids(json_path: Path, k: int) -> list[str]:
    d = json.load(open(json_path))
    ids = [str(s["id"]) for s in d["per_sample"]]  # per_sample is sorted by id
    return ids[:k]


def _acc_on_ids(json_path: Path, ids: list[str]) -> dict:
    d = json.load(open(json_path))
    keep = set(ids)
    rows = [s for s in d["per_sample"] if str(s["id"]) in keep]
    n = len(rows)
    correct = sum(1 for s in rows if s.get("correct"))
    parsed = sum(1 for s in rows if s.get("answer") not in (None, ""))
    return {"n": n, "correct": correct, "parsed": parsed,
            "acc": (correct / n if n else float("nan")),
            "parsed_rate": (parsed / n if n else float("nan"))}


def _summ(d: dict) -> dict:
    rows = d["per_sample"]
    n = len(rows)
    correct = sum(1 for s in rows if s.get("correct"))
    parsed = sum(1 for s in rows if s.get("answer") not in (None, ""))
    return {"n": n, "correct": correct, "parsed": parsed,
            "acc": (correct / n if n else float("nan")),
            "parsed_rate": (parsed / n if n else float("nan"))}


def run_cell_ids(task: str, out: Path, ids_file: Path, max_tokens: int,
                 conc: int = 32, n: int = 500) -> dict:
    cmd = [
        str(r.EVAL_PY), str(r.QE / "run_eval.py"), "--task", task, "--arm", "base_seqs1",
        "--out", str(out), "--seed", str(r.SEED), "--max-tokens", str(max_tokens),
        "--max-connections", str(conc), "--base-url", f"http://127.0.0.1:{r.PORT}/v1",
        "--model", "gemma-4-e4b-it", "--ids-file", str(ids_file),
    ]
    if task == "mmlu_pro":
        cmd += ["--n", str(n)]
    print(f"[diag] {task} -> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    d = json.load(open(out))
    print(f"[diag] {task} acc={d['accuracy']:.4f} scored={d['n_scored']} "
          f"correct={d['n_correct']} err={d['n_error']}", flush=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu-n", type=int, default=80)
    ap.add_argument("--gpqa-n", type=int, default=60)
    ap.add_argument("--conc", type=int, default=32)
    args = ap.parse_args()

    for note in r.paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    mmlu_ids = _first_ids(HERE / "base_mmlu_pro.json", args.mmlu_n)
    gpqa_ids = _first_ids(HERE / "base_gpqa.json", args.gpqa_n)
    (HERE / "seqs1_mmlu_ids.json").write_text(json.dumps(mmlu_ids))
    (HERE / "seqs1_gpqa_ids.json").write_text(json.dumps(gpqa_ids))
    print(f"[diag] paired subset: MMLU={len(mmlu_ids)} ids, GPQA={len(gpqa_ids)} ids", flush=True)

    submission_dir = (ROOT / "submissions" / r.SUBMISSION).resolve()
    manifest = r.harness.load_manifest(submission_dir)
    server_python = r.harness.ensure_server_venv(manifest["dependencies"])
    print(f"[diag] server_python={server_python}", flush=True)

    base_url = f"http://127.0.0.1:{r.PORT}"
    r.wait_gpu_free()
    proc = r.start_base_server(server_python, HERE / "server_base_seqs1.log", max_num_seqs=1)
    try:
        r.wait_ready(base_url, proc)
        print("[diag] plain base @ seqs=1 ready", flush=True)
        dm = run_cell_ids("mmlu_pro", HERE / "base_seqs1_mmlu.json",
                          HERE / "seqs1_mmlu_ids.json", max_tokens=2048, conc=args.conc)
        dg = run_cell_ids("gpqa_diamond", HERE / "base_seqs1_gpqa.json",
                          HERE / "seqs1_gpqa_ids.json", max_tokens=3072, conc=args.conc)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except Exception:
            pass

    # ---- paired comparison: seqs1 vs seqs16 on the SAME ids ----
    s16_m = _acc_on_ids(HERE / "base_mmlu_pro.json", mmlu_ids)
    s16_g = _acc_on_ids(HERE / "base_gpqa.json", gpqa_ids)
    s1_m = _summ(dm)
    s1_g = _summ(dg)

    report = {
        "pr": 542, "analysis_only": True, "official_tps": 0,
        "subset": {"mmlu_n": len(mmlu_ids), "gpqa_n": len(gpqa_ids)},
        "mmlu_pro": {"seqs16": s16_m, "seqs1": s1_m,
                     "acc_delta": s1_m["acc"] - s16_m["acc"],
                     "parsed_rate_delta": s1_m["parsed_rate"] - s16_m["parsed_rate"]},
        "gpqa_diamond": {"seqs16": s16_g, "seqs1": s1_g,
                         "acc_delta": s1_g["acc"] - s16_g["acc"],
                         "parsed_rate_delta": s1_g["parsed_rate"] - s16_g["parsed_rate"]},
        "anchor": {"mmlu": 0.668, "gpqa": 0.470},
    }
    (HERE / "base_seqs1_diag.json").write_text(json.dumps(report, indent=2))

    def line(tag, b):
        return (f"  {tag:13s} seqs16 acc={b['seqs16']['acc']:.4f} parsed={b['seqs16']['parsed_rate']:.3f}"
                f"  ->  seqs1 acc={b['seqs1']['acc']:.4f} parsed={b['seqs1']['parsed_rate']:.3f}"
                f"   (dAcc={b['acc_delta']:+.4f} dParsed={b['parsed_rate_delta']:+.3f})")

    print("\n==== base@seqs1 vs base@seqs16 (paired, identical ids) ====")
    print(line("mmlu_pro", report["mmlu_pro"]))
    print(line("gpqa_diamond", report["gpqa_diamond"]))
    print("  anchor: MMLU 0.668 / GPQA 0.470")
    print("DIAG-MARKER:", json.dumps({
        "mmlu_seqs16_acc": s16_m["acc"], "mmlu_seqs1_acc": s1_m["acc"],
        "mmlu_seqs16_parsed": s16_m["parsed_rate"], "mmlu_seqs1_parsed": s1_m["parsed_rate"],
        "gpqa_seqs16_acc": s16_g["acc"], "gpqa_seqs1_acc": s1_g["acc"],
        "gpqa_seqs16_parsed": s16_g["parsed_rate"], "gpqa_seqs1_parsed": s1_g["parsed_rate"],
    }))
    print("[diag] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
