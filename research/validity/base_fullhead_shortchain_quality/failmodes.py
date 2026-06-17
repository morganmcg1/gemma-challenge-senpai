#!/usr/bin/env python3
"""PR #542 — classify the failure mode of every zeroed item, per arm/axis.

The advisor's in-flight steer (the wirbel #541 first-token-EOS artifact) asks for
the per-arm immediate-EOS empty rate so we can tell whether base_fullhead's gate
is being depressed by a *recoverable serving artifact* (immediate-EOS empties,
fixable by request-level min_tokens) versus a real failure (max_tokens truncation
of a non-converging CoT, which min_tokens cannot touch).

For each run_eval.py output JSON we re-open the inspect `.eval` log it points at
(its `eval_log` field) and join each sample's stop_reason + completion length with
the scored answer. Items with NO parseable letter (`answer in {"", None}`) are the
"zeroed" items; we bucket them:

  - empty_eos   : completion is empty/whitespace -> immediate-EOS empty. The ONLY
                  bucket min_tokens=8 can recover (it masks EOS until N tokens).
  - truncation  : stop_reason in {max_tokens, model_length, length} -> ran out of
                  token budget before emitting "ANSWER:". min_tokens is a no-op here.
  - other_fail  : stopped normally with content but no parseable letter.

`empty_eos` is the recovery target set for the min_tokens=8 re-run.

Usage:
  failmodes.py --dir <here>          # classify the 4 standard as-served cells
  failmodes.py --json <out.json>     # classify a single cell
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_PY = Path("/tmp/eval-serve-venv/bin/python")
TRUNC_STOPS = {"max_tokens", "model_length", "length"}


def _read_eval_samples(eval_log_path: str) -> dict:
    """Return {id: {comp_len, comp_stripped_len, stop_reason}} from a .eval log.

    Runs under the eval venv (which has inspect_ai) via a subprocess so this module
    stays import-light and usable from any interpreter.
    """
    import subprocess

    code = (
        "import json,sys\n"
        "from inspect_ai.log import read_eval_log\n"
        "log=read_eval_log(sys.argv[1])\n"
        "out={}\n"
        "for s in (log.samples or []):\n"
        "    o=s.output\n"
        "    comp=(o.completion or '') if o else ''\n"
        "    out[str(s.id)]={'comp_len':len(comp),'comp_stripped_len':len(comp.strip()),"
        "'stop_reason':getattr(o,'stop_reason',None)}\n"
        "json.dump(out,sys.stdout)\n"
    )
    res = subprocess.run([str(EVAL_PY), "-c", code, eval_log_path],
                         check=True, text=True, capture_output=True)
    return json.loads(res.stdout)


def classify_eval(json_path: Path) -> dict:
    d = json.load(open(json_path))
    eval_log = d.get("eval_log")
    if not eval_log or not Path(eval_log).exists():
        raise SystemExit(f"[failmodes] eval_log missing for {json_path}: {eval_log}")
    by_id = _read_eval_samples(eval_log)

    empty_eos, truncation, other_fail, parsed, errored = [], [], [], [], []
    for r in d["per_sample"]:
        sid = str(r["id"])
        if r.get("error"):
            errored.append(sid)
            continue
        ans = r.get("answer")
        if ans not in (None, ""):
            parsed.append(sid)
            continue
        info = by_id.get(sid, {})
        slen = info.get("comp_stripped_len")
        stop = info.get("stop_reason")
        if slen == 0:
            empty_eos.append(sid)
        elif stop in TRUNC_STOPS:
            truncation.append(sid)
        else:
            other_fail.append(sid)

    n = len(d["per_sample"])
    nz = max(1, n)
    return {
        "task": d["task"], "arm": d["arm"], "eval_log": eval_log,
        "n": n, "n_scored": d["n_scored"], "accuracy": d["accuracy"],
        "n_parsed": len(parsed), "n_error": len(errored),
        "n_extract_fail": len(empty_eos) + len(truncation) + len(other_fail),
        "n_empty_eos": len(empty_eos),
        "n_truncation": len(truncation),
        "n_other_fail": len(other_fail),
        "empty_eos_rate": len(empty_eos) / nz,
        "truncation_rate": len(truncation) / nz,
        "extract_fail_rate": (len(empty_eos) + len(truncation) + len(other_fail)) / nz,
        "empty_eos_ids": sorted(empty_eos, key=lambda x: str(x)),
        "truncation_ids": sorted(truncation, key=lambda x: str(x)),
        "other_fail_ids": sorted(other_fail, key=lambda x: str(x)),
    }


STANDARD = {
    "fullhead_mmlu_pro.json": ("fullhead", "mmlu_pro"),
    "base_mmlu_pro.json": ("base", "mmlu_pro"),
    "fullhead_gpqa.json": ("fullhead", "gpqa_diamond"),
    "base_gpqa.json": ("base", "gpqa_diamond"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--json", default=None, help="classify a single run_eval output JSON")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    results = {}
    if a.json:
        c = classify_eval(Path(a.json))
        results[Path(a.json).name] = c
    else:
        d = Path(a.dir)
        for fname in STANDARD:
            p = d / fname
            if p.exists():
                results[fname] = classify_eval(p)
            else:
                print(f"[failmodes] (skip, not yet present) {fname}", file=sys.stderr)

    print(f"\n{'cell':28s} {'acc':>6s} {'n':>4s} {'parsed':>6s} "
          f"{'emptyEOS':>8s} {'trunc':>6s} {'other':>6s} {'err':>4s}")
    for name, c in results.items():
        print(f"{c['arm']+'/'+c['task']:28s} {c['accuracy']:6.4f} {c['n']:4d} "
              f"{c['n_parsed']:6d} {c['n_empty_eos']:8d} {c['n_truncation']:6d} "
              f"{c['n_other_fail']:6d} {c['n_error']:4d}")

    out = a.out or str(Path(a.dir) / "failmodes.json")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"\n[failmodes] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
