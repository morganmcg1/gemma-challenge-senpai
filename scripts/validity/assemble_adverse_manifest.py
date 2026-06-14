#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Assemble the PR #176 6-axis native-proxy manifest.

Imports the 3 #164 axes BYTE-IDENTICALLY from
`research/validity/descent_vs_bothbugs_private/proxies_native.json` (q_pub_sglang +
the code / casual / sharegpt component ladders) and appends the 2-3 NEW hard-tail
axes by parsing their freshly-measured `server_private_rerun.log` per-position scored
ladders (same sglang vllm-chat scored protocol, deployed fa2sw_precache_kenyan stack).
The shared public reference (q_pub_sglang) is REUSED banked -- the new axes are
count-pooled against the identical public component as #164, so the construction is
held fixed and only the distinct hard component varies. Self-contained: the parsed
ladders are embedded (server .log files are gitignored).

LOCAL/CPU only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_recon = _load("tree_private_drop_reconcile", ROOT / "scripts/validity/tree_private_drop_reconcile.py")
parse_server_ladder = _recon.parse_server_ladder

# new axes: name -> (axis label, probe out-dir, one-line provenance)
NEW_AXES = [
    ("native_multilingual", "non-latin-script", "native_multilingual",
     "fresh non-Latin-script (Cyrillic/CJK/Arabic/...) ShareGPT, code-excluded, nonlatin-ratio>=0.15"),
    ("native_math", "math-notation-chain", "native_math",
     "fresh math-notation ShareGPT (LaTeX/symbolic markers), code-excluded domain predicate"),
    ("native_longctx", "long-context-tail", "native_longctx",
     "fresh long-context tail (2.5x public length, code-excluded), the prefill-heavy hard tail"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-manifest",
                    default="research/validity/descent_vs_bothbugs_private/proxies_native.json")
    ap.add_argument("--probe-root", default="research/validity/private_gap_probe")
    ap.add_argument("--output",
                    default="research/validity/private_adverse_skew/proxies_native_6axis.json")
    args = ap.parse_args()

    base = json.loads(Path(args.base_manifest).read_text())
    out = {
        "_comment": (
            "PR #176 6-axis native-proxy manifest, SELF-CONTAINED. The first 3 axes "
            "(native_code / native_casual / native_sharegpt) and q_pub_sglang are imported "
            "BYTE-IDENTICALLY from #164 proxies_native.json. The next axes are NEW genuinely-"
            "distinct hard tails (non-Latin script / math-notation / long-context), each measured "
            "on the deployed fa2sw_precache_kenyan sglang vllm-chat scored stack via "
            "private_gap_probe.py (precache=off, bench=private) and count-pooled against the SAME "
            "shared public reference at the continuous weight that lands the DECODE-frame linear "
            "drop on GT-4.3% (<=0.5pp gate). Independence comes from the distinct hard components, "
            "not from re-weighting one tail. Raw server logs are gitignored under "
            "research/validity/private_gap_probe/native_{multilingual,math,longctx}/."),
        "q_pub_sglang": base["q_pub_sglang"],
        "proxies": list(base["proxies"]),  # 3 #164 axes byte-identical
    }

    for name, axis, subdir, prov in NEW_AXES:
        log = ROOT / args.probe_root / subdir / "server_private_rerun.log"
        if not log.exists():
            raise SystemExit(f"[assemble] MISSING measured ladder: {log}\n"
                             f"  run: private_gap_probe.py --private data/{name.replace('native_','private_proxy_native_')}.json "
                             f"--out-dir {args.probe_root}/{subdir} --no-decompose")
        lad = parse_server_ladder(log)
        if lad is None:
            raise SystemExit(f"[assemble] could not parse per-position ladder from {log}")
        out["proxies"].append({
            "name": name, "axis": axis, "mode": "pooled",
            "component": {
                "_src": f"{log.relative_to(ROOT)} ({prov})",
                "conditional_p_sglang": lad["conditional_p"],
                "num_drafts": float(lad["num_drafts"]),
            },
        })
        print(f"[assemble] {name:<20s} drafts={lad['num_drafts']:.0f} "
              f"ladder={[round(x,4) for x in lad['conditional_p']]}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[assemble] wrote {args.output} with {len(out['proxies'])} axes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
