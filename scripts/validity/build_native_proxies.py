#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Build INDEPENDENT-axis private-proxy HARD component sets for PR #164.

#156 read the tree's private drop off ONE deliberately-hard chat proxy
(`data/private_proxy_sharegpt.json`, generic length-matched chat, ~10.7% linear)
scaled toward public by a single `frac`. PR #164 removes that single-shape
assumption by measuring the tree drop under >=2 GENUINELY INDEPENDENT
organizer-faithful proxies that vary the CONSTRUCTION AXIS.

This builder produces the distinct HARD COMPONENT 128-sets (each measured on the
deployed stack via `private_gap_probe.py`, then count-pooled with the measured
public ladder to the GT-4.3% anchor in `descent_vs_bothbugs_native.py`). The
component just needs to be a DISTINCT-shaped, >=4.3%-hard realization; the pool
weight does the calibration. Axes here:

  * `code`   : code/technical-heavy chat (REQUIRE >=1 code marker). The drafter's
               acceptance shape on code (boilerplate/syntax high, identifiers low)
               is distinct from prose.
  * `casual` : code-EXCLUDED short-turn conversational chat (length-matched to a
               SHORTER target, 0.6x public) -- the "wide / chat-heavy" register the
               private re-run is believed to be, varying BOTH domain and length.

Both reuse the banked `build_private_proxy.py` machinery (tokenizer length match,
hard dedup vs public, reasoning-marker exclusion) so the output JSON loads
byte-identically through `sglang.bench_serving` exactly like the public set.

LOCAL/CPU only (download + tokenize). No GPU, no HF Job, no served-file change.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
_BPP = REPO / "scripts" / "validity" / "build_private_proxy.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bpp = _load("build_private_proxy", _BPP)

# Reuse the banked machinery (one source of truth).
load_tokenizer = _bpp.load_tokenizer
load_public = _bpp.load_public
templ_len = _bpp.templ_len
first_human_prompt = _bpp.first_human_prompt
text_hash = _bpp.text_hash
norm = _bpp.norm
nearest_match = _bpp.nearest_match
dist = _bpp.dist
REASONING_MARKERS = _bpp.REASONING_MARKERS
SHAREGPT_REPO = _bpp.SHAREGPT_REPO
SHAREGPT_FILE = _bpp.SHAREGPT_FILE

# A turn is "code-ish" if it carries at least one of these (kept broad/robust).
CODE_MARKERS = (
    "```", "def ", "import ", "class ", "function ", "console.log", "#include",
    "public static", "select ", "println", "<html", "</", "=>", "{\n", "();",
    "$ ", "npm ", "pip install", "std::", "printf(",
)


def has_marker(low: str, markers: tuple[str, ...]) -> bool:
    return any(m in low for m in markers)


def collect_axis_candidates(
    path: Path, public_hashes: set[str], public_prefixes: set[str], *,
    min_char: int, max_char: int, seed: int, cap: int,
    require_code: bool, exclude_code: bool,
) -> list[str]:
    """Stream ShareGPT first-human chat prompts, drop public-template / reasoning /
    dup turns, then keep only the requested DOMAIN slice (code-required or
    code-excluded). Returns a shuffled, capped candidate text list."""
    data = json.loads(path.read_text())
    seen: set[str] = set()
    out: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        prompt = first_human_prompt(item)
        if prompt is None:
            continue
        clen = len(prompt)
        if clen < min_char or clen > max_char:
            continue
        low = prompt.lower()
        if has_marker(low, REASONING_MARKERS):
            continue
        is_code = has_marker(low, CODE_MARKERS)
        if require_code and not is_code:
            continue
        if exclude_code and is_code:
            continue
        h = text_hash(prompt)
        if h in public_hashes or h in seen:
            continue
        if norm(prompt)[:80] in public_prefixes:
            continue
        seen.add(h)
        out.append(prompt)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:cap]


def build_axis(
    *, name: str, axis: str, sharegpt_path: Path, public, public_hashes, public_prefixes,
    tok, length_scale: float, require_code: bool, exclude_code: bool,
    seed: int, candidate_cap: int, out_path: Path,
) -> dict[str, Any]:
    public_lens = [r["tok_len"] for r in public]
    pmin, pmax = min(public_lens), max(public_lens)
    targets = [max(8, round(l * length_scale)) for l in public_lens]

    cands = collect_axis_candidates(
        sharegpt_path, public_hashes, public_prefixes,
        min_char=80, max_char=40000, seed=seed, cap=candidate_cap,
        require_code=require_code, exclude_code=exclude_code)
    print(f"[{name}] {len(cands)} candidates after axis filter; tokenizing", flush=True)

    tmin = max(8, min(targets) // 3)
    tmax = max(targets) * 3
    cand_lens: list[int] = []
    cand_texts: list[str] = []
    for i, text in enumerate(cands):
        tl = templ_len(tok, text)
        if tl < tmin or tl > tmax:
            continue
        cand_lens.append(tl)
        cand_texts.append(text)
        if (i + 1) % 2000 == 0:
            print(f"[{name}]   tokenized {i+1}/{len(cands)}", flush=True)
    print(f"[{name}] {len(cand_lens)} candidates in token band; matching", flush=True)
    if len(cand_lens) < len(targets):
        raise RuntimeError(
            f"[{name}] only {len(cand_lens)} candidates for {len(targets)} targets; "
            f"loosen the axis filter or raise candidate_cap")

    matched = nearest_match(targets, cand_lens, cand_texts)
    proxy_lens = [m[1] for m in matched]
    residuals = [abs(m[0] - m[1]) for m in matched]

    records = []
    for i, (_t, _l, text) in enumerate(matched):
        records.append({
            "id": f"{axis}-{i:04d}",
            "conversations": [
                {"from": "human", "value": text},
                {"from": "gpt", "value": "Sure -- here is a helpful response."},
            ],
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))

    meta = {
        "purpose": f"PR #164 native-proxy HARD component, axis={axis}",
        "axis": axis,
        "construction": {
            "require_code": require_code, "exclude_code": exclude_code,
            "length_scale": length_scale, "seed": seed,
            "code_markers": list(CODE_MARKERS) if (require_code or exclude_code) else [],
        },
        "source": {"repo": SHAREGPT_REPO, "file": SHAREGPT_FILE, "path": str(sharegpt_path)},
        "num": len(records), "tokenizer": "(shared with public)",
        "candidates_in_band": len(cand_lens),
        "public_token_len_dist": dist(public_lens),
        "proxy_token_len_dist": dist(proxy_lens),
        "length_match_residual_tokens": {
            "max": max(residuals), "mean": round(statistics.mean(residuals), 2),
            "p90": sorted(residuals)[int(len(residuals) * .9)],
        },
        "dedup": "hard text-hash + 80-char prefix guard vs public; unique within proxy",
        "reasoning_markers_excluded": list(REASONING_MARKERS),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] wrote {out_path} ({len(records)} records); "
          f"proxy dist {dist(proxy_lens)}", flush=True)
    return meta


AXES = [
    {"name": "code", "axis": "domain-code", "length_scale": 1.0,
     "require_code": True, "exclude_code": False, "seed": 1640,
     "out": "data/private_proxy_native_code.json"},
    {"name": "casual", "axis": "register-casual-short", "length_scale": 0.6,
     "require_code": False, "exclude_code": True, "seed": 16400,
     "out": "data/private_proxy_native_casual.json"},
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sharegpt", default="/tmp/sharegpt_src/" + SHAREGPT_FILE)
    ap.add_argument("--public", default=str(_bpp.PUBLIC_DEFAULT))
    ap.add_argument("--tokenizer", default="/tmp/osoi5-v0-baked")
    ap.add_argument("--num", type=int, default=128)
    ap.add_argument("--candidate-cap", type=int, default=12000)
    ap.add_argument("--only", default=None, help="comma-sep axis names to build (default all)")
    args = ap.parse_args()

    sharegpt_path = Path(args.sharegpt)
    if not sharegpt_path.exists():
        from huggingface_hub import hf_hub_download
        print(f"[build] downloading {SHAREGPT_FILE} ...", flush=True)
        sharegpt_path = Path(hf_hub_download(
            SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset", local_dir="/tmp/sharegpt_src"))

    print("[build] loading tokenizer", flush=True)
    tok = load_tokenizer(args.tokenizer)
    print("[build] loading public set (official seed=1)", flush=True)
    public = load_public(Path(args.public), tok, args.num, seed=1)
    public_hashes = {text_hash(r["prompt_text"]) for r in public}
    public_prefixes = {norm(r["prompt_text"])[:80] for r in public}
    print(f"[build] public lens: {dist([r['tok_len'] for r in public])}", flush=True)

    only = set(args.only.split(",")) if args.only else None
    built = {}
    for spec in AXES:
        if only and spec["name"] not in only:
            continue
        built[spec["name"]] = build_axis(
            name=spec["name"], axis=spec["axis"], sharegpt_path=sharegpt_path,
            public=public, public_hashes=public_hashes, public_prefixes=public_prefixes,
            tok=tok, length_scale=spec["length_scale"], require_code=spec["require_code"],
            exclude_code=spec["exclude_code"], seed=spec["seed"],
            candidate_cap=args.candidate_cap, out_path=REPO / spec["out"])
    print(f"[build] DONE: built {list(built)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
