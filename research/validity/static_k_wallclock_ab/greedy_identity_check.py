#!/usr/bin/env python3
"""Greedy-identity certificate for the static-K wall-clock A/B (PR #273).

Changing the static draft-depth K (``num_speculative_tokens``) is *proposal-only*:
MTP speculative decoding verifies every proposed token greedy-exactly, so the
emitted token-ids are identical across all K by construction. This script PROVES
that empirically: it reads each K-arm's decode capture (the official
``decode_outputs.py`` jsonl, one row per prompt with ``completion_token_ids`` +
``completion_token_sha256``) and compares it prompt-for-prompt against the K=7
reference capture.

A clean certificate is ``128/128`` token-id identity for every K. Any mismatch
would mean changing K changed what the model emits — a greedy-identity break that
must abort the lever.

Usage:
    .venv/bin/python research/validity/static_k_wallclock_ab/greedy_identity_check.py \
        --seed 1 --ks 3 4 5 6 7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUTROOT = ROOT / "research" / "validity" / "static_k_wallclock_ab"
REF_K = 7


def _load_decode_rows(path: Path) -> dict[int, dict[str, Any]]:
    """{prompt index -> row} from a decode_outputs.py jsonl."""
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[int(r["index"])] = r
    return rows


def _find_decode(seed: int, k: int) -> Path | None:
    """Locate a K-arm's first decode capture.

    The fresh K=7 baseline is saved inside the K=4 arm (``--baseline-label
    mtp_k7``); every candidate K writes its own ``mtp_k<K>/decode/run00.jsonl``.
    """
    candidates = [
        OUTROOT / f"seed{seed}_mtp_k{k}" / f"mtp_k{k}" / "decode" / "run00.jsonl",
    ]
    if k == REF_K:
        # K=7 reference lives as the baseline arm of whichever candidate ran fresh.
        candidates += sorted(OUTROOT.glob(f"seed{seed}_mtp_k*/mtp_k7/decode/run00.jsonl"))
    for c in candidates:
        if c.exists():
            return c
    return None


def compare_arm(ref: dict[int, dict], arm: dict[int, dict]) -> dict[str, Any]:
    shared = sorted(set(ref) & set(arm))
    identical, mismatched = 0, []
    for idx in shared:
        a, b = ref[idx], arm[idx]
        # Prefer the exact token-id list; fall back to the sha256 the harness stores.
        a_tok = a.get("completion_token_ids")
        b_tok = b.get("completion_token_ids")
        if a_tok is not None and b_tok is not None:
            same = a_tok == b_tok
        else:
            same = a.get("completion_token_sha256") == b.get("completion_token_sha256")
        # Same prompt too (defensive: identical workload).
        prompt_same = a.get("prompt_token_sha256") == b.get("prompt_token_sha256")
        if same and prompt_same:
            identical += 1
        else:
            mismatched.append(idx)
    return {
        "n_prompts": len(shared),
        "n_identical": identical,
        "n_mismatched": len(mismatched),
        "mismatched_indices": mismatched[:20],
        "all_identical": len(mismatched) == 0 and len(shared) > 0,
    }


def build_certificate(seed: int, ks: list[int]) -> dict[str, Any]:
    ref_path = _find_decode(seed, REF_K)
    if ref_path is None:
        return {"error": f"K={REF_K} reference decode capture not found under {OUTROOT}"}
    ref_rows = _load_decode_rows(ref_path)
    per_k: dict[str, Any] = {}
    all_pass = True
    for k in ks:
        path = _find_decode(seed, k)
        if path is None:
            per_k[str(k)] = {"error": "decode capture not found"}
            all_pass = False
            continue
        if k == REF_K:
            # Reference vs itself: trivially identical, but report n_prompts.
            per_k[str(k)] = {
                "n_prompts": len(ref_rows), "n_identical": len(ref_rows),
                "n_mismatched": 0, "all_identical": len(ref_rows) > 0,
                "decode_file": str(path.relative_to(ROOT)), "is_reference": True,
            }
            continue
        arm_rows = _load_decode_rows(path)
        cmp = compare_arm(ref_rows, arm_rows)
        cmp["decode_file"] = str(path.relative_to(ROOT))
        per_k[str(k)] = cmp
        all_pass = all_pass and cmp["all_identical"]
    return {
        "seed": seed,
        "reference_k": REF_K,
        "reference_decode_file": str(ref_path.relative_to(ROOT)),
        "reference_n_prompts": len(ref_rows),
        "per_k": per_k,
        "token_id_identity_all_k": all_pass,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cert = build_certificate(args.seed, args.ks)
    out = args.out or (OUTROOT / f"greedy_identity_certificate_seed{args.seed}.json")
    out.write_text(json.dumps(cert, indent=2))

    print(f"\n===== Greedy-identity certificate (seed={args.seed}) =====")
    if "error" in cert:
        print("  ERROR:", cert["error"])
        return 1
    print(f"  reference K={cert['reference_k']} ({cert['reference_n_prompts']} prompts)"
          f"  <- {cert['reference_decode_file']}")
    for k in sorted(cert["per_k"], key=int):
        c = cert["per_k"][k]
        if "error" in c:
            print(f"  K={k}: {c['error']}")
            continue
        tag = "REF" if c.get("is_reference") else ("IDENTICAL" if c["all_identical"] else "MISMATCH")
        print(f"  K={k}: {c['n_identical']}/{c['n_prompts']} token-id identical  [{tag}]"
              + (f"  mismatched={c['mismatched_indices']}" if c.get("n_mismatched") else ""))
    print(f"\n  >>> token_id_identity_all_k = {cert['token_id_identity_all_k']}")
    print(f"  >>> certificate -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
