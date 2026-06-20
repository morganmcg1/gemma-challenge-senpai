#!/usr/bin/env python
"""Byte-identity proof for the token-free PLE-dequant startup-quant serve path (PR #817).

Proves that quantizing the PUBLIC base lm_head to int4 g32 AND de-quantizing the
42 per_layer_input_gate projections to bf16 at serve startup (serve.py
LMHEAD_QUANT_AT_STARTUP=1 + LMHEAD_QUANT_DEQUANT_PLE=1 -> build_lmhead_quant.py
--dequant-ple) yields a checkpoint that is BIT-IDENTICAL to the validated #805
PLE-dequant checkpoint published to the private Hub repo (the one that scored
265.61 TPS / PPL 2.0031 / GSM8K 0.925). If the FULL model.safetensors sha256
matches, the served weights -- hence PPL, greedy decode, and the leaderboard
gate -- are identical by construction, so the token-free path needs no quality
re-validation beyond the local gate smoke.

Unlike #801's int4head proof (which compared ONLY the lm_head tensors, because
the rest of the checkpoint was copied byte-for-byte), the PLE-dequant build adds
a SECOND delta: 42 per_layer_input_gate modules switch from int4 packed/scale/
shape to bf16 .weight. A lm_head-only check would MISS a PLE divergence, so this
proof uses the WHOLE-FILE sha256 (covers int4 body + int4 lm_head + 42 bf16 PLE
gates + bf16 towers) plus the small config/tokenizer assets.

Reference = the private Hub repo's git-LFS sha256 (== sha256sum of the file
content), fetched with a token LOCALLY -- the serve path itself is token-free.

  --built <dir>   compare an already-built checkpoint dir (serve.py's
                  LMHEAD_QUANT_OUT) against the #805 private-repo reference.

Exit 0 iff model.safetensors + every compared asset is byte-identical, else 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REF_REPO = "gemma-challenge/gemma-4-e4b-it-int4-mtp-bi0-int4head-pledequant"
REF_REV = "f5a0dfd1caa52b429b6a0e973b53d2aac8e14a22"
ASSET_FILES = ("config.json", "tokenizer.json", "generation_config.json", "chat_template.jinja")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _ref_lfs_sha256(repo: str, rev: str, filename: str) -> tuple[str, int]:
    """git-LFS sha256 + byte size of a Hub file, WITHOUT downloading it."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=rev, files_metadata=True, token=True)
    for s in info.siblings:
        if s.rfilename == filename:
            lfs = s.lfs or {}
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            return sha, int(s.size or 0)
    raise KeyError(f"{filename} not found in {repo}@{rev}")


def _ref_asset_sha256(repo: str, rev: str, filename: str) -> str:
    """sha256 of a small (non-LFS) Hub asset, fetched with a token (local only)."""
    from huggingface_hub import hf_hub_download

    return _sha256_file(Path(hf_hub_download(repo, filename, revision=rev, token=True)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--built", required=True, help="built checkpoint dir (serve.py LMHEAD_QUANT_OUT)")
    ap.add_argument("--ref-repo", default=REF_REPO)
    ap.add_argument("--ref-rev", default=REF_REV)
    args = ap.parse_args()

    built = Path(args.built)
    st = built / "model.safetensors"
    print(f"[built] {st}", flush=True)
    built_sha = _sha256_file(st)
    built_size = st.stat().st_size
    ref_sha, ref_size = _ref_lfs_sha256(args.ref_repo, args.ref_rev, "model.safetensors")
    print(f"[ref]   hf://{args.ref_repo}@{args.ref_rev[:8]}  model.safetensors", flush=True)

    rows = []
    weights_ok = built_sha == ref_sha and built_size == ref_size
    mark = "OK " if weights_ok else "MISMATCH"
    print(
        f"  [{mark}] model.safetensors  built={built_sha[:16]}… ({built_size} B)  "
        f"ref={ref_sha[:16]}… ({ref_size} B)",
        flush=True,
    )
    rows.append(("model.safetensors", built_sha, ref_sha, built_size, ref_size, weights_ok))

    assets_ok = True
    for fn in ASSET_FILES:
        bpath = built / fn
        if not bpath.exists():
            print(f"  [SKIP] {fn} (not in built dir)", flush=True)
            continue
        b = _sha256_file(bpath)
        r = _ref_asset_sha256(args.ref_repo, args.ref_rev, fn)
        ok = b == r
        assets_ok = assets_ok and ok
        print(f"  [{'OK ' if ok else 'MISMATCH'}] {fn:<22} built={b[:16]}… ref={r[:16]}…", flush=True)
        rows.append((fn, b, r, None, None, ok))

    all_ok = weights_ok and assets_ok
    report = {
        "ref_repo": args.ref_repo,
        "ref_rev": args.ref_rev,
        "all_byte_identical": all_ok,
        "files": [
            {"name": n, "built_sha256": bs, "ref_sha256": rs,
             "built_bytes": bsz, "ref_bytes": rsz, "byte_identical": ok}
            for (n, bs, rs, bsz, rsz, ok) in rows
        ],
    }
    print(json.dumps(report, indent=2))
    print(
        f"\nBYTE-IDENTITY: {'PASS — full checkpoint bit-identical to validated #805' if all_ok else 'FAIL — checkpoint DIFFERS'}",
        flush=True,
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
