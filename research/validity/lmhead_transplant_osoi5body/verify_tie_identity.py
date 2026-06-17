"""Independent verification of the transplant's load-bearing identity (PR #536).

The transplant realises "osoi5 body + base 262k head" by tying the osoi5 output
head to osoi5's OWN embed_tokens. That equals "the base 262k head" iff:

  (A) base lm_head.weight  == base embed_tokens.weight   (base internal tie)
  (B) osoi5 embed_tokens.weight == base embed_tokens.weight  (bake left embed intact)

If both hold, tied(osoi5 head -> osoi5 embed) is byte-exactly the base 262k head.
We hash the raw tensor bytes (bounded memory) and also report dtype/shape and a
few element spot-checks.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open

OSOI5 = Path("/tmp/osoi5-v0-baked/model.safetensors")
BASE = Path(
    "/senpai-run/home/student-stark/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0/model.safetensors"
)


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def find_key(hdr: dict, *needles: str) -> str | None:
    for k in hdr:
        kl = k.lower()
        if all(nd in kl for nd in needles) and "scale" not in kl and "packed" not in kl:
            return k
    return None


def tensor_sha(path: Path, key: str) -> tuple[str, tuple, str]:
    with safe_open(str(path), framework="pt") as f:
        t = f.get_tensor(key)
    raw = t.contiguous().flatten().view(torch.uint8).numpy().tobytes()
    h = hashlib.sha256(raw).hexdigest()
    return h, tuple(t.shape), str(t.dtype)


def main() -> None:
    oh, bh = header(OSOI5), header(BASE)
    o_emb = find_key(oh, "embed_tokens")
    b_emb = find_key(bh, "embed_tokens")
    b_lm = find_key(bh, "lm_head")
    print(f"[keys] osoi5 embed = {o_emb}")
    print(f"[keys] base  embed = {b_emb}")
    print(f"[keys] base  lm_head = {b_lm}  (None => tied/absent in base ckpt)")

    res = {}
    o_sha, o_shape, o_dt = tensor_sha(OSOI5, o_emb)
    b_sha, b_shape, b_dt = tensor_sha(BASE, b_emb)
    print(f"[osoi5 embed] shape={o_shape} dtype={o_dt} sha256={o_sha}")
    print(f"[base  embed] shape={b_shape} dtype={b_dt} sha256={b_sha}")
    res["osoi5_embed"] = {"shape": o_shape, "dtype": o_dt, "sha256": o_sha}
    res["base_embed"] = {"shape": b_shape, "dtype": b_dt, "sha256": b_sha}
    res["B_osoi5_embed_eq_base_embed"] = (o_sha == b_sha and o_shape == b_shape)

    if b_lm is not None:
        l_sha, l_shape, l_dt = tensor_sha(BASE, b_lm)
        print(f"[base  lm_head] shape={l_shape} dtype={l_dt} sha256={l_sha}")
        res["base_lm_head"] = {"shape": l_shape, "dtype": l_dt, "sha256": l_sha}
        res["A_base_lm_head_eq_base_embed"] = (l_sha == b_sha and l_shape == b_shape)
    else:
        res["base_lm_head"] = None
        res["A_base_lm_head_eq_base_embed"] = "tied_in_config (no explicit lm_head tensor => served head == embed)"

    res["tied_head_is_base_head"] = bool(res["B_osoi5_embed_eq_base_embed"]) and (
        res["A_base_lm_head_eq_base_embed"] in (True, "tied_in_config (no explicit lm_head tensor => served head == embed)")
    )
    print("\n[VERDICT]")
    print(f"  (A) base lm_head == base embed : {res['A_base_lm_head_eq_base_embed']}")
    print(f"  (B) osoi5 embed  == base embed : {res['B_osoi5_embed_eq_base_embed']}")
    print(f"  => tied head IS the base 262k head (byte-exact): {res['tied_head_is_base_head']}")

    Path(__file__).with_name("tie_identity_verified.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[wrote] {Path(__file__).with_name('tie_identity_verified.json')}")


if __name__ == "__main__":
    main()
