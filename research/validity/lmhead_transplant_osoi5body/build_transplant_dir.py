"""Build the osoi5-body + base-262k-head transplant model dir (PR #536).

Mechanism (validated by the compatibility check):
  * base `lm_head.weight` is byte-identical to base `embed_tokens.weight`
    (base is `tie_word_embeddings: True`, lm_head in the quant `ignore` list),
    AND osoi5-v0-baked `embed_tokens.weight` is byte-identical to base
    `embed_tokens.weight`.
  => "the base 262k head" == base embed == **osoi5's own embed_tokens**.

So the transplant is realised by serving the osoi5 *body* (transformer + full
262k BF16 embed_tokens + PLE) with `tie_word_embeddings=True` and lm_head marked
unquantized. vLLM then ties the output head to embed_tokens (== base head),
emits true 262144 logits, and SKIPS the stale osoi5 16k int4 lm_head
(`lm_head.weight_packed [16384,320]`) via its tie skip_substrs. No head download,
no shard surgery, no checkpoint rewrite — only a config edit + symlinks.

The ONLY moved variable vs the osoi5-16k row is the head (16k pruned int4 ->
full 262k base, realised as the tied embed). Body weights, attention, PLE are
the osoi5 substrate, unchanged.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

SRC = Path("/tmp/osoi5-v0-baked")
DST = Path("/tmp/osoi5-transplant-tie")


def st_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main() -> None:
    assert SRC.exists(), f"missing osoi5 body at {SRC}"
    DST.mkdir(parents=True, exist_ok=True)

    # 1) sanity: confirm the osoi5 head we are about to bypass is the 16k int4 head,
    #    and embed_tokens is the full 262k BF16 matrix we will tie to.
    hdr = st_header(SRC / "model.safetensors")
    lm = hdr["lm_head.weight_packed"]
    emb = hdr["model.language_model.embed_tokens.weight"]
    assert lm["shape"] == [16384, 320], lm["shape"]
    assert emb["shape"] == [262144, 2560] and emb["dtype"] == "BF16", emb
    print(f"[build] osoi5 lm_head.weight_packed={lm['shape']} (bypassed); "
          f"embed_tokens={emb['shape']} {emb['dtype']} (tied head)")

    # 2) symlink everything needed to serve, except config.json (we rewrite it).
    serve_files = [
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
        "processor_config.json",
    ]
    for name in serve_files:
        src = SRC / name
        if not src.exists():
            print(f"[build] WARNING: {src} missing, skipping symlink")
            continue
        link = DST / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
    print(f"[build] symlinked: {serve_files}")

    # 3) edit config.json: tie_word_embeddings=True + lm_head unquantized.
    cfg = json.loads((SRC / "config.json").read_text())

    def set_tie(d: dict) -> None:
        if "tie_word_embeddings" in d:
            d["tie_word_embeddings"] = True
    set_tie(cfg)
    if "text_config" in cfg:
        set_tie(cfg["text_config"])
        # text_config may lack the key entirely; force it.
        cfg["text_config"]["tie_word_embeddings"] = True
    cfg["tie_word_embeddings"] = True

    qc = cfg.get("quantization_config", {})
    groups = qc.get("config_groups", {})
    removed = groups.pop("group_0_lmhead", None)
    ignore = qc.get("ignore", [])
    if "lm_head" not in ignore:
        ignore = list(ignore) + ["lm_head"]
        qc["ignore"] = ignore
    print(f"[build] removed group_0_lmhead={removed is not None}; "
          f"lm_head added to ignore (now unquantized, like base)")

    (DST / "config.json").write_text(json.dumps(cfg, indent=2))

    # 4) report final state
    final = json.loads((DST / "config.json").read_text())
    tc = final.get("text_config", {})
    print(f"[build] DST={DST}")
    print(f"[build] tie_word_embeddings: top={final.get('tie_word_embeddings')} "
          f"text={tc.get('tie_word_embeddings')}")
    print(f"[build] quant groups now: {list(final['quantization_config']['config_groups'].keys())}")
    print(f"[build] 'lm_head' in ignore: {'lm_head' in final['quantization_config']['ignore']}")
    print("[build] DONE")


if __name__ == "__main__":
    main()
