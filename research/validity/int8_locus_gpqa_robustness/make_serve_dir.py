#!/usr/bin/env python3
"""PR #717 -- build a TINY serve-dir for the int8-locus in-memory fake-quant.

NO disk model build: the 15.9 GB bf16 qat-unquantized base safetensors is SYMLINKED
(not copied); tokenizer/chat-template/generation-config are symlinked from the
qat-w4a16-ct cache (exactly the assets #696 served). The only real file written is a
~20 KB config.json that is the base (qat-unquantized, bf16, NO quantization_config)
config with `tie_word_embeddings=false` at both levels, so vLLM allocates a SEPARATE
`language_model.lm_head` that the in-memory RTN injector then fills with a synthetic
int4-g128 fake-quant of `embed_tokens` (fern #659's `lm_head = int4_g128 (locked)`),
while `embed_tokens` itself stays bf16.

The fake-quant of the 343 body Linear modules (int4-g128 skeleton + int8 on L14-27) is
applied at vLLM weight-load time by sitecustomize.py -- nothing is written to disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HOME = Path.home()
BASE = (HOME / ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized"
        / "snapshots/dfc5b925ddb1d41aaf1fe9679abdcfb0805e1aa6")
CT = (HOME / ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct"
      / "snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0")
OUT = Path(os.environ.get("SERVE_DIR", "/tmp/wirbel_int8locus_serve"))

ASSETS = [
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    assert (BASE / "model.safetensors").exists(), f"missing base weights {BASE}"
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. symlink the 15.9 GB bf16 base weights (NO copy -> not a disk model build)
    _link(BASE / "model.safetensors", OUT / "model.safetensors")

    # 2. config.json: base (bf16, NO quantization_config) with tie_word_embeddings=false
    cfg = json.load(open(BASE / "config.json"))
    cfg["tie_word_embeddings"] = False
    if "text_config" in cfg:
        cfg["text_config"]["tie_word_embeddings"] = False
    assert "quantization_config" not in cfg, "base must be plain bf16 (no quant config)"
    json.dump(cfg, open(OUT / "config.json", "w"), indent=2)

    # 3. tokenizer / chat-template / generation-config: symlink the #696-served ct assets
    for fn in ASSETS:
        src = CT / fn
        if src.exists():
            _link(src, OUT / fn)

    nlayers = cfg.get("text_config", cfg).get("num_hidden_layers")
    print(f"[serve-dir] {OUT} ready: tie=false, layers={nlayers}, "
          f"weights->{(BASE / 'model.safetensors').resolve()}")
    print(f"[serve-dir] assets: {[f for f in ASSETS if (OUT / f).exists()]}")


if __name__ == "__main__":
    main()
