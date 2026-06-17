"""PR #553 Stage 2 -- realize lawine #544's precision lever (int4 262k lm_head).

LOCAL, analysis-only. Builds a disk-cheap experimental checkpoint that is
BIT-IDENTICAL to the clean QAT int4 base in body+attention+embeddings, and
differs ONLY in the lm_head: the full 262,144-row head is quantized to int4
(group_size 32, symmetric, weight-only W4A16 -- the exact compressed-tensors
scheme the body already uses, so vLLM serves it through the same Marlin GEMV).

Why sharded-symlink: the QAT snapshot is a single 11.5GB model.safetensors and
local disk has <7GB free, so we cannot rewrite the whole checkpoint. Instead:
  shard-1 = symlink to the original 11.5GB blob (body + bf16 embeds, UNCHANGED)
  shard-2 = small (~0.4GB) file holding the int4 lm_head (weight_packed/scale/shape)
  index    = maps every original tensor EXCEPT lm_head.weight -> shard-1, and the
             three int4 lm_head tensors -> shard-2
  config   = remove 'lm_head' from quant ignore, add a lm_head target so vLLM's
             compressed-tensors path quantizes ParallelLMHead, untie embeddings.

The bf16 ``lm_head.weight`` is still PHYSICALLY present in the symlinked shard-1
(we cannot edit the blob). vLLM's safetensors iterator yields every physical key,
so a sibling ``sitecustomize.py`` (loaded via PYTHONPATH + LMHEAD_INT4_SKIP_STRAY=1)
makes vLLM skip that one bf16 tensor so it never collides with the int4 head.

Run under the SERVE venv::

    /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
        research/realized_anchor_tps/build_int4_head.py
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.compressors.pack_quantized import pack_to_int32, unpack_from_int32

GROUP_SIZE = 32
FULL_VOCAB = 262144
HIDDEN = 2560
OUT_DIR = Path("/tmp/base-int4-lmhead")
SHARD1 = "model-00001-of-00002.safetensors"  # symlink -> original blob
SHARD2 = "model-00002-of-00002.safetensors"  # int4 lm_head
ROW_CHUNK = 32768


def _resolve_snapshot() -> Path:
    base = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    snaps = sorted(p for p in base.glob("*") if (p / "config.json").exists())
    if not snaps:
        raise RuntimeError(f"no qat-w4a16-ct snapshot under {base}")
    return snaps[0]


def _read_st_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main() -> int:
    snap = _resolve_snapshot()
    blob = (snap / "model.safetensors").resolve()  # follow HF symlink to real blob
    print(f"[build] snapshot={snap}\n[build] blob={blob}", flush=True)

    args = QuantizationArgs(
        num_bits=4, type=QuantizationType.INT, symmetric=True,
        strategy=QuantizationStrategy.GROUP, group_size=GROUP_SIZE,
    )

    # ---- load the bf16 head + the tied embed (to confirm precision-only isolation) ----
    with safe_open(str(blob), framework="pt", device="cpu") as f:
        w = f.get_tensor("lm_head.weight")  # [V,H] bf16
        emb = f.get_tensor("model.language_model.embed_tokens.weight")
    V, H = w.shape
    assert (V, H) == (FULL_VOCAB, HIDDEN), f"unexpected head shape {(V, H)}"
    tie_maxdiff = (w.float() - emb.float()).abs().max().item()
    print(f"[build] head={tuple(w.shape)} dtype={w.dtype} | "
          f"|lm_head - embed_tokens|_max = {tie_maxdiff:.3e} "
          f"({'TIED/identical' if tie_maxdiff == 0 else 'DIFFER'})", flush=True)
    del emb

    # ---- chunked int4 quant + pack (RAM-safe) ----
    packed_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    rt_mean_rel = rt_max_abs = None
    for i in range(0, V, ROW_CHUNK):
        wc = w[i:i + ROW_CHUNK].float()  # [c,H]
        c = wc.shape[0]
        wg = wc.reshape(c, H // GROUP_SIZE, GROUP_SIZE)
        min_vals = wg.amin(dim=-1)
        max_vals = wg.amax(dim=-1)
        scale_c, zp_c = calculate_qparams(min_vals, max_vals, args)  # [c, H/gs]
        q_c = quantize(wc, scale_c, zp_c, args)  # int-valued float [c,H]
        q_int = q_c.round().to(torch.int8)
        packed_c = pack_to_int32(q_int, num_bits=4, packed_dim=1)  # [c, H/8] int32
        packed_chunks.append(packed_c)
        scale_chunks.append(scale_c.to(torch.bfloat16))
        if i == 0:
            # validate pack round-trip + dequant error on the first chunk
            u = unpack_from_int32(packed_c, num_bits=4, shape=torch.Size([c, H]), packed_dim=1)
            assert torch.equal(u.to(torch.int8), q_int), "pack/unpack mismatch"
            deq = dequantize(q_int, scale_c, zp_c, args).float()
            denom = wc.abs().mean().item()
            rt_mean_rel = (deq - wc).abs().mean().item() / denom
            rt_max_abs = (deq - wc).abs().max().item()
            del u, deq
        del wc, wg, q_c, q_int, packed_c
    del w

    packed = torch.cat(packed_chunks, dim=0).contiguous()
    scale = torch.cat(scale_chunks, dim=0).contiguous()
    del packed_chunks, scale_chunks
    assert packed.shape == (V, H // 8), f"packed shape {tuple(packed.shape)}"
    assert scale.shape == (V, H // GROUP_SIZE), f"scale shape {tuple(scale.shape)}"
    print(f"[build] int4 head: packed={tuple(packed.shape)} {packed.dtype} "
          f"scale={tuple(scale.shape)} {scale.dtype} | "
          f"roundtrip mean_rel_err={rt_mean_rel:.4e} max_abs={rt_max_abs:.4e}", flush=True)

    # ---- write OUT dir ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*"):
        if stale.is_symlink() or stale.is_file():
            stale.unlink()

    shard2 = {
        "lm_head.weight_packed": packed,
        "lm_head.weight_scale": scale,
        "lm_head.weight_shape": torch.tensor([V, H], dtype=torch.int64),
    }
    save_file(shard2, str(OUT_DIR / SHARD2))
    shard2_bytes = (OUT_DIR / SHARD2).stat().st_size
    print(f"[build] wrote {SHARD2} ({shard2_bytes/1e6:.1f} MB)", flush=True)

    # shard-1 = symlink to the real 11.5GB blob
    (OUT_DIR / SHARD1).symlink_to(blob)
    blob_bytes = blob.stat().st_size

    # index: every original key except lm_head.weight -> shard1; int4 keys -> shard2
    hdr = _read_st_header(blob)
    weight_map: dict[str, str] = {}
    for k in hdr:
        if k == "__metadata__" or k == "lm_head.weight":
            continue
        weight_map[k] = SHARD1
    for k in shard2:
        weight_map[k] = SHARD2
    index = {
        "metadata": {"total_size": int(blob_bytes + shard2_bytes)},
        "weight_map": weight_map,
    }
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"[build] wrote index ({len(weight_map)} tensors: "
          f"{sum(v==SHARD1 for v in weight_map.values())} -> shard1, "
          f"{sum(v==SHARD2 for v in weight_map.values())} -> shard2)", flush=True)

    # config: drop lm_head from ignore, add lm_head quant target, untie embeddings
    cfg = json.loads((snap / "config.json").read_text())
    qc = cfg["quantization_config"]
    ign = [x for x in qc.get("ignore", []) if x != "lm_head"]
    assert "lm_head" not in ign and len(ign) == len(qc["ignore"]) - 1, "ignore edit failed"
    qc["ignore"] = ign
    tgts = qc["config_groups"]["group_0"]["targets"]
    if "re:.*lm_head$" not in tgts:
        tgts.append("re:.*lm_head$")
    qc["config_groups"]["group_0"]["targets"] = tgts
    cfg["tie_word_embeddings"] = False
    if "text_config" in cfg:
        cfg["text_config"]["tie_word_embeddings"] = False
    (OUT_DIR / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[build] wrote config.json (ignore {len(qc['ignore'])} entries, "
          f"group_0.targets={tgts}, tie_word_embeddings=False)", flush=True)

    # symlink the rest of the model dir (tokenizer / generation / chat template / processor)
    for src in snap.iterdir():
        if src.name in {"model.safetensors", "config.json"}:
            continue
        if src.name.endswith(".safetensors") or src.name.endswith(".index.json"):
            continue
        dst = OUT_DIR / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())

    # ---- summarize the build for the report ----
    meta = {
        "out_dir": str(OUT_DIR),
        "snapshot": str(snap),
        "group_size": GROUP_SIZE,
        "vocab": V, "hidden": H,
        "tie_maxdiff": tie_maxdiff,
        "tie_identical": tie_maxdiff == 0,
        "roundtrip_mean_rel_err": rt_mean_rel,
        "roundtrip_max_abs": rt_max_abs,
        "shard2_bytes": int(shard2_bytes),
        "blob_bytes": int(blob_bytes),
        "packed_shape": list(packed.shape),
        "scale_shape": list(scale.shape),
    }
    out_json = Path(__file__).resolve().parent / "build_int4_head.json"
    out_json.write_text(json.dumps(meta, indent=2))
    print(f"[build] DONE -> {OUT_DIR}\n[build] meta -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
