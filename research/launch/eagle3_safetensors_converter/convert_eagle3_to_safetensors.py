#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""CPU-only converter: the {2,21,39} EAGLE-3 head -> a launch-ready safetensors dir (PR #333).

Closes ubel #328 blockers **2** (`weight_key_namespace_mismatch`) and **3**
(`container_format`) in ONE reviewed, deterministic step. It takes the 15-tensor
`Eagle3DraftHead.state_dict()` (`scripts/drafter/train_eagle3.py` ->
`research/eagle3_drafter/checkpoints/<run>/model_best.pt`, the fern #34 `gua9x68j` /
`56ksyxgw` head) and emits a two-file vLLM-loadable candidate directory
(`model.safetensors` + `config.json`) plus a `sha256` and an Approval-request snippet
for HUMAN review.

It does the three mechanical, lossless steps the #328 audit specified:
  1. **Finding-B rename** (audit §2): strip the leading `model.` from the body keys,
     `layers.0.` -> the canonical EAGLE-3 `midlayer.`, keep `lm_head.*`, and keep q/k/v
     + gate/up SEPARATE (vLLM fuses them via `stacked_params_mapping` at load).
  2. **bf16 cast**: body fp32 -> bf16; `embed_tokens`/`lm_head` already bf16 -> a uniform
     single-dtype `model.safetensors` (serving dtype; clean file, audit caveat C2).
  3. **config emit**: a vLLM EAGLE-3 `config.json` (`model_type:"llama"` for AutoConfig
     survival + a nested `eagle_config` backstop), matching #328 §3 + ubel #299 arch.

Then it ASSERTS — by porting vLLM 0.22.1rc1's `Eagle3LlamaForCausalLM.load_weights` /
`LlamaModel.load_weights` name+shape contract — that every published tensor lands on a
real vLLM parameter/shard with an exactly matching shape (so a bad export fails LOUDLY
here, on CPU, not at the one HF launch).

**0 GPU. NO model forward. NO publish / NO bucket write / NO manifest change / NO HF
job / NO submission / NO served-file change.** Publishing the produced artifact to the
`DRAFTER_BUCKET` and editing the manifest stays HUMAN-owned (audit blocker 1
`no_published_path`). This script only produces the reviewed converter + a
synthetic-shape-validated dry run.

Run the self-test with **0 GPU and no checkpoint present** (synthetic zero-tensors at
the exact #328 inventory shapes/dtypes):

    cd target/ && python research/launch/eagle3_safetensors_converter/convert_eagle3_to_safetensors.py \
        --synthetic-shapes --self-test \
        --wandb_group eagle3-safetensors-converter --wandb_name ubel/eagle3-safetensors-converter

Convert the REAL head (human, after locating the local `.pt`):

    python research/launch/eagle3_safetensors_converter/convert_eagle3_to_safetensors.py \
        --in research/eagle3_drafter/checkpoints/<run>/model_best.pt

PRIMARY metric : `converter_self_test_passes` (1 if all >=15 synthetic-mode checks hold).
TEST   metric  : `tensors_mapped_post_rename` (expect 15).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                       # .../target
CANDIDATE_DIR = HERE / "_candidate"

# --------------------------------------------------------------------------- #
# Analytic facts — Gemma-4-E4B EAGLE-3 draft head dims (authoritative: the module
# constants in scripts/drafter/train_eagle3.py; cross-checked vs ubel #299 arch).
# --------------------------------------------------------------------------- #
HID = 2560          # draft hidden size
VOCAB = 262144      # full Gemma vocab (identity draft<->target; no d2t/t2d)
N_AUX = 3           # aux layers {2, 21, 39}
HEAD_DIM = 256
N_HEADS = 8
N_KV = 2
INTER = 10240
EPS = 1e-6
ROPE_THETA = 1e6
AUX_LAYER_IDS = [2, 21, 39]

Q = N_HEADS * HEAD_DIM              # 2048  q-proj rows
KV = N_KV * HEAD_DIM               # 512   k/v-proj rows
QKV = (N_HEADS + 2 * N_KV) * HEAD_DIM  # 3072 fused qkv rows
GU = 2 * INTER                     # 20480 fused gate_up rows
TWO_H = 2 * HID                    # 5120  layer-0 qkv input (embeds ++ hidden)
FUSED_IN = N_AUX * HID             # 7680  fc input / input_norm dim

# --------------------------------------------------------------------------- #
# SOURCE: exactly what `Eagle3DraftHead.state_dict()` writes (15 tensors).
#   dtype: embed_tokens + lm_head are explicitly `.to(bfloat16)` (train lines
#   586-587/597-598); the other 13 body tensors are nn.Parameter at torch default
#   float32 -> the saved .pt is MIXED dtype. (name, [shape], saved-dtype)
# --------------------------------------------------------------------------- #
BF16, FP32 = "bfloat16", "float32"
SOURCE_INVENTORY: list[tuple[str, list[int], str]] = [
    ("model.embed_tokens.weight", [VOCAB, HID], BF16),
    ("model.input_norm.weight", [FUSED_IN], FP32),
    ("model.fc.weight", [HID, FUSED_IN], FP32),
    ("model.layers.0.self_attn.q_proj.weight", [Q, TWO_H], FP32),
    ("model.layers.0.self_attn.k_proj.weight", [KV, TWO_H], FP32),
    ("model.layers.0.self_attn.v_proj.weight", [KV, TWO_H], FP32),
    ("model.layers.0.self_attn.o_proj.weight", [HID, Q], FP32),
    ("model.layers.0.mlp.gate_proj.weight", [INTER, HID], FP32),
    ("model.layers.0.mlp.up_proj.weight", [INTER, HID], FP32),
    ("model.layers.0.mlp.down_proj.weight", [HID, INTER], FP32),
    ("model.layers.0.input_layernorm.weight", [HID], FP32),
    ("model.layers.0.hidden_norm.weight", [HID], FP32),
    ("model.layers.0.post_attention_layernorm.weight", [HID], FP32),
    ("model.norm.weight", [HID], FP32),
    ("lm_head.weight", [VOCAB, HID], BF16),
]
N_SOURCE_TENSORS = len(SOURCE_INVENTORY)          # 15

# --------------------------------------------------------------------------- #
# TARGET: vLLM `Eagle3LlamaForCausalLM` loadable params (named_parameters), AFTER the
# q/k/v->qkv_proj + gate/up->gate_up_proj stacked fusion. (llama_eagle3.py.)
# --------------------------------------------------------------------------- #
VLLM_PARAMS: dict[str, list[int]] = {
    "model.embed_tokens.weight": [VOCAB, HID],
    "model.input_norm.weight": [FUSED_IN],
    "model.fc.weight": [HID, FUSED_IN],
    "model.norm.weight": [HID],
    "model.layers.0.self_attn.qkv_proj.weight": [QKV, TWO_H],
    "model.layers.0.self_attn.o_proj.weight": [HID, Q],
    "model.layers.0.mlp.gate_up_proj.weight": [GU, HID],
    "model.layers.0.mlp.down_proj.weight": [HID, INTER],
    "model.layers.0.input_layernorm.weight": [HID],
    "model.layers.0.hidden_norm.weight": [HID],
    "model.layers.0.post_attention_layernorm.weight": [HID],
    "lm_head.weight": [VOCAB, HID],
}
# Shard decomposition of the fused params (shard_id -> expected shard shape).
VLLM_SHARDS: dict[str, dict[Any, list[int]]] = {
    "model.layers.0.self_attn.qkv_proj.weight": {"q": [Q, TWO_H], "k": [KV, TWO_H], "v": [KV, TWO_H]},
    "model.layers.0.mlp.gate_up_proj.weight": {0: [INTER, HID], 1: [INTER, HID]},
}
# vLLM stacked_params_mapping (llama_eagle3.py LlamaModel.load_weights).
STACKED = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]


# --------------------------------------------------------------------------- #
# Finding-B rename + vLLM load-path port (faithful to #328 audit §2).
# --------------------------------------------------------------------------- #
def published_name(src: str) -> str:
    """The lossless rename a vLLM-loadable converter applies to a saved key.

    Strip the leading `model.` from body keys; rename the single decoder layer
    `layers.0.` -> the canonical EAGLE-3 `midlayer.`; keep `lm_head.*` as-is. q/k/v +
    gate/up are kept SEPARATE (vLLM fuses them itself).
    """
    if src.startswith("lm_head."):
        return src
    body = src[len("model."):] if src.startswith("model.") else src
    if body.startswith("layers.0."):
        body = "midlayer." + body[len("layers.0."):]
    return body


def vllm_remap(name: str) -> tuple[str, Any]:
    """Faithful port of vLLM 0.22.1rc1 weight-name resolution for a published key.

    Replicates `Eagle3LlamaForCausalLM.load_weights` then `LlamaModel.load_weights` for
    the `model.*` subtree. Returns (internal_param_name, shard_id); shard_id is None for
    a direct param, or the sentinel ("<skip>", reason) for a dropped key.
    """
    if "t2d" in name:
        return ("<skip>", "t2d dropped")
    if "d2t" in name:
        name = name.replace("d2t", "draft_id_to_target_id")
    elif "mask_hidden" in name:
        return ("<skip>", "mask_hidden (parallel-draft only)")
    elif "lm_head" not in name:
        name = "model." + name  # unconditional model. prepend
    if name.startswith("model.") and name != "model.draft_id_to_target_id":
        sub = name[len("model."):]
        if "midlayer." in sub:
            sub = sub.replace("midlayer.", "layers.0.")
        for param_name, weight_name, shard_id in STACKED:
            if weight_name in sub:
                sub = sub.replace(weight_name, param_name)
                return ("model." + sub, shard_id)
        return ("model." + sub, None)
    return (name, None)


def required_targets() -> set[tuple[str, Any]]:
    """Every (param, shard) load-target a complete EAGLE-3 head must fill."""
    req: set[tuple[str, Any]] = set()
    for p in VLLM_PARAMS:
        if p in VLLM_SHARDS:
            for s in VLLM_SHARDS[p]:
                req.add((p, s))
        else:
            req.add((p, None))
    return req


def check_mapping(named_shapes: list[tuple[str, list[int]]]) -> dict[str, Any]:
    """Run each (published_name, shape) through vLLM's resolver; record where it lands
    and whether the shape matches the target param/shard."""
    rows = []
    filled: set[tuple[str, Any]] = set()
    for name, shape in named_shapes:
        internal, shard = vllm_remap(name)
        if internal == "<skip>":
            rows.append({"src": name, "internal": None, "status": "skipped", "detail": shard})
            continue
        if shard is not None and internal in VLLM_SHARDS:
            want = VLLM_SHARDS[internal].get(shard)
            ok = want == shape
            rows.append({"src": name, "internal": internal, "shard": shard,
                         "src_shape": shape, "want_shape": want,
                         "status": "ok" if ok else "shape_mismatch"})
            if ok:
                filled.add((internal, shard))
        elif internal in VLLM_PARAMS:
            want = VLLM_PARAMS[internal]
            ok = want == shape
            rows.append({"src": name, "internal": internal, "shard": None,
                         "src_shape": shape, "want_shape": want,
                         "status": "ok" if ok else "shape_mismatch"})
            if ok:
                filled.add((internal, None))
        else:
            rows.append({"src": name, "internal": internal, "shard": shard,
                         "src_shape": shape, "want_shape": None, "status": "unexpected"})
    req = required_targets()
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_unexpected = sum(1 for r in rows if r["status"] == "unexpected")
    n_shape_mismatch = sum(1 for r in rows if r["status"] == "shape_mismatch")
    missing = sorted(f"{p}::{s}" for (p, s) in (req - filled))
    return {
        "rows": rows, "n_ok": n_ok, "n_unexpected": n_unexpected,
        "n_shape_mismatch": n_shape_mismatch, "missing_targets": missing,
        "covers_all_targets": (filled == req) and not (n_unexpected or n_shape_mismatch),
    }


# --------------------------------------------------------------------------- #
# Config emit (vLLM EAGLE-3 draft config — #328 §3 + ubel #299/#322 arch fields).
# --------------------------------------------------------------------------- #
def build_config() -> dict[str, Any]:
    return {
        "architectures": ["Eagle3LlamaForCausalLM"],
        "model_type": "llama",  # AutoConfig-survivable; vLLM reads EAGLE knobs separately
        "hidden_size": HID,
        "intermediate_size": INTER,
        "num_hidden_layers": 1,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "draft_vocab_size": VOCAB,
        "rms_norm_eps": EPS,
        "rope_theta": ROPE_THETA,
        "max_position_embeddings": 131072,
        "norm_before_fc": True,
        "target_hidden_size": HID,
        "num_aux_hidden_states": N_AUX,
        "eagle_aux_hidden_state_layer_ids": list(AUX_LAYER_IDS),
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        # Belt-and-suspenders: vLLM reads `eagle_config` first (LlamaModel.__init__),
        # so nest the EAGLE knobs here as the robust backstop if AutoConfig drops the
        # top-level custom fields (audit caveat C1).
        "eagle_config": {
            "norm_before_fc": True,
            "target_hidden_size": HID,
            "num_aux_hidden_states": N_AUX,
            "eagle_aux_hidden_state_layer_ids": list(AUX_LAYER_IDS),
        },
    }


REQUIRED_CONFIG_FIELDS = [
    "architectures", "model_type", "hidden_size", "intermediate_size",
    "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
    "vocab_size", "draft_vocab_size", "rms_norm_eps", "rope_theta", "norm_before_fc",
    "target_hidden_size", "num_aux_hidden_states", "eagle_aux_hidden_state_layer_ids",
    "tie_word_embeddings",
]


class _StubAutoConfig:
    """Minimal stand-in for HF `PretrainedConfig`: stores every kwarg as an attribute
    (exactly what `AutoConfig(model_type=llama)` does with unknown fields). Lets the
    self-test verify config-field survival with 0 GPU and no transformers dependency."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def parse_config_through_autoconfig(config: dict[str, Any]) -> tuple[bool, str]:
    """Parse the emitted config the way vLLM will (real transformers if importable,
    else the stub). Return (custom_fields_survived, backend)."""
    custom = ["norm_before_fc", "target_hidden_size", "num_aux_hidden_states",
              "eagle_aux_hidden_state_layer_ids"]
    backend = "stub"
    try:
        from transformers import LlamaConfig  # type: ignore

        cfg_obj: Any = LlamaConfig(**config)
        backend = "transformers.LlamaConfig"
    except Exception:
        cfg_obj = _StubAutoConfig(**config)
    survived = all(getattr(cfg_obj, f, None) is not None for f in custom)
    survived = survived and getattr(cfg_obj, "eagle_aux_hidden_state_layer_ids", None) == AUX_LAYER_IDS
    # The nested eagle_config backstop must round-trip regardless of the top-level fate.
    nested = getattr(cfg_obj, "eagle_config", None)
    nested_ok = isinstance(nested, dict) and nested.get("eagle_aux_hidden_state_layer_ids") == AUX_LAYER_IDS
    return (bool(survived and nested_ok), backend)


# --------------------------------------------------------------------------- #
# State-dict sources.
# --------------------------------------------------------------------------- #
def build_synthetic_state_dict() -> "dict[str, Any]":
    """Zero-tensors at the EXACT #328 inventory shapes + saved dtypes (mixed: embed/
    lm_head bf16, 13 body fp32). Tests the fp32->bf16 cast path with 0 GPU and no
    checkpoint present."""
    import torch

    dtmap = {BF16: torch.bfloat16, FP32: torch.float32}
    return {name: torch.zeros(*shape, dtype=dtmap[dt]) for name, shape, dt in SOURCE_INVENTORY}


def load_state_dict(path: str) -> "dict[str, Any]":
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) and "state_dict" in obj else obj
    if not isinstance(state, dict):
        raise ValueError(f"{path} did not load to a state_dict mapping (got {type(state)})")
    return state


# --------------------------------------------------------------------------- #
# The conversion (deterministic): rename -> bf16 -> assert-map -> save 2-file dir.
# --------------------------------------------------------------------------- #
def convert(state_dict: "dict[str, Any]", out_dir: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)

    renamed: dict[str, Any] = {}
    collisions: list[str] = []
    for key, tensor in state_dict.items():
        new_key = published_name(key)
        if new_key in renamed:
            collisions.append(new_key)
        # bf16 cast + clone (contiguous, fresh storage -> no shared-tensor save error,
        # deterministic round-to-nearest-even for the fp32 body tensors).
        renamed[new_key] = tensor.to(torch.bfloat16).contiguous().clone()
    if collisions:
        raise ValueError(f"rename collision(s): {collisions}")

    # Deterministic serialization: sort keys so insertion order (and thus the
    # safetensors header + data layout) is identical across runs -> stable sha256.
    ordered = {k: renamed[k] for k in sorted(renamed)}
    named_shapes = [(k, list(v.shape)) for k, v in ordered.items()]
    mapping = check_mapping(named_shapes)
    if mapping["n_unexpected"] or mapping["n_shape_mismatch"] or not mapping["covers_all_targets"]:
        raise AssertionError(
            "vLLM load-map check FAILED — the export would not load: "
            f"unexpected={mapping['n_unexpected']} shape_mismatch={mapping['n_shape_mismatch']} "
            f"missing={mapping['missing_targets']}. Re-derive the rename (audit §2)."
        )

    # NaN/Inf guard on the cast tensors.
    nan_keys = [k for k, v in ordered.items() if not bool(torch.isfinite(v).all())]
    nan_clean = not nan_keys

    st_path = out_dir / "model.safetensors"
    save_file(ordered, str(st_path), metadata={"format": "pt"})

    config = build_config()
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sha256 = _sha256(st_path)
    (out_dir / "model.safetensors.sha256").write_text(
        f"{sha256}  model.safetensors\n", encoding="utf-8")

    all_bf16 = all(str(v.dtype) == "torch.bfloat16" for v in ordered.values())
    return {
        "out_dir": str(out_dir),
        "n_tensors": len(ordered),
        "published_keys": list(ordered.keys()),
        "tensors_mapped_post_rename": mapping["n_ok"],
        "n_unexpected": mapping["n_unexpected"],
        "n_shape_mismatch": mapping["n_shape_mismatch"],
        "covers_all_targets": mapping["covers_all_targets"],
        "missing_targets": mapping["missing_targets"],
        "all_bf16": all_bf16,
        "nan_clean": nan_clean,
        "nan_keys": nan_keys,
        "sha256": sha256,
        "safetensors_path": str(st_path),
        "config_path": str(cfg_path),
        "mapping_rows": mapping["rows"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def roundtrip_check(st_path: Path, expected: list[tuple[str, list[int]]]) -> dict[str, Any]:
    """Reload model.safetensors and confirm every key/shape/dtype survives the file."""
    from safetensors import safe_open

    seen: dict[str, tuple[list[int], str]] = {}
    with safe_open(str(st_path), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            seen[k] = (list(t.shape), str(t.dtype))
    exp_keys = {k for k, _ in expected}
    keys_match = set(seen.keys()) == exp_keys
    shapes_match = all(seen.get(k, (None, None))[0] == shp for k, shp in expected)
    all_bf16 = all(dt == "torch.bfloat16" for _, dt in seen.values())
    return {"keys_match": keys_match, "shapes_match": shapes_match,
            "all_bf16": all_bf16, "n_keys": len(seen)}


# --------------------------------------------------------------------------- #
# Self-test (>=15 checks; synthetic, 0 GPU, no checkpoint).
# --------------------------------------------------------------------------- #
def run_self_test() -> dict[str, Any]:
    import torch  # noqa: F401

    synth = build_synthetic_state_dict()

    with tempfile.TemporaryDirectory() as td:
        d1 = Path(td) / "run1"
        d2 = Path(td) / "run2"
        r1 = convert(build_synthetic_state_dict(), d1)
        r2 = convert(build_synthetic_state_dict(), d2)

        published_shapes = [(published_name(n), s) for n, s, _ in SOURCE_INVENTORY]
        rt = roundtrip_check(Path(r1["safetensors_path"]), published_shapes)

        # Raw (un-renamed) keys must NOT all load — only `lm_head.weight` is exempt from
        # vLLM's model.-prepend, so the rename is provably necessary (audit Finding A).
        raw_named = [(n, s) for n, s, _ in SOURCE_INVENTORY]
        raw_map = check_mapping(raw_named)

        config = build_config()
        cfg_survived, cfg_backend = parse_config_through_autoconfig(config)

        qkv_ok = (VLLM_PARAMS["model.layers.0.self_attn.qkv_proj.weight"] == [QKV, TWO_H]
                  and QKV == Q + 2 * KV)
        gu_ok = (VLLM_PARAMS["model.layers.0.mlp.gate_up_proj.weight"] == [GU, HID]
                 and GU == 2 * INTER)

        candidate_files = sorted(p.name for p in d1.iterdir())

        checks = {
            "source_has_15_tensors": len(synth) == N_SOURCE_TENSORS == 15,
            "all_15_map_post_rename": r1["tensors_mapped_post_rename"] == 15,
            "zero_unexpected_post_rename": r1["n_unexpected"] == 0,
            "zero_shape_mismatch_post_rename": r1["n_shape_mismatch"] == 0,
            "covers_all_vllm_load_targets": bool(r1["covers_all_targets"]),
            "no_missing_targets": r1["missing_targets"] == [],
            "raw_state_dict_not_loadable_as_is": (raw_map["n_ok"] == 1
                                                  and raw_map["n_unexpected"] == 14),
            "safetensors_round_trips_keys": bool(rt["keys_match"]),
            "safetensors_round_trips_shapes": bool(rt["shapes_match"]),
            "uniform_bf16_after_cast": bool(r1["all_bf16"] and rt["all_bf16"]),
            "nan_clean": bool(r1["nan_clean"]),
            "config_parses_through_autoconfig": bool(cfg_survived),
            "config_has_required_fields": all(f in config for f in REQUIRED_CONFIG_FIELDS),
            "config_nested_eagle_config_present": (
                isinstance(config.get("eagle_config"), dict)
                and config["eagle_config"].get("eagle_aux_hidden_state_layer_ids") == AUX_LAYER_IDS),
            "architectures_is_eagle3": config["architectures"] == ["Eagle3LlamaForCausalLM"],
            "vocab_divisible_by_64": VOCAB % 64 == 0,
            "fused_shard_shapes_consistent": bool(qkv_ok and gu_ok),
            "sha256_deterministic_across_runs": r1["sha256"] == r2["sha256"],
            "candidate_is_two_file_dir": (
                "model.safetensors" in candidate_files and "config.json" in candidate_files),
        }

    self_test_passes = all(checks.values())
    return {
        "checks": checks,
        "n_checks": len(checks),
        "n_checks_pass": sum(1 for v in checks.values() if v),
        "converter_self_test_passes": int(self_test_passes),
        "tensors_mapped_post_rename": r1["tensors_mapped_post_rename"],
        "sha256_synthetic": r1["sha256"],
        "config_backend": cfg_backend,
        "raw_map_ok": raw_map["n_ok"],
    }


# --------------------------------------------------------------------------- #
# Approval-request snippet (filled; references the artifact — does NOT publish).
# --------------------------------------------------------------------------- #
def approval_request_snippet(summary: dict[str, Any], source_desc: str) -> str:
    sha = summary["sha256"]
    return f"""# Approval request: HF job for eagle3-319-measured-read (in-repo {{2,21,39}} head)

**PR/branch:** #333 / `ubel/eagle3-safetensors-converter` (converter only — NOT a launch)
**What this card produced (CPU-only, 0 GPU):** a vLLM-loadable EAGLE-3 candidate dir
converted from {source_desc} via the #328 Finding-B rename + uniform-bf16 cast + config
emit. Two files, ready for the human to publish + smoke-test before the one #319 launch.

- `model.safetensors` sha256: `{sha}`
- tensors mapped post-rename: {summary['tensors_mapped_post_rename']}/15 (0 unexpected, 0 shape mismatch)
- artifact: `research/launch/eagle3_safetensors_converter/_candidate/`

**Human-owned next steps (NOT done here — audit blocker 1 `no_published_path`):**
1. Re-run this converter on the REAL `model_best.pt`:
   `python research/launch/eagle3_safetensors_converter/convert_eagle3_to_safetensors.py --in <path>/model_best.pt`
2. Publish the two-file `_candidate/` dir to
   `hf://buckets/gemma-challenge/gemma-senpai/weights/eagle3-inrepo-251head/`.
3. Set `DRAFTER_SHA256` to the printed sha256 in
   `submissions/fa2sw_precache_kenyan/manifest.json` and apply the ubel #322 §1 manifest
   delta (`method` mtp->eagle3, `SPECULATIVE_CONFIG.model` + `LOCAL_DRAFTER_DIR` +
   `DRAFTER_BUCKET` -> the eagle3 dir; keep `num_speculative_tokens:7`).
4. Local AWS A10G smoke (no HF Job): boot serve.py, confirm 0 missing/unexpected tensors,
   `/v1/models` 200, greedy token-identity, tiny PPL sane (ubel #322 §0 step 5).
5. Only after the smoke PASS + human approval on Issue #319: launch exactly one
   `a10g-small` measured read.

**Expected metric movement / risk:** this converter adds 0 TPS and changes no served file.
It de-risks the #319 launch (head conversion becomes one reviewed command). PPL/greedy
risk is deferred to the human smoke (step 4); a name/shape slip fails LOUDLY here on CPU.
"""


# --------------------------------------------------------------------------- #
# W&B logging (mirrors ubel #322/#299; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args: argparse.Namespace, payload: dict[str, Any]) -> list[str]:
    run_ids: list[str] = []
    if getattr(args, "no_wandb", False):
        return run_ids
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.append(repo)
    _w = sys.modules.get("wandb")
    if _w is not None and not hasattr(_w, "init"):
        del sys.modules["wandb"]
    try:
        import wandb as _wb

        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub wandb with no .init -> this venv lacks the wheel")
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-converter] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return run_ids

    st = payload["self_test"]
    try:
        run = init_wandb_run(
            job_type="analysis", agent="ubel",
            name=args.wandb_name or "ubel/eagle3-safetensors-converter",
            group=args.wandb_group,
            notes="CPU-only EAGLE-3 head -> launch-ready safetensors converter + synthetic "
                  "self-test (PR #333; closes #328 blockers 2+3). 0 GPU, 0 TPS, no publish.",
            tags=["eagle3", "safetensors-converter", "launch-prep", "0-gpu", "0-tps",
                  "issue-319", "pr-333"],
            config={"pr": 333, "issue": 319, "wandb_group": args.wandb_group,
                    "source_mode": payload["source_mode"]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-converter] wandb init failed (analysis unaffected): {exc}", flush=True)
        return run_ids
    if run is None:
        print("[eagle3-converter] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return run_ids

    summary: dict[str, Any] = {
        "converter_self_test_passes": int(st["converter_self_test_passes"]) if st else None,
        "tensors_mapped_post_rename": payload["conversion"]["tensors_mapped_post_rename"],
        "n_unexpected_post_rename": payload["conversion"]["n_unexpected"],
        "n_shape_mismatch_post_rename": payload["conversion"]["n_shape_mismatch"],
        "covers_all_vllm_load_targets": int(bool(payload["conversion"]["covers_all_targets"])),
        "all_bf16": int(bool(payload["conversion"]["all_bf16"])),
        "nan_clean": int(bool(payload["conversion"]["nan_clean"])),
        "n_checks": st["n_checks"] if st else None,
        "n_checks_pass": st["n_checks_pass"] if st else None,
        "tps_added_by_this_card": 0,
        "blockers_closed": 2,  # #328 blockers 2 (namespace) + 3 (container)
    }
    if st:
        summary.update({f"selftest_{k}": int(bool(v)) for k, v in st["checks"].items()})
    summary = {k: v for k, v in summary.items() if v is not None}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="eagle3_safetensors_converter_result",
                          artifact_type="analysis", data=payload)
        run_ids.append(getattr(run, "id", "") or "")
        print(f"[eagle3-converter] wandb run logged: {getattr(run, 'id', '?')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-converter] wandb summary/artifact skipped: {exc}", flush=True)
    finish_wandb(run)
    return [r for r in run_ids if r]


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--in", dest="in_path", default=None,
                     help="path to the real Eagle3DraftHead state_dict (.pt / model_best.pt)")
    src.add_argument("--synthetic-shapes", action="store_true",
                     help="build zero-tensors at the exact #328 inventory shapes (0 GPU, "
                          "no checkpoint) — the default when no --in is given")
    ap.add_argument("--out-dir", type=Path, default=CANDIDATE_DIR,
                    help="candidate output dir (default: ./_candidate)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the PRIMARY >=15-check self-validation (synthetic, 0 GPU)")
    ap.add_argument("--no-wandb", action="store_true", help="skip W&B logging")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="ubel/eagle3-safetensors-converter")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="eagle3-safetensors-converter")
    args = ap.parse_args(argv)

    synthetic = args.synthetic_shapes or (args.in_path is None)
    source_mode = "real_pt" if args.in_path else "synthetic_shapes"
    if args.in_path:
        source_desc = f"the real head state_dict `{args.in_path}`"
        state = load_state_dict(args.in_path)
    else:
        source_desc = "synthetic zero-tensors at the #328 inventory shapes (DRY RUN — "\
                      "not real weights; for human structural review)"
        state = build_synthetic_state_dict()

    bar = "=" * 78
    print(bar, flush=True)
    print("EAGLE-3 {2,21,39} head -> launch-ready safetensors converter (PR #333)", flush=True)
    print(f"source: {source_mode}  ->  out: {args.out_dir}", flush=True)
    print(bar, flush=True)

    conversion = convert(state, args.out_dir)
    print(f"[convert] {conversion['n_tensors']} tensors  "
          f"mapped_post_rename={conversion['tensors_mapped_post_rename']}/15  "
          f"unexpected={conversion['n_unexpected']}  shape_mismatch={conversion['n_shape_mismatch']}  "
          f"covers_all_targets={conversion['covers_all_targets']}", flush=True)
    print(f"[convert] uniform_bf16={conversion['all_bf16']}  nan_clean={conversion['nan_clean']}  "
          f"sha256={conversion['sha256']}", flush=True)
    print(f"[convert] wrote {conversion['safetensors_path']}", flush=True)
    print(f"[convert] wrote {conversion['config_path']}", flush=True)

    self_test = None
    if args.self_test:
        self_test = run_self_test()
        failed = [k for k, v in self_test["checks"].items() if not v]
        print("-" * 78, flush=True)
        print(f"[self-test] converter_self_test_passes={self_test['converter_self_test_passes']} "
              f"({self_test['n_checks_pass']}/{self_test['n_checks']} checks) "
              f"config_backend={self_test['config_backend']}", flush=True)
        if failed:
            print(f"[self-test] FAILED checks: {failed}", flush=True)
        else:
            print("[self-test] all checks PASS", flush=True)

    # Approval-request snippet (filled; references the artifact — does NOT publish).
    snippet = approval_request_snippet(conversion, source_desc)
    (args.out_dir / "APPROVAL_REQUEST.md").write_text(snippet, encoding="utf-8")

    payload: dict[str, Any] = {
        "card": "eagle3_safetensors_converter",
        "pr": 333, "author": "ubel",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "synthetic": synthetic,
        "no_publish": True, "no_bucket_write": True, "no_manifest_change": True,
        "no_hf_job": True, "no_served_file_change": True, "gpu_used": False,
        "closes_audit_blockers": ["weight_key_namespace_mismatch", "container_format"],
        "conversion": {k: v for k, v in conversion.items() if k != "mapping_rows"},
        "mapping_rows": conversion["mapping_rows"],
        "config": build_config(),
        "self_test": self_test,
    }
    nan_paths = _assert_finite(payload)
    payload["nan_clean_payload"] = not nan_paths

    # Log to W&B first so the run id can be embedded in the committed record below.
    run_ids = _maybe_log_wandb(args, payload)
    payload["wandb_run_ids"] = run_ids

    # _results.json is the committed machine-readable record (top-level, like #328);
    # the heavy/derived artifact (model.safetensors + config + sha + snippet) stays in
    # the gitignored _candidate/ dir, regenerable on demand. Written last so it carries
    # wandb_run_ids -> the PR diff record is self-contained and links to its W&B run.
    results_path = HERE / "_results.json"
    results_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[convert] wrote {results_path}", flush=True)
    print(f"[convert] wrote {args.out_dir / 'APPROVAL_REQUEST.md'}", flush=True)

    primary = self_test["converter_self_test_passes"] if self_test else None
    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": run_ids,
        "primary_metric": {"name": "converter_self_test_passes",
                           "value": primary if primary is not None else None},
        "test_metric": {"name": "tensors_mapped_post_rename",
                        "value": conversion["tensors_mapped_post_rename"]},
    }
    print("SENPAI-RESULT: " + json.dumps(marker), flush=True)

    if args.self_test:
        return 0 if self_test["converter_self_test_passes"] == 1 else 1
    return 0


def _assert_finite(obj: Any, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += _assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += _assert_finite(v, f"{path}[{i}]")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
