#!/usr/bin/env python
"""Build the g32-locus recovery checkpoint for PR #720 (#319 self-consistency).

Config (advisor Gap-2b, the fern #713-priced contiguous band):
  - language_model layers **14-27** (inclusive): int4 **group_size=32**  (118 modules)
  - everything else quantized: int4 **group_size=128**                   (225 modules)
  - lm_head: int4 g128, **untied**                                        (1 module)
  - tie_word_embeddings = False

The two int4 source checkpoints are byte-compatible (same module set, same
weight_packed *shape*, identical weight_shape); only the per-group ``weight_scale``
granularity and the quantized integer values differ between g32 and g128. So the
build is pure tensor surgery -- no re-quantization, no dense bf16 source needed:

  base   = anchor ``int4_g128_lmhead`` (g128 everywhere + untied int4 lm_head)
  override = for each of the 118 L14-27 modules, swap in the **g32**
             ``weight_packed`` + ``weight_scale`` from the official QAT-CT
             checkpoint (google/gemma-4-E4B-it-qat-w4a16-ct, g32).

The config is rewritten from the anchor's single ``['Linear']`` body group into
three explicit groups (g128-body / g32-locus / lm_head) with explicit per-module
target lists so each module resolves to exactly one group (no ``Linear`` catch-all
ambiguity). Mixed group-size Marlin serving was proven by land #708.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ANCHOR = Path("/workspace/gemma_build/int4_g128_lmhead")
CT_G32 = Path(
    "/senpai-run/home/student-land/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
DST = Path("/workspace/gemma_build/g32_locus")
LOCUS_LAYERS = set(range(14, 28))  # L14..L27 inclusive
LAYER_RE = re.compile(r"language_model\.layers\.(\d+)\.")


def _free_gib(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024**3


def _is_locus(module_name: str) -> bool:
    m = LAYER_RE.search(module_name)
    return bool(m) and int(m.group(1)) in LOCUS_LAYERS


# L14-27 inclusive, matched against vLLM's module prefix (which contains
# ``layers.<N>.``). ``re:`` targets are preserved verbatim by
# CompressedTensorsConfig.apply_vllm_mapper; explicit dotted HF paths are NOT --
# they get remapped and then fail to match the queried prefix, leaving the module
# unquantized (KeyError on weight_packed at load). So the locus MUST be a regex.
LOCUS_REGEX = r"re:.*layers\.(1[4-9]|2[0-7])\..*"


def build_config(anchor_cfg: dict) -> dict:
    cfg = json.loads(json.dumps(anchor_cfg))  # deep copy
    cfg["tie_word_embeddings"] = False
    qc = cfg["quantization_config"]
    # Template a weights spec off the anchor's body group (int4 sym group).
    base_group = json.loads(json.dumps(qc["config_groups"]["group_0"]))

    def mk(group_size: int, targets):
        g = json.loads(json.dumps(base_group))
        g["weights"]["group_size"] = group_size
        g["targets"] = targets
        return g

    # find_matched_target tries a NAME/regex match across all targets first, then
    # falls back to the ``Linear`` CLASS match -- so the locus regex (name match)
    # wins over the ``Linear`` catch-all (class match) for L14-27, and everything
    # else falls through to ``Linear`` g128. lm_head matches its own regex.
    qc["config_groups"] = {
        "group_0": mk(128, ["Linear"]),              # everything int4 g128 by class
        "group_1": mk(32, [LOCUS_REGEX]),            # L14-27 int4 g32 (regex override)
        "group_2": mk(128, ["re:.*lm_head"]),        # untied lm_head int4 g128
    }
    # ignore list unchanged (vision tower etc. stay bf16)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", type=Path, default=ANCHOR)
    ap.add_argument("--ct", type=Path, default=CT_G32)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    from safetensors import safe_open
    from safetensors.torch import save_file

    if not args.anchor.exists():
        raise SystemExit(f"anchor not found: {args.anchor}")
    if not args.ct.exists():
        raise SystemExit(f"ct not found: {args.ct}")
    if args.dst.exists():
        if not args.force:
            raise SystemExit(f"dst exists: {args.dst} (use --force)")
        import shutil
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True)

    anchor_st = args.anchor / "model.safetensors"
    ct_st = args.ct / "model.safetensors"

    # --- enumerate modules ---
    with safe_open(anchor_st, framework="pt") as f:
        anchor_keys = list(f.keys())
    quant_mods = sorted(k[: -len(".weight_packed")] for k in anchor_keys if k.endswith(".weight_packed"))
    locus_mods = [m for m in quant_mods if _is_locus(m)]
    lmhead_mods = [m for m in quant_mods if m.endswith("lm_head")]
    body_mods = [m for m in quant_mods if m not in set(locus_mods) and m not in set(lmhead_mods)]
    print(f"[buildg32] quant modules: {len(quant_mods)} = locus {len(locus_mods)} + body {len(body_mods)} + lmhead {len(lmhead_mods)}", flush=True)
    assert len(locus_mods) == 118, f"expected 118 locus modules, got {len(locus_mods)}"
    assert len(quant_mods) == len(locus_mods) + len(body_mods) + len(lmhead_mods)

    # --- disk guard ---
    size_gib = anchor_st.stat().st_size / 1024**3
    free = _free_gib(args.dst)
    print(f"[buildg32] anchor safetensors {size_gib:.2f} GiB, free {free:.2f} GiB", flush=True)
    if free < size_gib + 1.5:
        raise SystemExit(f"insufficient disk: need ~{size_gib + 1.5:.1f} GiB, have {free:.1f} GiB")

    # --- config surgery ---
    anchor_cfg = json.loads((args.anchor / "config.json").read_text())
    new_cfg = build_config(anchor_cfg)
    (args.dst / "config.json").write_text(json.dumps(new_cfg, indent=2))
    print(f"[buildg32] wrote config.json: group_0=Linear g128, group_1={LOCUS_REGEX} g32 (covers {len(locus_mods)} locus mods), group_2=lm_head g128, tie=False", flush=True)

    # --- aux files (everything except config.json + model.safetensors) ---
    for p in sorted(args.anchor.iterdir()):
        if p.name in ("config.json", "model.safetensors"):
            continue
        (args.dst / p.name).symlink_to(p.resolve())
        print(f"[buildg32] symlink {p.name}", flush=True)

    # --- tensor surgery: base=anchor, override locus weight_packed+weight_scale from ct(g32) ---
    locus_set = set(locus_mods)
    override_packed = {m + ".weight_packed" for m in locus_set}
    override_scale = {m + ".weight_scale" for m in locus_set}

    tensors = {}
    with safe_open(anchor_st, framework="pt") as fa:
        meta = fa.metadata() or {}
        for k in fa.keys():
            tensors[k] = fa.get_tensor(k)
    n_over = 0
    with safe_open(ct_st, framework="pt") as fc:
        ct_keys = set(fc.keys())
        for k in sorted(override_packed | override_scale):
            if k not in ct_keys:
                raise SystemExit(f"ct missing expected tensor: {k}")
            tensors[k] = fc.get_tensor(k)
            n_over += 1
    print(f"[buildg32] overrode {n_over} tensors ({len(override_packed)} packed + {len(override_scale)} scale) from g32 ct", flush=True)

    # sanity: a locus scale should now have g32 granularity (4x groups of anchor g128)
    probe = "model.language_model.layers.14.self_attn.q_proj.weight_scale"
    print(f"[buildg32] probe {probe}: shape={tuple(tensors[probe].shape)} (g32 expected wider than g128)", flush=True)

    out = args.dst / "model.safetensors"
    save_file(tensors, str(out), metadata=meta)
    print(f"[buildg32] wrote {out} ({out.stat().st_size/1024**3:.2f} GiB)", flush=True)
    print(f"[buildg32] DONE -> {args.dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
