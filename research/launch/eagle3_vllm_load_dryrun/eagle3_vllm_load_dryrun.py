#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Static vLLM load dry-run: which of #328's 4 residual caveats close at 0 GPU? (PR #338).

ubel #333 (merged) closed #328 blockers 2 (`weight_key_namespace_mismatch`) and 3
(`container_format`) by emitting a vLLM-loadable two-file EAGLE-3 candidate dir. #328 §5
left **4 "verify-at-smoke" caveats** (C1-C4) it asserted are closeable only by the human's
post-publish §4 A10G GPU smoke:

  C1  config-field survival + class registration
  C2  fp32->bf16 inference numerics + live served greedy-identity
  C3  absent-`d2t` -> identity-map default
  C4  vLLM-fork version / schema pin

**Claim under test (PR #338):** C1, C3, C4 are *static* — class registration, param-name/
shape matching, code-path defaults, and config-schema compatibility, none of which need a
GPU forward. Only C2 (live numerics + served greedy-identity) genuinely needs the GPU smoke.
This card runs a CPU-only static load-readiness audit against the **in-repo vLLM fork**
(`vllm 0.22.1rc1.dev307`, `Eagle3LlamaForCausalLM` in `model_executor/models/llama_eagle3.py`
— the same fork that serves the deployed frontier) and marks each caveat CLOSED-AT-0-GPU vs
REQUIRES-GPU-SMOKE, shrinking the human's §4 smoke to only the irreducible C2 numerics check.

It validates against the **real fork source** (not a hand-port): it locates the installed
fork (no GPU, no model construction, no forward), `ast`-extracts the authoritative load
contract (the `_SPECULATIVE_DECODING_MODELS` registry entry, `LlamaModel.load_weights`'s
`stacked_params_mapping`, and the `__init__`/`load_weights`/`compute_logits` `d2t` logic),
reads the pinned version from the fork's `_version.py`, and asserts the merged #333 converter's
published key set maps 1:1 onto every vLLM load-target. The #333 converter module is reused as
the single source of truth for the published manifest (it imports stdlib-only at module level;
torch lives inside its functions), so this script needs **neither torch nor a vLLM import** and
runs under the plain `.venv` interpreter.

**0 GPU. NO model forward. NO checkpoint load. NO publish / NO bucket write / NO manifest
change / NO HF job / NO submission / NO served-file change.** Publishing the artifact + the §4
GPU smoke stay HUMAN-owned. This is a STATIC load-readiness audit only; it does not touch
emission, PPL, or served greedy-identity (those are the human's §4 smoke — caveat C2).

Reproduce (CPU-only, 0 GPU):
    cd target/ && .venv/bin/python research/launch/eagle3_vllm_load_dryrun/eagle3_vllm_load_dryrun.py \
        --self-test --wandb_group eagle3-load-dryrun --wandb_name ubel/eagle3-vllm-load-dryrun

PRIMARY metric : `vllm_load_dryrun_self_test_passes` (1 iff C1/C3/C4 all close at 0 GPU and
                 C2 is correctly held residual, with no check erroring).
TEST   metric  : `caveats_closed_at_0gpu` (int 0-4; expected 3 — C1/C3/C4, C2 stays GPU).
"""
from __future__ import annotations

import argparse
import ast
import glob
import importlib.util
import json
import re
import struct
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]  # .../target
CONVERTER_PY = (
    REPO_ROOT / "research/launch/eagle3_safetensors_converter/convert_eagle3_to_safetensors.py"
)
ONDISK_CANDIDATE = CONVERTER_PY.parent / "_candidate" / "model.safetensors"

# The fork the candidate must load under (the wheel pinned in fa2sw_precache_kenyan/manifest.json
# and read by #328 §2). The audit fails loudly if the located fork's _version.py disagrees.
VLLM_VERSION_TOKEN = "0.22.1rc1"
ARCH_NAME = "Eagle3LlamaForCausalLM"
FORK_MODULE = "llama_eagle3"

# Caveat C2 is the irreducible GPU-numerics residual: a CPU static audit cannot exercise the
# fp32->bf16 forward or the served greedy-token-identity contract (#192 HARD gate).
GPU_ONLY_CAVEAT = "C2"


# --------------------------------------------------------------------------- #
# Locate the installed in-repo vLLM fork WITHOUT importing/executing it (0 GPU,
# no CUDA probe). importlib.util.find_spec resolves the package path without
# running its __init__; a filesystem glob is the cross-interpreter fallback (the
# reproduce interpreter `.venv` has no vllm — the fork lives in the serving venv).
# --------------------------------------------------------------------------- #
def locate_fork() -> dict[str, Any]:
    info: dict[str, Any] = {"method": None, "vllm_dir": None, "version": None,
                            "eagle3_src": None, "registry_src": None, "version_src": None}
    vllm_dir: str | None = None
    try:
        spec = importlib.util.find_spec("vllm")  # does NOT exec vllm
        if spec is not None and spec.submodule_search_locations:
            cand = spec.submodule_search_locations[0]
            if (Path(cand) / "model_executor/models/llama_eagle3.py").exists():
                vllm_dir, info["method"] = cand, "find_spec"
    except (ImportError, ValueError, AttributeError):
        pass
    if vllm_dir is None:
        patterns = [
            "/tmp/senpai-venvs/*/lib/python*/site-packages/vllm",
            str(REPO_ROOT / ".venv/lib/python*/site-packages/vllm"),
            str(Path(sys.prefix) / "lib/python*/site-packages/vllm"),
            "/usr/lib/python*/*-packages/vllm",
            "/usr/local/lib/python*/*-packages/vllm",
        ]
        for pat in patterns:
            for cand in sorted(glob.glob(pat)):
                if (Path(cand) / "model_executor/models/llama_eagle3.py").exists():
                    vllm_dir, info["method"] = cand, "filesystem"
                    break
            if vllm_dir:
                break
    if vllm_dir is None:
        return info
    info["vllm_dir"] = vllm_dir
    info["eagle3_src"] = str(Path(vllm_dir) / "model_executor/models/llama_eagle3.py")
    info["registry_src"] = str(Path(vllm_dir) / "model_executor/models/registry.py")
    vfile = Path(vllm_dir) / "_version.py"
    if vfile.exists():
        info["version_src"] = str(vfile)
        text = vfile.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"__version__\s*=\s*version\s*=\s*['\"]([^'\"]+)['\"]", text)
        if not m:
            m = re.search(r"__version__\s*[:=].*?['\"]([^'\"]+)['\"]", text)
        if m:
            info["version"] = m.group(1)
    return info


# --------------------------------------------------------------------------- #
# Reuse the merged, reviewed #333 converter as the single source of truth for the
# published key manifest + config + load-map resolver (NO re-hardcoding -> no drift).
# It is stdlib-only at module scope; torch is imported lazily inside its functions.
# --------------------------------------------------------------------------- #
def load_converter() -> Any:
    spec = importlib.util.spec_from_file_location("eagle3_converter", str(CONVERTER_PY))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load converter module at {CONVERTER_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Tiny AST helpers — read the REAL fork contract as data (0 GPU, no execution).
# --------------------------------------------------------------------------- #
def _parse(path: str) -> tuple[ast.Module, str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return ast.parse(text), text


def assign_literal(tree: ast.AST, name: str) -> Any:
    """`ast.literal_eval` the first module/function-level `name = <literal>` assignment."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        return None
    return None


def find_class(tree: ast.AST, classname: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == classname:
            return node
    return None


def method_node(cls: ast.ClassDef | None, method: str) -> ast.FunctionDef | None:
    if cls is None:
        return None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == method:
            return node
    return None


def method_source(cls: ast.ClassDef | None, method: str, src_text: str) -> tuple[str, int | None]:
    node = method_node(cls, method)
    if node is None:
        return "", None
    seg = ast.get_source_segment(src_text, node)
    return (seg or ""), node.lineno


def _is_config_base(node: ast.AST) -> bool:
    """True for `self.config` or a bare `config` (the LlamaConfig), not `eagle_config` etc."""
    if isinstance(node, ast.Attribute) and node.attr == "config":
        return isinstance(node.value, ast.Name) and node.value.id == "self"
    return isinstance(node, ast.Name) and node.id == "config"


def collect_config_fields(tree: ast.AST) -> dict[str, list[str]]:
    """Every config field the fork READS. Direct `self.config.X` / `config.X` reads are
    REQUIRED (no default); `getattr(<config>, "X", default)` is OPTIONAL."""
    required: set[str] = set()
    optional: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)
                and _is_config_base(node.args[0])):
            (optional if len(node.args) >= 3 else required).add(node.args[1].value)
        elif (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
              and _is_config_base(node.value)):
            required.add(node.attr)
    return {"required": sorted(required), "optional": sorted(optional - required)}


# --------------------------------------------------------------------------- #
# Caveat checks — each returns a dict with `passed` and the evidence behind it.
# --------------------------------------------------------------------------- #
def check_registration(fork: dict[str, Any], conv: Any) -> dict[str, Any]:
    """C1a — the converter's `architectures` arch resolves through the fork registry to the
    EAGLE-3 class, and the class is defined in the fork source."""
    arch = conv.build_config()["architectures"][0]
    detail: dict[str, Any] = {"caveat": "registration", "arch_name": arch}
    if not fork.get("registry_src") or not fork.get("eagle3_src"):
        detail.update(passed=False, reason="fork source not located")
        return detail
    reg_tree, _ = _parse(fork["registry_src"])
    spec_models = assign_literal(reg_tree, "_SPECULATIVE_DECODING_MODELS") or {}
    entry = spec_models.get(arch)
    entry = tuple(entry) if isinstance(entry, (list, tuple)) else entry
    registry_ok = entry is not None and entry[0] == FORK_MODULE and entry[1] == ARCH_NAME
    eag_tree, _ = _parse(fork["eagle3_src"])
    class_defined = find_class(eag_tree, ARCH_NAME) is not None
    # Sibling arch aliases that also route to the class (informational).
    aliases = sorted(k for k, v in spec_models.items()
                     if (tuple(v) if isinstance(v, (list, tuple)) else v) == (FORK_MODULE, ARCH_NAME))
    detail.update(
        passed=bool(registry_ok and class_defined),
        registry_entry=list(entry) if isinstance(entry, tuple) else entry,
        registry_resolves=registry_ok,
        class_defined_in_fork=class_defined,
        n_registry_aliases=len(aliases),
        registry_aliases=aliases,
        registry_src=fork["registry_src"],
        note=(f"{arch} -> {entry} in _SPECULATIVE_DECODING_MODELS; class defined in {FORK_MODULE}.py"
              if registry_ok and class_defined else "registry/class resolution FAILED"),
    )
    return detail


def published_manifest(conv: Any, regen: bool) -> dict[str, Any]:
    """The #333 candidate's published (name, shape, dtype) set — the dry-run input.

    Prefers the on-disk #333 `_candidate/model.safetensors` header (the exact artifact #333
    emitted; deterministic sha256), else regenerates it via the converter when torch is
    available, else derives it analytically from the converter's reviewed constants. All three
    routes are equivalent by construction; whichever is used is recorded + cross-checked."""
    analytic = [(conv.published_name(n), list(s), d) for n, s, d in conv.SOURCE_INVENTORY]
    # The converter casts every tensor to bf16, so the published FILE is uniform bf16 while the
    # analytic source dtypes are mixed — match on (name, shape) and track bf16-uniformity apart.
    analytic_ns = sorted((n, s) for n, s, _ in analytic)
    src = "analytic_from_converter_constants"
    header_keys: list[str] | None = None

    def _read_header(path: Path) -> list[tuple[str, list[int], str]]:
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            hdr = json.loads(fh.read(n))
        dt = {"BF16": conv.BF16, "F32": conv.FP32, "F16": "float16"}
        return [(k, list(v["shape"]), dt.get(v["dtype"], v["dtype"]))
                for k, v in hdr.items() if k != "__metadata__"]

    tensors = analytic
    if regen:
        try:
            with tempfile.TemporaryDirectory() as td:
                conv.convert(conv.build_synthetic_state_dict(), Path(td) / "regen")
                tensors = _read_header(Path(td) / "regen" / "model.safetensors")
                src, header_keys = "regenerated_via_converter", [t[0] for t in tensors]
        except Exception as exc:  # noqa: BLE001 — torch absent under .venv; fall through
            src = f"analytic (regen unavailable: {type(exc).__name__})"
    elif ONDISK_CANDIDATE.exists():
        tensors = _read_header(ONDISK_CANDIDATE)
        src, header_keys = "ondisk_#333_candidate_header", [t[0] for t in tensors]

    # Cross-check: the route used must carry the analytic (name, shape) manifest exactly.
    matches_analytic = sorted((n, s) for n, s, _ in tensors) == analytic_ns
    # bf16-uniformity is only observable when a real file was read (header/regen route).
    all_bf16 = (all(str(d) in ("bfloat16", conv.BF16) for _, _, d in tensors)
                if header_keys is not None else None)
    return {"tensors": tensors, "source": src, "header_keys": header_keys,
            "matches_analytic": matches_analytic, "all_bf16": all_bf16, "n_tensors": len(tensors)}


def check_param_manifest(fork: dict[str, Any], conv: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    """C1b — the real fork `stacked_params_mapping` matches the converter's port, and every
    published tensor maps 1:1 onto a vLLM load-target with an exact shape (0 missing/unexpected)."""
    detail: dict[str, Any] = {"caveat": "param_manifest"}
    if not fork.get("eagle3_src"):
        detail.update(passed=False, reason="fork source not located")
        return detail
    eag_tree, _ = _parse(fork["eagle3_src"])
    real_stacked = assign_literal(eag_tree, "stacked_params_mapping")
    real_norm = [tuple(x) for x in real_stacked] if real_stacked else None
    port_norm = [tuple(x) for x in conv.STACKED]
    port_matches_fork = real_norm == port_norm

    tensors = manifest["tensors"]
    mapping = conv.check_mapping([(n, s) for n, s, _ in tensors])
    one_to_one = (mapping["n_ok"] == len(tensors)
                  and mapping["n_unexpected"] == 0
                  and mapping["n_shape_mismatch"] == 0
                  and bool(mapping["covers_all_targets"])
                  and mapping["missing_targets"] == [])
    detail.update(
        passed=bool(port_matches_fork and one_to_one and manifest["matches_analytic"]),
        real_stacked_params_mapping=real_stacked,
        port_matches_fork=port_matches_fork,
        tensors_total=len(tensors),
        tensors_mapped=mapping["n_ok"],
        n_unexpected=mapping["n_unexpected"],
        n_shape_mismatch=mapping["n_shape_mismatch"],
        load_targets_required=len(conv.required_targets()),
        covers_all_targets=bool(mapping["covers_all_targets"]),
        missing_targets=mapping["missing_targets"],
        candidate_source=manifest["source"],
        candidate_matches_analytic=manifest["matches_analytic"],
        candidate_all_bf16=manifest["all_bf16"],
        note=(f"{mapping['n_ok']}/{len(tensors)} published keys map 1:1 onto "
              f"{len(conv.required_targets())} vLLM load-targets; fork stacked_params_mapping "
              f"== converter port ({port_matches_fork}); uniform_bf16={manifest['all_bf16']}"),
    )
    return detail


def check_config_survival(fork: dict[str, Any], conv: Any) -> dict[str, Any]:
    """C1c — the 4 custom EAGLE fields survive an AutoConfig/LlamaConfig parse (top-level +
    nested `eagle_config` backstop), every converter REQUIRED_CONFIG_FIELD is emitted, and no
    fork-required config field is left unemitted."""
    detail: dict[str, Any] = {"caveat": "config_survival"}
    config = conv.build_config()
    survived, backend = conv.parse_config_through_autoconfig(config)
    required_emitted = all(f in config for f in conv.REQUIRED_CONFIG_FIELDS)
    nested_ok = (isinstance(config.get("eagle_config"), dict)
                 and config["eagle_config"].get("eagle_aux_hidden_state_layer_ids") is not None)

    ast_fields: dict[str, list[str]] = {"required": [], "optional": []}
    flagged_missing: list[str] = []
    if fork.get("eagle3_src"):
        eag_tree, _ = _parse(fork["eagle3_src"])
        ast_fields = collect_config_fields(eag_tree)
        emitted = set(config) | set(conv.REQUIRED_CONFIG_FIELDS)
        # Standard LlamaConfig fields carry constructor defaults -> present even if not emitted.
        standard = {"vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "head_dim", "intermediate_size", "rms_norm_eps",
                    "rope_theta", "max_position_embeddings", "attention_bias", "logit_scale",
                    "tie_word_embeddings", "draft_vocab_size", "torch_dtype"}
        flagged_missing = [f for f in ast_fields["required"] if f not in emitted and f not in standard]

    detail.update(
        passed=bool(survived and required_emitted and nested_ok and not flagged_missing),
        custom_fields_survive_autoconfig=bool(survived),
        autoconfig_backend=backend,
        required_config_fields_emitted=required_emitted,
        nested_eagle_config_backstop=nested_ok,
        fork_config_fields=ast_fields,
        fork_required_fields_not_emitted=flagged_missing,
        note=(f"custom EAGLE fields survive {backend}; {len(conv.REQUIRED_CONFIG_FIELDS)} required "
              f"fields emitted; 0 fork-required fields unemitted"
              if survived and required_emitted and not flagged_missing
              else "config survival/emission FAILED"),
    )
    return detail


def check_absent_d2t(fork: dict[str, Any], conv: Any) -> dict[str, Any]:
    """C3 — absent `draft_id_to_target_id` -> identity. Verified at its three real source
    locations: zeros-init in __init__, skip-on-absent in load_weights, identity scatter in
    compute_logits — plus draft_vocab == target_vocab so the identity scatter is full-coverage."""
    detail: dict[str, Any] = {"caveat": "absent_d2t"}
    if not fork.get("eagle3_src"):
        detail.update(passed=False, reason="fork source not located")
        return detail
    eag_tree, src_text = _parse(fork["eagle3_src"])
    cls = find_class(eag_tree, ARCH_NAME)
    init_src, init_ln = method_source(cls, "__init__", src_text)
    lw_src, lw_ln = method_source(cls, "load_weights", src_text)
    cl_src, cl_ln = method_source(cls, "compute_logits", src_text)

    init_zeros = ("draft_id_to_target_id" in init_src and "nn.Parameter" in init_src
                  and "torch.zeros" in init_src)
    skip_on_absent = ("includes_draft_id_mapping" in lw_src
                      and 'skip_substrs.append("draft_id_to_target_id")' in lw_src)
    identity_scatter = ("draft_id_to_target_id" in cl_src and "arange" in cl_src
                        and "+ self.draft_id_to_target_id" in cl_src)

    config = conv.build_config()
    vocab_identity = config["draft_vocab_size"] == config["vocab_size"]

    detail.update(
        passed=bool(init_zeros and skip_on_absent and identity_scatter and vocab_identity),
        init_zeros_default=init_zeros,
        load_weights_skips_when_absent=skip_on_absent,
        compute_logits_identity_scatter=identity_scatter,
        draft_vocab_equals_target_vocab=vocab_identity,
        vocab_size=config["vocab_size"],
        source_locations={
            "init_zeros": f"{FORK_MODULE}.py:{init_ln} ({ARCH_NAME}.__init__)",
            "skip_substrs": f"{FORK_MODULE}.py:{lw_ln} ({ARCH_NAME}.load_weights)",
            "identity_scatter": f"{FORK_MODULE}.py:{cl_ln} ({ARCH_NAME}.compute_logits)",
        },
        note=("draft_id_to_target_id zero-init + skip-on-absent + identity scatter confirmed; "
              f"draft_vocab==target_vocab=={config['vocab_size']} -> full-coverage identity"
              if init_zeros and skip_on_absent and identity_scatter and vocab_identity
              else "d2t identity default NOT fully confirmed in source"),
    )
    return detail


def check_version_pin(fork: dict[str, Any], conv: Any, registration: dict[str, Any]) -> dict[str, Any]:
    """C4 — record the fork version the candidate must load under, confirm the converted config
    schema registers under it (parse + arch resolution), and flag any required-but-unemitted field."""
    detail: dict[str, Any] = {"caveat": "version_pin"}
    version = fork.get("version")
    version_ok = bool(version) and VLLM_VERSION_TOKEN in version
    config = conv.build_config()
    survived, backend = conv.parse_config_through_autoconfig(config)
    arch_resolves = bool(registration.get("registry_resolves"))
    detail.update(
        passed=bool(version_ok and survived and arch_resolves),
        fork_version=version,
        expected_version_token=VLLM_VERSION_TOKEN,
        version_token_present=version_ok,
        version_src=fork.get("version_src"),
        config_schema_parses=bool(survived),
        autoconfig_backend=backend,
        arch_resolves_under_fork=arch_resolves,
        note=(f"candidate must load under vLLM {version}; config schema parses ({backend}) and "
              f"{ARCH_NAME} resolves in this fork's registry"
              if version_ok and survived and arch_resolves
              else "version/schema pin NOT confirmed"),
    )
    return detail


# --------------------------------------------------------------------------- #
# #328 caveat ledger (deliverable 5): map C1-C4 -> CLOSED-AT-0-GPU vs REQUIRES-GPU-SMOKE.
# --------------------------------------------------------------------------- #
def build_ledger(reg: dict, manifest: dict, cfg: dict, d2t: dict, ver: dict) -> dict[str, Any]:
    c1_closed = bool(reg["passed"] and manifest["passed"] and cfg["passed"])
    c3_closed = bool(d2t["passed"])
    c4_closed = bool(ver["passed"])
    ledger = {
        "C1": {
            "name": "config-field survival + class registration",
            "status": "CLOSED-AT-0-GPU" if c1_closed else "REQUIRES-GPU-SMOKE",
            "closed": c1_closed,
            "evidence": "registry resolution + 1:1 param-manifest map + AutoConfig field survival",
            "sub_checks": {"registration": reg["passed"], "param_manifest": manifest["passed"],
                           "config_survival": cfg["passed"]},
        },
        "C2": {
            "name": "fp32->bf16 inference numerics + live served greedy-identity",
            "status": "REQUIRES-GPU-SMOKE",
            "closed": False,
            "evidence": "irreducible: needs a live GPU forward + served greedy-token-identity "
                        "(#192 HARD gate); a CPU static audit cannot exercise it",
        },
        "C3": {
            "name": "absent-d2t -> identity-map default",
            "status": "CLOSED-AT-0-GPU" if c3_closed else "REQUIRES-GPU-SMOKE",
            "closed": c3_closed,
            "evidence": "zero-init + skip-on-absent + identity scatter, draft_vocab==target_vocab",
        },
        "C4": {
            "name": "vLLM-fork version / schema pin",
            "status": "CLOSED-AT-0-GPU" if c4_closed else "REQUIRES-GPU-SMOKE",
            "closed": c4_closed,
            "evidence": "fork _version.py pin + config schema parse + arch resolution",
        },
    }
    closed = [k for k, v in ledger.items() if v["closed"]]
    residual = [f"{k}: {v['name']}" for k, v in ledger.items() if not v["closed"]]
    return {"ledger": ledger, "caveats_closed_at_0gpu": len(closed),
            "closed_caveats": closed, "residual_for_gpu_smoke": residual}


# --------------------------------------------------------------------------- #
# REPORT.md (deliverable) — generated so it always matches the live result.
# --------------------------------------------------------------------------- #
def write_report(path: Path, payload: dict[str, Any]) -> None:
    led = payload["ledger"]["ledger"]
    checks = payload["checks"]
    fork = payload["fork"]
    n_closed = payload["ledger"]["caveats_closed_at_0gpu"]
    verdict = ("🟢 GREEN — C1/C3/C4 CLOSED at 0 GPU; only C2 (GPU numerics) remains"
               if payload["self_test_passes"] and n_closed == 3
               else "🟡 PARTIAL — not all of C1/C3/C4 closed; see residual set")

    def row(cid: str) -> str:
        v = led[cid]
        mark = "✅ CLOSED-AT-0-GPU" if v["closed"] else (
            "🖥️ REQUIRES-GPU-SMOKE" if cid == GPU_ONLY_CAVEAT else "❌ NOT CLOSED")
        return f"| {cid} | {v['name']} | {mark} | {v['evidence']} |"

    d2t = checks["absent_d2t"]
    locs = d2t.get("source_locations", {})
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: 2026 CoreWeave, Inc.",
        "SPDX-License-Identifier: Apache-2.0",
        "SPDX-PackageName: senpai",
        "-->",
        "",
        "# Static vLLM load dry-run — which of #328's 4 residual caveats close at 0 GPU?",
        "",
        f"**PR:** #338 · **Author:** ubel · **Issue:** #319 · **Generated:** {payload['generated_utc']}",
        f"· **W&B group:** `eagle3-load-dryrun`",
        "",
        "**0-GPU static load-readiness audit. NO model forward, NO checkpoint load, NO publish,**",
        "**NO bucket write, NO manifest/served-file change, NO HF job, NO submission, NO GPU.**",
        "",
        f"Reproduce: `cd target/ && .venv/bin/python {Path(*HERE.parts[-3:])}/eagle3_vllm_load_dryrun.py --self-test`",
        "",
        "---",
        "",
        f"## Verdict: {verdict}",
        "",
        f"`vllm_load_dryrun_self_test_passes = {payload['self_test_passes']}` · "
        f"`caveats_closed_at_0gpu = {n_closed}` (of 4; C2 is GPU-only by construction)",
        "",
        f"Audited fork: **vLLM `{fork.get('version')}`** (`{ARCH_NAME}` in "
        f"`model_executor/models/{FORK_MODULE}.py`), located via `{fork.get('method')}`. This is the",
        "same fork that serves the deployed frontier; the candidate must load under it. The audit",
        "reads the **real fork source** (registry dict, `stacked_params_mapping`, the `d2t` logic) as",
        "data — it does not construct the model or run a forward.",
        "",
        "## #328 caveat ledger (deliverable 5)",
        "",
        "| # | caveat | status | evidence |",
        "|---|---|---|---|",
        row("C1"), row("C2"), row("C3"), row("C4"),
        "",
        "**Residual set the human's §4 A10G smoke must still cover:**",
    ]
    for r in payload["ledger"]["residual_for_gpu_smoke"]:
        lines.append(f"- {r}")
    lines += [
        "",
        "C2 stays residual by construction: the fp32→bf16 forward numerics and the served",
        "greedy-token-identity contract (#192 HARD gate) cannot be exercised without a live GPU",
        "forward. Everything else needed to *load* the head is verified statically below.",
        "",
        "---",
        "",
        "## Step 1 — registration / class resolution (C1)",
        "",
        f"- `architectures[0]` = `{checks['registration']['arch_name']}` resolves through the fork's",
        f"  `_SPECULATIVE_DECODING_MODELS` to `{checks['registration'].get('registry_entry')}`"
        f" (`registry_resolves={checks['registration'].get('registry_resolves')}`).",
        f"- Class defined in fork source: `{checks['registration'].get('class_defined_in_fork')}`;"
        f" {checks['registration'].get('n_registry_aliases')} registry aliases route to it.",
        "",
        "## Step 2 — param-manifest 1:1 (C1)",
        "",
        f"- Real fork `stacked_params_mapping` == #333 converter port: "
        f"`{checks['param_manifest'].get('port_matches_fork')}`.",
        f"- `{checks['param_manifest'].get('tensors_mapped')}`/"
        f"`{checks['param_manifest'].get('tensors_total')}` published keys map 1:1 onto "
        f"`{checks['param_manifest'].get('load_targets_required')}` vLLM load-targets "
        f"(`{checks['param_manifest'].get('n_unexpected')}` unexpected, "
        f"`{checks['param_manifest'].get('n_shape_mismatch')}` shape-mismatch, "
        f"`{len(checks['param_manifest'].get('missing_targets', []))}` missing).",
        f"- Dry-run input: `{checks['param_manifest'].get('candidate_source')}` "
        f"(matches analytic manifest: `{checks['param_manifest'].get('candidate_matches_analytic')}`).",
        "",
        "## Step 3 — absent-`d2t` → identity (C3)",
        "",
        f"- zero-init default: `{locs.get('init_zeros')}`",
        f"- skip-when-absent: `{locs.get('skip_substrs')}`",
        f"- identity scatter: `{locs.get('identity_scatter')}`",
        f"- `draft_vocab == target_vocab == {d2t.get('vocab_size')}` → the identity scatter is",
        "  full-coverage (no silent token remap). Absent `d2t` is therefore safe, not a blocker.",
        "",
        "## Step 4 — version / schema pin (C4)",
        "",
        f"- Fork version (from `_version.py`): `{checks['version_pin'].get('fork_version')}` "
        f"(token `{VLLM_VERSION_TOKEN}` present: `{checks['version_pin'].get('version_token_present')}`).",
        f"- Converted config schema parses via `{checks['version_pin'].get('autoconfig_backend')}` "
        f"and `{ARCH_NAME}` resolves under this fork.",
        f"- Fork-required config fields not emitted by the converter: "
        f"`{checks['config_survival'].get('fork_required_fields_not_emitted')}` (empty = none).",
        "",
        "---",
        "",
        "## Honesty note",
        "",
        "This is a STATIC load-readiness audit, not a runtime check. It does not touch emission,",
        "PPL, or served greedy-identity (those are the human's §4 GPU smoke = C2). It validates",
        "against the real installed fork source by reading its load contract as data; it never",
        "constructs the model, loads a checkpoint, or runs a forward. Publishing the artifact and",
        "running the §4 smoke stay HUMAN-owned.",
        "",
        "## Public evidence used",
        "",
        "- **ubel #328 / `eagle3_ckpt_publish_readiness/REPORT.md` (`27y5xxce`)** — the §5 C1-C4",
        "  'verify-at-smoke' caveats this card statically closes; this card operationalizes that §5.",
        "- **ubel #333 / `eagle3_safetensors_converter` (`quzi85y0`)** — the converter whose published",
        "  manifest + config are reused here as the dry-run input (single source of truth).",
        f"- **vLLM `{fork.get('version')}` `{FORK_MODULE}.py`** — the `{ARCH_NAME}` / `LlamaModel` load",
        "  contract read as the authoritative target (registry, `stacked_params_mapping`, `d2t` logic).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# W&B logging (mirrors ubel #333; never fatal).
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args: argparse.Namespace, payload: dict[str, Any]) -> list[str]:
    if getattr(args, "no_wandb", False):
        return []
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
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
        print(f"[eagle3-load-dryrun] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return []
    try:
        run = init_wandb_run(
            job_type="analysis", agent="ubel",
            name=args.wandb_name or "ubel/eagle3-vllm-load-dryrun",
            group=args.wandb_group,
            notes="CPU-only static vLLM load dry-run for the {2,21,39} EAGLE-3 head (PR #338). "
                  "Marks #328's 4 residual caveats CLOSED-AT-0-GPU vs REQUIRES-GPU-SMOKE. "
                  "0 GPU, 0 TPS, no forward, no publish.",
            tags=["eagle3", "vllm-load-dryrun", "launch-prep", "0-gpu", "0-tps",
                  "issue-319", "pr-338"],
            config={"pr": 338, "issue": 319, "wandb_group": args.wandb_group,
                    "vllm_version": payload["fork"].get("version"),
                    "candidate_source": payload["checks"]["param_manifest"].get("candidate_source")},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-load-dryrun] wandb init failed (analysis unaffected): {exc}", flush=True)
        return []
    if run is None:
        print("[eagle3-load-dryrun] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return []
    summary: dict[str, Any] = {
        "vllm_load_dryrun_self_test_passes": payload["self_test_passes"],
        "caveats_closed_at_0gpu": payload["ledger"]["caveats_closed_at_0gpu"],
        "tps_added_by_this_card": 0,
        "caveat_C1_closed": int(payload["ledger"]["ledger"]["C1"]["closed"]),
        "caveat_C2_closed": int(payload["ledger"]["ledger"]["C2"]["closed"]),
        "caveat_C3_closed": int(payload["ledger"]["ledger"]["C3"]["closed"]),
        "caveat_C4_closed": int(payload["ledger"]["ledger"]["C4"]["closed"]),
    }
    summary.update({f"check_{k}_passed": int(bool(v.get("passed")))
                    for k, v in payload["checks"].items()})
    run_ids: list[str] = []
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="eagle3_vllm_load_dryrun_result",
                          artifact_type="analysis", data=payload)
        run_ids.append(getattr(run, "id", "") or "")
        print(f"[eagle3-load-dryrun] wandb run logged: {getattr(run, 'id', '?')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[eagle3-load-dryrun] wandb summary/artifact skipped: {exc}", flush=True)
    finish_wandb(run)
    return [r for r in run_ids if r]


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", "--self_test", dest="self_test", action="store_true",
                    help="exit non-zero unless C1/C3/C4 all close at 0 GPU (self-test gate)")
    ap.add_argument("--regen", action="store_true",
                    help="regenerate the #333 candidate via the converter (needs torch) instead "
                         "of reading the on-disk header")
    ap.add_argument("--no-wandb", "--no_wandb", dest="no_wandb", action="store_true",
                    help="skip W&B logging")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="ubel/eagle3-vllm-load-dryrun")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="eagle3-load-dryrun")
    ap.add_argument("--out", default=None, help="path for _results.json (default: alongside script)")
    args = ap.parse_args(argv)

    bar = "=" * 80
    print(bar, flush=True)
    print("Static vLLM load dry-run — #328 C1-C4 caveat ledger (PR #338, 0 GPU)", flush=True)
    print(bar, flush=True)

    fork = locate_fork()
    conv = load_converter()
    if not fork.get("eagle3_src"):
        print("[eagle3-load-dryrun] WARNING: in-repo vLLM fork not located — C1/C3/C4 cannot "
              "close (honest failure; the source must be present to audit the load path).",
              flush=True)
    else:
        print(f"[fork] vLLM {fork.get('version')} via {fork.get('method')}: {fork.get('vllm_dir')}",
              flush=True)

    manifest = published_manifest(conv, regen=args.regen)
    registration = check_registration(fork, conv)
    param_manifest = check_param_manifest(fork, conv, manifest)
    config_survival = check_config_survival(fork, conv)
    absent_d2t = check_absent_d2t(fork, conv)
    version_pin = check_version_pin(fork, conv, registration)

    checks = {
        "registration": registration,
        "param_manifest": param_manifest,
        "config_survival": config_survival,
        "absent_d2t": absent_d2t,
        "version_pin": version_pin,
    }
    ledger = build_ledger(registration, param_manifest, config_survival, absent_d2t, version_pin)

    # self-test PASSES iff the 3 claimed-static caveats all close (C2 is residual by construction)
    # and no check raised. caveats_closed_at_0gpu counts C1/C3/C4 (max 3).
    self_test_passes = int(ledger["caveats_closed_at_0gpu"] == 3)

    print("-" * 80, flush=True)
    for cid in ("C1", "C2", "C3", "C4"):
        v = ledger["ledger"][cid]
        print(f"  [{v['status']:>20}] {cid}: {v['name']}", flush=True)
    print("-" * 80, flush=True)
    for name, c in checks.items():
        flag = "PASS" if c.get("passed") else "FAIL"
        print(f"  ({flag}) {name}: {c.get('note', c.get('reason', ''))}", flush=True)
    print("-" * 80, flush=True)
    print(f"caveats_closed_at_0gpu={ledger['caveats_closed_at_0gpu']}/4 (C2 GPU-only)  "
          f"self_test_passes={self_test_passes}", flush=True)
    if ledger["residual_for_gpu_smoke"]:
        print(f"residual for §4 GPU smoke: {ledger['residual_for_gpu_smoke']}", flush=True)
    print(bar, flush=True)

    payload: dict[str, Any] = {
        "card": "eagle3_vllm_load_dryrun",
        "pr": 338, "issue": 319, "author": "ubel",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_used": False, "model_forward": False, "checkpoint_loaded": False,
        "no_publish": True, "no_bucket_write": True, "no_manifest_change": True,
        "no_hf_job": True, "no_served_file_change": True,
        "fork": fork,
        "candidate_manifest": {k: v for k, v in manifest.items() if k != "tensors"},
        "checks": checks,
        "ledger": ledger,
        "self_test_passes": self_test_passes,
    }

    run_ids = maybe_log_wandb(args, payload)
    payload["wandb_run_ids"] = run_ids

    out_path = Path(args.out) if args.out else HERE / "_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[eagle3-load-dryrun] wrote {out_path}", flush=True)
    report_path = HERE / "REPORT.md"
    write_report(report_path, payload)
    print(f"[eagle3-load-dryrun] wrote {report_path}", flush=True)

    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": run_ids,
        "primary_metric": {"name": "vllm_load_dryrun_self_test_passes", "value": self_test_passes},
        "test_metric": {"name": "caveats_closed_at_0gpu",
                        "value": ledger["caveats_closed_at_0gpu"]},
    }
    print("SENPAI-RESULT: " + json.dumps(marker), flush=True)

    if args.self_test and self_test_passes != 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
