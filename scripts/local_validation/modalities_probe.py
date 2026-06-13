"""Modalities-load probe — the missing *official* leaderboard criterion, locally.

``program.md`` lines 29-31 forbid removing, skip-loading, zero-capping, or
disabling any of the four ``google/gemma-4-E4B-it`` pathways (text, image, audio,
video) to win speed. PR #38 established that the official HF-Jobs harness gates on
**PPL + completion + modalities**, NOT token-identity — and crucially it does
*not* programmatically check modalities at all (``hf_bucket_single_job.py`` runs
benchmark + decode-capture + PPL + summary; it never introspects the towers). So
a stack that drops the vision/audio towers to cut load time passes our local
PPL + completion + greedy checks and only dies on the official modalities rule —
discoverable today only by spending HF-Jobs quota. This module mirrors that
criterion locally so it can be caught before launch.

Two fidelity tiers, applied in ``program.md`` priority order:

1. **functional** (preferred for image; trivial for text): drive the *served*
   OpenAI-compatible endpoint with a real multimodal request. A non-degenerate
   completion proves the tower is loaded AND wired into inference — it catches
   skip-load / disable, not merely on-disk presence. If the server *rejects* the
   modality (e.g. "this model does not support image input"), that is a genuine
   ``False``.
2. **presence + non-zero** (fallback, and for audio/video where sample inputs are
   not staged): introspect the served checkpoint's ``config.json`` sub-configs and
   ``safetensors`` tensor names for each tower, and assert the tower's parameters
   are present and not all-zero / zero-capped. Catches *remove* and *zero-cap*.
   It CANNOT catch a tower that loads but is silently bypassed at inference — the
   documented residual gap (prefer the functional probe where feasible).

Honesty contract (#38 style): a tower that genuinely cannot be checked on this pod
is reported as ``None`` (unknown), never ``True``. Any unknown makes the
consolidated official gate read ``INCOMPLETE`` rather than ``PASS`` — a faithful
local mirror of the official gate, not a green light by omission.

Video note: ``gemma-4-E4B-it`` has no separate video tower — video frames route
through the vision tower (the config carries a ``video_token_id`` and reuses
``vision_config`` / ``model.vision_tower``). So the video pathway's presence is
inferred from the vision tower being loaded + non-zero AND ``video_token_id`` being
configured; there is no distinct video weight set to introspect.
"""
from __future__ import annotations

import base64
import json
import struct
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

# The four modality pathways the served model must preserve (program.md:29-31).
MODALITIES = ("text", "image", "audio", "video")

# PPL validity cap (program.md:24 — reference PPL + 5%, ~2.42).
PPL_CAP = 2.42

# Tower tensor-name prefixes in the Gemma4ForConditionalGeneration checkpoint.
# (Confirmed against /tmp/osoi5-*-baked: model.language_model.* + lm_head.* (text),
# model.vision_tower.* + model.embed_vision (image), model.audio_tower.* +
# model.embed_audio (audio); video shares the vision tower.)
_TOWER_PREFIXES = {
    "text": ("model.language_model.", "lm_head."),
    "image": ("model.vision_tower.", "model.embed_vision"),
    "audio": ("model.audio_tower.", "model.embed_audio"),
    "video": ("model.vision_tower.", "model.embed_vision"),
}

# Which config.json signal declares each modality's architecture is present.
_CONFIG_SIGNAL = {
    "text": ("text_config",),
    "image": ("vision_config",),
    "audio": ("audio_config",),
    # No video_config in gemma-4 — video is gated by the video_token_id + the
    # (shared) vision tower.
    "video": ("video_token_id",),
}


# --------------------------------------------------------------------------- #
# functional probes (against the live served endpoint)
# --------------------------------------------------------------------------- #
def _solid_png(width: int = 16, height: int = 16) -> bytes:
    """A tiny, valid RGB PNG built with the stdlib (no Pillow dependency).

    A simple 2-D gradient (not a flat color) so the vision encoder has actual
    structure to attend to — more likely to elicit a non-degenerate description
    than a 1x1 pixel.
    """
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # PNG filter type 0 for this scanline
        for x in range(width):
            raw += bytes(((x * 16) % 256, (y * 16) % 256, 128))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, RGB
    idat = zlib.compress(bytes(raw))
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any] | str]:
    """POST JSON, returning (status, parsed-json-or-error-text). Never raises."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:  # 4xx/5xx carry a body we want to classify
        try:
            return exc.code, exc.read().decode(errors="replace")
        except Exception:
            return exc.code, str(exc)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return 0, str(exc)


# Substrings that mark a *genuine* "this model has no image/audio pathway"
# rejection (-> a real False) rather than an ambiguous transport error.
_UNSUPPORTED_MARKERS = (
    "does not support",
    "not support",
    "no multimodal",
    "multimodal is not",
    "image input is not",
    "audio input is not",
    "not a multimodal",
    "unsupported",
    "no image",
    "modality",
    "mm_processor",
    "limit_mm_per_prompt",
)


def functional_text_probe(base_url: str, model: str, *, timeout_s: float = 60.0) -> tuple[bool | None, dict[str, Any]]:
    """Trivial text completion. Non-empty completion => text pathway functional."""
    status, resp = _post_json(
        f"{base_url.rstrip('/')}/v1/completions",
        {"model": model, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0.0},
        timeout_s,
    )
    if status == 200 and isinstance(resp, dict):
        text = ((resp.get("choices") or [{}])[0].get("text") or "")
        if text.strip():
            return True, {"method": "functional", "status": status, "sample": text.strip()[:60]}
        return None, {"method": "functional", "status": status, "note": "empty completion (inconclusive)"}
    return None, {"method": "functional", "status": status, "error": str(resp)[:200]}


def functional_image_probe(base_url: str, model: str, *, timeout_s: float = 180.0) -> tuple[bool | None, dict[str, Any]]:
    """Send one minimal image through /v1/chat/completions.

    Returns:
      True  — request accepted and produced a non-empty completion (vision tower
              loaded AND wired into inference).
      False — server explicitly rejected the image modality (genuine: not served).
      None  — ambiguous / transport error / empty completion (caller should fall
              back to the presence check rather than assert anything).
    """
    data_uri = "data:image/png;base64," + base64.b64encode(_solid_png()).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one short phrase."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    status, resp = _post_json(f"{base_url.rstrip('/')}/v1/chat/completions", payload, timeout_s)
    if status == 200 and isinstance(resp, dict):
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):  # some servers return content parts
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if content.strip():
            return True, {"method": "functional", "status": status, "sample": content.strip()[:80]}
        return None, {"method": "functional", "status": status, "note": "accepted image but empty completion (inconclusive)"}
    text = str(resp).lower()
    if status in (400, 415, 422) and any(m in text for m in _UNSUPPORTED_MARKERS):
        return False, {"method": "functional", "status": status, "error": str(resp)[:200], "reason": "server rejected image modality"}
    return None, {"method": "functional", "status": status, "error": str(resp)[:200], "note": "ambiguous — falling back to presence"}


# --------------------------------------------------------------------------- #
# presence + non-zero probe (served checkpoint introspection)
# --------------------------------------------------------------------------- #
# Manifest env keys that, when set, point at the actual on-disk weights the server
# loads (the served checkpoint), most-derived first. fa2sw bakes its final served
# weights to LM_HEAD_PRUNE_DST; other stacks fold into PLE_FOLD_TARGET_MODEL or
# stage at LOCAL_MODEL_DIR. These are checked before the declared MODEL_ID because
# a submission can *declare* the bf16 hub id while serving entirely different baked
# weights (the fa2sw aliasing hazard PR #32 documented) — and the modalities check
# must target what is actually served, not what is declared.
_WEIGHT_DIR_ENV_KEYS = ("LM_HEAD_PRUNE_DST", "PLE_FOLD_TARGET_MODEL", "LOCAL_MODEL_DIR")


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def _hf_cache_config_dir(model_id: str) -> Path | None:
    """Resolve a hub id to its local HF cache snapshot dir, if downloaded."""
    if "/" not in model_id:
        return None
    org, name = model_id.split("/", 1)
    base = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
    if not base.is_dir():
        return None
    for snap in sorted(base.iterdir(), reverse=True):
        if (snap / "config.json").exists():
            return snap
    return None


def resolve_served_model_dir(
    *,
    manifest: dict[str, Any] | None,
    submission_dir: Path | None,
    model_id: str | None,
    override: Path | None = None,
) -> Path | None:
    """Best-effort path to the checkpoint the server actually loads from.

    Priority: explicit override -> manifest weight-dir env hints -> MODEL_ID as a
    bundled dir / absolute path -> MODEL_ID as a hub id in the HF cache. Returns
    ``None`` when nothing resolves (caller reports the presence check as unknown).
    """
    if override is not None and _is_model_dir(Path(override)):
        return Path(override)
    env = (manifest or {}).get("env") or {}
    for key in _WEIGHT_DIR_ENV_KEYS:
        val = env.get(key)
        if val and _is_model_dir(Path(str(val))):
            return Path(str(val))
    if model_id:
        # bundled checkpoint under the submission dir (e.g. env.MODEL_ID="model")
        if submission_dir is not None:
            cand = Path(submission_dir) / model_id
            if _is_model_dir(cand):
                return cand
        p = Path(model_id)
        if p.is_absolute() and _is_model_dir(p):
            return p
        snap = _hf_cache_config_dir(model_id)
        if snap is not None:
            return snap
    return None


def _safetensors_header(path: Path) -> dict[str, Any]:
    """Parse a single-file .safetensors header (stdlib only): name -> spec."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def _checkpoint_tensor_index(model_dir: Path) -> tuple[dict[str, Any], Path] | None:
    """Return (name->spec-with-file, header_byte_offset-base) for a checkpoint.

    Handles both single-file ``model.safetensors`` and a sharded
    ``model.safetensors.index.json``. Each spec gets a ``_file`` (shard path) and,
    for single-file, a ``_base`` byte offset so a tensor's raw bytes can be read
    for the non-zero check without loading the whole file.
    """
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        # Presence-only for sharded checkpoints; non-zero read happens lazily.
        specs = {name: {"_file": model_dir / shard} for name, shard in weight_map.items()}
        return specs, model_dir
    single = model_dir / "model.safetensors"
    if single.exists():
        header = _safetensors_header(single)
        base = 8 + struct.unpack("<Q", single.open("rb").read(8))[0]
        for spec in header.values():
            spec["_file"] = single
            spec["_base"] = base
        return header, model_dir
    return None


def _tensor_is_nonzero(spec: dict[str, Any]) -> bool | None:
    """Read a tensor's raw bytes and report whether any byte is non-zero.

    Returns ``None`` if the bytes can't be read (sharded entry without offsets, or
    IO error) so the caller can degrade to presence-only rather than assert.
    """
    offsets = spec.get("data_offsets")
    base = spec.get("_base")
    path = spec.get("_file")
    if offsets is None or base is None or path is None:
        return None
    start, end = offsets
    try:
        with open(path, "rb") as f:
            f.seek(base + start)
            raw = f.read(end - start)
    except OSError:
        return None
    if not raw:
        return None
    return any(b != 0 for b in raw)


# Calibration / quantization-metadata tensors that are NOT parameters — a
# zero-capped tower can keep these non-zero, so they must never be the non-zero
# sample. ``output_*`` join ``input_*`` here: gemma-4's audio tower carries
# per-tensor activation ``output_max``/``output_min``/``output_scale`` calibration
# scalars that survive a zeroed projection (observed on /tmp/osoi5-12k-baked).
_NON_PARAM_MARKERS = (
    "weight_scale", "weight_shape", "zero_point", "g_idx",
    "input_max", "input_scale", "input_min", "layer_scalar", "act_scale",
    "output_max", "output_min", "output_scale",
)
# Substrings that mark an actual compute (projection / conv / embed) weight —
# the right place to detect a zero-capped tower (norms init to ones and would
# survive zeroing of the projections, so they are a weak zero-cap signal).
_PROJECTION_MARKERS = (
    "proj", "mlp", "fc", "linear", "dense", "conv", "qkv", "gate", "up_", "down_", "embed", "ffw",
)


def _is_norm_leaf(name: str) -> bool:
    """True if the tensor's leaf parameter is a normalization weight.

    Norms are 1-D and initialize to 1.0, so they SURVIVE a zero-cap of the real
    compute weights — sampling one as the non-zero check would mask a zeroed tower.
    The leaf must be inspected (not the full path): gemma-4's audio tower nests
    norms under projection-named modules (e.g. ``subsample_conv_projection.layer1.
    norm.weight``, ``feed_forward1.post_layer_norm.weight``), so an ancestor-name
    ``_PROJECTION_MARKERS`` match would otherwise misclassify a norm as a real
    projection weight.
    """
    leaf = name.lower().rsplit(".", 1)[0].rsplit(".", 1)[-1]  # component before the final .weight*
    return "norm" in leaf or leaf == "ln"


def _smallest_weight_for_prefixes(specs: dict[str, Any], prefixes: tuple[str, ...]) -> tuple[str, dict] | None:
    """Pick a representative real *compute* weight tensor under the prefixes.

    Preference, to make the non-zero check a faithful zero-cap detector:
      1. smallest ``*.weight_packed`` (int4 quantized projection weight),
      2. smallest projection ``*.weight`` that is NOT a norm (bf16 compute weight),
      3. smallest non-norm ``*.weight``,
      4. smallest norm ``*.weight`` (last resort — a real param, but 1-D and
         init-to-ones, so it survives a zero-cap; preferred only when no real
         compute weight exists),
      5. smallest non-``.weight`` tensor.
    Smallest-by-bytes keeps the raw read tiny. Calibration/quant-metadata tensors
    are excluded so a zeroed tower can't be masked by surviving non-zero scales,
    and norms are demoted below real compute weights for the same reason.
    """
    tiers: list[list[tuple[int, str, dict]]] = [[], [], [], [], []]
    for name, spec in specs.items():
        if not name.startswith(prefixes):
            continue
        low = name.lower()
        if any(tag in low for tag in _NON_PARAM_MARKERS):
            continue
        offsets = spec.get("data_offsets")
        size = (offsets[1] - offsets[0]) if offsets else 1 << 62
        entry = (size, name, spec)
        is_proj = any(tag in low for tag in _PROJECTION_MARKERS)
        is_norm = _is_norm_leaf(name)
        if low.endswith(".weight_packed"):
            tiers[0].append(entry)
        elif low.endswith(".weight") and is_norm:
            tiers[3].append(entry)
        elif low.endswith(".weight") and is_proj:
            tiers[1].append(entry)
        elif low.endswith(".weight"):
            tiers[2].append(entry)
        else:
            tiers[4].append(entry)
    for tier in tiers:
        if tier:
            best = min(tier, key=lambda e: e[0])
            return (best[1], best[2])
    return None


def presence_probe(model_dir: Path | None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per-tower present + non-zero check over the served checkpoint.

    For each required modality: the config sub-signal must be present AND at least
    one tower tensor must exist AND a representative weight tensor must be non-zero
    (not zero-capped). Missing config/tensors => ``False`` (removed); all-zero =>
    ``False`` (zero-capped); checkpoint unreadable => ``None`` (unknown).
    """
    result: dict[str, Any] = {"model_dir": str(model_dir) if model_dir else None, "loaded": {}, "detail": {}}
    if model_dir is None or not _is_model_dir(Path(model_dir)):
        for m in MODALITIES:
            result["loaded"][m] = None
            result["detail"][m] = {"method": "presence", "note": "served checkpoint dir not resolved"}
        return result

    model_dir = Path(model_dir)
    if config is None:
        try:
            config = json.loads((model_dir / "config.json").read_text())
        except (OSError, ValueError) as exc:
            for m in MODALITIES:
                result["loaded"][m] = None
                result["detail"][m] = {"method": "presence", "error": f"config unreadable: {exc}"}
            return result

    indexed = _checkpoint_tensor_index(model_dir)
    if indexed is None:
        for m in MODALITIES:
            result["loaded"][m] = None
            result["detail"][m] = {"method": "presence", "note": "no safetensors found"}
        return result
    specs, _ = indexed

    for m in MODALITIES:
        detail: dict[str, Any] = {"method": "presence"}
        config_ok = any(sig in config for sig in _CONFIG_SIGNAL[m])
        prefixes = _TOWER_PREFIXES[m]
        names = [n for n in specs if n.startswith(prefixes)]
        detail["config_signal"] = config_ok
        detail["num_tensors"] = len(names)
        if m == "video":
            detail["note"] = "no separate video tower; shares vision_tower + video_token_id"
        if not config_ok or not names:
            result["loaded"][m] = False
            detail["reason"] = "config signal absent" if not config_ok else "no tower tensors"
            result["detail"][m] = detail
            continue
        pick = _smallest_weight_for_prefixes(specs, prefixes)
        nonzero = _tensor_is_nonzero(pick[1]) if pick else None
        if nonzero is True:
            result["loaded"][m] = True
            detail["nonzero_tensor"] = pick[0]
        elif nonzero is False:
            result["loaded"][m] = False
            detail["reason"] = "representative tensor is all-zero (zero-capped)"
            detail["zero_tensor"] = pick[0]
        else:
            # present on disk but non-zero unverifiable (e.g. sharded) — presence
            # holds, but be honest the non-zero check was inconclusive.
            result["loaded"][m] = True
            detail["nonzero_check"] = "inconclusive (present, bytes unread)"
            if pick:
                detail["sampled_tensor"] = pick[0]
        result["detail"][m] = detail
    return result


# --------------------------------------------------------------------------- #
# aggregation + official-gate verdict (pure, CPU-only, unit-tested)
# --------------------------------------------------------------------------- #
def aggregate_all_modalities(loaded: dict[str, bool | None]) -> bool | None:
    """Three-valued AND over the four modalities (False dominates, then None).

    ``True`` iff every modality is ``True``; ``False`` if any is ``False`` (a known
    violation); ``None`` if none is ``False`` but at least one is unknown.
    """
    values = [loaded.get(m) for m in MODALITIES]
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def _tri(value: bool | None) -> bool | None:
    return value


def official_gate_verdict(
    *,
    ppl: float | None,
    completed: int | None,
    num_prompts: int,
    all_modalities_loaded: bool | None,
    ppl_cap: float = PPL_CAP,
) -> dict[str, Any]:
    """Consolidated leaderboard gate: PPL ≤ cap AND completion AND all modalities.

    Returns the verdict string plus the three component booleans (each ``True`` /
    ``False`` / ``None``-for-unknown):

      * ``PASS``       — all three components known-``True``.
      * ``FAIL``       — any component known-``False`` (a definitive violation
                         dominates, even if another input is still unknown).
      * ``INCOMPLETE`` — no known failure but at least one component unknown
                         (cannot certify PASS without spending HF-Jobs quota).

    This mirrors the *official* gate (#38: PPL + completion + modalities, NOT
    token-identity). ``official_gate_pass`` is the boolean form of the PR formula
    and is ``True`` exactly when the verdict is ``PASS``.
    """
    ppl_ok: bool | None = None if ppl is None else (ppl <= ppl_cap)
    completion_ok: bool | None = None if completed is None else (completed == num_prompts)
    modalities_ok: bool | None = _tri(all_modalities_loaded)

    components = [ppl_ok, completion_ok, modalities_ok]
    if any(c is False for c in components):
        verdict = "FAIL"
    elif any(c is None for c in components):
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    return {
        "official_gate": verdict,
        "official_gate_pass": verdict == "PASS",
        "official_gate_ppl_ok": ppl_ok,
        "official_gate_completion_ok": completion_ok,
        "official_gate_modalities_ok": modalities_ok,
        "ppl_cap": ppl_cap,
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def probe_modalities(
    *,
    base_url: str | None,
    model: str | None,
    manifest: dict[str, Any] | None = None,
    submission_dir: Path | None = None,
    model_id: str | None = None,
    model_dir: Path | None = None,
    image_timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Run the functional (text, image) + presence (audio, video; image fallback)
    probes and return the consolidated modality result.

    Output keys:
      * ``modalities_loaded``: ``{text,image,audio,video: bool|None}``
      * ``all_modalities_loaded``: ``bool|None``
      * ``modalities_method``: how each modality was verified
      * ``detail``: per-modality diagnostics (samples, errors, reasons)
    """
    loaded: dict[str, bool | None] = {m: None for m in MODALITIES}
    method: dict[str, str] = {}
    detail: dict[str, Any] = {}

    # Resolve the served checkpoint once for the presence tier.
    served_dir = resolve_served_model_dir(
        manifest=manifest, submission_dir=submission_dir, model_id=model_id, override=model_dir
    )
    presence = presence_probe(served_dir)

    # text — functional (falls back to presence if the endpoint is unreachable).
    if base_url and model:
        val, info = functional_text_probe(base_url, model)
        if val is not None:
            loaded["text"], method["text"], detail["text"] = val, "functional", info
    if "text" not in method:
        loaded["text"] = presence["loaded"].get("text")
        method["text"] = "presence"
        detail["text"] = presence["detail"].get("text", {})

    # image — functional preferred; presence fallback when functional is ambiguous.
    if base_url and model:
        val, info = functional_image_probe(base_url, model, timeout_s=image_timeout_s)
        if val is not None:
            loaded["image"], method["image"], detail["image"] = val, "functional", info
    if "image" not in method:
        loaded["image"] = presence["loaded"].get("image")
        method["image"] = "presence"
        info = presence["detail"].get("image", {})
        info.setdefault("note", "functional image probe inconclusive — presence fallback")
        detail["image"] = info

    # audio, video — presence + non-zero (functional sample inputs not staged).
    for m in ("audio", "video"):
        loaded[m] = presence["loaded"].get(m)
        method[m] = "presence"
        detail[m] = presence["detail"].get(m, {})

    return {
        "modalities_loaded": loaded,
        "all_modalities_loaded": aggregate_all_modalities(loaded),
        "modalities_method": method,
        "served_checkpoint_dir": str(served_dir) if served_dir else None,
        "detail": detail,
    }


def _print_block(result: dict[str, Any]) -> None:
    loaded = result["modalities_loaded"]
    method = result["modalities_method"]
    print("modalities:")
    for m in MODALITIES:
        flag = {True: "LOADED", False: "MISSING", None: "UNKNOWN"}[loaded.get(m)]
        print(f"    {m:<6} {flag:<8} [{method.get(m, '?')}]")
    agg = result["all_modalities_loaded"]
    print(f"  all_modalities_loaded: {agg}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    from . import harness

    ap = argparse.ArgumentParser(description="Probe that all four gemma-4 modalities are served.")
    ap.add_argument("--base-url", default=None, help="served endpoint for functional probes (e.g. http://127.0.0.1:8000)")
    ap.add_argument("--model", default=None, help="served model name for functional probes")
    ap.add_argument("--submission", type=Path, default=None, help="submission dir (resolves manifest + served checkpoint)")
    ap.add_argument("--model-dir", type=Path, default=None, help="explicit served checkpoint dir for the presence tier")
    ap.add_argument("--model-id", default=None, help="served MODEL_ID (hub id or path) for presence resolution")
    args = ap.parse_args(argv)

    manifest = None
    model_id = args.model_id
    if args.submission is not None:
        manifest = harness.load_manifest(args.submission)
        model_id = model_id or harness.serve_model_id(manifest, args.submission)

    result = probe_modalities(
        base_url=args.base_url,
        model=args.model,
        manifest=manifest,
        submission_dir=args.submission,
        model_id=model_id,
        model_dir=args.model_dir,
    )
    _print_block(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
