#!/usr/bin/env python3
"""Unit tests for the local OFFICIAL-gate preflight (PR #45).

Pins three pure, CPU-only pieces of the leaderboard-faithful preflight:

  * ``official_gate_verdict`` — the consolidated PASS / FAIL / INCOMPLETE gate
    (PPL ≤ 2.42 AND completion AND all-modalities-loaded; #38 established the
    official gate is PPL + completion + modalities, NOT token-identity).
  * ``aggregate_all_modalities`` — three-valued AND over the four pathways
    (False dominates a None; all-True is the only True).
  * the modalities **presence + non-zero** probe — config-signal + tower-tensor
    presence + a representative *weight* being non-zero, exercised against
    hand-built tiny checkpoints (removed tower, zero-capped tower, intact tower).
  * functional-probe response classification (mocked endpoint, no GPU/network).

No GPU, no model load, no real server. Run:
    python -m pytest scripts/tests/test_official_gate.py -v
  or: python scripts/tests/test_official_gate.py    (no-pytest fallback)
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.local_validation import modalities_probe as mp  # noqa: E402

PASS, FAIL, INCOMPLETE = "PASS", "FAIL", "INCOMPLETE"


# --------------------------------------------------------------------------- #
# official_gate_verdict truth table
# --------------------------------------------------------------------------- #
def _verdict(ppl, completed, all_mod, n=128):
    return mp.official_gate_verdict(
        ppl=ppl, completed=completed, num_prompts=n, all_modalities_loaded=all_mod
    )


def test_official_gate_pass_when_all_three_hold():
    r = _verdict(2.019, 128, True)
    assert r["official_gate"] == PASS
    assert r["official_gate_pass"] is True
    assert r["official_gate_ppl_ok"] is True
    assert r["official_gate_completion_ok"] is True
    assert r["official_gate_modalities_ok"] is True


def test_official_gate_fail_on_ppl_over_cap():
    r = _verdict(2.50, 128, True)  # 2.50 > 2.42
    assert r["official_gate"] == FAIL
    assert r["official_gate_pass"] is False
    assert r["official_gate_ppl_ok"] is False


def test_official_gate_pass_at_exactly_the_cap():
    """PPL == 2.42 is within the cap (<=), so it must not fail on PPL."""
    r = _verdict(2.42, 128, True)
    assert r["official_gate"] == PASS
    assert r["official_gate_ppl_ok"] is True


def test_official_gate_fail_on_incomplete_completion():
    r = _verdict(2.019, 120, True)  # 120 != 128
    assert r["official_gate"] == FAIL
    assert r["official_gate_completion_ok"] is False


def test_official_gate_fail_on_modality_false():
    r = _verdict(2.019, 128, False)
    assert r["official_gate"] == FAIL
    assert r["official_gate_modalities_ok"] is False


def test_official_gate_incomplete_on_modality_unknown():
    """One modality unknown (all_modalities_loaded=None) with no known failure
    must read INCOMPLETE — never a false PASS by omission."""
    r = _verdict(2.019, 128, None)
    assert r["official_gate"] == INCOMPLETE
    assert r["official_gate_pass"] is False
    assert r["official_gate_modalities_ok"] is None


def test_official_gate_incomplete_on_missing_ppl_or_completion():
    assert _verdict(None, 128, True)["official_gate"] == INCOMPLETE
    assert _verdict(2.019, None, True)["official_gate"] == INCOMPLETE


def test_official_gate_fail_dominates_unknown():
    """A known failure on one component dominates an unknown on another (a
    definitive violation can't be rescued by an unmeasured input)."""
    assert _verdict(2.50, None, None)["official_gate"] == FAIL          # ppl fail
    assert _verdict(None, 120, None)["official_gate"] == FAIL           # completion fail
    assert _verdict(None, None, False)["official_gate"] == FAIL         # modality fail


# --------------------------------------------------------------------------- #
# aggregate_all_modalities
# --------------------------------------------------------------------------- #
def test_aggregate_all_true():
    assert mp.aggregate_all_modalities({"text": True, "image": True, "audio": True, "video": True}) is True


def test_aggregate_any_false_is_false():
    assert mp.aggregate_all_modalities({"text": True, "image": False, "audio": True, "video": True}) is False


def test_aggregate_any_none_is_none():
    assert mp.aggregate_all_modalities({"text": True, "image": None, "audio": True, "video": True}) is None


def test_aggregate_false_dominates_none():
    """A known-missing modality must read False even when another is unknown."""
    assert mp.aggregate_all_modalities({"text": True, "image": None, "audio": False, "video": True}) is False


def test_aggregate_missing_key_is_none():
    """A modality absent from the dict is unknown, not assumed loaded."""
    assert mp.aggregate_all_modalities({"text": True, "image": True, "audio": True}) is None


# --------------------------------------------------------------------------- #
# presence + non-zero probe over hand-built checkpoints
# --------------------------------------------------------------------------- #
_FULL_CONFIG = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "text_config": {"model_type": "gemma4_text"},
    "vision_config": {"model_type": "gemma4_vision"},
    "audio_config": {"model_type": "gemma4_audio"},
    "video_token_id": 258884,
}


def _write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    """Write a minimal single-file .safetensors the stdlib parser can read.

    dtype is irrelevant to modalities_probe (it reads raw bytes via data_offsets),
    so we declare U8 and pack the given bytes contiguously.
    """
    header: dict = {}
    offset = 0
    for name, raw in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(raw)], "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for raw in tensors.values():
            f.write(raw)


def _checkpoint(dirpath: Path, *, config: dict, tensors: dict[str, bytes]) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "config.json").write_text(json.dumps(config))
    _write_safetensors(dirpath / "model.safetensors", tensors)
    return dirpath


NZ = b"\x01\x02\x03\x04\x05\x06\x07\x08"  # non-zero weight bytes
Z = b"\x00\x00\x00\x00\x00\x00\x00\x00"   # all-zero (zero-capped) bytes


def test_presence_all_towers_loaded():
    with tempfile.TemporaryDirectory() as tmp:
        d = _checkpoint(
            Path(tmp) / "ck",
            config=_FULL_CONFIG,
            tensors={
                "model.language_model.layers.0.mlp.down_proj.weight": NZ,
                "model.vision_tower.encoder.layers.0.self_attn.k_proj.weight": NZ,
                "model.audio_tower.layers.0.mlp.fc.weight": NZ,
                "model.embed_vision.weight": NZ,
                "model.embed_audio.weight": NZ,
            },
        )
        out = mp.presence_probe(d)
        assert out["loaded"] == {"text": True, "image": True, "audio": True, "video": True}


def test_presence_zero_capped_tower_is_false():
    """An audio tower whose projection weight is all-zero is zero-capped -> False,
    even though the tensor is present (the residual-risk case the probe must catch
    when it samples a real weight, not a calibration scalar)."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _checkpoint(
            Path(tmp) / "ck",
            config=_FULL_CONFIG,
            tensors={
                "model.language_model.layers.0.mlp.down_proj.weight": NZ,
                "model.vision_tower.encoder.layers.0.self_attn.k_proj.weight": NZ,
                "model.audio_tower.layers.0.mlp.fc.weight": Z,  # zero-capped
            },
        )
        out = mp.presence_probe(d)
        assert out["loaded"]["audio"] is False
        assert "zero" in out["detail"]["audio"].get("reason", "").lower()
        assert out["loaded"]["image"] is True


def test_presence_removed_tower_is_false():
    """No vision-tower tensors at all -> image (and video, which shares it) False."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _checkpoint(
            Path(tmp) / "ck",
            config=_FULL_CONFIG,
            tensors={
                "model.language_model.layers.0.mlp.down_proj.weight": NZ,
                "model.audio_tower.layers.0.mlp.fc.weight": NZ,
            },
        )
        out = mp.presence_probe(d)
        assert out["loaded"]["image"] is False
        assert out["loaded"]["video"] is False
        assert out["loaded"]["text"] is True
        assert out["loaded"]["audio"] is True


def test_presence_missing_config_signal_is_false():
    """A config with vision_config stripped -> image reads False (modality removed
    at the architecture level), regardless of any stray tensors."""
    cfg = dict(_FULL_CONFIG)
    cfg.pop("vision_config")
    cfg.pop("video_token_id")
    with tempfile.TemporaryDirectory() as tmp:
        d = _checkpoint(
            Path(tmp) / "ck",
            config=cfg,
            tensors={"model.language_model.layers.0.mlp.down_proj.weight": NZ},
        )
        out = mp.presence_probe(d)
        assert out["loaded"]["image"] is False
        assert out["loaded"]["video"] is False
        assert "config" in out["detail"]["image"].get("reason", "").lower()


def test_presence_unresolved_dir_is_none():
    out = mp.presence_probe(None)
    assert out["loaded"] == {m: None for m in mp.MODALITIES}
    out2 = mp.presence_probe(Path("/nonexistent/model/dir"))
    assert out2["loaded"]["text"] is None


def test_smallest_weight_prefers_real_projection_over_metadata():
    """The non-zero sampler must pick a real weight (weight_packed / projection
    .weight), never a quant-metadata scalar that survives a zero-cap."""
    specs = {
        "model.audio_tower.x.weight_scale": {"data_offsets": [0, 4]},      # metadata (skip)
        "model.audio_tower.x.input_max": {"data_offsets": [4, 8]},          # metadata (skip)
        "model.audio_tower.x.proj.weight_packed": {"data_offsets": [8, 40]},  # real (tier 0)
        "model.audio_tower.x.norm.weight": {"data_offsets": [40, 44]},      # norm .weight (demoted)
    }
    pick = mp._smallest_weight_for_prefixes(specs, ("model.audio_tower.",))
    assert pick is not None and pick[0].endswith(".weight_packed")


def test_smallest_weight_skips_norm_leaf_under_projection_module():
    """A norm leaf nested under a conv/projection-named module must NOT be the
    non-zero sample when a real compute weight exists. gemma-4's audio tower nests
    norms under projection modules (``subsample_conv_projection.layer1.norm.weight``);
    a norm inits to 1.0 and survives a zero-cap, so sampling it would mask a zeroed
    tower. The smaller norm must lose to the larger real projection weight."""
    specs = {
        # tiny norm leaves whose ANCESTOR module name matches a projection marker
        "model.audio_tower.subsample_conv_projection.layer1.norm.weight": {"data_offsets": [0, 64]},
        "model.audio_tower.layers.0.feed_forward1.post_layer_norm.weight": {"data_offsets": [64, 64 + 2048]},
        # the real compute weight (larger, but the only faithful zero-cap sample)
        "model.audio_tower.layers.0.self_attn.k_proj.weight": {"data_offsets": [2112, 2112 + 8192]},
    }
    pick = mp._smallest_weight_for_prefixes(specs, ("model.audio_tower.",))
    assert pick is not None and pick[0].endswith("k_proj.weight"), pick


def test_smallest_weight_skips_output_minmax_calibration_scalars():
    """``output_max``/``output_min`` are activation-range calibration scalars (not
    parameters) and survive a zero-cap, so they must be skipped like ``input_*``."""
    specs = {
        "model.audio_tower.x.k_proj.output_max": {"data_offsets": [0, 2]},
        "model.audio_tower.x.k_proj.output_min": {"data_offsets": [2, 4]},
        "model.audio_tower.x.k_proj.output_scale": {"data_offsets": [4, 6]},
        "model.audio_tower.x.k_proj.weight": {"data_offsets": [6, 6 + 1024]},
    }
    pick = mp._smallest_weight_for_prefixes(specs, ("model.audio_tower.",))
    assert pick is not None and pick[0].endswith("k_proj.weight")


def test_smallest_weight_norm_is_last_resort_when_no_compute_weight():
    """When a tower genuinely has only norm .weights (no real compute weight),
    the norm is still returned (a real param beats nothing) rather than None."""
    specs = {
        "model.audio_tower.layers.0.input_layernorm.weight": {"data_offsets": [0, 32]},
    }
    pick = mp._smallest_weight_for_prefixes(specs, ("model.audio_tower.",))
    assert pick is not None and pick[0].endswith("input_layernorm.weight")


# --------------------------------------------------------------------------- #
# resolve_served_model_dir
# --------------------------------------------------------------------------- #
def test_resolve_override_wins():
    with tempfile.TemporaryDirectory() as tmp:
        d = _checkpoint(Path(tmp) / "served", config=_FULL_CONFIG, tensors={"a.weight": NZ})
        got = mp.resolve_served_model_dir(manifest=None, submission_dir=None, model_id=None, override=d)
        assert got == d


def test_resolve_from_manifest_env_weight_dir():
    with tempfile.TemporaryDirectory() as tmp:
        d = _checkpoint(Path(tmp) / "baked", config=_FULL_CONFIG, tensors={"a.weight": NZ})
        manifest = {"env": {"LM_HEAD_PRUNE_DST": str(d)}}
        got = mp.resolve_served_model_dir(manifest=manifest, submission_dir=None, model_id="google/gemma-4-E4B-it", override=None)
        assert got == d


def test_resolve_bundled_model_dir():
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / "sub"
        _checkpoint(sub / "model", config=_FULL_CONFIG, tensors={"a.weight": NZ})
        got = mp.resolve_served_model_dir(manifest={"env": {}}, submission_dir=sub, model_id="model", override=None)
        assert got == sub / "model"


def test_resolve_unresolvable_is_none():
    got = mp.resolve_served_model_dir(manifest={"env": {}}, submission_dir=None, model_id="totally/missing-xyz", override=None)
    assert got is None


# --------------------------------------------------------------------------- #
# functional-probe response classification (mocked endpoint)
# --------------------------------------------------------------------------- #
class _patched_post:
    """Swap modalities_probe._post_json for a canned (status, body), restoring it."""

    def __init__(self, status, body):
        self._reply = (status, body)

    def __enter__(self):
        self._saved = mp._post_json
        mp._post_json = lambda *a, **k: self._reply
        return self

    def __exit__(self, *exc):
        mp._post_json = self._saved


def test_functional_image_accepted_is_true():
    body = {"choices": [{"message": {"content": "a colorful gradient"}}]}
    with _patched_post(200, body):
        val, info = mp.functional_image_probe("http://x", "m")
    assert val is True and info["method"] == "functional"


def test_functional_image_rejected_modality_is_false():
    with _patched_post(400, "This model does not support image input."):
        val, info = mp.functional_image_probe("http://x", "m")
    assert val is False


def test_functional_image_ambiguous_error_is_none():
    """A transport error / opaque 500 is unknown (caller falls back to presence),
    never a False that would wrongly read as a modality violation."""
    with _patched_post(0, "timed out"):
        val, _ = mp.functional_image_probe("http://x", "m")
    assert val is None
    with _patched_post(500, "internal server error"):
        val2, _ = mp.functional_image_probe("http://x", "m")
    assert val2 is None


def test_functional_image_empty_completion_is_none():
    body = {"choices": [{"message": {"content": "   "}}]}
    with _patched_post(200, body):
        val, _ = mp.functional_image_probe("http://x", "m")
    assert val is None


def test_functional_text_nonempty_is_true():
    with _patched_post(200, {"choices": [{"text": " Paris"}]}):
        val, _ = mp.functional_text_probe("http://x", "m")
    assert val is True


# --------------------------------------------------------------------------- #
# integration: real baked fa2sw checkpoint, when present (skipped in CI)
# --------------------------------------------------------------------------- #
def test_presence_on_real_baked_fa2sw_when_present():
    baked = Path("/tmp/osoi5-12k-baked")
    if not (baked / "config.json").exists():
        return  # baked weights not staged in this environment; nothing to check
    out = mp.presence_probe(baked)
    assert out["loaded"] == {"text": True, "image": True, "audio": True, "video": True}


# --------------------------------------------------------------------------- #
# no-pytest fallback runner
# --------------------------------------------------------------------------- #
def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface setup errors too
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
