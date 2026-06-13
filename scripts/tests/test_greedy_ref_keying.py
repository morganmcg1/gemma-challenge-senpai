#!/usr/bin/env python3
"""Regression guard for greedy-reference KEYING (PR #32).

The greedy-identity gate resolves a candidate's exact-greedy reference by a tag
derived from the served model identity. The served ``MODEL_ID`` is NOT safe to
key on:

  * several int4 submissions all set ``env.MODEL_ID="model"`` (a relative
    literal) — so they collided onto one ``research/greedy_reference/model/``
    tag, and a validate run for submission B could silently compare against
    submission A's reference (a confident *wrong* GREEDY_IDENTICAL/DIVERGENT);
  * a bucket-weights submission (``fa2sw_precache_kenyan``) nominally reports the
    bf16 hub id ``google/gemma-4-E4B-it`` while serving entirely different baked
    int4 weights — keying on that id aliases it onto the *bf16 baseline*
    reference.

The fix (``harness.reference_identity``) anchors a submission's reference to the
submission directory, so two distinct submissions can NEVER share a tag, and the
generator writes to exactly the tag the validator reads. These tests pin both
properties. They are CPU-only: no GPU, no model load, no network — they only
exercise the pure path-keying logic.

Run: ``python -m pytest scripts/tests/test_greedy_ref_keying.py -v``
  or: ``python scripts/tests/test_greedy_ref_keying.py``   (no-pytest fallback)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.local_validation import gen_greedy_reference as gen  # noqa: E402
from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

SUBMISSIONS = REPO_ROOT / "submissions"

# fa2sw resolves its served id from the manifest ``model_id`` (hub id) only when
# no ambient MODEL_ID leaks in; clear it so keying is deterministic here exactly
# as it is under the harness (which never exports MODEL_ID into its own env).
os.environ.pop("MODEL_ID", None)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ref_path(identity: str) -> Path:
    """The resolved reference file for a canonical identity (what the gate reads)."""
    return greedy_gate.reference_for(identity)


def _write_submission(dirpath: Path, *, env_model_id: str | None, bundle: bool) -> Path:
    """Write a minimal manifest (and optional bundled ``model/``) for keying tests."""
    env = {"MODEL_ID": env_model_id} if env_model_id is not None else {}
    manifest = {
        "name": dirpath.name,
        "dependencies": [],
        "model_id": env_model_id or "google/gemma-4-E4B-it",
        "served_model_name": "gemma-4-e4b-it",
        "serve": ["python", "serve.py"],
        "env": env,
    }
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "manifest.json").write_text(json.dumps(manifest))
    if bundle:
        (dirpath / "model").mkdir(exist_ok=True)
    return dirpath


# --------------------------------------------------------------------------- #
# the core collision class: two distinct submissions, same env.MODEL_ID="model"
# --------------------------------------------------------------------------- #
def test_two_model_literal_submissions_distinct_with_bundle():
    """Two int4-style submissions (bundled ``model/``, env.MODEL_ID="model")
    must resolve to DIFFERENT reference tags — the exact silent-collision class
    that gated int4_g128_lmhead vs any sibling int4 head submission."""
    with tempfile.TemporaryDirectory() as tmp:
        a = _write_submission(Path(tmp) / "subA", env_model_id="model", bundle=True)
        b = _write_submission(Path(tmp) / "subB", env_model_id="model", bundle=True)
        id_a = harness.reference_identity("model", a)
        id_b = harness.reference_identity("model", b)
        assert id_a != id_b, "bundled model/ submissions collided"
        assert _ref_path(id_a) != _ref_path(id_b)
        # both anchored to an absolute checkpoint path -> never the bare 'model' tag
        assert paths.model_tag(id_a) != "model"
        assert paths.model_tag(id_b) != "model"


def test_two_model_literal_submissions_distinct_without_bundle():
    """Same collision class but weights NOT yet materialized on disk (e.g. a
    bucket-weights submission at test time). resolve_model_id alone would fall
    back to the bare literal "model" and collide; the submission-dir anchor must
    keep them distinct anyway."""
    with tempfile.TemporaryDirectory() as tmp:
        a = _write_submission(Path(tmp) / "subA", env_model_id="model", bundle=False)
        b = _write_submission(Path(tmp) / "subB", env_model_id="model", bundle=False)
        id_a = harness.reference_identity("model", a)
        id_b = harness.reference_identity("model", b)
        assert id_a != id_b, "unmaterialized model/ submissions collided"
        assert _ref_path(id_a) != _ref_path(id_b)
        assert paths.model_tag(id_a) != "model" and paths.model_tag(id_b) != "model"


# --------------------------------------------------------------------------- #
# the two REAL submissions named in the PR
# --------------------------------------------------------------------------- #
def test_named_pair_int4_vs_fa2sw_distinct():
    """submissions/int4_g128_lmhead and submissions/fa2sw_precache_kenyan must
    auto-resolve to DISTINCT tags (no shared 'model/' tag)."""
    int4 = SUBMISSIONS / "int4_g128_lmhead"
    fa2sw = SUBMISSIONS / "fa2sw_precache_kenyan"
    if not (int4 / "manifest.json").exists() or not (fa2sw / "manifest.json").exists():
        return  # submissions not present in this checkout; nothing to guard
    id_int4 = gen.reference_key_for_submission(int4)
    id_fa2sw = gen.reference_key_for_submission(fa2sw)
    assert id_int4 != id_fa2sw
    assert _ref_path(id_int4) != _ref_path(id_fa2sw)
    # neither lands on the bare, collision-prone "model" tag
    assert paths.model_tag(id_int4) != "model"
    assert paths.model_tag(id_fa2sw) != "model"


def test_fa2sw_distinct_from_bf16_baseline():
    """fa2sw nominally reports the bf16 hub id but serves baked int4 weights, so
    its reference must NOT alias the bare bf16 baseline reference (the deeper
    hazard the dir-anchor closes)."""
    fa2sw = SUBMISSIONS / "fa2sw_precache_kenyan"
    if not (fa2sw / "manifest.json").exists():
        return
    id_fa2sw = gen.reference_key_for_submission(fa2sw)
    id_baseline = harness.reference_identity(paths.BF16_MODEL, None)  # bare '--model-id' anchor
    assert id_fa2sw != id_baseline
    assert _ref_path(id_fa2sw) != _ref_path(id_baseline)


# --------------------------------------------------------------------------- #
# generator writes EXACTLY where the validator reads (closes NO_REFERENCE)
# --------------------------------------------------------------------------- #
def test_generator_and_validator_agree():
    """gen_greedy_reference's write tag must equal LocalServer.reference_model_id
    (the tag validate_submission reads). Constructing LocalServer does no IO
    beyond reading the manifest, so this runs CPU-only."""
    with tempfile.TemporaryDirectory() as tmp:
        sub = _write_submission(Path(tmp) / "sub", env_model_id="model", bundle=True)
        # validator side: LocalServer.__init__ builds the serve env (no server is
        # started — __enter__ would do that) and exposes reference_model_id.
        srv = harness.LocalServer(sub, server_python=Path("/usr/bin/python3"), port=8000)
        validator_key = srv.reference_model_id
        # generator side:
        generator_key = gen.reference_key_for_submission(sub)
        assert validator_key == generator_key
        # and the resolved files are the same path
        assert greedy_gate.reference_for(validator_key) == gen.served_reference_path(generator_key)
        # serve env is genuinely untouched: serve.py still receives the raw literal
        assert srv.model_id == "model"


def test_bare_model_id_preserves_baseline_tag():
    """The bare '--model-id' anchor (no submission) must keep keying purely by
    model id, so the existing committed bf16 baseline reference is not orphaned."""
    identity = harness.reference_identity(paths.BF16_MODEL, None)
    assert identity == paths.BF16_MODEL
    expected = paths.REFERENCE_ROOT / paths.model_tag(paths.BF16_MODEL) / "decode_outputs.jsonl"
    assert greedy_gate.reference_for(identity) == expected


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
        except Exception as exc:  # noqa: BLE001 - surface any setup error too
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
