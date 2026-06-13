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

import contextlib
import importlib.util
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
# runtime guard against bare-'model' submission tags (PR #40)
# --------------------------------------------------------------------------- #
def _raises_assertion(fn, *args) -> bool:
    """True iff calling fn(*args) raises AssertionError (pytest-free)."""
    try:
        fn(*args)
    except AssertionError:
        return True
    return False


def test_bare_model_tag_assertion_guards_submission_refs():
    """harness.assert_submission_reference_tag must REJECT the bare 'model'
    collision tag (what an empty/None submission_dir collapses to) and any tag
    missing the '::' anchor, while ACCEPTing a real <submission_dir>::<model_id>
    tag. This pins the PR #40 runtime guard that defends the #32 keying fix
    against a silent regression to bare-model_id keying."""
    # An empty/None submission_dir collapses reference_identity to the bare model
    # id — exactly the pre-#32 collision key the guard must reject.
    bare = harness.reference_identity("model", None)
    assert bare == "model"
    assert _raises_assertion(harness.assert_submission_reference_tag, bare), (
        "guard did not reject the bare 'model' collision tag"
    )
    # A plausible hub id with no '::' anchor must also be rejected.
    assert _raises_assertion(harness.assert_submission_reference_tag, "google/gemma-4-E4B-it"), (
        "guard did not reject an unanchored model-id tag"
    )
    # A real submission-anchored tag passes and round-trips unchanged.
    with tempfile.TemporaryDirectory() as tmp:
        sub = _write_submission(Path(tmp) / "sub", env_model_id="model", bundle=True)
        good = gen.reference_key_for_submission(sub)
        assert "::" in good and good != "model"
        assert harness.assert_submission_reference_tag(good) == good


def test_localserver_init_runs_bare_tag_guard():
    """Constructing LocalServer for a real submission resolves a '::'-anchored
    reference tag and passes the in-__init__ guard (no GPU; __init__ does no IO
    beyond reading the manifest). Guards against the guard ever rejecting a
    legitimate submission."""
    with tempfile.TemporaryDirectory() as tmp:
        sub = _write_submission(Path(tmp) / "sub", env_model_id="model", bundle=True)
        srv = harness.LocalServer(sub, server_python=Path("/usr/bin/python3"), port=8000)
        assert "::" in srv.reference_model_id and srv.reference_model_id != "model"


# --------------------------------------------------------------------------- #
# reference-mode contract: spec submissions honor SENPAI_REFERENCE_MODE (PR #42)
# --------------------------------------------------------------------------- #
def _load_serve_module(submission_name: str):
    """Import a submission's serve.py by path. Module-level code in every spec
    serve.py is stdlib-only (no torch/vLLM, no model load, no network), so this is
    CPU-only — it exercises the reference-mode arg-gating logic in isolation."""
    path = SUBMISSIONS / submission_name / "serve.py"
    spec = importlib.util.spec_from_file_location(f"_serve_{submission_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def _env(**overrides):
    """Temporarily set (str) or clear (None) env vars, restoring prior state."""
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


SPEC_JSON = '{"method":"mtp","model":"/tmp/qat-assistant","num_speculative_tokens":7}'


def test_fa2sw_spec_off_clears_speculative_config():
    """fa2sw serve.py honors the contract: SENPAI_REFERENCE_MODE truthy clears
    SPECULATIVE_CONFIG so vLLM serves M=1 AR (speculative_config=None); unset
    leaves the drafter config intact so the leaderboard serving path is untouched.
    This is the silent-no-op footgun PR #40 documented, now fixed at the root."""
    mod = _load_serve_module("fa2sw_precache_kenyan")
    with _env(SENPAI_REFERENCE_MODE="1", SPECULATIVE_CONFIG=SPEC_JSON):
        assert mod.reference_mode_active() is True
        assert mod.disable_speculation_for_reference_mode() is True
        assert os.environ.get("SPECULATIVE_CONFIG") == ""
        args: list[str] = []
        mod.append_env_arg(args, "SPECULATIVE_CONFIG", "--speculative-config")
        assert "--speculative-config" not in args, "drafter not actually disabled"
    with _env(SENPAI_REFERENCE_MODE=None, SPECULATIVE_CONFIG=SPEC_JSON):
        assert mod.reference_mode_active() is False
        assert mod.disable_speculation_for_reference_mode() is False
        assert os.environ.get("SPECULATIVE_CONFIG") == SPEC_JSON
        args = []
        mod.append_env_arg(args, "SPECULATIVE_CONFIG", "--speculative-config")
        assert args == ["--speculative-config", SPEC_JSON], "leaderboard path changed"


def test_lf29_spec_off_clears_speculative_config():
    """lf29cap444_pupa_check (same append_env_arg drafter knob as fa2sw) honors the
    contract identically."""
    mod = _load_serve_module("lf29cap444_pupa_check")
    with _env(SENPAI_REFERENCE_MODE="1", SPECULATIVE_CONFIG=SPEC_JSON):
        assert mod.disable_speculation_for_reference_mode() is True
        assert os.environ.get("SPECULATIVE_CONFIG") == ""
    with _env(SENPAI_REFERENCE_MODE=None, SPECULATIVE_CONFIG=SPEC_JSON):
        assert mod.disable_speculation_for_reference_mode() is False
        assert os.environ.get("SPECULATIVE_CONFIG") == SPEC_JSON


def test_int4_mtp_batchinv_spec_off_forces_zero_tokens():
    """int4_mtp_batchinv uses a different knob (NUM_SPECULATIVE_TOKENS -> a JSON
    --speculative-config dict), so the contract forces num_speculative_tokens=0
    under reference mode (skipping the spec-config block -> speculative_config=None)
    and leaves the served default intact otherwise."""
    mod = _load_serve_module("int4_mtp_batchinv")
    with _env(SENPAI_REFERENCE_MODE="1"):
        assert mod.reference_mode_active() is True
        assert mod.reference_mode_num_spec(6) == 0
        assert mod.reference_mode_num_spec(0) == 0
    with _env(SENPAI_REFERENCE_MODE=None):
        assert mod.reference_mode_active() is False
        assert mod.reference_mode_num_spec(6) == 6


def test_reference_mode_active_truthiness():
    """The predicate matches the documented 'truthy' contract (paths.py): '1' (what
    gen_greedy_reference --spec-off injects) and other non-empty non-'0' values are
    active; unset / '' / '0' are not — so a stray falsy value never disables the
    leaderboard drafter."""
    mod = _load_serve_module("fa2sw_precache_kenyan")
    for val, expected in [("1", True), ("reference", True), ("0", False), ("", False), (None, False)]:
        with _env(SENPAI_REFERENCE_MODE=val):
            assert mod.reference_mode_active() is expected, f"value {val!r}"


# --------------------------------------------------------------------------- #
# validator N-mismatch legibility (PR #42)
# --------------------------------------------------------------------------- #
def test_reference_num_records_reads_meta_then_summary():
    """greedy_gate.reference_num_records reads num_records from a reference's
    sibling meta.json, falling back to decode_summary.json, and returns None when
    neither is present."""
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "decode_outputs.jsonl"
        ref.write_text("")  # content irrelevant; only the sibling metas are read
        assert greedy_gate.reference_num_records(ref) is None
        (ref.parent / "decode_summary.json").write_text(json.dumps({"num_records": 32}))
        assert greedy_gate.reference_num_records(ref) == 32
        # meta.json (the canonical reference metadata) wins over decode_summary.json
        (ref.parent / "meta.json").write_text(json.dumps({"num_records": 128}))
        assert greedy_gate.reference_num_records(ref) == 128


def test_reference_n_mismatch_trip_condition():
    """A reference holding fewer records than the requested --num-prompts is the
    INCOMPARABLE footgun the validator now warns about. ref_n < num_prompts is the
    exact trip condition that writes reference_n_mismatch=true; an equal/greater
    count does not trip."""
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "decode_outputs.jsonl"
        ref.write_text("")
        (ref.parent / "meta.json").write_text(json.dumps({"num_records": 32}))
        ref_n = greedy_gate.reference_num_records(ref)
        assert ref_n == 32
        assert ref_n < 128, "32-record reference must trip the mismatch at --num-prompts 128"
        assert not ref_n < 32, "a matched record count must NOT trip the mismatch"


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
