"""Unit tests for the check_greedy_identity CLI wrapper.

These exercise the CLI surface (arg parsing, exit codes, human + JSON output)
only. The comparison library internals are covered by
tests/test_greedy_identity.py and are not re-tested here.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

# Make the work directory (containing check_greedy_identity.py) importable
# regardless of the runner's start dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_greedy_identity as cli  # noqa: E402
import greedy_identity as gi  # noqa: E402


def _write_jsonl(lines):
    """Write raw text lines to a temp .jsonl file; return its path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
    return path


def _record(key, ids, sha=None):
    """Build a JSON line. sha=True computes a correct sha; a literal value is
    used as-is; None omits the field."""
    rec = {"id": key, "completion_token_ids": ids}
    if sha is True:
        rec["completion_token_sha256"] = gi.sha256_tokens(ids)
    elif sha is not None:
        rec["completion_token_sha256"] = sha
    return json.dumps(rec)


def _run(argv):
    """Invoke cli.main(argv), capturing stdout/stderr. Returns (code, out, err)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliExitCodeTests(unittest.TestCase):
    def setUp(self):
        self._paths = []

    def tearDown(self):
        for p in self._paths:
            try:
                os.remove(p)
            except OSError:
                pass

    def _make(self, lines):
        path = _write_jsonl(lines)
        self._paths.append(path)
        return path

    def test_identical_returns_zero(self):
        ref = self._make([
            _record("a", [1, 2, 3], sha=True),
            _record("b", [4, 5], sha=True),
        ])
        cand = self._make([
            _record("a", [1, 2, 3], sha=True),
            _record("b", [4, 5], sha=True),
        ])
        code, out, _ = _run(["--reference", ref, "--candidate", cand])
        self.assertEqual(code, 0)
        self.assertIn("GREEDY_IDENTICAL", out)

    def test_divergent_returns_one_and_names_prompt(self):
        ref = self._make([_record("a", [1, 2, 3, 4], sha=True)])
        cand = self._make([_record("a", [1, 2, 9, 4], sha=True)])
        code, out, _ = _run(["--reference", ref, "--candidate", cand])
        self.assertEqual(code, 1)
        self.assertIn("DIVERGENT", out)
        # Human output must emit the divergent prompt's example line, formatted
        # as "    - {key}: first divergence at index {i}". Assert on that line
        # specifically (not a bare "a", which appears in banner words too).
        self.assertIn("- a:", out)
        self.assertIn("first divergence at index", out)

    def test_incomparable_prompt_set_mismatch_returns_two(self):
        ref = self._make([
            _record("a", [1, 2], sha=True),
            _record("b", [3, 4], sha=True),
        ])
        cand = self._make([
            _record("a", [1, 2], sha=True),
            _record("c", [5, 6], sha=True),
        ])
        code, out, _ = _run(["--reference", ref, "--candidate", cand])
        self.assertEqual(code, 2)
        self.assertIn("INCOMPARABLE", out)

    def test_incomparable_integrity_failure_returns_two(self):
        ref = self._make([_record("a", [1, 2, 3], sha=True)])
        cand = self._make([_record("a", [1, 2, 3], sha="deadbeef")])
        code, out, _ = _run(["--reference", ref, "--candidate", cand])
        self.assertEqual(code, 2)
        self.assertIn("INCOMPARABLE", out)

    def test_missing_file_returns_two_without_traceback(self):
        ref = self._make([_record("a", [1, 2, 3], sha=True)])
        missing = ref + ".does-not-exist"
        code, out, err = _run(["--reference", ref, "--candidate", missing])
        self.assertEqual(code, 2)
        # A clean error message on stderr, no traceback leaked.
        self.assertNotIn("Traceback", out)
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip())

    def test_json_identical_returns_zero_and_valid_json(self):
        ref = self._make([_record("a", [1, 2, 3], sha=True)])
        cand = self._make([_record("a", [1, 2, 3], sha=True)])
        code, out, _ = _run(
            ["--reference", ref, "--candidate", cand, "--json"]
        )
        self.assertEqual(code, 0)
        parsed = json.loads(out)
        self.assertEqual(parsed["verdict"], "GREEDY_IDENTICAL")

    def test_json_divergent_returns_one_and_json_only(self):
        # Consistent shas so the verdict is genuinely DIVERGENT, not
        # INCOMPARABLE on an integrity failure.
        ref = self._make([_record("a", [1, 2, 3, 4], sha=True)])
        cand = self._make([_record("a", [1, 2, 9, 4], sha=True)])
        code, out, _ = _run(
            ["--reference", ref, "--candidate", cand, "--json"]
        )
        self.assertEqual(code, 1)
        parsed = json.loads(out)
        self.assertEqual(parsed["verdict"], "DIVERGENT")
        # --json output must be JSON-only: the human banner must not leak.
        self.assertNotIn("VERDICT:", out)

    def test_malformed_input_returns_two_without_traceback(self):
        # A malformed JSON line must surface as a clean ValueError -> exit 2,
        # not an uncaught traceback.
        ref = self._make([_record("a", [1, 2, 3], sha=True)])
        cand = self._make(["{not valid json"])
        code, out, err = _run(["--reference", ref, "--candidate", cand])
        self.assertEqual(code, 2)
        self.assertTrue(err.strip())
        self.assertNotIn("Traceback", out)
        self.assertNotIn("Traceback", err)

    def test_max_examples_limits_listed_divergent_prompts(self):
        ref = self._make([
            _record("a", [1], sha=True),
            _record("b", [1], sha=True),
            _record("c", [1], sha=True),
        ])
        cand = self._make([
            _record("a", [9], sha=True),
            _record("b", [9], sha=True),
            _record("c", [9], sha=True),
        ])
        code, out, _ = _run(
            ["--reference", ref, "--candidate", cand, "--max-examples", "1"]
        )
        self.assertEqual(code, 1)
        # Only one divergent prompt listed as an example line.
        listed = [ln for ln in out.splitlines()
                  if ln.strip().startswith("- ")]
        self.assertEqual(len(listed), 1)


class CliArgParsingTests(unittest.TestCase):
    def test_missing_required_args_exits_two(self):
        # argparse exits via SystemExit(2) on missing required options.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                cli.main([])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
