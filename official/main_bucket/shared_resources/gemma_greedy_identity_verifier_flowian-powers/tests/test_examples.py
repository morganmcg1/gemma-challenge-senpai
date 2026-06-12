"""End-to-end example tests against the committed fixtures.

These exercise the exact commands documented in README.md by importing the CLI
and calling main([...]) against the real, committed fixture files. They verify
the worked-example exit codes stay accurate over time.

The synthetic-fixture CLI behaviours (arg parsing, error handling, --max-examples,
etc.) are covered by tests/test_cli.py and are not duplicated here.
"""

import contextlib
import io
import json
import os
import sys
import unittest

# Repo dir holds check_greedy_identity.py and fixtures/; make it importable and
# resolve fixture paths relative to it (independent of the runner's start dir).
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_DIR)
_FIXTURES = os.path.join(_REPO_DIR, "fixtures")

import check_greedy_identity as cli  # noqa: E402


def _fixture(name):
    return os.path.join(_FIXTURES, name)


def _run(argv):
    """Invoke cli.main(argv), capturing stdout/stderr. Returns (code, out, err)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class ExampleFixtureTests(unittest.TestCase):
    def test_valid_candidate_exits_zero(self):
        code, out, _ = _run([
            "--reference", _fixture("reference.jsonl"),
            "--candidate", _fixture("candidate_valid.jsonl"),
        ])
        self.assertEqual(code, 0)
        self.assertIn("GREEDY_IDENTICAL", out)

    def test_divergent_candidate_exits_one(self):
        code, out, _ = _run([
            "--reference", _fixture("reference.jsonl"),
            "--candidate", _fixture("candidate_divergent.jsonl"),
        ])
        self.assertEqual(code, 1)
        self.assertIn("DIVERGENT", out)

    def test_json_valid_parses_and_reports_identical(self):
        code, out, _ = _run([
            "--reference", _fixture("reference.jsonl"),
            "--candidate", _fixture("candidate_valid.jsonl"),
            "--json",
        ])
        self.assertEqual(code, 0)
        parsed = json.loads(out)
        self.assertEqual(parsed["verdict"], "GREEDY_IDENTICAL")


if __name__ == "__main__":
    unittest.main()
