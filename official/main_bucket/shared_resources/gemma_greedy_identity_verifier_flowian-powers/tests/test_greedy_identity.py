"""Unit tests for the greedy-identity comparison library."""

import os
import sys
import tempfile
import unittest

# Ensure the work directory (containing greedy_identity.py) is importable
# regardless of how the test runner sets the start dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import greedy_identity as gi  # noqa: E402


def _write_jsonl(lines):
    """Write the given list of raw text lines to a temp .jsonl file, return its
    path. Caller is responsible for deletion."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
    return path


def _record(key, ids, sha=None):
    """Build a JSON line for a record. If sha is None it is omitted; if sha is
    True it is computed correctly; otherwise the literal value is used."""
    import json

    rec = {"id": key, "completion_token_ids": ids}
    if sha is True:
        rec["completion_token_sha256"] = gi.sha256_tokens(ids)
    elif sha is not None:
        rec["completion_token_sha256"] = sha
    return json.dumps(rec)


class Sha256TokensTests(unittest.TestCase):
    def test_matches_precomputed_literal(self):
        # Precomputed once: sha256("1,2,3".encode("ascii")).hexdigest()
        expected = (
            "8a6ae15122001229edb8866f56e342af12ae8187203c3e3b33931743e7c0c48d"
        )
        self.assertEqual(gi.sha256_tokens([1, 2, 3]), expected)

    def test_empty_list(self):
        # sha256 of the empty string
        expected = (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        self.assertEqual(gi.sha256_tokens([]), expected)


class CompareTests(unittest.TestCase):
    def _build(self, ref_recs, cand_recs):
        ref = {k: {"completion_token_ids": v} for k, v in ref_recs.items()}
        cand = {k: {"completion_token_ids": v} for k, v in cand_recs.items()}
        return ref, cand

    def test_identical_is_greedy_identical(self):
        ref, cand = self._build(
            {"a": [1, 2, 3], "b": [4, 5]},
            {"a": [1, 2, 3], "b": [4, 5]},
        )
        report = gi.compare(ref, cand)
        self.assertEqual(report.verdict, "GREEDY_IDENTICAL")
        self.assertEqual(report.num_divergent, 0)
        self.assertEqual(report.num_identical, 2)
        self.assertEqual(report.total_divergent_tokens, 0)

    def test_one_flipped_token_is_divergent(self):
        ref, cand = self._build(
            {"a": [1, 2, 3, 4]},
            {"a": [1, 2, 9, 4]},
        )
        report = gi.compare(ref, cand)
        self.assertEqual(report.verdict, "DIVERGENT")
        self.assertEqual(report.num_divergent, 1)
        pc = report.per_prompt[0]
        self.assertEqual(pc.first_divergence_index, 2)
        self.assertEqual(pc.num_divergent_tokens, 1)
        self.assertFalse(pc.length_mismatch)
        self.assertFalse(pc.identical)

    def test_candidate_strict_prefix_is_divergent(self):
        ref, cand = self._build(
            {"a": [1, 2, 3, 4, 5]},
            {"a": [1, 2, 3]},
        )
        report = gi.compare(ref, cand)
        self.assertEqual(report.verdict, "DIVERGENT")
        pc = report.per_prompt[0]
        self.assertTrue(pc.length_mismatch)
        self.assertEqual(pc.first_divergence_index, 3)  # shorter length
        self.assertEqual(pc.num_compared, 3)
        self.assertEqual(pc.num_divergent_tokens, 2)  # abs(5-3)
        self.assertFalse(pc.identical)

    def test_candidate_longer_than_reference_is_divergent(self):
        ref, cand = self._build(
            {"a": [1, 2, 3]},
            {"a": [1, 2, 3, 4, 5]},
        )
        report = gi.compare(ref, cand)
        self.assertEqual(report.verdict, "DIVERGENT")
        pc = report.per_prompt[0]
        self.assertTrue(pc.length_mismatch)
        # First divergence is at the shorter (reference) length.
        self.assertEqual(pc.first_divergence_index, 3)
        self.assertEqual(pc.num_compared, 3)
        # num_divergent_tokens includes the length delta abs(3-5)==2.
        self.assertEqual(pc.num_divergent_tokens, 2)
        self.assertFalse(pc.identical)

    def test_prompt_set_mismatch_is_incomparable(self):
        ref, cand = self._build(
            {"a": [1, 2], "b": [3, 4]},
            {"a": [1, 2], "c": [5, 6]},
        )
        report = gi.compare(ref, cand)
        self.assertEqual(report.verdict, "INCOMPARABLE")
        self.assertEqual(report.missing_in_candidate, ["b"])
        self.assertEqual(report.missing_in_reference, ["c"])

    def test_stored_sha_integrity_failure_is_incomparable(self):
        ref = {
            "a": {
                "completion_token_ids": [1, 2, 3],
                "completion_token_sha256": gi.sha256_tokens([1, 2, 3]),
            }
        }
        cand = {
            "a": {
                "completion_token_ids": [1, 2, 3],
                "completion_token_sha256": "deadbeef",  # wrong
            }
        }
        report = gi.compare(ref, cand)
        self.assertEqual(report.verdict, "INCOMPARABLE")
        self.assertEqual(report.integrity_failures, ["a"])

    def test_stored_sha_integrity_failure_on_reference_is_incomparable(self):
        ref = {
            "a": {
                "completion_token_ids": [1, 2, 3],
                "completion_token_sha256": "deadbeef",  # wrong on REFERENCE
            }
        }
        cand = {
            "a": {
                "completion_token_ids": [1, 2, 3],
                "completion_token_sha256": gi.sha256_tokens([1, 2, 3]),
            }
        }
        report = gi.compare(ref, cand)
        self.assertEqual(report.verdict, "INCOMPARABLE")
        self.assertEqual(report.integrity_failures, ["a"])

    def test_stored_sha_absent_yields_none(self):
        ref, cand = self._build({"a": [1, 2, 3]}, {"a": [1, 2, 3]})
        report = gi.compare(ref, cand)
        self.assertIsNone(report.per_prompt[0].stored_sha_consistent)
        # No integrity failures when sha is simply absent.
        self.assertEqual(report.integrity_failures, [])
        self.assertEqual(report.verdict, "GREEDY_IDENTICAL")

    def test_to_dict_is_json_serializable(self):
        import json

        ref, cand = self._build({"a": [1, 2]}, {"a": [1, 2]})
        report = gi.compare(ref, cand)
        d = report.to_dict()
        # Round-trips through JSON without error.
        json.loads(json.dumps(d))
        self.assertEqual(d["verdict"], "GREEDY_IDENTICAL")
        self.assertIsInstance(d["per_prompt"], list)


class LoadDecodeOutputsTests(unittest.TestCase):
    def test_loads_and_keys_by_id(self):
        path = _write_jsonl([
            _record("a", [1, 2, 3], sha=True),
            _record("b", [4, 5], sha=True),
        ])
        try:
            recs = gi.load_decode_outputs(path)
            self.assertEqual(set(recs), {"a", "b"})
            self.assertEqual(recs["a"]["completion_token_ids"], [1, 2, 3])
        finally:
            os.remove(path)

    def test_blank_lines_ignored(self):
        path = _write_jsonl([
            _record("a", [1, 2], sha=True),
            "",
            "   ",
            _record("b", [3, 4], sha=True),
        ])
        try:
            recs = gi.load_decode_outputs(path)
            self.assertEqual(set(recs), {"a", "b"})
        finally:
            os.remove(path)

    def test_fallback_to_prompt_sha256(self):
        import json

        path = _write_jsonl([
            json.dumps({"prompt_sha256": "ph1", "completion_token_ids": [1]}),
        ])
        try:
            recs = gi.load_decode_outputs(path)
            self.assertEqual(set(recs), {"ph1"})
        finally:
            os.remove(path)

    def test_malformed_json_raises_valueerror(self):
        path = _write_jsonl([
            _record("a", [1, 2], sha=True),
            "{not valid json",
        ])
        try:
            with self.assertRaises(ValueError):
                gi.load_decode_outputs(path)
        finally:
            os.remove(path)

    def test_empty_file_raises_valueerror(self):
        path = _write_jsonl([])  # creates an empty file
        try:
            with self.assertRaises(ValueError):
                gi.load_decode_outputs(path)
        finally:
            os.remove(path)

    def test_blank_only_file_raises_valueerror(self):
        path = _write_jsonl(["", "   "])
        try:
            with self.assertRaises(ValueError):
                gi.load_decode_outputs(path)
        finally:
            os.remove(path)

    def test_duplicate_key_raises_valueerror(self):
        path = _write_jsonl([
            _record("a", [1, 2], sha=True),
            _record("a", [3, 4], sha=True),
        ])
        try:
            with self.assertRaises(ValueError):
                gi.load_decode_outputs(path)
        finally:
            os.remove(path)

    def test_missing_completion_token_ids_raises_valueerror(self):
        import json

        path = _write_jsonl([
            json.dumps({"id": "a"}),  # no completion_token_ids
        ])
        try:
            with self.assertRaises(ValueError):
                gi.load_decode_outputs(path)
        finally:
            os.remove(path)

    def test_non_list_completion_token_ids_raises_valueerror(self):
        import json

        # A string would otherwise be iterated char-by-char downstream.
        path = _write_jsonl([
            json.dumps({"id": "a", "completion_token_ids": "12"}),
        ])
        try:
            with self.assertRaises(ValueError):
                gi.load_decode_outputs(path)
        finally:
            os.remove(path)

    def test_null_completion_token_ids_raises_valueerror(self):
        import json

        path = _write_jsonl([
            json.dumps({"id": "a", "completion_token_ids": None}),
        ])
        try:
            with self.assertRaises(ValueError):
                gi.load_decode_outputs(path)
        finally:
            os.remove(path)

    def test_non_int_element_raises_valueerror(self):
        import json

        path = _write_jsonl([
            json.dumps({"id": "a", "completion_token_ids": [1, "2", 3]}),
        ])
        try:
            with self.assertRaises(ValueError):
                gi.load_decode_outputs(path)
        finally:
            os.remove(path)

    def test_loader_error_includes_line_number(self):
        import json

        path = _write_jsonl([
            _record("a", [1, 2], sha=True),
            json.dumps({"id": "b", "completion_token_ids": "oops"}),
        ])
        try:
            with self.assertRaises(ValueError) as ctx:
                gi.load_decode_outputs(path)
            self.assertIn("line 2", str(ctx.exception))
            self.assertIn("b", str(ctx.exception))
        finally:
            os.remove(path)

    def test_mixed_type_keys_are_coerced_to_str(self):
        import json

        # One record uses an int id, another a str id. Without coercion the
        # downstream sorted() would crash; coercion makes them both str.
        path = _write_jsonl([
            json.dumps({"id": 1, "completion_token_ids": [1, 2]}),
            json.dumps({"id": "a", "completion_token_ids": [3, 4]}),
        ])
        try:
            recs = gi.load_decode_outputs(path)
            self.assertEqual(set(recs), {"1", "a"})
        finally:
            os.remove(path)


class CompareFilesTests(unittest.TestCase):
    def test_compare_files_end_to_end(self):
        ref_path = _write_jsonl([
            _record("a", [1, 2, 3], sha=True),
            _record("b", [4, 5], sha=True),
        ])
        cand_path = _write_jsonl([
            _record("a", [1, 2, 3], sha=True),
            _record("b", [4, 9], sha=True),
        ])
        try:
            report = gi.compare_files(ref_path, cand_path)
            self.assertEqual(report.verdict, "DIVERGENT")
            self.assertEqual(report.num_divergent, 1)
        finally:
            os.remove(ref_path)
            os.remove(cand_path)

    def test_record_order_does_not_affect_verdict(self):
        # Same records in different line orders must compare identical.
        ref_path = _write_jsonl([
            _record("a", [1, 2, 3], sha=True),
            _record("b", [4, 5], sha=True),
            _record("c", [6], sha=True),
        ])
        cand_path = _write_jsonl([
            _record("c", [6], sha=True),
            _record("a", [1, 2, 3], sha=True),
            _record("b", [4, 5], sha=True),
        ])
        try:
            report = gi.compare_files(ref_path, cand_path)
            self.assertEqual(report.verdict, "GREEDY_IDENTICAL")
            self.assertEqual(report.num_identical, 3)
            self.assertEqual(report.num_divergent, 0)
            self.assertEqual(report.missing_in_candidate, [])
            self.assertEqual(report.missing_in_reference, [])
        finally:
            os.remove(ref_path)
            os.remove(cand_path)


if __name__ == "__main__":
    unittest.main()
