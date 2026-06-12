"""Core comparison library for the Gemma greedy-identity verifier.

Checks the challenge's hard validity rule: a served endpoint's greedy decode
must be TOKEN-IDENTICAL to plain greedy decode of the same checkpoint. We
compare two harness ``decode_outputs.jsonl`` files (a CANDIDATE submission vs an
EXACT-GREEDY REFERENCE) and emit a verdict.

Standard library only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional


def sha256_tokens(token_ids: list[int]) -> str:
    """Authoritative harness recipe: sha256 of the comma-joined decimal token
    ids, ascii-encoded.

    ``sha256(",".join(str(t) for t in token_ids).encode("ascii")).hexdigest()``
    """
    joined = ",".join(str(t) for t in token_ids)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def load_decode_outputs(path: "str | os.PathLike") -> dict[str, dict]:
    """Read a ``decode_outputs.jsonl`` file into a dict keyed by record ``id``
    (falling back to ``prompt_sha256`` when ``id`` is absent). Keys are coerced
    to ``str`` so mixed int/str ids across records sort and dedupe consistently.

    Blank lines are ignored. Raises a clear ``ValueError`` on an empty file (no
    records), any malformed JSON line, a missing/duplicate key, or a record
    whose ``completion_token_ids`` is absent or is not a list of ints.
    """
    records: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {lineno} of {path}: {exc}"
                ) from exc

            key = record.get("id")
            if key is None:
                key = record.get("prompt_sha256")
            if key is None:
                raise ValueError(
                    f"Record on line {lineno} of {path} has neither 'id' nor "
                    "'prompt_sha256'"
                )
            # Coerce to str so mixed-type ids (e.g. int 1 vs str "1") don't
            # crash sorted() downstream and match the PromptComparison.key hint.
            key = str(key)

            # Require completion_token_ids and validate it is a list of ints.
            # bool is a subclass of int but is not a valid token id.
            ids = record.get("completion_token_ids")
            if ids is None:
                raise ValueError(
                    f"Record {key!r} on line {lineno} of {path} is missing "
                    "required field 'completion_token_ids'"
                )
            if not isinstance(ids, list):
                raise ValueError(
                    f"Record {key!r} on line {lineno} of {path} has "
                    f"'completion_token_ids' that is not a list (got "
                    f"{type(ids).__name__})"
                )
            for pos, tok in enumerate(ids):
                if not isinstance(tok, int) or isinstance(tok, bool):
                    raise ValueError(
                        f"Record {key!r} on line {lineno} of {path} has a "
                        f"non-int token at index {pos} in "
                        f"'completion_token_ids' (got {type(tok).__name__})"
                    )

            if key in records:
                raise ValueError(
                    f"Duplicate key {key!r} on line {lineno} of {path}"
                )
            records[key] = record

    if not records:
        raise ValueError(f"No records found in {path} (empty file)")

    return records


@dataclass
class PromptComparison:
    """Per-prompt comparison result."""

    key: str
    identical: bool
    ref_len: int
    cand_len: int
    length_mismatch: bool
    first_divergence_index: Optional[int]
    num_divergent_tokens: int
    num_compared: int
    stored_sha_consistent: Optional[bool]


@dataclass
class ComparisonReport:
    """Aggregate comparison result across all prompts."""

    verdict: str  # "GREEDY_IDENTICAL" | "DIVERGENT" | "INCOMPARABLE"
    num_prompts_compared: int
    num_identical: int
    num_divergent: int
    total_tokens_compared: int
    total_divergent_tokens: int
    missing_in_candidate: list[str]
    missing_in_reference: list[str]
    integrity_failures: list[str]
    per_prompt: list[PromptComparison]

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return dataclasses.asdict(self)


def _stored_sha_consistent(record: dict) -> Optional[bool]:
    """True/False if the record carries a stored completion_token_sha256, else
    None when the field is absent."""
    stored = record.get("completion_token_sha256")
    if stored is None:
        return None
    ids = record.get("completion_token_ids", [])
    return stored == sha256_tokens(ids)


def compare(reference: dict, candidate: dict) -> ComparisonReport:
    """Compare two loaded decode-output dicts and produce a ComparisonReport.

    Verdict semantics:
      * GREEDY_IDENTICAL: same set of keys AND every prompt's
        completion_token_ids are exactly equal.
      * DIVERGENT: key sets match but >=1 prompt differs (length differences
        count as divergence).
      * INCOMPARABLE: key sets differ, OR any stored-sha integrity failure
        exists (data can't be trusted).
    """
    ref_keys = set(reference)
    cand_keys = set(candidate)

    missing_in_candidate = sorted(ref_keys - cand_keys)
    missing_in_reference = sorted(cand_keys - ref_keys)

    # Detect stored-sha integrity failures across every record we have.
    integrity_failures: list[str] = []
    for key, record in reference.items():
        if _stored_sha_consistent(record) is False:
            integrity_failures.append(key)
    for key, record in candidate.items():
        consistent = _stored_sha_consistent(record)
        if consistent is False and key not in integrity_failures:
            integrity_failures.append(key)
    integrity_failures.sort()

    # Compare the prompts present in both files.
    common_keys = sorted(ref_keys & cand_keys)
    per_prompt: list[PromptComparison] = []
    num_identical = 0
    num_divergent = 0
    total_tokens_compared = 0
    total_divergent_tokens = 0

    for key in common_keys:
        ref_ids = reference[key].get("completion_token_ids", [])
        cand_ids = candidate[key].get("completion_token_ids", [])
        ref_len = len(ref_ids)
        cand_len = len(cand_ids)
        num_compared = min(ref_len, cand_len)
        length_mismatch = ref_len != cand_len

        # Count divergent positions over the compared range.
        diff_positions = 0
        first_divergence_index: Optional[int] = None
        for idx in range(num_compared):
            if ref_ids[idx] != cand_ids[idx]:
                diff_positions += 1
                if first_divergence_index is None:
                    first_divergence_index = idx

        length_delta = abs(ref_len - cand_len)
        num_divergent_tokens = diff_positions + length_delta

        # If one is a strict prefix of the other (no diffs in compared range
        # but lengths differ), first divergence is at the shorter length.
        if first_divergence_index is None and length_mismatch:
            first_divergence_index = num_compared

        identical = (first_divergence_index is None) and not length_mismatch

        # stored_sha_consistent for the per-prompt row reflects the candidate's
        # own record self-check (the file under test).
        stored_sha_consistent = _stored_sha_consistent(candidate[key])

        per_prompt.append(
            PromptComparison(
                key=key,
                identical=identical,
                ref_len=ref_len,
                cand_len=cand_len,
                length_mismatch=length_mismatch,
                first_divergence_index=first_divergence_index,
                num_divergent_tokens=num_divergent_tokens,
                num_compared=num_compared,
                stored_sha_consistent=stored_sha_consistent,
            )
        )

        total_tokens_compared += num_compared
        total_divergent_tokens += num_divergent_tokens
        if identical:
            num_identical += 1
        else:
            num_divergent += 1

    # Determine the verdict.
    key_sets_match = not missing_in_candidate and not missing_in_reference
    if integrity_failures or not key_sets_match:
        verdict = "INCOMPARABLE"
    elif num_divergent == 0:
        verdict = "GREEDY_IDENTICAL"
    else:
        verdict = "DIVERGENT"

    return ComparisonReport(
        verdict=verdict,
        num_prompts_compared=len(common_keys),
        num_identical=num_identical,
        num_divergent=num_divergent,
        total_tokens_compared=total_tokens_compared,
        total_divergent_tokens=total_divergent_tokens,
        missing_in_candidate=missing_in_candidate,
        missing_in_reference=missing_in_reference,
        integrity_failures=integrity_failures,
        per_prompt=per_prompt,
    )


def compare_files(
    ref_path: "str | os.PathLike", cand_path: "str | os.PathLike"
) -> ComparisonReport:
    """Load both files then compare them."""
    reference = load_decode_outputs(ref_path)
    candidate = load_decode_outputs(cand_path)
    return compare(reference, candidate)
