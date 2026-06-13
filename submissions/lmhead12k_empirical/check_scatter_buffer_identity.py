#!/usr/bin/env python
"""Bit-identity + token-identity test for the PR #41 persistent scatter buffer.

Validates the REAL plugin helper ``scatter_kept_to_full`` (vllm_lmhead12k/model.py)
against a naive ``torch.full(-inf) + scatter`` reference over random partial
logits and a varying-M stream that exercises BOTH the persistent decode/verify
path (M <= max_persist_M) and the fresh-alloc prefill path (M > max_persist_M),
in bf16 and fp32, using the COMMITTED strictly-ascending kept_ids.

Asserts, for every call:
  (1) output rows[:M] are BIT-IDENTICAL to the fresh reference (dead columns
      -inf, kept columns carry partial) -> greedy argmax and prompt_logprobs are
      unchanged by construction;
  (2) argmax(output) == kept_ids[argmax(partial)] (the PR #41 scatter-free
      token identity), incl. forced exact ties;
  (3) the persistent buffer stays correct across a grow/shrink/dtype-switch
      stream (no stale rows or columns leak between calls).

Runs on GPU if available, else CPU. No model load.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "submissions/lmhead12k_empirical/vllm_plugin"
KEPT_IDS = ROOT / "submissions/lmhead12k_empirical/kept_ids.json"
sys.path.insert(0, str(PLUGIN))

from vllm_lmhead12k.model import scatter_kept_to_full  # noqa: E402

FULL_VOCAB = 262144
MAX_PERSIST_M = 64


def naive(partial, kept_ids):
    M = partial.shape[0]
    full = torch.full((M, FULL_VOCAB), float("-inf"),
                      dtype=partial.dtype, device=partial.device)
    full.scatter_(1, kept_ids.unsqueeze(0).expand(M, -1), partial)
    return full


def main() -> int:
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kept = json.load(open(KEPT_IDS))["kept_ids"]
    assert all(kept[i] < kept[i + 1] for i in range(len(kept) - 1)), \
        "kept_ids must be strictly ascending for the token-identity proof"
    kept_ids = torch.tensor(kept, dtype=torch.long, device=dev)
    KEPT = len(kept)
    g = torch.Generator(device="cpu").manual_seed(0)

    # varying-M stream: small (persistent), grows past the cap (fresh), shrinks,
    # boundary at the cap, then dtype switch. Mirrors decode<->prefill interleave.
    stream = [1, 45, 7, 64, 65, 128, 17, 1, 64, 49, 200, 4, 33]
    n_bit = n_tok = n_calls = 0
    for dtype in (torch.bfloat16, torch.float32):
        buf = None
        for M in stream:
            partial = (torch.randn(M, KEPT, generator=g, dtype=torch.float32) * 6.0).to(
                dev, dtype=dtype)
            # inject an exact tie in a few rows: copy the row max into another col
            if M >= 2:
                am = partial.argmax(dim=-1)
                rowmax = partial.gather(1, am.unsqueeze(1))
                other = (am + 1) % KEPT
                partial.scatter_(1, other.unsqueeze(1), rowmax)
            out, buf = scatter_kept_to_full(partial, kept_ids, FULL_VOCAB, buf,
                                            MAX_PERSIST_M)
            ref = naive(partial, kept_ids)
            # (1) bit-identity
            assert out.shape == ref.shape
            assert torch.equal(out, ref), f"BIT MISMATCH dtype={dtype} M={M}"
            # (2) token identity vs kept_ids[argmax(partial)]
            tok_out = out.argmax(dim=-1)
            tok_remap = kept_ids[partial.argmax(dim=-1)]
            assert torch.equal(tok_out, tok_remap), \
                f"TOKEN MISMATCH dtype={dtype} M={M}"
            # buffer invariant: persistent path returns a view into a [cap, V] buf
            if M <= MAX_PERSIST_M:
                assert buf is not None and buf.shape[0] == MAX_PERSIST_M
            n_bit += int(torch.equal(out, ref))
            n_tok += int(torch.equal(tok_out, tok_remap))
            n_calls += 1

    report = {
        "verdict": "PASS",
        "device": str(dev),
        "calls": n_calls,
        "bit_identical_calls": n_bit,
        "token_identical_calls": n_tok,
        "max_persist_M": MAX_PERSIST_M,
        "stream_M": stream,
        "kept_strictly_ascending": True,
    }
    print(json.dumps(report, indent=2))
    ok = n_bit == n_calls and n_tok == n_calls
    print("[scatter-buffer] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
