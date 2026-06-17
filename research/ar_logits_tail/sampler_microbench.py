#!/usr/bin/env python
"""Greedy-argmax sampler microbench for the AR logits->token tail card (#604).

The temp=0 served sampler reduces the final [1, 262144] logits row to one token.
This bounds the GPU cost of the sampler tail and tests whether a faster
byte-identical argmax exists. Also probes FlashInfer-sampler tie-break parity
(the card flags FLASHINFER_SAMPLER=1 as likely NOT byte-identical).

LOCAL GPU microbench. Run when the server is idle (shares the GPU).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import torch

VOCAB = 262144
DEV = "cuda"
ITERS = 2000
WARM = 200


def bench(fn, *a):
    for _ in range(WARM):
        fn(*a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000.0  # ms


def main() -> int:
    torch.manual_seed(0)
    res = {"vocab": VOCAB, "iters": ITERS}

    # realistic single-row decode logits (bf16 compute -> fp32 logits is typical)
    logits32 = torch.randn(1, VOCAB, device=DEV, dtype=torch.float32)
    logits16 = logits32.to(torch.bfloat16)

    res["argmax_fp32_ms"] = bench(lambda x: torch.argmax(x, dim=-1), logits32)
    res["argmax_bf16_ms"] = bench(lambda x: torch.argmax(x, dim=-1), logits16)
    # max+indices (what some samplers call) and topk(1)
    res["max_fp32_ms"] = bench(lambda x: torch.max(x, dim=-1), logits32)
    res["topk1_fp32_ms"] = bench(lambda x: torch.topk(x, 1, dim=-1), logits32)
    # full greedy "vllm-ish": argmax then .item() host sync (per-step host roundtrip)
    def greedy_with_sync(x):
        t = torch.argmax(x, dim=-1)
        return int(t.item())
    res["argmax_plus_item_sync_fp32_ms"] = bench(greedy_with_sync, logits32)

    # FlashInfer sampler tie-break parity probe (byte-identity gate input)
    fi = {"available": False}
    try:
        import flashinfer.sampling as fis  # noqa
        fi["available"] = True
        # Build a logits row with an exact tie at two indices to test tie-break.
        tie = torch.full((1, VOCAB), -10.0, device=DEV, dtype=torch.float32)
        i_lo, i_hi = 100, 200000
        tie[0, i_lo] = 5.0
        tie[0, i_hi] = 5.0  # exact tie; torch.argmax returns the FIRST (lowest idx)
        torch_pick = int(torch.argmax(tie, dim=-1).item())
        fi["torch_argmax_tie_pick"] = torch_pick
        fi["torch_picks_lowest_index"] = (torch_pick == i_lo)
        try:
            # greedy via flashinfer: argmax over probs (temp=0 path)
            probs = torch.softmax(tie, dim=-1)
            fipick = fis.sampling_from_probs(probs, deterministic=True)
            fi["flashinfer_tie_pick"] = int(fipick.item())
            fi["flashinfer_matches_torch_tiebreak"] = (int(fipick.item()) == torch_pick)
        except Exception as e:
            fi["flashinfer_call_error"] = str(e)
    except Exception as e:
        fi["import_error"] = str(e)
    res["flashinfer"] = fi

    Path("research/ar_logits_tail/sampler_microbench.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
