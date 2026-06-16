"""PR #500 deliverable (2): diagnose the broken serve-identity gate.

Finding (static, from vLLM source): VLLM_BATCH_INVARIANT=1 patches ONLY the
matmul family (mm/addmm/matmul/linear/bmm/_log_softmax) + sets NCCL/cuBLAS/TF32
determinism env. It does NOT touch the attention kernel. So `full_flag`'s Gemma
global full-attn layer stays the STOCK adaptive 3D split-KV -> M-dependent ->
full_flag is NOT byte-exact-by-construction on the attention axis (the axis #496
is about). Hence full_flag(M=8) vs full_flag_ref(M=1 AR) cannot read ~1.0.

This script measures the END-TO-END M=8-verify vs M=1-AR token agreement for the
four arms, WITHIN each serve session, using the warmest available rounds:
  serve_run/: byteexact (fixed split-KV, M-inv attn) ; full_flag (batch-inv matmul, adaptive attn)
  control/  : surgical  (2D attn, M-inv)            ; deployed  (adaptive attn, M-dep)

Hypothesis: attention is the dominant M-axis. M-invariant-attn arms (byteexact,
surgical) should have FEW flipped seqs with LONG common prefixes; M-dependent-attn
arms (deployed, full_flag) should flip early & often. The residual end-to-end gap
on the M-inv arms is the bf16-ULP-tie AR-cascade amplifier (a single tie at the
argmax flips greedy and the rest of the seq cascades) -- which is why even
byte-exact arms cannot read 1.0 at the free-running-greedy serve level, and why
the kernel-level microbench (#496: 0/8 raw attn-output byte flips) is the correct
byte-exactness observable.

CPU-only over saved decode jsonls. No GPU/serve.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_round(base: Path, arm: str, rnd: int) -> dict[str, list[int]]:
    f = base / arm / f"decode_round{rnd:02d}.jsonl"
    seqs: dict[str, list[int]] = {}
    if not f.exists():
        return seqs
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        seqs[str(o.get("id"))] = [int(t) for t in o.get("completion_token_ids", [])]
    return seqs


def n_rounds(base: Path, arm: str) -> int:
    return len(list((base / arm).glob("decode_round*.jsonl")))


def compare(sa: dict[str, list[int]], sb: dict[str, list[int]]) -> dict:
    common = sorted(set(sa) & set(sb))
    total = matched = nflip = 0
    prefix_lens, first_div = [], []
    for k in common:
        ta, tb = sa[k], sb[k]
        n = min(len(ta), len(tb))
        p = 0
        while p < n and ta[p] == tb[p]:
            p += 1
        seq_flips = sum(1 for i in range(n) if ta[i] != tb[i])
        total += n
        matched += n - seq_flips
        if seq_flips or len(ta) != len(tb):
            nflip += 1
            prefix_lens.append(p)
            first_div.append(p)
    return {
        "n_prompts": len(common),
        "rate": (matched / total) if total else None,
        "n_flipped": nflip,
        "prefix_min": min(prefix_lens) if prefix_lens else None,
        "prefix_med": statistics.median(prefix_lens) if prefix_lens else None,
        "prefix_max": max(prefix_lens) if prefix_lens else None,
        "first_div_sorted": sorted(first_div),
    }


def warmest_pair(base: Path, arm: str, ref: str):
    """X(warmest speed round) vs X_ref(warmest ref round)."""
    ns, nr = n_rounds(base, arm), n_rounds(base, ref)
    sa = load_round(base, arm, ns - 1)
    sb = load_round(base, ref, nr - 1)
    return compare(sa, sb), ns - 1, nr - 1


def main():
    serve_run, control = HERE / "serve_run", HERE / "control"
    cases = [
        ("byteexact", "byteexact_ref", serve_run, "FIXED split-KV (M-inv attn) + fast Marlin matmul"),
        ("surgical", "surgical_ref", control, "2D attn (M-inv) + fast Marlin matmul"),
        ("full_flag", "full_flag_ref", serve_run, "ADAPTIVE 3D attn (M-DEP) + batch-inv matmul"),
        ("deployed", "deployed_ref", control, "ADAPTIVE 3D attn (M-DEP) + fast Marlin matmul"),
    ]
    print("=" * 92)
    print("M=8 spec-VERIFY  vs  M=1 AR reference   (end-to-end greedy, warmest rounds, within session)")
    print("=" * 92)
    rows = []
    for arm, ref, base, note in cases:
        if not (base / arm).exists() or not (base / ref).exists():
            print(f"  {arm:11s} -- MISSING ({base.name})")
            continue
        c, ra, rb = warmest_pair(base, arm, ref)
        rows.append((arm, c, note))
        print(f"  {arm:10s} vs {ref:15s} [{base.name}]  speed(r{ra}) vs ref(r{rb})")
        print(f"     rate={c['rate']:.4f}  flipped={c['n_flipped']}/{c['n_prompts']}  "
              f"common_prefix(min/med/max)={c['prefix_min']}/{c['prefix_med']}/{c['prefix_max']}")
        print(f"     {note}")
        print(f"     first_div={c['first_div_sorted']}")
        print()

    print("=" * 92)
    print("REF-ARM self consistency  (r0 cold vs r1; only 2 rounds exist -> cold-vs-warm, NOT warm-vs-warm)")
    print("=" * 92)
    for arm, ref, base, _ in cases:
        if not (base / ref).exists():
            continue
        if n_rounds(base, ref) >= 2:
            c = compare(load_round(base, ref, 0), load_round(base, ref, 1))
            print(f"  {ref:15s} [{base.name:9s}] r0-vs-r1 rate={c['rate']:.4f} flipped={c['n_flipped']}/{c['n_prompts']}")

    print()
    print("=" * 92)
    print("RANKING by M=8-vs-M=1AR agreement  (attention-axis test)")
    print("=" * 92)
    for arm, c, note in sorted(rows, key=lambda r: -(r[1]["rate"] or 0)):
        tag = "M-inv attn" if arm in ("byteexact", "surgical") else "M-DEP attn"
        print(f"  {c['rate']:.4f}  {arm:11s} [{tag}]  flipped={c['n_flipped']:2d}  med_prefix={c['prefix_med']}")


if __name__ == "__main__":
    main()
