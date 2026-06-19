#!/usr/bin/env python
"""PR #746: compose the route-b net-TPS K-sweep from MEASURED served arms.

Reads the real served arm_result.json files produced by run_arm.py:
  - arref/arm_result.json            -> wall_tps_AR (M=1 AR target, spec OFF)
  - batched_k{K}/arm_result.json     -> wall_tps_batched(K) + accept_stats(K)

and builds, per K, the route-b (strict byte-exact, K sequential M=1 verify)
net-TPS estimate plus the byte-exactness tax vs the batched fire.

Cost model (see PLAN.md). Per spec step the batched fire does:
    t_step_batched(K) = t_drafter(K) + t_verify_batched(K+1)      (one M=K+1 fwd)
and emits (a+1) tokens, so  wall_tps_batched = (a+1) / t_step_batched.
Route-b keeps the SAME drafter + accept pattern but replaces the single batched
M=K+1 verify forward with K+1 SEQUENTIAL M=1 forwards (byte-identical to decode):
    t_step_routeb(K) = t_drafter(K) + (K+1) * t_M1
With t_verify_batched(K+1) ~= t_M1 (int4 Marlin GEMM is M-invariant, PLAN/#736),
the drafter term cancels and the whole thing is measured:
    t_step_routeb(K) = t_step_batched(K) + K * t_M1
    route_b_tps(K)   = (a+1) / [ (a+1)/wall_tps_batched(K) + K / wall_tps_AR ]
where t_M1 = 1/wall_tps_AR.

Honest caveat (my #642): real in-loop M=1 verify forwards break the decode
cudagraph, so the realized route-b lands BELOW this composition. We therefore
label route_b_tps as an OPTIMISTIC estimate / soft upper bound. The decisive
fact is structural and needs no model: route-b does (K+1)/(a+1) >= 1 target
forwards per emitted token vs plain AR's 1, plus drafter drag, so
route_b_tps(K) < wall_tps_AR for all K -> the AR anchor is route-b's hard ceiling.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BAR = 126.378  # int4_g128_lmhead official AR rung (PR #601)
TAU = 1.03524  # local wall_tps -> official scalar (#267)


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None


def reparse_serve_log(log_path: Path) -> dict:
    """Re-derive aggregate spec acceptance straight from a vLLM v1 serve log.

    Source of truth (independent of any stored accept_stats, which an earlier
    buggy parser may have miswritten). Sums the per-interval 'Accepted: N tokens'
    / 'Drafted: N tokens' counts; mean acceptance length is kept as a cross-check
    (it equals emitted/step = a+1)."""
    out: dict = {}
    if not log_path or not log_path.exists():
        return out
    text = log_path.read_text(errors="ignore")
    acc = [int(x) for x in re.findall(r"Accepted:\s*(\d+)\s*tokens", text)]
    drf = [int(x) for x in re.findall(r"Drafted:\s*(\d+)\s*tokens", text)]
    mal = [float(x) for x in re.findall(r"Mean acceptance length:\s*([0-9.]+)",
                                        text, flags=re.IGNORECASE)]
    if acc:
        out["num_accepted_tokens"] = float(sum(acc))
    if drf:
        out["num_draft_tokens"] = float(sum(drf))
    if mal:
        out["mean_acceptance_length"] = sum(mal) / len(mal)
    out["spec_metric_intervals"] = float(len(acc))
    return out


def accept_len(stats: dict, k: int) -> tuple[float | None, str]:
    """Mean accepted DRAFT tokens per step `a` (emitted/step = a+1).

    Primary: the exact aggregate ``a = K * total_accepted / total_drafted`` from
    the summed per-interval counts (each step drafts K tokens). Cross-check:
    vLLM's reported 'Mean acceptance length' is emitted/step = a+1, so
    ``a = mean_acceptance_length - 1``. Returns (a, source-tag)."""
    na, nd = stats.get("num_accepted_tokens"), stats.get("num_draft_tokens")
    if na is not None and nd:
        return float(k) * float(na) / float(nd), "K*accepted/drafted"
    if "mean_acceptance_length" in stats:  # vLLM value = emitted/step = a+1
        return max(0.0, float(stats["mean_acceptance_length"]) - 1.0), "mean_accept_len-1"
    return None, "MISSING"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="research/strict_clean_routeb_m1verify")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--ar-dir", default="arref",
                    help="subdir holding the AR-ceiling arm_result.json. Use "
                         "'arref_fastkern' for route-b's TRUE (fast-kernel, no "
                         "batchinv tax) ceiling; 'arref' is the batchinv-ON floor.")
    ap.add_argument("--out", default=None, help="write the composed table JSON here")
    args = ap.parse_args(argv)

    root = Path(args.root)
    ar = _load(root / args.ar_dir / "arm_result.json")
    if ar is None or not ar.get("wall_tps"):
        print(f"[compose] missing AR anchor at {root/args.ar_dir/'arm_result.json'}")
        return 1
    tps_ar = float(ar["wall_tps"])
    t_m1 = 1.0 / tps_ar
    print(f"[compose] AR anchor wall_tps={tps_ar:.3f} (official_proj={tps_ar*TAU:.3f}); "
          f"t_M1={t_m1*1e3:.3f} ms/tok; route-b HARD CEILING={tps_ar:.3f}")
    print(f"[compose] bar={BAR}  AR clears bar? {tps_ar > BAR}\n")

    rows = []
    hdr = (f"{'K':>2} {'accept_len a':>12} {'wall_tps_bat':>12} {'tax_vs_AR':>10} "
           f"{'rb_upper':>9} {'rb_est':>9} {'rb_up>bar':>9} {'bat>bar':>8} {'a_src':>18}")
    print(hdr)
    print("-" * len(hdr))
    for k in args.ks:
        rec = _load(root / f"batched_k{k}" / "arm_result.json")
        if rec is None or not rec.get("wall_tps"):
            print(f"{k:>2} {'(missing batched arm)':>12}")
            continue
        tps_bat = float(rec["wall_tps"])
        # Re-parse the serve log as source of truth; fall back to stored stats.
        serve_log = root / f"batched_k{k}" / f"serve_batched_k{k}.log"
        stats = reparse_serve_log(serve_log) or (rec.get("accept_stats", {}) or {})
        if not stats.get("num_accepted_tokens"):
            stats = rec.get("accept_stats", {}) or stats
        a, src = accept_len(stats, k)
        if a is None:
            routeb = routeb_upper = None
            routeb_s = "n/a(no a)"
        else:
            emitted = a + 1.0
            # Drafter-inclusive estimate (cost model: replace the one batched
            # M=K+1 verify with K+1 sequential M=1 verifies, keeping the drafter).
            t_step_routeb = emitted / tps_bat + k * t_m1
            routeb = emitted / t_step_routeb
            # Drafter-FREE optimistic upper bound: route-b can never beat
            # tps_AR * (a+1)/(K+1) -- it does (K+1) M=1 target forwards to emit
            # (a+1) tokens vs plain AR's 1 forward/token, and the drafter only
            # adds cost. If even THIS < bar, route-b is dead at K.
            routeb_upper = tps_ar * emitted / (k + 1.0)
            routeb_s = f"{routeb:.3f}"
        tax = tps_bat - tps_ar  # batched advantage over the byte-exact AR floor
        row = {
            "k": k, "accept_len": a, "accept_len_src": src,
            "num_accepted_tokens": stats.get("num_accepted_tokens"),
            "num_draft_tokens": stats.get("num_draft_tokens"),
            "mean_acceptance_length_vllm": stats.get("mean_acceptance_length"),
            "wall_tps_batched": tps_bat, "wall_tps_batched_official_proj": tps_bat * TAU,
            "batched_minus_AR": tax,
            "routeb_tps_est": routeb,
            "routeb_tps_est_official_proj": (routeb * TAU) if routeb else None,
            "routeb_tps_upper": routeb_upper,
            "routeb_tps_upper_official_proj": (routeb_upper * TAU) if routeb_upper else None,
            "routeb_clears_bar": (bool(routeb > BAR) if routeb else None),
            "routeb_upper_clears_bar": (bool(routeb_upper > BAR) if routeb_upper else None),
            "batched_clears_bar": bool(tps_bat > BAR),
            "ppl_batched": rec.get("ppl"),
        }
        rows.append(row)
        a_s = f"{a:.3f}" if a is not None else "n/a"
        rb_up_s = f"{routeb_upper:.3f}" if routeb_upper else "n/a"
        rb_up_bar = (str(bool(routeb_upper > BAR)) if routeb_upper else "n/a")
        print(f"{k:>2} {a_s:>12} {tps_bat:>12.3f} {tax:>+10.3f} {rb_up_s:>9} "
              f"{routeb_s:>9} {rb_up_bar:>9} {str(tps_bat>BAR):>8} {src:>18}")

    summary = {
        "bar_official_tps": BAR, "tau": TAU,
        "wall_tps_AR": tps_ar, "wall_tps_AR_official_proj": tps_ar * TAU,
        "AR_clears_bar": bool(tps_ar > BAR),
        "routeb_hard_ceiling_tps": tps_ar,
        "ppl_AR": ar.get("ppl"),
        "rows": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\n[compose] wrote {args.out}")
    else:
        print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
