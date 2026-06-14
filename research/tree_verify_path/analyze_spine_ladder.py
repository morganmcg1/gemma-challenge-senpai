#!/usr/bin/env python3
"""STAGE-2b-spine analysis: turn the salvage-probe verdict JSON into the report
numbers for the advisor's lambda gate (PR #71).

Reads the verdict written by sitecustomize.py::_salvage_probe_dump (the
spine_ladder block) and reports, per spine depth d=1..K:
  - q[d]              measured conditional accept rate (GIVEN the chain reached d)
  - lambda_vs_top1[d] = q[d] / TOP1_MEASURED(0.729)   <- the self-KV recovery
and summary scalars over q[2..K] (the advisor logs q[2..9]; the deployed linear
chain only reaches K=7 -> this run covers q[2..7], q[8..9] need the tree spine).

Also computes the LINEAR-chain E[T] (expected committed tokens incl. bonus) under
the measured ladder vs the flat-0.729 assumption, to isolate the spine self-KV
effect on accept length.

This script does NOT grade GO/NO-GO. It reports the measured tuple honestly;
fern #185's launch_decision(tuple) (run by the advisor) is the official grader.
The forward map anchors below are the advisor's RELAYED anchors, used only for a
ballpark, clearly labelled as interpolation -- not the official calculator.
"""
import json
import sys

TOP1 = 0.729  # TOP1_MEASURED (tree_spec.py)
LAMBDA_BAR = 0.9052  # denken #183 finite-sample LCB bar (advisor relay 2026-06-14T16:19Z)

# advisor's RELAYED forward map anchors (measured-lambda -> predicted LCB-TPS).
FWD_MAP = [(0.342, 404.0), (0.838, 486.0), (0.9052, 500.0), (1.0, 520.95)]


def interp_tps(lam: float) -> float:
    pts = FWD_MAP
    if lam <= pts[0][0]:
        return pts[0][1]
    if lam >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= lam <= x1:
            t = (lam - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return float("nan")


def main(path: str) -> None:
    with open(path) as f:
        v = json.load(f)
    K = v.get("K")
    ladder = v.get("spine_ladder", {})
    decided = v.get("decided_steps")
    skipped = v.get("skipped", {})
    print(f"=== STAGE-2b-spine ladder  (verdict: {path}) ===")
    print(f"tree_M={v.get('tree_M')}  K={K}  decided_steps={decided}  "
          f"aligned_steps={v.get('aligned_steps')}")
    print(f"skipped: {skipped}")
    print(f"top1_accept(measured q[1])={v.get('top1_accept')}  "
          f"(reference TOP1_MEASURED={TOP1})")
    print(f"mean_accepted_len={v.get('mean_accepted_len')}")
    print()
    print(f"{'depth':>5} {'reached':>8} {'accepted':>8} {'q[d]':>8} "
          f"{'lam=q/0.729':>12}")
    lam_vals = []
    for d in range(1, (K or 0) + 1):
        e = ladder.get(str(d), {})
        q = e.get("q")
        lam = e.get("lambda_vs_top1")
        qs = f"{q:.4f}" if q is not None else "  --  "
        lams = f"{lam:.4f}" if lam is not None else "  --  "
        print(f"{d:>5} {e.get('reached', 0):>8} {e.get('accepted', 0):>8} "
              f"{qs:>8} {lams:>12}")
        if d >= 2 and lam is not None and e.get("reached", 0) > 0:
            lam_vals.append((d, lam, e.get("reached", 0)))
    print()
    if lam_vals:
        lams_only = [l for _, l, _ in lam_vals]
        lam_min = min(lams_only)
        d_min = [d for d, l, _ in lam_vals if l == lam_min][0]
        lam_mean = sum(lams_only) / len(lams_only)
        # reached-weighted mean: weights deep (sparser) depths less.
        wsum = sum(r for _, _, r in lam_vals)
        lam_wmean = sum(l * r for _, l, r in lam_vals) / wsum if wsum else float("nan")
        print(f"lambda over q[2..{K}] (the gate range we cover this run):")
        print(f"  min        = {lam_min:.4f}  (at depth {d_min})")
        print(f"  mean       = {lam_mean:.4f}")
        print(f"  reached-wtd= {lam_wmean:.4f}")
        print(f"  BAR        = {LAMBDA_BAR:.4f}  (denken #183 LCB)")
        for name, val in (("min", lam_min), ("mean", lam_mean),
                          ("reached-wtd", lam_wmean)):
            verdict = "PASS" if val >= LAMBDA_BAR else "FAIL"
            print(f"  [{verdict}] lambda_{name}={val:.4f} vs {LAMBDA_BAR} "
                  f"-> ballpark TPS~{interp_tps(val):.1f} (interp, NOT official)")
    print()
    # Linear-chain E[T] (incl. bonus): flat-0.729 vs measured-q ladder.
    et_flat = 1.0
    et_meas = 1.0
    prod_flat = 1.0
    prod_meas = 1.0
    for d in range(1, (K or 0) + 1):
        q = ladder.get(str(d), {}).get("q")
        prod_flat *= TOP1
        et_flat += prod_flat
        if q is not None:
            prod_meas *= q
            et_meas += prod_meas
    print("Linear-chain E[T] (expected committed tokens incl. bonus, depths 1..K):")
    print(f"  flat-0.729 assumption : {et_flat:.4f}")
    print(f"  measured ladder       : {et_meas:.4f}")
    if et_flat > 1:
        print(f"  retention (meas/flat) : {et_meas / et_flat:.4f}")
    print()
    print("NOTE: q[8],q[9] (tree spine depths beyond linear K=7) and branch-interior")
    print("lambda are NOT in this run -> they need STAGE-2b-branch (scratch-KV tree")
    print("forward). This is the SPINE portion of the lambda gate only.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "research/tree_verify_path/comp_salvage_probe_stage2b_spine_M16.json")
