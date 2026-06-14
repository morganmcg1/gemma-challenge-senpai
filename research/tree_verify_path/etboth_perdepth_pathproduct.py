"""PR #71 STAGE-2b: E[T]_both from the MEASURED per-depth acceptance ladder.

The validated closed form ``tree_spec.expected_committed_tokens`` applies one
FLAT per-rank ``p`` to every edge regardless of depth. Our STAGE-2b measurement
shows the spine accept ``q[d]`` RISES with depth (survivorship), so the flat form
UNDER-counts. This script feeds the measured per-depth ladder + measured
branch-hit into the SAME path-product formula
(``F = Sum_v prod_{u on root->v} p_edge(u)``) so the reported E[T] reflects the
real stack, not the borrowed/flat projection.

Inputs (all measured locally on the deployed frontier stack, this PR):
  * deployed-faithful spine ladder q[1..7]      <- salvage probe (reads production
    verify argmaxes), comp_salvage_probe_stage1_verdict_long.json  (8650 steps)
  * deep spine q[8],q[9]                          <- live M=16 scratch tree forward,
    comp_verify_probe_stage2b_m16_long.json (8700 steps). scratch pos d == q[d+1].
    These are DEFLATED lower bounds (near-tie realization diff); a deflation-
    corrected variant is also reported.
  * branch-hit pos0, pos1                         <- salvage probe per_position
    (rank-2 catch | rank-1 miss at a width>=2 divergence).

No GPU, no torch. Pure replay of the validated tree_spec path-product on the
measured numbers. Run:  python3 research/tree_verify_path/etboth_perdepth_pathproduct.py
"""

from __future__ import annotations

import json
import sys
from importlib import util as _u
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SALVAGE = ROOT / "research/tree_verify_path/comp_salvage_probe_stage1_verdict_long.json"
SCRATCH = (
    ROOT
    / "submissions/fa2sw_treeverify_kenyan/research/tree_verify_path"
    / "comp_verify_probe_stage2b_m16_long.json"
)

# --- import the validated tree_spec (topology + flat-p closed form anchor) -----
_spec = _u.spec_from_file_location("tree_spec", ROOT / "scripts/profiler/tree_spec.py")
ts = _u.module_from_spec(_spec)
sys.modules["tree_spec"] = ts
_spec.loader.exec_module(ts)

TOP1 = ts.TOP1_MEASURED  # 0.729 -- the lambda normaliser (binding bar is on lambda=q/0.729)
BAR = 0.9780  # binding private both-bugs bar (stark #208 / #191)


def load_measured_ladder():
    """Return q[1..9] (deployed-faithful 1..7, scratch deep 8..9) + branch-hit."""
    sal = json.loads(SALVAGE.read_text())
    scr = json.loads(SCRATCH.read_text())
    q = {}
    for d_str, rung in sal["spine_ladder"].items():
        q[int(d_str)] = rung["q"]  # deployed-faithful q[1..7]
    # scratch pos d corresponds to deployed depth d+1; take the deep tail 8,9
    scr_ladder = scr["spine_ladder"]
    q[8] = scr_ladder["7"]["q"]  # scratch pos 7 == q[8]  (n=268)
    q[9] = scr_ladder["8"]["q"]  # scratch pos 8 == q[9]  (n=192)
    branch_hit = {
        0: sal["per_position"]["0"]["branch_hit"],
        1: sal["per_position"]["1"]["branch_hit"],
    }
    # reached counts for the deep tail (for the conservative-LCB note)
    deep_n = {8: scr_ladder["7"]["reached"], 9: scr_ladder["8"]["reached"]}
    return q, branch_hit, deep_n, sal, scr


def deflation_overlap(sal, scr):
    """Per-depth scratch-vs-deployed lambda deflation over the q[1..7] overlap.
    scratch pos d == deployed depth d+1. Returns the typical DEEP-overlap delta
    used to correct the scratch deep tail upward (conservative lower bound)."""
    deltas = {}
    for d in range(1, 8):
        dep = sal["spine_ladder"][str(d)]["q"] / TOP1
        sc = scr["spine_ladder"][str(d - 1)]["q"] / TOP1
        deltas[d] = dep - sc  # positive: deployed sits above scratch
    # deep-overlap-typical = mean of the two deepest overlap rungs (q6,q7)
    deep_typ = (deltas[6] + deltas[7]) / 2.0
    return deltas, deep_typ


def et_perdepth(tree, q, branch_hit):
    """Per-depth path-product E[T] decomposed into spine-only (descent) and
    spine+branch (both). Each rank-1 edge into depth d uses q[d]; each rank-2
    edge into depth d uses (1-q[d])*branch_hit[divergence_position]. Returns
    (E_descent_committed, E_both_committed, contributions)."""
    pp = [0.0] * tree.num_nodes
    pp[0] = 1.0
    spine_set = set(tree.spine)
    e_descent = 1.0  # bonus token
    e_both = 1.0
    contrib = []
    # map a rank-2 branch root to a divergence-position index (0-based along spine)
    # node's parent's depth == the spine depth at which the rank-1 sibling diverges
    for i in range(1, tree.num_nodes):
        par = tree.parent[i]
        d = tree.depth[i]  # this node sits one below its parent's spine depth
        rank = tree.rank_in_parent[i]
        if rank == 1:
            p_edge = q[d]
        else:
            # rank-2: rank-1 sibling (at same depth d) missed, then rank-2 caught.
            div_pos = tree.depth[par]  # 0-based spine divergence position
            bh = branch_hit.get(div_pos, branch_hit[max(branch_hit)])
            p_edge = (1.0 - q[d]) * bh
        pp[i] = pp[par] * p_edge
        e_both += pp[i]
        if i in spine_set:
            e_descent += pp[i]
        contrib.append((i, rank, d, round(p_edge, 5), round(pp[i], 6)))
    return e_descent, e_both, contrib


def chain_et(q, k):
    """Deployed linear chain committed E[T] over the measured ladder q[1..k]."""
    tot, prod = 1.0, 1.0
    for d in range(1, k + 1):
        prod *= q[d]
        tot += prod
    return tot


def report(q, branch_hit, deep_n, label):
    t16 = ts.TreeSpec(ts.PARENT_M16)
    e_desc, e_both, contrib = et_perdepth(t16, q, branch_hit)
    chain7 = chain_et(q, 7)  # deployed K=7 reference, same measured rates
    # flat-p closed-form cross-check (the established 3.974 anchor)
    p_flat = [TOP1, (1 - TOP1) * 0.4165]
    f_flat = ts.expected_committed_tokens(t16, p_flat)
    print(f"\n===== {label} =====")
    print(f"  q ladder (1..9): " + " ".join(f"{q[d]:.3f}" for d in range(1, 10)))
    print(f"  lambda (q/{TOP1}): " + " ".join(f"{q[d]/TOP1:.3f}" for d in range(1, 10)))
    print(f"  min lambda over q[2..9] = {min(q[d]/TOP1 for d in range(2,10)):.4f}  (bar {BAR})")
    print(f"  deep n: q8={deep_n[8]}  q9={deep_n[9]}")
    print(f"  deployed K=7 chain  E[T]committed = {chain7:.4f}  (accepted {chain7-1:.4f})")
    print(f"  M16 descent (spine) E[T]committed = {e_desc:.4f}  (+{100*(e_desc/chain7-1):.1f}% vs chain)")
    print(f"  M16 BOTH (spine+br) E[T]committed = {e_both:.4f}  (+{100*(e_both/chain7-1):.1f}% vs chain)")
    print(f"    [flat-p closed-form anchor M16 = {f_flat:.4f}; per-depth lifts it via rising ladder]")
    print(f"  relative gain  E[T]_both / chain   = {e_both/chain7:.4f}")
    print(f"  COST-FREE upper-bound TPS (481.53 x rel.gain, ignores verify+drafter cost) = {481.53*e_both/chain7:.1f}")
    print(f"    [realistic: fern #185 cost-aware map caps lambda=1 at 520.95 TPS; trigger 512.41 -> clears +8.5]")
    return e_desc, e_both, chain7, f_flat


def main():
    q, branch_hit, deep_n, sal, scr = load_measured_ladder()
    deltas, deep_typ = deflation_overlap(sal, scr)
    print("scratch-vs-deployed lambda deflation over q[1..7] overlap:")
    print("  " + "  ".join(f"q{d}:{deltas[d]:+.3f}" for d in range(1, 8)))
    print(f"  deep-overlap-typical (mean q6,q7) = {deep_typ:+.3f}  -> deep-tail upward correction")
    print(f"  branch-hit: pos0={branch_hit[0]:.4f}  pos1={branch_hit[1]:.4f}")

    # (A) CONSERVATIVE: raw scratch deep tail (deflated lower bound)
    e_desc_c, e_both_c, chain_c, f_flat = report(q, branch_hit, deep_n, "CONSERVATIVE (raw scratch q8,q9)")

    # (B) DEFLATION-CORRECTED: lift q8,q9 by the deep-overlap-typical lambda delta.
    # No lambda=1 cap: survivorship makes lambda>1 physical (deployed q[2..7] reach
    # lambda 1.167), and the raw scratch is a strict LOWER bound, so correction only
    # adds; capping would spuriously pull q9 (raw lambda 1.064) back down.
    q_corr = dict(q)
    q_corr[8] = (q[8] / TOP1 + deep_typ) * TOP1
    q_corr[9] = (q[9] / TOP1 + deep_typ) * TOP1
    e_desc_d, e_both_d, chain_d, _ = report(q_corr, branch_hit, deep_n, "DEFLATION-CORRECTED (q8,q9 += deep-typ)")

    # robustness: how much does E[T]_both move between the two deep-tail variants?
    print(f"\n  E[T]_both robustness to deep-tail (q8,q9) uncertainty: "
          f"{e_both_c:.4f} (raw) -> {e_both_d:.4f} (corrected), delta {e_both_d-e_both_c:+.4f}")

    verdict = {
        "stage": "2b_etboth_perdepth",
        "topology": "PARENT_M16",
        "bar_lambda": BAR,
        "top1_normaliser": TOP1,
        "branch_hit_pos0": branch_hit[0],
        "branch_hit_pos1": branch_hit[1],
        "deep_n": deep_n,
        "q_ladder_raw": {d: q[d] for d in range(1, 10)},
        "q_ladder_lambda": {d: round(q[d] / TOP1, 4) for d in range(1, 10)},
        "min_lambda_q2_q9": round(min(q[d] / TOP1 for d in range(2, 10)), 4),
        "deflation_overlap_lambda": {d: round(deltas[d], 4) for d in range(1, 8)},
        "deep_overlap_typical_lambda": round(deep_typ, 4),
        "conservative": {
            "chain7_committed": round(chain_c, 4),
            "et_descent_committed": round(e_desc_c, 4),
            "et_both_committed": round(e_both_c, 4),
            "rel_gain_both": round(e_both_c / chain_c, 4),
            "proj_tps_costfree_upperbound": round(481.53 * e_both_c / chain_c, 1),
        },
        "deflation_corrected": {
            "chain7_committed": round(chain_d, 4),
            "et_descent_committed": round(e_desc_d, 4),
            "et_both_committed": round(e_both_d, 4),
            "rel_gain_both": round(e_both_d / chain_d, 4),
            "proj_tps_costfree_upperbound": round(481.53 * e_both_d / chain_d, 1),
        },
        "flat_p_closedform_anchor_m16": round(f_flat, 4),
        "fern185_costaware_ceiling_lambda1_tps": 520.95,
        "launch_trigger_tps": 512.41,
        "notes": (
            "Per-depth path-product on the validated tree_spec topology + measured "
            "ladder. E[T]_both/descent are committed tokens/step (incl bonus). "
            "proj_tps_costfree_upperbound IGNORES the verify+drafter per-step cost "
            "(assumes pure E[T] scaling) -> it is an UPPER bound only. fern #185's "
            "cost-aware launch_decision is authoritative: it caps the lambda=1 "
            "ceiling at 520.95 TPS (clears the 512.41 trigger by +8.5) and owns the "
            "GO/NO-GO vs the 0.9780 bar. Launch HELD on Issue #192 regardless of E[T]."
        ),
    }
    out = ROOT / "research/tree_verify_path/comp_etboth_perdepth_verdict.json"
    out.write_text(json.dumps(verdict, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
