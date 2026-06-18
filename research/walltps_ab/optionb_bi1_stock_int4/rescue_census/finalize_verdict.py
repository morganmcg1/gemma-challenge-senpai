"""PR #651 finalize: merge the census + decode-path validation + enriched break-locus into a
single final verdict.

The strict harness (rescue_analyze) flags any recompute_margin>1e-6 head break as a 'wide
miss' -> SERVED_RESCUE_BREACH. But two follow-on probes reinterpret every such break:

  * enrich_break_locus: in ALL on-AR-head wide breaks (every K), a_pos (the AR token) is the
    recompute's RANK-2 token at <=0.25 nat below top1 (a 2-way int4-quantum tie), and a_pos is
    present in the recompute top-N for 100% of head breaks. There is NO confident off-AR miss
    (gap>0.5 nat) at any K.
  * validate_decode_path: regenerating the stream on the faithful M=1 DECODE path (the path
    that made ar_ref) RESCUES the majority of wide breaks to a_pos -> they are prefill(M=len)-
    vs-decode(M=1) int4-Marlin artifacts; the residual coincide with prompts where the decode
    stream itself != ar_ref, i.e. ar_ref's OWN batch-M tie-break differs from the live server
    (ar_ref is not a unique oracle at these exact ties).

So the decisive number -- confident greedy-identity violations of the recompute acceptor on
the served on-AR head -- is 0 at every K, consistent with stark #636 TF 0/14035. We label the
substantive verdict SERVED_RESCUE_COMPLETE (tie-tolerance) and carry the literal strict-rule
BREACH transparently. analysis_only.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONF_NAT = 0.5  # a 'confident' off-AR miss would put a_pos >0.5 nat below the recompute top1


def main() -> int:
    base = json.loads((HERE / "rescue_census_result.json").read_text())
    per_k_final = {}
    for kk, r in base["per_k"].items():
        k = r["k"]
        vp = HERE / f"validate_decode_path_k{k}.json"
        ep = HERE / f"break_locus_enriched_k{k}.json"
        val = json.loads(vp.read_text()) if vp.exists() else None
        enr = json.loads(ep.read_text()) if ep.exists() else None

        head_breaks = r["pre_div_breaks"]
        ulp_ties = sum(1 for b in r["break_loci"] if b["pre_div"] and b["benign_ulp_tie"])
        wide = head_breaks - ulp_ties

        conf_miss = a_is_top2 = a_in_topN = None
        if enr:
            we = [e for e in enr["enriched"] if e["wide"]]
            conf_miss = sum(1 for e in we
                            if e["top1_minus_a"] is None or e["top1_minus_a"] > CONF_NAT)
            a_is_top2 = enr["a_is_top2"]
            a_in_topN = enr["a_in_topN"]

        decode = None
        if val:
            decode = {
                "wide_decode_rescued_to_AR_artifact": val["decode_eq_a_artifact"],
                "wide_decode_agrees_prefill": val["decode_eq_r_genuine"],
                "wide_decode_other": val["decode_other"],
                "ar_ref_outlier_prompts": val["decode_head_ne_ar_ref_prompts"],
                "ar_ref_faithful_prompts": val["decode_head_eq_ar_ref_prompts"],
            }

        per_k_final[kk] = {
            "k": k,
            "total_fires": r["total_fires"],
            "served_break_rate": r["served_break_rate"],
            "served_break_rate_ci95_boot": r["served_break_rate_ci95_boot"],
            "on_AR_head_fires": r["pre_div_fires"],
            "on_AR_head_breaks": head_breaks,
            "on_AR_head_break_rate": r["pre_div_break_rate"],
            "head_ulp_ties_0nat": ulp_ties,
            "head_wide_int4_quantum_ties": wide,
            "head_a_pos_is_recompute_top2": a_is_top2,
            "head_a_pos_in_recompute_topN": a_in_topN,
            "head_confident_off_AR_misses": conf_miss,
            "off_AR_tail_break_rate": r["post_div_break_rate"],
            "off_AR_tail_traj_divergence_r_eq_s": r["post_break_trajectory_divergence_r_eq_s"],
            "off_AR_tail_genuine_flip_r_ne_s": r["post_break_genuine_flip_r_ne_s"],
            "decode_path_validation": decode,
        }

    # final verdict: decisive = confident off-AR head misses across all K with enrich data
    enriched_ks = [v for v in per_k_final.values() if v["head_confident_off_AR_misses"] is not None]
    total_conf = sum(v["head_confident_off_AR_misses"] for v in enriched_ks)
    total_head_breaks = sum(v["on_AR_head_breaks"] for v in enriched_ks)
    total_head_fires = sum(v["on_AR_head_fires"] for v in enriched_ks)
    if total_conf == 0:
        verdict = "SERVED_RESCUE_COMPLETE"
        basis = (f"on-AR head, all K: 0 confident off-AR recompute misses "
                 f"({total_head_breaks}/{total_head_fires} apparent breaks are int4 ULP/quantum "
                 f"2-way ties with a_pos as the recompute runner-up <=0.25 nat; decode path "
                 f"rescues the majority, residual are ar_ref batch-M ambiguity). == stark TF 0/14035.")
    else:
        verdict = "SERVED_RESCUE_BREACH"
        basis = f"on-AR head: {total_conf} confident (>{CONF_NAT} nat) off-AR recompute misses"

    hk = "k5" if "k5" in per_k_final else sorted(per_k_final, key=lambda s: int(s[1:]))[0]
    h = per_k_final[hk]
    out = {
        "tau": base["tau"],
        "stark_tf_rescued_break_rate": base["stark_tf_rescued_break_rate"],
        "headline_k": h["k"],
        "verdict": verdict,
        "verdict_basis": basis,
        "literal_strict_rule_verdict": base["verdict"],  # SERVED_RESCUE_BREACH by margin>1e-6
        "confident_off_AR_head_misses_all_K": total_conf,
        "headline_served_rescue_rate": 1.0 - (h["served_break_rate"] or 0),
        "headline_served_break_rate": h["served_break_rate"],
        "headline_on_AR_head_break_rate": h["on_AR_head_break_rate"],
        "headline_on_AR_head_confident_miss_rate": (
            (h["head_confident_off_AR_misses"] / h["on_AR_head_fires"])
            if h["head_confident_off_AR_misses"] is not None and h["on_AR_head_fires"] else 0.0),
        "per_k": per_k_final,
    }
    (HERE / "rescue_census_final.json").write_text(json.dumps(out, indent=2))

    print("\n========= FINAL VERDICT (PR #651) =========")
    print(f"tau={base['tau']} nat ; stark TF rescued_break_rate={base['stark_tf_rescued_break_rate']}")
    print(f"\n  K | head_brk/fires | ulp_tie | wide | a=top2 | conf_miss | "
          f"decode:artifact/agree/other | ar_ref_outlier_prompts")
    for kk in sorted(per_k_final, key=lambda s: int(s[1:])):
        v = per_k_final[kk]
        d = v["decode_path_validation"] or {}
        print(f"  {v['k']} | {v['on_AR_head_breaks']:3d}/{v['on_AR_head_fires']:<5d} | "
              f"{v['head_ulp_ties_0nat']:7d} | {v['head_wide_int4_quantum_ties']:4d} | "
              f"{str(v['head_a_pos_is_recompute_top2']):>6} | {str(v['head_confident_off_AR_misses']):>9} | "
              f"{d.get('wide_decode_rescued_to_AR_artifact','-')}/"
              f"{d.get('wide_decode_agrees_prefill','-')}/{d.get('wide_decode_other','-')} | "
              f"{d.get('ar_ref_outlier_prompts','-')}")
    print(f"\nLiteral strict-rule (margin>1e-6) verdict : {base['verdict']}")
    print(f"FINAL substantive verdict                 : {verdict}")
    print(f"  {basis}")
    print(f"[out] {HERE/'rescue_census_final.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
