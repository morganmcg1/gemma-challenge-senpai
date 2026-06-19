#!/usr/bin/env python
"""PR #729 -- aggregate fp8-KV vs bf16(auto)-KV arms into a verdict.

Inputs: per-arm JSONs from run_kv_arm.py (auto is the reference / baseline KV).
Computes:
  * TPS(L) delta fp8 vs auto, scored point L=512.
  * PPL gate (<=2.42) both arms.
  * matched-state self-consistency: per-position argmax agreement on the SAME
    teacher-forced prefixes (cascade-free); confident_genuine_flips at tau=0.3
    (flip where the auto top1-top2 gap > tau nats) vs near-tie flips.
  * free-run served greedy-identity gate (official byte-exact compare).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TAU = 0.3
MATERIAL_TPS_PCT = 2.0  # >2% counts as a material speed lever


def byte_compare(ref, cand):
    rmap = {r["id"]: r["completion_token_ids"] for r in ref}
    cmap = {r["id"]: r["completion_token_ids"] for r in cand}
    keys = sorted(set(rmap) & set(cmap))
    num_identical = num_divergent = total = total_div = 0
    first_div_idxs = []
    for k in keys:
        r, c = rmap[k], cmap[k]
        n = min(len(r), len(c))
        diff = sum(1 for i in range(n) if r[i] != c[i])
        total += n
        total_div += diff + abs(len(r) - len(c))
        if diff == 0 and len(r) == len(c):
            num_identical += 1
        else:
            num_divergent += 1
            idx = next((i for i in range(n) if r[i] != c[i]), n)
            first_div_idxs.append(idx)
    verdict = "GREEDY_IDENTICAL" if num_divergent == 0 else "DIVERGENT"
    return {
        "verdict": verdict, "num_prompts": len(keys),
        "num_identical": num_identical, "num_divergent": num_divergent,
        "total_tokens": total, "total_divergent_tokens": total_div,
        "token_identical_frac": (total - total_div) / total if total else None,
        "first_divergence_index_mean": (sum(first_div_idxs) / len(first_div_idxs)) if first_div_idxs else None,
        "first_divergence_index_min": min(first_div_idxs) if first_div_idxs else None,
    }


def matched_state(ref_arm, cand_arm):
    """Align by record id; ref provides argmax+gap, cand provides argmax."""
    rmap = {r["id"]: r for r in ref_arm["matched_state"]["per_record"]}
    cmap = {r["id"]: r for r in cand_arm["matched_state"]["per_record"]}
    keys = sorted(set(rmap) & set(cmap))
    total = flips = confident_flips = near_tie_flips = 0
    confident_positions = 0
    flip_gaps = []
    for k in keys:
        ra, ca = rmap[k]["argmax_ids"], cmap[k]["argmax_ids"]
        gaps = rmap[k]["gaps"]
        n = min(len(ra), len(ca), len(gaps))
        for i in range(n):
            total += 1
            if gaps[i] > TAU:
                confident_positions += 1
            if ra[i] != ca[i]:
                flips += 1
                flip_gaps.append(gaps[i])
                if gaps[i] > TAU:
                    confident_flips += 1
                else:
                    near_tie_flips += 1
    return {
        "n_positions": total,
        "flips": flips,
        "flip_rate": flips / total if total else None,
        "confident_genuine_flips": confident_flips,
        "confident_genuine_flip_rate": confident_flips / total if total else None,
        "confident_genuine_flip_rate_among_confident": (
            confident_flips / confident_positions if confident_positions else None),
        "near_tie_flips": near_tie_flips,
        "frac_flips_that_are_near_tie": (near_tie_flips / flips) if flips else None,
        "confident_positions": confident_positions,
        "max_flip_gap": max(flip_gaps) if flip_gaps else 0.0,
        "tau": TAU,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", required=True)
    ap.add_argument("--fp8", required=True)
    ap.add_argument("--fp8-e5m2", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deployed-max-model-len", type=int, default=4096)
    args = ap.parse_args()

    auto = json.loads(Path(args.auto).read_text())
    fp8 = json.loads(Path(args.fp8).read_text())
    arms = {"auto_bf16": auto, "fp8_e4m3": fp8}
    if args.fp8_e5m2:
        arms["fp8_e5m2"] = json.loads(Path(args.fp8_e5m2).read_text())

    # TPS table
    Ls = sorted({int(k) for k in auto["tps"]})
    tps_table = {}
    for L in Ls:
        row = {a: arms[a]["tps"][str(L)]["output_tps"] for a in arms}
        d = row["fp8_e4m3"] - row["auto_bf16"]
        row["delta_fp8_minus_auto"] = d
        row["delta_pct"] = 100.0 * d / row["auto_bf16"]
        row["reachable_in_deployed_cap"] = (L + 300) <= args.deployed_max_model_len
        tps_table[str(L)] = row

    scored = tps_table["512"]
    fp8_material_at_512 = scored["delta_pct"] > MATERIAL_TPS_PCT
    # crossover: smallest measured L where fp8 materially beats auto
    crossover = None
    for L in Ls:
        if tps_table[str(L)]["delta_pct"] > MATERIAL_TPS_PCT:
            crossover = L
            break

    ms = matched_state(auto, fp8)
    gate = byte_compare(auto["greedy_freerun"], fp8["greedy_freerun"])

    ppl_auto = auto["ppl"]["ppl"]
    ppl_fp8 = fp8["ppl"]["ppl"]
    ppl_safe = (ppl_fp8 <= 2.42) and (ppl_auto <= 2.42)

    verdict = {
        "pr": 729, "lever": "fp8_kv_cache_decode",
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True, "no_hf_job": True,
        "anchor": {"submission": "int4_g128_lmhead", "official_tps": 126.378, "ppl": 2.019,
                   "wandb": "905tbujn", "scored_output_len": 512},
        "int8_kv_supported_in_vllm_0_22": False,
        "tps_table": tps_table,
        "scored_point_512": {
            "auto_bf16_tps": scored["auto_bf16"], "fp8_e4m3_tps": scored["fp8_e4m3"],
            "delta_tps": scored["delta_fp8_minus_auto"], "delta_pct": scored["delta_pct"],
            "fp8_material_at_512": fp8_material_at_512,
        },
        "crossover_length_material_fp8_win": crossover,
        "ppl": {"auto_bf16": ppl_auto, "fp8_e4m3": ppl_fp8, "cap": 2.42, "ppl_safe": ppl_safe},
        "matched_state_self_consistency": ms,
        "freerun_served_gate": gate,
        "peak_gib": {a: arms[a].get("peak_gib") for a in arms},
        "load_s": {a: arms[a].get("load_s") for a in arms},
    }

    # primary metric = scored-point delta_pct (what the leaderboard would see)
    verdict["primary_metric_name"] = "fp8kv_tps_delta_pct_at_512"
    verdict["primary_metric_value"] = scored["delta_pct"]
    verdict["kv_lever_is_green_for_official"] = bool(fp8_material_at_512 and ppl_safe)

    # self-tests
    st = {
        "ppl_finite": all(math.isfinite(arms[a]["ppl"]["ppl"]) for a in arms),
        "tps_positive": all(tps_table[str(L)]["auto_bf16"] > 0 and tps_table[str(L)]["fp8_e4m3"] > 0 for L in Ls),
        "matched_positions_match": ms["n_positions"] > 0,
        "gate_well_defined": gate["verdict"] in ("GREEDY_IDENTICAL", "DIVERGENT"),
        "flip_rate_in_unit": 0.0 <= (ms["flip_rate"] or 0) <= 1.0,
        "confident_le_total_flips": ms["confident_genuine_flips"] <= ms["flips"],
        "ppl_records_128": auto["ppl"]["num_records"] == 128 and fp8["ppl"]["num_records"] == 128,
    }
    st["self_test_passes"] = all(st.values())
    verdict["self_test"] = st

    Path(args.out).write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: verdict[k] for k in (
        "scored_point_512", "crossover_length_material_fp8_win", "ppl",
        "matched_state_self_consistency", "freerun_served_gate",
        "kv_lever_is_green_for_official", "self_test")}, indent=2))


if __name__ == "__main__":
    main()
