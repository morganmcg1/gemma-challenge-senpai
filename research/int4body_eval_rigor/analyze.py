"""Verdict synthesis for PR #693 (int4-body eval-rigor).

Reads the six decode-basis re-measures of the SHIPPED int4_g128_lmhead
(AIME-2024 maj@8 + gpqa_diamond, each at greedy / #31-sampled / #31+min_tokens=8),
computes Wilson 95% CIs, compares to the PR-stated #515 gates
(AIME 0.420, gpqa_diamond 0.471) under the ubel-#672 upper95 discipline, and
renders the GAP_* verdict. Writes verdict.json + a markdown table. Analysis-only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

WD = Path("/workspace/senpai/target/research/int4body_eval_rigor")

# --- PR-stated #515 gates (advisor authority, PR #693 Instruction 3 + Baseline) ---
GATE_AIME = 0.420  # >=90% of vanilla bf16 base, AIME-2024 maj@k
GATE_GPQA = 0.471  # >=90% of vanilla bf16 base, gpqa_diamond

# --- references (authorized cross-reads: ubel #672/#650, wirbel #682, bars #614) ---
# AIME gate basis is 12288 max_tokens / 60 problems (2024+2025) / min_tokens=8 --
# NOT my 3072/2024-only local basis. ubel ALREADY measured the int4 body under #31
# SAMPLING at that basis, so the AIME #31-compliant number is on record and MISSES.
# My local AIME (3072/30) is same-server CORROBORATION, not the gate comparison.
REF = {
    # --- AIME (gate basis: ubel #672/#650, 12288 max_tokens, 60 problems) ---
    "aime_base_bf16_greedy": 0.4833,        # ubel #672: bf16 greedy AIME 29/60, STABLE control
    "aime_base_for_gate": 0.4667,           # ubel #672: base AIME -> gate 0.420 = 0.9 x 0.4667
    "aime_int4_greedy_band_upper95": 0.38832,  # ubel #672 VERDICT run: int4 greedy band upper95, BLOCKER_ROBUST (< 0.420)
    "aime_int4_greedy_band": [0.350, 0.3833],  # ubel #672: 4 sessions x 60, mean 0.3667
    "aime_int4_greedy_band_mean": 0.3667,   # ubel #672: int4 greedy band mean (the "current" AIME number)
    "aime_int4_sampled12288": 0.3467,       # ubel #650 (per #672 body): int4 #31-SAMPLED maj@8 @12288/60
    "aime_int4_sampled12288_ci": [0.2951, 0.4022],  # ubel #650 CI; upper95 0.4022 < 0.420 -> compliant MISS
    "aime_base_local3072_sampled": 0.40,    # base_aime.json, 3072/30 (my local-basis denominator, maj@8)
    # --- GPQA-Diamond (gate basis: bars #614, 4096 max_tokens / 6144 mml / seed 12345 / n=198) ---
    "gpqa_base_greedy_4096": 0.5051,        # bars #614: bf16 base greedy @4096/n198
    "gpqa_base_sampled_mean_4096": 0.5313,  # bars #614: bf16 base #31-sampled 5-seed mean @4096/n198
    "gpqa_base515_denominator": 0.5236,     # wirbel #682 / #581: gate denom -> 0.471 = 0.9 x 0.5236
    "gpqa_int4_greedy_wirbel682": 0.4697,   # wirbel #682 ezvgx3et AR-arm int4 greedy = 46/99 @6144 (near-miss of 0.471)
    "gpqa_finish_len_rate_2048": 0.373,     # bars #614: 37% truncate at 2048 -> MUST score >=4096
    "gpqa_finish_len_rate_4096": 0.035,     # bars #614: 3.5% truncate at 4096 (truncation-clean)
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval. Returns (point, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, center - half), min(1.0, center + half))


def classify(point: float, lo: float, hi: float, gate: float) -> str:
    """Pass discipline: clear pass needs lo>=gate; clear miss needs hi<gate."""
    if lo >= gate:
        return "CLEAR_PASS"
    if hi < gate:
        return "CLEAR_MISS"
    if point >= gate:
        return "AMBIGUOUS_POINT_OVER"  # point clears but CI spans below -> NOT a clear pass
    return "AMBIGUOUS_POINT_UNDER"      # point misses, CI spans above


def load(name: str) -> dict | None:
    p = WD / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def row_aime(tag: str, d: dict | None) -> dict | None:
    if d is None:
        return None
    k = d["n_correct_maj"]
    n = d["n_problems"]
    point, lo, hi = wilson(k, n)
    return {
        "bench": "aime_2024",
        "basis": tag,
        "metric": "maj@%d" % d.get("maj_k", d.get("k", 0)),
        "k_correct": k,
        "n": n,
        "point": round(point, 4),
        "wilson_lo": round(lo, 4),
        "wilson_hi": round(hi, 4),
        "gate": GATE_AIME,
        "verdict": classify(point, lo, hi, GATE_AIME),
        "mean_pass_rate": round(d.get("mean_pass_rate", float("nan")), 4),
        "extract_fail_rate": round(d.get("extract_fail_rate", float("nan")), 4),
        "sampling": d.get("sampling"),
    }


def row_gpqa(tag: str, d: dict | None) -> dict | None:
    if d is None:
        return None
    k = d["n_correct"]
    n = d["n_scored"]
    point, lo, hi = wilson(k, n)
    return {
        "bench": "gpqa_diamond",
        "basis": tag,
        "metric": "accuracy",
        "k_correct": k,
        "n": n,
        "point": round(point, 4),
        "wilson_lo": round(lo, 4),
        "wilson_hi": round(hi, 4),
        "gate": GATE_GPQA,
        "verdict": classify(point, lo, hi, GATE_GPQA),
        "finish_length_rate": round(d.get("finish_length_rate", float("nan")), 4),
        "n_length_truncated": d.get("n_length_truncated"),
        "n_error": d.get("n_error"),
        "decode": d.get("decode"),
        "min_tokens": d.get("min_tokens"),
    }


def main() -> None:
    rows: list[dict] = []
    for tag, fn in [("greedy", "aime_greedy.json"),
                    ("sampled_31", "aime_sampled.json"),
                    ("sampled_31_mintok8", "aime_sampled_mintok8.json")]:
        r = row_aime(tag, load(fn))
        if r:
            rows.append(r)
    for tag, fn in [("greedy", "gpqa_greedy.json"),
                    ("sampled_31", "gpqa_sampled.json"),
                    ("sampled_31_mintok8", "gpqa_sampled_mintok8.json")]:
        r = row_gpqa(tag, load(fn))
        if r:
            rows.append(r)

    def pick(bench: str, basis: str) -> dict | None:
        for r in rows:
            if r["bench"] == bench and r["basis"] == basis:
                return r
        return None

    # --- GPQA leg: my OWN on-basis re-measure (4096/n198/seed12345 == the gate basis) ---
    # The card's artifact test is "does #31-compliant decode RECOVER the gap vs greedy?"
    # => a recovery REQUIRES the compliant number to IMPROVE over greedy. Pure #31 is the
    # no-guard generation_config sampling (sampled_31); the min_tokens=8 arm is the
    # supplementary triad-#541 guard (inert here, empty_rate=0). Use the MORE favorable of
    # the two compliant arms for the gate test (conservative for CONFIRMED_REAL).
    gpqa_g = pick("gpqa_diamond", "greedy")
    gpqa_c_pure = pick("gpqa_diamond", "sampled_31")           # canonical #31 (no guard)
    gpqa_c_guard = pick("gpqa_diamond", "sampled_31_mintok8")  # +EOS-guard (arm c)
    _gpqa_cands = [c for c in (gpqa_c_pure, gpqa_c_guard) if c]
    # canonical compliant = pure #31 if present, else the guard arm
    gpqa_c = gpqa_c_pure or gpqa_c_guard
    # most-favorable compliant point drives the recovery test
    gpqa_c_best = max(_gpqa_cands, key=lambda r: r["point"]) if _gpqa_cands else None

    def gpqa_leg() -> dict:
        if gpqa_c_best is None:
            return {"leg_verdict": "PENDING", "note": "gpqa compliant arm not yet measured"}
        gp = gpqa_g["point"] if gpqa_g else None
        move = round(gpqa_c_best["point"] - gp, 4) if gp is not None else None
        improves_vs_greedy = bool(gp is not None and gpqa_c_best["point"] > gp)
        # CI-separated material lift over greedy's own uncertainty (not n-noise).
        ci_separated_up = bool(gp is not None and gpqa_c_best["wilson_lo"] > gp)
        # RECOVERY requires the compliant number to improve OVER greedy. A point that sits
        # above the gate only because GREEDY was already above it (here greedy 0.49 >
        # compliant 0.47) is NOT a greedy-vs-#31 artifact recovery.
        if not improves_vs_greedy:
            lv = "NOT_RECOVERED"            # compliant <= greedy: no greedy-vs-#31 artifact
        elif gpqa_c_best["wilson_lo"] >= GATE_GPQA:
            lv = "RECOVERED_CLEAR"          # improved AND Wilson-lo >= gate: clean pass
        elif gpqa_c_best["point"] >= GATE_GPQA:
            lv = "RECOVERED_POINT_CIWIDE"   # improved, point clears, CI spans below
        elif ci_separated_up:
            lv = "PARTIAL_MOVE"             # improved beyond greedy noise, stays < gate
        else:
            lv = "NOT_RECOVERED"
        # Separately: where does the compliant point sit vs the gate (independent of recovery)?
        if gpqa_c_best["wilson_lo"] >= GATE_GPQA:
            gate_pos = "CLEAR_PASS"
        elif gpqa_c_best["wilson_hi"] < GATE_GPQA:
            gate_pos = "CLEAR_MISS"
        elif gpqa_c_best["point"] >= GATE_GPQA:
            gate_pos = "TIE_POINT_OVER"     # point >= gate but CI spans below -> marginal tie
        else:
            gate_pos = "TIE_POINT_UNDER"    # point < gate but CI spans above -> marginal tie
        return {
            "leg_verdict": lv, "gate_position": gate_pos,
            "point": gpqa_c["point"],                 # canonical #31 (no-guard) headline
            "wilson": [gpqa_c["wilson_lo"], gpqa_c["wilson_hi"]],
            "basis_arm": gpqa_c["basis"],
            "best_compliant_point": gpqa_c_best["point"], "best_compliant_arm": gpqa_c_best["basis"],
            "pure31_point": gpqa_c_pure["point"] if gpqa_c_pure else None,
            "guard_point": gpqa_c_guard["point"] if gpqa_c_guard else None,
            "greedy_point": gp,
            "greedy_to_compliant_move": move, "improves_vs_greedy": improves_vs_greedy,
            "ci_separated_up": ci_separated_up,
            "gate": GATE_GPQA,
            "vs_base_sampled": round(gpqa_c["point"] - REF["gpqa_base_sampled_mean_4096"], 4),
        }

    # --- AIME leg: gate-basis is ubel #672/#650 (12288/n60). The #31-compliant
    # (sampled) number is ALREADY on record there and MISSES -- so AIME is not a
    # greedy-masquerade case. My local 3072/30 arms are same-server CORROBORATION. ---
    aime_compliant_pt = REF["aime_int4_sampled12288"]            # 0.3467
    aime_compliant_ci = REF["aime_int4_sampled12288_ci"]         # [0.2951, 0.4022]
    aime_greedy_upper95 = REF["aime_int4_greedy_band_upper95"]   # 0.38832
    aime_leg_verdict = "CONFIRMED_REAL" if aime_compliant_ci[1] < GATE_AIME else "RECOVERED"
    aime_local_g = pick("aime_2024", "greedy")
    aime_local_c = pick("aime_2024", "sampled_31_mintok8") or pick("aime_2024", "sampled_31")
    aime_local_move = (round(aime_local_c["point"] - aime_local_g["point"], 4)
                       if (aime_local_c and aime_local_g) else None)
    aime_leg = {
        "leg_verdict": aime_leg_verdict,
        "compliant_point_gatebasis": aime_compliant_pt,
        "compliant_ci_gatebasis": aime_compliant_ci,
        "greedy_band_upper95_gatebasis": aime_greedy_upper95,
        "gate": GATE_AIME,
        "compliant_clears": aime_compliant_ci[0] >= GATE_AIME,
        "compliant_upper95_below_gate": aime_compliant_ci[1] < GATE_AIME,
        "local_corroboration": {
            "greedy_point_3072_30": aime_local_g["point"] if aime_local_g else None,
            "compliant_point_3072_30": aime_local_c["point"] if aime_local_c else None,
            "compliant_basis_arm": aime_local_c["basis"] if aime_local_c else None,
            "greedy_to_compliant_move": aime_local_move,
            "note": "3072/2024-only basis (heavy truncation); corroborates no-recovery, NOT the gate comparison",
        },
    }

    gl = gpqa_leg()

    # --- Overall synthesis ---
    # Binding leg = AIME (worst, robustly confirmed on the #31-compliant basis).
    # The #515 panel needs ALL gates; AIME confirmed-real => the int4-body fails the
    # panel on the correct basis regardless of GPQA. GPQA is characterised but is not
    # the binding wall. Verdict reflects whether ANY of the gap is a basis artifact.
    gpqa_recovers = gl["leg_verdict"] in ("RECOVERED_CLEAR", "RECOVERED_POINT_CIWIDE")
    gpqa_moves = gl["leg_verdict"] == "PARTIAL_MOVE"
    if aime_leg_verdict == "CONFIRMED_REAL" and not gpqa_recovers and not gpqa_moves:
        verdict = "GAP_CONFIRMED_REAL"
    elif gpqa_recovers or gpqa_moves:
        # AIME wall stands but some GPQA gap is a greedy-vs-#31 artifact -> partial.
        verdict = "GAP_PARTIALLY_RECOVERABLE"
    else:
        verdict = "GAP_CONFIRMED_REAL"

    out = {
        "pr": 693,
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": 1,
        "fires": False,
        "model": "int4_g128_lmhead (shipped: int4 g128 body + int4 lm_head)",
        "serve_stack": "vllm 0.22.0, eval MAX_MODEL_LEN=6144 (gpqa@4096 truncation-clean), MAX_NUM_BATCHED_TOKENS=512",
        "gates": {"aime_2024": GATE_AIME, "gpqa_diamond": GATE_GPQA},
        "gate_basis": {
            "aime": "ubel #672/#650: 12288 max_tokens / 60 problems (2024+2025) / min_tokens=8",
            "gpqa": "bars #614: 4096 max_tokens / 6144 mml / seed 12345 / n=198",
        },
        "references": REF,
        "rows": rows,
        "aime_leg": aime_leg,
        "gpqa_leg": gl,
        # headline scalars (also logged to W&B)
        "aime_compliant": aime_compliant_pt,
        "aime_compliant_wilson": aime_compliant_ci,
        "aime_compliant_verdict": aime_leg_verdict,
        "aime_compliant_basis": (
            "gate-basis #31-sampled maj@8 (12288 max_tokens / 60 problems), "
            "banked advisor-branch commit 1b00c31 (ubel #650/#672); my on-server "
            "local re-measure (3072/30 maj@8) corroborates at 0.30 vs base 0.40 (~75%)"
        ),
        # AIME greedy->compliant move at the gate basis: #31-sampled minus greedy-band
        # mean. Negative => compliant decode does NOT recover (same direction as GPQA).
        "aime_greedy_to_compliant_delta": round(
            REF["aime_int4_sampled12288"] - REF["aime_int4_greedy_band_mean"], 4),
        "aime_local_greedy_to_compliant_move": aime_local_move,
        "gpqa_compliant": gl.get("point"),
        "gpqa_compliant_wilson": gl.get("wilson"),
        "gpqa_compliant_basis": gl.get("basis_arm"),
        "gpqa_compliant_verdict": gl.get("leg_verdict"),
        "gpqa_greedy_to_compliant_delta": gl.get("greedy_to_compliant_move"),
        "verdict": verdict,
    }
    (WD / "verdict.json").write_text(json.dumps(out, indent=2))

    # markdown basis table
    md = ["| bench | basis | metric | point | Wilson95 | gate | call |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s | %s | **%.4f** (%d/%d) | [%.4f, %.4f] | %.3f | %s |" % (
            r["bench"], r["basis"], r["metric"], r["point"], r["k_correct"], r["n"],
            r["wilson_lo"], r["wilson_hi"], r["gate"], r["verdict"]))
    (WD / "basis_table.md").write_text("\n".join(md) + "\n")

    print(json.dumps(out, indent=2))
    print("\n".join(md))
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
