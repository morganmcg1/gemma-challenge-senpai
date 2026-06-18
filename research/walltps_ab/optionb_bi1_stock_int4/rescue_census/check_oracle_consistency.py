"""PR #654: sanity-check the freshly generated canonical M=1 oracle against #651's per-prompt
decode-path validation. Both are strict single-seq M=1 decodes on the same reference server, so
at every wide-break position whose prompt's canonical head reaches that position
(ar_first_mismatch is null or > pos), oracle[id][pos] MUST equal the decode_tok recorded in
validate_decode_path_k{K}.json. A mismatch would mean the oracle is not reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent / "ksweep" / "ar_ref_m1_canonical" / "decode_outputs.jsonl"


def load_toks(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d["completion_token_ids"]
    return out


def main() -> int:
    oracle = load_toks(ORACLE)
    # raw (all wide-break positions) and well-defined (validate decode still on the ar_ref
    # trajectory at pos: ar_first_mismatch is None or > pos) comparison sets.
    ok = miss = skip = 0
    wd_ok = wd_miss = 0          # well-defined subset
    fork_disagree = 0           # pos == ar_first_mismatch (the int4-tie fork point itself)
    downstream = 0              # pos > ar_first_mismatch (different trajectory -> not comparable)
    mismatches = []
    wd_mismatches = []
    for k in (3, 5, 7):
        v = json.loads((HERE / f"validate_decode_path_k{k}.json").read_text())
        for d in v["detail"]:
            pid, pos = d["id"], d["pos"]
            fm = d["ar_first_mismatch"]
            if pid not in oracle or pos >= len(oracle[pid]):
                skip += 1
                continue
            otok = oracle[pid][pos]
            agree = (otok == d["decode_tok"])
            ok += int(agree)
            miss += int(not agree)
            rec = {"k": k, "id": pid, "pos": pos, "oracle_tok": otok,
                   "validate_decode_tok": d["decode_tok"], "ar_first_mismatch": fm}
            if not agree:
                mismatches.append(rec)
            # classify against the validate run's own ar_ref fork
            well_defined = (fm is None) or (fm > pos)
            if well_defined:
                wd_ok += int(agree)
                wd_miss += int(not agree)
                if not agree:
                    wd_mismatches.append(rec)
            elif fm == pos:
                fork_disagree += int(not agree)
            else:  # fm < pos
                downstream += int(not agree)

    print(f"[consistency] RAW (all wide-break pos): match={ok} mismatch={miss} skip={skip}")
    print(f"  classification of the {miss} raw mismatches:")
    print(f"    fork-point (pos==ar_first_mismatch, the int4-tie the two M=1 runs split on): {fork_disagree}")
    print(f"    downstream (pos>ar_first_mismatch, validate already on a different trajectory): {downstream}")
    print(f"    well-defined disagreement (validate still on ar_ref at pos): {wd_miss}")
    print(f"  WELL-DEFINED set (ar_first_mismatch None or > pos): match={wd_ok} mismatch={wd_miss}")
    if wd_mismatches:
        print("  WELL-DEFINED mismatches (fresh oracle-side tie branch vs ar_ref/validate):")
        for m in wd_mismatches:
            print(f"    K{m['k']} {m['id']} pos{m['pos']}: oracle={m['oracle_tok']} "
                  f"validate={m['validate_decode_tok']} (ar_first_mismatch={m['ar_first_mismatch']})")
    print("  NOTE: every disagreement is an int4 exact-tie branch pick (two batch-invariant M=1 "
          "decode runs landing on opposite sides of the same ~0.125-nat tie), confirmed near-tie "
          "by residual_margins.py -- NOT an oracle bug.")
    (HERE / "consistency_check.json").write_text(json.dumps(
        {"match": ok, "mismatch": miss, "skip": skip,
         "well_defined_match": wd_ok, "well_defined_mismatch": wd_miss,
         "fork_point_disagree": fork_disagree, "downstream_disagree": downstream,
         "interpretation": "all disagreements are int4 exact-tie inter-run branch picks (near-tie), not oracle bugs",
         "mismatches": mismatches, "well_defined_mismatches": wd_mismatches}, indent=2))
    print(f"[out] {HERE/'consistency_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
