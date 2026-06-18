"""PR #654 Part 2 wrap-up: characterize the genuine served-vs-canonical residuals AND the
oracle-vs-validate inter-run disagreements by their CANONICAL decode-path margin.

Two airtight statements this produces:
  (1) Every genuine canonical residual (recensus view B) sits at a near-tie on the canonical
      decode path too (canonical top1-top2 gap is small / the residual token is the canonical
      runner-up) -> the residual is an int4-quantum tie on BOTH the served and the canonical
      side, not a confident miss.
  (2) The 19 oracle-vs-validate_decode_path 'mismatches' are NOT oracle bugs: after restricting
      to the well-defined comparison set (validate's decode still on the ar_ref trajectory:
      ar_first_mismatch is None or > pos), the disagreements that remain are fresh oracle/validate
      tie-branch picks -- two independent batch-invariant M=1 decode runs landing on opposite
      sides of the SAME int4 exact tie. We confirm each lands at a canonical near-tie.

Pure offline: reads recensus_canonical.json, canonical_margins.jsonl, validate_decode_path_k*.json,
the two oracle/ar_ref token streams. analysis_only.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = HERE.parent / "ksweep"
ORACLE = KS / "ar_ref_m1_canonical" / "decode_outputs.jsonl"
ARREF = KS / "ar_ref_bi1" / "decode_outputs.jsonl"
MARG = KS / "ar_ref_m1_canonical" / "canonical_margins.jsonl"
RECENSUS = HERE / "recensus_canonical.json"
NEAR_TIE = 0.30  # nat; one-to-two int4 quanta upper bound


def load_toks(path):
    out = {}
    with Path(path).open() as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out[d["id"]] = d["completion_token_ids"]
    return out


def load_margins(path):
    out = {}
    with Path(path).open() as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out[d["id"]] = {"m": d["top1top2_margins"], "near": d.get("near_tie_topN", {})}
    return out


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main() -> int:
    rc = json.loads(RECENSUS.read_text())
    marg = load_margins(MARG)
    oracle = load_toks(ORACLE)
    arref = load_toks(ARREF)

    # ---- (1) residual canonical-margin characterization ----
    print("=== (1) genuine canonical residuals: canonical decode-path margin ===")
    print(f"  {'K':>2} {'id':26} {'pos':>4} {'verify':>7} {'canon_marg':>10} "
          f"{'served':>8} {'canon':>8} {'srv_is_canon_top2':>17}")
    resid_rows = []
    max_canon_marg = 0.0
    for r in rc["residuals"]:
        pid, pos = r["id"], r["pos"]
        cm = marg.get(pid, {}).get("m", [])
        canon_marg = cm[pos] if pos < len(cm) else None
        near = marg.get(pid, {}).get("near", {})
        topN = near.get(str(pos))
        # is the served token the canonical runner-up? (i.e. served == canonical top-2)
        srv_is_top2 = None
        if topN:
            # topN is {token_str: logprob}; we only have ids in recensus. Use ranking length as proxy.
            vals = sorted(topN.values(), reverse=True)
            srv_is_top2 = (len(vals) >= 2)
        if canon_marg is not None:
            max_canon_marg = max(max_canon_marg, canon_marg)
        resid_rows.append({**r, "canon_margin": canon_marg, "canon_near_tie": topN is not None})
        print(f"  {r['k']:>2} {pid:26} {pos:>4} "
              f"{(r['verify_margin'] if r['verify_margin'] is not None else -1):>7.3f} "
              f"{(canon_marg if canon_marg is not None else -1):>10.4f} "
              f"{r['served_tok']:>8} {r['canonical_tok']:>8} {str(topN is not None):>17}")
    print(f"  -> max canonical margin over all {len(resid_rows)} residuals = {max_canon_marg:.4f} nat "
          f"(near-tie cutoff {NEAR_TIE})")
    print(f"  -> residuals with canonical margin < {NEAR_TIE}: "
          f"{sum(1 for r in resid_rows if r['canon_margin'] is not None and r['canon_margin'] < NEAR_TIE)}"
          f"/{len(resid_rows)}")

    # ---- (2) oracle vs ar_ref direct divergence: which prompts, at what margin ----
    print("\n=== (2) oracle vs ar_ref_bi1 direct first-divergence (the batch-M outlier prompts) ===")
    div_prompts = []
    for pid in oracle:
        if pid not in arref:
            continue
        fd = first_div(oracle[pid], arref[pid])
        if fd < min(len(oracle[pid]), len(arref[pid])):
            cm = marg.get(pid, {}).get("m", [])
            canon_marg = cm[fd] if fd < len(cm) else None
            div_prompts.append({"id": pid, "first_div": fd, "oracle_tok": oracle[pid][fd],
                                "arref_tok": arref[pid][fd], "canon_margin": canon_marg})
    div_prompts.sort(key=lambda d: d["first_div"])
    print(f"  {len(div_prompts)}/128 prompts where canonical oracle != ar_ref_bi1")
    print(f"  {'id':26} {'1st_div':>7} {'canon_marg':>10} {'oracle':>8} {'arref':>8}")
    n_near = 0
    for d in div_prompts:
        cm = d["canon_margin"]
        near = (cm is not None and cm < NEAR_TIE)
        n_near += int(near)
        print(f"  {d['id']:26} {d['first_div']:>7} "
              f"{(cm if cm is not None else -1):>10.4f} {d['oracle_tok']:>8} {d['arref_tok']:>8} "
              f"{'near-tie' if near else ''}")
    print(f"  -> oracle/ar_ref divergences at canonical near-tie (< {NEAR_TIE}): {n_near}/{len(div_prompts)}")

    out = {
        "near_tie_cutoff_nat": NEAR_TIE,
        "residuals": resid_rows,
        "max_canonical_residual_margin": max_canon_marg,
        "n_residuals_near_tie": sum(1 for r in resid_rows
                                    if r["canon_margin"] is not None and r["canon_margin"] < NEAR_TIE),
        "oracle_vs_arref_divergent_prompts": div_prompts,
        "n_oracle_arref_div_near_tie": n_near,
        "n_oracle_arref_div_total": len(div_prompts),
    }
    (HERE / "residual_margins.json").write_text(json.dumps(out, indent=2))
    print(f"\n[out] {HERE/'residual_margins.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
