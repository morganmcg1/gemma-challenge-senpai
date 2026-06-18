"""PR #654 Part 2: re-run the #651 on-AR-head census against the batch-M-stable canonical
M=1 oracle (ar_ref_m1_canonical), and measure served-vs-canonical identity directly.

Two views per K:

(A) REFERENCE-SWAP CENSUS  -- literal "re-run the #651 census against the canonical oracle":
    keep the stored prefill recompute r_pos (the acceptor's mechanism), swap the reference
    a_pos := canonical_oracle[pos], recompute the head boundary first_div(served, oracle).
    head break := r_pos != oracle[pos] on the head ; confident miss := head break with
    recompute_margin > 0.5 nat. -> canonical_oracle_on_AR_head_break_rate_k5 + confident misses.

(B) SERVED-vs-CANONICAL IDENTITY  -- the decisive measurement. For each prompt compare the
    served stream to BOTH references:
      old_head = first_div(served, ar_ref_bi1)        (the #651 head)
      new_head = first_div(served, canonical_oracle)   (the canonical head)
    A genuine canonical residual is a position where served == ar_ref_bi1 but
    served != canonical_oracle (i.e. new_head < old_head -> the served stream leaves the
    canonical M=1 greedy path earlier than it leaves ar_ref_bi1). We record each such
    divergence's served/canonical token ids, the served verify_margin (M=K+1 top1-top2 gap),
    whether the served token wins the lowest-index tie-break vs the canonical token, and the
    byte-exact-subset membership. residual positions are emitted for a targeted logprobs pass.

analysis_only. Pure offline read (oracle + ar_ref + margins + #651 capture).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = HERE.parent / "ksweep"
MARGINS_DIR = HERE.parent / "fire_census"
ARREF = KS / "ar_ref_bi1" / "decode_outputs.jsonl"
ORACLE = KS / "ar_ref_m1_canonical" / "decode_outputs.jsonl"
TAU = 0.5
CONF = 0.5  # confident-miss threshold (nat)


def load_toks(path):
    out = {}
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["id"]] = d["completion_token_ids"]
    return out


def load_margins(path):
    out = {}
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["id"]] = {"index": d["index"], "sha_ok": d["sha_ok"],
                                "served": d["completion_token_ids"], "margins": d["margins"]}
    return out


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main() -> int:
    oracle = load_toks(ORACLE)
    arref = load_toks(ARREF)
    print(f"[load] canonical oracle prompts={len(oracle)}  ar_ref_bi1 prompts={len(arref)}")

    summary = {}
    all_residuals = []
    for k in (3, 5, 7):
        marg = load_margins(MARGINS_DIR / f"per_prompt_margins_k{k}.jsonl")
        cap = [json.loads(l) for l in (HERE / f"rescue_k{k}.jsonl").read_text().splitlines() if l.strip()]

        # ---- (B) served-vs-canonical identity ----
        res_k = []
        n_oracle_eq_arref_prompts = 0
        head_shorter = head_longer = head_same = 0
        for pid, m in marg.items():
            served = m["served"]
            a = arref.get(pid)
            o = oracle.get(pid)
            if a is None or o is None:
                continue
            if a == o:
                n_oracle_eq_arref_prompts += 1
            old_head = first_div(served, a)
            new_head = first_div(served, o)
            if new_head < old_head:
                head_shorter += 1
                pos = new_head  # the canonical residual position (served leaves canonical here)
                s_tok = served[pos]
                o_tok = o[pos]
                vm = m["margins"][pos] if pos < len(m["margins"]) else None
                res = {
                    "k": k, "id": pid, "index": m["index"], "pos": pos,
                    "served_tok": s_tok, "canonical_tok": o_tok, "arref_tok": a[pos],
                    "served_eq_arref": bool(s_tok == a[pos]),
                    "verify_margin": vm, "sha_ok": bool(m["sha_ok"]),
                    "served_wins_lowest_index": bool(s_tok < o_tok),
                    "old_head": old_head, "new_head": new_head,
                }
                res_k.append(res)
                all_residuals.append(res)
            elif new_head > old_head:
                head_longer += 1
            else:
                head_same += 1

        # ---- (A) reference-swap census (prefill r_pos vs canonical oracle) ----
        new_head_cache = {}
        for pid, m in marg.items():
            o = oracle.get(pid)
            new_head_cache[pid] = first_div(m["served"], o) if o is not None else 0
        fires = [r for r in cap if r["is_fire"]]
        head_fires = head_breaks = head_conf_miss = 0
        bx_head_fires = bx_head_breaks = 0
        for r in fires:
            o = oracle.get(r["id"])
            if o is None or r["pos"] >= len(o):
                continue
            otok = o[r["pos"]]
            pre = r["pos"] < new_head_cache[r["id"]]
            if not pre:
                continue
            head_fires += 1
            brk = (r["r_pos"] != otok)
            if brk:
                head_breaks += 1
                rcm = r.get("recompute_margin")
                if rcm is not None and rcm != float("inf") and rcm > CONF:
                    head_conf_miss += 1
            if r["sha_ok"]:
                bx_head_fires += 1
                bx_head_breaks += int(brk)

        summary[f"k{k}"] = {
            "B_oracle_eq_arref_prompts": n_oracle_eq_arref_prompts,
            "B_head_shorter_on_canonical": head_shorter,
            "B_head_longer_on_canonical": head_longer,
            "B_head_same": head_same,
            "B_n_canonical_residuals": len(res_k),
            "B_residuals_served_wins_idx": sum(1 for r in res_k if r["served_wins_lowest_index"]),
            "B_residuals_verify_tie_lt_tau": sum(1 for r in res_k
                                                 if r["verify_margin"] is not None and r["verify_margin"] < TAU),
            "B_residuals_verify_confident_ge_tau": sum(1 for r in res_k
                                                       if r["verify_margin"] is not None and r["verify_margin"] >= CONF),
            "A_head_fires": head_fires,
            "A_head_breaks": head_breaks,
            "A_head_break_rate": (head_breaks / head_fires) if head_fires else None,
            "A_head_confident_miss": head_conf_miss,
            "A_bx_head_fires": bx_head_fires,
            "A_bx_head_breaks": bx_head_breaks,
            "A_bx_head_break_rate": (bx_head_breaks / bx_head_fires) if bx_head_fires else None,
        }

    out = {"tau": TAU, "confident_thresh": CONF, "per_k": summary, "residuals": all_residuals}
    (HERE / "recensus_canonical.json").write_text(json.dumps(out, indent=2))

    print("\n========== PR #654 Part 2: census vs canonical M=1 oracle ==========")
    for k in (3, 5, 7):
        s = summary[f"k{k}"]
        print(f"\n--- K={k} ---")
        print(f"  (B) prompts where canonical==ar_ref_bi1: {s['B_oracle_eq_arref_prompts']}/128 "
              f"| canonical head shorter/longer/same = "
              f"{s['B_head_shorter_on_canonical']}/{s['B_head_longer_on_canonical']}/{s['B_head_same']}")
        print(f"      genuine canonical residuals (served==ar_ref but served!=canonical): "
              f"{s['B_n_canonical_residuals']}")
        print(f"        served wins lowest-index vs canonical: {s['B_residuals_served_wins_idx']}/{s['B_n_canonical_residuals']}")
        print(f"        verify_margin < {TAU} (near-tie): {s['B_residuals_verify_tie_lt_tau']}/{s['B_n_canonical_residuals']}"
              f"  | verify_margin >= {CONF} (confident): {s['B_residuals_verify_confident_ge_tau']}")
        print(f"  (A) reference-swap head break rate: {s['A_head_breaks']}/{s['A_head_fires']} = "
              f"{(s['A_head_break_rate'] or 0)*100:.4f}%  | confident misses (rcmarg>{CONF}): {s['A_head_confident_miss']}")
        print(f"      bx-subset head break rate: {s['A_bx_head_breaks']}/{s['A_bx_head_fires']} = "
              f"{(s['A_bx_head_break_rate'] or 0)*100:.4f}%")
    print(f"\n[out] {HERE/'recensus_canonical.json'}")
    print(f"[residuals to logprobs-probe] {len(all_residuals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
