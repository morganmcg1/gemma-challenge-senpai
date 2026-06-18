"""PR #645 step 1: extract ROOT divergences (first fork position) from #632 data.

Reuses the #632 captures (no generation):
  - AR M=1 reference: ar_ref_bi1/decode_outputs.jsonl   (served spec-OFF, BASELINE.md L10)
  - K=7 spec lane:    k3/k7/decode/run00.jsonl           (served Option-B BI=1 spec)
  - gate_k7.json:     official greedy_gate per-prompt first_divergence_index

For each diverged prompt: root position p, spec token A = K7[p], AR token B = AR[p],
plus the shared prefix (prompt_token_ids + completion[:p]) needed to re-serve.
"""
from __future__ import annotations

import json
from pathlib import Path

KS = Path(__file__).resolve().parent.parent / "ksweep"
AR = KS / "ar_ref_bi1" / "decode_outputs.jsonl"
K7 = KS / "k3" / "k7" / "decode" / "run00.jsonl"
GATE = KS / "gate_k7.json"
OUT = Path(__file__).resolve().parent / "root_forks.jsonl"


def load_jsonl_by_id(path: Path) -> dict:
    out = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["id"]] = d
    return out


def main() -> int:
    ar = load_jsonl_by_id(AR)
    k7 = load_jsonl_by_id(K7)
    gate = json.loads(GATE.read_text())

    # alignment sanity
    ar_ids, k7_ids = set(ar), set(k7)
    assert ar_ids == k7_ids, f"id-set mismatch ar={len(ar_ids)} k7={len(k7_ids)} sym={len(ar_ids ^ k7_ids)}"
    gate_ids = {p["key"] for p in gate["per_prompt"]}
    assert gate_ids == ar_ids, f"gate-key set != decode-id set (sym={len(gate_ids ^ ar_ids)})"
    print(f"[align] {len(ar_ids)} prompts, ids match across AR / K7 / gate")

    # per-id prompt identity (same input fed both lanes)
    for pid in ar_ids:
        assert ar[pid]["prompt_token_ids"] == k7[pid]["prompt_token_ids"], f"prompt mismatch {pid}"

    n_div = 0
    n_ident = 0
    forks = []
    for pr in gate["per_prompt"]:
        pid = pr["key"]
        a_rec, k_rec = ar[pid], k7[pid]
        a_tok = a_rec["completion_token_ids"]
        k_tok = k_rec["completion_token_ids"]
        # independently recompute first divergence (don't just trust the gate)
        p_indep = None
        for i in range(min(len(a_tok), len(k_tok))):
            if a_tok[i] != k_tok[i]:
                p_indep = i
                break
        if pr["identical"]:
            assert p_indep is None, f"{pid}: gate says identical but forks at {p_indep}"
            n_ident += 1
            continue
        n_div += 1
        p = pr["first_divergence_index"]
        assert p == p_indep, f"{pid}: gate p={p} != independent p={p_indep}"
        # shared prefix must be byte-identical up to the root (definition of ROOT fork)
        assert a_tok[:p] == k_tok[:p], f"{pid}: prefix differs before root p={p}"
        A = k_tok[p]   # spec lane emitted token (M=8 argmax)
        B = a_tok[p]   # AR M=1 emitted token (M=1 argmax)
        assert A != B, f"{pid}: A==B at root p={p}"
        forks.append({
            "id": pid,
            "root_pos": p,
            "spec_token_A": A,
            "ar_token_B": B,
            "prompt_token_ids": k_rec["prompt_token_ids"],
            "shared_completion_prefix": k_tok[:p],   # == a_tok[:p]
            "spec_completion_sha256": k_rec["completion_token_sha256"],
            "num_completion_tokens": len(k_tok),
        })

    with OUT.open("w") as fh:
        for f in forks:
            fh.write(json.dumps(f) + "\n")

    # scale stats on the root positions themselves
    ps = sorted(f["root_pos"] for f in forks)
    n = len(ps)

    def pct(q):
        return ps[min(n - 1, int(q * n))]

    print(f"[forks] n_prompts_diverged = {n_div}  (n_identical = {n_ident}, total {n_div + n_ident})")
    print(f"[forks] n_root_forks       = {n} (one root per diverged prompt)")
    print(f"[forks] root_pos min/median/p95/max = {ps[0]} / {pct(0.5)} / {pct(0.95)} / {ps[-1]}")
    print(f"[forks] wrote {OUT}")
    # cross-check vs #632 banked numbers
    print(f"[xcheck] gate verdict={gate['verdict']} num_identical={gate['num_identical']} "
          f"num_divergent={gate['num_divergent']} (expect 20 / 108)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
