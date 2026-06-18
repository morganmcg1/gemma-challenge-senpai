"""PR #645: deterministic forced-prefix re-measure of root-fork margins.

For the 2 root forks whose run-to-run faithful replay flipped off-trajectory
(emit_eq_A=False), feed the EXACT stored shared prefix (prompt_token_ids +
shared_completion_prefix[:p]) and read the verify-forward logprobs at the first
generated position p. Greedy + a forced byte-identical context is deterministic,
so this reproduces the stored-context margin without depending on a lucky
run-to-run trajectory match.

Validation: run the SAME method on 3 forks that DID replay faithfully and check
the forced-prefix top1/top2/gap reproduces the faithful mid-stream values
(0.125 / 0.25 / 0.0). If it matches, the prefix-context margin is a sound proxy
and we trust it for the 2 flipped forks.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT_FORKS = HERE / "root_forks.jsonl"
BASE = "http://127.0.0.1:8000"
MODEL = "gemma-4-e4b-it"

# forks to re-measure (the 2 that flipped off-trajectory) + 3 faithful anchors
FLIPPED = ["mmlu_pro-006f3a2112", "mmlu_pro-012f0d5c8d"]
ANCHORS = {  # id -> faithful mid-stream top1-top2 gap (from margin_records.jsonl)
    "mmlu_pro-005aa2e50a": 0.125,
    "mmlu_pro-00a3fcc287": 0.250,
    "aime2026-0e674f51e8": 0.000,
}


def post(payload, timeout_s=300):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def load_forks():
    out = {}
    with ROOT_FORKS.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["id"]] = d
    return out


def measure(fork, n_logprobs=20):
    """Feed prompt + shared_prefix[:p]; read logprobs at the first output pos (=p)."""
    p = fork["root_pos"]
    prefix = fork["prompt_token_ids"] + fork["shared_completion_prefix"][:p]
    assert len(fork["shared_completion_prefix"]) == p, \
        f"{fork['id']}: stored prefix len {len(fork['shared_completion_prefix'])} != p {p}"
    payload = {
        "model": MODEL,
        "prompt": prefix,
        "max_tokens": 2,           # 1st output = position p
        "temperature": 0.0,
        "stream": False,
        "add_special_tokens": False,
        "ignore_eos": True,
        "return_token_ids": True,
        "logprobs": n_logprobs,
    }
    resp = post(payload)
    ch = resp["choices"][0]
    lp = ch["logprobs"]
    top = lp["top_logprobs"][0]               # dict token_str -> logprob at pos p
    emitted_lp = lp["token_logprobs"][0]
    items = sorted(top.items(), key=lambda kv: kv[1], reverse=True)
    # emitted token id at the first generated position
    emit_id = None
    for cand in ("token_ids", "output_token_ids", "completion_token_ids"):
        v = ch.get(cand)
        if isinstance(v, list) and v:
            # strip prompt echo if present
            if len(v) >= len(prefix) and v[:len(prefix)] == prefix:
                v = v[len(prefix):]
            emit_id = v[0] if v else None
            break
    return {
        "emit_id": emit_id, "emitted_lp": emitted_lp,
        "top1": items[0], "top2": items[1], "raw_top": items,
    }


def main() -> int:
    forks = load_forks()
    print("=== VALIDATION: forced-prefix vs faithful mid-stream gap ===")
    ok = True
    for pid, faithful_gap in ANCHORS.items():
        f = forks[pid]
        m = measure(f)
        gap = m["top1"][1] - m["top2"][1]
        match = abs(gap - faithful_gap) < 1e-6
        ok = ok and match
        print(f"{pid} p={f['root_pos']} A={f['spec_token_A']} "
              f"emit_id={m['emit_id']} top1={m['top1'][0]!r}/{m['top1'][1]:.7f} "
              f"top2={m['top2'][0]!r}/{m['top2'][1]:.7f} gap={gap:.7f} "
              f"(faithful {faithful_gap}) {'OK' if match else 'MISMATCH'}")
    print(f"\nvalidation: {'PASS' if ok else 'FAIL'}")

    print("\n=== RE-MEASURE flipped forks (deterministic forced-prefix) ===")
    out = {}
    for pid in FLIPPED:
        f = forks[pid]
        m = measure(f)
        A, B = f["spec_token_A"], f["ar_token_B"]
        gap = m["top1"][1] - m["top2"][1]
        emit_eq_A = (m["emit_id"] == A)
        print(f"{pid} p={f['root_pos']} A={A} B={B} emit_id={m['emit_id']} "
              f"emit_eq_A={emit_eq_A} top1={m['top1'][0]!r}/{m['top1'][1]:.7f} "
              f"top2={m['top2'][0]!r}/{m['top2'][1]:.7f} gap={gap:.7f}")
        out[pid] = {
            "id": pid, "root_pos": f["root_pos"], "A": A, "B": B,
            "emit_id": m["emit_id"], "emitted_eq_A": emit_eq_A,
            "logp_A": m["emitted_lp"], "raw_top": m["raw_top"],
            "method": "forced_prefix",
        }
    (HERE / "forced_prefix_records.jsonl").write_text(
        "\n".join(json.dumps(out[pid]) for pid in FLIPPED) + "\n")
    print(f"\n[out] wrote {HERE / 'forced_prefix_records.jsonl'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
