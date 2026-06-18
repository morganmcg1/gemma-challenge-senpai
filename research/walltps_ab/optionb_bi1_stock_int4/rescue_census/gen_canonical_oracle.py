"""PR #654 Part 2: regenerate a batch-M-STABLE single-seq (M=1) decode ar_ref oracle.

ar_ref_bi1 was captured via the served reference path; #651's decode-path validation proved
it is batch-M-AMBIGUOUS at exact int4 ties (4-6 prompts/K where a strict single-seq M=1 decode
!= the stored ar_ref). This script pins the canonical reference: for each of the 128 prompts we
send ONE decode request at a time to the spec-OFF MAX_NUM_SEQS=1 reference server
(boot_ref_server.sh) -> every generation step is a true M=1 decode with NO cross-prompt
batching, so the int4 Marlin GEMM M is fixed at 1 (the batch-invariant decode regime). This is
the same launcher/env that made ar_ref_bi1 (SENPAI_REFERENCE_MODE=1, BI=1, FlashInfer-sampler
off, temp 0, add_special_tokens=false, ignore_eos), minus the served-path batching.

Saved as a NEW artifact ar_ref_m1_canonical/ (does NOT overwrite ar_ref_bi1). Reusable oracle
for all future served-identity census (#645/#648/#651/#654). analysis_only.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = HERE.parent / "ksweep"
REF_K5 = KS / "k5" / "k5" / "decode" / "run00.jsonl"   # 128 prompt_token_ids (K-independent prompts)
OUT_DIR = KS / "ar_ref_m1_canonical"
BASE = "http://127.0.0.1:8000"
MODEL = "gemma-4-e4b-it"
OUTPUT_LEN = 512


NEAR_TIE_NAT = 1.0  # store full top-N at positions whose canonical top1-top2 gap is below this


def post(payload, timeout_s=900):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def extract_token_ids(choice):
    for v in (choice.get("token_ids"), choice.get("output_token_ids"),
              choice.get("completion_token_ids")):
        if isinstance(v, list) and v and isinstance(v[0], int):
            return v
    raise ValueError(f"no generated token_ids in choice keys={list(choice.keys())}")


def per_pos_margins(choice):
    """canonical DECODE-path top1-top2 gap (nat) at every generated position, plus the full
    top-N {str: logprob} at near-tie positions (gap < NEAR_TIE_NAT). These are the M=1
    decode-path logits captured DURING generation -- the faithful canonical margins (a prefill
    re-probe would flip at exactly these int4 ties)."""
    tl = choice["logprobs"]["top_logprobs"]  # list[ {token_str: logprob} ] per position
    margins = []
    near = {}
    for i, top in enumerate(tl):
        vals = sorted(top.values(), reverse=True)
        gap = (vals[0] - vals[1]) if len(vals) >= 2 else float("inf")
        margins.append(gap if gap != float("inf") else None)
        if gap < NEAR_TIE_NAT:
            near[str(i)] = top
    return margins, near


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUT_DIR / "decode_outputs.jsonl"
    marg_jsonl = OUT_DIR / "canonical_margins.jsonl"

    # load 128 prompts (id -> prompt_token_ids), preserve stream order
    prompts = []
    with REF_K5.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                prompts.append((d["id"], d["prompt_token_ids"]))
    if args.smoke:
        prompts = prompts[: args.smoke]
    print(f"[oracle] {len(prompts)} prompts -> {out_jsonl} (output_len={args.output_len})")

    t0 = time.time()
    n_tok = 0
    with out_jsonl.open("w") as out, marg_jsonl.open("w") as mout:
        for i, (pid, ptoks) in enumerate(prompts):
            payload = {
                "model": MODEL, "prompt": ptoks,
                "max_tokens": args.output_len, "temperature": 0.0, "stream": False,
                "add_special_tokens": False, "ignore_eos": True, "return_token_ids": True,
                "logprobs": 20,
            }
            resp = post(payload)
            choice = resp["choices"][0]
            toks = extract_token_ids(choice)
            margins, near = per_pos_margins(choice)
            n_tok += len(toks)
            out.write(json.dumps({"id": pid, "completion_token_ids": toks,
                                  "n_tokens": len(toks)}) + "\n")
            out.flush()
            mout.write(json.dumps({"id": pid, "top1top2_margins": margins,
                                   "near_tie_topN": near}) + "\n")
            mout.flush()
            if (i + 1) % 8 == 0 or args.smoke:
                dt = time.time() - t0
                print(f"  [{i+1}/{len(prompts)}] {pid} len={len(toks)} "
                      f"({n_tok} tok, {n_tok/dt:.1f} tok/s, {dt:.0f}s)", flush=True)

    wall = time.time() - t0
    meta = {
        "kind": "canonical_single_seq_M1_decode_oracle",
        "purpose": "PR#654 batch-M-stable ar_ref; do NOT overwrite ar_ref_bi1",
        "num_records": len(prompts), "output_len": args.output_len,
        "num_completion_tokens": n_tok, "capture_wall_s": wall,
        "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "served_model_name": MODEL,
        "ref_env": {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
                    "SENPAI_REFERENCE_MODE": "1", "MAX_NUM_SEQS": "1"},
        "request": {"temperature": 0.0, "add_special_tokens": False, "ignore_eos": True,
                    "one_request_at_a_time": True},
        "served_via": "submission:submissions/int4_mtp_batchinv [spec-off, MAX_NUM_SEQS=1]",
        "prompt_source": str(REF_K5),
        "supersedes_for_canonical_reference": "ar_ref_bi1/decode_outputs.jsonl",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[oracle] done: {n_tok} tokens in {wall:.0f}s ({n_tok/wall:.1f} tok/s)")
    print(f"[oracle] artifact -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
