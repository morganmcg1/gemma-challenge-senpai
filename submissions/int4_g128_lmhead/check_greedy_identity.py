#!/usr/bin/env python
"""Greedy-identity self-check for the int4 g128 + untied int4 lm_head submission.

Reproduces the OFFICIAL gate (shared_resources/gemma_greedy_identity_verifier):
the served endpoint's greedy decode must be TOKEN-IDENTICAL to a plain greedy
autoregressive decode of the SAME submitted checkpoint. The official harness
captures one ``decode_outputs.jsonl`` per endpoint over HTTP (``decode_outputs.py``)
and compares two such files byte-for-byte -- so BOTH sides run the same int4
Marlin kernel, and the comparison is only well defined when reference and
candidate share the serving config. The canonical config is the baseline
manifest / README "default env" (``MAX_NUM_BATCHED_TOKENS=512``), which this
submission adopts verbatim with no token-changing optimizations.

Usage (two captures + one compare; servers run one at a time on a single GPU):

  # 1. REFERENCE: honest plain-vLLM decode at the standard config
  vllm serve <ckpt> --served-model-name gemma-4-e4b-it --dtype bfloat16 \
      --max-model-len 4096 --gpu-memory-utilization 0.90 \
      --max-num-batched-tokens 512 --trust-remote-code --no-enable-log-requests
  python check_greedy_identity.py --phase capture --out ref.jsonl
  # (stop server)

  # 2. CANDIDATE: this submission's serve.py (same standard config)
  MODEL_ID=<ckpt> MAX_NUM_BATCHED_TOKENS=512 python serve.py
  python check_greedy_identity.py --phase capture --out cand.jsonl
  # (stop server)

  # 3. COMPARE (byte-exact, official semantics)
  python check_greedy_identity.py --phase compare --reference ref.jsonl --candidate cand.jsonl
  #   -> VERDICT: GREEDY_IDENTICAL (valid)  [exit 0]

NOTE: a NUMERICALLY DIFFERENT reference (e.g. a dequantized fake-quant HF model,
or a different prefill-chunk size) will show near-tie argmax flips that CASCADE
in greedy decode -- that is a config/kernel mismatch, not a gate failure. This
int4 checkpoint is near-tie dense, so the reference must use the same Marlin
kernel and the same standard serving config as the candidate (as the harness does).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

DEFAULT_DATASET = (
    "official/main_bucket/shared_resources/speed_benchmark/data/"
    "eval_prompts_sharegpt.json"
)


# ---- official sha + byte-exact comparison (vendored from greedy_identity.py) ----
def sha256_tokens(token_ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(t) for t in token_ids).encode("ascii")).hexdigest()


def load_decode_outputs(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        key = rec.get("id")
        if key is None:
            key = rec.get("prompt_sha256")
        key = str(key)
        ids = rec.get("completion_token_ids")
        if not isinstance(ids, list) or any(not isinstance(t, int) or isinstance(t, bool) for t in ids):
            raise ValueError(f"{path}:{lineno} bad completion_token_ids")
        if key in records:
            raise ValueError(f"{path}:{lineno} duplicate key {key!r}")
        records[key] = rec
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def compare(reference: dict[str, dict], candidate: dict[str, dict]) -> dict:
    ref_keys, cand_keys = set(reference), set(candidate)
    missing_cand = sorted(ref_keys - cand_keys)
    missing_ref = sorted(cand_keys - ref_keys)
    num_identical = num_divergent = total = total_div = 0
    first_div: dict | None = None
    for key in sorted(ref_keys & cand_keys):
        r = reference[key]["completion_token_ids"]
        c = candidate[key]["completion_token_ids"]
        n = min(len(r), len(c))
        diff = sum(1 for i in range(n) if r[i] != c[i])
        identical = (diff == 0) and (len(r) == len(c))
        total += n
        total_div += diff + abs(len(r) - len(c))
        if identical:
            num_identical += 1
        else:
            num_divergent += 1
            if first_div is None:
                idx = next((i for i in range(n) if r[i] != c[i]), n)
                first_div = {"key": key, "first_divergence_index": idx}
    key_sets_match = not missing_cand and not missing_ref
    if not key_sets_match:
        verdict = "INCOMPARABLE"
    elif num_divergent == 0:
        verdict = "GREEDY_IDENTICAL"
    else:
        verdict = "DIVERGENT"
    return {
        "verdict": verdict,
        "num_prompts_compared": len(ref_keys & cand_keys),
        "num_identical": num_identical,
        "num_divergent": num_divergent,
        "total_tokens_compared": total,
        "total_divergent_tokens": total_div,
        "missing_in_candidate": missing_cand,
        "missing_in_reference": missing_ref,
        "first_divergence": first_div,
    }


# ---- candidate capture (official decode_outputs.py request shape, HTTP only) ----
def to_id_list(x) -> list[int]:
    if hasattr(x, "input_ids"):
        x = x.input_ids
    if isinstance(x, dict):
        x = x.get("input_ids", x)
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, list) and x and isinstance(x[0], list):
        x = x[0]
    return [int(t) for t in x]


def read_sharegpt_prompts(path: Path, num_prompts: int, seed: int) -> list[dict]:
    data = json.loads(Path(path).read_text())
    recs = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        conv = item.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2 or not isinstance(conv[0], dict):
            continue
        p = conv[0].get("value")
        if isinstance(p, str) and p:
            recs.append({"id": str(item.get("id", index)), "index": index, "prompt_text": p})
    random.Random(seed).shuffle(recs)
    return recs[:num_prompts]


def post(url: str, payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract_completion_ids(resp: dict, prompt_ids: list[int]) -> list[int]:
    ch = resp["choices"][0]
    for v in (ch.get("token_ids"), ch.get("output_token_ids"),
              (ch.get("logprobs") or {}).get("token_ids") if isinstance(ch.get("logprobs"), dict) else None):
        if isinstance(v, list) and all(isinstance(t, int) and t >= 0 for t in v):
            if len(v) >= len(prompt_ids) and v[: len(prompt_ids)] == prompt_ids:
                return v[len(prompt_ids):]
            return v
    raise ValueError("endpoint returned no integer completion token IDs (need return_token_ids: true)")


def phase_capture(args) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    recs = read_sharegpt_prompts(Path(args.dataset), args.num_prompts, args.seed)
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for index, rec in enumerate(recs):
            ids = to_id_list(tok.apply_chat_template(
                [{"role": "user", "content": rec["prompt_text"]}], add_generation_prompt=True, tokenize=True))
            resp = post(f"{args.base_url}/v1/completions", {
                "model": args.model, "prompt": ids, "max_tokens": args.output_len,
                "temperature": 0.0, "stream": False, "add_special_tokens": False,
                "ignore_eos": True, "return_token_ids": True,
            })
            comp = extract_completion_ids(resp, ids)
            fh.write(json.dumps({
                "id": rec["id"], "index": index,
                "completion_token_ids": comp,
                "completion_token_sha256": sha256_tokens(comp),
                "num_completion_tokens": len(comp),
            }) + "\n")
            if (index + 1) % 16 == 0:
                print(f"  captured {index + 1}/{len(recs)}", flush=True)
    print(f"[capture] wrote {len(recs)} records -> {args.out}")


def phase_compare(args) -> None:
    rep = compare(load_decode_outputs(Path(args.reference)), load_decode_outputs(Path(args.candidate)))
    print(json.dumps(rep, indent=2))
    suffix = {"GREEDY_IDENTICAL": " (valid)", "DIVERGENT": " (invalid)"}.get(rep["verdict"], "")
    print(f"VERDICT: {rep['verdict']}{suffix} "
          f"[{rep['num_identical']}/{rep['num_prompts_compared']} prompts, "
          f"{rep['total_tokens_compared'] - rep['total_divergent_tokens']}/"
          f"{rep['total_tokens_compared']} tokens identical]")
    sys.exit({"GREEDY_IDENTICAL": 0, "DIVERGENT": 1}.get(rep["verdict"], 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["capture", "compare"])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--tokenizer", default="/workspace/gemma_build/int4_g128_lmhead")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="cand.jsonl")
    ap.add_argument("--reference")
    ap.add_argument("--candidate")
    args = ap.parse_args()
    if args.phase == "capture":
        phase_capture(args)
    else:
        if not (args.reference and args.candidate):
            ap.error("--phase compare requires --reference and --candidate")
        phase_compare(args)


if __name__ == "__main__":
    main()
