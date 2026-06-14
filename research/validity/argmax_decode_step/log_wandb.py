#!/usr/bin/env python3
"""Append PR #180 propagation + self-test metrics to the A/B wandb run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb


def _flat(prefix, d, out):
    for k, val in d.items():
        key = f"{prefix}{k}"
        if isinstance(val, dict):
            _flat(key + "/", val, out)
        elif isinstance(val, bool):
            out[key] = int(val)
        elif isinstance(val, (int, float)):
            out[key] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project", default="gemma-challenge-senpai")
    ap.add_argument("--entity", default="wandb-applied-ai-team")
    ap.add_argument("--group", default="argmax-decode-step-realization")
    args = ap.parse_args()

    res = json.loads(args.result.read_text())

    run = wandb.init(
        id=args.run_id, project=args.project, entity=args.entity,
        group=args.group, resume="allow",
    )
    flat: dict = {}
    for top in ("measurement", "step", "self_test_legs"):
        if top in res:
            _flat(f"prop/{top}/", res[top], flat)
    _flat("prop/verdict/", res.get("verdict_descent_only", {}).get("at_realized", {}), flat)
    flat.update({
        "prop/argmax_decode_step_self_test_passes": int(res["argmax_decode_step_self_test_passes"]),
        "prop/descent_only_lcb_with_argmax_decode": res["descent_only_lcb_with_argmax_decode"],
        "prop/descent_only_clears_500_with_argmax": int(res["descent_only_clears_500_with_argmax"]),
        "prop/output_neutral": int(res["output_neutral"]),
        "prop/token_identity_rate": res["token_identity_rate"],
        "prop/n_identical": res["n_identical"],
        "prop/n_shared": res["n_shared"],
    })
    if res.get("ppl_argmax") is not None:
        flat["prop/ppl_argmax"] = res["ppl_argmax"]
    if res.get("ppl_control") is not None:
        flat["prop/ppl_control"] = res["ppl_control"]
    if res.get("ppl_num_tokens") is not None:
        flat["prop/ppl_num_tokens"] = res["ppl_num_tokens"]

    run.summary.update(flat)
    run.config.update({"pr": 180, "leg": "argmax-decode-step-realization"}, allow_val_change=True)
    print(f"[wandb] updated run {args.run_id} with {len(flat)} summary metrics")
    print(f"[wandb] PRIMARY argmax_decode_step_self_test_passes="
          f"{res['argmax_decode_step_self_test_passes']}  "
          f"TEST descent_only_lcb={res['descent_only_lcb_with_argmax_decode']:.4f}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
