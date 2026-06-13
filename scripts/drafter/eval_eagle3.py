# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Offline acceptance eval for the EAGLE-3 draft head (PR #16, Step 4).

Loads a trained checkpoint (`config.json` + `model_*.pt` from train_eagle3.py) and
scores teacher-forced top-1 acceptance on a held-out corpus:

    tf_acceptance_rate = mean( argmax(head_logits[j]) == next_token_ids[j] )

over all non-IGNORE positions, using the same feature/embedding alignment
(`--feature_shift`, default from the checkpoint config) as serving. This is the
single-step, teacher-forced proxy for vLLM draft acceptance (arch_notes S5/S6).

Interpretation bands (PR #16): <0.50 underfit/broken, 0.50-0.70 expected,
>=0.70 strong.

Run (from target/):
  HF_HOME=/senpai-run/home/student-fern/.cache/huggingface \
  python scripts/drafter/eval_eagle3.py \
      --checkpoint research/eagle3_drafter/checkpoints/debug_1k_2ep/ \
      --eval_corpus research/eagle3_drafter/train_data/debug_1k_eval_corpus.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_eagle3 import (  # noqa: E402
    Eagle3DraftHead,
    evaluate,
    load_corpus,
)


def band(acc: float) -> str:
    if acc < 0.50:
        return "underfit/broken (<0.50)"
    if acc < 0.70:
        return "expected (0.50-0.70)"
    return "strong (>=0.70)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="dir with config.json + model_*.pt, or a .pt file")
    ap.add_argument("--eval_corpus", required=True)
    ap.add_argument("--weights", default="model_best.pt",
                    help="weights file inside --checkpoint dir (falls back to last)")
    ap.add_argument("--feature_shift", type=int, default=None,
                    help="override; default reads checkpoint config")
    ap.add_argument("--batch_tokens", type=int, default=4096)
    ap.add_argument("--rope_theta", type=float, default=None)
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "senpai-v1"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", default="eagle3-drafter-training")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    if os.path.isdir(args.checkpoint):
        cfg_path = os.path.join(args.checkpoint, "config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        wpath = os.path.join(args.checkpoint, args.weights)
        if not os.path.exists(wpath):
            wpath = os.path.join(args.checkpoint, "model_last.pt")
    else:
        cfg, wpath = {}, args.checkpoint

    shift = args.feature_shift if args.feature_shift is not None else cfg.get("feature_shift", 1)
    rope_theta = args.rope_theta if args.rope_theta is not None else cfg.get("rope_theta", 1e6)
    norm_before_fc = bool(cfg.get("norm_before_fc", True))

    print(f"[eval] checkpoint: {wpath}", flush=True)
    print(f"[eval] feature_shift={shift} rope_theta={rope_theta:g} "
          f"norm_before_fc={norm_before_fc}", flush=True)

    head = Eagle3DraftHead(norm_before_fc=norm_before_fc).to(device)
    state = torch.load(wpath, map_location=device, weights_only=False)
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[eval] load: missing={list(missing)} unexpected={list(unexpected)}",
              flush=True)
    # Ensure frozen tables are bf16 (match training).
    head.model.embed_tokens.weight.data = head.model.embed_tokens.weight.data.to(dtype)
    head.lm_head.weight.data = head.lm_head.weight.data.to(dtype)

    records, _ = load_corpus(args.eval_corpus)
    res = evaluate(head, records, shift, args.batch_tokens, device, dtype, rope_theta)
    acc = res["tf_acceptance_rate"]

    print("=" * 60, flush=True)
    print(f"[eval] tf_acceptance_rate = {acc:.4f}  ({band(acc)})", flush=True)
    print(f"[eval] loss = {res['loss']:.4f}   positions scored = {res['n']}", flush=True)
    print("=" * 60, flush=True)

    out = {"eval/tf_acceptance_rate": acc, "eval/loss": res["loss"],
           "eval/n": res["n"], "eval/feature_shift": shift, "eval/band": band(acc)}
    if args.wandb_name:
        try:
            import wandb

            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                             group=args.wandb_group, name=args.wandb_name,
                             config={"checkpoint": wpath, "feature_shift": shift,
                                     "rope_theta": rope_theta})
            run.log(out)
            run.summary.update(out)
            run.finish()
        except Exception as e:  # noqa: BLE001
            print(f"[eval] wandb disabled ({e!r})", flush=True)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
