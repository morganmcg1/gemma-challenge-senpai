#!/usr/bin/env python3
"""Decisive probe for the v1b train<->serve interface-fidelity finding.

At SERVE time, HF's SinglePositionMultiTokenCandidateGenerator feeds the draft a
step-0 hidden state taken from the target's verify-forward output, sliced at
`n_last_matches`. Our TRAINING feeds the draft `target_prefill_hidden[:, j]` (the
target's hidden at the SAME position as the seen token j). If those differ, every
offline fine-tune drifts the draft off the serving optimum -- the v0/v1b -5..-6%
native regression while tf went +10..+16%.

This script monkeypatches get_candidates to capture, at the start of every real
drafting round, the exact (last_token_id, step0_hidden) the draft receives. It
then runs a fresh clean target prefill over the realized output sequence and, for
each captured round, finds which prefill position's hidden the served step-0
hidden matches (argmin L2). The reported offset = matched_prefill_pos - last_token_pos
tells us the serve-faithful pairing: training must feed hidden at (token_pos + offset).
"""
from __future__ import annotations

import argparse
import json

import torch

import mtp_common as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", default="google/gemma-4-E4B-it-assistant")
    ap.add_argument("--target", default="google/gemma-4-E4B-it")
    ap.add_argument("--heldout", default="research/wide_drafter/corpus/heldout.jsonl")
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--max-prompt-tokens", type=int, default=768)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from transformers.generation import candidate_generator as CG

    tok = AutoTokenizer.from_pretrained(args.target)
    target = M.load_target(args.target, args.device).eval()
    drafter = M.load_drafter(args.drafter, args.device).eval()
    drafter.generation_config.num_assistant_tokens = args.K
    drafter.generation_config.num_assistant_tokens_schedule = "constant"

    CGED = CG.SinglePositionMultiTokenCandidateGenerator
    orig = CGED.get_candidates
    captures = []

    def patched(self, input_ids, model_kwargs, model_outputs, is_first_iteration,
                n_last_matches, **kw):
        if (not is_first_iteration and model_outputs is not None
                and getattr(model_outputs, "hidden_states", None) is not None):
            lh = model_outputs.hidden_states[-1]
            step0_hidden = lh[:, n_last_matches:n_last_matches + 1, :].detach().float().cpu()
            captures.append({
                "cur_len": int(input_ids.shape[1]),
                "n_last_matches": int(n_last_matches),
                "last_tok": int(input_ids[0, -1].item()),
                "verify_seqlen": int(lh.shape[1]),
                "step0_hidden": step0_hidden,            # [1,1,H]
            })
        return orig(self, input_ids, model_kwargs=model_kwargs, model_outputs=model_outputs,
                    is_first_iteration=is_first_iteration, n_last_matches=n_last_matches, **kw)

    CGED.get_candidates = patched

    rows = [json.loads(l) for l in open(args.heldout)][:args.n_prompts]
    all_offsets = []
    for pi, r in enumerate(rows):
        captures.clear()
        ct = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                     add_generation_prompt=True, return_tensors="pt")
        ids = (ct["input_ids"] if hasattr(ct, "keys") else ct)
        if ids.shape[1] > args.max_prompt_tokens:
            ids = ids[:, -args.max_prompt_tokens:]
        ids = ids.to(args.device)

        with torch.no_grad():
            out = target.generate(ids, do_sample=False, max_new_tokens=args.new_tokens,
                                  use_cache=True, assistant_model=drafter)
        full = out  # [1, L]
        # fresh clean prefill over the realized output -> target hidden at every position
        with torch.no_grad():
            prefill_hidden, _, _ = M.target_prefill(target, full)
        ph = prefill_hidden[0].float().cpu()  # [L, H]

        for ci, c in enumerate(captures):
            last_tok_pos = c["cur_len"] - 1          # 0-indexed position of input_ids[-1]
            sh = c["step0_hidden"][0, 0]             # [H]
            # match the served step-0 hidden against every clean-prefill position
            d = (ph - sh.unsqueeze(0)).pow(2).sum(-1).sqrt()   # [L]
            best = int(d.argmin().item())
            offset = best - last_tok_pos
            # token at the matched prefill position + at last_tok_pos (sanity)
            all_offsets.append(offset)
            if ci < 4:
                # what training currently feeds: hidden at last_tok_pos itself
                d_same = float((ph[last_tok_pos] - sh).pow(2).sum().sqrt().item())
                print(f"[p{pi} round{ci}] cur_len={c['cur_len']} n_last_matches={c['n_last_matches']} "
                      f"verify_seqlen={c['verify_seqlen']} last_tok_pos={last_tok_pos} "
                      f"-> best_match_pos={best} offset={offset:+d} L2@best={float(d[best]):.4f} "
                      f"L2@same_pos(train)={d_same:.4f}", flush=True)
        print(f"[p{pi}] {len(captures)} rounds, out_len={full.shape[1]}", flush=True)

    CGED.get_candidates = orig
    from collections import Counter
    hist = Counter(all_offsets)
    print("\n=== OFFSET HISTOGRAM (matched_prefill_pos - last_token_pos) ===", flush=True)
    print(json.dumps({str(k): v for k, v in sorted(hist.items())}, indent=2), flush=True)
    print(f"dominant offset = {hist.most_common(1)[0][0]:+d}  "
          f"({hist.most_common(1)[0][1]}/{len(all_offsets)} rounds)", flush=True)
    print("PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
