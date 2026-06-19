"""Ground-truth greedy AIME via HuggingFace ``transformers.generate`` (vLLM-independent).

PR #724. program.md L27-28 defines the #319 gate as token-identity to *"plain greedy
autoregressive decode."* HuggingFace ``transformers.generate(do_sample=False,
num_beams=1)`` **IS** plain greedy autoregressive decode, by definition. This pins the
canonical base AIME denominator, immune to the vLLM-engine corruption kanna #699 found
(``/tmp/vllm0220-srv`` gone; ``.venvs/vllm022`` corrupts greedy for both bodies).

Design choices that keep this a *gold reference*:
  * **Backend swap only.** We import ``aime_eval.py``'s dataset loader (``load_aime``),
    prompt builder (``build_messages``) and answer extractor (``extract_answer``)
    verbatim, so scoring is apples-to-apples with the served harness. Only the
    generation backend changes: vLLM-served -> local ``transformers.generate``.
  * **Sequential, batch=1.** The gold reference must not inherit any batch artifact, so
    each problem is generated alone. (``--mode batchcheck`` separately probes whether
    left-padded batching diverges from sequential; see that path.)
  * **One 8192 pass, derive 6144.** Greedy decode is deterministic, so the first 6144
    generated tokens of an 8192-budget run ARE exactly the 6144-budget completion. We
    generate once at the larger budget and derive every smaller budget by truncating the
    saved token ids and re-extracting. This is not an approximation; it is exact, and it
    guarantees the two budgets are perfectly nested (no run-to-run variation).
  * **eager attention.** The PR specifies eager; we additionally verified eager and sdpa
    give byte-identical greedy tokens here, so the reference is attention-kernel-invariant.
  * **Resumable.** Each finished problem is appended to a JSONL immediately; ``--resume``
    skips ids already present, so a recycled pod continues instead of restarting.

NOTHING here launches an HF Job, submission, or ``train.py``. Pure local analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the served harness's loader + extractor verbatim (apples-to-apples scoring).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "downstream_quality_aime"))
from aime_eval import build_messages, extract_answer, load_aime  # noqa: E402

# Model-level greedy stop set for google/gemma-4-E4B-it: <eos>=1 and the turn
# terminator <turn|>=106 (config.json top-level eos_token_id = [1, 106]). There is no
# generation_config.json in the snapshot, so we pass this explicitly or generation would
# only stop on <eos> (id 1, which chat rarely emits) and run every problem to the cap.
EOS_IDS = [1, 106]
PAD_ID = 0


def _truncated_at(gen_len: int, budget: int, stopped_on_eos: bool) -> bool:
    """Would a native ``budget``-token run have been cut off (hit the cap, no stop token)?

    Exact for deterministic greedy: a native-budget run emits ``min(natural_stop, budget)``
    tokens. If our (larger-budget) run produced ``gen_len > budget`` tokens, the native run
    is truncated at ``budget``. If ``gen_len <= budget`` it must have ended on a stop token
    (it stopped before the cap), so it is not truncated.
    """
    return (gen_len > budget) or (not stopped_on_eos and gen_len >= budget)


def load_model(model_id: str, attn: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation=attn
    ).to("cuda").eval()
    return tok, model


def encode_prompt(tok, problem_text: str) -> dict[str, torch.Tensor]:
    enc = tok.apply_chat_template(
        build_messages(problem_text),
        add_generation_prompt=True,
        enable_thinking=False,  # --no-thinking gate protocol
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v.to("cuda") for k, v in enc.items()}


@torch.no_grad()
def generate_one(model, enc: dict[str, torch.Tensor], *, min_new: int, max_new: int) -> list[int]:
    prompt_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        do_sample=False,
        num_beams=1,
        min_new_tokens=min_new,
        max_new_tokens=max_new,
        eos_token_id=EOS_IDS,
        pad_token_id=PAD_ID,
        use_cache=True,
    )
    return out[0][prompt_len:].tolist()


def score_record(prob: dict[str, Any], gen_ids: list[int], tok, budgets: list[int]) -> dict[str, Any]:
    gen_len = len(gen_ids)
    stopped_on_eos = gen_len > 0 and gen_ids[-1] in EOS_IDS
    gold = prob["answer"]
    rec: dict[str, Any] = {
        "id": prob["id"],
        "year": prob["year"],
        "gold": gold,
        "gen_len": gen_len,
        "stopped_on_eos": stopped_on_eos,
        "by_budget": {},
    }
    for b in budgets:
        ids_b = gen_ids[:b]
        text_b = tok.decode(ids_b, skip_special_tokens=True)
        ans_b = extract_answer(text_b)
        rec["by_budget"][str(b)] = {
            "answer": ans_b,
            "correct": (ans_b is not None and ans_b == gold),
            "truncated": _truncated_at(gen_len, b, stopped_on_eos),
            "extract_ok": ans_b is not None,
        }
    # Keep the full max-budget completion text + token ids for coherence audit / re-derive.
    rec["text"] = tok.decode(gen_ids, skip_special_tokens=True)
    rec["gen_ids"] = gen_ids
    return rec


def aggregate(records: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    n = len(records)
    agg: dict[str, Any] = {"n_problems": n, "by_budget": {}}
    for b in budgets:
        bs = str(b)
        ncorr = sum(int(r["by_budget"][bs]["correct"]) for r in records)
        ntrunc = sum(int(r["by_budget"][bs]["truncated"]) for r in records)
        nfail = sum(int(not r["by_budget"][bs]["extract_ok"]) for r in records)
        acc = ncorr / n if n else 0.0
        agg["by_budget"][bs] = {
            "accuracy": acc,
            "n_correct": ncorr,
            "n_truncated": ntrunc,
            "extract_fail": nfail,
            "gate_bar_0p9": 0.9 * acc,
        }
    return agg


def run_eval(args) -> int:
    years = [y.strip() for y in args.years.split(",") if y.strip()]
    budgets = sorted({int(b) for b in args.derive_budgets.split(",")}, reverse=True)
    assert budgets[0] <= args.max_new_tokens, "derive budgets must be <= max_new_tokens"
    problems = load_aime(years, limit=args.limit)
    print(f"[hf-aime] loaded {len(problems)} problems years={years}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{args.label}_perproblem.jsonl"
    done_ids: set[str] = set()
    if args.resume and jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
        print(f"[hf-aime] resume: {len(done_ids)} problems already done", flush=True)

    run = _init_wandb(args, years, budgets, len(problems))

    tok, model = load_model(args.model, args.attn)
    print(
        f"[hf-aime] model={args.model} attn={args.attn} "
        f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    # Re-load any already-done records so the final aggregate is complete on resume.
    if done_ids and jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    t_start = time.time()
    gen_tokens_total = 0
    with jsonl.open("a") as fh:
        for i, prob in enumerate(problems):
            if prob["id"] in done_ids:
                continue
            t0 = time.time()
            enc = encode_prompt(tok, prob["problem"])
            gen_ids = generate_one(model, enc, min_new=args.min_new_tokens, max_new=args.max_new_tokens)
            rec = score_record(prob, gen_ids, tok, budgets)
            rec["wall_s"] = round(time.time() - t0, 1)
            rec["prompt_len"] = int(enc["input_ids"].shape[1])
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            records.append(rec)
            gen_tokens_total += rec["gen_len"]
            top = budgets[0]
            tps = rec["gen_len"] / max(rec["wall_s"], 1e-6)
            print(
                f"[hf-aime] {i+1}/{len(problems)} id={prob['id']} gold={prob['gold'] if 'gold' in prob else prob['answer']} "
                f"len={rec['gen_len']} {'TRUNC' if rec['by_budget'][str(top)]['truncated'] else 'stop'} "
                f"ans@{top}={rec['by_budget'][str(top)]['answer']} "
                f"{'OK' if rec['by_budget'][str(top)]['correct'] else 'x'} "
                f"{rec['wall_s']}s ({tps:.1f} tok/s)",
                flush=True,
            )
            _log_running(run, records, budgets)

    agg = aggregate(records, budgets)
    agg["wall_s"] = round(time.time() - t_start, 1)
    agg["model"] = args.model
    agg["attn"] = args.attn
    agg["years"] = years
    agg["max_new_tokens"] = args.max_new_tokens
    agg["gen_tokens_total"] = gen_tokens_total

    summary_path = out_dir / f"{args.label}_summary.json"
    summary_path.write_text(json.dumps({**agg, "per_problem": records}, indent=2))
    print(f"[hf-aime] wrote {summary_path}", flush=True)
    for b in budgets:
        bb = agg["by_budget"][str(b)]
        print(
            f"[hf-aime] DONE {args.label} @ {b}: acc={bb['accuracy']:.4f} "
            f"({bb['n_correct']}/{agg['n_problems']}) trunc={bb['n_truncated']} "
            f"extract_fail={bb['extract_fail']} gate_bar(0.9x)={bb['gate_bar_0p9']:.4f}",
            flush=True,
        )
    _finalize_wandb(run, agg, args, budgets)
    return 0


# --------------------------------------------------------------------------- #
# Batch-vs-sequential integrity probe (canonical-decode guard, PR step 2)
# --------------------------------------------------------------------------- #
def run_batchcheck(args) -> int:
    years = [y.strip() for y in args.years.split(",") if y.strip()]
    problems = load_aime(years, limit=args.batch_n)
    print(f"[batchcheck] {len(problems)} problems, budget={args.batchcheck_tokens}", flush=True)
    tok, model = load_model(args.model, args.attn)

    # Sequential (the reference).
    seq: list[list[int]] = []
    for prob in problems:
        enc = encode_prompt(tok, prob["problem"])
        seq.append(generate_one(model, enc, min_new=args.min_new_tokens, max_new=args.batchcheck_tokens))

    # Batched, left-padded (the suspect path).
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = PAD_ID
    prompts = [
        tok.apply_chat_template(
            build_messages(p["problem"]), add_generation_prompt=True,
            enable_thinking=False, tokenize=False,
        )
        for p in problems
    ]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    plen = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc, do_sample=False, num_beams=1, min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.batchcheck_tokens, eos_token_id=EOS_IDS, pad_token_id=PAD_ID, use_cache=True,
        )
    batched = [out[r][plen:].tolist() for r in range(len(problems))]

    n_div = 0
    for i, prob in enumerate(problems):
        a, b = seq[i], batched[i]
        # trim trailing pads from batched row and compare up to min length
        L = min(len(a), len(b))
        first = next((j for j in range(L) if a[j] != b[j]), None)
        identical = (a == b)
        if not identical:
            n_div += 1
        coherent = "qlql" not in tok.decode(a, skip_special_tokens=True)[:200].lower()
        print(
            f"[batchcheck] id={prob['id']} seq_len={len(a)} bat_len={len(b)} "
            f"identical={identical} first_div={first} coherent={coherent}",
            flush=True,
        )
    verdict = "BATCH_IDENTICAL" if n_div == 0 else "BATCH_DIVERGES"
    print(f"[batchcheck] VERDICT {verdict} ({n_div}/{len(problems)} diverged)", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batchcheck.json").write_text(json.dumps({
        "verdict": verdict, "n_diverged": n_div, "n": len(problems),
        "budget": args.batchcheck_tokens,
        "seq_lens": [len(s) for s in seq], "bat_lens": [len(b) for b in batched],
    }, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# W&B (optional, never fatal to the eval)
# --------------------------------------------------------------------------- #
def _init_wandb(args, years, budgets, n):
    if args.no_wandb:
        return None
    try:
        import wandb
    except Exception as e:  # pragma: no cover
        print(f"[hf-aime] wandb unavailable: {e}", flush=True)
        return None
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=args.wandb_name,
            group=args.wandb_group,
            job_type="hf-groundtruth-aime",
            config={
                "pr": 724, "model": args.model, "attn": args.attn, "years": years,
                "max_new_tokens": args.max_new_tokens, "derive_budgets": budgets,
                "min_new_tokens": args.min_new_tokens, "n_problems": n, "label": args.label,
                "decode": "greedy do_sample=False num_beams=1", "seed": args.seed,
                # machine-checkable guard flags (no HF job / no submission / no fire)
                "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            },
        )
        # mirror guard flags into summary immediately (machine-checkable on partial runs)
        run.summary["analysis_only"] = 1
        run.summary["official_tps"] = 0
        run.summary["no_hf_job"] = 1
        run.summary["fires"] = 0
        return run
    except Exception as e:  # pragma: no cover
        print(f"[hf-aime] wandb.init failed (continuing without): {e}", flush=True)
        return None


def _log_running(run, records, budgets):
    if run is None:
        return
    try:
        n = len(records)
        top = budgets[0]
        ncorr = sum(int(r["by_budget"][str(top)]["correct"]) for r in records)
        run.log({
            "n_done": n,
            f"running_acc@{top}": ncorr / n if n else 0.0,
            "last_gen_len": records[-1]["gen_len"],
        })
    except Exception:
        pass


def _finalize_wandb(run, agg, args, budgets):
    if run is None:
        return
    try:
        import wandb
        s = run.summary
        for b in budgets:
            bb = agg["by_budget"][str(b)]
            s[f"aime_acc@{b}"] = bb["accuracy"]
            s[f"n_correct@{b}"] = bb["n_correct"]
            s[f"n_truncated@{b}"] = bb["n_truncated"]
            s[f"extract_fail@{b}"] = bb["extract_fail"]
            s[f"gate_bar_0p9@{b}"] = bb["gate_bar_0p9"]
        s["n_problems"] = agg["n_problems"]
        s["wall_s"] = agg["wall_s"]
        s["gen_tokens_total"] = agg["gen_tokens_total"]
        # per-problem table for later analysis
        cols = ["id", "year", "gold", "gen_len", "stopped_on_eos"] + [f"ans@{b}" for b in budgets] + [f"correct@{b}" for b in budgets] + [f"trunc@{b}" for b in budgets]
        tbl = wandb.Table(columns=cols)
        for r in agg.get("per_problem", []):
            row = [r["id"], r["year"], r["gold"], r["gen_len"], r["stopped_on_eos"]]
            for b in budgets:
                row.append(r["by_budget"][str(b)]["answer"])
            for b in budgets:
                row.append(r["by_budget"][str(b)]["correct"])
            for b in budgets:
                row.append(r["by_budget"][str(b)]["truncated"])
            tbl.add_data(*row)
        run.log({"per_problem": tbl})
        run.finish()
    except Exception as e:  # pragma: no cover
        print(f"[hf-aime] wandb finalize failed: {e}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["eval", "batchcheck"], default="eval")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--label", default="base")
    ap.add_argument("--years", default="2024,2025-I,2025-II")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--derive-budgets", default="6144,8192")
    ap.add_argument("--min-new-tokens", type=int, default=8)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--wandb-name", default="stark/hf-groundtruth-aime-base")
    ap.add_argument("--wandb-group", default="hf-groundtruth-aime-stark")
    ap.add_argument("--no-wandb", action="store_true")
    # batchcheck-only
    ap.add_argument("--batch-n", type=int, default=3)
    ap.add_argument("--batchcheck-tokens", type=int, default=1024)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)  # greedy is deterministic; set for full reproducibility
    if args.mode == "batchcheck":
        return run_batchcheck(args)
    return run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
