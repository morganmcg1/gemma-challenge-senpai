#!/usr/bin/env python3
"""PR #590 -- MMLU-Pro max_tokens=2048 truncation DE-BIAS via same-subset splice.

Diagnosis: under the lewtun #31 sampling protocol (T=1.0), ~12% of MMLU-Pro CoT
completions hit stop_reason=max_tokens at the 2048 cap and score WRONG only because
the `ANSWER: X` line is cut off (244/245 wrong on seed-1). Binding limit is the OUTPUT
cap (input tokens max=1604, well under max_model_len), so the fix is more output budget.

Clean controlled A/B (only the output budget changes; same N=2000 subset, same prompts,
same sampling seed):
  * non-truncated samples (stop_reason=stop) are output-budget-independent -> KEEP as-is;
  * truncated samples are re-run at --max-tokens 4096 against the 6144-model-len server
    (1604+4096=5700<6144) -> take the adequate-budget score (recovers, or stays wrong if
    genuinely needs >4096).
The spliced per-sample matrix feeds aggregate_ci.py unchanged.

Per seed:
  base   = runs/mmlu_base_fullhead_n{N}_s{S}.json     (the 2048 run, has per_sample)
  eval   = logs/mmlu_n{N}_s{S}/*.eval                 (stop_reason per sample)
  redo   = runs/_redo{MT}_s{S}.json                   (re-run of truncated ids at MT toks)
  out    = runs/mmlu_debias_n{N}_s{S}.json            (spliced, full N samples)
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys
from inspect_ai.log import read_eval_log

TRUNC = {"max_tokens", "length", "model_length"}
PYI = "/tmp/eval-serve-venv/bin/python"
RUN_EVAL = "research/validity/downstream_quality_eval/run_eval.py"


def find_eval(log_dir: str, n: int) -> str:
    """The N=2000 .eval (largest sample count), not the smoke .eval."""
    cands = sorted(glob.glob(os.path.join(log_dir, "*.eval")))
    if not cands:
        raise SystemExit(f"[debias] no .eval in {log_dir}")
    best, best_n = None, -1
    for c in cands:
        try:
            log = read_eval_log(c)
            ns = len(log.samples or [])
        except Exception:
            ns = -1
        if ns > best_n:
            best, best_n = c, ns
    if best_n < n:
        print(f"[debias] WARN {log_dir}: best .eval has {best_n} samples (< n={n})")
    return best


def trunc_ids_and_stops(eval_path: str):
    """{id: stop_reason} for samples that hit a truncation stop_reason."""
    log = read_eval_log(eval_path)
    trunc, all_stops = {}, {}
    for s in log.samples or []:
        sr = None
        if s.output and s.output.choices:
            sr = s.output.choices[0].stop_reason
        all_stops[str(s.id)] = sr
        if sr in TRUNC:
            trunc[str(s.id)] = sr
    return trunc, all_stops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--subset-seed", type=int, default=12345)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--runs", default="research/validity/quality_gates_ci/runs")
    ap.add_argument("--logs", default="research/validity/quality_gates_ci/logs")
    args = ap.parse_args()

    print(f"[debias] seeds={args.seeds} N={args.n} redo_max_tokens={args.max_tokens}")
    for s in args.seeds:
        base_path = f"{args.runs}/mmlu_base_fullhead_n{args.n}_s{s}.json"
        if not os.path.exists(base_path):
            print(f"[debias] seed {s}: base run MISSING ({base_path}) -- SKIP"); continue
        base = json.load(open(base_path))
        log_dir = f"{args.logs}/mmlu_n{args.n}_s{s}"
        eval_path = find_eval(log_dir, args.n)
        trunc, _ = trunc_ids_and_stops(eval_path)
        ids = sorted(trunc.keys())
        base_correct_trunc = sum(1 for r in base["per_sample"] if str(r["id"]) in trunc and r["correct"])
        print(f"\n[debias] seed {s}: base_acc={base['accuracy']:.4f} n_trunc={len(ids)} "
              f"({100*len(ids)/len(base['per_sample']):.2f}%) of which base-correct={base_correct_trunc}")
        if not ids:
            print(f"[debias] seed {s}: no truncated ids -> debiased == base");
            json.dump(base, open(f"{args.runs}/mmlu_debias_n{args.n}_s{s}.json", "w"), indent=2); continue

        ids_file = f"{args.runs}/_trunc_ids_s{s}.json"
        json.dump(ids, open(ids_file, "w"))
        redo_out = f"{args.runs}/_redo{args.max_tokens}_s{s}.json"
        redo_log = f"{args.logs}/redo{args.max_tokens}_s{s}"

        need_redo = True
        if os.path.exists(redo_out):
            try:
                rj = json.load(open(redo_out))
                if {str(r["id"]) for r in rj["per_sample"]} >= set(ids):
                    print(f"[debias] seed {s}: redo exists & covers all ids -> reuse"); need_redo = False
            except Exception:
                need_redo = True
        if need_redo:
            cmd = [PYI, RUN_EVAL, "--task", "mmlu_pro", "--arm", "base_fullhead",
                   "--n", str(args.n), "--seed", str(args.subset_seed),
                   "--sampling-seed", str(s), "--temperature", "1.0", "--top-p", "0.95",
                   "--top-k", "64", "--min-tokens", "8", "--max-tokens", str(args.max_tokens),
                   "--max-connections", "32", "--base-url", args.base_url,
                   "--ids-file", ids_file, "--out", redo_out, "--log-dir", redo_log]
            print(f"[debias] seed {s}: re-running {len(ids)} truncated ids @ max_tokens={args.max_tokens} ...", flush=True)
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"[debias] seed {s}: redo FAILED rc={rc} -- SKIP seed"); continue

        redo = json.load(open(redo_out))
        redo_map = {str(r["id"]): r for r in redo["per_sample"]}
        # count how many redo samples STILL truncate at the larger budget
        still = {}
        try:
            re_eval = find_eval(redo_log, len(ids))
            _, _ = trunc_ids_and_stops(re_eval)  # ensure readable
            still, _ = trunc_ids_and_stops(re_eval)
        except Exception as e:
            print(f"[debias] seed {s}: (could not read redo .eval for still-trunc count: {e})")

        spliced, recovered, missing, sha_mismatch = [], 0, 0, 0
        for r in base["per_sample"]:
            sid = str(r["id"])
            if sid in trunc:
                r2 = redo_map.get(sid)
                if r2 is None:
                    missing += 1; spliced.append(r); continue
                if r.get("prompt_sha") and r2.get("prompt_sha") and r["prompt_sha"] != r2["prompt_sha"]:
                    sha_mismatch += 1
                if (not r["correct"]) and r2["correct"]:
                    recovered += 1
                rr = dict(r2); rr["spliced_from_redo"] = True
                spliced.append(rr)
            else:
                spliced.append(r)
        n_correct = sum(1 for r in spliced if r["correct"])
        acc = n_correct / len(spliced)
        out = dict(base)
        out["per_sample"] = sorted(spliced, key=lambda r: str(r["id"]))
        out["accuracy"] = acc
        out["n_correct"] = n_correct
        out["debias"] = {"redo_max_tokens": args.max_tokens, "n_trunc": len(ids),
                         "n_recovered": recovered, "still_trunc_at_redo": len(still),
                         "missing_in_redo": missing, "prompt_sha_mismatch": sha_mismatch,
                         "base_accuracy": base["accuracy"]}
        out_path = f"{args.runs}/mmlu_debias_n{args.n}_s{s}.json"
        json.dump(out, open(out_path, "w"), indent=2)
        if sha_mismatch:
            print(f"[debias] seed {s}: !! prompt_sha mismatch on {sha_mismatch} ids (subset construction differs!)")
        print(f"[debias] seed {s}: recovered={recovered}/{len(ids)} still_trunc@{args.max_tokens}={len(still)} "
              f"missing={missing} -> debiased_acc={acc:.4f} (base {base['accuracy']:.4f}, "
              f"+{acc-base['accuracy']:+.4f}) -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
