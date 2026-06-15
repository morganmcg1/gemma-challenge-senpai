#!/usr/bin/env python3
"""Parse [steptime] agg/raw lines from a serve log, compute verify = exec - draft.

Usage: parse_steptime.py <serve_log> [num_spec]
verify_rows = 1 + num_spec. verify_gpu = exec_gpu - draft_gpu (propose is a subset of execute_model).
"""
import re, sys, statistics

def last_agg(path):
    exec_a = draft_a = None
    pat = re.compile(r"\[steptime\] agg n=(\d+) kind=(\w+).*?gpu p50=([\d.]+) p90=([\d.]+) mean=([\d.]+)")
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if not m:
            continue
        n, kind, p50, p90, mean = int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)), float(m.group(5))
        rec = dict(n=n, p50=p50, p90=p90, mean=mean)
        if kind == "exec":
            exec_a = rec
        elif kind == "draft":
            draft_a = rec
    return exec_a, draft_a

def raw_arrays(path):
    ex, dr = {}, {}
    pat = re.compile(r"\[steptime\] raw i=(\d+) kind=(\w+).*?gpu=([\d.]+)")
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if not m:
            continue
        i, kind, gpu = int(m.group(1)), m.group(2), float(m.group(3))
        (ex if kind == "exec" else dr)[i] = gpu
    return ex, dr

def main():
    path = sys.argv[1]
    nspec = int(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = (1 + nspec) if nspec is not None else None
    ea, da = last_agg(path)
    print(f"== {path}  verify_rows={rows} ==")
    if ea and da:
        v50 = ea["p50"] - da["p50"]
        vmean = ea["mean"] - da["mean"]
        print(f"  exec  gpu p50={ea['p50']:.3f} mean={ea['mean']:.3f} (n={ea['n']})")
        print(f"  draft gpu p50={da['p50']:.3f} mean={da['mean']:.3f} (n={da['n']})")
        print(f"  VERIFY gpu p50={v50:.3f} ms  mean={vmean:.3f} ms")
    else:
        print("  (no agg lines found)")
    ex, dr = raw_arrays(path)
    common = sorted(set(ex) & set(dr))
    if common:
        vs = [ex[i] - dr[i] for i in common]
        print(f"  raw verify (n={len(vs)}): p50={statistics.median(vs):.3f} "
              f"min={min(vs):.3f} max={max(vs):.3f} mean={statistics.fmean(vs):.3f}")
    return v50 if (ea and da) else None

if __name__ == "__main__":
    main()
