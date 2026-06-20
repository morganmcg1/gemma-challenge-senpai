"""Pin the owner of the #809 17% bf16 aten::mm (114.75ms x59) / ampere_bf16
GEMM (111.83ms x40) by correlating the saved decode_window trace: kernel ->
launching cpu_op (via External id) -> Input Dims (M,K,N) + Call stack.

The trace is the STEADY-STATE decode window captured AFTER the one-time
centroid-graph capture, so whatever launches the ampere_bf16 GEMM is a per-step
decode cost. Reads the existing gz trace; no GPU needed."""
from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "research" / "cudagraph_sampling_capture" / "decode_window.pt.trace.json.gz"
OUT = ROOT / "research" / "bf16_gemm_attribution" / "trace_owner.json"


def main() -> int:
    with gzip.open(TRACE, "rt") as f:
        data = json.load(f)
    ev = data.get("traceEvents", data if isinstance(data, list) else [])
    print(f"[tr] {len(ev)} trace events", flush=True)

    # index by External id
    by_extid_cpu: dict = collections.defaultdict(list)
    kernels: list = []
    cpu_ops: list = []
    for e in ev:
        cat = e.get("cat", "")
        args = e.get("args", {}) or {}
        extid = args.get("External id", args.get("external id"))
        if cat in ("cpu_op", "user_annotation"):
            cpu_ops.append(e)
            if extid is not None:
                by_extid_cpu[extid].append(e)
        elif cat == "kernel":
            kernels.append(e)

    print(f"[tr] {len(kernels)} kernel events, {len(cpu_ops)} cpu_op events", flush=True)

    def kdur(e):
        return float(e.get("dur", 0.0))

    # 1) ampere_bf16 GEMM kernels -> launching cpu_op + shapes + stack
    amp = [k for k in kernels if "ampere_bf16_s16816gemm" in k.get("name", "")]
    amp_tot = sum(kdur(k) for k in amp)
    print(f"\n[tr] === ampere_bf16_s16816gemm kernels: {len(amp)}, total {amp_tot/1e3:.2f} ms ===", flush=True)
    owner = collections.Counter()
    shapes_seen = collections.Counter()
    stack_seen = collections.Counter()
    grid_seen = collections.Counter()
    for k in amp:
        a = k.get("args", {}) or {}
        extid = a.get("External id", a.get("external id"))
        grid = tuple(a.get("grid", []))
        grid_seen[grid] += 1
        ops = by_extid_cpu.get(extid, [])
        # choose the most specific (deepest / aten::) op for this extid
        names = [o.get("name", "") for o in ops]
        owner[tuple(names)] += 1
        for o in ops:
            dims = (o.get("args", {}) or {}).get("Input Dims")
            if dims:
                shapes_seen[(o.get("name", ""), json.dumps(dims))] += 1
            st = (o.get("args", {}) or {}).get("Call stack")
            if st:
                # keep last few frames that aren't torch internals
                frames = [x for x in st.split(";") if "site-packages/torch/" not in x]
                stack_seen[" || ".join(frames[-6:])] += 1

    print("  -- launching cpu_op name-chains (by extid) --", flush=True)
    for names, c in owner.most_common(12):
        print(f"     x{c:4d}  {names}", flush=True)
    print("  -- grid dims --", flush=True)
    for g, c in grid_seen.most_common(8):
        print(f"     x{c:4d}  grid={g}", flush=True)
    print("  -- Input Dims of launching ops --", flush=True)
    for (nm, dims), c in shapes_seen.most_common(12):
        print(f"     x{c:4d}  {nm}  {dims}", flush=True)
    print("  -- Call stacks (last frames) --", flush=True)
    for st, c in stack_seen.most_common(8):
        print(f"     x{c:4d}  {st}", flush=True)

    # 2) aten::mm cpu_ops: dims + the kernels they own
    mm_ops = [o for o in cpu_ops if o.get("name") == "aten::mm"]
    print(f"\n[tr] === aten::mm cpu_ops: {len(mm_ops)} ===", flush=True)
    mm_dims = collections.Counter()
    mm_stack = collections.Counter()
    for o in mm_ops:
        a = o.get("args", {}) or {}
        dims = a.get("Input Dims")
        if dims:
            mm_dims[json.dumps(dims)] += 1
        st = a.get("Call stack")
        if st:
            frames = [x for x in st.split(";") if "site-packages/torch/" not in x]
            mm_stack[" || ".join(frames[-6:])] += 1
    print("  -- aten::mm Input Dims --", flush=True)
    for dims, c in mm_dims.most_common(12):
        print(f"     x{c:4d}  {dims}", flush=True)
    print("  -- aten::mm Call stacks --", flush=True)
    for st, c in mm_stack.most_common(8):
        print(f"     x{c:4d}  {st}", flush=True)
    if not mm_stack:
        print("     (no Call stack args — trace was captured without with_stack)", flush=True)

    # 3) any cpu_op referencing einsum / scatter / logits / centroid / get_top
    print("\n[tr] === cpu_ops of interest (einsum/scatter/bmm/index/logits) ===", flush=True)
    interest = collections.Counter()
    for o in cpu_ops:
        nm = o.get("name", "")
        if any(t in nm for t in ("einsum", "scatter", "bmm", "index_select", "index", "logits", "gather", "topk")):
            interest[nm] += 1
    for nm, c in interest.most_common(20):
        print(f"     x{c:4d}  {nm}", flush=True)

    OUT.write_text(json.dumps({
        "ampere_bf16_count": len(amp), "ampere_bf16_total_ms": amp_tot / 1e3,
        "owner_name_chains": {str(k): v for k, v in owner.most_common(20)},
        "grid_dims": {str(k): v for k, v in grid_seen.most_common(20)},
        "launch_op_dims": {f"{nm} {d}": v for (nm, d), v in shapes_seen.most_common(20)},
        "aten_mm_count": len(mm_ops),
        "aten_mm_dims": dict(mm_dims.most_common(20)),
        "interest_ops": dict(interest.most_common(40)),
    }, indent=2, default=str))
    print(f"\n[tr] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
