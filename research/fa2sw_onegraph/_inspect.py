import json, sys
p = "serve_runs/base/decode_outputs.jsonl"
with open(p) as f:
    lines = f.readlines()
first = json.loads(lines[0])
print("KEYS:", list(first.keys()))
for k, v in first.items():
    if isinstance(v, list):
        print(f"  {k}: list[{len(v)}] head={v[:8]}")
    else:
        s = str(v)
        print(f"  {k}: {s[:90]}")
print("records:", len(lines))
