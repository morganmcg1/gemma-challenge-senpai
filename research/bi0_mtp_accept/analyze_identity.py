import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent


def load(path):
    d = {}
    with open(OUT / path) as f:
        for line in f:
            r = json.loads(line)
            d[str(r["id"])] = {
                "sha": r["completion_token_sha256"],
                "toks": r["completion_token_ids"],
            }
    return d


c32 = load("decode_ctk32.jsonl")
c64 = load("decode_ctk64.jsonl")
c128 = load("decode_ctk128.jsonl")


def mism(ctrl, var):
    return sorted(i for i in ctrl if ctrl[i]["sha"] != var[i]["sha"])


m64 = mism(c32, c64)
m128 = mism(c32, c128)
print("ctk64 mismatched ids (%d): %s" % (len(m64), m64))
print("ctk128 mismatched ids (%d): %s" % (len(m128), m128))
print("\noverlap ctk64 & ctk128:", sorted(set(m64) & set(m128)))
print("ctk64-only:", sorted(set(m64) - set(m128)))
print("ctk128-only:", sorted(set(m128) - set(m64)))

print("\n=== first-divergence index (identical prefix length) ctk64 vs ctk32 ===")
for i in m64:
    a = c32[i]["toks"]
    b = c64[i]["toks"]
    n = min(len(a), len(b))
    fd = next((k for k in range(n) if a[k] != b[k]), n)
    print(f"  {i}: len32={len(a)} len64={len(b)} first_diff_idx={fd}")

print("\n=== first-divergence index ctk128 vs ctk32 ===")
for i in m128:
    a = c32[i]["toks"]
    b = c128[i]["toks"]
    n = min(len(a), len(b))
    fd = next((k for k in range(n) if a[k] != b[k]), n)
    print(f"  {i}: len32={len(a)} len128={len(b)} first_diff_idx={fd}")


def subset(i):
    return i.split("-")[0]


print("\n=== mismatch by eval subset (ctk64) ===", dict(Counter(subset(i) for i in m64)))
print("=== mismatch by eval subset (ctk128) ===", dict(Counter(subset(i) for i in m128)))
print("=== total prompts by subset ===", dict(Counter(subset(i) for i in c32)))
