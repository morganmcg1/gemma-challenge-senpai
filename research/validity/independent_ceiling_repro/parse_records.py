import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "run/records.jsonl"
rows = [json.loads(l) for l in open(path)]

hdr = ["name", "kind", "K", "wall_tps", "reps", "spread%", "served",
       "attn", "eager", "cudagraph", "spec", "flash_off", "ppl"]
print("{:13} {:17} {:>4} {:>9} {:>16} {:>8} {:>6} {:>12} {:>6} {:>20} {:>5} {:>9} {}".format(*hdr))
for r in rows:
    reps = r.get("rep_wall_tps") or []
    repstr = "/".join("{:.2f}".format(x) for x in reps) if reps else "-"
    wt = r.get("wall_tps")
    wt = wt if wt is not None else float("nan")
    print("{:13} {:17} {:>4} {:>9.2f} {:>16} {:>8.3f} {:>6} {:>12} {:>6} {:>20} {:>5} {:>9} {}".format(
        r["name"], r["kind"], str(r.get("K")), wt, repstr,
        r.get("wall_tps_spread_pct", 0.0), str(r.get("served_ok")),
        str(r.get("attn_backend")), str(r.get("enforce_eager")),
        str(r.get("cudagraph_mode")), str(r.get("spec_method")),
        str(r.get("flashinfer_sampler_disabled")), r.get("ppl")))

print("\n=== resolved_env per arm ===")
for r in rows:
    e = r.get("resolved_env", {})
    print("{:13} FLASH_SAMPLER={} ATTN={} ENFORCE_EAGER={} NUM_SPEC={} BATCH_INV={} MAX_SEQS={} MODEL={}".format(
        r["name"], e.get("VLLM_USE_FLASHINFER_SAMPLER"), e.get("VLLM_ATTENTION_BACKEND"),
        e.get("ENFORCE_EAGER"), e.get("NUM_SPECULATIVE_TOKENS"), e.get("VLLM_BATCH_INVARIANT"),
        e.get("MAX_NUM_SEQS"), e.get("MODEL_ID") or r.get("model_id")))

print("\n=== extra fields (fire/break/identity if present) ===")
for r in rows:
    keys = ["spec_fire_rate", "break_rate", "fire_rate", "n_break", "greedy_identical",
            "spec_num_tokens", "flashinfer_sampler_crash", "attn_patch_failed"]
    present = {k: r.get(k) for k in keys if k in r}
    print(r["name"], present)
