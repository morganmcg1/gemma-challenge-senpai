import json

rows = [json.loads(l) for l in open("run/records.jsonl")]
print("=== spec-decode acceptance metrics per arm ===")
for r in rows:
    m = r.get("metrics", {}) or {}
    if r["kind"] not in ("spec", "spec_ar_anchor"):
        continue
    acc = m.get("vllm:spec_decode_num_accepted_tokens_total") or m.get("vllm:spec_decode_num_accepted_tokens")
    drf = m.get("vllm:spec_decode_num_draft_tokens_total") or m.get("vllm:spec_decode_num_draft_tokens")
    emitted = m.get("vllm:spec_decode_num_emitted_tokens_total") or m.get("vllm:spec_decode_num_emitted_tokens")
    rate = (acc / drf) if (acc and drf) else None
    name = r["name"]
    K = r.get("K")
    # mean accepted tokens per draft step ~= acc/(num draft steps); draft tokens = K * num_steps
    accept_len = None
    if acc is not None and drf and K:
        steps = drf / K
        accept_len = (acc + steps) / steps  # accepted draft + the always-emitted target token
    print("%-12s K=%s acc=%s draft=%s emitted=%s accept_rate=%s mean_accept_len(approx)=%s"
          % (name, K, acc, drf, emitted, round(rate, 4) if rate else None,
             round(accept_len, 4) if accept_len else None))
    spec_keys = {k: v for k, v in m.items() if "spec_decode" in k or "accepted" in k or "draft" in k}
    print("             spec_keys=%s" % spec_keys)
