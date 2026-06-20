"""Compute the exact input-token length distribution over the sampled 128-prompt
ShareGPT decode set, mirroring decode_outputs.read_sharegpt_prompts (seed=1) +
encode_prompt (chat template). Sets the true MAX_MODEL_LEN floor = max(input)+output_len.
CPU-only (tokenizer), no GPU. Usage: promptlen.py [num_prompts] [output_len]
"""
import json, random, sys
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path("/workspace/senpai/target")
DATASET = ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json"
MODEL = "google/gemma-4-E4B-it"  # DEFAULT_TOKENIZER the benchmark actually uses
NUM_PROMPTS = int(sys.argv[1]) if len(sys.argv) > 1 else 128
OUTPUT_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 512
SEED = 1

def read_sharegpt_prompts(path, num_prompts, seed):
    data = json.loads(path.read_text())
    records = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        conv = item.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        first = conv[0]
        if not isinstance(first, dict):
            continue
        prompt = first.get("value")
        if not isinstance(prompt, str) or not prompt:
            continue
        records.append(prompt)
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:num_prompts]

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
prompts = read_sharegpt_prompts(DATASET, NUM_PROMPTS, SEED)
lens = []
for p in prompts:
    enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True, tokenize=True)
    # apply_chat_template(tokenize=True) returns a BatchEncoding; the real harness
    # extracts input_ids via normalize_token_ids. Mirror that.
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    lens.append(len(ids))
lens.sort()
n = len(lens)
mx = lens[-1]
print(f"n_prompts={n} (requested {NUM_PROMPTS})")
print(f"input tokens: min={lens[0]} median={lens[n//2]} p90={lens[int(n*0.9)]} max={mx}")
print(f"top10 input lengths: {lens[-10:]}")
print(f"output_len={OUTPUT_LEN}")
print(f"=> longest total (input+output) = {mx} + {OUTPUT_LEN} = {mx + OUTPUT_LEN}")
print(f"=> MIN MAX_MODEL_LEN that truncates nothing = {mx + OUTPUT_LEN}")
print(f"=> control MAX_MODEL_LEN=4096 headroom over floor = {4096 - (mx + OUTPUT_LEN)} tokens")
# how many prompts would fail at each candidate max_model_len
for cap in (1024, 2048, 3072, 4096):
    n_fail = sum(1 for L in lens if L + OUTPUT_LEN > cap)
    print(f"   MAX_MODEL_LEN={cap}: {n_fail}/{n} prompts exceed context (input+512)")
