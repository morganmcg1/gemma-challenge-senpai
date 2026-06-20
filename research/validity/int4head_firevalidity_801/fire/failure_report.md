STUDENT land:
SENPAI-RESULT: {"terminal":false,"status":"blocked","pending_arms":true,"wandb_run_ids":["ftds7gll"],"primary_metric":{"name":"local_official_gate_ppl","value":2.00256},"test_metric":{"name":"local_official_gate_completed","value":128}}

## ⛔ FIRE FAILED — HF job errored at server startup (private-repo 401). NOT an int4head metric/quality failure. Re-fire is BLOCKED on a model-hosting fix.

**The two numbers in the marker are the on-pod local gate (PPL 2.00256 / 128-of-128), NOT leaderboard results.** The official benchmark produced **no** `summary.json` / `ppl_summary.json` because the vLLM server never reached readiness.

### What happened
The one human-approved int4head HF submission launched at **18:03:10Z** (job `6a36d5de3093dba73ce2b016`) and **errored at 18:09:52Z**, before serving a single prompt. Authoritative `job_status.json`:
```
"status": "error", "stage": "ERROR",
"message": "job ended in error: Job failed with exit code: 1. Reason: Error"
```
`job_logs.txt` root cause — the runner's vLLM could not download the model:
```
huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error.
Repository Not Found for url:
  https://huggingface.co/gemma-challenge/gemma-4-e4b-it-int4-mtp-bi0-int4head/resolve/main/config.json
... If you are trying to access a private or gated repo, make sure you are authenticated ...
OSError: gemma-challenge/...-int4head is not a valid model identifier
RuntimeError: server exited before readiness with code 1
```

### Root cause (diagnosed, not assumed): private-repo auth on the runner
- The model repo `gemma-challenge/gemma-4-e4b-it-int4-mtp-bi0-int4head` is **fine**: `private=True`, `sha=ad42984`, all 8 files present (`config.json`/tokenizer/`model.safetensors` 10.55 GB). From **this pod** it loads cleanly — `whoami=morgan` has read access (so the pre-fire local Hub serve-verify steps passed).
- The HF Job runs **as `morgan` too** (`job_status.hf_user=morgan`) — yet the model download came back **401 / unauthenticated**. So the failure is **not** "morgan lacks access"; it's that the **runner's serve subprocess has no HF token in its env** for the model pull. `serve.py` launches `vllm ... --model $MODEL_ID` via `execvpe(os.environ)` and **passes no explicit token** (line 67 / 147) — it relies on an ambient `HF_TOKEN`, which the sandboxed HF Job does not inject into the submission process.
- **The contrast that proves it:** shipped bi0 (`int4_mtp_bi0_surgattn`) uses a **PUBLIC** `model_id=google/gemma-4-E4B-it-qat-w4a16-ct` → no token needed → it fired fine (#769). int4head is the **first** submission to point `model_id`/`MODEL_ID` at a **private** Hub repo, so it's the first to hit this. Every local pre-fire gate ran in a token-present environment and was therefore **blind to the token-less runner context** — a real local↔runner gap, not a flake.

### Why I did NOT re-fire
The 401 is a **deterministic auth failure**, not a transient blip — re-firing the identical (private-repo, token-less-runner) setup will 401 again and burn one of the **2 remaining agent fires** (`quota.agent_remaining=2` at launch). Per the card's own "report back instead of retrying speculatively" and the human's "fire **when ready**", the honest read is: int4head is technically fire-clear (your accepted disposition stands) but it is **not hosting-ready** until the runner can load the weights. So I stopped and escalated instead of blind-retrying into a known 401.

### Recommended fix (your call — it's the #802 private repo + a competition-visibility decision)
One of:
1. **Make `gemma-challenge/gemma-4-e4b-it-int4-mtp-bi0-int4head` PUBLIC** (cleanest; matches how the challenge serves the public google checkpoints — zero token needed on the runner). Visibility of a competition checkpoint is your/Morgan's strategic call, so I did **not** flip it.
2. **OR** have the benchmark runner expose an `HF_TOKEN` with read scope to the serve subprocess (harness/infra change). I will **not** put a token in the committed manifest (secret-in-repo).

I recommend **#1** unless the harness already has a private-model token path I'm not seeing.

### Also: the re-sync landmine recurred — fixed durably this time
On this invocation my working-tree `manifest.json` had **again reverted** to the stale local path (`model_id=/workspace/gemma_build/bi0_int4head_g32`, no `env.MODEL_ID`) — the 18:09Z merge (`ad392c7`) was discarded by a pod re-checkout (it's in no ref). **The fire itself used the correct Hub-pointed manifest** (the runner tried the Hub repo — that's how we got the 401), so this did not cause the failure. But to stop it biting the re-fire, I restored the Hub-pointed manifest from advisor HEAD and **committed + pushed it** (`997a16d`) so a re-checkout can't discard it again. Verified: `model_id` AND `env.MODEL_ID` = the private repo; the rest of the submission dir is byte-identical to advisor HEAD (empty diff).

### Status / quota / staging
- Job `6a36d5de3093dba73ce2b016` — **ERROR** (no `summary.json` / `ppl_summary.json`). run_prefix `results/senpai/int4-mtp-bi0-int4head-20260620T180305Z`. Launch W&B `ftds7gll`.
- Agent fire quota remaining: **2** (this consumed 1). GPU clean (0 MiB, 0% util — no leak).
- Build `/workspace/gemma_build/bi0_int4head_g32` and serve config remain staged; manifest re-Hub-pointed and pushed → re-fire is **instant** once the repo is runner-loadable.
- int4head technical disposition unchanged (local: PPL 2.00256, 128/128, all-4-modalities; your accepted greedy-noise-floor + private-gap dispositions stand). **The only blocker is model hosting/auth on the runner.**

**Awaiting your call on the hosting fix (public repo vs runner token). On your go — after the repo is runner-loadable — I re-fire immediately (new launch, so it needs your explicit go).**
