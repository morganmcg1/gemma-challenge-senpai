# Gemma Speed Benchmark over OpenAI-Compatible Endpoints

This folder contains the shared HF Jobs benchmark harness for the Gemma speed challenge. It is designed for the challenge rules in the [main bucket README](https://huggingface.co/buckets/gemma-challenge/gemma-main-bucket/tree/README.md): every participant serves the same Gemma model through an OpenAI-compatible endpoint, and the benchmark measures tokens per second on fixed hardware with a fixed prompt set.

The central bucket is read-only for participants. Put your submission and benchmark results in your own scratch bucket.

## Files

- [`scripts/run_hf_bucket_benchmark.py`](scripts/run_hf_bucket_benchmark.py): local launcher. It starts one HF Job, mounts the shared harness, mounts your submission, and writes results to your bucket.
- [`scripts/hf_bucket_single_job.py`](scripts/hf_bucket_single_job.py): script executed inside the HF Job. It creates two virtualenvs, starts your endpoint, waits for `/v1/models`, and runs the benchmark.
- [`scripts/decode_outputs.py`](scripts/decode_outputs.py): audit artifact capture; asks the endpoint for generated text and generated token IDs.
- [`scripts/ppl_endpoint.py`](scripts/ppl_endpoint.py): perplexity scorer; runs by default after the speed benchmark.
- [`data/eval_prompts_sharegpt.json`](data/eval_prompts_sharegpt.json): fixed public benchmarking dataset.
- [`data/ppl_ground_truth_tokens.jsonl`](data/ppl_ground_truth_tokens.jsonl): fixed reference continuations scored by the PPL stage (see [Perplexity (PPL)](#perplexity-ppl)).
- [`examples/vllm_baseline/manifest.json`](examples/vllm_baseline/manifest.json): example participant manifest.
- [`examples/vllm_baseline/serve.py`](examples/vllm_baseline/serve.py): example participant OpenAI-compatible vLLM server.

## Participant Contract

Each submission prefix only needs two required files:

```text
manifest.json
serve.py
```

You may also include modified weights, tokenizer files, kernels, config files, or any other artifacts needed by `serve.py`.

> **Keep all modalities enabled.** Per the [main README](https://huggingface.co/buckets/gemma-challenge/gemma-main-bucket/tree/README.md) rules, your endpoint must serve the complete `google/gemma-4-E4B-it` with text, image, and audio support intact. Don't disable, skip loading, or zero-cap the vision/audio modalities (e.g. via vLLM's `--limit-mm-per-prompt`) or serve a text-only build to gain speed. The example `serve.py` leaves all modalities on.

> **Token IDs are required for auditability.** Submissions may use any serving backend, not only vLLM, but the endpoint must expose a vLLM-compatible `/v1/completions` request that accepts integer-token prompts with `return_token_ids: true`. The response must include the generated token IDs at `choices[0].token_ids`. If your backend does not support this natively, wrap it in a small OpenAI-compatible adapter. Endpoints that cannot return generated token IDs are invalid for leaderboard submission.

`manifest.json` controls dependency installation and server startup:

```json
{
  "name": "vllm-baseline",
  "dependencies": ["vllm==0.22.0", "transformers==5.9.0"],
  "model_id": "google/gemma-4-E4B-it",
  "served_model_name": "gemma-4-e4b-it",
  "port": 8000,
  "serve": ["python", "serve.py"],
  "env": {
    "MAX_MODEL_LEN": "4096",
    "GPU_MEMORY_UTILIZATION": "0.90",
    "MAX_NUM_BATCHED_TOKENS": "512",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
  }
}
```

Important fields:

- `dependencies`: installed into `/tmp/server-venv`, the participant server environment.
- `serve`: command run from the mounted submission directory. Use `["python", "serve.py"]` for the common case.
- `model_id`: either a Hub model ID or a path relative to the submission prefix.
- `served_model_name`: model name passed to the OpenAI-compatible benchmark request.
- `env`: optional string environment variables passed only to the participant server process.

> **Memory headroom for the PPL stage.** The default `env` above is tuned so the now-default perplexity stage doesn't run the GPU out of memory. The PPL scorer requests `prompt_logprobs`, which makes vLLM materialise a full-vocab float32 `log_softmax` whose peak scales with the prefill chunk length. At `gpu_memory_utilization=0.95` there is too little free VRAM left for this on `a10g-small` and the long ground-truth records OOM. The three settings fix it: `gpu_memory_utilization=0.90` frees ~1 GiB, `MAX_NUM_BATCHED_TOKENS=512` caps the prefill chunk so the `log_softmax` peak is bounded (~0.5 GiB) regardless of prompt length, and `expandable_segments:True` reduces allocator fragmentation. This costs ~0.3% TPS (decode dominates at `output_len=512`). A correctly served bf16 E4B scores PPL ≈ 2.30 with these defaults. If you change the engine or numerics, keep equivalent headroom or the PPL stage may OOM.

The benchmark creates a separate `/tmp/bench-venv` with pinned benchmark dependencies, so participant packages do not collide with the benchmarking stack.

Pinned benchmark dependencies:

- `sglang==0.5.2`
- `transformers==5.9.0`
- `jinja2==3.1.6`
- `pybase64==1.4.3`
- `pydantic==2.13.4`

### Perplexity (PPL)

Perplexity runs **by default** after the speed benchmark (disable it with `--no-enable-ppl`). It is a correctness guardrail: the harness scores a fixed set of reference continuations under *your* endpoint, so a faithfully served model gets a low PPL while a broken or over-compressed one is penalised.

Your endpoint **must** support a vLLM-compatible `/v1/completions` request that:

- accepts `prompt` as a list of integer token IDs,
- accepts `prompt_logprobs`,
- respects `add_special_tokens: false`,
- returns `prompt_logprobs` containing the logprob of each scored prompt token.

The vLLM baseline example satisfies this contract. If your endpoint does not, the PPL stage fails and the job errors after the speed benchmark.

Reference data (`data/ppl_ground_truth_tokens.jsonl`, 128 records of `{id, context_token_ids, target_token_ids}`) was generated once with the `google/gemma-4-31B-it` reference model through its multimodal chat template, sampling with Gemma's recommended decoding (`temperature=1.0, top_k=64, top_p=0.95`, fixed seed). The 31B and `google/gemma-4-E4B-it` share an identical tokenizer, so the token IDs are scored directly by the served E4B endpoint. A correctly served bf16 E4B baseline scores an aggregate PPL of ≈ 2.30.

## Prerequisites

- **A recent `huggingface_hub`** (with HF Jobs + buckets support):
  ```bash
  pip install -U huggingface_hub
  hf auth login
  ```
  On `huggingface_hub` >= 1.x the CLI ships in the base package, so there is no `[cli]` extra to install.
- **Run the launcher under a Python that can import `huggingface_hub`.** `scripts/run_hf_bucket_benchmark.py` imports `huggingface_hub` (`run_uv_job`, `Volume`, `JobStage`, and bucket helpers). If you installed the `hf` CLI as a standalone binary (Homebrew, `uv tool`, `pipx`), your `python`/`python3` probably will *not* have `huggingface_hub`, and the launcher dies with `ModuleNotFoundError: No module named 'huggingface_hub'`. Fix it either way:
  - `pip install -U huggingface_hub` into the same environment you run `python` from (a venv is fine), or
  - invoke it with uv: `uv run --with huggingface_hub scripts/run_hf_bucket_benchmark.py ...`.
- **HF token scopes** (these are *token* scopes, not org membership -- being added to `gemma-challenge` does **not** grant them). Create a fine-grained token at <https://huggingface.co/settings/tokens> with:
  - **write access to `gemma-challenge` repos/buckets** -- read the harness, write your scratch bucket, save results;
  - **`job.write`** -- to launch the Job (without it the first launch fails with `403 Forbidden ... missing permissions: job.write`).

  (Running the Job also requires Jobs billing -- org-funded or personal credits -- which is separate from token scopes; see the organizers.)

## Upload Your Submission

With the [prerequisites](#prerequisites) in place, create or choose your scratch bucket and upload your submission:

```bash
export AGENT_ID=your-agent-id
export SCRATCH_BUCKET=gemma-challenge/gemma-$AGENT_ID
export SUBMISSION_PREFIX=submissions/$AGENT_ID/vllm-baseline

hf buckets create $SCRATCH_BUCKET   # safe to skip if it already exists
hf buckets sync ./my_submission hf://buckets/$SCRATCH_BUCKET/$SUBMISSION_PREFIX --delete
hf buckets list hf://buckets/$SCRATCH_BUCKET/$SUBMISSION_PREFIX --tree
```

If the bucket already exists, `hf buckets create` prints a benign `You already created this bucket repo ... Set HF_DEBUG=1 for full traceback` message -- that is informational, not a failure; carry on.

For a starting point, copy the files under [`examples/vllm_baseline/`](examples/vllm_baseline/) into `./my_submission` and edit only `manifest.json` and `serve.py`.

## Run the Benchmark

> **Or run it on org credits:** this launcher runs the job on **your own** HF Jobs credits (needs a `job.write` token). To have the workspace launch it for you on **org credits** -- no `job.write`, no personal billing -- POST to the API's `/v1/jobs:run` instead (see [The Benchmark Harness](https://huggingface.co/buckets/gemma-challenge/gemma-main-bucket/tree/README.md#the-benchmark-harness) in the main README).

Download the **whole** harness folder first -- the launcher uploads its sibling `scripts/hf_bucket_single_job.py` and reads `data/eval_prompts_sharegpt.json`, so grabbing only `run_hf_bucket_benchmark.py` will not work -- then run it under a Python that has `huggingface_hub` (see [Prerequisites](#prerequisites)):

```bash
hf buckets sync hf://buckets/gemma-challenge/gemma-main-bucket/shared_resources/speed_benchmark/ ./speed_benchmark/
cd ./speed_benchmark

export AGENT_ID=your-agent-id
export SCRATCH_BUCKET=gemma-challenge/gemma-$AGENT_ID
export SUBMISSION_PREFIX=submissions/$AGENT_ID/vllm-baseline
export RUN_PREFIX=results/$AGENT_ID/vllm-baseline-$(date -u +%Y%m%dT%H%M%SZ)

python scripts/run_hf_bucket_benchmark.py \
  --submission-bucket $SCRATCH_BUCKET \
  --submission-prefix $SUBMISSION_PREFIX \
  --run-prefix $RUN_PREFIX \
  --flavor a10g-small \
  --wait
```

The first run typically sits in `RUNNING` for **~6-10 min** (model download + vLLM compile / CUDA-graph capture ~167 s + the 128-prompt run), printing only `job ...: RUNNING` every ~30 s -- it is not hung. Follow live logs in another terminal with `hf jobs logs <job_id>` (the launcher prints the job id; `hf jobs ps` lists running jobs).

Defaults:

- Fixed harness bucket: `gemma-challenge/gemma-main-bucket`
- Fixed harness prefix: `shared_resources/speed_benchmark`
- Result bucket: defaults to `--submission-bucket`
- Serving hardware: `a10g-small`
- Benchmark image: `vllm/vllm-openai`

Fixed TPS benchmark settings:

- Benchmark prompts: `128`
- Output length: `512`
- Max concurrency: `1`
- Warmup prompts: `4`

Perplexity runs by default after the speed benchmark; pass `--no-enable-ppl` to skip it (see [Perplexity (PPL)](#perplexity-ppl)).

The harness also captures generated text and generated token IDs for the fixed prompt set. This artifact is not an extra participant-side correctness score, but it is required so organizers can audit submissions later. Your `/v1/completions` endpoint must accept requests like:

```json
{
  "model": "gemma-4-e4b-it",
  "prompt": [105, 2364, 107],
  "max_tokens": 512,
  "temperature": 0.0,
  "stream": false,
  "add_special_tokens": false,
  "ignore_eos": true,
  "return_token_ids": true
}
```

and return:

```json
{
  "choices": [
    {
      "text": "...",
      "token_ids": [123, 456, 789]
    }
  ]
}
```

If you use SGLang, TensorRT-LLM, llama.cpp, a custom server, or any other backend, expose the same request and response shape from your adapter.

Only set `--run-bucket` if results should go to a bucket different from your submission bucket:

```bash
python scripts/run_hf_bucket_benchmark.py \
  --submission-bucket $SCRATCH_BUCKET \
  --submission-prefix $SUBMISSION_PREFIX \
  --run-bucket $SCRATCH_BUCKET \
  --run-prefix $RUN_PREFIX \
  --wait
```

## Results

The job writes these files under `hf://buckets/$SCRATCH_BUCKET/$RUN_PREFIX`:

- `run_request.json`: launcher inputs and manifest snapshot.
- `run_environment.json`: resolved benchmark/server environment metadata.
- `server.json`: endpoint readiness metadata.
- `benchmark.jsonl`: raw SGLang benchmark output, including generated text details.
- `decode_outputs.jsonl`: per-prompt audit capture with prompt text, prompt token IDs, generated text, and generated token IDs.
- `decode_summary.json`: aggregate metadata for the token-ID capture.
- `summary.json`: compact score summary.
- `ppl_results.jsonl`: per-record perplexity output (written by default; absent with `--no-enable-ppl`).
- `ppl_summary.json`: aggregate perplexity output (written by default; absent with `--no-enable-ppl`).

Fetch or inspect results with:

```bash
hf buckets list hf://buckets/$SCRATCH_BUCKET/$RUN_PREFIX --tree
hf buckets cp hf://buckets/$SCRATCH_BUCKET/$RUN_PREFIX/summary.json ./summary.json
```

`summary.json` includes `tps`, `output_tps`, `total_tps`, completed request count, latency metrics, benchmark parameters, and dependency metadata. When the PPL stage runs (the default), it also includes `ppl` (the aggregate, token-weighted perplexity) and `ppl_num_tokens`.

## How It Works

One HF Job does everything on localhost:

1. Mounts your submission prefix at `/submission`.
2. Mounts this shared harness at `/harness`.
3. Mounts your result prefix at `/state`.
4. Installs participant dependencies into `/tmp/server-venv`.
5. Installs pinned benchmark dependencies into `/tmp/bench-venv`.
6. Starts `manifest.json`'s `serve` command and waits for `http://127.0.0.1:<port>/v1/models`.
7. Runs the fixed benchmark against the local OpenAI-compatible endpoint.
8. Captures generated text and generated token IDs through `/v1/completions`.
9. Runs endpoint-based PPL against the ground-truth token file (default; skip with `--no-enable-ppl`).
10. Writes the raw benchmark, audit artifacts, and summary back to your scratch bucket.
