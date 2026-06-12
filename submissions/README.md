# Submissions

Each subdirectory is a self-contained challenge submission package. A runnable
submission contains at least:

```text
manifest.json
serve.py
```

Develop and review these packages in GitHub. Upload the selected package to the
HF scratch bucket before benchmarking:

```bash
python scripts/upload_submission.py --path submissions/vllm_baseline --name vllm-baseline
```

The canonical HF destination is:

```text
hf://buckets/gemma-challenge/gemma-senpai/submissions/senpai/<submission-name>
```
