"""Build and analyze static kept sets for the greedy-safe sparse verifier.

The kept set ``S`` is the vocabulary the lm_head / spec-verify GEMM is restricted
to. Its *source does not affect correctness* — the certificate guarantees greedy
identity for any ``S`` — it only affects the fallback rate, i.e. the size of the
win. We therefore build several candidate kept sets and report the complement
geometry (``R = max_{j not in S} ||W_j||``) that drives certification:

* ``norm-topk``   : top-k token rows by L2 norm. Model-derived, fully prompt
                    invariant, zero external data. In a tied-embedding model the
                    row norm tracks output-prediction utility, so this keeps the
                    high-norm tokens and leaves a low-norm complement (tight bound).
* ``freq-topk``   : top-k tokens by frequency over a *broad* multi-domain corpus
                    (never the 128 public prompts). This is the classic "lmhead12k"
                    construction; built only if a corpus is available.
* ``freq+norm``   : union of freq-topk and a high-norm guard band, so every
                    high-norm token is verified and the complement is guaranteed
                    low-norm (tightest certificate for a given emit budget).

The kept set is intentionally derived from broad/prompt-invariant signals, never
from ``eval_prompts_sharegpt.json``, so it generalizes to the private re-run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


TIED_EMBED_KEY = "model.language_model.embed_tokens.weight"


def load_lm_head_weight(model: str, key: str = TIED_EMBED_KEY) -> torch.Tensor:
    """Load only the tied lm_head / embedding tensor ``[V, H]`` from the model's
    safetensors, avoiding a full (multimodal, ~16 GB) model instantiation.

    Resolves a single-file ``model.safetensors`` or a sharded checkpoint via its
    index, downloading from the hub only if not already cached.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    if Path(model).exists():
        index = Path(model) / "model.safetensors.index.json"
        if index.exists():
            shard = json.loads(index.read_text())["weight_map"][key]
            path = str(Path(model) / shard)
        else:
            path = str(Path(model) / "model.safetensors")
    else:
        try:
            index_path = hf_hub_download(model, "model.safetensors.index.json")
            shard = json.loads(Path(index_path).read_text())["weight_map"][key]
            path = hf_hub_download(model, shard)
        except Exception:
            path = hf_hub_download(model, "model.safetensors")

    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def row_norms(weight: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """L2 norm of every row of ``weight`` ([V, H]) in fp32, chunked for memory."""
    v = weight.shape[0]
    out = torch.empty(v, dtype=torch.float32)
    for start in range(0, v, chunk):
        end = min(start + chunk, v)
        out[start:end] = weight[start:end].to(torch.float32).norm(dim=1).cpu()
    return out


def norm_topk(norms: torch.Tensor, k: int) -> torch.Tensor:
    return torch.sort(torch.topk(norms, k).indices).values


def freq_topk_from_corpus(tokenizer, k: int, *, max_docs: int = 20000, seed: int = 0):
    """Top-k token ids by frequency over a broad multi-domain corpus.

    Returns ``(kept_ids_tensor, provenance_dict)`` or ``(None, reason)`` if no
    corpus can be loaded. Deliberately excludes the eval prompt set.
    """
    sources = [
        ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text"),
        ("allenai/c4", "en", "train", "text"),
        ("bigcode/the-stack-smol", None, "train", "content"),
    ]
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        return None, {"reason": f"datasets import failed: {exc}"}

    counts = np.zeros(len(tokenizer), dtype=np.int64)
    used = []
    per_source = max(1, max_docs // max(1, len(sources)))
    for name, config, split, field in sources:
        try:
            ds = load_dataset(name, config, split=split, streaming=True)
        except Exception as exc:  # pragma: no cover
            used.append({"source": name, "config": config, "error": str(exc)[:160]})
            continue
        n = 0
        try:
            for row in ds:
                text = row.get(field)
                if not isinstance(text, str) or not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                if ids:
                    np.add.at(counts, np.asarray(ids, dtype=np.int64), 1)
                n += 1
                if n >= per_source:
                    break
        except Exception as exc:  # pragma: no cover
            used.append({"source": name, "config": config, "docs": n, "error": str(exc)[:160]})
            continue
        used.append({"source": name, "config": config, "docs": n})
    total = int(counts.sum())
    if total == 0:
        return None, {"reason": "no corpus documents tokenized", "attempts": used}
    kept = np.argsort(counts)[::-1][:k]
    kept = np.sort(kept).astype(np.int64)
    prov = {
        "method": "corpus-frequency",
        "k": int(k),
        "total_tokens_counted": total,
        "distinct_tokens_seen": int((counts > 0).sum()),
        "sources": used,
        "coverage_of_kept": float(counts[kept].sum() / total),
    }
    return torch.from_numpy(kept), prov


def complement_analysis(norms: torch.Tensor, kept_ids: torch.Tensor) -> dict:
    """Geometry that determines certification: complement max norm vs kept norms."""
    v = norms.shape[0]
    mask = torch.zeros(v, dtype=torch.bool)
    mask[kept_ids] = True
    kept_n = norms[mask]
    comp_n = norms[~mask]
    q = torch.tensor([0.5, 0.9, 0.99, 0.999])
    return {
        "kept_size": int(mask.sum()),
        "complement_size": int((~mask).sum()),
        "R_complement_max_norm": float(comp_n.max()) if comp_n.numel() else 0.0,
        "kept_min_norm": float(kept_n.min()),
        "kept_median_norm": float(kept_n.median()),
        "kept_max_norm": float(kept_n.max()),
        "complement_median_norm": float(comp_n.median()) if comp_n.numel() else 0.0,
        "complement_quantile_norms": {
            f"q{int(qq * 1000)/1000}": float(torch.quantile(comp_n, qq)) for qq in q
        }
        if comp_n.numel()
        else {},
        "global_max_norm": float(norms.max()),
        # how many kept tokens does the complement max-norm exceed? (guard-band sizing)
        "kept_below_R": int((kept_n < (comp_n.max() if comp_n.numel() else 0.0)).sum()),
    }


def build_freq_plus_norm(freq_ids: torch.Tensor, norms: torch.Tensor, guard: int) -> torch.Tensor:
    """Union freq-topk with the top-``guard`` highest-norm tokens (certificate guard band)."""
    high_norm = torch.topk(norms, guard).indices
    return torch.unique(torch.cat([freq_ids.long(), high_norm.long()]))


def save_kept_set(name: str, kept_ids: torch.Tensor, provenance: dict) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ids = np.sort(np.asarray(kept_ids.cpu(), dtype=np.int64))
    np.save(ARTIFACTS / f"kept_ids_{name}.npy", ids)
    (ARTIFACTS / f"kept_ids_{name}.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    return ARTIFACTS / f"kept_ids_{name}.npy"


def load_kept_set(name: str) -> np.ndarray:
    return np.load(ARTIFACTS / f"kept_ids_{name}.npy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--k", type=int, default=12000)
    parser.add_argument("--guard", type=int, default=2000, help="high-norm guard band for freq+norm")
    parser.add_argument("--with-corpus", action="store_true", help="also build corpus-frequency kept set")
    parser.add_argument("--corpus-docs", type=int, default=20000)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"loading {args.model} tied lm_head embedding only...", flush=True)
    weight = load_lm_head_weight(args.model).detach()
    print(f"lm_head weight: {tuple(weight.shape)} dtype={weight.dtype}", flush=True)

    norms = row_norms(weight)
    report = {"model": args.model, "vocab_size": int(weight.shape[0]), "hidden_size": int(weight.shape[1]), "k": args.k}

    norm_ids = norm_topk(norms, args.k)
    report["norm_topk"] = complement_analysis(norms, norm_ids)
    save_kept_set("norm_topk", norm_ids, {"method": "row-norm-topk", "k": args.k, **report["norm_topk"]})

    if args.with_corpus:
        freq_ids, prov = freq_topk_from_corpus(tok, args.k, max_docs=args.corpus_docs)
        if freq_ids is not None:
            report["freq_topk"] = {**complement_analysis(norms, freq_ids), "provenance": prov}
            save_kept_set("freq_topk", freq_ids, {"method": "corpus-frequency", "k": args.k, **prov})
            fn_ids = build_freq_plus_norm(freq_ids, norms, args.guard)
            report["freq_plus_norm"] = complement_analysis(norms, fn_ids)
            save_kept_set(
                "freq_plus_norm",
                fn_ids,
                {"method": "corpus-frequency + high-norm guard", "k": args.k, "guard": args.guard, **report["freq_plus_norm"]},
            )
        else:
            report["freq_topk_error"] = prov

    (ARTIFACTS / "kept_set_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
