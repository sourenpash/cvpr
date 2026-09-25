#!/usr/bin/env python3
"""Kimodo's text embeddings with the original ``llm2vec`` package (``llm2vec`` env).

Kimodo conditions on LLM2Vec-Meta-Llama-3-8B-Instruct (MNTP + supervised adapters, mean pooling,
no instruction). Its bundled copy returns all-zero embeddings under transformers 5.1 (see
``generate_kimodo_motions.py``); the reference implementation, in its supported transformers
range (4.44), is used here instead. Runs on the CPU in float32 (~32 GB RAM, seconds per prompt).

    python scripts/r1/kimodo_text_embeddings.py data/kimodo_g1/prompts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompts", type=Path, help="json list of prompts")
    ap.add_argument("--out", type=Path, default=None, help="default: <prompts>.npz")
    args = ap.parse_args()

    from llm2vec import LLM2Vec
    import torch

    texts = json.loads(args.prompts.read_text())
    l2v = LLM2Vec.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    emb = l2v.encode(texts, batch_size=1, show_progress_bar=True, convert_to_numpy=True)
    emb = np.asarray(emb, dtype=np.float32)
    assert emb.shape == (len(texts), 4096) and np.isfinite(emb).all()
    norms = np.linalg.norm(emb, axis=1)
    assert norms.min() > 0, "zero embedding: the encoder is broken"
    unit = emb / norms[:, None]
    off_diag = unit @ unit.T - np.eye(len(texts))
    print(
        f"{len(texts)} embeddings; norm {norms.min():.2f}-{norms.max():.2f}; "
        f"max cosine between prompts {off_diag.max():.3f}"
    )
    out = args.out or args.prompts.with_suffix(".npz")
    np.savez(out, texts=np.array(texts), emb=emb)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
