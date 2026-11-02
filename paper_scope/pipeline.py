"""Standalone embedding pipeline: text files + titles → vectors.

This module provides a single function ``embed_papers`` that takes a list of
paper dicts and returns (vectors, records) ready for ``index_store.save_index``.

Every embedding path (local add, batch, Kaggle) should use this function so
that the same model, prefix, dtype, and normalization are applied consistently.
"""

from __future__ import annotations

from datetime import datetime, UTC
from typing import Any

import numpy as np

from .embed import Embedder, EmbeddingError


def embed_papers(
    papers: list[dict[str, Any]],
    *,
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    batch_size: int = 32,
    max_tokens: int = 120_000,
    max_seq_length: int = 8192,
    device: str | None = None,
    prefix: str = "search_document: ",
    printer=print,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Embed a list of papers into vectors.

    Each paper dict must have keys: ``key``, ``title``, ``text``.
    Optional keys: ``path``, ``text_path``, ``page_count``, ``body_pages``,
    ``removed_references``, ``arxiv_id``, ``year``, ``topic``.

    Returns:
        (vectors, records) where vectors is a (N, dim) float32 array and
        records is a list of dicts ready for index_store.
    """
    if not papers:
        raise ValueError("No papers to embed")

    embedder = Embedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_tokens=max_tokens,
    )

    # Prefix all texts
    texts_to_encode = [f"{prefix}{p['text']}" for p in papers]

    # Encode in batches
    printer(f"Encoding {len(texts_to_encode)} papers (batch_size={batch_size})...")
    vectors = embedder.encode_one_batch(texts_to_encode, prefix="")

    # Build records
    records: list[dict[str, Any]] = []
    for i, paper in enumerate(papers):
        text = paper["text"]
        token_estimate = embedder.estimate_tokens(text)
        truncated = token_estimate > max_seq_length
        truncation_rate = (
            (token_estimate - max_seq_length) / token_estimate
            if truncated and token_estimate > 0
            else 0.0
        )

        records.append({
            "key": paper["key"],
            "title": paper["title"],
            "path": paper.get("path", ""),
            "text_path": paper.get("text_path", ""),
            "arxiv_id": paper.get("arxiv_id"),
            "year": paper.get("year"),
            "topic": paper.get("topic"),
            "page_count": paper.get("page_count", 0),
            "body_pages": paper.get("body_pages", 0),
            "removed_references": paper.get("removed_references", False),
            "token_estimate": token_estimate,
            "truncated": truncated,
            "truncation_rate": round(truncation_rate, 6),
        })

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]

    printer(f"Encoded {len(records)} papers → shape {vectors.shape}")

    return vectors, records


def build_manifest(
    records: list[dict[str, Any]],
    vectors: np.ndarray,
    *,
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    max_seq_length: int = 8192,
    max_tokens: int = 120_000,
) -> dict[str, Any]:
    """Build a manifest dict from records and vectors."""
    return {
        "schema_version": 2,
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": model_name,
        "dimension": vectors.shape[1],
        "max_seq_length": max_seq_length,
        "normalized": True,
        "similarity": "cosine (dot product of L2-normalised vectors)",
        "paper_count": len(records),
        "max_tokens": max_tokens,
        "papers": records,
    }
