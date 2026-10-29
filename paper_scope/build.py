"""Build and update the whole-paper vector index from a user-built collection."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from .catalog import PaperEntry, load_collection, save_collection
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    INDEX_DIR,
    SCHEMA_VERSION,
    TEXT_DIR,
)
from .download import copy_pdf
from .embed import Embedder
from .extract import extract_paper
from .index_store import save_index


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def add_paper(
    key: str,
    title: str,
    source_path: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    index_dir: Path = INDEX_DIR,
    collection_file: Path | None = None,
    printer=print,
) -> dict:
    """Add one paper to the collection and append its vector to the index."""

    if collection_file is None:
        from .config import COLLECTION_FILE

        collection_file = COLLECTION_FILE

    index_dir.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    local_pdf = copy_pdf(source_path, key)

    extracted = extract_paper(local_pdf)
    text_path = TEXT_DIR / f"{key}.txt"
    text_path.write_text(extracted.text, encoding="utf-8")

    embedder = Embedder(model_name, batch_size=batch_size, max_tokens=max_tokens)
    token_estimate = embedder.estimate_tokens(extracted.text)

    truncated = token_estimate > embedder.max_seq_length
    truncation_rate = (
        (token_estimate - embedder.max_seq_length) / token_estimate
        if truncated and token_estimate > 0
        else 0.0
    )

    vector = embedder.encode_one(extracted.text)

    # Load existing records if present.
    records_path = index_dir / "papers.jsonl"
    records: list[dict] = []
    vectors: list = []
    if records_path.is_file():
        try:
            with records_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            records = []
    vectors_path = index_dir / "embeddings.npy"
    if vectors_path.is_file():
        import numpy as np

        old = np.load(vectors_path, allow_pickle=False)
        vectors = list(old)

    # Remove any existing record with the same key, then append.
    records = [row for row in records if row.get("key") != key]
    vectors = [vec for vec, row in zip(vectors, records) if row.get("key") != key]

    record = {
        "key": key,
        "title": title,
        "path": str(local_pdf),
        "text_path": str(text_path),
        "arxiv_id": None,
        "year": None,
        "topic": None,
        "page_count": extracted.page_count,
        "body_pages": extracted.body_pages,
        "removed_references": extracted.removed_references,
        "token_estimate": token_estimate,
        "truncated": truncated,
        "truncation_rate": round(truncation_rate, 6),
    }
    records.append(record)
    vectors.append(vector)

    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float32)
    matrix = np.ascontiguousarray(matrix)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at": _now(),
        "model_name": embedder.model_name,
        "dimension": embedder.dimension,
        "max_seq_length": embedder.max_seq_length,
        "normalized": True,
        "similarity": "cosine (dot product of L2-normalised vectors)",
        "paper_count": len(records),
        "max_tokens": max_tokens,
        "papers": records,
    }

    save_index(records, matrix, manifest, index_dir=index_dir)
    save_collection(
        [
            PaperEntry(
                key=row["key"],
                title=row["title"],
                path=row["path"],
                arxiv_id=row.get("arxiv_id"),
                year=row.get("year"),
                topic=row.get("topic"),
            )
            for row in records
        ],
        collection_file=collection_file,
    )

    printer(
        f"Added {key:11s} | pages {extracted.body_pages:2d}/{extracted.page_count:2d} "
        f"| tokens {token_estimate:6d} | truncated {truncated} | "
        f"rate {truncation_rate:.4f}"
    )
    return manifest


def build_index(
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    index_dir: Path = INDEX_DIR,
    collection_file: Path | None = None,
    printer=print,
) -> dict:
    """Rebuild the whole index from the saved collection."""

    if collection_file is None:
        from .config import COLLECTION_FILE

        collection_file = COLLECTION_FILE

    entries = load_collection(collection_file)
    if not entries:
        raise RuntimeError("No papers in the collection. Add papers first.")

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    vectors: list = []

    started = time.perf_counter()
    printer(f"Building index: {len(entries)} papers")
    printer("")

    for position, entry in enumerate(entries, start=1):
        pdf_path = Path(entry.path)
        if not pdf_path.is_file():
            printer(f"[{position}/{len(entries)}] {entry.key:11s} missing PDF, skipping")
            continue

        printer(f"[{position}/{len(entries)}] {entry.key:11s} reading PDF")
        extracted = extract_paper(pdf_path)
        text_path = TEXT_DIR / f"{entry.key}.txt"
        text_path.write_text(extracted.text, encoding="utf-8")

        embedder = Embedder(model_name, batch_size=batch_size, max_tokens=max_tokens)
        token_estimate = embedder.estimate_tokens(extracted.text)

        truncated = token_estimate > embedder.max_seq_length
        truncation_rate = (
            (token_estimate - embedder.max_seq_length) / token_estimate
            if truncated and token_estimate > 0
            else 0.0
        )

        vector = embedder.encode_one(extracted.text)

        record = {
            "key": entry.key,
            "title": entry.title,
            "path": str(pdf_path),
            "text_path": str(text_path),
            "arxiv_id": entry.arxiv_id,
            "year": entry.year,
            "topic": entry.topic,
            "page_count": extracted.page_count,
            "body_pages": extracted.body_pages,
            "removed_references": extracted.removed_references,
            "token_estimate": token_estimate,
            "truncated": truncated,
            "truncation_rate": round(truncation_rate, 6),
        }
        records.append(record)
        vectors.append(vector)

        printer(
            f"             {extracted.body_pages:2d}/{extracted.page_count:2d} pages "
            f"| tokens {token_estimate:6d} | truncated {truncated} | "
            f"rate {truncation_rate:.4f}"
        )

    if not records:
        raise RuntimeError("No papers were extracted successfully")

    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float32)
    matrix = np.ascontiguousarray(matrix)

    embedder = Embedder(model_name, batch_size=batch_size, max_tokens=max_tokens)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at": _now(),
        "model_name": embedder.model_name,
        "dimension": embedder.dimension,
        "max_seq_length": embedder.max_seq_length,
        "normalized": True,
        "similarity": "cosine (dot product of L2-normalised vectors)",
        "paper_count": len(records),
        "max_tokens": max_tokens,
        "papers": records,
    }

    save_index(records, matrix, manifest, index_dir=index_dir)
    save_collection(
        [
            PaperEntry(
                key=row["key"],
                title=row["title"],
                path=row["path"],
                arxiv_id=row.get("arxiv_id"),
                year=row.get("year"),
                topic=row.get("topic"),
            )
            for row in records
        ],
        collection_file=collection_file,
    )

    elapsed = time.perf_counter() - started
    printer("")
    printer(f"Saved {len(records)} vectors to {index_dir}")
    printer(f"Build finished in {elapsed / 60:.1f} minutes")
    return manifest
