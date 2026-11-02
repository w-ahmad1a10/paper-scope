"""Batch add papers from arXiv by title using true batched encoding.

Pipeline:
1. Download all PDFs from arXiv (network-bound, sequential).
2. Extract and clean text from all PDFs (CPU/file I/O bound).
3. Encode ALL texts in batched forward passes (batch_size=4, chunks to avoid OOM).
4. Merge with existing index and save once.
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import PaperEntry, load_collection, save_collection
from .config import (
    COLLECTION_FILE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    INDEX_DIR,
    PAPERS_DIR,
    SCHEMA_VERSION,
    TEXT_DIR,
)
from .download import is_valid_pdf
from .embed import Embedder
from .pipeline import embed_papers, build_manifest
from .extract import extract_paper
from .index_store import save_index


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s).lower().strip()


def _key_from_title(title: str) -> str:
    """Derive the paper identifier key from its title automatically."""
    key = title.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    return "".join(c for c in key if c.isalnum() or c == "_")


def _progress_bar(current: int, total: int, label: str = "", width: int = 40) -> str:
    """Return a single-line progress bar string."""
    if total == 0:
        return f"{label} 0/0"
    frac = current / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = frac * 100
    return f"{label} [{bar}] {current}/{total} ({pct:.0f}%)"


def _print_progress(current: int, total: int, label: str = "", width: int = 40) -> None:
    """Print a progress bar that updates in place."""
    line = _progress_bar(current, total, label, width)
    sys.stdout.write(f"\r{line}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def download_by_title(title: str, dirpath: Path = PAPERS_DIR) -> Path:
    """Download one paper PDF from arXiv by exact title match."""

    dirpath.mkdir(parents=True, exist_ok=True)
    key = _key_from_title(title)
    target = dirpath / f"{key}.pdf"

    # If PDF already exists locally and is valid, skip arXiv entirely.
    if target.is_file() and is_valid_pdf(target):
        return target

    import arxiv

    client = arxiv.Client()
    search = arxiv.Search(query=f'ti:"{title}"', max_results=5)
    results = list(client.results(search))

    if not results:
        raise RuntimeError(f"Paper not found on arXiv: {title}")

    norm = _norm
    best = next((r for r in results if norm(r.title) == norm(title)), results[0])

    pdf_url = best.pdf_url
    urllib.request.urlretrieve(pdf_url, str(target))

    if not is_valid_pdf(target):
        raise RuntimeError(f"Downloaded file for {title} is not a valid PDF")

    return target


def batch_add(
    titles: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    index_dir: Path = INDEX_DIR,
    printer=print,
) -> dict:
    """Download, extract, and encode many papers with true batched encoding.

    Phases:
      1. Download every PDF (network).
      2. Extract text from every PDF (CPU/file I/O).
      3. Encode ALL texts in batched forward passes (batch_size=4).
      4. Merge with existing index and save once.
    """

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(model_name, batch_size=batch_size, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Phase 1: Download all PDFs
    # ------------------------------------------------------------------

    printer(f"Phase 1/4: Downloading {len(titles)} PDFs from arXiv")
    printer("")

    downloaded: list[tuple[str, str, Path]] = []
    skipped: list[str] = []

    for position, title in enumerate(titles, start=1):
        _print_progress(position, len(titles), "Downloading")

        try:
            pdf_path = download_by_title(title, PAPERS_DIR)
        except Exception as exc:  # noqa: BLE001 - report and continue.
            skipped.append(title)
            continue

        key = _key_from_title(title)
        downloaded.append((title, key, pdf_path))

    printer(f"Downloaded {len(downloaded)} of {len(titles)} papers")
    if skipped:
        printer(f"Skipped {len(skipped)}: {', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}")
    printer("")

    # ------------------------------------------------------------------
    # Phase 2: Extract text from all PDFs
    # ------------------------------------------------------------------

    printer(f"Phase 2/4: Extracting text from {len(downloaded)} PDFs")
    printer("")

    extracted: list[dict[str, Any]] = []

    for position, (title, key, pdf_path) in enumerate(downloaded, start=1):
        _print_progress(position, len(downloaded), "Extracting")

        text_path = TEXT_DIR / f"{key}.txt"

        if text_path.is_file() and text_path.stat().st_size > 0:
            # Text already extracted — reuse it, skip re-extraction.
            text = text_path.read_text(encoding="utf-8")
            # We don't have page_count/body_pages from extraction; set defaults.
            page_count = 0
            body_pages = 0
            removed_references = True
        else:
            extracted_paper = extract_paper(pdf_path)
            text = extracted_paper.text
            text_path.write_text(text, encoding="utf-8")
            page_count = extracted_paper.page_count
            body_pages = extracted_paper.body_pages
            removed_references = extracted_paper.removed_references

        token_estimate = embedder.estimate_tokens(text)

        truncated = token_estimate > embedder.max_seq_length
        truncation_rate = (
            (token_estimate - embedder.max_seq_length) / token_estimate
            if truncated and token_estimate > 0
            else 0.0
        )

        extracted.append(
            {
                "key": key,
                "title": title,
                "path": str(pdf_path),
                "text_path": str(text_path),
                "text": text,
                "page_count": page_count,
                "body_pages": body_pages,
                "removed_references": removed_references,
                "token_estimate": token_estimate,
                "truncated": truncated,
                "truncation_rate": round(truncation_rate, 6),
            }
        )

    printer(f"Extracted {len(extracted)} papers")
    printer("")

    # ------------------------------------------------------------------
    # Phase 3: Encode ALL texts in batched forward passes
    # ------------------------------------------------------------------

    texts = [item["text"] for item in extracted]
    total_chunks = (len(texts) + 3) // 4

    printer(f"Phase 3/4: Encoding {len(texts)} texts in batches of 4 ({total_chunks} chunks)")
    printer("")

    encode_start = time.perf_counter()
    vectors = embedder.encode_one_batch(texts, prefix="search_document: ")
    encode_elapsed = time.perf_counter() - encode_start

    printer(f"Encoding finished in {encode_elapsed:.1f}s")
    printer("")

    # ------------------------------------------------------------------
    # Phase 4: Merge with existing index and save
    # ------------------------------------------------------------------

    printer("Phase 4/4: Merging and saving index")

    records_path = index_dir / "papers.jsonl"
    vectors_path = index_dir / "embeddings.npy"

    existing_records: list[dict[str, Any]] = []
    existing_vectors: list = []
    if records_path.is_file():
        import json

        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing_records.append(json.loads(line))
    if vectors_path.is_file():
        old = np.load(vectors_path, allow_pickle=False)
        existing_vectors = list(old)

    new_keys = {item["key"] for item in extracted}
    existing_records = [row for row in existing_records if row.get("key") not in new_keys]
    existing_vectors = [
        vec
        for vec, row in zip(existing_vectors, existing_records)
        if row.get("key") not in new_keys
    ]

    records: list[dict[str, Any]] = list(existing_records)
    vectors_list: list = list(existing_vectors)
    entries: list[PaperEntry] = []

    for vector_index, item in enumerate(extracted):
        record = {
            "key": item["key"],
            "title": item["title"],
            "path": item["path"],
            "text_path": item["text_path"],
            "arxiv_id": None,
            "year": None,
            "topic": None,
            "page_count": item["page_count"],
            "body_pages": item["body_pages"],
            "removed_references": item["removed_references"],
            "token_estimate": item["token_estimate"],
            "truncated": item["truncated"],
            "truncation_rate": item["truncation_rate"],
        }
        records.append(record)
        vectors_list.append(vectors[vector_index])
        entries.append(
            PaperEntry(
                key=item["key"],
                title=item["title"],
                path=item["path"],
                arxiv_id=None,
                year=None,
                topic=None,
            )
        )

    if len(records) == len(existing_records):
        raise RuntimeError("No papers were downloaded successfully")

    matrix = np.asarray(vectors_list, dtype=np.float32)
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

    all_entries: list[PaperEntry] = []
    for row in records:
        all_entries.append(
            PaperEntry(
                key=row["key"],
                title=row["title"],
                path=row["path"],
                arxiv_id=row.get("arxiv_id"),
                year=row.get("year"),
                topic=row.get("topic"),
            )
        )
    save_collection(all_entries, collection_file=COLLECTION_FILE)

    elapsed = time.perf_counter() - started

    printer("")
    printer(f"Saved {len(records)} vectors to {index_dir}")
    printer(f"Batch finished in {elapsed / 60:.1f} minutes")
    return manifest
