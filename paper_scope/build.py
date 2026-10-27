"""Build the full paper-to-vector index."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .catalog import CATALOG, Paper
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    INDEX_DIR,
    MAX_CHARS,
    MIN_CHARS,
    PAPERS_FILE,
    SCHEMA_VERSION,
    TARGET_CHARS,
    TEXT_DIR,
)
from .download import download_all
from .embed import Embedder
from .extract import Paragraph, extract_paper
from .index_store import save_index


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_index(
    *,
    papers: tuple[Paper, ...] | None = None,
    limit: int | None = None,
    force_download: bool = False,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    index_dir: Path = INDEX_DIR,
    printer=print,
) -> dict:
    selected = tuple(papers or CATALOG)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        selected = selected[:limit]

    started = time.perf_counter()
    printer(f"PaperScope build: {len(selected)} papers")
    printer("")

    pdf_paths = download_all(selected, force=force_download, printer=printer)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_paragraphs: list[Paragraph] = []
    paper_stats: list[dict] = []

    printer("")
    printer("Extracting text and building paragraphs")
    for position, (paper, pdf_path) in enumerate(zip(selected, pdf_paths), start=1):
        printer(f"[{position}/{len(selected)}] {paper.key:11s} reading PDF")
        extracted = extract_paper(pdf_path, paper)
        (TEXT_DIR / f"{paper.key}.txt").write_text(extracted.text, encoding="utf-8")
        all_paragraphs.extend(extracted.paragraphs)

        stat = {
            **asdict(paper),
            "pages": extracted.page_count,
            "body_pages": extracted.body_pages,
            "paragraphs": len(extracted.paragraphs),
        }
        paper_stats.append(stat)
        printer(
            f"             {extracted.body_pages:2d}/{extracted.page_count:2d} pages, "
            f"{len(extracted.paragraphs):4d} paragraphs"
        )

    if not all_paragraphs:
        raise RuntimeError("No paragraphs were extracted from any paper")

    PAPERS_FILE.write_text(
        json.dumps(paper_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    printer("")
    printer(f"Paragraphs: {len(all_paragraphs)}")
    printer(f"Loading embedding model: {model_name}")
    embedder = Embedder(model_name, batch_size=batch_size)
    printer(
        f"Model ready: {embedder.dimension} dimensions, "
        f"max {embedder.max_seq_length} tokens"
    )

    texts = [paragraph.text for paragraph in all_paragraphs]
    printer(f"Embedding {len(texts)} paragraphs on {embedder.device}")
    vectors = embedder.encode(texts, show_progress_bar=True, batch_size=batch_size)
    if vectors.shape != (len(all_paragraphs), embedder.dimension):
        raise RuntimeError(
            f"Embedding shape mismatch: {vectors.shape} for {len(all_paragraphs)} paragraphs"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at": _now(),
        "model_name": embedder.model_name,
        "query_prefix": embedder.query_prefix,
        "dimension": embedder.dimension,
        "max_seq_length": embedder.max_seq_length,
        "normalized": True,
        "similarity": "cosine (dot product of L2-normalised vectors)",
        "paragraph_count": len(all_paragraphs),
        "paper_count": len(selected),
        "chunking": {
            "target_chars": TARGET_CHARS,
            "max_chars": MAX_CHARS,
            "min_chars": MIN_CHARS,
            "boundary": "sentence",
        },
        "papers": paper_stats,
    }

    save_index(all_paragraphs, vectors, manifest, index_dir=index_dir)
    elapsed = time.perf_counter() - started

    printer("")
    printer(f"Saved {len(all_paragraphs)} vectors to {index_dir}")
    printer(f"Build finished in {elapsed / 60:.1f} minutes")
    return manifest
