"""Persist and load the paragraph vectors, and run exact cosine search."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    EMBEDDINGS_FILE,
    INDEX_DIR,
    MANIFEST_FILE,
    PARAGRAPHS_FILE,
    SCHEMA_VERSION,
)
from .extract import Paragraph


class IndexStoreError(RuntimeError):
    """Raised when the saved semantic index is missing or inconsistent."""


@dataclass(frozen=True)
class SearchHit:
    score: float
    paragraph: Paragraph


def _paragraph_to_dict(paragraph: Paragraph) -> dict[str, Any]:
    return {
        "id": paragraph.id,
        "paper_key": paragraph.paper_key,
        "paper_title": paragraph.paper_title,
        "arxiv_id": paragraph.arxiv_id,
        "year": paragraph.year,
        "page": paragraph.page,
        "text": paragraph.text,
    }


def _paragraph_from_dict(row: dict[str, Any]) -> Paragraph:
    try:
        return Paragraph(
            id=str(row["id"]),
            paper_key=str(row["paper_key"]),
            paper_title=str(row["paper_title"]),
            arxiv_id=str(row["arxiv_id"]),
            year=int(row["year"]),
            page=int(row["page"]),
            text=str(row["text"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexStoreError(f"Malformed paragraph record: {exc}") from exc


class SemanticIndex:
    def __init__(
        self,
        paragraphs: list[Paragraph],
        vectors: np.ndarray,
        manifest: dict[str, Any],
    ) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise IndexStoreError(f"Embedding matrix must be 2-D, got shape {matrix.shape}")
        if len(paragraphs) != matrix.shape[0]:
            raise IndexStoreError(
                f"Have {len(paragraphs)} paragraphs but {matrix.shape[0]} vectors"
            )
        if matrix.shape[0] == 0:
            raise IndexStoreError("The semantic index is empty")

        norms = np.linalg.norm(matrix, axis=1)
        if not np.isfinite(norms).all() or np.any(norms < 1e-12):
            raise IndexStoreError("Embedding matrix contains zero or invalid vectors")

        self.paragraphs = paragraphs
        self.vectors = np.ascontiguousarray(matrix)
        self.manifest = dict(manifest)
        self.model_name = str(
            self.manifest.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.dimension = int(self.manifest.get("dimension", matrix.shape[1]))
        if self.vectors.shape[1] != self.dimension:
            raise IndexStoreError(
                f"Manifest dimension {self.dimension} does not match vectors "
                f"{self.vectors.shape[1]}"
            )

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> SemanticIndex:
        manifest_path = index_dir / MANIFEST_FILE.name
        paragraphs_path = index_dir / PARAGRAPHS_FILE.name
        vectors_path = index_dir / EMBEDDINGS_FILE.name

        if not manifest_path.is_file():
            raise IndexStoreError(
                f"No index found in {index_dir}. Build the index before searching."
            )
        for path in (paragraphs_path, vectors_path):
            if not path.is_file():
                raise IndexStoreError(f"Index file is missing: {path}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexStoreError(f"Could not read the index manifest: {exc}") from exc

        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise IndexStoreError(
                "The saved index uses a different schema version. Rebuild it."
            )

        try:
            vectors = np.load(vectors_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise IndexStoreError(f"Could not read embedding vectors: {exc}") from exc

        paragraphs: list[Paragraph] = []
        try:
            with paragraphs_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        paragraphs.append(_paragraph_from_dict(json.loads(line)))
                    except json.JSONDecodeError as exc:
                        raise IndexStoreError(
                            f"Bad paragraph JSON on line {line_number}: {exc}"
                        ) from exc
        except OSError as exc:
            raise IndexStoreError(f"Could not read paragraph records: {exc}") from exc

        expected = manifest.get("paragraph_count")
        if expected is not None and int(expected) != len(paragraphs):
            raise IndexStoreError(
                f"Manifest says {expected} paragraphs but file has {len(paragraphs)}"
            )

        return cls(paragraphs, vectors, manifest)

    def search_vector(self, query_vector: np.ndarray, top_k: int = 1) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimension:
            raise IndexStoreError(
                f"Query vector has {query.shape[0]} dimensions; index needs {self.dimension}"
            )
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm < 1e-12:
            raise IndexStoreError("Query embedding has zero length")
        query = query / norm

        scores = self.vectors @ query
        limit = min(int(top_k), len(self.paragraphs))
        order = np.argsort(-scores, kind="stable")[:limit]
        return [
            SearchHit(score=float(scores[position]), paragraph=self.paragraphs[position])
            for position in order
        ]


def save_index(
    paragraphs: list[Paragraph],
    vectors: np.ndarray,
    manifest: dict[str, Any],
    index_dir: Path = INDEX_DIR,
) -> None:
    if len(paragraphs) != vectors.shape[0]:
        raise IndexStoreError(
            f"Refusing to save {len(paragraphs)} paragraphs and {vectors.shape[0]} vectors"
        )

    index_dir.mkdir(parents=True, exist_ok=True)
    paragraphs_path = index_dir / PARAGRAPHS_FILE.name
    vectors_path = index_dir / EMBEDDINGS_FILE.name
    manifest_path = index_dir / MANIFEST_FILE.name

    temporary_paragraphs = paragraphs_path.with_suffix(".jsonl.tmp")
    temporary_vectors = vectors_path.with_suffix(".npy.tmp")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")

    try:
        with temporary_paragraphs.open("w", encoding="utf-8", newline="\n") as handle:
            for paragraph in paragraphs:
                handle.write(
                    json.dumps(_paragraph_to_dict(paragraph), ensure_ascii=False) + "\n"
                )

        with temporary_vectors.open("wb") as handle:
            np.save(handle, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        os.replace(temporary_vectors, vectors_path)
        os.replace(temporary_paragraphs, paragraphs_path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_paragraphs.unlink(missing_ok=True)
        temporary_vectors.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
