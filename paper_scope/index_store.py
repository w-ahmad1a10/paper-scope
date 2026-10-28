"""Persist and load whole-paper vectors, and run exact cosine search."""

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
    PAPER_RECORDS_FILE,
    SCHEMA_VERSION,
)


class IndexError(RuntimeError):
    """Raised when the saved semantic index is missing or inconsistent."""


@dataclass(frozen=True)
class SearchHit:
    score: float
    paper: dict[str, Any]


def _paper_from_dict(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return {
            "key": str(row["key"]),
            "title": str(row["title"]),
            "path": str(row["path"]),
            "text_path": str(row.get("text_path", "")),
            "arxiv_id": row.get("arxiv_id"),
            "year": row.get("year"),
            "topic": row.get("topic"),
            "page_count": int(row.get("page_count", 0) or 0),
            "body_pages": int(row.get("body_pages", 0) or 0),
            "removed_references": bool(row.get("removed_references", False)),
            "token_estimate": row.get("token_estimate"),
            "truncated": bool(row.get("truncated", False)),
            "truncation_rate": row.get("truncation_rate"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexError(f"Malformed paper record: {exc}") from exc


class SemanticIndex:
    def __init__(
        self,
        papers: list[dict[str, Any]],
        vectors: np.ndarray,
        manifest: dict[str, Any],
    ) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise IndexError(f"Embedding matrix must be 2-D, got shape {matrix.shape}")
        if len(papers) != matrix.shape[0]:
            raise IndexError(
                f"Have {len(papers)} papers but {matrix.shape[0]} vectors"
            )
        if matrix.shape[0] == 0:
            raise IndexError("The semantic index is empty")

        norms = np.linalg.norm(matrix, axis=1)
        if not np.isfinite(norms).all() or np.any(norms < 1e-12):
            raise IndexError("Embedding matrix contains zero or invalid vectors")

        self.papers = papers
        self.vectors = np.ascontiguousarray(matrix)
        self.manifest = dict(manifest)
        self.model_name = str(
            self.manifest.get("model_name", "voyageai/voyage-4-nano")
        )
        self.dimension = int(self.manifest.get("dimension", matrix.shape[1]))
        if self.vectors.shape[1] != self.dimension:
            raise IndexError(
                f"Manifest dimension {self.dimension} does not match vectors "
                f"{self.vectors.shape[1]}"
            )

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> SemanticIndex:
        manifest_path = index_dir / MANIFEST_FILE.name
        papers_path = index_dir / PAPER_RECORDS_FILE.name
        vectors_path = index_dir / EMBEDDINGS_FILE.name

        if not manifest_path.is_file():
            raise IndexError(
                f"No index found in {index_dir}. Add papers before searching."
            )
        for path in (papers_path, vectors_path):
            if not path.is_file():
                raise IndexError(f"Index file is missing: {path}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexError(f"Could not read the index manifest: {exc}") from exc

        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise IndexError(
                "The saved index uses a different schema version. Rebuild it."
            )

        try:
            vectors = np.load(vectors_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise IndexError(f"Could not read embedding vectors: {exc}") from exc

        papers: list[dict[str, Any]] = []
        try:
            with papers_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        papers.append(_paper_from_dict(row))
                    except json.JSONDecodeError as exc:
                        raise IndexError(
                            f"Bad paper JSON on line {line_number}: {exc}"
                        ) from exc
        except OSError as exc:
            raise IndexError(f"Could not read paper records: {exc}") from exc

        expected = manifest.get("paper_count")
        if expected is not None and int(expected) != len(papers):
            raise IndexError(
                f"Manifest says {expected} papers but file has {len(papers)}"
            )

        return cls(papers, vectors, manifest)

    def search_vector(self, query_vector: np.ndarray, top_k: int = 1) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimension:
            raise IndexError(
                f"Query vector has {query.shape[0]} dimensions; index needs {self.dimension}"
            )
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm < 1e-12:
            raise IndexError("Query embedding has zero length")
        query = query / norm

        scores = self.vectors @ query
        limit = min(int(top_k), len(self.papers))
        order = np.argsort(-scores, kind="stable")[:limit]
        return [
            SearchHit(score=float(scores[position]), paper=self.papers[position])
            for position in order
        ]

    def all_scores(self, query_vector: np.ndarray) -> list[SearchHit]:
        """Return every paper ranked by similarity to the question."""

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimension:
            raise IndexError(
                f"Query vector has {query.shape[0]} dimensions; index needs {self.dimension}"
            )
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm < 1e-12:
            raise IndexError("Query embedding has zero length")
        query = query / norm

        scores = self.vectors @ query
        order = np.argsort(-scores, kind="stable")
        return [
            SearchHit(score=float(scores[position]), paper=self.papers[position])
            for position in order
        ]


def save_index(
    papers: list[dict[str, Any]],
    vectors: np.ndarray,
    manifest: dict[str, Any],
    index_dir: Path = INDEX_DIR,
) -> None:
    if len(papers) != vectors.shape[0]:
        raise IndexError(
            f"Refusing to save {len(papers)} papers and {vectors.shape[0]} vectors"
        )

    index_dir.mkdir(parents=True, exist_ok=True)
    papers_path = index_dir / PAPER_RECORDS_FILE.name
    vectors_path = index_dir / EMBEDDINGS_FILE.name
    manifest_path = index_dir / MANIFEST_FILE.name

    temporary_papers = papers_path.with_suffix(".jsonl.tmp")
    temporary_vectors = vectors_path.with_suffix(".npy.tmp")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")

    try:
        with temporary_papers.open("w", encoding="utf-8", newline="\n") as handle:
            for paper in papers:
                handle.write(json.dumps(paper, ensure_ascii=False) + "\n")

        with temporary_vectors.open("wb") as handle:
            np.save(handle, np.asarray(vectors, dtype=np.float32), allow_pickle=False)

        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        os.replace(temporary_vectors, vectors_path)
        os.replace(temporary_papers, papers_path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_papers.unlink(missing_ok=True)
        temporary_vectors.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
