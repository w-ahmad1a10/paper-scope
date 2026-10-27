from __future__ import annotations

import numpy as np
import pytest

from paper_scope.extract import Paragraph
from paper_scope.index_store import (
    IndexStoreError,
    SemanticIndex,
    save_index,
)


def _paragraph(key: str, text: str) -> Paragraph:
    return Paragraph(
        id=f"{key}:p001:c00",
        paper_key=key,
        paper_title=f"Paper {key}",
        arxiv_id="1234.5678",
        year=2024,
        page=1,
        text=text,
    )


def _manifest(count: int, dimension: int = 2) -> dict:
    return {
        "schema_version": 1,
        "model_name": "test-model",
        "dimension": dimension,
        "paragraph_count": count,
    }


def _index() -> SemanticIndex:
    paragraphs = [
        _paragraph("a", "Alpha paragraph"),
        _paragraph("b", "Beta paragraph"),
        _paragraph("c", "Gamma paragraph"),
    ]
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.70710677, 0.70710677],
        ],
        dtype=np.float32,
    )
    return SemanticIndex(paragraphs, vectors, _manifest(3))


def test_exact_search_returns_most_similar_paragraph() -> None:
    index = _index()

    hits = index.search_vector(np.array([3.0, 0.0]), top_k=1)

    assert len(hits) == 1
    assert hits[0].paragraph.paper_key == "a"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_clamps_top_k_and_orders_scores() -> None:
    index = _index()

    hits = index.search_vector(np.array([1.0, 1.0]), top_k=99)

    assert len(hits) == 3
    assert hits[0].paragraph.paper_key == "c"
    assert hits[0].score > hits[1].score


def test_zero_query_is_rejected() -> None:
    index = _index()

    with pytest.raises(IndexStoreError):
        index.search_vector(np.array([0.0, 0.0]), top_k=1)


def test_save_and_load_round_trip(tmp_path) -> None:
    index = _index()

    save_index(index.paragraphs, index.vectors, index.manifest, tmp_path)
    loaded = SemanticIndex.load(tmp_path)

    assert loaded.dimension == 2
    assert len(loaded.paragraphs) == 3
    assert loaded.paragraphs[1].paper_key == "b"
    assert np.allclose(loaded.vectors, index.vectors)
