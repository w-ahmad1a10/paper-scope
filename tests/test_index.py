from __future__ import annotations

import numpy as np
import pytest

from paper_scope.index_store import (
    IndexError,
    SemanticIndex,
    save_index,
)


def _paper(key: str, title: str = "Paper X") -> dict:
    return {
        "key": key,
        "title": title,
        "path": f"/tmp/{key}.pdf",
        "arxiv_id": "1234.5678",
        "year": 2024,
        "topic": "test",
        "page_count": 10,
        "body_pages": 9,
        "removed_references": True,
        "token_estimate": 5000,
        "truncated": False,
        "truncation_rate": 0.0,
    }


def _manifest(count: int, dimension: int = 2) -> dict:
    return {
        "schema_version": 2,
        "model_name": "test-model",
        "dimension": dimension,
        "max_seq_length": 512,
        "normalized": True,
        "similarity": "cosine (dot product of L2-normalised vectors)",
        "paper_count": count,
        "max_tokens": 120000,
        "papers": [_paper(f"p{i}") for i in range(count)],
    }


def _index() -> SemanticIndex:
    papers = [
        _paper("a", "Alpha Paper"),
        _paper("b", "Beta Paper"),
        _paper("c", "Gamma Paper"),
    ]
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.70710677, 0.70710677],
        ],
        dtype=np.float32,
    )
    return SemanticIndex(papers, vectors, _manifest(3))


def test_exact_search_returns_most_similar_paper() -> None:
    index = _index()

    hits = index.search_vector(np.array([3.0, 0.0]), top_k=1)

    assert len(hits) == 1
    assert hits[0].paper["key"] == "a"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_clamps_top_k_and_orders_scores() -> None:
    index = _index()

    hits = index.search_vector(np.array([1.0, 1.0]), top_k=99)

    assert len(hits) == 3
    assert hits[0].paper["key"] == "c"
    assert hits[0].score > hits[1].score


def test_zero_query_is_rejected() -> None:
    index = _index()

    with pytest.raises(IndexError):
        index.search_vector(np.array([0.0, 0.0]), top_k=1)


def test_all_scores_returns_every_paper() -> None:
    index = _index()

    hits = index.all_scores(np.array([1.0, 0.0]))

    assert len(hits) == 3
    assert hits[0].paper["key"] == "a"
    assert hits[1].paper["key"] == "c"
    assert hits[2].paper["key"] == "b"


def test_save_and_load_round_trip(tmp_path) -> None:
    index = _index()

    save_index(index.papers, index.vectors, index.manifest, tmp_path)
    loaded = SemanticIndex.load(tmp_path)

    assert loaded.dimension == 2
    assert len(loaded.papers) == 3
    assert loaded.papers[1]["key"] == "b"
    assert np.allclose(loaded.vectors, index.vectors)
