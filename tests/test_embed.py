from __future__ import annotations

from paper_scope.config import DEFAULT_QUERY_PREFIX
from paper_scope.embed import query_prefix_for_model


def test_bge_models_get_the_query_instruction() -> None:
    prefix = query_prefix_for_model("BAAI/bge-small-en-v1.5")

    assert prefix == DEFAULT_QUERY_PREFIX
    assert prefix.endswith("passages: ")


def test_other_models_do_not_get_a_query_instruction() -> None:
    assert query_prefix_for_model("sentence-transformers/all-MiniLM-L6-v2") == ""
