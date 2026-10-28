"""High-level semantic search over the saved paper index."""

from __future__ import annotations

from pathlib import Path

from .config import INDEX_DIR
from .embed import Embedder
from .index_store import SearchHit, SemanticIndex


class Searcher:
    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self.index = SemanticIndex.load(index_dir)
        self.embedder = Embedder(self.index.model_name)

    def search(self, query: str, top_k: int = 1) -> list[SearchHit]:
        """Return the top_k most similar papers to the question."""

        query = " ".join(query.split())
        if not query:
            raise ValueError("The question is empty")
        query_vector = self.embedder.encode_one(query)
        return self.index.search_vector(query_vector, top_k=top_k)

    def search_all(self, query: str) -> list[SearchHit]:
        """Return every paper ranked by similarity to the question."""

        query = " ".join(query.split())
        if not query:
            raise ValueError("The question is empty")
        query_vector = self.embedder.encode_one(query)
        return self.index.all_scores(query_vector)

    @property
    def manifest(self) -> dict:
        return self.index.manifest
