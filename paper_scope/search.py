"""High-level semantic search over the saved paper index."""

from __future__ import annotations

import time
from pathlib import Path

from .config import INDEX_DIR
from .embed import Embedder
from .index_store import SearchHit, SemanticIndex


class Searcher:
    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self.index = SemanticIndex.load(index_dir)
        self.embedder = Embedder(self.index.model_name)
        self.last_encode_time: float = 0.0
        self.last_compare_time: float = 0.0

    def search(self, query: str, top_k: int = 1) -> list[SearchHit]:
        """Return the top_k most similar papers to the question."""

        query = " ".join(query.split())
        if not query:
            raise ValueError("The question is empty")

        encode_start = time.perf_counter()
        query_vector = self.embedder.encode_one(query, prefix="search_query: ")
        self.last_encode_time = time.perf_counter() - encode_start

        compare_start = time.perf_counter()
        results = self.index.search_vector(query_vector, top_k=top_k)
        self.last_compare_time = time.perf_counter() - compare_start

        return results

    def search_all(self, query: str) -> list[SearchHit]:
        """Return every paper ranked by similarity to the question."""

        query = " ".join(query.split())
        if not query:
            raise ValueError("The question is empty")

        encode_start = time.perf_counter()
        query_vector = self.embedder.encode_one(query, prefix="search_query: ")
        self.last_encode_time = time.perf_counter() - encode_start

        compare_start = time.perf_counter()
        results = self.index.all_scores(query_vector)
        self.last_compare_time = time.perf_counter() - compare_start

        return results

    @property
    def manifest(self) -> dict:
        return self.index.manifest

    @property
    def last_total_time(self) -> float:
        """Total time for the last search, in seconds."""
        return self.last_encode_time + self.last_compare_time
