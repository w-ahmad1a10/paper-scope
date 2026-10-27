"""Sentence-embedding model loading and batch encoding."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .config import (
    CACHE_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_QUERY_PREFIX,
)


def _can_write_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _configure_model_cache() -> None:
    """Keep model files in the project when a global cache path is broken."""

    fallback = CACHE_DIR / "huggingface"
    if not _can_write_directory(fallback):
        # Let the Hugging Face libraries report a useful error themselves.
        return

    hub_fallback = fallback / "hub"
    _can_write_directory(hub_fallback)

    for variable in ("HF_HOME", "SENTENCE_TRANSFORMERS_HOME"):
        configured = os.environ.get(variable)
        if not configured or not _can_write_directory(Path(configured).expanduser()):
            os.environ[variable] = str(fallback)

    configured_hub = os.environ.get("HF_HUB_CACHE")
    if not configured_hub or not _can_write_directory(
        Path(configured_hub).expanduser()
    ):
        os.environ["HF_HUB_CACHE"] = str(hub_fallback)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_configure_model_cache()


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


def query_prefix_for_model(model_name: str) -> str:
    if "bge" in model_name.lower():
        return DEFAULT_QUERY_PREFIX
    return ""


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "The sentence-transformers package is not installed"
            ) from exc

        self.model_name = model_name
        self.query_prefix = query_prefix_for_model(model_name)
        self.batch_size = batch_size
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model '{model_name}': {exc}"
            ) from exc

        get_dimension = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self.dimension = int(get_dimension())
        self.max_seq_length = int(self._model.max_seq_length)

    @property
    def device(self) -> str:
        return str(self._model.device)

    def encode(
        self,
        texts: Sequence[str],
        *,
        show_progress_bar: bool = False,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if not texts:
            raise EmbeddingError("Cannot encode an empty list of texts")

        try:
            vectors = self._model.encode(
                list(texts),
                batch_size=batch_size or self.batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise EmbeddingError(
                f"Unexpected embedding shape {matrix.shape}; expected (*, {self.dimension})"
            )
        if not np.isfinite(matrix).all():
            raise EmbeddingError("Embedding model produced non-finite values")
        return np.ascontiguousarray(matrix)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
