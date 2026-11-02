"""Sentence-embedding model loading and whole-document encoding."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .config import CACHE_DIR, DEFAULT_BATCH_SIZE, DEFAULT_MAX_TOKENS, DEFAULT_MODEL


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


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


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "The sentence-embedding package is not installed"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        try:
            self._model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.float32, "rotary_scaling_factor": 2},
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model '{model_name}': {exc}"
            ) from exc

        # Determine actual embedding dimension by encoding a test string,
        # since get_embedding_dimension() may not match the actual output
        # for some models (e.g., nomic-embed-text reports a nominal size that
        # differs from the real output width).
        test_vec = self._model.encode(
            ["search_document: dimension check"],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        test_matrix = np.asarray(test_vec, dtype=np.float32)
        if test_matrix.ndim == 1:
            test_matrix = test_matrix[None, :]
        self.dimension = int(test_matrix.shape[1])
        self.max_seq_length = int(self._model.max_seq_length)

    @property
    def device(self) -> str:
        return str(self._model.device)

    def encode_one(self, text: str, prefix: str = "") -> np.ndarray:
        """Encode one whole document and return its vector.

        If prefix is given (e.g. "search_document: " or "search_query: "),
        it is prepended to the text before encoding, as required by the model.
        """

        if prefix:
            text = f"{prefix}{text}"
        vectors = self.encode_one_batch([text], prefix="")
        return np.ascontiguousarray(vectors)[0]

    def encode_one_batch(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """Encode many whole documents in small batches to avoid memory exhaustion.

        If prefix is given (e.g. "search_document: "), it is prepended to
        every text before encoding, as required by the model.

        Returns a 2-D array of shape (len(texts), dimension).
        """

        if prefix:
            texts = [f"{prefix}{t}" for t in texts]

        all_vectors: list[np.ndarray] = []
        chunk = max(1, min(4, self.batch_size))  # small chunks to avoid OOM
        total_chunks = (len(texts) + chunk - 1) // chunk

        for chunk_idx, start in enumerate(range(0, len(texts), chunk), start=1):
            piece = texts[start : start + chunk]
            print(
                f"  Encoding chunk {chunk_idx}/{total_chunks} "
                f"({len(piece)} texts, {start + len(piece)}/{len(texts)} done)",
                flush=True,
            )
            try:
                vectors = self._model.encode(
                    piece,
                    batch_size=chunk,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    truncate_dim=self.dimension,
                )
            except TypeError:
                vectors = self._model.encode(
                    piece,
                    batch_size=chunk,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

            matrix = np.asarray(vectors, dtype=np.float32)
            if matrix.ndim == 1:
                matrix = matrix[None, :]
            all_vectors.append(matrix)

        if not all_vectors:
            raise EmbeddingError("No vectors produced")

        matrix = np.concatenate(all_vectors, axis=0)
        if matrix.shape[1] != self.dimension:
            raise EmbeddingError(
                f"Unexpected embedding shape {matrix.shape}; expected (*, {self.dimension})"
            )
        if not np.isfinite(matrix).all():
            raise EmbeddingError("Embedding model produced non-finite values")
        return np.ascontiguousarray(matrix)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using the model tokenizer when available."""

        try:
            tokenizer = self._model.tokenizer
        except AttributeError:
            return max(len(text) // 4, 1)

        try:
            tokens = tokenizer(text, return_tensors=None, add_special_tokens=True)
            ids = tokens["input_ids"] if isinstance(tokens, dict) else tokens
            if isinstance(ids, list):
                return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
            return int(ids.shape[-1])
        except Exception:  # noqa: BLE001 - tokenizer fallback for missing tokenizer.
            return max(len(text) // 4, 1)
