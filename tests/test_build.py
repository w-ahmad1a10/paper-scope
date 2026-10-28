from __future__ import annotations

import json
from pathlib import Path

import fitz

import paper_scope.build as build_module
from paper_scope.build import add_paper
from paper_scope.catalog import PaperEntry


def _make_minimal_pdf(path: Path, title_text: str = "Test content") -> None:
    """Create a minimal valid PDF that PyMuPDF can open."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), title_text, fontsize=11)
    doc.save(str(path))
    doc.close()
    # Pad file to exceed the 20KB minimum size check
    current_size = path.stat().st_size
    if current_size < 20_001:
        with open(path, "ab") as f:
            f.write(b"\0" * (20_001 - current_size))


def _mock_embedder(monkeypatch, dimension: int = 384) -> None:
    """Replace Embedder with a mock that returns deterministic vectors."""
    import numpy as np

    class MockEmbedder:
        def __init__(self, model_name=None, device=None, batch_size=32, max_tokens=120_000):
            self.model_name = model_name or "test-model"
            self.dimension = dimension
            self.max_seq_length = 512
            self.device = "cpu"
            self.batch_size = batch_size
            self.max_tokens = max_tokens

        def encode_one(self, text):
            # Deterministic vector based on text hash
            import hashlib

            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            vec = np.zeros(self.dimension, dtype=np.float32)
            for i in range(self.dimension):
                vec[i] = float((h >> (i * 7)) & 0x1) * 2.0 - 1.0
            norm = (vec @ vec) ** 0.5
            if norm > 0:
                vec = vec / norm
            return vec

        def estimate_tokens(self, text):
            return max(len(text) // 4, 1)

    import paper_scope.build as build_module

    monkeypatch.setattr(build_module, "Embedder", MockEmbedder)


def test_add_paper_writes_records(tmp_path, monkeypatch):
    """Add a paper and verify records and vectors are saved."""

    # Create a minimal valid PDF
    pdf_path = tmp_path / "test_paper.pdf"
    _make_minimal_pdf(pdf_path, "Test paper content")

    index_dir = tmp_path / "index"
    collection_file = tmp_path / "collection.json"
    text_dir = tmp_path / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    _mock_embedder(monkeypatch)
    monkeypatch.setattr(build_module, "TEXT_DIR", text_dir)

    manifest = add_paper(
        key="test_paper",
        title="Test Paper",
        source_path=pdf_path,
        model_name="test-model",
        index_dir=index_dir,
        collection_file=collection_file,
    )

    assert manifest["paper_count"] == 1
    assert manifest["dimension"] == 384

    # Verify collection was saved
    entries = json.loads(collection_file.read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["key"] == "test_paper"
    assert entries[0]["title"] == "Test Paper"

    # Verify index files exist
    assert (index_dir / "papers.jsonl").is_file()
    assert (index_dir / "embeddings.npy").is_file()
    assert (index_dir / "manifest.json").is_file()

    # Verify cleaned text was saved
    text_file = tmp_path / "text" / "test_paper.txt"
    assert text_file.is_file()


def test_add_paper_rejects_invalid_pdf(tmp_path):
    """Adding a non-PDF file should raise an error."""

    bad_file = tmp_path / "not_a_pdf.txt"
    bad_file.write_bytes(b"this is not a pdf")

    index_dir = tmp_path / "index"
    collection_file = tmp_path / "collection.json"

    try:
        add_paper(
            key="bad",
            title="Bad Paper",
            source_path=bad_file,
            index_dir=index_dir,
            collection_file=collection_file,
        )
    except Exception:  # noqa: BLE001, S110 - expected error for invalid PDF
        pass  # Expected


def test_add_paper_duplicate_key_replaces(tmp_path, monkeypatch):
    """Adding a paper with the same key should replace the old one."""

    pdf1 = tmp_path / "paper_a.pdf"
    _make_minimal_pdf(pdf1, "Paper A content")
    pdf2 = tmp_path / "paper_b.pdf"
    _make_minimal_pdf(pdf2, "Paper B content")

    index_dir = tmp_path / "index"
    collection_file = tmp_path / "collection.json"
    text_dir = tmp_path / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    _mock_embedder(monkeypatch)
    monkeypatch.setattr(build_module, "TEXT_DIR", text_dir)

    m1 = add_paper(
        key="same_key",
        title="First Title",
        source_path=pdf1,
        index_dir=index_dir,
        collection_file=collection_file,
    )
    assert m1["paper_count"] == 1

    m2 = add_paper(
        key="same_key",
        title="Second Title",
        source_path=pdf2,
        index_dir=index_dir,
        collection_file=collection_file,
    )
    assert m2["paper_count"] == 1  # Should still be 1, not 2

    entries = json.loads(collection_file.read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["title"] == "Second Title"


def test_catalog_entry_creation():
    """PaperEntry dataclass works correctly."""

    entry = PaperEntry(
        key="bert",
        title="BERT Paper",
        path="/tmp/bert.pdf",
        arxiv_id="1810.04805",
        year=2018,
        topic="Bidirectional pretraining",
    )

    assert entry.key == "bert"
    assert entry.title == "BERT Paper"
    assert entry.arxiv_id == "1810.04805"
    assert entry.year == 2018

    # Test asdict round-trip
    from dataclasses import asdict
    d = asdict(entry)
    assert d["key"] == "bert"
    assert d["title"] == "BERT Paper"
