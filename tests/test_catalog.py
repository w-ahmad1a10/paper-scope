from __future__ import annotations

from paper_scope.catalog import PaperEntry, load_collection, save_collection


def test_save_and_load_collection_round_trip(tmp_path) -> None:
    collection_file = tmp_path / "collection.json"
    entries = [
        PaperEntry(
            key="bert",
            title="BERT Paper",
            path="/tmp/bert.pdf",
            arxiv_id="1810.04805",
            year=2018,
            topic="Bidirectional pretraining",
        ),
        PaperEntry(
            key="electra",
            title="ELECTRA Paper",
            path="/tmp/electra.pdf",
            arxiv_id="2003.10555",
            year=2020,
            topic="Sample-efficient pretraining",
        ),
    ]

    save_collection(entries, collection_file)
    loaded = load_collection(collection_file)

    assert len(loaded) == 2
    assert loaded[0].key == "bert"
    assert loaded[0].title == "BERT Paper"
    assert loaded[0].arxiv_id == "1810.04805"
    assert loaded[1].key == "electra"
    assert loaded[1].year == 2020


def test_load_collection_when_missing_returns_empty(tmp_path) -> None:
    collection_file = tmp_path / "nonexistent.json"
    entries = load_collection(collection_file)
    assert entries == []
