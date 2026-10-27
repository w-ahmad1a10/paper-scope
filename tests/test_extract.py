from __future__ import annotations

from paper_scope.catalog import CATALOG_BY_KEY
from paper_scope.config import MAX_CHARS
from paper_scope.extract import (
    chunk_page_text,
    clean_page_text,
    truncate_at_references,
)


def test_clean_page_text_repairs_wrapping_and_removes_artefacts() -> None:
    raw = (
        "arXiv:1810.04805v1 [cs.CL] 3 Oct 2018\n"
        "7\n"
        "This is a hy-\n"
        "phenated word [12] in a sentence.\n"
        "Published as a conference paper at ICLR 2020\n"
    )

    cleaned = clean_page_text(raw)

    assert "hyphenated" in cleaned
    assert "[12]" not in cleaned
    assert "arXiv" not in cleaned
    assert "Published as a conference paper" not in cleaned
    assert "\n" not in cleaned


def test_truncate_at_references_keeps_only_body() -> None:
    raw = "\n".join(
        [
            "Title",
            "Abstract",
            "Body one.",
            "Body two.",
            "Body three.",
            "Body four.",
            "References",
            "[1] A paper. 2020.",
        ]
    )

    kept, found = truncate_at_references(raw)

    assert found is True
    assert "Body four." in kept
    assert "References" not in kept
    assert "[1]" not in kept


def test_truncate_handles_heading_sharing_a_line_with_first_reference() -> None:
    raw = "\n".join(
        [
            "References                                      Kevin Clark, Minh-Thang Luong, 2018.",
            "Body of the first reference.",
        ]
    )

    kept, found = truncate_at_references(raw)

    assert found is True
    assert kept == ""


def test_chunking_uses_sentence_boundaries_and_respects_maximum() -> None:
    sentence = (
        "Masked language modeling asks the model to recover randomly hidden "
        "tokens from the surrounding context. "
    )
    text = " ".join([sentence.strip()] * 24)
    paper = CATALOG_BY_KEY["bert"]

    paragraphs = chunk_page_text(text, paper, page=3)

    assert len(paragraphs) >= 3
    assert all(paragraph.paper_key == "bert" for paragraph in paragraphs)
    assert all(paragraph.page == 3 for paragraph in paragraphs)
    assert all(len(paragraph.text) <= MAX_CHARS for paragraph in paragraphs)
    assert all(paragraph.text.endswith(".") for paragraph in paragraphs)
    assert [paragraph.id for paragraph in paragraphs] == [
        f"bert:p003:c{index:02d}" for index in range(len(paragraphs))
    ]


def test_short_page_tail_is_dropped_or_absorbed() -> None:
    text = "A useful sentence about pretraining. " * 20 + "Tiny tail."
    paper = CATALOG_BY_KEY["electra"]

    paragraphs = chunk_page_text(text, paper, page=1)

    assert paragraphs
    assert all(len(paragraph.text) >= 250 for paragraph in paragraphs)
