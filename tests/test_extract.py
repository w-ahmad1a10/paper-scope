from __future__ import annotations

from paper_scope.extract import clean_page_text, truncate_at_references


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
