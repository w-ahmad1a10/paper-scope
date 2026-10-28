"""Turn full paper PDFs into clean running text with references removed."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf
except ImportError:  # Older PyMuPDF releases expose only the fitz name.
    import fitz as pymupdf  # type: ignore[no-redefinition]


class ExtractionError(RuntimeError):
    """Raised when a PDF cannot be read or yields no usable text."""


@dataclass
class ExtractedPaper:
    text: str
    page_count: int
    body_pages: int
    removed_references: bool


_REFERENCE_HEADING = re.compile(
    r"^\s*(?:\d{1,2}\s+)?(references|bibliography|works cited)\s*$", re.IGNORECASE
)
_REFERENCE_PREFIX = re.compile(
    r"^\s*(?:\d{1,2}\s+)?(references|bibliography|works cited)\b", re.IGNORECASE
)
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.IGNORECASE)
_ARXIV_STAMP = re.compile(
    r"^\s*arxiv:\s*\d{4}\.\d{4,5}(?:v\d+)?(?:\s*\[[^\]]+\])?(?:\s+\d{1,2}\s+\w+\s+\d{4})?\s*$",
    re.IGNORECASE,
)
_BOILERPLATE = re.compile(
    r"^\s*(?:preprint\.?\s*(?:under review)?|published as a conference paper at .+)$",
    re.IGNORECASE,
)
_CITATION = re.compile(r"\[\s*\d+(?:\s*[-,–]\s*\d+)*\s*\]")


def _normalise_characters(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u00ad": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb00": "ff",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def truncate_at_references(raw_page: str) -> tuple[str, bool]:
    """Cut a page at the start of its references section, if present."""

    lines = raw_page.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _REFERENCE_HEADING.fullmatch(stripped):
            return "\n".join(lines[:index]), True

        prefix = _REFERENCE_PREFIX.match(stripped)
        if prefix:
            remainder = stripped[prefix.end() :]
            tail = remainder.strip()
            # PDF extraction sometimes puts the first reference on the same
            # line as the heading, separated only by layout spaces.
            if (
                not tail
                or len(remainder) - len(remainder.lstrip()) >= 2
                or re.search(r"\b(?:19|20)\d{2}\b", tail)
            ):
                return "\n".join(lines[:index]), True
    return raw_page, False


def clean_page_text(raw_page: str) -> str:
    """Remove PDF artefacts and rejoin wrapped lines into running text."""

    text = _normalise_characters(raw_page)

    # Join words that were split with a hyphen at the end of a printed line.
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)

    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _PAGE_NUMBER.fullmatch(stripped):
            continue
        if _ARXIV_STAMP.fullmatch(stripped):
            continue
        if _BOILERPLATE.fullmatch(stripped):
            continue
        if len(stripped) <= 2 and not stripped.isalnum():
            continue
        kept_lines.append(stripped)

    text = " ".join(kept_lines)
    text = _CITATION.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_paper(pdf_path: Path) -> ExtractedPaper:
    """Read a PDF, stop before the references, and return clean full text."""

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:  # PyMuPDF raises several unrelated exception types.
        raise ExtractionError(f"Could not open {pdf_path.name}: {exc}") from exc

    try:
        if document.needs_pass:
            raise ExtractionError(f"{pdf_path.name} is password protected")
        if document.page_count == 0:
            raise ExtractionError(f"{pdf_path.name} has no pages")

        cleaned_pages: list[tuple[int, str]] = []
        removed_references = False

        for page_index in range(document.page_count):
            raw = document.load_page(page_index).get_text("text", sort=False)
            if not raw.strip():
                continue
            raw, found_references = truncate_at_references(raw)
            cleaned = clean_page_text(raw)
            if cleaned:
                cleaned_pages.append((page_index + 1, cleaned))
            if found_references:
                removed_references = True
                break

        if not cleaned_pages:
            raise ExtractionError(
                f"No selectable text found in {pdf_path.name}; it may be a scanned PDF"
            )

        body_text = "\n\n".join(page_text for _, page_text in cleaned_pages)
        return ExtractedPaper(
            text=body_text,
            page_count=document.page_count,
            body_pages=len(cleaned_pages),
            removed_references=removed_references,
        )
    finally:
        document.close()
