"""Turn arXiv PDFs into clean, sentence-aligned paragraph chunks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf
except ImportError:  # Older PyMuPDF releases expose only the fitz name.
    import fitz as pymupdf

from .catalog import Paper
from .config import MAX_CHARS, MIN_CHARS, TARGET_CHARS


class ExtractionError(RuntimeError):
    """Raised when a PDF cannot be read or yields no usable text."""


@dataclass(frozen=True)
class Paragraph:
    id: str
    paper_key: str
    paper_title: str
    arxiv_id: str
    year: int
    page: int
    text: str


@dataclass
class ExtractedPaper:
    paper: Paper
    text: str
    page_count: int
    body_pages: int
    paragraphs: list[Paragraph]


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
_ABBREVIATION_END = re.compile(
    r"(?:\b(?:e\.g|i\.e|et al|fig|figs|eq|eqs|sec|secs|no|vs|cf|approx|ref|refs|dr|mr|ms|prof)\.|(?:\b[A-Z]\.){2,})$",
    re.IGNORECASE,
)


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


def _split_sentences(text: str) -> list[str]:
    """Split on sentence endings while protecting common abbreviations."""

    pieces = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“(])', text)
    sentences: list[str] = []
    buffer = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{buffer} {piece}".strip() if buffer else piece
        if _ABBREVIATION_END.search(piece):
            buffer = candidate
            continue
        sentences.append(candidate)
        buffer = ""

    if buffer:
        sentences.append(buffer)
    return sentences


def _hard_split(text: str, maximum: int) -> list[str]:
    words = text.split()
    parts: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > maximum:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _build_chunks(
    sentences: list[str],
    *,
    target: int,
    maximum: int,
    minimum: int,
) -> list[str]:
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > maximum:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(sentence, maximum))
            continue

        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    if (
        len(chunks) >= 2
        and len(chunks[-1]) < minimum
        and len(chunks[-2]) + 1 + len(chunks[-1]) <= maximum
    ):
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}"
        chunks.pop()

    return chunks


def _is_useful(text: str, minimum: int) -> bool:
    if len(text) < minimum:
        return False
    letters = sum(character.isalpha() for character in text)
    if letters < 160:
        return False
    return letters / max(len(text), 1) >= 0.35


def _dedup_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())[:180]


def chunk_page_text(
    text: str,
    paper: Paper,
    page: int,
    *,
    target_chars: int = TARGET_CHARS,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[Paragraph]:
    """Split one cleaned page into paragraph-sized, sentence-aligned chunks."""

    raw_chunks = _build_chunks(
        _split_sentences(text),
        target=target_chars,
        maximum=max_chars,
        minimum=min_chars,
    )

    paragraphs: list[Paragraph] = []
    for index, chunk in enumerate(raw_chunks):
        if not _is_useful(chunk, min_chars):
            continue
        paragraphs.append(
            Paragraph(
                id=f"{paper.key}:p{page:03d}:c{index:02d}",
                paper_key=paper.key,
                paper_title=paper.title,
                arxiv_id=paper.arxiv_id,
                year=paper.year,
                page=page,
                text=chunk,
            )
        )
    return paragraphs


def extract_paper(pdf_path: Path, paper: Paper) -> ExtractedPaper:
    """Read a PDF, stop before the references, and produce paragraph chunks."""

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
        for page_index in range(document.page_count):
            raw = document.load_page(page_index).get_text("text", sort=False)
            if not raw.strip():
                continue
            raw, found_references = truncate_at_references(raw)
            cleaned = clean_page_text(raw)
            if cleaned:
                cleaned_pages.append((page_index + 1, cleaned))
            if found_references:
                break

        if not cleaned_pages:
            raise ExtractionError(
                f"No selectable text found in {pdf_path.name}; it may be a scanned PDF"
            )

        paragraphs: list[Paragraph] = []
        seen: set[str] = set()
        for page_number, page_text in cleaned_pages:
            for paragraph in chunk_page_text(page_text, paper, page_number):
                key = _dedup_key(paragraph.text)
                if key in seen:
                    continue
                seen.add(key)
                paragraphs.append(paragraph)

        if not paragraphs:
            raise ExtractionError(f"No usable text chunks found in {pdf_path.name}")

        body_text = "\n\n".join(page_text for _, page_text in cleaned_pages)
        return ExtractedPaper(
            paper=paper,
            text=body_text,
            page_count=document.page_count,
            body_pages=len(cleaned_pages),
            paragraphs=paragraphs,
        )
    finally:
        document.close()
