"""Validate and cache local paper PDFs."""

from __future__ import annotations

from pathlib import Path

from .config import PAPERS_DIR

MIN_PDF_BYTES = 20_000


class DownloadError(RuntimeError):
    """Raised when a local PDF cannot be used."""


def is_valid_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < MIN_PDF_BYTES:
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def copy_pdf(source: Path, key: str) -> Path:
    """Copy a local PDF into the project papers folder under a safe key name."""

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPERS_DIR / f"{key}.pdf"

    if not is_valid_pdf(source):
        raise DownloadError(
            f"The file at {source} is not a usable PDF or is too small"
        )

    try:
        data = source.read_bytes()
    except OSError as exc:
        raise DownloadError(f"Could not read {source}: {exc}") from exc

    target.write_bytes(data)
    if not is_valid_pdf(target):
        raise DownloadError(f"Copied file for {key} is not a valid PDF")

    return target
