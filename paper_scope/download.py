"""Download paper PDFs from arXiv with retries and basic validation."""

from __future__ import annotations

import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .catalog import Paper
from .config import PAPERS_DIR

USER_AGENT = "paper-scope/0.1 (local semantic-search research tool)"
MIN_PDF_BYTES = 20_000


class DownloadError(RuntimeError):
    """Raised when a paper cannot be downloaded or is not a usable PDF."""


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,*/*;q=0.8",
        }
    )
    return session


def is_valid_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < MIN_PDF_BYTES:
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def download_pdf(
    paper: Paper,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> tuple[Path, bool]:
    """Return the local PDF path and whether it was downloaded in this call."""

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPERS_DIR / f"{paper.key}.pdf"

    if not force and is_valid_pdf(target):
        return target, False

    client = session or _session()
    temporary = target.with_suffix(".pdf.part")
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            with client.get(
                paper.pdf_url,
                stream=True,
                timeout=(20, 180),
                allow_redirects=True,
            ) as response:
                if response.status_code != 200:
                    raise DownloadError(
                        f"arXiv returned HTTP {response.status_code} for {paper.arxiv_id}"
                    )

                expected = response.headers.get("Content-Length")
                total = int(expected) if expected and expected.isdigit() else None
                written = 0
                with temporary.open("wb") as handle:
                    for block in response.iter_content(chunk_size=256 * 1024):
                        if not block:
                            continue
                        handle.write(block)
                        written += len(block)
                        if total and written > 80 * 1024 * 1024:
                            raise DownloadError(
                                f"PDF for {paper.arxiv_id} is unexpectedly large"
                            )

            if not is_valid_pdf(temporary):
                raise DownloadError(
                    f"Downloaded file for {paper.arxiv_id} is not a valid PDF"
                )

            temporary.replace(target)
            return target, True
        except (requests.RequestException, OSError, DownloadError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2.0 * attempt)

    raise DownloadError(f"Could not download {paper.arxiv_id}: {last_error}")


def download_all(
    papers: tuple[Paper, ...],
    *,
    force: bool = False,
    printer=print,
) -> list[Path]:
    client = _session()
    paths: list[Path] = []
    for position, paper in enumerate(papers, start=1):
        printer(f"[{position}/{len(papers)}] {paper.key:11s} {paper.arxiv_id}")
        path, downloaded = download_pdf(paper, force=force, session=client)
        size_mb = path.stat().st_size / (1024 * 1024)
        state = "downloaded" if downloaded else "cached"
        printer(f"             {state:10s} {size_mb:5.1f} MB")
        paths.append(path)
    return paths
