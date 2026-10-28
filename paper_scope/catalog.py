"""The paper collection is built by the user, not fixed in code."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import COLLECTION_FILE


@dataclass
class PaperEntry:
    key: str
    title: str
    path: str
    arxiv_id: str | None = None
    year: int | None = None
    topic: str | None = None


def load_collection(collection_file: Path = COLLECTION_FILE) -> list[PaperEntry]:
    if not collection_file.is_file():
        return []
    try:
        data = json.loads(collection_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries: list[PaperEntry] = []
    for row in data:
        try:
            entries.append(PaperEntry(
                key=str(row["key"]),
                title=str(row["title"]),
                path=str(row["path"]),
                arxiv_id=row.get("arxiv_id"),
                year=row.get("year"),
                topic=row.get("topic"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return entries


def save_collection(
    entries: list[PaperEntry],
    collection_file: Path = COLLECTION_FILE,
) -> None:
    collection_file.parent.mkdir(parents=True, exist_ok=True)
    collection_file.write_text(
        json.dumps(
            [asdict(entry) for entry in entries],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
