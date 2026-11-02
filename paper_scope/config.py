"""Project-wide settings and filesystem locations."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.environ.get("PAPER_SCOPE_DATA", str(PROJECT_ROOT / "data"))
).expanduser().resolve()
PAPERS_DIR = DATA_DIR / "papers"
TEXT_DIR = DATA_DIR / "text"
INDEX_DIR = DATA_DIR / "index"
COLLECTION_FILE = DATA_DIR / "collection.json"
CACHE_DIR = DATA_DIR / "cache"

PAPER_RECORDS_FILE = INDEX_DIR / "papers.jsonl"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
MANIFEST_FILE = INDEX_DIR / "manifest.json"

DEFAULT_MODEL = os.environ.get(
    "PAPER_SCOPE_MODEL", "Alibaba-NLP/gte-modernbert-base"
)
DEFAULT_BATCH_SIZE = 32

# The embedding model must have a context window large enough for whole papers.
# If a paper exceeds this token count, the tail is cut off and the loss is reported.
DEFAULT_MAX_TOKENS = 120_000

WEAK_MATCH_SCORE = 0.68
SCHEMA_VERSION = 2
