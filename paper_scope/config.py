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
CACHE_DIR = DATA_DIR / "cache"

PAPERS_FILE = PAPERS_DIR / "papers.json"
PARAGRAPHS_FILE = INDEX_DIR / "paragraphs.jsonl"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
MANIFEST_FILE = INDEX_DIR / "manifest.json"

DEFAULT_MODEL = os.environ.get(
    "PAPER_SCOPE_MODEL", "BAAI/bge-small-en-v1.5"
)
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_BATCH_SIZE = 32

# Paragraphs are built from complete sentences, aiming for this many characters.
# The embedding model has a 256-token limit, so chunks are kept comfortably below it.
TARGET_CHARS = 650
MAX_CHARS = 900
MIN_CHARS = 250

WEAK_MATCH_SCORE = 0.68
SCHEMA_VERSION = 1
