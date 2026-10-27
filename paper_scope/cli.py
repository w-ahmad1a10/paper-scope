"""Command-line interface for building and querying the paper index."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import traceback

from .build import build_index
from .catalog import CATALOG
from .config import DEFAULT_MODEL, INDEX_DIR, WEAK_MATCH_SCORE
from .download import DownloadError
from .embed import EmbeddingError
from .extract import ExtractionError
from .index_store import IndexStoreError, SearchHit
from .search import Searcher


def _print_hit(hit: SearchHit, rank: int = 1) -> None:
    paragraph = hit.paragraph
    print()
    print(f"Match {rank}  |  similarity {hit.score:.3f}")
    print(
        f"{paragraph.paper_title} ({paragraph.year})  "
        f"|  arXiv {paragraph.arxiv_id}  |  page {paragraph.page}"
    )
    print("-" * 88)
    print(textwrap.fill(paragraph.text, width=88))
    if hit.score < WEAK_MATCH_SCORE:
        print()
        print(
            "Note: this is a weak match. The collection may not contain a direct answer."
        )


def _hit_to_dict(hit: SearchHit) -> dict:
    paragraph = hit.paragraph
    return {
        "score": round(hit.score, 6),
        "paper_key": paragraph.paper_key,
        "title": paragraph.paper_title,
        "year": paragraph.year,
        "arxiv_id": paragraph.arxiv_id,
        "page": paragraph.page,
        "paragraph": paragraph.text,
    }


def _load_searcher() -> Searcher:
    print("Loading the embedding model and saved vectors...")
    return Searcher()


def _ensure_searcher() -> Searcher:
    try:
        return _load_searcher()
    except IndexStoreError:
        print()
        print("No usable index was found, so the collection will be built now.")
        print("This downloads the papers and the embedding model once.")
        print()
        build_index()
        return _load_searcher()


def cmd_build(args: argparse.Namespace) -> int:
    build_index(
        limit=args.limit,
        force_download=args.force_download,
        model_name=args.model,
        batch_size=args.batch_size,
        index_dir=INDEX_DIR,
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    searcher = _load_searcher()
    hits = searcher.search(args.query, top_k=args.top)

    if args.json:
        payload = {
            "query": args.query,
            "model": searcher.index.model_name,
            "results": [_hit_to_dict(hit) for hit in hits],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    for rank, hit in enumerate(hits, start=1):
        _print_hit(hit, rank)
    return 0


def _print_welcome(searcher: Searcher) -> None:
    manifest = searcher.manifest
    papers = manifest.get("papers", [])
    print()
    print("PaperScope")
    print("=" * 88)
    print(
        f"{manifest.get('paragraph_count', 0)} paragraphs from "
        f"{manifest.get('paper_count', len(papers))} AI research papers."
    )
    print(f"Embedding model: {searcher.index.model_name}")
    print("Ask a question in ordinary English. Type /quit to stop.")
    print()


def cmd_ask(args: argparse.Namespace) -> int:
    searcher = _ensure_searcher()
    _print_welcome(searcher)

    while True:
        try:
            query = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in {"/quit", "quit", "exit", "/exit"}:
            break
        if query.lower() in {"/help", "help", "?"}:
            print("Type a question, or /quit to stop.")
            continue

        try:
            hits = searcher.search(query, top_k=1)
        except Exception as exc:  # noqa: BLE001 - one failed query must not end the loop.
            print(f"Could not answer that question: {exc}")
            continue

        _print_hit(hits[0])
        print()

    return 0


def cmd_papers(args: argparse.Namespace) -> int:
    print(f"{len(CATALOG)} papers in the collection:")
    print()
    for paper in sorted(CATALOG, key=lambda item: (item.year, item.title)):
        print(f"{paper.year}  {paper.arxiv_id:12s}  {paper.title}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-scope",
        description=(
            "Semantic search over paragraphs from famous AI research papers. "
            "Run without a command for an interactive question loop."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser(
        "build", help="Download papers, extract paragraphs, and compute vectors"
    )
    build.add_argument(
        "--force-download",
        action="store_true",
        help="Download every PDF again even when a cached copy exists",
    )
    build.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N papers (useful for a quick trial)",
    )
    build.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face embedding model (default: {DEFAULT_MODEL})",
    )
    build.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of paragraphs encoded at once (default: 32)",
    )
    build.set_defaults(func=cmd_build)

    search = subparsers.add_parser("search", help="Search the saved index once")
    search.add_argument("query", help="Question or topic to search for")
    search.add_argument(
        "--top", type=int, default=1, help="Number of paragraphs to return (default: 1)"
    )
    search.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON"
    )
    search.set_defaults(func=cmd_search)

    ask = subparsers.add_parser("ask", help="Open the interactive question loop")
    ask.set_defaults(func=cmd_ask)

    papers = subparsers.add_parser("papers", help="List the papers in the collection")
    papers.set_defaults(func=cmd_papers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        args = argparse.Namespace(func=cmd_ask)

    try:
        result = args.func(args)
        return 0 if result is None else int(result)
    except KeyboardInterrupt:
        print()
        return 130
    except (DownloadError, ExtractionError, EmbeddingError, IndexStoreError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - final user-facing error boundary.
        if os.environ.get("PAPER_SCOPE_DEBUG"):
            traceback.print_exc()
        else:
            print(f"Unexpected error: {exc}", file=sys.stderr)
            print("Set PAPER_SCOPE_DEBUG=1 to see technical details.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
