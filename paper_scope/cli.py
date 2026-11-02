"""Command-line interface for whole-paper semantic search."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import traceback
from pathlib import Path

from .batch import batch_add
from .build import add_paper, build_index
from .config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, INDEX_DIR, WEAK_MATCH_SCORE
from .sync import sync_from_hf
from .download import DownloadError
from .embed import EmbeddingError
from .extract import ExtractionError
from .index_store import IndexError, SearchHit
from .search import Searcher


def _print_paper_hit(hit: SearchHit, rank: int = 1, show_text: bool = False) -> None:
    paper = hit.paper
    print()
    print(f"Match {rank}  |  similarity {hit.score:.3f}")
    print(f"{paper['title']}")
    if paper.get("year"):
        print(f"Year: {paper['year']}")
    if paper.get("arxiv_id"):
        print(f"arXiv: {paper['arxiv_id']}")
    print(f"Key: {paper['key']}")
    if paper.get("truncated"):
        print(
            f"Truncated: yes (rate {paper.get('truncation_rate', 0):.4f}, "
            f"tokens {paper.get('token_estimate', 0)})"
        )
    print("-" * 88)
    if show_text and paper.get("text_path"):
        try:
            text = Path(paper["text_path"]).read_text(encoding="utf-8")
            print(textwrap.fill(text[:2000], width=88))
            if len(text) > 2000:
                print("... (text truncated for display)")
        except OSError:
            print("(Could not load paper text for display)")
    if hit.score < WEAK_MATCH_SCORE:
        print()
        print(
            "Note: this is a weak match. The collection may not contain a direct answer."
        )


def _print_all_scores(hits: list[SearchHit]) -> None:
    print()
    print("All papers ranked by similarity:")
    print("-" * 88)
    for rank, hit in enumerate(hits, start=1):
        paper = hit.paper
        truncated_flag = " [truncated]" if paper.get("truncated") else ""
        print(
            f"{rank:3d}. {hit.score:.3f}  {paper['title']}{truncated_flag}  "
            f"(key: {paper['key']})"
        )


def _load_searcher() -> Searcher:
    print("Loading the embedding model and saved vectors...")
    return Searcher()


def _ensure_searcher() -> Searcher:
    try:
        return _load_searcher()
    except IndexError:
        print()
        print("No usable index was found. Add papers first, then ask questions.")
        print()
        raise


def cmd_add(args: argparse.Namespace) -> int:
    """Add a paper from a local PDF file."""

    source = Path(args.path)
    title = args.name or source.stem
    key = title.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    key = "".join(c for c in key if c.isalnum() or c == "_")

    try:
        manifest = add_paper(
            title=title,
            source_path=source,
            model_name=args.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            index_dir=INDEX_DIR,
        )
    except (DownloadError, ExtractionError, EmbeddingError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Index now has {manifest['paper_count']} papers.")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Rebuild the whole index from the saved collection."""

    try:
        build_index(
            model_name=args.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            index_dir=INDEX_DIR,
        )
    except (ExtractionError, EmbeddingError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search the saved index once and show the closest paper."""

    searcher = _load_searcher()
    hits = searcher.search(args.query, top_k=args.top)

    if args.json:
        payload = {
            "query": args.query,
            "model": searcher.index.model_name,
            "results": [
                {
                    "score": round(hit.score, 6),
                    "key": hit.paper["key"],
                    "title": hit.paper["title"],
                    "year": hit.paper.get("year"),
                    "arxiv_id": hit.paper.get("arxiv_id"),
                    "page_count": hit.paper.get("page_count"),
                    "body_pages": hit.paper.get("body_pages"),
                    "truncated": hit.paper.get("truncated"),
                    "truncation_rate": hit.paper.get("truncation_rate"),
                }
                for hit in hits
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    for rank, hit in enumerate(hits, start=1):
        _print_paper_hit(hit, rank, show_text=True)
    print()
    print(
        f"Search time: encode {searcher.last_encode_time:.3f}s "
        f"| compare {searcher.last_compare_time:.3f}s "
        f"| total {searcher.last_total_time:.3f}s"
    )
    return 0


def cmd_search_all(args: argparse.Namespace) -> int:
    """Search and show scores for all papers."""

    searcher = _load_searcher()
    hits = searcher.search_all(args.query)

    if args.json:
        payload = {
            "query": args.query,
            "model": searcher.index.model_name,
            "results": [
                {
                    "score": round(hit.score, 6),
                    "key": hit.paper["key"],
                    "title": hit.paper["title"],
                    "year": hit.paper.get("year"),
                    "arxiv_id": hit.paper.get("arxiv_id"),
                    "page_count": hit.paper.get("page_count"),
                    "body_pages": hit.paper.get("body_pages"),
                    "truncated": hit.paper.get("truncated"),
                    "truncation_rate": hit.paper.get("truncation_rate"),
                }
                for hit in hits
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    _print_all_scores(hits)
    print()
    print(
        f"Search time: encode {searcher.last_encode_time:.3f}s "
        f"| compare {searcher.last_compare_time:.3f}s "
        f"| total {searcher.last_total_time:.3f}s"
    )
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Download many papers by title and build the index in one pass."""

    titles = []
    if getattr(args, "titles", None):
        titles = [t.strip() for t in args.titles.split(",") if t.strip()]
    if getattr(args, "titles_file", None):
        from pathlib import Path
        import re
        p = Path(args.titles_file)
        if p.is_file():
            titles = [t.strip() for t in p.read_text(encoding="utf-8").splitlines() if t.strip()]
            titles = [re.sub(r"^\\d+\\.\\s*", "", t) for t in titles]
            titles = [t for t in titles if t]
    if not titles:
        print("Error: no titles provided", file=sys.stderr)
        return 1

    try:
        manifest = batch_add(
            titles,
            model_name=args.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            index_dir=INDEX_DIR,
        )
    except Exception as exc:  # noqa: BLE001 - surface batch errors.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Index now has {manifest['paper_count']} papers.")
    return 0


def cmd_kaggle_run(args: argparse.Namespace) -> int:
    """Prepare and push a Kaggle kernel for batch processing."""

    import os
    import shutil
    import subprocess
    import time

    kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
    if not kaggle_token:
        print(
            "Error: KAGGLE_API_TOKEN environment variable is required.",
            file=sys.stderr,
        )
        print(
            "Set it with: $env:KAGGLE_API_TOKEN = 'your_token'",
            file=sys.stderr,
        )
        return 1

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    hf_repo = args.hf_repo or os.environ.get("HF_REPO", "w-ahmad/paper-scope-vectors")

    # Prepare kernel package directory
    kernel_dir = Path(__file__).parent.parent / "kaggle"
    if not kernel_dir.is_dir():
        print("Error: kaggle/ directory not found.", file=sys.stderr)
        print(f"  Expected at: {kernel_dir}", file=sys.stderr)
        return 1

    # Get titles from comma-separated arg
    titles = []
    if args.titles:
        titles = [t.strip() for t in args.titles.split(",") if t.strip()]
    if not titles and args.titles_file and Path(args.titles_file).is_file():
        titles = [
            t.strip() for t in Path(args.titles_file).read_text(encoding="utf-8").splitlines() if t.strip()
        ]
        titles = [re.sub(r"^\d+\.\s*", "", t) for t in titles]
        titles = [t for t in titles if t]

    if not titles:
        print("Error: no titles provided.", file=sys.stderr)
        return 1

    print(f"Pushing {len(titles)} paper(s) to Kaggle GPU")
    for t in titles:
        print(f"  - {t}")

    # Set env vars for the kaggle process (and the kernel itself)
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = kaggle_token
    env["HF_TOKEN"] = hf_token or ""
    env["HF_REPO"] = hf_repo
    env["PAPER_TITLES"] = ",".join(titles)
def cmd_sync(args: argparse.Namespace) -> int:
    """Sync vectors from Hugging Face to local index."""

    import os

    repo_id = args.repo
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if not token:
        print(
            "Error: HF_TOKEN environment variable is required.",
            file=sys.stderr,
        )
        print(
            "Set it with: $env:HF_TOKEN = 'your_token'",
            file=sys.stderr,
        )
        return 1

    try:
        manifest = sync_from_hf(
            repo_id=repo_id,
            token=token,
            index_dir=INDEX_DIR,
            printer=print,
        )
    except Exception as exc:  # noqa: BLE001 - surface sync errors.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Index now has {manifest['paper_count']} papers.")
    return 0


def cmd_papers(args: argparse.Namespace) -> int:
    """List the papers in the collection."""

    searcher = _load_searcher()
    papers = searcher.index.papers
    print(f"{len(papers)} papers in the collection:")
    print()
    for paper in papers:
        truncated_flag = " [truncated]" if paper.get("truncated") else ""
        print(
            f"{paper['key']:12s}  {paper['title']}{truncated_flag}"
        )
        if paper.get("arxiv_id"):
            print(f"{'':14s} arXiv: {paper['arxiv_id']}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Open the interactive question-and-add screen."""

    try:
        searcher = _ensure_searcher()
    except IndexError:
        return 1

    print()
    print("PaperScope - Whole Paper Search")
    print("=" * 88)
    print(f"{searcher.index.manifest.get('paper_count', 0)} papers in the collection.")
    print(f"Embedding model: {searcher.index.model_name}")
    print()
    print("Commands:")
    print("  ask <question>   - Find the closest paper and show scores for all papers")
    print("  add <pdf_path> <name>  - Add a paper from your laptop")
    print("  list             - List all papers in the collection")
    print("  quit             - Stop the program")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split(None, 1)
        command = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if command in {"/quit", "quit", "exit", "/exit"}:
            break
        elif command in {"/help", "help", "?"}:
            print(
                "Commands: ask <question>, add <pdf_path> <name>, list, quit"
            )
            continue
        elif command == "ask":
            if not rest:
                print("Please provide a question after 'ask'.")
                continue
            try:
                hits = searcher.search_all(rest)
            except Exception as exc:  # noqa: BLE001 - keep prompt alive.
                print(f"Could not answer that question: {exc}")
                continue

            _print_paper_hit(hits[0], 1, show_text=False)
            print()
            print("Closest match details above. Here are all paper scores:")
            _print_all_scores(hits)
            print()
            print(
                f"Search time: encode {searcher.last_encode_time:.3f}s "
                f"| compare {searcher.last_compare_time:.3f}s "
                f"| total {searcher.last_total_time:.3f}s"
            )
        elif command == "add":
            add_parts = rest.split(None, 1)
            if len(add_parts) < 2:
                print("Usage: add <pdf_path> <name>")
                continue
            pdf_path = Path(add_parts[0])
            name = add_parts[1]
            try:
                manifest = add_paper(
                    title=name,
                    source_path=pdf_path,
                    model_name=DEFAULT_MODEL,
                    batch_size=32,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    index_dir=INDEX_DIR,
                )
                # Reload searcher to include new paper.
                searcher = Searcher()
                print(f"Index now has {manifest['paper_count']} papers.")
            except (DownloadError, ExtractionError, EmbeddingError, OSError) as exc:
                print(f"Error: {exc}")
            continue
        elif command == "list":
            papers = searcher.index.papers
            print(f"{len(papers)} papers in the collection:")
            print()
            for paper in papers:
                truncated_flag = " [truncated]" if paper.get("truncated") else ""
                print(
                    f"{paper['key']:12s}  {paper['title']}{truncated_flag}"
                )
            print()
        else:
            print(f"Unknown command: {command}. Type /help for commands.")
            continue

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-scope",
        description=(
            "Whole-paper semantic search over AI research papers. "
            "Run without a command to see the available commands."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    add = subparsers.add_parser(
        "add", help="Add a paper from a local PDF file"
    )
    add.add_argument("path", help="Path to the PDF file")
    add.add_argument(
        "name", nargs="?", default=None, help="Display title (defaults to filename)"
    )
    add.add_argument(
        "--model", default=DEFAULT_MODEL, help="Embedding model to use"
    )
    add.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    add.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Max tokens per paper (default: 120000)"
    )
    add.set_defaults(func=cmd_add)

    build = subparsers.add_parser(
        "build", help="Rebuild the index from the saved collection"
    )
    build.add_argument(
        "--model", default=DEFAULT_MODEL, help="Embedding model to use"
    )
    build.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    build.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Max tokens per paper (default: 120000)"
    )
    build.set_defaults(func=cmd_build)

    batch = subparsers.add_parser(
        "batch", help="Download many papers by title and build the index in one pass"
    )
    batch.add_argument(
        "titles", nargs="?", default=None,
        help="Comma-separated list of paper titles to download from arXiv",
    )
    batch.add_argument(
        "--titles-file", default=None,
        help="Path to a file with one paper title per line",
    )
    batch.add_argument(
        "--model", default=DEFAULT_MODEL, help="Embedding model to use"
    )
    batch.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    batch.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Max tokens per paper (default: 120000)"
    )
    batch.set_defaults(func=cmd_batch)

    kaggle_run = subparsers.add_parser(
        "kaggle-run", help="Prepare and push a Kaggle kernel for batch processing"
    )
    kaggle_run.add_argument(
        "titles", nargs="?", default=None,
        help="Comma-separated list of paper titles (or use --titles-file)",
    )
    kaggle_run.add_argument(
        "--titles-file", default=None,
        help="Path to a file with one title per line",
    )
    kaggle_run.add_argument(
        "--hf-repo", default=None,
        help="Hugging Face repo name (default: w-ahmad/paper-scope-vectors)",
    )
    kaggle_run.set_defaults(func=cmd_kaggle_run)

    hf_sync = subparsers.add_parser(
        "hf-sync", help="Sync vectors from Hugging Face to local index"
    )
    hf_sync.add_argument(
        "repo", nargs="?", default="w-ahmad/paper-scope-vectors",
        help="Hugging Face repo name (default: w-ahmad/paper-scope-vectors)",
    )
    hf_sync.set_defaults(func=cmd_sync)

    search = subparsers.add_parser("search", help="Search the saved index once")
    search.add_argument("query", help="Question or topic to search for")
    search.add_argument(
        "--top", type=int, default=1, help="Number of papers to return (default: 1)"
    )
    search.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON"
    )
    search.set_defaults(func=cmd_search)

    search_all = subparsers.add_parser(
        "search-all", help="Search and show scores for all papers"
    )
    search_all.add_argument("query", help="Question or topic to search for")
    search_all.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON"
    )
    search_all.set_defaults(func=cmd_search_all)

    papers = subparsers.add_parser("papers", help="List papers in the collection")
    papers.set_defaults(func=cmd_papers)

    ask = subparsers.add_parser("ask", help="Open the interactive question screen")
    ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    try:
        result = args.func(args)
        return 0 if result is None else int(result)
    except KeyboardInterrupt:
        print()
        return 130
    except (DownloadError, ExtractionError, EmbeddingError, IndexError, OSError) as exc:
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
