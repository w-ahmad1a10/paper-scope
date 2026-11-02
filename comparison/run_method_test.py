"""Method-as-Query Test for PaperScope.

For each paper:
  1. Download PDF from arXiv
  2. Extract method section (Method/Approach/Methodology heading)
  3. Embed method text as a query (search_query: prefix)
  4. Search against ALL papers in the full-text index
  5. Record: rank of correct paper, similarity score, point (1 if rank==1),
     and top-10 similarity values

Aggregate metrics: Recall@1/3/5/10, MRR, Mean Rank

Results saved to comparison/results/method_as_query.json
             and comparison/results/method_metrics.json
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import arxiv
import fitz
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_scope.config import INDEX_DIR
from paper_scope.embed import Embedder
from paper_scope.index_store import SemanticIndex


# Section heading patterns for method extraction
METHOD_HEADINGS = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?(method(?:s|ology)?|approach|our approach|proposed approach|"
    r"proposed method|model|architecture|framework|system|technical approach|"
    r"method details|methodology)\s*$",
)

NEXT_SECTION_HEADINGS = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?(experiment(?:s|al)?|result(?:s)?|evaluation|"
    r"conclusion(?:s)?|discussion|limitation(?:s)?|reference(?:s)?|"
    r"bibliography|appendix|ablation|analysis)\s*$",
)


def fetch_pdf_url(title: str) -> str | None:
    """Get PDF URL from arXiv by title."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(query='ti:"' + title + '"', max_results=3)
        results = list(client.results(search))
        if not results:
            return None
        import re as _re
        norm = lambda s: _re.sub(r"\W+", " ", s).lower().strip()
        target = norm(title)
        for r in results:
            if norm(r.title) == target:
                return r.pdf_url
        return results[0].pdf_url
    except Exception:
        return None


def download_pdf(url: str, dest: Path) -> bool:
    """Download PDF from URL to dest. Returns True on success."""
    import urllib.request
    max_retries = 3
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, str(dest))
            if dest.is_file() and dest.stat().st_size > 20000:
                return True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return False


def extract_full_text(pdf_path: Path) -> str:
    """Extract full text from PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text


def extract_method_section(full_text: str) -> str | None:
    """Extract method section from full text using heading patterns."""
    lines = full_text.splitlines()
    method_start = None
    method_end = None

    # Find method section start
    for i, line in enumerate(lines):
        stripped = line.strip()
        if METHOD_HEADINGS.match(stripped):
            method_start = i
            break

    if method_start is None:
        return None

    # Find next section after method
    for i in range(method_start + 1, len(lines)):
        stripped = lines[i].strip()
        if NEXT_SECTION_HEADINGS.match(stripped):
            method_end = i
            break

    if method_end is None:
        method_end = min(method_start + 200, len(lines))  # cap at 200 lines

    method_text = "\n".join(lines[method_start:method_end]).strip()

    # Remove the heading itself
    method_lines = method_text.splitlines()
    if method_lines:
        method_text = "\n".join(method_lines[1:]).strip()

    # Check minimum length
    word_count = len(method_text.split())
    if word_count < 50:
        return None

    return method_text


def run_method_test(
    index_dir: Path = INDEX_DIR,
    output_dir: Path = Path("comparison/results"),
    pdf_dir: Path = Path("comparison/pdfs"),
    method_dir: Path = Path("comparison/method_texts"),
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    top_k: int = 10,
) -> dict:
    """Run the method-as-query test and save results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    index = SemanticIndex.load(index_dir)
    papers = index.papers
    print(f"Loaded index: {len(papers)} papers, dim={index.dimension}")

    # Load embedder
    embedder = Embedder(model_name)

    per_paper_results = []
    ranks = []
    points = 0
    skipped = []

    for idx, paper in enumerate(papers):
        key = paper["key"]
        title = paper["title"]

        pdf_path = pdf_dir / f"{key}.pdf"
        method_text_path = method_dir / f"{key}.txt"

        # Download PDF if not cached
        if not pdf_path.is_file():
            url = fetch_pdf_url(title)
            if not url:
                print(f"  [{idx+1}/{len(papers)}] SKIP (no PDF URL): {title[:50]}")
                skipped.append(key)
                continue
            if not download_pdf(url, pdf_path):
                print(f"  [{idx+1}/{len(papers)}] SKIP (download failed): {title[:50]}")
                skipped.append(key)
                continue

        # Extract method section
        if method_text_path.is_file():
            method_text = method_text_path.read_text(encoding="utf-8")
        else:
            try:
                full_text = extract_full_text(pdf_path)
                method_text = extract_method_section(full_text)
                if method_text:
                    method_text_path.write_text(method_text, encoding="utf-8")
            except Exception as e:
                print(f"  [{idx+1}/{len(papers)}] SKIP (extraction error): {title[:50]} - {e}")
                skipped.append(key)
                continue

        if not method_text:
            print(f"  [{idx+1}/{len(papers)}] SKIP (no method section): {title[:50]}")
            skipped.append(key)
            continue

        # Truncate if too long
        words = method_text.split()
        if len(words) > 2000:
            method_text = " ".join(words[:2000])

        # Embed method as query
        method_vec = embedder.encode_one(method_text, prefix="search_query: ")

        # Search against ALL papers
        hits = index.search_vector(method_vec, top_k=len(papers))

        # Find rank of correct paper
        rank = None
        score = None
        for i, hit in enumerate(hits):
            if hit.paper["key"] == key:
                rank = i + 1
                score = float(hit.score)
                break

        if rank is None:
            rank = len(papers) + 1
            score = 0.0

        point = 1 if rank == 1 else 0
        points += point
        ranks.append(rank)

        top10 = [
            {"key": hit.paper["key"], "score": round(float(hit.score), 6)}
            for hit in hits[:top_k]
        ]

        per_paper_results.append({
            "key": key,
            "title": title,
            "rank": rank,
            "score": round(score, 6),
            "point": point,
            "method_word_count": len(method_text.split()),
            "top10": top10,
        })

        print(
            f"  [{idx+1}/{len(papers)}] rank={rank:3d}  score={score:.4f}  "
            f"words={len(method_text.split()):5d}  point={point}  {title[:40]}"
        )

        time.sleep(0.5)

    # Compute aggregate metrics
    n = len(ranks)
    recall_at = {}
    for k in [1, 3, 5, 10]:
        recall_at[k] = sum(1 for r in ranks if r <= k) / n if n > 0 else 0.0

    mrr = sum(1.0 / r for r in ranks) / n if n > 0 else 0.0
    mean_rank = sum(ranks) / n if n > 0 else 0.0

    metrics = {
        "test": "method_as_query",
        "total_papers": len(papers),
        "tested": n,
        "skipped": len(skipped),
        "skipped_keys": skipped,
        "recall_at_1": round(recall_at[1], 4),
        "recall_at_3": round(recall_at[3], 4),
        "recall_at_5": round(recall_at[5], 4),
        "recall_at_10": round(recall_at[10], 4),
        "mrr": round(mrr, 4),
        "mean_rank": round(mean_rank, 2),
        "total_points": points,
        "max_points": n,
    }

    # Save results
    results_path = output_dir / "method_as_query.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(per_paper_results, f, indent=2, ensure_ascii=False)

    metrics_path = output_dir / "method_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("METHOD-AS-QUERY TEST RESULTS")
    print("=" * 60)
    print(f"  Papers tested:  {n}")
    print(f"  Skipped:        {len(skipped)}")
    print(f"  Recall@1:       {metrics['recall_at_1']:.4f}  ({sum(1 for r in ranks if r <= 1)}/{n})")
    print(f"  Recall@3:       {metrics['recall_at_3']:.4f}  ({sum(1 for r in ranks if r <= 3)}/{n})")
    print(f"  Recall@5:       {metrics['recall_at_5']:.4f}  ({sum(1 for r in ranks if r <= 5)}/{n})")
    print(f"  Recall@10:      {metrics['recall_at_10']:.4f}  ({sum(1 for r in ranks if r <= 10)}/{n})")
    print(f"  MRR:            {metrics['mrr']:.4f}")
    print(f"  Mean Rank:      {metrics['mean_rank']:.2f}")
    print(f"  Points:         {points}/{n}")
    print("=" * 60)
    print(f"  Results saved to: {results_path}")
    print(f"  Metrics saved to: {metrics_path}")

    return metrics


if __name__ == "__main__":
    run_method_test()
