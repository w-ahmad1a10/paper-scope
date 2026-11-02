"""Abstract-as-Query Test for PaperScope.

For each paper:
  1. Fetch its abstract from arXiv
  2. Embed the abstract as a query (search_query: prefix)
  3. Search against ALL papers in the full-text index
  4. Record: rank of correct paper, similarity score, point (1 if rank==1),
     and top-10 similarity values

Aggregate metrics: Recall@1/3/5/10, MRR, Mean Rank

Results saved to comparison/results/abstract_as_query.json
             and comparison/results/abstract_metrics.json
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import arxiv
import numpy as np

from paper_scope.config import INDEX_DIR
from paper_scope.embed import Embedder
from paper_scope.index_store import SemanticIndex


def fetch_abstract(title: str) -> str | None:
    """Fetch abstract from arXiv by exact title match."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(query='ti:"' + title + '"', max_results=3)
        results = list(client.results(search))
        if not results:
            return None
        # Find best match by normalized title
        import re
        norm = lambda s: re.sub(r"\W+", " ", s).lower().strip()
        target = norm(title)
        for r in results:
            if norm(r.title) == target:
                return r.summary
        return results[0].summary
    except Exception:
        return None


def run_abstract_test(
    index_dir: Path = INDEX_DIR,
    output_dir: Path = Path("comparison/results"),
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    top_k: int = 10,
) -> dict:
    """Run the abstract-as-query test and save results."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    index = SemanticIndex.load(index_dir)
    papers = index.papers
    print(f"Loaded index: {len(papers)} papers, dim={index.dimension}")

    # Load embedder
    embedder = Embedder(model_name)

    # Per-paper results
    per_paper_results = []
    ranks = []
    points = 0
    skipped = []

    for idx, paper in enumerate(papers):
        key = paper["key"]
        title = paper["title"]

        # Fetch abstract from arXiv
        abstract = fetch_abstract(title)
        if not abstract:
            print(f"  [{idx+1}/{len(papers)}] SKIP (no abstract): {title[:60]}")
            skipped.append(key)
            continue

        # Embed abstract as query
        abstract_vec = embedder.encode_one(abstract, prefix="search_query: ")

        # Search against ALL papers (including itself)
        hits = index.search_vector(abstract_vec, top_k=len(papers))

        # Find rank of correct paper
        rank = None
        score = None
        for i, hit in enumerate(hits):
            if hit.paper["key"] == key:
                rank = i + 1
                score = float(hit.score)
                break

        if rank is None:
            rank = len(papers) + 1  # not found in top-N
            score = 0.0

        point = 1 if rank == 1 else 0
        points += point
        ranks.append(rank)

        # Top-10 results
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
            "top10": top10,
        })

        print(
            f"  [{idx+1}/{len(papers)}] rank={rank:3d}  score={score:.4f}  "
            f"point={point}  {title[:50]}"
        )

        # Rate limit for arXiv API
        time.sleep(1.0)

    # Compute aggregate metrics
    n = len(ranks)
    recall_at = {}
    for k in [1, 3, 5, 10]:
        recall_at[k] = sum(1 for r in ranks if r <= k) / n if n > 0 else 0.0

    mrr = sum(1.0 / r for r in ranks) / n if n > 0 else 0.0
    mean_rank = sum(ranks) / n if n > 0 else 0.0

    metrics = {
        "test": "abstract_as_query",
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

    # Save per-paper results
    results_path = output_dir / "abstract_as_query.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(per_paper_results, f, indent=2, ensure_ascii=False)

    # Save aggregate metrics
    metrics_path = output_dir / "abstract_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("ABSTRACT-AS-QUERY TEST RESULTS")
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
    run_abstract_test()
