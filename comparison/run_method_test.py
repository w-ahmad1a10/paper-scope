"""Method-as-Query Test for PaperScope.

For each paper:
  1. Read its extracted text from Hugging Face (text/{key}.txt)
  2. Extract method section (Method/Approach/Methodology heading)
  3. Embed method text as a query
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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from paper_scope.config import INDEX_DIR, TEXT_DIR
from paper_scope.embed import Embedder
from paper_scope.index_store import SemanticIndex
from huggingface_hub import hf_hub_download

HF_REPO = "w-ahmad/paper-scope-vectors"

# Section heading patterns for method extraction from running text.
# Headings appear inline (not on separate lines) in HF extracted text.
# Headings are inline in running text, preceded by period+space or newline or start
METHOD_HEADINGS = re.compile(
    r"(?:^|\.\s+|\n\s+)(method(?:s|ology)?|approach|our approach|proposed approach|"
    r"proposed method|architecture|framework|design|overview)\s+(?=[A-Z0-9])",
    re.IGNORECASE,
)

NEXT_SECTION_HEADINGS = re.compile(
    r"(?:^|\.\s+|\n\s+)(experiment(?:s|al)?|result(?:s)?|evaluation|"
    r"conclusion(?:s)?|discussion(?:s)?|limitation(?:s)?|reference(?:s)?|"
    r"bibliography|appendix|ablation|analysis|acknowledgement(?:s)?|related work)\s+(?=[A-Z0-9])",
    re.IGNORECASE,
)


def extract_method_section(full_text: str) -> str | None:
    """Extract method section from running text using heading patterns.

    Works on HF extracted text (running text with inline headings).
    Finds the first method-related heading and captures text until the
    next known section heading.
    """
    # Find method section start
    method_match = METHOD_HEADINGS.search(full_text)
    if method_match is None:
        return None

    # Start after the heading
    method_start = method_match.end()

    # Find next section after method
    next_match = NEXT_SECTION_HEADINGS.search(full_text, method_start)
    if next_match is None:
        method_end = min(method_start + 8000, len(full_text))
    else:
        method_end = next_match.start()

    method_text = full_text[method_start:method_end].strip()

    # Check minimum length
    word_count = len(method_text.split())
    if word_count < 50:
        return None

    # Truncate if too long
    if word_count > 2000:
        method_text = " ".join(method_text.split()[:2000])

    return method_text


def get_text_path(key: str) -> Path:
    """Get local path to the HF text file, downloading if needed."""
    local_path = TEXT_DIR / f"{key}.txt"
    if local_path.is_file() and local_path.stat().st_size > 0:
        return local_path
    # Download from HF
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            HF_REPO, f"text/{key}.txt", repo_type="dataset",
            local_dir=str(TEXT_DIR.parent),
            local_dir_use_symlinks=False,
        )
        return Path(downloaded)
    except Exception:
        return local_path


def detect_device() -> str:
    """Detect best available device (GPU preferred)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def run_method_test(
    index_dir: Path = INDEX_DIR,
    output_dir: Path = Path("comparison/results"),
    model_name: str = "Alibaba-NLP/gte-modernbert-base",
    top_k: int = 10,
) -> dict:
    """Run the method-as-query test and save results."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Device detection
    device = detect_device()

    # Print hyperparameters
    print("=" * 60)
    print("METHOD-AS-QUERY TEST - CONFIGURATION")
    print("=" * 60)
    print(f"  Model:          {model_name}")
    print(f"  Device:         {device}")
    print(f"  Top-K:          {top_k}")
    print(f"  Index dir:      {index_dir}")
    if device == "cuda":
        try:
            import torch
            print(f"  GPU:            {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception:
            pass
    print("=" * 60)
    print("")

    # Load index
    index = SemanticIndex.load(index_dir)
    papers = index.papers
    print(f"Loaded index: {len(papers)} papers, dim={index.dimension}")

    # Load embedder
    embedder = Embedder(model_name, device=device)
    print(f"Embedder loaded on: {embedder.device}")
    print("")

    per_paper_results = []
    ranks = []
    points = 0
    skipped = []

    for idx, paper in enumerate(papers):
        key = paper["key"]
        title = paper["title"]

        # Read text from HF
        text_path = get_text_path(key)
        if not text_path.is_file():
            print(f"  [{idx+1}/{len(papers)}] SKIP (no text file): {title[:50]}")
            skipped.append(key)
            continue

        try:
            full_text = text_path.read_text(encoding="utf-8")
        except Exception:
            print(f"  [{idx+1}/{len(papers)}] SKIP (read error): {title[:50]}")
            skipped.append(key)
            continue

        # Extract method section
        method_text = extract_method_section(full_text)
        if not method_text:
            print(f"  [{idx+1}/{len(papers)}] SKIP (no method section): {title[:50]}")
            skipped.append(key)
            continue

        # Truncate if too long
        words = method_text.split()
        if len(words) > 2000:
            method_text = " ".join(words[:2000])

        # Embed method as query
        method_vec = embedder.encode_one(method_text)

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

    # Compute aggregate metrics
    n = len(ranks)
    recall_at = {}
    for k in [1, 3, 5, 10]:
        recall_at[k] = sum(1 for r in ranks if r <= k) / n if n > 0 else 0.0

    mrr = sum(1.0 / r for r in ranks) / n if n > 0 else 0.0
    mean_rank = sum(ranks) / n if n > 0 else 0.0

    metrics = {
        "test": "method_as_query",
        "model": model_name,
        "device": device,
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
    print(f"  Model:          {model_name}")
    print(f"  Device:         {device}")
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
