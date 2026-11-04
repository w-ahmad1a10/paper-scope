"""Idea -> nearest paper search over LAION Scientific-Summaries (chunk-based, CLI).

Run from a Colab cell:
    !python app.py "your idea here" --k 10

No web server, no Gradio. Results print to stdout.

Credit: LAION Project Alexandria, CC-BY-4.0.
"""
import argparse
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import faiss
import requests
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

REPO_ID = "laion/Scientific-Summaries"
LOCAL_DIR = "./scientific-summaries"

INDEX_REL = "search/faiss/chunks.faiss"
IVFDATA_REL = "search/faiss/chunks.faiss.ivfdata"
MAP_REL = "search/mappings/paper_id_order.parquet"
CHUNK_TO_PAPER_NPY = "./chunk_to_paper_global_idx.npy"

CHUNK_PARQUETS = [
    "search/chunks/arxiv_chunks.parquet",
    "search/chunks/bethgelab_chunks.parquet",
    "search/chunks/pubmed_chunks.parquet",
    "search/chunks/web-papers_chunks.parquet",
]

SUBSET_BASE = {
    "arxiv": 0,
    "bethgelab": 2591107,
    "pubmed": 2925813,
    "web-papers": 4657998,
}

title_cache = {}
model = None
index = None
paper_order = None
chunk_to_paper = None


def ensure_files():
    patterns = [INDEX_REL, IVFDATA_REL, MAP_REL]
    missing = [p for p in patterns if not os.path.exists(os.path.join(LOCAL_DIR, p))]
    if missing:
        print("Downloading chunks index + mapping from Hugging Face (one-time, ~3.4 GB)...")
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            allow_patterns=patterns,
        )
        print("Download complete.")


def build_chunk_mapping():
    global chunk_to_paper
    if os.path.exists(CHUNK_TO_PAPER_NPY):
        print("Loading cached chunk->paper mapping...")
        chunk_to_paper = np.load(CHUNK_TO_PAPER_NPY, mmap_mode="r")
        print(f"Loaded. Chunks: {len(chunk_to_paper):,}")
        return
    print("Building chunk->paper mapping by streaming parquets (first run, a few minutes)...")
    parts = []
    total = 0
    for rel in CHUNK_PARQUETS:
        url = f"hf://datasets/{REPO_ID}/{rel}"
        print(f"  reading {rel} ...", flush=True)
        df = pd.read_parquet(url, columns=["paper_global_idx"])
        arr = df["paper_global_idx"].to_numpy(dtype=np.int64)
        parts.append(arr)
        total += len(arr)
        print(f"    {len(arr):,} chunks (running total {total:,})", flush=True)
        del df
    chunk_to_paper = np.concatenate(parts)
    np.save(CHUNK_TO_PAPER_NPY, chunk_to_paper)
    print(f"Saved mapping: {len(chunk_to_paper):,} chunks -> {CHUNK_TO_PAPER_NPY}")


def load_resources():
    global model, index, paper_order, chunk_to_paper
    print("Loading embedding model (gte-modernbert-base)...")
    model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
    print("Loading FAISS chunks index (memory-mapped)...")
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(os.path.join(LOCAL_DIR, INDEX_REL), faiss.IO_FLAG_MMAP)
    index.nprobe = 16
    print("Loading paper id mapping (only needed columns)...")
    paper_order = pd.read_parquet(os.path.join(LOCAL_DIR, MAP_REL), columns=["paper_id", "subset"])
    global SUBSET_BASE
    SUBSET_BASE = {}
    for s in ["arxiv", "bethgelab", "pubmed", "web-papers"]:
        matches = np.where(paper_order["subset"].values == s)[0]
        SUBSET_BASE[s] = int(matches[0]) if len(matches) else 0
    build_chunk_mapping()
    print(f"Ready. Papers: {len(paper_order):,}  Chunks: {len(chunk_to_paper):,}")


def fetch_title(subset, paper_id, local_offset):
    key = (subset, int(local_offset))
    if key in title_cache:
        return title_cache[key]
    title = None
    doi = None
    try:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset=laion/Scientific-Summaries&config={subset}"
            f"&split=train&offset={int(local_offset)}&length=1"
        )
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            rows = r.json().get("rows", [])
            if rows:
                row = rows[0]["row"]
                title = row.get("source_title") or row.get("summary_title")
                doi = row.get("source_doi")
    except Exception as e:
        print("HF rows fetch failed:", repr(e))
    if not title and str(subset).startswith("arxiv"):
        try:
            arxiv_id = str(paper_id).replace("v1", "").replace("v2", "").replace("v3", "")
            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and "<title>" in r.text:
                m = re.search(r"<title>(.*?)</title>", r.text, re.S)
                if m:
                    title = m.group(1).strip()
        except Exception as e:
            print("arXiv API fetch failed:", repr(e))
    if not title:
        title = paper_id
    title_cache[key] = (title, doi)
    return title_cache[key]


def make_link(subset, paper_id, doi):
    if str(subset).startswith("arxiv"):
        return f"https://arxiv.org/abs/{paper_id}"
    if doi:
        return f"https://doi.org/{doi}"
    return None


def search(query, k):
    query = (query or "").strip()
    if not query:
        return []
    k = int(k)
    N = 50
    qvec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    print("Searching index...", flush=True)
    scores, idxs = index.search(qvec, N)
    scores, idxs = scores[0], idxs[0]
    print(f"Got {len(idxs)} chunks. Grouping by paper...", flush=True)

    paper_best = {}
    for s, ci in zip(scores, idxs):
        if ci is None or int(ci) < 0:
            continue
        ci = int(ci)
        if ci >= len(chunk_to_paper):
            continue
        pgid = int(chunk_to_paper[ci])
        if pgid >= len(paper_order):
            continue
        prow = paper_order.iloc[pgid]
        paper_id = str(prow["paper_id"])
        if paper_id not in paper_best or float(s) > paper_best[paper_id][0]:
            paper_best[paper_id] = (float(s), ci, pgid)

    if len(paper_best) < k:
        scores2, idxs2 = index.search(qvec, 100)
        for s, ci in zip(scores2[0], idxs2[0]):
            if ci is None or int(ci) < 0:
                continue
            ci = int(ci)
            if ci >= len(chunk_to_paper):
                continue
            pgid = int(chunk_to_paper[ci])
            if pgid >= len(paper_order):
                continue
            prow = paper_order.iloc[pgid]
            paper_id = str(prow["paper_id"])
            if paper_id not in paper_best or float(s) > paper_best[paper_id][0]:
                paper_best[paper_id] = (float(s), ci, pgid)

    ranked = sorted(paper_best.items(), key=lambda kv: -kv[1][0])[:k]

    out = []
    for rank, (paper_id, (score, ci, pgid)) in enumerate(ranked, start=1):
        prow = paper_order.iloc[pgid]
        subset = str(prow["subset"])
        base = SUBSET_BASE.get(subset, 0)
        local_offset = pgid - base
        title, doi = fetch_title(subset, paper_id, local_offset)
        link = make_link(subset, paper_id, doi)
        out.append((rank, round(float(score), 4), title or paper_id, subset, paper_id, link or ""))
        print(f"  {rank}. [{round(float(score),4)}] {title or paper_id}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Idea -> nearest paper (chunk-based)")
    parser.add_argument("query", nargs="+", help="your idea / research question")
    parser.add_argument("--k", type=int, default=10, help="top-k papers")
    args = parser.parse_args()
    query = " ".join(args.query)

    ensure_files()
    load_resources()

    print(f"\nQuery: {query}")
    print(f"Top {args.k}:\n")
    t0 = time.time()
    results = search(query, args.k)
    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s. {len(results)} results.")

    # also print a tab-separated block for easy copy
    print("\n--- RESULTS ---")
    for r in results:
        print(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]}")


if __name__ == "__main__":
    main()
