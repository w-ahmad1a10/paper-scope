"""Idea -> nearest paper search over LAION Scientific-Summaries (chunk-based).

Runs in Google Colab. Exposes a Gradio UI via a public share link.

Method: oversample N chunks from the chunks FAISS index, group by paper,
max-pool scores (best passage per paper), dedupe, take top-k. This finds
papers whose best matching passage matches the user's idea.

Credit: LAION Project Alexandria, CC-BY-4.0.
"""
import os
import re
import time
import numpy as np
import pandas as pd
import faiss
import gradio as gr
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
    """Build (and cache) an int64 array: chunk_idx -> paper_global_idx."""
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
        print(f"  reading {rel} ...")
        df = pd.read_parquet(url, columns=["paper_global_idx"])
        arr = df["paper_global_idx"].to_numpy(dtype=np.int64)
        parts.append(arr)
        total += len(arr)
        print(f"    {len(arr):,} chunks (running total {total:,})")
        del df
    chunk_to_paper = np.concatenate(parts)
    np.save(CHUNK_TO_PAPER_NPY, chunk_to_paper)
    print(f"Saved mapping: {len(chunk_to_paper):,} chunks -> {CHUNK_TO_PAPER_NPY}")


def load_resources():
    global model, index, paper_order, chunk_to_paper
    print("Loading embedding model (gte-modernbert-base)...")
    model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
    print("Loading FAISS chunks index (memory-mapped)...")
    index = faiss.read_index(os.path.join(LOCAL_DIR, INDEX_REL), faiss.IO_FLAG_MMAP)
    index.nprobe = 32
    print("Loading paper id mapping (only needed columns)...")
    paper_order = pd.read_parquet(os.path.join(LOCAL_DIR, MAP_REL), columns=["paper_id", "subset"])
    global SUBSET_BASE
    SUBSET_BASE = {}
    for s in ["arxiv", "bethgelab", "pubmed", "web-papers"]:
        matches = np.where(paper_order["subset"].values == s)[0]
        SUBSET_BASE[s] = int(matches[0]) if len(matches) else 0
    build_chunk_mapping()
    print(f"Ready. Papers in mapping: {len(paper_order):,}  Chunks: {len(chunk_to_paper):,}")


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
        r = requests.get(url, timeout=30)
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
            r = requests.get(url, timeout=30)
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


def search_ui(query, k):
    query = (query or "").strip()
    if not query:
        return []
    k = int(k)
    try:
        N = 100
        qvec = model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, idxs = index.search(qvec, N)
        scores, idxs = scores[0], idxs[0]

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
            scores2, idxs2 = index.search(qvec, 500)
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
            try:
                title, doi = fetch_title(subset, paper_id, local_offset)
            except Exception:
                title, doi = paper_id, None
            link = make_link(subset, paper_id, doi)
            out.append([rank, round(float(score), 4), title or paper_id, subset, paper_id, link or ""])
        print(f"Search done: {len(out)} results")
        return out
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


def main():
    ensure_files()
    load_resources()
    with gr.Blocks(title="Idea -> Nearest Paper (chunks)") as demo:
        gr.Markdown(
            "# Idea → Nearest Paper (chunk-based)\n"
            "Searches 60.3M text chunks from LAION Scientific-Summaries, groups by "
            "paper, keeps each paper's best matching passage. Uses "
            "`Alibaba-NLP/gte-modernbert-base`.\n\n"
            "Credit: LAION Project Alexandria, CC-BY-4.0."
        )
        with gr.Row():
            query = gr.Textbox(label="Your idea / research question", lines=3, scale=4)
            k = gr.Slider(1, 20, value=10, step=1, label="Top-k papers", scale=1)
        btn = gr.Button("Search")
        out = gr.Dataframe(headers=["Rank", "Score", "Title", "Subset", "Paper ID", "Link"], wrap=True)
        btn.click(search_ui, inputs=[query, k], outputs=out)
        query.submit(search_ui, inputs=[query, k], outputs=out)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
