"""Idea -> nearest paper search over LAION Scientific-Summaries.

Runs in Google Colab. Exposes a Gradio UI via a public share link.
Credit: LAION Project Alexandria, CC-BY-4.0.
"""
import os
import time
import requests
import numpy as np
import pandas as pd
import faiss
import gradio as gr
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

REPO_ID = "laion/Scientific-Summaries"
LOCAL_DIR = "./scientific-summaries"
INDEX_REL = "search/faiss/summaries/concatenated_summary.faiss"
MAP_REL = "search/mappings/paper_id_order.parquet"
INDEX_PATH = os.path.join(LOCAL_DIR, INDEX_REL)
MAP_PATH = os.path.join(LOCAL_DIR, MAP_REL)

# global_idx offsets inside paper_id_order.parquet (subset order)
SUBSET_BASE = {
    "arxiv": 0,
    "bethgelab": 2591107,
    "pubmed": 2925813,
    "web-papers": 4657998,
}

title_cache = {}


def ensure_files():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(MAP_PATH)):
        print("Downloading index + mapping from Hugging Face (one-time, ~1.5 GB)...")
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            allow_patterns=[INDEX_REL, MAP_REL],
        )
        print("Download complete.")


def load_resources():
    global model, index, paper_order
    print("Loading embedding model (gte-modernbert-base)...")
    model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
    print("Loading FAISS index...")
    # Memory-map the index so the 1.23 GB file is not fully loaded into RAM.
    index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
    index.nprobe = 16
    print("Loading paper id mapping (only needed columns)...")
    paper_order = pd.read_parquet(MAP_PATH, columns=["paper_id", "subset"])
    # Compute per-subset base offsets from the actual mapping (robust to counts).
    global SUBSET_BASE
    SUBSET_BASE = {}
    for s in ["arxiv", "bethgelab", "pubmed", "web-papers"]:
        matches = np.where(paper_order["subset"].values == s)[0]
        SUBSET_BASE[s] = int(matches[0]) if len(matches) else 0
    print(f"Ready. Papers in index mapping: {len(paper_order):,}")
    print("Subset bases:", SUBSET_BASE)


def fetch_title(subset, paper_id, local_offset):
    key = (subset, int(local_offset))
    if key in title_cache:
        return title_cache[key]
    title = None
    doi = None
    # 1) Hugging Face datasets-server (single row, no big download)
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
    # 2) arXiv API fallback for arxiv subset
    if not title and subset.startswith("arxiv"):
        try:
            arxiv_id = paper_id.replace("v1", "").replace("v2", "").replace("v3", "")
            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and "<title>" in r.text:
                import re
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
    if subset.startswith("arxiv"):
        return f"https://arxiv.org/abs/{paper_id}"
    if doi:
        return f"https://doi.org/{doi}"
    return None


def search(query, k):
    query = (query or "").strip()
    if not query:
        return []
    qvec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(qvec, int(k))
    out = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        if idx is None or int(idx) < 0:
            continue
        idx = int(idx)
        if idx >= len(paper_order):
            continue
        prow = paper_order.iloc[idx]
        paper_id = str(prow["paper_id"])
        subset = str(prow["subset"])
        base = SUBSET_BASE.get(subset, 0)
        local_offset = idx - base
        title, doi = fetch_title(subset, paper_id, local_offset)
        link = make_link(subset, paper_id, doi)
        out.append({
            "Rank": rank,
            "Score": round(float(score), 4),
            "Title": title,
            "Subset": subset,
            "Paper ID": paper_id,
            "Link": link or "",
        })
    return out


def search_ui(query, k):
    rows = search(query, k)
    if not rows:
        return []
    return [[r["Rank"], r["Score"], r["Title"], r["Subset"], r["Paper ID"], r["Link"]] for r in rows]


def main():
    ensure_files()
    load_resources()
    with gr.Blocks(title="Idea -> Nearest Paper") as demo:
        gr.Markdown(
            "# Idea → Nearest Paper\n"
            "Semantic search over ~22M papers from LAION Scientific-Summaries "
            "(Project Alexandria), using `Alibaba-NLP/gte-modernbert-base`.\n\n"
            "Credit: LAION, CC-BY-4.0. Titles/links are resolved on demand."
        )
        with gr.Row():
            query = gr.Textbox(label="Your idea / research question", lines=3, scale=4)
            k = gr.Slider(1, 20, value=10, step=1, label="Top-k", scale=1)
        btn = gr.Button("Search")
        out = gr.Dataframe(
            headers=["Rank", "Score", "Title", "Subset", "Paper ID", "Link"],
            wrap=True,
        )
        btn.click(search_ui, inputs=[query, k], outputs=out)
        query.submit(search_ui, inputs=[query, k], outputs=out)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
