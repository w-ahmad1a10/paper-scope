"""Idea -> nearest paper search over LAION Scientific-Summaries.

Runs in Google Colab. Exposes a Gradio UI via a public share link.

Searches TWO views:
  1. concatenated_summary  -> FAISS semantic search (whole paper), top 10.
  2. methodological_details -> re-rank of the top 50 concat results by keyword
     overlap in the methodological_details field (fetched on demand). Top 10.

Note: the dataset ships only ONE FAISS summary index (concatenated_summary).
methodological_details has a BM25 index (~19 GB) which is too heavy for Colab,
so we re-rank the concat candidates by methods text instead.

Credit: LAION Project Alexandria, CC-BY-4.0.
"""
import os
import re
import numpy as np
import pandas as pd
import faiss
import gradio as gr
import requests
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

REPO_ID = "laion/Scientific-Summaries"
LOCAL_DIR = "./scientific-summaries"

INDEX_REL = "search/faiss/summaries/concatenated_summary.faiss"
MAP_REL = "search/mappings/paper_id_order.parquet"

SUBSET_BASE = {
    "arxiv": 0,
    "bethgelab": 2591107,
    "pubmed": 2925813,
    "web-papers": 4657998,
}

title_cache = {}
row_cache = {}
model = None
index = None
paper_order = None


def ensure_files():
    patterns = [INDEX_REL, MAP_REL]
    missing = [p for p in patterns if not os.path.exists(os.path.join(LOCAL_DIR, p))]
    if missing:
        print("Downloading index + mapping from Hugging Face (one-time, ~1.5 GB)...")
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            allow_patterns=patterns,
        )
        print("Download complete.")


def load_resources():
    global model, index, paper_order
    print("Loading embedding model (gte-modernbert-base)...")
    model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
    print("Loading FAISS index (memory-mapped)...")
    index = faiss.read_index(os.path.join(LOCAL_DIR, INDEX_REL), faiss.IO_FLAG_MMAP)
    index.nprobe = 16
    print("Loading paper id mapping (only needed columns)...")
    paper_order = pd.read_parquet(os.path.join(LOCAL_DIR, MAP_REL), columns=["paper_id", "subset"])
    global SUBSET_BASE
    SUBSET_BASE = {}
    for s in ["arxiv", "bethgelab", "pubmed", "web-papers"]:
        matches = np.where(paper_order["subset"].values == s)[0]
        SUBSET_BASE[s] = int(matches[0]) if len(matches) else 0
    print(f"Ready. Papers in mapping: {len(paper_order):,}")


def fetch_row(subset, local_offset):
    key = (subset, int(local_offset))
    if key in row_cache:
        return row_cache[key]
    data = {"title": None, "doi": None, "methods": None}
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
                data["title"] = row.get("source_title") or row.get("summary_title")
                data["doi"] = row.get("source_doi")
                data["methods"] = row.get("methodological_details")
    except Exception as e:
        print("HF rows fetch failed:", repr(e))
    if not data["title"] and str(subset).startswith("arxiv"):
        try:
            # fallback arxiv API for title only
            pass
        except Exception:
            pass
    if not data["title"]:
        data["title"] = None
    row_cache[key] = data
    return data


def make_link(subset, paper_id, doi):
    if str(subset).startswith("arxiv"):
        return f"https://arxiv.org/abs/{paper_id}"
    if doi:
        return f"https://doi.org/{doi}"
    return None


def keyword_score(query, text):
    if not text:
        return 0.0
    q_tokens = [t.lower() for t in query.split() if len(t) > 2]
    if not q_tokens:
        return 0.0
    t = text.lower()
    return sum(t.count(tok) for tok in q_tokens)


def search_ui(query, k):
    query = (query or "").strip()
    if not query:
        return [], []
    k = int(k)
    qvec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(qvec, max(k, 50))

    concat_rows = []
    candidates = []
    for rank, (score, i) in enumerate(zip(scores[0], idxs[0]), start=1):
        if i is None or int(i) < 0:
            continue
        i = int(i)
        if i >= len(paper_order):
            continue
        prow = paper_order.iloc[i]
        paper_id = str(prow["paper_id"])
        subset = str(prow["subset"])
        base = SUBSET_BASE.get(subset, 0)
        local_offset = i - base
        data = fetch_row(subset, local_offset)
        title = data["title"] or paper_id
        doi = data["doi"]
        link = make_link(subset, paper_id, doi)
        row = [rank, round(float(score), 4), title, subset, paper_id, link or ""]
        candidates.append((row, data.get("methods") or ""))
        if len(concat_rows) < k:
            concat_rows.append(row)

    # re-rank candidates by methodological_details keyword overlap
    scored = []
    for row, methods in candidates:
        scored.append((keyword_score(query, methods), row))
    scored.sort(key=lambda x: x[0], reverse=True)
    method_rows = []
    for rank, (sc, row) in enumerate(scored[:k], start=1):
        method_rows.append([rank, row[1], row[2], row[3], row[4], row[5]])

    return concat_rows, method_rows


def main():
    ensure_files()
    load_resources()
    with gr.Blocks(title="Idea -> Nearest Paper") as demo:
        gr.Markdown(
            "# Idea → Nearest Paper\n"
            "Semantic search over ~22M papers from LAION Scientific-Summaries "
            "(Project Alexandria), using `Alibaba-NLP/gte-modernbert-base`.\n\n"
            "Two views: **concatenated_summary** (FAISS, whole paper) and "
            "**methodological_details** (re-ranked from top-50 by methods text).\n\n"
            "Credit: LAION, CC-BY-4.0."
        )
        with gr.Row():
            query = gr.Textbox(label="Your idea / research question", lines=3, scale=4)
            k = gr.Slider(1, 20, value=10, step=1, label="Top-k per view", scale=1)
        btn = gr.Button("Search")
        gr.Markdown("### concatenated_summary (FAISS, whole paper)")
        out1 = gr.Dataframe(headers=["Rank", "Score", "Title", "Subset", "Paper ID", "Link"], wrap=True)
        gr.Markdown("### methodological_details (re-ranked by methods text)")
        out2 = gr.Dataframe(headers=["Rank", "Score", "Title", "Subset", "Paper ID", "Link"], wrap=True)
        btn.click(search_ui, inputs=[query, k], outputs=[out1, out2])
        query.submit(search_ui, inputs=[query, k], outputs=[out1, out2])
    demo.launch(share=True)


if __name__ == "__main__":
    main()
