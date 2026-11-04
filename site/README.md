# Idea -> Nearest Paper

Gradio app for semantic search over LAION Scientific-Summaries (~22M papers).

## Run in Google Colab

```python
!git clone https://github.com/w-ahmad1a10/paper-scope.git
%cd paper-scope/site
!pip install -r requirements.txt
!python app.py
```

Gradio prints a public `https://....gradio.live` link you can open.

First run downloads ~1.5 GB (FAISS index + paper id mapping) plus the embedding model.

Tip: In Colab, if you hit memory limits, switch to Runtime -> Change runtime type -> High-RAM (or GPU). The index is memory-mapped to keep RAM low.

## What it does
1. You type an idea.
2. It is embedded with `Alibaba-NLP/gte-modernbert-base`.
3. Searched against the `concatenated_summary` FAISS index (21.8M papers).
4. Similarity scores (cosine, 0-1) are returned for the top-k nearest papers.
5. Each result's title and link are fetched on demand from Hugging Face (no big data download).
6. Results shown as: Rank, Score, Title, Subset, Paper ID, Link.

## Credit
Data and indices: LAION Scientific-Summaries (Project Alexandria), CC-BY-4.0.
Embedding model: Alibaba-NLP/gte-modernbert-base.