# Idea -> Nearest Paper (chunk-based)

Gradio app for semantic search over LAION Scientific-Summaries (~22M papers, 60.3M chunks).

## Run in Google Colab

```python
!git clone https://github.com/w-ahmad1a10/paper-scope.git
%cd paper-scope/site
!pip install -r requirements.txt
!python app.py
```

Gradio prints a public `https://....gradio.live` link you can open.

First run downloads ~3.4 GB (chunks FAISS index) + builds a ~480 MB chunk->paper mapping
(streamed from Hugging Face, a few minutes) + the embedding model. Later runs are fast.

## What it does
1. You type an idea.
2. It is embedded with `Alibaba-NLP/gte-modernbert-base`.
3. Oversamples N=200 nearest chunks from the chunks FAISS index (60.3M vectors).
4. Groups chunks by paper and keeps each paper's best matching passage (max-pool).
5. Dedupes and returns top-k papers by best passage score.
6. Titles and links are fetched on demand from Hugging Face (no big data download).

## Credit
Data and indices: LAION Scientific-Summaries (Project Alexandria), CC-BY-4.0.
Embedding model: Alibaba-NLP/gte-modernbert-base.