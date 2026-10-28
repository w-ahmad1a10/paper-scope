# PaperScope

PaperScope is a small, local research tool. It turns AI research papers into vectors
and finds the paper whose meaning is closest to your question. It does not generate
an answer. It retrieves the closest paper and shows its score, along with scores for
every other paper in the collection so you can see how each paper ranked.

## What the program does

- You add papers from your laptop by placing a PDF and giving it a name.
- The program reads the full text of each paper, removes the references section, and
  converts the text to plain running text.
- A whole-paper embedding model turns each paper into one vector of numbers.
- When you ask a question, the program turns your question into a vector and compares
  it with every paper vector using cosine similarity.
- The closest paper is shown first, followed by every other paper ranked by score.

## Quick start on Windows

On this machine, the local Python environment has already been created.
Double-click `run.cmd`.

The interactive screen opens. You can:

1. Ask a question: `ask <your question>`
2. Add a paper: `add <path to PDF> <name>`
3. List papers: `list`
4. Stop: `quit`

## Manual setup on a new machine

The environment is managed with `uv`, which resolves and installs packages much
faster than the standard installer.

```powershell
uv venv .venv --python 3.13
uv pip install --python .\.venv\Scripts\python.exe "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

A CPU build of PyTorch is enough. No GPU is required.

If `uv` uses a cache directory on a drive that does not exist, point
`UV_CACHE_DIR` to a writable folder before running the install commands. The same
applies to `HF_HOME`, `HF_HUB_CACHE`, and `SENTENCE_TRANSFORMERS_HOME` when the
Hugging Face cache path is broken. The program automatically falls back to a
project-local model cache if those paths cannot be created.

## Embedding model

The default model is `voyageai/voyage-4-nano`, downloaded from Hugging Face. It is
used to embed each whole paper into one vector. The model must have a context window
large enough for whole papers. If a paper exceeds the model's context limit, the end
of the paper is cut off and the truncation rate is reported.

You can use a different model with `PAPER_SCOPE_MODEL` or the `--model` option.

## How it works

### 1. Add papers

Through the interactive screen, you provide a path to a PDF on your laptop and a name
for the paper. The program copies the PDF into the project's papers folder, extracts
the full text, and removes the references section if it can find it.

### 2. Extract and clean

The extractor reads the PDF text layer page by page using a natural block order so
two-column text is not mixed. It:

- normalizes ligatures and quotation marks,
- rejoins words split by a line-end hyphen,
- removes page numbers, arXiv stamps, and repeated conference headers,
- removes numeric citation brackets,
- stops reading when it reaches the references section.

Scanned PDFs without a text layer are rejected.

### 3. Embed whole papers

Each cleaned paper is sent as one document to the embedding model. The model produces
one vector per paper. Every vector is scaled to length 1 so that dot product equals
cosine similarity.

### 4. Search

A question becomes one vector using the same model. The program computes the dot
product between that vector and every paper vector. Because all vectors have length 1,
this dot product equals cosine similarity. The highest scores are sorted and every
paper is shown in ranked order.

## Output

Each result shows:

- the cosine similarity score,
- the paper title,
- the paper key,
- the publication year and arXiv ID if available,
- page count and body pages,
- whether the paper was truncated and by what rate,
- the retrieved paper text preview (in search mode).

Scores below 0.68 are marked as weak matches. The score is a relative ranking signal,
not a calibrated probability that the paper answers the question.

## Project layout

```text
paper_scope/
  catalog.py       dynamic paper collection management
  download.py      local PDF validation and copying
  extract.py       full-paper text cleaning and reference removal
  embed.py         embedding model loading and whole-document encoding
  build.py         add papers and rebuild the vector index
  index_store.py   save/load vectors and exact cosine search
  search.py        model + index query helper
  cli.py           command-line interface and interactive screen
data/
  papers/          copied PDFs                        (ignored by Git)
  text/            cleaned paper text                 (ignored by Git)
  index/           paper records, vectors, manifest   (ignored by Git)
  cache/           local Hugging Face model cache     (ignored by Git)
tests/             unit tests for cleaning, chunking, and search
```

The index manifest stores the model name, vector dimension, paper count, chunk
settings, per-paper statistics, and truncation information. A saved index is rejected
when its schema or dimensions do not match the code.

## Tests

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

The tests do not download papers or the model. They check text cleanup, reference
removal, vector search, and index round-tripping.

## Limits and research notes

- **This is retrieval, not question answering.** A language model does not write a
  new answer. The output is the closest stored paper.
- **"Vector" means one vector per whole paper.** Paper length affects what can be
  retrieved. Very long papers may be truncated if they exceed the model's context
  window.
- **PDF extraction is imperfect.** Two-column layouts, equations, tables, and figure
  captions can produce noisy text. The embedding model is robust to some noise, but
  not all of it.
- **The embedding model has a context limit.** If a paper exceeds it, the tail is
  cut off and the truncation rate is reported.
- **Similarity is not factual correctness.** A high-scoring paper can be closely
  related to the question without containing the exact answer.
- **The collection is user-built.** Papers are added by you through the interactive
  screen. There is no fixed reading list.
- **No baseline accuracy test exists.** There are no labeled retrieval benchmarks
  comparing this system against other methods.

## Data and licensing

The papers remain the property of their authors and are subject to their licenses.
This repository does not redistribute the PDFs or the generated text; both are
downloaded or copied locally and ignored by Git. The source code is released under
the MIT License.
