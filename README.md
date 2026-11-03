# PaperScope

PaperScope is a local semantic search tool over AI research papers. It turns whole
papers into vectors using a Hugging Face embedding model and finds the paper whose
meaning is closest to your question. It does not generate answers — it retrieves the
closest paper and shows its score, along with scores for every other paper in the
collection.

## What the program does

- You add papers from your laptop by placing a PDF and giving it a name, or download
  hundreds of papers at once from arXiv via Kaggle GPU.
- The program reads the full text of each paper, removes the references section, and
  converts the text to plain running text.
- A whole-paper embedding model turns each paper into one vector of numbers.
- When you ask a question, the program turns your question into a vector and compares
  it with every paper vector using cosine similarity.
- The closest paper is shown first, followed by every other paper ranked by score.

## Quick start on Windows

On this machine, the local Python environment has already been created.
Double-click `run.cmd`.

Running the program without a command opens an interactive terminal screen. You can:

1. Ask a question: `ask <your question>`
2. Add a paper: `add <path to PDF> <name>`
3. List papers: `list`
4. Stop: `quit`

Or use the plain command line directly:

```powershell
.\.venv\Scripts\python.exe -m paper_scope ask "How does masked language modeling work?"
.\.venv\Scripts\python.exe -m paper_scope search-all "reinforcement learning for reasoning"
.\.venv\Scripts\python.exe -m paper_scope papers
```

## Manual setup on a new machine

The environment is managed with `uv`, which resolves and installs packages much
faster than the standard installer.

```powershell
uv venv .venv --python 3.13
uv pip install --python .\.venv\Scripts\python.exe "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

A CPU build of PyTorch is enough. No GPU is required for local search.

If `uv` uses a cache directory on a drive that does not exist, point
`UV_CACHE_DIR` to a writable folder before running the install commands. The same
applies to `HF_HOME`, `HF_HUB_CACHE`, and `SENTENCE_TRANSFORMERS_HOME` when the
Hugging Face cache path is broken. The program automatically falls back to a
project-local model cache if those paths cannot be created.

## Embedding model

The default model is `Alibaba-NLP/gte-modernbert-base`, downloaded from Hugging Face. It is
used to embed each whole paper into one vector. The model must have a context window
large enough for whole papers (8192 tokens). If a paper exceeds the model's context
limit, the end of the paper is cut off and the truncation rate is reported.

You can use a different model with `PAPER_SCOPE_MODEL` or the `--model` option.

## How it works

### 1. Add papers

You provide a path to a PDF on your laptop and a name for the paper. The program copies
the PDF into the project's papers folder, extracts the full text, and removes the
references section if it can find it.

Alternatively, download hundreds of papers from arXiv at once using Kaggle GPU
(see "Kaggle batch processing" below).

### 2. Extract and clean

The extractor reads the PDF text layer page by page. It:

- normalizes ligatures and quotation marks,
- rejoins words split by a line-end hyphen,
- removes page numbers, arXiv stamps, and repeated conference headers,
- removes numeric citation brackets,
- stops reading when it reaches the references section.

Scanned PDFs without a text layer are rejected.

### 3. Embed whole papers

Each cleaned paper is sent as one document to the embedding model. The model produces
one vector per paper. Every vector is scaled to length 1 so that dot product equals
cosine similarity. Encoding uses `torch.float16`.

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
- whether the paper was truncated and by what rate.

Scores below 0.68 are marked as weak matches. The score is a relative ranking signal,
not a calibrated probability that the paper answers the question.

## Commands

| Command | Description |
|---------|-------------|
| `add <pdf> <name>` | Add a paper from a local PDF |
| `build` | Rebuild the index from the saved collection |
| `batch` | Download many papers by title and build the index |
| `kaggle-run` | Prepare and push a Kaggle kernel for batch processing |
| `hf-sync <repo>` | Sync vectors from Hugging Face to local index |
| `search <query>` | Search the saved index once |
| `search-all <query>` | Search and show scores for all papers |
| `papers` | List papers in the collection |
| `ask` | Open the interactive question screen |

## Kaggle batch processing

For large collections (100+ papers), use Kaggle GPU to download and encode:

1. Prepare a `titles.txt` file with one paper title per line.
2. Upload it to a Hugging Face dataset repo.
3. Push the kernel: `kaggle kernels push -p ./kaggle --accelerator NvidiaTeslaT4`
4. Wait for completion, then download output: `kaggle kernels output <slug> -p ./kaggle_output -o`
5. Sync to local: `.\.venv\Scripts\python.exe -m paper_scope hf-sync <hf-repo>`

The kernel downloads PDFs from arXiv, extracts text with PyMuPDF, encodes with
Alibaba-NLP/gte-modernbert-base on GPU in fp16, and uploads vectors to Hugging Face.

## Working copy and watcher

This machine keeps a full working copy of the code. All edits happen in that
local copy. A background watcher checks for changes every 1 minute and sends
them to the online code site. Start the watcher once by running the start file.
If the watcher is not running, local changes stay only on this machine.

## Project layout

```text
paper_scope/
  catalog.py       dynamic paper collection management
  download.py      local PDF validation and copying
  extract.py       full-paper text cleaning and reference removal
  embed.py         embedding model loading and whole-document encoding
  build.py         add papers and rebuild the vector index
  batch.py         batch download and encode papers from arXiv
  index_store.py   save/load vectors and exact cosine search
  search.py        model + index query helper
  sync.py          sync vectors from Hugging Face to local index
  cli.py           command-line interface
  config.py        configuration and paths
kaggle/
  run.py           Kaggle GPU kernel script
  titles.txt       paper titles for batch processing
  kernel-metadata.json  Kaggle kernel manifest
data/
  papers/          copied PDFs                        (ignored by Git)
  text/            cleaned paper text                 (ignored by Git)
  index/           paper records, vectors, manifest   (ignored by Git)
  cache/           local Hugging Face model cache     (ignored by Git)
tests/             unit tests for cleaning, chunking, and search
100papers.txt      list of 100 AI research paper titles
```

The index manifest stores the model name, vector dimension, paper count, per-paper
statistics, and truncation information. A saved index is rejected when its schema or
dimensions do not match the code.

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
  window (8192 tokens).
- **PDF extraction is imperfect.** Two-column layouts, equations, tables, and figure
  captions can produce noisy text. The embedding model is robust to some noise, but
  not all of it.
- **The embedding model has a context limit.** If a paper exceeds it, the tail is
  cut off and the truncation rate is reported.
- **Similarity is not factual correctness.** A high-scoring paper can be closely
  related to the question without containing the exact answer.
- **The collection is user-built.** Papers are added by you through the `add` command
  or downloaded via Kaggle. There is no fixed reading list.
- **No baseline accuracy test exists.** There are no labeled retrieval benchmarks
  comparing this system against other methods.

## Data and licensing

The papers remain the property of their authors and are subject to their licenses.
This repository does not redistribute the PDFs or the generated text; both are
downloaded or copied locally and ignored by Git. The source code is released under
the MIT License.