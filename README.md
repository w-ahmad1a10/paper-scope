# PaperScope

PaperScope is a small, local research tool. It downloads 12 well-known AI
papers from arXiv, turns their text into paragraph-level vectors with a
Hugging Face embedding model, and finds the paragraph that best matches a
plain-English question.

It does not generate an answer. It retrieves the closest passage and shows the
paper, year, arXiv identifier, page number, and similarity score. That makes the
retrieval step easy to inspect and reason about.

## The collection

| Year | Paper | arXiv |
| ---: | --- | --- |
| 2017 | Attention Is All You Need | 1706.03762 |
| 2018 | BERT | 1810.04805 |
| 2019 | RoBERTa | 1907.11692 |
| 2019 | T5 | 1910.10683 |
| 2020 | ELECTRA | 2003.10555 |
| 2020 | GPT-3: Language Models are Few-Shot Learners | 2005.14165 |
| 2022 | InstructGPT | 2203.02155 |
| 2022 | Chinchilla | 2203.15556 |
| 2023 | LLaMA | 2302.13971 |
| 2023 | Llama 2 | 2307.09288 |
| 2023 | Mistral 7B | 2310.06825 |
| 2023 | GPT-4 Technical Report | 2303.08774 |

The list is fixed in the project source, so builds are reproducible. PDFs are
cached after the first download.

## Quick start on Windows

On this machine, the local Python environment has already been created.
Double-click `run.cmd`.

On the first run, the program:

1. downloads the 12 PDFs from arXiv,
2. extracts and chunks their body text,
3. downloads the embedding model from Hugging Face,
4. embeds every paragraph and saves the vector index,
5. opens a question prompt.

Later runs load the saved index and model from the local cache.

### Manual setup on a new machine

The environment is managed with `uv`, which resolves and installs packages much
faster than the standard installer.

```powershell
uv venv .venv --python 3.13
uv pip install --python .\.venv\Scripts\python.exe "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m paper_scope build
```

A CPU build of PyTorch is enough. No GPU is required.

If `uv` uses a cache directory on a drive that does not exist, point
`UV_CACHE_DIR` to a writable folder before running the install commands. The
same applies to `HF_HOME`, `HF_HUB_CACHE`, and `SENTENCE_TRANSFORMERS_HOME` when
the Hugging Face cache path is broken. The program automatically falls back to a
project-local model cache if those paths cannot be created.

## Use

Build or rebuild the collection:

```powershell
.\.venv\Scripts\python.exe -m paper_scope build
```

Ask one question:

```powershell
.\.venv\Scripts\python.exe -m paper_scope search "Why does ELECTRA need fewer training examples than BERT?"
```

Return the five closest paragraphs:

```powershell
.\.venv\Scripts\python.exe -m paper_scope search "How is Chinchilla trained differently from GPT-3?" --top 5
```

Machine-readable output:

```powershell
.\.venv\Scripts\python.exe -m paper_scope search "masked language modeling" --json
```

Open the interactive prompt:

```powershell
.\.venv\Scripts\python.exe -m paper_scope ask
```

Running with no command does the same as `ask`. If no index exists, the program
offers to build it.

Each one-question command starts a new process and reloads the model. On this
CPU machine that startup takes about 20 to 30 seconds. The interactive prompt
loads the model once, so repeated questions are much faster.

List the paper collection without loading the model:

```powershell
.\.venv\Scripts\python.exe -m paper_scope papers
```

Build a quick trial index from the first three papers:

```powershell
.\.venv\Scripts\python.exe -m paper_scope build --limit 3
```

## How it works

### 1. Download

Each paper has a fixed arXiv identifier. The downloader requests the PDF with
retries, verifies the PDF signature, and caches the file under `data/papers/`.
A cached file is reused unless `--force-download` is used.

### 2. Extract and clean

PyMuPDF reads the text layer page by page. The extractor:

- normalises ligatures and quotation marks,
- rejoins words broken by a line-end hyphen,
- removes page numbers, arXiv stamps, and repeated conference headers,
- removes numeric citation brackets,
- stops reading when it reaches the references section.

Scanned PDFs without a text layer are rejected with a clear error. All 12
papers in the collection have a text layer.

### 3. Make paragraphs

The cleaned page text is split into sentences, protecting common abbreviations
such as "et al." and "e.g." Sentences are grouped into chunks that aim for
about 650 characters, never exceed 900 characters, and are at least 250
characters long. Each chunk records its paper, year, arXiv ID, and page.

These chunks are the retrieval unit. They are paragraph-like, but they are not
always the exact paragraphs chosen by the paper's authors. A chunk can begin or
end at a sentence boundary inside a long paragraph, and it never crosses a page
boundary.

### 4. Embed

The default model is `BAAI/bge-small-en-v1.5`, downloaded from Hugging Face. It
maps text to 384 numbers and supports inputs up to 512 tokens. Paragraphs are
encoded in batches of 32.

BGE models expect a short instruction in front of a search question. The
program adds that instruction to questions but not to stored paragraphs. This
is not a trick: the model was trained with this format, and using it improves
retrieval.

Every vector is scaled to length 1. The same encoder and the same normalisation
are used for questions. This matters because the search compares vectors from
the same coordinate system.

### 5. Search

A question becomes one 384-number vector. The program computes the dot product
between that vector and every paragraph vector. Because all vectors have length
1, this dot product equals cosine similarity. The highest scores are sorted and
the requested number of paragraphs is printed.

With roughly 1,500 paragraphs, an exact scan over the full matrix is fast and
needs no approximate nearest-neighbour library.

## Output

Each result shows:

- the cosine similarity, usually between 0 and 1 for related text,
- the paper title, publication year, arXiv ID, and page,
- the retrieved paragraph.

Scores below 0.68 are marked as weak matches. The score is a relative ranking
signal, not a calibrated probability that the paragraph answers the question.
The 0.68 cut-off was chosen for this model after testing unrelated questions;
it is not a universal threshold.

## Project layout

```text
paper_scope/
  catalog.py       fixed list of papers
  download.py      PDF download, validation, and cache
  extract.py       PDF text cleaning and paragraph chunking
  embed.py         Hugging Face model loading and batch encoding
  build.py         full build pipeline
  index_store.py   save/load vectors and exact cosine search
  search.py        model + index query helper
  cli.py           command-line interface
data/
  papers/          downloaded PDFs                      (ignored by Git)
  text/            cleaned paper text                   (ignored by Git)
  index/           paragraphs, vectors, manifest        (ignored by Git)
  cache/           local Hugging Face model cache       (ignored by Git)
tests/             unit tests for cleaning, chunking, and search
```

The index manifest stores the model name, vector dimension, paragraph count,
chunk settings, and per-paper statistics. A saved index is rejected when its
schema or dimensions do not match the code.

## Tests

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

The tests do not download papers or the model. They check text cleanup,
reference removal, chunk sizes, vector search, index round-tripping, and the
model-specific question prefix.

## Limits and research notes

- **This is retrieval, not question answering.** A language model does not write
  a new answer. The output is the closest stored passage.
- **"Paragraph" means a sentence-aligned chunk.** Chunk size affects what can be
  retrieved. Small chunks are precise but may miss context. Large chunks carry
  context but mix several ideas.
- **PDF extraction is imperfect.** Two-column layouts, equations, tables, and
  figure captions can produce noisy text. The embedding model is robust to some
  noise, but not all of it.
- **The embedding model has a 512-token limit.** The 900-character cap keeps most
  chunks below that limit. A few technical chunks may still be truncated.
- **Similarity is not factual correctness.** A high-scoring paragraph can be
  closely related to the question without containing the exact answer.
- **The collection is small and fixed.** Twelve papers cannot cover the whole
  field. Adding papers means adding catalog entries and rebuilding the index.
- **arXiv versions can change.** The build does not pin a version suffix, so a
  later rebuild may use a newer revision of a paper.

## Data and licensing

The papers remain the property of their authors and are subject to their arXiv
licenses. This repository does not redistribute the PDFs or the generated text;
both are downloaded locally and ignored by Git. The source code is released
under the MIT License.
