# PaperScope — Workflow Guide

## Overview

PaperScope is a local semantic search tool over AI research papers. It turns whole papers into vectors using a Hugging Face embedding model and finds the paper whose meaning is closest to your question.

## Architecture

```
This computer (local copy)  →  Online code site  →  Kaggle (compute)  →  Hugging Face (storage)
     │                    │                      │
     │                    │                      │
  Source of truth    GPU/CPU execution      Vectors + results
```

- **This computer**: Working copy of all code. All edits happen here.
- **Online code site**: Source of truth for code. Updated by the watcher.
- **Kaggle**: Execution environment (GPU for heavy tasks, CPU for tests)
- **Hugging Face**: Storage for vectors, results, and paper titles

## How It Works

### 1. Code Changes
All code lives in a working copy on this computer. When I (the agent) make edits:
1. I read and edit files in the local working copy.
2. The watcher checks for changes every 1 minute.
3. If the local copy has changes, the watcher sends them to the online code site.
4. The online code site is updated by the watcher.

**IMPORTANT: Edits happen in the local working copy only.**
**The watcher is the only path from this computer to the online code site.**
**The watcher never brings changes from the online code site down to this computer.**

### 2. Kaggle Execution
Kaggle kernels are thin wrappers that:
1. Clone the latest code from the online code site
2. Install dependencies
3. Run the appropriate command
4. Upload results to Hugging Face

### 3. Results Storage
All results are stored on Hugging Face:
- Vectors: `w-ahmad/paper-scope-vectors/embeddings.npy`
- Paper records: `w-ahmad/paper-scope-vectors/papers.jsonl`
- Test results: `w-ahmad/paper-scope-vectors/comparison_results/`

## Kaggle Kernels

### GPU Kernels (enable_gpu: true)

#### Batch Processing (`kaggle/batch/`)
- **Purpose**: Download 100+ papers from arXiv, extract text, encode vectors
- **Runtime**: ~15-20 minutes
- **Command**: `python -m paper_scope batch --titles-file titles.txt`
- **Output**: Vectors uploaded to HF

#### Nomic Variants (`kaggle/nomic_variants/`)
- **Purpose**: Compare 5 nomic model variants
- **Runtime**: ~50 minutes
- **Output**: Results uploaded to HF

### CPU Kernels (enable_gpu: false)

#### Tests (`kaggle/tests/`)
- **Purpose**: Run 23 unit tests + abstract/method/chunk comparison tests
- **Runtime**: ~5-10 minutes
- **Commands**: `pytest` + `comparison/run_abstract_test.py` + `comparison/run_method_test.py`

## Workflow: Adding New Papers

### Step 1: Update titles.txt on HF
Upload a new `titles.txt` with paper titles (one per line) to HF repo.

### Step 2: Run Kaggle batch kernel
The kernel will:
1. Clone latest code from GitHub
2. Download titles.txt from HF
3. Download PDFs from arXiv
4. Extract text
5. Encode vectors (float32)
6. Merge with existing HF index (append-only, never overwrites)
7. Upload merged index to HF

### Step 3: Sync locally (optional)
```bash
python -m paper_scope hf-sync w-ahmad/paper-scope-vectors
```

## Workflow: Running Tests

### On Kaggle CPU
1. Push `kaggle/tests/` kernel to Kaggle
2. Kernel auto-clones latest code from GitHub
3. Runs `pytest` + comparison tests
4. Uploads results to HF

### Results
- `comparison_results/abstract_as_query.json`
- `comparison_results/abstract_metrics.json`
- `comparison_results/method_as_query.json`
- `comparison_results/method_metrics.json`
- `comparison_results/chunk_as_query.json`
- `comparison_results/chunk_metrics.json`

## Key Design Decisions

### 1. Embedding Model
- **Model**: `Alibaba-NLP/gte-modernbert-base`
- **Precision**: float32 (not float16)
- **Max sequence length**: 8192 tokens

### 2. Vector Storage
- **Format**: float32 numpy arrays, L2-normalized
- **Similarity**: Cosine (dot product of normalized vectors)
- **Append-only**: New vectors are merged, never overwriting existing ones
- **HF as source of truth**: Vectors stored on Hugging Face, synced locally

### 3. Prefix Convention
- Documents: `search_document: {text}`
- Queries: `search_query: {text}`

### 4. Truncation
- Papers exceeding 8192 tokens are truncated
- Truncation rate is reported per paper and in aggregate

## How to Make Changes

### For the Agent (AI) — LOCAL WORKING COPY
1. Read and edit files in the local working copy.
2. The watcher sends local changes to the online code site every 1 minute.
3. The online code site is updated by the watcher.

**RULE: All edits happen in the local working copy.**
**RULE: The watcher is the only path to the online code site.**
**RULE: The watcher never brings changes from the online code site down.**

### For the User (Manual)
1. Edit files in the local working copy.
2. The watcher sends the changes to the online code site.
3. Or ask the agent to make changes.

### For Kaggle
1. Re-run the relevant kernel.
2. Kernel auto-clones latest code from the online code site.
3. No manual code copying needed.

## Important Notes

- **HF uploads need a Hugging Face token**: Kernels that upload to Hugging Face read the token from the private dataset `deep-research-1a10/hf-token-store`, which contains a file named `hf_token.txt`. The code reads the token from `/kaggle/input/hf-token-store/hf_token.txt`.
- **Local working copy**: All code lives on this computer. The watcher keeps the online code site in sync.
- **Watcher always on**: The watcher runs every 1 minute and sends local changes to the online code site.
- **No GitHub Actions**: Tests run on Kaggle CPU
- **Append-only vectors**: Never delete existing vectors, only add new ones
- **Temporary files**: PDFs and extracted text are temporary (deleted after processing)
- **HF as source of truth**: Vectors and results are on Hugging Face

## Repository Structure

```
paper-scope/ (GitHub)
├── paper_scope/          # Main package
│   ├── embed.py          # float32
│   ├── pipeline.py       # Standalone embedding
│   ├── batch.py          # Batch processing
│   ├── build.py          # Add papers
│   ├── search.py         # Search helper
│   ├── sync.py           # HF sync (append-only)
│   ├── cli.py            # CLI interface
│   └── ...
├── comparison/           # Test scripts
│   ├── run_abstract_test.py
│   └── run_method_test.py
├── tests/                # 23 unit tests
├── kaggle/               # Thin Kaggle wrappers
│   ├── batch/            # GPU kernel
│   └── tests/            # CPU kernel
├── git_watcher.py         # Watches local changes and sends them to the online code site
├── run_watcher.cmd        # Starts the watcher in the background
├── 100papers.txt         # Paper titles
├── NOMIC_VARIANTS_COMPARISON.md
├── WORKFLOW.md           # This file
└── README.md
```
