# PaperScope — Workflow Guide

## Overview

PaperScope is a local semantic search tool over AI research papers. It turns whole papers into vectors using a Hugging Face embedding model and finds the paper whose meaning is closest to your question.

## Architecture

```
GitHub (code)  →  Kaggle (compute)  →  Hugging Face (storage)
     │                    │                      │
     │                    │                      │
  Source of truth    GPU/CPU execution      Vectors + results
```

- **GitHub**: Single source of truth for all code
- **Kaggle**: Execution environment (GPU for heavy tasks, CPU for tests)
- **Hugging Face**: Storage for vectors, results, and paper titles

## How It Works

### 1. Code Changes
All code lives on GitHub. When I (the agent) make edits:
1. I read the file from GitHub via MCP tools
2. I edit the content
3. I commit directly to GitHub via MCP tools
4. GitHub is updated immediately

**IMPORTANT: Only MCP GitHub tools are used for all GitHub operations.**
**The GitHub REST API is NEVER used directly. All reads, writes, commits, and deletions go through MCP tools.**

### 2. Kaggle Execution
Kaggle kernels are thin wrappers that:
1. Clone the latest code from GitHub
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
5. Encode vectors (with rotary scaling, float32)
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
- **Model**: `nomic-ai/nomic-embed-text-v1.5`
- **Precision**: float32 (not float16)
- **Trust remote code**: True (required for rotary scaling)
- **Rotary scaling**: `rotary_scaling_factor=2` (extends 2048 → 8192 tokens)
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

### For the Agent (AI) — MCP TOOLS ONLY
1. Read file from GitHub via MCP tools (mcp__github__get_file_contents or github_file_read)
2. Edit content in memory
3. Commit directly to GitHub via MCP tools (mcp__github__create_or_update_file or mcp__github__push_files)
4. GitHub is updated immediately

**RULE: Never use the GitHub REST API directly. All GitHub operations go through MCP tools only.**

### For the User (Manual)
1. Edit on GitHub web (press `.` for VS Code in browser)
2. Or ask the agent to make changes

### For Kaggle
1. Re-run the relevant kernel
2. Kernel auto-clones latest code from GitHub
3. No manual code copying needed

## Important Notes

- **No local repo**: All code lives on GitHub
- **No GitHub Actions**: Tests run on Kaggle CPU
- **Append-only vectors**: Never delete existing vectors, only add new ones
- **Temporary files**: PDFs and extracted text are temporary (deleted after processing)
- **HF as source of truth**: Vectors and results are on Hugging Face
- **MCP tools only**: All GitHub operations use MCP tools, never the REST API

## Repository Structure

```
paper-scope/ (GitHub)
├── paper_scope/          # Main package
│   ├── embed.py          # float32, trust_remote_code, rotary=2
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
├── 100papers.txt         # Paper titles
├── NOMIC_VARIANTS_COMPARISON.md
├── WORKFLOW.md           # This file
└── README.md
```
