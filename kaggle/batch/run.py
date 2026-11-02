"""Kaggle kernel: Batch process papers from arXiv.

Thin wrapper that clones the paper-scope repo and runs the batch command.
All logic is in the repo's paper_scope/ package.
"""

import os

REPO_URL = "https://github.com/w-ahmad1a10/paper-scope.git"
REPO_DIR = "/kaggle/working/repo"

# Clone repo
os.system(f"git clone {REPO_URL} {REPO_DIR}")

# Install dependencies
os.system(f"pip install -q -r {REPO_DIR}/requirements.txt")

# Change to repo directory
os.chdir(REPO_DIR)

# Download titles from Hugging Face
os.system("python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('w-ahmad/paper-scope-vectors', 'titles.txt', repo_type='dataset', local_dir='.')\"")

# Run batch processing (downloads PDFs, extracts text, encodes vectors, uploads to HF)
os.system("python -m paper_scope batch --titles-file titles.txt")

print("PAPERS_DONE_MARKER")
"