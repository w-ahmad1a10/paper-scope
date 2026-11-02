"""Kaggle kernel: Run comparison tests on papers.

Thin wrapper that clones the paper-scope repo and runs test scripts.
All logic is in the repo's comparison/ directory.
"""

import os

REPO_URL = "https://github.com/w-ahmad1a10/paper-scope.git"
REPO_DIR = "/kaggle/working/repo"

# Clone repo
os.system(f"git clone {REPO_URL} {REPO_DIR}")

# Install dependencies (individual packages to avoid version conflicts)
os.system("pip install -q sentence-transformers arxiv huggingface_hub PyMuPDF 2>&1 | tail -3")

# Change to repo directory
os.chdir(REPO_DIR)

# Sync vectors from HF to local
os.system("python -m paper_scope hf-sync w-ahmad/paper-scope-vectors")

# Run abstract-as-query test
os.system("python comparison/run_abstract_test.py")

# Run method-as-query test
os.system("python comparison/run_method_test.py")

# Upload results to HF
os.system("python -c \"from huggingface_hub import upload_file; import os; [upload_file(path_or_fileobj=f'comparison/results/{f}', path_in_repo=f'comparison_results/{f}', repo_id='w-ahmad/paper-scope-vectors', repo_type='dataset') for f in os.listdir('comparison/results') if f.endswith('.json')]\"")

print("TESTS_DONE_MARKER")
