"""Kaggle kernel: Run comparison tests on papers.

Thin wrapper that clones the paper-scope repo and runs test scripts.
All logic is in the repo's comparison/ directory.

Runs on GPU. No unit tests (pytest) are run.
Abstract and method tests read extracted text from Hugging Face.
"""
import os

REPO_URL = "https://github.com/w-ahmad1a10/paper-scope.git"
REPO_DIR = "/kaggle/working/repo"

# Clone repo
os.system(f"git clone {REPO_URL} {REPO_DIR}")

# Install dependencies
os.system("pip install -q sentence-transformers huggingface_hub PyMuPDF 2>&1 | tail -3")

# Change to repo directory
os.chdir(REPO_DIR)

# Read HF token from private dataset and set as env var
HF_TOKEN_PATH = "/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt"
hf_token = None
if os.path.isfile(HF_TOKEN_PATH):
    with open(HF_TOKEN_PATH, "r", encoding="utf-8") as f:
        hf_token = f.read().strip()
    os.environ["HF_TOKEN"] = hf_token
    print("HF token loaded, length:", len(hf_token))
else:
    print("HF token file not found at", HF_TOKEN_PATH)

# GPU check
os.system("python -c \"import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\"")

# Sync vectors from HF to local (needs HF_TOKEN env var)
os.system("python -m paper_scope hf-sync w-ahmad/paper-scope-vectors")

# Print settings
os.system("python -c \"from paper_scope.config import INDEX_DIR; from paper_scope.index_store import SemanticIndex; idx = SemanticIndex.load(INDEX_DIR); print('Papers:', len(idx.papers), 'Dim:', idx.dimension, 'Model:', idx.model_name)\"")

# Run abstract-as-query test
os.system("python comparison/run_abstract_test.py")

# Run method-as-query test
os.system("python comparison/run_method_test.py")

# Upload results to HF
os.system("""python -c "
import os
from huggingface_hub import upload_file
token = os.environ.get('HF_TOKEN')
for f in os.listdir('comparison/results'):
    if f.endswith('.json'):
        upload_file(path_or_fileobj=f'comparison/results/{f}', path_in_repo=f'comparison_results/{f}', repo_id='w-ahmad/paper-scope-vectors', repo_type='dataset', token=token)
        print('Uploaded', f)
" """)

print("TESTS_DONE_MARKER")
