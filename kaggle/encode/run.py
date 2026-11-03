"""Kaggle GPU kernel: Download text from HF, encode vectors, upload to HF.

No arXiv calls. No PDF downloads. Just read text + encode + upload.
"""

import os, time, re, sys
os.system("git clone https://github.com/w-ahmad1a10/paper-scope.git /kaggle/working/repo")
os.system("pip install -q sentence-transformers huggingface_hub 2>&1 | tail -3")
REPO_DIR = "/kaggle/working/repo"
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

from paper_scope.embed import Embedder
from paper_scope.config import INDEX_DIR, TEXT_DIR
from paper_scope.index_store import save_index
from huggingface_hub import HfApi, hf_hub_download, upload_file
import json, numpy as np

# Download titles
hf_hub_download('w-ahmad/paper-scope-vectors', 'titles.txt', repo_type='dataset', local_dir='.')

titles = [l.strip() for l in open('titles.txt') if l.strip()]
titles = [re.sub(r'^\d+\.\s*', '', t) for t in titles]
titles = [t for t in titles if t]

# Download text files from HF
TEXT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

for t in titles:
    key = t.lower().replace(' ', '_').replace('-', '_').replace('.', '_')
    key = ''.join(c for c in key if c.isalnum() or c == '_')
    try:
        hf_hub_download('w-ahmad/paper-scope-vectors', f'text/{key}.txt', repo_type='dataset', local_dir='data')
    except Exception as e:
        print(f'SKIP (no text): {key} - {str(e)[:60]}')

# Read text files
papers = []
for t in titles:
    key = t.lower().replace(' ', '_').replace('-', '_').replace('.', '_')
    key = ''.join(c for c in key if c.isalnum() or c == '_')
    text_path = TEXT_DIR / f'{key}.txt'
    if text_path.is_file():
        text = text_path.read_text(encoding='utf-8')
        papers.append({'key': key, 'title': t, 'text': text, 'path': '', 'text_path': str(text_path)})
    else:
        print(f'SKIP (no text file): {key}')

print(f'Loaded {len(papers)} papers with text')

# Encode
embedder = Embedder('Alibaba-NLP/gte-modernbert-base', batch_size=1, max_tokens=120000)

records = []
vectors = []
for i, paper in enumerate(papers, 1):
    text = paper['text']
    token_est = embedder.estimate_tokens(text)
    truncated = token_est > embedder.max_seq_length
    trunc_rate = (token_est - embedder.max_seq_length) / token_est if truncated and token_est > 0 else 0.0

    vec = embedder.encode_one(text)
    vectors.append(vec)

    records.append({
        'key': paper['key'],
        'title': paper['title'],
        'path': '',
        'text_path': paper['text_path'],
        'arxiv_id': None,
        'year': None,
        'topic': None,
        'page_count': 0,
        'body_pages': 0,
        'removed_references': True,
        'token_estimate': token_est,
        'truncated': truncated,
        'truncation_rate': round(trunc_rate, 6),
    })
    print(f'[{i}/{len(papers)}] Encoded: {paper["title"][:50]} (tokens={token_est}, truncated={truncated})')

# Merge with existing index on HF
existing_records = []
existing_vectors = []
try:
    from huggingface_hub import hf_hub_download
    hf_hub_download('w-ahmad/paper-scope-vectors', 'papers.jsonl', repo_type='dataset', local_dir='data/index_tmp')
    hf_hub_download('w-ahmad/paper-scope-vectors', 'embeddings.npy', repo_type='dataset', local_dir='data/index_tmp')
    with open('data/index_tmp/papers.jsonl') as f:
        for line in f:
            if line.strip():
                existing_records.append(json.loads(line))
    existing_vectors = list(np.load('data/index_tmp/embeddings.npy', allow_pickle=False))
    print(f'Merging with {len(existing_records)} existing vectors')
except Exception as e:
    print(f'No existing index on HF (first run): {str(e)[:60]}')

new_keys = {r['key'] for r in records}
kept_records = [r for r in existing_records if r.get('key') not in new_keys]
kept_vectors = [v for v, r in zip(existing_vectors, existing_records) if r.get('key') not in new_keys]
merged_records = kept_records + records
merged_vectors = kept_vectors + vectors

matrix = np.asarray(merged_vectors, dtype=np.float32)
if matrix.ndim == 1:
    matrix = matrix[None, :]

# Save
INDEX_DIR.mkdir(parents=True, exist_ok=True)
np.save(INDEX_DIR / 'embeddings.npy', matrix)
with open(INDEX_DIR / 'papers.jsonl', 'w', encoding='utf-8') as f:
    for r in merged_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

import datetime
manifest = {
    'schema_version': 2,
    'built_at': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'model_name': 'Alibaba-NLP/gte-modernbert-base',
    'dimension': matrix.shape[1],
    'max_seq_length': embedder.max_seq_length,
    'normalized': True,
    'similarity': 'cosine (dot product of L2-normalised vectors)',
    'paper_count': len(merged_records),
    'max_tokens': 120000,
    'papers': merged_records,
}
with open(INDEX_DIR / 'manifest.json', 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f'Saved {len(merged_records)} vectors, shape={matrix.shape}')

# Upload to HF
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    token = os.environ.get('HF_TOKEN')
api = HfApi(token=token)
api.create_repo(repo_id='w-ahmad/paper-scope-vectors', repo_type='dataset', exist_ok=True)
for fname in ['embeddings.npy', 'papers.jsonl', 'manifest.json']:
    upload_file(path_or_fileobj=str(INDEX_DIR / fname), path_in_repo=fname, repo_id='w-ahmad/paper-scope-vectors', repo_type='dataset', token=token)
    print(f'Uploaded {fname}')

print(f'HF now has {len(merged_records)} vectors')
print("ENCODE_DONE_MARKER")
