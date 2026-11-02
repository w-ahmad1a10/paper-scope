"""Kaggle CPU kernel: Download PDFs from arXiv, extract text, upload to HF.

No GPU needed. No model loading. Just I/O.
"""

import os, time, re
os.system("git clone https://github.com/w-ahmad1a10/paper-scope.git /kaggle/working/repo")
os.system("pip install -q arxiv huggingface_hub PyMuPDF 2>&1 | tail -3")
os.chdir("/kaggle/working/repo")

from paper_scope.extract import extract_paper
from paper_scope.download import download_by_title
from paper_scope.config import PAPERS_DIR, TEXT_DIR
from huggingface_hub import HfApi, hf_hub_download, upload_file

# Download titles
hf_hub_download('w-ahmad/paper-scope-vectors', 'titles.txt', repo_type='dataset', local_dir='.')

titles = [l.strip() for l in open('titles.txt') if l.strip()]
titles = [re.sub(r'^\d+\.\s*', '', t) for t in titles]
titles = [t for t in titles if t]

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

for i, title in enumerate(titles, 1):
    key = title.lower().replace(' ', '_').replace('-', '_').replace('.', '_')
    key = ''.join(c for c in key if c.isalnum() or c == '_')
    pdf_path = PAPERS_DIR / f'{key}.pdf'
    text_path = TEXT_DIR / f'{key}.txt'

    if text_path.is_file() and text_path.stat().st_size > 0:
        print(f'[{i}/{len(titles)}] SKIP (already extracted): {title[:50]}')
        continue

    if not pdf_path.is_file():
        try:
            pdf_path = download_by_title(title, PAPERS_DIR)
            time.sleep(5)
        except Exception as e:
            print(f'[{i}/{len(titles)}] SKIP (download failed): {title[:50]} - {str(e)[:60]}')
            continue

    try:
        extracted = extract_paper(pdf_path)
        text_path.write_text(extracted.text, encoding='utf-8')
        print(f'[{i}/{len(titles)}] Extracted: {title[:50]} ({extracted.page_count} pages)')
    except Exception as e:
        print(f'[{i}/{len(titles)}] SKIP (extract failed): {title[:50]} - {str(e)[:60]}')

# Upload text files to HF
token = os.environ.get('HF_TOKEN')
api = HfApi(token=token)
api.create_repo(repo_id='w-ahmad/paper-scope-vectors', repo_type='dataset', exist_ok=True)
uploaded = 0
for txt_file in TEXT_DIR.glob('*.txt'):
    try:
        upload_file(path_or_fileobj=str(txt_file), path_in_repo=f'text/{txt_file.name}',
                    repo_id='w-ahmad/paper-scope-vectors', repo_type='dataset', token=token)
        uploaded += 1
    except Exception as e:
        print(f'  Upload failed: {txt_file.name} - {str(e)[:60]}')
print(f'Uploaded {uploaded} text files to HF')

print("EXTRACT_DONE_MARKER")
