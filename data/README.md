# Generated data

This folder is filled by the build step and is intentionally ignored by Git.

- `papers/` contains the downloaded arXiv PDFs and a small metadata snapshot.
- `text/` contains cleaned paper text with the references removed.
- `index/` contains the paragraph records, embedding matrix, and manifest.

The index can always be recreated from the fixed paper list in the project
source. Cached PDFs are reused unless a full re-download is requested.
