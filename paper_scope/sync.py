"""Sync vectors from Hugging Face to local index.

Downloads embeddings.npy, papers.jsonl, and manifest.json from a HF dataset repo
and merges them into the local index. Existing vectors are preserved and
deduplicated by key (new replaces old if same key).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

from .catalog import PaperEntry, load_collection, save_collection
from .config import COLLECTION_FILE, INDEX_DIR, SCHEMA_VERSION


def sync_from_hf(
    repo_id: str,
    token: str | None = None,
    index_dir: Path = INDEX_DIR,
    printer=print,
) -> dict:
    """Download vectors from HF and merge into local index."""

    index_dir.mkdir(parents=True, exist_ok=True)

    printer(f"Syncing from Hugging Face: {repo_id}")
    printer("")

    # Download the three files from HF
    downloaded: dict[str, Path] = {}
    for fname in ["embeddings.npy", "papers.jsonl", "manifest.json"]:
        printer(f"  Downloading {fname}...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            token=token,
            local_dir=str(index_dir),
            local_dir_use_symlinks=False,
        )
        downloaded[fname] = Path(local_path)
        printer(f"  Downloaded {fname}")

    printer("")

    # Load downloaded records and vectors
    hf_records: list[dict[str, Any]] = []
    with downloaded["papers.jsonl"].open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                hf_records.append(json.loads(line))

    hf_vectors = np.load(downloaded["embeddings.npy"], allow_pickle=False)
    hf_manifest = json.loads(downloaded["manifest.json"].read_text(encoding="utf-8"))

    printer(f"  HF index has {len(hf_records)} papers (model: {hf_manifest.get('model_name', '?')})")

    # Load existing local records and vectors
    local_records_path = index_dir / "papers.jsonl"
    local_vectors_path = index_dir / "embeddings.npy"

    existing_records: list[dict[str, Any]] = []
    existing_vectors: list = []
    if local_records_path.is_file():
        with local_records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing_records.append(json.loads(line))
    if local_vectors_path.is_file():
        old = np.load(local_vectors_path, allow_pickle=False)
        existing_vectors = list(old)

    printer(f"  Local index has {len(existing_records)} papers")

    # Merge: new HF records replace existing ones with same key
    hf_keys = {r.get("key") for r in hf_records}
    original_existing_records = list(existing_records)
    existing_records = [row for row in existing_records if row.get("key") not in hf_keys]
    existing_vectors = [
        vec for vec, row in zip(existing_vectors, original_existing_records) if row.get("key") not in hf_keys
    ]

    merged_records = existing_records + hf_records
    merged_vectors = existing_vectors + list(hf_vectors)

    if not merged_records:
        raise RuntimeError("No papers to sync")

    matrix = np.asarray(merged_vectors, dtype=np.float32)
    matrix = np.ascontiguousarray(matrix)

    # Build manifest from HF manifest (updated paper count)
    manifest = dict(hf_manifest)
    manifest["paper_count"] = len(merged_records)
    manifest["papers"] = merged_records
    manifest["built_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Save merged index
    from .index_store import save_index

    save_index(merged_records, matrix, manifest, index_dir=index_dir)

    # Update collection
    all_entries = [
        PaperEntry(
            key=row["key"],
            title=row["title"],
            path=row["path"],
            arxiv_id=row.get("arxiv_id"),
            year=row.get("year"),
            topic=row.get("topic"),
        )
        for row in merged_records
    ]
    save_collection(all_entries, collection_file=COLLECTION_FILE)

    printer("")
    printer(f"Synced {len(hf_records)} papers from HF")
    printer(f"Total papers in local index: {len(merged_records)}")
    printer(f"  (was {len(existing_records)} before sync)")

    # Vector safety: if local has more vectors than what HF had, re-upload
    # merged index to HF so HF stays the complete source of truth.
    if token and len(merged_records) > len(hf_records):
        try:
            from huggingface_hub import HfApi, upload_file

            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            printer("")
            printer(f"  Re-uploading merged index to HF ({len(hf_records)} -> {len(merged_records)} vectors)...")
            for fname in ["embeddings.npy", "papers.jsonl", "manifest.json"]:
                fpath = index_dir / fname
                upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )
                printer(f"    Uploaded {fname}")
            printer("  HF is now up to date.")
        except Exception as exc:  # noqa: BLE001
            printer(f"  Warning: could not re-upload to HF: {exc}")

    return manifest
