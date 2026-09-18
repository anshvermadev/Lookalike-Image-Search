from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import faiss
import numpy as np

from .config import DATA, FEATURE_VERSIONS, INDEXES, MODEL_PATH
from .dataset import Record, file_hash, fingerprint, image_path, load_records
from .features import Encoder, normalize, open_rgb

faiss.omp_set_num_threads(1)


@lru_cache(maxsize=4)
def _model_hash(path: str, mtime: int, size: int) -> str:
    return file_hash(Path(path))


def model_hash() -> str:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Pretrained CLIP model is missing. Run scripts/download_model.py.")
    stat = MODEL_PATH.stat()
    return _model_hash(str(MODEL_PATH), stat.st_mtime_ns, stat.st_size)


def build_index(method: str, data_dir: Path = DATA, index_dir: Path = INDEXES,
                progress: Callable[[int, int], None] | None = None) -> dict:
    records = load_records(data_dir)
    gallery = [row for row in records if row.split == "gallery"]
    encoder = Encoder(method)
    start = perf_counter()
    batches = []
    for offset in range(0, len(gallery), 8):
        images = [open_rgb(image_path(row, data_dir)) for row in gallery[offset:offset + 8]]
        batches.append(encoder.encode(images))
        if progress:
            progress(min(offset + 8, len(gallery)), len(gallery))
    vectors = normalize(np.concatenate(batches))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    directory = index_dir / method
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / "search.faiss.tmp"))
    with (directory / "vectors.npy.tmp").open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    (directory / "records.json.tmp").write_text(json.dumps([asdict(row) for row in gallery]), encoding="utf-8")
    metadata = {
        "method": method, "feature_version": FEATURE_VERSIONS[method],
        "dataset_fingerprint": fingerprint(records), "count": len(gallery),
        "dimension": vectors.shape[1], "model_sha256": model_hash() if method == "clip" else None,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": perf_counter() - start, "index_type": "FAISS IndexFlatIP (exact cosine search)",
    }
    metadata["files"] = {name: file_hash(directory / f"{name}.tmp")
                         for name in ["search.faiss", "vectors.npy", "records.json"]}
    for name in metadata["files"]:
        (directory / f"{name}.tmp").replace(directory / name)
    metadata["index_bytes"] = (directory / "search.faiss").stat().st_size
    (directory / "metadata.json.tmp").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (directory / "metadata.json.tmp").replace(directory / "metadata.json")
    return metadata


class SearchIndex:
    def __init__(self, method: str, records: list[Record] | None = None,
                 index_dir: Path = INDEXES):
        directory = index_dir / method
        if not (directory / "metadata.json").exists():
            raise FileNotFoundError(f"The {method} index has not been built yet.")
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        current = load_records() if records is None else records
        if self.metadata["dataset_fingerprint"] != fingerprint(current):
            raise ValueError("The dataset has changed. Rebuild the search indexes.")
        if self.metadata["feature_version"] != FEATURE_VERSIONS[method]:
            raise ValueError("Feature extraction changed. Rebuild the search index.")
        if method == "clip" and self.metadata["model_sha256"] != model_hash():
            raise ValueError("The CLIP model changed. Rebuild the neural index.")
        for name, digest in self.metadata["files"].items():
            if file_hash(directory / name) != digest:
                raise ValueError("Search index is incomplete or changed. Rebuild it.")
        self.records = [Record(**row) for row in json.loads((directory / "records.json").read_text(encoding="utf-8"))]
        if self.records != [row for row in current if row.split == "gallery"]:
            raise ValueError("Index records do not match the searchable gallery.")
        self.vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        self.index = faiss.read_index(str(directory / "search.faiss"))
        if self.vectors.shape != (len(self.records), self.index.d) or self.index.ntotal != len(self.records):
            raise ValueError("Index dimensions are inconsistent. Rebuild it.")
        self.method = method

    def search(self, query: np.ndarray, k: int = 10) -> list[dict]:
        if k < 1:
            raise ValueError("Choose at least one result.")
        vector = normalize(np.asarray(query).reshape(1, -1))
        if vector.shape[1] != self.index.d:
            raise ValueError("Query features do not match the selected index.")
        scores, ids = self.index.search(vector, min(k, len(self.records)))
        return [{"rank": rank + 1, "index_id": int(idx), "score": float(np.clip(score, -1, 1)),
                 **asdict(self.records[idx])}
                for rank, (score, idx) in enumerate(zip(scores[0], ids[0]))]


def refine_query(query: np.ndarray, positive_vectors: np.ndarray, weight: float = 0.3) -> np.ndarray:
    if not 0 <= weight <= 1:
        raise ValueError("Feedback weight must be between 0 and 1.")
    if len(positive_vectors) == 0:
        raise ValueError("Select at least one relevant result.")
    return normalize((1 - weight) * normalize(query) + weight * np.mean(normalize(positive_vectors), axis=0))
