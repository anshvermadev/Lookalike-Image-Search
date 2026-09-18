from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .config import DATA


@dataclass(frozen=True)
class Record:
    path: str
    category: str
    split: str
    sha256: str


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def image_path(record: Record, data_dir: Path = DATA) -> Path:
    root = data_dir.resolve()
    path = (root / record.path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path must stay inside the dataset folder.")
    return path


def load_records(data_dir: Path = DATA, verify: bool = True) -> list[Record]:
    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError("Dataset not prepared. See the dataset setup section in README.md.")
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [Record(**{key: row[key] for key in Record.__dataclass_fields__})
                for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("Dataset manifest is empty.")
    hashes, paths = set(), set()
    for row in rows:
        if row.split not in {"gallery", "queries"} or not row.category:
            raise ValueError("Invalid dataset split or category.")
        if row.path in paths or row.sha256 in hashes:
            raise ValueError("Duplicate images in the dataset manifest; recreate the split.")
        paths.add(row.path)
        hashes.add(row.sha256)
        path = image_path(row, data_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset image: {row.path}")
        if verify and file_hash(path) != row.sha256:
            raise ValueError(f"Dataset image changed: {row.path}. Recreate the manifest and indexes.")
    if not any(row.split == "gallery" for row in rows):
        raise ValueError("The dataset has no gallery images to search.")
    return rows


def fingerprint(records: list[Record]) -> str:
    content = json.dumps([asdict(row) for row in records], sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()
