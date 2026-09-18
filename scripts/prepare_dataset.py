"""Download the chosen Kaggle archive and prepare a reproducible small collection.

Uses only Python's standard library. No Kaggle credentials are saved or required
when the public download endpoint is available.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://www.kaggle.com/datasets/imbikramsaha/caltech-101"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/imbikramsaha/caltech-101"
CATEGORIES = ["airplanes", "butterfly", "car_side", "cellphone", "chair",
              "cup", "elephant", "laptop", "Motorbikes", "watch"]


def download_archive(target: Path) -> None:
    if target.exists():
        if zipfile.is_zipfile(target):
            print(f"Using existing archive: {target}", flush=True)
            return
        raise ValueError(f"Existing archive is not a ZIP: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".zip.part")
    request = urllib.request.Request(DOWNLOAD, headers={"User-Agent": "ISR-PBL/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        location = response.geturl()
    if total <= 0:
        raise ValueError("Download size unavailable; download the archive manually from Kaggle.")
    # Range requests avoid one slow connection delaying the complete archive.
    # Signed redirect URLs remain in memory and are never printed or persisted.
    block_size = 4 * 1024 * 1024
    parts_dir = target.parent / (target.name + ".parts")
    parts_dir.mkdir(exist_ok=True)

    def fetch(start):
        end = min(start + block_size, total) - 1
        part = parts_dir / f"{start:012d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(3):
            try:
                req = urllib.request.Request(location, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    expected = f"bytes {start}-{end}/{total}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor range requests; download manually from Kaggle.")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError("Incomplete download chunk")
                part.write_bytes(data)
                return part
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    starts = list(range(0, total, block_size))
    received = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, start) for start in starts]
        for future in as_completed(futures):
            received += future.result().stat().st_size
            print(f"Downloaded {received / 1024**2:.0f} / {total / 1024**2:.0f} MiB", flush=True)
    with partial.open("wb") as output:
        for start in starts:
            output.write((parts_dir / f"{start:012d}.part").read_bytes())
    if not zipfile.is_zipfile(partial):
        raise ValueError("Kaggle did not return a ZIP. Download it manually from " + SOURCE)
    partial.replace(target)
    for start in starts:
        (parts_dir / f"{start:012d}.part").unlink()
    parts_dir.rmdir()


def prepare(archive: Path, output: Path, seed: int = 42) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}. Choose a new --output directory.")
    selected = []
    hashes = set()
    with zipfile.ZipFile(archive) as source:
        for category in CATEGORIES:
            candidates = sorted(
                name for name in source.namelist()
                if PurePosixPath(name).parent.name == category
                and PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            random.Random(f"{seed}:{category}").shuffle(candidates)
            unique = []
            for name in candidates:
                data = source.read(name)
                digest = hashlib.sha256(data).hexdigest()
                if digest in hashes:
                    continue
                hashes.add(digest)
                unique.append((name, data, digest))
                if len(unique) == 50:
                    break
            if len(unique) < 50:
                raise ValueError(f"{category}: only {len(unique)} unique images; need 50.")
            for index, (name, data, digest) in enumerate(unique):
                split = "gallery" if index < 40 else "queries"
                filename = PurePosixPath(name).name
                if not re.fullmatch(r"[\w.-]+", filename):
                    raise ValueError(f"Unexpected filename: {filename}")
                relative = Path(split) / category / filename
                selected.append((relative, data, {
                    "path": relative.as_posix(), "category": category, "split": split,
                    "sha256": digest, "source_member": name,
                }))
            print(f"Selected {category}: 40 gallery + 10 queries", flush=True)

    # Selection succeeds completely before any output images are written.
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for relative, data, row in selected:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        rows.append(row)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with archive.open("rb") as handle:
        archive_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    summary = {
        "dataset": "Caltech-101: 10-category PBL subset", "source": SOURCE,
        "archive_sha256": archive_digest,
        "seed": seed, "categories": CATEGORIES, "total_images": len(rows),
        "gallery_images": 400, "query_images": 100,
        "total_image_bytes": sum(len(data) for _, data, _ in selected),
        "selection": f"Sorted filenames shuffled independently per category with seed {seed}; first 50 unique file hashes.",
        "relevance": "A gallery image is relevant when its category matches the query category.",
        "limitations": ["Exact file duplicates removed; near duplicates are not automatically detected.",
                        "Category relevance is a proxy for visual similarity.",
                        "This is a small educational subset, not the official benchmark split."],
    }
    (output / "dataset_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the specified Kaggle dataset")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/caltech-101.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "data/caltech101-small")
    args = parser.parse_args()
    if args.download:
        download_archive(args.archive)
    if not args.archive.is_file():
        parser.error("Archive missing. Use --download or --archive PATH_TO_DOWNLOADED_ZIP.")
    prepare(args.archive, args.output)


if __name__ == "__main__":
    main()
