"""Verify counts, file hashes, image decoding, and query/gallery separation.

Requires Pillow: python -m pip install Pillow
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/caltech101-small"


def main():
    with (DATA / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 500, f"Expected 500 images, got {len(rows)}"
    counts = Counter((row["category"], row["split"]) for row in rows)
    categories = list(dict.fromkeys(row["category"] for row in rows))
    assert len(categories) == 10
    hashes = set()
    paths = set()
    for row in rows:
        path = DATA / row["path"]
        assert path.resolve().is_relative_to(DATA.resolve())
        assert row["path"] not in paths, "Duplicate path"
        paths.add(row["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == row["sha256"], f"Hash mismatch: {path}"
        assert digest not in hashes, f"Duplicate file: {path}"
        hashes.add(digest)
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            assert min(img.size) > 0
    for category in categories:
        assert counts[category, "gallery"] == 40
        assert counts[category, "queries"] == 10
    actual = {p.relative_to(DATA).as_posix() for split in ["gallery", "queries"]
              for p in (DATA / split).rglob("*") if p.is_file()}
    assert paths == actual, "Unexpected or missing image files"

    # Two held-out query examples per category, for a quick visual inspection.
    sheet = Image.new("RGB", (1000, 620), "#f4f5f7")
    draw = ImageDraw.Draw(sheet)
    for idx, category in enumerate(categories):
        x, y = (idx % 5) * 200, (idx // 5) * 310
        draw.text((x + 12, y + 12), category, fill="#18243a")
        examples = [row for row in rows if row["category"] == category and row["split"] == "queries"][:2]
        for j, row in enumerate(examples):
            with Image.open(DATA / row["path"]) as img:
                tile = ImageOps.contain(ImageOps.exif_transpose(img).convert("RGB"), (180, 125))
                sheet.paste(tile, (x + (200 - tile.width) // 2, y + 38 + j * 135))
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    sheet.save(output / "dataset-preview.jpg", quality=90)
    report = {"images_decoded": 500, "unique_file_hashes": 500,
              "gallery_images": 400, "query_images": 100,
              "categories": categories, "exact_duplicate_overlap": 0,
              "near_duplicates_checked": False}
    (output / "dataset-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
