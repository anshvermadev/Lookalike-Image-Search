import csv
import hashlib
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from isr.dataset import Record, image_path, load_records
from isr.evaluation import metrics
from isr.features import clip_pixels, histogram, normalize, open_rgb
from isr.retrieval import SearchIndex, build_index, refine_query


@pytest.fixture
def collection(tmp_path):
    data = tmp_path / "data"
    rows = []
    for name, color, split in [("red", (255, 0, 0), "gallery"), ("blue", (0, 0, 255), "gallery"),
                               ("green", (0, 255, 0), "gallery"), ("query", (240, 0, 0), "queries")]:
        path = data / split / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color).save(path)
        rows.append({"path": path.relative_to(data).as_posix(), "category": "red" if name == "query" else name,
                     "split": split, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with (data / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return data, tmp_path / "indexes"


def test_ranked_search_and_persistence_excludes_queries(collection):
    data, indexes = collection
    build_index("histogram", data_dir=data, index_dir=indexes)
    index = SearchIndex("histogram", load_records(data), indexes)
    results = index.search(histogram(Image.new("RGB", (25, 25), (245, 0, 0))), k=20)
    assert len(results) == 3  # requesting more matches than the gallery remains valid
    assert results[0]["category"] == "red"
    assert results[0]["score"] == pytest.approx(1)
    assert all(row["split"] == "gallery" for row in results)
    assert results[0]["score"] >= results[-1]["score"]
    with pytest.raises(ValueError, match="at least one"):
        index.search(np.ones(512), 0)
    with pytest.raises(ValueError, match="do not match"):
        index.search(np.ones(16))


def test_changed_dataset_and_corrupted_index_rejected(collection):
    data, indexes = collection
    build_index("histogram", data, indexes)
    records = load_records(data)
    (indexes / "histogram" / "vectors.npy").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="incomplete or changed"):
        SearchIndex("histogram", records, indexes)
    image_path(records[0], data).write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        load_records(data)


def test_dataset_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        image_path(Record("../outside.jpg", "x", "gallery", "x"), tmp_path)


def test_duplicate_across_splits_rejected(collection):
    data, _ = collection
    with (data / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[-1]["sha256"] = rows[0]["sha256"]
    with (data / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Duplicate"):
        load_records(data, verify=False)


def test_metrics_known_ranking():
    values = metrics([True, False, True] + [False] * 7, 40)
    assert values["precision_at_5"] == pytest.approx(0.4)
    assert values["precision_at_10"] == pytest.approx(0.2)
    assert values["recall_at_10"] == pytest.approx(0.05)
    assert values["ap_at_10"] == pytest.approx((1 + 2 / 3) / 10)
    assert values["mrr_at_10"] == 1
    assert values["ndcg_at_10"] == pytest.approx(1.5 / sum(1 / np.log2(np.arange(2, 12))))
    perfect = metrics([True] * 10, 40)
    assert perfect["ap_at_10"] == 1
    assert perfect["recall_at_10"] == 0.25
    assert all(value == 0 for value in metrics([False] * 10, 40).values())
    with pytest.raises(ValueError):
        metrics([True] * 10, 0)


def test_image_input_validation_and_clip_preprocessing():
    with pytest.raises(ValueError, match="could not be read"):
        open_rgb(b"this is not an image")
    with pytest.raises(ValueError, match="smaller than 10 MB"):
        open_rgb(b"x" * (10 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="long or narrow"):
        open_rgb(Image.new("RGB", (1000, 1)))
    for mode in ["L", "RGB", "RGBA"]:
        img = Image.new(mode, (120, 80))
        buffer = BytesIO()
        img.save(buffer, "PNG")
        decoded = open_rgb(buffer.getvalue())
        assert decoded.mode == "RGB"
        pixels = clip_pixels(decoded)
        assert pixels.shape == (3, 224, 224)
        assert pixels.dtype == np.float32
        assert np.isfinite(pixels).all()
    with pytest.raises(ValueError):
        normalize(np.zeros(512))


def test_feedback_moves_toward_selected_vectors():
    q = np.array([1., 0.])
    positives = np.array([[0., 1.]])
    refined = refine_query(q, positives)
    assert refined[1] > q[1]
    assert refined[0] > refined[1]
    assert np.linalg.norm(refined) == pytest.approx(1)
    with pytest.raises(ValueError, match="Select"):
        refine_query(q, np.empty((0, 2)))
