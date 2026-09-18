# Validation and Troubleshooting

Use this guide after the [README setup](../README.md#installation-from-a-fresh-clone). Run commands from the repository root using the project virtual environment. The [project guide](PROJECT_GUIDE.md) defines the algorithms and metrics being checked.

## 1. Verify the dataset

```powershell
.venv\Scripts\python.exe scripts/verify_dataset.py
```

Expected: 500 decodable images across 10 categories, with 40 gallery images and 10 query images per category. Every manifest path must exist, match its recorded SHA-256, and remain inside the dataset directory. No identical file hash may appear twice.

The script produces `artifacts/dataset-validation.json` and `artifacts/dataset-preview.jpg`. Inspect the preview to confirm the categories look plausible. Hash validation establishes exact-file uniqueness; it does not establish the absence of near duplicates.

## 2. Verify model acquisition and indexing

```powershell
.venv\Scripts\python.exe scripts/download_model.py
.venv\Scripts\python.exe scripts/build_index.py --method all
```

The downloader checks the pinned model hash. Each method should produce an index with 400 vectors of dimension 512. Inspect `indexes/clip/metadata.json` and `indexes/histogram/metadata.json` for counts, dimension, feature version, dataset fingerprint, timing, and file hashes.

Each directory contains:

| File | Purpose |
|---|---|
| `search.faiss` | Exact inner-product search index |
| `vectors.npy` | Saved vectors, also used by feedback |
| `records.json` | Vector ID to gallery image mapping |
| `metadata.json` | Identity and integrity checks |

Loading rejects changed datasets, incompatible feature versions, changed CLIP weights, modified index files, and inconsistent dimensions. Rebuild after intentional changes rather than editing metadata to bypass a check.

## 3. Generate evaluation outputs

```powershell
.venv\Scripts\python.exe scripts/evaluate.py --method all
```

Expected: 100 queries evaluated under each method against the same 400-image gallery. Outputs under `artifacts/` are:

- `evaluation.json`: aggregate and category summaries with protocol metadata.
- `evaluation-queries.csv`: per-query scores and timings.
- `evaluation-categories.csv`: category averages.
- `evaluation-summary.csv`: method averages and index information.

Sanity checks: metrics should be within [0,1], Recall@10 cannot exceed 0.25 for this split, and all measured durations should be nonnegative. The reported mAP@10 uses the min(R,10) denominator explained in the project guide. Evaluation does not apply user feedback.

Historical reference values from 18 September 2026 are:

| Method | P@10 | mAP@10 | Mean total query time |
|---|---:|---:|---:|
| Histogram | 0.297 | 0.21266 | 6.11 ms |
| CLIP | 0.985 | 0.97898 | 40.93 ms |

These are observations, not assertions that must hold for every environment. Different data, weights, or numerical runtimes may change rankings. Latency varies with hardware and load. These values do not represent general classification accuracy.

## 4. Run automated tests

For the full suite, prepare the dataset, model, both indexes, and evaluation report first:

```powershell
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

The current suite contains **13 tests**. The last full run after removal of the obsolete Streamlit prototype passed all 13. That historical run was on Windows with Python 3.13; the documentation rewrite itself does not constitute a new full test run.

| Test file | Coverage |
|---|---|
| `tests/test_retrieval.py` | Known rankings, index persistence, query/gallery separation, dataset/index corruption, safe paths, duplicate rejection, hand-calculated metrics, image validation, preprocessing, feedback |
| `tests/test_api.py` | Frontend and overview, search, upload comparison, feedback/reset, invalid requests, collection, evaluation, downloads, image access, request guards |

For the core tests without downloading real runtime assets:

```powershell
.venv\Scripts\python.exe -B -m pytest tests/test_retrieval.py -q -p no:cacheprovider
```

The core suite builds small temporary fixtures. API tests use the prepared project assets. Do not interpret an incomplete or skipped test run as verification of the entire app.

In a restricted execution sandbox, Windows may deny pytest temporary-folder access. That occurred during development; the same suite passed when run with permitted filesystem access. Distinguish temporary-directory errors from failed ranking assertions.

## 5. Manual browser checks

Start the server:

```powershell
.venv\Scripts\python.exe server.py
```

Open http://127.0.0.1:8000 and perform these checks:

| Action | Expected behavior |
|---|---|
| Select a Cups example and search with CLIP | Ranked gallery cards appear with scores and a best-match badge |
| Open the first result | Preview percentage matches the card, with raw cosine also shown |
| Change result count | The next search returns the selected top K |
| Compare methods | Both methods use the same query and produce separate rankings |
| Select relevant cards and refine | Search uses the refined vector; results may change but are not guaranteed to improve |
| Reset feedback | Original-query retrieval is restored |
| Upload a valid photo | Its preview replaces the example and search works |
| Try an unreadable or oversized image | An explanatory validation error appears |
| Open Evaluation and export CSV | Saved metrics render and the export is usable |
| Browse Collection | Categories and pagination display dataset images |
| Resize the window | Hero fits the first viewport; controls and results remain usable |

Recorded browser checks covered example search, uploaded laptop search, feedback, comparison, evaluation, and responsive layouts. The percentage display was checked with a live cup query whose best match had cosine approximately 0.849 and displayed 84.9%.

## 6. Troubleshooting

| Problem | What to check or do |
|---|---|
| Missing manifest or dataset | Prepare the ZIP into the default `data/caltech101-small` directory |
| Preparation refuses a nonempty folder | Verify and reuse the existing subset; do not overwrite it blindly |
| A custom preparation output is not used | Runtime paths in `isr/config.py` and the verification script assume the default dataset location |
| Direct Kaggle download fails | Download the exact ZIP manually through Kaggle |
| Model download fails | Check connectivity; retry with `--workers 1` for the nonparallel path |
| Model checksum mismatch | Rerun the downloader; do not bypass checksum checks |
| Missing or stale index | Verify the dataset, rebuild the affected method, and restart the server |
| Evaluation page has no report | Generate evaluation after both indexes are built |
| A locked dependency has no compatible wheel | Use a supported Python/platform combination or try `requirements.txt`, then verify the environment |
| Port 8000 is busy | Stop the existing server with Ctrl+C before starting another |
| Feedback token expired | Run a new search; tokens expire and are cleared on restart |
| Frontend edit is not visible | Refresh the browser; restart for Python code changes |
| An unfamiliar object returns unrelated images | The system searches only the ten-category gallery and has no rejection threshold |

## 7. What has not been established

The checks do not establish exhaustive near-duplicate removal, absence of CLIP pretraining overlap, performance on arbitrary categories, equivalence to full-precision CLIP, or public-deployment security. No macOS/Linux validation has been recorded.

The pretrained model folder is included in the repository. Dataset files, generated indexes, and reports are ignored by Git. A fresh clone must prepare those assets before full verification can succeed.
