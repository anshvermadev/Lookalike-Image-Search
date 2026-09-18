from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config import ARTIFACTS, DATA, INDEXES, METHODS
from .dataset import fingerprint, image_path, load_records
from .features import Encoder, open_rgb
from .retrieval import SearchIndex


def metrics(relevance: list[bool], total_relevant: int) -> dict:
    """AP@10 uses min(total relevant, 10); MRR is explicitly truncated at 10."""
    if total_relevant <= 0:
        raise ValueError("Evaluation requires at least one relevant gallery image.")
    rel = np.asarray(relevance[:10], dtype=np.float64)
    if len(rel) < 10:
        raise ValueError("At least 10 results are required for this evaluation.")
    p5, p10, recall = float(rel[:5].mean()), float(rel.mean()), float(rel.sum() / total_relevant)
    ranks = np.arange(1, 11)
    ap = float(np.sum(np.cumsum(rel) / ranks * rel) / min(total_relevant, 10))
    relevant_positions = np.flatnonzero(rel)
    mrr = float(1 / (relevant_positions[0] + 1)) if len(relevant_positions) else 0.0
    dcg = np.sum(rel / np.log2(ranks + 1))
    ideal = np.sum(1 / np.log2(np.arange(1, min(total_relevant, 10) + 1) + 1))
    return {"precision_at_5": p5, "precision_at_10": p10, "recall_at_10": recall,
            "f1_at_10": 2 * p10 * recall / (p10 + recall) if p10 + recall else 0.0,
            "ap_at_10": ap, "mrr_at_10": mrr, "ndcg_at_10": float(dcg / ideal)}


def evaluate(methods=("histogram", "clip"), data_dir: Path = DATA, index_dir: Path = INDEXES,
             output_dir: Path = ARTIFACTS, progress=None) -> dict:
    records = load_records(data_dir)
    queries = [row for row in records if row.split == "queries"]
    totals = Counter(row.category for row in records if row.split == "gallery")
    if not queries:
        raise ValueError("No held-out queries are available.")
    rows, summaries = [], []
    for method in methods:
        index = SearchIndex(method, records, index_dir)
        encoder = Encoder(method)
        # Exclude model/session initialization and one warm-up from measured latency.
        encoder.encode([open_rgb(image_path(queries[0], data_dir))])
        method_rows = []
        for number, query in enumerate(queries):
            start = perf_counter()
            image = open_rgb(image_path(query, data_dir))
            decoded = perf_counter()
            vector = encoder.encode([image])[0]
            encoded = perf_counter()
            results = index.search(vector, k=10)
            finished = perf_counter()
            relevant = [result["category"] == query.category for result in results]
            row = {"method": method, "query": query.path, "category": query.category,
                   **metrics(relevant, totals[query.category]),
                   "decode_ms": (decoded - start) * 1000,
                   "encode_ms": (encoded - decoded) * 1000,
                   "search_ms": (finished - encoded) * 1000,
                   "total_ms": (finished - start) * 1000}
            method_rows.append(row)
            if progress:
                progress(method, number + 1, len(queries))
        frame = pd.DataFrame(method_rows)
        summary = {"method": method, "name": METHODS[method], "queries": len(queries),
                   **frame.select_dtypes(include="number").mean().to_dict(),
                   "p95_total_ms": float(frame.total_ms.quantile(0.95)),
                   "index_bytes": index.metadata["index_bytes"],
                   "build_seconds": index.metadata["build_seconds"],
                   "index_built_at": index.metadata["built_at"]}
        summary["map_at_10"] = summary.pop("ap_at_10")
        summaries.append(summary)
        rows.extend(method_rows)
    frame = pd.DataFrame(rows)
    per_category = frame.groupby(["method", "category"], as_index=False).mean(numeric_only=True)
    report = {"evaluated_at": datetime.now(timezone.utc).isoformat(),
              "dataset_fingerprint": fingerprint(records), "gallery_count": sum(totals.values()),
              "query_count": len(queries), "relevance": "Same category as the query image.",
              "ap_definition": "AP@10 = sum(precision at each relevant rank <=10) / min(R, 10).",
              "latency_definition": "Warm single-image decoding + encoding + exact search; excludes model loading and UI rendering.",
              "summaries": summaries, "per_category": per_category.to_dict(orient="records")}
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "evaluation-queries.csv", index=False)
    per_category.to_csv(output_dir / "evaluation-categories.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "evaluation-summary.csv", index=False)
    target = output_dir / "evaluation.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(target)
    return report
