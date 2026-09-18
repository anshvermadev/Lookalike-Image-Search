"""Download a pinned pretrained CLIP vision encoder; verify its SHA-256 hash."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isr.config import MODEL_DIR, MODEL_PATH, MODEL_REPO

REVISION = "d15189d7028b43f1d3e65039190477f6af591c2a"
FILENAME = "onnx/vision_model_quantized.onnx"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    api = f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(api, timeout=60) as response:
        info = json.load(response)
    sibling = next(row for row in info["siblings"] if row["rfilename"] == FILENAME)
    expected = sibling["lfs"]["sha256"]
    total = sibling["size"]
    revision = info["sha"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and sha256(MODEL_PATH) == expected:
        print("Verified existing pretrained CLIP model.", flush=True)
    else:
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{revision}/{FILENAME}?download=true"
        partial = MODEL_PATH.with_suffix(".onnx.part")
        if args.workers == 1:
            with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as output:
                downloaded = 0
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    print(f"CLIP: {downloaded / 1024**2:.0f} / {total / 1024**2:.0f} MiB", flush=True)
        else:
            block = 4 * 1024 * 1024
            starts = list(range(0, total, block))
            parts = MODEL_DIR / f"parts-{expected[:12]}"
            parts.mkdir(exist_ok=True)

            def fetch(start):
                end = min(start + block, total) - 1
                part = parts / f"{start}.part"
                if part.exists() and part.stat().st_size == end - start + 1:
                    return part
                for attempt in range(3):
                    try:
                        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                        with urllib.request.urlopen(req, timeout=90) as response:
                            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                                raise ValueError("Server did not honor ranges. Retry with --workers 1.")
                            data = response.read(end - start + 2)
                        if len(data) != end - start + 1:
                            raise ValueError("Incomplete model chunk; rerun the download.")
                        part.write_bytes(data)
                        return part
                    except (OSError, ValueError):
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)

            completed = 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for future in as_completed([pool.submit(fetch, start) for start in starts]):
                    completed += future.result().stat().st_size
                    print(f"CLIP: {completed / 1024**2:.0f} / {total / 1024**2:.0f} MiB", flush=True)
            with partial.open("wb") as output:
                for start in starts:
                    output.write((parts / f"{start}.part").read_bytes())
        if sha256(partial) != expected:
            raise ValueError("Model checksum mismatch. Remove incomplete model parts and download again.")
        partial.replace(MODEL_PATH)
    metadata = {"repo": MODEL_REPO, "revision": revision, "file": FILENAME,
                "sha256": expected, "bytes": total, "runtime": "ONNX Runtime CPU",
                "description": "Pretrained CLIP ViT-B/32 vision encoder, INT8 quantization; no project fine-tuning."}
    (MODEL_DIR / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Ready: {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
