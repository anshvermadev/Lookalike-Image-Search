import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isr.retrieval import build_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build exact image retrieval indexes from gallery images only.")
    parser.add_argument("--method", choices=["histogram", "clip", "all"], default="all")
    args = parser.parse_args()
    for method in (["histogram", "clip"] if args.method == "all" else [args.method]):
        def progress(done, total):
            if done % 80 == 0 or done == total:
                print(f"{method}: {done}/{total}", flush=True)
        print(json.dumps(build_index(method, progress=progress), indent=2), flush=True)
