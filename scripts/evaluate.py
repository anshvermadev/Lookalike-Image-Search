import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isr.evaluation import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate held-out queries; save per-query and aggregate results.")
    parser.add_argument("--method", choices=["histogram", "clip", "all"], default="all")
    args = parser.parse_args()
    methods = ("histogram", "clip") if args.method == "all" else (args.method,)
    def progress(method, done, total):
        if done % 20 == 0 or done == total:
            print(f"{method}: {done}/{total} queries", flush=True)
    report = evaluate(methods, progress=progress)
    print(json.dumps(report["summaries"], indent=2))
