"""Regenerate a Kinetix run's per-cell latency-quality summaries."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from robotics_bench.kinetix.results import write_summary  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.input / "case-manifest.json").read_text())
    complete = manifest["status"] == "completed"
    if not complete and not args.allow_partial:
        parser.error(
            "Run is incomplete; use --allow-partial to inspect completed episodes"
        )
    rows = [
        json.loads(line)
        for line in (args.input / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    write_summary(args.input, rows, complete=complete)
    print((args.input / "summary.md").read_text())


if __name__ == "__main__":
    main()
