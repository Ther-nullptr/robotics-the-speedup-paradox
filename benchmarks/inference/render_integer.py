"""Render an existing integer tuning cohort; never launches GPU work."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Completed bench_integer output directory",
    )
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--render-python", default=sys.executable)
    args = parser.parse_args(argv)
    directory = args.input.resolve()
    metadata = json.loads((directory / "manifest.json").read_text())
    records = json.loads((directory / "measurements.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("Render only a completed tuning cohort")
    data = {
        "schema_version": 1,
        "title": "Integer Linear tactic tuning",
        "evidence_kind": "measured",
        "protocols": [],
        "optimizations": [],
        "states": [],
        "runs": [],
        "comparisons": [],
        "notes": [
            "GPU event timings per repeated graph node; not policy-service or task latency.",
            "Online activation preparation is included. Weight packing, compilation and graph capture are excluded.",
            "Tactics match the same quantized reference. Model quality versus BF16 is unmeasured here.",
            "Every tested tactic is retained. Best single-shape results do not establish model speedup.",
        ],
    }
    tactics = sorted({r["tactic"] for r in records if "tactic" in r})
    data["optimizations"] = [
        {"id": f"tactic-{t}", "label": f"Tactic {t}", "scope": "precision"}
        for t in tactics
    ]
    keys = list(dict.fromkeys((r["case"], tuple(r["shape"])) for r in records))
    for case, shape in keys:
        prefix = case + "-" + "x".join(map(str, shape))
        rows = [
            r
            for r in records
            if r["case"] == case
            and r["shape"] == list(shape)
            and r["scope"] == "complete_linear"
        ]
        rows.sort(
            key=lambda r: (
                r["precision"] != "bf16",
                r["precision"],
                r.get("tactic", -1),
            )
        )
        anchor = prefix + "-bf16"
        workload = {
            "shape_MNK": list(shape),
            "case_shape_source": case,
            "boundary": metadata["boundary"],
            "scope": "complete_linear",
            "graph_inner": metadata["graph_inner"],
            "pack_reuse": metadata["pack_reuse"],
        }
        data["protocols"].append(
            {
                "id": prefix,
                "label": prefix,
                "original_state_id": anchor,
                "workload": workload,
                "environment": {
                    k: metadata[k] for k in ("device", "capability", "torch", "cuda")
                },
                "metric": {
                    "name": "Complete Linear GPU time",
                    "unit": "ms",
                    "statistic": "median",
                },
            }
        )
        for row in rows:
            baseline = row["precision"] == "bf16"
            rid = (
                anchor if baseline else prefix + f"-{row['precision']}-t{row['tactic']}"
            )
            state = {
                "id": rid,
                "label": "BF16"
                if baseline
                else f"{row['precision'].upper()} tactic {row['tactic']}",
                "revision": "Source fingerprints in manifest.json",
                "precision": row["precision"],
                "classification": "reference" if baseline else "lossy",
                "status": "accepted" if baseline else "experimental",
                "description": "Reference math checked separately from model action quality",
                "shared_config": workload,
                "active_optimizations": [] if baseline else [f"tactic-{row['tactic']}"],
            }
            if not baseline:
                state["parent_id"] = anchor
                data["comparisons"].append(
                    {
                        "id": "vs-" + rid,
                        "label": "Versus BF16 Linear",
                        "kind": "cumulative",
                        "baseline_run_id": anchor,
                        "candidate_run_id": rid,
                    }
                )
            data["states"].append(state)
            data["runs"].append(
                {
                    "id": rid,
                    "state_id": rid,
                    "protocol_id": prefix,
                    "cohort_id": directory.name,
                    "recorded_at": metadata["recorded_at"],
                    "samples": row["samples_ms"],
                    "sources": ["measurements.json", "manifest.json"],
                    "quality": {"status": "unmeasured"},
                }
            )
    data["current_run_id"] = data["runs"][0]["id"]
    ledger = directory / "ledger.json"
    ledger.write_text(json.dumps(data, indent=2) + "\n")
    subprocess.run(
        [
            args.render_python,
            str(args.skill.resolve() / "scripts/render_optimization_history.py"),
            str(ledger),
            "--output-dir",
            str(directory / "figures"),
            "--format",
            "both",
            "--update-index",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
