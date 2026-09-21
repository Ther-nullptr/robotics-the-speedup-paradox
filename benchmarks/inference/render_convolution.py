#!/usr/bin/env python3
"""Plot measured convolution tactics without starting a GPU workload."""

import argparse
from datetime import datetime
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve()
    records = json.loads((source / "measurements.json").read_text())
    metadata = json.loads((source / "manifest.json").read_text())
    complete = [row for row in records if row["scope"] == "complete_conv"]
    baselines = {
        row["case_id"]: row for row in complete if row["backend"] == "native_cudnn"
    }
    candidates = {
        (row["case_id"], row["tactic"]): row for row in complete if "tactic" in row
    }
    ids = sorted(baselines)
    tactics = sorted({tactic for _, tactic in candidates})
    if not ids or not tactics:
        raise ValueError("No comparable native and CUTLASS records")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    shape_by_id = {row["id"]: row["input_shape"] for row in metadata["cases"]}
    fig, axes = plt.subplots(
        1, 2, figsize=(15, max(4.8, 0.68 * len(ids) + 2.6)), sharey=True
    )
    rows = []
    for axis, (key, label) in zip(
        axes,
        [
            ("median_ms", "Complete convolution: CUDA events"),
            ("median_eager_wall_ms", "Complete convolution: synchronized eager wall"),
        ],
    ):
        values = np.full((len(ids), len(tactics)), np.nan)
        for i, cid in enumerate(ids):
            for j, tactic in enumerate(tactics):
                candidate = candidates.get((cid, tactic))
                if not candidate or key not in candidate:
                    continue
                ratio = baselines[cid][key] / candidate[key]
                values[i, j] = ratio
                accepted = candidate.get("numerical", {}).get(
                    "allclose_rtol_002_atol_002"
                )
                text = f"{ratio:.2f}x" + (" *" if accepted is False else "")
                axis.text(j, i, text, ha="center", va="center", fontsize=9)
                rows.append(
                    {
                        "case_id": cid,
                        "tactic": tactic,
                        "metric": key,
                        "baseline_ms": baselines[cid][key],
                        "candidate_ms": candidate[key],
                        "ratio": ratio,
                        "operator_tolerance_pass": accepted,
                    }
                )
        image = axis.imshow(values, cmap="RdYlGn", vmin=0.5, vmax=1.5, aspect="auto")
        axis.set_title(label, fontsize=11)
        axis.set_xticks(range(len(tactics)), [f"T{tactic}" for tactic in tactics])
        axis.set_xlabel("Explicit CUTLASS tactic")
    axes[0].set_yticks(
        range(len(ids)), [cid + "  " + str(shape_by_id[cid]) for cid in ids], fontsize=9
    )
    warmup = metadata.get("warmup_ms")
    if warmup is None:
        note = "Exploratory short-warmup records; clock drift observed. Do not treat these as steady-state gains."
    else:
        note = f"Sustained warmup: {warmup:g} ms per scope. Repeated medians; sample spread retained in source JSON."
    fig.suptitle(
        f"{source.name} | cuDNN benchmark={metadata['cudnn_benchmark']}", fontsize=13
    )
    fig.text(0.5, 0.045, note, ha="center", fontsize=9)
    fig.text(
        0.5,
        0.013,
        "Ratios = same-cohort native / CUTLASS; >1 is faster. Layout conversion included. Color range 0.5–1.5x; labels are unclipped. * = tolerance failure.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.10, 0.96, 0.94))
    color_axis = fig.add_axes((0.965, 0.23, 0.012, 0.5))
    fig.colorbar(image, cax=color_axis)
    output = (args.output_dir or source / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    prefix = output / f"{stamp}_convolution_tactics"
    for suffix in ("png", "svg"):
        fig.savefig(prefix.with_suffix("." + suffix), dpi=160)
    plt.close(fig)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(source),
                "new_gpu_measurement": False,
                "notes": note,
                "comparisons": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "png": str(prefix.with_suffix(".png")),
                "svg": str(prefix.with_suffix(".svg")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
