"""Plot executed Cosmos precision coverage and measured policy latency on CPU."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Completed progressive inference output directory",
    )
    args = parser.parse_args(argv)
    directory = args.input.resolve()
    source = directory / "measurements.json"
    records = json.loads(source.read_text())
    metadata = json.loads((directory / "manifest.json").read_text())
    if metadata.get("status") != "completed":
        raise ValueError("Only completed progressive cohorts can be plotted")
    from robotics_bench.optimizations.progressive import cosmos_candidate_sites
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import numpy as np

    roles = [
        f"{a}.{p}"
        for a in ("self_attn", "cross_attn")
        for p in ("q_proj", "k_proj", "v_proj", "output_proj")
    ] + ["mlp.layer1", "mlp.layer2"]
    ordered = [f"blocks.{block}.{role}" for role in roles for block in range(28)]
    rows, labels, counts = [], [], []
    for record in records:
        if record.get("quant_tier") is None:
            continue
        modules = record["coverage"]["quantization"]["modules"]
        modules = {
            name.replace("._checkpoint_wrapped_module", ""): value
            for name, value in modules.items()
        }
        if set(modules) != set(cosmos_candidate_sites()):
            raise ValueError(
                "Recorded candidate coverage does not match the Cosmos protocol"
            )
        bits = []
        for name in ordered:
            backend = modules[name]["backend"]
            if backend.startswith("cutlass_sm80_int4_"):
                bits.append(4)
            elif backend.startswith("cutlass_sm80_int8_"):
                bits.append(8)
            else:
                raise ValueError(f"Unverified backend in coverage: {backend}")
        rows.append(bits)
        counts.append(bits.count(4))
        labels.append(f"{record['id']} ({bits.count(4)}/280)")
    if not rows:
        raise ValueError("No measured progressive tiers in this cohort")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        gridspec_kw={"height_ratios": [1.3, 1]},
        layout="constrained",
    )
    colors = ["#397c9d", "#d98243"]
    axes[0].imshow(
        np.asarray(rows) == 4,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(colors),
        vmin=0,
        vmax=1,
    )
    axes[0].set_yticks(range(len(rows)), labels, fontsize=9)
    axes[0].set_xticks(
        [28 * i + 13.5 for i in range(10)],
        [
            "SA.Q",
            "SA.K",
            "SA.V",
            "SA.Out",
            "CA.Q",
            "CA.K",
            "CA.V",
            "CA.Out",
            "MLP.1",
            "MLP.2",
        ],
    )
    for role in range(10):
        if role:
            axes[0].axvline(role * 28 - 0.5, color="white", linewidth=1.5)
        for boundary in (9.5, 19.5):
            axes[0].axvline(
                role * 28 + boundary, color="white", linestyle=":", linewidth=0.8
            )
    axes[0].set_title("Executed Linear precision (count of W4A4 sites)", loc="left")
    axes[0].set_xlabel(
        "Within each role: B0 to B27; dashed boundaries after B9 and B19"
    )
    axes[0].legend(
        handles=[
            Patch(color=colors[0], label="W8A8"),
            Patch(color=colors[1], label="W4A4"),
        ],
        loc="upper right",
        bbox_to_anchor=(1, 1.13),
        ncol=2,
        frameon=False,
    )
    values = np.asarray([r["median_ms"] for r in records])
    low = np.asarray([min(r["samples_ms"]) for r in records])
    high = np.asarray([max(r["samples_ms"]) for r in records])
    x = np.arange(len(records))
    axes[1].bar(
        x,
        values,
        color=[
            "#71808c"
            if r["precision"] == "bf16"
            else colors[0]
            if r.get("quant_tier") == 0
            else colors[1]
            for r in records
        ],
    )
    axes[1].errorbar(
        x,
        values,
        yerr=[values - low, high - values],
        fmt="none",
        ecolor="#24333d",
        capsize=2,
    )
    axes[1].set_xticks(
        x, [r["id"] for r in records], rotation=25, ha="right", fontsize=9
    )
    axes[1].set_ylabel("Complete policy latency (ms)")
    axes[1].set_title(
        "Measured medians; whiskers show sample ranges, not confidence intervals",
        loc="left",
        fontsize=11,
    )
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_axisbelow(True)
    steps = metadata["model"]["steps"]
    figure.suptitle(
        f"Cosmos progressive quantization: {steps}-step policy inference", fontsize=16
    )
    figure.supxlabel(
        "One fixed observation, seed and GPU. Loading, weight packing and warmup excluded. Task outcomes are reported separately.",
        fontsize=9,
    )
    output = directory / "figures"
    output.mkdir(exist_ok=True)
    stamp = datetime.now().astimezone()
    stem = stamp.strftime("%Y%m%d_%H%M%S") + "_cosmos_precision_coverage"
    if (output / (stem + ".png")).exists():
        raise FileExistsError("A figure with this timestamp already exists")
    for extension in ("png", "svg"):
        figure.savefig(output / (stem + "." + extension), dpi=160)
    plt.close(figure)
    (output / (stem + ".json")).write_text(
        json.dumps(
            {
                "recorded_at": stamp.isoformat(),
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "w4_counts": counts,
                "boundary": "Executed per-module backend coverage and measured policy-service latency",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "png": str(output / (stem + ".png")),
                "svg": str(output / (stem + ".svg")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
