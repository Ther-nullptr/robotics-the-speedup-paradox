"""Plot measured endpoint recovery and internal-grid continuity probes."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compare-run", type=Path)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("The research run must be complete")
    toy = manifest["toy"]
    delays = [r["delay_ms"] for r in toy]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7), constrained_layout=True)
    axes[0].plot(
        delays,
        [r["one_sided_dx"] for r in toy],
        "o--",
        label="New-anchor formula",
        color="#c44e52",
        markersize=3,
    )
    axes[0].plot(
        delays,
        [r["balanced_dx"] for r in toy],
        "o-",
        label="Two-endpoint formula",
        color="#0072b2",
        markersize=3,
    )
    axes[0].scatter(
        [delays[-1]],
        [0],
        marker="x",
        s=60,
        color="black",
        label="Native old-action endpoint",
    )
    period = delays[-1] / 1000
    axes[0].plot(
        delays,
        [0.5 * r["expected_vx"] * (period - r["delay_ms"] / 1000) for r in toy],
        ":",
        color="0.4",
        label="Continuous free-flight solution",
    )
    axes[0].set(
        title="Isolated thruster: endpoint bias",
        xlabel="Simulated delay (ms)",
        ylabel="Displacement (world units)",
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(
        delays,
        [r["expected_vx"] for r in toy],
        "--",
        color="black",
        label="Analytical impulse response",
    )
    axes[1].scatter(
        delays,
        [r["actual_vx"] for r in toy],
        s=20,
        color="#0072b2",
        label="Two-endpoint result",
    )
    axes[1].set(
        title="Velocity response remains delay-dependent",
        xlabel="Simulated delay (ms)",
        ylabel="Velocity (world units/s)",
    )
    axes[1].legend(fontsize=8)
    boundaries = []
    sources = [(args.run_dir, manifest, "-")]
    if args.compare_run:
        compared = json.loads((args.compare_run / "manifest.json").read_text())
        if compared["status"] != "complete" or compared["factor"] != manifest["factor"]:
            raise ValueError(
                "Comparison requires a complete run at the same refinement factor"
            )
        fingerprints = {
            task["task"]: task["input_tape_sha256"] for task in manifest["tasks"]
        }
        if any(
            fingerprints.get(task["task"]) != task["input_tape_sha256"]
            for task in compared["tasks"]
        ):
            raise ValueError("Comparison runs must use identical task action tapes")
        sources.append((args.compare_run, compared, "--"))
    for folder, source, style in sources:
        for task, color in zip(source["tasks"], ("#e69f00", "#009e73"), strict=False):
            rows = [r for r in task["probes"] if r["native_tick"] == 0]
            if not rows:
                continue
            boundary = (
                rows[0]["plan"]["control_dt_seconds"] * 1000 / (2 * source["factor"])
            )
            center = min(
                range(len(rows)),
                key=lambda index: abs(rows[index]["delay_ms"] - boundary),
            )
            with np.load(
                folder / f"{task['task']}_tick0_delay{center}.npz", allow_pickle=False
            ) as data:
                baseline = {
                    k: data[k]
                    for k in data
                    if k.startswith("balanced_") and k.endswith("_position")
                }
            offsets, gaps = [], []
            for index, row in enumerate(rows):
                offset = row["delay_ms"] - rows[center]["delay_ms"]
                if not 0 < offset < 0.002:
                    continue
                with np.load(
                    folder / f"{task['task']}_tick0_delay{index}.npz",
                    allow_pickle=False,
                ) as data:
                    gap = max(
                        float(np.max(np.linalg.norm(data[k] - baseline[k], axis=-1)))
                        for k in baseline
                    )
                offsets.append(offset)
                gaps.append(gap)
            if offsets:
                axes[2].loglog(
                    offsets,
                    gaps,
                    "o" + style,
                    label=task["task"] + " / " + source.get("fine_model", "split"),
                    color=color,
                )
                boundaries.append(
                    {
                        "task": task["task"],
                        "fine_model": source.get("fine_model", "split"),
                        "native_tick": 0,
                        "boundary_ms": boundary,
                        "offset_ms": offsets,
                        "max_body_position_gap": gaps,
                    }
                )
    axes[2].set(
        title="Internal grid boundary: proposals only",
        xlabel="Positive offset from the boundary (ms)",
        ylabel="Max body position difference (world units)",
    )
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Native baseline + coupled fine-step residuals | factor 2 | endpoint recovery is not physical equivalence",
        fontsize=12,
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.run_dir / f"calibration-study.{extension}", dpi=160)
    plt.close(fig)
    summary = {
        "max_toy_velocity_abs_error": max(
            abs(r["actual_vx"] - r["expected_vx"]) for r in toy
        ),
        "single_anchor_full_delay_displacement_bias": toy[-1]["one_sided_dx"],
        "two_endpoint_full_delay_displacement_bias": toy[-1]["balanced_dx"],
        "boundary_probes": boundaries,
    }
    (args.run_dir / "plot-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(args.run_dir / "calibration-study.png")


if __name__ == "__main__":
    main()
