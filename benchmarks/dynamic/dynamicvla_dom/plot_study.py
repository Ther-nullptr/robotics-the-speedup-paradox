"""Plot audited streaming delay studies without launching models or simulation."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    rows = [r for r in data["task_conditions"] if r["status"] == "completed"]
    if not rows:
        raise ValueError("No completed, audited task conditions to plot")
    steps = {r["num_steps"] for r in rows}
    if len(steps) != 1:
        raise ValueError("Plot one fixed-refinement pure-delay study at a time")
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    groups = [(task, [r for r in rows if r["task"] == task]) for task in tasks]
    pooled = [r for r in data["pooled_conditions"] if r["status"] == "completed"]
    if pooled:
        groups.append(("Pooled", pooled))
    dense = "included_blocks" in data
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2 if dense else 4.8))
    colors = ["#4477AA", "#EE6677", "#228833", "#AA3377", "#66CCEE"]
    handles = []
    for index, (name, selected) in enumerate(groups):
        selected = sorted(selected, key=lambda r: r["extra_delay_ms"])
        color = "#222222" if name == "Pooled" else colors[index % len(colors)]
        marker = "s" if name == "Pooled" else "o"
        label = (
            name
            if name == "Pooled"
            else (name.split("_")[3] if len(name.split("_")) > 3 else name)
        )
        handles.append(Line2D([0], [0], color=color, marker=marker, label=label))
        metrics = [r["metrics"] for r in selected]
        x = [m["effective_service_ms_median"] for m in metrics]
        y = np.array([m["success_rate"] * 100 for m in metrics])
        bounds = np.array(
            [
                [m["success_rate_wilson95_low"] * 100 for m in metrics],
                [m["success_rate_wilson95_high"] * 100 for m in metrics],
            ]
        )
        axes[0].errorbar(
            x,
            y,
            yerr=np.vstack([y - bounds[0], bounds[1] - y]),
            color=color,
            marker=marker,
            linewidth=1.3,
            capsize=2,
            alpha=0.8,
        )
        delay = [r["extra_delay_ms"] for r in selected]
        for ax, key in (
            (axes[1], "worker_compute_ms_median"),
            (axes[2], "action_age_sim_ms_median"),
        ):
            values = [np.nan if m[key] is None else m[key] for m in metrics]
            ax.plot(delay, values, color=color, marker=marker, linewidth=1.3, alpha=0.8)
    labels = [
        ("Effective service latency (ms)", "Task success (%)"),
        ("Added wall delay (ms)", "Worker compute median (ms)"),
        ("Added wall delay (ms)", "Generating-observation age (sim ms)"),
    ]
    for ax, (x, y) in zip(axes, labels):
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(-3, 103)
    title = f"Native streaming: latency and task success ({next(iter(steps))} refinement steps)"
    if dense:
        title += f"\n{'Interim' if data['interim'] else 'Final'}: {data['included_blocks']}/{data['planned_blocks']} balanced blocks, {data['included_episodes']} episodes"
    fig.suptitle(title, y=0.99, fontsize=13 if dense else None)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88 if dense else 0.91),
        ncol=len(handles),
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "Extra delay is synthetic. Compute excludes pacing/delay. Error bars: descriptive Wilson 95% intervals. Missing ages mean no applied model actions.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.065, right=0.99, top=0.73 if dense else 0.77, bottom=0.19, wspace=0.33
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            args.output_dir / f"latency-success.{ext}", dpi=180, bbox_inches="tight"
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
