"""Plot Cartesian trajectories and finite-difference diagnostics without a GUI."""

import argparse
from pathlib import Path

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def create_figure(reports, labels, *, projection="3d", frame="unspecified", title=None):
    """Plot reports returned by analyze_trajectory; all positions are in meters."""
    if not reports or len(reports) != len(labels):
        raise ValueError("provide one label for every trajectory")
    if projection not in {"3d", "xy", "xz", "yz"}:
        raise ValueError("projection must be 3d, xy, xz, or yz")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = (
        "Noto Sans CJK SC"
        if any(f.name == "Noto Sans CJK SC" for f in font_manager.fontManager.ttflist)
        else "DejaVu Sans"
    )
    with plt.rc_context(
        {
            "font.family": font,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#708090",
            "text.color": "#19334b",
            "axes.labelcolor": "#19334b",
            "grid.color": "#dce4eb",
            "grid.alpha": 0.7,
            "svg.fonttype": "none",
        }
    ):
        fig = plt.figure(figsize=(12.8, 8.6), layout="constrained")
        grid = fig.add_gridspec(3, 2, width_ratios=[1.08, 1])
        path_ax = fig.add_subplot(
            grid[:, 0], projection="3d" if projection == "3d" else None
        )
        diagnostic_axes = [fig.add_subplot(grid[row, 1]) for row in range(3)]
        coordinates = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        all_positions = []
        missing = {"velocity": [], "acceleration": [], "jerk": []}

        for index, (report, label) in enumerate(zip(reports, labels, strict=True)):
            color = COLORS[index % len(COLORS)]
            position = report["series"]["position"]
            xyz = position["xyz"]
            all_positions.extend(xyz)
            origin = position["time_s"][0]
            if projection == "3d":
                columns = list(zip(*xyz, strict=True))
                path_ax.plot(*columns, color=color, linewidth=1.7, label=label)
                path_ax.scatter(*xyz[0], color=color, marker="o", s=35)
                path_ax.scatter(*xyz[-1], color=color, marker="X", s=55)
            else:
                first, second = coordinates[projection]
                path_ax.plot(
                    [p[first] for p in xyz],
                    [p[second] for p in xyz],
                    color=color,
                    linewidth=1.7,
                    label=label,
                )
                path_ax.scatter(
                    xyz[0][first], xyz[0][second], color=color, marker="o", s=35
                )
                path_ax.scatter(
                    xyz[-1][first], xyz[-1][second], color=color, marker="X", s=55
                )

            for kind, axis in zip(missing, diagnostic_axes, strict=True):
                series = report["series"][kind]
                if series is None:
                    missing[kind].append(label)
                    continue
                axis.plot(
                    [t - origin for t in series["time_s"]],
                    series["norm"],
                    color=color,
                    linewidth=1.6,
                    marker="o" if len(series["time_s"]) == 1 else None,
                    markersize=4,
                    label=label,
                )

        if projection == "3d":
            bounds = [
                (min(p[i] for p in all_positions), max(p[i] for p in all_positions))
                for i in range(3)
            ]
            half_span = max(*(high - low for low, high in bounds), 0.01) * 0.56
            for setter, (low, high) in zip(
                (path_ax.set_xlim, path_ax.set_ylim, path_ax.set_zlim),
                bounds,
                strict=True,
            ):
                center = (low + high) / 2
                setter(center - half_span, center + half_span)
            path_ax.set_box_aspect((1, 1, 1))
            path_ax.set_xlabel("X (m)", labelpad=8)
            path_ax.set_ylabel("Y (m)", labelpad=8)
            path_ax.set_zlabel("Z (m)", labelpad=8)
            path_ax.view_init(elev=24, azim=-58)
        else:
            first, second = coordinates[projection]
            path_ax.set_xlabel(f"{'XYZ'[first]} (m)")
            path_ax.set_ylabel(f"{'XYZ'[second]} (m)")
            path_ax.set_aspect("equal", adjustable="datalim")
            path_ax.grid(True)
        path_ax.set_title("Cartesian path   |   circle: start   X: end", pad=18)
        path_ax.legend(loc="upper left", frameon=False)

        duration = max(
            report["series"]["position"]["time_s"][-1]
            - report["series"]["position"]["time_s"][0]
            for report in reports
        )
        for kind, axis, heading, unit in zip(
            missing,
            diagnostic_axes,
            ("Speed", "Acceleration magnitude", "Jerk magnitude"),
            ("m/s", "m/s²", "m/s³"),
            strict=True,
        ):
            axis.set_title(heading, loc="left", fontweight="bold")
            axis.set_ylabel(unit)
            axis.set_xlabel("Time from each trajectory start (s)")
            axis.grid(True)
            axis.set_ylim(bottom=0)
            axis.set_xlim(0, duration if duration > 0 else 1)
            if axis.lines:
                axis.legend(loc="upper right", fontsize=8, frameon=False)
            if missing[kind]:
                axis.text(
                    0.02,
                    0.07,
                    "Unavailable: " + ", ".join(missing[kind]),
                    transform=axis.transAxes,
                    fontsize=8,
                    color="#8a5d19",
                    bbox={"facecolor": "#fff8e8", "edgecolor": "none", "alpha": 0.9},
                )
        fig.suptitle(
            f"{title or 'Trajectory diagnostics'}  |  frame: {frame}\n"
            "Raw samples · no smoothing or resampling · derivative magnitudes use their own timestamps",
            fontsize=13,
        )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="PNG or SVG destination"
    )
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--projection", choices=("3d", "xy", "xz", "yz"), default="3d")
    parser.add_argument("--time-unit", choices=("s", "ms"), default="s")
    parser.add_argument("--position-unit", choices=("m", "cm", "mm"), default="m")
    parser.add_argument("--frame", default="unspecified")
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".png", ".svg"}:
        parser.error("--output must end with .png or .svg")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.labels is not None and len(args.labels) != len(args.input):
        parser.error("--labels must contain one label per input")
    if args.output.resolve() in {p.resolve() for p in args.input}:
        parser.error("output cannot overwrite an input trajectory")
    if __package__:
        from .trajectory_metrics import analyze_trajectory, load_trajectory
    else:
        from trajectory_metrics import analyze_trajectory, load_trajectory

    figure = None
    try:
        reports = [
            analyze_trajectory(
                *load_trajectory(
                    path, time_unit=args.time_unit, position_unit=args.position_unit
                )
            )
            for path in args.input
        ]
        figure = create_figure(
            reports,
            args.labels or [p.stem for p in args.input],
            projection=args.projection,
            frame=args.frame,
            title=args.title,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            args.output, dpi=args.dpi, facecolor="white", bbox_inches="tight"
        )
    except ImportError as exc:
        parser.exit(2, f"error: install tools/embodied/requirements-plot.txt ({exc})\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    finally:
        if figure is not None:
            import matplotlib.pyplot as plt

            plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
