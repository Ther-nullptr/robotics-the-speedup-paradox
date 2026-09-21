"""Plot analytic integration error separately from native-benchmark trajectory deviation."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run", type=Path, required=True)
    parser.add_argument("--freefall-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = json.loads((args.matrix_run / "manifest.json").read_text())
    gravity = json.loads((args.freefall_run / "manifest.json").read_text())
    setup = gravity.get("freefall_setup", {})
    if setup.get("active_polygons") != 0 or setup.get("active_circles") != 1:
        raise ValueError(
            "Analytic freefall comparison requires the verified isolated scene"
        )
    rows = [
        json.loads(line)
        for line in (args.matrix_run / "trials.jsonl").read_text().splitlines()
    ]
    expected = (
        len(matrix["levels"])
        * len(matrix["patterns"])
        * (1 + 2 * (len(matrix["factors"]) - 1))
    )
    if len(rows) != expected:
        raise ValueError("Trajectory matrix is incomplete")
    factors = matrix["factors"]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9))
    ff = gravity["freefall"]
    axes[0].loglog(
        [r["factor"] for r in ff],
        [r["absolute_error"] for r in ff],
        "o-",
        color="#0072B2",
        label="Jax2D vs analytic solution",
    )
    expected_errors = [
        0.5 * 9.81 * r["duration_seconds"] * r["physics_dt_seconds"] for r in ff
    ]
    axes[0].loglog(
        [r["factor"] for r in ff],
        expected_errors,
        "--",
        color="#D55E00",
        label="Semi-implicit Euler error",
    )
    axes[0].set_title("Free fall: integration accuracy")
    axes[0].set_ylabel("Absolute position error")
    axes[0].legend(fontsize=8)
    for ax, task, pattern, title in zip(
        axes[1:],
        ["mjc_walker", "mjc_half_cheetah"],
        ["zero", "fixed-motor"],
        ["Walker: zero commands", "Half cheetah: fixed motor commands"],
        strict=True,
    ):
        chosen = [r for r in rows if r["task"] == task and r["pattern"] == pattern]
        trajectories = {
            r["trajectory"]: np.load(args.matrix_run / r["trajectory"])["position"]
            for r in chosen
        }
        common = min(len(x) for x in trajectories.values()) - 1
        base = next(r for r in chosen if r["factor"] == 1)
        reference = trajectories[base["trajectory"]][common]
        for rule, color, marker in [
            ("fixed", "#0072B2", "o"),
            ("scaled_collision", "#D55E00", "s"),
        ]:
            selected = [base] + [
                r for r in chosen if r["factor"] != 1 and r["collision_rule"] == rule
            ]
            deviations = [
                float(
                    np.sqrt(
                        np.mean(
                            (trajectories[r["trajectory"]][common] - reference) ** 2
                        )
                    )
                )
                for r in selected
            ]
            ax.plot(
                [r["factor"] for r in selected],
                deviations,
                marker=marker,
                color=color,
                linestyle="-" if rule == "fixed" else "--",
                label="Fixed collision beta"
                if rule == "fixed"
                else "Collision beta / factor",
            )
        ax.set_xscale("log")
        ax.set_title(f"{title}\nMatched control boundary {common}")
        ax.set_ylabel("Position RMSE vs native (world units)")
        ax.legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(factors, [str(f) for f in factors])
        ax.set_xlabel("Refinement factor (native = 1)")
        ax.grid(alpha=0.22)
    fig.suptitle(
        "Smaller physics steps: numerical accuracy and benchmark fidelity are separate",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        "No model, injected delay or action noise; control period fixed at 1/30 s.\nDeviation from a native trajectory is not error against physical ground truth.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            args.output_dir / f"timestep-study.{suffix}", dpi=220, bbox_inches="tight"
        )
    plt.close(fig)
    print("Saved timestep-study.png/.svg/.pdf")


if __name__ == "__main__":
    main()
