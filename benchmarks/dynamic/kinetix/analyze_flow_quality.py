"""Validate paired native Kinetix trials and plot flow-iteration quality curves."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from robotics_bench.kinetix.protocol import normalize_mapping  # noqa: E402
from robotics_bench.kinetix.results import summarize  # noqa: E402


def wilson(successes, count):
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def analyze(
    rows,
    *,
    levels,
    flows,
    seeds,
    bootstrap_samples=5000,
    bootstrap_seed=20260922,
    allow_partial=False,
    run_completed=True,
):
    import numpy as np

    if (
        not levels
        or not flows
        or not seeds
        or any(len(x) != len(set(x)) for x in (levels, flows, seeds))
    ):
        raise ValueError("Matrix dimensions must be nonempty and unique")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rows = [
        {**row, "delay_mapping": normalize_mapping(row["delay_mapping"])}
        for row in rows
    ]
    expected = {
        (task, flow, seed) for task in levels for flow in flows for seed in seeds
    }
    observed, initial = {}, {}
    for row in rows:
        key = (row["task"], row["flow_steps"], row["env_seed"])
        if key in observed:
            raise ValueError(f"Duplicate episode key: {key}")
        if key not in expected:
            raise ValueError(f"Unexpected episode key: {key}")
        if (
            row["requested_latency_ms"] != 0
            or row["effective_latency_ms"] != 0
            or row["delay_mapping"] != "fine"
        ):
            raise ValueError("Quality-only analysis requires zero-delay fine trials")
        if (
            type(row["success"]) is not bool
            or row["max_primitive_steps"] != 256
            or not 1 <= row["primitive_steps"] <= 256
        ):
            raise ValueError("Invalid outcome or native episode budget")
        init_key = (row["task"], row["env_seed"])
        digest = row["initial_observation_sha256"]
        if initial.setdefault(init_key, digest) != digest:
            raise ValueError(
                f"Paired cells have different initial observations: {init_key}"
            )
        observed[key] = row
    complete = set(observed) == expected and run_completed
    if not complete and not allow_partial:
        raise ValueError(f"Incomplete matrix: {len(observed)}/{len(expected)} episodes")
    cells = summarize(rows) if rows else []
    for cell in cells:
        cell["ci95_low"], cell["ci95_high"] = wilson(
            cell["successes"], cell["episodes"]
        )
    report = {
        "format": "kinetix-flow-quality-analysis-v1",
        "complete": complete,
        "observed_episodes": len(observed),
        "expected_episodes": len(expected),
        "levels": levels,
        "flow_steps": flows,
        "reference_flow_steps": max(flows),
        "seeds": seeds,
        "cells": cells,
        "macro": [],
        "paired": [],
        "cell_interval": "95% Wilson",
        "macro_interval": "95% shared-seed-block bootstrap over fixed tasks",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    # Partial cell rates remain inspectable; macro and paired inference require
    # the entire prespecified task/step/seed matrix, not whichever jobs finish first.
    if not complete:
        return report
    flows = sorted(flows)
    reference_index = flows.index(max(flows))
    rng = np.random.default_rng(bootstrap_seed)
    macro_bootstrap = np.zeros((bootstrap_samples, len(flows)))
    # The runner reuses seed-derived noise streams across tasks as well as N.
    # Resample entire seed blocks to retain both forms of dependence.
    indices = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    success = np.array(
        [
            [
                [observed[(task, flow, seed)]["success"] for flow in flows]
                for seed in seeds
            ]
            for task in levels
        ],
        dtype=float,
    )
    for task_index, task in enumerate(levels):
        task_bootstrap = success[task_index][indices].mean(axis=1)
        macro_bootstrap += task_bootstrap / len(levels)
        for index, flow in enumerate(flows):
            if index == reference_index:
                continue
            delta = (
                success[task_index, :, index] - success[task_index, :, reference_index]
            )
            wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
            discordant = wins + losses
            p_value = (
                min(
                    1.0,
                    2
                    * sum(
                        math.comb(discordant, k) for k in range(min(wins, losses) + 1)
                    )
                    / (2**discordant),
                )
                if discordant
                else 1.0
            )
            limits = np.quantile(
                task_bootstrap[:, index] - task_bootstrap[:, reference_index],
                [0.025, 0.975],
            )
            report["paired"].append(
                {
                    "task": task,
                    "flow_steps": flow,
                    "reference_flow_steps": max(flows),
                    "paired_episodes": len(seeds),
                    "delta_success_rate": float(delta.mean()),
                    "ci95_low": float(limits[0]),
                    "ci95_high": float(limits[1]),
                    "wins": wins,
                    "losses": losses,
                    "exact_mcnemar_p": p_value,
                }
            )
    previous = 0.0
    ordered = sorted(report["paired"], key=lambda row: row["exact_mcnemar_p"])
    for index, row in enumerate(ordered):
        previous = min(
            1.0, max(previous, row["exact_mcnemar_p"] * (len(ordered) - index))
        )
        row["holm_adjusted_p"] = previous
    means = success.mean(axis=(0, 1))
    for index, flow in enumerate(flows):
        limits = np.quantile(macro_bootstrap[:, index], [0.025, 0.975])
        delta = means[index] - means[reference_index]
        delta_limits = np.quantile(
            macro_bootstrap[:, index] - macro_bootstrap[:, reference_index],
            [0.025, 0.975],
        )
        selected = [r for r in rows if r["flow_steps"] == flow]
        successes = [r for r in selected if r["success"]]
        report["macro"].append(
            {
                "flow_steps": flow,
                "tasks": len(levels),
                "episodes": len(selected),
                "successes": len(successes),
                "success_rate": float(means[index]),
                "ci95_low": float(limits[0]),
                "ci95_high": float(limits[1]),
                "paired_delta_vs_reference": float(delta),
                "paired_delta_ci95_low": float(delta_limits[0]),
                "paired_delta_ci95_high": float(delta_limits[1]),
                "failure_budget_mean_control_steps": sum(
                    r["primitive_steps"] if r["success"] else r["max_primitive_steps"]
                    for r in selected
                )
                / len(selected),
                "success_mean_control_steps": sum(
                    r["primitive_steps"] for r in successes
                )
                / len(successes)
                if successes
                else None,
            }
        )
    return report


def load_sweep(root, allow_partial):
    plan = json.loads((root / "sweep-manifest.json").read_text())
    if plan["status"] != "completed" and not allow_partial:
        raise ValueError(
            "Sweep is not completed; use --allow-partial for progress only"
        )
    rows, manifests, source_identity = [], {}, None
    for task in plan["levels"]:
        path = root / task / "case-manifest.json"
        if not path.exists():
            if allow_partial:
                continue
            raise ValueError(f"Missing task manifest: {task}")
        manifest = json.loads(path.read_text())
        if manifest["status"] != "completed" and not allow_partial:
            raise ValueError(f"Task is incomplete: {task}")
        params = manifest["tasks"][task]
        env, static = params["native_env_params"], params["native_static_env_params"]
        if (
            env["dt"] != 1 / 60
            or static["frame_skip"] != 2
            or env["max_timesteps"] != 256
            or env["baumgarte_coefficient_collision"] != 0.2
            or static["num_solver_iterations"] != 10
        ):
            raise ValueError("Task changed the native physics protocol")
        if (
            manifest["execute_horizon"] != 4
            or manifest["action_noise_std"] != 0.1
            or normalize_mapping(manifest["mapping"]) != "fine"
        ):
            raise ValueError(
                "Incompatible execution horizon, action noise or delay mapping"
            )
        runtime = manifest.get("runtime", {})
        packages = runtime.get("packages", {})
        policy = params.get("runtime", {}).get("policy", {})
        if (
            not runtime.get("python")
            or not all(
                packages.get(name) for name in ("jax", "jaxlib", "flax", "numpy")
            )
            or not policy.get("parameter_dtypes")
            or not manifest.get("source_sha256")
            or not manifest.get("entry_code_sha256")
        ):
            # The manifest exists before GPU imports and model loading finish.
            if allow_partial and manifest["status"] != "completed":
                continue
            raise ValueError(f"Missing evaluator/runtime evidence: {task}")
        identity = (
            manifest["source_sha256"],
            manifest["entry_code_sha256"],
            runtime["python"],
            packages,
        )
        if source_identity is None:
            source_identity = identity
        elif identity != source_identity:
            raise ValueError(
                "Tasks used different evaluator or runtime source versions"
            )
        if policy["parameter_dtypes"] != ["float32"]:
            raise ValueError("This study requires the fixed FP32 policy")
        manifests[task] = {
            "path": str(path),
            "status": manifest["status"],
            "gpu": manifest["gpu"],
            "repository_commit": manifest["repository_commit"],
            "runtime": manifest.get("runtime", {}),
        }
        ledger = root / task / "episodes.jsonl"
        if ledger.exists():
            lines = ledger.read_text().splitlines(keepends=True)
            if allow_partial and lines and not lines[-1].endswith("\n"):
                lines = lines[:-1]
            rows.extend(json.loads(line) for line in lines if line.strip())
    return plan, rows, manifests


def write_csv(path, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(
        math.ceil(len(report["levels"]) / 4),
        4,
        figsize=(14, 3 * math.ceil(len(report["levels"]) / 4)),
        squeeze=False,
    )
    for ax, task in zip(axes.flat, report["levels"]):
        cells = sorted(
            [r for r in report["cells"] if r["task"] == task],
            key=lambda r: r["flow_steps"],
        )
        if cells:
            x = [r["flow_steps"] for r in cells]
            y = np.array([r["success_rate"] for r in cells])
            ax.errorbar(
                x,
                y,
                yerr=np.array(
                    [
                        y - [r["ci95_low"] for r in cells],
                        [r["ci95_high"] for r in cells] - y,
                    ]
                ).clip(0),
                fmt="o-",
                capsize=3,
                color="#276e9c",
            )
        ax.set(
            title=task,
            xlabel="Flow iterations N",
            ylabel="Task success rate",
            ylim=(-0.03, 1.03),
            xticks=report["flow_steps"],
        )
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(report["levels"]) :]:
        ax.set_visible(False)
    figure.suptitle(
        "Native KINETIX: flow iterations vs task success\nZero injected delay; 95% Wilson intervals"
        + (" [PARTIAL]" if not report["complete"] else "")
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"task_success_vs_flow_steps.{suffix}", dpi=180)
    plt.close(figure)
    if report["macro"]:
        figure, ax = plt.subplots(figsize=(7, 4.5))
        rows = report["macro"]
        y = np.array([r["success_rate"] for r in rows])
        ax.errorbar(
            [r["flow_steps"] for r in rows],
            y,
            yerr=np.array(
                [y - [r["ci95_low"] for r in rows], [r["ci95_high"] for r in rows] - y]
            ).clip(0),
            fmt="o-",
            capsize=4,
            color="#276e9c",
        )
        ax.set(
            title=f"Native KINETIX: equal-weight mean over {len(report['levels'])} tasks",
            xlabel="Flow iterations N",
            ylabel="Mean task success rate",
            xticks=report["flow_steps"],
            ylim=(0, 1),
        )
        ax.grid(alpha=0.2)
        figure.text(
            0.5,
            0.01,
            "Zero delay; 95% shared-seed-block bootstrap over fixed tasks",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.035, 1, 1))
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"macro_success_vs_flow_steps.{suffix}", dpi=180)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    plan, rows, manifests = load_sweep(args.input, args.allow_partial)
    report = analyze(
        rows,
        levels=plan["levels"],
        flows=plan["flow_steps"],
        seeds=list(
            range(plan["start_seed"], plan["start_seed"] + plan["episodes_per_cell"])
        ),
        bootstrap_samples=args.bootstrap_samples,
        allow_partial=args.allow_partial,
        run_completed=plan["status"] == "completed"
        and len(manifests) == len(plan["levels"])
        and all(m["status"] == "completed" for m in manifests.values()),
    )
    report["task_manifests"] = manifests
    report["sweep_host_seconds"] = plan.get("host_seconds")
    output = args.output_dir or args.input / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    if not report["complete"]:
        for suffix in ("png", "pdf"):
            (output / f"macro_success_vs_flow_steps.{suffix}").unlink(missing_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for filename, key in (
        ("cell_metrics.csv", "cells"),
        ("macro_metrics.csv", "macro"),
        ("paired_vs_reference.csv", "paired"),
    ):
        write_csv(output / filename, report[key])
    lines = [
        "# Native KINETIX flow-step quality",
        "",
        f"Status: {'complete' if report['complete'] else 'partial'}; {report['observed_episodes']}/{report['expected_episodes']} episodes.",
        "",
        "Native physics, 30 Hz control, execute horizon 4, action noise 0.1, zero injected delay, FP32 RTC checkpoints.",
        "",
        "| Task | N | Successes / episodes | Success rate | 95% Wilson interval |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["cells"]:
        lines.append(
            f"| {row['task']} | {row['flow_steps']} | {row['successes']}/{row['episodes']} | {row['success_rate']:.2%} | [{row['ci95_low']:.2%}, {row['ci95_high']:.2%}] |"
        )
    if report["macro"]:
        lines += [
            "",
            "Equal-weight task means; intervals jointly resample shared seed blocks across all fixed tasks and N values.",
            "",
            "| N | Success rate | 95% interval | Paired delta vs highest N | Failure-budget mean steps |",
            "| ---: | ---: | --- | ---: | ---: |",
        ]
        for row in report["macro"]:
            lines.append(
                f"| {row['flow_steps']} | {row['success_rate']:.2%} | [{row['ci95_low']:.2%}, {row['ci95_high']:.2%}] | {row['paired_delta_vs_reference']:+.2%} | {row['failure_budget_mean_control_steps']:.3f} |"
            )
        lines += [
            "",
            "Paired task comparisons use exact McNemar tests with Holm correction over all task/step comparisons. Quality is not constrained to be monotonic. This is one checkpoint family; no hardware-latency trade-off is inferred from zero-delay quality alone.",
        ]
    else:
        lines += [
            "",
            "Macro and paired comparisons are withheld until all prespecified cells are complete.",
        ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    if not args.no_plots and report["cells"]:
        plot(report, output)
    print(
        f"{'Complete' if report['complete'] else 'Partial'}: {report['observed_episodes']}/{report['expected_episodes']} episodes. Report: {output / 'report.md'}"
    )


if __name__ == "__main__":
    main()
