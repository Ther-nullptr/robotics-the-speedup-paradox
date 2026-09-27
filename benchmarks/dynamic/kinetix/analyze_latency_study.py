"""Freeze calibration predictions and evaluate held-out hardware-delay cells."""

import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latency_study as study_tools  # noqa: E402


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    divisor = 1 + z * z / count
    middle = (p + z * z / (2 * count)) / divisor
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / divisor
    return max(0.0, middle - radius), min(1.0, middle + radius)


def summarize_cell(condition, rows):
    count = len(rows)
    successes = sum(row["success"] for row in rows)
    successful_steps = sum(row["primitive_steps"] for row in rows if row["success"])
    penalized_steps = sum(
        row["primitive_steps"] if row["success"] else row["max_primitive_steps"]
        for row in rows
    )
    low, high = wilson(successes, count)
    return {
        **condition,
        "episodes": count,
        "successes": successes,
        "success_rate": successes / count,
        "ci95_low": low,
        "ci95_high": high,
        "failure_budget_total_control_steps": penalized_steps,
        "failure_budget_mean_control_steps": penalized_steps / count,
        "success_total_control_steps": successful_steps,
        "success_mean_control_steps": successful_steps / successes
        if successes
        else None,
    }


def collect(directory, roles):
    study = json.loads((directory / "study.json").read_text())
    study_tools.verify_inputs(study)
    status = json.loads((directory / "status.json").read_text())
    conditions = [row for row in study["conditions"] if row["role"] in roles]
    source_rows, evidence = {}, {}
    for task, source in study["baseline"].items():
        rows = study_tools.read_jsonl(source["episodes"])
        for n in range(1, 6):
            source_rows[("baseline", f"{task}/N{n}")] = sorted(
                [row for row in rows if row["flow_steps"] == n],
                key=lambda r: r["env_seed"],
            )
        evidence[source["episodes"]] = source["episodes_sha256"]
    needed = {
        row["source"]["id"]
        for row in conditions
        if row["source"]["kind"] == "execution"
    }
    executions = {row["id"]: row for row in study["executions"]}
    keys = {row["weight_key"]: row["id"] for row in study["executions"]}
    jobs = {row["id"]: row for row in study["jobs"]}
    for job_id in sorted({executions[key]["job"] for key in needed}):
        record = status["jobs"].get(job_id, {})
        if record.get("status") != "completed":
            raise ValueError(f"Required job is incomplete: {job_id}")
        output = Path(record["output"])
        verification = study_tools.verify_job(study, jobs[job_id], output)
        if verification != record["verification"]:
            raise ValueError("Job artifacts changed after verification")
        ledger = output / "episodes.jsonl"
        evidence[str(ledger)] = verification["episodes_sha256"]
        for row in study_tools.read_jsonl(ledger):
            plan = study_tools.delay_plan(
                row["requested_latency_ms"],
                physics_dt=1 / 60,
                frame_skip=2,
                execute_horizon=4,
                mapping=row["delay_mapping"],
            )
            key = study_tools.weight_key(row["task"], row["flow_steps"], plan)
            source_rows.setdefault(("execution", keys[key]), []).append(row)
    summaries, rows_by_condition = [], {}
    for condition in conditions:
        source = condition["source"]
        rows = sorted(
            source_rows[(source["kind"], source["id"])], key=lambda r: r["env_seed"]
        )
        study_tools.check_coverage(
            rows,
            [condition["flow_steps"]],
            start_seed=study["start_seed"],
            episodes=study["episodes_per_cell"],
        )
        summaries.append(summarize_cell(condition, rows))
        rows_by_condition[condition["id"]] = rows
    return study, summaries, rows_by_condition, evidence


def fit_task_model(cells):
    """Only quality/calibration cells enter fitting, even if other rows are passed."""
    import numpy as np
    from scipy.optimize import minimize

    quality = sorted(
        (row for row in cells if row["role"] == "quality"),
        key=lambda r: r["flow_steps"],
    )
    if [row["flow_steps"] for row in quality] != list(range(1, 6)):
        raise ValueError("Calibration requires one complete zero-delay N=1..5 grid")
    probabilities = [row["successes"] / row["episodes"] for row in quality]
    anchor = probabilities[-1]
    if anchor <= 0:
        raise ValueError(
            "Delay response is unidentifiable with zero N5 baseline success"
        )
    delay_models = {}
    for mode in ("fine", "coarse"):
        calibration = sorted(
            (
                row
                for row in cells
                if row["role"] == "calibration" and row["mode"] == mode
            ),
            key=lambda r: r["effective_latency_ms"],
        )
        delays = np.array([r["effective_latency_ms"] for r in calibration], dtype=float)
        successes = np.array([r["successes"] for r in calibration], dtype=float)
        trials = np.array([r["episodes"] for r in calibration], dtype=float)
        if len(delays) < 3 or delays[0] != 0 or len(set(delays)) != len(delays):
            raise ValueError("Calibration requires distinct delays including zero")
        rates = successes / trials
        if rates[0] != anchor or trials[0] != quality[-1]["episodes"]:
            raise ValueError("Zero-delay calibration anchor differs from quality data")
        model = {
            "delays_ms": delays.tolist(),
            "observed_probabilities": rates.tolist(),
            "anchor_probability": anchor,
            "largest_upward_change": float(max(0.0, np.diff(rates).max())),
        }
        if mode == "coarse":
            model.update(kind="effective_delay_grid", calibration_rmse=0.0)
        else:

            def nll(predicted):
                predicted = np.clip(predicted, 1e-10, 1 - 1e-10)
                return float(
                    -np.sum(
                        successes * np.log(predicted)
                        + (trials - successes) * np.log1p(-predicted)
                    )
                )

            def objective(log_parameters):
                tau, power = np.exp(log_parameters)
                return nll(anchor / (1 + (delays / tau) ** power))

            candidates = [
                minimize(
                    objective,
                    np.log([30.0, power]),
                    method="L-BFGS-B",
                    bounds=[
                        (math.log(0.01), math.log(10000)),
                        (math.log(0.05), math.log(10)),
                    ],
                )
                for power in (0.5, 1.0, 2.0, 4.0)
            ]
            successful = [
                result
                for result in candidates
                if result.success and math.isfinite(result.fun)
            ]
            if not successful:
                raise RuntimeError("Delay model fitting did not converge")
            fit = min(successful, key=lambda result: result.fun)
            constant_aic, hill_aic = (
                2 * nll(np.full_like(rates, anchor)),
                2 * float(fit.fun) + 4,
            )
            if hill_aic + 2 < constant_aic:
                tau, power = np.exp(fit.x)
                predicted = anchor / (1 + (delays / tau) ** power)
                model.update(kind="hill", tau_ms=float(tau), exponent=float(power))
            else:
                predicted = np.full_like(rates, anchor)
                model.update(kind="constant")
            model.update(
                constant_aic=constant_aic,
                hill_aic=hill_aic,
                calibration_rmse=float(np.sqrt(np.mean((predicted - rates) ** 2))),
            )
        model["requires_review"] = (
            model["calibration_rmse"] > 0.1 or model["largest_upward_change"] > 0.1
        )
        delay_models[mode] = model
    return {
        "quality_steps": list(range(1, 6)),
        "quality_probabilities": probabilities,
        "quality_interpolation": "shape-preserving PCHIP; no monotonicity constraint",
        "delay_models": delay_models,
    }


def predict_success(model, mode, flow_steps, latency_ms):
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    mode = study_tools.normalize_mapping(mode)
    flow = np.asarray(flow_steps, dtype=float)
    latency = np.asarray(latency_ms, dtype=float)
    if np.any((flow < 1) | (flow > 5)) or np.any(latency < 0):
        raise ValueError("Prediction is restricted to the calibrated flow/delay domain")
    delay = model["delay_models"][mode]
    effective = (
        np.rint(latency / (1000 / 30)) * (1000 / 30) if mode == "coarse" else latency
    )
    if np.any(effective > max(delay["delays_ms"]) + 1e-8):
        raise ValueError("Prediction would extrapolate beyond delay calibration")
    quality = PchipInterpolator(model["quality_steps"], model["quality_probabilities"])(
        flow
    )
    if delay["kind"] == "hill":
        factor = 1 / (1 + (effective / delay["tau_ms"]) ** delay["exponent"])
    elif delay["kind"] == "effective_delay_grid":
        factor = (
            np.interp(effective, delay["delays_ms"], delay["observed_probabilities"])
            / delay["anchor_probability"]
        )
    else:
        factor = np.ones_like(effective)
    result = np.clip(quality * factor, 0, 1)
    return float(result) if result.ndim == 0 else result


def validation_metrics(cells):
    heldout = [row for row in cells if not row["calibration_overlap"]]
    best = max(row["success_rate"] for row in cells)
    predicted = max(row["predicted_success"] for row in cells)
    selected = min(
        row["flow_steps"]
        for row in cells
        if abs(row["predicted_success"] - predicted) < 1e-12
    )
    selected_actual = next(
        row["success_rate"] for row in cells if row["flow_steps"] == selected
    )
    return {
        "heldout_cells": len(heldout),
        "calibration_overlap_cells": len(cells) - len(heldout),
        "heldout_rmse": math.sqrt(
            sum(
                (row["success_rate"] - row["predicted_success"]) ** 2 for row in heldout
            )
            / len(heldout)
        )
        if heldout
        else None,
        "observed_best_integers": [
            row["flow_steps"] for row in cells if row["success_rate"] == best
        ],
        "predicted_best_integer": selected,
        "observed_best_success": best,
        "selected_observed_success": selected_actual,
        "empirical_selection_gap": best - selected_actual,
    }


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            for row in rows
        )


def build_plot_data(study, models):
    """Freeze every plotted prediction, including non-integer visualization points."""
    import numpy as np

    flow = np.linspace(1, 5, 801)
    data = {}
    for task, model in models.items():
        data[task] = {
            "flow_steps": flow.tolist(),
            "quality_success": predict_success(model, "fine", flow, 0).tolist(),
            "modes": {},
        }
        for mode in ("fine", "coarse"):
            delays = np.linspace(0, max(model["delay_models"][mode]["delays_ms"]), 801)
            curves = {
                "requested_delays_ms": delays.tolist(),
                "delay_success": predict_success(model, mode, 5, delays).tolist(),
                "hardware_success": {},
            }
            for hardware, values in study["hardware_profiles"].items():
                latency = np.interp(
                    flow, range(1, 6), [values[str(n)] for n in range(1, 6)]
                )
                curves["hardware_success"][hardware] = predict_success(
                    model, mode, flow, latency
                ).tolist()
            data[task]["modes"][mode] = curves
    return data


def frozen_plot_data(frozen, task):
    if task not in frozen.get("plot_data", {}):
        raise ValueError(
            "Frozen plot data is missing; do not recompute predictions after validation"
        )
    return frozen["plot_data"][task]


def freeze(args):
    directory = args.study_dir.expanduser().resolve()
    output = directory / "calibration"
    if output.exists():
        raise ValueError(
            "Calibration output already exists; frozen models are immutable"
        )
    status = json.loads((directory / "status.json").read_text())
    if (
        status["phases"].get("calibration") != "completed"
        or "validation" in status["phases"]
    ):
        raise ValueError("Freeze after calibration and before validation begins")
    study, cells, _, evidence = collect(directory, {"quality", "calibration"})
    models = {
        task: fit_task_model([row for row in cells if row["task"] == task])
        for task in study["tasks"]
    }
    predictions = []
    for condition in study["conditions"]:
        if condition["role"] != "validation":
            continue
        predictions.append(
            {
                "condition_id": condition["id"],
                "predicted_success": predict_success(
                    models[condition["task"]],
                    condition["mode"],
                    condition["flow_steps"],
                    condition["requested_latency_ms"],
                ),
            }
        )
    result = {
        "format": "kinetix-frozen-latency-model-v1",
        "frozen_at": study_tools.utc_now(),
        "study_sha256": study_tools.digest(directory / "study.json"),
        "analysis_source_sha256": study_tools.digest(__file__),
        "fit_runtime": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy")
        },
        "calibration_sources": evidence,
        "task_models": models,
        "plot_data": build_plot_data(study, models),
        "predictions": predictions,
        "prediction_scope": "Condition holdout with shared paired seeds; no claim of seed or scene generalization.",
        "calibration_method": "Empirical zero-delay quality with PCHIP interpolation; binomial Hill/no-effect fine delay; effective-bin lookup for coarse delay.",
    }
    output.mkdir()
    write_csv(output / "cells.csv", cells)
    write_csv(output / "predictions.csv", predictions)
    study_tools.write_json(output / "frozen-model.json", result)
    lines = [
        "# KINETIX delay calibration",
        "",
        "All calibration cells are complete. Hardware-mapped outcomes were not used in fitting.",
        "",
        "| Task | Mode | Delay model | Calibration RMSE | Review flag |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for task, model in models.items():
        for mode, delay in model["delay_models"].items():
            lines.append(
                f"| {task} | {mode} | {delay['kind']} | {delay['calibration_rmse']:.4f} | {delay['requires_review']} |"
            )
    lines += [
        "",
        "The review flag is a descriptive fit diagnostic, not a statistical acceptance test. Zero-delay quality may be non-monotonic. Coarse grid cells are calibration lookups, not independent fit validation.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    print(f"Frozen {len(predictions)} predictions before mapped validation: {output}")


def analyze(args):
    import numpy as np

    directory = args.study_dir.expanduser().resolve()
    status = json.loads((directory / "status.json").read_text())
    if status["phases"].get("validation") != "completed":
        raise ValueError("Complete all validation jobs before final analysis")
    frozen_path = directory / "calibration/frozen-model.json"
    if study_tools.digest(frozen_path) != status["frozen_model_sha256"]:
        raise ValueError("Calibration model changed after validation started")
    frozen = json.loads(frozen_path.read_text())
    if frozen["study_sha256"] != study_tools.digest(directory / "study.json"):
        raise ValueError("Study plan changed after calibration")
    for path, expected in frozen["calibration_sources"].items():
        if study_tools.digest(path) != expected:
            raise ValueError("Calibration data changed after fitting")
    study, cells, rows, evidence = collect(
        directory, {"quality", "calibration", "validation"}
    )
    predictions = {
        row["condition_id"]: row["predicted_success"] for row in frozen["predictions"]
    }
    for cell in cells:
        cell["predicted_success"] = predictions.get(cell["id"])
    metrics = []
    rng = np.random.default_rng(args.bootstrap_seed)
    for task in study["tasks"]:
        # Share bootstrap seed blocks across every hardware/mode cell for this task.
        indices = rng.integers(
            0,
            study["episodes_per_cell"],
            size=(args.bootstrap, study["episodes_per_cell"]),
        )
        for mode in ("fine", "coarse"):
            for hardware in study_tools.HARDWARE:
                group = sorted(
                    (
                        row
                        for row in cells
                        if row["task"] == task
                        and row["mode"] == mode
                        and row["hardware"] == hardware
                        and row["role"] == "validation"
                    ),
                    key=lambda r: r["flow_steps"],
                )
                metric = {
                    "task": task,
                    "mode": mode,
                    "hardware": hardware,
                    **validation_metrics(group),
                }
                outcomes = np.array(
                    [
                        [episode["success"] for episode in rows[cell["id"]]]
                        for cell in group
                    ],
                    dtype=float,
                )
                boot = outcomes[:, indices].mean(axis=2)
                selected = metric["predicted_best_integer"] - 1
                gaps = boot.max(axis=0) - boot[selected]
                interval = np.quantile(gaps, [0.025, 0.975])
                metric.update(
                    selection_gap_ci95_low=float(interval[0]),
                    selection_gap_ci95_high=float(interval[1]),
                )
                metrics.append(metric)
    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    result = {
        "format": "kinetix-latency-analysis-v1",
        "status": "complete",
        "created_at": study_tools.utc_now(),
        "study_sha256": study_tools.digest(directory / "study.json"),
        "frozen_model_sha256": study_tools.digest(frozen_path),
        "analysis_source_sha256": study_tools.digest(__file__),
        "episodes_per_cell": study["episodes_per_cell"],
        "tasks": study["tasks"],
        "unique_new_cells": len(study["executions"]),
        "new_episodes": len(study["executions"]) * study["episodes_per_cell"],
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "uncertainty_scope": "Wilson intervals for observed rates; paired-seed bootstrap selection gaps conditional on frozen predictions. Fit and hardware-profile uncertainty are not included.",
        "metrics": metrics,
        "cells": cells,
        "sources": evidence,
    }
    study_tools.write_json(output / "analysis.json", result)
    write_csv(output / "cells.csv", cells)
    write_csv(output / "validation.csv", metrics)
    write_report(output, study, result)
    if args.plots:
        make_plots(output, study, frozen, cells)
    print(f"Complete analysis: {output / 'report.md'}")


def write_report(output, study, result):
    lines = [
        "# KINETIX latency-quality study",
        "",
        f"Complete: {result['new_episodes']} new episodes; {study['episodes_per_cell']} paired seeds per cell.",
        "",
        study["hardware_interpretation"],
        "",
        result["uncertainty_scope"],
        "",
        "Equivalent conditions explicitly share an execution; references do not add independent samples. Validation cells that reuse calibration are excluded from held-out RMSE.",
        "",
        "| Task | Mode | Hardware | Predicted integer | Observed best integers | Held-out cells | RMSE | Selection gap (pp) |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in result["metrics"]:
        rmse = "n/a" if row["heldout_rmse"] is None else f"{row['heldout_rmse']:.4f}"
        lines.append(
            f"| {row['task']} | {row['mode']} | {row['hardware']} | {row['predicted_best_integer']} | {row['observed_best_integers']} | {row['heldout_cells']} | {rmse} | {100 * row['empirical_selection_gap']:.2f} |"
        )
    lines += [
        "",
        "Observed best integers are exact empirical ties, not a claim of statistical superiority or equivalence. Predicted integers are selected before mapped outcomes are observed; ties choose the lower step count.",
        "",
        "## Observed cells",
        "",
        "| Task | Mode | Hardware / role | N | Requested delay (ms) | Effective delay (ms) | Successes / episodes | Success (%) | Failure-budget mean controls | Success-only mean controls |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in result["cells"]:
        successful = (
            "n/a"
            if row["success_mean_control_steps"] is None
            else f"{row['success_mean_control_steps']:.3f}"
        )
        lines.append(
            f"| {row['task']} | {row['mode']} | {row['hardware']} / {row['role']} | {row['flow_steps']} | {row['requested_latency_ms']:.6f} | {row['effective_latency_ms']:.6f} | {row['successes']}/{row['episodes']} | {100 * row['success_rate']:.2f} | {row['failure_budget_mean_control_steps']:.3f} | {successful} |"
        )
    lines += [
        "",
        "Physics remains dt=1/60 s, frame_skip=2, 30 Hz control, horizon8/execute4 and budget256. Seeds vary policy/action noise on the fixed task initialization. Fine uses time-averaged processed commands within a native physics slot; coarse rounds to native control intervals.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def make_plots(output, study, frozen, cells):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {
        "local_ada": "#0072B2",
        "rtx3090": "#009E73",
        "agx15": "#D55E00",
        "agx30": "#E69F00",
    }
    names = {
        "local_ada": "RTX 6000 Ada",
        "rtx3090": "RTX 3090",
        "agx15": "AGX Orin 15W",
        "agx30": "AGX Orin 30W",
    }

    def observations(axis, x, selected, color, label=None, hollow=False):
        mean = np.array([row["success_rate"] for row in selected])
        low = np.array([row["ci95_low"] for row in selected])
        high = np.array([row["ci95_high"] for row in selected])
        axis.errorbar(
            x,
            mean * 100,
            yerr=np.maximum(0, np.array([mean - low, high - mean]) * 100),
            fmt="o",
            markersize=4,
            color=color,
            mfc="white" if hollow else color,
            capsize=2,
            label=label,
            zorder=4,
        )

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "pdf.fonttype": 42,
        }
    ):
        for task in study["tasks"]:
            model = frozen["task_models"][task]
            plot_data = frozen_plot_data(frozen, task)
            fine_grid = plot_data["flow_steps"]
            quality = sorted(
                (
                    row
                    for row in cells
                    if row["task"] == task and row["role"] == "quality"
                ),
                key=lambda r: r["flow_steps"],
            )
            for mode in ("fine", "coarse"):
                fig, panels = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
                latency_ax, quality_ax, delay_ax, mapped_ax = panels.flat
                for hardware, values in study["hardware_profiles"].items():
                    x = sorted(int(n) for n in values)
                    y = [values[str(n)] for n in x]
                    latency_ax.plot(
                        x,
                        y,
                        "o-",
                        color=colors[hardware],
                        label=names[hardware],
                        markersize=4,
                    )
                latency_ax.set(
                    title="(1a) Historical hardware latency",
                    xlabel="Flow steps N",
                    ylabel="Complete policy latency (ms)",
                    xticks=range(1, 6),
                )
                latency_ax.legend(fontsize=8)
                observations(
                    quality_ax,
                    range(1, 6),
                    quality,
                    "#333333",
                    "Observed, 95% Wilson CI",
                )
                quality_ax.plot(
                    fine_grid,
                    100 * np.asarray(plot_data["quality_success"]),
                    color="#333333",
                    label="Quality interpolation",
                )
                quality_ax.set(
                    title="(1b) Zero-delay quality",
                    xlabel="Flow steps N",
                    ylabel="Success (%)",
                    xticks=range(1, 6),
                    ylim=(-2, 103),
                )
                quality_ax.legend(fontsize=8)
                calibration = sorted(
                    (
                        row
                        for row in cells
                        if row["task"] == task
                        and row["role"] == "calibration"
                        and row["mode"] == mode
                    ),
                    key=lambda r: r["requested_latency_ms"],
                )
                observations(
                    delay_ax,
                    [row["requested_latency_ms"] for row in calibration],
                    calibration,
                    "#333333",
                    "Calibration observations",
                )
                curves = plot_data["modes"][mode]
                delays = curves["requested_delays_ms"]
                delay_ax.plot(
                    delays,
                    100 * np.asarray(curves["delay_success"]),
                    color="#882255",
                    label=model["delay_models"][mode]["kind"],
                )
                delay_ax.set(
                    title=f"(1c) Delay response at N=5 ({mode})",
                    xlabel="Requested injected delay (ms)",
                    ylabel="Success (%)",
                    ylim=(-2, 103),
                )
                delay_ax.legend(fontsize=8)
                for hardware, values in study["hardware_profiles"].items():
                    mapped_ax.plot(
                        fine_grid,
                        100 * np.asarray(curves["hardware_success"][hardware]),
                        color=colors[hardware],
                        label=names[hardware],
                    )
                    observed = sorted(
                        (
                            row
                            for row in cells
                            if row["task"] == task
                            and row["mode"] == mode
                            and row["hardware"] == hardware
                            and row["role"] == "validation"
                        ),
                        key=lambda r: r["flow_steps"],
                    )
                    for overlap in (False, True):
                        subset = [
                            row
                            for row in observed
                            if row["calibration_overlap"] == overlap
                        ]
                        if subset:
                            observations(
                                mapped_ax,
                                [row["flow_steps"] for row in subset],
                                subset,
                                colors[hardware],
                                hollow=overlap,
                            )
                mapped_ax.set(
                    title="(2) Frozen predictions and observed outcomes",
                    xlabel="Flow steps N",
                    ylabel="Success (%)",
                    xticks=range(1, 6),
                    ylim=(-2, 103),
                )
                mapped_ax.legend(fontsize=8, loc="best")
                fig.suptitle(
                    f"{task} | {mode} delay replay | {study['episodes_per_cell']} paired seeds per cell\nNative physics: 60 Hz; control: 30 Hz; original JAX FP32 policy",
                    fontsize=14,
                )
                fig.supxlabel(
                    "Curves: calibrated predictions. Filled points: held-out conditions. Hollow points: reused calibration.\nHistorical Torch FP16 latencies define replay scenarios; fit/profile uncertainty is not included in the observation error bars.",
                    fontsize=9,
                )
                stem = output / f"{task}-{mode}-four-panel"
                fig.savefig(stem.with_suffix(".png"), dpi=180)
                fig.savefig(stem.with_suffix(".pdf"))
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser(
        "freeze", help="Fit calibration and freeze predictions before validation"
    )
    calibrate.add_argument("--study-dir", type=Path, required=True)
    calibrate.set_defaults(handler=freeze)
    report = commands.add_parser(
        "report", help="Verify complete coverage and analyze frozen predictions"
    )
    report.add_argument("--study-dir", type=Path, required=True)
    report.add_argument("--bootstrap", type=int, default=2000)
    report.add_argument("--bootstrap-seed", type=int, default=20260927)
    report.add_argument("--plots", action="store_true")
    report.set_defaults(handler=analyze)
    args = parser.parse_args()
    if getattr(args, "bootstrap", 1) < 1:
        parser.error("bootstrap must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
