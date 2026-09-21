"""Summarize each quality/latency cell with the shared failure-budget statistics."""

from collections import defaultdict
import csv
import json
from pathlib import Path

from robotics_bench.statistics import summarize_episodes


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["task"],
                row["flow_steps"],
                row["requested_latency_ms"],
                row["delay_mapping"],
            )
        ].append(row)
    cells = []
    for (task, flow, delay, mapping), records in sorted(groups.items()):
        budgets = {r["max_primitive_steps"] for r in records}
        if len(budgets) != 1:
            raise ValueError("Cannot combine different task budgets in one cell")
        overall = summarize_episodes(records, max_steps=budgets.pop())["overall"]
        cells.append(
            {
                "task": task,
                "flow_steps": flow,
                "requested_latency_ms": delay,
                "effective_latency_ms": records[0]["effective_latency_ms"],
                "delay_mapping": mapping,
                "episodes": overall["episodes"],
                "successes": overall["successes"],
                "success_rate": overall["success_rate"],
                "failure_budget_total_control_steps": overall["failure_penalized"][
                    "total_control_steps"
                ],
                "failure_budget_mean_control_steps": overall["failure_penalized"][
                    "mean_control_steps"
                ],
                "success_total_control_steps": overall["successful_episodes"][
                    "total_control_steps"
                ],
                "success_mean_control_steps": overall["successful_episodes"][
                    "mean_control_steps"
                ],
                "mean_return": sum(r["episode_return"] for r in records) / len(records),
                "mean_inference_calls": sum(r["inference_calls"] for r in records)
                / len(records),
            }
        )
    if not cells:
        raise ValueError("No completed episodes to summarize")
    return cells


def write_summary(output, rows, *, complete):
    output = Path(output)
    cells = summarize(rows)
    report = {
        "format": "kinetix-latency-quality-summary-v1",
        "complete": complete,
        "step_unit": "kinetix_control_steps",
        "time_domain": "virtual_native_physics",
        "measured_speedup": None,
        "cells": cells,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    lines = [
        "# Kinetix latency-quality evaluation",
        "",
        f"Status: {'complete' if complete else 'partial'}. Failed episodes use their declared control-step budgets.",
        "",
        "Virtual latency replay; host runtime is not a measured target-hardware speedup.",
        "",
        "| Task | Flow steps | Delay (ms) | Episodes | Success rate | Failure-budget mean steps | Success-only mean steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in cells:
        success_steps = cell["success_mean_control_steps"]
        lines.append(
            f"| {cell['task']} | {cell['flow_steps']} | {cell['requested_latency_ms']:g} | {cell['episodes']} | {cell['success_rate']:.2%} | {cell['failure_budget_mean_control_steps']:.6g} | {'N/A' if success_steps is None else f'{success_steps:.6g}'} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return report
