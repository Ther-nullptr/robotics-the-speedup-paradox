#!/usr/bin/env python3
"""Audit a dense latency campaign and summarize only its balanced completed prefix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from robotics_bench.dynamicvla_dom import analysis as base  # noqa: E402


FORMAT = "dynamicvla-dense-study-v1"


def normalized_sources(sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Campaign/source identity must contain actor hashes")
    normalized = {}
    for path, digest in sources.items():
        name = path
        for prefix in ("src/robotics_bench/", "robotics_bench/"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        if name in normalized or not digest:
            raise ValueError("Duplicate or missing normalized actor source identity")
        normalized[name] = digest
    return normalized


def declared_identity(campaign):
    identity = campaign.get("identity")
    required = (
        "resources",
        "tasks",
        "source_sha256",
        "checkpoint_sha256",
        "checkpoint_config_sha256",
        "model_gpu",
        "sim_gpu",
        "rotation",
        "num_steps",
    )
    if not isinstance(identity, dict) or any(key not in identity for key in required):
        raise ValueError("Dense campaign requires a complete declared identity")
    if not identity["resources"] or not identity["tasks"]:
        raise ValueError("Campaign identity requires resources and task hashes")
    if campaign.get("object_speed_scale", 1.0) != identity.get(
        "object_speed_scale", 1.0
    ):
        raise ValueError("Campaign object speed scale differs from declared identity")
    if identity["num_steps"] != campaign["num_steps"]:
        raise ValueError("Campaign num_steps differs from declared identity")
    tasks = {}
    for path, digest in identity["tasks"].items():
        task = Path(path).stem
        if not Path(path).is_absolute() or task in tasks or not digest:
            raise ValueError(
                "Campaign task identity requires unique absolute task paths"
            )
        tasks[task] = (str(Path(path).resolve()), digest)
    if set(tasks) != set(campaign["expected_tasks"]):
        raise ValueError("Campaign declared task identity differs from expected tasks")
    return identity, tasks, normalized_sources(identity["source_sha256"])


def bind_completed_record(row, identity, tasks, sources):
    observed = row["comparison_identity"]
    if observed.get("object_speed_scale", 1.0) != identity.get(
        "object_speed_scale", 1.0
    ):
        raise ValueError(
            "Completed cell object speed differs from declared campaign identity"
        )
    for key in ("checkpoint_sha256", "checkpoint_config_sha256"):
        if observed[key] != identity[key]:
            raise ValueError(
                f"Completed cell {key} differs from declared campaign identity"
            )
    if normalized_sources(observed["source_sha256"]) != sources:
        raise ValueError(
            "Completed cell actor source differs from declared campaign identity"
        )
    for key in ("model_gpu", "sim_gpu", "rotation", "num_steps"):
        if observed["model_options"].get(key) != identity[key]:
            raise ValueError(
                f"Completed cell {key} differs from declared campaign identity"
            )
    task_path, task_hash = tasks[row["task"]]
    if row["task_identity"]["task_config_sha256"] != task_hash:
        raise ValueError(
            "Completed cell task hash differs from declared campaign identity"
        )
    manifest = base.read(Path(row["run_dir"]) / "case-manifest.json")
    resources = manifest.get("resources", {})
    if any(resources.get(key) != value for key, value in identity["resources"].items()):
        raise ValueError(
            "Completed cell resources differ from declared campaign identity"
        )
    if (
        not resources.get("env_cfg")
        or str(Path(resources["env_cfg"]).resolve()) != task_path
    ):
        raise ValueError(
            "Completed cell task path differs from declared campaign identity"
        )


def analyze(campaign_path):
    campaign_path = Path(campaign_path).resolve()
    campaign = base.read(campaign_path)
    if campaign.get("format") != FORMAT or not campaign.get("id"):
        raise ValueError("Invalid dense campaign format/id")
    identity, declared_tasks, declared_sources = declared_identity(campaign)
    tasks = campaign["expected_tasks"]
    delays = campaign["delays_ms"]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Campaign tasks must be nonempty and unique")
    if not delays or len(set(delays)) != len(delays) or 0 not in delays:
        raise ValueError("Campaign delays must be unique and include zero")
    for value in delays:
        base.nonnegative(value, "extra delay")
    num_steps = base.integer(campaign["num_steps"], "num_steps", 1)
    target = base.integer(campaign["episodes_per_task_condition"], "target episodes", 1)
    block_size = base.integer(campaign["block_episodes"], "block episodes", 1)
    first_seed = base.integer(campaign["seed"], "campaign seed")
    blocks = campaign["blocks"]
    if target % block_size or len(blocks) != target // block_size:
        raise ValueError("Block count and size must exactly cover the campaign target")
    expected_keys = {(task, delay) for task in tasks for delay in delays}
    block_ids, study_paths, seeds, run_paths = set(), set(), set(), set()
    records, data_by_block, progress = [], [], []
    for block_index, block in enumerate(blocks):
        block_id = block["id"]
        seed = base.integer(block["seed"], "block seed")
        episodes = base.integer(block["episodes"], "block episodes", 1)
        study_path = (campaign_path.parent / block["study"]).resolve()
        block_seeds = set(range(seed, seed + episodes))
        if block_id in block_ids or study_path in study_paths or seeds & block_seeds:
            raise ValueError("Duplicate block, study, or overlapping seeds")
        if episodes != block_size or seed != first_seed + block_index * block_size:
            raise ValueError("Block seed range/size differs from campaign sequence")
        block_ids.add(block_id)
        study_paths.add(study_path)
        seeds.update(block_seeds)
        if not study_path.exists():
            data_by_block.append(None)
            progress.append(
                {
                    "id": block_id,
                    "seed": seed,
                    "episodes": episodes,
                    "study": str(study_path),
                    "status": "planned",
                    "completed_cells": 0,
                    "expected_cells": len(expected_keys),
                    "completed_episodes": 0,
                }
            )
            continue
        study = base.read(study_path)
        if (
            study.get("format") != base.FORMAT
            or not study.get("id")
            or set(study["expected_tasks"]) != set(tasks)
            or len(study["expected_tasks"]) != len(tasks)
        ):
            raise ValueError(f"{block_id}: invalid study/task identity")
        seen_keys, seen_ids = set(), set()
        block_data = {}
        completed_count = 0
        for cell in study["cells"]:
            key = cell["task"], cell["extra_delay_ms"]
            run = (study_path.parent / cell["run_dir"]).resolve()
            if key in seen_keys or cell["id"] in seen_ids or run in run_paths:
                raise ValueError(
                    "Duplicate task/condition/cell or reused run directory"
                )
            if (
                key not in expected_keys
                or cell["num_steps"] != num_steps
                or cell["episodes"] != episodes
                or cell["seed"] != seed
            ):
                raise ValueError(
                    f"{block_id}: cell differs from exact campaign task/delay/steps/seeds"
                )
            seen_keys.add(key)
            seen_ids.add(cell["id"])
            run_paths.add(run)
            row, data = base.audit_cell(cell, study_path)
            row.update(
                block_id=block_id, cell_id=row["id"], id=f"{block_id}/{row['id']}"
            )
            records.append(row)
            if data is not None:
                block_data[key] = data
                completed_count += 1
        if seen_keys != expected_keys:
            raise ValueError(
                f"{block_id}: study does not declare the complete condition grid"
            )
        full = completed_count == len(expected_keys)
        data_by_block.append(block_data if full else None)
        progress.append(
            {
                "id": block_id,
                "seed": seed,
                "episodes": episodes,
                "study": str(study_path),
                "status": "completed" if full else "partial",
                "completed_cells": completed_count,
                "expected_cells": len(expected_keys),
                "completed_episodes": completed_count * episodes,
            }
        )
    completed_records = [row for row in records if row["status"] == "completed"]
    if (
        len({base.canonical(row["comparison_identity"]) for row in completed_records})
        > 1
    ):
        raise ValueError(
            "Campaign model/checkpoint/source/device identity differs across blocks"
        )
    for row in completed_records:
        bind_completed_record(row, identity, declared_tasks, declared_sources)
    for task in tasks:
        if (
            len(
                {
                    base.canonical(row["task_identity"])
                    for row in completed_records
                    if row["task"] == task
                }
            )
            > 1
        ):
            raise ValueError(
                f"Campaign task/config/physics identity differs for {task}"
            )
    prefix = 0
    for data in data_by_block:
        if data is None:
            break
        prefix += 1
    included_block_ids = {block["id"] for block in blocks[:prefix]}
    for row in records:
        row["included_in_aggregate"] = row["block_id"] in included_block_ids
    for row in progress:
        row["included_in_aggregate"] = row["id"] in included_block_ids
    interim = prefix != len(blocks)
    condition_data = {
        key: base.combine([data_by_block[index][key] for index in range(prefix)])
        if prefix
        else None
        for key in expected_keys
    }
    task_rows = []
    for task in tasks:
        for delay in sorted(delays):
            data = condition_data[task, delay]
            baseline = condition_data[task, 0]
            task_rows.append(
                {
                    "task": task,
                    "extra_delay_ms": delay,
                    "num_steps": num_steps,
                    "status": "completed" if data else "partial",
                    "interim": interim,
                    "included_blocks": prefix,
                    "planned_blocks": len(blocks),
                    "included_episodes": prefix * block_size,
                    "expected_episodes": target,
                    "metrics": base.metrics(data) if data else None,
                    "paired_vs_zero": base.paired(baseline, data) if data else None,
                }
            )
    pooled_data = {
        delay: base.combine([condition_data[task, delay] for task in tasks])
        if prefix
        else None
        for delay in delays
    }
    pooled_rows = [
        {
            "extra_delay_ms": delay,
            "num_steps": num_steps,
            "status": "completed" if prefix else "partial",
            "interim": interim,
            "included_blocks": prefix,
            "planned_blocks": len(blocks),
            "included_episodes": prefix * block_size * len(tasks),
            "expected_episodes": target * len(tasks),
            "completed_tasks": len(tasks) if prefix else 0,
            "expected_tasks": len(tasks),
            "metrics": base.metrics(pooled_data[delay]) if prefix else None,
            "paired_vs_zero": base.paired(pooled_data[0], pooled_data[delay])
            if prefix
            else None,
        }
        for delay in sorted(delays)
    ]
    return {
        "format": "dynamicvla-dense-study-results-v1",
        "id": campaign["id"],
        "object_speed_scale": identity.get("object_speed_scale", 1.0),
        "study": str(campaign_path),
        "status": "partial" if interim else "completed",
        "interim": interim,
        "completed_blocks": sum(row["status"] == "completed" for row in progress),
        "planned_blocks": len(blocks),
        "included_blocks": prefix,
        "included_block_ids": [block["id"] for block in blocks[:prefix]],
        "included_episodes": prefix * block_size * len(expected_keys),
        "expected_episodes": target * len(expected_keys),
        "completed_episodes": sum(row["completed_episodes"] for row in progress),
        "block_progress": progress,
        "cells": records,
        "task_conditions": task_rows,
        "pooled_conditions": pooled_rows,
        "notes": [
            f"{'INTERIM' if interim else 'FINAL'}: balanced prefix includes {prefix}/{len(blocks)} blocks and {prefix * block_size}/{target} episodes per task/condition.",
            "Completed row status describes the included balanced subset, not completion of the full campaign; check interim and expected_episodes.",
            "Partial blocks and completed blocks after the first incomplete block are excluded from all task/pooled aggregates; their raw cell progress remains visible.",
            "All completed cells, including excluded cells, must share model, checkpoint, actor source, device and per-task scene/physics identity.",
            "Disjoint consecutive seed blocks are paired against zero added delay within each task. No significance claims are made.",
            "Added delay is synthetic wall-clock waiting; model compute remains separately measured with fixed refinement steps.",
            "Effective service is release minus worker start, including native pacing, measured host overhead and actual extra delay.",
            "Timing samples include completed generations even if not applied. Action ages include held model controls and exclude initial holds without model provenance.",
            "Expiry fractions use dequeued chunks/actions. Failure-budget means charge unsuccessful episodes the full task budget; successful chunk means condition on success.",
        ],
    }


def save(result, output_dir):
    base.save(result, output_dir)
    path = Path(output_dir) / "report.md"
    label = (
        "INTERIM BALANCED PREFIX" if result["interim"] else "FINAL COMPLETE CAMPAIGN"
    )
    notice = f"**{label}: {result['included_blocks']}/{result['planned_blocks']} blocks; {result['included_episodes']}/{result['expected_episodes']} episodes included.** Fully completed blocks: {result['completed_blocks']}. Partial/non-prefix blocks are excluded from aggregated statistics.\n\n"
    path.write_text(notice + path.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.campaign)
    save(result, args.output_dir)
    print(
        f"Campaign {result['id']}: {result['status']}; balanced blocks {result['included_blocks']}/{result['planned_blocks']}, episodes {result['included_episodes']}/{result['expected_episodes']}"
    )


if __name__ == "__main__":
    main()
