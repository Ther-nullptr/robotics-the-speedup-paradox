"""CPU-only audit and summary of native streaming synthetic latency studies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


FORMAT = "dynamicvla-streaming-study-v1"
TIMINGS = (
    "model_compute_ms",
    "worker_compute_ms",
    "native_pacing_ms",
    "extra_delay_actual_ms",
    "post_compute_overhead_ms",
    "effective_service_ms",
    "action_age_sim_ms",
    "action_age_wall_ms",
)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nonnegative(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid nonnegative {label}: {value!r}")
    return value


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid integer {label}: {value!r}")
    return value


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def wilson(successes, count):
    if not count:
        return None, None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    )
    return max(0, center - radius), min(1, center + radius)


def audit_cell(cell, study_path):
    run = (study_path.parent / cell["run_dir"]).resolve()
    result = {
        key: cell[key]
        for key in ("id", "task", "extra_delay_ms", "num_steps", "episodes", "seed")
    }
    result.update(run_dir=str(run), status="planned", metrics=None)
    manifest_path = run / "case-manifest.json"
    if not manifest_path.exists():
        return result, None
    manifest = read(manifest_path)
    result["status"] = manifest.get("status", "unknown")
    if result["status"] != "completed":
        result["error"] = manifest.get("error")
        return result, None
    options = manifest["options"]
    for key in ("extra_delay_ms", "num_steps", "episodes", "seed"):
        if options.get(key) != cell[key]:
            raise ValueError(f"{cell['id']}: manifest {key} differs from study")
    if (
        options.get("mode") != "native-streaming"
        or options.get("measure_inference") is not True
        or options.get("episode_seed_mode") is not True
    ):
        raise ValueError(
            f"{cell['id']}: native streaming measured inference and episode seeding are required"
        )
    if manifest.get("case_id") != "dynamicvla_dom":
        raise ValueError("Unexpected case identity")
    identity = manifest["identity"]
    for key in ("checkpoint_config_sha256", "task_config_sha256"):
        if not identity.get(key):
            raise ValueError(f"Missing final {key}")
    if not manifest.get("checkpoint_sha256") or not manifest.get("source_sha256"):
        raise ValueError("Missing final checkpoint/source identity")
    effective = read(run / "effective-config.json")
    coverage = read(run / "coverage.json")
    episodes = lines(run / "episodes.jsonl")
    count = cell["episodes"]
    if (
        coverage.get("status") != "passed"
        or coverage.get("expected_episodes") != count
        or coverage.get("completed_episodes") != count
    ):
        raise ValueError(f"{cell['id']}: coverage is incomplete")
    if len(episodes) != count or {row["episode_id"] for row in episodes} != set(
        range(count)
    ):
        raise ValueError(f"{cell['id']}: duplicate or missing episodes")
    data = {
        "episodes": [],
        "samples": {key: [] for key in TIMINGS},
        "received_chunks": 0,
        "fully_expired_chunks": 0,
        "expired_actions": 0,
        "dequeued_generated_actions": 0,
        "generated_chunks": 0,
        "generated_actions": 0,
        "enqueued_chunks": 0,
        "policy_starved_episodes": 0,
        "held_control_steps": 0,
        "control_steps": 0,
    }
    generated, dequeued = {}, set()
    requests = lines(run / "requests.jsonl")
    for event in requests:
        if event.get("kind") != "chunk_generated":
            continue
        key = event["episode_id"], event["chunk_id"]
        if key in generated or key[0] not in range(count):
            raise ValueError("Duplicate generation or unknown episode")
        integer(key[1], "chunk_id")
        integer(event["generated_actions"], "generated_actions", 1)
        if type(event.get("result_enqueued")) is not bool:
            raise ValueError("Generation lacks result_enqueued evidence")
        for name in TIMINGS[:4]:
            data["samples"][name].append(nonnegative(event[name], name))
        started = nonnegative(event["worker_started_wall_s"], "worker start time")
        ready = nonnegative(event["compute_ready_wall_s"], "compute ready time")
        delay_started = nonnegative(event["delay_started_wall_s"], "delay start time")
        released = nonnegative(event["released_wall_s"], "release time")
        if not started <= ready <= delay_started <= released:
            raise ValueError("Worker timing boundaries are not ordered")
        requested = nonnegative(
            event["extra_delay_requested_ms"], "requested extra delay"
        )
        if requested != cell["extra_delay_ms"]:
            raise ValueError("Generation requested extra delay differs from study")
        # Each interval has explicit endpoints: host preemption between compute
        # completion and delay start is measured overhead, never hidden tolerance.
        delay_ms = (released - delay_started) * 1000
        overhead_ms = (delay_started - ready) * 1000
        checks = [
            (delay_ms, event["extra_delay_actual_ms"], "extra delay"),
            (
                (ready - started) * 1000,
                event["worker_compute_ms"] + event["native_pacing_ms"],
                "worker compute/pacing",
            ),
        ]
        if "post_compute_overhead_ms" in event:
            checks.append(
                (
                    overhead_ms,
                    nonnegative(
                        event["post_compute_overhead_ms"], "post-compute overhead"
                    ),
                    "post-compute overhead",
                )
            )
        for measured, recorded, label in checks:
            if not math.isclose(measured, recorded, abs_tol=1e-6, rel_tol=1e-9):
                raise ValueError(f"Recorded {label} disagrees with timing boundaries")
        data["samples"]["post_compute_overhead_ms"].append(overhead_ms)
        data["samples"]["effective_service_ms"].append((released - started) * 1000)
        generated[key] = event
        data["generated_actions"] += event["generated_actions"]
        data["enqueued_chunks"] += int(event["result_enqueued"])
    data["generated_chunks"] = len(generated)
    for event in requests:
        kind = event.get("kind")
        if kind not in ("chunk_received", "chunk_expired"):
            continue
        key = event["episode_id"], event["chunk_id"]
        if (
            key not in generated
            or key in dequeued
            or not generated[key]["result_enqueued"]
        ):
            raise ValueError("Dequeued chunk is duplicate, unknown, or not enqueued")
        dequeued.add(key)
        expired = integer(event["expired_actions"], "expired_actions")
        length = generated[key]["generated_actions"]
        if expired > length or (kind == "chunk_expired" and expired != length):
            raise ValueError("Expired action count disagrees with generated chunk")
        data["received_chunks"] += int(kind == "chunk_received")
        data["fully_expired_chunks"] += int(kind == "chunk_expired")
        data["expired_actions"] += expired
        data["dequeued_generated_actions"] += length
    for episode in episodes:
        episode_id = integer(episode["episode_id"], "episode_id")
        if (
            episode["task"] != cell["task"]
            or episode["env_seed"] != cell["seed"] + episode_id
        ):
            raise ValueError("Episode task/seed differs from declared cell")
        if type(episode["success"]) is not bool:
            raise ValueError("Episode success must be boolean")
        if not any(key[0] == episode_id for key in generated):
            raise ValueError(
                "Completed episode has no completed model generation evidence"
            )
        starved = (
            episode.get("policy_starved", False)
            or episode.get("termination_reason") == "policy_starved"
        )
        if starved and episode["success"]:
            raise ValueError("A policy-starved episode cannot be successful")
        data["policy_starved_episodes"] += int(starved)
        steps = integer(episode["primitive_steps"], "primitive_steps", 1)
        budget = integer(episode["max_primitive_steps"], "max_primitive_steps", 1)
        if steps > budget:
            raise ValueError("Episode exceeds declared control budget")
        controls = lines(run / f"episode-{episode_id:06d}-controls.jsonl")
        if len(controls) != steps or [row["control_index"] for row in controls] != list(
            range(steps)
        ):
            raise ValueError("Control ledger coverage differs from actual steps")
        applied, held = set(), 0
        for control in controls:
            if type(control["held"]) is not bool:
                raise ValueError("Control held flag must be boolean")
            held += int(control["held"])
            chunk = control.get("chunk_id")
            if chunk is None:
                if not control["held"]:
                    raise ValueError("Applied model action has no chunk provenance")
                if any(control.get(name) is not None for name in TIMINGS[-2:]):
                    raise ValueError(
                        "Initial held target cannot have model observation age"
                    )
                continue
            if (episode_id, chunk) not in generated:
                raise ValueError("Applied control refers to unknown chunk")
            generated_event = generated[episode_id, chunk]
            for source_key in (
                "chunk_observation_index",
                "chunk_observation_sim_time_s",
                "chunk_observation_wall_s",
            ):
                if (
                    source_key in generated_event
                    and control.get(source_key) != generated_event[source_key]
                ):
                    raise ValueError(
                        "Applied control observation provenance differs from generated chunk"
                    )
            applied.add(chunk)
            for name, clock, source in (
                ("action_age_sim_ms", "sim_time_s", "chunk_observation_sim_time_s"),
                ("action_age_wall_ms", "wall_time_s", "chunk_observation_wall_s"),
            ):
                age = nonnegative(control[name], name)
                expected_age = (control[clock] - control[source]) * 1000
                if not math.isclose(age, expected_age, abs_tol=0.01, rel_tol=1e-6):
                    raise ValueError(f"Inconsistent {name}")
                data["samples"][name].append(age)
        if (
            sorted(applied) != sorted(episode["applied_chunk_ids"])
            or len(applied) != episode["applied_chunks"]
        ):
            raise ValueError("Applied chunk summary disagrees with actual controls")
        if (
            held != episode["held_control_steps"]
            or steps - held != episode["actions_accepted"]
        ):
            raise ValueError("Held/accepted control counts disagree")
        data["held_control_steps"] += held
        data["control_steps"] += steps
        data["episodes"].append(
            {
                "task": cell["task"],
                "seed": episode["env_seed"],
                "success": episode["success"],
                "steps": steps,
                "budget": budget,
                "applied_chunks": len(applied),
                "generated_chunks": sum(key[0] == episode_id for key in generated),
            }
        )
    if len({row["budget"] for row in data["episodes"]}) != 1:
        raise ValueError("One task has inconsistent episode budgets")
    if coverage.get("successes") != sum(row["success"] for row in episodes):
        raise ValueError("Coverage success count differs from raw episodes")
    result["artifact_sha256"] = {
        name: sha(run / name)
        for name in (
            "case-manifest.json",
            "coverage.json",
            "episodes.jsonl",
            "requests.jsonl",
            "effective-config.json",
        )
    }
    result["control_ledger_sha256"] = {
        f"episode-{episode_id:06d}-controls.jsonl": sha(
            run / f"episode-{episode_id:06d}-controls.jsonl"
        )
        for episode_id in range(count)
    }
    result["comparison_identity"] = {
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_config_sha256": identity["checkpoint_config_sha256"],
        "source_sha256": manifest["source_sha256"],
        "model_options": {
            key: options.get(key)
            for key in (
                "num_steps",
                "rotation",
                "n_action_steps",
                "action_horizon",
                "n_obs_steps",
                "model_gpu",
                "sim_gpu",
            )
        },
    }
    result["task_identity"] = {
        "task_config_sha256": identity["task_config_sha256"],
        "simulation": {
            key: effective.get(key)
            for key in (
                "physics_dt_s",
                "control_dt_s",
                "decimation",
                "camera_update_periods_s",
                "episode_length_s",
                "control_protocol",
            )
        },
        "budget": data["episodes"][0]["budget"],
    }
    result["metrics"] = metrics(data)
    return result, data


def metrics(data):
    episodes = data["episodes"]
    successful = [row for row in episodes if row["success"]]
    n = len(episodes)
    low, high = wilson(len(successful), n)
    result = {
        key: data[key]
        for key in (
            "policy_starved_episodes",
            "received_chunks",
            "fully_expired_chunks",
            "expired_actions",
            "dequeued_generated_actions",
            "generated_chunks",
            "generated_actions",
            "enqueued_chunks",
            "held_control_steps",
            "control_steps",
        )
    }
    dequeued = data["received_chunks"] + data["fully_expired_chunks"]
    result.update(
        episodes=n,
        successes=len(successful),
        success_rate=len(successful) / n,
        success_rate_wilson95_low=low,
        success_rate_wilson95_high=high,
        mean_generated_chunks_success=mean(
            row["generated_chunks"] for row in successful
        )
        if successful
        else None,
        mean_applied_chunks_success=mean(row["applied_chunks"] for row in successful)
        if successful
        else None,
        mean_steps_success=mean(row["steps"] for row in successful)
        if successful
        else None,
        mean_steps_failure_budget=mean(
            row["steps"] if row["success"] else row["budget"] for row in episodes
        ),
        held_control_fraction=data["held_control_steps"] / data["control_steps"],
        dequeued_chunks=dequeued,
        fully_expired_chunk_fraction_dequeued=data["fully_expired_chunks"] / dequeued
        if dequeued
        else None,
        expired_action_fraction_dequeued=data["expired_actions"]
        / data["dequeued_generated_actions"]
        if data["dequeued_generated_actions"]
        else None,
        received_chunk_fraction_generated=data["received_chunks"]
        / data["generated_chunks"]
        if data["generated_chunks"]
        else None,
    )
    for name, values in data["samples"].items():
        result[f"{name}_samples"] = len(values)
        result[f"{name}_median"] = percentile(values, 0.5)
        result[f"{name}_p95"] = percentile(values, 0.95)
    return result


def combine(data):
    combined = {
        key: sum(item[key] for item in data)
        for key in data[0]
        if key not in ("episodes", "samples")
    }
    combined["episodes"] = [row for item in data for row in item["episodes"]]
    combined["samples"] = {
        key: [value for item in data for value in item["samples"][key]]
        for key in TIMINGS
    }
    return combined


def paired(baseline, candidate):
    left = {(row["task"], row["seed"]): row["success"] for row in baseline["episodes"]}
    right = {
        (row["task"], row["seed"]): row["success"] for row in candidate["episodes"]
    }
    if set(left) != set(right):
        return {
            "status": "unmatched_seeds",
            "matched_episodes": len(set(left) & set(right)),
        }
    return {
        "status": "paired",
        "episodes": len(left),
        "gains": sum(not left[key] and right[key] for key in left),
        "losses": sum(left[key] and not right[key] for key in left),
        "both_success": sum(left[key] and right[key] for key in left),
        "both_failure": sum(not left[key] and not right[key] for key in left),
        "success_rate_change": (sum(right.values()) - sum(left.values())) / len(left),
    }


def analyze(path):
    path = Path(path).resolve()
    study = read(path)
    if study.get("format") != FORMAT or not study.get("id"):
        raise ValueError("Invalid study descriptor format/id")
    tasks = study["expected_tasks"]
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("Expected tasks must be nonempty and unique")
    records, raw, seen, runs = [], {}, set(), set()
    for cell in study["cells"]:
        integer(cell["episodes"], "episodes", 1)
        integer(cell["seed"], "seed")
        integer(cell["num_steps"], "num_steps", 1)
        nonnegative(cell["extra_delay_ms"], "extra_delay_ms")
        key = cell["task"], cell["extra_delay_ms"], cell["num_steps"]
        run = (path.parent / cell["run_dir"]).resolve()
        if cell["task"] not in tasks or key in seen or run in runs or cell["id"] in raw:
            raise ValueError("Duplicate or unexpected task/condition/cell/run")
        seen.add(key)
        runs.add(run)
        record, data = audit_cell(cell, path)
        records.append(record)
        raw[cell["id"]] = data
    completed = [row for row in records if row["status"] == "completed"]
    if len({canonical(row["comparison_identity"]) for row in completed}) > 1:
        raise ValueError(
            "Completed cells differ in model/source/device comparison identity"
        )
    for task in tasks:
        if (
            len(
                {
                    canonical(row["task_identity"])
                    for row in completed
                    if row["task"] == task
                }
            )
            > 1
        ):
            raise ValueError(
                f"Task {task} has incompatible task/config/physics identity"
            )
    for row in records:
        baseline = next(
            (
                item
                for item in completed
                if item["task"] == row["task"]
                and item["extra_delay_ms"] == 0
                and item["num_steps"] == row["num_steps"]
            ),
            None,
        )
        row["paired_vs_zero"] = (
            paired(raw[baseline["id"]], raw[row["id"]])
            if baseline and raw[row["id"]]
            else None
        )
    pooled = []
    pooled_raw = {}
    for delay, num_steps in sorted(
        {(row["extra_delay_ms"], row["num_steps"]) for row in records}
    ):
        available = [
            row
            for row in completed
            if row["extra_delay_ms"] == delay and row["num_steps"] == num_steps
        ]
        full = set(row["task"] for row in available) == set(tasks)
        if full and len({(row["seed"], row["episodes"]) for row in available}) != 1:
            raise ValueError(
                "Pooled tasks must share the declared seed list and episode count"
            )
        data = combine([raw[row["id"]] for row in available]) if full else None
        pooled_raw[delay, num_steps] = data
        pooled.append(
            {
                "extra_delay_ms": delay,
                "num_steps": num_steps,
                "status": "completed" if full else "partial",
                "completed_tasks": len(available),
                "expected_tasks": len(tasks),
                "metrics": metrics(data) if data else None,
            }
        )
    for row in pooled:
        baseline = pooled_raw.get((0, row["num_steps"]))
        data = pooled_raw[row["extra_delay_ms"], row["num_steps"]]
        row["paired_vs_zero"] = paired(baseline, data) if baseline and data else None
    return {
        "format": "dynamicvla-streaming-study-results-v1",
        "id": study["id"],
        "study": str(path),
        "status": "completed"
        if all(row["status"] == "completed" for row in pooled) and bool(pooled)
        else "partial",
        "cells": records,
        "task_conditions": records,
        "pooled_conditions": pooled,
        "notes": [
            "Extra delay is synthetic wall-clock waiting after model compute; it is not lower precision or a changed model.",
            "Model compute timings are distinct from worker compute and native pacing; effective service is release minus worker start, including worker compute, native pacing, measured post-compute host overhead, and actual extra delay.",
            "Timing samples include completed generations not eventually applied; counts and sample denominators are explicit.",
            "Observation-age statistics include held model controls and exclude initial holds with no model chunk.",
            "Expiry fractions use dequeued chunks/actions; generated-but-not-dequeued chunks are not labelled expired.",
            "Failure-budget steps charge every unsuccessful episode its full native task budget.",
            "Paired success changes are descriptive; no statistical significance claims are made.",
        ],
    }


def flatten(row):
    result = {
        key: value
        for key, value in row.items()
        if key
        not in (
            "metrics",
            "paired_vs_zero",
            "comparison_identity",
            "task_identity",
            "artifact_sha256",
            "control_ledger_sha256",
        )
    }
    result.update(row.get("metrics") or {})
    result.update(
        {
            f"paired_{key}": value
            for key, value in (row.get("paired_vs_zero") or {}).items()
        }
    )
    return result


def dump_csv(path, rows):
    rows = [flatten(row) for row in rows]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    for filename, key in (
        ("cells.csv", "cells"),
        ("task-conditions.csv", "task_conditions"),
        ("pooled-conditions.csv", "pooled_conditions"),
    ):
        dump_csv(output / filename, result[key])
    text = [
        f"# DynamicVLA streaming latency study: {result['id']}",
        "",
        f"Status: {result['status']}",
        "",
        "Synthetic extra delay does not change the model or its refinement steps.",
        "",
        "| Scope | Added wall delay (ms) | Status | Episodes | Success rate (Wilson 95% CI) | Successful generated/applied chunks | Model compute median/P95 (ms) | Effective service median/P95 (ms) | Held controls |",
        "| --- | ---: | --- | ---: | --- | --- | --- | --- | --- |",
    ]

    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"

    for row in [*result["task_conditions"], *result["pooled_conditions"]]:
        m = row.get("metrics")
        scope = row.get("task", "All tasks")
        if m is None:
            text.append(
                f"| {scope} | {row['extra_delay_ms']} | {row['status']} | — | — | — | — | — | — |"
            )
            continue
        text.append(
            f"| {scope} | {row['extra_delay_ms']} | {row['status']} | {m['episodes']} | {m['success_rate']:.3f} [{m['success_rate_wilson95_low']:.3f}, {m['success_rate_wilson95_high']:.3f}] | {fmt(m['mean_generated_chunks_success'])}/{fmt(m['mean_applied_chunks_success'])} | {fmt(m['model_compute_ms_median'])}/{fmt(m['model_compute_ms_p95'])} | {fmt(m['effective_service_ms_median'])}/{fmt(m['effective_service_ms_p95'])} | {m['held_control_fraction']:.3f} |"
        )
    text.extend(
        [
            "",
            *[f"- {note}" for note in result["notes"]],
            "",
            "See CSV/JSON for full expiry counts, action ages, failure-budget steps, paired seed changes, sample counts and artifact identities.",
        ]
    )
    (output / "report.md").write_text("\n".join(text) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.study)
    save(result, args.output_dir)
    print(
        f"Study {result['id']}: {result['status']}; records saved to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
