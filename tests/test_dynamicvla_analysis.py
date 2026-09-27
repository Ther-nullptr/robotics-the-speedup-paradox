"""Audits for measured streaming studies; all fixtures are CPU-only synthetic ledgers."""

import json

import pytest

from robotics_bench.dynamicvla_dom.analysis import FORMAT, analyze, save, wilson


def write(path, value):
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_run(root, task, delay, successes=(True, False)):
    run = root / f"{task}-{delay}"
    run.mkdir()
    cell = {
        "id": run.name,
        "task": task,
        "extra_delay_ms": delay,
        "num_steps": 10,
        "episodes": 2,
        "seed": 4,
        "run_dir": run.name,
    }
    options = {
        key: cell[key] for key in ("extra_delay_ms", "num_steps", "episodes", "seed")
    }
    options.update(
        mode="native-streaming", measure_inference=True, episode_seed_mode=True
    )
    manifest = {
        "case_id": "dynamicvla_dom",
        "status": "completed",
        "options": options,
        "identity": {"checkpoint_config_sha256": "config", "task_config_sha256": task},
        "checkpoint_sha256": "weights",
        "source_sha256": {"server.py": "source"},
    }
    write(run / "case-manifest.json", manifest)
    write(
        run / "coverage.json",
        {
            "status": "passed",
            "expected_episodes": 2,
            "completed_episodes": 2,
            "successes": sum(successes),
        },
    )
    write(
        run / "effective-config.json",
        {
            "physics_dt_s": 0.04,
            "control_dt_s": 0.04,
            "decimation": 1,
            "control_protocol": "native_hold_last",
        },
    )
    episodes, requests = [], []
    for episode, success in enumerate(successes):
        episodes.append(
            {
                "episode_id": episode,
                "task": task,
                "env_seed": 4 + episode,
                "success": success,
                "primitive_steps": 3,
                "max_primitive_steps": 10,
                "applied_chunk_ids": [0],
                "applied_chunks": 1,
                "held_control_steps": 2,
                "actions_accepted": 1,
            }
        )
        for chunk in (0, 1):
            requests.append(
                {
                    "kind": "chunk_generated",
                    "episode_id": episode,
                    "chunk_id": chunk,
                    "generated_actions": 20,
                    "model_compute_ms": 5 + chunk,
                    "worker_compute_ms": 8,
                    "native_pacing_ms": 2,
                    "extra_delay_actual_ms": delay,
                    "extra_delay_requested_ms": delay,
                    "worker_started_wall_s": 0.99,
                    "compute_ready_wall_s": 1,
                    "delay_started_wall_s": 1,
                    "post_compute_overhead_ms": 0,
                    "released_wall_s": 1 + delay / 1000,
                    "result_enqueued": True,
                }
            )
        requests += [
            {
                "kind": "chunk_received",
                "episode_id": episode,
                "chunk_id": 0,
                "expired_actions": 2,
            },
            {
                "kind": "chunk_expired",
                "episode_id": episode,
                "chunk_id": 1,
                "expired_actions": 20,
            },
        ]
        controls = [{"control_index": 0, "held": True}]
        for i in (1, 2):
            controls.append(
                {
                    "control_index": i,
                    "held": i == 2,
                    "chunk_id": 0,
                    "sim_time_s": i * 0.04,
                    "wall_time_s": i,
                    "chunk_observation_index": 0,
                    "chunk_observation_sim_time_s": 0,
                    "chunk_observation_wall_s": 0.5,
                    "action_age_sim_ms": i * 40,
                    "action_age_wall_ms": (i - 0.5) * 1000,
                }
            )
        jsonl(run / f"episode-{episode:06d}-controls.jsonl", controls)
    jsonl(run / "episodes.jsonl", episodes)
    jsonl(run / "requests.jsonl", requests)
    return cell


def study(root, cells, tasks=("pick", "place")):
    path = root / "study.json"
    write(
        path,
        {
            "format": FORMAT,
            "id": "latency-example",
            "expected_tasks": tasks,
            "cells": cells,
        },
    )
    return path


def test_full_study_recomputes_denominators_and_timing(tmp_path):
    cells = [
        make_run(tmp_path, task, delay, successes=(True, delay == 0))
        for task in ("pick", "place")
        for delay in (0, 100)
    ]
    result = analyze(study(tmp_path, cells))
    assert result["status"] == "completed"
    delayed = result["pooled_conditions"][1]
    m = delayed["metrics"]
    assert (m["episodes"], m["successes"], m["success_rate"]) == (4, 2, 0.5)
    assert m["mean_generated_chunks_success"] == 2
    assert m["mean_applied_chunks_success"] == 1
    assert m["mean_steps_failure_budget"] == 6.5
    assert m["held_control_fraction"] == pytest.approx(2 / 3)
    assert m["effective_service_ms_median"] == pytest.approx(110)
    assert m["model_compute_ms_median"] == 5.5
    assert m["expired_action_fraction_dequeued"] == 22 / 40
    assert m["fully_expired_chunk_fraction_dequeued"] == 0.5
    assert m["action_age_sim_ms_samples"] == 8
    assert delayed["paired_vs_zero"]["losses"] == 2
    assert delayed["paired_vs_zero"]["gains"] == 0
    output = tmp_path / "analysis"
    save(result, output)
    assert len(list(output.iterdir())) == 5
    assert "Synthetic extra delay" in (output / "report.md").read_text()


def test_partial_and_failed_cells_never_create_pooled_metrics(tmp_path):
    complete = make_run(tmp_path, "pick", 0)
    failed = make_run(tmp_path, "place", 0)
    path = tmp_path / failed["run_dir"] / "case-manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(status="failed", error="simulator unavailable")
    write(path, manifest)
    result = analyze(study(tmp_path, [complete, failed]))
    assert result["status"] == "partial"
    assert result["cells"][1]["metrics"] is None
    assert result["pooled_conditions"][0]["metrics"] is None
    assert result["pooled_conditions"][0]["completed_tasks"] == 1


def test_zero_success_returns_null_conditioned_chunks(tmp_path):
    cell = make_run(tmp_path, "pick", 0, successes=(False, False))
    result = analyze(study(tmp_path, [cell], tasks=["pick"]))
    m = result["pooled_conditions"][0]["metrics"]
    assert m["mean_generated_chunks_success"] is None
    assert m["mean_applied_chunks_success"] is None
    assert m["mean_steps_failure_budget"] == 10
    assert wilson(0, 2)[0] == 0


@pytest.mark.parametrize(
    "corruption",
    [
        "source",
        "task",
        "coverage",
        "applied",
        "timing",
        "age",
        "requested_delay",
        "release_gap",
    ],
)
def test_completed_invalid_evidence_fails_closed(tmp_path, corruption):
    cells = [make_run(tmp_path, "pick", delay) for delay in (0, 100)]
    run = tmp_path / cells[1]["run_dir"]
    if corruption in ("source", "task"):
        path = run / "case-manifest.json"
        value = json.loads(path.read_text())
        if corruption == "source":
            value["source_sha256"]["server.py"] = "different"
        else:
            value["identity"]["task_config_sha256"] = "different"
        write(path, value)
    elif corruption == "coverage":
        path = run / "coverage.json"
        value = json.loads(path.read_text())
        value["completed_episodes"] = 1
        write(path, value)
    elif corruption == "applied":
        path = run / "episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["applied_chunks"] = 2
        jsonl(path, rows)
    elif corruption in ("timing", "requested_delay", "release_gap"):
        path = run / "requests.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if corruption == "timing":
            rows[0]["model_compute_ms"] = -1
        elif corruption == "requested_delay":
            rows[0]["extra_delay_requested_ms"] += 1
        else:
            rows[0]["released_wall_s"] += 0.1
        jsonl(path, rows)
    else:
        path = run / "episode-000000-controls.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["action_age_wall_ms"] = 100
        jsonl(path, rows)
    with pytest.raises(ValueError):
        analyze(study(tmp_path, cells, tasks=["pick"]))


def test_duplicate_run_reuse_rejected(tmp_path):
    cell = make_run(tmp_path, "pick", 0)
    copied = {**cell, "id": "reused", "extra_delay_ms": 100}
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(study(tmp_path, [cell, copied], tasks=["pick"]))


def test_post_compute_preemption_is_measured_not_rejected(tmp_path):
    cell = make_run(tmp_path, "pick", 100)
    path = tmp_path / cell["run_dir"] / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["kind"] == "chunk_generated":
            row["delay_started_wall_s"] = 1.1
            row["released_wall_s"] = 1.2
            row["post_compute_overhead_ms"] = 100
    jsonl(path, rows)
    result = analyze(study(tmp_path, [cell], tasks=["pick"]))
    metrics = result["pooled_conditions"][0]["metrics"]
    assert metrics["post_compute_overhead_ms_median"] == pytest.approx(100)
    assert metrics["extra_delay_actual_ms_median"] == 100
    assert metrics["effective_service_ms_median"] == pytest.approx(210)


@pytest.mark.parametrize("field", ["delay_started_wall_s", "post_compute_overhead_ms"])
def test_delay_boundary_or_overhead_corruption_fails(tmp_path, field):
    cell = make_run(tmp_path, "pick", 100)
    path = tmp_path / cell["run_dir"] / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = 0.5 if field == "delay_started_wall_s" else 100
    jsonl(path, rows)
    with pytest.raises(ValueError):
        analyze(study(tmp_path, [cell], tasks=["pick"]))


def test_speed_mismatch_is_not_pooled_into_latency_curve(tmp_path):
    cell = make_run(tmp_path, "pick", 0)
    p = tmp_path / cell["run_dir"] / "case-manifest.json"
    manifest = json.loads(p.read_text())
    manifest["options"]["object_speed_scale"] = 1.25
    write(p, manifest)
    with pytest.raises(ValueError, match="object speed scale"):
        analyze(study(tmp_path, [cell], tasks=["pick"]))


def test_configured_speed_must_reach_native_default_velocity(tmp_path):
    from robotics_bench.dynamicvla_dom.object_motion import scale_object_speed

    cell = make_run(tmp_path, "pick", 0)
    cell["object_speed_scale"] = 1.25
    run = tmp_path / cell["run_dir"]
    config, motion = scale_object_speed(
        {"scene": {"object": {"init_state": {"lin_vel": [0.2, 0.0, 0.0]}}}}, 1.25
    )
    manifest = json.loads((run / "case-manifest.json").read_text())
    manifest["options"]["object_speed_scale"] = 1.25
    manifest["object_motion"] = motion
    write(run / "case-manifest.json", manifest)
    effective = json.loads((run / "effective-config.json").read_text())
    effective.update(object_speed_scale=1.25, object_motion=motion, input_config=config)
    write(run / "effective-config.json", effective)
    rows = [
        json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()
    ]
    for row in rows:
        row.update(
            object_speed_scale=1.25,
            object_motion={**motion, "default_lin_vel_mps": [0.25, 0.0, 0.0]},
        )
    jsonl(run / "episodes.jsonl", rows)
    path = study(tmp_path, [cell], tasks=["pick"])
    descriptor = json.loads(path.read_text())
    descriptor["identity"] = {"object_speed_scale": 1.25}
    write(path, descriptor)
    assert analyze(path)["pooled_conditions"][0]["metrics"]["success_rate"] == 0.5
    rows[0]["object_motion"]["default_lin_vel_mps"] = [0.2, 0.0, 0.0]
    jsonl(run / "episodes.jsonl", rows)
    with pytest.raises(ValueError, match="Native default object velocity"):
        analyze(path)
