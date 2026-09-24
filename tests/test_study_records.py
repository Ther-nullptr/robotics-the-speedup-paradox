"""Record completed studies without pooling unrelated or reused runs."""

import importlib.util
import json
from pathlib import Path

import pytest


def tool():
    path = Path(__file__).resolve().parents[1] / "tools/study_records.py"
    assert path.exists(), "Unified study recorder is missing"
    spec = importlib.util.spec_from_file_location("_study_records", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def study(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        dict(
            task="Task",
            init_state_id=0,
            env_seed=0,
            success=True,
            primitive_steps=7,
            inference_calls=4,
        ),
        dict(
            task="Task",
            init_state_id=1,
            env_seed=1,
            success=False,
            primitive_steps=10,
            inference_calls=5,
        ),
    ]
    options = dict(
        task="Task",
        episodes=2,
        seed=195,
        env_seed=0,
        layout_id=1,
        style_id=1,
        n_action_steps=2,
        num_inference_steps=5,
        action_horizon=32,
        schedule="sync",
        overlap_actions=0,
        model_runtime="owned",
        optimizations={
            "precision": "bf16",
            "switches": [],
            "scopes": ["dit"],
            "tactic": 1,
            "quant_tier": None,
        },
    )
    manifest = {
        "case_id": "cosmos_robocasa",
        "status": "completed",
        "options": options,
        "identity": {"case": options},
        "case_fingerprint": "identity",
        "gpu": "3",
        "resources": {"checkpoint": "policy.pt"},
    }
    (run / "case-manifest.json").write_text(json.dumps(manifest))
    (run / "coverage.json").write_text(
        json.dumps(
            dict(
                status="passed",
                expected_episodes=2,
                completed_episodes=2,
                successes=1,
                max_primitive_steps=10,
            )
        )
    )
    (run / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    requests = []
    for row in rows:
        for i in range(row["inference_calls"]):
            requests.append(
                dict(
                    task="Task",
                    init_state_id=row["init_state_id"],
                    env_seed=row["env_seed"],
                    inference_index=i + 1,
                    control_step=i * 2,
                    observation_step=i * 2,
                    history_offset_steps=0,
                )
            )
    (run / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in requests)
    )
    descriptor = {
        "format": "robotics-study-v1",
        "id": "study-a",
        "case_id": "cosmos_robocasa",
        "code_commit": "commit-a",
        "cells": [
            {
                "id": "Task/bf16",
                "task": "Task",
                "role": "reference",
                "run_dir": str(run),
                "expected_episodes": 2,
            }
        ],
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(descriptor))
    return path, run


def test_uses_success_conditioned_recorded_chunks(study):
    path, run = study
    result = tool().collect(path)
    row = result["records"][0]
    assert row["success_rate"] == 0.5
    assert row["mean_chunks_success"] == 4
    assert row["mean_control_steps_failure_budget"] == 8.5
    assert row["policy_median_ms"] is None
    assert row["timing_status"] == "not_requested"


def test_unfinished_cells_have_no_aggregate_metrics(study):
    path, run = study
    manifest = json.loads((run / "case-manifest.json").read_text())
    manifest["status"] = "running"
    (run / "case-manifest.json").write_text(json.dumps(manifest))
    result = tool().collect(path)
    assert not result["complete"]
    assert result["records"][0]["success_rate"] is None


def test_rejects_duplicate_ledgers_and_stale_chunk_counts(study):
    path, run = study
    descriptor = json.loads(path.read_text())
    descriptor["cells"].append({**descriptor["cells"][0], "id": "duplicate"})
    path.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="same run"):
        tool().collect(path)
    descriptor["cells"].pop()
    path.write_text(json.dumps(descriptor))
    rows = [
        json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()
    ]
    rows[0]["inference_calls"] = 3
    (run / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="chunk"):
        tool().collect(path)


def test_timing_binding_checks_workload_and_uses_raw_samples(study, tmp_path):
    path, run = study
    timing = tmp_path / "timing"
    timing.mkdir()
    meta = {
        "status": "completed",
        "case": "cosmos_robocasa",
        "model": {
            "steps": 5,
            "horizon": 32,
            "checkpoint": "policy.pt",
            "runtime": "owned",
        },
        "gpu": "Ada",
        "gpu_uuid": "UUID",
        "gpu_selector": "3",
    }
    (timing / "manifest.json").write_text(json.dumps(meta))
    (timing / "measurements.json").write_text(
        json.dumps(
            [
                {
                    "id": "original",
                    "precision": "bf16",
                    "switches": [],
                    "quant_tier": None,
                    "samples_ms": [10.0, 20.0, 30.0],
                    "median_ms": 20.0,
                }
            ]
        )
    )
    descriptor = json.loads(path.read_text())
    descriptor["cells"][0]["timing"] = {
        "run_dir": str(timing),
        "variant_id": "original",
    }
    path.write_text(json.dumps(descriptor))
    row = tool().collect(path)["records"][0]
    assert row["policy_median_ms"] == 20
    assert row["policy_p95_ms"] == 29
    initial = run / "initializations/000000"
    initial.mkdir(parents=True)
    observation = initial / "rendered-observation.npz"
    observation.write_bytes(b"fixed-input")
    (initial / "episode.json").write_text(json.dumps({"description": "Close drawer"}))
    meta.update(task="Close drawer", input_sha256=[tool().digest(observation)])
    (timing / "manifest.json").write_text(json.dumps(meta))
    assert tool().collect(path)["records"][0]["policy_median_ms"] == 20
    meta["input_sha256"].insert(0, "different-timed-input")
    (timing / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Timing input"):
        tool().collect(path)
    meta["input_sha256"].pop(0)
    meta["model"]["steps"] = 1
    (timing / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sampling"):
        tool().collect(path)


def test_registry_keeps_study_boundaries(study, tmp_path):
    path, run = study
    module = tool()
    registry = tmp_path / "records"
    module.register(path, registry)
    second = tmp_path / "second.json"
    descriptor = json.loads(path.read_text())
    descriptor["id"] = "study-b"
    descriptor["cells"][0]["quality_reused_from"] = "study-a"
    second.write_text(json.dumps(descriptor))
    module.register(second, registry)
    data = json.loads((registry / "records.json").read_text())
    assert {r["study_id"] for r in data["records"]} == {"study-a", "study-b"}
    assert len(data["records"]) == 2
    assert (registry / "records.csv").is_file()
    assert (registry / "README.md").is_file()
    descriptor["cells"][0].pop("quality_reused_from")
    second.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="explicit reuse"):
        module.register(second, registry)


def test_progressive_timing_rejects_wrong_integer_tactic(study, tmp_path):
    _, run = study
    manifest = json.loads((run / "case-manifest.json").read_text())
    timing = tmp_path / "timing"
    timing.mkdir()
    (timing / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "case": "cosmos_robocasa",
                "model": {
                    "steps": 5,
                    "horizon": 32,
                    "checkpoint": "policy.pt",
                    "runtime": "owned",
                },
            }
        )
    )
    (timing / "measurements.json").write_text(
        json.dumps(
            [
                {
                    "id": "tier-01",
                    "precision": "w4-t1",
                    "quant_tier": 1,
                    "scopes": ["dit"],
                    "tactic": 7,
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="integer tactic"):
        tool().timing(
            {"run_dir": str(timing), "variant_id": "tier-01"},
            tmp_path,
            manifest["options"],
            manifest,
            1,
            run,
        )
