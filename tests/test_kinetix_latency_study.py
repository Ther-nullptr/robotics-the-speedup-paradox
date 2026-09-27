"""Coverage and provenance boundaries for hardware-delay studies."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def load_study():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/dynamic/kinetix/latency_study.py"
    )
    spec = importlib.util.spec_from_file_location("kinetix_latency_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_preserves_logical_conditions_without_inflating_samples():
    study = load_study()
    profiles = study.read_profiles(study.DEFAULT_ARCHIVE)
    matrix = study.make_matrix(["catcher_v3"], profiles)
    assert len(matrix["conditions"]) == 57
    assert len(matrix["executions"]) == 31
    assert len(matrix["jobs"]) == 6
    assert sum(len(job["execution_ids"]) for job in matrix["jobs"]) == 31
    coarse = [
        row
        for row in matrix["conditions"]
        if row["role"] == "validation" and row["mode"] == "coarse"
    ]
    fast = [row for row in coarse if row["hardware"] in {"local_ada", "rtx3090"}]
    assert len(fast) == 10
    assert all(row["source"]["kind"] == "baseline" for row in fast)
    assert all(row["calibration_overlap"] for row in fast)
    fine = [
        row
        for row in matrix["conditions"]
        if row["role"] == "validation" and row["mode"] == "fine"
    ]
    assert len(fine) == 20
    assert not any(row["calibration_overlap"] for row in fine)
    ids = {row["id"] for row in matrix["executions"]}
    assert all(
        row["source"]["id"] in ids
        for row in matrix["conditions"]
        if row["source"]["kind"] == "execution"
    )


def test_duplicate_or_missing_seed_is_not_accepted_as_complete_coverage():
    study = load_study()
    rows = [{"flow_steps": 1, "env_seed": seed} for seed in range(3)]
    study.check_coverage(rows, [1], start_seed=0, episodes=3)
    with pytest.raises(ValueError, match="coverage|Duplicate"):
        study.check_coverage(rows + rows[:1], [1], start_seed=0, episodes=3)
    with pytest.raises(ValueError, match="coverage"):
        study.check_coverage(rows[:-1], [1], start_seed=0, episodes=3)


def test_profile_rejects_excluded_or_ambiguous_hardware(tmp_path):
    study = load_study()
    original = study.DEFAULT_ARCHIVE.read_text()
    path = tmp_path / "profiles.csv"
    path.write_text(original + original.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        study.read_profiles(path)
    path.write_text(original.replace("agx30", "agx50"))
    with pytest.raises(ValueError, match="profile|hardware"):
        study.read_profiles(path)


def test_baseline_accepts_audited_rename_but_rejects_changed_model(tmp_path):
    study = load_study()
    task = "catcher_v3"
    directory = tmp_path / task
    directory.mkdir()
    checkpoint = tmp_path / f"worlds_l_{task}.pkl"
    checkpoint.write_bytes(b"test checkpoint identity")
    level = study.SOURCE / "levels" / f"{task}.json"
    native = json.loads(level.read_text())
    hashes = {
        str(p.relative_to(study.SOURCE)): study.digest(p)
        for p in study.SOURCE.rglob("*.py")
    }
    for name, (old, _) in study.NAMING_MIGRATION.items():
        hashes[name] = old
    manifest = {
        "status": "completed",
        "episodes_per_cell": 2,
        "start_seed": 0,
        "execute_horizon": 4,
        "action_noise_std": 0.1,
        "mapping": "native-blend",
        "source_sha256": hashes,
        "resource_sha256": {
            str(checkpoint): study.digest(checkpoint),
            str(level): study.digest(level),
        },
        "tasks": {
            task: {
                "native_env_params": native["env_params"],
                "native_static_env_params": native["static_env_params"],
                "max_steps": 256,
                "checkpoint": str(checkpoint),
                "level_path": str(level),
                "runtime": {"model_config": {"dtype": "float32"}},
            }
        },
        "runtime": {"packages": {}},
        "protocol": {
            "host_time_advances_simulation": False,
            "initial_policy_prefetches": 0,
            "initial_previous_action": "zeros",
            "physics_refinement": False,
            "blend_domain": "processed_actuator_command",
        },
    }
    rows = [
        {
            "task": task,
            "flow_steps": n,
            "env_seed": seed,
            "requested_latency_ms": 0,
            "effective_latency_ms": 0,
            "delay_mapping": "native-blend",
            "success": True,
            "max_primitive_steps": 256,
            "primitive_steps": 1,
            "physics_steps": 2,
        }
        for n in range(1, 6)
        for seed in range(2)
    ]
    path = directory / "case-manifest.json"
    path.write_text(json.dumps(manifest))
    (directory / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    evidence = study.verify_baseline(
        tmp_path, [task], tmp_path, episodes=2, start_seed=0
    )
    assert evidence[task]["checkpoint_sha256"] == study.digest(checkpoint)
    manifest["source_sha256"]["flow_model.py"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="flow_model"):
        study.verify_baseline(tmp_path, [task], tmp_path, episodes=2, start_seed=0)


def test_running_job_reuse_rejects_different_checkpoint(tmp_path):
    tools = load_study()
    task = "catapult"
    level = tools.SOURCE / "levels" / f"{task}.json"
    native = json.loads(level.read_text())
    study = {
        "source_sha256": {"src/robotics_bench/kinetix/flow_model.py": "same"},
        "episodes_per_cell": 512,
        "start_seed": 0,
        "baseline": {
            task: {
                "runtime_packages": {},
                "model_config": {},
                "checkpoint_sha256": "a" * 64,
            }
        },
    }
    job = {"id": "catapult/delay", "task": task, "mode": "fine", "phase": "calibration"}
    manifest = {
        "status": "running",
        "source_sha256": {"flow_model.py": "same"},
        "episodes_per_cell": 512,
        "start_seed": 0,
        "action_noise_std": 0.1,
        "execute_horizon": 4,
        "mapping": "fine",
        "protocol": {"host_time_advances_simulation": False},
        "runtime": {"packages": {}},
        "tasks": {
            task: {
                "runtime": {"model_config": {}},
                "checkpoint": "different.pkl",
                "level_path": str(level),
                "max_steps": 256,
                "native_env_params": native["env_params"],
                "native_static_env_params": native["static_env_params"],
            }
        },
        "resource_sha256": {"different.pkl": "b" * 64, str(level): tools.digest(level)},
    }
    (tmp_path / "case-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checkpoint"):
        tools.verify_job(study, job, tmp_path, allow_running=True)


def test_runtime_executable_keeps_virtual_environment_symlink(tmp_path):
    tools = load_study()
    executable = tmp_path / "venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    assert tools.interpreter_path(executable) == executable.absolute()
    assert tools.interpreter_path(executable) != executable.resolve()
