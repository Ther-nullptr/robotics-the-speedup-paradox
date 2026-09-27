"""Coverage and provenance boundaries for hardware-delay studies."""

import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import threading
import time
from types import SimpleNamespace

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


def test_coarse_sources_are_available_before_fine_for_exact_grid_profiles():
    tools = load_study()
    profiles = tools.read_profiles(tools.DEFAULT_ARCHIVE)
    profiles["local_ada"][3] = 1000 / 30
    matrix = tools.make_matrix(["catapult"], profiles)
    executions = {row["id"]: row for row in matrix["executions"]}
    for condition in matrix["conditions"]:
        if condition["mode"] != "coarse" or condition["source"]["kind"] != "execution":
            continue
        source = executions[condition["source"]["id"]]
        assert source["phase"] == "calibration" or source["mode"] == "coarse"


def test_legacy_cross_mode_dependency_is_rejected_before_scheduling(
    tmp_path, monkeypatch
):
    tools = load_study()
    matrix = tools.make_matrix(["catapult"], tools.read_profiles(tools.DEFAULT_ARCHIVE))
    # A legacy fine-first plan can assign a shared exact-grid cell to fine.
    shared = next(row for row in matrix["executions"] if row["mode"] == "coarse")
    shared["mode"] = "fine"
    (tmp_path / "study.json").write_text(json.dumps(matrix))
    (tmp_path / "calibration").mkdir()
    tools.write_json(
        tmp_path / "calibration/frozen-model.json",
        {"study_sha256": tools.digest(tmp_path / "study.json")},
    )
    tools.write_json(
        tmp_path / "status.json",
        {
            "phases": {"verification": "completed", "calibration": "completed"},
            "jobs": {},
        },
    )
    monkeypatch.setattr(tools, "verify_inputs", lambda study: None)
    monkeypatch.setattr(
        tools,
        "run_job_batch",
        lambda *args, **kwargs: pytest.fail("Incompatible plan reached GPU scheduling"),
    )
    args = SimpleNamespace(
        study_dir=tmp_path,
        python=Path(sys.executable),
        gpus="0",
        cpus_per_worker=1,
        phase="validation",
        reuse_study=None,
    )
    with pytest.raises(ValueError, match="coarse-first"):
        tools.run_phase(args)


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


def test_baseline_accepts_audited_rename_but_rejects_changed_model(
    tmp_path, monkeypatch
):
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
    copied_source = tmp_path / "owned_source"
    copied_level = copied_source / "levels" / f"{task}.json"
    copied_level.parent.mkdir(parents=True)
    copied_level.write_bytes(level.read_bytes())
    with monkeypatch.context() as context:
        context.setattr(study, "SOURCE", copied_source)
        plan = {"source_sha256": {}, "baseline": evidence, "policy_dir": str(tmp_path)}
        study.verify_inputs(plan)
        copied_level.write_text(copied_level.read_text() + "\n")
        with pytest.raises(ValueError, match="level"):
            study.verify_inputs(plan)
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


def test_resume_attaches_matching_live_local_worker(tmp_path):
    tools = load_study()
    output = str(tmp_path / "attempt-001")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; print('ready', flush=True); sys.stdin.read()",
            output,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        record = {"status": "running", "pid": process.pid, "output": output}
        assert tools.attach_live_worker(record)
        assert record["status"] == "external_running"
        assert not tools.worker_is_alive(
            {**record, "output": str(tmp_path / "attempt")}
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_imported_completion_must_match_original_certificate():
    tools = load_study()
    original = {
        "episodes": 512,
        "manifest_sha256": "manifest",
        "episodes_sha256": "original",
    }
    record = {"status": "completed", "verification": original}
    tools.verify_completion_evidence(record, dict(original))
    with pytest.raises(ValueError, match="changed"):
        tools.verify_completion_evidence(
            record, {**original, "episodes_sha256": "changed"}
        )


@pytest.mark.parametrize("detached_coarse", [False, True])
@pytest.mark.parametrize("failed_coarse", [False, True])
def test_validation_waits_for_all_coarse_workers(
    tmp_path, monkeypatch, detached_coarse, failed_coarse
):
    tools = load_study()
    coarse_ids = {"a/coarse", "b/coarse"}
    jobs = [
        {"id": name, "phase": "validation", "mode": mode}
        for name, mode in (
            ("a/fine", "fine"),
            ("a/coarse", "coarse"),
            ("b/fine", "fine"),
            ("b/coarse", "coarse"),
        )
    ]
    (tmp_path / "study.json").write_text(json.dumps({"jobs": jobs}))
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration/frozen-model.json").write_text(
        json.dumps({"study_sha256": tools.digest(tmp_path / "study.json")})
    )
    status = {
        "phases": {"verification": "completed", "calibration": "completed"},
        "jobs": {},
    }
    if detached_coarse:
        status["jobs"]["b/coarse"] = {
            "status": "external_running",
            "pid": 123,
            "output": str(tmp_path / "existing-coarse"),
        }
    tools.write_json(tmp_path / "status.json", status)
    launched, verified = [], set()
    fast_finished = threading.Event()
    monkeypatch.setattr(tools, "verify_inputs", lambda study: None)
    monkeypatch.setattr(tools, "gpu_is_empty", lambda gpu: True)
    monkeypatch.setattr(tools, "recover_completed_job", lambda *args: False)
    monkeypatch.setattr(
        tools,
        "attach_live_worker",
        lambda record: record.get("status") == "external_running",
    )
    monkeypatch.setattr(
        tools,
        "job_command",
        lambda study, job, directory, **kwargs: ["fake-worker", job["id"]],
    )

    def verify(study, job, output, **kwargs):
        verified.add(job["id"])
        return {"episodes": 512}

    monkeypatch.setattr(tools, "verify_job", verify)

    class Worker:
        def __init__(self, command, **kwargs):
            self.job_id = command[-1]
            if self.job_id.endswith("/fine"):
                assert coarse_ids <= verified, "Fine started before coarse completion"
            launched.append(self.job_id)
            self.pid = 1000 + len(launched)

        def wait(self):
            if self.job_id == "a/coarse":
                fast_finished.set()
                return int(failed_coarse)
            if self.job_id == "b/coarse":
                assert fast_finished.wait(2)
                time.sleep(0.05)
            return 0

    monkeypatch.setattr(tools.subprocess, "Popen", Worker)
    args = SimpleNamespace(
        study_dir=tmp_path,
        python=Path(sys.executable),
        gpus="0,1",
        cpus_per_worker=1,
        phase="validation",
        reuse_study=None,
    )
    if failed_coarse:
        with pytest.raises(RuntimeError, match="incomplete"):
            tools.run_phase(args)
        assert not any(name.endswith("/fine") for name in launched)
    else:
        tools.run_phase(args)
        assert set(launched) == {job["id"] for job in jobs} - (
            {"b/coarse"} if detached_coarse else set()
        )
    final = json.loads((tmp_path / "status.json").read_text())
    assert final["phases"]["validation"] == ("failed" if failed_coarse else "completed")
    assert final["validation_modes"]["coarse"] == (
        "failed" if failed_coarse else "completed"
    )
    assert final["validation_modes"].get("fine") == (
        None if failed_coarse else "completed"
    )
