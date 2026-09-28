"""Ownership and capacity checks for independent KINETIX GPU processes."""

import importlib.util
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest


def load_tools():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/dynamic/kinetix/latency_study.py"
    )
    spec = importlib.util.spec_from_file_location("kinetix_multiworker_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_gpu_requires_owned_processes_and_counts_starting_workers(monkeypatch):
    tools = load_tools()
    status = {"jobs": {"one": {"pid": 101, "gpu": 3, "status": "running"}}}
    monkeypatch.setattr(tools, "worker_is_alive", lambda record: True)
    monkeypatch.setattr(tools, "gpu_is_empty", lambda gpu: False)
    gpu_pids = "101\n"
    monkeypatch.setattr(
        tools.subprocess, "check_output", lambda *args, **kwargs: gpu_pids
    )
    assert tools.gpu_has_capacity(3, status, 2)
    assert not tools.gpu_has_capacity(3, status, 1)
    gpu_pids = "101\n999\n"
    assert not tools.gpu_has_capacity(3, status, 2)
    gpu_pids = ""  # Both owned processes may still be importing, before CUDA init.
    status["jobs"]["two"] = {"pid": 102, "gpu": 3, "status": "running"}
    assert not tools.gpu_has_capacity(3, status, 2)
    assert not tools.gpu_has_capacity(2, status, 2)


def test_cpu_allocation_avoids_adopted_and_new_worker_affinities(monkeypatch):
    tools = load_tools()
    status = {
        "jobs": {
            "legacy": {"pid": 101, "gpu": 1, "status": "external_running"},
            "new": {"pid": 102, "gpu": 3, "status": "running", "cpu_affinity": [2, 3]},
        }
    }
    monkeypatch.setattr(tools, "worker_is_alive", lambda record: True)
    monkeypatch.setattr(tools.os, "sched_getaffinity", lambda pid: {0, 1})
    assert tools.allocate_worker_cpus(status, list(range(6)), 2) == [4, 5]
    assert tools.allocate_worker_cpus(status, list(range(4)), 2) is None


@pytest.mark.parametrize("adopted", [False, True])
def test_parallel_dispatch_enforces_capacity_with_adopted_workers(
    tmp_path, monkeypatch, adopted
):
    tools = load_tools()
    jobs = [
        {"id": f"task/job-{i}", "phase": "validation", "mode": "fine"} for i in range(4)
    ]
    status = {"jobs": {}, "phases": {"validation": "running"}}
    alive, launched = set(), []
    if adopted:
        jobs.insert(0, {"id": "task/existing", "phase": "validation", "mode": "fine"})
        status["jobs"]["task/existing"] = {
            "pid": 100,
            "gpu": 3,
            "status": "external_running",
            "output": str(tmp_path / "existing"),
            "cpu_affinity": [0],
        }
        alive.add(100)
    monkeypatch.setattr(
        tools, "worker_is_alive", lambda record: record.get("pid") in alive
    )
    monkeypatch.setattr(tools, "gpu_is_empty", lambda gpu: True)
    monkeypatch.setattr(
        tools.subprocess,
        "check_output",
        lambda *args, **kwargs: "\n".join(map(str, alive)),
    )
    monkeypatch.setattr(tools.os, "sched_getaffinity", lambda pid: set(range(4)))
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
        if job["id"] == "task/existing":
            alive.remove(100)
        return {"episodes": 512}

    monkeypatch.setattr(tools, "verify_job", verify)

    class FastEvent(threading.Event):
        def wait(self, timeout=None):
            return super().wait(None if timeout is None else min(timeout, 0.01))

    monkeypatch.setattr(
        tools, "threading", SimpleNamespace(Lock=threading.Lock, Event=FastEvent)
    )
    peak = 0

    class Worker:
        def __init__(self, command, **kwargs):
            nonlocal peak
            self.pid = 101 + len(launched)
            launched.append(command[-1])
            alive.add(self.pid)
            peak = max(peak, len(alive))

        def wait(self):
            time.sleep(0.03)
            alive.remove(self.pid)
            return 0

    monkeypatch.setattr(tools.subprocess, "Popen", Worker)
    args = SimpleNamespace(
        phase="validation", python="python", workers_per_gpu=2, cpus_per_worker=1
    )
    assert tools.run_job_batch(args, {}, jobs, tmp_path, status, gpus=[3])
    assert peak == 2
    assert set(launched) == {f"task/job-{i}" for i in range(4)}
    assert all(record["status"] == "completed" for record in status["jobs"].values())
