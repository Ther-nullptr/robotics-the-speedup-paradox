"""Native DOM launches remain offline, isolated and correctly accounted."""

import json
from pathlib import Path

import pytest

from robotics_bench.dynamicvla_dom.runner import build_plan, parser, summarize


@pytest.fixture
def options(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "dynamicvla",
                "n_action_steps": 20,
                "chunk_size": 20,
                "n_obs_steps": 2,
                "use_delta_action": True,
            }
        )
    )
    for name in ("scenes", "objects"):
        (tmp_path / name).mkdir()
    (tmp_path / "scene.json").write_text(
        json.dumps({"seed": 42, "episode_length_s": 20})
    )
    (tmp_path / "panda.usd").write_text("fixture")
    return parser().parse_args(
        [
            "--model-python",
            "/usr/bin/python3",
            "--sim-python",
            "/usr/bin/python3",
            "--checkpoint",
            str(checkpoint),
            "--scene-dir",
            str(tmp_path / "scenes"),
            "--object-dir",
            str(tmp_path / "objects"),
            "--env-cfg",
            str(tmp_path / "scene.json"),
            "--franka-usd",
            str(tmp_path / "panda.usd"),
            "--output-dir",
            str(tmp_path / "out"),
            "--streaming",
            "--episodes",
            "2",
            "--dry-run",
        ]
    )


def test_plan_is_offline_owned_and_dry_run_does_not_write(options):
    plan = build_plan(options)
    assert plan["case_id"] == "dynamicvla_dom"
    assert plan["options"]["mode"] == "native-streaming"
    assert "--streaming" in plan["commands"]["client"]
    assert "robotics_bench.dynamicvla_dom.server" in plan["commands"]["server"]
    assert plan["environments"]["client"]["HF_HUB_OFFLINE"] == "1"
    assert not Path(plan["output_dir"]).exists()
    assert "paper_sync" not in str(plan)


def test_reject_incompatible_checkpoint_and_ports(options):
    options.img_port = options.act_port
    with pytest.raises(ValueError, match="ports"):
        build_plan(options)
    options.img_port += 1
    config = Path(options.checkpoint) / "config.json"
    config.write_text(json.dumps({"type": "cosmos"}))
    with pytest.raises(ValueError, match="DynamicVLA"):
        build_plan(options)


def test_chunks_are_counted_from_generation_not_control_steps(tmp_path):
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps(
            dict(
                task="scene",
                init_state_id=0,
                env_seed=42,
                episode_id=0,
                success=True,
                primitive_steps=100,
                max_primitive_steps=500,
            )
        )
        + "\n"
    )
    (tmp_path / "requests.jsonl").write_text(
        "\n".join(
            json.dumps(dict(kind="chunk_generated", episode_id=0, chunk_id=i))
            for i in range(3)
        )
        + "\n"
    )
    result = summarize(tmp_path, expected_episodes=1)
    assert result["mean_chunks_success"] == 3
    assert result["success_rate"] == 1


def test_completed_status_cannot_mask_failed_worker_exit(options, monkeypatch):
    from robotics_bench.dynamicvla_dom import runner

    plan = build_plan(options)
    monkeypatch.setattr(runner, "require_idle_gpus", lambda selectors: [])

    class FailedWorker:
        pid = 999999999
        returncode = 7

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *args, **kwargs: FailedWorker()
    )
    with pytest.raises(RuntimeError, match="exited with 7"):
        runner.execute(plan)
    manifest = json.loads((Path(plan["output_dir"]) / "case-manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_sigterm_cleans_up_detached_workers(options, tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    plan = build_plan(options)
    for role in ("client", "server"):
        pid_file = tmp_path / f"{role}.pid"
        plan["commands"][role] = [
            sys.executable,
            "-c",
            f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)",
        ]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    script = (
        "from robotics_bench.dynamicvla_dom import runner; "
        "runner.require_idle_gpus=lambda selectors: []; "
        f"runner.execute(runner.read({str(plan_path)!r}))"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    parent = subprocess.Popen(
        [sys.executable, "-c", script],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while not all(
            (tmp_path / f"{role}.pid").exists() for role in ("client", "server")
        ):
            assert time.monotonic() < deadline, "Workers did not start"
            time.sleep(0.05)
        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=15)
        for role in ("client", "server"):
            pid = int((tmp_path / f"{role}.pid").read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert (
            json.loads((Path(plan["output_dir"]) / "case-manifest.json").read_text())[
                "status"
            ]
            == "failed"
        )
    finally:
        if parent.poll() is None:
            parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=15)


def test_latency_options_require_streaming_and_keep_compute_separate(options):
    options.measure_inference = True
    options.extra_delay_ms = 200
    options.episode_seed_mode = True
    plan = build_plan(options)
    assert plan["options"]["extra_delay_ms"] == 200
    assert "--measure-inference" in plan["commands"]["client"]
    assert "--episode-seed-mode" in plan["commands"]["client"]
    options.streaming = False
    with pytest.raises(ValueError, match="require streaming"):
        build_plan(options)


def test_study_dry_run_declares_paired_zero_delay_matrix(options, tmp_path, capsys):
    import importlib.util

    source = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/dynamic/dynamicvla_dom/study.py"
    )
    spec = importlib.util.spec_from_file_location("_dynamicvla_study", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = []
    for key in (
        "model_python",
        "sim_python",
        "checkpoint",
        "scene_dir",
        "object_dir",
        "franka_usd",
    ):
        command.extend(["--" + key.replace("_", "-"), getattr(options, key)])
    destination = tmp_path / "study-output"
    assert (
        module.main(
            [
                *command,
                "--task-json",
                options.env_cfg,
                "--delays-ms",
                "0,100,800",
                "--episodes",
                "2",
                "--output-dir",
                str(destination),
                "--dry-run",
            ]
        )
        == 0
    )
    study = json.loads(capsys.readouterr().out)
    assert len(study["cells"]) == 3
    assert {cell["extra_delay_ms"] for cell in study["cells"]} == {0, 100, 800}
    assert all(cell["episodes"] == 2 and cell["seed"] == 42 for cell in study["cells"])
    assert study["identity"]["checkpoint_sha256"]
    assert not destination.exists()


def test_no_action_episode_needs_actual_completed_generation(tmp_path):
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps(
            dict(
                task="scene",
                init_state_id=0,
                env_seed=42,
                episode_id=0,
                success=False,
                primitive_steps=5,
                max_primitive_steps=300,
                applied_chunks=0,
                applied_chunk_ids=[],
            )
        )
        + "\n"
    )
    (tmp_path / "requests.jsonl").write_text("")
    with pytest.raises(ValueError, match="generation evidence"):
        summarize(tmp_path, expected_episodes=1)
    (tmp_path / "requests.jsonl").write_text(
        json.dumps(dict(kind="chunk_generated", episode_id=0, chunk_id=0)) + "\n"
    )
    assert summarize(tmp_path, expected_episodes=1)["success_rate"] == 0
