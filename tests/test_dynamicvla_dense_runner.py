"""Dense experiment planning preserves actor code and disjoint paired seeds."""

import fcntl
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/dynamic/dynamicvla_dom/dense_study.py"
    )
    spec = importlib.util.spec_from_file_location("_dense_runner", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def options(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "config.json").write_text(
        json.dumps(
            dict(
                type="dynamicvla",
                use_delta_action=True,
                n_action_steps=20,
                chunk_size=20,
                n_obs_steps=2,
                num_steps=10,
            )
        )
    )
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    objects = tmp_path / "objects"
    objects.mkdir()
    robot = tmp_path / "panda.usd"
    robot.write_text("fixture")
    argv = [
        "--model-python",
        sys.executable,
        "--sim-python",
        sys.executable,
        "--checkpoint",
        str(checkpoint),
        "--scene-dir",
        str(scenes),
        "--object-dir",
        str(objects),
        "--franka-usd",
        str(robot),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    for name in ("apple", "avocado", "can"):
        task = tmp_path / f"{name}.json"
        task.write_text(
            json.dumps(
                {"scene": {"object": {"init_state": {"lin_vel": [0.2, 0.1, 0.0]}}}}
            )
        )
        argv.extend(["--task-json", str(task)])
    return module().parser().parse_args(argv)


def test_default_dense_plan_has_3600_paired_episodes(options):
    m = module()
    plan, output = m.plan_campaign(options)
    assert plan["expected_episodes"] == 3600
    assert plan["delays_ms"] == list(range(0, 501, 50)) + [800]
    assert [b["seed"] for b in plan["blocks"]] == list(range(42, 142, 10))
    assert all(b["episodes"] == 10 for b in plan["blocks"])
    command = m.block_command(plan, plan["blocks"][1], output)
    assert command[command.index("--seed") + 1] == "52"
    assert command[command.index("--episodes") + 1] == "10"
    assert command[command.index("--model-gpu") + 1] == "0"
    assert command[command.index("--order-seed") + 1] == "20260927"
    assert not output.exists()


def test_unbalanced_blocks_and_weight_replacement_rejected(options):
    m = module()
    first, _ = m.plan_campaign(options)
    (Path(options.checkpoint) / "model.safetensors").write_bytes(b"changed weights")
    second, _ = m.plan_campaign(options)
    assert first["fingerprint"] != second["fingerprint"]
    options.episodes = 103
    with pytest.raises(ValueError, match="divisible"):
        m.plan_campaign(options)


def test_dense_supervisor_requires_exclusive_lock(options):
    m = module()
    plan, output = m.plan_campaign(options)
    output.mkdir()
    with (output / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="output lock"):
            m.execute(plan, output)
    assert not (output / "campaign.json").exists()


def test_speed_change_is_forwarded_and_changes_campaign_identity(options):
    m = module()
    baseline, _ = m.plan_campaign(options)
    options.object_speed_scale = 1.25
    faster, output = m.plan_campaign(options)
    assert faster["fingerprint"] != baseline["fingerprint"]
    command = m.block_command(faster, faster["blocks"][0], output)
    assert command[command.index("--object-speed-scale") + 1] == "1.25"
    assert faster["identity"]["object_speed_scale"] == 1.25
