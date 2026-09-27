"""Native-time delay accounting, action handoff and isolated episode semantics."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("name", "alias", "effective"),
    [("fine", "native-blend", 20.0), ("coarse", "legacy-round", 100 / 3)],
)
def test_readable_delay_modes_preserve_alias_physics(name, alias, effective):
    from robotics_bench.kinetix.protocol import delay_plan, normalize_mapping

    arguments = dict(physics_dt=1 / 60, frame_skip=2, execute_horizon=4)
    current = delay_plan(20, mapping=name, **arguments)
    historical = delay_plan(20, mapping=alias, **arguments)
    assert normalize_mapping(alias) == name
    assert current["mapping"] == name
    assert historical["mapping"] == alias
    assert current["effective_latency_ms"] == pytest.approx(effective)
    assert current["old_weights"] == historical["old_weights"]
    assert current["physics_steps_per_cycle"] == 8


@pytest.mark.parametrize("latency", [0.0, 0.417, 7.465, 21.938, 36.738, 400 / 3])
def test_native_blend_preserves_fractional_delay_and_native_solver_count(latency):
    from robotics_bench.kinetix.protocol import delay_plan

    plan = delay_plan(latency, physics_dt=1 / 60, frame_skip=2, execute_horizon=4)
    weights = np.asarray(plan["old_weights"])
    assert weights.shape == (4, 2)
    assert np.all((weights >= 0) & (weights <= 1))
    assert np.all(np.diff(weights.ravel()) <= 0)
    assert ((weights > 0) & (weights < 1)).sum() <= 1
    assert weights.sum() * 1000 / 60 == pytest.approx(latency)
    assert plan["effective_latency_ms"] == pytest.approx(latency)
    assert plan["physics_steps_per_cycle"] == 8


def test_legacy_round_changes_availability_only_and_preserves_requested_delay():
    from robotics_bench.kinetix.protocol import delay_plan

    plan = delay_plan(
        20, physics_dt=1 / 60, frame_skip=2, execute_horizon=4, mapping="legacy-round"
    )
    assert plan["requested_latency_ms"] == 20
    assert plan["effective_latency_ms"] == pytest.approx(100 / 3)
    assert plan["old_weights"] == [[1, 1], [0, 0], [0, 0], [0, 0]]


@pytest.mark.parametrize("latency", [-1, float("nan"), float("inf"), 134])
def test_delay_rejects_invalid_values_or_unavailable_old_action_tail(latency):
    from robotics_bench.kinetix.protocol import delay_plan

    with pytest.raises(ValueError):
        delay_plan(latency, physics_dt=1 / 60, frame_skip=2, execute_horizon=4)
    with pytest.raises(ValueError, match="tail|horizon"):
        delay_plan(34, physics_dt=1 / 60, frame_skip=2, execute_horizon=7)


def test_boundary_blends_processed_commands_and_has_exact_endpoints():
    from robotics_bench.kinetix.protocol import mix_processed_commands

    raw_old, raw_new = np.array([-2.0]), np.array([0.5])

    def process(value):
        return np.clip(value, 0, 1)

    mixed = mix_processed_commands(
        process(raw_old), process(raw_new), np.array([1, 0.5, 0]), np
    )
    np.testing.assert_array_equal(mixed[:, 0], [0, 0.25, 0.5])
    assert mixed[1, 0] != process((raw_old + raw_new) / 2)[0]
    constant = mix_processed_commands(
        np.array([0.3]), np.array([0.3]), np.array([1, 0.1, 0]), np
    )
    np.testing.assert_array_equal(constant, np.full((3, 1), 0.3))


@pytest.mark.parametrize("terminal", [True, False])
def test_rollout_has_no_prefetch_retains_tail_and_stops_first_episode(terminal):
    from robotics_bench.kinetix.runner import run_episode

    calls, actions, events, frames = [], [], [], []
    env = SimpleNamespace(
        physics_dt=1 / 60, frame_skip=2, max_steps=12, action_dim=1, count=0
    )

    def reset(seed):
        env.count = 0
        return np.array([0.0])

    def step(old, new, weights):
        actions.append((old.copy(), new.copy(), weights.copy()))
        env.count += 1
        done = terminal and env.count == 5
        return np.array([float(env.count)]), float(done), done, done

    env.reset, env.step, env.render = reset, step, lambda: env.count
    policy = SimpleNamespace(action_horizon=8, reset=lambda seed: None)

    def infer(obs, flow_steps):
        calls.append(float(obs[0]))
        return (np.arange(8, dtype=float) + 10 * len(calls))[:, None]

    policy.infer = infer
    row = run_episode(
        policy,
        env,
        seed=3,
        flow_steps=5,
        latency_ms=20,
        execute_horizon=4,
        max_steps=6,
        on_event=events.append,
        on_frame=frames.append,
    )
    assert calls == [0, 4]
    assert row["primitive_steps"] == (5 if terminal else 6)
    assert row["success"] is terminal
    assert row["max_primitive_steps"] == 6
    assert len(frames) == row["primitive_steps"] + 1
    assert [float(a[0][0]) for a in actions[:4]] == [0, 0, 0, 0]
    assert float(actions[4][0][0]) == 14
    assert float(actions[4][1][0]) == 20
    assert row["inference_calls"] == 2
    assert row["physics_steps"] == row["primitive_steps"] * 2
    requests = [e for e in events if e["kind"] == "inference"]
    assert [r["observation_step"] for r in requests] == [0, 4]
    assert requests[0]["release_sim_seconds"] == pytest.approx(0.02)
    assert frames == list(range(row["primitive_steps"] + 1))


def test_invalid_policy_output_is_an_infrastructure_error():
    from robotics_bench.kinetix.runner import run_episode

    env = SimpleNamespace(
        physics_dt=1 / 60,
        frame_skip=2,
        max_steps=12,
        action_dim=1,
        reset=lambda seed: np.zeros(1),
    )
    policy = SimpleNamespace(
        action_horizon=8,
        reset=lambda seed: None,
        infer=lambda obs, n: np.full((8, 1), np.nan),
    )
    with pytest.raises(ValueError, match="finite"):
        run_episode(policy, env, seed=0, flow_steps=5, latency_ms=0, execute_horizon=4)


def test_summary_partitions_cells_and_penalizes_early_native_failures():
    from robotics_bench.kinetix.results import summarize

    common = {
        "task": "car_launch",
        "flow_steps": 5,
        "delay_mapping": "native-blend",
        "max_primitive_steps": 12,
        "inference_calls": 2,
    }
    rows = [
        {
            **common,
            "init_state_id": 0,
            "env_seed": 0,
            "requested_latency_ms": 0,
            "effective_latency_ms": 0,
            "success": True,
            "primitive_steps": 5,
            "episode_return": 1,
        },
        {
            **common,
            "init_state_id": 1,
            "env_seed": 1,
            "requested_latency_ms": 0,
            "effective_latency_ms": 0,
            "success": False,
            "primitive_steps": 2,
            "episode_return": -1,
        },
        {
            **common,
            "init_state_id": 0,
            "env_seed": 0,
            "requested_latency_ms": 20,
            "effective_latency_ms": 20,
            "success": False,
            "primitive_steps": 3,
            "episode_return": -1,
        },
    ]
    cells = summarize(rows)
    assert cells[0]["success_rate"] == 0.5
    assert cells[0]["failure_budget_mean_control_steps"] == 8.5
    assert cells[0]["success_mean_control_steps"] == 5
    assert cells[1]["failure_budget_mean_control_steps"] == 12
    assert cells[1]["success_mean_control_steps"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + [rows[0]])


def test_runtime_source_is_owned_and_cannot_import_external_kinetix_packages():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src/robotics_bench/kinetix"
    for relative in (
        "flow_model.py",
        "native/kinetix/environment/env.py",
        "native/jax2d/engine.py",
    ):
        assert (root / relative).is_file(), relative
    assert len(list((root / "levels").glob("*.json"))) == 12
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(
                    item.name.split(".")[0] not in {"kinetix", "jax2d", "model"}
                    for item in node.names
                ), path
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] not in {
                    "kinetix",
                    "jax2d",
                    "model",
                }, path


def test_cli_preflight_uses_bundled_levels_without_model_runtime_dependencies(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    (tmp_path / "worlds_l_car_launch.pkl").write_bytes(b"preflight-only fixture")
    command = [
        sys.executable,
        str(root / "benchmarks/dynamic/kinetix/run.py"),
        "--policy-dir",
        str(tmp_path),
        "--levels",
        "car_launch",
        "--flow-steps",
        "1,5",
        "--latencies-ms",
        "0,21.938",
        "--output-dir",
        str(tmp_path / "run"),
        "--dry-run",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert len(plan["cells"]) == 4
    assert Path(plan["source"]).is_relative_to(root)
    assert plan["implementation"] == "repository_owned_model_environment_and_physics"
    assert not (tmp_path / "run").exists()
    rejected = subprocess.run(
        command + ["--source", "/external/code"], text=True, capture_output=True
    )
    assert rejected.returncode == 2
    assert "unrecognized arguments: --source" in rejected.stderr
