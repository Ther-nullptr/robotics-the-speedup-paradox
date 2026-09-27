"""Quality reports must retain the complete paired matrix and uncertainty."""

import importlib.util
from pathlib import Path

import pytest


def module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/dynamic/kinetix/analyze_flow_quality.py"
    )
    spec = importlib.util.spec_from_file_location("flow_analysis_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def trials():
    rows = []
    for task, successes in (
        ("a", {1: [True, False], 5: [True, True]}),
        ("b", {1: [False, False], 5: [False, True]}),
    ):
        for flow, outcomes in successes.items():
            for seed, success in enumerate(outcomes):
                steps = 8 if success else 256
                rows.append(
                    dict(
                        task=task,
                        flow_steps=flow,
                        env_seed=seed,
                        init_state_id=seed,
                        success=success,
                        primitive_steps=steps,
                        max_primitive_steps=256,
                        requested_latency_ms=0,
                        effective_latency_ms=0,
                        delay_mapping="native-blend",
                        initial_observation_sha256=f"{task}-{seed}",
                        episode_return=float(success),
                        inference_calls=(steps + 3) // 4,
                    )
                )
    return rows


def test_macro_and_paired_differences_use_all_tasks_and_matching_seeds():
    report = module().analyze(
        trials(),
        levels=["a", "b"],
        flows=[1, 5],
        seeds=[0, 1],
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    assert report["complete"]
    assert [r["success_rate"] for r in report["macro"]] == [0.25, 0.75]
    assert report["macro"][0]["episodes"] == 4
    assert all(r["wins"] == 0 and r["losses"] == 1 for r in report["paired"])
    assert report["macro"][0]["paired_delta_vs_reference"] == -0.5


def test_fine_aliases_share_one_cell_without_mutating_raw_records():
    rows = trials()
    for row in rows[::2]:
        row["delay_mapping"] = "fine"
    report = module().analyze(
        rows,
        levels=["a", "b"],
        flows=[1, 5],
        seeds=[0, 1],
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    assert report["complete"]
    assert len(report["cells"]) == 4
    assert all(row["episodes"] == 2 for row in report["cells"])
    assert {row["delay_mapping"] for row in rows} == {"fine", "native-blend"}


def test_duplicate_or_incomplete_coverage_is_not_silently_reported_as_complete():
    analyze = module().analyze
    settings = dict(
        levels=["a", "b"],
        flows=[1, 5],
        seeds=[0, 1],
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(trials() + [trials()[0]], **settings)
    with pytest.raises(ValueError, match="Incomplete"):
        analyze(trials()[:-1], **settings)
    report = analyze(trials()[:-1], allow_partial=True, **settings)
    assert not report["complete"]
    assert report["macro"] == []


def test_incompatible_initial_state_or_delay_is_rejected():
    analyze = module().analyze
    settings = dict(
        levels=["a", "b"],
        flows=[1, 5],
        seeds=[0, 1],
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    rows = trials()
    rows[2]["initial_observation_sha256"] = "different"
    with pytest.raises(ValueError, match="initial"):
        analyze(rows, **settings)
    rows = trials()
    rows[0]["requested_latency_ms"] = 1
    with pytest.raises(ValueError, match="zero-delay"):
        analyze(rows, **settings)


def test_macro_bootstrap_preserves_shared_seed_dependence_between_tasks():
    rows = trials()
    for row in rows:
        row["success"] = (row["env_seed"] == 0) == (row["task"] == "a")
    report = module().analyze(
        rows,
        levels=["a", "b"],
        flows=[1, 5],
        seeds=[0, 1],
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    # Each seed block has one success across two tasks. Independently drawing
    # task seeds would create variation that the paired experiment did not have.
    for row in report["macro"]:
        assert row["ci95_low"] == row["ci95_high"] == 0.5


def test_full_ledgers_do_not_hide_a_failed_or_unfinished_run():
    report = module().analyze(
        trials(),
        levels=["a", "b"],
        flows=[1, 5],
        seeds=[0, 1],
        bootstrap_samples=100,
        run_completed=False,
        allow_partial=True,
    )
    assert report["observed_episodes"] == report["expected_episodes"]
    assert not report["complete"]
    assert report["macro"] == report["paired"] == []


@pytest.mark.parametrize("mapping", ["fine", "native-blend"])
def test_completed_manifest_requires_runtime_evidence_but_loading_can_be_partial(
    tmp_path,
    mapping,
):
    import json

    plan = {"status": "completed", "levels": ["a"]}
    (tmp_path / "sweep-manifest.json").write_text(json.dumps(plan))
    (tmp_path / "a").mkdir()
    path = tmp_path / "a/case-manifest.json"
    manifest = {
        "status": "completed",
        "tasks": {
            "a": {
                "native_env_params": {
                    "dt": 1 / 60,
                    "max_timesteps": 256,
                    "baumgarte_coefficient_collision": 0.2,
                },
                "native_static_env_params": {
                    "frame_skip": 2,
                    "num_solver_iterations": 10,
                },
                "runtime": {"policy": {"parameter_dtypes": ["float32"]}},
            }
        },
        "runtime": {
            "python": "3.11",
            "packages": {
                "jax": "0.4.35",
                "jaxlib": "0.4.34",
                "numpy": "1.26.4",
                "flax": "0.10.2",
            },
        },
        "source_sha256": {"model.py": "source-hash"},
        "entry_code_sha256": {"run.py": "entry-hash"},
        "gpu": 0,
        "repository_commit": "commit",
        "execute_horizon": 4,
        "action_noise_std": 0.1,
        "mapping": mapping,
    }
    path.write_text(json.dumps(manifest))
    load = module().load_sweep
    assert len(load(tmp_path, False)[2]) == 1
    del manifest["tasks"]["a"]["runtime"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="runtime evidence"):
        load(tmp_path, False)
    manifest["status"] = "running"
    plan["status"] = "running"
    (tmp_path / "sweep-manifest.json").write_text(json.dumps(plan))
    path.write_text(json.dumps(manifest))
    assert load(tmp_path, True)[1:] == ([], {})
