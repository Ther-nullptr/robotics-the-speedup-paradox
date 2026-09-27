"""CPU checks for manually launched static evaluation matrices."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver():
    path = ROOT / "benchmarks/static/full_evaluation.py"
    assert path.is_file(), "The full evaluation entry is missing"
    spec = importlib.util.spec_from_file_location("_full_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(driver, tmp_path, case="cosmos_libero", *extra):
    args = driver.parser().parse_args(
        [
            "--case",
            case,
            "--gpu",
            "0",
            "--output-dir",
            str(tmp_path / "matrix"),
            *extra,
        ]
    )
    return driver.build_plan(args)


def value(command, flag):
    return command[command.index(flag) + 1]


def test_cosmos_recipes_keep_matched_shared_optimizations(driver, tmp_path):
    matrix = plan(driver, tmp_path)
    assert len(matrix["cells"]) == 8
    cells = {c["id"]: c for c in matrix["cells"]}
    assert (
        value(cells["libero_object/original-bf16/sync"]["command"], "--episodes")
        == "500"
    )
    assert "--enable" not in cells["libero_object/original-bf16/sync"]["command"]
    for recipe in ("optimized-bf16", "int8", "int4"):
        command = cells[f"libero_object/{recipe}/paper_async"]["command"]
        assert "vae_condition_prefix" in command
        assert value(command, "--overlap-actions") == "2"
        assert value(command, "--n-action-steps") == "16"


def test_pi05_keeps_its_own_protocol_defaults(driver, tmp_path):
    matrix = plan(driver, tmp_path, "pi05_libero")
    assert len(matrix["cells"]) == 2
    command = matrix["cells"][1]["command"]
    for flag, expected in [
        ("--seed", "42"),
        ("--batch-size", "10"),
        ("--n-action-steps", "5"),
        ("--num-inference-steps", "10"),
    ]:
        assert value(command, flag) == expected
    assert "--enable" not in command
    assert "--env-seed" not in command
    assert value(command, "--schedule") == "paper_async"
    with pytest.raises(ValueError, match="PI0.5"):
        plan(driver, tmp_path, "pi05_libero", "--recipes", "int4")


def test_robocasa_references_original_sync_for_each_task(driver, tmp_path):
    matrix = plan(
        driver, tmp_path, "cosmos_robocasa", "--tasks", "TurnOffMicrowave,OpenDrawer"
    )
    assert len(matrix["cells"]) == 16
    for cell in matrix["cells"]:
        assert value(cell["command"], "--episodes") == "50"
        if cell["recipe"] == "original-bf16" and cell["schedule"] == "sync":
            assert cell["reference"] is None
        else:
            assert cell["reference"] == f"{cell['task']}/original-bf16/sync"


def complete_cell(driver, root, cell, rows, budget=280):
    run = root / cell["id"] / "attempt-001"
    run.mkdir(parents=True)
    manifest = {
        "status": "completed",
        "case_fingerprint": "matched",
        "identity": {"case": {"episodes": len(rows)}},
    }
    coverage = {
        "status": "passed",
        "expected_episodes": len(rows),
        "completed_episodes": len(rows),
        "successes": sum(r["success"] for r in rows),
        "max_primitive_steps": budget,
    }
    for name, data in [("case-manifest.json", manifest), ("coverage.json", coverage)]:
        (run / name).write_text(json.dumps(data))
    (run / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {
        "status": "completed",
        "output_dir": str(run),
        "case_fingerprint": "matched",
    }


def test_summary_uses_budget_penalties_and_rejects_partial(driver, tmp_path):
    matrix = plan(
        driver, tmp_path, "pi05_libero", "--episodes", "2", "--batch-size", "1"
    )
    rows = [
        dict(
            task="task", init_state_id=0, env_seed=42, success=True, primitive_steps=10
        ),
        dict(
            task="task", init_state_id=1, env_seed=42, success=False, primitive_steps=20
        ),
    ]
    state = {"plan": matrix, "cells": {}}
    for cell in matrix["cells"]:
        state["cells"][cell["id"]] = complete_cell(driver, tmp_path, cell, rows)
    report = driver.summarize(state)
    assert report["complete"] is True
    row = report["rows"][0]
    assert row["success_rate"] == 0.5
    assert row["failure_budget_total_steps"] == 290
    assert row["failure_budget_mean_steps"] == 145
    assert row["success_mean_steps"] == 10
    state["cells"][matrix["cells"][1]["id"]]["status"] = "failed"
    report = driver.summarize(state)
    assert report["complete"] is False
    assert report["rows"][1]["success_rate"] is None


def test_summary_rejects_mismatched_episode_sets(driver, tmp_path):
    matrix = plan(
        driver, tmp_path, "pi05_libero", "--episodes", "1", "--batch-size", "1"
    )
    state = {"plan": matrix, "cells": {}}
    for i, cell in enumerate(matrix["cells"]):
        rows = [
            dict(
                task="task",
                init_state_id=i,
                env_seed=42,
                success=True,
                primitive_steps=10,
            )
        ]
        state["cells"][cell["id"]] = complete_cell(driver, tmp_path, cell, rows)
    with pytest.raises(ValueError, match="episode"):
        driver.summarize(state)


def test_resume_rejects_changed_plan_and_keeps_failed_attempt(driver, tmp_path):
    matrix = plan(driver, tmp_path)
    state = driver.open_state(matrix, resume=False)
    cell = matrix["cells"][0]
    run = Path(matrix["output_dir"]) / cell["id"] / "attempt-001"
    run.mkdir(parents=True)
    (run / "keep.txt").write_text("failed evidence")
    state["cells"][cell["id"]] = {"status": "failed", "output_dir": str(run)}
    driver.save_state(state)
    resumed = driver.open_state(matrix, resume=True)
    assert resumed["cells"][cell["id"]]["status"] == "failed"
    assert driver.next_attempt(Path(matrix["output_dir"]), cell).name == "attempt-002"
    matrix["cells"][0]["command"].append("changed")
    with pytest.raises(ValueError, match="plan"):
        driver.open_state(matrix, resume=True)
    assert (run / "keep.txt").read_text() == "failed evidence"


def test_next_attempt_skips_early_launcher_failures(driver, tmp_path):
    cell = {"id": "task/recipe/sync"}
    directory = tmp_path / cell["id"]
    directory.mkdir(parents=True)
    (directory / "attempt-001.launcher.log").write_text(
        "Python failed before creating output"
    )
    assert driver.next_attempt(tmp_path, cell).name == "attempt-002"


def test_matrix_lock_rejects_a_second_driver(driver, tmp_path):
    assert hasattr(driver, "matrix_lock"), "Matrix ownership lock is missing"
    root = tmp_path / "matrix"
    with driver.matrix_lock(root):
        with pytest.raises(ValueError, match="already running"):
            with driver.matrix_lock(root):
                pass
    with driver.matrix_lock(root):
        pass


def test_summary_rejects_budget_changes_between_variants(driver, tmp_path):
    matrix = plan(
        driver, tmp_path, "pi05_libero", "--episodes", "1", "--batch-size", "1"
    )
    state = {"plan": matrix, "cells": {}}
    for i, cell in enumerate(matrix["cells"]):
        rows = [
            dict(
                task="task",
                init_state_id=0,
                env_seed=42,
                success=True,
                primitive_steps=10,
            )
        ]
        state["cells"][cell["id"]] = complete_cell(
            driver, tmp_path, cell, rows, budget=280 + i
        )
    with pytest.raises(ValueError, match="budget"):
        driver.summarize(state)


def test_failed_child_stops_matrix_and_resume_uses_new_attempt(driver, tmp_path):
    import sys

    stub = tmp_path / "case_stub.py"
    stub.write_text("""import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
root = a.output_dir
root.mkdir(parents=True)
if 'sync' == root.parent.name and root.name == 'attempt-001':
    raise SystemExit(9)
row = dict(task='task', init_state_id=0, env_seed=42, success=True, primitive_steps=12)
identity = {'frozen': 'yes', 'case': {'episodes': 1}}
(root/'case-manifest.json').write_text(json.dumps(dict(status='completed', case_fingerprint='matched', identity=identity)))
(root/'coverage.json').write_text(json.dumps(dict(status='passed', expected_episodes=1, completed_episodes=1, successes=1, max_primitive_steps=280)))
(root/'episodes.jsonl').write_text(json.dumps(row)+'\\n')
""")
    matrix = plan(
        driver, tmp_path, "pi05_libero", "--episodes", "1", "--batch-size", "1"
    )
    for cell in matrix["cells"]:
        cell["command"] = [sys.executable, str(stub)]
    state = driver.open_state(matrix, resume=False)
    state["resource_identity"] = {"frozen": "yes"}
    with pytest.raises(RuntimeError, match="code 9"):
        driver.execute(state)
    assert len(state["cells"]) == 1
    assert state["status"] == "failed"
    resumed = driver.open_state(matrix, resume=True)
    driver.execute(resumed)
    assert resumed["status"] == "completed"
    first = resumed["cells"][matrix["cells"][0]["id"]]
    assert Path(first["output_dir"]).name == "attempt-002"
    failed = Path(first["output_dir"]).with_name("attempt-001")
    assert failed.is_dir()
    # Re-running a completed matrix launches nothing and revalidates its results.
    stub.unlink()
    driver.execute(resumed)
    assert driver.summarize(resumed)["complete"] is True
