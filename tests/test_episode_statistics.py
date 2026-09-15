"""Synthetic, CPU-only checks for pooled episode and failure-budget statistics."""

import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "tools/episode_statistics.py"


def summarize(records, **kwargs):
    assert MODULE.is_file(), "episode_statistics.py is not implemented"
    spec = importlib.util.spec_from_file_location("episode_statistics_tested", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize_episodes(records, **kwargs)


def record(task, initial, success, steps, seed=7):
    return {
        "task": task,
        "init_state_id": initial,
        "env_seed": seed,
        "success": success,
        "primitive_steps": steps,
    }


def mixed_records():
    return [
        record("A", 0, True, 10),
        record("A", 1, False, 4),
        record("B", 0, True, 20),
        record("B", 1, True, 30),
        record("B", 2, True, 40),
    ]


def test_pooled_statistics_and_failure_penalty_have_distinct_denominators():
    result = summarize(mixed_records(), max_steps=100)
    assert result["unit"] == "primitive_control_steps"
    assert result["aggregation"] == "episode_weighted"
    overall = result["overall"]
    assert {
        key: overall[key]
        for key in ("episodes", "successes", "failures", "success_rate")
    } == {
        "episodes": 5,
        "successes": 4,
        "failures": 1,
        "success_rate": 0.8,
    }
    assert overall["all_episodes"] == {
        "total_control_steps": 104,
        "mean_control_steps": 20.8,
    }
    assert overall["successful_episodes"] == {
        "total_control_steps": 100,
        "mean_control_steps": 25,
    }
    assert overall["failed_episodes"] == {
        "total_control_steps": 4,
        "mean_control_steps": 4,
    }
    assert overall["failure_penalized"] == {
        "status": "available",
        "total_control_steps": 200,
        "mean_control_steps": 40,
        "successful_contribution_mean": 20,
        "failed_contribution_mean": 20,
        "missing_budget_tasks": [],
    }
    a, b = result["tasks"]
    assert a["task"] == "A"
    assert a["trial_weight"] == 2 / 5
    assert a["success_weight"] == 1 / 4
    assert a["failure_penalized"]["mean_control_steps"] == 55
    assert b["trial_weight"] == 3 / 5
    assert b["success_weight"] == 3 / 4
    assert b["successful_episodes"]["mean_control_steps"] == 30
    # Pooled means differ from the unweighted mean of task means.
    assert overall["successful_episodes"]["mean_control_steps"] != (10 + 30) / 2


def test_no_success_keeps_null_success_mean_and_weight():
    result = summarize([record("A", 0, False, 4)], max_steps=20)
    assert result["overall"]["success_rate"] == 0
    assert result["overall"]["successful_episodes"] == {
        "total_control_steps": 0,
        "mean_control_steps": None,
    }
    assert result["tasks"][0]["success_weight"] is None
    assert result["overall"]["failure_penalized"]["successful_contribution_mean"] == 0
    assert result["overall"]["failure_penalized"]["mean_control_steps"] == 20


def test_all_success_needs_no_failure_budget_and_accepts_zero_steps():
    result = summarize([record("A", 0, True, 0), record("A", 1, True, 10)])
    stats = result["overall"]
    assert stats["failed_episodes"] == {
        "total_control_steps": 0,
        "mean_control_steps": None,
    }
    assert stats["failure_penalized"] == {
        "status": "available",
        "total_control_steps": 10,
        "mean_control_steps": 5,
        "successful_contribution_mean": 5,
        "failed_contribution_mean": 0,
        "missing_budget_tasks": [],
    }


def test_missing_failure_budget_does_not_hide_measured_steps():
    result = summarize(mixed_records())
    penalty = result["overall"]["failure_penalized"]
    assert penalty == {
        "status": "missing_failure_budget",
        "total_control_steps": None,
        "mean_control_steps": None,
        "successful_contribution_mean": 20,
        "failed_contribution_mean": None,
        "missing_budget_tasks": ["A"],
    }
    assert result["overall"]["failed_episodes"]["mean_control_steps"] == 4
    assert result["tasks"][1]["failure_penalized"]["status"] == "available"


def test_task_budgets_override_global_and_pool_per_failed_episode():
    rows = [
        record("A", 0, True, 10),
        record("A", 1, False, 3),
        record("B", 0, False, 5),
    ]
    result = summarize(rows, max_steps=100, task_max_steps={"A": 20, "B": 50})
    penalty = result["overall"]["failure_penalized"]
    assert penalty["total_control_steps"] == 80
    assert penalty["mean_control_steps"] == pytest.approx(80 / 3)
    assert penalty["failed_contribution_mean"] == pytest.approx(70 / 3)
    assert result["tasks"][1]["success_weight"] == 0


def test_partial_task_budgets_report_only_unbudgeted_failed_tasks():
    rows = [record("C", 0, True, 8), record("B", 0, False, 5), record("A", 0, False, 3)]
    result = summarize(rows, task_max_steps={"A": 20})
    assert result["overall"]["failure_penalized"]["missing_budget_tasks"] == ["B"]
    assert [task["task"] for task in result["tasks"]] == ["A", "B", "C"]


def test_global_budget_is_fallback_for_tasks_without_override():
    rows = [record("A", 0, False, 3), record("B", 0, False, 5)]
    assert (
        summarize(rows, max_steps=100, task_max_steps={"A": 20})["overall"][
            "failure_penalized"
        ]["total_control_steps"]
        == 120
    )


def test_same_task_and_initial_state_with_different_seeds_are_distinct_trials():
    result = summarize(
        [record("A", 0, True, 2, seed=None), record("A", 0, True, 4, seed=8)]
    )
    assert result["overall"]["episodes"] == 2


def test_duplicate_episode_identity_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        summarize([record("A", 0, True, 2), record("A", 0, False, 4)])


@pytest.mark.parametrize("bad", [[], [None], ["not a record"]])
def test_empty_or_nonobject_records_are_rejected(bad):
    with pytest.raises(ValueError):
        summarize(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", ""),
        ("task", "  "),
        ("task", 1),
        ("init_state_id", -1),
        ("init_state_id", 0.5),
        ("init_state_id", True),
        ("env_seed", -1),
        ("env_seed", 1.5),
        ("env_seed", False),
        ("success", 1),
        ("success", "true"),
        ("success", None),
        ("primitive_steps", -1),
        ("primitive_steps", 1.5),
        ("primitive_steps", True),
    ],
)
def test_invalid_record_fields_are_rejected(field, value):
    row = record("A", 0, True, 2)
    row[field] = value
    with pytest.raises(ValueError, match=field):
        summarize([row])


@pytest.mark.parametrize(
    "field", ["task", "init_state_id", "success", "primitive_steps"]
)
def test_missing_required_record_fields_are_rejected(field):
    row = record("A", 0, True, 2)
    del row[field]
    with pytest.raises(ValueError, match=field):
        summarize([row])


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5])
def test_invalid_global_budgets_are_rejected_even_for_all_success(budget):
    with pytest.raises(ValueError, match="max_steps"):
        summarize([record("A", 0, True, 0)], max_steps=budget)


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5, None])
def test_invalid_task_budgets_are_rejected(budget):
    with pytest.raises(ValueError, match="task_max_steps"):
        summarize([record("A", 0, True, 0)], task_max_steps={"A": budget})


def test_unknown_task_budget_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        summarize([record("A", 0, True, 1)], task_max_steps={"typo": 20})


@pytest.mark.parametrize("success", [True, False])
def test_observed_steps_must_not_exceed_effective_budget(success):
    with pytest.raises(ValueError, match="budget"):
        summarize(
            [record("A", 0, success, 11)], max_steps=100, task_max_steps={"A": 10}
        )


def test_equal_budget_is_allowed_and_inputs_are_not_mutated():
    rows = [record("A", 0, True, 10)]
    budgets = {"A": 10}
    before = [dict(row) for row in rows]
    result = summarize(rows, task_max_steps=budgets)
    assert result["overall"]["all_episodes"]["total_control_steps"] == 10
    assert rows == before
    assert budgets == {"A": 10}
