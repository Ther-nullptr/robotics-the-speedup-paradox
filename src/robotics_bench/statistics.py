"""Pooled episode statistics in primitive control steps, using only the CPU.

Failure penalties replace each failed episode's observed steps with its declared
maximum budget. They remain separate from observed totals and are not elapsed
time, model inference counts, or a measured speedup.
"""

from collections.abc import Mapping


def _integer(value, name, *, positive=False):
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _validated_records(records):
    try:
        records = list(records)
    except TypeError as exc:
        raise ValueError("records must be an iterable of episode objects") from exc
    if not records:
        raise ValueError("records must not be empty")
    seen = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ValueError(f"record {index} must be an object")
        task = row.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"record {index}: task must be a nonempty string")
        initial = _integer(row.get("init_state_id"), f"record {index}: init_state_id")
        seed = row.get("env_seed")
        if seed is not None:
            _integer(seed, f"record {index}: env_seed")
        if type(row.get("success")) is not bool:
            raise ValueError(f"record {index}: success must be boolean")
        _integer(row.get("primitive_steps"), f"record {index}: primitive_steps")
        identity = task, initial, seed
        if identity in seen:
            raise ValueError(f"duplicate episode identity: {identity}")
        seen.add(identity)
    return records


def _budgets(records, max_steps, task_max_steps):
    if max_steps is not None:
        _integer(max_steps, "max_steps", positive=True)
    if task_max_steps is None:
        task_max_steps = {}
    if not isinstance(task_max_steps, Mapping):
        raise ValueError("task_max_steps must map task names to positive integers")
    for task, budget in task_max_steps.items():
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task_max_steps keys must be nonempty task names")
        _integer(budget, f"task_max_steps[{task!r}]", positive=True)
    tasks = {row["task"] for row in records}
    unknown = set(task_max_steps) - tasks
    if unknown:
        raise ValueError(f"task_max_steps contains unknown tasks: {sorted(unknown)}")
    budgets = {task: task_max_steps.get(task, max_steps) for task in tasks}
    for row in records:
        budget = budgets[row["task"]]
        if budget is not None and row["primitive_steps"] > budget:
            raise ValueError(
                f"observed primitive_steps exceeds budget for task {row['task']!r}, "
                f"init_state_id={row['init_state_id']}: {row['primitive_steps']} > {budget}"
            )
    return budgets


def _step_stats(rows):
    total = sum(row["primitive_steps"] for row in rows)
    return {
        "total_control_steps": total,
        "mean_control_steps": total / len(rows) if rows else None,
    }


def _stats(rows, budgets):
    successful = [row for row in rows if row["success"]]
    failed = [row for row in rows if not row["success"]]
    successful_stats = _step_stats(successful)
    success_steps = successful_stats["total_control_steps"]
    missing_tasks = sorted(
        {row["task"] for row in failed if budgets[row["task"]] is None}
    )
    failure_budget_total = (
        None if missing_tasks else sum(budgets[row["task"]] for row in failed)
    )
    penalized_total = None if missing_tasks else success_steps + failure_budget_total
    count = len(rows)
    return {
        "episodes": count,
        "successes": len(successful),
        "failures": len(failed),
        "success_rate": len(successful) / count,
        "all_episodes": _step_stats(rows),
        "successful_episodes": successful_stats,
        "failed_episodes": _step_stats(failed),
        "failure_penalized": {
            "status": "missing_failure_budget" if missing_tasks else "available",
            "total_control_steps": penalized_total,
            "mean_control_steps": penalized_total / count
            if penalized_total is not None
            else None,
            "successful_contribution_mean": success_steps / count,
            "failed_contribution_mean": failure_budget_total / count
            if failure_budget_total is not None
            else None,
            "missing_budget_tasks": missing_tasks,
        },
    }


def summarize_episodes(records, *, max_steps=None, task_max_steps=None):
    """Validate records and return episode-weighted control-step statistics.

    Records require task, init_state_id, success, and primitive_steps. An omitted
    env_seed is treated as None; identity is (task, init_state_id, env_seed).
    task_max_steps overrides max_steps for that task. Every supplied budget must
    cover all observed steps of its task, including successful episodes.

    Without a budget for any failed episode, only the failure-penalized total,
    mean, and failed contribution are unavailable. An all-success group needs no
    budget. Successful means divide by success count; contribution means divide
    by all episodes, so their sum is the failure-penalized mean when available.
    """
    records = _validated_records(records)
    budgets = _budgets(records, max_steps, task_max_steps)
    overall = _stats(records, budgets)
    tasks = []
    for task in sorted(budgets):
        rows = [row for row in records if row["task"] == task]
        stats = _stats(rows, budgets)
        tasks.append(
            {
                "task": task,
                "trial_weight": len(rows) / len(records),
                "success_weight": stats["successes"] / overall["successes"]
                if overall["successes"]
                else None,
                **stats,
            }
        )
    return {
        "unit": "primitive_control_steps",
        "aggregation": "episode_weighted",
        "overall": overall,
        "tasks": tasks,
    }
