"""Constant condition reuse and finite failure-budget completion."""

import importlib
from pathlib import Path


def test_registered_constants_reuse_projection_but_other_inputs_do_not():
    path = (
        Path(__file__).parents[1]
        / "src/robotics_bench/optimizations/constant_projection.py"
    )
    assert path.exists(), "Condition projection cache is missing"
    api = importlib.import_module("robotics_bench.optimizations.constant_projection")

    class Value:
        def __init__(self, value):
            self.value = value

    constant, dynamic = Value(3), Value(4)
    calls = []

    def project(x):
        calls.append(x)
        return Value(x.value * 2)

    registered = [constant]
    cache = api.ConstantProjectionCache(
        project, lambda x: any(x is y for y in registered)
    )
    first = cache(constant)
    assert cache(constant) is first and len(calls) == 1
    assert cache(dynamic).value == 8
    dynamic.value = 5
    assert cache(dynamic).value == 10 and len(calls) == 3
    assert cache.retained_tensors() == (first,)
    registered.clear()
    constant.value = 6
    assert cache(constant).value == 12 and len(calls) == 4


def test_budget_completion_preserves_observed_time_and_only_fills_missing_work():
    from robotics_bench.profiling import task_report as module

    assert hasattr(module, "failure_budget_time"), "Budget completion helper is missing"
    completed = {
        "success": False,
        "primitive_steps": 10,
        "max_primitive_steps": 10,
        "inference_calls": 3,
        "paper_model_task_ms": 800,
    }
    assert (
        module.failure_budget_time(
            completed, inference_ms=20, action_steps=4, action_time_ms=50
        )
        == 800
    )
    early = {
        **completed,
        "primitive_steps": 5,
        "inference_calls": 2,
        "paper_model_task_ms": 400,
    }
    assert (
        module.failure_budget_time(
            early, inference_ms=20, action_steps=4, action_time_ms=50
        )
        == 670
    )
