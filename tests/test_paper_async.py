"""Paper overlap timing is a CPU estimate with explicit missing observations."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "benchmarks/static/pi05_libero/paper_async.py"
SPEEDUPS = ROOT / "tools/compare_speedups.py"


def load_file(path, name):
    assert path.is_file(), f"{path.name} is not implemented"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def api():
    return load_file(CONTRACT, "tested_paper_async")


def options(**changes):
    return {
        "schedule": "paper_async",
        "overlap_actions": 2,
        "n_action_steps": 5,
        "paper_action_time_ms": 20,
        "paper_inference_time_ms": 100,
        "paper_inference_time_source": "synthetic hand calculation",
        "delay_state_with_observation": True,
        **changes,
    }


def test_hand_calculation_and_explicit_time_semantics():
    result = api().paper_async_contract(options())
    assert result["n_prime"] == 2
    assert result["n_actions"] == 5
    assert result["implementation"] == "history_observation"
    assert result["history_unit"] == "primitive_control_steps"
    assert result["state_policy"] == "same_snapshot"
    assert result["history_fallback"] == "current_until_available"
    assert "2606.28529v2" in result["paper_reference"]
    timing = result["timing"]
    assert timing["status"] == "estimated"
    assert timing["clock_domain"] == "paper_model"
    assert timing["hidden_inference_time_ms"] == 40
    assert timing["residual_inference_time_ms"] == 60
    assert timing["cycle_time_ms"] == 160
    assert timing["inference_time_source"] == "synthetic hand calculation"
    assert "not_host_duration_or_observation_age" in timing["interpretation"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "inference,overlap,hidden,residual,cycle",
    [(100, 0, 0, 100, 200), (100, 5, 100, 0, 100), (30, 2, 30, 0, 100)],
)
def test_zero_and_saturated_overlap(inference, overlap, hidden, residual, cycle):
    result = api().paper_async_contract(
        options(paper_inference_time_ms=inference, overlap_actions=overlap)
    )["timing"]
    assert result["hidden_inference_time_ms"] == hidden
    assert result["residual_inference_time_ms"] == residual
    assert result["cycle_time_ms"] == cycle


def test_sync_metadata_and_current_state_choice_are_explicit():
    result = api().paper_async_contract(
        options(schedule="sync", overlap_actions=0, delay_state_with_observation=False)
    )
    assert result["schedule"] == "sync"
    assert result["implementation"] == "sync"
    assert result["n_prime"] == 0
    assert result["state_policy"] == "current_state"
    assert result["timing"]["cycle_time_ms"] == 200


def test_unknown_inference_is_not_inferred_from_episode_steps():
    supplied = options(
        paper_inference_time_ms=None,
        paper_inference_time_source=None,
        episode_steps_mean=157,
        episode_time_mean_ms=12345,
    )
    result = api().paper_async_contract(supplied)
    assert result["timing"]["status"] == "requires_inference_profile"
    assert result["timing"]["inference_time_ms"] is None
    for key in (
        "hidden_inference_time_ms",
        "residual_inference_time_ms",
        "cycle_time_ms",
    ):
        assert result["timing"][key] is None
    assert "task_speedup" not in json.dumps(result)
    assert "successful_chunk_count_mean" not in json.dumps(result)
    assert supplied["episode_steps_mean"] == 157


@pytest.mark.parametrize(
    "changes",
    [
        {"schedule": "real_async"},
        {"schedule": None},
        {"schedule": "sync", "overlap_actions": 1},
        {"overlap_actions": -1},
        {"overlap_actions": 6},
        {"overlap_actions": True},
        {"overlap_actions": 1.5},
        {"n_action_steps": 0},
        {"n_action_steps": True},
        {"paper_action_time_ms": 0},
        {"paper_action_time_ms": -1},
        {"paper_action_time_ms": float("nan")},
        {"paper_action_time_ms": float("inf")},
        {"paper_action_time_ms": False},
        {"paper_action_time_ms": 1e308, "paper_inference_time_ms": None},
        {"paper_inference_time_ms": 0},
        {"paper_inference_time_ms": -1},
        {"paper_inference_time_ms": float("nan")},
        {"paper_inference_time_ms": float("inf")},
        {"paper_inference_time_ms": True},
        {"paper_inference_time_source": None},
        {"paper_inference_time_source": " "},
        {"paper_inference_time_source": False},
        {"delay_state_with_observation": 1},
    ],
)
def test_rejects_invalid_contract_options(changes):
    with pytest.raises(ValueError):
        api().paper_async_contract(options(**changes))


def test_required_inputs_are_not_silently_defaulted():
    supplied = options()
    del supplied["n_action_steps"]
    with pytest.raises(ValueError, match="n_action_steps"):
        api().paper_async_contract(supplied)


def test_contract_requires_mapping():
    with pytest.raises(ValueError):
        api().paper_async_contract([])


def test_shared_chunk_helper_has_same_hand_calculation():
    speedups = load_file(SPEEDUPS, "tested_speedups_for_paper")
    assert hasattr(speedups, "chunk_timing"), "shared chunk_timing is not implemented"
    assert speedups.chunk_timing(100, 5, 20, 2) == {
        "hidden_inference_time_ms": 40,
        "residual_inference_time_ms": 60,
        "cycle_time_ms": 160,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 5, 20, 2),
        (float("nan"), 5, 20, 2),
        (True, 5, 20, 2),
        (100, 0, 20, 0),
        (100, True, 20, 0),
        (100, 5, 0, 0),
        (100, 5, float("inf"), 0),
        (100, 5, True, 0),
        (100, 5, 20, -1),
        (100, 5, 20, 6),
        (100, 5, 20, False),
        (100, 5, 20, 1.5),
        (100, 5, 1e308, 0),
    ],
)
def test_shared_chunk_helper_validates_direct_callers(arguments):
    speedups = load_file(SPEEDUPS, "tested_speedups_validation")
    assert hasattr(speedups, "chunk_timing"), "shared chunk_timing is not implemented"
    with pytest.raises(ValueError):
        speedups.chunk_timing(*arguments)


def test_shared_helper_uses_repository_path_not_external_import(monkeypatch, tmp_path):
    fake = ModuleType("compare_speedups")
    fake.chunk_timing = lambda *args: {"cycle_time_ms": -1}
    monkeypatch.setitem(sys.modules, "compare_speedups", fake)
    monkeypatch.setitem(sys.modules, "tools.compare_speedups", fake)
    (tmp_path / "compare_speedups.py").write_text(
        "raise AssertionError('wrong source')"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    result = api().paper_async_contract(options())
    assert result["timing"]["cycle_time_ms"] == 160
