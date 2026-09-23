"""Parallel evaluation cannot duplicate, drop, or hide changed episodes."""

import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest


def module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/dynamic/kinetix/batched_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("batch_eval_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_seed_groups_cover_the_same_prespecified_work_once():
    for size in (1, 8, 16, 32):
        groups = module().seed_groups(list(range(128)), size)
        assert [seed for group in groups for seed in group] == list(range(128))
        assert all(len(group) == size for group in groups)
    with pytest.raises(ValueError):
        module().seed_groups([0, 0], 2)
    with pytest.raises(ValueError):
        module().seed_groups([0, 1, 2], 2)


def test_coverage_rejects_missing_or_duplicate_results():
    index = module().episode_index
    rows = [{"result": {"env_seed": 0}}, {"result": {"env_seed": 1}}]
    assert set(index(rows, [0, 1])) == {0, 1}
    with pytest.raises(ValueError, match="coverage"):
        index(rows[:1], [0, 1])
    with pytest.raises(ValueError, match="Duplicate"):
        index([rows[0], rows[0]], [0, 1])


def test_window_guard_protects_completion_not_only_enqueue(monkeypatch):
    candidate = module()
    counters = {"active": 0, "peak": 0}
    counter_lock = threading.Lock()

    class PendingWindow:
        complete = False

        def wait(self):
            if not self.complete:
                # The executable has returned, but its device work is pending.
                time.sleep(0.01)
                with counter_lock:
                    counters["active"] -= 1
                self.complete = True
            return self

    def advance(*args):
        with counter_lock:
            counters["active"] += 1
            counters["peak"] = max(counters["peak"], counters["active"])
        return PendingWindow()

    def episode(policy, env, window, *, seed, **kwargs):
        window(seed).wait()
        return {"result": {"env_seed": seed}}

    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(block_until_ready=lambda value: value.wait()),
    )
    monkeypatch.setattr(candidate, "resident_episode", episode)
    evaluator = candidate.BatchedEvaluator.__new__(candidate.BatchedEvaluator)
    evaluator.env = SimpleNamespace()
    evaluator.policy = SimpleNamespace(_compiled={5: None})
    evaluator.flow_steps, evaluator.advance = 5, advance
    evaluator.threaded(list(range(16)), 4)
    assert counters["peak"] > 1
    counters.update(active=0, peak=0)
    rows = evaluator.threaded(list(range(16)), 4, serialize_windows=True)
    assert counters == {"active": 0, "peak": 1}
    assert len(rows["episodes"]) == 16
