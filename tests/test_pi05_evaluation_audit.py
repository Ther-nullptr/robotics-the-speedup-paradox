"""Coverage must follow actual environment resets, including terminal padding."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/static/pi05_libero/evaluation_audit.py"
)


def api():
    assert SOURCE.exists(), "evaluation audit not implemented"
    spec = importlib.util.spec_from_file_location("test_evaluation_audit", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scalar_seeds_reach_each_environment_without_extra_resets():
    class Vector:
        def __init__(self):
            self.calls = []

        def __len__(self):
            return 2

        def reset(self, id, seed):
            self.calls.append((id, seed))
            return [{"state": id}], [{"seed": seed}]

    env = Vector()
    obs, infos = api().SeededResetVector(env).reset(seed=[42, 43])
    assert env.calls == [(0, 42), (1, 43)]
    assert obs == [{"state": 0}, {"state": 1}]
    assert infos == [{"seed": 42}, {"seed": 43}]


def test_ledger_records_actual_ids_and_stops_success_at_first_done(tmp_path):
    ledger = api().EpisodeLedger(tmp_path, 2)
    ledger.begin([[("task", "instruction", [0, 1])]])
    for index in range(2):
        ledger.record(
            0,
            index,
            ["task"],
            [index],
            {
                "done": [[False, True, True]],
                "success": [[False, index == 0, True]],
            },
            [42],
        )
    result = {
        "overall": {"pc_successes": 50.0, "avg_episode_length": 2.0},
        "task": {"pc_successes": 50.0, "avg_episode_length": 2.0},
    }
    ledger.finish(result)
    rows = [
        json.loads(line)
        for line in (tmp_path / "episodes.jsonl").read_text().splitlines()
    ]
    assert [row["init_state_id"] for row in rows] == [0, 1]
    assert [row["success"] for row in rows] == [True, False]
    assert [row["primitive_steps"] for row in rows] == [2, 2]
    assert all(row["env_seed"] == 42 for row in rows)
    coverage = json.loads((tmp_path / "coverage.json").read_text())
    assert coverage["status"] == "passed"
    assert coverage["completed_episodes"] == 2
    assert coverage["successes"] == 1


@pytest.mark.parametrize("names,ids", [(["task"], [1]), (["other"], [0])])
def test_ledger_rejects_skipped_initial_states_or_wrong_task(tmp_path, names, ids):
    ledger = api().EpisodeLedger(tmp_path, 1)
    ledger.begin([[("task", "instruction", [0])]])
    with pytest.raises(RuntimeError, match="identity"):
        ledger.record(0, 0, names, ids, {"done": [[True]], "success": [[True]]}, [42])


def test_missing_episodes_cannot_pass(tmp_path):
    ledger = api().EpisodeLedger(tmp_path, 2)
    ledger.begin([[("task", "instruction", [0, 1])]])
    ledger.record(0, 0, ["task"], [0], {"done": [[True]], "success": [[True]]}, [42])
    with pytest.raises(RuntimeError, match="coverage"):
        ledger.finish({})


def test_duplicate_episode_cannot_pass(tmp_path):
    ledger = api().EpisodeLedger(tmp_path, 1)
    ledger.begin([[("task", "instruction", [0])]])
    args = (0, 0, ["task"], [0], {"done": [[True]], "success": [[True]]}, [42])
    ledger.record(*args)
    with pytest.raises(RuntimeError, match="duplicate"):
        ledger.record(*args)


def test_summary_cannot_disagree_with_episode_results(tmp_path):
    ledger = api().EpisodeLedger(tmp_path, 1)
    ledger.begin([[("task", "instruction", [0])]])
    ledger.record(0, 0, ["task"], [0], {"done": [[True]], "success": [[False]]}, [42])
    with pytest.raises(RuntimeError, match="summary"):
        ledger.finish({"overall": {"pc_successes": 100, "avg_episode_length": 1}})


def test_capped_budget_is_not_silently_called_complete(tmp_path):
    ledger = api().EpisodeLedger(tmp_path, 500)
    with pytest.raises(RuntimeError, match="budget"):
        ledger.begin([[("task", "instruction", [0])]])


def test_context_audits_batches_fixes_descriptions_and_restores_hooks(tmp_path):
    class Vector:
        def __init__(self, name):
            self.name = name
            self.closed = False
            self.resets = 0

        def __len__(self):
            return 1

        def reset(self, id, seed):
            assert id == 0 and seed == 42
            self.resets += 1
            return [{}], [{}]

        def get_env_attr(self, key):
            return [self.name if key == "evaluation_task_name" else 0]

        def close(self):
            self.closed = True

    envs = [Vector("a"), Vector("b")]

    def rollout(env, policy, task_description=None, seeds=None):
        env.reset(seed=seeds)
        assert task_description == [env.name + " instruction"]
        return {"done": [[True]], "success": [[True]]}

    def policy(env_cfg, policy, schedule):
        for env in envs:
            evaluator.rollout(
                env, None, task_description=["wrong accumulated list"], seeds=[42]
            )
        return {
            key: {"pc_successes": 100, "avg_episode_length": 1}
            for key in ("overall", "a", "b")
        }

    original_factory = object()
    evaluator = SimpleNamespace(
        eval_policy=policy, rollout=rollout, make_lerobot_libero_env=original_factory
    )
    replacement = object()
    with api().evaluation_audit(evaluator, replacement, tmp_path, 2):
        assert evaluator.make_lerobot_libero_env is replacement
        evaluator.eval_policy(
            None, None, [[("a", "a instruction", [0])], [("b", "b instruction", [0])]]
        )
    assert evaluator.eval_policy is policy and evaluator.rollout is rollout
    assert evaluator.make_lerobot_libero_env is original_factory
    assert all(env.closed and env.resets == 1 for env in envs)
