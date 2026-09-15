"""Audit actual LIBERO task/initial-state coverage without default GPU imports."""

from contextlib import contextmanager
from datetime import datetime, timezone
import inspect
import json
import math
from numbers import Integral


class SeededResetVector:
    """Split vector seeds through the public per-worker reset API."""

    def __init__(self, env):
        self.env = env

    def __len__(self):
        return len(self.env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None, **kwargs):
        if seed is None or isinstance(seed, Integral):
            return self.env.reset(seed=seed, **kwargs)
        seeds = list(seed)
        if len(seeds) != len(self):
            raise ValueError("one seed is required for each vector environment")
        observations, infos = [], []
        for index, value in enumerate(seeds):
            obs, info = self.env.reset(id=index, seed=value, **kwargs)
            observations.append(obs[0])
            infos.append(info[0])
        return observations, infos


def _rows(value):
    return value.tolist() if hasattr(value, "tolist") else value


class EpisodeLedger:
    """Check actual episode identities, flush progress, and verify summary rates."""

    def __init__(self, output, expected_episodes):
        self.output = output
        self.expected_episodes = expected_episodes
        self.schedule = None
        self.expected = set()
        self.rows = []
        self.seen = set()

    def begin(self, schedule):
        if self.schedule is not None:
            raise RuntimeError("evaluation audit accepts one schedule per run")
        self.schedule = schedule
        pairs = [
            (name, int(ep))
            for batch in schedule
            for name, _, ids in batch
            for ep in ids
        ]
        self.expected = set(pairs)
        if len(pairs) != self.expected_episodes or len(self.expected) != len(pairs):
            raise RuntimeError(
                "episode budget is capped, duplicated, or differs from requested count"
            )
        (self.output / "episodes.jsonl").touch(exist_ok=False)
        self.write_status("running")

    def record(self, batch_index, rollout_index, names, ids, data, seeds):
        batch = self.schedule[batch_index]
        done, success = _rows(data["done"]), _rows(data["success"])
        if not (len(names) == len(ids) == len(done) == len(success) == len(batch)):
            raise RuntimeError("environment identity/result batch sizes disagree")
        for index, (task, _, planned_ids) in enumerate(batch):
            if rollout_index >= len(planned_ids):
                continue
            pair = (names[index], int(ids[index]))
            if pair != (task, planned_ids[rollout_index]):
                raise RuntimeError(
                    f"episode identity mismatch: {pair}, expected {(task, planned_ids[rollout_index])}"
                )
            if pair in self.seen:
                raise RuntimeError(f"duplicate episode: {pair}")
            flags = [bool(value) for value in done[index]]
            if not any(flags) or len(flags) != len(success[index]):
                raise RuntimeError(
                    "rollout must include a terminal step and aligned success flags"
                )
            steps = flags.index(True) + 1
            row = {
                "task": pair[0],
                "init_state_id": pair[1],
                "success": any(bool(value) for value in success[index][:steps]),
                "primitive_steps": steps,
                "env_seed": int(seeds[index]) if seeds is not None else None,
                "batch_index": batch_index,
                "rollout_index": rollout_index,
            }
            with (self.output / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.seen.add(pair)
            self.rows.append(row)
        self.write_status("running")

    def write_status(self, status, error=None):
        tasks = {}
        for task in sorted({name for name, _ in self.expected}):
            rows = [row for row in self.rows if row["task"] == task]
            tasks[task] = {
                "episodes": len(rows),
                "successes": sum(row["success"] for row in rows),
                "init_state_ids": sorted(row["init_state_id"] for row in rows),
                "expected_init_state_ids": sorted(
                    ep for name, ep in self.expected if name == task
                ),
            }
        report = {
            "status": status,
            "expected_episodes": self.expected_episodes,
            "completed_episodes": len(self.rows),
            "successes": sum(row["success"] for row in self.rows),
            "tasks": tasks,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if error is not None:
            report["error"] = error
        temporary = self.output / "coverage.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(self.output / "coverage.json")

    def finish(self, results):
        if self.seen != self.expected or len(self.rows) != self.expected_episodes:
            raise RuntimeError("incomplete episode coverage")
        for task in ["overall", *sorted({name for name, _ in self.expected})]:
            rows = (
                self.rows
                if task == "overall"
                else [row for row in self.rows if row["task"] == task]
            )
            expected = {
                "pc_successes": 100 * sum(row["success"] for row in rows) / len(rows),
                "avg_episode_length": sum(row["primitive_steps"] for row in rows)
                / len(rows),
            }
            for key, value in expected.items():
                observed = results.get(task, {}).get(key)
                if not isinstance(observed, (int, float)) or not math.isclose(
                    observed, value, abs_tol=1e-6
                ):
                    raise RuntimeError(
                        f"summary disagrees with episode ledger: {task}.{key}"
                    )
        self.write_status("passed")


@contextmanager
def evaluation_audit(evaluator, factory, output, expected_episodes):
    """Install the explicit-reset adapter and audit rollouts, restoring on exit."""
    original_policy = evaluator.eval_policy
    original_rollout = evaluator.rollout
    original_factory = evaluator.make_lerobot_libero_env
    ledger = EpisodeLedger(output, expected_episodes)
    groups = []
    environments = []

    def audited_rollout(*args, **kwargs):
        if not groups:
            raise RuntimeError("unexpected extra rollout")
        batch_index, rollout_index = groups.pop(0)
        bound = inspect.signature(original_rollout).bind(*args, **kwargs)
        env = bound.arguments["env"]
        if not any(env is previous for previous in environments):
            environments.append(env)
        bound.arguments["env"] = SeededResetVector(env)
        # External evaluator accumulates descriptions across task batches.
        bound.arguments["task_description"] = [
            entry[1] for entry in ledger.schedule[batch_index]
        ]
        data = original_rollout(*bound.args, **bound.kwargs)
        ledger.record(
            batch_index,
            rollout_index,
            env.get_env_attr("evaluation_task_name"),
            env.get_env_attr("evaluation_episode_id"),
            data,
            bound.arguments.get("seeds"),
        )
        return data

    def audited_policy(*args, **kwargs):
        bound = inspect.signature(original_policy).bind(*args, **kwargs)
        ledger.begin(bound.arguments["schedule"])
        groups.extend(
            (index, episode)
            for index, batch in enumerate(ledger.schedule)
            for episode in range(max(len(ids) for _, _, ids in batch))
        )
        result = original_policy(*args, **kwargs)
        ledger.finish(result)
        return result

    evaluator.eval_policy = audited_policy
    evaluator.rollout = audited_rollout
    evaluator.make_lerobot_libero_env = factory
    try:
        yield ledger
        if ledger.schedule is None:
            raise RuntimeError("evaluation did not reach the audited policy loop")
    except BaseException as exc:
        ledger.write_status("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        evaluator.eval_policy = original_policy
        evaluator.rollout = original_rollout
        evaluator.make_lerobot_libero_env = original_factory
        for env in environments:
            env.close()
