"""Initialization retries preserve exact reference acceptance before policy calls."""

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = (
        Path(__file__).resolve().parents[1] / "benchmarks/static/cosmos_robocasa/run.py"
    )
    spec = importlib.util.spec_from_file_location("_retry_run", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def setup(tmp_path):
    reference = tmp_path / "reference" / "initializations" / "000000"
    reference.mkdir(parents=True)
    metadata = dict(
        task="OpenDrawer",
        init_state_id=0,
        env_seed=0,
        layout_id=1,
        style_id=1,
        description="open drawer",
        initial_state_sha256="state",
        initial_xml_sha256="xml",
        initial_observation_sha256="good",
    )
    (reference / "episode.json").write_text(json.dumps(metadata))
    output = tmp_path / "output"
    output.mkdir()
    return reference.parents[1], output, metadata


class Simulator:
    def __init__(self, metadata, changes):
        self.metadata = metadata
        self.changes = iter(changes)
        self.calls = 0

    def reset(self, init_state_id, seed=None):
        self.calls += 1
        self.episode_metadata = {**self.metadata, **next(self.changes)}
        return self.episode_metadata["initial_observation_sha256"]


@pytest.mark.parametrize(
    "changes,retries,message",
    [
        ([{"initial_observation_sha256": "bad"}, {}], 1, None),
        ([{"initial_observation_sha256": "bad"}] * 2, 1, "observation"),
        ([{"initial_state_sha256": "wrong"}], 8, "initial_state"),
    ],
)
def test_retry_accepts_only_exact_initialization(setup, changes, retries, message):
    reference, output, metadata = setup
    sim = Simulator(metadata, changes)
    code = module()
    assert hasattr(code, "ReferenceReset"), (
        "The bounded reference reset adapter is missing"
    )
    adapter = code.ReferenceReset(sim, reference, output, retries)
    if message:
        with pytest.raises(RuntimeError, match=message):
            adapter.reset(0, seed=0)
    else:
        assert adapter.reset(0, seed=0) == "good"
    assert sim.calls == len(changes)
    audit = [
        json.loads(line)
        for line in (output / "initialization-attempts.jsonl").read_text().splitlines()
    ]
    assert len(audit) == len(changes)
    assert audit[-1]["accepted"] is (message is None)
