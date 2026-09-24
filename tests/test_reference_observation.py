"""Reference images may be replayed only under a narrow, audited reset contract."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/robotics_bench/simulators/reference_observation.py"
    )
    assert path.exists(), "Reference observation validation is missing"
    spec = importlib.util.spec_from_file_location("_reference_obs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hash_obs(obs):
    h = hashlib.sha256()
    for key in ("primary_image", "secondary_image", "wrist_image", "proprio"):
        h.update(key.encode())
        h.update(str(obs[key].dtype).encode())
        h.update(obs[key].tobytes())
    return h.hexdigest()


@pytest.fixture
def sample(tmp_path):
    obs = {
        key: np.full((2, 2, 3), 100, dtype=np.uint8)
        for key in ("primary_image", "secondary_image", "wrist_image")
    }
    obs["proprio"] = np.zeros(9, dtype=np.float64)
    ref = dict(
        task="OpenDrawer",
        init_state_id=7,
        env_seed=7,
        layout_id=1,
        style_id=1,
        description="open drawer",
        initial_state_sha256="state",
        initial_xml_sha256="xml",
        initial_observation_sha256=hash_obs(obs),
    )
    np.savez(tmp_path / "000007.npz", **obs)
    cert = {
        "format": "robocasa-reference-observation-v1",
        "reference": ref,
        "allowed_pixel_variations": [
            {"key": "secondary_image", "index": [0, 0, 0], "values": [100, 101]}
        ],
    }
    (tmp_path / "000007.json").write_text(json.dumps(cert))
    native = {k: v.copy() for k, v in obs.items()}
    native["secondary_image"][0, 0, 0] = 101
    actual = {**ref, "initial_observation_sha256": hash_obs(native)}
    return tmp_path, ref, actual, native, obs


def test_replays_exact_reference_without_mutating_native(sample):
    directory, ref, actual, native, expected = sample
    code = load_module()
    replay, audit = code.restore_reference_observation(
        directory, 7, ref, actual, native
    )
    assert hash_obs(replay) == ref["initial_observation_sha256"]
    assert native["secondary_image"][0, 0, 0] == 101
    assert audit["changed_channels"] == 1
    assert audit["rendered_observation_sha256"] == actual["initial_observation_sha256"]
    assert all(np.array_equal(replay[k], v) for k, v in expected.items())


@pytest.mark.parametrize(
    "change",
    [
        "outside_pixel",
        "larger_delta",
        "proprio",
        "physics",
        "tampered_snapshot",
        "tampered_certificate",
    ],
)
def test_rejects_changes_outside_the_certificate(sample, change):
    directory, ref, actual, native, expected = sample
    if change == "outside_pixel":
        native["primary_image"][0, 0, 0] = 101
    elif change == "larger_delta":
        native["secondary_image"][0, 0, 0] = 102
    elif change == "proprio":
        native["proprio"][0] = 0.001
    elif change == "physics":
        actual["initial_state_sha256"] = "changed"
    elif change == "tampered_snapshot":
        expected["primary_image"][0, 1, 0] = 0
        np.savez(directory / "000007.npz", **expected)
    else:
        cert = json.loads((directory / "000007.json").read_text())
        cert["allowed_pixel_variations"][0]["values"] = [100, 102]
        (directory / "000007.json").write_text(json.dumps(cert))
    actual["initial_observation_sha256"] = hash_obs(native)
    with pytest.raises(ValueError):
        load_module().restore_reference_observation(directory, 7, ref, actual, native)


def test_reset_replay_only_replaces_initial_input_and_first_video_frame(
    sample, tmp_path
):
    directory, reference, actual, native, expected = sample
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_snapshot_runner", root / "benchmarks/static/cosmos_robocasa/run.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    baseline = tmp_path / "baseline"
    target = baseline / "initializations" / "000007"
    target.mkdir(parents=True)
    (target / "episode.json").write_text(json.dumps(reference))
    output = tmp_path / "run"
    output.mkdir()

    class Sim:
        episode_metadata = actual

        def reset(self, index, seed=None):
            return native

        def render(self):
            return np.full((2, 6, 3), 9, dtype=np.uint8)

        def step(self, action):
            return "live-observation", False, False

        def save_initialization(self, path):
            path.mkdir()
            (path / "episode.json").write_text(json.dumps(actual))

    adapter = runner.ReferenceReset(Sim(), baseline, output, 0, snapshots=directory)
    result = adapter.reset(7, seed=7)
    assert hash_obs(result) == reference["initial_observation_sha256"]
    assert (
        adapter.episode_metadata["initial_rendered_observation_sha256"]
        == actual["initial_observation_sha256"]
    )
    assert np.all(adapter.render() == 100)
    adapter.save_initialization(output / "initial")
    saved = json.loads((output / "initial/episode.json").read_text())
    assert (
        saved["initial_observation_sha256"] == reference["initial_observation_sha256"]
    )
    assert (output / "initial/rendered-observation.npz").is_file()
    assert adapter.step(None)[0] == "live-observation"
    assert np.all(adapter.render() == 9)


def test_rejected_snapshot_attempt_is_logged(sample, tmp_path):
    directory, reference, actual, native, expected = sample
    native["primary_image"][0, 0, 0] = 102
    actual["initial_observation_sha256"] = hash_obs(native)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_rejected_snapshot_runner", root / "benchmarks/static/cosmos_robocasa/run.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    baseline = tmp_path / "baseline"
    target = baseline / "initializations" / "000007"
    target.mkdir(parents=True)
    (target / "episode.json").write_text(json.dumps(reference))
    output = tmp_path / "run"
    output.mkdir()

    class Sim:
        episode_metadata = actual

        def reset(self, index, seed=None):
            return native

    with pytest.raises(ValueError, match="Uncertified"):
        runner.ReferenceReset(Sim(), baseline, output, 16, snapshots=directory).reset(
            7, seed=7
        )
    records = [
        json.loads(line)
        for line in (output / "initialization-attempts.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1 and records[0]["accepted"] is False
    assert "Uncertified" in records[0]["observation_replay_error"]
