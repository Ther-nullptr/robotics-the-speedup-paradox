"""Keep original reset inputs available for later paired-run auditing."""

import numpy as np
from robotics_bench.simulators.robocasa import NativeRoboCasaSimulator


def test_initialization_saves_raw_observation_arrays(tmp_path):
    sim = NativeRoboCasaSimulator(tmp_path, tmp_path / "controller.pkl", "CloseDrawer")
    sim._ready = True
    sim._initial_state = np.array([1.0, 2.0])
    sim._initial_xml = "<mujoco/>"
    sim.episode_metadata = {"task": "CloseDrawer"}
    sim._initial_observation = {
        "primary_image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "proprio": np.zeros(9, dtype=np.float64),
    }
    target = tmp_path / "initial"
    sim.save_initialization(target)
    assert (target / "rendered-observation.npz").is_file(), (
        "Reset observations must be persisted"
    )
    with np.load(target / "rendered-observation.npz", allow_pickle=False) as data:
        for key, value in sim._initial_observation.items():
            assert data[key].dtype == value.dtype
            np.testing.assert_array_equal(data[key], value)


def test_rejected_initialization_is_saved_before_retry(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "benchmarks/static/cosmos_robocasa/run.py"
    )
    spec = importlib.util.spec_from_file_location("_reset_record", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    reference = tmp_path / "reference"
    folder = reference / "initializations/000000"
    folder.mkdir(parents=True)
    expected = dict(
        task="CloseDrawer",
        init_state_id=0,
        env_seed=0,
        layout_id=1,
        style_id=1,
        description="close drawer",
        initial_state_sha256="state",
        initial_xml_sha256="xml",
        initial_observation_sha256="good",
    )
    (folder / "episode.json").write_text(json.dumps(expected))
    output = tmp_path / "run"
    output.mkdir()

    class Sim:
        count = 0
        saved = []

        def reset(self, *args, **kwargs):
            self.count += 1
            self.episode_metadata = {
                **expected,
                "initial_observation_sha256": "bad" if self.count == 1 else "good",
            }
            return {}

        def save_initialization(self, directory):
            directory.mkdir()
            self.saved.append(self.episode_metadata["initial_observation_sha256"])
            (directory / "episode.json").write_text(json.dumps(self.episode_metadata))

    sim = Sim()
    runner.ReferenceReset(sim, reference, output, 1).reset(0, seed=0)
    assert sim.saved == ["bad"]
    assert (
        json.loads(
            (output / "initialization-rejections/000000/01/episode.json").read_text()
        )["initial_observation_sha256"]
        == "bad"
    )
