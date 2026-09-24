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
