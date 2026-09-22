"""Inference replay must keep Cosmos case assets and horizons separate."""

import importlib.util
from pathlib import Path

import pytest


def benchmark():
    path = Path(__file__).parents[1] / "benchmarks/inference/run.py"
    spec = importlib.util.spec_from_file_location("inference_case_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("suite,horizon", [("libero", 16), ("robocasa", 32)])
def test_case_resources_are_not_cross_bound(tmp_path, monkeypatch, suite, horizon):
    api = benchmark()
    args = api.parser().parse_args(
        [
            "--case",
            "cosmos_" + suite,
            "--input",
            "input.npz",
            "--task",
            "task",
            "--output-dir",
            "output",
            "--gpu",
            "0",
        ]
    )
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("ROBOTICS_COSMOS_SOURCE", str(source))
    for case, prefix in [
        ("libero", "ROBOTICS_COSMOS"),
        ("robocasa", "ROBOTICS_COSMOS_ROBOCASA"),
    ]:
        for key in ("CHECKPOINT", "DATASET_STATS", "TEXT_EMBEDDINGS", "VAE_CHECKPOINT"):
            path = tmp_path / (case + "-" + key)
            path.touch()
            monkeypatch.setenv(prefix + "_" + key, str(path))
    options = api.cosmos_engine_options(args)
    assert options["suite"] == suite
    assert options["action_horizon"] == horizon
    assert options["num_inference_steps"] == 5
    for key in ("checkpoint", "dataset_stats", "text_embeddings", "vae_checkpoint"):
        assert options[key].name == suite + "-" + key.upper()


def test_robocasa_does_not_fall_back_to_libero_assets(tmp_path, monkeypatch):
    api = benchmark()
    args = api.parser().parse_args(
        [
            "--case",
            "cosmos_robocasa",
            "--input",
            "input.npz",
            "--task",
            "task",
            "--output-dir",
            "output",
            "--gpu",
            "0",
        ]
    )
    monkeypatch.setenv("ROBOTICS_COSMOS_SOURCE", str(tmp_path))
    monkeypatch.setenv("ROBOTICS_COSMOS_CHECKPOINT", str(tmp_path))
    monkeypatch.delenv("ROBOTICS_COSMOS_ROBOCASA_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="ROBOTICS_COSMOS_ROBOCASA_CHECKPOINT"):
        api.cosmos_engine_options(args)
