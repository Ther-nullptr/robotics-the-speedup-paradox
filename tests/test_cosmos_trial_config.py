"""CPU-only configuration checks with deliberately unimportable fake sources."""

import builtins
import importlib.util
import json
from pathlib import Path

import pytest


CONFIG = (
    Path(__file__).resolve().parents[1] / "benchmarks/static/cosmos_libero/config.py"
)
SOURCE_FILES = (
    "cosmos_policy/experiments/robot/cosmos_utils.py",
    "cosmos_policy/_src/predict2/utils/model_loader.py",
    "cosmos_policy/config/config.py",
)
SIGNATURE = (
    "cfg, model, dataset_stats, obs, task_label_or_embedding, seed=1, "
    "randomize_seed=False, num_denoising_steps_action=5, "
    "generate_future_state_and_value_in_parallel=True, worker_id=0, batch_size=1"
)


@pytest.fixture
def config(monkeypatch):
    for name in (
        "SOURCE",
        "CHECKPOINT",
        "DATASET_STATS",
        "TEXT_EMBEDDINGS",
        "VAE_CHECKPOINT",
        "LIBERO_CONFIG_DIR",
    ):
        monkeypatch.delenv("ROBOTICS_COSMOS_" + name, raising=False)
    monkeypatch.delenv("COSMOS_SMOKE", raising=False)
    original_import = builtins.__import__

    def cpu_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "numpy", "libero", "cosmos_policy"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_import)
    assert CONFIG.is_file(), "Cosmos CLI configuration module is not implemented"
    spec = importlib.util.spec_from_file_location("cosmos_trial_config_test", CONFIG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    for relative in SOURCE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise AssertionError('external source was imported')\n")
    (source / SOURCE_FILES[0]).write_text(
        "raise AssertionError('external source was imported')\n"
        f"def get_action({SIGNATURE}): pass\n"
    )
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"not a pickle; must only be hashed")
    embeddings = tmp_path / "text.pkl"
    embeddings.write_bytes(b"not a pickle; must only be hashed")
    vae = tmp_path / "vae.pth"
    vae.write_bytes(b"not a pickle; must only be hashed")
    stats = tmp_path / "statistics.json"
    stats.write_text(
        json.dumps(
            {
                "actions_min": [-1] * 7,
                "actions_max": [1] * 7,
                "proprio_min": [-1] * 9,
                "proprio_max": [1] * 9,
            }
        )
    )
    libero = tmp_path / "libero-config"
    libero.mkdir()
    (libero / "config.yaml").write_text("{}\n")
    return {
        "cosmos_source": source,
        "checkpoint": checkpoint,
        "dataset_stats": stats,
        "text_embeddings": embeddings,
        "vae_checkpoint": vae,
        "libero_config_dir": libero,
        "output_dir": tmp_path / "output",
    }


def arguments(inputs, *extra):
    result = []
    for name, value in inputs.items():
        result.extend(["--" + name.replace("_", "-"), str(value)])
    return result + list(extra)


def plan(config, inputs, *extra):
    return config.build_plan(
        config.build_parser().parse_args(arguments(inputs, "--dry-run", *extra))
    )


def test_dry_run_is_cpu_only_and_does_not_create_output(config, inputs, monkeypatch):
    original = builtins.__import__

    def cpu_only(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "numpy", "libero", "cosmos_policy"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_only)
    result = plan(config, inputs)
    assert result["format"] == "cosmos-libero-run-v1"
    assert result["case_id"] == "cosmos_libero"
    assert result["gpu"] is None
    assert result["output_dir"] == str(inputs["output_dir"])
    assert result["resources"] == {
        key: str(value) for key, value in inputs.items() if key != "output_dir"
    }
    assert result["options"] == {
        "suite": "libero_object",
        "episodes": 1,
        "task_ids": None,
        "seed": 195,
        "env_seed": 0,
        "n_action_steps": 16,
        "num_inference_steps": 5,
        "schedule": "sync",
        "overlap_actions": 0,
        "model_config": "cosmos_predict2_2b_480p_libero__inference_only",
        "quant": "none",
        "max_steps": 280,
        "video_episodes_per_task": 0,
        "video_fps": 30.0,
        "paper_action_time_ms": 50.0,
        "paper_inference_time_ms": None,
        "paper_inference_time_source": None,
        "settle_steps": 10,
        "action_horizon": 16,
        "delay_state_with_observation": True,
    }
    assert result["paper_async"]["timing"]["status"] == "requires_inference_profile"
    assert not inputs["output_dir"].exists()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "suite,limit",
    [
        ("libero_spatial", 220),
        ("libero_object", 280),
        ("libero_goal", 300),
        ("libero_10", 520),
    ],
)
def test_suite_step_limits(config, inputs, suite, limit):
    assert plan(config, inputs, "--suite", suite)["options"]["max_steps"] == limit


def test_overlap_selects_paper_async_and_preserves_timing_contract(config, inputs):
    result = plan(
        config,
        inputs,
        "--overlap-actions",
        "2",
        "--paper-inference-time-ms",
        "120",
        "--paper-inference-time-source",
        "profile.json",
    )
    assert result["options"]["schedule"] == "paper_async"
    assert result["paper_async"]["n_prime"] == 2
    assert result["paper_async"]["timing"]["cycle_time_ms"] == 820


@pytest.mark.parametrize(
    "flags",
    [
        ("--episodes", "0"),
        ("--seed", "-1"),
        ("--env-seed", "-1"),
        ("--n-action-steps", "0"),
        ("--n-action-steps", "17"),
        ("--num-inference-steps", "0"),
        ("--overlap-actions", "-1"),
        ("--overlap-actions", "17"),
        ("--schedule", "sync", "--overlap-actions", "1"),
        ("--max-steps", "0"),
        ("--video-episodes-per-task", "-1"),
        ("--video-fps", "0"),
        ("--video-fps", "nan"),
        ("--video-fps", "inf"),
        ("--paper-action-time-ms", "0"),
        ("--paper-inference-time-ms", "20"),
        (
            "--paper-inference-time-ms",
            "nan",
            "--paper-inference-time-source",
            "profile",
        ),
        ("--paper-inference-time-ms", "20", "--paper-inference-time-source", " "),
        ("--task-ids", ""),
        ("--task-ids", "1,,2"),
        ("--task-ids", "-1"),
        ("--task-ids", "10"),
        ("--task-ids", "2,2"),
        ("--task-ids", "foo"),
        ("--gpu", "0,1"),
        ("--gpu", "-1"),
        ("--gpu", "GPU-foo"),
        ("--model-config", " "),
    ],
)
def test_invalid_options_fail_before_execution(config, inputs, flags):
    with pytest.raises(ValueError):
        plan(config, inputs, *flags)
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize(
    "flags",
    [
        ("--suite", "libero_90"),
        ("--quant", "w8a8"),
        ("--unknown", "x"),
        ("--ep", "2"),
        ("--record-video", "--no-record-video"),
        ("--record-video", "--video-episodes-per-task", "2"),
    ],
)
def test_parser_rejects_unknown_or_conflicting_options(config, inputs, flags):
    with pytest.raises(SystemExit):
        config.build_parser().parse_args(arguments(inputs, *flags))


def test_task_selection_and_video_modes(config, inputs):
    result = plan(
        config, inputs, "--task-ids", "2,0,9", "--record-video", "--max-steps", "8"
    )
    assert result["options"]["task_ids"] == [2, 0, 9]
    assert result["options"]["video_episodes_per_task"] == "all"
    assert result["options"]["max_steps"] == 8
    assert (
        plan(config, inputs, "--no-record-video")["options"]["video_episodes_per_task"]
        == 0
    )
    assert (
        plan(config, inputs, "--video-episodes-per-task", "3")["options"][
            "video_episodes_per_task"
        ]
        == 3
    )


def test_execution_requires_explicit_single_gpu(config, inputs):
    args = config.build_parser().parse_args(arguments(inputs))
    with pytest.raises(ValueError, match="explicit --gpu"):
        config.build_plan(args)
    for gpu in ["0", "12", "GPU-12345678-1234-1234-1234-123456789abc"]:
        assert plan(config, inputs, "--gpu", gpu)["gpu"] == gpu


def test_environment_resources_are_overridden_by_cli(config, inputs, monkeypatch):
    for key, value in inputs.items():
        if key != "output_dir":
            name = "SOURCE" if key == "cosmos_source" else key.upper()
            monkeypatch.setenv("ROBOTICS_COSMOS_" + name, str(value))
    result = config.build_plan(
        config.build_parser().parse_args(
            [
                "--output-dir",
                str(inputs["output_dir"]),
                "--dry-run",
            ]
        )
    )
    assert result["resources"]["cosmos_source"] == str(inputs["cosmos_source"])
    monkeypatch.setenv("ROBOTICS_COSMOS_CHECKPOINT", "/does/not/exist")
    assert plan(config, inputs)["resources"]["checkpoint"] == str(inputs["checkpoint"])


@pytest.mark.parametrize(
    "resource", ["checkpoint", "dataset_stats", "text_embeddings", "vae_checkpoint"]
)
def test_missing_resources_are_rejected(config, inputs, resource):
    inputs[resource].unlink()
    with pytest.raises(ValueError, match="missing"):
        plan(config, inputs)


@pytest.mark.parametrize("relative", SOURCE_FILES)
def test_missing_source_contract_files_are_rejected(config, inputs, relative):
    (inputs["cosmos_source"] / relative).unlink()
    with pytest.raises(ValueError, match="missing"):
        plan(config, inputs)


def test_incompatible_action_signature_is_rejected(config, inputs):
    (inputs["cosmos_source"] / SOURCE_FILES[0]).write_text(
        "def get_action(cfg): pass\n"
    )
    with pytest.raises(ValueError, match="get_action"):
        plan(config, inputs)


def test_extra_required_action_parameter_is_rejected(config, inputs):
    (inputs["cosmos_source"] / SOURCE_FILES[0]).write_text(
        f"def get_action(unhandled_required_input, {SIGNATURE}): pass\n"
    )
    with pytest.raises(ValueError, match="get_action"):
        plan(config, inputs)


def test_checkpoint_requires_pt_extension(config, inputs):
    renamed = inputs["checkpoint"].with_suffix(".pkl")
    inputs["checkpoint"].rename(renamed)
    inputs["checkpoint"] = renamed
    with pytest.raises(ValueError, match=r"\.pt"):
        plan(config, inputs)


def test_parser_requires_resource_paths(config):
    with pytest.raises(SystemExit):
        config.build_parser().parse_args(["--output-dir", "unused", "--dry-run"])


@pytest.mark.parametrize(
    "value", [[0] * 6, [True] * 7, [float("nan")] * 7, ["0"] * 7, None]
)
def test_invalid_statistics_are_rejected(config, inputs, value):
    stats = json.loads(inputs["dataset_stats"].read_text())
    stats["actions_min"] = value
    inputs["dataset_stats"].write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="actions_min"):
        plan(config, inputs)


def test_existing_output_and_missing_libero_config_fail(config, inputs):
    inputs["output_dir"].mkdir()
    with pytest.raises(ValueError, match="already exists"):
        plan(config, inputs)
    inputs["output_dir"].rmdir()
    (inputs["libero_config_dir"] / "config.yaml").unlink()
    with pytest.raises(ValueError, match="config.yaml"):
        plan(config, inputs)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_cosmos_smoke_environment_is_rejected(config, inputs, monkeypatch, value):
    monkeypatch.setenv("COSMOS_SMOKE", value)
    with pytest.raises(ValueError, match="COSMOS_SMOKE"):
        plan(config, inputs)


def test_fingerprint_uses_content_and_options_without_fabricated_git(config, inputs):
    first = plan(config, inputs)
    assert first["identity"]["source"]["git_available"] is False
    assert "git_head" not in first["identity"]["source"]
    assert first["identity"]["source"]["python_tree_sha256"]
    assert first["identity"]["entry_code_sha256"]
    assert first["case_fingerprint"] == plan(config, inputs)["case_fingerprint"]
    assert (
        first["case_fingerprint"]
        != plan(config, inputs, "--seed", "196")["case_fingerprint"]
    )
    inputs["checkpoint"].write_bytes(b"changed weights")
    changed = plan(config, inputs)
    assert first["case_fingerprint"] != changed["case_fingerprint"]
    source = inputs["cosmos_source"] / "cosmos_policy/new_module.py"
    source.write_text("raise AssertionError('do not import')\n")
    assert changed["case_fingerprint"] != plan(config, inputs)["case_fingerprint"]


@pytest.mark.parametrize(
    "resource", ["text_embeddings", "vae_checkpoint", "libero_config_dir"]
)
def test_auxiliary_resource_content_changes_fingerprint(config, inputs, resource):
    first = plan(config, inputs)["case_fingerprint"]
    path = inputs[resource]
    if resource == "libero_config_dir":
        path = path / "config.yaml"
    path.write_bytes(b"changed resource")
    assert first != plan(config, inputs)["case_fingerprint"]
